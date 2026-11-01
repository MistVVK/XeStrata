// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// setup.py --json's lines (setup.py's "output" section): one JSON object a line, or plain text (the server's own
/// lines once setup starts it).
module Xestrata.Launcher.Protocol

open System
open System.Text.Json

type Cpu = { Name: string; Avx2: bool; Avx512: bool }

type Gpu =
    { Index: int
      Name: string
      Vendor: string
      VramGb: float
      Integrated: bool
      Pci: string
      /// why setup cannot use it, or null
      Problem: string }

type Model =
    { Id: string
      About: string
      DownloadGb: float
      RamGb: float
      /// "yes", "low-ram" (part of the experts on the GPU, the rest from the SSD), "budget" (a RAM budget of the
      /// experts), "slow" or "no"
      Fits: string
      Tested: bool
      Experimental: bool
      Recommended: bool
      /// the recommended context for this PC
      Context: int
      /// whether it can read pictures (setup's "Do you want images?", --vision)
      Vision: bool
      /// whether setup takes a RAM budget of its experts, the rest read from the SSD (--resident-budget-gib)
      Budget: bool }

type Family =
    { Id: string
      Title: string
      By: string
      About: string
      License: string
      Experimental: bool
      /// whether setup can add image generation to its models
      Images: bool
      /// whether setup asks the KV cache's precision above 8K of context (--kv)
      Kv: bool
      /// whether setup asks for the experimental speed projection (--experimental-speed-projection)
      SpeedProjection: bool
      /// the precisions --kv takes (where setup does not ask: 16-bit unless one is given)
      KvTakes: string[]
      Contexts: int[]
      Models: Model[] }

/// An image model setup can add (--images), its -uncensored text encoder a choice of its own where it has one.
type ImageModel =
    { Id: string
      Title: string
      About: string
      DownloadGb: float
      /// besides the text model's
      RamGb: float
      Edits: bool
      License: string
      Uncensored: bool
      /// the sizes one can choose (--images name:size), largest first; empty: one size
      Quants: string[] }

type Installed =
    { Config: string
      ModelName: string
      Family: string
      Model: string
      Context: Nullable<int>
      Port: int
      /// its image models as --images names them, the default first (empty: none)
      Images: string[] }

type Describe =
    { Version: string
      Packaged: bool
      RamGb: float
      Cpu: Cpu
      /// the GPU setup takes by default, or null when none can be used
      Gpu: Nullable<int>
      Gpus: Gpu[]
      DataDir: string
      Families: Family[]
      Installed: Installed[]
      Images: ImageModel[]
      /// the image models a model set up for the first time is offered (--images names, the default first)
      ImageDefault: string[] }

type Update =
    { Current: string
      Latest: string
      Newer: bool
      Url: string
      How: string }

type Message =
    | Say of string
    | Ok of string
    | Warn of string
    | Step of int * string
    | Fail of text: string * hint: string
    | Progress of what: string * done': int64 * total: int64
    | Log of string
    | Ask of question: string * choices: string list * defaultChoice: string
    | Described of Describe
    | Updated of Update
    /// a line that is not one of setup's objects
    | Text of string

let private options = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower)

let private str (e: JsonElement) (name: string) =
    match e.TryGetProperty name with
    | true, v when v.ValueKind = JsonValueKind.String -> v.GetString()
    | true, v when v.ValueKind <> JsonValueKind.Null -> v.ToString()
    | _ -> ""

let private int64Of (e: JsonElement) (name: string) =
    match e.TryGetProperty name with
    | true, v when v.ValueKind = JsonValueKind.Number -> v.GetInt64()
    | _ -> 0L

let parse (line: string) : Message =
    if not (line.StartsWith "{") then
        Text line
    else
        try
            use doc = JsonDocument.Parse line
            let e = doc.RootElement
            match str e "type" with
            | "say" -> Say(str e "text")
            | "ok" -> Ok(str e "text")
            | "warn" -> Warn(str e "text")
            | "step" -> Step(int (int64Of e "n"), str e "title")
            | "fail" -> Fail(str e "text", str e "hint")
            | "progress" -> Progress(str e "what", int64Of e "done", int64Of e "total")
            | "log" -> Log(str e "text")
            | "ask" ->
                let choices =
                    match e.TryGetProperty "choices" with
                    | true, v when v.ValueKind = JsonValueKind.Array -> [ for c in v.EnumerateArray() -> c.ToString() ]
                    | _ -> []
                Ask(str e "question", choices, str e "default")
            | "describe" -> Described(e.Deserialize<Describe> options)
            | "update" -> Updated(e.Deserialize<Update> options)
            | _ -> Text line
        with :? JsonException ->
            Text line
