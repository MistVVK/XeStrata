// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// setup.py --json's lines (setup.py's "output" section): one JSON object a line, or plain text (the server's own
/// lines once setup starts it).
module Xestrata.Launcher.Protocol

open System
open System.Text.Json

type Cpu = { Name: string; Avx2: bool; Avx512: bool }

/// SAM 3.1's license (setup's --image-sam3-license) and whether it was agreed to.
type Sam3 = { License: string; Agreed: bool }

/// A level of the lists' entry: its id and its name.
type Tier = { Id: string; Title: string }

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
      Experimental: bool
      Recommended: bool
      /// the recommended context for this PC
      Context: int
      /// whether it can read pictures (setup's "Do you want images?", --vision)
      Vision: bool
      /// whether setup takes a RAM budget of its experts, the rest read from the SSD (--resident-budget-gib)
      Budget: bool
      /// the quantization it is of (Describe.Quantizers), its family's or its own
      Quantizer: string }

/// One of setup's options as a field (setup.py's OPTIONS: what each part means is written there).
type OptionSpec =
    { Name: string
      Label: string
      /// choice, int, number, text, path, flag, password or gpus
      Kind: string
      /// install, advanced, images or start
      Place: string
      /// [value, label] each
      Choices: string[][]
      /// a choice's value or a check box's state, as setup offers it now (a string, or true and false)
      Default: JsonElement
      When: string[]
      Min: Nullable<float>
      Max: Nullable<float>
      Pattern: string
      Hint: string
      Takes: string
      About: string
      Vars: Collections.Generic.Dictionary<string, string>
      /// "index" or "pci": the GPUs are choices too
      GpuChoices: string
      PathChoice: string
      /// folder or file
      Pick: string
      /// a check box's values, checked first
      Values: string[]
      /// passed only when changed
      Keep: bool
      /// a password's environment variable, whether its key is saved, and why it cannot be left blank
      Env: string
      Set: bool
      Needed: string }

/// The choices a family takes of one of the options, and the one offered first.
type FamilyOption = { Choices: string[]; Default: string }

type Family =
    { Id: string
      Title: string
      By: string
      About: string
      /// its place in the lists: the model (Describe.Bases), the version (Describe.Variants) and the quantization
      /// (Describe.Quantizers); null from a setup without the levels
      Base: string
      Variant: string
      Quantizer: string
      /// the size setup recommends of it on this PC
      RecommendedSize: string
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
      /// the options whose choices are the family's own (setup's --kv of a plain family)
      Options: Collections.Generic.Dictionary<string, FamilyOption>
      Contexts: int[]
      Models: Model[] }

type ImageSize =
    { Id: string
      DownloadGb: float
      RamGb: float }

/// An image model setup can add (--images), its -uncensored text encoder a choice of its own where it has one.
type ImageModel =
    { Id: string
      Title: string
      Family: string
      Version: string
      Origin: string
      OriginKey: string
      Own: bool
      Sizes: ImageSize[]
      UncensoredSizes: ImageSize[]
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

/// What setup read of the model files given with --describe --gguf (setup.py's custom_info).
type CustomModel =
    { /// its model id, custom-NAME
      Name: string
      Arch: string
      Title: string
      DownloadGb: float }

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
      ImageDefault: string[]
      /// web search for every model as setup set it up: "deepseek", "tinyfish", or null (none)
      WebSearch: string
      /// with --gguf: what setup read of those files (their family, "custom", is in Families), else null
      Custom: CustomModel
      /// the models in setup's order
      Bases: Tier[]
      /// the family a first install is offered
      RecommendedFamily: string
      /// the versions' and the quantizations' names
      Variants: Collections.Generic.Dictionary<string, string>
      Quantizers: Collections.Generic.Dictionary<string, string>
      /// null from a setup without it
      Sam3: Sam3
      /// the options the window shows as fields
      Options: OptionSpec[] }

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
    /// a start found the model already served on its port (setup's attach_running): its address, and whether the
    /// systemd user service serves it
    | Attached of url: string * service: bool
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
            | "attached" ->
                let service = match e.TryGetProperty "service" with | true, v -> v.ValueKind = JsonValueKind.True | _ -> false
                Attached(str e "url", service)
            | _ -> Text line
        with :? JsonException ->
            Text line
