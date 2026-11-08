// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// setup.py run by the bundled Python: with --json for the GUI (its lines as messages, its questions answered on
/// stdin), or on the console as it is for the CLI.
module Xestrata.Launcher.Setup

open System
open System.Diagnostics
open System.Text
open System.Threading.Tasks

let private startInfo (script: string) (args: string seq) =
    let si = ProcessStartInfo(Places.python, WorkingDirectory = Places.app, UseShellExecute = false)
    // The bundled Python sees its own packages only: not a Python installed on the PC (PYTHONHOME, PYTHONPATH) nor the
    // user's site-packages.  It has no ._pth file, which would also leave the script's folder off sys.path (setup.py
    // imports serve/ from beside it, tools/ their neighbours).
    si.Environment.Remove "PYTHONHOME" |> ignore
    si.Environment.Remove "PYTHONPATH" |> ignore
    si.Environment["PYTHONNOUSERSITE"] <- "1"
    si.ArgumentList.Add script
    for a in args do
        si.ArgumentList.Add a
    si

/// `script` (setup.py, chat.py, tools/image/install.py) on this console, interactive as on Linux; its exit code.
let runOnConsole (script: string) (args: string seq) =
    use p = Process.Start(startInfo script args)
    p.WaitForExit()
    p.ExitCode

/// A run the window follows on a page: its answers and its end.
type IRun =
    /// The answer to an Ask, one line.
    abstract Answer: string -> unit
    /// Ends the run and what it started (the server and its engine).
    abstract Stop: unit -> unit

/// setup.py --json with `args` (or `script`, another of the app's scripts that speaks setup.py's --json:
/// tools/image/install.py): each line to `onMessage` (on a thread of its own), the exit code to `onExit` once both
/// streams have ended.
type Run(script: string, args: string seq, onMessage: Protocol.Message -> unit, onExit: int -> unit) =
    let si = startInfo script (Seq.append [ "--json" ] args)
    do
        si.RedirectStandardInput <- true
        si.RedirectStandardOutput <- true
        si.RedirectStandardError <- true
        si.CreateNoWindow <- true
        si.StandardOutputEncoding <- UTF8Encoding false
        si.StandardErrorEncoding <- UTF8Encoding false
        si.StandardInputEncoding <- UTF8Encoding false
        // Python's own streams in UTF-8 (the console's code page otherwise: 932 on a Japanese Windows)
        si.Environment["PYTHONIOENCODING"] <- "utf-8"
        si.Environment["PYTHONUTF8"] <- "1"

    let proc = Process.Start si

    let pump (reader: IO.StreamReader) (wrap: string -> Protocol.Message) =
        Task.Run(fun () ->
            let mutable line = reader.ReadLine()
            while not (isNull line) do
                onMessage (wrap line)
                line <- reader.ReadLine())

    let out = pump proc.StandardOutput Protocol.parse
    let err = pump proc.StandardError Protocol.Log

    do
        Task
            .WhenAll(out, err)
            .ContinueWith(fun (_: Task) ->
                proc.WaitForExit()
                onExit proc.ExitCode)
        |> ignore

    /// The answer to an Ask, one line.
    member _.Answer(answer: string) =
        proc.StandardInput.WriteLine answer
        proc.StandardInput.Flush()

    /// Ends setup and what it started (the server and its engine).
    member _.Stop() =
        try
            if not proc.HasExited then
                proc.Kill true
        with :? InvalidOperationException ->
            ()

    member _.HasExited = proc.HasExited

    interface IRun with
        member this.Answer answer = this.Answer answer
        member this.Stop() = this.Stop()

    new(args: string seq, onMessage: Protocol.Message -> unit, onExit: int -> unit) =
        Run(Places.setupPy, args, onMessage, onExit)

/// The messages of a short run that asks nothing (--describe, --check-update), and its exit code.
let collect (args: string seq) : Task<Protocol.Message list * int> =
    let done' = TaskCompletionSource<Protocol.Message list * int>()
    let messages = Collections.Concurrent.ConcurrentQueue<Protocol.Message>()
    Run(args, messages.Enqueue, (fun code -> done'.SetResult(List.ofSeq messages, code))) |> ignore
    done'.Task

/// How the window reaches setup: the short runs that ask nothing, and the runs of `script` it follows on a page (the
/// launcher's tests give canned ones).
type Access =
    { Collect: string list -> Task<Protocol.Message list * int>
      Start: string -> string list -> (Protocol.Message -> unit) -> (int -> unit) -> IRun }

/// setup.py itself, through the bundled Python.
let real =
    { Collect = collect
      Start = fun script args onMessage onExit -> Run(script, args, onMessage, onExit) :> IRun }
