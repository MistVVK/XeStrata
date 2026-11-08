// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// xestrata-cli.exe: setup.py with the command line given, on this console (`xestrata-cli --help` lists the options),
/// as `xestrata` on Linux: `xestrata-cli chat` the chat in the console (chat.py), `xestrata-cli images` image
/// generation for an installed model (tools/image/install.py, the window's Images ...), `xestrata-cli gui` the window
/// (xestrata.exe beside it).
module Xestrata.Launcher.Cli

[<EntryPoint>]
let main argv =
    match Runtime.problem () with
    | Some why ->
        eprintfn "%s" why
        1
    | None ->
        match List.ofArray argv with
        | "chat" :: rest -> Setup.runOnConsole Places.chatPy rest
        | "images" :: rest -> Setup.runOnConsole Places.imageInstallPy rest
        | "gui" :: rest ->
            let si = System.Diagnostics.ProcessStartInfo(System.IO.Path.Combine(System.AppContext.BaseDirectory, "xestrata.exe"))
            for a in rest do
                si.ArgumentList.Add a
            System.Diagnostics.Process.Start si |> ignore
            0
        | args -> Setup.runOnConsole Places.setupPy args
