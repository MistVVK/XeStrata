// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// What the launchers need before they start: the program's files, and on Windows the Microsoft Visual C++ runtime,
/// which the engine, the SYCL runtime and Python link but the 7z does not carry (it is not free software).
module Xestrata.Launcher.Runtime

open System.Runtime.InteropServices

/// The installer of the runtime, Microsoft's permanent link.
let vcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

/// The runtime's DLLs this PC cannot load (empty: it has them all; Linux has no such runtime).
let vcMissing () =
    if not Platform.isWindows then [] else
    [ "vcruntime140.dll"; "vcruntime140_1.dll"; "msvcp140.dll"; "msvcp140_atomic_wait.dll" ]
    |> List.filter (fun dll ->
        match NativeLibrary.TryLoad dll with
        | true, handle ->
            NativeLibrary.Free handle
            false
        | _ -> true)

/// Why the launcher cannot start (None: it can).
let problem () =
    match Places.missing (), vcMissing () with
    | [], [] -> None
    | [], dlls ->
        Some(
            I18n.trf "runtime.vc" "The Microsoft Visual C++ runtime is not installed (missing: {dlls}).\nInstall it from {url} and start XeStrata again."
                [ "dlls", box (String.concat ", " dlls); "url", box vcRedistUrl ]
        )
    | files, _ when Platform.isWindows ->
        Some(
            I18n.trf "runtime.files" "XeStrata's files are incomplete (missing: {files}).\nUnpack the whole 7z again, into an empty folder."
                [ "files", box (String.concat ", " files) ]
        )
    | files, _ ->
        Some(
            I18n.trf "runtime.filesLinux" "XeStrata's files are not where the window looks for them (missing: {files}).\nOpen the window with xestrata gui (the packages) or ./setup.sh gui (the source)."
                [ "files", box (String.concat ", " files) ]
        )
