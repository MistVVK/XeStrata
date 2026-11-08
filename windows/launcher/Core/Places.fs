// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// Where the program's parts are in the unpacked 7z: the launchers beside app\ (XeStrata's setup.py, serve\, tools\,
/// data\ and engine\, read-only, with its PACKAGED record) and python\ (Python's embeddable distribution with the
/// packages setup.py needs).  The user's settings, logs and models live elsewhere (serve/places.py).
module Xestrata.Launcher.Places

open System
open System.IO

/// The folder the launcher runs from: the 7z's top.
let root = AppContext.BaseDirectory

let app = Path.Combine(root, "app")

let setupPy = Path.Combine(app, "setup.py")

/// The chat in the console (`xestrata chat` on Linux).
let chatPy = Path.Combine(app, "chat.py")

/// An installed model's image generation changed (setup.py's setup_images, the config's "images" rewritten).
let imageInstallPy = Path.Combine(app, "tools", "image", "install.py")

let python = Path.Combine(root, "python", "python.exe")

/// The launcher's own log of the errors that stop it (#50: a crash left no trace), in the folder serve/places.py keeps
/// the logs in (%LOCALAPPDATA%\xestrata\state).
let launcherLog =
    let local =
        match Environment.GetEnvironmentVariable "LOCALAPPDATA" with
        | null | "" -> Path.Combine(Environment.GetFolderPath Environment.SpecialFolder.UserProfile, "AppData", "Local")
        | d -> d
    Path.Combine(local, "xestrata", "state", "launcher.log")

/// What is missing of the 7z's parts (empty: none), for the message that stops the launcher.
let missing () =
    [ setupPy; python; Path.Combine(app, "PACKAGED") ] |> List.filter (File.Exists >> not)
