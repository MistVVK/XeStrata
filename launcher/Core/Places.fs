// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// Where the program's parts are (Platform.layout: the 7z's on Windows, what the entry passes on Linux).  The user's
/// settings, logs and models live elsewhere (serve/places.py).
module Xestrata.Launcher.Places

open System.IO

let app = Platform.current.App

let setupPy = Path.Combine(app, "setup.py")

/// The chat in the console (`xestrata chat`).
let chatPy = Path.Combine(app, "chat.py")

/// An installed model's image generation changed (setup.py's setup_images, the config's "images" rewritten).
let imageInstallPy = Path.Combine(app, "tools", "image", "install.py")

let python = Platform.current.Python

/// The launcher's own log of the errors that stop it (#50: a crash left no trace).
let launcherLog = Platform.current.Log

/// What is missing of the program's parts (empty: none), for the message that stops the launcher.
let missing () =
    Platform.current.Required |> List.filter (File.Exists >> not)
