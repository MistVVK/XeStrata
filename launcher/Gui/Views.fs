// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

/// The window's building blocks, made in code (no XAML): text, cards, badges and rows, in colours that read in the
/// light and the dark theme alike.
module Xestrata.Launcher.Views

open Avalonia
open Avalonia.Controls
open Avalonia.Controls.Templates
open Avalonia.Layout
open Avalonia.Media
open Xestrata.Launcher.I18n

let private brush (hex: string) = SolidColorBrush(Color.Parse hex) :> IBrush

/// A neutral tint over the theme's background, for cards.
let cardBrush = brush "#14808080"
let cardBorder = brush "#30808080"
let accentTint = brush "#330078D4"
let warnBrush = brush "#C77700"
let failBrush = brush "#D13438"
let okBrush = brush "#107C10"
let mutedBrush = brush "#99808080"

let text (s: string) =
    TextBlock(Text = s, TextWrapping = TextWrapping.Wrap)

let muted (s: string) =
    TextBlock(Text = s, TextWrapping = TextWrapping.Wrap, Foreground = mutedBrush, FontSize = 12.0)

let heading (s: string) =
    TextBlock(Text = s, FontSize = 20.0, FontWeight = FontWeight.SemiBold, TextWrapping = TextWrapping.Wrap)

let subheading (s: string) =
    TextBlock(Text = s, FontSize = 15.0, FontWeight = FontWeight.SemiBold, TextWrapping = TextWrapping.Wrap)

let coloured (b: IBrush) (s: string) =
    TextBlock(Text = s, TextWrapping = TextWrapping.Wrap, Foreground = b)

let stack (spacing: float) (children: Control seq) =
    let p = StackPanel(Spacing = spacing)
    p.Children.AddRange children
    p

let row (spacing: float) (children: Control seq) =
    let p = StackPanel(Orientation = Orientation.Horizontal, Spacing = spacing)
    p.Children.AddRange children
    p

let card (child: Control) =
    Border(
        Child = child,
        Background = cardBrush,
        BorderBrush = cardBorder,
        BorderThickness = Thickness 1.0,
        CornerRadius = CornerRadius 8.0,
        Padding = Thickness 14.0
    )

let badge (b: IBrush) (s: string) =
    Border(
        Child = TextBlock(Text = s, FontSize = 11.0, Foreground = b, FontWeight = FontWeight.SemiBold),
        BorderBrush = b,
        BorderThickness = Thickness 1.0,
        CornerRadius = CornerRadius 4.0,
        Padding = Thickness(6.0, 1.0),
        VerticalAlignment = VerticalAlignment.Center
    )

let button (label: string) (onClick: unit -> unit) =
    let b = Button(Content = label, MinWidth = 96.0, HorizontalContentAlignment = HorizontalAlignment.Center)
    b.Click.Add(fun _ -> onClick ())
    b

let accentButton (label: string) (onClick: unit -> unit) =
    let b = button label onClick
    b.Classes.Add "accent"
    b

/// Buttons at the right end of a row.
let actions (buttons: Control seq) =
    let r = row 8.0 buttons
    r.HorizontalAlignment <- HorizontalAlignment.Right
    r

/// A list's item template.  It takes no null item: a list that scrolls clears its recycled item's content while the
/// template is still set, and Avalonia then builds the template for null (#50: the install page closed on scrolling).
let itemTemplate<'T> (build: 'T -> Control) =
    FuncDataTemplate<'T>((fun x -> not (isNull (box x))), (fun x _ -> build x))

/// A label above a control.
let field (label: string) (control: Control) =
    stack 4.0 [ muted label; control ]

let gb (x: float) =
    trf "unit.gb" "{n} GB" [ "n", box (if x >= 10.0 then x.ToString("0") else x.ToString("0.0")) ]

/// A model's files: to a tenth of a GB at any size (setup's files_gb; 83.5 and 75.8 GB, not 84 and 76)
let gbTenth (x: float) = trf "unit.gb" "{n} GB" [ "n", box (x.ToString("0.0")) ]

let contextLabel (tokens: int) = trf "unit.ktokens" "{n}K tokens" [ "n", box (tokens / 1024) ]
