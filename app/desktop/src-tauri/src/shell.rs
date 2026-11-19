//! The shell's own choices and the two things the host asks of it besides updates (#3402, #3477).
//!
//! `<Strata data folder>/shell-prefs.json` is written by the App Host when the person changes a setting
//! (`app/host/server/shell-prefs.mjs`) and read here when Strata starts, before any page exists:
//!
//!   { "port": 0 | 1024..65535, "tray": false, "language": "en" }
//!
//! * `port`: the port the window's origin uses. 0 is the default, 47831. When the port is busy the shell takes a free one and tells the
//!   host (`STRATA_PORT_NOTICE=busy:<wanted>:<used>`), which shows the person a warning: the chats and settings of the page are kept per
//!   port, so a different port starts empty.
//! * `tray`: closing the window hides it and keeps Strata running in the tray (Windows, macOS). The tray menu follows `language`.
//! * `/shell/notify` on the loopback bridge shows a desktop notification (an approval is waiting) whether the window is open or hidden.

use std::path::PathBuf;

use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prefs {
    pub port: u16,
    pub tray: bool,
    pub language: String,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { port: 0, tray: false, language: "en".into() }
    }
}

pub const LANGUAGES: [&str; 10] = ["en", "zh", "ja", "de", "fr", "es", "ru", "pt", "ko", "he"];

pub fn prefs_path() -> PathBuf {
    crate::desktoplog::data_dir().join("shell-prefs.json")
}

/// The file's text as choices; anything missing or out of range is the default.
pub fn parse_prefs(text: &str) -> Prefs {
    let value: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap_or(Value::Null);
    let port = value.get("port").and_then(Value::as_u64).filter(|p| (1024..=65535).contains(p)).unwrap_or(0) as u16;
    let tray = value.get("tray").and_then(Value::as_bool).unwrap_or(false);
    let language = value
        .get("language")
        .and_then(Value::as_str)
        .filter(|l| LANGUAGES.contains(l))
        .unwrap_or("en")
        .to_string();
    Prefs { port, tray, language }
}

pub fn read_prefs() -> Prefs {
    std::fs::read_to_string(prefs_path()).map(|t| parse_prefs(&t)).unwrap_or_default()
}

/// The port to ask for: the person's choice, else the app's usual one.
pub fn wanted_port(prefs: &Prefs, usual: u16) -> u16 {
    if prefs.port != 0 { prefs.port } else { usual }
}

/// The notice the host shows when the wanted port was taken.
pub fn busy_notice(wanted: u16, used: u16) -> String {
    format!("busy:{wanted}:{used}")
}

/// `%41%20b` -> "A b" (a query value from the host). Bad escapes are kept as typed.
pub fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(v) = std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The value of `key` in the query of a request path (`/shell/notify?title=..&body=..`).
pub fn query_value(path: &str, key: &str) -> Option<String> {
    let query = path.split_once('?')?.1;
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

/// A short single-line text for a notification: no control characters, bounded.
pub fn clean_text(text: &str, max: usize) -> String {
    text.chars().filter(|c| !c.is_control()).take(max).collect::<String>().trim().to_string()
}

/// The tray menu: (open, quit, tooltip), in the app's language.
pub fn tray_labels(language: &str) -> (&'static str, &'static str, &'static str) {
    match language {
        "zh" => ("打开 Strata", "退出 Strata", "Strata"),
        "ja" => ("Strata を開く", "Strata を終了", "Strata"),
        "de" => ("Strata öffnen", "Strata beenden", "Strata"),
        "fr" => ("Ouvrir Strata", "Quitter Strata", "Strata"),
        "es" => ("Abrir Strata", "Salir de Strata", "Strata"),
        "ru" => ("Открыть Strata", "Выйти из Strata", "Strata"),
        "pt" => ("Abrir o Strata", "Sair do Strata", "Strata"),
        "ko" => ("Strata 열기", "Strata 종료", "Strata"),
        "he" => ("פתיחת Strata", "יציאה מ-Strata", "Strata"),
        _ => ("Open Strata", "Quit Strata", "Strata"),
    }
}

#[cfg(not(target_os = "linux"))]
mod native {
    use tauri::menu::{Menu, MenuItem};
    use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
    use tauri::{AppHandle, Manager};
    use tauri_plugin_notification::NotificationExt;

    const TRAY_ID: &str = "strata-tray";

    fn show_main(app: &AppHandle) {
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
        }
    }

    /// Hide the window and keep Strata running in the tray (made on the first need, with the menu in the app's language).
    pub fn hide_to_tray(app: &AppHandle, language: &str) {
        if app.tray_by_id(TRAY_ID).is_none() {
            let (open, quit, tip) = super::tray_labels(language);
            let built = (|| -> tauri::Result<()> {
                let open_item = MenuItem::with_id(app, "open", open, true, None::<&str>)?;
                let quit_item = MenuItem::with_id(app, "quit", quit, true, None::<&str>)?;
                let menu = Menu::with_items(app, &[&open_item, &quit_item])?;
                let mut builder = TrayIconBuilder::with_id(TRAY_ID)
                    .tooltip(tip)
                    .menu(&menu)
                    .show_menu_on_left_click(false)
                    .on_menu_event(|app, event| match event.id().as_ref() {
                        "open" => show_main(app),
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                            show_main(tray.app_handle());
                        }
                    });
                if let Some(icon) = app.default_window_icon() {
                    builder = builder.icon(icon.clone());
                }
                builder.build(app)?;
                Ok(())
            })();
            if built.is_err() {
                // No tray could be made (a desktop without one): do not hide a window nobody could bring back.
                return;
            }
        }
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.hide();
        }
    }

    pub fn tray_exists(app: &AppHandle) -> bool {
        app.tray_by_id(TRAY_ID).is_some()
    }

    /// Show a desktop notification; a hidden or minimised window also asks for attention (the taskbar button flashes).
    pub fn notify(app: &AppHandle, title: &str, body: &str) -> bool {
        if title.is_empty() {
            return false;
        }
        if let Some(window) = app.get_webview_window("main") {
            let hidden = !window.is_visible().unwrap_or(true) || window.is_minimized().unwrap_or(false);
            if hidden {
                let _ = window.request_user_attention(Some(tauri::UserAttentionType::Informational));
            }
        }
        app.notification().builder().title(title).body(body).show().is_ok()
    }
}

#[cfg(not(target_os = "linux"))]
pub use native::{hide_to_tray, notify, tray_exists};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_gives_choices_and_anything_wrong_is_the_default() {
        assert_eq!(parse_prefs(""), Prefs::default());
        assert_eq!(parse_prefs("{ not json"), Prefs::default());
        let p = parse_prefs(r#"{"port":47900,"tray":true,"language":"de"}"#);
        assert_eq!((p.port, p.tray, p.language.as_str()), (47900, true, "de"));
        assert_eq!(parse_prefs("\u{feff}{\"port\":47900}").port, 47900);
        assert_eq!(parse_prefs(r#"{"port":80}"#).port, 0, "a port below 1024 is not used");
        assert_eq!(parse_prefs(r#"{"port":70000}"#).port, 0);
        assert_eq!(parse_prefs(r#"{"port":"47831"}"#).port, 0, "a text is not a number");
        assert_eq!(parse_prefs(r#"{"tray":"yes","language":"xx"}"#), Prefs::default());
    }

    #[test]
    fn the_wanted_port_is_the_choice_or_the_usual_one() {
        assert_eq!(wanted_port(&Prefs::default(), 47831), 47831);
        assert_eq!(wanted_port(&Prefs { port: 47900, ..Prefs::default() }, 47831), 47900);
        assert_eq!(busy_notice(47831, 51234), "busy:47831:51234");
    }

    #[test]
    fn query_values_are_decoded_and_bounded() {
        let path = "/shell/notify?title=Strata%20needs%20your%20answer&body=a%20%26%20b%3F+ok";
        assert_eq!(query_value(path, "title").as_deref(), Some("Strata needs your answer"));
        assert_eq!(query_value(path, "body").as_deref(), Some("a & b? ok"));
        assert_eq!(query_value(path, "missing"), None);
        assert_eq!(query_value("/shell/notify", "title"), None);
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz%41"), "%zzA");
        assert_eq!(percent_decode("%C3%A4"), "ä");
        assert_eq!(clean_text("a\nb\u{7}c", 10), "abc");
        assert_eq!(clean_text("abcdef", 3), "abc");
    }

    #[test]
    fn every_language_has_tray_labels() {
        for language in LANGUAGES {
            let (open, quit, tip) = tray_labels(language);
            assert!(!open.is_empty() && !quit.is_empty() && !tip.is_empty(), "{language}");
            if language != "en" {
                assert_ne!(open, tray_labels("en").0, "{language}");
            }
        }
        assert_eq!(tray_labels("xx"), tray_labels("en"));
    }
}
