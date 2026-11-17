//! Linux: keep the colour emoji font that the AppImage's WebKitGTK cannot draw away from the app window (#4533).
//!
//! The WebKitGTK inside the AppImage paints a COLRv1 colour font through Skia, and its `colrv1_configure_skpaint` reads past the end of a
//! gradient's colour-stop list. The library is built with libstdc++ range checks on, so the read becomes `abort()` and `WebKitWebProcess`
//! dies: the window goes grey the first time a chat title or a message draws an emoji (Fedora 44 and newer Ubuntu ship
//! `Noto-COLRv1.ttf` as the default colour emoji font; the reporters on #4533 name the same assertion and the same workaround).
//!
//! fontconfig reads one file when `FONTCONFIG_FILE` is set, and that file replaces the normal configuration, so the file written here
//! includes the system configuration first and only then rejects the COLRv1 faces. Emoji are then drawn from whatever else the system
//! has (the outline "Noto Emoji" is preferred); nothing else about the fonts changes.
//!
//! Only the window process is told (main.rs run_window_process, before WebKit starts its own processes). The shell owns the host, the
//! engine and the agent, and they keep the fonts the person chose. A machine with no COLRv1 font is not touched at all.
//!
//! This file is plain std so its decisions are unit-tested on every platform; only the window process on Linux uses it.
//! The idea is that of community PR #4485; the code is written for this tree.

use std::path::{Path, PathBuf};

/// File name patterns of the COLRv1 faces (fontconfig globs, case sensitive, so both spellings). Bitmap (`CBDT`) and outline faces are not
/// listed: they draw fine, and hiding them would leave an empty box where an emoji should be.
pub const COLRV1_GLOBS: &[&str] = &["*COLRv1*", "*colrv1*", "*COLRV1*"];

/// The outline family preferred in place of the colour one.
const OUTLINE_EMOJI: &str = "Noto Emoji";

/// fontconfig's own configuration. `FONTCONFIG_FILE` replaces it, so the written file pulls it back in.
pub const SYSTEM_CONFIG: &str = "/etc/fonts/fonts.conf";

/// `STRATA_FONT_CONFIG=off` turns this off; `STRATA_FONT_CONFIG=/path/to/fonts.conf` uses the person's own file.
const OFF: &str = "off";

/// The name of the file written under the data folder.
const FILE_NAME: &str = "fonts-no-colrv1.conf";

/// Folders fontconfig scans by default, where a colour emoji font is found: the system ones, the person's own, and the XDG data folders.
pub fn font_dirs(env: &dyn Fn(&str) -> Option<String>) -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from("/usr/share/fonts"), PathBuf::from("/usr/local/share/fonts")];
    let home = env("HOME").filter(|h| !h.is_empty()).map(PathBuf::from);
    match env("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
        Some(data) => dirs.push(PathBuf::from(data).join("fonts")),
        None => {
            if let Some(h) = &home {
                dirs.push(h.join(".local/share/fonts"));
            }
        }
    }
    if let Some(h) = &home {
        dirs.push(h.join(".fonts"));
    }
    if let Some(list) = env("XDG_DATA_DIRS") {
        for d in list.split(':').filter(|d| !d.is_empty()) {
            dirs.push(PathBuf::from(d).join("fonts"));
        }
    }
    dirs.dedup();
    dirs
}

/// A COLRv1 font file installed under `dirs`, if any. The scan is shallow and stops at the first one found, so a window that starts on a
/// machine without such a font costs a few directory reads and nothing else.
pub fn colrv1_font_present(dirs: &[PathBuf]) -> Option<PathBuf> {
    fn scan(dir: &Path, depth: u8, found: &mut Option<PathBuf>) {
        if found.is_some() || depth == 0 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan(&path, depth - 1, found);
            }
            else if path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase().contains("colrv1")).unwrap_or(false) {
                *found = Some(path);
                return;
            }
            if found.is_some() {
                return;
            }
        }
    }
    let mut found = None;
    for dir in dirs {
        scan(dir, 5, &mut found);
        if found.is_some() {
            break;
        }
    }
    found
}

/// The configuration file's text: the system's own rules, then the rejection of the COLRv1 faces, then the outline family preferred.
pub fn config_text(system_config: &str) -> String {
    let mut lines: Vec<String> = vec![
        "<?xml version=\"1.0\"?>".into(),
        "<!DOCTYPE fontconfig SYSTEM \"urn:fontconfig:fonts.dtd\">".into(),
        "<!-- Written by Strata for its own window only. The WebKitGTK inside the AppImage aborts while it draws a COLRv1 colour font,".into(),
        "     so those faces are hidden from the window; every other font is the system's own setup, included first.".into(),
        "     Off: STRATA_FONT_CONFIG=off. Your own file: set FONTCONFIG_FILE yourself. -->".into(),
        "<fontconfig>".into(),
        format!("  <include ignore_missing=\"yes\">{system_config}</include>"),
        "  <selectfont>".into(),
    ];
    // fontconfig rejects a font only inside <selectfont>
    for glob in COLRV1_GLOBS {
        lines.push(format!("    <rejectfont><glob>{glob}</glob></rejectfont>"));
    }
    lines.push("  </selectfont>".into());
    lines.push("  <alias binding=\"strong\">".into());
    lines.push("    <family>Noto Color Emoji</family>".into());
    lines.push(format!("    <prefer><family>{OUTLINE_EMOJI}</family></prefer>"));
    lines.push("  </alias>".into());
    lines.push("</fontconfig>".into());
    lines.join("\n") + "\n"
}

/// Write the file under `dir` and return its path, or None when it cannot be written or there is no system configuration to include
/// (then the window would be left without any font). Rewritten only when the contents differ, so a start-up touches the disk once, and a
/// file left by an older Strata is put right. Written through a temporary file and moved, so a second instance starting at the same moment
/// never hands fontconfig half a file.
pub fn ensure_config(dir: &Path, system_config: &Path) -> Option<PathBuf> {
    if !system_config.exists() {
        return None;
    }
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join(FILE_NAME);
    let want = config_text(&system_config.to_string_lossy());
    if matches!(std::fs::read_to_string(&path), Ok(have) if have == want) {
        return Some(path);
    }
    let tmp = dir.join(format!("{FILE_NAME}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, want.as_bytes()).ok()?;
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        // another writer got there first: take its file only if it is the one wanted
        return (std::fs::read_to_string(&path).ok()? == want).then_some(path);
    }
    Some(path)
}

/// The file the window process should hand to fontconfig, or None to leave the environment exactly as it is. In this order, so a person's
/// own choice always wins and a machine without the font is untouched:
///   1. `FONTCONFIG_FILE` already set: theirs.
///   2. `STRATA_FONT_CONFIG=off`: nothing.
///   3. `STRATA_FONT_CONFIG=<path>`: that file, when it exists.
///   4. no COLRv1 font installed: nothing.
pub fn config_for(env: &dyn Fn(&str) -> Option<String>, dir: &Path, dirs: &[PathBuf], system_config: &Path) -> Option<PathBuf> {
    if env("FONTCONFIG_FILE").is_some() || env("STRATA_FONT_CONFIG").as_deref() == Some(OFF) {
        return None;
    }
    if let Some(custom) = env("STRATA_FONT_CONFIG") {
        let p = PathBuf::from(custom);
        return p.is_file().then_some(p);
    }
    colrv1_font_present(dirs)?;
    ensure_config(dir, system_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k: &str| owned.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone())
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("strata-fontguard-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_file_includes_the_system_config_first_and_rejects_only_colrv1_faces_inside_selectfont() {
        let text = config_text("/etc/fonts/fonts.conf");
        assert!(text.starts_with("<?xml"));
        let include = text.find("/etc/fonts/fonts.conf").expect("the system's fonts are kept");
        let select = text.find("<selectfont>").expect("a rejection needs <selectfont>");
        let select_end = text.find("</selectfont>").expect("closed");
        assert!(include < select && select < select_end);
        assert_eq!(text.matches("<rejectfont>").count(), COLRV1_GLOBS.len());
        for glob in COLRV1_GLOBS {
            let at = text.find(&format!("<glob>{glob}</glob>")).unwrap_or_else(|| panic!("{glob} is rejected"));
            assert!(at > select && at < select_end, "{glob} is inside <selectfont>");
        }
        // the bitmap and the outline faces stay
        assert!(!text.contains("NotoColorEmoji.ttf") && !text.contains("NotoEmoji-Regular.ttf"));
        assert!(text.contains("<prefer><family>Noto Emoji</family></prefer>"));
        assert!(!text.contains('{') && !text.contains('}'), "no placeholder left: {text}");
        // well formed enough: every opened tag is closed
        for tag in ["fontconfig", "selectfont", "alias"] {
            assert_eq!(text.matches(&format!("<{tag}")).count(), text.matches(&format!("</{tag}>")).count(), "{tag}");
        }
    }

    #[test]
    fn the_globs_match_the_real_file_names_and_not_the_faces_that_draw_fine() {
        // fontconfig's glob is `*` anywhere; this is the same rule on the file names that matter
        let hit = |name: &str| COLRV1_GLOBS.iter().any(|g| { let core = g.trim_matches('*'); name.contains(core) });
        for bad in ["/usr/share/fonts/google-noto-color-emoji-fonts/Noto-COLRv1.ttf", "/usr/share/fonts/truetype/noto/NotoColorEmoji-COLRv1.ttf",
                    "/home/me/.fonts/Twemoji-colrv1.ttf", "/usr/share/fonts/NotoSansCOLRv1-Regular.ttf"] {
            assert!(hit(bad), "{bad}");
        }
        for good in ["/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
                     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"] {
            assert!(!hit(good), "{good}");
        }
    }

    #[test]
    fn a_persons_own_fontconfig_or_an_opt_out_leaves_the_environment_alone() {
        let dir = scratch("own");
        let system = dir.join("system.conf");
        std::fs::write(&system, b"<fontconfig/>").unwrap();
        let fonts = dir.join("fonts");
        std::fs::create_dir_all(&fonts).unwrap();
        std::fs::write(fonts.join("Noto-COLRv1.ttf"), b"x").unwrap();
        let dirs = vec![fonts];
        assert_eq!(config_for(&env_from(&[("FONTCONFIG_FILE", "/home/me/fonts.conf")]), &dir, &dirs, &system), None);
        assert_eq!(config_for(&env_from(&[("STRATA_FONT_CONFIG", "off")]), &dir, &dirs, &system), None);
        let theirs = dir.join("theirs.conf");
        std::fs::write(&theirs, b"<fontconfig/>").unwrap();
        assert_eq!(config_for(&env_from(&[("STRATA_FONT_CONFIG", theirs.to_str().unwrap())]), &dir, &dirs, &system), Some(theirs));
        assert_eq!(config_for(&env_from(&[("STRATA_FONT_CONFIG", "/no/such/fonts.conf")]), &dir, &dirs, &system), None);
        // no setting at all: the font is there, so the file is written
        assert!(config_for(&env_from(&[]), &dir.join("out"), &dirs, &system).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_machine_without_a_colrv1_font_is_not_touched() {
        let dir = scratch("none");
        let system = dir.join("system.conf");
        std::fs::write(&system, b"<fontconfig/>").unwrap();
        let fonts = dir.join("fonts");
        std::fs::create_dir_all(fonts.join("noto")).unwrap();
        std::fs::write(fonts.join("noto/NotoColorEmoji.ttf"), b"bitmap").unwrap();
        std::fs::write(fonts.join("noto/NotoEmoji-Regular.ttf"), b"outline").unwrap();
        std::fs::write(fonts.join("noto/NotoSansArabic-Regular.ttf"), b"arabic").unwrap();
        assert_eq!(colrv1_font_present(&[fonts.clone()]), None);
        assert_eq!(config_for(&env_from(&[]), &dir.join("out"), &[fonts.clone()], &system), None);
        assert!(!dir.join("out").exists(), "nothing is written");
        std::fs::write(fonts.join("noto/Noto-COLRv1.ttf"), b"colour").unwrap();
        assert!(colrv1_font_present(&[fonts]).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_scan_is_shallow_and_a_missing_folder_is_fine() {
        let dir = scratch("deep");
        let mut deep = dir.clone();
        for i in 0..9 { deep = deep.join(format!("d{i}")); }
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("Noto-COLRv1.ttf"), b"x").unwrap();
        assert_eq!(colrv1_font_present(&[dir.clone(), dir.join("missing")]), None, "nine levels down is not a font folder");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_folders_follow_the_xdg_variables() {
        let none = font_dirs(&env_from(&[("HOME", "/home/me")]));
        assert!(none.contains(&PathBuf::from("/usr/share/fonts")));
        assert!(none.contains(&PathBuf::from("/home/me/.local/share/fonts")));
        assert!(none.contains(&PathBuf::from("/home/me/.fonts")));
        let set = font_dirs(&env_from(&[("HOME", "/home/me"), ("XDG_DATA_HOME", "/data/me"), ("XDG_DATA_DIRS", "/usr/local/share:/var/lib/flatpak/exports/share")]));
        assert!(set.contains(&PathBuf::from("/data/me/fonts")));
        assert!(!set.contains(&PathBuf::from("/home/me/.local/share/fonts")));
        assert!(set.contains(&PathBuf::from("/var/lib/flatpak/exports/share/fonts")));
    }

    #[test]
    fn the_file_is_written_once_an_old_one_is_put_right_and_nothing_is_left_behind() {
        let dir = scratch("write");
        let system = dir.join("system.conf");
        std::fs::write(&system, b"<fontconfig/>").unwrap();
        let out = dir.join("out");
        let first = ensure_config(&out, &system).expect("written");
        assert_eq!(ensure_config(&out, &system).unwrap(), first, "the same path every start");
        std::fs::write(&first, "<fontconfig/>").unwrap();
        assert_eq!(ensure_config(&out, &system), Some(first.clone()));
        assert_eq!(std::fs::read_to_string(&first).unwrap(), config_text(&system.to_string_lossy()));
        assert_eq!(std::fs::read_dir(&out).unwrap().count(), 1, "one file, no temporary");
        // no system configuration to include: nothing is written (the window would have no font at all)
        assert_eq!(ensure_config(&dir.join("out2"), &dir.join("absent.conf")), None);
        assert!(!dir.join("out2").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
