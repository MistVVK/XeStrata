//! The shell's half of the app update: Tauri's signed updater behind a loopback bridge for the App Host.
//!
//! The UI is served by the App Host, so it cannot call Tauri's updater plugin; and the host is Node, which has no updater.
//! The shell therefore opens one loopback listener, protected by a random token it hands only to the host it started
//! (`STRATA_SHELL_BRIDGE`, `STRATA_SHELL_TOKEN`), and the host's `app-updater.mjs` talks to it:
//!
//!   GET  /app-update/check    -> { available, version, current, notes }   (signature-verified metadata from the feed)
//!   POST /app-update/install  -> starts download + install in the background
//!   GET  /app-update/status   -> { state: idle|downloading|installing|done|error, downloaded, total, version, error }
//!
//! The update itself is Tauri's: it downloads the installer, checks its minisign signature against the public key in
//! `tauri.conf.json` (`plugins.updater.pubkey`, the owner's separate updater key) and installs. A check or an install happens
//! only when the user presses "Check for updates" / "Update Strata"; nothing here runs on start, and nothing here touches
//! engines or models (those are Control's component updater: `strata update`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use tauri::AppHandle;
use tauri_plugin_updater::UpdaterExt;

/// Where an endpoint other than the one in `tauri.conf.json` may come from: a local test feed. Only loopback or https.
const ENDPOINT_ENV: &str = "STRATA_UPDATE_APP_ENDPOINT";

/// Environment variables that point Strata Desktop at a custom profile/storage layout (e.g. a user launcher script).
const CUSTOM_PROFILE_ENVS: [&str; 6] = [
    "STRATA_DESKTOP_DATA_DIR",
    "STRATA_HOME",
    "STRATA_MODELS_DIR",
    "STRATA_WORKSPACE",
    "STRATA_INSTALL_DIR",
    "STRATA_DATA_DIR",
];

/// Strip Windows verbatim prefix (`\\?\`) and surrounding quotes from a path string.
fn strip_win_prefix(raw: &str) -> String {
    let trimmed = raw.trim().trim_matches('"');
    if let Some(rest) = trimmed.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = trimmed.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Normalize a Windows directory path for case-insensitive equality comparison (Issue #6573).
pub(crate) fn normalize_install_dir(raw: &str) -> String {
    let clean = strip_win_prefix(raw).replace('/', "\\");
    let trimmed = clean.trim_end_matches('\\');
    if trimmed.len() == 2 && trimmed.ends_with(':') {
        format!("{}\\", trimmed.to_ascii_lowercase())
    } else {
        trimmed.to_ascii_lowercase()
    }
}

/// True when `running` and `registered` are both known and point to different folders.
pub(crate) fn install_path_mismatch(running: Option<&Path>, registered: Option<&Path>) -> bool {
    match (running, registered) {
        (Some(a), Some(b)) => {
            let na = normalize_install_dir(&a.to_string_lossy());
            let nb = normalize_install_dir(&b.to_string_lossy());
            !na.is_empty() && !nb.is_empty() && na != nb
        }
        _ => false,
    }
}

/// Return active custom profile override environment variable names (non-empty) on this process.
pub(crate) fn custom_profile_env_vars() -> Vec<String> {
    CUSTOM_PROFILE_ENVS
        .iter()
        .filter(|k| std::env::var_os(k).map(|v| !v.is_empty()).unwrap_or(false))
        .map(|k| (*k).to_string())
        .collect()
}

/// Validate that a directory is a safe, real target for NSIS `/D=<dir>` (absolute, no quotes/newlines, not inside temp or `%LOCALAPPDATA%\Strata`).
pub(crate) fn safe_installer_target_dir(dir: &Path, temp_dir: Option<&Path>, local_app_data: Option<&Path>) -> Option<String> {
    let clean = strip_win_prefix(&dir.to_string_lossy());
    let clean = clean.trim_end_matches(['\\', '/']).to_string();
    if clean.is_empty() || clean.contains('"') || clean.chars().any(|c| c.is_control()) {
        return None;
    }
    let looks_abs = clean.starts_with(r"\\")
        || (clean.len() >= 3 && clean.as_bytes()[0].is_ascii_alphabetic() && clean.as_bytes()[1] == b':' && (clean.as_bytes()[2] == b'\\' || clean.as_bytes()[2] == b'/'))
        || dir.is_absolute();
    if !looks_abs {
        return None;
    }
    let norm = normalize_install_dir(&clean);
    if let Some(tmp) = temp_dir {
        let tn = normalize_install_dir(&tmp.to_string_lossy());
        if !tn.is_empty() && (norm == tn || norm.starts_with(&format!("{tn}\\"))) {
            return None;
        }
    }
    if let Some(lad) = local_app_data {
        let data_dir = normalize_install_dir(&lad.join("Strata").to_string_lossy());
        if !data_dir.is_empty() && norm == data_dir {
            return None;
        }
    }
    Some(clean)
}

/// Build the NSIS updater arguments: `/P /R /UPDATE`, plus an optional unquoted `/D=<dir>` (must be the final NSIS argument).
pub(crate) fn installer_cli_args(target_dir: Option<&Path>, temp_dir: Option<&Path>, local_app_data: Option<&Path>) -> (Vec<&'static str>, Option<String>) {
    let base = vec!["/P", "/R", "/UPDATE"];
    let d_arg = target_dir
        .and_then(|d| safe_installer_target_dir(d, temp_dir, local_app_data))
        .map(|clean| format!("/D={clean}"));
    (base, d_arg)
}

/// Directory containing the currently running desktop executable.
fn running_install_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let parent = exe.parent()?;
    Some(PathBuf::from(strip_win_prefix(&parent.to_string_lossy())))
}

/// Windows: read the registered install folder from HKCU (`Software\Strata\Strata` or `Uninstall\Strata\InstallLocation`).
#[cfg(windows)]
fn registered_install_dir() -> Option<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    const HKEY_CURRENT_USER: isize = 0x8000_0001_u32 as isize;
    const RRF_RT_REG_SZ: u32 = 0x0000_0002;
    const RRF_RT_REG_EXPAND_SZ: u32 = 0x0000_0004;

    #[link(name = "advapi32")]
    extern "system" {
        fn RegGetValueW(
            hkey: isize,
            subkey: *const u16,
            value: *const u16,
            flags: u32,
            pdw_type: *mut u32,
            pv_data: *mut u16,
            pcb_data: *mut u32,
        ) -> i32;
    }

    fn read_hkcu_str(subkey: &str, value_name: Option<&str>) -> Option<String> {
        let wide = |s: &str| std::ffi::OsStr::new(s).encode_wide().chain(std::iter::once(0)).collect::<Vec<u16>>();
        let sk = wide(subkey);
        let vn = value_name.map(wide);
        let vn_ptr = vn.as_ref().map_or(std::ptr::null(), |v| v.as_ptr());
        let mut bytes: u32 = 0;
        let flags = RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ;
        let rc = unsafe { RegGetValueW(HKEY_CURRENT_USER, sk.as_ptr(), vn_ptr, flags, std::ptr::null_mut(), std::ptr::null_mut(), &mut bytes) };
        if rc != 0 || bytes < 2 {
            return None;
        }
        let mut buf = vec![0u16; (bytes as usize + 1) / 2];
        let rc2 = unsafe { RegGetValueW(HKEY_CURRENT_USER, sk.as_ptr(), vn_ptr, flags, std::ptr::null_mut(), buf.as_mut_ptr(), &mut bytes) };
        if rc2 != 0 {
            return None;
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        let s = OsString::from_wide(&buf[..len]).to_string_lossy().trim().trim_matches('"').to_string();
        if s.is_empty() { None } else { Some(s) }
    }

    read_hkcu_str(r"Software\Strata\Strata", None)
        .or_else(|| read_hkcu_str(r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Strata", Some("InstallLocation")))
        .map(|s| PathBuf::from(strip_win_prefix(&s)))
}

#[cfg(not(windows))]
fn registered_install_dir() -> Option<PathBuf> {
    None
}

fn install_location_fields() -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    let running = running_install_dir();
    let registered = registered_install_dir();
    let mismatch = install_path_mismatch(running.as_deref(), registered.as_deref());
    if let Some(ref r) = running {
        map.insert("installDir".into(), serde_json::json!(r.to_string_lossy()));
    }
    if let Some(ref reg) = registered {
        map.insert("registeredInstallDir".into(), serde_json::json!(reg.to_string_lossy()));
    }
    map.insert("installPathMismatch".into(), serde_json::json!(mismatch));
    let custom_env = custom_profile_env_vars();
    if !custom_env.is_empty() {
        map.insert("customProfileEnv".into(), serde_json::json!(custom_env));
    }
    map
}

#[derive(Clone, Default)]
struct Progress {
    state: &'static str,
    downloaded: u64,
    total: u64,
    version: Option<String>,
    error: Option<String>,
}

impl Progress {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "state": if self.state.is_empty() { "idle" } else { self.state },
            "downloaded": self.downloaded, "total": self.total, "version": self.version, "error": self.error,
        })
    }
}

/// What the host needs to find the bridge.
pub struct Bridge {
    pub url: String,
    pub token: String,
}

fn endpoint_override() -> Option<String> {
    let url = std::env::var(ENDPOINT_ENV).ok()?;
    if url.starts_with("https://") || url.starts_with("http://127.0.0.1:") || url.starts_with("http://localhost:") {
        Some(url)
    } else {
        None
    }
}

fn updater_for(app: &AppHandle) -> Result<tauri_plugin_updater::Updater, String> {
    match endpoint_override() {
        Some(url) => {
            let url: tauri::Url = url.parse().map_err(|e| format!("bad update endpoint: {e}"))?;
            app.updater_builder().endpoints(vec![url]).map_err(|e| e.to_string())?.build().map_err(|e| e.to_string())
        }
        None => app.updater().map_err(|e| e.to_string()),
    }
}

async fn check(app: &AppHandle) -> Result<serde_json::Value, String> {
    let updater = updater_for(app)?;
    let mut out = match updater.check().await.map_err(|e| e.to_string())? {
        Some(update) => serde_json::json!({
            "available": true, "version": update.version, "current": update.current_version, "notes": update.body,
        }),
        None => serde_json::json!({ "available": false, "current": app.package_info().version.to_string() }),
    };
    if let Some(obj) = out.as_object_mut() {
        obj.extend(install_location_fields());
    }
    Ok(out)
}

async fn install(app: AppHandle, progress: Arc<Mutex<Progress>>) {
    let fail = |message: String| {
        let mut p = progress.lock().unwrap();
        p.state = "error";
        p.error = Some(message);
    };
    let updater = match updater_for(&app) {
        Ok(u) => u,
        Err(e) => return fail(e),
    };
    let update = match updater.check().await {
        Ok(Some(update)) => update,
        Ok(None) => return fail("There is no newer version of the Strata app.".into()),
        Err(e) => return fail(e.to_string()),
    };
    {
        let mut p = progress.lock().unwrap();
        *p = Progress { state: "downloading", version: Some(update.version.clone()), ..Default::default() };
    }
    let on_chunk = {
        let progress = progress.clone();
        move |chunk: usize, total: Option<u64>| {
            let mut p = progress.lock().unwrap();
            p.downloaded += chunk as u64;
            p.total = total.unwrap_or(p.total);
        }
    };
    let on_finish = {
        let progress = progress.clone();
        move || progress.lock().unwrap().state = "installing"
    };
    // Tauri checks the installer's minisign signature when it downloads it. Installing is done here, not by the plugin's
    // download_and_install: on Windows this app lives in a kill-on-close Job Object, and an installer the plugin starts is in
    // that job, so it is ended the moment the app exits and the update never happens. `launch_installer` starts it outside
    // the job; the app then exits and the installer (passive: a progress window, no questions) replaces it and starts it again.
    let bytes = match update.download(on_chunk, on_finish).await {
        Ok(bytes) => bytes,
        Err(e) => return fail(e.to_string()),
    };
    progress.lock().unwrap().state = "installing";
    match launch_installer(&update.version, &bytes) {
        Ok(()) => {
            progress.lock().unwrap().state = "done";
            app.exit(0);
        }
        Err(first) => match update.install(&bytes) {
            Ok(()) => {
                progress.lock().unwrap().state = "done";
                app.restart();
            }
            Err(e) => fail(format!("{first}; {e}")),
        },
    }
}

/// Windows: write the verified installer to a temp folder and start it detached and outside the app's Job Object.
/// Passes `/D=<running_install_dir>` as the final unquoted NSIS argument (#6573) so a manually relocated installation
/// updates in place and refreshes Windows's registered install path instead of updating a stale old folder.
#[cfg(windows)]
fn launch_installer(version: &str, bytes: &[u8]) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    let temp = std::env::temp_dir();
    let dir = temp.join(format!("strata-app-update-{version}"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let installer = dir.join("installer.exe");
    std::fs::write(&installer, bytes).map_err(|e| e.to_string())?;
    let running = running_install_dir();
    let local_app_data = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    let (base_args, d_arg) = installer_cli_args(running.as_deref(), Some(&temp), local_app_data.as_deref());
    let mut cmd = std::process::Command::new(&installer);
    cmd.args(&base_args);
    if let Some(d) = d_arg {
        // NSIS requires `/D=...` to be the final unquoted argument even when the path contains spaces.
        cmd.raw_arg(d);
    }
    cmd.creation_flags(CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("could not start the installer outside the app: {e}"))
}

/// Elsewhere the plugin's own install is used (the installer replaces the app in place).
#[cfg(not(windows))]
fn launch_installer(_version: &str, _bytes: &[u8]) -> Result<(), String> {
    Err("not Windows".into())
}

fn respond(stream: &mut TcpStream, status: &str, body: &serde_json::Value) {
    let text = body.to_string();
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
            text.len()
        )
        .as_bytes(),
    );
}

fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The loopback bridge is served on its own thread. Window construction is an
/// OS UI operation, so wait for Tauri's main thread to finish it before
/// answering the host. In particular, WebView2 and macOS reject window work
/// from this listener thread.
fn open_helper_window_on_main_thread(app: &AppHandle, host_port: u16, session: &str, helper: &str, name: &str) -> Result<(), String> {
    let (sent, received) = mpsc::sync_channel(1);
    let app_for_main = app.clone();
    let session = session.to_owned();
    let helper = helper.to_owned();
    let name = name.to_owned();
    app.run_on_main_thread(move || {
        let result = crate::open_helper_window(&app_for_main, host_port, &session, &helper, &name);
        let _ = sent.send(result);
    }).map_err(|e| format!("could not schedule the helper window: {e}"))?;
    received.recv_timeout(Duration::from_secs(5))
        .map_err(|_| "the helper window did not open on the UI thread in time".to_string())?
}

fn handle(mut stream: TcpStream, app: &AppHandle, token: &str, host_port: u16, progress: &Arc<Mutex<Progress>>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") && buf.len() < 8192 {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let head = String::from_utf8_lossy(&buf).to_string();
    let mut lines = head.lines();
    let request = lines.next().unwrap_or_default().to_string();
    let supplied = lines
        .find_map(|l| l.strip_prefix("authorization: ").or_else(|| l.strip_prefix("Authorization: ")))
        .map(|v| v.trim().trim_start_matches("Bearer ").to_string())
        .unwrap_or_default();
    if !same(&supplied, token) {
        return respond(&mut stream, "401 Unauthorized", &serde_json::json!({ "error": "not allowed" }));
    }
    let mut parts = request.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    match (method, path) {
        ("GET", "/app-update/check") => match tauri::async_runtime::block_on(check(app)) {
            Ok(value) => respond(&mut stream, "200 OK", &value),
            Err(e) => respond(&mut stream, "502 Bad Gateway", &serde_json::json!({ "error": e })),
        },
        ("POST", "/app-update/install") => {
            {
                let mut p = progress.lock().unwrap();
                if p.state == "downloading" || p.state == "installing" {
                    return respond(&mut stream, "409 Conflict", &serde_json::json!({ "error": "An app update is already running." }));
                }
                *p = Progress { state: "downloading", ..Default::default() };
            }
            tauri::async_runtime::spawn(install(app.clone(), progress.clone()));
            respond(&mut stream, "202 Accepted", &progress.lock().unwrap().json());
        }
        ("GET", "/app-update/status") => respond(&mut stream, "200 OK", &progress.lock().unwrap().json()),
        ("POST", p) if p.starts_with("/shell/helper?") => {
            let session = crate::shell::query_value(p, "sessionId").unwrap_or_default();
            let helper = crate::shell::query_value(p, "helperId").unwrap_or_default();
            let name = crate::shell::clean_text(&crate::shell::query_value(p, "name").unwrap_or_default(), 80);
            match open_helper_window_on_main_thread(app, host_port, &session, &helper, &name) {
                Ok(()) => respond(&mut stream, "200 OK", &serde_json::json!({ "opened": true })),
                Err(e) => respond(&mut stream, "400 Bad Request", &serde_json::json!({ "error": e })),
            }
        }
        // The host asks for a desktop notification (an approval is waiting): title and body are in the query, bounded and cleaned here.
        ("POST", p) if p.starts_with("/shell/notify") => {
            let title = crate::shell::clean_text(&crate::shell::query_value(p, "title").unwrap_or_default(), 120);
            let body = crate::shell::clean_text(&crate::shell::query_value(p, "body").unwrap_or_default(), 300);
            #[cfg(not(target_os = "linux"))]
            let shown = crate::shell::notify(app, &title, &body);
            #[cfg(target_os = "linux")]
            let shown = false;
            respond(&mut stream, "200 OK", &serde_json::json!({ "shown": shown }))
        }
        _ => respond(&mut stream, "404 Not Found", &serde_json::json!({ "error": "unknown route" })),
    }
}

/// Open the bridge on a free loopback port. Returns None if the listener cannot be bound (the app then simply has no
/// in-app update; the Terminal's `strata update --app` still works).
pub fn start(app: &AppHandle, token: String, host_port: u16) -> Option<Bridge> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).ok()?;
    let port = listener.local_addr().ok()?.port();
    let progress = Arc::new(Mutex::new(Progress::default()));
    let app = app.clone();
    let secret = token.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            handle(stream, &app, &secret, host_port, &progress);
        }
    });
    Some(Bridge { url: format!("http://127.0.0.1:{port}"), token })
}

/// Environment for the App Host: where the bridge is, which app version this is.
#[allow(dead_code)]
pub fn apply_env(cmd: &mut std::process::Command, app: &AppHandle, bridge: Option<&Bridge>) {
    cmd.env("STRATA_APP_VERSION", app.package_info().version.to_string());
    if let Some(bridge) = bridge {
        cmd.env("STRATA_SHELL_BRIDGE", &bridge.url).env("STRATA_SHELL_TOKEN", &bridge.token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_comparison_needs_every_byte() {
        assert!(same("abc123", "abc123"));
        assert!(!same("abc123", "abc124"));
        assert!(!same("abc123", "abc12"));
        assert!(!same("", "x"));
    }

    #[test]
    fn progress_reports_idle_until_something_starts() {
        let idle = Progress::default().json();
        assert_eq!(idle["state"], "idle");
        let p = Progress { state: "downloading", downloaded: 10, total: 100, version: Some("0.2.0".into()), error: None };
        assert_eq!(p.json()["downloaded"], 10);
        assert_eq!(p.json()["version"], "0.2.0");
    }

    #[test]
    fn only_https_or_loopback_endpoints_may_override_the_configured_one() {
        for (value, ok) in [
            ("https://updates.example.com/latest.json", true),
            ("http://127.0.0.1:19080/app/latest.json", true),
            ("http://localhost:19080/latest.json", true),
            ("http://example.com/latest.json", false),
            ("file:///etc/passwd", false),
        ] {
            std::env::set_var(ENDPOINT_ENV, value);
            assert_eq!(endpoint_override().is_some(), ok, "{value}");
        }
        std::env::remove_var(ENDPOINT_ENV);
        assert!(endpoint_override().is_none());
    }

    #[test]
    fn install_path_mismatch_normalizes_case_slashes_and_verbatim_prefix() {
        assert!(!install_path_mismatch(
            Some(Path::new(r"\\?\C:\Strata\App\")),
            Some(Path::new(r#""c:/strata/app""#))
        ));
        assert!(install_path_mismatch(
            Some(Path::new(r"C:\Strata\App")),
            Some(Path::new(r"D:\Strata"))
        ));
        assert!(!install_path_mismatch(Some(Path::new(r"C:\Strata\App")), None));
        assert!(!install_path_mismatch(None, Some(Path::new(r"D:\Strata"))));
    }

    #[test]
    fn installer_cli_args_appends_unquoted_d_flag_for_safe_relocated_paths() {
        let temp = Path::new(r"C:\Users\alice\AppData\Local\Temp");
        let lad = Path::new(r"C:\Users\alice\AppData\Local");
        let (base, d_arg) = installer_cli_args(Some(Path::new(r"C:\Strata\My App\")), Some(temp), Some(lad));
        assert_eq!(base, vec!["/P", "/R", "/UPDATE"]);
        assert_eq!(d_arg.as_deref(), Some(r"/D=C:\Strata\My App"));

        // Temp folder, %LOCALAPPDATA%\Strata (user data), and paths with quotes are rejected.
        assert_eq!(installer_cli_args(Some(Path::new(r"C:\Users\alice\AppData\Local\Temp\test")), Some(temp), Some(lad)).1, None);
        assert_eq!(installer_cli_args(Some(Path::new(r"C:\Users\alice\AppData\Local\Strata")), Some(temp), Some(lad)).1, None);
        assert_eq!(installer_cli_args(Some(Path::new(r#"C:\Bad"Dir"#)), Some(temp), Some(lad)).1, None);
    }

    #[test]
    fn custom_profile_env_vars_detects_active_launcher_overrides() {
        std::env::set_var("STRATA_DESKTOP_DATA_DIR", r"C:\Strata\Data");
        let vars = custom_profile_env_vars();
        assert!(vars.iter().any(|v| v == "STRATA_DESKTOP_DATA_DIR"), "{vars:?}");
        std::env::remove_var("STRATA_DESKTOP_DATA_DIR");
    }
}
