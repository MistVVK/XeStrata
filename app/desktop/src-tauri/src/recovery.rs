//! A page whose renderer dies is brought back, not left on the browser's error page (#5079).
//!
//! WebView2 (Windows) reports a dead renderer through its ProcessFailed event, and WKWebView (macOS) through
//! `webViewWebContentProcessDidTerminate`; until now the shell listened to neither, so the window kept the browser's own "This page has a
//! problem" page, nothing reached the log, and the person's Reload brought back the same chat that had just crashed. Linux does not need it:
//! there the window is its own process and the shell moves to browser mode when it dies (supervisor.rs).
//!
//! This file is only decisions, so it builds and tests everywhere; the platform hooks are in main.rs.
//!   - which failures are worth a reload (the page's own renderer; not a GPU process that restarts by itself, not a frame of the page);
//!   - how many automatic recoveries are allowed in a window of time (a renderer that dies again at once is not reloaded for ever);
//!   - where the reload goes: the same screen first, with the sidebar previews closed once; the home screen the second time.
//! The page reads `?recovery=<n>` in apps/host/src/lib/recovery.ts. The chats are the host's: nothing here touches them.

/// Automatic recoveries allowed inside `WINDOW_MS`; the next failure leaves the window as it is for the person to reload.
pub const MAX_AUTOMATIC: usize = 2;
/// The window of time the limit counts over.
pub const WINDOW_MS: u64 = 120_000;
/// Let the browser start a new renderer before the page is asked for again.
pub const RELOAD_DELAY_MS: u64 = 400;

/// WebView2's COREWEBVIEW2_PROCESS_FAILED_KIND values, by number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    BrowserExited,
    /// The renderer of the page itself.
    RendererExited,
    Unresponsive,
    /// The renderer of a frame inside the page (a website preview); the page itself is alive.
    FrameRendererExited,
    Gpu,
    Other(i32),
}

impl Failure {
    pub fn from_webview2(kind: i32) -> Failure {
        match kind {
            0 => Failure::BrowserExited,
            1 => Failure::RendererExited,
            2 => Failure::Unresponsive,
            3 => Failure::FrameRendererExited,
            6 => Failure::Gpu,
            other => Failure::Other(other),
        }
    }

    /// Only the page's own renderer dying leaves a window that has to be loaded again. A GPU process is restarted by the browser, a frame
    /// shows its own sad face, an unresponsive page may only be busy, and a dead browser process cannot navigate at all.
    pub fn needs_reload(self) -> bool {
        self == Failure::RendererExited
    }

    pub fn name(self) -> String {
        match self {
            Failure::BrowserExited => "browser process exited".into(),
            Failure::RendererExited => "renderer exited".into(),
            Failure::Unresponsive => "renderer unresponsive".into(),
            Failure::FrameRendererExited => "frame renderer exited".into(),
            Failure::Gpu => "gpu process exited".into(),
            Failure::Other(n) => format!("process failure kind {n}"),
        }
    }
}

/// WebView2's COREWEBVIEW2_PROCESS_FAILED_REASON, for the log.
pub fn reason_name(reason: i32) -> &'static str {
    match reason {
        0 => "unexpected",
        1 => "unresponsive",
        2 => "terminated",
        3 => "crashed",
        4 => "launch failed",
        5 => "out of memory",
        6 => "profile deleted",
        _ => "unknown",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Load the page again; `attempt` is 1 for the first recovery of the window, 2 for the second.
    Reload { attempt: usize },
    /// Too many in a short time: leave the window alone.
    Stop,
}

/// Counts recoveries over a sliding window of time. `now_ms` is any clock that only goes forward (the caller's uptime).
#[derive(Default, Debug)]
pub struct Guard {
    times: Vec<u64>,
}

impl Guard {
    pub const fn new() -> Guard {
        Guard { times: Vec::new() }
    }

    pub fn on_failure(&mut self, now_ms: u64) -> Action {
        self.times.retain(|t| now_ms.saturating_sub(*t) < WINDOW_MS);
        if self.times.len() >= MAX_AUTOMATIC {
            return Action::Stop;
        }
        self.times.push(now_ms);
        Action::Reload { attempt: self.times.len() }
    }
}

/// The address a recovery loads. The first goes back to the screen that died (its `#/...` part) with `?recovery=1`, which makes the page
/// start with the sidebar previews closed once; the second goes to the home screen, in case the screen itself is what crashes.
/// `failed` is the address the window was showing; it is used only when it is on `origin`.
pub fn recovery_url(origin: &str, attempt: usize, failed: Option<&str>) -> String {
    let origin = origin.trim_end_matches('/');
    let base = format!("{origin}/?recovery={}", attempt.clamp(1, 9));
    if attempt > 1 {
        return base;
    }
    let Some(failed) = failed else { return base };
    // on this origin: after it comes the end, a path, a query or a fragment (not another digit of a longer port)
    let rest = match failed.strip_prefix(origin) {
        Some(rest) if rest.is_empty() || rest.starts_with(['/', '?', '#']) => rest,
        _ => return base,
    };
    match rest.find('#') {
        Some(at) => {
            let hash = &rest[at..];
            // only what a page address holds: no spaces, quotes, angle brackets or control characters
            let safe = hash.len() < 2048 && hash.starts_with("#/") && hash.chars().all(|c| c.is_ascii_graphic() && !matches!(c, '"' | '<' | '>' | '\\' | '`' | '{' | '}' | '|' | '^'));
            if safe { format!("{base}{hash}") } else { base }
        }
        None => base,
    }
}

/// What the shell does about one failure report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Not a failure that leaves the window needing a reload: only logged.
    Ignore,
    /// A reload is due, to `url`.
    Reload { attempt: usize, url: String },
    /// Too many recent recoveries: the window is left as it is.
    Stop,
}

/// One failure report in, the decision out. The platform hooks (main.rs) only collect the facts and carry the decision out.
pub fn plan(guard: &mut Guard, now_ms: u64, origin: &str, failure: Failure, failed: Option<&str>) -> Plan {
    if !failure.needs_reload() {
        return Plan::Ignore;
    }
    match guard.on_failure(now_ms) {
        Action::Stop => Plan::Stop,
        Action::Reload { attempt } => Plan::Reload { attempt, url: recovery_url(origin, attempt, failed) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_pages_own_renderer_is_reloaded() {
        assert!(Failure::from_webview2(1).needs_reload());
        for kind in [0, 2, 3, 4, 5, 6, 7, 8, 9, 42] {
            assert!(!Failure::from_webview2(kind).needs_reload(), "kind {kind}");
        }
        assert_eq!(Failure::from_webview2(6), Failure::Gpu);
        assert_eq!(Failure::from_webview2(1).name(), "renderer exited");
        assert_eq!(Failure::from_webview2(42).name(), "process failure kind 42");
    }

    #[test]
    fn reasons_are_named_for_the_log() {
        assert_eq!(reason_name(5), "out of memory");
        assert_eq!(reason_name(3), "crashed");
        assert_eq!(reason_name(99), "unknown");
    }

    #[test]
    fn two_recoveries_in_two_minutes_then_the_window_is_left_alone() {
        let mut g = Guard::new();
        assert_eq!(g.on_failure(1_000), Action::Reload { attempt: 1 });
        assert_eq!(g.on_failure(5_000), Action::Reload { attempt: 2 });
        assert_eq!(g.on_failure(6_000), Action::Stop);
        assert_eq!(g.on_failure(100_000), Action::Stop);
        // the first has left the window: one more is allowed, and it is the second of the window
        assert_eq!(g.on_failure(1_000 + WINDOW_MS), Action::Reload { attempt: 2 });
        assert_eq!(g.on_failure(1_000 + WINDOW_MS + 1), Action::Stop);
    }

    #[test]
    fn a_renderer_that_dies_later_starts_over_at_the_first_recovery() {
        let mut g = Guard::new();
        assert_eq!(g.on_failure(0), Action::Reload { attempt: 1 });
        assert_eq!(g.on_failure(10), Action::Reload { attempt: 2 });
        assert_eq!(g.on_failure(WINDOW_MS + 100), Action::Reload { attempt: 1 });
    }

    #[test]
    fn a_stopped_loop_stays_stopped_until_the_window_has_passed() {
        let mut g = Guard::new();
        let mut reloads = 0;
        // a renderer that dies every second for ten minutes: never more than two reloads in any two-minute window
        let mut loaded = Vec::new();
        for second in 0..600u64 {
            if let Action::Reload { .. } = g.on_failure(second * 1_000) {
                reloads += 1;
                loaded.push(second);
            }
        }
        assert!(reloads <= 2 * (600 / (WINDOW_MS / 1_000) as usize + 1));
        for w in loaded.windows(3) {
            assert!(w[2] - w[0] >= WINDOW_MS / 1_000, "three reloads inside one window: {w:?}");
        }
    }

    #[test]
    fn the_first_recovery_keeps_the_screen_and_the_second_goes_home() {
        let o = "http://127.0.0.1:47831";
        assert_eq!(recovery_url(o, 1, Some("http://127.0.0.1:47831/#/chat/s_abc")), "http://127.0.0.1:47831/?recovery=1#/chat/s_abc");
        assert_eq!(recovery_url(o, 1, Some("http://127.0.0.1:47831/?recovery=1#/chat/s_abc")), "http://127.0.0.1:47831/?recovery=1#/chat/s_abc");
        assert_eq!(recovery_url(o, 2, Some("http://127.0.0.1:47831/#/chat/s_abc")), "http://127.0.0.1:47831/?recovery=2");
        assert_eq!(recovery_url(o, 1, None), "http://127.0.0.1:47831/?recovery=1");
        assert_eq!(recovery_url("http://127.0.0.1:47831/", 1, None), "http://127.0.0.1:47831/?recovery=1");
    }

    #[test]
    fn an_address_that_is_not_ours_or_not_plain_is_not_carried_over() {
        let o = "http://127.0.0.1:47831";
        for failed in [
            "https://example.com/#/chat/x",
            "http://127.0.0.1:47832/#/chat/x",
            "http://127.0.0.1:47831/#/chat/x y",
            "http://127.0.0.1:47831/#/chat/\"onload=",
            "http://127.0.0.1:47831/#/chat/<script>",
            "http://127.0.0.1:47831/#nothing",
            "http://127.0.0.1:47831/#/chat/\u{e9}",
            "chrome-error://chromewebdata/",
        ] {
            assert_eq!(recovery_url(o, 1, Some(failed)), "http://127.0.0.1:47831/?recovery=1", "{failed}");
        }
        let long = format!("http://127.0.0.1:47831/#/{}", "a".repeat(3000));
        assert_eq!(recovery_url(o, 1, Some(&long)), "http://127.0.0.1:47831/?recovery=1");
        // a look-alike host is not the origin
        assert_eq!(recovery_url(o, 1, Some("http://127.0.0.1:478310/#/chat/x")), "http://127.0.0.1:47831/?recovery=1");
    }

    #[test]
    fn a_failure_report_becomes_one_decision() {
        let o = "http://127.0.0.1:47831";
        let mut g = Guard::new();
        let chat = Some("http://127.0.0.1:47831/#/chat/s_1");
        // a GPU process or a frame going down never costs the person their page, and does not use up the allowance
        assert_eq!(plan(&mut g, 0, o, Failure::Gpu, chat), Plan::Ignore);
        assert_eq!(plan(&mut g, 1, o, Failure::FrameRendererExited, chat), Plan::Ignore);
        assert_eq!(plan(&mut g, 2, o, Failure::Unresponsive, chat), Plan::Ignore);
        assert_eq!(plan(&mut g, 3, o, Failure::BrowserExited, chat), Plan::Ignore);
        assert_eq!(plan(&mut g, 1_000, o, Failure::RendererExited, chat), Plan::Reload { attempt: 1, url: "http://127.0.0.1:47831/?recovery=1#/chat/s_1".into() });
        assert_eq!(plan(&mut g, 4_000, o, Failure::RendererExited, chat), Plan::Reload { attempt: 2, url: "http://127.0.0.1:47831/?recovery=2".into() });
        assert_eq!(plan(&mut g, 7_000, o, Failure::RendererExited, chat), Plan::Stop);
    }
}
