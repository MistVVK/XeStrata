//! Who answers on a port (#6278).
//!
//! The shell starts its own App Host and then waits for `/api/health`. The Terminal's model server (`strata launch`,
//! default port 8080) answers `/api/health` with 200 too, so a port that the Terminal's server holds - or that an
//! earlier app session had and the server took over - made the shell take the server's web page for its own host:
//! the window showed another product's page that asks for an API key. The two answers differ: the App Host says
//! `"uptimeSec"`, the model server says `"service": "strata"`. The shell checks which one it has.

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Answer {
    /// The App Host (this app's own server).
    AppHost,
    /// The Strata model server started from the Terminal (`strata launch` / `strata serve`).
    ModelServer,
    /// Anything else, or no good answer.
    Other,
}

fn ok_status(status_line: &str) -> bool {
    status_line.split_whitespace().nth(1).map(|code| code.starts_with('2')).unwrap_or(false)
}

/// True when `"key"` is followed (after optional spaces and a colon) by the quoted `value`.
fn json_text_member(body: &str, key: &str, value: &str) -> bool {
    let needle = format!("\"{key}\"");
    let mut rest = body;
    while let Some(i) = rest.find(&needle) {
        let after = rest[i + needle.len()..].trim_start();
        if let Some(after) = after.strip_prefix(':') {
            if after.trim_start().strip_prefix('"').map(|t| t.starts_with(value) && t[value.len()..].starts_with('"')).unwrap_or(false) {
                return true;
            }
        }
        rest = &rest[i + needle.len()..];
    }
    false
}

/// Classify the answer to `GET /api/health` (the status line and the body as read).
pub fn classify_health(status_line: &str, body: &str) -> Answer {
    if !ok_status(status_line) {
        return Answer::Other;
    }
    if json_text_member(body, "service", "strata") {
        return Answer::ModelServer;
    }
    if body.contains("\"uptimeSec\"") {
        return Answer::AppHost;
    }
    Answer::Other
}

/// The notice the host gets when the port it wanted was taken by the Terminal's model server: "busy:<wanted>:<used>:terminal".
pub fn terminal_notice(wanted: u16, used: u16) -> String {
    format!("busy:{wanted}:{used}:terminal")
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOST: &str = r#"{"ok":true,"uptimeSec":3,"backend":"none","engine":{"state":"idle"},"clients":0}"#;
    const SERVER: &str = r#"{"status": "ok", "max_context": 65536, "model": "swift-1.5-iq3_xxs", "images": false, "api_key": true, "loaded": true, "service": "strata"}"#;

    #[test]
    fn the_app_host_is_known_by_its_uptime() {
        assert_eq!(classify_health("HTTP/1.1 200 OK", HOST), Answer::AppHost);
    }

    #[test]
    fn the_terminals_model_server_is_not_taken_for_the_host() {
        assert_eq!(classify_health("HTTP/1.1 200 OK", SERVER), Answer::ModelServer);
        // compact JSON too
        assert_eq!(classify_health("HTTP/1.1 200 OK", r#"{"service":"strata","status":"ok"}"#), Answer::ModelServer);
    }

    #[test]
    fn a_failed_or_foreign_answer_is_neither() {
        assert_eq!(classify_health("HTTP/1.1 503 Service Unavailable", HOST), Answer::Other);
        assert_eq!(classify_health("HTTP/1.1 200 OK", "<html>hello</html>"), Answer::Other);
        assert_eq!(classify_health("", ""), Answer::Other);
        assert_eq!(classify_health("HTTP/1.1 200 OK", r#"{"service":"strata-other"}"#), Answer::Other);
    }

    #[test]
    fn the_terminal_notice_names_both_ports() {
        assert_eq!(terminal_notice(8080, 51234), "busy:8080:51234:terminal");
    }
}
