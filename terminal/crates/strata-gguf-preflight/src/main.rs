//! `strata-gguf-preflight --strata-protocol 1 inspect <shard.gguf>`: the native-helper contract of strata_control/native_helpers.py.
//! Exactly one JSON object on stdout: {"protocol":1,"ok":true,"result":{...}} or {"protocol":1,"ok":false,"error":{"code":N,"message":"..."}}.
//! Any other command line is a usage error (exit 2, nothing on stdout), so a caller of another protocol falls back instead of misreading.

use std::env;
use std::path::Path;
use std::process;

use strata_gguf_preflight::{inspect_metadata, json_escape};

const PROTOCOL: &str = "1";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 || args[0] != "--strata-protocol" || args[1] != PROTOCOL || args[2] != "inspect" {
        eprintln!("usage: strata-gguf-preflight --strata-protocol 1 inspect <shard.gguf>");
        process::exit(2);
    }
    match inspect_metadata(Path::new(&args[3])) {
        Ok(report) => println!("{{\"protocol\":1,\"ok\":true,\"result\":{}}}", report.to_json()),
        Err(err) => {
            let code = if (1..=255).contains(&err.code) { err.code } else { 5 };
            println!("{{\"protocol\":1,\"ok\":false,\"error\":{{\"code\":{},\"message\":{}}}}}", code, json_escape(&err.message));
        }
    }
}
