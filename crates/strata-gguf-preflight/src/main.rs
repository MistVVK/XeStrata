use std::env;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: strata-gguf-preflight <shard.gguf>");
        process::exit(2);
    }
    match strata_gguf_preflight::inspect_metadata(Path::new(&args[0])) {
        Ok(report) => println!("{}", report.to_json()),
        Err(err) => {
            eprintln!("{err}");
            process::exit(err.code);
        }
    }
}
