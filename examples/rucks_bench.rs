//! Timing harness for the rucks instrumentation overhead.
//!
//! Usage: cargo run --release --example rucks_bench -- <file.mps>...
//! Prints one line per instance: wall-clock seconds, objective, and (with
//! the `rucks` feature) the loop counter for the solve. Build it twice — with
//! and without `--features rucks` — and diff the times to measure the
//! instrumentation overhead. `BENCH_NODES` (default 3000) caps the branch and
//! bound search so both builds walk the identical deterministic tree.

use lio_highs::{Model, SolvedModel};
use std::time::Instant;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    for path in &paths {
        let mut model = Model::default();
        model.read(path);
        model.set_option("threads", 1i32);
        model.set_option("output_flag", false);
        model.set_option("time_limit", 600.0);
        // Cap the search so both builds explore the identical deterministic
        // tree — otherwise a time limit makes each build stop somewhere else
        // and the wall times are not comparable.
        let nodes: i32 = std::env::var("BENCH_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3000);
        model.set_option("mip_max_nodes", nodes);

        #[cfg(feature = "rucks")]
        lio_highs::rucks::reset();

        let t = Instant::now();
        let solved: SolvedModel = model.solve();
        let secs = t.elapsed().as_secs_f64();

        let name = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());

        #[cfg(feature = "rucks")]
        let counters = format!(" loops={}", lio_highs::rucks::loop_count());
        #[cfg(not(feature = "rucks"))]
        let counters = String::new();

        // status() is not printed: the node-limit stop code has no mapping
        // in the bindings' HighsModelStatus and unwrapping it panics.
        println!("{name}\t{secs:.3}\tobj={}{counters}", solved.obj_val());
    }
}
