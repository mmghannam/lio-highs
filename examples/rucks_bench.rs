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

        // rucks::reset() clears the base bucket counters; in pf mode the loop
        // counts live in the per-function table, so that must be reset instead
        // — otherwise Model::default()/read()/set_option above get counted too.
        #[cfg(feature = "rucks-pf")]
        lio_highs::rucks::rucks_pf::reset();
        #[cfg(all(feature = "rucks", not(feature = "rucks-pf")))]
        lio_highs::rucks::reset();

        let t = Instant::now();
        let solved: SolvedModel = model.solve();
        let secs = t.elapsed().as_secs_f64();

        let name = std::path::Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());

        // With rucks-pf the base bucket total reads 0 (loops go into the
        // per-function table), so the summary line shows the pf total instead.
        #[cfg(feature = "rucks-pf")]
        let counters = format!(" loops={}", lio_highs::rucks::rucks_pf::loop_count());
        #[cfg(all(feature = "rucks", not(feature = "rucks-pf")))]
        let counters = format!(" loops={}", lio_highs::rucks::loop_count());
        #[cfg(not(feature = "rucks"))]
        let counters = String::new();

        // status() is not printed: the node-limit stop code has no mapping
        // in the bindings' HighsModelStatus and unwrapping it panics.
        println!("{name}\t{secs:.3}\tobj={}{counters}", solved.obj_val());

        // Per-function attribution (--features rucks-pf): dump every tracked
        // HiGHS function's loop iterations, largest first.
        #[cfg(feature = "rucks-pf")]
        {
            let mut counts = lio_highs::rucks::rucks_pf::counts();
            counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (name, c) in &counts {
                println!("\tRUCKS_PFLOOPS_{name}={c}");
            }
            let total: u64 = counts.iter().map(|(_, c)| c).sum();
            println!("\tRUCKS_LOOPS_highs={total} ({} functions)", counts.len());
        }
    }
}
