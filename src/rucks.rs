//! rucks work-counter API — exposes loop counters inserted into HiGHS C++ source.
//!
//! HiGHS is instrumented under bucket "highs" (see HiGHS/highs/CMakeLists.txt
//! `LIO_RUCKS` block). The bucket runtime tracks per-thread TLS counters and
//! aggregates them on demand; the rucks_core registry shares one report/reset
//! API across all buckets in the binary.
//!
//! Loops only — `rucks_instrument` is called without `FUNCS`, so no function
//! counter exists to read. On a C++ codebase most function entries are
//! accessors and constructors that cost more to count than to run, and
//! measured over node-limited MIPLIB solves the HiGHS function count is not
//! reproducible run to run (it drifts by a few hundred out of ~10^8,
//! presumably via timer-polled paths) while the loop count is bit-identical.
//!
//! Headers are instrumented alongside the sources, so the templates and
//! inline code HiGHS keeps in them — HighsHashTree, the red-black tree, the
//! matrix slices, pdqsort — count too. They are a large share of the work:
//! it is 8% more counted iterations overall than a sources-only build, and
//! 39% more on gen-ip054.
//!
//! Loop counting is deferred: each instrumented function accumulates its
//! iterations in a local the compiler keeps in a register, and flushes once
//! on scope exit. That is what makes the counters nearly free — 3.1% wall
//! clock over node-limited MIPLIB solves, 0.042 ns per counted iteration.
//! `examples/rucks_bench.rs` reproduces the measurement.
//!
//! Only available when the `rucks` feature is enabled.

extern "C" {
    fn rucks_loops_highs() -> u64;
    fn rucks_reset_highs();
    fn rucks_report();
}

/// Loop iterations executed inside HiGHS since the last [`reset`], summed
/// over every thread.
pub fn loop_count() -> u64 {
    unsafe { rucks_loops_highs() }
}

/// Zero the HiGHS bucket's counters on all threads.
pub fn reset() {
    unsafe { rucks_reset_highs() }
}

/// Print every bucket's counters (HiGHS and any other instrumented library
/// linked into the same binary) to stderr.
pub fn report() {
    unsafe { rucks_report() }
}
