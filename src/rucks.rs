//! rucks work-counter API — exposes counters inserted into HiGHS C++ source.
//!
//! HiGHS is instrumented under bucket "highs" (see HiGHS/highs/CMakeLists.txt
//! `LIO_RUCKS` block). The bucket runtime tracks per-thread TLS counters and
//! aggregates them on demand; the rucks_core registry shares one report/reset
//! API across all buckets in the binary.
//!
//! Only available when the `rucks` feature is enabled.

extern "C" {
    fn rucks_loops_highs() -> u64;
    fn rucks_funcs_highs() -> u64;
    fn rucks_total_highs() -> u64;
    fn rucks_reset_highs();
    fn rucks_report();
}

pub fn loop_count() -> u64 {
    unsafe { rucks_loops_highs() }
}

pub fn func_count() -> u64 {
    unsafe { rucks_funcs_highs() }
}

pub fn total_count() -> u64 {
    unsafe { rucks_total_highs() }
}

pub fn reset() {
    unsafe { rucks_reset_highs() }
}

pub fn report() {
    unsafe { rucks_report() }
}
