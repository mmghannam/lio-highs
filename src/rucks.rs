//! rucks work-counter API — exposes counters inserted into HiGHS C++ source.
//!
//! Only available when the `rucks` feature is enabled.

extern "C" {
    fn rucks_loop_count() -> i64;
    fn rucks_func_count() -> i64;
    fn rucks_total_count() -> i64;
    fn rucks_reset();
    fn rucks_report();
}

pub fn loop_count() -> i64 {
    unsafe { rucks_loop_count() }
}

pub fn func_count() -> i64 {
    unsafe { rucks_func_count() }
}

pub fn total_count() -> i64 {
    unsafe { rucks_total_count() }
}

pub fn reset() {
    unsafe { rucks_reset() }
}

pub fn report() {
    unsafe { rucks_report() }
}
