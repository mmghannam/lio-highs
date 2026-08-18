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
//! `highs/parallel/` is the one exception, excluded from instrumentation.
//! Its loops are the task scheduler's spin and work-stealing waits, and they
//! are wall-clock gated by construction: HighsTaskExecutor::random_steal_loop
//! and sync_stolen_task, HighsBinarySemaphore, HighsMutex and
//! HighsSplitDeque's wait-for-stolen-task all double their spin budget until
//! a microsecond deadline. Counting them makes the total a function of how
//! long the spinning happened to last rather than of the work done — on
//! markshare_4_0 at threads=4 they were 3.5% of the count and doubled its
//! run-to-run spread, while at threads=1 they contribute 13 iterations.
//!
//! Loop counting is deferred: each instrumented function accumulates its
//! iterations in a local the compiler keeps in a register, and flushes once
//! on scope exit. That is what makes the counters nearly free — 3.1% wall
//! clock over node-limited MIPLIB solves, 0.042 ns per counted iteration.
//! `examples/rucks_bench.rs` reproduces the measurement.
//!
//! Reproducible only at `threads=1`, which callers must set explicitly:
//! HiGHS's `threads` option defaults to 0, which HighsParallel.h resolves to
//! `(hardware_concurrency() + 1) / 2`, so an unconfigured solve is
//! multithreaded and its count moves both run to run and machine to machine.
//! With threads pinned to 1 the count is bit-identical across separate
//! processes and across repeated in-process solves with [`reset`] between —
//! verified on instances up to 1.4e9 counted iterations, and no counted loop
//! anywhere in HiGHS derives a trip count from a pointer address.
//!
//! Above one thread it is not reproducible and cannot be made so: HiGHS runs
//! the analytic centre in a task that races the root node and is cancelled by
//! its progress (HighsMipSolverData::startAnalyticCenterComputation), and
//! switches the clique table to a parallel merge only when
//! `num_threads() > 1`. The work itself differs run to run, so the counter
//! differs with it — on markshare_4_0 at threads=4, by about 0.03%.
//!
//! Only available when the `rucks` feature is enabled.

extern "C" {
    fn rucks_loops_highs() -> u64;
    fn rucks_reset_highs();
    fn rucks_report();
}

/// Loop iterations executed inside HiGHS since the last [`reset`], summed
/// over every thread.
///
/// Reproducible run to run only if HiGHS's `threads` option is set to 1
/// before solving; the default of 0 means "as many as the machine has". See
/// the module docs.
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
