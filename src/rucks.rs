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
//! Everything above describes the default `rucks` feature: a single
//! reproducible bucket total. The `rucks-pf` feature swaps HiGHS into the
//! per-function attribution mode instead — [`rucks_pf`] — where every loop
//! increment is attributed to the function that contains it and the bucket
//! total [`loop_count`] reads 0. See that submodule for its own caveats
//! (sources-only, not deferred, calling-thread reads).
//!
//! Only available when the `rucks` feature is enabled (and `rucks-pf`, which
//! implies it, for per-function attribution).

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

/// Per-function loop attribution for HiGHS (`rucks-pf` feature).
///
/// With the `rucks-pf` feature, `rucks_instrument` runs with `PER_FUNCTION`
/// (see HiGHS/highs/CMakeLists.txt): every loop increment lands in a
/// per-function table instead of the single bucket total, keyed by the
/// function that contains the loop as `"<filestem>::<funcname>"` (e.g.
/// `"HighsMipSolverData::trivialHeuristics"`). [`rucks_pf::counts`] reads
/// that table so loop iterations can be attributed to specific HiGHS
/// functions.
///
/// Mode-specific caveats that the loops-only build does not have:
///
/// * **`rucks::loop_count()` reads 0.** Loops are routed into the
///   per-function table, not the bucket total. Use [`rucks_pf::loop_count`]
///   (this module) for the sum, which equals `sum(counts())`.
/// * **Sources only.** The rucks CMake module cannot yet compose
///   `PER_FUNCTION` with `HEADERS` (two `rucks inst` runs would build
///   disjoint function-index maps and the second would clobber the first's
///   runtime), so the header templates — HighsHashTree, the red-black tree,
///   the matrix slices, pdqsort — are not instrumented in this mode. Their
///   iterations are still counted, attributed to the `.cpp` function that
///   instantiates them.
/// * **Not deferred.** Per-function counting increments the table directly
///   on every iteration (the deferred per-function accumulator does not
///   exist yet), so the counters are not free like the loops-only build's —
///   they are in the +50–100% wall-clock band, not +3%.
/// * **Counts are read from the calling thread's table.** There is no
///   aggregate per-function accessor in the rucks runtime (only the
///   printed report and the calling thread's pointer), so
///   [`rucks_pf::counts`] and [`rucks_pf::counters`] see only what the
///   current thread counted, and [`rucks_pf::loop_count`] sums every thread
///   in the process. Set HiGHS's `threads` option to 1 — the reproducible
///   configuration anyway — and the whole solve runs on this thread, so
///   [`rucks_pf::counts`] is complete and matches
///   [`rucks_pf::loop_count`]. In a multithreaded host process (tests, a
///   solver service) other threads' counted work lands in
///   [`rucks_pf::loop_count`] but not [`rucks_pf::counts`]; pair
///   [`rucks_pf::counts`] with [`rucks_pf::thread_loop_count`] in that case.
/// * **Reset after setup, not before it.** Every HiGHS call is instrumented,
///   including `Model::default()`/`read()`/`set_option()` — they run real
///   HiGHS C++ and increment the counters. `rucks::reset()` (the base
///   module) does **not** clear the pf table, so to measure only the solve
///   you must call [`rucks_pf::reset`] after configuring the model and
///   before `solve()`. On `chip.mps` this is the difference between 1262
///   (setup + solve, if you forget) and 1096 (solve only) — on bigger
///   instances the setup pollution is a large share of the count.
///
/// At `threads=1` the per-function counts are bit-identical across separate
/// processes and across repeated in-process solves with
/// [`rucks_pf::reset`] between (verified on `chip.mps` to the last
/// iteration of the last function).
///
/// Only available when the `rucks-pf` feature is enabled (which implies
/// `rucks`).
#[cfg(feature = "rucks-pf")]
pub mod rucks_pf {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int};

    extern "C" {
        fn rucks_pf_loop_count_highs() -> u64;
        fn rucks_pf_loop_count_thread_highs() -> u64;
        fn rucks_pf_reset_highs();
        fn rucks_pf_report_highs();
        fn rucks_pf_num_funcs_highs() -> c_int;
        fn rucks_pf_name_highs(idx: c_int) -> *const c_char;
        fn rucks_pf_counters_ptr_highs() -> *mut u64;
    }

    /// Number of tracked functions (length of the [`counters`] table).
    pub fn num_funcs() -> usize {
        unsafe { rucks_pf_num_funcs_highs() }.max(0) as usize
    }

    /// The attributed name for function slot `idx`, e.g.
    /// `"HighsMipSolverData::trivialHeuristics"`, or `None` for an
    /// out-of-range index. Names live in the rucks runtime for the life of
    /// the process, so the returned `&str` is `'static`.
    pub fn name(idx: usize) -> Option<&'static str> {
        let p = unsafe { rucks_pf_name_highs(idx as c_int) };
        if p.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(p) }.to_str().ok()
    }

    /// Loop iterations since the last [`reset`], summed over every thread in
    /// the process. In a single-threaded HiGHS run (`threads=1`) this equals
    /// the sum of [`counts`]; in a multithreaded host process (e.g. a test
    /// harness that solves on several threads at once) it also includes any
    /// other thread's counted work — use [`thread_loop_count`] for the
    /// calling thread alone.
    pub fn loop_count() -> u64 {
        unsafe { rucks_pf_loop_count_highs() }
    }

    /// The calling thread's own loop total — the sum of what [`counts`]
    /// returns, no matter how many other threads are counting in the same
    /// process. This is the right check against [`counts`] in a
    /// multithreaded host.
    pub fn thread_loop_count() -> u64 {
        unsafe { rucks_pf_loop_count_thread_highs() }
    }

    /// Zero the per-function table on all threads (see the base module's
    /// [`reset`](super::reset) for the bucket counters).
    pub fn reset() {
        unsafe { rucks_pf_reset_highs() }
    }

    /// Print every nonzero function's `RUCKS_PFLOOPS_<name>=<count>` line,
    /// aggregated over all threads, plus the grand total, to stderr.
    pub fn report() {
        unsafe { rucks_pf_report_highs() }
    }

    /// The calling thread's per-function counter array, indexed by function.
    /// Length [`num_funcs`]; pair with [`name`] to interpret the indices.
    /// Only reflects work done on the calling thread — complete when HiGHS
    /// runs at `threads=1`. See the module docs.
    pub fn counters() -> &'static [u64] {
        let n = num_funcs();
        if n == 0 {
            return &[];
        }
        let p = unsafe { rucks_pf_counters_ptr_highs() };
        if p.is_null() {
            return &[];
        }
        // rucks_pf_counters_ptr_highs() registers the calling thread and
        // returns a pointer into its TLS block, which outlives this call.
        unsafe { std::slice::from_raw_parts(p, n) }
    }

    /// Per-function loop iterations as `(name, count)` pairs, in function
    /// index order, skipping functions that ran no counted loop. Sums to
    /// [`loop_count`] when every counted loop ran on the calling thread
    /// (i.e. at `threads=1`).
    pub fn counts() -> Vec<(&'static str, u64)> {
        counters()
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if c == 0 {
                    None
                } else {
                    Some((name(i).unwrap_or("<unnamed>"), c))
                }
            })
            .collect()
    }
}
