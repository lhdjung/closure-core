//! Absolute timings and peak heap use for CLOSURE and SPRITE on the canonical
//! case list.
//!
//! Unlike the other benchmarks, this one compares nothing by itself: it prints a
//! plain TSV so the same workload can be timed on two different revisions and
//! diffed. See `scripts/compare_versions.sh`.
#[path = "common/mod.rs"]
mod common;
use common::CASES;

use closure_core::{closure_parallel, sprite_parallel, RestrictionsOption};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

/// The system allocator, plus a running total of live heap bytes and its peak.
/// Heap bytes rather than RSS: RSS only ever grows within a process, so it
/// can't be split between cases, and it includes memory the allocator holds
/// on to after it was freed.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            grow(new_size - layout.size());
        } else {
            LIVE.fetch_sub(layout.size() - new_size, Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Timings are medians over this many repetitions (SPRITE is randomized, so a
/// single run is noisy).
const REPS: usize = 3;
/// SPRITE searches at random and never terminates on its own, so it is always
/// asked for a fixed number of distributions.
const SPRITE_STOP_AFTER: usize = 200;

/// Median time in ms and median peak heap in KiB above what was live before
/// the call, plus the result count of the last run.
fn median_ms(mut f: impl FnMut() -> usize) -> (f64, f64, usize) {
    let mut times = Vec::with_capacity(REPS);
    let mut peaks = Vec::with_capacity(REPS);
    let mut count = 0;
    for _ in 0..REPS {
        let before = LIVE.load(Relaxed);
        PEAK.store(before, Relaxed);
        let start = Instant::now();
        count = f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        peaks.push((PEAK.load(Relaxed) - before) as f64 / 1024.0);
    }
    times.sort_by(f64::total_cmp);
    peaks.sort_by(f64::total_cmp);
    (times[REPS / 2], peaks[REPS / 2], count)
}

fn main() {
    println!("case\tclosure_n\tclosure_ms\tsprite_n\tsprite_ms\tclosure_kib\tsprite_kib");
    for c in CASES {
        let (closure_ms, closure_kib, closure_n) = median_ms(|| {
            closure_parallel::<f64, i32>(
                c.mean,
                c.sd,
                c.n,
                c.scale_min,
                c.scale_max,
                c.re_mean,
                c.re_sd,
                1,
                None,
                None,
            )
            .unwrap()
            .results
            .len()
        });
        let (sprite_ms, sprite_kib, sprite_n) = median_ms(|| {
            sprite_parallel::<f64, i32>(
                c.mean,
                c.sd,
                c.n,
                c.scale_min,
                c.scale_max,
                c.re_mean,
                c.re_sd,
                1,
                None,
                RestrictionsOption::Default,
                None,
                Some(SPRITE_STOP_AFTER),
            )
            .unwrap()
            .results
            .len()
        });
        println!(
            "{}\t{}\t{:.3}\t{}\t{:.3}\t{:.1}\t{:.1}",
            c.label(),
            closure_n,
            closure_ms,
            sprite_n,
            sprite_ms,
            closure_kib,
            sprite_kib
        );
    }
}
