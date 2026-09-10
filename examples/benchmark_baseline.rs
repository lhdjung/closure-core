//! Absolute timings for CLOSURE and SPRITE on the canonical case list.
//!
//! Unlike the other benchmarks, this one compares nothing by itself: it prints a
//! plain TSV so the same workload can be timed on two different revisions and
//! diffed. See `scripts/compare_versions.sh`.
#[path = "common/mod.rs"]
mod common;
use common::CASES;

use closure_core::{closure_parallel, sprite_parallel, RestrictionsOption};
use std::time::Instant;

/// Timings are medians over this many repetitions (SPRITE is randomized, so a
/// single run is noisy).
const REPS: usize = 3;
/// SPRITE searches at random and never terminates on its own, so it is always
/// asked for a fixed number of distributions.
const SPRITE_STOP_AFTER: usize = 200;

fn median_ms(mut f: impl FnMut() -> usize) -> (f64, usize) {
    let mut times = Vec::with_capacity(REPS);
    let mut count = 0;
    for _ in 0..REPS {
        let start = Instant::now();
        count = f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    (times[REPS / 2], count)
}

fn main() {
    println!("case\tclosure_n\tclosure_ms\tsprite_n\tsprite_ms");
    for c in CASES {
        let (closure_ms, closure_n) = median_ms(|| {
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
        let (sprite_ms, sprite_n) = median_ms(|| {
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
            "{}\t{}\t{:.3}\t{}\t{:.3}",
            c.label(),
            closure_n,
            closure_ms,
            sprite_n,
            sprite_ms
        );
    }
}
