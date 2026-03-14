#[path = "common/mod.rs"]
mod common;
use common::CASES;

use closure_core::{closure_count, closure_parallel};
use std::time::Instant;

fn main() {
    println!("closure_count() vs closure_parallel() — head-to-head benchmark");
    println!("All cases use items=1, parquet_config=None, stop_after=None");
    println!("* = wider rounding tolerance (0.1 instead of 0.05)");
    println!();
    println!(
        "{:<15} {:>10} {:>12} {:>12} {:>8}",
        "Case", "Count", "count()", "parallel()", "Speedup"
    );
    println!("{}", "-".repeat(61));

    for c in CASES {
        let start = Instant::now();
        let count = closure_count(
            c.mean, c.sd, c.n, c.scale_min, c.scale_max, c.re_mean, c.re_sd,
        );
        let t_count = start.elapsed();

        let start = Instant::now();
        let results = closure_parallel::<f64, i32>(
            c.mean, c.sd, c.n, c.scale_min, c.scale_max,
            c.re_mean, c.re_sd, 1, None, None,
        )
        .unwrap();
        let t_parallel = start.elapsed();

        let enumerated = results.results.sample.len() as u64;
        assert_eq!(count, enumerated, "Mismatch on {}", c.label);

        let speedup = t_parallel.as_secs_f64() / t_count.as_secs_f64();

        println!(
            "{:<15} {:>10} {:>9.3} ms {:>9.3} ms {:>7.0}x",
            c.label,
            count,
            t_count.as_secs_f64() * 1000.0,
            t_parallel.as_secs_f64() * 1000.0,
            speedup,
        );
    }
}
