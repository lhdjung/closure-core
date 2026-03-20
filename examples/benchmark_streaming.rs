#[path = "common/mod.rs"]
mod common;
use common::CASES;

use closure_core::{closure_parallel, closure_parallel_streaming, StreamingConfig};
use std::time::Instant;

fn main() {
    let out_dir = std::env::temp_dir().join("closure_benchmark_streaming");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out_path = format!("{}/", out_dir.to_string_lossy());

    println!("closure_parallel() vs closure_parallel_streaming() — head-to-head benchmark");
    println!("All cases use items=1, batch_size=1000, stop_after=None");
    println!("* = wider rounding tolerance (0.1 instead of 0.05)");
    println!();
    println!(
        "{:<15} {:>10} {:>12} {:>12} {:>10}",
        "Case", "Count", "parallel()", "streaming()", "Overhead"
    );
    println!("{}", "-".repeat(65));

    for c in CASES {
        let start = Instant::now();
        let results = closure_parallel::<f64, i32>(
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
        .unwrap();
        let t_parallel = start.elapsed();
        let count = results.results.sample.len();

        let config = StreamingConfig {
            file_path: out_path.clone(),
            batch_size: 1000,
            show_progress: false,
        };
        let start = Instant::now();
        closure_parallel_streaming::<f64, i32>(
            c.mean,
            c.sd,
            c.n,
            c.scale_min,
            c.scale_max,
            c.re_mean,
            c.re_sd,
            1,
            config,
            None,
        )
        .unwrap();
        let t_streaming = start.elapsed();

        let overhead = (t_streaming.as_secs_f64() / t_parallel.as_secs_f64() - 1.0) * 100.0;

        println!(
            "{:<15} {:>10} {:>9.3} ms {:>9.3} ms {:>+9.1}%",
            c.label,
            count,
            t_parallel.as_secs_f64() * 1000.0,
            t_streaming.as_secs_f64() * 1000.0,
            overhead,
        );
    }

    std::fs::remove_dir_all(&out_dir).ok();
}
