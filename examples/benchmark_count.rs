use closure_core::{closure_count, closure_parallel};
use std::time::Instant;

struct Case {
    mean: f64,
    sd: f64,
    n: i32,
    scale_min: i32,
    scale_max: i32,
    re_mean: f64,
    re_sd: f64,
    label: &'static str,
}

fn main() {
    let cases = [
        Case { mean: 4.0, sd: 1.50, n: 10,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=10,  [1,7]" },
        Case { mean: 3.0, sd: 1.00, n: 12,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=12,  [1,5]" },
        Case { mean: 4.0, sd: 2.00, n: 15,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=15,  [1,7]" },
        Case { mean: 3.0, sd: 1.20, n: 20,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=20,  [1,5]" },
        Case { mean: 4.0, sd: 1.50, n: 20,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=20,  [1,7]" },
        Case { mean: 3.5, sd: 1.20, n: 20,  scale_min: 1, scale_max: 7, re_mean: 0.10, re_sd: 0.10, label: "n=20,  [1,7]*" },
        Case { mean: 3.0, sd: 1.00, n: 25,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=25,  [1,5]" },
        Case { mean: 4.0, sd: 1.80, n: 25,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=25,  [1,7]" },
        Case { mean: 3.5, sd: 1.00, n: 30,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=30,  [1,5]" },
        Case { mean: 4.0, sd: 1.50, n: 30,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=30,  [1,7]" },
        Case { mean: 3.0, sd: 1.00, n: 40,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=40,  [1,5]" },
        Case { mean: 4.0, sd: 1.50, n: 40,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=40,  [1,7]" },
        Case { mean: 4.0, sd: 1.50, n: 50,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=50,  [1,7]" },
        Case { mean: 3.0, sd: 1.00, n: 50,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=50,  [1,5]" },
        Case { mean: 4.0, sd: 1.80, n: 60,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=60,  [1,7]" },
        Case { mean: 3.0, sd: 1.00, n: 60,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=60,  [1,5]" },
        Case { mean: 4.0, sd: 1.50, n: 75,  scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=75,  [1,7]" },
        Case { mean: 3.0, sd: 1.00, n: 75,  scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=75,  [1,5]" },
        Case { mean: 4.0, sd: 1.50, n: 100, scale_min: 1, scale_max: 7, re_mean: 0.05, re_sd: 0.05, label: "n=100, [1,7]" },
        Case { mean: 3.0, sd: 1.00, n: 100, scale_min: 1, scale_max: 5, re_mean: 0.05, re_sd: 0.05, label: "n=100, [1,5]" },
    ];

    println!("closure_count() vs closure_parallel() — head-to-head benchmark");
    println!("All cases use items=1, parquet_config=None, stop_after=None");
    println!("* = wider rounding tolerance (0.1 instead of 0.05)");
    println!();
    println!(
        "{:<15} {:>10} {:>12} {:>12} {:>8}",
        "Case", "Count", "count()", "parallel()", "Speedup"
    );
    println!("{}", "-".repeat(61));

    for c in &cases {
        // closure_count
        let start = Instant::now();
        let count = closure_count(
            c.mean, c.sd, c.n, c.scale_min, c.scale_max, c.re_mean, c.re_sd,
        );
        let t_count = start.elapsed();

        // closure_parallel with the same overlapping inputs
        let start = Instant::now();
        let results = closure_parallel::<f64, i32>(
            c.mean, c.sd, c.n, c.scale_min, c.scale_max,
            c.re_mean, c.re_sd,
            1,    // items (must be 1 for CLOSURE)
            None, // parquet_config
            None, // stop_after
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
