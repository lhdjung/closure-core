use closure_core::dfs_parallel;
/// Benchmark to demonstrate the performance improvement of stop_after with the fast path
use std::time::Instant;

fn main() {
    println!("Benchmarking stop_after performance improvements\n");
    println!("Parameters: mean=3.5, sd=0.5, n=52, scale=[1,5], errors=0.05\n");

    // Benchmark with stop_after = 1
    println!("Testing stop_after = 1 (uses sequential fast path):");
    let start = Instant::now();
    let result = dfs_parallel::<f64, i32>(3.5, 0.5, 52, 1, 5, 0.05, 0.05, None, Some(1));
    let duration_1 = start.elapsed();
    println!("  Duration: {:?}", duration_1);
    println!("  Samples found: {}", result.results.sample.len());
    println!();

    // Benchmark with stop_after = 10
    println!("Testing stop_after = 10 (uses sequential fast path):");
    let start = Instant::now();
    let result = dfs_parallel::<f64, i32>(3.5, 0.5, 52, 1, 5, 0.05, 0.05, None, Some(10));
    let duration_10 = start.elapsed();
    println!("  Duration: {:?}", duration_10);
    println!("  Samples found: {}", result.results.sample.len());
    println!();

    // Benchmark with stop_after = 100
    println!("Testing stop_after = 100 (uses sequential fast path):");
    let start = Instant::now();
    let result = dfs_parallel::<f64, i32>(3.5, 0.5, 52, 1, 5, 0.05, 0.05, None, Some(100));
    let duration_100 = start.elapsed();
    println!("  Duration: {:?}", duration_100);
    println!("  Samples found: {}", result.results.sample.len());
    println!();

    // Benchmark with stop_after = 1000 (uses parallel path)
    println!("Testing stop_after = 1000 (uses parallel path):");
    let start = Instant::now();
    let result = dfs_parallel::<f64, i32>(3.5, 0.5, 52, 1, 5, 0.05, 0.05, None, Some(1000));
    let duration_1000 = start.elapsed();
    println!("  Duration: {:?}", duration_1000);
    println!("  Samples found: {}", result.results.sample.len());
    println!();

    // Benchmark without limit (full search, parallel)
    println!("Testing without stop_after (full parallel search):");
    let start = Instant::now();
    let result = dfs_parallel::<f64, i32>(3.5, 0.5, 52, 1, 5, 0.05, 0.05, None, None);
    let duration_full = start.elapsed();
    println!("  Duration: {:?}", duration_full);
    println!("  Samples found: {}", result.results.sample.len());
    println!();

    println!("Summary:");
    println!(
        "  stop_after=1 is {}x faster than full search",
        duration_full.as_secs_f64() / duration_1.as_secs_f64()
    );
    println!(
        "  stop_after=10 is {}x faster than full search",
        duration_full.as_secs_f64() / duration_10.as_secs_f64()
    );
    println!(
        "  stop_after=100 is {}x faster than full search",
        duration_full.as_secs_f64() / duration_100.as_secs_f64()
    );
}
