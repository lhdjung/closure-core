//! CLOSURE: complete listing of original samples of underlying raw evidence
//! 
//! Crate closure-core implements the CLOSURE technique for efficiently reconstructing
//! all possible distributions of raw data from summary statistics. It is not
//! about the Rust feature called closure.
//! 
//! The only API users are likely to need is `dfs_parallel()`. This function applies
//! the lower-level `dfs_branch()` in parallel and writes results to disk (currently
//! into a CSV file, but this may change in the future.)
//! 
//! Most of the code was written by Claude 3.5, translating Python code by Nathanael Larigaldie.


use std::collections::VecDeque;
use rayon::prelude::*;
use std::time::Instant;
use std::fs::File;
use std::io;
use csv::WriterBuilder;
use indicatif::{ProgressBar, ProgressStyle};


/// Result type containing all valid CLOSURE combinations and execution metadata
#[derive(Debug)]
pub struct ClosureResult {
    pub combinations: Vec<Vec<i32>>,
    pub execution_time_secs: f64,
    pub initial_combinations_count: i32,
}

/// Struct to hold combinations of possible raw data during processing
#[derive(Clone)]
struct Combination {
    values: Vec<i32>,
    running_sum: f64,
    running_m2: f64,
}

/// Calculates the number of initial combinations that will be processed in parallel
#[inline]
fn count_initial_combinations(min_scale: i32, max_scale: i32) -> i32 {
    let range_size = max_scale - min_scale + 1;
    (range_size * (range_size + 1)) / 2
}

/// Collect all valid combinations from a starting point
#[inline]
fn dfs_branch(
    start_combination: Vec<i32>,
    running_sum_init: f64,
    running_m2_init: f64,
    n: usize,
    target_sum_upper: f64,
    target_sum_lower: f64,
    target_sd_upper: f64,
    target_sd_lower: f64,
    min_scale_sum: &[i32],
    max_scale_sum: &[i32],
    n_minus_1: usize,
    max_scale_plus_1: i32,
) -> Vec<Vec<i32>> {
    let mut stack = VecDeque::with_capacity(n * 2); // Preallocate with reasonable capacity
    let mut results = Vec::new();
    
    stack.push_back(Combination {
        values: start_combination,
        running_sum: running_sum_init,
        running_m2: running_m2_init,
    });
    
    while let Some(current) = stack.pop_back() {
        if current.values.len() >= n {
            let current_std = (current.running_m2 / n_minus_1 as f64).sqrt();
            if current_std >= target_sd_lower {
                results.push(current.values);
            }
            continue;
        }

        let n_left = n_minus_1 - current.values.len();
        let next_n = current.values.len() + 1;
        let last_value = *current.values.last().unwrap();
        let current_mean = current.running_sum / current.values.len() as f64;

        // Use iterator for better performance than range
        (last_value..max_scale_plus_1)
            .take_while(|&next_value| {
                let next_sum = current.running_sum + next_value as f64;
                let minmean = next_sum + min_scale_sum[n_left] as f64;
                minmean <= target_sum_upper
            })
            .filter(|&next_value| {
                let next_sum = current.running_sum + next_value as f64;
                let maxmean = next_sum + max_scale_sum[n_left] as f64;
                maxmean >= target_sum_lower
            })
            .for_each(|next_value| {
                let next_sum = current.running_sum + next_value as f64;
                let next_mean = next_sum / next_n as f64;
                let delta = next_value as f64 - current_mean;
                let delta2 = next_value as f64 - next_mean;
                let next_m2 = current.running_m2 + delta * delta2;
                
                let min_sd = (next_m2 / n_minus_1 as f64).sqrt();
                if min_sd <= target_sd_upper {
                    let mut new_values = current.values.clone();
                    new_values.push(next_value);
                    stack.push_back(Combination {
                        values: new_values,
                        running_sum: next_sum,
                        running_m2: next_m2,
                    });
                }
            });
    }

    results
}

/// Run CLOSURE across starting combinations and return results
pub fn dfs_parallel(
    min_scale: i32,
    max_scale: i32,
    n: usize,
    target_sum: f64,
    target_sd: f64,
    rounding_error_sums: f64,
    rounding_error_sds: f64,
) -> ClosureResult {
    let start_time = Instant::now();
    
    let initial_count = count_initial_combinations(min_scale, max_scale);
    
    let target_sum_upper = target_sum + rounding_error_sums;
    let target_sum_lower = target_sum - rounding_error_sums;
    let target_sd_upper = target_sd + rounding_error_sds;
    let target_sd_lower = target_sd - rounding_error_sds;
    
    // Precompute scale sums for optimization
    let min_scale_sum: Vec<i32> = (0..n).map(|x| min_scale * x as i32).collect();
    let max_scale_sum: Vec<i32> = (0..n).map(|x| max_scale * x as i32).collect();
    
    let n_minus_1 = n - 1;
    let max_scale_plus_1 = max_scale + 1;

    // Generate initial combinations - now using iterators
    let initial_combinations: Vec<_> = (min_scale..=max_scale)
        .flat_map(|i| {
            (i..=max_scale).map(move |j| {
                let initial_combination = vec![i, j];
                let running_sum = (i + j) as f64;
                let current_mean = running_sum / 2.0;
                let current_m2 = (i as f64 - current_mean).powi(2) + 
                                (j as f64 - current_mean).powi(2);
                (initial_combination, running_sum, current_m2)
            })
        })
        .collect();

    // Process combinations in parallel and collect results
    let combinations: Vec<Vec<i32>> = initial_combinations
        .par_iter()
        .flat_map(|(combo, running_sum, running_m2)| {
            dfs_branch(
                combo.clone(),
                *running_sum,
                *running_m2,
                n,
                target_sum_upper,
                target_sum_lower,
                target_sd_upper,
                target_sd_lower,
                &min_scale_sum,
                &max_scale_sum,
                n_minus_1,
                max_scale_plus_1,
            )
        })
        .collect();

    ClosureResult {
        combinations,
        execution_time_secs: start_time.elapsed().as_secs_f64(),
        initial_combinations_count: initial_count,
    }
}



/// Write CLOSURE results to disk with progress tracking
pub fn write_closure_csv(
    min_scale: i32,
    max_scale: i32,
    n: usize,
    target_sum: f64,
    target_sd: f64,
    rounding_error_sums: f64,
    rounding_error_sds: f64,
    output_file: &str,
) -> io::Result<()> {
    // Setup progress bar
    let initial_count = count_initial_combinations(min_scale, max_scale);
    let bar = ProgressBar::new(initial_count as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Computing...")
            .unwrap()
            .progress_chars("=>-")
    );

    // Compute results (only this part is timed)
    let result = dfs_parallel(
        min_scale,
        max_scale,
        n,
        target_sum,
        target_sd,
        rounding_error_sums,
        rounding_error_sds,
    );
    
    println!("Computation time: {:.2} seconds", result.execution_time_secs);
    bar.set_message("Writing to disk...");

    // Initialize CSV file
    let file = File::create(output_file)?;
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(file);

    // Write header
    let header: Vec<String> = (1..=n).map(|i| format!("n{}", i)).collect();
    writer.write_record(&header)?;

    // Write combinations with progress updates
    let chunk_size = if result.combinations.len() >= 100 {
        result.combinations.len() / 100  // Update every 1%
    } else {
        1  // Update for every combination if fewer than 100
    };
    
    for (i, combination) in result.combinations.iter().enumerate() {
        writer.write_record(
            &combination
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
        )?;
        
        if i % chunk_size == 0 {
            bar.inc(1);
        }
    }

    bar.finish_with_message("Done!");
    
    println!("Number of valid combinations: {}", result.combinations.len());
    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_initial_combinations() {
        assert_eq!(count_initial_combinations(1, 3), 6);
        assert_eq!(count_initial_combinations(1, 4), 10);
    }
}

// fn main() -> io::Result<()> {
//     let min_scale = 1;
//     let max_scale = 7;
//     let n = 30;
//     let target_mean = 5.0;
//     let target_sum = target_mean * n as f64;
//     let target_sd = 2.78;
//     let rounding_error_means = 0.01;
//     let rounding_error_sums = rounding_error_means * n as f64;
//     let rounding_error_sds = 0.01;
//     let output_file = "parallel_results.csv";
// 
//     dfs_parallel(
//         min_scale,
//         max_scale,
//         n,
//         target_sum,
//         target_sd,
//         rounding_error_sums,
//         rounding_error_sds,
//         output_file,
//     )
// }