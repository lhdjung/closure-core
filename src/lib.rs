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


/// Struct to hold combinations of possible raw data during processing
#[derive(Clone)]
struct Combination {
    values: Vec<i32>,
    running_sum: f64,
    running_m2: f64,
}

/// Calculates the number of initial combinations that will be processed in parallel
#[inline]
pub fn count_initial_combinations(scale_min: i32, scale_max: i32) -> i32 {
    let range_size = scale_max - scale_min + 1;
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
    sd_upper: f64,
    sd_lower: f64,
    scale_min_sum: &[i32],
    scale_max_sum: &[i32],
    n_minus_1: usize,
    scale_max_plus_1: i32,
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
            if current_std >= sd_lower {
                results.push(current.values);
            }
            continue;
        }

        let n_left = n_minus_1 - current.values.len();
        let next_n = current.values.len() + 1;
        let last_value = *current.values.last().unwrap();
        let current_mean = current.running_sum / current.values.len() as f64;

        for next_value in last_value..scale_max_plus_1 {
            let next_sum = current.running_sum + next_value as f64;
            let minmean = next_sum + scale_min_sum[n_left] as f64;
            if minmean > target_sum_upper {
                break; // Early termination - better than take_while!
            }
            
            let maxmean = next_sum + scale_max_sum[n_left] as f64;
            if maxmean < target_sum_lower {
                continue;
            }
        
            let next_mean = next_sum / next_n as f64;
            let delta = next_value as f64 - current_mean;
            let delta2 = next_value as f64 - next_mean;
            let next_m2 = current.running_m2 + delta * delta2;
            
            let min_sd = (next_m2 / n_minus_1 as f64).sqrt();
            if min_sd <= sd_upper {
                let mut new_values = current.values.clone();
                new_values.push(next_value);
                stack.push_back(Combination {
                    values: new_values,
                    running_sum: next_sum,
                    running_m2: next_m2,
                });
            }
        }
    }

    results
}

/// Run CLOSURE across starting combinations and return results
pub fn dfs_parallel(
    mean: f64,
    sd: f64,
    n: usize,
    scale_min: i32,
    scale_max: i32,
    // target_sum: f64,
    rounding_error_mean: f64,
    rounding_error_sd: f64,
) -> Vec<Vec<i32>> {
    
    // Remember: target_sum == mean * n
    let target_sum = mean * n as f64;
    let rounding_error_sum = rounding_error_mean * n as f64;
    
    let target_sum_upper = target_sum + rounding_error_sum;
    let target_sum_lower = target_sum - rounding_error_sum;
    let sd_upper = sd + rounding_error_sd;
    let sd_lower = sd - rounding_error_sd;
    
    // Precompute scale sums for optimization
    let scale_min_sum: Vec<i32> = (0..n).map(|x| scale_min * x as i32).collect();
    let scale_max_sum: Vec<i32> = (0..n).map(|x| scale_max * x as i32).collect();
    
    let n_minus_1 = n - 1;
    let scale_max_plus_1 = scale_max + 1;

    // Generate initial combinations
    (scale_min..=scale_max)
        .flat_map(|i| {
            (i..=scale_max).map(move |j| {
                let initial_combination = vec![i, j];
                let running_sum = (i + j) as f64;
                let current_mean = running_sum / 2.0;
                let current_m2 = (i as f64 - current_mean).powi(2) + (j as f64 - current_mean).powi(2);
                (initial_combination, running_sum, current_m2)
            })
        })
        .collect::<Vec<_>>()
    // Process combinations in parallel
        .par_iter()
        .flat_map(|(combo, running_sum, running_m2)| {
            dfs_branch(
                combo.clone(),
                *running_sum,
                *running_m2,
                n,
                target_sum_upper,
                target_sum_lower,
                sd_upper,
                sd_lower,
                &scale_min_sum,
                &scale_max_sum,
                n_minus_1,
                scale_max_plus_1,
            )
        })
        .collect::<Vec<Vec<i32>>>()
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


