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
use std::slice::SliceIndex;
//use std::iter::Step;
use num::{Float, Integer, NumCast, FromPrimitive, ToPrimitive};
use rayon::prelude::*;


/// Struct to hold combinations of possible raw data during processing
#[derive(Clone)]
struct Combination<T, U>
where
    T: Float + FromPrimitive + Send + Sync,
    U: Integer + NumCast + ToPrimitive + Copy + Send + Sync + std::slice::SliceIndex<[U]>,
{
    values: Vec<U>,
    running_sum: T,
    running_m2: T,
}

/// Calculates the number of initial combinations that will be processed in parallel
#[inline]
pub fn count_initial_combinations(scale_min: i32, scale_max: i32) -> i32 {
    let range_size = scale_max - scale_min + 1;
    (range_size * (range_size + 1)) / 2
}

/// Collect all valid combinations from a starting point
#[inline]
fn dfs_branch<T, U>(
    start_combination: Vec<U>,
    running_sum_init: T,
    running_m2_init: T,
    n: usize,  // Use usize for the length
    target_sum_upper: T,
    target_sum_lower: T,
    sd_upper: T,
    sd_lower: T,
    scale_min_sum: &[U],
    scale_max_sum: &[U],
    n_minus_1: U,
    scale_max_plus_1: U,
) -> Vec<Vec<U>>
where
    T: Float + FromPrimitive + Send + Sync,
    U: Integer + NumCast + ToPrimitive + Copy + Send + Sync,
{
    let mut stack = VecDeque::with_capacity(n * 2); // Preallocate with reasonable capacity
    let mut results = Vec::new();
    
    stack.push_back(Combination {
        values: start_combination.clone(),
        running_sum: running_sum_init,
        running_m2: running_m2_init,
    });
    
    while let Some(current) = stack.pop_back() {
        if current.values.len() >= n {
            let current_std = (current.running_m2 / T::from(n).unwrap()).sqrt();
            if current_std >= sd_lower {
                results.push(current.values);
            }
            continue;
        }

        // Calculate remaining items to add
        let current_len = current.values.len();
        let n_left = n - current_len - 1; // How many more items after the next one
        let next_n = current_len + 1;

        // Get current mean
        let current_mean = current.running_sum / T::from(current_len).unwrap();

        // Get the last value and convert to i32 for range operations
        let last_value = U::to_i32(&current.values[current_len - 1]).unwrap();
        let scale_max_plus_1_i32 = U::to_i32(&scale_max_plus_1).unwrap();

        // Use concrete i32 type for the range
        for next_value_i32 in last_value..scale_max_plus_1_i32 {
            let next_value = U::from(next_value_i32).unwrap();
            let next_value_as_t = T::from(next_value_i32).unwrap();
            let next_sum = current.running_sum + next_value_as_t;
            
            // Safe indexing with bounds check
            if n_left < scale_min_sum.len() {
                let scale_min_sum_value = scale_min_sum[n_left];
                let scale_min_sum_as_t = T::from(U::to_i32(&scale_min_sum_value).unwrap()).unwrap();
                
                let minmean = next_sum + scale_min_sum_as_t;
                if minmean > target_sum_upper {
                    break; // Early termination - better than take_while!
                }
                
                // Safe indexing with bounds check
                if n_left < scale_max_sum.len() {
                    let scale_max_sum_value = scale_max_sum[n_left];
                    let scale_max_sum_as_t = T::from(U::to_i32(&scale_max_sum_value).unwrap()).unwrap();
                    
                    let maxmean = next_sum + scale_max_sum_as_t;
                    if maxmean < target_sum_lower {
                        continue;
                    }
                    
                    let next_mean = next_sum / T::from(next_n).unwrap();
                    let delta  = next_value_as_t - current_mean;
                    let delta2 = next_value_as_t - next_mean;
                    let next_m2 = current.running_m2 + delta * delta2;
                    
                    let min_sd = (next_m2 / T::from(n).unwrap()).sqrt();
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
        }
    }

    results
}

/// Run CLOSURE across starting combinations and return results
pub fn dfs_parallel<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
) -> Vec<Vec<U>>
where
    T: Float + FromPrimitive + Send + Sync,
    U: Integer + NumCast + ToPrimitive + Copy + Send + Sync,
{
    use std::collections::VecDeque;
    use rayon::prelude::*;
    
    // Convert integer `n` to float to enable multiplication with other floats
    let n_float = T::from(U::to_i32(&n).unwrap()).unwrap();
    
    // Target sum calculations
    let target_sum = mean * n_float;
    let rounding_error_sum = rounding_error_mean * n_float;
    
    let target_sum_upper = target_sum + rounding_error_sum;
    let target_sum_lower = target_sum - rounding_error_sum;
    let sd_upper = sd + rounding_error_sd;
    let sd_lower = sd - rounding_error_sd;

    // Convert to concrete types for range operations
    let n_usize = U::to_usize(&n).unwrap();
    let scale_min_i32 = U::to_i32(&scale_min).unwrap();
    let scale_max_i32 = U::to_i32(&scale_max).unwrap();
    
    // Precompute scale sums using concrete types first, then convert back to U
    let scale_min_sum: Vec<U> = (0..n_usize)
        .map(|x| U::from(scale_min_i32 * (x as i32)).unwrap())
        .collect();
    
    let scale_max_sum: Vec<U> = (0..n_usize)
        .map(|x| U::from(scale_max_i32 * (x as i32)).unwrap())
        .collect();
    
    let n_minus_1 = n - U::one();
    let scale_max_plus_1 = scale_max + U::one();

    // Generate initial combinations using concrete types for the ranges
    let combinations = (scale_min_i32..=scale_max_i32)
        .flat_map(|i| {
            (i..=scale_max_i32).map(move |j| {
                // Convert back to type U for the combination
                let i_u = U::from(i).unwrap();
                let j_u = U::from(j).unwrap();
                let initial_combination = vec![i_u, j_u];
                
                // Convert to T for calculations
                let i_float = T::from(i).unwrap();
                let j_float = T::from(j).unwrap();
                let sum = i_float + j_float;
                let two = T::from(2).unwrap();
                let current_mean = sum / two;
                let diff_i = i_float - current_mean;
                let diff_j = j_float - current_mean;
                let current_m2 = diff_i * diff_i + diff_j * diff_j;
                
                (initial_combination, sum, current_m2)
            })
        })
        .collect::<Vec<_>>();

    // Define the Combination struct
    #[derive(Clone)]
    struct Combination<T, U> {
        values: Vec<U>,
        running_sum: T,
        running_m2: T,
    }

    // Process combinations in parallel
    combinations.par_iter()
        .flat_map(|(combo, running_sum, running_m2)| {
            dfs_branch(
                combo.clone(),
                *running_sum,
                *running_m2,
                n_usize,
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
        .collect()
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


