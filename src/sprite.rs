//! Creating a space to experiment with a Rust translation of SPRITE

use std::cmp::max;
use crate::grimmer::{decimal_places_scalar, grim_scalar_rust, grimmer_scalar, rust_round, GrimReturn};
use crate::sprite_types::{RestrictionsOption, RestrictionsMinimum, OccurrenceConstraints};
use core::f64;
use std::collections::{HashMap, HashSet};
use std::iter::once;
use statrs::statistics::Statistics;
use rand::prelude::*;
use thiserror::Error;

// Loop iteration limits, u32 is a good choice for these counts.
const MAX_DELTA_LOOPS_LOWER: u32 = 20_000;
const MAX_DELTA_LOOPS_UPPER: u32 = 1_000_000;
const MAX_DUP_LOOPS: u32 = 20;

// This is identical to the `FUZZ_VALUE` constant already in the Canvas.
// You can either use `FUZZ_VALUE` or rename it to `DUST` if you prefer.
const DUST: f64 = 1e-12;

// A very large number, represented as a 64-bit float.
// not actually used in our code or in the original rsprite2
// const HUGE: f64 = 1e15;

#[derive(Debug, Error)]
pub enum ParameterError {
    #[error("{0}")]
    InputValidation(String),
    #[error("{0}")]
    Consistency(String),
    #[error("{0}")]
    Conflict(String),
}

// A struct to hold the final, validated parameters.
#[derive(Debug)]
pub struct SpriteParameters {
    pub mean: f64,
    pub sd: f64,
    pub n_obs: u32,
    pub min_val: i32,
    pub max_val: i32,
    pub m_prec: i32,
    pub sd_prec: i32,
    pub n_items: u32,
    /// Values that can be used for the remaining, unrestricted observations.
    pub possible_values: Vec<f64>,
    /// The initial set of responses that are fixed by restrictions.
    pub fixed_responses: Vec<f64>,
    /// The number of fixed responses.
    pub n_fixed: usize,
}


#[allow(clippy::too_many_arguments)]
pub fn set_parameters(
    mean: f64, sd: f64, n_obs: u32, min_val: i32, max_val: i32, m_prec: Option<i32>,
    sd_prec: Option<i32>, n_items: u32, restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption, dont_test: bool,
) -> Result<SpriteParameters, ParameterError> {
    if min_val >= max_val {
        return Err(ParameterError::InputValidation("max_val must be greater than min_val".to_string()));
    }
    let m_prec = m_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean.to_string()), ".").unwrap());
    let sd_prec = sd_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean.to_string()), ".").unwrap());

    if !dont_test {
        if (n_obs * n_items) as f64 <= 10.0f64.powi(m_prec) {
            let grim_result = grim_scalar_rust(
                &mean.to_string(), n_obs, vec![false, false, false], n_items,
                "up_or_down", 5.0, f64::EPSILON.sqrt(),
            );
            let consistent = match grim_result {
                Ok(GrimReturn::Bool(b)) => b,
                Ok(GrimReturn::List(b, _, _, _, _, _)) => b,
                Err(_) => false, // Treat error as inconsistency
            };
            if !consistent {
                return Err(ParameterError::Consistency("Mean fails GRIM test.".to_string()));
            }
        }
        
        let grimmer_consistent = grimmer_scalar(
            &mean.to_string(), &sd.to_string(), n_obs, n_items, vec![false, false, false],
            "up_or_down", 5.0, f64::EPSILON.sqrt(),
        );
        if !grimmer_consistent {
            return Err(ParameterError::Consistency("SD fails GRIMMER test.".to_string()));
        }
    }

    let sd_lims = sd_limits(n_obs, mean, min_val, max_val, Some(sd_prec), n_items);
    if !(sd >= sd_lims.0 && sd <= sd_lims.1) {
        return Err(ParameterError::InputValidation(format!("SD is outside the possible range: [{}, {}]", sd_lims.0, sd_lims.1)));
    }
    if !(mean >= min_val as f64 && mean <= max_val as f64) {
        return Err(ParameterError::InputValidation("Mean is outside the possible [min_val, max_val] range.".to_string()));
    }

    let restrictions_exact = restrictions_exact.unwrap_or_default();
    let restrictions_minimum = match restrictions_minimum {
        RestrictionsOption::Default => Some(RestrictionsMinimum::from_range(min_val * 100, max_val * 100).extract()),
        RestrictionsOption::Opt(opt_map) => opt_map.map(|rm| rm.extract()),
        RestrictionsOption::Null => None,
    };

    if let Some(ref min_map) = restrictions_minimum {
        let constraints = OccurrenceConstraints::new(restrictions_exact.clone(), min_map.clone(), None);
        if constraints.check_conflicts() {
            let exact_keys: HashSet<_> = constraints.exact.keys().collect();
            let min_keys: HashSet<_> = constraints.minimum.keys().collect();
            let conflict_keys: Vec<_> = exact_keys.intersection(&min_keys).map(|&&k| k as f64 / 100.0).collect();
            return Err(ParameterError::Conflict(format!("Value(s) in both exact and minimum restrictions: {conflict_keys:? }")));
        }
    }

    let mut poss_values: Vec<f64> = Vec::new();
    for i in 1..=n_items {
        for val in min_val..max_val {
            poss_values.push(val as f64 + (i - 1) as f64 / n_items as f64);
        }
    }
    poss_values.push(max_val as f64);
    poss_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    poss_values.dedup();
    let poss_values_keys: HashSet<i32> = poss_values.iter().map(|&v| (v * 100.0).round() as i32).collect();

    let mut fixed_responses: Vec<f64> = Vec::new();
    let mut fixed_values_keys: HashSet<i32> = HashSet::new();

    for (&key, &count) in &restrictions_exact {
        if !poss_values_keys.contains(&key) {
            return Err(ParameterError::InputValidation(format!("Invalid key in restrictions_exact: {}", key as f64 / 100.0)));
        }
        fixed_values_keys.insert(key);
        for _ in 0..count { fixed_responses.push(key as f64 / 100.0); }
    }

    if let Some(min_map) = restrictions_minimum {
        for (&key, &count) in &min_map {
            if !poss_values_keys.contains(&key) {
                return Err(ParameterError::InputValidation(format!("Invalid key in restrictions_minimum: {}", key as f64 / 100.0)));
            }
            for _ in 0..count { fixed_responses.push(key as f64 / 100.0); }
        }
    }

    let final_possible_values: Vec<f64> = poss_values.into_iter().filter(|v| {
        !fixed_values_keys.contains(&((v * 100.0).round() as i32))
    }).collect();
    let n_fixed = fixed_responses.len();

    Ok(SpriteParameters {
        mean, sd, n_obs, min_val, max_val, m_prec, sd_prec, n_items,
        possible_values: final_possible_values,
        fixed_responses,
        n_fixed,
    })
}


pub fn sd_limits(n_obs: u32, mean: f64, min_val: i32, max_val: i32, sd_prec: Option<i32>, n_items: u32) -> (f64, f64) {

    // double check that this is actually the right implementation, getting the sd precision from
    // mean, not sd as we do in set_parameters
    let sd_prec: i32 = sd_prec.unwrap_or_else(|| 
        max(decimal_places_scalar(Some(&mean.to_string()), ".").unwrap() - 1, 0)
    );

    let a_max = min_val;
    let a_min = (mean * n_items as f64).floor()/n_items as f64;
    let b_max = [max_val as f64, min_val as f64 + 1.0, a_min + 1.0].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let b_min = a_min + 1.0 / n_items as f64;
    let total = rust_round(mean * n_obs as f64 * n_items as f64, 0)/n_items as f64;

    let mut poss_values = vec![max_val as f64];

    for i in 0..n_items {
        generate_sequence(min_val, max_val, n_items, i).append(&mut poss_values)
    }
    poss_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // assuming that what the above is supposed to be doing is creating a vector which we merge into
    // poss_values before sorting everything
    // So poss_values should still be a flat vector?
    

    let combined_vec_1: Vec<f64> = abm_internal(total, n_obs as f64, a_min, b_min);
    
    if rust_round(combined_vec_1.iter().sum::<f64>()/combined_vec_1.len() as f64, sd_prec) != rust_round(mean, sd_prec) {
        panic!("Error in calculating range of possible standard deviations")
    }

    let combined_vec_2: Vec<f64> = abm_internal(total, n_obs as f64, a_max as f64, b_max);
    
    if rust_round(combined_vec_2.iter().sum::<f64>()/combined_vec_2.len() as f64, sd_prec) != rust_round(mean, sd_prec) {
        panic!("Error in calculating range of possible standard deviations")
    }

    (rust_round(combined_vec_1.std_dev(), sd_prec), rust_round(combined_vec_2.std_dev(), sd_prec))
}

fn abm_internal(total: f64, n_obs: f64, a: f64, b: f64) -> Vec<f64>{

    let k1 = rust_round((total - (n_obs * b)) / (a - b), 0);
    let k = f64::min(f64::max(k1, 1.0), n_obs - 1.0);

    let mut v1 = vec![a; k.floor() as usize];
    let mut v2 = vec![b; (n_obs - k).floor() as usize];

    v2.append(&mut v1);

    let diff = v2.iter().sum::<f64>() - total;

    // assuming we're cmpletely overwriting the contents of the vector we made before, so I've
    // disambiguated the variable names
    //

    let combined_vec: Vec<f64> = match diff < 0.0 {
        true => vec![a; (k - 1.0).floor() as usize]
            .into_iter()
            .chain(once(a + diff.abs()))
            .chain(vec![b; (n_obs - k).floor() as usize])
            .collect(),
        false => vec![a; k as usize]
            .into_iter()
            .chain(once(b - diff))
            .chain(vec![b; (n_obs - k - 1.0).floor() as usize]).collect(),
    };

    combined_vec
}


#[derive(Debug, PartialEq, Clone)]
pub enum Outcome {
    Success,
    Failure,
}

/// Holds the detailed results of a single distribution search.
#[derive(Debug, Clone)]
pub struct DistributionResult {
    pub outcome: Outcome,
    pub values: Vec<f64>,
    pub mean: f64,
    pub sd: f64,
    pub iterations: u32,
}

/// Iteratively adjusts a vector's values to match a target mean
///
/// This function randomly selects elements and nudges them up or down according
/// to a list of possible values until the rounded mean of the full dataset
/// matches the target
pub fn adjust_mean(
    vec: &mut [f64],
    fixed_vals: &[f64],
    poss_values: &[f64],
    target_mean: f64,
    m_prec: i32,
    max_iter: u32,
    rng: &mut impl Rng,
) -> Result<(), String> {
    if poss_values.is_empty() {
        return Err("Cannot adjust mean with no possible values.".to_string());
    }

    let sum_fixed: f64 = fixed_vals.iter().sum();
    let len_fixed = fixed_vals.len();
    let total_len = vec.len() + len_fixed;

    for _ in 0..max_iter {
        let sum_vec: f64 = vec.iter().sum();
        let current_mean = (sum_vec + sum_fixed) / total_len as f64;

        if equalish(rust_round(current_mean, m_prec), target_mean) {
            return Ok(()); // Success
        }

        let increase_mean = current_mean < target_mean;
        let min_poss_val = poss_values[0];
        let max_poss_val = poss_values[poss_values.len() - 1];

        // A reasonable number of attempts to find a valid value to bump
        let max_attempts = vec.len().max(1) * 4;
        let mut changed = false;

        for _ in 0..max_attempts {
            let index_to_try = rng.random_range(0..vec.len());
            let current_val = vec[index_to_try];

            let is_valid_to_bump = if increase_mean {
                current_val < max_poss_val
            } else {
                current_val > min_poss_val
            };

            if is_valid_to_bump {
                if let Some(pos) = poss_values.iter().position(|&p| equalish(p, current_val)) {
                    let new_pos = if increase_mean { pos + 1 } else { pos.saturating_sub(1) };
                    if let Some(&new_val) = poss_values.get(new_pos) {
                        vec[index_to_try] = new_val;
                        changed = true;
                        break; // Found a valid change, break from attempt loop
                    }
                }
            }
        }

        if !changed {
            // If after many attempts we couldn't find a value to change, we're probably stuck
            break;
        }
    }

    let err_msg = if !fixed_vals.is_empty() {
        "Couldn't initialize data with correct mean. This *might* be because the restrictions cannot be satisfied."
    } else {
        "Couldn't initialize data with correct mean. This might indicate a coding error if the mean is in range."
    };
    Err(err_msg.to_string())
}


/// The main function to find a distribution matching the given parameters.
///
/// It works by:
/// 1. Generating a random set of starting data.
/// 2. Iteratively adjusting the data to match the target mean.
/// 3. Iteratively shifting values to match the target standard deviation.
pub fn find_possible_distribution(
    params: &SpriteParameters,
    rng: &mut impl Rng,
) -> Result<DistributionResult, String> {

    let r_n = params.n_obs - params.n_fixed as u32;
    if params.possible_values.is_empty() && r_n > 0 {
        return Err("No possible values to sample from for initialization.".to_string());
    }
    let mut vec: Vec<f64> = (0..r_n)
        .map(|_| *params.possible_values.choose(rng).unwrap())
        .collect();

    // Initial mean adjustment
    let max_loops_mean = params.n_obs * params.possible_values.len() as u32;
    adjust_mean(&mut vec, &params.fixed_responses, &params.possible_values, params.mean, params.m_prec, max_loops_mean, rng)?;

    let max_loops_sd = (params.n_obs * (params.possible_values.len().pow(2) as u32)).clamp(MAX_DELTA_LOOPS_LOWER, MAX_DELTA_LOOPS_UPPER);
    let granule_sd = (0.1f64.powi(params.sd_prec)) / 2.0 + DUST;

    let sum_fixed: f64 = params.fixed_responses.iter().sum();
    let len_fixed: usize = params.fixed_responses.len();

    for i in 1..=max_loops_sd {

        let sum_vec: f64 = vec.iter().sum();
        let len_vec: usize = vec.len();
        let total_len = len_vec + len_fixed;
        let total_sum = sum_vec + sum_fixed;
        let combined_mean = total_sum / total_len as f64;

       // Check for success
        let sum_sq_diff_vec: f64 = vec.iter().map(|&v| (v - combined_mean).powi(2)).sum();
        let sum_sq_diff_fixed: f64 = params.fixed_responses.iter().map(|&v| (v - combined_mean).powi(2)).sum();
        
        let variance = (sum_sq_diff_vec + sum_sq_diff_fixed) / (total_len - 1) as f64;
        let current_sd = variance.sqrt();
        
        if (current_sd - params.sd).abs() <= granule_sd {
            // Only now, on success, do we allocate the final vector.
            let mut full_vec = vec.clone();
            full_vec.extend_from_slice(&params.fixed_responses);
            return Ok(DistributionResult {
                outcome: Outcome::Success,
                values: full_vec,
                mean: combined_mean,
                sd: current_sd,
                iterations: i,
            });
        }

        // --- Main Loop Logic ---
        // 1. Shift values to adjust SD
        shift_values(&mut vec, params, rng);

        // 2. Check for and correct mean drift
        let current_mean = mean(&[vec.as_slice(), params.fixed_responses.as_slice()].concat());
        if !equalish(rust_round(current_mean, params.m_prec), params.mean) {
            // The mean has drifted. Pull it back.
            adjust_mean(
                &mut vec, 
                &params.fixed_responses, 
                &params.possible_values, 
                params.mean, 
                params.m_prec, 
                20, 
                rng
            ).unwrap_or(());
        }
    }

    // If loop finishes, we have failed to find a solution.
    let mut final_vec = vec;
    final_vec.extend_from_slice(&params.fixed_responses);
    let final_sd = std_dev(&final_vec).unwrap_or(0.0);
    let vec_mean = mean(&final_vec);
    Ok(DistributionResult {
        outcome: Outcome::Failure,
        values: final_vec,
        mean: vec_mean,
        sd: final_sd,
        iterations: max_loops_sd,
    })
}

/// Finds multiple possible distributions that match the given parameters.
///
/// This function repeatedly calls `find_possible_distribution` to gather a set of
/// unique, successful distributions. It includes logic to stop if the search stalls
/// or finds too many consecutive duplicates.
pub fn find_possible_distributions(
    params: &SpriteParameters,
    n_distributions: usize,
    return_failures: bool,
    rng: &mut impl Rng, // Require a thread-safe RNG
) -> Vec<DistributionResult> {
    let mut results: Vec<DistributionResult> = Vec::new();
    let mut unique_distributions = HashSet::<Vec<i64>>::new();
    let mut consecutive_failures = 0;
    let mut consecutive_duplicates = 0;
    for _ in 0..(n_distributions * MAX_DUP_LOOPS as usize) {
        if unique_distributions.len() >= n_distributions { break; }
        if consecutive_failures >= 10 { println!("Warning: No successful distribution found in the last 10 attempts. Exiting."); break; }
        let n_found = unique_distributions.len() as f64;
        let max_duplications = if n_found > 0.0 { (0.00001f64.ln() / (n_found / (n_found + 1.0)).ln()).round() as u32 } else { 100 }.max(100);
        if consecutive_duplicates > max_duplications { println!("Warning: Found too many consecutive duplicate distributions. Exiting."); break; }
        if let Ok(mut res) = find_possible_distribution(params, rng) {
            match res.outcome {
                Outcome::Success => {
                    res.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let hashable_values: Vec<i64> = res.values.iter().map(|v| (v * 1_000_000.0).round() as i64).collect();
                    if unique_distributions.insert(hashable_values) {
                        results.push(res);
                        consecutive_failures = 0;
                        consecutive_duplicates = 0;
                    } else { consecutive_duplicates += 1; }
                }
                Outcome::Failure => {
                    consecutive_failures += 1;
                    if return_failures { results.push(res); }
                }
            }
        }
    }
    if unique_distributions.len() < n_distributions { println!("Only {} matching distributions could be found.", unique_distributions.len()); }
    if !return_failures { results.retain(|r| r.outcome == Outcome::Success); }
    results
}

/// Attempts to shift values within a vector to better match a target standard deviation,
/// while keeping the mean approximately constant.
///
/// This is a core part of the SPRITE algorithm's search process.
pub fn shift_values(vec: &mut [f64], params: &SpriteParameters, rng: &mut impl Rng) -> bool {
    // In-place SD calculation
    let sum_vec: f64 = vec.iter().sum();
    let sum_fixed: f64 = params.fixed_responses.iter().sum();
    let total_len = vec.len() + params.fixed_responses.len();
    let combined_mean = (sum_vec + sum_fixed) / total_len as f64;
    let sum_sq_diff_vec: f64 = vec.iter().map(|&v| (v - combined_mean).powi(2)).sum();
    let sum_sq_diff_fixed: f64 = params.fixed_responses.iter().map(|&v| (v - combined_mean).powi(2)).sum();
    let variance = (sum_sq_diff_vec + sum_sq_diff_fixed) / (total_len - 1) as f64;
    let current_sd = variance.sqrt();
    let increase_sd = current_sd < params.sd;

    // A reasonable number of attempts to find a valid swap
    let max_attempts = vec.len() * 4; 

    for _ in 0..max_attempts {
        // Randomly sample two distinct indices
        let i = rng.random_range(0..vec.len());
        let mut j = rng.random_range(0..vec.len());
        while i == j {
            j = rng.random_range(0..vec.len());
        }

        let val1 = vec[i];
        let val2 = vec[j];

        let val1_new_opt = params.possible_values.iter().find(|&&v| v > val1 && !equalish(v, val1));
        let val2_new_opt = params.possible_values.iter().rev().find(|&&v| v < val2 && !equalish(v, val2));

        if let (Some(&val1_new), Some(&val2_new)) = (val1_new_opt, val2_new_opt) {
            if equalish(val1_new - val1, val2 - val2_new) {
                let sd_before = std_dev(&[val1, val2]).unwrap_or(0.0);
                let sd_after = std_dev(&[val1_new, val2_new]).unwrap_or(0.0);
                
                let is_pointless = if increase_sd { sd_after <= sd_before } else { sd_after >= sd_before };

                if !is_pointless {
                    vec[i] = val1_new;
                    vec[j] = val2_new;
                    return true;
                }
            }
        }
    }
    false
}



fn equalish(x: f64, y: f64) -> bool {
  x <= (y + DUST) &&
    x >= (y - DUST)
}

/// Calculates the mean (average) of a slice of f64 values.
/// Returns 0.0 if the slice is empty.
pub fn mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    if data.is_empty() {
        0.0
    } else {
        sum / (data.len() as f64)
    }
}

/// Calculates the sample standard deviation of a slice of f64 values.
/// Returns `None` if the slice has fewer than two elements, as SD is not defined.
pub fn std_dev(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let data_mean = mean(data);
    let variance = data.iter().map(|value| {
        let diff = value - data_mean;
        diff * diff
    }).sum::<f64>() / (n - 1) as f64; // Use n-1 for sample standard deviation

    Some(variance.sqrt())
}

/// Calculates the difference between adjacent elements in a slice.
/// `diff(c(a, b, c))` in R is equivalent to `c(b-a, c-b)`.
pub fn diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 {
        return Vec::new();
    }
    // `windows(2)` creates an iterator over overlapping sub-slices of length 2.
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// A struct to hold the results of a run-length encoding.
#[derive(Debug, PartialEq)]
pub struct Rle<T> {
    pub values: Vec<T>,
    pub lengths: Vec<usize>,
}

/// Performs run-length encoding on a slice.
/// It identifies consecutive runs of identical values and returns a struct
/// containing the values and the length of each run.
pub fn rle<T: PartialEq + Copy>(data: &[T]) -> Option<Rle<T>> {
    if data.is_empty() {
        return None;
    }

    let mut values = Vec::new();
    let mut lengths = Vec::new();

    let mut current_val = data[0];
    let mut current_len = 1;

    for &item in &data[1..] {
        if item == current_val {
            current_len += 1;
        } else {
            values.push(current_val);
            lengths.push(current_len);
            current_val = item;
            current_len = 1;
        }
    }

    // Push the last run
    values.push(current_val);
    lengths.push(current_len);

    Some(Rle { values, lengths })
}

fn generate_sequence(min_val: i32, max_val: i32, n_items: u32, i: u32) -> Vec<f64> {
    let increment = (1.0 / n_items as f64) * (i as f64 - 1.0);
    (min_val..max_val)
        .map(|val| val as f64 + increment)
        .collect()
}

