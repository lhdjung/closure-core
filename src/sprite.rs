//! Creating a space to experiment with a Rust translation of SPRITE

use core::f64;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use num_traits::Float;
use rand::prelude::*;
use thiserror::Error;
use bigdecimal::{BigDecimal, FromPrimitive, ToPrimitive, Zero};

use crate::grimmer::{decimal_places_scalar, grim_scalar_rust, grimmer_scalar, rust_round, GrimReturn, is_near};
use crate::sprite_types::{RestrictionsOption, RestrictionsMinimum, OccurrenceConstraints};

const MAX_DELTA_LOOPS_LOWER: u32 = 20_000;
const MAX_DELTA_LOOPS_UPPER: u32 = 1_000_000;
const MAX_DUP_LOOPS: u32 = 20;
const DUST: f64 = 1e-12;

#[derive(Debug, Error)]
pub enum ParameterError {
    #[error("{0}")]
    InputValidation(String),
    #[error("{0}")]
    Consistency(String),
    #[error("{0}")]
    Conflict(String),
}

// A struct to hold the final, validated parameters
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
    /// Values that can be used for the remaining, unrestricted observations
    pub possible_values: Vec<f64>,
    /// The initial set of responses that are fixed by restrictions
    pub fixed_responses: Vec<f64>,
    /// The number of fixed responses
    pub n_fixed: usize,
}

fn abm_internal(total: &BigDecimal, n_obs: &BigDecimal, a: &BigDecimal, b: &BigDecimal) -> Vec<f64> {
    // [BigDecimal] All calculations are now decimal-based.
    let k1 = ((total - (n_obs * b)) / (a - b)).with_scale_round(0, bigdecimal::RoundingMode::HalfUp);
    let k = k1.max(BigDecimal::from(1)).min(n_obs - BigDecimal::from(1));
    
    let k_floor = k.with_scale_round(0, bigdecimal::RoundingMode::Floor);
    let n_minus_k = n_obs - &k_floor;

    let mut v = Vec::with_capacity(n_obs.to_usize().unwrap());
    for _ in 0..k_floor.to_usize().unwrap() { v.push(a.clone()); }
    for _ in 0..n_minus_k.to_usize().unwrap() { v.push(b.clone()); }

    let current_sum: BigDecimal = v.iter().sum();
    let diff = current_sum - total;

    if diff.is_zero() {
        return v.iter().map(|i| i.to_f64().unwrap()).collect();
    }
    
    // Correction logic
    if let Some(val_to_change) = v.iter_mut().find(|val| *val == a) {
        *val_to_change -= diff;
    }
    v.iter().map(|i| i.to_f64().unwrap()).collect()
}

pub fn sd_limits(n_obs: u32, mean: f64, sd: f64, min_val: i32, max_val: i32, sd_prec_opt: Option<i32>, n_items: u32) -> (f64, f64) {
    // [BigDecimal] Convert inputs for high-precision calculation.
    let mean_bd = BigDecimal::from_f64(mean).unwrap();
    let n_obs_bd = BigDecimal::from_u32(n_obs).unwrap();
    let n_items_bd = BigDecimal::from_u32(n_items).unwrap();

    let sd_prec: i32 = sd_prec_opt.unwrap_or_else(|| max(decimal_places_scalar(Some(&sd.to_string()), ".").unwrap() - 1, 0));

    let a_max = BigDecimal::from_i32(min_val).unwrap();
    let a_min = (&mean_bd * &n_items_bd).with_scale_round(0, bigdecimal::RoundingMode::Floor) / &n_items_bd;

    let b_max_cand1 = BigDecimal::from_i32(max_val).unwrap();
    let b_max_cand2 = BigDecimal::from_i32(min_val + 1).unwrap();
    let b_max_cand3 = &a_min + BigDecimal::from(1);
    let b_max = std::cmp::max(b_max_cand1, std::cmp::max(b_max_cand2, b_max_cand3));

    let b_min = &a_min + (BigDecimal::from(1) / &n_items_bd);
    let total = (&mean_bd * &n_obs_bd * &n_items_bd).with_scale_round(0, bigdecimal::RoundingMode::HalfUp) / &n_items_bd;

    // --- Min SD Calculation ---
    let vec_min_sd_bd = abm_internal(&total, &n_obs_bd, &a_min, &b_min);
    let vec_min_sd_f64: Vec<f64> = vec_min_sd_bd.iter().map(|v| v.to_f64().unwrap()).collect();
    let min_sd = std_dev(&vec_min_sd_f64).unwrap_or(0.0);

    // --- Max SD Calculation ---
    let vec_max_sd_bd = abm_internal(&total, &n_obs_bd, &a_max, &b_max);
    let vec_max_sd_f64: Vec<f64> = vec_max_sd_bd.iter().map(|v| v.to_f64().unwrap()).collect();
    let max_sd = std_dev(&vec_max_sd_f64).unwrap_or(0.0);

    (rust_round(min_sd, sd_prec), rust_round(max_sd, sd_prec))
}

#[allow(clippy::too_many_arguments)]
pub fn set_parameters(
    mean: f64, sd: f64, n_obs: u32, min_val: i32, max_val: i32, m_prec: Option<i32>,
    sd_prec: Option<i32>, n_items: u32, restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption, mut dont_test: bool,
) -> Result<SpriteParameters, ParameterError> {
    if min_val >= max_val {
        return Err(ParameterError::InputValidation("max_val must be greater than min_val".to_string()));
    }
    let m_prec = m_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean.to_string()), ".").unwrap());
    let sd_prec = sd_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean.to_string()), ".").unwrap());

    if n_obs * n_items <= 10.0.powi(m_prec) as u32 {
        dont_test = true;
    }

    if !dont_test {
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
        
        let grimmer_consistent = grimmer_scalar(
            &mean.to_string(), &sd.to_string(), n_obs, n_items, vec![false, false, false],
            "up_or_down", 5.0, f64::EPSILON.sqrt(),
        );
        if !grimmer_consistent {
            return Err(ParameterError::Consistency("SD fails GRIMMER test.".to_string()));
        }
    }

    let sd_lims = sd_limits(n_obs, mean, sd, min_val, max_val, Some(sd_prec), n_items);
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


#[derive(Debug, PartialEq, Clone)]
pub enum Outcome {
    Success,
    Failure,
}

/// Holds the detailed results of a single distribution search
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

        if is_near(rust_round(current_mean, m_prec), target_mean, DUST) {
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
                if let Some(pos) = poss_values.iter().position(|&p| is_near(p, current_val, DUST)) {
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

/// Attempts to shift values within a vector to better match a target standard deviation,
/// while keeping the mean approximately constant
///
/// This is a core part of the SPRITE algorithm's search process
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
    let max_attempts = vec.len() * 2; 

    for _ in 0..max_attempts {
        // Randomly sample two distinct indices
        let i = rng.random_range(0..vec.len());
        let mut j = rng.random_range(0..vec.len());
        while i == j {
            j = rng.random_range(0..vec.len());
        }

        let val1 = vec[i];
        let val2 = vec[j];

        let val1_new_opt = params.possible_values.iter().find(|&&v| v > val1 && !is_near(v, val1, DUST));
        let val2_new_opt = params.possible_values.iter().rev().find(|&&v| v < val2 && !is_near(v, val2, DUST));

        if let (Some(&val1_new), Some(&val2_new)) = (val1_new_opt, val2_new_opt) {
            if is_near(val1_new - val1, val2 - val2_new, DUST) {
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

        // 1. Shift values to adjust SD
        shift_values(&mut vec, params, rng);

        // 2. Check for and correct mean drift
        let current_mean = mean(&[vec.as_slice(), params.fixed_responses.as_slice()].concat());
        if !is_near(rust_round(current_mean, params.m_prec), params.mean, DUST) {
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
    rng: &mut impl Rng,
) -> Vec<DistributionResult> {

    let mut results: Vec<DistributionResult> = Vec::new();
    let mut unique_distributions = HashSet::<Vec<i64>>::new();
    let mut consecutive_failures = 0;
    let mut consecutive_duplicates = 0;

    for _ in 0..(n_distributions * MAX_DUP_LOOPS as usize) {
        if unique_distributions.len() >= n_distributions {
            break;
        }

        // Stop if the search seems to be stalled
        if consecutive_failures >= 10 {
            println!("Warning: No successful distribution found in the last 10 attempts. Exiting.");
            break;
        }
        
        // Calculate max duplicates allowed before stopping
        let n_found = unique_distributions.len() as f64;
        let max_duplications = if n_found > 0.0 {
            (0.00001f64.ln() / (n_found / (n_found + 1.0)).ln()).round() as u32
        } else {
            100 // Default if no successes yet
        }.max(100);

        if consecutive_duplicates > max_duplications {
            println!("Warning: Found too many consecutive duplicate distributions. Exiting.");
            break;
        }

        match find_possible_distribution(params, rng) {
            Ok(mut res) => {
                match res.outcome {
                    Outcome::Success => {

                        res.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        // Convert to scaled integers for hashing
                        let hashable_values: Vec<i64> = res.values.iter().map(|v| (v * 1_000_000.0).round() as i64).collect();

                        if unique_distributions.insert(hashable_values) {
                            // This is a new, unique distribution
                            results.push(res);
                            consecutive_failures = 0;
                            consecutive_duplicates = 0;

                        } else {
                            // This is a duplicate
                            consecutive_duplicates += 1;
                        }
                    }
                    Outcome::Failure => {
                        consecutive_failures += 1;
                        if return_failures {
                            results.push(res);
                        }
                    }
                }
            }

            Err(e) => {
                // This indicates a fatal error in the setup, like mean adjustment failing
                eprintln!("Fatal error during distribution search: {e}");
                break;
            }
        }
    }

    if unique_distributions.len() < n_distributions {
        println!("Only {} matching distributions could be found.", unique_distributions.len());
    }
    
    if !return_failures {
        results.retain(|r| r.outcome == Outcome::Success);
    }

    results
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


#[cfg(test)]
pub mod tests {
    use super::*;
    use rand::rngs::StdRng;

    #[test]
    fn sprite_test_mean() {
        let sprite_parameters = set_parameters(2.2, 1.3, 20, 1, 5, None, None, 1, None, RestrictionsOption::Default, false).unwrap();
        let results = find_possible_distributions(&sprite_parameters, 5, false, &mut StdRng::seed_from_u64(1234));

        assert_eq!(rust_round(results[0].mean, 1), 2.2);
    }

    #[test]
    fn sprite_test_sd() {
        let sprite_parameters = set_parameters(2.2, 1.3, 20, 1, 5, None, None, 1, None, RestrictionsOption::Default, false).unwrap();
        let results = find_possible_distributions(&sprite_parameters, 5, false, &mut StdRng::seed_from_u64(1234));

        for result in results {
            assert_eq!(rust_round(result.sd, 1), 1.3);
        }
    }
    #[test]
    fn sprite_test_big() {
        let params = set_parameters(26.281, 14.6339, 1000, 1, 50, Some(3), Some(4), 1, None, RestrictionsOption::Default, true).unwrap();
        // let params = set_parameters(50.5565, 29.1503, 1231, 1, 100, None, None, 1, None, RestrictionsOption::Default, false).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        let results = &find_possible_distributions(&params, 10, false, &mut rng);

        for result in results {
            assert_eq!(rust_round(result.mean, 3), 26.281);
            assert_eq!(rust_round(result.sd, 4), 14.6339);
        }
    }
}

