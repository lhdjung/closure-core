//! Creating a space to experiment with a Rust translation of SPRITE

use std::cmp::max;
use crate::grimmer::{decimal_places_scalar, grim_scalar_rust, grimmer_scalar, rust_round, GrimReturn};
use crate::sprite_types::{RestrictionsOption, RestrictionsMinimum, OccurrenceConstraints};
use core::f64;
use std::collections::{HashMap, HashSet};
use std::iter::once;
use statrs::statistics::Statistics;
use rand::prelude::*;


// Loop iteration limits, u32 is a good choice for these counts.
const MAX_DELTA_LOOPS_LOWER: u32 = 20_000;
const MAX_DELTA_LOOPS_UPPER: u32 = 1_000_000;
const MAX_DUP_LOOPS: u32 = 20;

// This is identical to the `FUZZ_VALUE` constant already in the Canvas.
// You can either use `FUZZ_VALUE` or rename it to `DUST` if you prefer.
const DUST: f64 = 1e-12;

// A very large number, represented as a 64-bit float.
const HUGE: f64 = 1e15;
// rSprite.maxDeltaLoopsLower <- 20000
// rSprite.maxDeltaLoopsUpper <- 1000000
// rSprite.maxDupLoops <- 20
//
// rSprite.dust <- 1e-12 #To account for rounding errors
// rSprite.huge <- 1e15 #Should this not be Inf?


#[derive(Debug)]
pub enum ParameterError {
    InputValidation(String),
    Consistency(String),
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
        RestrictionsOption::Default() => Some(RestrictionsMinimum::from_range(min_val * 100, max_val * 100).extract()),
        RestrictionsOption::Opt(opt_map) => opt_map.map(|rm| rm.extract()),
        RestrictionsOption::Null() => None,
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



// #[allow(clippy::too_many_arguments)]
// #[allow(unused_variables)]
// pub fn set_parameters(
//     mean: f64, 
//     sd: f64, 
//     n_obs: u32, 
//     min_val: i32, 
//     max_val: i32, 
//     m_prec: Option<i32>, 
//     sd_prec: Option<i32>, 
//     n_items: u32, 
//     restrictions_exact: Option<HashMap<i32, usize>>,
//     restrictions_minimum: RestrictionsOption,
//     dont_test: bool) {
//
//     let m_prec: i32 = m_prec.unwrap_or_else(|| 
//         max(decimal_places_scalar(Some(&mean.to_string()), ".").unwrap() - 1, 0)
//     );
//     let sd_prec: i32 = sd_prec.unwrap_or_else(|| 
//         max(decimal_places_scalar(Some(&sd.to_string()), ".").unwrap() - 1, 0)
//     );
//
//     assert!(min_val < max_val, "min_val was not less than max_val");
//
//
//     if !dont_test 
//     && n_obs as f64 * n_items as f64 <= 10.0f64.powi(m_prec) 
//     && !grim_rust(
//         vec![&mean.to_string()], 
//         vec![n_obs], 
//         vec![false, false, false], 
//         vec![n_items], 
//         "up_or_down", 
//         5.0, 
//         f64::EPSILON.powf(0.5)
//     )[0] {
//         panic!("The mean is not consistent with this number of observations (fails GRIM test). 
// You can use grim_scalar() to identify the closest possible mean and try again")
//     } else if !dont_test
//     && !grimmer_scalar(&mean.to_string(), &sd.to_string(), n_obs, n_items, vec![false, false, false], "up_or_down", 5.0, f64::EPSILON.powf(0.5)) {
//         panic!("The standard deviation is not consistent with this mean and number of observations (fails GRIMMER test)")
//     };
//
//     let sd_limits: (f64, f64) = sd_limits(n_obs, mean, min_val, max_val, Some(sd_prec), n_items);
//
//     if !(sd > sd_limits.0 && sd <= sd_limits.1) {
//         panic!("The standard deviation is outside the possible range, givent he other parameters. 
// It should be between {} and {}.", sd_limits.0, sd_limits.1)
//     }
//
//     if !(mean >= min_val as f64 && mean <= max_val as f64) {
//         panic!("The mean is outside the possible range, which is impossible - please check inputs.")
//     };
//
//     // if it's default, construct from range. 
//     // if it's a Some(T), extract T
//     // if it's a None, do nothing?
//
//
//     let restrictions_minimum = match restrictions_minimum {
//         RestrictionsOption::Default() => Some(restrictions_minimum.construct_from_default(min_val, max_val).extract()),
//         RestrictionsOption::Opt(t) => t,
//         RestrictionsOption::Null() => None,
//     };
//
//
//
//
//
//
//
//
//     // // awkward, replicating an R pattern. Once we have a good idea what 
//     // // restrictions_minimum is used for, translate to more idiomatic form
//     // let new_restrictions_min: Option<(i32, i32)> = if let Some(res_min) = restrictions_minimum.clone() {
//     //     if res_min == "range".to_string() {
//     //         Some((1, 1))
//     //     } else {
//     //         None
//     //     }
//     // } else {
//     //     None
//     // };
//
//
//     let mut poss_values = vec![max_val as f64];
//
//     for i in 0..n_items {
//         generate_sequence(min_val, max_val, n_items, i).append(&mut poss_values)
//     }
//
//     poss_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
//
//     let poss_values_chr = poss_values.iter().map(
//         |&x| rust_round(x, 2)
//     ).collect::<Vec<f64>>();
//
//
//
//     let exact_keys: HashSet<i32> = restrictions_exact.clone().unwrap().keys().copied().collect();
//     let min_keys: HashSet<i32> = restrictions_minimum.clone().unwrap().keys().copied().collect();
//
//     if restrictions_minimum.is_some() && restrictions_exact.is_some() {
//         // check to see if their keys overlap
//
//         let conflicts: Vec<i32> = exact_keys.intersection(&min_keys).copied().collect();
//         if !conflicts.is_empty() {
//             panic!("Conflict: values with both exact and minimum restrictions: {:?}", conflicts);
//         }
//     }
//
//     if restrictions_minimum.is_some() {
//
//         let allowed_values: HashSet<i32> = poss_values_chr
//             .iter()
//             .map(|f| (f * 100.0).round() as i32) 
//             // emulate rounding to 2 decimal places
//             .collect();
//
//         let min_keys: Vec<f64> = restrictions_minimum.clone().unwrap().keys().map(|&k| k as f64).collect();
//
//         let invalid_keys: Vec<i32> = min_keys.iter().map(|k| (k * 100.0).round() as i32 / 100).filter(|k| !allowed_values.contains(k)).collect();
//
//
//         if !invalid_keys.is_empty() {
//             panic!("Invalid keys in restrictions_minimum: {:?}", invalid_keys);
//         }
//
//         // ensure restrictions are ordered
//         // take the values of poss_values_chr which also exist as the keys of restrictions_minimum
//
//         let restrictions_minimum = filter_restrictions_exact(&restrictions_minimum.unwrap().extract(), &poss_values_chr);
//
//         let fixed_responses = fixed_responses_from_minimum(&restrictions_minimum, poss_values_chr, poss_values);
//     }
//
//     if restrictions_exact.is_some() {
//         // get keys, round them to second place, if any is NOT in poss_values_chr, then ...
//         let j = restrictions_exact.unwrap().keys().map();
//         // ... translating line 154, restrictions_exact <- restrictions_exact[as.character(poss_values_chr[poss_values_chr %in% names(restrictions_exact)])]
//
//     }
//     // do the same for restrictions_exact
//
//
//
//     // if new_restrictions_min.is_some() && restrictions_exact.is_some() {
//     //
//     // }
//
//
//
//
// //restrictions_minimum <- restrictions_minimum[as.character(poss_values_chr[poss_values_chr %in% names(restrictions_minimum)])]
//
//
//     // double check implementation here
//
//
//     //poss_values <- max_val
//     //  for (i in seq_len(n_items)) {
//     //    poss_values <- c(poss_values, min_val:(max_val-1) + (1 / n_items) * (i - 1))
//     //  }
//     //  poss_values <- sort(poss_values)
//     //
//     //  poss_values_chr <- round(poss_values, 2)
//     //
//     //  fixed_responses <- numeric()
//     //  fixed_values <- NA
//
// }


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


/// Represents the outcome of the `find_possible_distribution` function.
#[derive(Debug)]
pub enum FindDistributionResult {
    Success {
        values: Vec<f64>,
        mean: f64,
        sd: f64,
        iterations: u32,
    },
    Failure {
        values: Vec<f64>,
        mean: f64,
        sd: f64,
        iterations: u32,
    },
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
) -> Result<FindDistributionResult, String> {
    // 1. Generate random starting data.
    let r_n = params.n_obs - params.n_fixed as u32;
    if params.possible_values.is_empty() && r_n > 0 {
        return Err("No possible values to sample from for initialization.".to_string());
    }
    let mut vec: Vec<f64> = (0..r_n)
        .map(|_| *params.possible_values.choose(rng).unwrap())
        .collect();

    // 2. Adjust mean of starting data.
    let max_loops_mean = params.n_obs * params.possible_values.len() as u32;
    adjust_mean(
        vec.clone(),
        &params.fixed_responses,
        &params.possible_values,
        params.mean,
        params.m_prec,
        max_loops_mean,
        rng,
    )?; // Propagate error if mean adjustment fails

    // 3. Find distribution that also matches SD.
    let max_loops_sd = (params.n_obs * (params.possible_values.len().pow(2) as u32)).clamp(MAX_DELTA_LOOPS_LOWER, MAX_DELTA_LOOPS_UPPER);
        
    let granule_sd = (0.1f64.powi(params.sd_prec)) / 2.0 + DUST;

    for i in 1..=max_loops_sd {
        let mut full_vec = vec.clone();
        full_vec.extend_from_slice(&params.fixed_responses);
        
        if let Some(current_sd) = std_dev(&full_vec) {
            if (current_sd - params.sd).abs() <= granule_sd {
                // Success!
                return Ok(FindDistributionResult::Success {
                    values: full_vec.clone(),
                    mean: mean(&full_vec),
                    sd: current_sd,
                    iterations: i,
                });
            }
        }

        // If not successful, shift values and try again.
        shift_values(&mut vec, params, rng);
    }

    // If loop finishes, we have failed to find a solution.
    let mut final_vec = vec;
    final_vec.extend_from_slice(&params.fixed_responses);
    let final_sd = std_dev(&final_vec).unwrap_or(0.0);

    Ok(FindDistributionResult::Failure {
        values: final_vec.clone(),
        mean: mean(&final_vec),
        sd: final_sd,
        iterations: max_loops_sd,
    })
}



/// Attempts to shift values within a vector to better match a target standard deviation,
/// while keeping the mean approximately constant.
///
/// This is a core part of the SPRITE algorithm's search process.
#[allow(clippy::too_many_arguments)]
pub fn shift_values(
    vec: &mut Vec<f64>,
    params: &SpriteParameters,
    rng: &mut impl Rng,
) -> bool { // Returns true if vec was modified, false otherwise.
    
    let vec_original = vec.clone();

    // Combine all possible values and sort them.
    let mut poss_values = [params.possible_values.as_slice(), params.fixed_responses.as_slice()].concat();
    poss_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    poss_values.dedup();

    if poss_values.len() < 2 { return false; } // Cannot shift if there aren't at least two options.

    // Determine direction of shift.
    let inc_first: bool = rng.random();
    let mut full_vec = vec.clone();
    full_vec.extend_from_slice(&params.fixed_responses);
    
    let current_sd = match std_dev(&full_vec) {
        Some(sd) => sd,
        None => return false, // Not enough data to calculate SD.
    };
    let increase_sd = current_sd < params.sd;

    let max_to_inc = poss_values[poss_values.len() - 2];
    let min_to_dec = poss_values[1];

    // --- First Bump ---
    
    // Find unique elements that are eligible for the first bump.
    let mut seen = HashSet::new();

    // TODO: resolve this hacky solution to the hash trait error
    let unique_indices: Vec<usize> = (0..vec.len())
        .filter(|&i| {
            let scaled_val = (vec[i] * 1_000_000.0).round() as i64;
            seen.insert(scaled_val)
        })
        .collect();
    // let unique_indices: Vec<usize> = (0..vec.len()).filter(|&i| seen.insert(vec[i])).collect();

    let mut index_can_bump1: Vec<usize> = unique_indices.into_iter().filter(|&i| {
        if inc_first { vec[i] <= max_to_inc } else { vec[i] >= min_to_dec }
    }).collect();

    if index_can_bump1.is_empty() { return false; }

    // Filter out "pointless" moves if possible.
    let pointless_filter = |i: &usize| {
        let val = vec[*i];
        if increase_sd {
            if inc_first { !equalish(val, vec.iter().copied().fold(f64::INFINITY, f64::min)) } 
            else { !equalish(val, vec.iter().copied().fold(f64::NEG_INFINITY, f64::max)) }
        } else if inc_first { 
            !equalish(val, max_to_inc) 
        } else { 
            !equalish(val, min_to_dec) 
        }
        
    };

    let better_options: Vec<usize> = index_can_bump1.iter().copied().filter(pointless_filter).collect();
    if !better_options.is_empty() {
        index_can_bump1 = better_options;
    }

    // Select a value to change.
    let &which_will_bump1 = index_can_bump1.choose(rng).unwrap();
    let will_bump1 = vec[which_will_bump1];

    // Find the new value from the list of non-restricted possibilities.
    let new1 = match params.possible_values.iter().position(|&v| equalish(v, will_bump1)) {
        Some(pos) => {
            let new_pos = if inc_first { pos + 1 } else { pos - 1 };
            params.possible_values.get(new_pos).copied()
        },
        None => None,
    };
    
    let new1 = if let Some(val) = new1 { val } else { return false; }; // Could not find a value to shift to.
    
    vec[which_will_bump1] = new1;

    // --- Check Mean and Decide on Second Bump ---

    full_vec = vec.clone();
    full_vec.extend_from_slice(&params.fixed_responses);
    let new_mean = mean(&full_vec);
    let mean_changed = !equalish(rust_round(new_mean, params.m_prec), params.mean);

    // If mean is now inconsistent OR with some probability, perform a second, compensating bump.
    if mean_changed || rng.random::<f64>() < 0.4 {
        todo!();
        // ... (Second bump logic would go here)
        // For now, if a second bump is needed but fails, we revert.
        // The full logic for the second bump is complex and involves the gap resolution.
        // A simplified approach is to revert if the mean changed.
        if mean_changed {
            *vec = vec_original;
            return false;
        }
    }

    // --- Final Check ---
    // The R code has a final check for mean drift.
    full_vec = vec.clone();
    full_vec.extend_from_slice(&params.fixed_responses);
    let final_mean = mean(&full_vec);
    if !equalish(rust_round(final_mean, params.m_prec), params.mean) {
        *vec = vec_original; // Revert on mean drift.
        return false;
    }

    // If we've reached here, the vector was successfully modified.
    true
}

/// Iteratively adjusts a vector's values to match a target mean.
///
/// This function randomly selects elements and nudges them up or down according
/// to a list of possible values until the rounded mean of the full dataset
/// matches the target.
pub fn adjust_mean(
    mut vec: Vec<f64>,
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

    for _ in 0..max_iter {
        let mut full_vec = vec.clone();
        full_vec.extend_from_slice(fixed_vals);
        let current_mean = mean(&full_vec);

        if equalish(rust_round(current_mean, m_prec), target_mean) {
            return Ok(()); // Success
        }

        let increase_mean = current_mean < target_mean;
        
        let min_poss_val = poss_values[0];
        let max_poss_val = poss_values[poss_values.len() - 1];

        // Find indices of elements in `vec` that can be bumped.
        let possible_bump_indices: Vec<usize> = vec
            .iter()
            .enumerate()
            .filter(|(_, &val)| {
                if increase_mean { val < max_poss_val } else { val > min_poss_val }
            })
            .map(|(i, _)| i)
            .collect();

        if possible_bump_indices.is_empty() {
            // No more values can be adjusted in the desired direction.
            // Break the loop and let the final error handle it.
            break;
        }

        // Randomly select an index to change.
        if let Some(&index_to_bump) = possible_bump_indices.choose(rng) {
            let current_val = vec[index_to_bump];
            
            // Find the position of the current value in the `poss_values` list.
            if let Some(pos) = poss_values.iter().position(|&p| equalish(p, current_val)) {
                let new_pos = if increase_mean { pos + 1 } else { pos.saturating_sub(1) };
                
                // Get the new value and update the vector.
                if let Some(&new_val) = poss_values.get(new_pos) {
                    vec[index_to_bump] = new_val;
                }
            }
        }
    }

    // If the loop finishes without success, return an error.
    let err_msg = if !fixed_vals.is_empty() {
        "Couldn't initialize data with correct mean. This *might* be because the restrictions cannot be satisfied."
    } else {
        "Couldn't initialize data with correct mean. This might indicate a coding error if the mean is in range."
    };
    Err(err_msg.to_string())
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
// fn scale_2dp(x: f64) -> i32 {
//     (x * 100.0).round() as i32
// }
//
// fn filter_restrictions_exact(
//     restrictions_exact: &HashMap<i32, usize>,
//     poss_values_chr: &Vec<f64>,
// ) -> HashMap<i32, usize> {
//     // Convert allowed f64 values to scaled integers
//     let allowed_scaled: HashSet<i32> = poss_values_chr.iter().map(|&v| scale_2dp(v)).collect();
//
//     // Filter: retain only keys that, when scaled, match allowed values
//     restrictions_exact
//         .iter()
//         .filter_map(|(&k, &v)| {
//             let k_scaled = scale_2dp(k as f64);
//             if allowed_scaled.contains(&k_scaled) {
//                 Some((k, v))
//             } else {
//                 None
//             }
//         })
//         .collect()
// }


// fn fixed_responses_from_minimum(
//     restrictions_minimum: &HashMap<i32, usize>,
//     poss_values_chr: Vec<f64>,
//     poss_values: Vec<f64>,
// ) -> Vec<f64> {
//     poss_values_chr
//         .into_iter()
//         .zip(poss_values.into_iter())
//         .filter_map(|(chr, val)| {
//             let scaled = scale_2dp(chr);
//             restrictions_minimum.get(&scaled).map(|&count| vec![val; count])
//         })
//         .flatten()
//         .collect()
// }
