//! Creating a space to experiment with a Rust translation of SPRITE

use std::cmp::max;
use crate::grimmer::{decimal_places_scalar, grim_scalar_rust, grimmer_scalar, rust_round, GrimReturn};
use crate::sprite_types::{RestrictionsOption, RestrictionsMinimum, OccurrenceConstraints};
use core::f64;
use std::collections::{HashMap, HashSet};
use std::iter::once;
use statrs::statistics::Statistics;


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

fn equalish(x: f64, y: f64) -> bool {
  x <= (y + DUST) &&
    x >= (y - DUST)
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
