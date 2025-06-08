//! Creating a space to experiment with a Rust translation of SPRITE

use std::cmp::max;
use crate::grimmer::{decimal_places_scalar, grim_rust, grimmer_scalar, rust_round};
use crate::sprite_types::RestrictionsOption;
use core::f64;
use std::collections::{HashMap, HashSet};
use std::iter::once;
use statrs::statistics::Statistics;


#[allow(clippy::too_many_arguments)]
#[allow(unused_variables)]
pub fn set_parameters(
    mean: f64, 
    sd: f64, 
    n_obs: u32, 
    min_val: i32, 
    max_val: i32, 
    m_prec: Option<i32>, 
    sd_prec: Option<i32>, 
    n_items: u32, 
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    dont_test: bool) {

    let m_prec: i32 = m_prec.unwrap_or_else(|| 
        max(decimal_places_scalar(Some(&mean.to_string()), ".").unwrap() - 1, 0)
    );
    let sd_prec: i32 = sd_prec.unwrap_or_else(|| 
        max(decimal_places_scalar(Some(&sd.to_string()), ".").unwrap() - 1, 0)
    );
    
    assert!(min_val < max_val, "min_val was not less than max_val");


    if !dont_test 
    && n_obs as f64 * n_items as f64 <= 10.0f64.powi(m_prec) 
    && !grim_rust(
        vec![&mean.to_string()], 
        vec![n_obs], 
        vec![false, false, false], 
        vec![n_items], 
        "up_or_down", 
        5.0, 
        f64::EPSILON.powf(0.5)
    )[0] {
        panic!("The mean is not consistent with this number of observations (fails GRIM test). 
You can use grim_scalar() to identify the closest possible mean and try again")
    } else if !dont_test
    && !grimmer_scalar(&mean.to_string(), &sd.to_string(), n_obs, n_items, vec![false, false, false], "up_or_down", 5.0, f64::EPSILON.powf(0.5)) {
        panic!("The standard deviation is not consistent with this mean and number of observations (fails GRIMMER test)")
    };

    let sd_limits: (f64, f64) = sd_limits(n_obs, mean, min_val, max_val, Some(sd_prec), n_items);

    if !(sd > sd_limits.0 && sd <= sd_limits.1) {
        panic!("The standard deviation is outside the possible range, givent he other parameters. 
It should be between {} and {}.", sd_limits.0, sd_limits.1)
    }

    if !(mean >= min_val as f64 && mean <= max_val as f64) {
        panic!("The mean is outside the possible range, which is impossible - please check inputs.")
    };

    // if it's default, construct from range. 
    // if it's a Some(T), extract T
    // if it's a None, do nothing?


    let restrictions_minimum = match restrictions_minimum {
        RestrictionsOption::Default() => Some(restrictions_minimum.construct_from_default(min_val, max_val).extract()),
        RestrictionsOption::Opt(t) => t,
        RestrictionsOption::Null() => None,
    };


    
    

    


    // // awkward, replicating an R pattern. Once we have a good idea what 
    // // restrictions_minimum is used for, translate to more idiomatic form
    // let new_restrictions_min: Option<(i32, i32)> = if let Some(res_min) = restrictions_minimum.clone() {
    //     if res_min == "range".to_string() {
    //         Some((1, 1))
    //     } else {
    //         None
    //     }
    // } else {
    //     None
    // };
    

    let mut poss_values = vec![max_val as f64];

    for i in 0..n_items {
        generate_sequence(min_val, max_val, n_items, i).append(&mut poss_values)
    }

    poss_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let poss_values_chr = poss_values.iter().map(
        |&x| rust_round(x, 2)
    ).collect::<Vec<f64>>();



    let exact_keys: HashSet<i32> = restrictions_exact.clone().unwrap().keys().copied().collect();
    let min_keys: HashSet<i32> = restrictions_minimum.clone().unwrap().keys().copied().collect();

    if restrictions_minimum.is_some() && restrictions_exact.is_some() {
        // check to see if their keys overlap

        let conflicts: Vec<i32> = exact_keys.intersection(&min_keys).copied().collect();
        if !conflicts.is_empty() {
            panic!("Conflict: values with both exact and minimum restrictions: {:?}", conflicts);
        }
    }

    if restrictions_minimum.is_some() {

        let allowed_values: HashSet<i32> = poss_values_chr
            .iter()
            .map(|f| (f * 100.0).round() as i32) 
            // emulate rounding to 2 decimal places
            .collect();

        let min_keys: Vec<f64> = restrictions_minimum.clone().unwrap().keys().map(|&k| k as f64).collect();

        let invalid_keys: Vec<i32> = min_keys.iter().map(|k| (k * 100.0).round() as i32 / 100).filter(|k| !allowed_values.contains(k)).collect();


        if !invalid_keys.is_empty() {
            panic!("Invalid keys in restrictions_minimum: {:?}", invalid_keys);
        }

        // ensure restrictions are ordered
        // take the values of poss_values_chr which also exist as the keys of restrictions_minimum
        
        let restrictions_minimum = filter_restrictions_exact(&restrictions_minimum.unwrap().extract(), &poss_values_chr);

        let fixed_responses = fixed_responses_from_minimum(&restrictions_minimum, poss_values_chr, poss_values);
    }

    if restrictions_exact.is_some() {
        // get keys, round them to second place, if any is NOT in poss_values_chr, then ...
        let j = restrictions_exact.unwrap().keys().map();

    }
    // do the same for restrictions_exact



    // if new_restrictions_min.is_some() && restrictions_exact.is_some() {
    //
    // }

    


//restrictions_minimum <- restrictions_minimum[as.character(poss_values_chr[poss_values_chr %in% names(restrictions_minimum)])]


    // double check implementation here
    

    //poss_values <- max_val
    //  for (i in seq_len(n_items)) {
    //    poss_values <- c(poss_values, min_val:(max_val-1) + (1 / n_items) * (i - 1))
    //  }
    //  poss_values <- sort(poss_values)
    //
    //  poss_values_chr <- round(poss_values, 2)
    //
    //  fixed_responses <- numeric()
    //  fixed_values <- NA

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

fn generate_sequence(min_val: i32, max_val: i32, n_items: u32, i: u32) -> Vec<f64> {
    let increment = (1.0 / n_items as f64) * (i as f64 - 1.0);
    (min_val..max_val)
        .map(|val| val as f64 + increment)
        .collect()
}
fn scale_2dp(x: f64) -> i32 {
    (x * 100.0).round() as i32
}

fn filter_restrictions_exact(
    restrictions_exact: &HashMap<i32, usize>,
    poss_values_chr: &Vec<f64>,
) -> HashMap<i32, usize> {
    // Convert allowed f64 values to scaled integers
    let allowed_scaled: HashSet<i32> = poss_values_chr.iter().map(|&v| scale_2dp(v)).collect();

    // Filter: retain only keys that, when scaled, match allowed values
    restrictions_exact
        .iter()
        .filter_map(|(&k, &v)| {
            let k_scaled = scale_2dp(k as f64);
            if allowed_scaled.contains(&k_scaled) {
                Some((k, v))
            } else {
                None
            }
        })
        .collect()
}


fn fixed_responses_from_minimum(
    restrictions_minimum: &HashMap<i32, usize>,
    poss_values_chr: Vec<f64>,
    poss_values: Vec<f64>,
) -> Vec<f64> {
    poss_values_chr
        .into_iter()
        .zip(poss_values.into_iter())
        .filter_map(|(chr, val)| {
            let scaled = scale_2dp(chr);
            restrictions_minimum.get(&scaled).map(|&count| vec![val; count])
        })
        .flatten()
        .collect()
}
