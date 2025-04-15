//! Creating a space to experiment with a Rust translation of SPRITE

use crate::grimmer::{decimal_places_scalar, rust_round, grim_rust};
use core::f64;
use std::cmp::max;
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
    restrictions_exact: Option<f64>,
    restrictions_minimum: Option<f64>, 
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
    }

    // add GRIMMER test warning
    
    let sd_limits = sd_limits(n_obs, mean, min_val, max_val, Some(sd_prec), n_items);

    if !(sd > sd_limits.0 && sd <= sd_limits.1) {
        panic!("The standard deviation is outside the possible range, givent he other parameters. 
It should be between {} and {}.", sd_limits.0, sd_limits.1)
    }

    if !(mean >= min_val as f64 && mean <= max_val as f64) {
        panic!("The mean is outside the possible range, which is impossible - please check inputs.")
    }

    // implement checkmate warning commented below
    //  if (isTRUE(checkmate::check_choice(restrictions_minimum, "range"))) {
    //    restrictions_minimum <- list(1, 1)
    //    names(restrictions_minimum) <- c(min_val, max_val)
    //  }
    
    let mut poss_values = vec![max_val as f64];

    for i in 0..n_items {
        generate_sequence(min_val, max_val, n_items, i).append(&mut poss_values)
    }

    poss_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
