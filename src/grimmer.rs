use core::f64;
use regex::Regex;
use thiserror::Error;

use bigdecimal::{BigDecimal, FromPrimitive, Pow, ToPrimitive, Zero};
use std::str::FromStr;

const _EPS: f64 = f64::EPSILON;
const FUZZ_VALUE: f64 = 1e-12;
// bool params in this case takes show_reason, default false, and symmetric, default false
// no getting around it, the original contains lots of arguments, even if we condense the bools
// into a vec, we still have 8

/// Returns the number of values after the decimal point, or else None if there are no such values,
/// no decimal, or if the string cannot be converted to a numeric type
///
/// Note that this function will only record the number of values after the first decimal point
pub(crate) fn decimal_places_scalar(x: Option<&str>, sep: &str) -> Option<i32> {
    let s = x?;

    let pattern = format!("{sep}(\\d+)");
    let re = Regex::new(&pattern).ok()?;
    let caps = re.captures(s)?;

    caps.get(1).map(|c| c.as_str().len() as i32)
}

/// Determine whether the two provided numbers are within a given tolerance of each other
pub fn is_near(num_1: f64, num_2: f64, tolerance: f64) -> bool {
    (num_1 - num_2).abs() <= tolerance
}

/// rust does not have a native function that rounds binary floating point numbers to a set number
/// of decimals. This is a hacky workaround that nevertheless seems to be the best option
pub fn rust_round(x: f64, y: i32) -> f64 {
    (x * 10.0f64.powi(y)).round() / 10.0f64.powi(y)
}

pub(crate) fn check_threshold_specified(threshold: f64) {
    if threshold == 5.0 {
        panic!("Threshold must be set to some number other than its default, 5.0");
    }
}

/// round down function
pub(crate) fn round_down(number: f64, decimals: i32) -> f64 {
    let to_round =
        (number * 10.0f64.powi(decimals + 1)) - (number * 10f64.powi(decimals)).floor() * 10.0;

    match to_round {
        5.0 => (number * 10f64.powi(decimals)).floor() / 10f64.powi(decimals),
        _ => rust_round(number, decimals),
    }
}

pub(crate) fn round_up(number: f64, decimals: i32) -> f64 {
    let to_round =
        (number * 10.0f64.powi(decimals + 1)) - (number * 10f64.powi(decimals)).floor() * 10.0;

    match to_round {
        5.0 => (number * 10f64.powi(decimals)).ceil() / 10f64.powi(decimals),
        _ => rust_round(number, decimals),
    }
}

pub(crate) fn round_trunc(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);

    //For symmetry between positive and negative numbers, use the absolute value:
    // the rust f64::trunc() function may have different properties than the R trunc() function, check to make sure

    let core = (x.abs() * p10).trunc() / p10;
    match x < 0.0 {
        true => -core,
        false => core,
    }

    //If `x` is negative, its truncated version should be negative or zero.
    //Therefore, in this case, the function returns the negative of `core`, the
    //absolute value; otherwise it simply returns `core` itself:
}

pub(crate) fn anti_trunc(x: f64) -> f64 {
    let core = x.abs().trunc() + 1.0;

    match x < 0.0 {
        true => -core,
        false => core,
    }
}

pub(crate) fn round_anti_trunc(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);
    anti_trunc(x * p10) / p10
}

pub(crate) fn round_ceiling(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);
    (x * p10).ceil() / p10
}

pub(crate) fn round_floor(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);
    (x * p10).floor() / p10
}
// not sure if x and rounding are meant to be scalars or vectors, because it seems elsewhere like
// we can pass multiple arguments to rounding, and that x can also be a vector????
// But the example values for the above function look exclusively scalars

pub(crate) fn round_up_from(x: Vec<f64>, digits: i32, threshold: f64, symmetric: bool) -> Vec<f64> {
    let p10 = 10.0f64.powi(digits);
    let threshold = threshold - f64::MIN_POSITIVE.powf(0.5);

    x.iter()
        .map(|i| round_up_from_scalar(*i, p10, threshold, symmetric))
        .collect()
}

pub(crate) fn round_up_from_scalar(x: f64, p10: f64, threshold: f64, symmetric: bool) -> f64 {
    if symmetric {
        match x < 0.0 {
            true => -(x.abs() * p10 + (1.0 - (threshold / 10.0))).floor() / p10, // - (floor(abs(x) * p10 + (1 - (threshold / 10))) / p10)
            false => (x * p10 + (1.0 - (threshold / 10.0))).floor() / p10,
        }
    } else {
        (x * p10 + (1.0 - (threshold / 10.0))).floor() / p10
    }
}

pub(crate) fn round_down_from(x: Vec<f64>, digits: i32, threshold: f64, symmetric: bool) -> Vec<f64> {
    let p10 = 10.0f64.powi(digits);
    let threshold = threshold - f64::EPSILON.powf(0.5);

    // let's make a round_down_from_scalar function that we can .map onto the vector
    //
    //
    x.iter()
        .map(|i| round_down_from_scalar(*i, p10, threshold, symmetric))
        .collect()
}

pub(crate) fn round_down_from_scalar(x: f64, p10: f64, threshold: f64, symmetric: bool) -> f64 {
    if symmetric {
        match x < 0.0 {
            true => -(x.abs() * p10 - (1.0 - (threshold / 10.0))).ceil() / p10,
            false => (x * p10 - (1.0 - (threshold / 10.0))).ceil() / p10,
        }
    } else {
        (x * p10 - (1.0 - (threshold / 10.0))).ceil() / p10

        //(x * p10 - (1-(threshold / 10))).ceil() / p10
        // ceiling(x * p10 - (1 - (threshold / 10))) / p10
    }
}

/// reconstruct_rounded_numbers fn for reround
/// notably the R version of this function can take a vector as x or a scalar, we
/// should probably turn this into taking a vector with at least one element, and that means the
/// rounding functions also need to be updated to do that
/// but both this and the vectorized version return doubles, and the same number of them, just in a
/// different format
pub(crate) fn reconstruct_rounded_numbers_scalar(
    x: f64,
    digits: i32,
    rounding: &str,
    threshold: f64,
    symmetric: bool,
) -> Vec<f64> {
    // requires the round_up and round_down functions
    match rounding {
        "up_or_down" => vec![round_up(x, digits), round_down(x, digits)], // this is supposed to
        // contain a `symmetric` argument in the R code, but that's not present in the definition
        // for round up and round down ??
        "up_from_or_down_from" => {
            check_threshold_specified(threshold); // untested
            vec![
                round_up_from(vec![x], digits, threshold, symmetric)[0], // untested
                round_down_from(vec![x], digits, threshold, symmetric)[0], // this is a hacky // untested
                                                                           // solution to suppress the errors while we're migrating this from scalar to
            ]
        }
        "ceiling_or_floor" => vec![round_ceiling(x, digits), round_floor(x, digits)],
        "even" => vec![rust_round(x, digits)],
        "up" => vec![round_up(x, digits)], // supposed to have a symmetric keyword, but round up
        // definition doesn't have it, ???
        "down" => vec![round_down(x, digits)], // supposed to have a symmetric keyword, but round down definition doesn't have it ??? // untested
        "up_from" => {
            check_threshold_specified(threshold);
            round_up_from(vec![x], digits, threshold, symmetric)
        }
        "down_from" => {
            check_threshold_specified(threshold);
            vec![round_down_from(vec![x], digits, threshold, symmetric)[0]]
        }
        "ceiling" => vec![round_ceiling(x, digits)],
        "floor" => vec![round_floor(x, digits)], // untested
        "trunc" => vec![round_trunc(x, digits)], // untested
        "anti_trunc" => vec![round_anti_trunc(x, digits)], // untested
        _ => panic!("`rounding` must be one of the designated string keywords"), // untested
    }
}

pub(crate) fn reround(
    x: Vec<f64>,
    digits: i32,
    rounding: &str,
    threshold: f64,
    symmetric: bool,
) -> Vec<f64> {
    x.iter()
        .flat_map(|&x| {
            reconstruct_rounded_numbers_scalar(x, digits, rounding, threshold, symmetric)
        })
        .collect()
}

#[allow(dead_code)]
pub enum GrimReturn {
    Bool(bool),
    List(bool, f64, Vec<f64>, Vec<f64>, f64, f64),
}

#[derive(Debug, Error)]
pub enum GrimScalarError {
    #[error("Could not parse x into a number")]
    ParseFloatError,
    #[error("Could not extract decimal places")]
    DecimalNullError(String),
}

/// Performs GRIM test of a single number
///
/// We test whether the provided mean is within a plausible rounding of any possible means given
/// the number of samples
pub fn grim_scalar_rust(
    x: &str,
    n: u32,
    bool_params: Vec<bool>, // includes percent, show_rec, and symmetric
    items: u32,
    rounding: &str,
    threshold: f64,
    tolerance: f64,
) -> Result<GrimReturn, GrimScalarError> {
    let percent: bool = bool_params[0];
    let show_rec: bool = bool_params[1];
    let symmetric: bool = bool_params[2];

    // [BigDecimal] Parse the input string directly into a high-precision decimal
    // This avoids any initial f64 representation error
    let mut x_bd = match BigDecimal::from_str(x) {
        Ok(val) => val,
        Err(_) => return Err(GrimScalarError::ParseFloatError),
    };

    let Some(mut digits): Option<i32> = decimal_places_scalar(Some(x), ".") else {
        return Err(GrimScalarError::DecimalNullError(x.to_string()));
    };

    if percent {
        // [BigDecimal] Perform division with high precision.
        x_bd /= 100;
        digits += 2;
    };

    // [BigDecimal] Convert other numbers to BigDecimal for the calculation
    let n_bd = BigDecimal::from_u32(n).unwrap();
    let items_bd = BigDecimal::from_u32(items).unwrap();
    let n_items_bd = &n_bd * &items_bd;

    // [BigDecimal] The core multiplication is now done with decimal arithmetic,
    // preventing the amplification of binary floating-point errors
    let rec_sum_bd = &x_bd * &n_items_bd;

    // [BigDecimal] Perform ceiling and floor operations on the high-precision result
    let rec_sum_ceil = rec_sum_bd.with_scale_round(0, bigdecimal::RoundingMode::Ceiling);
    let rec_sum_floor = rec_sum_bd.with_scale_round(0, bigdecimal::RoundingMode::Floor);

    // [BigDecimal] Calculate the reconstructed means, still in high precision
    let rec_x_upper_bd = &rec_sum_ceil / &n_items_bd;
    let rec_x_lower_bd = &rec_sum_floor / &n_items_bd;

    // Convert back to f64 to interface with the existing `dustify` and `reround` logic
    let rec_x_upper = dustify(rec_x_upper_bd.to_f64().unwrap());
    let rec_x_lower = dustify(rec_x_lower_bd.to_f64().unwrap());

    let conc: Vec<f64> = rec_x_upper
        .iter()
        .cloned()
        .chain(rec_x_lower.iter().cloned())
        .collect();

    let grains_rounded = reround(conc, digits, rounding, threshold, symmetric);

    // The final check still uses the original f64 value of the mean, which is correct
    let x_num: f64 = x.parse().unwrap();
    let bools: Vec<bool> = grains_rounded
        .clone()
        .into_iter()
        .map(|val| is_near(val, x_num, tolerance))
        .collect();

    let grain_is_x: bool = bools.iter().any(|&b| b);

    if !show_rec {
        Ok(GrimReturn::Bool(grain_is_x))
    } else {
        let length_2ers = ["up_or_down", "up_from_or_down_from", "ceiling_or_floor"];

        if length_2ers.contains(&rounding) {
            Ok(GrimReturn::List(
                grain_is_x,
                rec_sum_bd.to_f64().unwrap(), // Convert final sum back to f64 for return
                rec_x_upper,
                rec_x_lower,
                grains_rounded[0],
                grains_rounded[1],
            ))
        } else {
            Ok(GrimReturn::Bool(grain_is_x))
        }
    }
}

/// Determines whether a standard deviation is possible from the listed mean and sample size
///
/// Implements L. Jung's adaptation of A. Allard's A-GRIMMER algorithm for testing the possibility
/// of standard deviations. (https://aurelienallard.netlify.app/post/anaytic-grimmer-possibility-standard-deviations/).
///
/// # Arguments
/// x: the sample mean
/// sd: sample standard deviation
/// n: sample size
/// items: number of items
/// bool_params: booleans for options in GRIMMER and the underlying GRIM function, in the form
/// [percent, show_reason, symmetric]
/// rounding: method of rounding
/// threshold: rounding threshold, ordinarily 5.0
/// tolerance: rounding tolerance usually the square root of machine epsilon
///
/// # Panics
/// If x or sd are given as numbers instead of strings. This is necessary in order to
///     preserve trailing 0s
/// If items is not 1. Items > 1 may be implemented in a later update
///
/// # Returns
#[allow(clippy::too_many_arguments)]
pub fn grimmer_scalar(
    x: &str,
    sd: &str,
    n: u32,
    items: u32,
    bool_params: Vec<bool>,
    rounding: &str,
    threshold: f64,
    tolerance: f64,
) -> bool {
    if items != 1 {
        todo!("GRIMMER for items > 1 is not yet implemented");
    };

    let show_rec: bool = bool_params[1];
    let symmetric: bool = bool_params[2];

    // Pass-through to GRIM test, which is assumed to be correct
    if let Ok(GrimReturn::Bool(false)) = grim_scalar_rust(
        x,
        n,
        bool_params.clone(),
        items,
        rounding,
        threshold,
        tolerance,
    ) {
        if show_rec {
            println!("{x} is GRIM inconsistent");
        }
        return false;
    };

    // [BigDecimal] Convert all inputs to BigDecimal for high-precision decimal arithmetic
    let x_bd = BigDecimal::from_str(x).unwrap();
    let sd_bd = BigDecimal::from_str(sd).unwrap();
    let n_bd = BigDecimal::from_u32(n).unwrap();
    let items_bd = BigDecimal::from_u32(items).unwrap();
    let n_items_bd = &n_bd * &items_bd;

    // [BigDecimal] Calculate the real sum and mean using decimal math
    let sum = &x_bd * &n_items_bd;
    let sum_real = sum.with_scale_round(0, bigdecimal::RoundingMode::HalfUp); // Round to 0 decimal places
    let x_real = &sum_real / &n_items_bd;

    let digits_sd = decimal_places_scalar(Some(sd), ".").unwrap();

    // [BigDecimal] Calculate sd bounds using decimal math

    let p10 = Pow::pow(10u32, digits_sd as u32 + 1);

    let p10_frac = BigDecimal::from(5) / p10;
    let sd_lower = (&sd_bd - &p10_frac).max(BigDecimal::zero());
    let sd_upper = &sd_bd + &p10_frac;

    // [BigDecimal] Calculate sum of squares bounds using decimal math
    let n_minus_1_bd = BigDecimal::from_u32(n - 1).unwrap();
    let items_pow2_bd = items_bd.square();

    let sum_squares_lower =
        (&n_minus_1_bd * sd_lower.square() + &n_bd * x_real.square()) * &items_pow2_bd;
    let sum_squares_upper =
        (&n_minus_1_bd * sd_upper.square() + &n_bd * x_real.square()) * &items_pow2_bd;

    // [BigDecimal] Use ceil() and floor() on BigDecimal
    let pass_test1 = sum_squares_lower.with_scale_round(0, bigdecimal::RoundingMode::Ceiling)
        <= sum_squares_upper.with_scale_round(0, bigdecimal::RoundingMode::Floor);

    if !pass_test1 {
        if show_rec {
            println!("Failed test 1");
        }
        return false;
    };

    // The rest of the logic involves converting back to f64 for the rerounding part,
    // as that logic is complex and less sensitive to the initial amplification.
    // The critical amplification step has been fixed.
    let integers_possible: Vec<u32> = (sum_squares_lower
        .with_scale_round(0, bigdecimal::RoundingMode::Ceiling)
        .to_u32()
        .unwrap()
        ..=sum_squares_upper
            .with_scale_round(0, bigdecimal::RoundingMode::Floor)
            .to_u32()
            .unwrap())
        .collect();
    let x_real_f64 = x_real.to_f64().unwrap();

    let sd_predicted: Vec<f64> = integers_possible
        .iter()
        .map(|x| {
            (((*x as f64 / items.pow(2) as f64) - n as f64 * x_real_f64.powi(2)) / (n as f64 - 1.0))
                .powf(0.5)
        })
        .collect();

    let sd_rec_rounded = reround(sd_predicted, digits_sd, rounding, threshold, symmetric);
    let sd_f64: f64 = sd.parse().unwrap();
    let sd_dusted = dustify(sd_f64);
    let sd_rec_rounded_dusted: Vec<f64> = sd_rec_rounded.into_iter().flat_map(dustify).collect();
    let matches_sd: Vec<bool> = sd_dusted
        .iter()
        .zip(sd_rec_rounded_dusted.iter())
        .map(|(i, sdr)| is_near(*i, *sdr, f64::EPSILON.powf(0.5)))
        .collect();
    let pass_test2: bool = matches_sd.iter().any(|&b| b);

    if !pass_test2 {
        if show_rec {
            println!("Failed test 2");
        };
        return false;
    }

    let sum_parity = sum_real % BigDecimal::from(2);

    let matches_parity: Vec<bool> = integers_possible
        .iter()
        .map(|&n| BigDecimal::from(n) % BigDecimal::from(2) == sum_parity)
        .collect();

    let matches_sd_and_parity: Vec<bool> = matches_sd
        .iter()
        .zip(matches_parity)
        .map(|(s, p)| s & p)
        .collect();

    let pass_test3 = matches_sd_and_parity.iter().any(|&b| b);

    if !pass_test3 {
        if show_rec {
            println!("Failed test 3");
        };
        return false;
    }

    if show_rec {
        println!("Passed all tests");
        true
    } else {
        true
    }
}

#[allow(clippy::too_many_arguments)]

#[allow(clippy::too_many_arguments)]
/// Determines the possibility of standard deviations from given means and sample sizes using the A-GRIMMER algorithm.
///
/// This function implements L. Jung's adaptation of A. Allard's A-GRIMMER algorithm for testing the possibility of standard deviations. It processes multiple sets of means, standard deviations, and sample sizes, returning a boolean vector indicating the possibility for each set.
///
/// # Arguments
/// * `xs` - A vector of strings representing the sample means. Trailing zeros are preserved by using strings.
/// * `sds` - A vector of strings representing the sample standard deviations. Trailing zeros are preserved by using strings.
/// * `ns` - A vector of unsigned integers representing the sample sizes.
/// * `rounding` - A string specifying the method of rounding to be used.
/// * `items` - A vector of unsigned integers representing the number of items. Default is a vector with a single element [1].
/// * `percent` - A boolean indicating whether to treat the means as percentages. Default is false.
/// * `show_reason` - A boolean indicating whether to print the reason for failure if the tests do not pass. Default is false.
/// * `threshold` - A floating-point number representing the rounding threshold. Default is 5.0.
/// * `symmetric` - A boolean indicating whether to use symmetric rounding. Default is false.
/// * `tolerance` - A floating-point number representing the rounding tolerance, usually the square root of machine epsilon. Default is `f64::EPSILON.powf(0.5)`.
///
/// # Returns
/// A vector of booleans where each element corresponds to a set of inputs, indicating whether the standard deviation is possible for that set.
///
/// # Panics
/// The function will panic if the lengths of `xs`, `sds`, and `ns` do not match.

/// Fuzzes the value of a float by 1e-12
///
/// Parameters:
///     x: floating-point number
///
/// Returns:
///     a vector of 2 floating-point numbers
///
/// Raises:
///     ValueError: If x is not a floating-point number
pub(crate) fn dustify(x: f64) -> Vec<f64> {
    vec![x - FUZZ_VALUE, x + FUZZ_VALUE]
}
