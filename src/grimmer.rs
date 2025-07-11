//! A place to store the GRIMMER and helper functions in Rust, copied over from scrutipy 


use core::f64;

//  146-148, 150, 169-171, 173, 180, 186, 198-201, 204-211, 217-219
const EPS: f64 = f64::EPSILON;

use regex::Regex;
// 124-125, 127-128, 130-132, 134-138
const FUZZ_VALUE: f64 = 1e-12;

use std::num::ParseFloatError;

use thiserror::Error;
// bool params in this case takes show_reason, default false, and symmetric, default false
// no getting around it, the original contains lots of arguments, even if we condense the bools
// into a vec, we still have 8

/// Determines whether a standard deviation is possible from the listed mean and sample size
///
/// Implements L. Jung's adaptation of A. Allard's A-GRIMMER algorithm for testing the possibility
/// of standard deviations. (https://aurelienallard.netlify.app/post/anaytic-grimmer-possibility-standard-deviations/).
///
/// # Arguments
///     x: the sample mean
///     sd: sample standard deviation
///     n: sample size
///     items: number of items
///     bool_params: booleans for options in GRIMMER and the underlying GRIM function, in the form
///     [percent, show_reason, symmetric]
///     rounding: method of rounding
///     threshold: rounding threshold, ordinarily 5.0
///     tolerance: rounding tolerance usually the square root of machine epsilon
///
/// # Panics
///     - If x or sd are given as numbers instead of strings. This is necessary in order to
///     preserve trailing 0s
///     - If items is not 1. Items > 1 may be implemented in a later update
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
    // in the original, items does not work as intended, message Jung about this once I have more
    // time
    if items != 1 {
        todo!("GRIMMER for items > 1 is not yet implemented")
    };

    let digits_sd = decimal_places_scalar(Some(sd), ".").unwrap();

    let _percent: bool = bool_params[0];
    let show_rec: bool = bool_params[1];
    let symmetric: bool = bool_params[2];

    let grim_return = grim_scalar_rust(
        x,
        n,
        bool_params.clone(),
        items,
        rounding,
        threshold,
        tolerance,
    );

    let pass_grim = match grim_return {
        Ok(grim_return) => match grim_return {
            GrimReturn::Bool(b) => b,
            GrimReturn::List(a, _, _, _, _, _) => a,
        },
        Err(_) => panic!(),
    };

    let n_items = n * items;

    let x: f64 = x.parse().unwrap();

    let sum = x * f64::from(n_items) ;
    let sum_real = rust_round(sum, 0);
    let x_real = sum_real / f64::from(n_items) ;

    if !pass_grim {
        if show_rec {
            println!("{x} is GRIM inconsistent")
        };
        return false;
    };

    let p10 = 10.0f64.powi(digits_sd + 1i32);
    let p10_frac = 5.0 / p10;

    let sd: f64 = sd.parse().unwrap(); // why can't this be a ?

    let sd_lower = (sd - p10_frac).max(0.0); // returning 0 if p10_frac is greater than sd and
                                             // would thus return a negative number

    let sd_upper = sd + p10_frac;

    let sum_squares_lower =
        (f64::from(n - 1) * sd_lower.powi(2) + f64::from(n) * x_real.powi(2)) * f64::from(items.pow(2)) ;
    let sum_squares_upper =
        (f64::from(n - 1) * sd_upper.powi(2) + f64::from(n) * x_real.powi(2)) * f64::from(items.pow(2)) ;

    let pass_test1: bool = sum_squares_lower.ceil() <= sum_squares_upper.floor();

    if !pass_test1 {
        if show_rec {
            println!("Failed test 1");
            return false;
        };
        return false;
    };

    let integers_possible: Vec<u32> =
        (sum_squares_lower.ceil() as u32..=sum_squares_upper.floor() as u32).collect();

    let sd_predicted: Vec<f64> = integers_possible
        .iter()
        .map(|x| {
            (((*x as f64 / items.pow(2) as f64) - n as f64 * x_real.powi(2)) / (n as f64 - 1.0))
                .powf(0.5)
        })
        .collect();

    let sd_rec_rounded = reround(sd_predicted, digits_sd, rounding, threshold, symmetric);

    let sd = dustify(sd);

    let sd_rec_rounded: Vec<f64> = sd_rec_rounded.into_iter().flat_map(dustify).collect();

    let matches_sd: Vec<bool> = sd
        .iter()
        .zip(sd_rec_rounded.iter())
        .map(|(i, sdr)| is_near(*i, *sdr, EPS.powf(0.5)))
        .collect();

    let pass_test2: bool = matches_sd.iter().any(|&b| b);

    if !pass_test2 {
        if show_rec {
            println!("Failed test 2");
            return false;
        };
        return false;
    }

    let sum_parity = sum_real % 2.0;

    let matches_parity: Vec<bool> = integers_possible
        .iter()
        .map(|&n| n as f64 % 2.0 == sum_parity)
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
            return false;
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
pub fn grimmer(
    xs: Vec<String>,
    sds: Vec<String>,
    ns: Vec<u32>,
    rounding: String,
    items: Vec<u32>,
    percent: bool,
    show_reason: bool,
    threshold: f64,
    symmetric: bool,
    tolerance: f64,
) -> Vec<bool> {
    let bool_params = vec![percent, show_reason, symmetric];
    let xs: Vec<&str> = xs.iter().map(|s| &**s).collect();
    let sds: Vec<&str> = sds.iter().map(|s| &**s).collect();

    grimmer_rust(
        xs,
        sds,
        ns,
        items,
        bool_params,
        rounding.as_str(),
        threshold,
        tolerance,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn grimmer_rust(
    xs: Vec<&str>,
    sds: Vec<&str>,
    ns: Vec<u32>,
    items: Vec<u32>,
    bool_params: Vec<bool>,
    rounding: &str,
    threshold: f64,
    tolerance: f64,
) -> Vec<bool> {
    xs.iter()
        .zip(sds.iter())
        .zip(ns.iter())
        .zip(items.iter())
        .map(|(((x, sd), n), item)| {
            grimmer_scalar(
                x,
                sd,
                *n,
                *item,
                bool_params.clone(),
                rounding,
                threshold,
                tolerance,
            )
        })
        .collect()
}


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
pub fn dustify(x: f64) -> Vec<f64> {
    vec![x - FUZZ_VALUE, x + FUZZ_VALUE]
}

/// Returns the number of values after the decimal point, or else None if there are no such values,
/// no decimal, or if the string cannot be converted to a numeric type
///
/// Note that this function will only record the number of values after the first decimal point
/// ```
/// let num_decimals = crate::decimal_places_scalar(Some("1.52.1"), ".");
/// assert_eq!(num_decimals, Some(2));
/// ```
pub fn decimal_places_scalar(x: Option<&str>, sep: &str) -> Option<i32> {
    let s = x?;

    let pattern = format!("{sep}(\\d+)");
    let re = Regex::new(&pattern).ok()?;
    let caps = re.captures(s)?;

    caps.get(1).map(|c| c.as_str().len() as i32)
}


#[derive(Debug, Error)]
pub enum ReconstructSdError {
    #[error("{0} is not a number")]
    NotANumber(String),
    #[error("{0} is not a formula")]
    NotAFormula(String),
    #[error("Inputs to reconstruct_sd_scalar failed. The reconstruction formula {0} was called but resulted in a {1}")]
    SdBinaryError(String, SdBinaryError),
    // complete this so that the error propagates from the function below
}

pub fn reconstruct_sd_scalar(
    formula: &str,
    x: &str,
    n: u32,
    zeros: u32,
    ones: u32,
) -> Result<f64, ReconstructSdError> {
    let x_num: f64 = match x.parse() {
        Ok(num) => num,
        Err(string) => return Err(ReconstructSdError::NotANumber(string.to_string())),
    };

    let sd_rec: Result<f64, SdBinaryError> = match formula {
        "mean_n" => sd_binary_mean_n(x_num, n),
        "mean" => sd_binary_mean_n(x_num, n), // convenient aliases
        "0_n" => sd_binary_0_n(zeros, n),
        "0" => sd_binary_0_n(zeros, n), // convenient aliases
        "1_n" => sd_binary_1_n(ones, n),
        "1" => sd_binary_1_n(ones, n), // convenient aliases
        "groups" => sd_binary_groups(zeros, ones),
        "group" => sd_binary_groups(zeros, ones), // convenient aliases
        _ => return Err(ReconstructSdError::NotAFormula(formula.to_string())),
    };

    match sd_rec {
        Ok(num) => Ok(num),
        Err(e) => Err(ReconstructSdError::SdBinaryError(formula.to_string(), e)),
    }
}

//to_round <- number * 10^(decimals + 1) - floor(number * 10^(decimals)) * 10
//    number_rounded <- ifelse(to_round == 5,
//                             floor(number * 10^decimals) / 10^decimals,
//                             round(number, digits = decimals))
//    return(number_rounded)

pub fn check_threshold_specified(threshold: f64) {
    if threshold == 5.0 {
        panic!("Threshold must be set to some number other than its default, 5.0");
    }
}

/// reconstruct_rounded_numbers fn for reround
/// notably the R version of this function can take a vector as x or a scalar, we
/// should probably turn this into taking a vector with at least one element, and that means the
/// rounding functions also need to be updated to do that
/// but both this and the vectorized version return doubles, and the same number of them, just in a
/// different format
pub fn reconstruct_rounded_numbers_scalar(
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

/// the reround function
/// redo so that it just takes a vec instead of a vec of vec, the R version can take arbitrarily
/// nested vectors, but it's the same as just flattening the input into a single vector
pub fn reround(
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
        // this is the root, we need to have it zip across multiple rounding options, returning a
    // vec of vec, where the inner vector includes all the rounded numbers for a given rounding
    // scheme, and the outer vector includes a vector for each rounding scheme
   
}

/// check rounding singular, necessary for the reround function
pub fn check_rounding_singular(
    rounding: Vec<&str>,
    bad: &str,
    good1: &str,
    good2: &str,
) -> Result<(), String> {
    if rounding.contains(&bad) {
        Err(format!("If rounding has length > 1, only single rounding procedures are supported, such as {good1} and {good2}. Instead, rounding was given as {bad} plus others. You can still concatenate multiple of them; just leave out those with 'or'."))
    } else {
        Ok(())
    }
}

/// round down function
pub fn round_down(number: f64, decimals: i32) -> f64 {
    let to_round =
        (number * 10.0f64.powi(decimals + 1)) - (number * 10f64.powi(decimals)).floor() * 10.0;

    match to_round {
        5.0 => (number * 10f64.powi(decimals)).floor() / 10f64.powi(decimals),
        _ => rust_round(number, decimals),
    }
}

pub fn round_up(number: f64, decimals: i32) -> f64 {
    let to_round =
        (number * 10.0f64.powi(decimals + 1)) - (number * 10f64.powi(decimals)).floor() * 10.0;

    match to_round {
        5.0 => (number * 10f64.powi(decimals)).ceil() / 10f64.powi(decimals),
        _ => rust_round(number, decimals),
    }
}

/// rust does not have a native function that rounds binary floating point numbers to a set number
/// of decimals. This is a hacky workaround that nevertheless seems to be the best option.
pub fn rust_round(x: f64, y: i32) -> f64 {
    (x * 10.0f64.powi(y)).round() / 10.0f64.powi(y)
}

pub fn round_trunc(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);

    //For symmetry between positive and negative numbers, use the absolute value:
    let core = (x.abs() * p10).trunc() / p10; // the rust f64::trunc() function may have
                                              // different properties than the R trunc() function, check to make sure
    match x < 0.0 {
        true => -core,
        false => core,
    }

    //If `x` is negative, its truncated version should be negative or zero.
    //Therefore, in this case, the function returns the negative of `core`, the
    //absolute value; otherwise it simply returns `core` itself:
}

pub fn anti_trunc(x: f64) -> f64 {
    let core = x.abs().trunc() + 1.0;

    match x < 0.0 {
        true => -core,
        false => core,
    }
}

/// a function to return any function to its decimal portion, used in unit tests in the original R
/// library
pub fn trunc_reverse(x: f64) -> f64 {
    x - x.trunc()
}

pub fn round_anti_trunc(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);
    anti_trunc(x * p10) / p10
}

pub fn round_ceiling(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);
    (x * p10).ceil() / p10
}

pub fn round_floor(x: f64, digits: i32) -> f64 {
    let p10 = 10.0f64.powi(digits);
    (x * p10).floor() / p10
}
// not sure if x and rounding are meant to be scalars or vectors, because it seems elsewhere like
// we can pass multiple arguments to rounding, and that x can also be a vector????
// But the example values for the above function look exclusively scalars

pub fn round_up_from(x: Vec<f64>, digits: i32, threshold: f64, symmetric: bool) -> Vec<f64> {
    let p10 = 10.0f64.powi(digits);
    let threshold = threshold - f64::MIN_POSITIVE.powf(0.5);

    x.iter()
        .map(|i| round_up_from_scalar(*i, p10, threshold, symmetric))
        .collect()
}

pub fn round_up_from_scalar(x: f64, p10: f64, threshold: f64, symmetric: bool) -> f64 {
    if symmetric {
        match x < 0.0 {
            true => -(x.abs() * p10 + (1.0 - (threshold / 10.0))).floor() / p10, // - (floor(abs(x) * p10 + (1 - (threshold / 10))) / p10)
            false => (x * p10 + (1.0 - (threshold / 10.0))).floor() / p10,
        }
    } else {
        (x * p10 + (1.0 - (threshold / 10.0))).floor() / p10
    }
}

pub fn round_down_from(x: Vec<f64>, digits: i32, threshold: f64, symmetric: bool) -> Vec<f64> {
    let p10 = 10.0f64.powi(digits);
    let threshold = threshold - f64::EPSILON.powf(0.5);

    // let's make a round_down_from_scalar function that we can .map onto the vector
    //
    //
    x.iter()
        .map(|i| round_down_from_scalar(*i, p10, threshold, symmetric))
        .collect()
}

pub fn round_down_from_scalar(x: f64, p10: f64, threshold: f64, symmetric: bool) -> f64 {
    //let p10 = 10.0f64.powi(digits);
    //let threshold = threshold - f64::MIN_POSITIVE.powf(0.5);

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

// now do the sd_binary functions, originally from the sd-binary.R file not utils, but for now we
// can keep them here, they're short enough
// 66, 70, 94, 98, 122, 126
/// Returns the standard deviation of binary value counts
///
/// Parameters:
///     zeros: count of observations in the 0-binary condition
///     ones: count of observations in the 1-binary condition
///
/// Returns:
///     the floating-point standard deviation of the binary groups
///
/// Raises:
///     ValueError is zeros or ones are not usigned integers
///
/// Panics:
///     If the total number of observations is not greater than one
pub fn sd_binary_groups(zeros: u32, ones: u32) -> Result<f64, SdBinaryError> {
    // though we take in the counts as unsigned integers, we transform them into
    // floating point values in order to perform the
    let n: f64 = f64::from(zeros) + f64::from(ones);

    if n < 2.0 {
        return Err(SdBinaryError::InsufficientObservationsError);
    }
    // sqrt((n / (n - 1)) * ((group_0 * group_1) / (n ^ 2)))
    Ok((n / (n - 1.0) * (f64::from(zeros * ones) / n.powi(2))).sqrt())

    //sqrt((n / (n - 1)) * ((group_0 * group_1) / (n ^ 2)))
}

#[derive(Debug, Error, PartialEq)]
pub enum SdBinaryError {
    #[error("There cannot be more observations {0} in one condition than in the whole system {1}")]
    ObservationCountError(u32, u32),
    #[error("There must be at least two observations")]
    InsufficientObservationsError,
    #[error("The mean of binary observations cannot be less than 0.0")]
    NegativeBinaryMeanError,
    #[error("The mean of binary observations cannot be greater than 1.0")]
    InvalidBinaryError,
}

/// Returns the standard deviation of binary variables from the count of zero values and the total
///
/// Parameters:
///     zeros: count of observations in the 0-binary condition
///     n: count of total observations
///
/// Returns:
///     the floating-point standard deviation of the binary groups
///
/// Raises:
///     ValueError: if zeros or n are not unsigned integers
///
/// Panics:
///     If there are more observations in the zero condition than in the total
///     If the total number of observations is not greater than one
pub fn sd_binary_0_n(zeros: u32, n: u32) -> Result<f64, SdBinaryError> {
    let ones: f64 = f64::from(n) - f64::from(zeros);

    if n < zeros {
        return Err(SdBinaryError::ObservationCountError(zeros, n));
    }

    if n < 2 {
        return Err(SdBinaryError::InsufficientObservationsError);
    }

    Ok(((f64::from(n) / f64::from(n - 1) ) * ((f64::from(zeros) * ones) / (f64::from(n) ).powi(2))).sqrt())
}
/// Returns the standard deviation of binary variables from the count of one values and the total
///
/// Parameters:
///     ones: count of observations in the 1-binary condition
///     n: count of total observations
///
/// Returns:
///     the floating-point standard deviation of the binary groups
///
/// Raises:
///     ValueError: if ones or n are not unsigned integers
///
/// Panics:
///     If there are more observations in the one condition than in the total
///     If the total number of observations is not greater than one
pub fn sd_binary_1_n(ones: u32, n: u32) -> Result<f64, SdBinaryError> {
    let zeros: f64 = f64::from(n) - f64::from(ones);

    if n < ones {
        return Err(SdBinaryError::ObservationCountError(ones, n));
    }

    if n < 2 {
        return Err(SdBinaryError::InsufficientObservationsError);
    }

    Ok(((f64::from(n) / f64::from(n - 1) ) * ((zeros * f64::from(ones) ) / (f64::from(n) ).powi(2))).sqrt())
}
/// Returns the standard deviation of binary variables from the mean and the total
///
/// Parameters:
///     mean: mean of the binary observations, namely the proportion of values in the 1-binary
///     condition
///     n: count of total observations
///
/// Returns:
///     the floating-point standard deviation of the binary system
///
/// Raises:
///     ValueError: if mean is not a floating-point number
///     ValueError: if n is not an unsigned integer
///
/// Panics:
///     if the mean is greater than one or less than zero
pub fn sd_binary_mean_n(mean: f64, n: u32) -> Result<f64, SdBinaryError> {
    if mean < 0.0 {
        return Err(SdBinaryError::NegativeBinaryMeanError);
    }

    if mean > 1.0 {
        return Err(SdBinaryError::InvalidBinaryError);
    }

    Ok(((f64::from(n) / f64::from(n - 1) ) * (mean * (1.0 - mean))).sqrt())
}


pub enum GRIMInput {
    Str(String),
    Num(f64), // this captures input integer and coerces it into a string if possible, in order to
              // deal with user error on the Python interface
}
/// reproducing scrutiny's grim_scalar() function, albeit with slightly different order of
/// arguments, because unlike R, Python requires that all the positional parameters be provided up
/// front before optional arguments with defaults
#[allow(clippy::too_many_arguments)]
pub fn grim_scalar(
    x: GRIMInput,
    n: u32,
    rounding: String,
    items: u32,
    percent: bool,
    show_rec: bool,
    threshold: f64,
    symmetric: bool,
    tolerance: f64,
) -> bool {
    let x: String = match x {
        GRIMInput::Str(s) => s,
        GRIMInput::Num(n) => format!("{n}"),
    };
    // accounting for the possibility that we might receive either a String or numeric type,
    // turning the numeric possibility into a String, which we later turn into a &str to
    // pass into grim_scalar_rust()

    //let round: &str = rounding.as_str();
                                                                     // turn Vec<String> to Vec<&str>
    let val = grim_scalar_rust(
        x.as_str(),
        n,
        vec![percent, show_rec, symmetric],
        items,
        rounding.as_str(),
        threshold,
        tolerance,
    );

    match val {
        Ok(r) => match r {
            GrimReturn::Bool(b) => b,
            GrimReturn::List(a, _, _, _, _, _) => a,
        },
        Err(_) => panic!(),
    }
}

pub enum GrimReturn {
    Bool(bool),
    List(bool, f64, Vec<f64>, Vec<f64>, f64, f64),
    //
    //
    //
    //List(
    // bool,
    //f64,
    //Vec<f64>,
    //Vec<f64>,
    //Vec<f64>,
    //Vec<f64>,
    //Vec<f64>,
    //Vec<f64>,
    //),
    //
}

// vector wrapper for grim_scalar_rust
pub fn grim_rust(
    xs: Vec<&str>,
    ns: Vec<u32>,
    bool_params: Vec<bool>,
    items: Vec<u32>,
    rounding: &str,
    threshold: f64,
    tolerance: f64,
) -> Vec<bool> {

    let vals: Vec<Result<GrimReturn, GrimScalarError>> = xs
        .iter()
        .zip(ns.iter())
        .zip(items.iter())
        .map(|((x, num), item)| {
            grim_scalar_rust(
                x,
                *num,
                bool_params.clone(),
                *item,
                rounding,
                threshold,
                tolerance,
            )
        })
        .collect();
    vals.iter()
        .map(|grim_result| match grim_result {
            Ok(grim_return) => match grim_return {
                GrimReturn::Bool(b) => *b,
                GrimReturn::List(a, _, _, _, _, _) => *a,
            },
            Err(_) => panic!(),
        })
        .collect()
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

    // let mut x_num: f64 = x.parse()?;

    let Ok(mut x_num): Result<f64, ParseFloatError> = x.parse() else {
        return Err(GrimScalarError::ParseFloatError)
    };

    //let mut digits: i32 = decimal_places_scalar(Some(x), ".").unwrap();

    let Some(mut digits): Option<i32> = decimal_places_scalar(Some(x), ".") else {
        return Err(GrimScalarError::DecimalNullError("".to_string()));
    };

    if percent {
        x_num /= 100.0;
        digits += 2;
    };

    let n_items = n * items;

    let rec_sum = x_num * f64::from(n_items) ;

    let rec_x_upper = dustify(rec_sum.ceil() / f64::from(n_items) );
    let rec_x_lower = dustify(rec_sum.floor() / f64::from(n_items) );

    let conc: Vec<f64> = rec_x_upper
        .iter()
        .cloned()
        .chain(rec_x_lower.iter().cloned())
        .collect();
    //note that this modifies in place, so we just use rec_x_upper as the input to grains_rounded

    let grains_rounded = reround(conc, digits, rounding, threshold, symmetric);

    let bools: Vec<bool> = grains_rounded
        .clone()
        .into_iter()
        .map(|x| is_near(x, x_num, tolerance))
        .collect();

    let grain_is_x: bool = bools.iter().any(|&b| b);

    if !show_rec {
        Ok(GrimReturn::Bool(grain_is_x))
    } else {
        let consistency: bool = grain_is_x;

        let length_2ers = ["up_or_down", "up_from_or_down_from", "ceiling_or_floor"];

        if length_2ers.contains(&rounding) {
            Ok(GrimReturn::List(
                consistency,
                rec_sum,
                rec_x_upper,
                rec_x_lower,
                grains_rounded[0],
                grains_rounded[1],
                //grains_rounded[4].clone(),
                //grains_rounded[5].clone(),
            ))
        } else {
            Ok(GrimReturn::Bool(grain_is_x))
        }
    }
}

/// Determine whether the two provided numbers are within a given tolerance of each other
pub fn is_near(num_1: f64, num_2: f64, tolerance: f64) -> bool {
    (num_1 - num_2).abs() <= tolerance
}


/// Automatically unpacks and tests the output of grim_scalar_rust and checks whether its main bool
/// result matches the expected bool
pub fn grim_tester(grim_result: Result<GrimReturn, GrimScalarError>, expected: bool) {
    match grim_result {
        Ok(grim_return) => match grim_return {
            GrimReturn::Bool(b) => match expected {
                true => assert!(b),
                false => assert!(!b),
            },
            GrimReturn::List(a, _, _, _, _, _) => assert!(!a),
        },
        Err(_) => panic!(),
    };
}

