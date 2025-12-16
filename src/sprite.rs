//! Creating a space to experiment with a Rust translation of SPRITE

use bigdecimal::{
    BigDecimal, FromPrimitive as BigDecimalFromPrimitive, ToPrimitive as BigDecimalToPrimitive,
    Zero,
};
use core::f64;
use num::NumCast;
use num_traits::Float;
use rand::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::grimmer::{
    decimal_places_scalar, grim_scalar_rust, grimmer_scalar, is_near, rust_round, GrimReturn,
};
use crate::sprite_types::{OccurrenceConstraints, RestrictionsMinimum, RestrictionsOption};
use crate::{
    calculate_all_statistics, create_results_writer, create_stats_writers, results_to_record_batch,
    FloatType, IntegerType, ParquetConfig, ResultListFromMeanSdN, StreamingConfig,
    StreamingFrequencyState, StreamingResult,
};

use arrow::array::{Float64Array, Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::channel;
use std::thread;

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

// Internal struct to hold the final, validated parameters (generic version)
#[derive(Debug)]
struct SpriteParams<T, U>
where
    T: FloatType,
    U: IntegerType,
{
    mean: T,
    sd: T,
    n_obs: u32,
    m_prec: i32,
    sd_prec: i32,
    /// Scale factor used to convert floating-point values to integers
    scale_factor: u32,
    /// Scaled possible values (integers)
    possible_values_scaled: Vec<U>,
    /// Scaled fixed responses (integers)
    fixed_responses_scaled: Vec<U>,
    /// The number of fixed responses
    n_fixed: usize,
}

// Legacy struct maintained for backward compatibility (deprecated)
#[deprecated(since = "0.2.0", note = "Use sprite_parallel() instead")]
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

#[allow(deprecated)]
fn abm_internal(
    total: &BigDecimal,
    n_obs: &BigDecimal,
    a: &BigDecimal,
    b: &BigDecimal,
) -> Vec<f64> {
    // [BigDecimal] All calculations are now decimal-based.
    let k1 =
        ((total - (n_obs * b)) / (a - b)).with_scale_round(0, bigdecimal::RoundingMode::HalfUp);
    let k = k1.max(BigDecimal::from(1)).min(n_obs - BigDecimal::from(1));

    let k_floor = k.with_scale_round(0, bigdecimal::RoundingMode::Floor);
    let n_minus_k = n_obs - &k_floor;

    let mut v = Vec::with_capacity(n_obs.to_usize().unwrap());
    for _ in 0..k_floor.to_usize().unwrap() {
        v.push(a.clone());
    }
    for _ in 0..n_minus_k.to_usize().unwrap() {
        v.push(b.clone());
    }

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

#[allow(deprecated)]
pub fn sd_limits(
    n_obs: u32,
    mean: f64,
    sd: f64,
    min_val: i32,
    max_val: i32,
    sd_prec_opt: Option<i32>,
    n_items: u32,
) -> (f64, f64) {
    // [BigDecimal] Convert inputs for high-precision calculation.
    let mean_bd = BigDecimal::from_f64(mean).unwrap();
    let n_obs_bd = BigDecimal::from_u32(n_obs).unwrap();
    let n_items_bd = BigDecimal::from_u32(n_items).unwrap();

    let sd_prec: i32 = sd_prec_opt.unwrap_or_else(|| {
        max(
            decimal_places_scalar(Some(&sd.to_string()), ".").unwrap() - 1,
            0,
        )
    });

    let a_max = BigDecimal::from_i32(min_val).unwrap();
    let a_min =
        (&mean_bd * &n_items_bd).with_scale_round(0, bigdecimal::RoundingMode::Floor) / &n_items_bd;

    let b_max_cand1 = BigDecimal::from_i32(max_val).unwrap();
    let b_max_cand2 = BigDecimal::from_i32(min_val + 1).unwrap();
    let b_max_cand3 = &a_min + BigDecimal::from(1);
    let b_max = std::cmp::max(b_max_cand1, std::cmp::max(b_max_cand2, b_max_cand3));

    let b_min = &a_min + (BigDecimal::from(1) / &n_items_bd);
    let total = (&mean_bd * &n_obs_bd * &n_items_bd)
        .with_scale_round(0, bigdecimal::RoundingMode::HalfUp)
        / &n_items_bd;

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

#[allow(deprecated)]
#[allow(clippy::too_many_arguments)]
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
    mut dont_test: bool,
) -> Result<SpriteParameters, ParameterError> {
    if min_val >= max_val {
        return Err(ParameterError::InputValidation(
            "max_val must be greater than min_val".to_string(),
        ));
    }
    let m_prec =
        m_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean.to_string()), ".").unwrap());
    let sd_prec =
        sd_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean.to_string()), ".").unwrap());

    if n_obs * n_items <= 10.0.powi(m_prec) as u32 {
        dont_test = true;
    }

    if !dont_test {
        let grim_result = grim_scalar_rust(
            &mean.to_string(),
            n_obs,
            vec![false, false, false],
            n_items,
            "up_or_down",
            5.0,
            f64::EPSILON.sqrt(),
        );
        let consistent = match grim_result {
            Ok(GrimReturn::Bool(b)) => b,
            Ok(GrimReturn::List(b, _, _, _, _, _)) => b,
            Err(_) => false, // Treat error as inconsistency
        };
        if !consistent {
            return Err(ParameterError::Consistency(
                "Mean fails GRIM test.".to_string(),
            ));
        }

        let grimmer_consistent = grimmer_scalar(
            &mean.to_string(),
            &sd.to_string(),
            n_obs,
            n_items,
            vec![false, false, false],
            "up_or_down",
            5.0,
            f64::EPSILON.sqrt(),
        );
        if !grimmer_consistent {
            return Err(ParameterError::Consistency(
                "SD fails GRIMMER test.".to_string(),
            ));
        }
    }

    let sd_lims = sd_limits(n_obs, mean, sd, min_val, max_val, Some(sd_prec), n_items);
    if !(sd >= sd_lims.0 && sd <= sd_lims.1) {
        return Err(ParameterError::InputValidation(format!(
            "SD is outside the possible range: [{}, {}]",
            sd_lims.0, sd_lims.1
        )));
    }
    if !(mean >= min_val as f64 && mean <= max_val as f64) {
        return Err(ParameterError::InputValidation(
            "Mean is outside the possible [min_val, max_val] range.".to_string(),
        ));
    }

    let restrictions_exact = restrictions_exact.unwrap_or_default();
    let restrictions_minimum = match restrictions_minimum {
        RestrictionsOption::Default => {
            Some(RestrictionsMinimum::from_range(min_val * 100, max_val * 100).extract())
        }
        RestrictionsOption::Opt(opt_map) => opt_map.map(|rm| rm.extract()),
        RestrictionsOption::Null => None,
    };

    if let Some(ref min_map) = restrictions_minimum {
        let constraints =
            OccurrenceConstraints::new(restrictions_exact.clone(), min_map.clone(), None);
        if constraints.check_conflicts() {
            let exact_keys: HashSet<_> = constraints.exact.keys().collect();
            let min_keys: HashSet<_> = constraints.minimum.keys().collect();
            let conflict_keys: Vec<_> = exact_keys
                .intersection(&min_keys)
                .map(|&&k| k as f64 / 100.0)
                .collect();
            return Err(ParameterError::Conflict(format!(
                "Value(s) in both exact and minimum restrictions: {conflict_keys:? }"
            )));
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
    let poss_values_keys: HashSet<i32> = poss_values
        .iter()
        .map(|&v| (v * 100.0).round() as i32)
        .collect();

    let mut fixed_responses: Vec<f64> = Vec::new();
    let mut fixed_values_keys: HashSet<i32> = HashSet::new();

    for (&key, &count) in &restrictions_exact {
        if !poss_values_keys.contains(&key) {
            return Err(ParameterError::InputValidation(format!(
                "Invalid key in restrictions_exact: {}",
                key as f64 / 100.0
            )));
        }
        fixed_values_keys.insert(key);
        for _ in 0..count {
            fixed_responses.push(key as f64 / 100.0);
        }
    }

    if let Some(min_map) = restrictions_minimum {
        for (&key, &count) in &min_map {
            if !poss_values_keys.contains(&key) {
                return Err(ParameterError::InputValidation(format!(
                    "Invalid key in restrictions_minimum: {}",
                    key as f64 / 100.0
                )));
            }
            for _ in 0..count {
                fixed_responses.push(key as f64 / 100.0);
            }
        }
    }

    let final_possible_values: Vec<f64> = poss_values
        .into_iter()
        .filter(|v| !fixed_values_keys.contains(&((v * 100.0).round() as i32)))
        .collect();
    let n_fixed = fixed_responses.len();

    Ok(SpriteParameters {
        mean,
        sd,
        n_obs,
        min_val,
        max_val,
        m_prec,
        sd_prec,
        n_items,
        possible_values: final_possible_values,
        fixed_responses,
        n_fixed,
    })
}

// Legacy types (deprecated)
#[deprecated(since = "0.2.0", note = "Use sprite_parallel() instead")]
#[derive(Debug, PartialEq, Clone)]
pub enum Outcome {
    Success,
    Failure,
}

/// Holds the detailed results of a single distribution search (deprecated)
#[deprecated(since = "0.2.0", note = "Use sprite_parallel() instead")]
#[allow(deprecated)]
#[derive(Debug, Clone)]
pub struct DistributionResult {
    pub outcome: Outcome,
    pub values: Vec<f64>,
    pub mean: f64,
    pub sd: f64,
    pub iterations: u32,
}

/// Internal function to build and validate SPRITE parameters
#[allow(clippy::too_many_arguments)]
fn build_sprite_params<T, U>(
    mean: T,
    sd: T,
    n_obs: u32,
    min_val: i32,
    max_val: i32,
    m_prec: Option<i32>,
    sd_prec: Option<i32>,
    n_items: u32,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    mut dont_test: bool,
) -> Result<SpriteParams<T, U>, ParameterError>
where
    T: FloatType,
    U: IntegerType,
{
    if min_val >= max_val {
        return Err(ParameterError::InputValidation(
            "max_val must be greater than min_val".to_string(),
        ));
    }

    let mean_str = T::to_f64(&mean).unwrap().to_string();
    let sd_str = T::to_f64(&sd).unwrap().to_string();

    let m_prec = m_prec.unwrap_or_else(|| decimal_places_scalar(Some(&mean_str), ".").unwrap());
    let sd_prec = sd_prec.unwrap_or_else(|| decimal_places_scalar(Some(&sd_str), ".").unwrap());

    // Determine scale factor
    let precision = max(m_prec, sd_prec);
    let scale_factor = 10_u32.pow(precision as u32);

    if n_obs * n_items <= 10_u32.pow(m_prec as u32) {
        dont_test = true;
    }

    // GRIM/GRIMMER tests (still done on f64)
    if !dont_test {
        let mean_f64 = T::to_f64(&mean).unwrap();
        let sd_f64 = T::to_f64(&sd).unwrap();

        let grim_result = grim_scalar_rust(
            &mean_str,
            n_obs,
            vec![false, false, false],
            n_items,
            "up_or_down",
            5.0,
            f64::EPSILON.sqrt(),
        );
        let consistent = match grim_result {
            Ok(GrimReturn::Bool(b)) => b,
            Ok(GrimReturn::List(b, _, _, _, _, _)) => b,
            Err(_) => false,
        };
        if !consistent {
            return Err(ParameterError::Consistency(
                "Mean fails GRIM test.".to_string(),
            ));
        }

        let grimmer_consistent = grimmer_scalar(
            &mean_str,
            &sd_str,
            n_obs,
            n_items,
            vec![false, false, false],
            "up_or_down",
            5.0,
            f64::EPSILON.sqrt(),
        );
        if !grimmer_consistent {
            return Err(ParameterError::Consistency(
                "SD fails GRIMMER test.".to_string(),
            ));
        }

        // Check SD limits
        let sd_lims = sd_limits(
            n_obs,
            mean_f64,
            sd_f64,
            min_val,
            max_val,
            Some(sd_prec),
            n_items,
        );
        if !(sd_f64 >= sd_lims.0 && sd_f64 <= sd_lims.1) {
            return Err(ParameterError::InputValidation(format!(
                "SD is outside the possible range: [{}, {}]",
                sd_lims.0, sd_lims.1
            )));
        }
    }

    // Check mean is in range
    let mean_f64 = T::to_f64(&mean).unwrap();
    if !(mean_f64 >= min_val as f64 && mean_f64 <= max_val as f64) {
        return Err(ParameterError::InputValidation(
            "Mean is outside the possible [min_val, max_val] range.".to_string(),
        ));
    }

    // Handle restrictions
    let restrictions_exact = restrictions_exact.unwrap_or_default();
    let restrictions_minimum = match restrictions_minimum {
        RestrictionsOption::Default => {
            Some(RestrictionsMinimum::from_range(min_val * 100, max_val * 100).extract())
        }
        RestrictionsOption::Opt(opt_map) => opt_map.map(|rm| rm.extract()),
        RestrictionsOption::Null => None,
    };

    if let Some(ref min_map) = restrictions_minimum {
        let constraints =
            OccurrenceConstraints::new(restrictions_exact.clone(), min_map.clone(), None);
        if constraints.check_conflicts() {
            let exact_keys: HashSet<_> = constraints.exact.keys().collect();
            let min_keys: HashSet<_> = constraints.minimum.keys().collect();
            let conflict_keys: Vec<_> = exact_keys
                .intersection(&min_keys)
                .map(|&&k| k as f64 / 100.0)
                .collect();
            return Err(ParameterError::Conflict(format!(
                "Value(s) in both exact and minimum restrictions: {conflict_keys:?}"
            )));
        }
    }

    // Generate all possible scaled values
    let mut poss_values_scaled: Vec<U> = Vec::new();
    for i in 1..=n_items {
        for val in min_val..max_val {
            let float_val = val as f64 + (i - 1) as f64 / n_items as f64;
            let scaled_val = (float_val * scale_factor as f64).round() as i64;
            if let Some(u_val) = NumCast::from(scaled_val) {
                poss_values_scaled.push(u_val);
            }
        }
    }
    // Add max_val
    let max_scaled = (max_val as f64 * scale_factor as f64).round() as i64;
    if let Some(u_val) = NumCast::from(max_scaled) {
        poss_values_scaled.push(u_val);
    }

    poss_values_scaled.sort_by(|a, b| U::to_i64(a).unwrap().cmp(&U::to_i64(b).unwrap()));
    poss_values_scaled.dedup();

    let poss_values_keys: HashSet<i32> = poss_values_scaled
        .iter()
        .map(|v| (U::to_f64(v).unwrap() / (scale_factor as f64 / 100.0)).round() as i32)
        .collect();

    // Process fixed responses
    let mut fixed_responses_scaled: Vec<U> = Vec::new();
    let mut fixed_values_keys: HashSet<i32> = HashSet::new();

    for (&key, &count) in &restrictions_exact {
        if !poss_values_keys.contains(&key) {
            return Err(ParameterError::InputValidation(format!(
                "Invalid key in restrictions_exact: {}",
                key as f64 / 100.0
            )));
        }
        fixed_values_keys.insert(key);
        let scaled_val = (key as f64 / 100.0 * scale_factor as f64).round() as i64;
        if let Some(u_val) = NumCast::from(scaled_val) {
            for _ in 0..count {
                fixed_responses_scaled.push(u_val);
            }
        }
    }

    if let Some(min_map) = restrictions_minimum {
        for (&key, &count) in &min_map {
            if !poss_values_keys.contains(&key) {
                return Err(ParameterError::InputValidation(format!(
                    "Invalid key in restrictions_minimum: {}",
                    key as f64 / 100.0
                )));
            }
            let scaled_val = (key as f64 / 100.0 * scale_factor as f64).round() as i64;
            if let Some(u_val) = NumCast::from(scaled_val) {
                for _ in 0..count {
                    fixed_responses_scaled.push(u_val);
                }
            }
        }
    }

    // Filter out fixed values from possible values
    let final_possible_values_scaled: Vec<U> = poss_values_scaled
        .into_iter()
        .filter(|v| {
            let key = (U::to_f64(v).unwrap() / (scale_factor as f64 / 100.0)).round() as i32;
            !fixed_values_keys.contains(&key)
        })
        .collect();

    let n_fixed = fixed_responses_scaled.len();

    Ok(SpriteParams {
        mean,
        sd,
        n_obs,
        m_prec,
        sd_prec,
        scale_factor,
        possible_values_scaled: final_possible_values_scaled,
        fixed_responses_scaled,
        n_fixed,
    })
}

/// Main SPRITE API: Generate all valid distributions matching summary statistics
///
/// This is the SPRITE equivalent of `dfs_parallel()`. It finds multiple possible
/// distributions of raw data that match the given mean and standard deviation.
///
/// # Arguments
/// * `mean` - The target mean
/// * `sd` - The target standard deviation
/// * `n_obs` - The number of observations
/// * `min_val` - The minimum value on the scale
/// * `max_val` - The maximum value on the scale
/// * `m_prec` - Optional precision for the mean (decimal places)
/// * `sd_prec` - Optional precision for the SD (decimal places)
/// * `n_items` - Number of items averaged (default 1)
/// * `n_distributions` - Number of unique distributions to find
/// * `restrictions_exact` - Optional exact count requirements for specific values
/// * `restrictions_minimum` - Optional minimum count requirements for specific values
/// * `dont_test` - Skip GRIM/GRIMMER validation (default false)
/// * `parquet_config` - Optional configuration for writing results to Parquet files
/// * `rng` - Random number generator
///
/// # Returns
/// A `ResultListFromMeanSdN<U>` containing all distributions and comprehensive statistics
/// including horns metrics and frequency distributions.
///
/// # Example
/// ```ignore
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let results = sprite_parallel(
///     2.2_f64, 1.3_f64, 20, 1, 5,
///     None, None, 1, 5,
///     None, RestrictionsOption::Default,
///     false, None, &mut rng
/// ).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn sprite_parallel<T, U>(
    mean: T,
    sd: T,
    n_obs: u32,
    min_val: i32,
    max_val: i32,
    m_prec: Option<i32>,
    sd_prec: Option<i32>,
    n_items: u32,
    n_distributions: usize,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    dont_test: bool,
    parquet_config: Option<ParquetConfig>,
    rng: &mut impl Rng,
) -> Result<ResultListFromMeanSdN<U>, ParameterError>
where
    T: FloatType + Send + Sync,
    U: IntegerType + Send + Sync + 'static,
{
    // Build and validate parameters
    let params = build_sprite_params(
        mean,
        sd,
        n_obs,
        min_val,
        max_val,
        m_prec,
        sd_prec,
        n_items,
        restrictions_exact,
        restrictions_minimum,
        dont_test,
    )?;

    // Find distributions (scaled integer values)
    let results_scaled = find_distributions_internal(&params, n_distributions, rng);

    // Convert scaled values to the 100x scale for compatibility with CLOSURE statistics
    // SPRITE internally uses scale_factor (e.g., 10000 for precision 4), but statistics
    // expect values in the 100x scale (multiply original values by 100)
    let scale_factor_f64 = params.scale_factor as f64;
    let results_100x: Vec<Vec<U>> = results_scaled
        .iter()
        .map(|dist| {
            dist.iter()
                .map(|&val| {
                    // Convert to original value, then to 100x scale
                    let original_f64 = U::to_i64(&val).unwrap() as f64 / scale_factor_f64;
                    let value_100x = (original_f64 * 100.0).round() as i64;
                    NumCast::from(value_100x).unwrap()
                })
                .collect()
        })
        .collect();

    // Use min_val and max_val in 100x scale for statistics
    let scale_min: U = NumCast::from(min_val * 100).unwrap();
    let scale_max: U = NumCast::from(max_val * 100).unwrap();

    // Calculate all statistics using the shared function from lib.rs
    let sprite_results = calculate_all_statistics(results_100x, scale_min, scale_max);

    // Write to Parquet if configured
    if let Some(config) = parquet_config {
        write_sprite_parquet(&sprite_results, &config, params.scale_factor)?;
    }

    Ok(sprite_results)
}

/// Write SPRITE results to Parquet files
fn write_sprite_parquet<U>(
    results: &ResultListFromMeanSdN<U>,
    config: &ParquetConfig,
    _scale_factor: u32,
) -> Result<(), ParameterError>
where
    U: IntegerType + Send + Sync,
{
    use std::sync::Arc;

    // Ensure base_path ends with / for consistent file naming
    let base_path = if config.file_path.ends_with('/') {
        config.file_path.clone()
    } else {
        format!("{}/", config.file_path)
    };

    // Write results table
    let results_path = format!("{}results.parquet", base_path);
    if let Ok(mut writer) = create_results_writer(&results_path) {
        let batch_size = config.batch_size;
        let total_samples = results.results.sample.len();

        for start in (0..total_samples).step_by(batch_size) {
            let end = (start + batch_size).min(total_samples);

            if let Ok(record_batch) = results_to_record_batch(&results.results, start, end) {
                let _ = writer.write(&record_batch);
            }
        }
        let _ = writer.close();
    }

    // Write statistics tables
    if let Ok((mut mm_writer, mut mh_writer, mut freq_writer, mm_schema, mh_schema, freq_schema)) =
        create_stats_writers(&base_path)
    {
        // Write metrics_main
        let mm_batch = RecordBatch::try_new(
            mm_schema,
            vec![
                Arc::new(Float64Array::from(vec![
                    results.metrics_main.samples_initial,
                ])),
                Arc::new(Float64Array::from(vec![results.metrics_main.samples_all])),
                Arc::new(Float64Array::from(vec![results.metrics_main.values_all])),
            ],
        );
        if let Ok(batch) = mm_batch {
            let _ = mm_writer.write(&batch);
        }
        let _ = mm_writer.close();

        // Write metrics_horns
        let mh_batch = RecordBatch::try_new(
            mh_schema,
            vec![
                Arc::new(Float64Array::from(vec![results.metrics_horns.mean])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.uniform])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.sd])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.cv])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.mad])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.min])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.median])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.max])),
                Arc::new(Float64Array::from(vec![results.metrics_horns.range])),
            ],
        );
        if let Ok(batch) = mh_batch {
            let _ = mh_writer.write(&batch);
        }
        let _ = mh_writer.close();

        // Write frequency table
        let freq_batch = RecordBatch::try_new(
            freq_schema,
            vec![
                Arc::new(StringArray::from(results.frequency.samples.clone())),
                Arc::new(Int32Array::from(results.frequency.value.clone())),
                Arc::new(Float64Array::from(results.frequency.f_average.clone())),
                Arc::new(Float64Array::from(results.frequency.f_absolute.clone())),
                Arc::new(Float64Array::from(results.frequency.f_relative.clone())),
            ],
        );
        if let Ok(batch) = freq_batch {
            let _ = freq_writer.write(&batch);
        }
        let _ = freq_writer.close();
    }

    Ok(())
}

/// SPRITE streaming API: Generate distributions and stream to Parquet files
///
/// This is the SPRITE equivalent of `dfs_parallel_streaming()`. It finds distributions
/// matching the given mean and SD and streams them directly to Parquet files without
/// keeping all results in memory.
///
/// Use this when:
/// - Result sets are very large (> 1GB in memory)
/// - You only need file output, not in-memory processing
/// - Memory efficiency is critical
///
/// # Arguments
/// * `mean` - The target mean
/// * `sd` - The target standard deviation
/// * `n_obs` - The number of observations
/// * `min_val` - The minimum value on the scale
/// * `max_val` - The maximum value on the scale
/// * `m_prec` - Optional precision for the mean (decimal places)
/// * `sd_prec` - Optional precision for the SD (decimal places)
/// * `n_items` - Number of items averaged (default 1)
/// * `restrictions_exact` - Optional exact count requirements for specific values
/// * `restrictions_minimum` - Optional minimum count requirements for specific values
/// * `dont_test` - Skip GRIM/GRIMMER validation (default false)
/// * `config` - Streaming configuration (file path, batch size, progress reporting)
/// * `stop_after` - Optional limit on number of distributions to find
///
/// # Returns
/// A `StreamingResult` with the total count and file path.
#[allow(clippy::too_many_arguments)]
pub fn sprite_parallel_streaming<T, U>(
    mean: T,
    sd: T,
    n_obs: u32,
    min_val: i32,
    max_val: i32,
    m_prec: Option<i32>,
    sd_prec: Option<i32>,
    n_items: u32,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    dont_test: bool,
    config: StreamingConfig,
    stop_after: Option<usize>,
) -> StreamingResult
where
    T: FloatType + Send + Sync,
    U: IntegerType + Send + Sync + 'static,
{
    use crate::{
        create_horns_writer, create_samples_writer, horns_to_record_batch, samples_to_record_batch,
        write_streaming_statistics,
    };
    use std::collections::HashMap;

    // Build and validate parameters
    let params = match build_sprite_params(
        mean,
        sd,
        n_obs,
        min_val,
        max_val,
        m_prec,
        sd_prec,
        n_items,
        restrictions_exact,
        restrictions_minimum,
        dont_test,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ERROR: Parameter validation failed: {}", e);
            return StreamingResult {
                total_combinations: 0,
                file_path: config.file_path,
            };
        }
    };

    let _scale_factor = params.scale_factor;
    let n_obs_usize = n_obs as usize;

    // Setup channels for streaming results
    let (tx_results, rx_results) = channel::<Vec<(Vec<U>, f64)>>();
    let (tx_stats, rx_stats) = channel::<(Vec<f64>, HashMap<i32, i64>)>();

    // Add a flag to track writer thread status
    let writer_failed = Arc::new(AtomicUsize::new(0)); // 0 = ok, 1 = failed
    let writer_failed_for_compute = writer_failed.clone();
    let writer_failed_for_thread = writer_failed.clone();

    // Counter for total distributions found
    let total_counter = Arc::new(AtomicUsize::new(0));

    // Shared state for tracking min/max horns frequencies
    let freq_state = Arc::new(Mutex::new(StreamingFrequencyState {
        current_min_horns: f64::INFINITY,
        current_max_horns: f64::NEG_INFINITY,
        all_freq: HashMap::new(),
        min_freq: HashMap::new(),
        max_freq: HashMap::new(),
        min_count: 0,
        max_count: 0,
    }));
    let freq_state_for_thread = freq_state.clone();

    // Handle file paths
    let base_path = if config.file_path.ends_with('/') {
        config.file_path.clone()
    } else if std::path::Path::new(&config.file_path).is_dir() {
        format!("{}/", config.file_path)
    } else {
        format!("{}_", config.file_path)
    };

    // Create parent directory if needed
    if let Some(parent) = std::path::Path::new(&base_path).parent() {
        if !parent.to_str().unwrap_or("").is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("Warning: Could not create directory {:?}: {}", parent, e);
            }
        }
    }

    // Spawn dedicated writer thread
    let samples_path = format!("{}samples.parquet", base_path);
    let horns_path = format!("{}horns.parquet", base_path);

    let writer_handle = thread::spawn(move || {
        let mut samples_writer = match create_samples_writer(&samples_path, n_obs_usize) {
            Ok(w) => w,
            Err(e) => {
                eprintln!(
                    "ERROR: Failed to create samples writer for file '{}': {}",
                    samples_path, e
                );
                writer_failed_for_thread.store(1, Ordering::Relaxed);
                return 0;
            }
        };

        let mut horns_writer = match create_horns_writer(&horns_path) {
            Ok(w) => w,
            Err(e) => {
                eprintln!(
                    "ERROR: Failed to create horns writer for file '{}': {}",
                    horns_path, e
                );
                writer_failed_for_thread.store(1, Ordering::Relaxed);
                return 0;
            }
        };

        let mut samples_buffer: Vec<Vec<U>> = Vec::with_capacity(config.batch_size * 2);
        let mut horns_buffer: Vec<f64> = Vec::with_capacity(config.batch_size * 2);

        let mut total_written = 0;
        let mut last_progress_report = 0;

        loop {
            match rx_results.recv() {
                Ok(batch) => {
                    for (sample, horns) in batch {
                        samples_buffer.push(sample);
                        horns_buffer.push(horns);
                    }

                    if samples_buffer.len() >= config.batch_size {
                        match samples_to_record_batch(&samples_buffer) {
                            Ok(record_batch) => {
                                if let Err(e) = samples_writer.write(&record_batch) {
                                    eprintln!("ERROR: Failed to write samples batch: {}", e);
                                    return total_written;
                                }
                            }
                            Err(e) => {
                                eprintln!("ERROR: Failed to create samples batch: {}", e);
                                return total_written;
                            }
                        }

                        match horns_to_record_batch(&horns_buffer) {
                            Ok(record_batch) => {
                                if let Err(e) = horns_writer.write(&record_batch) {
                                    eprintln!("ERROR: Failed to write horns batch: {}", e);
                                    return total_written;
                                }
                            }
                            Err(e) => {
                                eprintln!("ERROR: Failed to create horns batch: {}", e);
                                return total_written;
                            }
                        }

                        total_written += samples_buffer.len();

                        if config.show_progress && total_written - last_progress_report >= 1000 {
                            eprintln!("Progress: {} distributions written...", total_written);
                            last_progress_report = total_written;
                        }

                        samples_buffer.clear();
                        horns_buffer.clear();
                    }
                }
                Err(_) => break,
            }
        }

        // Write remaining results
        if !samples_buffer.is_empty() {
            if let Ok(record_batch) = samples_to_record_batch(&samples_buffer) {
                if let Err(e) = samples_writer.write(&record_batch) {
                    eprintln!("ERROR: Failed to write final samples batch: {}", e);
                } else if let Ok(record_batch) = horns_to_record_batch(&horns_buffer) {
                    if let Err(e) = horns_writer.write(&record_batch) {
                        eprintln!("ERROR: Failed to write final horns batch: {}", e);
                    } else {
                        total_written += samples_buffer.len();
                    }
                }
            }
        }

        if let Err(e) = samples_writer.close() {
            eprintln!("ERROR: Failed to close samples file: {}", e);
        }
        if let Err(e) = horns_writer.close() {
            eprintln!("ERROR: Failed to close horns file: {}", e);
        }

        if config.show_progress {
            eprintln!(
                "Streaming complete: {} total distributions written",
                total_written
            );
        }

        total_written
    });

    // Spawn statistics collector thread
    let freq_state_for_stats = freq_state.clone();
    let stats_handle = thread::spawn(move || {
        let mut all_horns = Vec::new();

        while let Ok((horns_batch, _)) = rx_stats.recv() {
            all_horns.extend(horns_batch);
        }

        (all_horns, freq_state_for_stats)
    });

    // Find distributions using streaming approach
    find_distributions_streaming(
        &params,
        stop_after,
        tx_results,
        tx_stats,
        &total_counter,
        &freq_state_for_thread,
        &writer_failed_for_compute,
        &config,
    );

    // Wait for threads to complete
    let total_written = writer_handle.join().unwrap_or_else(|_| {
        eprintln!("ERROR: Writer thread panicked unexpectedly");
        0
    });

    let scale_min_i32 = min_val;
    let scale_max_i32 = max_val;

    let (all_horns, final_freq_state) = stats_handle.join().unwrap_or_else(|_| {
        eprintln!("ERROR: Statistics thread panicked unexpectedly");
        (Vec::new(), freq_state)
    });

    if total_written == 0 {
        eprintln!("\nERROR: No results were written to disk.");
        return StreamingResult {
            total_combinations: 0,
            file_path: config.file_path,
        };
    }

    // Write statistics files
    write_streaming_statistics(
        &base_path,
        &all_horns,
        n_obs_usize,
        scale_min_i32,
        scale_max_i32,
        final_freq_state,
    );

    StreamingResult {
        total_combinations: total_written,
        file_path: config.file_path,
    }
}

/// Find distributions using streaming approach (parallelized with rayon)
fn find_distributions_streaming<T, U>(
    params: &SpriteParams<T, U>,
    stop_after: Option<usize>,
    tx_results: std::sync::mpsc::Sender<Vec<(Vec<U>, f64)>>,
    tx_stats: std::sync::mpsc::Sender<(Vec<f64>, HashMap<i32, i64>)>,
    total_counter: &Arc<AtomicUsize>,
    freq_state: &Arc<Mutex<StreamingFrequencyState>>,
    writer_failed: &Arc<AtomicUsize>,
    config: &StreamingConfig,
) where
    T: FloatType + Send + Sync,
    U: IntegerType + Send + Sync + 'static,
{
    use crate::calculate_horns;

    let should_stop = Arc::new(AtomicBool::new(false));
    let unique_distributions = Arc::new(Mutex::new(HashSet::<Vec<i64>>::new()));
    let total_failures = Arc::new(AtomicU32::new(0));
    let total_duplicates = Arc::new(AtomicU32::new(0));

    let batch_size = 100;
    let max_iterations = stop_after.unwrap_or(usize::MAX).min(100_000_000);

    let scale_min_i32 = params
        .possible_values_scaled
        .first()
        .map(|v| {
            (U::to_i64(v).unwrap() as f64 / (params.scale_factor as f64 / 100.0)).round() as i32
        })
        .unwrap_or(0);
    let scale_max_i32 = params
        .possible_values_scaled
        .last()
        .map(|v| {
            (U::to_i64(v).unwrap() as f64 / (params.scale_factor as f64 / 100.0)).round() as i32
        })
        .unwrap_or(100);
    let nrow_frequency = (scale_max_i32 - scale_min_i32 + 1) as usize;

    for batch_start in (0..max_iterations).step_by(batch_size) {
        if should_stop.load(Ordering::Relaxed) || writer_failed.load(Ordering::Relaxed) == 1 {
            break;
        }

        {
            let unique = unique_distributions.lock().unwrap();
            if let Some(limit) = stop_after {
                if unique.len() >= limit {
                    break;
                }
            }
        }

        let batch_end = (batch_start + batch_size).min(max_iterations);
        let batch_range = batch_start..batch_end;

        batch_range.into_par_iter().for_each(|_| {
            if should_stop.load(Ordering::Relaxed) {
                return;
            }

            if let Some(limit) = stop_after {
                if total_counter.load(Ordering::Relaxed) >= limit {
                    should_stop.store(true, Ordering::Relaxed);
                    return;
                }
            }

            let mut thread_rng = rand::rng();

            match find_distribution_internal(params, &mut thread_rng) {
                Ok(mut distribution) => {
                    distribution.sort_by(|a, b| U::to_i64(a).unwrap().cmp(&U::to_i64(b).unwrap()));

                    let hashable_values: Vec<i64> =
                        distribution.iter().map(|v| U::to_i64(v).unwrap()).collect();

                    let mut unique = unique_distributions.lock().unwrap();
                    if unique.insert(hashable_values) {
                        drop(unique);

                        // Calculate horns for this distribution
                        let scale_f = params.scale_factor as f64;
                        let mut freqs = vec![0.0; nrow_frequency];
                        let mut sample_freq: HashMap<i32, i64> = HashMap::new();

                        for &value in &distribution {
                            let val_f64 = U::to_i64(&value).unwrap() as f64 / scale_f;
                            let idx = ((val_f64 * 100.0).round() as i32 - scale_min_i32) as usize;
                            if idx < freqs.len() {
                                freqs[idx] += 1.0;
                            }
                            *sample_freq
                                .entry((val_f64 * 100.0).round() as i32)
                                .or_insert(0) += 1;
                        }

                        let horns = calculate_horns(&freqs, scale_min_i32, scale_max_i32);

                        // Update frequency state
                        {
                            let mut state = freq_state.lock().unwrap();

                            for (&val, &count) in &sample_freq {
                                *state.all_freq.entry(val).or_insert(0) += count;
                            }

                            if (horns - state.current_min_horns).abs() < 1e-10 {
                                for (&val, &count) in &sample_freq {
                                    *state.min_freq.entry(val).or_insert(0) += count;
                                }
                                state.min_count += 1;
                            } else if horns < state.current_min_horns {
                                state.current_min_horns = horns;
                                state.min_freq.clear();
                                for (&val, &count) in &sample_freq {
                                    state.min_freq.insert(val, count);
                                }
                                state.min_count = 1;
                            }

                            if (horns - state.current_max_horns).abs() < 1e-10 {
                                for (&val, &count) in &sample_freq {
                                    *state.max_freq.entry(val).or_insert(0) += count;
                                }
                                state.max_count += 1;
                            } else if horns > state.current_max_horns {
                                state.current_max_horns = horns;
                                state.max_freq.clear();
                                for (&val, &count) in &sample_freq {
                                    state.max_freq.insert(val, count);
                                }
                                state.max_count = 1;
                            }
                        }

                        // Send to writer
                        if tx_results.send(vec![(distribution, horns)]).is_ok() {
                            let _ = tx_stats.send((vec![horns], HashMap::new()));
                            total_counter.fetch_add(1, Ordering::Relaxed);
                        }

                        total_failures.store(0, Ordering::Relaxed);
                        total_duplicates.store(0, Ordering::Relaxed);
                    } else {
                        total_duplicates.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    total_failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        // Check early stopping conditions
        let failures = total_failures.load(Ordering::Relaxed);
        let duplicates = total_duplicates.load(Ordering::Relaxed);

        if failures >= 1000 {
            if config.show_progress {
                eprintln!(
                    "Warning: Too many failed attempts ({}). Stopping search.",
                    failures
                );
            }
            should_stop.store(true, Ordering::Relaxed);
            break;
        }

        let n_found = unique_distributions.lock().unwrap().len() as f64;
        let max_duplications = if n_found > 0.0 {
            (0.00001f64.ln() / (n_found / (n_found + 1.0)).ln()).round() as u32
        } else {
            1000
        }
        .max(1000);

        if duplicates > max_duplications {
            if config.show_progress {
                eprintln!(
                    "Warning: Found too many duplicate distributions ({}). Stopping search.",
                    duplicates
                );
            }
            should_stop.store(true, Ordering::Relaxed);
            break;
        }
    }
}

/// Internal function to find multiple distributions (parallelized with rayon)
fn find_distributions_internal<T, U>(
    params: &SpriteParams<T, U>,
    n_distributions: usize,
    _rng: &mut impl Rng,
) -> Vec<Vec<U>>
where
    T: FloatType,
    U: IntegerType,
{
    // Shared state protected by mutexes
    let unique_distributions = Arc::new(Mutex::new(HashSet::<Vec<i64>>::new()));
    let results = Arc::new(Mutex::new(Vec::<Vec<U>>::new()));

    // Atomic counters for tracking failures and duplicates
    let total_failures = Arc::new(AtomicU32::new(0));
    let total_duplicates = Arc::new(AtomicU32::new(0));
    let should_stop = Arc::new(AtomicBool::new(false));

    // Process in batches for better control and early termination
    let batch_size = 100;
    let max_iterations = n_distributions * MAX_DUP_LOOPS as usize;

    for batch_start in (0..max_iterations).step_by(batch_size) {
        if should_stop.load(Ordering::Relaxed) {
            break;
        }

        // Check if we already have enough distributions
        {
            let unique = unique_distributions.lock().unwrap();
            if unique.len() >= n_distributions {
                break;
            }
        }

        let batch_end = (batch_start + batch_size).min(max_iterations);
        let batch_range = batch_start..batch_end;

        // Parallel processing of the current batch using rayon
        batch_range.into_par_iter().for_each(|_| {
            if should_stop.load(Ordering::Relaxed) {
                return;
            }

            // Check whether the target number of distributions was reached
            {
                let unique = unique_distributions.lock().unwrap();
                if unique.len() >= n_distributions {
                    should_stop.store(true, Ordering::Relaxed);
                    return;
                }
            }

            // Use thread-local RNG for thread safety
            let mut thread_rng = rand::rng();

            match find_distribution_internal(params, &mut thread_rng) {
                Ok(mut distribution) => {
                    distribution.sort_by(|a, b| U::to_i64(a).unwrap().cmp(&U::to_i64(b).unwrap()));

                    // Convert to hashable format for uniqueness check
                    let hashable_values: Vec<i64> =
                        distribution.iter().map(|v| U::to_i64(v).unwrap()).collect();

                    let mut unique = unique_distributions.lock().unwrap();
                    if unique.insert(hashable_values) {
                        results.lock().unwrap().push(distribution);
                        total_failures.store(0, Ordering::Relaxed);
                        total_duplicates.store(0, Ordering::Relaxed);
                    } else {
                        total_duplicates.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    total_failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        // Check early stopping conditions after each batch
        let failures = total_failures.load(Ordering::Relaxed);
        let duplicates = total_duplicates.load(Ordering::Relaxed);

        if failures >= 1000 {
            eprintln!(
                "Warning: Too many failed attempts ({}). Stopping search.",
                failures
            );
            should_stop.store(true, Ordering::Relaxed);
            break;
        }

        let n_found = unique_distributions.lock().unwrap().len() as f64;
        let max_duplications = if n_found > 0.0 {
            (0.00001f64.ln() / (n_found / (n_found + 1.0)).ln()).round() as u32
        } else {
            1000
        }
        .max(1000);

        if duplicates > max_duplications {
            eprintln!(
                "Warning: Found too many duplicate distributions ({}). Stopping search.",
                duplicates
            );
            should_stop.store(true, Ordering::Relaxed);
            break;
        }
    }

    let final_results = results.lock().unwrap().clone();
    let unique_count = unique_distributions.lock().unwrap().len();

    if unique_count < n_distributions {
        eprintln!(
            "Only {} matching distributions could be found.",
            unique_count
        );
    }

    final_results
}

/// Internal function to find a single distribution
fn find_distribution_internal<T, U>(
    params: &SpriteParams<T, U>,
    rng: &mut impl Rng,
) -> Result<Vec<U>, String>
where
    T: FloatType,
    U: IntegerType,
{
    let r_n = params.n_obs as usize - params.n_fixed;
    if params.possible_values_scaled.is_empty() && r_n > 0 {
        return Err("No possible values to sample from for initialization.".to_string());
    }

    // Initialize with random scaled values
    let mut vec: Vec<U> = (0..r_n)
        .map(|_| *params.possible_values_scaled.choose(rng).unwrap())
        .collect();

    // Initial mean adjustment
    let max_loops_mean = params.n_obs * params.possible_values_scaled.len() as u32;
    adjust_mean_internal(&mut vec, params, max_loops_mean, rng)?;

    let max_loops_sd = (params.n_obs * (params.possible_values_scaled.len().pow(2) as u32))
        .clamp(MAX_DELTA_LOOPS_LOWER, MAX_DELTA_LOOPS_UPPER);
    let granule_sd = T::from(0.1f64).unwrap().powi(params.sd_prec) / T::from(2.0).unwrap()
        + T::from(DUST).unwrap();

    for _ in 1..=max_loops_sd {
        // Check for success
        let current_sd: T =
            compute_sd_scaled(&vec, &params.fixed_responses_scaled, params.scale_factor);

        if (current_sd - params.sd).abs() <= granule_sd {
            // Success - combine vec with fixed responses
            let mut full_vec = vec;
            full_vec.extend_from_slice(&params.fixed_responses_scaled);
            return Ok(full_vec);
        }

        // Shift values to adjust SD
        shift_values_internal(&mut vec, params, rng);

        // Check for and correct mean drift
        let current_mean =
            compute_mean_scaled(&vec, &params.fixed_responses_scaled, params.scale_factor);
        let target_mean_rounded =
            T::from(rust_round(T::to_f64(&params.mean).unwrap(), params.m_prec)).unwrap();
        let current_mean_rounded =
            T::from(rust_round(T::to_f64(&current_mean).unwrap(), params.m_prec)).unwrap();

        if !is_near(
            T::to_f64(&current_mean_rounded).unwrap(),
            T::to_f64(&target_mean_rounded).unwrap(),
            DUST,
        ) {
            adjust_mean_internal(&mut vec, params, 20, rng).unwrap_or(());
        }
    }

    // Failed to find a solution
    Err("Could not find a matching distribution within iteration limit".to_string())
}

/// Adjust mean of scaled integer values
fn adjust_mean_internal<T, U>(
    vec: &mut [U],
    params: &SpriteParams<T, U>,
    max_iter: u32,
    rng: &mut impl Rng,
) -> Result<(), String>
where
    T: FloatType,
    U: IntegerType,
{
    if params.possible_values_scaled.is_empty() {
        return Err("Cannot adjust mean with no possible values.".to_string());
    }

    let target_mean = params.mean;

    for _ in 0..max_iter {
        let current_mean =
            compute_mean_scaled(vec, &params.fixed_responses_scaled, params.scale_factor);
        let target_mean_rounded =
            T::from(rust_round(T::to_f64(&target_mean).unwrap(), params.m_prec)).unwrap();
        let current_mean_rounded =
            T::from(rust_round(T::to_f64(&current_mean).unwrap(), params.m_prec)).unwrap();

        if is_near(
            T::to_f64(&current_mean_rounded).unwrap(),
            T::to_f64(&target_mean_rounded).unwrap(),
            DUST,
        ) {
            return Ok(());
        }

        let increase_mean = current_mean < target_mean;
        let min_poss_val = params.possible_values_scaled[0];
        let max_poss_val = params.possible_values_scaled[params.possible_values_scaled.len() - 1];

        let max_attempts = vec.len().max(1) * 4;
        let mut changed = false;

        for _ in 0..max_attempts {
            let index_to_try = rng.random_range(0..vec.len());
            let current_val = vec[index_to_try];

            let is_valid_to_bump = if increase_mean {
                U::to_i64(&current_val).unwrap() < U::to_i64(&max_poss_val).unwrap()
            } else {
                U::to_i64(&current_val).unwrap() > U::to_i64(&min_poss_val).unwrap()
            };

            if is_valid_to_bump {
                if let Some(pos) = params
                    .possible_values_scaled
                    .iter()
                    .position(|&p| U::to_i64(&p).unwrap() == U::to_i64(&current_val).unwrap())
                {
                    let new_pos = if increase_mean {
                        pos + 1
                    } else {
                        pos.saturating_sub(1)
                    };
                    if let Some(&new_val) = params.possible_values_scaled.get(new_pos) {
                        vec[index_to_try] = new_val;
                        changed = true;
                        break;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    let err_msg = if !params.fixed_responses_scaled.is_empty() {
        "Couldn't initialize data with correct mean. This *might* be because the restrictions cannot be satisfied."
    } else {
        "Couldn't initialize data with correct mean. This might indicate a coding error if the mean is in range."
    };
    Err(err_msg.to_string())
}

/// Shift values to adjust standard deviation
fn shift_values_internal<T, U>(
    vec: &mut [U],
    params: &SpriteParams<T, U>,
    rng: &mut impl Rng,
) -> bool
where
    T: FloatType,
    U: IntegerType,
{
    let current_sd: T = compute_sd_scaled(vec, &params.fixed_responses_scaled, params.scale_factor);
    let increase_sd = current_sd < params.sd;

    let max_attempts = vec.len() * 2;

    for _ in 0..max_attempts {
        let i = rng.random_range(0..vec.len());
        let mut j = rng.random_range(0..vec.len());
        while i == j && vec.len() > 1 {
            j = rng.random_range(0..vec.len());
        }
        if i == j {
            continue;
        }

        let val1 = vec[i];
        let val2 = vec[j];
        let val1_i64 = U::to_i64(&val1).unwrap();
        let val2_i64 = U::to_i64(&val2).unwrap();

        // Find next larger value for val1
        let val1_new_opt = params
            .possible_values_scaled
            .iter()
            .find(|&&v| U::to_i64(&v).unwrap() > val1_i64);

        // Find next smaller value for val2
        let val2_new_opt = params
            .possible_values_scaled
            .iter()
            .rev()
            .find(|&&v| U::to_i64(&v).unwrap() < val2_i64);

        if let (Some(&val1_new), Some(&val2_new)) = (val1_new_opt, val2_new_opt) {
            let val1_new_i64 = U::to_i64(&val1_new).unwrap();
            let val2_new_i64 = U::to_i64(&val2_new).unwrap();

            // Check if the swap preserves mean
            if val1_new_i64 - val1_i64 == val2_i64 - val2_new_i64 {
                // Check if this swap increases/decreases SD as needed
                let scale_f = params.scale_factor as f64;
                let v1_f = val1_i64 as f64 / scale_f;
                let v2_f = val2_i64 as f64 / scale_f;
                let v1_new_f = val1_new_i64 as f64 / scale_f;
                let v2_new_f = val2_new_i64 as f64 / scale_f;

                let sd_before = std_dev(&[v1_f, v2_f]).unwrap_or(0.0);
                let sd_after = std_dev(&[v1_new_f, v2_new_f]).unwrap_or(0.0);

                let is_pointless = if increase_sd {
                    sd_after <= sd_before
                } else {
                    sd_after >= sd_before
                };

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

/// Compute mean from scaled integer values
fn compute_mean_scaled<T, U>(vec: &[U], fixed_vals: &[U], scale_factor: u32) -> T
where
    T: FloatType,
    U: IntegerType,
{
    let sum_vec: i64 = vec.iter().map(|v| U::to_i64(v).unwrap()).sum();
    let sum_fixed: i64 = fixed_vals.iter().map(|v| U::to_i64(v).unwrap()).sum();
    let total_len = vec.len() + fixed_vals.len();
    let total_sum = sum_vec + sum_fixed;

    T::from((total_sum as f64) / (scale_factor as f64) / (total_len as f64)).unwrap()
}

/// Compute standard deviation from scaled integer values
fn compute_sd_scaled<T, U>(vec: &[U], fixed_vals: &[U], scale_factor: u32) -> T
where
    T: FloatType,
    U: IntegerType,
{
    let scale_f = scale_factor as f64;
    let combined_mean = compute_mean_scaled(vec, fixed_vals, scale_factor);
    let combined_mean_f64 = T::to_f64(&combined_mean).unwrap();

    let sum_sq_diff_vec: f64 = vec
        .iter()
        .map(|v| {
            let val_f = U::to_i64(v).unwrap() as f64 / scale_f;
            (val_f - combined_mean_f64).powi(2)
        })
        .sum();

    let sum_sq_diff_fixed: f64 = fixed_vals
        .iter()
        .map(|v| {
            let val_f = U::to_i64(v).unwrap() as f64 / scale_f;
            (val_f - combined_mean_f64).powi(2)
        })
        .sum();

    let total_len = vec.len() + fixed_vals.len();
    let variance = (sum_sq_diff_vec + sum_sq_diff_fixed) / (total_len - 1) as f64;

    T::from(variance.sqrt()).unwrap()
}

/// Iteratively adjusts a vector's values to match a target mean (legacy, deprecated)
///
/// This function randomly selects elements and nudges them up or down according
/// to a list of possible values until the rounded mean of the full dataset
/// matches the target
#[allow(deprecated)]
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
                if let Some(pos) = poss_values
                    .iter()
                    .position(|&p| is_near(p, current_val, DUST))
                {
                    let new_pos = if increase_mean {
                        pos + 1
                    } else {
                        pos.saturating_sub(1)
                    };
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
#[allow(deprecated)]
pub fn shift_values(vec: &mut [f64], params: &SpriteParameters, rng: &mut impl Rng) -> bool {
    // In-place SD calculation
    let sum_vec: f64 = vec.iter().sum();
    let sum_fixed: f64 = params.fixed_responses.iter().sum();
    let total_len = vec.len() + params.fixed_responses.len();
    let combined_mean = (sum_vec + sum_fixed) / total_len as f64;
    let sum_sq_diff_vec: f64 = vec.iter().map(|&v| (v - combined_mean).powi(2)).sum();
    let sum_sq_diff_fixed: f64 = params
        .fixed_responses
        .iter()
        .map(|&v| (v - combined_mean).powi(2))
        .sum();
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

        let val1_new_opt = params
            .possible_values
            .iter()
            .find(|&&v| v > val1 && !is_near(v, val1, DUST));
        let val2_new_opt = params
            .possible_values
            .iter()
            .rev()
            .find(|&&v| v < val2 && !is_near(v, val2, DUST));

        if let (Some(&val1_new), Some(&val2_new)) = (val1_new_opt, val2_new_opt) {
            if is_near(val1_new - val1, val2 - val2_new, DUST) {
                let sd_before = std_dev(&[val1, val2]).unwrap_or(0.0);
                let sd_after = std_dev(&[val1_new, val2_new]).unwrap_or(0.0);

                let is_pointless = if increase_sd {
                    sd_after <= sd_before
                } else {
                    sd_after >= sd_before
                };

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
#[allow(deprecated)]
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
    adjust_mean(
        &mut vec,
        &params.fixed_responses,
        &params.possible_values,
        params.mean,
        params.m_prec,
        max_loops_mean,
        rng,
    )?;

    let max_loops_sd = (params.n_obs * (params.possible_values.len().pow(2) as u32))
        .clamp(MAX_DELTA_LOOPS_LOWER, MAX_DELTA_LOOPS_UPPER);
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
        let sum_sq_diff_fixed: f64 = params
            .fixed_responses
            .iter()
            .map(|&v| (v - combined_mean).powi(2))
            .sum();

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
                rng,
            )
            .unwrap_or(());
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
#[allow(deprecated)]
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
        }
        .max(100);

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
                        let hashable_values: Vec<i64> = res
                            .values
                            .iter()
                            .map(|v| (v * 1_000_000.0).round() as i64)
                            .collect();

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
        println!(
            "Only {} matching distributions could be found.",
            unique_distributions.len()
        );
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
    let variance = data
        .iter()
        .map(|value| {
            let diff = value - data_mean;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1) as f64; // Use n-1 for sample standard deviation

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
    use rand::SeedableRng;

    /// Helper to convert scaled integers to floats
    fn unscale_distribution(scaled_values: &[i32], scale_factor: u32) -> Vec<f64> {
        scaled_values
            .iter()
            .map(|&v| v as f64 / scale_factor as f64)
            .collect()
    }

    #[test]
    fn sprite_test_mean() {
        let mut rng = StdRng::seed_from_u64(1234);
        let results = sprite_parallel(
            2.2_f64,
            1.3_f64,
            20,
            1,
            5,
            None,
            None,
            1,
            5,
            None,
            RestrictionsOption::Default,
            false,
            None, // no parquet config
            &mut rng,
        )
        .unwrap();

        // Convert first result to floats and check mean
        // Results are in 100x scale (standard for CLOSURE/SPRITE statistics)
        let first_dist = unscale_distribution(&results.results.sample[0], 100);
        let computed_mean = mean(&first_dist);
        assert_eq!(rust_round(computed_mean, 1), 2.2);
    }

    #[test]
    fn sprite_test_sd() {
        let mut rng = StdRng::seed_from_u64(1234);
        let results = sprite_parallel(
            2.2_f64,
            1.3_f64,
            20,
            1,
            5,
            None,
            None,
            1,
            5,
            None,
            RestrictionsOption::Default,
            false,
            None, // no parquet config
            &mut rng,
        )
        .unwrap();

        // Check all distributions
        // Results are in 100x scale (standard for CLOSURE/SPRITE statistics)
        for dist_scaled in &results.results.sample {
            let dist = unscale_distribution(dist_scaled, 100);
            let computed_sd = std_dev(&dist).unwrap();
            assert_eq!(rust_round(computed_sd, 1), 1.3);
        }
    }

    #[test]
    fn sprite_test_big() {
        // What is the target mean and SD?
        let test_mean = 26.281;
        let test_sd = 14.6339;

        // To how many decimal places were they reported?
        let test_mean_digits = 3;
        let test_sd_digits = 4;

        // What is the target sample size?
        let test_n = 2000;

        // How many distributions should SPRITE generate?
        let target_runs = 5000;

        let mut rng = StdRng::seed_from_u64(42);

        let results = sprite_parallel(
            test_mean,
            test_sd,
            test_n,
            1,
            50,
            Some(test_mean_digits),
            Some(test_sd_digits),
            1,
            target_runs,
            None,
            RestrictionsOption::Default,
            true,
            None, // no parquet config
            &mut rng,
        )
        .unwrap();

        println!(
            "Number of target results: {:?}\n",
            results.results.sample.len()
        );
        println!("First result (scaled):\n");
        println!("{:?}", &results.results.sample[0][..10]); // Print first 10 values

        // Results are in 100x scale (standard for CLOSURE/SPRITE statistics)
        let scale_factor = 100;

        // Go through the SPRITE results and check if they conform to the
        // input mean and SD. If any result sample doesn't, throw an error.
        for dist_scaled in &results.results.sample {
            let dist = unscale_distribution(dist_scaled, scale_factor);
            let computed_mean = mean(&dist);
            let computed_sd = std_dev(&dist).unwrap();
            let rounded_mean = rust_round(computed_mean, test_mean_digits);
            let rounded_sd = rust_round(computed_sd, test_sd_digits);

            assert_eq!(rounded_mean, test_mean);
            assert_eq!(rounded_sd, test_sd);
        }
    }

    #[test]
    fn test_sprite_parallel_streaming() {
        // Test streaming mode with separate files
        let config = StreamingConfig {
            file_path: "test_sprite_streaming/".to_string(),
            batch_size: 100,
            show_progress: false,
        };

        let _ = std::fs::create_dir("test_sprite_streaming");

        let result = sprite_parallel_streaming::<f64, i32>(
            2.2,  // mean
            1.3,  // sd
            20,   // n_obs
            1,    // min_val
            5,    // max_val
            None, // m_prec
            None, // sd_prec
            1,    // n_items
            None, // restrictions_exact
            RestrictionsOption::Default,
            false, // dont_test
            config,
            Some(10), // stop_after - find 10 distributions
        );

        assert!(result.total_combinations > 0);
        assert_eq!(result.file_path, "test_sprite_streaming/");

        // Check that files were created
        assert!(std::path::Path::new("test_sprite_streaming/samples.parquet").exists());
        assert!(std::path::Path::new("test_sprite_streaming/horns.parquet").exists());

        // Clean up test files
        let _ = std::fs::remove_file("test_sprite_streaming/samples.parquet");
        let _ = std::fs::remove_file("test_sprite_streaming/horns.parquet");
        let _ = std::fs::remove_file("test_sprite_streaming/metrics_main.parquet");
        let _ = std::fs::remove_file("test_sprite_streaming/metrics_horns.parquet");
        let _ = std::fs::remove_file("test_sprite_streaming/frequency.parquet");
        let _ = std::fs::remove_dir("test_sprite_streaming");
    }

    #[test]
    fn test_sprite_parallel_with_parquet() {
        // Test in-memory mode with optional Parquet output
        let config = ParquetConfig {
            file_path: "test_sprite_parquet/".to_string(),
            batch_size: 100,
        };

        let _ = std::fs::create_dir("test_sprite_parquet");

        let mut rng = StdRng::seed_from_u64(42);

        let results = sprite_parallel::<f64, i32>(
            2.2_f64,
            1.3_f64,
            20,
            1,
            5,
            None,
            None,
            1,
            5,
            None,
            RestrictionsOption::Default,
            false,
            Some(config),
            &mut rng,
        )
        .unwrap();

        // Check that results were returned
        assert!(!results.results.sample.is_empty());

        // Check that files were created
        assert!(std::path::Path::new("test_sprite_parquet/results.parquet").exists());
        assert!(std::path::Path::new("test_sprite_parquet/metrics_main.parquet").exists());
        assert!(std::path::Path::new("test_sprite_parquet/metrics_horns.parquet").exists());
        assert!(std::path::Path::new("test_sprite_parquet/frequency.parquet").exists());

        // Clean up test files
        let _ = std::fs::remove_file("test_sprite_parquet/results.parquet");
        let _ = std::fs::remove_file("test_sprite_parquet/metrics_main.parquet");
        let _ = std::fs::remove_file("test_sprite_parquet/metrics_horns.parquet");
        let _ = std::fs::remove_file("test_sprite_parquet/frequency.parquet");
        let _ = std::fs::remove_dir("test_sprite_parquet");
    }
}
