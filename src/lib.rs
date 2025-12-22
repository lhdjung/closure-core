//! CLOSURE: complete listing of original samples of underlying raw evidence
//!
//! Crate closure-core implements the CLOSURE technique for efficiently reconstructing
//! all possible distributions of raw data from summary statistics. It is not
//! about the Rust feature called closure.
//!
//! The crate is mostly meant to serve as a backend for the R package [unsum](https://lhdjung.github.io/unsum/).
//! The main APIs users need are `dfs_parallel()` for in-memory results and
//! `dfs_parallel_streaming()` for memory-efficient file output.
//!
//! Most of the code was written by Claude 3.5, translating Python code by Nathanael Larigaldie.

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int32Builder, ListBuilder, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use num::{Float, FromPrimitive, Integer, NumCast, ToPrimitive};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};

/// Trait alias for floating-point types used in CLOSURE computations
pub trait FloatType: Float + FromPrimitive + Send + Sync {}
impl<T> FloatType for T where T: Float + FromPrimitive + Send + Sync {}

/// Trait alias for integer types used in CLOSURE computations
pub trait IntegerType: Integer + NumCast + ToPrimitive + Copy + Send + Sync {}
impl<T> IntegerType for T where T: Integer + NumCast + ToPrimitive + Copy + Send + Sync {}

pub mod distribution_finder;
pub mod grimmer;
pub mod sprite;
pub mod sprite_types;

/// Configuration for Parquet output in memory mode
/// Used with `dfs_parallel()` to optionally save results while returning them
pub struct ParquetConfig {
    pub file_path: String,
    pub batch_size: usize,
}

/// Configuration for streaming mode
/// Used with `dfs_parallel_streaming()` for memory-efficient processing
pub struct StreamingConfig {
    pub file_path: String,
    pub batch_size: usize,
    pub show_progress: bool,
}

/// Result of streaming operation
pub struct StreamingResult {
    pub total_combinations: usize,
    pub file_path: String,
}

pub struct SamplesSubset {
    pub subset: Vec<String>,
}

/// Sample category for frequency tables
///
/// Automatically provides iteration, count, and snake_case string conversion via strum.
///
/// **Important**: While strum makes iteration robust, these three variants have specific
/// semantic meaning in the CLOSURE algorithm:
/// - `All`: Frequencies across all samples
/// - `HornsMin`: Frequencies for samples with minimum horns index
/// - `HornsMax`: Frequencies for samples with maximum horns index
///
/// Adding new variants requires corresponding changes to the frequency calculation logic
/// in `samples_to_result_list()` and `write_streaming_statistics()` to define what
/// samples belong to the new category and how to calculate their frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCountMacro, EnumIter, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum SampleCategory {
    All,
    HornsMin,
    HornsMax,
}

impl SampleCategory {
    /// Convert to snake_case string representation
    pub fn as_str(&self) -> &'static str {
        self.into()
    }

    /// Returns an iterator over all sample category variants in declaration order
    pub fn all() -> impl Iterator<Item = Self> + Clone {
        Self::iter()
    }

    /// Returns an iterator over all sample category names in snake_case
    pub fn all_names() -> impl Iterator<Item = &'static str> + Clone {
        Self::iter().map(|variant| variant.as_str())
    }
}

/// Type-safe samples column for frequency tables
///
/// Enforces the structure: "all" repeated x times, then "horns_min" repeated x times,
/// then "horns_max" repeated x times, where x is the number of scale values.
///
/// # Example
/// ```ignore
/// // For scale 1..=5 (5 values), creates:
/// // ["all", "all", "all", "all", "all",
/// //  "horns_min", "horns_min", "horns_min", "horns_min", "horns_min",
/// //  "horns_max", "horns_max", "horns_max", "horns_max", "horns_max"]
/// let samples = FrequencySamplesColumn::new(5);
/// ```
#[derive(Clone, Debug)]
pub struct FrequencySamplesColumn {
    /// Number of times each category is repeated (number of scale values)
    repetitions: usize,
}

impl FrequencySamplesColumn {
    /// Create a new samples column with the given number of repetitions per category
    ///
    /// # Parameters
    /// - `repetitions`: Number of scale values (determines how many times each category appears)
    pub fn new(repetitions: usize) -> Self {
        Self { repetitions }
    }

    /// Convert to a Vec<String> for compatibility with existing code
    ///
    /// Returns a vector with category names repeated x times each, in declaration order.
    pub fn to_vec(&self) -> Vec<String> {
        let mut result = Vec::with_capacity(self.repetitions * SampleCategory::COUNT);

        for name in SampleCategory::all_names() {
            for _ in 0..self.repetitions {
                result.push(name.to_string());
            }
        }

        result
    }


    /// Get the total length of the samples column
    pub fn len(&self) -> usize {
        self.repetitions * SampleCategory::COUNT
    }

    /// Check if the column is empty (repetitions == 0)
    pub fn is_empty(&self) -> bool {
        self.repetitions == 0
    }

    /// Get the number of repetitions per category
    pub fn repetitions(&self) -> usize {
        self.repetitions
    }

    /// Get the category at a given index
    ///
    /// # Panics
    /// Panics if index >= self.len()
    pub fn get(&self, index: usize) -> SampleCategory {
        assert!(index < self.len(), "Index out of bounds");

        // Calculate which category this index belongs to
        let category_index = index / self.repetitions;

        // Use strum's iterator to get the variant at this position
        // This automatically adapts to any changes in SampleCategory
        SampleCategory::iter()
            .nth(category_index)
            .expect("category_index should always be valid based on len() check")
    }

    /// Get the category at a given index as a string
    pub fn get_str(&self, index: usize) -> &'static str {
        self.get(index).as_str()
    }
}

/// Combined frequency data for a set of samples
/// Each row represents frequency data for a specific value in a specific sample group
///
/// Invariant: All fields must have the same length to ensure valid data frame structure
#[derive(Clone, Debug)]
pub struct FrequencyTable {
    /// Sample categories: "all", "horns_min", "horns_max" each repeated for all scale values
    samples_group: FrequencySamplesColumn,
    value: Vec<i32>,
    f_average: Vec<f64>,
    f_absolute: Vec<f64>,
    f_relative: Vec<f64>,
}

impl FrequencyTable {
    /// Create a new FrequencyTable, validating that all fields have the same length
    ///
    /// # Panics
    /// Panics if the lengths of value, f_average, f_absolute, or f_relative don't match
    /// the length of samples_group
    pub fn new(
        samples_group: FrequencySamplesColumn,
        value: Vec<i32>,
        f_average: Vec<f64>,
        f_absolute: Vec<f64>,
        f_relative: Vec<f64>,
    ) -> Self {
        let expected_len = samples_group.len();

        let name_len_tuples = [
            ("value", value.len()),
            ("f_average", f_average.len()),
            ("f_absolute", f_absolute.len()),
            ("f_relative", f_relative.len()),
        ];

        // Validate all field lengths match
        for (name, len) in name_len_tuples {
            assert_eq!(
                len, expected_len,
                "can't create a FrequencyTable: `{}` length ({}) doesn't match `samples_group` length ({})",
                name, len, expected_len
            );
        }

        Self {
            samples_group,
            value,
            f_average,
            f_absolute,
            f_relative,
        }
    }

    /// Get the number of rows in this frequency table
    pub fn len(&self) -> usize {
        self.samples_group.len()
    }

    /// Check if the frequency table is empty
    pub fn is_empty(&self) -> bool {
        self.samples_group.is_empty()
    }

    /// Get a reference to the samples_group column
    pub fn samples_group(&self) -> &FrequencySamplesColumn {
        &self.samples_group
    }

    /// Get a reference to the value column
    pub fn value(&self) -> &[i32] {
        &self.value
    }

    /// Get a reference to the f_average column
    pub fn f_average(&self) -> &[f64] {
        &self.f_average
    }

    /// Get a reference to the f_absolute column
    pub fn f_absolute(&self) -> &[f64] {
        &self.f_absolute
    }

    /// Get a reference to the f_relative column
    pub fn f_relative(&self) -> &[f64] {
        &self.f_relative
    }
}

/// Main metrics about the CLOSURE results
#[derive(Clone, Debug)]
pub struct MetricsMain {
    pub samples_initial: f64,
    pub samples_all: f64,
    pub values_all: f64,
}

/// Horns-specific metrics
#[derive(Clone, Debug)]
pub struct MetricsHorns {
    pub mean: f64,
    pub uniform: f64,
    pub sd: f64,
    pub cv: f64,
    pub mad: f64,
    pub min: f64,
    pub median: f64,
    pub max: f64,
    pub range: f64,
}

/// Results table combining samples and their horns values
#[derive(Clone, Debug)]
pub struct ResultsTable<U> {
    pub id: Vec<usize>,
    pub sample: Vec<Vec<U>>,
    pub horns_values: Vec<f64>,
}

/// Complete CLOSURE results with all statistics
#[derive(Clone, Debug)]
pub struct ResultListFromMeanSdN<U> {
    pub metrics_main: MetricsMain,
    pub metrics_horns: MetricsHorns,
    pub frequency: FrequencyTable,
    pub results: ResultsTable<U>,
}

/// Context for CLOSURE search containing precomputed bounds and lookup tables
/// for efficient DFS pruning during sample space exploration
struct ClosureSearchContext<T, U> {
    /// Upper bound for target sum (mean * n + rounding_error_mean * n)
    target_sum_upper: T,
    /// Lower bound for target sum (mean * n - rounding_error_mean * n)
    target_sum_lower: T,
    /// Upper bound for standard deviation
    sd_upper: T,
    /// Lower bound for standard deviation
    sd_lower: T,
    /// Sample size as usize for array indexing
    n_usize: usize,
    /// Lookup table: min_scale_sum[value][n_left] = minimum possible sum for n_left remaining positions
    min_scale_sum: Vec<Vec<T>>,
    /// Lookup table: scale_max_sum[n_left] = maximum possible sum for n_left remaining positions
    scale_max_sum: Vec<T>,
    /// Sample size minus one (n - 1)
    n_minus_1: U,
    /// Maximum scale value plus one (scale_max + 1) for range operations
    scale_max_plus_1: U,
}

/// Implements range over Rint-friendly generic integer type U
struct IntegerRange<U>
where
    U: Integer + Copy,
{
    current: U,
    end: U,
}

impl<U> Iterator for IntegerRange<U>
where
    U: Integer + Copy,
{
    type Item = U;

    /// Increment over U type integers
    fn next(&mut self) -> Option<U> {
        if self.current < self.end {
            let next = self.current;
            self.current = self.current + U::one();
            Some(next)
        } else {
            None
        }
    }
}

/// Creates an iterator over the space of U type integers
fn range_u<U: Integer + Copy>(start: U, end: U) -> IntegerRange<U> {
    IntegerRange {
        current: start,
        end,
    }
}

// Define the Combination struct
#[derive(Clone)]
struct Combination<T, U> {
    values: Vec<U>,
    running_sum: T,
    running_m2: T,
}

/// Count first set of integers
///
/// The first set of integers that can be formed
/// given a range defined by `scale_min` and `scale_max`.
/// This function calculates the number of unique pairs (i, j) where i and j are integers
/// within the specified range, and i <= j.
/// # Arguments
/// * `scale_min` - The minimum value of the scale.
/// * `scale_max` - The maximum value of the scale.
/// # Returns
/// The total number of unique combinations of integers within the specified range.
pub fn count_initial_combinations(scale_min: i32, scale_max: i32) -> i32 {
    let range_size = scale_max - scale_min + 1;
    (range_size * (range_size + 1)) / 2
}

/// Calculate horns index for a frequency distribution
fn calculate_horns(freqs: &[f64], scale_min: i32, scale_max: i32) -> f64 {
    let scale_values: Vec<f64> = (scale_min..=scale_max).map(|v| v as f64).collect();

    let total: f64 = freqs.iter().sum();
    if total == 0.0 {
        return 0.0;
    }

    let freqs_relative: Vec<f64> = freqs.iter().map(|f| f / total).collect();

    // Calculate mean
    let mean: f64 = scale_values
        .iter()
        .zip(freqs_relative.iter())
        .map(|(v, f)| v * f)
        .sum();

    // Calculate weighted sum of squared deviations
    let numerator: f64 = scale_values
        .iter()
        .zip(freqs_relative.iter())
        .map(|(v, f)| f * (v - mean).powi(2))
        .sum();

    // Maximum possible variance given scale limits
    let denominator = ((scale_max - scale_min) as f64).powi(2) / 4.0;

    numerator / denominator
}

/// Calculate horns index for a uniform distribution
fn calculate_horns_uniform(scale_min: i32, scale_max: i32) -> f64 {
    let n_values = (scale_max - scale_min + 1) as usize;
    let uniform_freqs = vec![1.0; n_values];
    calculate_horns(&uniform_freqs, scale_min, scale_max)
}

/// Calculate frequency data for a specific set of samples
fn calculate_frequency_rows<U>(
    samples: &[Vec<U>],
    scale_min: U,
    scale_max: U,
) -> (Vec<i32>, Vec<f64>, Vec<f64>, Vec<f64>)
where
    U: Integer + ToPrimitive + Copy,
{
    let scale_min_i32 = U::to_i32(&scale_min).unwrap();
    let scale_max_i32 = U::to_i32(&scale_max).unwrap();

    let nrow_frequency = (scale_max_i32 - scale_min_i32 + 1) as usize;

    let mut f_absolute = vec![0.0; nrow_frequency];
    let n_samples = samples.len() as f64;

    // Count frequencies
    for sample in samples {
        for &value in sample {
            let idx = (U::to_i32(&value).unwrap() - scale_min_i32) as usize;
            f_absolute[idx] += 1.0;
        }
    }

    // Calculate derived values
    let total_values: f64 = f_absolute.iter().sum();
    let value: Vec<i32> = (scale_min_i32..=scale_max_i32).collect();
    let f_average: Vec<f64> = f_absolute.iter().map(|&f| f / n_samples).collect();
    let f_relative: Vec<f64> = f_absolute.iter().map(|&f| f / total_values).collect();

    (value, f_average, f_absolute, f_relative)
}

/// Calculate median of a sorted vector
fn median(sorted: &[f64]) -> f64 {
    let len = sorted.len();
    if len.is_multiple_of(2) {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Calculate median absolute deviation
fn mad(values: &[f64], median_val: f64) -> f64 {
    let mut deviations: Vec<f64> = values.iter().map(|&v| (v - median_val).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    median(&deviations)
}

/// Create an empty result list for cases with no valid distributions
///
/// Returns a ResultListFromMeanSdN with all metrics set to NaN or zero
/// (depending on the metric), and with empty result vectors. Used when no
/// distributions are found or when an algorithm returns early with no results.
///
/// # Parameters
/// - `scale_min`: Minimum value in the scale range
/// - `scale_max`: Maximum value in the scale range
///
/// # Returns
/// An empty ResultListFromMeanSdN<U> with appropriate structure
pub fn empty_result_list<U>(scale_min: U, scale_max: U) -> ResultListFromMeanSdN<U>
where
    U: Integer + ToPrimitive + Copy,
{
    let scale_min_i32 = U::to_i32(&scale_min).unwrap();
    let scale_max_i32 = U::to_i32(&scale_max).unwrap();
    
    let group_size = (scale_max_i32 - scale_min_i32 + 1) as usize;
    let nrow_frequency = group_size * 3;

    // Build value vector repeated 3 times (for all, horns_min, horns_max)
    let base_values: Vec<i32> = (scale_min_i32..=scale_max_i32).collect();
    let mut value = Vec::with_capacity(nrow_frequency);
    for _ in 0..3 {
        value.extend_from_slice(&base_values);
    }

    ResultListFromMeanSdN {
        metrics_main: MetricsMain {
            samples_initial: 0.0,
            samples_all: 0.0,
            values_all: 0.0,
        },
        metrics_horns: MetricsHorns {
            mean: f64::NAN,
            uniform: f64::NAN,
            sd: f64::NAN,
            cv: f64::NAN,
            mad: f64::NAN,
            min: f64::NAN,
            median: f64::NAN,
            max: f64::NAN,
            range: f64::NAN,
        },
        frequency: FrequencyTable::new(
            FrequencySamplesColumn::new(group_size),
            value,
            vec![f64::NAN; nrow_frequency],
            vec![0.0; nrow_frequency],
            vec![f64::NAN; nrow_frequency],
        ),
        results: ResultsTable {
            id: Vec::new(),
            sample: Vec::new(),
            horns_values: Vec::new(),
        },
    }
}

/// Calculate all statistics for the samples
fn samples_to_result_list<U>(
    samples: Vec<Vec<U>>,
    scale_min: U,
    scale_max: U,
) -> ResultListFromMeanSdN<U>
where
    U: Integer + ToPrimitive + Copy,
{
    let scale_min_i32 = U::to_i32(&scale_min).unwrap();
    let scale_max_i32 = U::to_i32(&scale_max).unwrap();

    let group_size = (scale_max_i32 - scale_min_i32 + 1) as usize;

    // Handle empty samples case
    if samples.is_empty() {
        return empty_result_list(scale_min, scale_max);
    }

    let n = samples[0].len();
    let samples_all = samples.len();
    let values_all = samples_all * n;

    // Calculate horns for each sample
    let mut horns_values = Vec::with_capacity(samples_all);
    for sample in &samples {
        let mut freqs = vec![0.0; group_size];
        for &value in sample {
            let idx = (U::to_i32(&value).unwrap() - scale_min_i32) as usize;
            freqs[idx] += 1.0;
        }
        horns_values.push(calculate_horns(&freqs, scale_min_i32, scale_max_i32));
    }

    // Calculate horns statistics
    let horns_mean = horns_values.iter().sum::<f64>() / samples_all as f64;
    let horns_sd = {
        let variance = horns_values
            .iter()
            .map(|&h| (h - horns_mean).powi(2))
            .sum::<f64>()
            / samples_all as f64;
        variance.sqrt()
    };

    let mut horns_sorted = horns_values.clone();
    horns_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let horns_min = horns_sorted[0];
    let horns_max = horns_sorted[samples_all - 1];
    let horns_median = median(&horns_sorted);
    let horns_mad = mad(&horns_values, horns_median);

    // Find indices of samples with min/max horns
    let min_indices: Vec<usize> = horns_values
        .iter()
        .enumerate()
        .filter(|(_, &h)| (h - horns_min).abs() < 1e-10)
        .map(|(i, _)| i)
        .collect();

    let max_indices: Vec<usize> = horns_values
        .iter()
        .enumerate()
        .filter(|(_, &h)| (h - horns_max).abs() < 1e-10)
        .map(|(i, _)| i)
        .collect();

    let min_samples: Vec<Vec<U>> = min_indices.iter().map(|&i| samples[i].clone()).collect();
    let max_samples: Vec<Vec<U>> = max_indices.iter().map(|&i| samples[i].clone()).collect();

    // Calculate frequency data for all three groups
    let (
        all_value,
        all_f_average,
        all_f_absolute,
        all_f_relative,
    ) = calculate_frequency_rows(&samples, scale_min, scale_max);

    let (
        min_value,
        min_f_average,
        min_f_absolute,
        min_f_relative,
    ) = calculate_frequency_rows(&min_samples, scale_min, scale_max);

    let (
        max_value,
        max_f_average,
        max_f_absolute,
        max_f_relative,
    ) = calculate_frequency_rows(&max_samples, scale_min, scale_max);

    // Combine all frequency data into a single table
    let mut combined_value = all_value;
    combined_value.extend(min_value);
    combined_value.extend(max_value);

    let mut combined_f_average = all_f_average;
    combined_f_average.extend(min_f_average);
    combined_f_average.extend(max_f_average);

    let mut combined_f_absolute = all_f_absolute;
    combined_f_absolute.extend(min_f_absolute);
    combined_f_absolute.extend(max_f_absolute);

    let mut combined_f_relative = all_f_relative;
    combined_f_relative.extend(min_f_relative);
    combined_f_relative.extend(max_f_relative);

    // Create ID column for results table
    let id: Vec<usize> = (1..=samples_all).collect();

    ResultListFromMeanSdN {
        metrics_main: MetricsMain {
            samples_initial: count_initial_combinations(scale_min_i32, scale_max_i32) as f64,
            samples_all: samples_all as f64,
            values_all: values_all as f64,
        },
        metrics_horns: MetricsHorns {
            mean: horns_mean,
            uniform: calculate_horns_uniform(scale_min_i32, scale_max_i32),
            sd: horns_sd,
            cv: horns_sd / horns_mean,
            mad: horns_mad,
            min: horns_min,
            median: horns_median,
            max: horns_max,
            range: horns_max - horns_min,
        },
        frequency: FrequencyTable::new(
            FrequencySamplesColumn::new(group_size),
            combined_value,
            combined_f_average,
            combined_f_absolute,
            combined_f_relative,
        ),
        results: ResultsTable {
            id,
            sample: samples,
            horns_values,
        },
    }
}

/// Create a Parquet writer for the results table with appropriate schema
/// Now stores samples as a list column instead of expanding them
fn create_results_writer(file_path: &str) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
    // Create schema with id column, samples as list column, plus horns column
    // Note: List items are marked as nullable to match what Arrow's ListBuilder produces
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "samples",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ), // true for nullable items
        Field::new("horns", DataType::Float64, false),
    ];

    let schema = Arc::new(Schema::new(fields));

    let file = File::create(file_path)?;
    let props = WriterProperties::builder().build();
    let writer = ArrowWriter::try_new(file, schema, Some(props))?;

    Ok(writer)
}

/// Create a simple Parquet writer for samples only
/// For streaming: each row is a sample, with positions as columns (pos1, pos2, ..., posN)
/// This allows streaming while maintaining a fixed schema
fn create_samples_writer(
    file_path: &str,
    sample_size: usize,
) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
    // Create schema where each position in the sample is a column
    // Column names will be pos1, pos2, pos3, etc.
    let fields: Vec<Field> = (1..=sample_size)
        .map(|i| Field::new(format!("pos{}", i), DataType::Int32, false))
        .collect();

    let schema = Arc::new(Schema::new(fields));

    let file = File::create(file_path)?;
    let props = WriterProperties::builder().build();
    let writer = ArrowWriter::try_new(file, schema, Some(props))?;

    Ok(writer)
}

/// Create a simple Parquet writer for horns values only
fn create_horns_writer(file_path: &str) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
    // Schema with just a horns column
    let fields = vec![Field::new("horns", DataType::Float64, false)];

    let schema = Arc::new(Schema::new(fields));

    let file = File::create(file_path)?;
    let props = WriterProperties::builder().build();
    let writer = ArrowWriter::try_new(file, schema, Some(props))?;

    Ok(writer)
}

/// Convert samples to a RecordBatch for the samples-only file
/// Each row is a sample, with positions as columns for R compatibility
fn samples_to_record_batch<U>(samples: &[Vec<U>]) -> Result<RecordBatch, Box<dyn std::error::Error>>
where
    U: Integer + ToPrimitive + Copy,
{
    if samples.is_empty() {
        return Err("No samples to write".into());
    }

    let sample_size = samples[0].len();

    // Create arrays for each position (column)
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(sample_size);

    // For each position in the samples
    for pos in 0..sample_size {
        // Collect values at this position from all samples
        let values: Vec<i32> = samples
            .iter()
            .map(|sample| U::to_i32(&sample[pos]).unwrap())
            .collect();
        arrays.push(Arc::new(Int32Array::from(values)));
    }

    // Create schema with column names pos1, pos2, pos3, etc.
    let fields: Vec<Field> = (1..=sample_size)
        .map(|i| Field::new(format!("pos{}", i), DataType::Int32, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(schema, arrays).map_err(|e| e.into())
}

/// Convert horns values to a RecordBatch
fn horns_to_record_batch(horns_values: &[f64]) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let horns_array = Arc::new(Float64Array::from(horns_values.to_vec()));

    // Create schema
    let fields = vec![Field::new("horns", DataType::Float64, false)];
    let schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(schema, vec![horns_array]).map_err(|e| e.into())
}

/// Create writers for statistics tables
fn create_stats_writers(
    base_path: &str,
) -> Result<
    (
        ArrowWriter<File>,
        ArrowWriter<File>,
        ArrowWriter<File>,
        Arc<Schema>,
        Arc<Schema>,
        Arc<Schema>,
    ),
    Box<dyn std::error::Error>,
> {
    // Metrics main writer
    let metrics_main_schema = Arc::new(Schema::new(vec![
        Field::new("samples_initial", DataType::Float64, false),
        Field::new("samples_all", DataType::Float64, false),
        Field::new("values_all", DataType::Float64, false),
    ]));
    let metrics_main_file = File::create(format!("{}metrics_main.parquet", base_path))?;
    let metrics_main_writer =
        ArrowWriter::try_new(metrics_main_file, metrics_main_schema.clone(), None)?;

    // Metrics horns writer
    let metrics_horns_schema = Arc::new(Schema::new(vec![
        Field::new("mean", DataType::Float64, false),
        Field::new("uniform", DataType::Float64, false),
        Field::new("sd", DataType::Float64, false),
        Field::new("cv", DataType::Float64, false),
        Field::new("mad", DataType::Float64, false),
        Field::new("min", DataType::Float64, false),
        Field::new("median", DataType::Float64, false),
        Field::new("max", DataType::Float64, false),
        Field::new("range", DataType::Float64, false),
    ]));
    let metrics_horns_file = File::create(format!("{}metrics_horns.parquet", base_path))?;
    let metrics_horns_writer =
        ArrowWriter::try_new(metrics_horns_file, metrics_horns_schema.clone(), None)?;

    // Frequency writer with samples column first
    let frequency_schema = Arc::new(Schema::new(vec![
        Field::new("samples", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
        Field::new("f_average", DataType::Float64, false),
        Field::new("f_absolute", DataType::Float64, false),
        Field::new("f_relative", DataType::Float64, false),
    ]));
    let frequency_file = File::create(format!("{}frequency.parquet", base_path))?;
    let frequency_writer = ArrowWriter::try_new(frequency_file, frequency_schema.clone(), None)?;

    Ok((
        metrics_main_writer,
        metrics_horns_writer,
        frequency_writer,
        metrics_main_schema,
        metrics_horns_schema,
        frequency_schema,
    ))
}

/// Convert results table to a RecordBatch for Parquet writing
/// Now properly handles samples as a list column
fn results_to_record_batch<U>(
    results: &ResultsTable<U>,
    start_idx: usize,
    end_idx: usize,
) -> Result<RecordBatch, Box<dyn std::error::Error>>
where
    U: Integer + ToPrimitive + Copy,
{
    // Create arrays for each column
    let mut arrays: Vec<ArrayRef> = Vec::new();

    // Add ID column
    let id_data: Vec<i32> = results.id[start_idx..end_idx]
        .iter()
        .map(|&id| id as i32)
        .collect();
    arrays.push(Arc::new(Int32Array::from(id_data)));

    // Add samples column as a list using the standard ListBuilder
    let mut list_builder = ListBuilder::new(Int32Builder::new());

    for sample in &results.sample[start_idx..end_idx] {
        // Append all values for this sample
        for &val in sample {
            list_builder.values().append_value(U::to_i32(&val).unwrap());
        }
        // Mark the end of this list
        list_builder.append(true);
    }

    arrays.push(Arc::new(list_builder.finish()));

    // Add horns column
    let horns_data: Vec<f64> = results.horns_values[start_idx..end_idx].to_vec();
    arrays.push(Arc::new(Float64Array::from(horns_data)));

    // Create schema - matching the schema from create_results_writer
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "samples",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ), // true for nullable items
        Field::new("horns", DataType::Float64, false),
    ];

    let schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(schema, arrays).map_err(|e| e.into())
}

/// Internal function to prepare computation parameters
/// Initialize the search context for CLOSURE computation
///
/// Prepares all necessary bounds, lookup tables, and derived parameters needed
/// for efficient depth-first search through the sample space. This includes:
/// - Computing target sum bounds accounting for rounding errors
/// - Computing SD bounds accounting for rounding errors
/// - Pre-computing lookup tables for O(1) pruning decisions
/// - Type conversions and convenience values
fn initialize_closure_search<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
) -> ClosureSearchContext<T, U>
where
    T: FloatType,
    U: IntegerType,
{
    // Convert integer `n` to float to enable multiplication with other floats
    let n_float = T::from(U::to_i32(&n).unwrap()).unwrap();

    // Target sum calculations
    let target_sum = mean * n_float;
    let rounding_error_sum = rounding_error_mean * n_float;

    let target_sum_upper = target_sum + rounding_error_sum;
    let target_sum_lower = target_sum - rounding_error_sum;
    let sd_upper = sd + rounding_error_sd;
    let sd_lower = sd - rounding_error_sd;

    // Convert to usize for range operations
    let n_usize = U::to_usize(&n).unwrap();

    // Create 2D array for min_scale_sum like in Python
    // min_scale_sum[value][n_left] = value * n_left
    let scale_range = U::to_usize(&(scale_max - scale_min + U::one())).unwrap();
    let mut min_scale_sum_t: Vec<Vec<T>> = Vec::with_capacity(scale_range);

    // For each possible value from scale_min to scale_max
    for value in range_u(scale_min, scale_max + U::one()) {
        let value_float = T::from(value).unwrap();
        let row: Vec<T> = (0..n_usize)
            .map(|n_left| value_float * T::from(n_left).unwrap())
            .collect();
        min_scale_sum_t.push(row);
    }

    // max_scale_sum remains 1D as in the original
    let scale_max_sum_t: Vec<T> = (0..n_usize)
        .map(|x| T::from(scale_max).unwrap() * T::from(x).unwrap())
        .collect();

    let n_minus_1 = n - U::one();
    let scale_max_plus_1 = scale_max + U::one();

    ClosureSearchContext {
        target_sum_upper,
        target_sum_lower,
        sd_upper,
        sd_lower,
        n_usize,
        min_scale_sum: min_scale_sum_t,
        scale_max_sum: scale_max_sum_t,
        n_minus_1,
        scale_max_plus_1,
    }
}

/// Generate initial combinations for parallel processing
fn generate_initial_combinations<T, U>(scale_min: U, scale_max_plus_1: U) -> Vec<(Vec<U>, T, T)>
where
    T: FloatType,
    U: IntegerType,
{
    range_u(scale_min, scale_max_plus_1)
        .flat_map(|i| {
            range_u(i, scale_max_plus_1).map(move |j| {
                let initial_combination = vec![i, j];

                let i_float = T::from(i).unwrap();
                let j_float = T::from(j).unwrap();
                let sum = i_float + j_float;
                let current_mean = sum / T::from(2).unwrap();

                let diff_i = i_float - current_mean;
                let diff_j = j_float - current_mean;
                let current_m2 = diff_i * diff_i + diff_j * diff_j;

                (initial_combination, sum, current_m2)
            })
        })
        .collect()
}

/// Generate all valid combinations (memory mode) with summary statistics
///
/// This function computes all valid combinations and returns them in memory
/// along with comprehensive statistics.
/// Optionally writes to a Parquet file if config is provided.
///
/// Use this when:
/// - Result sets are reasonably sized (< 1GB)
/// - You need to process results in memory after generation
/// - You want both file output and in-memory access
///
/// For large result sets, use `dfs_parallel_streaming()` instead.
///
/// # Parameters
/// - `stop_after`: Optional limit on number of samples to find. If None, finds all samples.
pub fn dfs_parallel<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    parquet_config: Option<ParquetConfig>,
    stop_after: Option<usize>,
) -> ResultListFromMeanSdN<U>
where
    T: FloatType,
    U: IntegerType + 'static,
{
    let ClosureSearchContext {
        target_sum_upper,
        target_sum_lower,
        sd_upper,
        sd_lower,
        n_usize,
        min_scale_sum,
        scale_max_sum,
        n_minus_1,
        scale_max_plus_1,
    } = initialize_closure_search(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
    );

    // Generate initial combinations
    let combinations = generate_initial_combinations(scale_min, scale_max_plus_1);

    // Process combinations in parallel with optional early termination
    let results: Vec<Vec<U>> = if let Some(limit) = stop_after {
        // Fast path for small limits: use sequential processing to avoid parallel overhead
        if limit <= 100 {
            let mut found = Vec::with_capacity(limit);
            for (combo, running_sum, running_m2) in combinations {
                if found.len() >= limit {
                    break;
                }

                let remaining = limit - found.len();
                let branch_results = dfs_branch(
                    combo,
                    running_sum,
                    running_m2,
                    n_usize,
                    target_sum_upper,
                    target_sum_lower,
                    sd_upper,
                    sd_lower,
                    &min_scale_sum,
                    &scale_max_sum,
                    n_minus_1,
                    scale_max_plus_1,
                    scale_min,
                    Some(remaining), // Pass the remaining count for early exit
                );

                found.extend(branch_results);
            }
            found
        } else {
            // Parallel path for larger limits
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::Arc;

            let found_count = Arc::new(AtomicUsize::new(0));

            combinations
                .par_iter()
                .flat_map(|(combo, running_sum, running_m2)| {
                    // Check if we've already found enough samples
                    if found_count.load(Ordering::Relaxed) >= limit {
                        return Vec::new();
                    }

                    // Calculate how many more results we need
                    let current = found_count.load(Ordering::Relaxed);
                    let remaining = if current >= limit {
                        return Vec::new();
                    } else {
                        limit - current
                    };

                    let branch_results = dfs_branch(
                        combo.clone(),
                        *running_sum,
                        *running_m2,
                        n_usize,
                        target_sum_upper,
                        target_sum_lower,
                        sd_upper,
                        sd_lower,
                        &min_scale_sum,
                        &scale_max_sum,
                        n_minus_1,
                        scale_max_plus_1,
                        scale_min,
                        Some(remaining), // Pass remaining count for early exit
                    );

                    // Update counter
                    if !branch_results.is_empty() {
                        found_count.fetch_add(branch_results.len(), Ordering::Relaxed);
                    }

                    branch_results
                })
                .collect()
        }
    } else {
        combinations
            .par_iter()
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
                    &min_scale_sum,
                    &scale_max_sum,
                    n_minus_1,
                    scale_max_plus_1,
                    scale_min,
                    None, // No limit for unlimited search
                )
            })
            .collect()
    };

    // Calculate all statistics
    let closure_results = samples_to_result_list(
        results,
        scale_min,
        scale_max
    );

    // Write to Parquet if configured
    if let Some(config) = parquet_config {
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
            let total_samples = closure_results.results.sample.len();

            for start in (0..total_samples).step_by(batch_size) {
                let end = (start + batch_size).min(total_samples);

                if let Ok(record_batch) =
                    results_to_record_batch(&closure_results.results, start, end)
                {
                    let _ = writer.write(&record_batch);
                }
            }
            let _ = writer.close();
        }

        // Write statistics tables
        if let Ok((
            mut mm_writer,
            mut mh_writer,
            mut freq_writer,
            mm_schema,
            mh_schema,
            freq_schema,
        )) = create_stats_writers(&base_path)
        {
            // Write metrics_main
            let mm_batch = RecordBatch::try_new(
                mm_schema,
                vec![
                    Arc::new(Float64Array::from(vec![
                        closure_results.metrics_main.samples_initial,
                    ])),
                    Arc::new(Float64Array::from(vec![
                        closure_results.metrics_main.samples_all,
                    ])),
                    Arc::new(Float64Array::from(vec![
                        closure_results.metrics_main.values_all,
                    ])),
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
                    Arc::new(Float64Array::from(vec![closure_results.metrics_horns.mean])),
                    Arc::new(Float64Array::from(vec![
                        closure_results.metrics_horns.uniform,
                    ])),
                    Arc::new(Float64Array::from(vec![closure_results.metrics_horns.sd])),
                    Arc::new(Float64Array::from(vec![closure_results.metrics_horns.cv])),
                    Arc::new(Float64Array::from(vec![closure_results.metrics_horns.mad])),
                    Arc::new(Float64Array::from(vec![closure_results.metrics_horns.min])),
                    Arc::new(Float64Array::from(vec![
                        closure_results.metrics_horns.median,
                    ])),
                    Arc::new(Float64Array::from(vec![closure_results.metrics_horns.max])),
                    Arc::new(Float64Array::from(vec![
                        closure_results.metrics_horns.range,
                    ])),
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
                    Arc::new(StringArray::from(closure_results.frequency.samples_group().to_vec())),
                    Arc::new(Int32Array::from(closure_results.frequency.value().to_vec())),
                    Arc::new(Float64Array::from(
                        closure_results.frequency.f_average().to_vec(),
                    )),
                    Arc::new(Float64Array::from(
                        closure_results.frequency.f_absolute().to_vec(),
                    )),
                    Arc::new(Float64Array::from(
                        closure_results.frequency.f_relative().to_vec(),
                    )),
                ],
            );
            if let Ok(batch) = freq_batch {
                let _ = freq_writer.write(&batch);
            }
            let _ = freq_writer.close();
        }
    }

    closure_results
}

/// Structure to hold streaming frequency state
pub struct StreamingFrequencyState {
    current_min_horns: f64,
    current_max_horns: f64,
    all_freq: HashMap<i32, i64>,
    min_freq: HashMap<i32, i64>,
    max_freq: HashMap<i32, i64>,
    min_count: usize,
    max_count: usize,
}

/// Helper function to write statistics files for streaming mode
pub fn write_streaming_statistics(
    base_path: &str,
    all_horns: &[f64],
    n_usize: usize,
    scale_min_i32: i32,
    scale_max_i32: i32,
    final_freq_state: Arc<Mutex<StreamingFrequencyState>>,
) {
    if let Ok((mut mm_writer, mut mh_writer, mut freq_writer, mm_schema, mh_schema, freq_schema)) =
        create_stats_writers(base_path)
    {
        // Calculate final statistics
        let samples_all = all_horns.len();
        if samples_all == 0 {
            return;
        }

        let values_all = samples_all * n_usize;

        // Calculate horns statistics
        let horns_mean = all_horns.iter().sum::<f64>() / samples_all as f64;
        let horns_sd = {
            let variance = all_horns
                .iter()
                .map(|&h| (h - horns_mean).powi(2))
                .sum::<f64>()
                / samples_all as f64;
            variance.sqrt()
        };

        let mut horns_sorted = all_horns.to_vec();
        horns_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let horns_min = horns_sorted[0];
        let horns_max = horns_sorted[samples_all - 1];
        let horns_median = median(&horns_sorted);
        let horns_mad = mad(all_horns, horns_median);

        // Write metrics_main
        let mm_batch = RecordBatch::try_new(
            mm_schema,
            vec![
                Arc::new(Float64Array::from(vec![count_initial_combinations(
                    scale_min_i32,
                    scale_max_i32,
                ) as f64])),
                Arc::new(Float64Array::from(vec![samples_all as f64])),
                Arc::new(Float64Array::from(vec![values_all as f64])),
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
                Arc::new(Float64Array::from(vec![horns_mean])),
                Arc::new(Float64Array::from(vec![calculate_horns_uniform(
                    scale_min_i32,
                    scale_max_i32,
                )])),
                Arc::new(Float64Array::from(vec![horns_sd])),
                Arc::new(Float64Array::from(vec![horns_sd / horns_mean])),
                Arc::new(Float64Array::from(vec![horns_mad])),
                Arc::new(Float64Array::from(vec![horns_min])),
                Arc::new(Float64Array::from(vec![horns_median])),
                Arc::new(Float64Array::from(vec![horns_max])),
                Arc::new(Float64Array::from(vec![horns_max - horns_min])),
            ],
        );
        if let Ok(batch) = mh_batch {
            let _ = mh_writer.write(&batch);
        }
        let _ = mh_writer.close();

        // Extract frequency data from final state and construct FrequencyTable
        let state = final_freq_state.lock().unwrap();
        let nrow_frequency = (scale_max_i32 - scale_min_i32 + 1) as usize;

        // Build frequency vectors for "all" category
        let mut all_value = Vec::with_capacity(nrow_frequency);
        let mut all_f_absolute = Vec::with_capacity(nrow_frequency);
        let mut all_f_average = Vec::with_capacity(nrow_frequency);
        let mut all_f_relative = Vec::with_capacity(nrow_frequency);

        let total_values = values_all as f64;
        let n_samples = samples_all as f64;

        for scale_value in scale_min_i32..=scale_max_i32 {
            let count = *state.all_freq.get(&scale_value).unwrap_or(&0) as f64;
            all_value.push(scale_value);
            all_f_absolute.push(count);
            all_f_average.push(count / n_samples);
            all_f_relative.push(count / total_values);
        }

        // Build frequency vectors for "horns_min" category
        let mut min_value = Vec::with_capacity(nrow_frequency);
        let mut min_f_absolute = Vec::with_capacity(nrow_frequency);
        let mut min_f_average = Vec::with_capacity(nrow_frequency);
        let mut min_f_relative = Vec::with_capacity(nrow_frequency);

        let min_n_samples = state.min_count as f64;
        let min_total_values = (state.min_count * n_usize) as f64;

        for scale_value in scale_min_i32..=scale_max_i32 {
            let count = *state.min_freq.get(&scale_value).unwrap_or(&0) as f64;
            min_value.push(scale_value);
            min_f_absolute.push(count);
            min_f_average.push(count / min_n_samples);
            min_f_relative.push(count / min_total_values);
        }

        // Build frequency vectors for "horns_max" category
        let mut max_value = Vec::with_capacity(nrow_frequency);
        let mut max_f_absolute = Vec::with_capacity(nrow_frequency);
        let mut max_f_average = Vec::with_capacity(nrow_frequency);
        let mut max_f_relative = Vec::with_capacity(nrow_frequency);

        let max_n_samples = state.max_count as f64;
        let max_total_values = (state.max_count * n_usize) as f64;

        for scale_value in scale_min_i32..=scale_max_i32 {
            let count = *state.max_freq.get(&scale_value).unwrap_or(&0) as f64;
            max_value.push(scale_value);
            max_f_absolute.push(count);
            max_f_average.push(count / max_n_samples);
            max_f_relative.push(count / max_total_values);
        }

        // Combine all frequency data into proper FrequencyTable structure
        let mut combined_value = all_value;
        combined_value.extend(min_value);
        combined_value.extend(max_value);

        let mut combined_f_average = all_f_average;
        combined_f_average.extend(min_f_average);
        combined_f_average.extend(max_f_average);

        let mut combined_f_absolute = all_f_absolute;
        combined_f_absolute.extend(min_f_absolute);
        combined_f_absolute.extend(max_f_absolute);

        let mut combined_f_relative = all_f_relative;
        combined_f_relative.extend(min_f_relative);
        combined_f_relative.extend(max_f_relative);

        // Create proper FrequencyTable with type-safe samples column
        let frequency_table = FrequencyTable::new(
            FrequencySamplesColumn::new(nrow_frequency),
            combined_value,
            combined_f_average,
            combined_f_absolute,
            combined_f_relative,
        );

        // Write frequency table as a single batch
        let freq_batch = RecordBatch::try_new(
            freq_schema,
            vec![
                Arc::new(StringArray::from(frequency_table.samples_group().to_vec())),
                Arc::new(Int32Array::from(frequency_table.value().to_vec())),
                Arc::new(Float64Array::from(frequency_table.f_average().to_vec())),
                Arc::new(Float64Array::from(frequency_table.f_absolute().to_vec())),
                Arc::new(Float64Array::from(frequency_table.f_relative().to_vec())),
            ],
        );
        if let Ok(batch) = freq_batch {
            let _ = freq_writer.write(&batch);
        }
        let _ = freq_writer.close();
    }
}

/// Generate all valid combinations (streaming mode) with summary statistics
///
/// This function computes all valid combinations and streams them directly to Parquet files
/// without keeping them in memory. Statistics are computed incrementally.
///
/// Use this when:
/// - Result sets are very large (> 1GB)
/// - You only need file output, not in-memory processing
/// - Memory efficiency is critical
///
/// Returns a StreamingResult with the total count and file path.
///
/// # Parameters
/// - `stop_after`: Optional limit on number of samples to find. If None, finds all samples.
pub fn dfs_parallel_streaming<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    config: StreamingConfig,
    stop_after: Option<usize>,
) -> StreamingResult
where
    T: FloatType,
    U: IntegerType + 'static,
{
    let ClosureSearchContext {
        target_sum_upper,
        target_sum_lower,
        sd_upper,
        sd_lower,
        n_usize,
        min_scale_sum,
        scale_max_sum,
        n_minus_1,
        scale_max_plus_1,
    } = initialize_closure_search(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
    );

    // Setup channels for streaming results
    let (tx_results, rx_results) = channel::<Vec<(Vec<U>, f64)>>();
    let (tx_stats, rx_stats) = channel::<(Vec<f64>, HashMap<i32, i64>)>();

    // Add a flag to track writer thread status
    let writer_failed = Arc::new(AtomicUsize::new(0)); // 0 = ok, 1 = failed
    let writer_failed_for_compute = writer_failed.clone();
    let writer_failed_for_thread = writer_failed.clone();

    // Counter for total combinations found
    let total_counter = Arc::new(AtomicUsize::new(0));
    let counter_for_thread = total_counter.clone();

    // Counter for tracking progress through initial combinations
    let initial_combo_counter = Arc::new(AtomicUsize::new(0));
    let initial_combo_total = count_initial_combinations(
        U::to_i32(&scale_min).unwrap(),
        U::to_i32(&scale_max).unwrap(),
    ) as usize;

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

    // Handle file paths more carefully
    let base_path = if config.file_path.ends_with('/') {
        config.file_path.clone()
    } else if std::path::Path::new(&config.file_path).is_dir() {
        format!("{}/", config.file_path)
    } else {
        // If it doesn't exist or isn't a directory, treat as a prefix
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

    // Spawn dedicated writer thread for two separate files
    let samples_path = format!("{}samples.parquet", base_path);
    let horns_path = format!("{}horns.parquet", base_path);

    let n_usize_for_writer = n_usize; // Capture n_usize for the writer thread

    let writer_handle = thread::spawn(move || {
        // Create two separate writers
        let mut samples_writer = match create_samples_writer(&samples_path, n_usize_for_writer) {
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

        // Buffers for batching
        let mut samples_buffer: Vec<Vec<U>> = Vec::with_capacity(config.batch_size * 2);
        let mut horns_buffer: Vec<f64> = Vec::with_capacity(config.batch_size * 2);

        let mut total_written = 0;
        let mut last_progress_report = 0;

        // Process incoming results
        loop {
            match rx_results.recv() {
                Ok(batch) => {
                    // Add to buffers, maintaining order
                    for (sample, horns) in batch {
                        samples_buffer.push(sample);
                        horns_buffer.push(horns);
                    }

                    // Write when buffers reach threshold
                    if samples_buffer.len() >= config.batch_size {
                        // Write samples batch
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

                        // Write horns batch
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

                        // Progress reporting
                        if config.show_progress && total_written - last_progress_report >= 100_000 {
                            eprintln!("Progress: {} combinations written...", total_written);
                            last_progress_report = total_written;
                        }

                        samples_buffer.clear();
                        horns_buffer.clear();
                    }
                }
                Err(_) => {
                    // Channel closed, write remaining data
                    break;
                }
            }
        }

        // Write any remaining results
        if !samples_buffer.is_empty() {
            // Write final samples batch
            if let Ok(record_batch) = samples_to_record_batch(&samples_buffer) {
                if let Err(e) = samples_writer.write(&record_batch) {
                    eprintln!("ERROR: Failed to write final samples batch: {}", e);
                } else {
                    // Write final horns batch
                    if let Ok(record_batch) = horns_to_record_batch(&horns_buffer) {
                        if let Err(e) = horns_writer.write(&record_batch) {
                            eprintln!("ERROR: Failed to write final horns batch: {}", e);
                        } else {
                            total_written += samples_buffer.len();
                        }
                    }
                }
            }
        }

        // Close both writers
        if let Err(e) = samples_writer.close() {
            eprintln!("ERROR: Failed to close samples file: {}", e);
        }
        if let Err(e) = horns_writer.close() {
            eprintln!("ERROR: Failed to close horns file: {}", e);
        }

        if config.show_progress {
            eprintln!(
                "Streaming complete: {} total combinations written",
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

    // Generate initial combinations
    let combinations = generate_initial_combinations(scale_min, scale_max_plus_1);

    let scale_min_i32 = U::to_i32(&scale_min).unwrap();
    let scale_max_i32 = U::to_i32(&scale_max).unwrap();

    // Fast path for small stop_after limits: use sequential processing
    if let Some(limit) = stop_after {
        if limit <= 100 {
            // Sequential processing for small limits
            let nrow_frequency = (scale_max_i32 - scale_min_i32 + 1) as usize;
            let mut found_count = 0;

            for (combo, running_sum, running_m2) in combinations {
                if found_count >= limit {
                    break;
                }

                let remaining = limit - found_count;
                let branch_results = dfs_branch(
                    combo,
                    running_sum,
                    running_m2,
                    n_usize,
                    target_sum_upper,
                    target_sum_lower,
                    sd_upper,
                    sd_lower,
                    &min_scale_sum,
                    &scale_max_sum,
                    n_minus_1,
                    scale_max_plus_1,
                    scale_min,
                    Some(remaining), // Pass remaining count for early exit
                );

                for sample in branch_results.into_iter() {
                    if found_count >= limit {
                        break;
                    }

                    let mut freqs = vec![0.0; nrow_frequency];
                    let mut sample_freq: HashMap<i32, i64> = HashMap::new();

                    for &value in &sample {
                        let idx = (U::to_i32(&value).unwrap() - scale_min_i32) as usize;
                        freqs[idx] += 1.0;
                        *sample_freq.entry(U::to_i32(&value).unwrap()).or_insert(0) += 1;
                    }

                    let horns = calculate_horns(&freqs, scale_min_i32, scale_max_i32);

                    // Update frequency state
                    {
                        let mut state = freq_state_for_thread.lock().unwrap();

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

                    // Send to writer and stats
                    if tx_results.send(vec![(sample.clone(), horns)]).is_ok() {
                        let _ = tx_stats.send((vec![horns], HashMap::new()));
                    }

                    found_count += 1;
                }
            }

            // Close channels and wait for completion
            drop(tx_results);
            drop(tx_stats);

            let total_written = writer_handle.join().unwrap_or(0);
            let (all_horns, final_freq_state) = stats_handle
                .join()
                .unwrap_or_else(|_| (Vec::new(), freq_state));

            // Write statistics files
            write_streaming_statistics(
                &base_path,
                &all_horns,
                n_usize,
                scale_min_i32,
                scale_max_i32,
                final_freq_state,
            );

            return StreamingResult {
                total_combinations: total_written,
                file_path: config.file_path,
            };
        }
    }

    // Process combinations in parallel (original path for unlimited or large limits)
    combinations
        .par_iter()
        .for_each(|(combo, running_sum, running_m2)| {
            // Check if writer has failed before doing expensive computation
            if writer_failed_for_compute.load(Ordering::Relaxed) == 1 {
                return;
            }

            // Check if we've reached the stop_after limit
            if let Some(limit) = stop_after {
                if counter_for_thread.load(Ordering::Relaxed) >= limit {
                    return;
                }
            }

            // Track progress through initial combinations
            let current_initial = initial_combo_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if config.show_progress && current_initial.is_multiple_of(10) {
                let percentage = (current_initial as f64 / initial_combo_total as f64) * 100.0;
                eprintln!(
                    "Progress: {:.1}% of initial combinations explored...",
                    percentage
                );
            }

            // Calculate how many more results we need
            let remaining = if let Some(limit) = stop_after {
                let current = counter_for_thread.load(Ordering::Relaxed);
                if current >= limit {
                    return; // Already have enough
                }
                Some(limit - current)
            } else {
                None
            };

            let branch_results = dfs_branch(
                combo.clone(),
                *running_sum,
                *running_m2,
                n_usize,
                target_sum_upper,
                target_sum_lower,
                sd_upper,
                sd_lower,
                &min_scale_sum,
                &scale_max_sum,
                n_minus_1,
                scale_max_plus_1,
                scale_min,
                remaining, // Pass remaining count for early exit
            );

            if !branch_results.is_empty() {
                // Check again before processing results
                if writer_failed_for_compute.load(Ordering::Relaxed) == 1 {
                    return;
                }

                let mut results_with_horns = Vec::with_capacity(branch_results.len());
                let mut horns_batch = Vec::with_capacity(branch_results.len());

                let nrow_frequency = (scale_max_i32 - scale_min_i32 + 1) as usize;

                // Calculate horns for each sample and update frequency state
                for sample in branch_results.into_iter() {
                    let mut freqs = vec![0.0; nrow_frequency];
                    let mut sample_freq: HashMap<i32, i64> = HashMap::new();

                    for &value in &sample {
                        let idx = (U::to_i32(&value).unwrap() - scale_min_i32) as usize;
                        freqs[idx] += 1.0;
                        *sample_freq.entry(U::to_i32(&value).unwrap()).or_insert(0) += 1;
                    }

                    let horns = calculate_horns(&freqs, scale_min_i32, scale_max_i32);

                    // Update frequency state with proper locking
                    {
                        let mut state = freq_state_for_thread.lock().unwrap();

                        // Update all frequencies
                        for (&val, &count) in &sample_freq {
                            *state.all_freq.entry(val).or_insert(0) += count;
                        }

                        // Check if this is a new min or max
                        if (horns - state.current_min_horns).abs() < 1e-10 {
                            // Equal to current min
                            for (&val, &count) in &sample_freq {
                                *state.min_freq.entry(val).or_insert(0) += count;
                            }
                            state.min_count += 1;
                        } else if horns < state.current_min_horns {
                            // New minimum found
                            state.current_min_horns = horns;
                            state.min_freq.clear();
                            for (&val, &count) in &sample_freq {
                                state.min_freq.insert(val, count);
                            }
                            state.min_count = 1;
                        }

                        if (horns - state.current_max_horns).abs() < 1e-10 {
                            // Equal to current max
                            for (&val, &count) in &sample_freq {
                                *state.max_freq.entry(val).or_insert(0) += count;
                            }
                            state.max_count += 1;
                        } else if horns > state.current_max_horns {
                            // New maximum found
                            state.current_max_horns = horns;
                            state.max_freq.clear();
                            for (&val, &count) in &sample_freq {
                                state.max_freq.insert(val, count);
                            }
                            state.max_count = 1;
                        }
                    }

                    horns_batch.push(horns);
                    results_with_horns.push((sample, horns));
                }

                // Update counter and potentially truncate results if we exceed limit
                let current_count =
                    counter_for_thread.fetch_add(results_with_horns.len(), Ordering::Relaxed);

                // Truncate results if we've exceeded the stop_after limit
                let final_results = if let Some(limit) = stop_after {
                    if current_count >= limit {
                        return; // We already have enough
                    } else if current_count + results_with_horns.len() > limit {
                        // Take only what we need to reach the limit
                        let take_count = limit - current_count;
                        results_with_horns.into_iter().take(take_count).collect()
                    } else {
                        results_with_horns
                    }
                } else {
                    results_with_horns
                };

                // Send to writer and stats collector
                if tx_results.send(final_results).is_err() {
                    // Channel is closed, writer must have failed
                }
            }
        });

    // Close channels
    drop(tx_results);
    drop(tx_stats);

    // Wait for threads to complete
    let total_written = writer_handle.join().unwrap_or_else(|_| {
        eprintln!("ERROR: Writer thread panicked unexpectedly");
        0
    });

    let (all_horns, final_freq_state) = stats_handle.join().unwrap_or_else(|_| {
        eprintln!("ERROR: Statistics thread panicked unexpectedly");
        (Vec::new(), freq_state)
    });

    // Check if we successfully wrote any results
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
        n_usize,
        scale_min_i32,
        scale_max_i32,
        final_freq_state,
    );

    StreamingResult {
        total_combinations: total_written,
        file_path: config.file_path,
    }
}

// Collect all valid combinations from a starting point
#[inline]
#[allow(clippy::too_many_arguments)]
fn dfs_branch<T, U>(
    start_combination: Vec<U>,
    running_sum_init: T,
    running_m2_init: T,
    n: usize, // Use usize for the length
    target_sum_upper: T,
    target_sum_lower: T,
    sd_upper: T,
    sd_lower: T,
    min_scale_sum_t: &[Vec<T>], // Now 2D array
    scale_max_sum_t: &[T],
    _n_minus_1: U,
    scale_max_plus_1: U,
    scale_min: U,              // Need this to calculate indices
    stop_after: Option<usize>, // Optional limit for early termination
) -> Vec<Vec<U>>
where
    T: FloatType,
    U: IntegerType,
{
    let mut stack = VecDeque::with_capacity(n * 2);
    let mut results = Vec::new();

    // Use usize::MAX as sentinel when no limit is specified
    let limit = stop_after.unwrap_or(usize::MAX);

    stack.push_back(Combination {
        values: start_combination.clone(),
        running_sum: running_sum_init,
        running_m2: running_m2_init,
    });

    while let Some(current) = stack.pop_back() {
        if current.values.len() >= n {
            let n_minus_1_float = T::from(n - 1).unwrap();
            let current_std = (current.running_m2 / n_minus_1_float).sqrt();
            if current_std >= sd_lower {
                results.push(current.values);
                // Early exit if we've reached the limit
                if results.len() >= limit {
                    results.truncate(limit);
                    return results;
                }
            }
            continue;
        }

        // Calculate remaining items to add
        let current_len = current.values.len();
        let n_left = n - current_len - 1; // How many more items after the next one
        let next_n = current_len + 1;

        // Get current mean
        let current_mean = current.running_sum / T::from(current_len).unwrap();

        // Get the last value
        let last_value = current.values[current_len - 1];

        for next_value in range_u(last_value, scale_max_plus_1) {
            let next_value_as_t = T::from(next_value).unwrap();
            let next_sum = current.running_sum + next_value_as_t;

            // Calculate index for 2D array access
            // Index is (next_value - scale_min) to map to 0-based array
            let value_index = U::to_usize(&(next_value - scale_min)).unwrap();

            // Safe indexing with bounds check
            if value_index < min_scale_sum_t.len() && n_left < min_scale_sum_t[value_index].len() {
                // Access as min_scale_sum_t[next_value-scale_min][n_left]
                let minmean = next_sum + min_scale_sum_t[value_index][n_left];
                if minmean > target_sum_upper {
                    break; // Early termination
                }

                // Safe indexing with bounds check
                if n_left < scale_max_sum_t.len() {
                    let maxmean = next_sum + scale_max_sum_t[n_left];
                    if maxmean < target_sum_lower {
                        continue;
                    }

                    let next_mean = next_sum / T::from(next_n).unwrap();
                    let delta = next_value_as_t - current_mean;
                    let delta2 = next_value_as_t - next_mean;
                    let next_m2 = current.running_m2 + delta * delta2;

                    let min_sd = (next_m2 / T::from(n - 1).unwrap()).sqrt();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_initial_combinations() {
        assert_eq!(count_initial_combinations(1, 3), 6);
        assert_eq!(count_initial_combinations(1, 4), 10);
    }

    #[test]
    fn test_frequency_samples_column() {
        // Test with 5 repetitions (like scale 1..=5)
        let samples = FrequencySamplesColumn::new(5);

        // Check length
        assert_eq!(samples.len(), 15); // 5 * 3
        assert!(!samples.is_empty());

        // Check repetitions
        assert_eq!(samples.repetitions(), 5);

        // Check structure via to_vec()
        let vec = samples.to_vec();
        assert_eq!(vec.len(), 15);

        // First 5 should be "all"
        for i in 0..5 {
            assert_eq!(vec[i], "all");
            assert_eq!(samples.get(i), SampleCategory::All);
            assert_eq!(samples.get_str(i), "all");
        }

        // Next 5 should be "horns_min"
        for i in 5..10 {
            assert_eq!(vec[i], "horns_min");
            assert_eq!(samples.get(i), SampleCategory::HornsMin);
            assert_eq!(samples.get_str(i), "horns_min");
        }

        // Last 5 should be "horns_max"
        for i in 10..15 {
            assert_eq!(vec[i], "horns_max");
            assert_eq!(samples.get(i), SampleCategory::HornsMax);
            assert_eq!(samples.get_str(i), "horns_max");
        }
    }

    #[test]
    fn test_frequency_samples_column_empty() {
        let samples = FrequencySamplesColumn::new(0);
        assert_eq!(samples.len(), 0);
        assert!(samples.is_empty());
        assert_eq!(samples.to_vec().len(), 0);
    }

    #[test]
    fn test_horns_calculation() {
        // Test uniform distribution
        let uniform_freqs = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let horns_uniform = calculate_horns(&uniform_freqs, 1, 5);
        assert!((horns_uniform - 0.5).abs() < 0.01);

        // Test extreme distribution
        let extreme_freqs = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let horns_extreme = calculate_horns(&extreme_freqs, 1, 5);
        assert!((horns_extreme - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sample_category_strum_integration() {
        // Test that strum correctly generates snake_case names
        assert_eq!(SampleCategory::All.as_str(), "all");
        assert_eq!(SampleCategory::HornsMin.as_str(), "horns_min");
        assert_eq!(SampleCategory::HornsMax.as_str(), "horns_max");

        // Test that all() returns all variants in order
        let all_variants: Vec<_> = SampleCategory::all().collect();
        assert_eq!(all_variants.len(), 3);
        assert_eq!(all_variants[0], SampleCategory::All);
        assert_eq!(all_variants[1], SampleCategory::HornsMin);
        assert_eq!(all_variants[2], SampleCategory::HornsMax);

        // Test that all_names() returns correct snake_case strings
        let all_names: Vec<_> = SampleCategory::all_names().collect();
        assert_eq!(all_names, vec!["all", "horns_min", "horns_max"]);

        // Test that COUNT matches the actual number of variants
        assert_eq!(SampleCategory::COUNT, SampleCategory::iter().count());
    }

    #[test]
    fn test_dfs_parallel_with_new_api() {
        // Test that the function returns valid statistics with new structure
        let results = dfs_parallel::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            None, // no parquet config
            None, // no stop_after limit
        );

        // Check that results table is properly formed
        assert!(!results.results.sample.is_empty());
        assert_eq!(
            results.results.sample.len(),
            results.results.horns_values.len()
        );
        assert_eq!(results.results.sample.len(), results.results.id.len());
        assert_eq!(results.results.id[0], 1);
        assert_eq!(
            results.results.id.last(),
            Some(&results.results.sample.len())
        );

        // Check metrics
        assert_eq!(
            results.metrics_main.samples_all,
            results.results.sample.len() as f64
        );
        assert_eq!(
            results.metrics_main.values_all,
            (results.results.sample.len() * 5) as f64
        );

        // Check horns metrics
        assert!(results.metrics_horns.min <= results.metrics_horns.mean);
        assert!(results.metrics_horns.mean <= results.metrics_horns.max);
        assert!(results.metrics_horns.range >= 0.0);

        // Check combined frequency table
        let n_values = 5; // scale_max - scale_min + 1
        let expected_rows = n_values * 3; // all, horns_min, horns_max
        assert_eq!(results.frequency.len(), expected_rows);
        assert_eq!(results.frequency.samples_group().len(), expected_rows);
        assert_eq!(results.frequency.value().len(), expected_rows);
        assert_eq!(results.frequency.f_average().len(), expected_rows);
        assert_eq!(results.frequency.f_absolute().len(), expected_rows);
        assert_eq!(results.frequency.f_relative().len(), expected_rows);

        // Check that samples column has correct values
        let samples_vec = results.frequency.samples_group().to_vec();
        let all_count = samples_vec.iter().filter(|&s| s == "all").count();
        let min_count = samples_vec.iter().filter(|&s| s == "horns_min").count();
        let max_count = samples_vec.iter().filter(|&s| s == "horns_max").count();
        assert_eq!(all_count, n_values);
        assert_eq!(min_count, n_values);
        assert_eq!(max_count, n_values);
    }

    #[test]
    fn test_dfs_parallel_with_file() {
        // Test with Parquet output
        let config = ParquetConfig {
            file_path: "test_output/".to_string(),
            batch_size: 100,
        };

        let _ = std::fs::create_dir("test_output");

        let results = dfs_parallel::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            Some(config),
            None, // no stop_after limit
        );

        assert!(!results.results.sample.is_empty());

        // Clean up test files
        let _ = std::fs::remove_file("test_output/results.parquet");
        let _ = std::fs::remove_file("test_output/metrics_main.parquet");
        let _ = std::fs::remove_file("test_output/metrics_horns.parquet");
        let _ = std::fs::remove_file("test_output/frequency.parquet");
        let _ = std::fs::remove_dir("test_output");
    }

    #[test]
    fn test_dfs_parallel_streaming_separate_files() {
        // Test streaming mode with separate files
        let config = StreamingConfig {
            file_path: "test_streaming/".to_string(),
            batch_size: 100,
            show_progress: false,
        };

        let _ = std::fs::create_dir("test_streaming");

        let result = dfs_parallel_streaming::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            config, None, // no stop_after limit
        );

        assert!(result.total_combinations > 0);
        assert_eq!(result.file_path, "test_streaming/");

        // Check that both files were created
        assert!(std::path::Path::new("test_streaming/samples.parquet").exists());
        assert!(std::path::Path::new("test_streaming/horns.parquet").exists());

        // Clean up test files
        let _ = std::fs::remove_file("test_streaming/samples.parquet");
        let _ = std::fs::remove_file("test_streaming/horns.parquet");
        let _ = std::fs::remove_file("test_streaming/metrics_main.parquet");
        let _ = std::fs::remove_file("test_streaming/metrics_horns.parquet");
        let _ = std::fs::remove_file("test_streaming/frequency.parquet");
        let _ = std::fs::remove_dir("test_streaming");
    }

    #[test]
    fn test_stop_after_parameter() {
        // Test that stop_after limits the number of results
        let results_unlimited = dfs_parallel::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            80,   // n (increased sample size)
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            None, // no parquet config
            None, // no stop_after limit
        );

        let total_samples = results_unlimited.results.sample.len();
        assert!(total_samples > 10); // Should have many samples

        // Test with limit of 10
        let results_limited = dfs_parallel::<f64, i32>(
            3.0,      // mean
            1.0,      // sd
            80,       // n (increased sample size)
            1,        // scale_min
            5,        // scale_max
            0.05,     // rounding_error_mean
            0.05,     // rounding_error_sd
            None,     // no parquet config
            Some(10), // stop after 10 samples
        );

        assert_eq!(results_limited.results.sample.len(), 10);
        assert_eq!(results_limited.results.horns_values.len(), 10);
        assert_eq!(results_limited.results.id.len(), 10);

        // Test with limit of 1
        let results_one = dfs_parallel::<f64, i32>(
            3.0,     // mean
            1.0,     // sd
            80,      // n (increased sample size)
            1,       // scale_min
            5,       // scale_max
            0.05,    // rounding_error_mean
            0.05,    // rounding_error_sd
            None,    // no parquet config
            Some(1), // stop after 1 sample
        );

        assert_eq!(results_one.results.sample.len(), 1);
        assert_eq!(results_one.results.horns_values.len(), 1);
        assert_eq!(results_one.results.id.len(), 1);
    }
}
