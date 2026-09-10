//! CLOSURE: complete listing of original samples of underlying raw evidence
//!
//! Crate closure-core implements the CLOSURE technique for efficiently reconstructing
//! all possible distributions of raw data from summary statistics. It is not
//! about the Rust feature called closure.
//!
//! The crate is mostly meant to serve as a backend for the R package [unsum](https://lhdjung.github.io/unsum/).
//! The main APIs users need are `closure_parallel()` for in-memory results and
//! `closure_parallel_streaming()` for memory-efficient file output.
//!
//! # Results are frequency tables
//!
//! Both techniques store a reconstructed sample as one count per scale value
//! rather than one integer per observation. Every sample they produce is
//! sorted, so the two representations carry exactly the same information —
//! see [`sample_counts`] for why, and [`ResultsTable::sample`] to get a sample
//! back as a vector of values.
//!
//! The payoff is in memory: a result set of `m` samples costs `m * k` counts
//! instead of `m` heap-allocated vectors of `n` values. For a 235k-sample run
//! over a 1-7 scale with `n = 60` that is about 6.6 MB against roughly 62 MB.
//! On disk the gain is much smaller — around 1.3x — because Parquet's
//! dictionary and run-length encoding already compressed most of the
//! redundancy out of the per-position layout.
//!
//! # Output layout
//!
//! One directory per run, written identically by CLOSURE and SPRITE:
//!
//! | File | Contents |
//! |---|---|
//! | `counts.parquet` | one row per sample: a count column per scale value, then `horns`. A sample's id is its row number. |
//! | `scale_values.parquet` | which scale value each count column stands for |
//! | `format.parquet` | format name, version, technique, `n`, `k`, `items`, scale bounds |
//! | `metrics_main`, `metrics_horns`, `frequency`, `frequency_dist` | summary statistics |
//! | `modality_counts`, `modality_pairs` | per-value count ranges across the whole result set, and which adjacent orderings are fixed |
//! | `modality_shapes` | per-value count ranges **within each shape class** — read as "if the data had this shape, then…" |
//! | `modality_summary` | one row: samples per shape class, whether the search was exhaustive, unimodality-deficit spread |
//! | `modality_prominence` | samples per shape class at each threshold in the prominence envelope |
//!
//! # Reading the shape output
//!
//! [`modality::ModalityShapes`] answers what the raw data could have looked
//! like. Three points decide how much its answers are worth:
//!
//! - A shape class with **zero** members means the raw data did not have that
//!   shape — but only if the search was exhaustive. A truncated run
//!   (`stop_after`) and SPRITE both see part of the space, so
//!   [`modality::ModalityShapes::can_be`] returns `None` for them rather than
//!   claiming a proof. It is also only a proof relative to a threshold for what
//!   counts as a mode, so `can_be` requires the class to be empty across a band
//!   of thresholds rather than at one; `modality_prominence` shows the band.
//! - A class with **many** members means very little on its own. Since one
//!   admissible dataset is enough for an author to point at, the useful output
//!   is the class's *conditional bounds*: what every dataset of that shape would
//!   also have to look like. Those are in `modality_shapes`.
//! - Class **proportions** weight admissible datasets equally, the same
//!   weighting `metrics_horns` already uses. That is a sensitivity analysis, not
//!   a posterior probability.
//!
//! Set [`OutputFormat::Samples`] or [`OutputFormat::Both`] on the config to
//! also write the older per-position layout (`sample.parquet` /
//! `results.parquet`), which cross-validation against the Python
//! implementation still uses.
//!
//! Most of the code was written by Claude 3.5, translating Python code by Nathanael Larigaldie.

use crate::modality::{ModalityShapes, ShapeAccumulator, ShapeClass, DEFAULT_MODE_PROMINENCE};
use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int32Builder, ListBuilder, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use num::{Float, FromPrimitive, Integer, NumCast, ToPrimitive};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
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

use thiserror::Error;

mod count;
pub mod modality;
pub mod sample_counts;
mod sprite;
mod sprite_types;

pub use count::closure_count;
pub use sample_counts::{
    count_data_type, counts_record_batch, counts_schema, create_counts_writer, OutputFormat,
    SampleCounts, SampleFormat, ValueGrid, OUTPUT_FORMAT_VERSION,
};

#[derive(Debug, Error)]
pub enum ParameterError {
    #[error("{0}")]
    InputValidation(String),
    #[error("{0}")]
    Consistency(String),
    #[error("{0}")]
    Conflict(String),
}

// Re-export sprite types needed for the public API
pub use sprite::{sprite_parallel, sprite_parallel_streaming, Sprite};
pub use sprite_types::{RestrictionsMinimum, RestrictionsOption};

/// Unified trait for sample reconstruction techniques (CLOSURE, SPRITE, etc.)
///
/// Requiring [`SampleFormat`] is what keeps the techniques' results
/// interchangeable: a technique chooses how it searches, never how its results
/// are shaped or written.
#[allow(clippy::too_many_arguments)]
pub trait Technique<T: FloatType, U: IntegerType + 'static>: SampleFormat<U> {
    fn run(
        &mut self,
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        items: u32,
        parquet_config: Option<ParquetConfig>,
        stop_after: Option<usize>,
    ) -> Result<ResultListFromMeanSdN<U>, ParameterError>;

    fn run_streaming(
        &mut self,
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        items: u32,
        config: StreamingConfig,
        stop_after: Option<usize>,
    ) -> Result<StreamingResult, ParameterError>;
}

/// CLOSURE technique: complete listing of original samples of underlying raw evidence
pub struct Closure;

impl<U: IntegerType> SampleFormat<U> for Closure {
    const TECHNIQUE: &'static str = "closure";
    /// CLOSURE emits whole scale points.
    const SAMPLE_SCALE_FACTOR: i32 = 100;
}

impl<T: FloatType, U: IntegerType + 'static> Technique<T, U> for Closure {
    fn run(
        &mut self,
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        items: u32,
        parquet_config: Option<ParquetConfig>,
        stop_after: Option<usize>,
    ) -> Result<ResultListFromMeanSdN<U>, ParameterError> {
        closure_parallel(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            items,
            parquet_config,
            stop_after,
        )
    }

    fn run_streaming(
        &mut self,
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        items: u32,
        config: StreamingConfig,
        stop_after: Option<usize>,
    ) -> Result<StreamingResult, ParameterError> {
        closure_parallel_streaming(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            items,
            config,
            stop_after,
        )
    }
}

/// Configuration for Parquet output in memory mode
/// Used with `closure_parallel()` to optionally save results while returning them
pub struct ParquetConfig {
    pub file_path: String,
    pub batch_size: usize,
    /// Which sample layout to write. Defaults to the counts layout.
    pub format: OutputFormat,
}

impl ParquetConfig {
    /// Write the counts layout to `file_path` in batches of `batch_size` rows.
    pub fn new(file_path: impl Into<String>, batch_size: usize) -> Self {
        Self {
            file_path: file_path.into(),
            batch_size,
            format: OutputFormat::default(),
        }
    }
}

/// Configuration for streaming mode
/// Used with `closure_parallel_streaming()` for memory-efficient processing
pub struct StreamingConfig {
    pub file_path: String,
    pub batch_size: usize,
    pub show_progress: bool,
    /// Which sample layout to write. Defaults to the counts layout.
    pub format: OutputFormat,
}

impl StreamingConfig {
    /// Stream the counts layout to `file_path` in batches of `batch_size` rows.
    pub fn new(file_path: impl Into<String>, batch_size: usize, show_progress: bool) -> Self {
        Self {
            file_path: file_path.into(),
            batch_size,
            show_progress,
            format: OutputFormat::default(),
        }
    }
}

/// Result of streaming operation
pub struct StreamingResult {
    pub total_combinations: usize,
    pub file_path: String,
}

/// Sample category for frequency tables
///
/// Automatically provides iteration, count, and snake_case string conversion via strum.
///
/// **Important**: While strum makes iteration robust, these three variants have specific
/// semantic meaning in the CLOSURE algorithm:
/// - `All`: Medoid of all samples (the sample closest by EMD to the grand
///   centroid). Formerly the grand mean; now an actual sample.
/// - `HornsMin`: Medoid of samples with the minimum horns index.
/// - `HornsMax`: Medoid of samples with the maximum horns index.
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

    /// Convert to a `Vec<String>` for compatibility with existing code
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
/// Two different summaries of the same group sit side by side, because they
/// answer different questions and neither substitutes for the other:
///
/// - `f_expected` is the mean count of the value across the group. It is a
///   central tendency over admissible datasets, not itself a dataset — it need
///   not be integral and generally does not satisfy the reported mean and SD.
/// - `f_representative` is the count in one *actual* member of the group, the
///   medoid (see [`medoid_of`]). It is a real reconstruction, but only one of
///   many, and the group can be very widely spread around it. Read it alongside
///   `modality_counts`, which gives the per-value range across the whole set.
///
/// `f_representative` is `NaN` in streaming mode, which cannot pick a medoid in
/// a single pass. That is deliberate: the column means the same thing in every
/// output file or is visibly absent, rather than quietly switching definition.
///
/// Invariant: All fields must have the same length to ensure valid data frame structure
#[derive(Clone, Debug)]
pub struct FrequencyTable {
    /// Sample categories: "all", "horns_min", "horns_max" each repeated for all scale values
    samples_group: FrequencySamplesColumn,
    /// Scale value of each row. Held as a float because a multi-item SPRITE
    /// grid has fractional values; at `items == 1` these are whole numbers.
    value: Vec<f64>,
    /// Mean count of the value across the group. Sums to `n` over a group.
    f_expected: Vec<f64>,
    /// Count of the value in the group's medoid. `NaN` when no medoid was
    /// computed, which is the case for every streaming run.
    f_representative: Vec<f64>,
    /// `f_expected / n`: the expected proportion of responses at this value.
    f_relative: Vec<f64>,
}

impl FrequencyTable {
    /// Create a new FrequencyTable, validating that all fields have the same length
    ///
    /// # Panics
    /// Panics if the lengths of value, f_expected, f_representative, or
    /// f_relative don't match the length of samples_group
    pub fn new(
        samples_group: FrequencySamplesColumn,
        value: Vec<f64>,
        f_expected: Vec<f64>,
        f_representative: Vec<f64>,
        f_relative: Vec<f64>,
    ) -> Self {
        let expected_len = samples_group.len();

        let name_len_tuples = [
            ("value", value.len()),
            ("f_expected", f_expected.len()),
            ("f_representative", f_representative.len()),
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
            f_expected,
            f_representative,
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
    pub fn value(&self) -> &[f64] {
        &self.value
    }

    /// Get a reference to the f_expected column
    pub fn f_expected(&self) -> &[f64] {
        &self.f_expected
    }

    /// Get a reference to the f_representative column
    pub fn f_representative(&self) -> &[f64] {
        &self.f_representative
    }

    /// Get a reference to the f_relative column
    pub fn f_relative(&self) -> &[f64] {
        &self.f_relative
    }
}

/// Distilled per-scale-value count distribution for efficient overlay rendering.
/// One row per (value, count) pair that actually occurred across all samples.
/// Precomputed to avoid expensive per-sample expansion on the R side.
#[derive(Clone, Debug)]
pub struct FrequencyDist {
    pub value: Vec<f64>,     // scale value (e.g. 1, 2, 3, 4, 5)
    pub count: Vec<i32>,     // raw integer count in a single sample (0..n)
    pub n_samples: Vec<u32>, // how many samples had this count at this scale value
}

/// Per-value count ranges derived from all samples.
///
/// One row per scale value: `count_lo` is the minimum count of that value
/// across all samples, `count_hi` is the maximum.
#[derive(Clone, Debug)]
pub struct ModalityCounts {
    /// Scale values
    pub value: Vec<f64>,
    /// Minimum count of each scale value across all samples
    pub count_lo: Vec<i32>,
    /// Maximum count of each scale value across all samples
    pub count_hi: Vec<i32>,
}

/// Pairwise frequency-ordering resolution for adjacent scale values.
///
/// One row per consecutive pair $(v_i, v_{i+1})$: `resolved` is true when
/// the ordering between the two values is the same in every sample, and
/// `a_greater` is true when `value_a` always has the higher count.
#[derive(Clone, Debug)]
pub struct ModalityPairs {
    /// Lower scale value of each adjacent pair
    pub value_a: Vec<f64>,
    /// Higher scale value of each adjacent pair
    pub value_b: Vec<f64>,
    /// True if the ordering between value_a and value_b is the same in every sample
    pub resolved: Vec<bool>,
    /// When resolved is true: true if value_a always has a higher count than value_b
    pub a_greater: Vec<bool>,
}

// Shape conclusions live in `crate::modality` as `ModalityShapes`, which
// records per-class sample counts and per-class count bounds instead of a fixed
// set of booleans. `j_shape_low` / `j_shape_high` are now
// `ModalityShapes::can_be_j_shape_low` / `_high`, derived from the same
// per-sample scan as every other class rather than from the `count_lo`/
// `count_hi` box, whose corners are not generally admissible samples.

/// Main metrics about the CLOSURE results
#[derive(Clone, Debug)]
pub struct MetricsMain {
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

/// Results table combining reconstructed samples and their horns values.
///
/// Samples are stored as frequency tables — one count per scale value rather
/// than one integer per observation. That is lossless: every sample a technique
/// in this crate produces is sorted, so its count vector determines it
/// completely. Call [`ResultsTable::sample`] to get one back as a vector of
/// values.
#[derive(Clone, Debug)]
pub struct ResultsTable<U> {
    pub id: Vec<f64>,
    pub counts: SampleCounts,
    pub horns: Vec<f64>,
    _marker: PhantomData<U>,
}

impl<U: IntegerType> ResultsTable<U> {
    /// Assemble a results table.
    ///
    /// # Panics
    /// Panics if `id`, `counts` and `horns` don't describe the same number of
    /// samples.
    pub fn new(id: Vec<f64>, counts: SampleCounts, horns: Vec<f64>) -> Self {
        assert_eq!(
            id.len(),
            counts.nrow(),
            "can't create a ResultsTable: `id` length ({}) doesn't match the number of samples ({})",
            id.len(),
            counts.nrow()
        );
        assert_eq!(
            horns.len(),
            counts.nrow(),
            "can't create a ResultsTable: `horns` length ({}) doesn't match the number of samples ({})",
            horns.len(),
            counts.nrow()
        );
        Self {
            id,
            counts,
            horns,
            _marker: PhantomData,
        }
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.counts.nrow()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Reconstruct sample `i` as a sorted vector of values.
    pub fn sample(&self, i: usize) -> Vec<U> {
        self.counts.sample_at(i)
    }

    /// Reconstruct every sample. Allocates the full expanded matrix, so prefer
    /// [`ResultsTable::sample`] when one sample at a time will do.
    pub fn samples(&self) -> Vec<Vec<U>> {
        self.counts.samples()
    }
}

/// Complete CLOSURE results with all statistics
#[derive(Clone, Debug)]
pub struct ResultListFromMeanSdN<U> {
    pub metrics_main: MetricsMain,
    pub metrics_horns: MetricsHorns,
    pub frequency: FrequencyTable,
    pub frequency_dist: FrequencyDist,
    pub modality_counts: ModalityCounts,
    pub modality_pairs: ModalityPairs,
    /// Per-shape-class sample counts and conditional per-value count bounds.
    pub modality_shapes: ModalityShapes,
    pub results: ResultsTable<U>,
}

impl<U: IntegerType> ResultListFromMeanSdN<U> {
    /// Create an empty result list for cases with no valid distributions
    ///
    /// Returns a `ResultListFromMeanSdN` with all metrics set to `NaN` or zero
    /// (depending on the metric), and with empty result vectors.
    /// Used when CLOSURE or SPRITE fail to find any distributions.
    ///
    /// Assumes a single-item scale. For a multi-item SPRITE grid, use
    /// [`ResultListFromMeanSdN::empty_on_grid`].
    pub fn empty(scale_min: U, scale_max: U) -> Self {
        let grid = ValueGrid::new(
            U::to_i32(&scale_min).unwrap(),
            U::to_i32(&scale_max).unwrap(),
            1,
            100,
        );
        empty_result_list(grid, 0)
    }

    /// Create an empty result list over an explicit value grid.
    pub fn empty_on_grid(grid: ValueGrid, n: usize) -> Self {
        empty_result_list(grid, n)
    }
}

/// Slack added to every float threshold before it is rounded to an integer.
///
/// Scale values are integers, so a sample's sum and sum of squares are exact
/// integers and the only rounding anywhere in the search is in the thresholds
/// themselves: `mean * n` is not exactly `81` for `mean = 3.24, n = 25`, and
/// `(sd + rounding_error_sd)²` is rarely exact either. A slack far below one
/// unit but far above f64 error keeps a sample whose statistic sits exactly on
/// a bound from being dropped by the last bit of that arithmetic.
const BOUND_TOLERANCE: f64 = 1e-6;

/// Integer-exact acceptance bounds for a CLOSURE search.
///
/// Shared by the DFS in this file and the DP counter in [`count`], so the two
/// accept exactly the same samples by construction rather than by luck.
///
/// A sample is admissible when `sum_lo <= Σx <= sum_hi` and `m2n_lo <=
/// n·Σx² − (Σx)² <= m2n_hi`. The second quantity is `n·(n−1)` times the sample
/// variance, which is why it is an integer.
pub(crate) struct SearchBounds {
    pub sum_lo: i64,
    pub sum_hi: i64,
    pub m2n_lo: i64,
    pub m2n_hi: i64,
    /// `(n−1)·(sd − rounding_error_sd)²`, floored at zero.
    pub var_nm1_lo: f64,
    /// `(n−1)·(sd + rounding_error_sd)²`.
    pub var_nm1_hi: f64,
}

impl SearchBounds {
    pub(crate) fn new(mean: f64, sd: f64, n: usize, re_mean: f64, re_sd: f64) -> Self {
        let n_f = n as f64;
        let target_sum = mean * n_f;
        let sum_slack = re_mean * n_f;
        let sd_lo = (sd - re_sd).max(0.0);
        let sd_hi = sd + re_sd;
        let var_nm1_lo = sd_lo * sd_lo * (n_f - 1.0);
        let var_nm1_hi = sd_hi * sd_hi * (n_f - 1.0);
        Self {
            sum_lo: (target_sum - sum_slack - BOUND_TOLERANCE).ceil() as i64,
            sum_hi: (target_sum + sum_slack + BOUND_TOLERANCE).floor() as i64,
            m2n_lo: (n_f * var_nm1_lo - BOUND_TOLERANCE).ceil() as i64,
            m2n_hi: (n_f * var_nm1_hi + BOUND_TOLERANCE).floor() as i64,
            var_nm1_lo,
            var_nm1_hi,
        }
    }
}

/// Everything the DFS needs, computed once per run.
///
/// The search runs entirely in integer arithmetic on a running sum and sum of
/// squares; see [`SearchBounds`] for why that is exact.
struct ClosureSearchContext {
    n: usize,
    scale_min: i64,
    scale_max: i64,
    sum_lo: i64,
    sum_hi: i64,
    /// Lower bound on `n·Σx² − (Σx)²` of a complete sample.
    m2n_lo: i64,
    /// `m2_hi[k]` bounds `k·Σx² − (Σx)²` over the first `k` values. The sum of
    /// squared deviations from the running mean never decreases as values are
    /// added, so a partial sample over the bound cannot be completed.
    m2_hi: Vec<i64>,
}

impl ClosureSearchContext {
    fn new<T: FloatType, U: IntegerType>(
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
    ) -> Result<Self, ParameterError> {
        let invalid = |msg: &str| Err(ParameterError::InputValidation(msg.to_string()));
        let (Some(n_i64), Some(scale_min), Some(scale_max)) =
            (U::to_i64(&n), U::to_i64(&scale_min), U::to_i64(&scale_max))
        else {
            return invalid("n, scale_min and scale_max must fit in an i64");
        };
        if n_i64 < 2 {
            return invalid("n must be at least 2");
        }
        if scale_min > scale_max {
            return invalid("scale_max must not be below scale_min");
        }
        let to_f64 = |x: T| T::to_f64(&x).filter(|v| v.is_finite());
        let (Some(mean), Some(sd), Some(re_mean), Some(re_sd)) = (
            to_f64(mean),
            to_f64(sd),
            to_f64(rounding_error_mean),
            to_f64(rounding_error_sd),
        ) else {
            return invalid("mean, sd and the rounding errors must be finite");
        };
        if sd < 0.0 || re_mean < 0.0 || re_sd < 0.0 {
            return invalid("sd and the rounding errors must not be negative");
        }
        let n = n_i64 as usize;
        let bounds = SearchBounds::new(mean, sd, n, re_mean, re_sd);
        let m2_hi = (0..=n)
            .map(|k| (k as f64 * bounds.var_nm1_hi + BOUND_TOLERANCE).floor() as i64)
            .collect();
        Ok(Self {
            n,
            scale_min,
            scale_max,
            sum_lo: bounds.sum_lo,
            sum_hi: bounds.sum_hi,
            m2n_lo: bounds.m2n_lo,
            m2_hi,
        })
    }

    /// Number of scale values.
    fn scale_range(&self) -> usize {
        (self.scale_max - self.scale_min + 1) as usize
    }

    /// Seed depth for splitting the search into independent branches: three
    /// values, or fewer when `n` is smaller than that. Two gave only 28
    /// very unevenly sized branches on a 7-point scale; three balances the
    /// load measurably better and four adds nothing.
    fn seed_depth(&self) -> usize {
        self.n.min(3)
    }
}

/// Count initial combinations with replacement
///
/// Computes the number of sorted combinations of length `depth` from
/// `scale_min..=scale_max`, i.e., the multiset coefficient
/// C(range_size + depth - 1, depth).
/// # Arguments
/// * `scale_min` - The minimum value of the scale.
/// * `scale_max` - The maximum value of the scale.
/// * `depth` - The length of each combination.
/// # Returns
/// The total number of unique combinations.
pub fn count_initial_combinations(scale_min: i32, scale_max: i32, depth: usize) -> i64 {
    let range_size = (scale_max - scale_min + 1) as i64;
    // C(range_size + depth - 1, depth) via iterative multiplication
    let k = depth as i64;
    let mut result: i64 = 1;
    for i in 0..k {
        result = result * (range_size + k - 1 - i) / (i + 1);
    }
    result
}

/// Calculate horns index for a frequency distribution
fn calculate_horns(freqs: &[f64], scale_min: i32, scale_max: i32) -> f64 {
    let total: f64 = freqs.iter().sum();
    if total == 0.0 {
        return 0.0;
    }

    // Called once per sample, so no allocation here.
    let value = |i: usize| (scale_min + i as i32) as f64;

    // Calculate mean
    let mean: f64 = freqs
        .iter()
        .enumerate()
        .map(|(i, f)| value(i) * (f / total))
        .sum();

    // Calculate weighted sum of squared deviations
    let numerator: f64 = freqs
        .iter()
        .enumerate()
        .map(|(i, f)| (f / total) * (value(i) - mean).powi(2))
        .sum();

    // Maximum possible variance given scale limits
    let denominator = ((scale_max - scale_min) as f64).powi(2) / 4.0;

    numerator / denominator
}

/// Calculate horns index for a uniform distribution over `k` grid values
fn horns_uniform(k: usize) -> f64 {
    horns_from_counts(&vec![1.0; k])
}

/// Horns index of a frequency vector laid out on a [`ValueGrid`].
///
/// Horns normalises by the maximum variance the scale allows, so it is
/// invariant under any affine relabelling of the value axis. That means the
/// grid *positions* give the same answer as the grid values do — which is what
/// makes the index directly comparable between a CLOSURE run on 1..7 and a
/// five-item SPRITE run on the same scale, whose grid is 31 values wide.
pub(crate) fn horns_from_counts(freqs: &[f64]) -> f64 {
    if freqs.len() < 2 {
        return 0.0;
    }
    calculate_horns(freqs, 0, freqs.len() as i32 - 1)
}

/// Earth Mover's Distance between two count vectors over the same grid.
///
/// For 1D distributions this equals the L1 distance between cumulative sums,
/// which respects the ordinal structure of the scale: a unit of mass moved one
/// step along the scale costs 1, correctly treating adjacent values as closer
/// than distant ones (unlike L1 or L2 on the raw vectors).
///
/// This is the definition [`medoid_of`] minimises. That function computes the
/// same quantity by a decomposition that avoids the quadratic pairwise loop, so
/// this direct form is kept as the reference the tests check it against.
#[cfg(test)]
fn emd_1d(f1: &[u32], f2: &[u32]) -> u64 {
    let mut cum1 = 0i64;
    let mut cum2 = 0i64;
    let mut total = 0u64;
    for (&a, &b) in f1.iter().zip(f2.iter()) {
        cum1 += a as i64;
        cum2 += b as i64;
        total += (cum1 - cum2).unsigned_abs();
    }
    total
}

/// The medoid of a group: the sample minimising the summed EMD to every other
/// sample in the group.
///
/// This is the real medoid, not the sample nearest the group mean. The two
/// differ — in CDF space the first is a median and the second is anchored on a
/// mean — and the medoid is the robust one, which is the entire reason for
/// preferring an actual sample over a centroid in the first place.
///
/// Ties are broken by taking the lexicographically smallest count vector, so
/// the answer does not depend on enumeration order. That matters because
/// enumeration order is not reproducible under `stop_after` or for SPRITE.
///
/// # Complexity
///
/// Naively this is `O(m² k)`. Since 1D EMD is the L1 distance between
/// cumulative counts, the total cost of a candidate `x` decomposes per grid
/// position into `Σ_y |F_x(j) − F_y(j)|`, and every `F(j)` is an integer in
/// `0..=n`. So one pass builds a histogram of cumulative counts per position
/// and a second pass answers each candidate from its prefix sums: `O(m k + k
/// n)` overall.
fn medoid_of<'a, I>(rows: I, k: usize, n: usize) -> Option<Vec<u32>>
where
    I: Iterator<Item = &'a [u32]> + Clone,
{
    if k == 0 {
        return None;
    }
    // Cumulative counts at the last position are always n, so they contribute
    // nothing to any distance and are skipped.
    let cuts = k - 1;
    let stride = n + 1;

    // hist[j * stride + c] = how many samples have cumulative count c at cut j.
    let mut hist = vec![0u64; cuts * stride];
    let mut m = 0u64;
    for row in rows.clone() {
        let mut cum = 0usize;
        for (j, &c) in row[..cuts].iter().enumerate() {
            cum += c as usize;
            hist[j * stride + cum.min(n)] += 1;
        }
        m += 1;
    }
    if m == 0 {
        return None;
    }

    // Per cut, prefix counts and prefix weighted sums over cumulative values.
    let mut pre_count = vec![0u64; cuts * (stride + 1)];
    let mut pre_weight = vec![0u64; cuts * (stride + 1)];
    for j in 0..cuts {
        for c in 0..stride {
            let h = hist[j * stride + c];
            pre_count[j * (stride + 1) + c + 1] = pre_count[j * (stride + 1) + c] + h;
            pre_weight[j * (stride + 1) + c + 1] = pre_weight[j * (stride + 1) + c] + h * c as u64;
        }
    }

    let mut best: Option<(u64, Vec<u32>)> = None;
    for row in rows {
        let mut cost = 0u64;
        let mut cum = 0usize;
        for (j, &c) in row[..cuts].iter().enumerate() {
            cum += c as usize;
            let v = cum.min(n);
            let base = j * (stride + 1);
            // Samples at or below v, and their summed cumulative counts.
            let lo_n = pre_count[base + v + 1];
            let lo_w = pre_weight[base + v + 1];
            let all_n = pre_count[base + stride];
            let all_w = pre_weight[base + stride];
            // Σ|v − c| split at v: those below contribute v·n − Σc, those above Σc − v·n.
            cost += (v as u64 * lo_n - lo_w) + ((all_w - lo_w) - v as u64 * (all_n - lo_n));
        }
        let better = match &best {
            None => true,
            Some((best_cost, best_row)) => {
                cost < *best_cost || (cost == *best_cost && row < best_row.as_slice())
            }
        };
        if better {
            best = Some((cost, row.to_vec()));
        }
    }
    best.map(|(_, row)| row)
}

/// Return the frequency-table rows for one group of samples.
///
/// Yields `(value, f_expected, f_representative, f_relative)`, where
/// `f_expected` is the mean count per sample, `f_representative` is the count in
/// the group medoid, and `f_relative` is `f_expected / n`. See
/// [`FrequencyTable`] for why both summaries are reported.
fn compute_frequency_rows<'a, I>(
    grid: &ValueGrid,
    n: usize,
    rows: I,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)
where
    I: Iterator<Item = &'a [u32]> + Clone,
{
    let k = grid.len();
    let value = grid.values().to_vec();

    // Summed counts and group size in one pass.
    let mut summed = vec![0u64; k];
    let mut m = 0usize;
    for row in rows.clone() {
        for (acc, &c) in summed.iter_mut().zip(row.iter()) {
            *acc += c as u64;
        }
        m += 1;
    }

    if m == 0 {
        return (
            value,
            vec![f64::NAN; k],
            vec![f64::NAN; k],
            vec![f64::NAN; k],
        );
    }

    let f_expected: Vec<f64> = summed.iter().map(|&s| s as f64 / m as f64).collect();
    let f_relative: Vec<f64> = f_expected.iter().map(|&f| f / n as f64).collect();
    let f_representative = match medoid_of(rows, k, n) {
        Some(medoid) => medoid.iter().map(|&c| c as f64).collect(),
        None => vec![f64::NAN; k],
    };

    (value, f_expected, f_representative, f_relative)
}

/// Calculate the distilled count distribution across all samples.
/// For each (scale value, raw count) pair that occurred in any sample,
/// records how many samples produced that count at that scale value.
fn calculate_frequency_dist(counts: &SampleCounts) -> FrequencyDist {
    if counts.is_empty() {
        return FrequencyDist {
            value: Vec::new(),
            count: Vec::new(),
            n_samples: Vec::new(),
        };
    }

    let grid = counts.grid();
    let k = grid.len();
    let n = counts.n();

    // Flat 2D array: dist[v_idx * (n+1) + count] = number of samples with that count.
    // Direct index arithmetic replaces HashMap hashing entirely.
    let mut dist = vec![0u32; k * (n + 1)];

    for row in counts.rows() {
        for (v_idx, &cnt) in row.iter().enumerate() {
            dist[v_idx * (n + 1) + cnt as usize] += 1;
        }
    }

    // Collect non-zero entries; iteration order is already (value asc, count asc)
    let mut value = Vec::new();
    let mut count = Vec::new();
    let mut n_samples = Vec::new();
    for v_idx in 0..k {
        for cnt in 0..=n {
            let n_samp = dist[v_idx * (n + 1) + cnt];
            if n_samp > 0 {
                value.push(grid.values()[v_idx]);
                count.push(cnt as i32);
                n_samples.push(n_samp);
            }
        }
    }

    FrequencyDist {
        value,
        count,
        n_samples,
    }
}

/// Compute the per-value count ranges and adjacent-pair orderings from a
/// `FrequencyDist`, returning the two result structs that map directly to the
/// R-level tibbles.
///
/// Shape conclusions are *not* computed here. They require a per-sample scan —
/// these bounds describe a box around the result set, and the box has corners
/// that are not admissible samples — and live in [`ModalityShapes`], built by
/// [`ShapeAccumulator`] as samples stream past.
pub(crate) fn compute_modality(
    freq_dist: &FrequencyDist,
    grid: &ValueGrid,
) -> (ModalityCounts, ModalityPairs) {
    let n_vals = grid.len();

    if n_vals == 0 || freq_dist.value.is_empty() {
        return (
            ModalityCounts {
                value: Vec::new(),
                count_lo: Vec::new(),
                count_hi: Vec::new(),
            },
            ModalityPairs {
                value_a: Vec::new(),
                value_b: Vec::new(),
                resolved: Vec::new(),
                a_greater: Vec::new(),
            },
        );
    }

    // --- per-value count ranges ---------------------------------------
    let mut count_lo = vec![i32::MAX; n_vals];
    let mut count_hi = vec![i32::MIN; n_vals];

    for (&val, &cnt) in freq_dist.value.iter().zip(freq_dist.count.iter()) {
        let Some(idx) = grid.index_of_value(val) else {
            continue;
        };
        if cnt < count_lo[idx] {
            count_lo[idx] = cnt;
        }
        if cnt > count_hi[idx] {
            count_hi[idx] = cnt;
        }
    }

    // Guard: if a value somehow has no rows, default its range to [0, 0]
    for i in 0..n_vals {
        if count_lo[i] == i32::MAX {
            count_lo[i] = 0;
            count_hi[i] = 0;
        }
    }

    let values: Vec<f64> = grid.values().to_vec();

    // --- pairwise adjacent orderings ----------------------------------
    let n_pairs = n_vals - 1;
    let mut value_a = Vec::with_capacity(n_pairs);
    let mut value_b = Vec::with_capacity(n_pairs);
    let mut resolved = Vec::with_capacity(n_pairs);
    let mut a_greater = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        // a_always_greater: lo_a > hi_b — value_a always has more counts than value_b
        let a_always_greater = count_lo[i] > count_hi[i + 1];
        let b_always_greater = count_lo[i + 1] > count_hi[i];
        value_a.push(values[i]);
        value_b.push(values[i + 1]);
        resolved.push(a_always_greater || b_always_greater);
        a_greater.push(a_always_greater);
    }

    (
        ModalityCounts {
            value: values,
            count_lo,
            count_hi,
        },
        ModalityPairs {
            value_a,
            value_b,
            resolved,
            a_greater,
        },
    )
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
    deviations.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    median(&deviations)
}

/// Used by `ResultListFromMeanSdN::empty()`
pub(crate) fn empty_result_list<U>(grid: ValueGrid, n: usize) -> ResultListFromMeanSdN<U>
where
    U: IntegerType,
{
    let group_size = grid.len();
    let nrow_frequency = group_size * SampleCategory::COUNT;

    // Build a vector of scale values repeated as many times as there are categories of samples.
    // Currently, this means 3 times (for "all", "horns_min", and "horns_max").
    let mut value = Vec::with_capacity(nrow_frequency);
    for _ in 0..SampleCategory::COUNT {
        value.extend_from_slice(grid.values());
    }

    ResultListFromMeanSdN {
        metrics_main: MetricsMain {
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
            vec![f64::NAN; nrow_frequency],
            vec![f64::NAN; nrow_frequency],
        ),
        frequency_dist: FrequencyDist {
            value: Vec::new(),
            count: Vec::new(),
            n_samples: Vec::new(),
        },
        modality_counts: ModalityCounts {
            value: Vec::new(),
            count_lo: Vec::new(),
            count_hi: Vec::new(),
        },
        modality_pairs: ModalityPairs {
            value_a: Vec::new(),
            value_b: Vec::new(),
            resolved: Vec::new(),
            a_greater: Vec::new(),
        },
        // Nothing was scanned, so nothing is ruled out: every `can_be` query on
        // this returns `None`, not `false`.
        modality_shapes: modality::empty_shapes(DEFAULT_MODE_PROMINENCE),
        results: ResultsTable::new(Vec::new(), SampleCounts::new(grid, n), Vec::new()),
    }
}

/// Rows per parallel task in the per-sample statistics pass.
const STATS_CHUNK_ROWS: usize = 4096;

/// Calculate all statistics for the samples.
///
/// `exhaustive` says whether `counts` is the complete set of samples matching
/// the reported statistics. It is false for a truncated CLOSURE search and for
/// SPRITE, which samples the solution space rather than enumerating it. Only
/// [`ModalityShapes`] uses it, and only to decide whether an empty shape class
/// is a proof of impossibility or merely an absence of evidence.
fn counts_to_result_list<U>(counts: SampleCounts, exhaustive: bool) -> ResultListFromMeanSdN<U>
where
    U: IntegerType,
{
    let grid = counts.grid().clone();
    let group_size = grid.len();

    // Handle empty samples case
    if counts.is_empty() {
        let n = counts.n();
        return empty_result_list(grid, n);
    }

    let n = counts.n();
    let samples_all = counts.nrow();
    let values_all = samples_all * n;

    // Calculate horns for each sample and classify its shape in the same
    // pass, in parallel over chunks of rows. This pass costs more than the
    // search itself on large result sets.
    let k = group_size;
    let chunks: Vec<(Vec<f64>, ShapeAccumulator)> = counts
        .as_flat()
        .par_chunks(k * STATS_CHUNK_ROWS)
        .map(|chunk| {
            let mut shapes = ShapeAccumulator::new(k, n, DEFAULT_MODE_PROMINENCE);
            let mut horns = Vec::with_capacity(chunk.len() / k);
            let mut freqs = vec![0.0f64; k];
            for row in chunk.chunks_exact(k) {
                for (slot, &c) in freqs.iter_mut().zip(row.iter()) {
                    *slot = c as f64;
                }
                horns.push(horns_from_counts(&freqs));
                shapes.update(row);
            }
            (horns, shapes)
        })
        .collect();
    let mut horns_values = Vec::with_capacity(samples_all);
    let mut shapes = ShapeAccumulator::new(k, n, DEFAULT_MODE_PROMINENCE);
    for (horns, part) in chunks {
        horns_values.extend(horns);
        shapes.merge(part);
    }
    let modality_shapes = shapes.finish(exhaustive);

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
    horns_sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
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

    // Frequency rows for all three groups. Each group reports both its mean
    // profile and its medoid; see `FrequencyTable` for why both are needed.
    let (all_value, all_f_exp, all_f_rep, all_f_rel) =
        compute_frequency_rows(&grid, n, counts.rows());

    let (min_value, min_f_exp, min_f_rep, min_f_rel) =
        compute_frequency_rows(&grid, n, min_indices.iter().map(|&i| counts.row(i)));

    let (max_value, max_f_exp, max_f_rep, max_f_rel) =
        compute_frequency_rows(&grid, n, max_indices.iter().map(|&i| counts.row(i)));

    // Combine all frequency data into a single table
    let mut combined_value = all_value;
    combined_value.extend(min_value);
    combined_value.extend(max_value);

    let mut combined_f_expected = all_f_exp;
    combined_f_expected.extend(min_f_exp);
    combined_f_expected.extend(max_f_exp);

    let mut combined_f_representative = all_f_rep;
    combined_f_representative.extend(min_f_rep);
    combined_f_representative.extend(max_f_rep);

    let mut combined_f_relative = all_f_rel;
    combined_f_relative.extend(min_f_rel);
    combined_f_relative.extend(max_f_rel);

    // Create ID column for results table
    let id: Vec<f64> = (1..=samples_all).map(|i| i as f64).collect();

    let frequency_dist = calculate_frequency_dist(&counts);
    let (modality_counts, modality_pairs) = compute_modality(&frequency_dist, &grid);

    ResultListFromMeanSdN {
        metrics_main: MetricsMain {
            samples_all: samples_all as f64,
            values_all: values_all as f64,
        },
        metrics_horns: MetricsHorns {
            mean: horns_mean,
            uniform: horns_uniform(group_size),
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
            combined_f_expected,
            combined_f_representative,
            combined_f_relative,
        ),
        frequency_dist,
        modality_counts,
        modality_pairs,
        modality_shapes,
        results: ResultsTable::new(id, counts, horns_values),
    }
}

/// Create a Parquet writer for the results table with appropriate schema
/// Now stores samples as a list column instead of expanding them
fn create_results_writer(file_path: &str) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
    // Create schema with id column, samples as list column, plus horns column
    // Note: List items are marked as nullable to match what Arrow's ListBuilder produces
    let fields = vec![
        Field::new("id", DataType::Float64, false),
        Field::new(
            "sample",
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
        Field::new("value", DataType::Float64, false),
        Field::new("f_expected", DataType::Float64, false),
        Field::new("f_representative", DataType::Float64, false),
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
    U: IntegerType,
{
    // Create arrays for each column
    let mut arrays: Vec<ArrayRef> = Vec::new();

    // Add ID column
    let id_data: Vec<f64> = results.id[start_idx..end_idx].to_vec();
    arrays.push(Arc::new(Float64Array::from(id_data)));

    // Add samples column as a list using the standard ListBuilder.
    // Samples are expanded from their counts one at a time, so the legacy
    // layout never needs the whole matrix in memory at once.
    let mut list_builder = ListBuilder::new(Int32Builder::new());

    for i in start_idx..end_idx {
        for &val in &results.sample(i) {
            list_builder.values().append_value(U::to_i32(&val).unwrap());
        }
        // Mark the end of this list
        list_builder.append(true);
    }

    arrays.push(Arc::new(list_builder.finish()));

    // Add horns column
    let horns_data: Vec<f64> = results.horns[start_idx..end_idx].to_vec();
    arrays.push(Arc::new(Float64Array::from(horns_data)));

    // Create schema - matching the schema from create_results_writer
    let fields = vec![
        Field::new("id", DataType::Float64, false),
        Field::new(
            "sample",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            false,
        ), // true for nullable items
        Field::new("horns", DataType::Float64, false),
    ];

    let schema = Arc::new(Schema::new(fields));

    RecordBatch::try_new(schema, arrays).map_err(|e| e.into())
}

/// Every sorted combination of `depth` scale-value indices from
/// `0..scale_range`, each the seed of one independent search branch.
fn generate_initial_combinations(scale_range: usize, depth: usize) -> Vec<Vec<usize>> {
    let mut seeds = Vec::new();
    let mut combo = Vec::with_capacity(depth);
    fn walk(
        scale_range: usize,
        min_idx: usize,
        left: usize,
        combo: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if left == 0 {
            out.push(combo.clone());
            return;
        }
        for idx in min_idx..scale_range {
            combo.push(idx);
            walk(scale_range, idx, left - 1, combo, out);
            combo.pop();
        }
    }
    walk(scale_range, 0, depth, &mut combo, &mut seeds);
    seeds
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
/// For large result sets, use `closure_parallel_streaming()` instead.
///
/// # Parameters
/// - `items`: Number of items averaged (must be 1 for CLOSURE)
/// - `stop_after`: Optional limit on number of samples to find. If None, finds all samples.
#[allow(clippy::too_many_arguments)]
pub fn closure_parallel<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    items: u32,
    parquet_config: Option<ParquetConfig>,
    stop_after: Option<usize>,
) -> Result<ResultListFromMeanSdN<U>, ParameterError>
where
    T: FloatType,
    U: IntegerType + 'static,
{
    if items != 1 {
        return Err(ParameterError::InputValidation(
            "CLOSURE requires items == 1".to_string(),
        ));
    }
    let ctx = ClosureSearchContext::new(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
    )?;
    let n_usize = ctx.n;
    let combinations = generate_initial_combinations(ctx.scale_range(), ctx.seed_depth());

    let k = ctx.scale_range();

    // Process combinations in parallel with optional early termination.
    // Every branch returns a flat row-major buffer of count vectors, one row
    // per valid sample.
    let branches: Vec<Vec<u32>> = match stop_after {
        // Small limits run sequentially: the parallel overhead outweighs the
        // work of finding a handful of samples.
        Some(limit) if limit <= 100 => {
            let mut found = Vec::with_capacity(limit * k);
            for combo in &combinations {
                let have = found.len() / k;
                if have >= limit {
                    break;
                }
                found.extend(closure_branch(combo, &ctx, Some(limit - have)));
            }
            vec![found]
        }
        Some(limit) => {
            // Branches stop early once the shared count passes the limit, but
            // several can still overshoot it together; the truncation below
            // makes `stop_after` an exact upper bound.
            let found_count = AtomicUsize::new(0);
            combinations
                .par_iter()
                .map(|combo| {
                    let current = found_count.load(Ordering::Relaxed);
                    if current >= limit {
                        return Vec::new();
                    }
                    let branch_results = closure_branch(combo, &ctx, Some(limit - current));
                    found_count.fetch_add(branch_results.len() / k, Ordering::Relaxed);
                    branch_results
                })
                .collect()
        }
        None => combinations
            .par_iter()
            .map(|combo| closure_branch(combo, &ctx, None))
            .collect(),
    };
    let mut results = Vec::with_capacity(branches.iter().map(Vec::len).sum());
    for branch in branches {
        results.extend_from_slice(&branch);
    }
    if let Some(limit) = stop_after {
        results.truncate(limit * k);
    }

    // Calculate all statistics
    let counts = SampleCounts::from_flat(
        <Closure as SampleFormat<U>>::value_grid(scale_min, scale_max, items),
        n_usize,
        results,
    );
    // `stop_after` truncates the enumeration, so the result set is then only
    // part of the solution space and an unseen shape is not an impossible one.
    let closure_results: ResultListFromMeanSdN<U> =
        counts_to_result_list(counts, stop_after.is_none());

    // Write to Parquet if configured
    if let Some(config) = parquet_config {
        <Closure as SampleFormat<U>>::write_result_list(
            &normalize_base_path(&config.file_path),
            &closure_results,
            &config,
        );
    }

    Ok(closure_results)
}

/// Append a trailing separator so `{base_path}name.parquet` lands inside the
/// directory the caller named.
fn normalize_base_path(file_path: &str) -> String {
    if file_path.ends_with('/') {
        file_path.to_string()
    } else {
        format!("{}/", file_path)
    }
}

/// Write the four statistics tables that accompany every result set.
///
/// Shared by both techniques and both modes, so a `metrics_horns.parquet` means
/// the same thing wherever it came from.
pub(crate) fn write_statistics_files<U>(base_path: &str, results: &ResultListFromMeanSdN<U>) {
    let Ok((mut mm_writer, mut mh_writer, mut freq_writer, mm_schema, mh_schema, freq_schema)) =
        create_stats_writers(base_path)
    else {
        return;
    };

    // Write metrics_main
    let mm_batch = RecordBatch::try_new(
        mm_schema,
        vec![
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
            Arc::new(StringArray::from(
                results.frequency.samples_group().to_vec(),
            )),
            Arc::new(Float64Array::from(results.frequency.value().to_vec())),
            Arc::new(Float64Array::from(results.frequency.f_expected().to_vec())),
            Arc::new(Float64Array::from(
                results.frequency.f_representative().to_vec(),
            )),
            Arc::new(Float64Array::from(results.frequency.f_relative().to_vec())),
        ],
    );
    if let Ok(batch) = freq_batch {
        let _ = freq_writer.write(&batch);
    }
    let _ = freq_writer.close();

    let _ = write_frequency_dist_to_parquet(
        &results.frequency_dist,
        &format!("{}frequency_dist.parquet", base_path),
    );
    let _ = write_modality_to_parquet(base_path, results);
}

/// Running frequency statistics for the streaming paths.
///
/// Everything here is indexed by grid position, so a sample's count vector —
/// the same one being written to disk — is the only per-sample structure a
/// streaming path needs to build.
pub(crate) struct StreamingFrequencyState {
    current_min_horns: f64,
    current_max_horns: f64,
    /// Summed counts per grid position across all samples.
    all_freq: Vec<i64>,
    /// Summed counts per grid position across the minimum-horns samples.
    min_freq: Vec<i64>,
    /// Summed counts per grid position across the maximum-horns samples.
    max_freq: Vec<i64>,
    min_count: usize,
    max_count: usize,
    /// Flat 2D array for frequency_dist: indexed by [v_idx * (n+1) + count]
    freq_dist: Vec<u32>,
    /// Per-sample shape classification. Streaming sees every sample exactly
    /// once, which is all the shape analysis needs, so it reports the same
    /// classes and the same conditional bounds as memory mode.
    shapes: ShapeAccumulator,
    k: usize,
    n: usize,
}

impl StreamingFrequencyState {
    /// Start tracking a run over `k` grid values with samples of size `n`.
    pub(crate) fn new(k: usize, n: usize) -> Self {
        Self {
            current_min_horns: f64::INFINITY,
            current_max_horns: f64::NEG_INFINITY,
            all_freq: vec![0; k],
            min_freq: vec![0; k],
            max_freq: vec![0; k],
            min_count: 0,
            max_count: 0,
            freq_dist: vec![0u32; k * (n + 1)],
            shapes: ShapeAccumulator::new(k, n, DEFAULT_MODE_PROMINENCE),
            k,
            n,
        }
    }

    /// Fold another state over the same grid into this one, so branches can
    /// accumulate locally and take the shared lock once.
    pub(crate) fn merge(&mut self, other: Self) {
        debug_assert_eq!((self.k, self.n), (other.k, other.n));
        for (a, b) in self.all_freq.iter_mut().zip(other.all_freq) {
            *a += b;
        }
        for (a, b) in self.freq_dist.iter_mut().zip(other.freq_dist) {
            *a += b;
        }
        self.shapes.merge(other.shapes);

        // The same tie rule as `update`, applied to a whole group at once.
        if other.min_count > 0 {
            if (other.current_min_horns - self.current_min_horns).abs() < 1e-10 {
                for (a, b) in self.min_freq.iter_mut().zip(other.min_freq) {
                    *a += b;
                }
                self.min_count += other.min_count;
            } else if other.current_min_horns < self.current_min_horns {
                self.current_min_horns = other.current_min_horns;
                self.min_freq = other.min_freq;
                self.min_count = other.min_count;
            }
        }
        if other.max_count > 0 {
            if (other.current_max_horns - self.current_max_horns).abs() < 1e-10 {
                for (a, b) in self.max_freq.iter_mut().zip(other.max_freq) {
                    *a += b;
                }
                self.max_count += other.max_count;
            } else if other.current_max_horns > self.current_max_horns {
                self.current_max_horns = other.current_max_horns;
                self.max_freq = other.max_freq;
                self.max_count = other.max_count;
            }
        }
    }

    /// Fold one sample's counts into the running statistics.
    pub(crate) fn update(&mut self, counts: &[u32], horns: f64) {
        self.shapes.update(counts);
        let stride = self.n + 1;
        for (v_idx, &cnt) in counts.iter().enumerate() {
            self.all_freq[v_idx] += cnt as i64;
            self.freq_dist[v_idx * stride + cnt as usize] += 1;
        }

        // Horns extremes. Ties accumulate into the group; a new extreme
        // replaces it.
        if (horns - self.current_min_horns).abs() < 1e-10 {
            for (acc, &cnt) in self.min_freq.iter_mut().zip(counts.iter()) {
                *acc += cnt as i64;
            }
            self.min_count += 1;
        } else if horns < self.current_min_horns {
            self.current_min_horns = horns;
            self.min_freq.clear();
            self.min_freq.extend(counts.iter().map(|&c| c as i64));
            self.min_count = 1;
        }

        if (horns - self.current_max_horns).abs() < 1e-10 {
            for (acc, &cnt) in self.max_freq.iter_mut().zip(counts.iter()) {
                *acc += cnt as i64;
            }
            self.max_count += 1;
        } else if horns > self.current_max_horns {
            self.current_max_horns = horns;
            self.max_freq.clear();
            self.max_freq.extend(counts.iter().map(|&c| c as i64));
            self.max_count = 1;
        }
    }
}

/// Write a FrequencyDist to a flat Parquet file at `path`.
fn write_frequency_dist_to_parquet(
    dist: &FrequencyDist,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Float64, false),
        Field::new("count", DataType::Int32, false),
        Field::new("n_samples", DataType::UInt32, false),
    ]));
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(dist.value.clone())),
            Arc::new(Int32Array::from(dist.count.clone())),
            Arc::new(UInt32Array::from(dist.n_samples.clone())),
        ],
    )?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write one Parquet file from a schema and its columns.
fn write_table(
    path: &str,
    schema: Arc<Schema>,
    columns: Vec<arrow::array::ArrayRef>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    writer.write(&RecordBatch::try_new(schema, columns)?)?;
    writer.close()?;
    Ok(())
}

/// Write the four modality tables.
///
/// These used to be computed and then dropped on the floor: no writer existed
/// for them, so the whole shape analysis was reachable only from Rust. The
/// tables are:
///
/// - `modality_counts`: per-value count range across the whole result set.
/// - `modality_pairs`: whether the ordering of each adjacent value pair is the
///   same in every sample.
/// - `modality_shapes`: **per-shape-class** count range — the conditional
///   bounds. One row per (class, value); `n_samples` is constant within a
///   class.
/// - `modality_summary`: one row. How many samples of each class, whether the
///   search was exhaustive, and the unimodality-deficit spread.
///
/// `modality_shapes` is the table that supports a claim of the form "if the
/// data had this shape, then the count of this value was in this range", which
/// is the only shape claim a single atypical sample cannot defeat.
fn write_modality_to_parquet<U>(
    base_path: &str,
    results: &ResultListFromMeanSdN<U>,
) -> Result<(), Box<dyn std::error::Error>> {
    let counts = &results.modality_counts;
    write_table(
        &format!("{}modality_counts.parquet", base_path),
        Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
            Field::new("count_lo", DataType::Int32, false),
            Field::new("count_hi", DataType::Int32, false),
        ])),
        vec![
            Arc::new(Float64Array::from(counts.value.clone())),
            Arc::new(Int32Array::from(counts.count_lo.clone())),
            Arc::new(Int32Array::from(counts.count_hi.clone())),
        ],
    )?;

    let pairs = &results.modality_pairs;
    write_table(
        &format!("{}modality_pairs.parquet", base_path),
        Arc::new(Schema::new(vec![
            Field::new("value_a", DataType::Float64, false),
            Field::new("value_b", DataType::Float64, false),
            Field::new("resolved", DataType::Boolean, false),
            Field::new("a_greater", DataType::Boolean, false),
        ])),
        vec![
            Arc::new(Float64Array::from(pairs.value_a.clone())),
            Arc::new(Float64Array::from(pairs.value_b.clone())),
            Arc::new(BooleanArray::from(pairs.resolved.clone())),
            Arc::new(BooleanArray::from(pairs.a_greater.clone())),
        ],
    )?;

    // Long format: one row per (class, value). Classes with no members
    // contribute no rows, which is how "impossible under this shape" reads.
    let shapes = &results.modality_shapes;
    let grid_values = &results.modality_counts.value;
    let mut s_class = Vec::new();
    let mut s_n = Vec::new();
    let mut s_value = Vec::new();
    let mut s_lo = Vec::new();
    let mut s_hi = Vec::new();
    for bounds in &shapes.bounds {
        for (i, &lo) in bounds.count_lo.iter().enumerate() {
            s_class.push(bounds.class.as_str());
            s_n.push(bounds.n_samples);
            s_value.push(grid_values.get(i).copied().unwrap_or(f64::NAN));
            s_lo.push(lo);
            s_hi.push(bounds.count_hi[i]);
        }
    }
    write_table(
        &format!("{}modality_shapes.parquet", base_path),
        Arc::new(Schema::new(vec![
            Field::new("class", DataType::Utf8, false),
            Field::new("n_samples", DataType::UInt64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("count_lo", DataType::Int32, false),
            Field::new("count_hi", DataType::Int32, false),
        ])),
        vec![
            Arc::new(StringArray::from(s_class)),
            Arc::new(UInt64Array::from(s_n)),
            Arc::new(Float64Array::from(s_value)),
            Arc::new(Int32Array::from(s_lo)),
            Arc::new(Int32Array::from(s_hi)),
        ],
    )?;

    // One row: per-class totals plus the scan's own provenance. `exhaustive` is
    // what tells a reader whether a zero count is a proof or just an absence.
    let mut fields = vec![
        Field::new("exhaustive", DataType::Boolean, false),
        Field::new("n_scanned", DataType::UInt64, false),
        Field::new("min_prominence", DataType::Float64, false),
        Field::new("deficit_min", DataType::UInt32, false),
        Field::new("deficit_mean", DataType::Float64, false),
        Field::new("deficit_max", DataType::UInt32, false),
    ];
    let mut columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![shapes.exhaustive])),
        Arc::new(UInt64Array::from(vec![shapes.n_scanned])),
        Arc::new(Float64Array::from(vec![shapes.min_prominence])),
        Arc::new(UInt32Array::from(vec![shapes.deficit_min])),
        Arc::new(Float64Array::from(vec![shapes.deficit_mean])),
        Arc::new(UInt32Array::from(vec![shapes.deficit_max])),
    ];
    for class in ShapeClass::all() {
        fields.push(Field::new(
            format!("n_{}", class.as_str()),
            DataType::UInt64,
            false,
        ));
        columns.push(Arc::new(UInt64Array::from(vec![shapes.n_of(class)])));
    }
    write_table(
        &format!("{}modality_summary.parquet", base_path),
        Arc::new(Schema::new(fields)),
        columns,
    )?;

    // The prominence envelope: per-class counts at every threshold in the band,
    // one row per (threshold, class). A zero count in `modality_summary` is a
    // proof of impossibility only as far as this table keeps it at zero, so the
    // two are meant to be read together.
    let mut p_prom = Vec::new();
    let mut p_counts = Vec::new();
    let mut p_primary = Vec::new();
    let mut p_class = Vec::new();
    let mut p_n = Vec::new();
    for rung in &shapes.ladder {
        for class in ShapeClass::all() {
            p_prom.push(rung.min_prominence);
            p_counts.push(rung.min_prominence_counts);
            p_primary.push(rung.primary);
            p_class.push(class.as_str());
            p_n.push(rung.n_of(class));
        }
    }
    write_table(
        &format!("{}modality_prominence.parquet", base_path),
        Arc::new(Schema::new(vec![
            Field::new("min_prominence", DataType::Float64, false),
            Field::new("min_prominence_counts", DataType::UInt32, false),
            Field::new("primary", DataType::Boolean, false),
            Field::new("class", DataType::Utf8, false),
            Field::new("n_samples", DataType::UInt64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(p_prom)),
            Arc::new(UInt32Array::from(p_counts)),
            Arc::new(BooleanArray::from(p_primary)),
            Arc::new(StringArray::from(p_class)),
            Arc::new(UInt64Array::from(p_n)),
        ],
    )?;

    Ok(())
}

/// Write the statistics files for a streaming run.
///
/// Rebuilds the same tables memory mode produces from the running state, then
/// hands them to the shared writer, so both modes emit identical files.
pub(crate) fn write_streaming_statistics(
    base_path: &str,
    all_horns: &[f64],
    n_usize: usize,
    grid: &ValueGrid,
    final_freq_state: Arc<Mutex<StreamingFrequencyState>>,
    exhaustive: bool,
) {
    let samples_all = all_horns.len();
    if samples_all == 0 {
        // Still create the files so a reader finds a complete, empty result set.
        let results: ResultListFromMeanSdN<i32> = empty_result_list(grid.clone(), n_usize);
        write_statistics_files(base_path, &results);
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
    horns_sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let horns_min = horns_sorted[0];
    let horns_max = horns_sorted[samples_all - 1];
    let horns_median = median(&horns_sorted);
    let horns_mad = mad(all_horns, horns_median);

    let mut state = final_freq_state.lock().unwrap();
    let k = grid.len();

    // One block of frequency rows per sample category. A single pass can build
    // `f_expected` exactly, but picking a medoid needs the whole group at once,
    // so `f_representative` is NaN here. Both columns therefore mean exactly
    // what they mean in memory mode; the one streaming cannot supply is
    // visibly absent rather than quietly redefined.
    let mut value = Vec::with_capacity(k * SampleCategory::COUNT);
    let mut f_expected = Vec::with_capacity(k * SampleCategory::COUNT);
    let mut f_relative = Vec::with_capacity(k * SampleCategory::COUNT);
    let f_representative = vec![f64::NAN; k * SampleCategory::COUNT];

    let groups: [(&[i64], usize); SampleCategory::COUNT] = [
        (&state.all_freq, samples_all),
        (&state.min_freq, state.min_count),
        (&state.max_freq, state.max_count),
    ];

    for (freqs, group_size) in groups {
        let group_samples = group_size as f64;
        let group_values = (group_size * n_usize) as f64;
        for v_idx in 0..k {
            let count = freqs.get(v_idx).copied().unwrap_or(0) as f64;
            value.push(grid.values()[v_idx]);
            f_expected.push(count / group_samples);
            f_relative.push(count / group_values);
        }
    }

    // Build frequency_dist from the flat 2D array
    let mut dist_value = Vec::new();
    let mut dist_count = Vec::new();
    let mut dist_n_samples = Vec::new();
    let stride = state.n + 1;
    for v_idx in 0..state.k {
        for cnt in 0..=state.n {
            let n_samp = state.freq_dist[v_idx * stride + cnt];
            if n_samp > 0 {
                dist_value.push(grid.values()[v_idx]);
                dist_count.push(cnt as i32);
                dist_n_samples.push(n_samp);
            }
        }
    }
    // Take the accumulator out so it can be finished; the state is discarded
    // immediately afterwards.
    let shapes = std::mem::replace(
        &mut state.shapes,
        ShapeAccumulator::new(0, 0, DEFAULT_MODE_PROMINENCE),
    );
    drop(state);
    let modality_shapes = shapes.finish(exhaustive);

    let frequency_dist = FrequencyDist {
        value: dist_value,
        count: dist_count,
        n_samples: dist_n_samples,
    };
    let (modality_counts, modality_pairs) = compute_modality(&frequency_dist, grid);

    let results: ResultListFromMeanSdN<i32> = ResultListFromMeanSdN {
        metrics_main: MetricsMain {
            samples_all: samples_all as f64,
            values_all: values_all as f64,
        },
        metrics_horns: MetricsHorns {
            mean: horns_mean,
            uniform: horns_uniform(k),
            sd: horns_sd,
            cv: horns_sd / horns_mean,
            mad: horns_mad,
            min: horns_min,
            median: horns_median,
            max: horns_max,
            range: horns_max - horns_min,
        },
        frequency: FrequencyTable::new(
            FrequencySamplesColumn::new(k),
            value,
            f_expected,
            f_representative,
            f_relative,
        ),
        frequency_dist,
        modality_counts,
        modality_pairs,
        modality_shapes,
        results: ResultsTable::new(
            Vec::new(),
            SampleCounts::new(grid.clone(), n_usize),
            Vec::new(),
        ),
    };

    write_statistics_files(base_path, &results);
}

/// Generate all valid combinations (streaming mode) with summary statistics
///
/// This function computes all valid combinations and streams them directly to
/// Parquet files without keeping them in memory. Statistics are computed
/// incrementally.
///
/// Use this when:
/// - Result sets are very large (> 1GB)
/// - You only need file output, not in-memory processing
/// - Memory efficiency is critical
///
/// Returns a StreamingResult with the total count and file path.
///
/// # Parameters
/// - `items`: Number of items averaged (must be 1 for CLOSURE)
/// - `stop_after`: Optional limit on number of samples to find. If None, finds
///   all samples.
#[allow(clippy::too_many_arguments)]
pub fn closure_parallel_streaming<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    items: u32,
    config: StreamingConfig,
    stop_after: Option<usize>,
) -> Result<StreamingResult, ParameterError>
where
    T: FloatType,
    U: IntegerType + 'static,
{
    if items != 1 {
        return Err(ParameterError::InputValidation(
            "CLOSURE requires items == 1".to_string(),
        ));
    }
    let ctx = ClosureSearchContext::new(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
    )?;
    let n_usize = ctx.n;

    // Setup channels for streaming results
    let (tx_results, rx_results) = channel::<Vec<(Vec<u32>, f64)>>();
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
    let combinations = generate_initial_combinations(ctx.scale_range(), ctx.seed_depth());
    let initial_combo_total = combinations.len();

    // The value grid every result in this run is indexed on.
    let grid = <Closure as SampleFormat<U>>::value_grid(scale_min, scale_max, items);
    let grid_len = grid.len();

    // Shared state for tracking min/max horns frequencies
    let freq_state = Arc::new(Mutex::new(StreamingFrequencyState::new(grid_len, n_usize)));
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

    // Spawn the shared writer thread. It owns the output format; this path
    // only feeds it (counts, horns) pairs.
    let writer_handle = <Closure as SampleFormat<U>>::spawn_streaming_writer(
        base_path.clone(),
        grid.clone(),
        n_usize,
        config.batch_size,
        config.format,
        config.show_progress.then_some((100_000, "combinations")),
        rx_results,
        writer_failed_for_thread,
    );

    // Spawn statistics collector thread
    let freq_state_for_stats = freq_state.clone();
    let stats_handle = thread::spawn(move || {
        let mut all_horns = Vec::new();

        while let Ok((horns_batch, _)) = rx_stats.recv() {
            all_horns.extend(horns_batch);
        }

        (all_horns, freq_state_for_stats)
    });

    // Fast path for small stop_after limits: use sequential processing
    if let Some(limit) = stop_after {
        if limit <= 100 {
            // Sequential processing for small limits
            let mut freqs: Vec<f64> = Vec::with_capacity(grid_len);
            let mut found_count = 0;

            for combo in &combinations {
                if found_count >= limit {
                    break;
                }

                let branch_results = closure_branch(combo, &ctx, Some(limit - found_count));

                for counts in branch_results.chunks_exact(grid_len) {
                    if found_count >= limit {
                        break;
                    }

                    // The counts vector already is the frequency vector, so
                    // there is nothing to tabulate here.
                    freqs.clear();
                    freqs.extend(counts.iter().map(|&c| c as f64));
                    let horns = horns_from_counts(&freqs);

                    freq_state_for_thread.lock().unwrap().update(counts, horns);

                    // Send to writer and stats
                    if tx_results.send(vec![(counts.to_vec(), horns)]).is_ok() {
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
                &grid,
                final_freq_state,
                // A truncated search saw only part of the space.
                stop_after.is_none(),
            );

            return Ok(StreamingResult {
                total_combinations: total_written,
                file_path: config.file_path,
            });
        }
    }

    // Process combinations in parallel (original path for unlimited or large limits)
    combinations.par_iter().for_each(|combo| {
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

        let mut branch_results = closure_branch(combo, &ctx, remaining);
        if branch_results.is_empty() || writer_failed_for_compute.load(Ordering::Relaxed) == 1 {
            return;
        }

        // Claim this branch's share of the limit before touching any
        // statistics, so the running statistics only ever see samples
        // that are also written.
        let found = branch_results.len() / grid_len;
        let current_count = counter_for_thread.fetch_add(found, Ordering::Relaxed);
        if let Some(limit) = stop_after {
            if current_count >= limit {
                return;
            }
            branch_results.truncate((limit - current_count).min(found) * grid_len);
        }

        let mut freqs: Vec<f64> = Vec::with_capacity(grid_len);
        let results_with_horns: Vec<(Vec<u32>, f64)> = branch_results
            .chunks_exact(grid_len)
            .map(|counts| {
                // The counts vector already is the frequency vector.
                freqs.clear();
                freqs.extend(counts.iter().map(|&c| c as f64));
                let horns = horns_from_counts(&freqs);
                (counts.to_vec(), horns)
            })
            .collect();
        let horns_batch: Vec<f64> = results_with_horns.iter().map(|(_, h)| *h).collect();

        // Accumulate locally and take the shared lock once per branch; the
        // per-sample shape classification is the expensive part.
        let mut local = StreamingFrequencyState::new(grid_len, n_usize);
        for (counts, horns) in &results_with_horns {
            local.update(counts, *horns);
        }
        freq_state_for_thread.lock().unwrap().merge(local);

        // A closed channel means the writer failed; it has already
        // reported that.
        let _ = tx_results.send(results_with_horns);
        let _ = tx_stats.send((horns_batch, HashMap::new()));
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

    // Written even when nothing was found, so a reader always finds a
    // complete result set, as the sequential path above already guarantees.
    write_streaming_statistics(
        &base_path,
        &all_horns,
        n_usize,
        &grid,
        final_freq_state,
        stop_after.is_none(),
    );

    Ok(StreamingResult {
        total_combinations: total_written,
        file_path: config.file_path,
    })
}

/// Collect every valid sample that extends one seed combination.
///
/// Samples are carried as a count per scale value rather than as an expanded
/// list of values. The DFS only ever appends values in non-decreasing order, so
/// the two encode the same thing; counts just make that explicit and shrink the
/// clone at every emitted leaf from `n` values to `k`.
///
/// The search is exact: it tracks an integer running sum and sum of squares and
/// compares them against the integer thresholds in [`ClosureSearchContext`],
/// so no floating-point error accumulates along a branch.
fn closure_branch(
    seed: &[usize],
    ctx: &ClosureSearchContext,
    stop_after: Option<usize>,
) -> Vec<u32> {
    let mut results = Vec::new();
    let mut counts = vec![0u32; ctx.scale_range()];
    let mut sum = 0i64;
    let mut ssq = 0i64;
    for &idx in seed {
        counts[idx] += 1;
        let v = ctx.scale_min + idx as i64;
        sum += v;
        ssq += v * v;
    }
    let min_idx = seed.last().copied().unwrap_or(0);
    // The limit in buffer elements, so the hot loop compares lengths directly.
    let limit = stop_after.map_or(usize::MAX, |limit| limit.saturating_mul(counts.len()));
    closure_branch_recurse(
        &mut counts,
        seed.len(),
        sum,
        ssq,
        min_idx,
        ctx,
        limit,
        &mut results,
    );
    results
}

/// Recursive backtracking core for [`closure_branch`].
///
/// Extends the sample one value at a time by incrementing a count on entry and
/// decrementing it on exit, reusing a single allocation across the entire
/// search tree. `k` is how many values the counts stand for.
#[allow(clippy::too_many_arguments)]
fn closure_branch_recurse(
    counts: &mut [u32],
    k: usize,
    sum: i64,
    ssq: i64,
    min_idx: usize,
    ctx: &ClosureSearchContext,
    limit: usize,
    results: &mut Vec<u32>,
) {
    if k >= ctx.n {
        // Every bound is checked here rather than relying on the pruning
        // below, which a seed as long as `n` never passes through.
        let m2n = ctx.n as i64 * ssq - sum * sum;
        if sum >= ctx.sum_lo && sum <= ctx.sum_hi && m2n >= ctx.m2n_lo && m2n <= ctx.m2_hi[ctx.n] {
            results.extend_from_slice(counts);
        }
        return;
    }

    let n_left = (ctx.n - k - 1) as i64;
    let next_k = (k + 1) as i64;
    let m2_hi = ctx.m2_hi[k + 1];

    for idx in min_idx..counts.len() {
        let v = ctx.scale_min + idx as i64;
        let next_sum = sum + v;

        // Remaining values are at least `v`, so the smallest reachable sum
        // only grows from here.
        if next_sum + v * n_left > ctx.sum_hi {
            break;
        }
        if next_sum + ctx.scale_max * n_left < ctx.sum_lo {
            continue;
        }

        let next_ssq = ssq + v * v;
        if next_k * next_ssq - next_sum * next_sum > m2_hi {
            // The partial M2 is a parabola in `v` with its minimum at the
            // running mean, so once `v` is past the mean every larger value
            // overshoots too.
            if v * k as i64 >= sum {
                break;
            }
            continue;
        }

        counts[idx] += 1;
        closure_branch_recurse(counts, k + 1, next_sum, next_ssq, idx, ctx, limit, results);
        counts[idx] -= 1;

        if results.len() >= limit {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    /// Column names of a Parquet file, in order.
    fn parquet_columns(path: &str) -> Vec<String> {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        builder
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    /// Total number of rows in a Parquet file.
    fn parquet_rows(path: &str) -> usize {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        reader.map(|b| b.unwrap().num_rows()).sum()
    }

    /// The medoid straight from the definition: minimise summed EMD to every
    /// other member, breaking ties lexicographically.
    fn medoid_naive(rows: &[Vec<u32>]) -> Option<Vec<u32>> {
        rows.iter()
            .map(|x| {
                let cost: u64 = rows.iter().map(|y| emd_1d(x, y)).sum();
                (cost, x.clone())
            })
            .min()
            .map(|(_, row)| row)
    }

    #[test]
    fn the_fast_medoid_matches_the_definition() {
        // `medoid_of` avoids the quadratic pairwise loop by decomposing the
        // summed EMD per grid position. Check the shortcut against the thing it
        // is a shortcut for, on real result sets.
        for (mean, sd, n) in [(3.0, 1.13, 60), (3.5, 1.0, 52), (2.2, 1.3, 40)] {
            let results =
                closure_parallel(mean, sd, n, 1i32, 5i32, 0.005, 0.005, 1, None, None).unwrap();
            let rows: Vec<Vec<u32>> = results.results.counts.rows().map(|r| r.to_vec()).collect();
            if rows.is_empty() {
                continue;
            }
            let fast = medoid_of(rows.iter().map(|r| r.as_slice()), 5, n as usize);
            assert_eq!(
                fast,
                medoid_naive(&rows),
                "fast medoid differs from the definition at mean={mean} sd={sd} n={n}"
            );
        }
    }

    #[test]
    fn the_medoid_does_not_depend_on_enumeration_order() {
        // Ties are broken lexicographically rather than by whichever candidate
        // the search happened to reach first, so a reshuffled input gives the
        // same answer. Enumeration order is not reproducible under `stop_after`
        // or for SPRITE, so this is what keeps the reported sample stable.
        let rows: Vec<Vec<u32>> = vec![
            vec![2, 0, 0, 0, 2],
            vec![0, 2, 0, 2, 0],
            vec![2, 0, 0, 0, 2],
            vec![0, 0, 4, 0, 0],
            vec![1, 1, 0, 1, 1],
        ];
        let forward = medoid_of(rows.iter().map(|r| r.as_slice()), 5, 4);
        let mut reversed = rows.clone();
        reversed.reverse();
        let backward = medoid_of(reversed.iter().map(|r| r.as_slice()), 5, 4);
        assert_eq!(forward, backward);
        assert_eq!(forward, medoid_naive(&rows));
    }

    #[test]
    fn modality_tables_reach_disk_with_conditional_bounds() {
        let dir = std::env::temp_dir().join("closure_modality_parquet");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = format!("{}/", dir.display());

        let results =
            closure_parallel(3.5, 1.5, 100, 1i32, 5i32, 0.01, 0.01, 1, None, None).unwrap();
        write_statistics_files(&base, &results);

        assert_eq!(
            parquet_columns(&format!("{base}frequency.parquet")),
            vec![
                "samples",
                "value",
                "f_expected",
                "f_representative",
                "f_relative"
            ]
        );
        assert_eq!(
            parquet_columns(&format!("{base}modality_shapes.parquet")),
            vec!["class", "n_samples", "value", "count_lo", "count_hi"]
        );
        // One row per (populated class, value); three of the six classes are
        // empty for this target, so 3 * 5 rows.
        assert_eq!(parquet_rows(&format!("{base}modality_shapes.parquet")), 15);
        assert_eq!(parquet_rows(&format!("{base}modality_counts.parquet")), 5);
        assert_eq!(parquet_rows(&format!("{base}modality_pairs.parquet")), 4);
        assert_eq!(parquet_rows(&format!("{base}modality_summary.parquet")), 1);

        // The prominence envelope behind every `Some(false)`: one row per
        // (threshold, class), so a reader can see how far the threshold would
        // have to move before an empty class stopped being empty.
        assert_eq!(
            parquet_columns(&format!("{base}modality_prominence.parquet")),
            vec![
                "min_prominence",
                "min_prominence_counts",
                "primary",
                "class",
                "n_samples"
            ]
        );
        assert_eq!(
            parquet_rows(&format!("{base}modality_prominence.parquet")),
            modality::DEFAULT_PROMINENCE_LADDER.len() * ShapeClass::all().count()
        );

        let summary = parquet_columns(&format!("{base}modality_summary.parquet"));
        assert!(summary.contains(&"exhaustive".to_string()));
        assert!(summary.contains(&"n_one_mode_interior".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn counts_are_a_lossless_encoding_of_closure_samples() {
        let results =
            closure_parallel::<f64, i32>(3.0, 1.0, 5, 1, 5, 0.05, 0.05, 1, None, None).unwrap();
        assert!(!results.results.is_empty());

        for i in 0..results.results.len() {
            let sample = results.results.sample(i);
            let counts = results.results.counts.row(i);

            // Expanding counts gives a sorted sample of the right size.
            assert_eq!(sample.len(), 5, "sample {i} has the wrong size");
            assert!(
                sample.windows(2).all(|w| w[0] <= w[1]),
                "sample {i} is not sorted: {sample:?}"
            );
            assert_eq!(counts.iter().sum::<u32>(), 5);

            // Re-tabulating the expanded sample returns the same counts, so the
            // map between the two representations is a bijection.
            let regrid = results.results.counts.grid().counts_of_sample(&sample);
            assert_eq!(regrid, counts, "round trip changed sample {i}");

            // And the sample still satisfies the search constraints.
            let mean = sample.iter().sum::<i32>() as f64 / 5.0;
            assert!((mean - 3.0).abs() <= 0.05, "sample {i} mean is {mean}");
        }
    }

    #[test]
    fn closure_and_sprite_write_the_same_layout() {
        let closure_dir = "test_format_closure";
        let sprite_dir = "test_format_sprite";
        let _ = std::fs::create_dir(closure_dir);
        let _ = std::fs::create_dir(sprite_dir);

        closure_parallel::<f64, i32>(
            3.0,
            1.0,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            Some(ParquetConfig::new(format!("{closure_dir}/"), 100)),
            Some(20),
        )
        .unwrap();

        sprite_parallel::<f64, i32>(
            3.0,
            1.0,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            RestrictionsOption::Default,
            Some(ParquetConfig::new(format!("{sprite_dir}/"), 100)),
            Some(20),
        )
        .unwrap();

        // Same files, and — because both techniques run on the same value grid
        // — the same count columns in the same order.
        for name in &[
            "counts",
            "scale_values",
            "format",
            "metrics_main",
            "metrics_horns",
            "frequency",
            "frequency_dist",
        ] {
            let closure_cols = parquet_columns(&format!("{closure_dir}/{name}.parquet"));
            let sprite_cols = parquet_columns(&format!("{sprite_dir}/{name}.parquet"));
            assert_eq!(
                closure_cols, sprite_cols,
                "{name}.parquet differs between the two techniques"
            );
        }

        let cols = parquet_columns(&format!("{closure_dir}/counts.parquet"));
        assert_eq!(cols, vec!["v1", "v2", "v3", "v4", "v5", "horns"]);

        let _ = std::fs::remove_dir_all(closure_dir);
        let _ = std::fs::remove_dir_all(sprite_dir);
    }

    #[test]
    fn sprite_tabulates_a_multi_item_grid() {
        // A five-item 1-5 scale: 21 grid values spaced 0.2 apart.
        let results = sprite_parallel::<f64, i32>(
            3.0,
            1.0,
            20,
            1,
            5,
            0.05,
            0.05,
            5,
            None,
            RestrictionsOption::Null,
            None,
            Some(5),
        )
        .unwrap();

        let grid = results.results.counts.grid();
        assert_eq!(grid.len(), 21);
        assert_eq!(grid.column_names()[1], "v1_2");
        assert_eq!(results.frequency.value().len(), 21 * SampleCategory::COUNT);

        // Every sample is fully accounted for on the grid, with nothing falling
        // off it — the failure the old integer binning would have produced.
        for i in 0..results.results.len() {
            assert_eq!(results.results.counts.row(i).iter().sum::<u32>(), 20);
        }
    }

    #[test]
    fn both_layouts_describe_the_same_samples() {
        let dir = "test_format_both";
        let _ = std::fs::create_dir(dir);

        let results = closure_parallel::<f64, i32>(
            3.0,
            1.0,
            5,
            1,
            5,
            0.05,
            0.05,
            1,
            Some(ParquetConfig {
                file_path: format!("{dir}/"),
                batch_size: 100,
                format: OutputFormat::Both,
            }),
            None,
        )
        .unwrap();

        let expected = results.results.len();
        assert_eq!(parquet_rows(&format!("{dir}/counts.parquet")), expected);
        assert_eq!(parquet_rows(&format!("{dir}/results.parquet")), expected);
        assert_eq!(parquet_rows(&format!("{dir}/scale_values.parquet")), 5);
        assert_eq!(parquet_rows(&format!("{dir}/format.parquet")), 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn horns_is_invariant_to_the_grid_labelling() {
        // Horns normalises by the scale's maximum variance, so labelling the
        // grid positions differently cannot change it. That equivalence is what
        // lets one horns implementation serve both a 1-7 CLOSURE run and a
        // multi-item SPRITE grid whose values are fractional.
        fn horns_at(values: &[f64], freqs: &[f64]) -> f64 {
            let total: f64 = freqs.iter().sum();
            let mean: f64 = values.iter().zip(freqs).map(|(v, f)| v * f / total).sum();
            let numerator: f64 = values
                .iter()
                .zip(freqs)
                .map(|(v, f)| (f / total) * (v - mean).powi(2))
                .sum();
            let span = values[values.len() - 1] - values[0];
            numerator / (span.powi(2) / 4.0)
        }

        let freqs = [3.0, 1.0, 0.0, 4.0, 2.0];
        let reference = horns_at(&[1.0, 2.0, 3.0, 4.0, 5.0], &freqs);

        // Same five positions, three different labellings.
        assert!((horns_from_counts(&freqs) - reference).abs() < 1e-12);
        assert!((calculate_horns(&freqs, 1, 5) - reference).abs() < 1e-12);
        assert!((horns_at(&[100.0, 200.0, 300.0, 400.0, 500.0], &freqs) - reference).abs() < 1e-12);
        // A five-item grid: values 0.2 apart rather than 1 apart.
        assert!((horns_at(&[1.0, 1.2, 1.4, 1.6, 1.8], &freqs) - reference).abs() < 1e-12);

        assert!((horns_uniform(5) - calculate_horns(&[1.0; 5], 1, 5)).abs() < 1e-12);
    }

    #[test]
    fn test_count_initial_combinations() {
        // Depth 2: C(3+1, 2) = 6, C(4+1, 2) = 10
        assert_eq!(count_initial_combinations(1, 3, 2), 6);
        assert_eq!(count_initial_combinations(1, 4, 2), 10);
        // Depth 3: C(3+2, 3) = 10, C(4+2, 3) = 20
        assert_eq!(count_initial_combinations(1, 3, 3), 10);
        assert_eq!(count_initial_combinations(1, 4, 3), 20);
        // Depth 1: C(range_size, 1) = range_size
        assert_eq!(count_initial_combinations(1, 7, 1), 7);
    }

    #[test]
    fn test_frequency_samples_column() {
        let repetitions = 5;

        // Test with 5 repetitions (like scale 1..=5)
        let samples = FrequencySamplesColumn::new(repetitions);

        // Check length -- currently, it should be 5 * 3 == 15
        assert_eq!(samples.len(), repetitions * SampleCategory::COUNT);
        assert!(!samples.is_empty());

        // Check repetitions
        assert_eq!(samples.repetitions(), repetitions);

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
    fn test_closure_parallel_with_new_api() {
        // Test that the function returns valid statistics with new structure
        let results = closure_parallel::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            1,    // items
            None, // no parquet config
            None, // no stop_after limit
        )
        .unwrap();

        // Check that results table is properly formed
        assert!(!results.results.is_empty());
        assert_eq!(results.results.len(), results.results.horns.len());
        assert_eq!(results.results.len(), results.results.id.len());
        assert_eq!(results.results.id[0], 1.0);
        assert_eq!(
            results.results.id.last(),
            Some(&(results.results.len() as f64))
        );

        // Check metrics
        assert_eq!(
            results.metrics_main.samples_all,
            results.results.len() as f64
        );
        assert_eq!(
            results.metrics_main.values_all,
            (results.results.len() * 5) as f64
        );

        // Check horns metrics
        assert!(results.metrics_horns.min <= results.metrics_horns.mean);
        assert!(results.metrics_horns.mean <= results.metrics_horns.max);
        assert!(results.metrics_horns.range >= 0.0);

        // Check combined frequency table
        let n_values = 5; // scale_max - scale_min + 1
        let expected_rows = n_values * SampleCategory::COUNT; // all, horns_min, horns_max
        assert_eq!(results.frequency.len(), expected_rows);
        assert_eq!(results.frequency.samples_group().len(), expected_rows);
        assert_eq!(results.frequency.value().len(), expected_rows);
        assert_eq!(results.frequency.f_expected().len(), expected_rows);
        assert_eq!(results.frequency.f_representative().len(), expected_rows);
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
    fn test_closure_parallel_with_file() {
        // Test with Parquet output
        let config = ParquetConfig {
            file_path: "test_output/".to_string(),
            batch_size: 100,
            format: OutputFormat::default(),
        };

        let _ = std::fs::create_dir("test_output");

        let results = closure_parallel::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            1,    // items
            Some(config),
            None, // no stop_after limit
        )
        .unwrap();

        assert!(!results.results.is_empty());

        // Verify parquet files are valid (more than just "PAR1" magic bytes)
        for name in &[
            "metrics_main",
            "metrics_horns",
            "frequency",
            "counts",
            "scale_values",
            "format",
        ] {
            let path = format!("test_output/{}.parquet", name);
            let size = std::fs::metadata(&path).unwrap().len();
            assert!(
                size > 4,
                "{}.parquet is only {} bytes (likely just PAR1 header)",
                name,
                size
            );
        }

        // The counts layout is the default, so no per-position file is written
        assert!(!std::path::Path::new("test_output/results.parquet").exists());

        // Clean up test files
        let _ = std::fs::remove_dir_all("test_output");
    }

    #[test]
    fn test_closure_parallel_streaming_separate_files() {
        // Test streaming mode with separate files
        let config = StreamingConfig {
            file_path: "test_streaming/".to_string(),
            batch_size: 100,
            show_progress: false,
            format: OutputFormat::default(),
        };

        let _ = std::fs::create_dir("test_streaming");

        let result = closure_parallel_streaming::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            1,    // items
            config, None, // no stop_after limit
        )
        .unwrap();

        assert!(result.total_combinations > 0);
        assert_eq!(result.file_path, "test_streaming/");

        // Verify parquet files are valid (more than just "PAR1" magic bytes)
        for name in &[
            "metrics_main",
            "metrics_horns",
            "frequency",
            "counts",
            "scale_values",
            "format",
        ] {
            let path = format!("test_streaming/{}.parquet", name);
            let size = std::fs::metadata(&path).unwrap().len();
            assert!(
                size > 4,
                "{}.parquet is only {} bytes (likely just PAR1 header)",
                name,
                size
            );
        }

        // Clean up test files
        let _ = std::fs::remove_dir_all("test_streaming");
    }

    #[test]
    fn test_streaming_with_stop_after_and_large_batch() {
        // Mimics the unsum R package scenario: stop_after=5, batch_size=1000
        let config = StreamingConfig {
            file_path: "test_streaming_unsum/".to_string(),
            batch_size: 1000,
            show_progress: false,
            format: OutputFormat::default(),
        };

        let _ = std::fs::create_dir("test_streaming_unsum");

        let result = closure_parallel_streaming::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            5,    // n
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            1,    // items
            config,
            Some(5), // stop_after = 5
        )
        .unwrap();

        assert!(result.total_combinations > 0);

        // Verify all parquet files are valid
        for name in &[
            "metrics_main",
            "metrics_horns",
            "frequency",
            "counts",
            "scale_values",
            "format",
        ] {
            let path = format!("test_streaming_unsum/{}.parquet", name);
            let size = std::fs::metadata(&path).unwrap().len();
            assert!(
                size > 4,
                "{}.parquet is only {} bytes (likely just PAR1 header)",
                name,
                size
            );
        }

        // Clean up
        let _ = std::fs::remove_dir_all("test_streaming_unsum");
    }

    #[test]
    fn test_streaming_large_n_no_stop_after() {
        // Tests the main parallel path (not the small-limit sequential path)
        let config = StreamingConfig {
            file_path: "test_streaming_large/".to_string(),
            batch_size: 1000,
            show_progress: false,
            format: OutputFormat::default(),
        };

        let _ = std::fs::create_dir("test_streaming_large");

        let result = closure_parallel_streaming::<f64, i32>(
            3.5,  // mean
            0.5,  // sd
            10,   // n (large enough for parallel path)
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            1,    // items
            config, None, // no stop_after → main parallel path
        )
        .unwrap();

        assert!(result.total_combinations > 0);

        // Verify all parquet files are valid
        for name in &[
            "metrics_main",
            "metrics_horns",
            "frequency",
            "counts",
            "scale_values",
            "format",
        ] {
            let path = format!("test_streaming_large/{}.parquet", name);
            let size = std::fs::metadata(&path).unwrap().len();
            assert!(
                size > 4,
                "{}.parquet is only {} bytes (likely just PAR1 header)",
                name,
                size
            );
        }

        // Clean up
        let _ = std::fs::remove_dir_all("test_streaming_large");
    }

    #[test]
    fn test_stop_after_parameter() {
        // Test that stop_after limits the number of results
        let results_unlimited = closure_parallel::<f64, i32>(
            3.0,  // mean
            1.0,  // sd
            80,   // n (increased sample size)
            1,    // scale_min
            5,    // scale_max
            0.05, // rounding_error_mean
            0.05, // rounding_error_sd
            1,    // items
            None, // no parquet config
            None, // no stop_after limit
        )
        .unwrap();

        let total_samples = results_unlimited.results.len();
        assert!(total_samples > 10); // Should have many samples

        // Test with limit of 10
        let results_limited = closure_parallel::<f64, i32>(
            3.0,      // mean
            1.0,      // sd
            80,       // n (increased sample size)
            1,        // scale_min
            5,        // scale_max
            0.05,     // rounding_error_mean
            0.05,     // rounding_error_sd
            1,        // items
            None,     // no parquet config
            Some(10), // stop after 10 samples
        )
        .unwrap();

        assert_eq!(results_limited.results.len(), 10);
        assert_eq!(results_limited.results.horns.len(), 10);
        assert_eq!(results_limited.results.id.len(), 10);

        // Test with limit of 1
        let results_one = closure_parallel::<f64, i32>(
            3.0,     // mean
            1.0,     // sd
            80,      // n (increased sample size)
            1,       // scale_min
            5,       // scale_max
            0.05,    // rounding_error_mean
            0.05,    // rounding_error_sd
            1,       // items
            None,    // no parquet config
            Some(1), // stop after 1 sample
        )
        .unwrap();

        assert_eq!(results_one.results.len(), 1);
        assert_eq!(results_one.results.horns.len(), 1);
        assert_eq!(results_one.results.id.len(), 1);
    }
}
