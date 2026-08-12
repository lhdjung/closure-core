//! Frequency-table representation of reconstructed samples.
//!
//! Every technique in this crate emits *sorted* samples: CLOSURE by
//! construction (its DFS never steps back down the scale) and SPRITE by an
//! explicit sort before deduplication. A sorted sample over a known value
//! lattice is a unary encoding of its own frequency vector, so storing counts
//! per value instead of one integer per observation is lossless — the map
//! sorted sample <-> count vector is a bijection.
//!
//! This module holds the three pieces that representation needs:
//!
//! - [`ValueGrid`] — the lattice of values a technique can emit, derived
//!   identically for CLOSURE and SPRITE from `scale_min`, `scale_max` and
//!   `items`.
//! - [`SampleCounts`] — a rectangular table with one row per sample and one
//!   column per grid value.
//! - [`SampleFormat`] — the trait that pins the on-disk layout. Every method
//!   that touches the format is a provided method, so techniques cannot drift
//!   apart in how they write results.

use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, StringArray, UInt16Array, UInt32Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use num::NumCast;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use crate::{IntegerType, ParquetConfig, ResultListFromMeanSdN};

/// Version stamp written into `format.parquet`.
///
/// 1 = one row per sample with one column per *position* (`pos1`..`posN`), or a
/// list column of raw values. 2 = one row per sample with one column per
/// *scale value*, this module's format.
pub const OUTPUT_FORMAT_VERSION: i32 = 2;

/// Which sample layout to write to disk.
///
/// The counts layout is the default; the sample layout is kept so that
/// cross-validation against the Python implementation and older readers of
/// `unsum` keep working through the version bump.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Write `counts.parquet` only (one column per scale value).
    #[default]
    Counts,
    /// Write the version 1 layout only (one column per position).
    Samples,
    /// Write both layouts.
    Both,
}

impl OutputFormat {
    /// Whether the counts layout should be written.
    pub fn writes_counts(&self) -> bool {
        matches!(self, OutputFormat::Counts | OutputFormat::Both)
    }

    /// Whether the legacy per-position layout should be written.
    pub fn writes_samples(&self) -> bool {
        matches!(self, OutputFormat::Samples | OutputFormat::Both)
    }
}

/// The lattice of values a technique can emit.
///
/// Values are `scale_min + j / items` for `j` in `0..=(scale_max - scale_min) *
/// items`, which is exactly the grid SPRITE builds internally and, at
/// `items == 1`, exactly the consecutive-integer scale CLOSURE walks. Deriving
/// both from one place is what keeps the two techniques' output comparable.
///
/// Values are also carried in hundredths of a scale point as exact integers.
/// That is the unit SPRITE emits samples in, and it makes the value of every
/// column expressible without floating-point ambiguity.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueGrid {
    scale_min: i32,
    scale_max: i32,
    items: u32,
    sample_scale_factor: i32,
    values: Vec<f64>,
    values_hundredths: Vec<i32>,
    column_names: Vec<String>,
}

impl ValueGrid {
    /// Build the grid for a scale.
    ///
    /// `sample_scale_factor` converts one unit of an emitted sample value into
    /// hundredths of a scale point: 100 for a technique that emits whole scale
    /// points (CLOSURE), 1 for one that already emits hundredths (SPRITE).
    ///
    /// # Panics
    /// Panics if `items` is zero or `scale_max < scale_min`.
    pub fn new(scale_min: i32, scale_max: i32, items: u32, sample_scale_factor: i32) -> Self {
        assert!(items >= 1, "can't build a ValueGrid: `items` must be >= 1");
        assert!(
            scale_max >= scale_min,
            "can't build a ValueGrid: `scale_max` ({}) is below `scale_min` ({})",
            scale_max,
            scale_min
        );

        let k = (scale_max - scale_min) as usize * items as usize + 1;
        let mut values = Vec::with_capacity(k);
        let mut values_hundredths = Vec::with_capacity(k);
        let mut column_names = Vec::with_capacity(k);

        for j in 0..k {
            let value = scale_min as f64 + j as f64 / items as f64;
            let hundredths = (value * 100.0).round() as i32;
            values.push(value);
            values_hundredths.push(hundredths);
            column_names.push(column_name_for(hundredths));
        }

        Self {
            scale_min,
            scale_max,
            items,
            sample_scale_factor,
            values,
            values_hundredths,
            column_names,
        }
    }

    /// Number of grid values, i.e. the width of a count vector.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// A grid is never empty in practice; provided for lint parity with `len`.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// True value of each grid position, ascending.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Value of each grid position in hundredths of a scale point.
    pub fn values_hundredths(&self) -> &[i32] {
        &self.values_hundredths
    }

    /// Count-column name of each grid position.
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Lowest scale value.
    pub fn scale_min(&self) -> i32 {
        self.scale_min
    }

    /// Highest scale value.
    pub fn scale_max(&self) -> i32 {
        self.scale_max
    }

    /// Number of items averaged into each observation.
    pub fn items(&self) -> u32 {
        self.items
    }

    /// Multiplier from emitted sample units to hundredths of a scale point.
    pub fn sample_scale_factor(&self) -> i32 {
        self.sample_scale_factor
    }

    /// Grid position of a value given in hundredths of a scale point.
    ///
    /// The grid is equally spaced, so the position follows from arithmetic
    /// rather than a lookup, which also keeps it exact for grids whose values
    /// are not representable in hundredths (`items` of 3, say).
    pub fn index_of_hundredths(&self, hundredths: i32) -> Option<usize> {
        let offset = (hundredths as f64 / 100.0 - self.scale_min as f64) * self.items as f64;
        let idx = offset.round();
        if idx < 0.0 || idx >= self.len() as f64 {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// Grid position of a true (unscaled) value.
    pub fn index_of_value(&self, value: f64) -> Option<usize> {
        let idx = ((value - self.scale_min as f64) * self.items as f64).round();
        if idx < 0.0 || idx >= self.len() as f64 {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// Grid position of a value in the units a technique emits samples in.
    pub fn index_of_sample_value<U: IntegerType>(&self, value: U) -> Option<usize> {
        let hundredths = U::to_i32(&value)? * self.sample_scale_factor;
        self.index_of_hundredths(hundredths)
    }

    /// Tabulate a sample into a count vector of length [`ValueGrid::len`].
    ///
    /// Values off the grid are dropped; that cannot happen for a sample a
    /// technique in this crate produced, but a caller-supplied one is not
    /// guaranteed to be on-grid.
    pub fn counts_of_sample<U: IntegerType>(&self, sample: &[U]) -> Vec<u32> {
        let mut counts = vec![0u32; self.len()];
        for &value in sample {
            if let Some(idx) = self.index_of_sample_value(value) {
                counts[idx] += 1;
            }
        }
        counts
    }

    /// Expand a count vector back into the sorted sample it encodes, in the
    /// units the technique emits samples in.
    ///
    /// This is the inverse of [`ValueGrid::counts_of_sample`] for any sample a
    /// technique in this crate produced, since those are already sorted.
    pub fn expand<U: IntegerType>(&self, counts: &[u32]) -> Vec<U> {
        let total: u32 = counts.iter().sum();
        let mut sample = Vec::with_capacity(total as usize);
        for (idx, &count) in counts.iter().enumerate() {
            let emitted = self.values_hundredths[idx] / self.sample_scale_factor;
            let value: U =
                NumCast::from(emitted).expect("grid value outside the sample's integer type");
            for _ in 0..count {
                sample.push(value);
            }
        }
        sample
    }
}

/// Canonical count-column name for a value given in hundredths.
///
/// Whole values get `v3`; fractional ones spell the fraction after an
/// underscore, `v1_5` or `v1_33`; negatives take an `n`, `vn2`. The authoritative
/// mapping is always `scale_values.parquet` — these names exist so a column is
/// readable without it, not so anyone has to parse them.
fn column_name_for(hundredths: i32) -> String {
    let sign = if hundredths < 0 { "n" } else { "" };
    let abs = hundredths.unsigned_abs();
    let whole = abs / 100;
    let frac = abs % 100;
    if frac == 0 {
        format!("v{}{}", sign, whole)
    } else {
        let frac_str = format!("{:02}", frac);
        format!("v{}{}_{}", sign, whole, frac_str.trim_end_matches('0'))
    }
}

/// One row per sample, one column per grid value.
///
/// Counts are held in a single flat row-major buffer, so a table of `m`
/// samples costs `m * k` values rather than the `m * n` an expanded sample
/// matrix would.
#[derive(Clone, Debug)]
pub struct SampleCounts {
    grid: ValueGrid,
    n: usize,
    data: Vec<u32>,
}

impl SampleCounts {
    /// Create an empty table for samples of size `n` over `grid`.
    pub fn new(grid: ValueGrid, n: usize) -> Self {
        Self {
            grid,
            n,
            data: Vec::new(),
        }
    }

    /// Create an empty table with room for `rows` samples.
    pub fn with_capacity(grid: ValueGrid, n: usize, rows: usize) -> Self {
        let k = grid.len();
        Self {
            grid,
            n,
            data: Vec::with_capacity(rows * k),
        }
    }

    /// Build a table from already-tabulated rows.
    ///
    /// # Panics
    /// Panics if any row's width differs from the grid's.
    pub fn from_rows(grid: ValueGrid, n: usize, rows: Vec<Vec<u32>>) -> Self {
        let k = grid.len();
        let mut data = Vec::with_capacity(rows.len() * k);
        for row in rows {
            assert_eq!(
                row.len(),
                k,
                "can't build SampleCounts: row width ({}) doesn't match grid width ({})",
                row.len(),
                k
            );
            data.extend_from_slice(&row);
        }
        Self { grid, n, data }
    }

    /// Append an already-tabulated row.
    ///
    /// # Panics
    /// Panics if the row's width differs from the grid's.
    pub fn push_row(&mut self, row: &[u32]) {
        assert_eq!(
            row.len(),
            self.grid.len(),
            "can't push a row into SampleCounts: row width ({}) doesn't match grid width ({})",
            row.len(),
            self.grid.len()
        );
        self.data.extend_from_slice(row);
    }

    /// Tabulate a sample and append it.
    pub fn push_sample<U: IntegerType>(&mut self, sample: &[U]) {
        let row = self.grid.counts_of_sample(sample);
        self.data.extend_from_slice(&row);
    }

    /// Number of samples.
    pub fn nrow(&self) -> usize {
        self.data.len().checked_div(self.grid.len()).unwrap_or(0)
    }

    /// Whether the table holds no samples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of grid values, i.e. the number of count columns.
    pub fn k(&self) -> usize {
        self.grid.len()
    }

    /// Sample size.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The value grid these counts are indexed on.
    pub fn grid(&self) -> &ValueGrid {
        &self.grid
    }

    /// Count vector of sample `i`.
    pub fn row(&self, i: usize) -> &[u32] {
        let k = self.grid.len();
        &self.data[i * k..(i + 1) * k]
    }

    /// Iterate over count vectors.
    pub fn rows(&self) -> std::slice::ChunksExact<'_, u32> {
        self.data.chunks_exact(self.grid.len().max(1))
    }

    /// The flat row-major counts of samples `start..end`, laid out the way
    /// [`counts_record_batch`] expects them.
    pub fn row_range(&self, start: usize, end: usize) -> &[u32] {
        let k = self.grid.len();
        &self.data[start * k..end * k]
    }

    /// All counts as one flat row-major buffer.
    pub fn as_flat(&self) -> &[u32] {
        &self.data
    }

    /// Reconstruct sample `i` as a sorted vector of values.
    ///
    /// Exact: the count vector is a complete encoding of the sample, and every
    /// sample a technique in this crate produces is already sorted.
    pub fn sample_at<U: IntegerType>(&self, i: usize) -> Vec<U> {
        self.grid.expand(self.row(i))
    }

    /// Reconstruct every sample. Allocates the full `m * n` matrix the counts
    /// exist to avoid, so prefer [`SampleCounts::sample_at`] where possible.
    pub fn samples<U: IntegerType>(&self) -> Vec<Vec<U>> {
        (0..self.nrow()).map(|i| self.sample_at(i)).collect()
    }
}

/// Arrow type for a count column given the sample size.
///
/// Counts are bounded by `n`, so most result sets fit in a byte per value
/// against the four the position layout spends.
pub fn count_data_type(n: usize) -> DataType {
    if n < u8::MAX as usize {
        DataType::UInt8
    } else if n < u16::MAX as usize {
        DataType::UInt16
    } else {
        DataType::UInt32
    }
}

/// Schema of `counts.parquet`: one column per grid value, then `horns`.
///
/// There is deliberately no `id` column. A sample's id is its row number, and
/// writing it out costs eight bytes per sample — on a 235k-sample run that one
/// column outweighed all the counts put together.
pub fn counts_schema(grid: &ValueGrid, n: usize) -> Arc<Schema> {
    let count_type = count_data_type(n);
    let mut fields = Vec::with_capacity(grid.len() + 1);
    for name in grid.column_names() {
        fields.push(Field::new(name, count_type.clone(), false));
    }
    fields.push(Field::new("horns", DataType::Float64, false));
    Arc::new(Schema::new(fields))
}

/// Build one column array from a flat row-major count buffer.
fn count_column(data: &[u32], k: usize, col: usize, count_type: &DataType) -> ArrayRef {
    let values = data.iter().skip(col).step_by(k).copied();
    match count_type {
        DataType::UInt8 => Arc::new(UInt8Array::from_iter_values(values.map(|c| c as u8))),
        DataType::UInt16 => Arc::new(UInt16Array::from_iter_values(values.map(|c| c as u16))),
        _ => Arc::new(UInt32Array::from_iter_values(values)),
    }
}

/// Assemble a `counts.parquet` batch from a flat row-major count buffer.
///
/// `horns` must have one entry per row in `data`.
pub fn counts_record_batch(
    grid: &ValueGrid,
    n: usize,
    data: &[u32],
    horns: &[f64],
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let k = grid.len();
    let nrow = data.len().checked_div(k).unwrap_or(0);
    if horns.len() != nrow {
        return Err(format!(
            "counts batch is ragged: {} rows of counts, {} horns values",
            nrow,
            horns.len()
        )
        .into());
    }

    let count_type = count_data_type(n);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(k + 1);
    for col in 0..k {
        arrays.push(count_column(data, k, col, &count_type));
    }
    arrays.push(Arc::new(Float64Array::from(horns.to_vec())));

    RecordBatch::try_new(counts_schema(grid, n), arrays).map_err(|e| e.into())
}

/// Create the Parquet writer for `counts.parquet`.
pub fn create_counts_writer(
    file_path: &str,
    grid: &ValueGrid,
    n: usize,
) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
    let file = File::create(file_path)?;
    let props = WriterProperties::builder().build();
    Ok(ArrowWriter::try_new(
        file,
        counts_schema(grid, n),
        Some(props),
    )?)
}

/// Write `scale_values.parquet`: the authoritative mapping from count column to
/// scale value.
pub fn write_scale_values_parquet(
    file_path: &str,
    grid: &ValueGrid,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("index", DataType::Int32, false),
        Field::new("column", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("value_hundredths", DataType::Int32, false),
    ]));

    let index: Vec<i32> = (0..grid.len() as i32).collect();
    let columns: Vec<&str> = grid.column_names().iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(index)),
            Arc::new(StringArray::from(columns)),
            Arc::new(Float64Array::from(grid.values().to_vec())),
            Arc::new(Int32Array::from(grid.values_hundredths().to_vec())),
        ],
    )?;

    let mut writer = ArrowWriter::try_new(File::create(file_path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Write `format.parquet`: a single row describing the layout of the other
/// files, so a reader can tell version 2 from version 1 without guessing.
pub fn write_format_parquet(
    file_path: &str,
    technique: &str,
    grid: &ValueGrid,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("format", DataType::Utf8, false),
        Field::new("version", DataType::Int32, false),
        Field::new("technique", DataType::Utf8, false),
        Field::new("n", DataType::Int32, false),
        Field::new("k", DataType::Int32, false),
        Field::new("items", DataType::Int32, false),
        Field::new("scale_min", DataType::Int32, false),
        Field::new("scale_max", DataType::Int32, false),
        Field::new("sample_scale_factor", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["counts"])),
            Arc::new(Int32Array::from(vec![OUTPUT_FORMAT_VERSION])),
            Arc::new(StringArray::from(vec![technique])),
            Arc::new(Int32Array::from(vec![n as i32])),
            Arc::new(Int32Array::from(vec![grid.len() as i32])),
            Arc::new(Int32Array::from(vec![grid.items() as i32])),
            Arc::new(Int32Array::from(vec![grid.scale_min()])),
            Arc::new(Int32Array::from(vec![grid.scale_max()])),
            Arc::new(Int32Array::from(vec![grid.sample_scale_factor()])),
        ],
    )?;

    let mut writer = ArrowWriter::try_new(File::create(file_path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Batching writer for `counts.parquet`, used by the streaming paths.
///
/// Rows are buffered until `batch_size` of them have arrived, then flushed as
/// one record batch. Ids are assigned in write order, so they match the ids a
/// memory-mode run of the same search would produce.
pub struct CountsFileWriter {
    writer: ArrowWriter<File>,
    grid: ValueGrid,
    n: usize,
    batch_size: usize,
    buffer: Vec<u32>,
    horns: Vec<f64>,
    total_written: usize,
}

impl CountsFileWriter {
    /// Open `counts.parquet` under `base_path`.
    pub fn create(
        base_path: &str,
        grid: ValueGrid,
        n: usize,
        batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let writer = create_counts_writer(&format!("{}counts.parquet", base_path), &grid, n)?;
        let batch_size = batch_size.max(1);
        Ok(Self {
            writer,
            grid,
            n,
            batch_size,
            buffer: Vec::new(),
            horns: Vec::new(),
            total_written: 0,
        })
    }

    /// Queue one sample's counts and its horns value.
    pub fn push(&mut self, counts: &[u32], horns: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.buffer.extend_from_slice(counts);
        self.total_written += 1;
        self.horns.push(horns);
        if self.horns.len() >= self.batch_size {
            self.flush()?;
        }
        Ok(())
    }

    /// Write out whatever is buffered.
    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.horns.is_empty() {
            return Ok(());
        }
        let batch = counts_record_batch(&self.grid, self.n, &self.buffer, &self.horns)?;
        self.writer.write(&batch)?;
        self.buffer.clear();
        self.horns.clear();
        Ok(())
    }

    /// How many samples have been queued so far.
    pub fn total_written(&self) -> usize {
        self.total_written
    }

    /// Flush and close, returning the number of samples written.
    pub fn finish(mut self) -> Result<usize, Box<dyn std::error::Error>> {
        self.flush()?;
        let total = self.total_written;
        self.writer.close()?;
        Ok(total)
    }
}

/// Batching writer for the version 1 layout: `sample.parquet` with one column
/// per position, plus `horns.parquet`.
///
/// Kept so cross-validation against the Python implementation and readers that
/// predate the counts format keep working through the version bump.
struct LegacySampleWriter {
    samples: ArrowWriter<File>,
    horns_writer: ArrowWriter<File>,
    batch_size: usize,
    samples_buffer: Vec<Vec<i32>>,
    horns_buffer: Vec<f64>,
}

impl LegacySampleWriter {
    fn create(
        base_path: &str,
        n: usize,
        batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            samples: crate::create_samples_writer(&format!("{}sample.parquet", base_path), n)?,
            horns_writer: crate::create_horns_writer(&format!("{}horns.parquet", base_path))?,
            batch_size: batch_size.max(1),
            samples_buffer: Vec::new(),
            horns_buffer: Vec::new(),
        })
    }

    fn push(&mut self, sample: &[i32], horns: f64) -> Result<(), Box<dyn std::error::Error>> {
        self.samples_buffer.push(sample.to_vec());
        self.horns_buffer.push(horns);
        if self.samples_buffer.len() >= self.batch_size {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.samples_buffer.is_empty() {
            return Ok(());
        }
        self.samples
            .write(&crate::samples_to_record_batch(&self.samples_buffer)?)?;
        self.horns_writer
            .write(&crate::horns_to_record_batch(&self.horns_buffer)?)?;
        self.samples_buffer.clear();
        self.horns_buffer.clear();
        Ok(())
    }

    fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.flush()?;
        self.samples.close()?;
        self.horns_writer.close()?;
        Ok(())
    }
}

/// The output format shared by every reconstruction technique.
///
/// A technique supplies two constants and nothing else: every method that
/// decides what the results look like, in memory or on disk, is a provided
/// method here. That is deliberate — CLOSURE and SPRITE results are meant to be
/// byte-comparable for the same scale, and the only way to guarantee that is to
/// leave the techniques no say in the matter.
pub trait SampleFormat<U: IntegerType> {
    /// Name recorded in `format.parquet`.
    const TECHNIQUE: &'static str;

    /// Multiplier from one unit of an emitted sample value to hundredths of a
    /// scale point. CLOSURE emits whole scale points (100); SPRITE emits
    /// hundredths already (1).
    const SAMPLE_SCALE_FACTOR: i32;

    /// The value lattice for a set of parameters.
    fn value_grid(scale_min: U, scale_max: U, items: u32) -> ValueGrid {
        ValueGrid::new(
            U::to_i32(&scale_min).expect("scale_min doesn't fit in i32"),
            U::to_i32(&scale_max).expect("scale_max doesn't fit in i32"),
            items,
            Self::SAMPLE_SCALE_FACTOR,
        )
    }

    /// An empty counts table for a set of parameters.
    fn counts_table(scale_min: U, scale_max: U, items: u32, n: usize) -> SampleCounts {
        SampleCounts::new(Self::value_grid(scale_min, scale_max, items), n)
    }

    /// Write the two files that describe the layout: the column-to-value key
    /// and the format stamp.
    fn write_grid_files(base_path: &str, grid: &ValueGrid, n: usize) {
        let _ = write_scale_values_parquet(&format!("{}scale_values.parquet", base_path), grid);
        let _ = write_format_parquet(
            &format!("{}format.parquet", base_path),
            Self::TECHNIQUE,
            grid,
            n,
        );
    }

    /// Create the streaming writer for `counts.parquet`.
    fn counts_writer(
        base_path: &str,
        grid: &ValueGrid,
        n: usize,
    ) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
        create_counts_writer(&format!("{}counts.parquet", base_path), grid, n)
    }

    /// Assemble one `counts.parquet` batch.
    fn counts_batch(
        grid: &ValueGrid,
        n: usize,
        data: &[u32],
        horns: &[f64],
    ) -> Result<RecordBatch, Box<dyn std::error::Error>> {
        counts_record_batch(grid, n, data, horns)
    }

    /// Write a complete counts table in one pass: the grid files plus
    /// `counts.parquet` in batches of `batch_size` rows.
    fn write_counts(
        base_path: &str,
        counts: &SampleCounts,
        horns: &[f64],
        batch_size: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let grid = counts.grid();
        Self::write_grid_files(base_path, grid, counts.n());

        let mut writer = Self::counts_writer(base_path, grid, counts.n())?;
        let nrow = counts.nrow();
        let step = batch_size.max(1);

        for start in (0..nrow).step_by(step) {
            let end = (start + step).min(nrow);
            let batch = Self::counts_batch(
                grid,
                counts.n(),
                counts.row_range(start, end),
                &horns[start..end],
            )?;
            writer.write(&batch)?;
        }
        writer.close()?;
        Ok(())
    }

    /// Open a batching writer for `counts.parquet` and write the grid files
    /// that go with it. Used by the streaming paths.
    fn counts_file_writer(
        base_path: &str,
        grid: ValueGrid,
        n: usize,
        batch_size: usize,
    ) -> Result<CountsFileWriter, Box<dyn std::error::Error>> {
        Self::write_grid_files(base_path, &grid, n);
        CountsFileWriter::create(base_path, grid, n, batch_size)
    }

    /// Spawn the writer thread that drains `rx` to disk.
    ///
    /// The thread owns every decision about what the sample files look like,
    /// so a technique's streaming path only has to produce `(counts, horns)`
    /// pairs and let this drain them. Returns the join handle; the thread
    /// finishes when the sending half of `rx` is dropped and yields the number
    /// of samples written.
    #[allow(clippy::too_many_arguments)]
    fn spawn_streaming_writer(
        base_path: String,
        grid: ValueGrid,
        n: usize,
        batch_size: usize,
        format: OutputFormat,
        progress: Option<(usize, &'static str)>,
        rx: Receiver<Vec<(Vec<u32>, f64)>>,
        writer_failed: Arc<AtomicUsize>,
    ) -> JoinHandle<usize> {
        thread::spawn(move || {
            let mut counts_writer = if format.writes_counts() {
                match Self::counts_file_writer(&base_path, grid.clone(), n, batch_size) {
                    Ok(w) => Some(w),
                    Err(e) => {
                        eprintln!(
                            "ERROR: Failed to create counts writer under '{}': {}",
                            base_path, e
                        );
                        writer_failed.store(1, Ordering::Relaxed);
                        return 0;
                    }
                }
            } else {
                None
            };

            let mut legacy = if format.writes_samples() {
                match LegacySampleWriter::create(&base_path, n, batch_size) {
                    Ok(w) => Some(w),
                    Err(e) => {
                        eprintln!(
                            "ERROR: Failed to create sample writer under '{}': {}",
                            base_path, e
                        );
                        writer_failed.store(1, Ordering::Relaxed);
                        return 0;
                    }
                }
            } else {
                None
            };

            let mut total_written = 0usize;
            let mut last_progress_report = 0usize;

            while let Ok(batch) = rx.recv() {
                for (counts, horns) in batch {
                    if let Some(writer) = counts_writer.as_mut() {
                        if let Err(e) = writer.push(&counts, horns) {
                            eprintln!("ERROR: Failed to write counts batch: {}", e);
                            return total_written;
                        }
                    }
                    if let Some(writer) = legacy.as_mut() {
                        if let Err(e) = writer.push(&grid.expand::<i32>(&counts), horns) {
                            eprintln!("ERROR: Failed to write samples batch: {}", e);
                            return total_written;
                        }
                    }
                    total_written += 1;
                }

                if let Some((every, label)) = progress {
                    if total_written - last_progress_report >= every {
                        eprintln!("Progress: {} {} written...", total_written, label);
                        last_progress_report = total_written;
                    }
                }
            }

            if let Some(writer) = counts_writer {
                if let Err(e) = writer.finish() {
                    eprintln!("ERROR: Failed to close counts file: {}", e);
                }
            }
            if let Some(writer) = legacy {
                if let Err(e) = writer.finish() {
                    eprintln!("ERROR: Failed to close sample file: {}", e);
                }
            }

            if let Some((_, label)) = progress {
                eprintln!(
                    "Streaming complete: {} total {} written",
                    total_written, label
                );
            }

            total_written
        })
    }

    /// Write a complete in-memory result list to `base_path`.
    ///
    /// This is the whole of memory-mode output: the samples in whichever
    /// layouts `config.format` asks for, plus the statistics tables. Both
    /// techniques go through here, which is what makes their output
    /// directories interchangeable.
    fn write_result_list(
        base_path: &str,
        results: &ResultListFromMeanSdN<U>,
        config: &ParquetConfig,
    ) where
        U: 'static,
    {
        let counts = &results.results.counts;
        let batch_size = config.batch_size.max(1);

        if config.format.writes_counts() {
            let _ = Self::write_counts(base_path, counts, &results.results.horns, batch_size);
        }

        if config.format.writes_samples() {
            let results_path = format!("{}results.parquet", base_path);
            if let Ok(mut writer) = crate::create_results_writer(&results_path) {
                let total_samples = results.results.len();
                for start in (0..total_samples).step_by(batch_size) {
                    let end = (start + batch_size).min(total_samples);
                    if let Ok(batch) = crate::results_to_record_batch(&results.results, start, end)
                    {
                        let _ = writer.write(&batch);
                    }
                }
                let _ = writer.close();
            }
        }

        crate::write_statistics_files(base_path, results);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_closure_scale() {
        let grid = ValueGrid::new(1, 7, 1, 100);
        assert_eq!(grid.len(), 7);
        assert_eq!(
            grid.values_hundredths(),
            &[100, 200, 300, 400, 500, 600, 700]
        );
        assert_eq!(
            grid.column_names(),
            &["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
        );
    }

    #[test]
    fn grid_matches_multi_item_sprite_scale() {
        // items = 5 over a 1-7 scale: 31 values, spaced 0.2 apart.
        let grid = ValueGrid::new(1, 7, 5, 1);
        assert_eq!(grid.len(), 31);
        assert_eq!(grid.values_hundredths()[1], 120);
        assert_eq!(grid.column_names()[1], "v1_2");
        assert_eq!(grid.column_names()[30], "v7");
    }

    #[test]
    fn column_names_spell_out_fractions_and_signs() {
        assert_eq!(column_name_for(150), "v1_5");
        assert_eq!(column_name_for(133), "v1_33");
        assert_eq!(column_name_for(300), "v3");
        assert_eq!(column_name_for(-250), "vn2_5");
        assert_eq!(column_name_for(0), "v0");
    }

    #[test]
    fn tabulate_and_expand_round_trip() {
        let grid = ValueGrid::new(1, 5, 1, 100);
        let sample: Vec<i32> = vec![1, 2, 2, 3, 5, 5, 5];
        let counts = grid.counts_of_sample(&sample);
        assert_eq!(counts, vec![1, 2, 1, 0, 3]);
        let expanded: Vec<i32> = grid.expand(&counts);
        assert_eq!(expanded, sample);
    }

    #[test]
    fn sprite_hundredths_round_trip() {
        // SPRITE emits hundredths, so the grid indexes them directly.
        let grid = ValueGrid::new(1, 3, 2, 1);
        assert_eq!(grid.values_hundredths(), &[100, 150, 200, 250, 300]);
        let sample: Vec<i32> = vec![100, 150, 150, 300];
        let counts = grid.counts_of_sample(&sample);
        assert_eq!(counts, vec![1, 2, 0, 0, 1]);
        let expanded: Vec<i32> = grid.expand(&counts);
        assert_eq!(expanded, sample);
    }

    #[test]
    fn counts_table_reconstructs_samples() {
        let grid = ValueGrid::new(1, 4, 1, 100);
        let mut table = SampleCounts::new(grid, 5);
        table.push_sample(&[1i32, 1, 2, 3, 4]);
        table.push_sample(&[2i32, 2, 2, 2, 2]);
        assert_eq!(table.nrow(), 2);
        assert_eq!(table.row(0), &[2, 1, 1, 1]);
        assert_eq!(table.sample_at::<i32>(1), vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn count_dtype_narrows_with_n() {
        assert_eq!(count_data_type(60), DataType::UInt8);
        assert_eq!(count_data_type(1000), DataType::UInt16);
        assert_eq!(count_data_type(100_000), DataType::UInt32);
    }
}
