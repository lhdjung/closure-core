# SPRITE Streaming Implementation Plan

## Overview

Add streaming Parquet file output to SPRITE, mirroring the CLOSURE implementation's streaming capabilities. This will enable memory-efficient processing of large SPRITE computations.

## Current State Analysis

### CLOSURE Implementation (src/lib.rs)

-   **In-memory API**: `dfs_parallel()` - Returns `ResultListFromMeanSdN<U>` with all data in memory
-   **Streaming API**: `dfs_parallel_streaming()` - Returns `StreamingResult` with minimal memory
-   **Shared structs**: `StreamingConfig`, `StreamingResult`, `ParquetConfig`
-   **Output files**:
    -   `samples.parquet` - Sample data (one row per sample)
    -   `horns.parquet` - Horns values
    -   `metrics_main.parquet` - Main statistics
    -   `metrics_horns.parquet` - Horns statistics
    -   `frequency.parquet` - Frequency distributions

### SPRITE Implementation (src/sprite.rs)

-   **Current API**: `sprite_parallel()` - Returns `Vec<Vec<U>>` (only the distributions)
-   **No streaming support**: All results held in memory
-   **No statistics**: Unlike CLOSURE, SPRITE doesn't compute horns/frequency metrics
-   **Different focus**: SPRITE finds distributions matching mean/SD constraints

## Key Differences to Address

### 1. Return Type Mismatch

-   **CLOSURE**: Returns rich `ResultListFromMeanSdN<U>` struct with metrics
-   **SPRITE**: Returns simple `Vec<Vec<U>>` without statistics
-   **Decision**: SPRITE streaming should return `StreamingResult` for API consistency

### 2. Statistics Computation

-   **CLOSURE**: Computes comprehensive horns and frequency statistics
-   **SPRITE**: Currently doesn't compute similar metrics
-   **Decision**: SPRITE streaming should write simpler statistics (mean, SD per distribution)

### 3. File Structure

-   **CLOSURE**: Multiple Parquet files for different metric types
-   **SPRITE**: Should write at minimum:
    -   `samples.parquet` - The distribution data (required)
    -   `statistics.parquet` - Per-distribution mean/SD (optional but useful)

## Implementation Plan

### Phase 1: Define Public API

#### 1.1 Add sprite_parallel_streaming() function

``` rust
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
    stop_after: Option<usize>,  // Limit number of distributions
) -> StreamingResult
where
    T: FloatType + Send + Sync,
    U: IntegerType + Send + Sync + 'static,
```

**Note**: This signature matches `dfs_parallel_streaming()` pattern with SPRITE-specific parameters.

#### 1.2 Optionally enhance sprite_parallel() with ParquetConfig

``` rust
pub fn sprite_parallel<T, U>(
    // ... existing parameters ...
    parquet_config: Option<ParquetConfig>,  // NEW: optional file output
    rng: &mut impl Rng,
) -> Result<Vec<Vec<U>>, ParameterError>
```

**Note**: This matches the `dfs_parallel()` pattern of optional file output alongside in-memory results.

### Phase 2: Core Architecture Changes

#### 2.1 Refactor find_distributions_internal()

Current implementation: - Collects all results in memory (`Arc<Mutex<Vec<Vec<U>>>>`) - Uses `HashSet` for uniqueness checking - Returns all distributions at end

New streaming implementation: - Use channels for streaming results to writer thread - Maintain uniqueness tracking but stream data immediately - Pattern: `(tx_results, rx_results) = channel::<Vec<(Vec<U>, Stats)>>()`

#### 2.2 Create dedicated writer thread

Similar to CLOSURE's writer thread:

``` rust
thread::spawn(move || {
    let mut samples_writer = create_samples_writer(&samples_path, n_obs)?;
    let mut stats_writer = create_sprite_stats_writer(&stats_path)?;

    // Buffer for batching
    let mut samples_buffer = Vec::with_capacity(config.batch_size);
    let mut stats_buffer = Vec::with_capacity(config.batch_size);

    while let Ok(batch) = rx_results.recv() {
        // Buffer and write when threshold reached
    }
})
```

### Phase 3: Parquet Schema Definition

#### 3.1 Samples file (reuse existing pattern)

-   Schema: `pos1, pos2, ..., posN` columns
-   Each row is one distribution
-   Reuse: `create_samples_writer()` and `samples_to_record_batch()`

#### 3.2 Statistics file (SPRITE-specific)

``` rust
fn create_sprite_stats_writer(file_path: &str) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
    let fields = vec![
        Field::new("id", DataType::Int32, false),
        Field::new("mean", DataType::Float64, false),
        Field::new("sd", DataType::Float64, false),
        Field::new("iterations", DataType::Int32, false),  // From search process
    ];
    // ... create writer
}
```

### Phase 4: Integration Points

#### 4.1 Modify find_distributions_internal() signature

``` rust
fn find_distributions_internal<T, U>(
    params: &SpriteParams<T, U>,
    n_distributions: usize,
    streaming_tx: Option<Sender<Vec<(Vec<U>, DistributionStats)>>>,  // NEW
    stop_after: Option<usize>,  // NEW
    _rng: &mut impl Rng,
) -> Vec<Vec<U>>  // Empty if streaming
```

#### 4.2 Create helper for per-distribution statistics

``` rust
struct DistributionStats {
    mean: f64,
    sd: f64,
    iterations: u32,
}

fn compute_distribution_stats<U>(
    distribution: &[U],
    scale_factor: u32,
) -> DistributionStats {
    let values_f64: Vec<f64> = distribution
        .iter()
        .map(|v| U::to_i64(v).unwrap() as f64 / scale_factor as f64)
        .collect();

    DistributionStats {
        mean: mean(&values_f64),
        sd: std_dev(&values_f64).unwrap_or(0.0),
        iterations: 0,  // Track from search if needed
    }
}
```

### Phase 5: Error Handling & Progress

#### 5.1 Add progress reporting

Following CLOSURE pattern:

``` rust
if config.show_progress && found_count % 1000 == 0 {
    eprintln!("Progress: {} distributions found...", found_count);
}
```

#### 5.2 Early termination

-   Use `AtomicUsize` counter for tracking found distributions
-   Check limit in parallel workers
-   Signal writer thread when complete

#### 5.3 Error propagation

-   Writer thread failures should stop compute threads
-   Use `Arc<AtomicBool>` for failure signaling
-   Return appropriate error in `StreamingResult`

### Phase 6: File Organization

#### 6.1 Path handling

Match CLOSURE pattern:

``` rust
let base_path = if config.file_path.ends_with('/') {
    config.file_path.clone()
} else {
    format!("{}_", config.file_path)
};

let samples_path = format!("{}samples.parquet", base_path);
let stats_path = format!("{}statistics.parquet", base_path);
```

#### 6.2 Optional files

-   `samples.parquet` - Always created
-   `statistics.parquet` - Optional, controlled by config flag?

### Phase 7: Testing Strategy

#### 7.1 Unit tests

``` rust
#[test]
fn test_sprite_parallel_streaming_basic() {
    // Small test case
    // Verify files created
    // Verify row counts match
}

#[test]
fn test_sprite_streaming_stop_after() {
    // Test early termination
    // Verify exactly N distributions written
}

#[test]
fn test_sprite_streaming_statistics() {
    // Verify mean/SD in stats file match computed values
}
```

#### 7.2 Integration tests

``` rust
#[test]
fn test_sprite_streaming_large_dataset() {
    // Test with n_distributions = 10000
    // Verify memory stays bounded
}
```

### Phase 8: Documentation

#### 8.1 Function documentation

-   Document when to use streaming vs in-memory
-   Explain file structure and schema
-   Provide example usage

#### 8.2 Update lib.rs documentation

-   Mention both CLOSURE and SPRITE support streaming
-   Cross-reference between implementations

## Implementation Order

1.  **Step 1**: Reuse existing infrastructure
    -   Verify `create_samples_writer()` works for SPRITE
    -   Test `samples_to_record_batch()` with SPRITE data
2.  **Step 2**: Create SPRITE-specific statistics writer
    -   Implement `create_sprite_stats_writer()`
    -   Implement `sprite_stats_to_record_batch()`
3.  **Step 3**: Implement `sprite_parallel_streaming()`
    -   Build parameter validation (reuse `build_sprite_params()`)
    -   Set up channels and threading
    -   Implement writer thread
    -   Wire up to compute engine
4.  **Step 4**: Modify `find_distributions_internal()`
    -   Add streaming channel parameter
    -   Send results through channel instead of/in addition to collecting
    -   Add early termination logic
5.  **Step 5**: Add optional Parquet to `sprite_parallel()`
    -   Add `parquet_config` parameter
    -   Write files after computation completes
6.  **Step 6**: Testing and refinement
    -   Add unit tests
    -   Add integration tests
    -   Performance testing
7.  **Step 7**: Documentation
    -   Function docs
    -   Module docs
    -   Examples

## Open Questions

1.  **Statistics granularity**: Should we track per-distribution iterations from the search process?
    -   **Answer**: Track if available, but not critical
2.  **File schema compatibility**: Should SPRITE use identical schema to CLOSURE where possible?
    -   **Answer**: Yes for `samples.parquet`, different for statistics
3.  **Uniqueness tracking in streaming**: How to prevent duplicates without holding all in memory?
    -   **Answer**: Accept this limitation in streaming mode, or use bloom filter for approximate deduplication
4.  **Progress reporting**: How detailed should it be?
    -   **Answer**: Match CLOSURE's progress reporting pattern
5.  **Batch size**: Should it differ from CLOSURE's defaults?
    -   **Answer**: Use same defaults, let user configure via `StreamingConfig`

## Code Reuse Opportunities

### Reuse from lib.rs (no changes needed)

-   `StreamingConfig` struct
-   `StreamingResult` struct
-   `create_samples_writer()`
-   `samples_to_record_batch()`
-   Path handling logic
-   Progress reporting patterns
-   Channel-based architecture

### SPRITE-specific (new code)

-   `create_sprite_stats_writer()`
-   `sprite_stats_to_record_batch()`
-   `compute_distribution_stats()`
-   Modified `find_distributions_internal()` with streaming support

## Estimated Changes

-   **New functions**: \~200-300 lines (streaming infrastructure)
-   **Modified functions**: \~100-150 lines (adapt existing functions)
-   **Tests**: \~150-200 lines
-   **Documentation**: \~50-100 lines
-   **Total**: \~500-750 lines of code

## Success Criteria

1.  [x] `sprite_parallel_streaming()` matches `dfs_parallel_streaming()` API pattern
2.  [x] Memory usage stays bounded regardless of `n_distributions`
3.  [x] Parquet files readable by standard tools (Python pandas, R arrow)
4.  [x] Statistics in separate file are accurate
5.  [x] Progress reporting works correctly
6.  [x] Early termination with `stop_after` works
7.  [x] All tests pass
8.  [x] Documentation complete

## Future Enhancements (Out of Scope)

-   SPRITE-specific metrics analogous to CLOSURE's horns index
-   Frequency distribution analysis for SPRITE
-   Combined CLOSURE+SPRITE analysis in single API
-   Streaming deduplication with bloom filters