# SPRITE Refactoring Plan: Align with CLOSURE Return Type

## Current State Analysis

### CLOSURE (src/lib.rs)

-   **Generic type `T`**: Floating-point type for calculations (bounds: `Float + FromPrimitive + Send + Sync`)
-   **Generic type `U`**: Integer type for actual values (bounds: `Integer + NumCast + ToPrimitive + Copy + Send + Sync`)
-   **Return type**: `dfs_parallel()` returns `Vec<Vec<U>>` - a collection of combinations where each combination is a vector of integer values
-   **Combination struct**: Stores `values: Vec<U>`, `running_sum: T`, `running_m2: T`

### SPRITE (src/sprite.rs)

-   **Concrete types**: Uses `f64` throughout, no generics
-   **Return type**: `find_possible_distributions()` returns `Vec<DistributionResult>` where `DistributionResult` contains:
    -   `values: Vec<f64>`
    -   `mean: f64`
    -   `sd: f64`
    -   `outcome: Outcome`
    -   `iterations: u32`
-   **Scale values**: Internally works with fractional values like 1.0, 1.25, 1.5, etc. (based on `n_items` parameter)
-   **Scaling for hashing**: Already scales values to integers (×1,000,000) for uniqueness checks at line 618-622

## Goal

Make SPRITE return `Vec<Vec<U>>` where `U` has the same meaning as in CLOSURE: the actual values in the distribution, and both techniques use consistent generic type parameters.

## Challenges

1.  **Value representation**: CLOSURE uses integer types for `U`, but SPRITE currently uses fractional `f64` values
2.  **Precision handling**: SPRITE values like 2.25, 3.5 need to be represented as integers
3.  **Backward compatibility**: Need to maintain existing SPRITE functionality while changing the API

## Proposed Solution

### Approach: Scale SPRITE Values to Integers

Convert SPRITE to internally represent all values as scaled integers, similar to how it already does for hashing. This allows `U` to maintain its `Integer` trait bound across both CLOSURE and SPRITE.

### Implementation Steps

#### 1. Define Common Trait Bounds

Create trait aliases in `src/lib.rs` for shared type constraints:

``` rust
// Trait alias for floating-point calculation type
pub trait FloatType: Float + FromPrimitive + Send + Sync {}
impl<T: Float + FromPrimitive + Send + Sync> FloatType for T {}

// Trait alias for value type (integers representing scaled values)
pub trait ValueType: Integer + NumCast + ToPrimitive + Copy + Send + Sync {}
impl<U: Integer + NumCast + ToPrimitive + Copy + Send + Sync> ValueType for U {}
```

#### 2. Refactor SPRITE Core Structures

**2.1. Update `SpriteParameters` to use generics:**

``` rust
#[derive(Debug)]
pub struct SpriteParameters<T, U>
where
    T: FloatType,
    U: ValueType,
{
    pub mean: T,
    pub sd: T,
    pub n_obs: u32,
    pub min_val_scaled: U,  // Scaled by precision factor
    pub max_val_scaled: U,  // Scaled by precision factor
    pub m_prec: i32,
    pub sd_prec: i32,
    pub n_items: u32,
    pub possible_values_scaled: Vec<U>,  // All values as scaled integers
    pub fixed_responses_scaled: Vec<U>,  // Fixed values as scaled integers
    pub n_fixed: usize,
    pub scale_factor: u32,  // 10^max(m_prec, sd_prec) for converting back
}
```

**2.2. Update `DistributionResult`:**

``` rust
#[derive(Debug, Clone)]
pub struct DistributionResult<T, U>
where
    T: FloatType,
    U: ValueType,
{
    pub outcome: Outcome,
    pub values_scaled: Vec<U>,  // Scaled integer values
    pub mean: T,
    pub sd: T,
    pub iterations: u32,
}
```

#### 3. Update Function Signatures

**3.1. Main entry point `find_possible_distributions`:**

Change from:

``` rust
pub fn find_possible_distributions(
    params: &SpriteParameters,
    n_distributions: usize,
    return_failures: bool,
    rng: &mut impl Rng,
) -> Vec<DistributionResult>
```

To:

``` rust
pub fn find_possible_distributions<T, U>(
    params: &SpriteParameters<T, U>,
    n_distributions: usize,
    return_failures: bool,
    rng: &mut impl Rng,
) -> Vec<Vec<U>>  // Returns only the values, matching CLOSURE API
where
    T: FloatType,
    U: ValueType,
```

**3.2. Helper functions:**

Add generic type parameters to: - `find_possible_distribution<T, U>()` - `adjust_mean<T, U>()` - `shift_values<T, U>()` - `set_parameters<T, U>()` (main parameter validation function)

#### 4. Internal Value Scaling

**4.1. Determine scale factor:**

``` rust
// In set_parameters
let precision = max(m_prec, sd_prec);
let scale_factor = 10_u32.pow(precision as u32);
```

**4.2. Scale all input values:**

-   Convert `min_val`, `max_val` from user input to scaled integers
-   Generate `possible_values_scaled` by multiplying by `scale_factor`
-   Convert restrictions to use scaled integer keys
-   Scale `fixed_responses` to integers

**4.3. Scale target statistics:**

When performing GRIM/GRIMMER tests and internal calculations: - Convert scaled mean/SD back to float for tests: `value_scaled as f64 / scale_factor as f64` - Perform calculations in float type `T` - Convert results back to scaled integers `U` when needed

#### 5. Update Algorithm Logic

**5.1. `adjust_mean` function:** - Work with `Vec<U>` (scaled integers) instead of `Vec<f64>` - Target mean converted to scaled integer: `(target_mean * scale_factor as f64) as U` - Increment/decrement by finding next/previous value in `possible_values_scaled`

**5.2. `shift_values` function:** - Operate on `Vec<U>` with scaled integer arithmetic - Calculate SD by converting to float: `(value_scaled as f64) / (scale_factor as f64)` - Return swapped indices, maintaining integer representation

**5.3. `find_possible_distribution` function:** - Initialize with random scaled integers from `possible_values_scaled` - All operations work on `Vec<U>` - On success, return just the `Vec<U>` (not the full `DistributionResult`)

#### 6. Return Type Transformation

**Main function change:**

``` rust
pub fn find_possible_distributions<T, U>(
    params: &SpriteParameters<T, U>,
    n_distributions: usize,
    return_failures: bool,
    rng: &mut impl Rng,
) -> Vec<Vec<U>>
where
    T: FloatType,
    U: ValueType,
{
    let mut results: Vec<Vec<U>> = Vec::new();
    // ... existing logic ...

    // When successful distribution found:
    if meets_criteria {
        results.push(values_scaled);  // Just push the Vec<U>
    }

    results
}
```

**Note**: If detailed results are needed (mean, SD, iterations), create a separate function:

``` rust
pub fn find_possible_distributions_detailed<T, U>(
    params: &SpriteParameters<T, U>,
    n_distributions: usize,
    return_failures: bool,
    rng: &mut impl Rng,
) -> Vec<DistributionResult<T, U>>
```

#### 7. Update Tests

**7.1. Update existing tests:** - Convert test assertions to work with scaled integers - Add conversion helpers for readability - Verify that scaled values, when converted back, match expected float values

**7.2. Add new tests:** - Test that `find_possible_distributions` returns `Vec<Vec<U>>` - Verify scaling/unscaling roundtrips correctly - Test with different precision levels - Ensure CLOSURE and SPRITE use compatible type parameters

#### 8. Public API Considerations

**8.1. Convenience functions:**

For users who want float values, provide conversion utilities:

``` rust
pub fn unscale_distribution<T, U>(scaled_values: &[U], scale_factor: u32) -> Vec<T>
where
    T: FloatType,
    U: ValueType,
{
    scaled_values
        .iter()
        .map(|&v| T::from(v).unwrap() / T::from(scale_factor).unwrap())
        .collect()
}

pub fn unscale_distributions<T, U>(
    scaled_distributions: Vec<Vec<U>>,
    scale_factor: u32
) -> Vec<Vec<T>>
where
    T: FloatType,
    U: ValueType,
{
    scaled_distributions
        .iter()
        .map(|dist| unscale_distribution(dist, scale_factor))
        .collect()
}
```

**8.2. Builder pattern (optional):**

Consider a builder pattern for constructing `SpriteParameters` that handles scaling automatically:

``` rust
pub struct SpriteParametersBuilder<T> {
    mean: T,
    sd: T,
    // ... other fields as floats
}

impl<T: FloatType> SpriteParametersBuilder<T> {
    pub fn build<U: ValueType>(self) -> Result<SpriteParameters<T, U>, ParameterError> {
        // Automatically handles scaling
    }
}
```

## Migration Path

1.  **Phase 1**: Add trait aliases to `lib.rs`
2.  **Phase 2**: Create new generic versions of SPRITE functions alongside existing ones
3.  **Phase 3**: Update internal logic to use scaled integers
4.  **Phase 4**: Add tests for generic SPRITE implementation
5.  **Phase 5**: Deprecate old concrete-type SPRITE functions (or remove if no external users)
6.  **Phase 6**: Update documentation and examples

## Testing Strategy

### Unit Tests

-   Test scaling/unscaling at various precisions
-   Verify `U` type works with `i32`, `i64`, `u32`, `u64`
-   Test edge cases (very large/small values, high precision)

### Integration Tests

-   Verify SPRITE distributions match statistical properties
-   Ensure CLOSURE and SPRITE can use the same `U` type (e.g., `i32`)
-   Test interoperability if distributions from both techniques need to be combined

### Regression Tests

-   Run existing SPRITE tests with new implementation
-   Verify numerical stability with scaling

## Benefits

1.  **Consistency**: Both CLOSURE and SPRITE return `Vec<Vec<U>>`
2.  **Type safety**: Generic types ensure compile-time correctness
3.  **Performance**: Integer operations are faster than floating-point
4.  **Precision**: Avoid floating-point rounding errors in comparisons
5.  **Interoperability**: Both techniques can work with the same value type

## Potential Issues and Mitigations

### Issue 1: Integer overflow with high precision

-   **Mitigation**: Document precision limits for each integer type, use `i64`/`u64` for high precision

### Issue 2: Loss of API simplicity

-   **Mitigation**: Provide convenience functions and good documentation with examples

### Issue 3: Breaking changes for existing users

-   **Mitigation**: If external users exist, use deprecation warnings; otherwise, just update

### Issue 4: Complexity in scaling logic

-   **Mitigation**: Encapsulate all scaling in helper functions, add comprehensive tests

## Open Questions

1.  Should `DistributionResult` be removed entirely, or kept as a separate detailed result type?
2.  What should be the default integer type for `U`? (`i32` or `i64`?)
3.  Should we support custom scale factors, or always derive from precision?
4.  Do we need to maintain backward compatibility with the current API?

## Next Steps

1.  Review and approve this plan
2.  Implement trait aliases
3.  Begin refactoring `SpriteParameters` struct
4.  Update one function at a time, maintaining tests
5.  Iterate and refine based on implementation discoveries