# SPRITE API Refactoring: Unification with CLOSURE

## Overview

This document summarizes the refactoring of the SPRITE API to match the CLOSURE API conventions, creating a unified and homogeneous interface for both distribution-finding algorithms.

**Date**: 2025-12-16
**Status**: Complete ✅
**Tests**: All 11 tests passing

## Motivation

The SPRITE and CLOSURE APIs had inconsistencies in parameter naming, ordering, and types. This refactoring ensures both algorithms present a consistent interface, making them feel like they were "made from one piece."

## Key Changes

### 1. Parameter Naming

| Old SPRITE | New SPRITE | CLOSURE | Rationale |
|-----------|-----------|---------|-----------|
| `n_obs: u32` | `n: U` | `n: U` | Generic integer type, consistent naming |
| `min_val: i32` | `scale_min: U` | `scale_min: U` | Generic type, clearer semantics |
| `max_val: i32` | `scale_max: U` | `scale_max: U` | Generic type, clearer semantics |
| `m_prec: Option<i32>` | `rounding_error_mean: T` | `rounding_error_mean: T` | Direct tolerance specification |
| `sd_prec: Option<i32>` | `rounding_error_sd: T` | `rounding_error_sd: T` | Direct tolerance specification |
| `n_distributions: usize` | `stop_after: Option<usize>` | `stop_after: Option<usize>` | Optional limit, consistent naming |

### 2. Parameter Order

**New unified order** (matching CLOSURE):
```rust
(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    // SPRITE-specific parameters
    n_items: u32,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    dont_test: bool,
    // Output configuration
    parquet_config/config: ...,
    stop_after: Option<usize>,
    // RNG (SPRITE only)
    rng: &mut impl Rng,
)
```

### 3. Type System Changes

#### Generic Integer Types
- **Before**: Fixed types (`u32` for n_obs, `i32` for min_val/max_val)
- **After**: Generic `U: IntegerType` for `n`, `scale_min`, `scale_max`
- **Benefit**: Works with any integer type (i32, i64, u32, etc.)

#### Internal Type Conversions
Added proper conversions throughout:
```rust
let n_u32 = U::to_u32(&n).unwrap();
let n_usize = U::to_usize(&n).unwrap();
let scale_min_i32 = U::to_i32(&scale_min).unwrap();
```

### 4. Precision ↔ Rounding Error Conversion

#### New Helper Functions

**`precision_from_rounding_error()`** (sprite.rs:379-386)
```rust
/// Converts rounding error to precision (decimal places)
/// Examples:
///   0.05 → 1 (precision 1)
///   0.005 → 2 (precision 2)
///   0.0005 → 3 (precision 3)
fn precision_from_rounding_error<T: FloatType>(rounding_error: T) -> i32 {
    let re_f64 = T::to_f64(&rounding_error).unwrap();
    if re_f64 <= 0.0 { return 0; }
    (-((re_f64 * 2.0).log10())).round() as i32
}
```

**`scale_factor_from_rounding_errors()`** (sprite.rs:401-409)
```rust
/// Calculates internal scale factor from rounding errors
/// scale_factor = 10^max(m_prec, sd_prec)
fn scale_factor_from_rounding_errors<T: FloatType>(
    rounding_error_mean: T,
    rounding_error_sd: T,
) -> u32
```

#### Conversion Table

| Precision | Rounding Error | Formula |
|-----------|----------------|---------|
| 0 | 0.5 | 10^(-0) / 2 |
| 1 | 0.05 | 10^(-1) / 2 |
| 2 | 0.005 | 10^(-2) / 2 |
| 3 | 0.0005 | 10^(-3) / 2 |
| 4 | 0.00005 | 10^(-4) / 2 |

### 5. Internal Struct Updates

**SpriteParams** (sprite.rs:51-71)
```rust
struct SpriteParams<T, U> {
    mean: T,
    sd: T,
    n: U,                      // Changed from n_obs: u32
    rounding_error_mean: T,    // New field (stored for reference)
    rounding_error_sd: T,      // New field (stored for reference)
    m_prec: i32,               // Internal: inferred from rounding_error_mean
    sd_prec: i32,              // Internal: inferred from rounding_error_sd
    scale_factor: u32,
    possible_values_scaled: Vec<U>,
    fixed_responses_scaled: Vec<U>,
    n_fixed: usize,
}
```

## Migration Guide

### For Users Migrating from Old API

#### Example 1: Basic Usage

**Before**:
```rust
let results = sprite_parallel(
    2.2_f64,     // mean
    1.3_f64,     // sd
    20,          // n_obs
    1,           // min_val
    5,           // max_val
    None,        // m_prec (auto-detected)
    None,        // sd_prec (auto-detected)
    1,           // n_items
    100,         // n_distributions
    None,
    RestrictionsOption::Default,
    false,
    None,
    &mut rng,
)?;
```

**After**:
```rust
let results = sprite_parallel(
    2.2_f64,     // mean
    1.3_f64,     // sd
    20,          // n (generic integer)
    1,           // scale_min (generic integer)
    5,           // scale_max (generic integer)
    0.05,        // rounding_error_mean (precision 1: 0.1^1/2)
    0.05,        // rounding_error_sd (precision 1: 0.1^1/2)
    1,           // n_items
    None,
    RestrictionsOption::Default,
    false,
    None,
    Some(100),   // stop_after (now optional)
    &mut rng,
)?;
```

#### Example 2: High Precision

**Before**:
```rust
sprite_parallel(
    26.281,      // mean
    14.6339,     // sd
    2000,        // n_obs
    1, 50,
    Some(3),     // m_prec: 3 decimal places
    Some(4),     // sd_prec: 4 decimal places
    1, 5000,
    None, RestrictionsOption::Default,
    true, None, &mut rng,
)?;
```

**After**:
```rust
// Convert precision to rounding errors
let rounding_error_mean = 0.1_f64.powi(3) / 2.0;  // 0.0005
let rounding_error_sd = 0.1_f64.powi(4) / 2.0;    // 0.00005

sprite_parallel(
    26.281,
    14.6339,
    2000,
    1, 50,
    rounding_error_mean,
    rounding_error_sd,
    1,
    None, RestrictionsOption::Default,
    true, None,
    Some(5000),  // stop_after
    &mut rng,
)?;
```

#### Example 3: Streaming Mode

**Before**:
```rust
sprite_parallel_streaming(
    2.2, 1.3,
    20,          // n_obs
    1, 5,        // min_val, max_val
    None, None,  // m_prec, sd_prec
    1,
    None, RestrictionsOption::Default,
    false,
    config,
    Some(10),
);
```

**After**:
```rust
sprite_parallel_streaming(
    2.2, 1.3,
    20,          // n (generic)
    1, 5,        // scale_min, scale_max (generic)
    0.05, 0.05,  // rounding_error_mean, rounding_error_sd
    1,
    None, RestrictionsOption::Default,
    false,
    config,
    Some(10),    // stop_after
);
```

### Quick Conversion Reference

```rust
// OLD → NEW parameter mapping
n_obs              → n
min_val            → scale_min
max_val            → scale_max
m_prec             → rounding_error_mean = 0.1^m_prec / 2
sd_prec            → rounding_error_sd = 0.1^sd_prec / 2
n_distributions    → Some(n_distributions)  // for stop_after
```

## Technical Details

### Files Modified

1. **src/sprite.rs**
   - Lines 51-71: Updated `SpriteParams` struct
   - Lines 367-410: Added helper functions
   - Lines 415-646: Refactored `build_sprite_params()`
   - Lines 687-758: Updated `sprite_parallel()` signature
   - Lines 884-1155: Updated `sprite_parallel_streaming()` signature
   - Lines 1479-1496: Updated `find_distribution_internal()`
   - Lines 2189-2395: Updated all tests

### Implementation Strategy

1. **Helper functions** calculate precision from rounding errors internally
2. **Type conversions** use trait methods: `U::to_i32()`, `U::to_u32()`, `U::to_usize()`
3. **Generic iteration** over scale ranges using integer arithmetic
4. **Internal compatibility** maintained: scale_factor still computed as `10^max(m_prec, sd_prec)`
5. **Two-scale system** preserved: internal scale_factor vs. output 100x scale

### Test Results

All 11 tests passing:
```
✓ sprite_test_mean
✓ sprite_test_sd
✓ sprite_test_big
✓ test_sprite_parallel_streaming
✓ test_sprite_parallel_with_parquet
✓ test_dfs_parallel_with_new_api
✓ test_dfs_parallel_streaming_separate_files
✓ test_dfs_parallel_with_file
✓ test_stop_after_parameter
✓ test_horns_calculation
✓ test_count_initial_combinations
```

## Benefits

### 1. Consistency
- Both algorithms now use identical parameter names, types, and order
- Users can switch between CLOSURE and SPRITE with minimal changes

### 2. Type Safety
- Generic types allow compile-time checking
- Works with any appropriate integer type (i32, i64, etc.)

### 3. Clarity
- Direct specification of tolerances via rounding errors
- `stop_after: Option<usize>` clearly indicates optional early termination

### 4. Maintainability
- Unified interface easier to document and maintain
- Reduces cognitive load when working with both algorithms

## Future Enhancements

### Potential Additions (Not Yet Implemented)

1. **DistributionFinder Trait**
   ```rust
   pub trait DistributionFinder<T, U, Config = ()> {
       fn find_distributions(...) -> ResultListFromMeanSdN<U>;
       fn find_distributions_streaming(...) -> StreamingResult;
       fn algorithm_name() -> &'static str;
   }
   ```

2. **SpriteConfig Struct**
   ```rust
   pub struct SpriteConfig {
       pub n_items: u32,
       pub restrictions_exact: Option<HashMap<i32, usize>>,
       pub restrictions_minimum: RestrictionsOption,
       pub dont_test: bool,
   }
   ```

3. **Trait Implementations**
   - `ClosureFinder` implementing `DistributionFinder<T, U, (), ()>`
   - `SpriteFinder` implementing `DistributionFinder<T, U, SpriteConfig, ()>`

These would further enforce API consistency and enable polymorphic usage.

## Compatibility Notes

### Breaking Changes

This is a **breaking change** for existing SPRITE users:
- Parameter names changed
- Parameter order changed
- Parameter types changed (u32/i32 → generic U)
- Precision specification changed (integer → floating-point rounding error)

### Recommended Migration Path

1. Update function calls to new signature
2. Convert precision values to rounding errors: `0.1^precision / 2`
3. Change `n_distributions` to `Some(n_distributions)` for `stop_after`
4. Ensure integer type inference works (may need explicit types in some cases)

## Conclusion

The SPRITE API has been successfully unified with CLOSURE, creating a consistent, type-safe, and maintainable interface. All tests pass, and the algorithms now feel like they were designed together from the start.

The refactoring maintains backward compatibility at the internal level while providing a cleaner, more consistent external API that aligns with Rust best practices and the existing CLOSURE conventions.
