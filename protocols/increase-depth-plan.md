# Plan: Configurable parallelization depth

## Background

The Python diff "Better optimization by  Increasing the depth of parallelization (and made it an argument to parallel_dfs)" (https://github.com/larigaldie-n/CLOSURE-Python/commit/7db4aa6f716025399be2d31b7f7532644517c83f) introduces a configurable `depth` parameter
for generating initial combinations, replacing the hardcoded depth of 2 (pairs).
Deeper initial splits create more work items, improving load balancing for large
`n` values.

## Formula

```
depth = min(round(n / 10), 15, n - 1)
```

Capped at 15 to keep the number of initial combinations manageable:
`combinations_with_replacement(range_size, depth)` grows fast.

## Changes

### 1. Generalize `generate_initial_combinations` (line 1052)

**Current**: hardcoded depth 2, nested loops over `(i, j)` pairs.

**New**: accept a `depth: usize` parameter. Generate all
`combinations_with_replacement(scale_min..=scale_max, depth)` and compute
`running_sum` and `running_m2` for each.

Implementation approach — iterative combination generator:
- Write a helper that yields all sorted combinations of length `depth` from
  `scale_min..=scale_max` (equivalent to `itertools.combinations_with_replacement`).
- For each combination, compute `sum` and Welford M2 in a loop over its elements.
- Return `Vec<(Vec<U>, T, T)>` as before.

No external crate needed; the generator is straightforward with a recursive or
stack-based approach (or use the `itertools` crate if already a dependency).

### 2. Update `count_initial_combinations` (line 465)

**Current**: `range_size * (range_size + 1) / 2` — correct only for depth 2.

**New**: compute `C(range_size + depth - 1, depth)` (multiset coefficient).
This is the number of combinations with replacement of `range_size` items taken
`depth` at a time. Implement via a small loop to avoid overflow with
factorials.

The function signature changes to accept `depth`:
```rust
pub fn count_initial_combinations(scale_min: i32, scale_max: i32, depth: usize) -> i64
```

Update all call sites (streaming progress tracking, test harness, tests).

### 3. Compute `depth` in `closure_parallel` and `closure_parallel_streaming`

Add the depth calculation before calling `generate_initial_combinations`:

```rust
let depth = (n_usize / 10).clamp(2, 15.min(n_usize - 1));
```

Note: keep `depth >= 2` to maintain the minimum parallelism that currently
exists. Pass `depth` to `generate_initial_combinations`.

### 4. Update call sites

- `closure_parallel` (line 1136): pass `depth`.
- `closure_parallel_streaming` (line 1796): pass `depth`.
- `count_initial_combinations` call in streaming (line 1612): pass `depth`.
- `test-harness.rs` (line 46): pass `depth`.
- Unit test `test_count_initial_combinations` (line 2221): update expected
  values and add depth parameter.

### 5. Update tests

- Verify `count_initial_combinations(1, 3, 2) == 6` (unchanged).
- Add `count_initial_combinations(1, 3, 3) == 10` (C(3+2, 3) = 10).
- Verify that `generate_initial_combinations` with depth 2 produces the same
  output as the current implementation (regression test).
- Run existing integration tests to confirm no behavioral change for small `n`.

## Files touched

| File | What changes |
|------|-------------|
| `src/lib.rs` | `generate_initial_combinations`, `count_initial_combinations`, `closure_parallel`, `closure_parallel_streaming`, tests |
| `src/bin/test-harness.rs` | `count_initial_combinations` call site |

## Risks and mitigations

- **Too many initial combinations at high depth + wide scale**: capped at 15,
  and for typical scales (1..7, range_size=7) depth 15 gives C(21,15) = 54264
  work items — reasonable.
- **Small `n`**: the `.clamp(2, ...)` ensures depth never exceeds `n-1` or
  drops below 2.
