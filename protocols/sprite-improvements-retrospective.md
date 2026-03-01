# SPRITE Improvements — Implementation Retrospective

All seven items from `sprite-improvements.md` were implemented in `src/sprite.rs`.
The test suite (`sprite_test_mean`, `sprite_test_sd`, `sprite_test_big`,
`test_sprite_parallel_streaming`, `test_sprite_parallel_with_parquet`) passes after every
change, and the release-mode runtime for `sprite_test_big` (n = 2000, 5 000 distributions)
dropped from roughly 3.5 s to 2.3 s.

---

## Item 2 — Incremental Statistics (O(n) → O(1))

**Planned:** maintain `running_sum` / `running_sum_sq` so that the SD and mean in the
inner loops are recomputed in O(1) rather than O(n).

**Implemented as planned.**  Two new `#[inline]` helpers were added after `std_dev`:

```rust
fn mean_from_running(total_sum: i64, n: usize, scale_factor: u32) -> f64
fn sd_from_running(total_sum: i64, total_sum_sq: i64, n: usize, scale_factor: u32) -> f64
```

`sd_from_running` uses the computational formula
`Var = (n·Σx² − (Σx)²) / (n·(n−1)·s²)` to avoid a division by `scale_factor²` inside
each element access.

`find_distribution_internal` now precomputes `fixed_sum` / `fixed_sum_sq` once and
maintains `running_sum` / `running_sum_sq` for the mutable slice throughout the SD loop.
`shift_values_internal` receives these as `&mut i64` parameters and updates them on every
accepted swap.  When `adjust_mean_internal` fires (rare mean-drift path), the running
sums are re-synced with a full O(n) pass — acceptable since that path is rare.

`adjust_mean_internal` was similarly changed to maintain a `running_sum` and increment it
by `(new_val − old_val)` on each bump, removing the per-iteration `compute_mean_scaled`
call.

`compute_mean_scaled` and `compute_sd_scaled` were annotated `#[allow(dead_code)]` and
kept for debugging purposes.

---

## Item 4 — Constant Hoisting

**Planned:** move `target_mean_rounded` (which depends only on `params.mean` and
`params.m_prec`) out of the SD loop.

**Implemented as planned** — `target_mean_rounded` is now computed once before the loop
in `find_distribution_internal`.  The same hoisting was applied to `adjust_mean_internal`
(its `target_mean_rounded` was also previously recomputed every iteration).

---

## Item 3 — O(k) → O(1) Value Lookups

**Planned:** add a `HashMap<i64, usize>` to `SpriteParams` mapping each scaled value to
its index in `possible_values_scaled`.

**Implemented as planned.**  `value_to_index` is built once in `build_sprite_params`
immediately after `final_possible_values_scaled` is assembled, with zero extra traversal
cost (a single `.enumerate().map().collect()` pass over a slice that just been created).

In `adjust_mean_internal`, `iter().position()` (O(k)) was replaced with
`value_to_index.get()` (O(1)).

In `shift_values_internal`, both `iter().find()` and `iter().rev().find()` (O(k) each)
were replaced with index lookups: `value_to_index.get(&val)` → `pv[idx + 1]` /
`pv[idx − 1]`.  Having the index available also directly enables the gap-resolution
logic below.

---

## Item 1 — Gap Resolution in `shift_values_internal`

**Planned:** when `delta1 ≠ delta2` (step sizes differ due to `items > 1` or
`restrictions_exact`), implement the two-stage fallback from R's `.shift_values`.

**Implemented as planned** with one minor deviation in Stage 2.

### Stage 1

Scans consecutive pairs `(pv[p], pv[p+1])` in `possible_values_scaled` for a pair whose
gap equals `delta1`.  Skips the pair that would undo step 1 (`high == val1_i64`).  For
the first matching pair whose `high` member exists in `vec` at some position `k ≠ i`,
applies `vec[i] = val1_new` and `vec[k] = pv[p]` — net sum change is zero.  The
`is_pointless` check (local 2-element SD test) is applied using `(val1, high)` as the
"before" pair and `(val1_new, low)` as the "after" pair.

### Stage 2

Tries splitting `delta1` into `num_steps` equal sub-steps for `num_steps` from 2 to
`min(delta1 / min_step, 10)`.  For each candidate `num_steps`, finds all consecutive
pairs in `possible_values_scaled` with gap `= delta1 / num_steps`, then searches for
`num_steps` distinct positions in `vec` (all `≠ i`) whose values are the "high" end of
such a pair.  If all `num_steps` positions are found, applies all changes atomically (no
backtracking needed — positions were verified before modification).

**Deviation from plan:** the plan used `longest_run` (the longest run of restricted
values in the full scale) as the upper bound for `num_steps`.  That would require
reconstructing the full unfiltered scale, which is not available in `SpriteParams`.
Instead the bound is `delta1 / min_step` capped at 10, where `min_step` is the smallest
gap between any two consecutive entries in `possible_values_scaled`.  This is at least as
conservative and avoids the need for extra stored state.

**Deviation from plan:** Stage 2 omits the `is_pointless` check.  Computing the local SD
change for a multi-element swap is non-trivial and Stage 2 is a rare recovery path; the
outer SD loop will correct any unhelpful moves in subsequent iterations.

### Closure refactor

The repeated `if increase_sd { sd_after <= sd_before } else { sd_after >= sd_before }`
pattern was extracted into a local closure `is_pointless(a_before, a_after, b_before,
b_after)`, keeping both the equal-gap and Stage 1 acceptance checks readable.

---

## Item 5 — Mutex Contention

**Planned:** reduce lock contention in `find_distributions_all_internal`.  Option A
(thread-local accumulation) was recommended.

**Implemented using Option A variant:** instead of thread-local buffers that flush
periodically, the parallel and serial phases are fully separated.

```
Parallel phase:  par_iter().filter().map().collect()
                 — zero shared-state access; only reads should_stop (AtomicBool)
Serial phase:    plain for loop over Vec<Result<...>>
                 — no locking, no atomics needed
```

`unique_distributions` and `results` are plain local `HashSet` and `Vec` values on the
main thread.  `total_failures` is a plain `u32`.  Only `should_stop: Arc<AtomicBool>` is
shared across the thread boundary so the filter can skip work early within a batch.

**Trade-off:** tasks that started before `should_stop` is set finish their
`find_distribution_internal` call; the serial loop discards excess results.  The
worst-case wasted work is `batch_size − 1 = 99` extra calls on the final batch.  This is
negligible compared to the elimination of per-thread mutex acquisitions on every attempt.

---

## Item 6 — Direct Mean Initialization

**Planned:** replace the O(n·k) random-fill + `adjust_mean_internal` initialization with
an O(n) direct construction using floor/ceil values.

**Implemented as planned.**  After the guard for empty `possible_values_scaled`, the code:

1. Computes `vec_target_sum = round(mean · n · scale_factor) − fixed_sum` using integer
   arithmetic.
2. Finds `floor_idx` via `partition_point` (binary-search semantics, O(log k)).
3. Computes `n_ceil = deficit / step` where `deficit = vec_target_sum − r_n · floor_val`
   and `step = ceil_val − floor_val`.
4. If `deficit % step == 0` and `n_ceil ≤ r_n`, constructs the vector directly and
   shuffles it.

Falls back to random fill if `deficit % step ≠ 0` or the arithmetic edge case where the
mean is at the maximum of `possible_values_scaled`.  The subsequent `adjust_mean_internal`
call is retained as a safety net; it returns `Ok(())` on its first iteration when the
direct path succeeded (since the mean is already exact).

---

## Item 7 — Lincoln-Petersen Stopping Criterion

**Planned:** replace the hard-coded `0.00001` probability formula with a Lincoln-Petersen
coverage estimate.

**Implemented as planned.**  A new constant `LP_COVERAGE_THRESHOLD: f64 = 0.99` was
added alongside the other module-level constants.  A `cumulative_duplicates: u64` counter
accumulates all duplicate draws across all batches (never reset).  After each batch the
criterion is:

```
N̂        = n_found + cumulative_duplicates
coverage = n_found / N̂
stop if  coverage ≥ LP_COVERAGE_THRESHOLD
```

The previous `total_duplicates` counter (which reset on each new unique find) was removed
entirely because the only consumer was the old formula.  The `total_failures ≥ 1000`
hard-stop is retained unchanged as a guard against degenerate inputs.

**Behavioural note:** the LP estimate is conservative when sampling density is high (many
unique distributions found per attempt and few duplicates), so the criterion fires only
when the duplicate rate starts to rise — i.e., precisely when the space is near-exhausted.
For large spaces (`sprite_test_big`: n = 2000, 5 000 distributions requested) the
`results.len() ≥ n_distributions` exit fires first and the LP criterion is never reached.

---

## Summary of Changes

| Plan # | Area | Functions changed |
|--------|------|-------------------|
| 2 | Incremental statistics | `find_distribution_internal`, `shift_values_internal`, `adjust_mean_internal`; new `mean_from_running`, `sd_from_running` |
| 4 | Constant hoisting | `find_distribution_internal`, `adjust_mean_internal` |
| 3 | O(1) value lookups | `SpriteParams` (new field), `build_sprite_params`, `adjust_mean_internal`, `shift_values_internal` |
| 1 | Gap resolution | `shift_values_internal` |
| 5 | Mutex elimination | `find_distributions_all_internal` |
| 6 | Direct mean init | `find_distribution_internal` |
| 7 | LP stopping criterion | `find_distributions_all_internal`; new `LP_COVERAGE_THRESHOLD` |
