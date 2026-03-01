# SPRITE Implementation Improvements

This document catalogues potential improvements to `src/sprite.rs`, grouped by kind.
Each entry references the relevant lines in the current Rust code and, where applicable,
the R reference in `protocols/rsprite2-ref/core-functions.R`.

---

## 1. Correctness Gap — Gap Resolution Missing in `shift_values_internal`

**Severity:** High
**Affects:** All inputs with `items > 1`; also inputs with `restrictions_exact` that create
non-uniform spacing in `possible_values_scaled`.

### What the code does

`shift_values_internal` (sprite.rs:1279–1353) tries to adjust the SD of a distribution by
swapping two values: incrementing one and decrementing another by one step each.  For the
swap to preserve the mean, the two steps must be equal in magnitude:

```rust
if val1_new_i64 - val1_i64 == val2_i64 - val2_new_i64 {
    // … perform swap
}
```

When this condition fails the function returns `false` without doing anything.

### Why this breaks for `items > 1`

When `items > 1`, `possible_values_scaled` contains fractional values spaced `1/items`
apart (e.g. for `items = 3`, scale 1–5: `[1, 1.33, 1.67, 2, 2.33, …]`).  Restrictions
can additionally remove values from this list, creating uneven gaps.

In such cases the "next larger" and "next smaller" steps from two randomly chosen
positions will often differ in size, so the equal-gap condition fails repeatedly and
`shift_values_internal` produces `false` for most attempts.  The outer SD loop then
exhausts all `MAX_DELTA_LOOPS_UPPER` iterations and returns `Err`, inflating the failure
counter in `find_distributions_all_internal` until early termination fires.

### What the R reference does

R's `.shift_values` (core-functions.R:385–573) handles mismatched gaps in two stages.

**Stage 1 — single alternative replacement (lines 483–511):**
If `abs(gap1) ≠ abs(gap2)`, scan `poss_non_restricted` for another contiguous pair whose
spacing equals `abs(gap1)`.  If at least one of those pairs appears in the current
vector, swap it instead, so that the total delta in the sum is cancelled.

**Stage 2 — multi-step replacement (lines 512–554):**
If no single replacement resolves the imbalance, find the length of the longest run of
restricted (fixed) values in the scale, then try up to that many steps.  At step `i`,
look for `i` pairs each with spacing `gap1 / i`; perform all `i` replacements
simultaneously.  Accept if all `i` replacements succeed; otherwise backtrack.

### Recommended fix

Implement both stages inside `shift_values_internal`, using the already-sorted
`possible_values_scaled` slice.  A value→index `HashMap` (see §4 below) makes the
"find a pair with spacing X that appears in vec" lookup fast.

---

## 2. Performance — O(n) Statistics Recomputed from Scratch Every Iteration

**Severity:** High
**Affects:** All inputs; impact scales with `n`.

### What the code does

At the top of every iteration of the SD-adjustment loop in `find_distribution_internal`
(sprite.rs:1159–1189), `compute_sd_scaled` (sprite.rs:1370–1399) is called.  This
function:

1. Calls `compute_mean_scaled`, iterating over the entire `vec` and `fixed_vals` to sum
   them — O(n).
2. Iterates over both slices again to compute the sum of squared deviations — O(n).

So every iteration costs O(n), and the loop runs up to `MAX_DELTA_LOOPS_UPPER = 1_000_000`
times per distribution attempt.  For `n = 2000` this is up to 2 × 10⁹ element visits per
attempt.

Similarly, `adjust_mean_internal` calls `compute_mean_scaled` at the start of each of
its up to `n * |possible_values|` iterations.

### Why O(1) updates are possible

`shift_values_internal` only ever changes two positions in `vec` — replacing `old1` with
`new1` and `old2` with `new2`.  Therefore:

```
Δsum     = (new1 - old1) + (new2 - old2)
Δsum_sq  = (new1² - old1²) + (new2² - old2²)
```

If the caller maintains `running_sum` and `running_sum_sq` alongside `vec`, both can be
updated in O(1) after each swap.  SD can then be derived in O(1):

```
mean     = (running_sum + fixed_sum) / scale_factor / n
variance = (running_sum_sq + fixed_sum_sq) / scale_factor² / (n - 1)
           − n * mean² / (n - 1)
```

The fixed-part contributions (`fixed_sum`, `fixed_sum_sq`) are constants computed once
at the start of each attempt and never change.

### Recommended fix

In `find_distribution_internal`, initialise `running_sum` and `running_sum_sq` after the
initial random fill and mean-adjustment phase.  Pass them into `shift_values_internal` as
`&mut` parameters; have it update them whenever a swap is accepted.  Remove the
`compute_sd_scaled` and `compute_mean_scaled` calls from the inner loop and replace them
with the O(1) formulas above.

Separately, `adjust_mean_internal` should maintain a running `current_sum` and update it
by `±scale_factor / items` (one step) whenever an element is bumped, avoiding the O(n)
`compute_mean_scaled` call.

---

## 3. Performance — O(k) Linear Searches in Hot Loops

**Severity:** Medium
**Affects:** All inputs; impact scales with `|possible_values|` (k) and `n`.

### What the code does

**`adjust_mean_internal`** (sprite.rs:1246–1259):
For each attempted bump it calls `.iter().position()` to find the index of the current
value inside `possible_values_scaled` — O(k) per call.  This runs inside a loop with up
to `max_iter * max_attempts = (n * k) * (4 * n)` total calls in the worst case: O(n²k).

**`shift_values_internal`** (sprite.rs:1309–1319):
Uses `.iter().find()` twice — once for the next-larger value and once for the
next-smaller — both O(k).  This runs inside a loop with up to `vec.len() * 2` attempts.

### Recommended fix

Build two lookup structures once inside `build_sprite_params` and store them in
`SpriteParams`:

```rust
/// Maps a scaled value to its index in `possible_values_scaled`
value_to_index: HashMap<i64, usize>,
```

Then:

- **Position lookup** (`adjust_mean_internal`): `value_to_index[&val]` — O(1).
- **Next/prev value** (`shift_values_internal`): `possible_values_scaled[idx ± 1]` — O(1)
  once the index is known.

Since `possible_values_scaled` is already sorted, a binary search (`slice::binary_search`)
is a lightweight alternative that avoids the HashMap entirely for the next/prev lookup.

---

## 4. Performance — Constant Expression Hoisted Inside SD Loop

**Severity:** Low–Medium
**Affects:** All inputs.

### What the code does

Inside the SD-adjustment `for` loop in `find_distribution_internal` (sprite.rs:1177–1180):

```rust
let target_mean_rounded =
    T::from(round_f64(T::to_f64(&params.mean).unwrap(), params.m_prec)).unwrap();
```

This value does not change across iterations — `params.mean` and `params.m_prec` are both
fixed for the lifetime of the attempt.  Yet it is recomputed on every iteration where
mean drift is detected.

### Recommended fix

Compute `target_mean_rounded` once before the loop and reference it as a local binding.

---

## 5. Performance — Mutex Contention in Parallel Distribution Search

**Severity:** Medium
**Affects:** All inputs; impact scales with core count and solution density.

### What the code does

`find_distributions_all_internal` (sprite.rs:1007–1127) uses rayon to run attempts in
parallel.  Every time a thread finds a valid distribution it:

1. Locks `unique_distributions` (an `Arc<Mutex<HashSet<Vec<i64>>>>`) to check for
   duplicates.
2. If new, locks `results` (a separate `Arc<Mutex<Vec<Vec<U>>>>`) to push the result.

These two separate lock acquisitions happen for every successful attempt, and every
*failed* uniqueness check also acquires the first lock.  With many rayon threads, this
is a serialisation bottleneck.

### Option A — Thread-local accumulation

Each rayon thread accumulates its own `Vec<Vec<U>>` locally and merges into the shared
set only when its local buffer reaches a threshold (e.g., 10 distributions).  This
reduces lock acquisition frequency by ~10×.

### Option B — `parking_lot::Mutex`

Replace `std::sync::Mutex` with `parking_lot::Mutex`.  It is a drop-in replacement with
lower overhead under contention (no OS-level futex on short paths, no poisoning).

### Option C — Bloom filter pre-screen

Before acquiring the mutex, run a fast probabilistic membership test using a thread-local
bloom filter seeded from the distribution's sorted values.  Most duplicate candidates are
rejected without touching the shared state.  Only potential non-duplicates (and bloom
false-positives) pay the lock cost.

---

## 6. Algorithm — Direct Mean Initialization

**Severity:** Low
**Affects:** All inputs.

### What the code does

`find_distribution_internal` initialises `vec` with `r_n` values chosen uniformly at
random from `possible_values_scaled`, then calls `adjust_mean_internal` to iteratively
bump individual values until the mean is correct.  For a uniform random initialisation,
the expected deviation from target mean is O(SD / √n), which requires O(n * k) bump
iterations to correct.

### A better approach

For GRIM-consistent inputs, `target_sum = mean * n * scale_factor` is an integer.  A
distribution with exactly that sum can be constructed in O(n) without iteration:

1. Fill all `r_n` slots with `floor_value` (the largest scaled value ≤ target mean).
2. Compute the deficit: `deficit = target_sum - r_n * floor_value`.
3. Divide the deficit into steps of `step_size` (the gap between floor and the next
   possible value).  The quotient gives how many elements to raise by one step.
4. Randomly place those raised elements in the vector.

This eliminates the `adjust_mean_internal` call for the initial setup.  Mean correction
during the SD loop (the `adjust_mean_internal(…, 20, rng)` call) is still needed as a
safety net, but fires much less often because shifts are mean-preserving by construction.

For `items > 1`, the same principle applies using the fractional step sizes already
present in `possible_values_scaled`.

---

## 7. Algorithm — Mark-Recapture Stopping Criterion

**Severity:** Low
**Affects:** Cases where the true number of valid distributions is small (and can
therefore be enumerated exhaustively) or very large (where coverage matters).

### What the code does

The stopping condition in `find_distributions_all_internal` (sprite.rs:1098–1113) is:

```rust
let max_duplications =
    (0.00001f64.ln() / (n_found / (n_found + 1.0)).ln()).round() as u32;
if duplicates > max_duplications { break; }
```

This derives from: "if every distribution is equally likely to be sampled, what is the
expected number of draws to re-find all `n_found` known ones with probability 0.99999?"
It assumes sampling with replacement from a pool of size `n_found + 1`.  If the true
pool size is much larger than `n_found`, the formula over-estimates the duplicate rate
and stops too early.

### A better approach — Lincoln-Petersen estimator

After collecting `n_found` unique distributions with `d` total duplicate draws, the
Lincoln-Petersen mark-recapture estimator gives:

```
N̂ = n_found * (n_found + d) / n_found  =  n_found + d
```

This estimates the total population size.  The fraction of undiscovered distributions is
then `1 - n_found / N̂`.  The search can be terminated when this fraction drops below a
configurable threshold (e.g., 0.01 = estimate ≥ 99 % of valid distributions found), or
when the estimate's confidence interval is sufficiently tight.

This replaces the hard-coded `0.00001` probability with an interpretable coverage
target.

---

## Summary Table

| # | Area | Severity | Function(s) | Nature |
|---|------|----------|-------------|--------|
| 1 | Gap resolution | High / Correctness | `shift_values_internal` | Missing R feature |
| 2 | Incremental statistics | High / Performance | `find_distribution_internal`, `compute_sd_scaled`, `compute_mean_scaled` | O(n) → O(1) per inner loop iteration |
| 3 | Value lookup | Medium / Performance | `adjust_mean_internal`, `shift_values_internal` | O(k) → O(1) searches |
| 4 | Constant hoisting | Low–Medium / Performance | `find_distribution_internal` | Redundant computation in loop |
| 5 | Mutex contention | Medium / Performance | `find_distributions_all_internal` | Parallelism bottleneck |
| 6 | Direct initialisation | Low / Algorithm | `find_distribution_internal`, `adjust_mean_internal` | O(n·k) init → O(n) |
| 7 | Stopping criterion | Low / Algorithm | `find_distributions_all_internal` | Heuristic → estimator |
