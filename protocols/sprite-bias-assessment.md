# SPRITE Bias Assessment

**Date:** 2026-02-24
**Source:** `protocols/sprite-bias-investigation.pdf` (generated from `unsum`, an R wrapper for
`closure-core`)

---

## Overview

The investigation compared `sprite_generate()` and `closure_generate()` on identical inputs across
three test cases. CLOSURE (exhaustive DFS) reliably finds every valid distribution. SPRITE (stochastic
random walk) is supposed to find a representative sample of the same space. In practice SPRITE finds a
tiny, systematically biased subset:

| Test case | SPRITE found | CLOSURE found | Coverage |
|---|---|---|---|
| mean=3.0, sd=1.0, n=100, scale 1–5 | 246 | 24,907 | ~1% |
| mean=4.7, sd=2.2, n=50, scale 1–7 | 2,430 | 81,020 | ~3% |
| mean=2.00, sd=1.45, n=65, scale 1–5 | 26 | 30 | ~87% |

Case 3 is the tell: when the valid space is tiny and highly constrained, SPRITE finds nearly
everything. Cases 1–2 show the problem most acutely.

The bimodality (horns index) of the returned distributions also differs markedly:

| Test case | SPRITE horns range | CLOSURE horns range |
|---|---|---|
| mean=3.0, sd=1.0, n=100 | 0.225–0.235 | 0.225–0.272 |
| mean=4.7, sd=2.2, n=50 | 0.508–0.526 | 0.504–0.550 |

SPRITE consistently misses the high-horns (bimodal/extreme) end of the space. CLOSURE covers the
full range.

There are two distinct causes: one is an implementation bug that is fully fixable; the other is an
inherent algorithmic property of SPRITE.

---

## Cause 1 (bug): Inverted Lincoln-Petersen stopping formula

### Where it lives

`src/sprite.rs`, function `find_distributions_all_internal`, around line 1091–1106 (same logic is
duplicated in `find_distributions_all_streaming`):

```rust
// Lincoln-Petersen coverage estimate:
// N̂ = n_found + cumulative_duplicates (all draws = n_found unique + d duplicate)
// coverage = n_found / N̂ — stop when we estimate ≥ LP_COVERAGE_THRESHOLD found.
let n_found = unique_distributions.len() as f64;
if n_found > 0.0 && cumulative_duplicates > 0 {
    let n_hat = n_found + cumulative_duplicates as f64;
    let coverage = n_found / n_hat;          // ← BUG: formula is inverted
    if coverage >= LP_COVERAGE_THRESHOLD {   // LP_COVERAGE_THRESHOLD = 0.99
        break;
    }
}
```

### What the formula actually computes

`n_found / (n_found + c_dup)` is the **fraction of total draws that were unique** (non-duplicate).
It starts close to 1.0 when duplicates are scarce and falls toward 0.0 as the space becomes
saturated. The condition `>= 0.99` fires when only **1% of draws have been duplicates**, which
happens almost immediately — after just 2–3 total duplicate encounters.

Plugging in case 1:
- After finding 246 unique distributions, the condition `246 / (246 + c_dup) >= 0.99` is satisfied
  when `c_dup <= 2.5`.
- SPRITE stops after the 2nd or 3rd duplicate draw, having found 246 of 24,907 valid distributions
  (≈ 1%).

### What the correct LP formula would compute

The Schnabel/Lincoln-Petersen estimator for total population size, given `n_found` unique
distributions already encountered and `c_dup` recaptures (duplicate draws), is:

```
N̂ = n_found × (n_found + c_dup) / c_dup
```

Under uniform sampling, the expected duplicate rate per draw is `n_found / N_total`. After many
draws the duplicate fraction `c_dup / (n_found + c_dup)` converges to `n_found / N_total`, so:

```
coverage  =  n_found / N̂
          =  n_found / [n_found × (n_found + c_dup) / c_dup]
          =  c_dup / (n_found + c_dup)
```

The correct stopping condition `coverage >= 0.99` therefore means:
**99% of draws are already-seen duplicates** — the space is nearly saturated.

The current code uses `n_found / (n_found + c_dup)`, which is exactly `1 − coverage_correct`. The
check `>= 0.99` fires when the duplicate rate is only 1% (the very start of saturation) rather than
when it reaches 99% (near-complete saturation). The formula is **inverted**.

### Numerical effect in theory (uniform sampling assumption)

| Case | n_found at stop | c_dup at stop (current) | c_dup needed for correct stop |
|---|---|---|---|
| mean=3.0, n=100 | 246 | ≤ 2 | ~24,417 (≈99% of 24,907 found) |
| mean=4.7, n=50 | 2,430 | ≤ 24 | ~80,266 (≈99% of 81,020 found) |

### Practical effect (non-uniform sampling)

Because SPRITE's random walk only reaches a biased subset of the full distribution space (see
Cause 2), duplicates accumulate quickly once that subset is exhausted — regardless of which LP
formula is used. In practice, both the old and corrected formulas fire at roughly the same point:
when the locally-accessible neighborhood is saturated. Testing confirms the corrected formula
also fires around n_found ≈ 168–300 for the n=100 case.

The practical improvement from the formula fix is therefore modest: the corrected formula is
*semantically* correct (it measures what the comment says it measures) and avoids the pathological
case where the first duplicate triggers immediate stopping before n_found even reaches the
accessible-neighborhood size. However, for typical SPRITE inputs the dominant limitation is Cause
2, not the LP formula.

The old formula does have one concrete failure mode: it fires at `c_dup = 1` as soon as
`n_found >= 99`, regardless of how large the accessible neighborhood is. If SPRITE's random walk
produces its first duplicate at n_found=50 (which can happen with highly non-uniform within-neighborhood
sampling), the old formula does not fire (50/51 < 0.99). But if it happens at n_found=130, the
old formula fires immediately even though there may be 200+ accessible distributions not yet found.
The corrected formula avoids this by waiting for the duplicate *rate* to be high.

---

## Cause 2 (inherent): Near-mean initialization biases the random walk

### How SPRITE initializes each attempt

`find_distribution_internal` always begins by filling the `n` slots with `floor_val` and `ceil_val`
— the two scale values bracketing the mean. For mean=3.0 on scale 1–5 this is entirely 3s. For
mean=4.7 this is a mix of 4s and 5s. The starting point is always a **maximally central,
low-variance distribution**.

### Why this biases sampling

The SD-adjustment loop (`shift_values_internal`) then moves pairs of values apart (to increase SD)
or together (to decrease SD), one scale step at a time, for up to `max_loops_sd` iterations (clamped
to 20,000–1,000,000). Reaching a distribution like `[1,1,1,...,5,5,5,...]` from `[3,3,...,3,3]`
requires many directed steps, and the random walk's budget is often exhausted before arriving. As a
result:

- Distributions near the mean (bell-shaped, low horns) are easily reachable and frequently found.
- Bimodal / heavily skewed distributions (high horns) require traversing a long path from the
  starting point; the walk often fails before reaching them.

This is why SPRITE's horns range is always a subset of CLOSURE's, anchored at the lower
(normal-like) end.

### Why case 3 is different

When mean=2.00, sd=1.45, n=65 on scale 1–5, there are only 30 valid distributions and they all have
high frequency at value 1 (60%+ of responses). This happens to be geometrically close to the
initialization even though the distribution looks unusual. The random walk can reach all 30 within
its budget. Consequently SPRITE finds 26 (87%) and the remaining 4 it misses are an artefact of the
failure-based stopping condition (high SD-adjustment failure rate for that constrained problem),
not of sampling bias.

### This is not a fixable bug

Near-mean initialization is a design choice of the original SPRITE algorithm (Brown & Heathers,
2017). The random walk provides a heuristic sample, not uniform coverage. No stopping criterion can
make SPRITE uniformly sample the full space; it would require either (a) exhaustive enumeration
(which is what CLOSURE does) or (b) a fundamentally different initialization scheme that seeds
attempts from diverse starting distributions.

---

## Plan for fixing Cause 1

The fix is small and localized: correct the coverage formula in both places it appears.

### Step 1 — Fix `find_distributions_all_internal`

Current (line ~1096–1098):
```rust
let n_hat = n_found + cumulative_duplicates as f64;
let coverage = n_found / n_hat;
if coverage >= LP_COVERAGE_THRESHOLD {
```

Replace with:
```rust
let n_hat = n_found + cumulative_duplicates as f64;
let coverage = cumulative_duplicates as f64 / n_hat;
if coverage >= LP_COVERAGE_THRESHOLD {
```

The variable name `n_hat` is now misleading (it is the total draw count, not a population estimate).
Rename for clarity:
```rust
let total_draws = n_found + cumulative_duplicates as f64;
let duplicate_rate = cumulative_duplicates as f64 / total_draws;
if duplicate_rate >= LP_COVERAGE_THRESHOLD {
```

Also update the `eprintln!` message below it, which currently reports the inverted number as if it
were a coverage percentage.

### Step 2 — Fix `find_distributions_all_streaming`

The same logic (lines ~997–1013 in the streaming function) uses an analogous early-stop block. Apply
the same substitution: replace `n_found / (n_found + duplicates)` with
`duplicates / (n_found + duplicates)`.

### Step 3 — Reconsider LP_COVERAGE_THRESHOLD value

With the corrected formula, `LP_COVERAGE_THRESHOLD = 0.99` means "stop when 99% of draws are
duplicates". Under uniform sampling from a space of N distributions, reaching a 99% duplicate rate
requires sampling roughly `N × ln(100) ≈ 4.6 × N` total draws, which is expensive. Consider:

- **0.99** is a reasonable upper bound for completeness (use when you want near-exhaustive coverage
  and can afford the compute).
- **0.95** is a practical default (stops at 95% duplicate rate, still finds the vast majority of
  the space).
- Since SPRITE is inherently non-uniform (Cause 2), any LP threshold is only an approximation;
  document this limitation.

### Step 4 — Update the warning message

The current message says "Estimated X% of all valid distributions found". After the fix this number
will be the duplicate rate, not the coverage. Either:
- Print both: "Duplicate rate: X% — stopping (estimate ≥ Y% coverage not possible to guarantee)."
- Or suppress the LP message and let the existing "Only N matching distributions could be found."
  message speak for itself.

### Step 5 — Add a regression test

Add a test (in `sprite.rs` or a dedicated test file) that:
1. Calls `sprite_parallel` with a case where CLOSURE is known to find N distributions.
2. With `stop_after = N × 10` or similar, checks that SPRITE finds at least a reasonable fraction
   (e.g., ≥ 50%) of the true population.

Concrete candidate: mean=3.0, sd=1.0, n=10, scale 1–5 (small enough for CLOSURE to enumerate
quickly, large enough to expose the bias).

---

## What fixing Cause 1 will and will not change

**Will change:**
- The LP formula now correctly measures the duplicate rate rather than the unique-draw fraction.
  The formula matches its own documentation comment.
- The stopping criterion no longer has the pathological behavior of firing immediately on the
  first duplicate when `n_found >= 99`, regardless of how large the accessible neighborhood is.
  With non-uniform within-neighborhood sampling, the corrected formula will sometimes find
  meaningfully more distributions before stopping.
- The warning message now reports the duplicate rate (the correct quantity) rather than the
  unique-draw fraction (which was misleadingly printed as "estimated X% of all valid distributions
  found").

**Will not change (confirmed by testing):**
- For typical SPRITE inputs, the number of unique distributions returned is still bounded by the
  size of the accessible neighborhood from near-mean initialization. Both the old and new formulas
  fire when that neighborhood is saturated, giving similar unique counts in practice.
- Coverage of the full valid space remains ~1–3% for the reported test cases. This is driven
  entirely by Cause 2 (sampling bias), not the LP formula.
- The horns range returned by SPRITE will still be narrower than CLOSURE's.
- For use cases where uniform coverage of the entire distribution space is required, CLOSURE remains
  the correct tool. SPRITE should be understood as a fast heuristic sampler, not an exhaustive one.
