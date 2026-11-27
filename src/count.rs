//! Fast count-in-advance algorithm for CLOSURE.
//!
//! Counts valid sorted combinations matching mean and SD constraints
//! without enumerating them, using dynamic programming over frequency
//! assignments for each scale value.

use crate::{ClosureInputs, ParameterError, SearchBounds};
use std::collections::HashMap;

/// Count valid sorted combinations that CLOSURE would find.
///
/// Returns the number of sorted (non-decreasing) sequences of `n` values
/// from `scale_min..=scale_max` whose sample mean and sample SD match the
/// targets within the given rounding tolerances.
///
/// This is equivalent to `closure_parallel(...).results.len()` but
/// orders of magnitude faster since it never constructs the actual samples.
///
/// # Algorithm
///
/// Dynamic programming over scale-value frequencies. For each scale value
/// `v` in `scale_min..=scale_max`, we decide how many of the `n` items
/// equal `v`. The DP state tracks `(items_remaining, running_sum,
/// running_sum_of_squares)`. Aggressive pruning on sum and sum-of-squares
/// bounds keeps the state space small.
///
/// # Parameters
/// - `mean`, `sd`: Target sample mean and sample standard deviation
/// - `n`: Sample size (must be >= 2)
/// - `scale_min`, `scale_max`: Scale range endpoints (inclusive)
/// - `rounding_error_mean`, `rounding_error_sd`: Allowed rounding tolerances
///
/// # Returns
/// The number of valid sorted combinations. Counts beyond `u64::MAX` (about
/// 1.8e19) saturate there rather than wrapping around to a small number.
///
/// # Errors
/// The same [`ParameterError`] `closure_parallel` returns for the same
/// input, such as a negative `sd` or `n < 2`.
pub fn closure_count(
    mean: f64,
    sd: f64,
    n: i32,
    scale_min: i32,
    scale_max: i32,
    rounding_error_mean: f64,
    rounding_error_sd: f64,
) -> Result<u64, ParameterError> {
    // The same validation `closure_parallel` runs, and the same integer
    // thresholds its DFS uses, so the two agree on every sample by
    // construction. Without the validation a negative `sd` was read as a band
    // of SDs from 0 up to |sd| minus the tolerance.
    let ClosureInputs { bounds, .. } = ClosureInputs::new(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
    )?;

    let n_i64 = n as i64;
    let Some(pruning) = Pruning::new(&bounds, n, scale_min, scale_max) else {
        return Ok(0);
    };

    // --- DP ---

    // State: (remaining_items, sum, sum_sq) -> count of frequency assignments
    let mut dp: HashMap<(u32, i64, i64), u64> = HashMap::new();
    dp.insert((n as u32, 0, 0), 1);

    // Accumulate valid complete states here (remaining == 0 and constraints met)
    let mut total: u64 = 0;

    for v in scale_min..=scale_max {
        let v_i64 = v as i64;
        let v_sq = v_i64 * v_i64;
        let is_last = v == scale_max;
        let ahead = pruning.ahead(v);

        let mut next_dp: HashMap<(u32, i64, i64), u64> = HashMap::with_capacity(dp.len());

        for (&(remaining, sum, sum_sq), &count) in &dp {
            if remaining == 0 {
                // Complete state; check constraints and accumulate
                accumulate_if_valid(&mut total, count, sum, sum_sq, n_i64, &bounds);
                continue; // Don't propagate — it would just pass through unchanged
            }

            let rem = remaining as usize;

            if is_last {
                // All remaining items must take this value
                let f = rem as i64;
                let slot = next_dp
                    .entry((0, sum + f * v_i64, sum_sq + f * v_sq))
                    .or_insert(0);
                *slot = slot.saturating_add(count);
                continue;
            }

            // Try each frequency f for value v
            for f in 0..=rem {
                let f_i64 = f as i64;
                let new_rem = (rem - f) as u32;
                let new_sum = sum + f_i64 * v_i64;
                let new_sq = sum_sq + f_i64 * v_sq;

                match ahead.verdict(new_rem, new_sum, new_sq) {
                    Verdict::Keep => {}
                    Verdict::Skip => continue,
                    Verdict::Stop => break,
                }

                let slot = next_dp.entry((new_rem, new_sum, new_sq)).or_insert(0);
                *slot = slot.saturating_add(count);
            }
        }

        dp = next_dp;
    }

    // Process remaining complete states from the final DP layer
    for (&(remaining, sum, sum_sq), &count) in &dp {
        debug_assert_eq!(
            remaining, 0,
            "All items should be placed after the last scale value"
        );
        if remaining != 0 {
            continue;
        }
        accumulate_if_valid(&mut total, count, sum, sum_sq, n_i64, &bounds);
    }

    Ok(total)
}

/// Check sum and variance constraints; if valid, add `count` to `total`.
#[inline]
fn accumulate_if_valid(
    total: &mut u64,
    count: u64,
    sum: i64,
    sum_sq: i64,
    n_i64: i64,
    bounds: &SearchBounds,
) {
    if sum < bounds.sum_lo || sum > bounds.sum_hi {
        return;
    }
    // n·Σx² − (Σx)² is n·(n−1) times the sample variance, and an integer.
    let m2n = n_i64 * sum_sq - sum * sum;
    if m2n < bounds.m2n_lo || m2n > bounds.m2n_hi {
        return;
    }
    *total = total.saturating_add(count);
}

/// Bounds on the totals any admissible sample can reach, for pruning partial
/// frequency assignments.
struct Pruning {
    sum_lower: i64,
    sum_upper: i64,
    global_sq_lo: i64,
    global_sq_hi: i64,
    scale_max: i64,
}

impl Pruning {
    /// `None` when no total is admissible, so that no sample can be.
    fn new(bounds: &SearchBounds, n: i32, scale_min: i32, scale_max: i32) -> Option<Self> {
        let n_i64 = n as i64;
        let sum_lower = bounds.sum_lo.max(n_i64 * scale_min as i64);
        let sum_upper = bounds.sum_hi.min(n_i64 * scale_max as i64);
        if sum_lower > sum_upper {
            return None;
        }

        // Global sum_sq bounds for pruning, across all valid sums. Since
        // `n·Σx² = m2n + (Σx)²` with `m2n` between the integer bounds, these
        // are exact in i128. They used to be f64, where the ±1 slack vanished
        // once `(Σx)²/n` passed 2^53, and valid states were pruned.
        let min_abs_sum = if sum_lower <= 0 && sum_upper >= 0 {
            0
        } else {
            sum_lower.unsigned_abs().min(sum_upper.unsigned_abs()) as i128
        };
        let max_abs_sum = sum_lower.unsigned_abs().max(sum_upper.unsigned_abs()) as i128;
        let n_i128 = n as i128;
        // The numerator is never negative, so this is a ceiling division.
        let global_sq_lo =
            (bounds.m2n_lo as i128 + min_abs_sum * min_abs_sum + n_i128 - 1) / n_i128;
        let global_sq_hi = (bounds.m2n_hi as i128 + max_abs_sum * max_abs_sum) / n_i128;
        // Every reachable sum of squares fits in i64, so clamping loses nothing.
        Some(Self {
            sum_lower,
            sum_upper,
            global_sq_lo: global_sq_lo.clamp(0, i64::MAX as i128) as i64,
            global_sq_hi: global_sq_hi.clamp(0, i64::MAX as i128) as i64,
            scale_max: scale_max as i64,
        })
    }

    /// The pruning test for frequencies of value `v`, given that the values
    /// still to come are `v + 1..=scale_max`.
    fn ahead(&self, v: i32) -> Ahead<'_> {
        let next_v = v as i64 + 1;
        let top = self.scale_max;
        // Squares of the values still available, `next_v..=scale_max`. With a
        // negative `next_v` the smallest square is not at `next_v` and the
        // largest may not be at `scale_max`, so both are taken over the range.
        let rem_sq_lo = if next_v <= 0 && top >= 0 {
            0
        } else {
            (next_v * next_v).min(top * top)
        };
        let rem_sq_hi = (next_v * next_v).max(top * top);
        Ahead {
            pruning: self,
            v_nonnegative: v >= 0,
            next_v,
            rem_sq_lo,
            rem_sq_hi,
        }
    }
}

struct Ahead<'a> {
    pruning: &'a Pruning,
    v_nonnegative: bool,
    next_v: i64,
    rem_sq_lo: i64,
    rem_sq_hi: i64,
}

/// What to do with a frequency of the current value, tried in increasing
/// order from 0.
enum Verdict {
    /// Some admissible sample may extend it.
    Keep,
    /// None can, but a larger frequency may still work.
    Skip,
    /// None can, and no larger frequency can either.
    Stop,
}

impl Ahead<'_> {
    /// Whether a partial assignment with `new_rem` items still to place and
    /// totals `new_sum` and `new_sq` so far can still become admissible.
    fn verdict(&self, new_rem: u32, new_sum: i64, new_sq: i64) -> Verdict {
        if new_rem == 0 {
            return Verdict::Keep;
        }
        let p = self.pruning;
        let nr = new_rem as i64;

        // Sum pruning. As f increases, min_total_sum and max_total_sum both
        // decrease (since v < next_v <= scale_max for non-last values).
        if new_sum + nr * p.scale_max < p.sum_lower {
            return Verdict::Stop; // Larger f only makes it worse
        }
        if new_sum + nr * self.next_v > p.sum_upper {
            return Verdict::Skip; // Larger f will decrease min_total_sum
        }

        // Sum-of-squares pruning. Monotone in f only when v² is no larger than
        // the remaining squares, i.e. for v >= 0; otherwise a larger f can
        // still be feasible, so skip rather than stop.
        if new_sq + nr * self.rem_sq_hi < p.global_sq_lo {
            return if self.v_nonnegative {
                Verdict::Stop
            } else {
                Verdict::Skip
            };
        }
        if new_sq + nr * self.rem_sq_lo > p.global_sq_hi {
            return Verdict::Skip;
        }
        Verdict::Keep
    }
}

/// How often each scale value occurs across a set of samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountTable {
    /// Number of samples in the set.
    pub n_samples: u64,
    /// `by_count[j][c]` is the number of samples in which the `j`-th scale
    /// value occurs `c` times, for `c` in `0..=n`. Each row sums to
    /// `n_samples`.
    pub by_count: Vec<Vec<u64>>,
}

impl CountTable {
    /// The smallest count of each scale value across the samples; empty when
    /// there are none.
    pub fn count_lo(&self) -> Vec<u32> {
        self.ends(|row| row.iter().position(|&s| s > 0))
    }

    /// The largest count of each scale value across the samples; empty when
    /// there are none.
    pub fn count_hi(&self) -> Vec<u32> {
        self.ends(|row| row.iter().rposition(|&s| s > 0))
    }

    /// The mean count of each scale value across the samples; empty when
    /// there are none. Sums to `n`.
    pub fn count_mean(&self) -> Vec<f64> {
        if self.n_samples == 0 {
            return Vec::new();
        }
        self.by_count
            .iter()
            .map(|row| {
                let total: f64 = row
                    .iter()
                    .enumerate()
                    .map(|(c, &s)| c as f64 * s as f64)
                    .sum();
                total / row.iter().map(|&s| s as f64).sum::<f64>()
            })
            .collect()
    }

    fn ends(&self, end: impl Fn(&[u64]) -> Option<usize>) -> Vec<u32> {
        if self.n_samples == 0 {
            return Vec::new();
        }
        self.by_count
            .iter()
            .map(|row| end(row).expect("every row holds all samples") as u32)
            .collect()
    }
}

/// Per-value counts of the samples [`closure_count`] counts, over all of them
/// and over the single-peaked ones, without listing any sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountDetails {
    /// The scale values, `scale_min..=scale_max`.
    pub values: Vec<i32>,
    /// All samples. `all.n_samples` is what [`closure_count`] returns.
    pub all: CountTable,
    /// The single-peaked samples: their counts rise to a maximum and then
    /// fall, with ties allowed and no dip at all, so that their
    /// [`unimodality_deficit`](crate::modality::unimodality_deficit) is 0.
    /// This needs no prominence cutoff.
    pub single_peak: CountTable,
}

/// Per-value counts of the CLOSURE samples, over all of them and over the
/// single-peaked ones, computed without listing any sample.
///
/// [`closure_count`] says how many samples there are; this also says, for each
/// scale value and each count `c`, in how many samples the value occurs `c`
/// times ([`CountTable::by_count`]). That gives the range of each value's count
/// across all samples, as `modality_counts` reports it after a CLOSURE run, and
/// the same for the single-peaked samples alone. So it settles whether any
/// single-peaked sample exists (`single_peak.n_samples > 0`), and what every
/// such sample would have to look like, at sample sizes where listing the
/// samples is out of reach.
///
/// # Algorithm
///
/// The dynamic program of [`closure_count`], run forwards and then backwards.
/// The forward pass counts the ways to reach each state (items still to place,
/// running sum, running sum of squares) after each scale value, keeping every
/// layer; the backward pass counts, for each state, the ways to finish it as an
/// admissible sample. A count `c` of value `j` is then in as many samples as the
/// products of the two counts across the transitions that give value `j` the
/// count `c`. For the single-peaked samples the state also records the last
/// count and whether the counts have started to fall, which only allows
/// sequences that rise and then fall.
///
/// Time and memory grow with the number of reachable states, so with `n`, the
/// number of scale values, and the width of the rounding intervals. It takes
/// about one and a half times as long as [`closure_count`]: on 1–7 at two
/// decimals, about half a second for `n = 100` and 40 seconds for `n = 300`
/// (`details_timing` in the tests).
///
/// # Returns
/// Counts beyond `u64::MAX` (about 1.8e19) saturate there; a nonzero count
/// never becomes zero, so the ranges stay exact.
///
/// # Errors
/// The same [`ParameterError`] as [`closure_count`].
pub fn closure_count_details(
    mean: f64,
    sd: f64,
    n: i32,
    scale_min: i32,
    scale_max: i32,
    rounding_error_mean: f64,
    rounding_error_sd: f64,
) -> Result<CountDetails, ParameterError> {
    let ClosureInputs { bounds, .. } = ClosureInputs::new(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
    )?;
    let values: Vec<i32> = (scale_min..=scale_max).collect();
    let pruning = Pruning::new(&bounds, n, scale_min, scale_max);
    let table = |shape| match &pruning {
        Some(p) => count_table(p, &bounds, n, &values, shape),
        None => CountTable {
            n_samples: 0,
            by_count: vec![vec![0; n as usize + 1]; values.len()],
        },
    };
    Ok(CountDetails {
        all: table(Shape::Any),
        single_peak: table(Shape::SinglePeak),
        values,
    })
}

#[derive(Clone, Copy)]
enum Shape {
    Any,
    SinglePeak,
}

/// Items still to place, running sum, running sum of squares, and the shape
/// tag: always 0 for [`Shape::Any`]; for [`Shape::SinglePeak`], twice the last
/// count, plus 1 once the counts have started to fall.
type State = (u32, i64, i64, u32);

impl Shape {
    /// The tag after giving the current value the count `f`, with `rem` items
    /// still to place on `left` more values; `None` if no single-peaked
    /// sample can follow.
    fn next_tag(self, tag: u32, f: u32, rem: u32, left: u32) -> Option<u32> {
        let Shape::SinglePeak = self else {
            return Some(0);
        };
        let (last, falling) = (tag / 2, tag % 2 == 1);
        let falling = match (falling, f >= last) {
            (false, true) => false,
            (false, false) => true,
            (true, _) if f <= last => true,
            (true, _) => return None,
        };
        // Once falling, no later count may exceed this one.
        if falling && rem as u64 > f as u64 * left as u64 {
            return None;
        }
        // A last count of at least `rem` no longer constrains anything that
        // follows, rising or falling, so all such states merge into one.
        Some(if f >= rem {
            2 * rem + 1
        } else {
            2 * f + falling as u32
        })
    }
}

/// Calls `visit(f, after)` for every count `f` of value `v` that some
/// admissible sample may give it from `state`, with the state after it.
fn transitions(
    shape: Shape,
    ahead: &Ahead,
    v: i64,
    left: u32,
    (rem, sum, sq, tag): State,
    mut visit: impl FnMut(u32, State),
) {
    // The last value takes every item still to place.
    let first = if left == 0 { rem } else { 0 };
    for f in first..=rem {
        let (new_rem, f_i64) = (rem - f, f as i64);
        let (new_sum, new_sq) = (sum + f_i64 * v, sq + f_i64 * v * v);
        match ahead.verdict(new_rem, new_sum, new_sq) {
            Verdict::Keep => {}
            Verdict::Skip => continue,
            Verdict::Stop => break,
        }
        if let Some(new_tag) = shape.next_tag(tag, f, new_rem, left) {
            visit(f, (new_rem, new_sum, new_sq, new_tag));
        }
    }
}

fn count_table(
    pruning: &Pruning,
    bounds: &SearchBounds,
    n: i32,
    values: &[i32],
    shape: Shape,
) -> CountTable {
    let k = values.len();
    let left = |j: usize| (k - j - 1) as u32;

    // Forward: layers[j] holds the states before value j, with the number of
    // ways to reach each.
    let mut layers: Vec<HashMap<State, u64>> = Vec::with_capacity(k + 1);
    layers.push(HashMap::from([((n as u32, 0, 0, 0), 1)]));
    for (j, &v) in values.iter().enumerate() {
        let ahead = pruning.ahead(v);
        let mut next: HashMap<State, u64> = HashMap::with_capacity(layers[j].len());
        for (&state, &ways) in &layers[j] {
            transitions(shape, &ahead, v as i64, left(j), state, |_, after| {
                let slot = next.entry(after).or_insert(0);
                *slot = slot.saturating_add(ways);
            });
        }
        layers.push(next);
    }

    // Backward: `ends` holds the states after value j with the number of ways
    // to finish each as an admissible sample.
    let mut ends: HashMap<State, u64> = layers[k]
        .keys()
        .filter(|&&(rem, sum, sq, _)| {
            let mut admissible = 0;
            accumulate_if_valid(&mut admissible, 1, sum, sq, n as i64, bounds);
            rem == 0 && admissible > 0
        })
        .map(|&state| (state, 1))
        .collect();
    let mut by_count = vec![vec![0u64; n as usize + 1]; k];
    for j in (0..k).rev() {
        let ahead = pruning.ahead(values[j]);
        let mut starts: HashMap<State, u64> = HashMap::new();
        for (&state, &ways) in &layers[j] {
            transitions(
                shape,
                &ahead,
                values[j] as i64,
                left(j),
                state,
                |f, after| {
                    let Some(&finishes) = ends.get(&after) else {
                        return;
                    };
                    let slot = starts.entry(state).or_insert(0);
                    *slot = slot.saturating_add(finishes);
                    let through = (ways as u128 * finishes as u128).min(u64::MAX as u128) as u64;
                    let cell = &mut by_count[j][f as usize];
                    *cell = cell.saturating_add(through);
                },
            );
        }
        layers.truncate(j + 1);
        ends = starts;
    }

    CountTable {
        n_samples: ends.values().fold(0, |a: u64, &b| a.saturating_add(b)),
        by_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_single_solution() {
        // n=3, scale=[1,3], mean=2.0, sd=1.0
        // Only {1,2,3}: sum=6, sum_sq=14, var_nm1=14-36/3=2, sd=sqrt(2/2)=1.0
        assert_eq!(closure_count(2.0, 1.0, 3, 1, 3, 0.0, 0.0).unwrap(), 1);
    }

    #[test]
    fn test_count_all_same() {
        // n=4, scale=[1,3], mean=2.0, sd=0.0
        // Only {2,2,2,2}
        assert_eq!(closure_count(2.0, 0.0, 4, 1, 3, 0.0, 0.0).unwrap(), 1);
    }

    #[test]
    fn test_count_no_solutions() {
        // mean=1.5 with n=3 requires sum=4.5, not an integer → 0 solutions
        assert_eq!(closure_count(1.5, 0.5, 3, 1, 3, 0.0, 0.0).unwrap(), 0);
    }

    #[test]
    fn test_count_with_rounding() {
        // mean=2.0 ± 0.5 with n=4, scale=[1,3]
        // sum must be in [6, 10], clamped to [4, 12] → [6, 10]
        // All sorted 4-combos from {1,2,3}:
        //   {1,1,1,1} sum=4 — out of range
        //   {1,1,1,2} sum=5 — out of range
        //   {1,1,1,3} sum=6 ✓
        //   {1,1,2,2} sum=6 ✓
        //   {1,1,2,3} sum=7 ✓
        //   {1,1,3,3} sum=8 ✓
        //   {1,2,2,2} sum=7 ✓
        //   {1,2,2,3} sum=8 ✓
        //   {1,2,3,3} sum=9 ✓
        //   {1,3,3,3} sum=10 ✓
        //   {2,2,2,2} sum=8 ✓
        //   {2,2,2,3} sum=9 ✓
        //   {2,2,3,3} sum=10 ✓
        //   {2,3,3,3} sum=11 — out of range
        //   {3,3,3,3} sum=12 — out of range
        // That's 11 in sum range. Now filter by SD <= 100 (wide tolerance):
        let count = closure_count(2.0, 1.0, 4, 1, 3, 0.5, 100.0).unwrap();
        assert_eq!(count, 11);
    }

    #[test]
    fn test_count_n2() {
        // n=2, scale=[1,5], mean=3.0, sd=0
        // Only {3,3}
        assert_eq!(closure_count(3.0, 0.0, 2, 1, 5, 0.0, 0.0).unwrap(), 1);

        // n=2, scale=[1,5], mean=3.0, sd=sqrt(2) ≈ 1.4142
        // Need sum=6, var_nm1 = sum_sq - 36/2 = sum_sq - 18 = 2
        // So sum_sq = 20 → pairs with sum=6 and sum_sq=20:
        //   {2,4}: sum_sq=4+16=20 ✓
        // That's 1 solution
        let sd = (2.0_f64).sqrt();
        assert_eq!(closure_count(3.0, sd, 2, 1, 5, 0.0, 0.0).unwrap(), 1);
    }

    #[test]
    fn test_count_edge_cases() {
        assert!(closure_count(1.0, 0.0, 0, 1, 5, 0.0, 0.0).is_err()); // n=0
        assert!(closure_count(1.0, 0.0, 1, 1, 5, 0.0, 0.0).is_err()); // n=1
        assert!(closure_count(1.0, 0.0, 5, 3, 2, 0.0, 0.0).is_err()); // invalid scale
    }

    #[test]
    fn test_count_matches_closure_parallel() {
        // Cross-validate with closure_parallel on a small case.
        use crate::closure_parallel;

        let mean = 3.0_f64;
        let sd = 1.0_f64;
        let n = 5_i32;
        let scale_min = 1_i32;
        let scale_max = 5_i32;
        let re_mean = 0.05;
        let re_sd = 0.05;

        let results = closure_parallel::<f64, i32>(
            mean, sd, n, scale_min, scale_max, re_mean, re_sd, 1, None, None, None,
        )
        .unwrap();
        let expected = results.results.len() as u64;

        let counted = closure_count(mean, sd, n, scale_min, scale_max, re_mean, re_sd).unwrap();
        assert_eq!(
            counted, expected,
            "count={counted} but closure_parallel found {expected}"
        );
    }

    #[test]
    fn test_count_matches_closure_parallel_with_rounding() {
        use crate::closure_parallel;

        let mean = 2.5_f64;
        let sd = 0.8_f64;
        let n = 6_i32;
        let scale_min = 1_i32;
        let scale_max = 4_i32;
        let re_mean = 0.1_f64;
        let re_sd = 0.1_f64;

        let results = closure_parallel::<f64, i32>(
            mean, sd, n, scale_min, scale_max, re_mean, re_sd, 1, None, None, None,
        )
        .unwrap();
        let expected = results.results.len() as u64;

        let counted = closure_count(mean, sd, n, scale_min, scale_max, re_mean, re_sd).unwrap();
        assert_eq!(
            counted, expected,
            "count={counted} but closure_parallel found {expected}"
        );
    }

    #[test]
    fn test_count_moderate_case() {
        // n=10, scale=[1,7], mean=3.0, sd=2.0 — the benchmark from the plan
        // We just verify it completes quickly and returns a nonzero count.
        let count = closure_count(3.0, 2.0, 10, 1, 7, 0.0, 0.0).unwrap();
        assert!(count > 0, "Expected nonzero count for n=10 benchmark case");
    }

    /// Helper: assert closure_count and closure_parallel agree for given parameters
    fn assert_count_matches_parallel(
        mean: f64,
        sd: f64,
        n: i32,
        smin: i32,
        smax: i32,
        re_m: f64,
        re_s: f64,
        label: &str,
    ) {
        use crate::closure_parallel;

        let counted = closure_count(mean, sd, n, smin, smax, re_m, re_s).unwrap();

        let results =
            closure_parallel::<f64, i32>(mean, sd, n, smin, smax, re_m, re_s, 1, None, None, None)
                .unwrap();
        let enumerated = results.results.len() as u64;

        assert_eq!(
            counted, enumerated,
            "{label}: closure_count={counted}, closure_parallel={enumerated}"
        );
    }

    /// Sweep small-n cases (n=3..8) with varied scales and statistics
    #[test]
    fn test_count_vs_closure_parallel_small_n() {
        let cases: &[(f64, f64, i32, i32, i32, f64, f64, &str)] = &[
            (2.0, 0.5, 3, 1, 3, 0.05, 0.05, "n=3, [1,3], centered"),
            (2.0, 1.0, 3, 1, 3, 0.05, 0.05, "n=3, [1,3], high sd"),
            (2.0, 0.5, 4, 1, 3, 0.05, 0.05, "n=4, [1,3]"),
            (3.0, 1.0, 4, 1, 5, 0.05, 0.05, "n=4, [1,5]"),
            (2.0, 0.5, 5, 1, 4, 0.05, 0.05, "n=5, [1,4]"),
            (3.0, 1.5, 5, 1, 5, 0.05, 0.05, "n=5, [1,5], high sd"),
            (2.0, 0.0, 4, 1, 3, 0.05, 0.05, "n=4, zero sd"),
            (3.0, 0.0, 5, 1, 5, 0.05, 0.05, "n=5, zero sd"),
            (2.5, 1.0, 5, 1, 4, 0.2, 0.2, "n=5, wide rounding"),
            (3.0, 1.0, 6, 1, 5, 0.15, 0.15, "n=6, [1,5]"),
            (4.0, 1.5, 8, 1, 7, 0.05, 0.05, "n=8, [1,7]"),
            (3.5, 2.0, 8, 1, 7, 0.05, 0.05, "n=8, [1,7], high sd"),
            (1.0, 3.0, 5, 1, 3, 0.05, 0.05, "impossible sd"),
            (6.5, 0.5, 4, 1, 7, 0.05, 0.05, "near-top mean"),
        ];
        for &(mean, sd, n, smin, smax, re_m, re_s, label) in cases {
            assert_count_matches_parallel(mean, sd, n, smin, smax, re_m, re_s, label);
        }
    }

    /// Cross-validate with realistic n=10..30, typical Likert-scale survey parameters
    #[test]
    fn test_count_vs_closure_parallel_medium_n() {
        let cases: &[(f64, f64, i32, i32, i32, f64, f64, &str)] = &[
            // n=10: 7-point Likert, various means and SDs
            (4.0, 1.50, 10, 1, 7, 0.05, 0.05, "n=10, [1,7], centered"),
            (
                3.0,
                2.00,
                10,
                1,
                7,
                0.05,
                0.05,
                "n=10, [1,7], plan benchmark",
            ),
            (2.5, 1.00, 10, 1, 5, 0.05, 0.05, "n=10, [1,5], low mean"),
            // n=12-15: 5-point and 7-point scales
            (3.0, 1.00, 12, 1, 5, 0.05, 0.05, "n=12, [1,5]"),
            (4.0, 2.00, 15, 1, 7, 0.05, 0.05, "n=15, [1,7]"),
            (
                3.0,
                1.00,
                15,
                1,
                5,
                0.10,
                0.10,
                "n=15, [1,5], wider rounding",
            ),
            (4.0, 2.50, 15, 1, 7, 0.05, 0.05, "n=15, [1,7], high sd"),
            // n=20: covers common small-group research
            (3.0, 1.20, 20, 1, 5, 0.05, 0.05, "n=20, [1,5]"),
            (4.0, 1.50, 20, 1, 7, 0.05, 0.05, "n=20, [1,7]"),
            (2.5, 0.80, 20, 1, 4, 0.05, 0.05, "n=20, [1,4]"),
            (3.0, 0.50, 20, 1, 5, 0.05, 0.05, "n=20, [1,5], low sd"),
            (5.5, 1.00, 20, 1, 7, 0.05, 0.05, "n=20, [1,7], high mean"),
            (1.5, 0.50, 20, 1, 5, 0.05, 0.05, "n=20, [1,5], low mean"),
            (
                3.5,
                1.20,
                20,
                1,
                7,
                0.10,
                0.10,
                "n=20, [1,7], wider rounding",
            ),
            // n=25-30: medium groups
            (3.0, 1.00, 25, 1, 5, 0.05, 0.05, "n=25, [1,5]"),
            (4.0, 1.80, 25, 1, 7, 0.05, 0.05, "n=25, [1,7]"),
            (3.5, 1.00, 30, 1, 5, 0.05, 0.05, "n=30, [1,5]"),
            (4.0, 1.50, 30, 1, 7, 0.05, 0.05, "n=30, [1,7]"),
        ];
        for &(mean, sd, n, smin, smax, re_m, re_s, label) in cases {
            assert_count_matches_parallel(mean, sd, n, smin, smax, re_m, re_s, label);
        }
    }

    /// Cross-validate at n=40..50 — realistic group sizes where CLOSURE is slow
    /// but still finishes within a test timeout
    #[test]
    fn test_count_vs_closure_parallel_large_n() {
        let cases: &[(f64, f64, i32, i32, i32, f64, f64, &str)] = &[
            (3.0, 1.00, 40, 1, 5, 0.05, 0.05, "n=40, [1,5]"),
            (4.0, 1.50, 50, 1, 7, 0.05, 0.05, "n=50, [1,7]"),
        ];
        for &(mean, sd, n, smin, smax, re_m, re_s, label) in cases {
            assert_count_matches_parallel(mean, sd, n, smin, smax, re_m, re_s, label);
        }
    }

    /// The per-value tables, built from the samples `closure_parallel` lists:
    /// over all of them, and over those with no dip.
    fn tables_by_listing(
        mean: f64,
        sd: f64,
        n: i32,
        smin: i32,
        smax: i32,
        re_m: f64,
        re_s: f64,
    ) -> (CountTable, CountTable) {
        use crate::closure_parallel;
        use crate::modality::unimodality_deficit;

        let results =
            closure_parallel::<f64, i32>(mean, sd, n, smin, smax, re_m, re_s, 1, None, None, None)
                .unwrap();
        let k = (smax - smin + 1) as usize;
        let empty = || CountTable {
            n_samples: 0,
            by_count: vec![vec![0; n as usize + 1]; k],
        };
        let (mut all, mut single_peak) = (empty(), empty());
        for row in results.results.counts.rows() {
            let mut tables = vec![&mut all];
            if unimodality_deficit(row) == 0 {
                tables.push(&mut single_peak);
            }
            for table in tables {
                table.n_samples += 1;
                for (j, &c) in row.iter().enumerate() {
                    table.by_count[j][c as usize] += 1;
                }
            }
        }
        (all, single_peak)
    }

    #[test]
    fn test_details_match_listing() {
        let cases: &[(f64, f64, i32, i32, i32, f64, f64)] = &[
            (3.0, 1.0, 5, 1, 5, 0.05, 0.05),
            (2.5, 0.8, 6, 1, 4, 0.1, 0.1),
            (2.0, 1.0, 3, 1, 3, 0.05, 0.05),
            (3.0, 1.5, 10, 1, 5, 0.05, 0.05),
            (4.0, 1.5, 10, 1, 7, 0.05, 0.05),
            (3.0, 2.0, 10, 1, 7, 0.05, 0.05),
            (3.0, 1.0, 15, 1, 5, 0.1, 0.1),
            (4.0, 2.5, 15, 1, 7, 0.05, 0.05),
            (3.0, 1.2, 20, 1, 5, 0.05, 0.05),
            (1.5, 0.5, 20, 1, 5, 0.05, 0.05),
            (5.5, 1.0, 20, 1, 7, 0.05, 0.05),
            (3.5, 1.0, 30, 1, 5, 0.05, 0.05),
            (4.0, 1.5, 30, 1, 7, 0.05, 0.05),
            // Wide SDs on few values: mostly or only samples with a dip.
            (3.0, 1.9, 20, 1, 5, 0.05, 0.05),
            (2.5, 1.49, 40, 1, 5, 0.005, 0.005),
            (4.0, 1.73, 5, 1, 7, 0.005, 0.005),
            // Negative and zero-crossing scales.
            (0.0, 1.5, 12, -3, 3, 0.05, 0.05),
            (-2.0, 1.0, 10, -5, -1, 0.05, 0.05),
            // Zero SD, and statistics no sample meets.
            (3.0, 0.0, 6, 1, 5, 0.05, 0.05),
            (1.0, 3.0, 5, 1, 3, 0.05, 0.05),
        ];
        for &(mean, sd, n, smin, smax, re_m, re_s) in cases {
            let details = closure_count_details(mean, sd, n, smin, smax, re_m, re_s).unwrap();
            let (all, single_peak) = tables_by_listing(mean, sd, n, smin, smax, re_m, re_s);
            let label = format!("mean {mean}, sd {sd}, n {n}, scale {smin}..={smax}");
            assert_eq!(details.values, (smin..=smax).collect::<Vec<_>>(), "{label}");
            assert_eq!(details.all, all, "{label}: all samples");
            assert_eq!(details.single_peak, single_peak, "{label}: single-peaked");
            assert_eq!(
                details.all.n_samples,
                closure_count(mean, sd, n, smin, smax, re_m, re_s).unwrap(),
                "{label}"
            );
        }
    }

    #[test]
    fn test_details_ranges_and_means() {
        // n = 5 on 1–7 at mean 4.00 and SD 1.73: samples exist, but every one
        // has a dip, since no single-peaked sample of 5 values reaches that SD.
        let d = closure_count_details(4.0, 1.73, 5, 1, 7, 0.005, 0.005).unwrap();
        assert!(d.all.n_samples > 0);
        assert_eq!(d.single_peak.n_samples, 0);
        assert!(d.single_peak.count_lo().is_empty());
        assert!(d.single_peak.count_mean().is_empty());

        // {2, 2, 2, 2} is the only sample.
        let d = closure_count_details(2.0, 0.0, 4, 1, 3, 0.0, 0.0).unwrap();
        assert_eq!(d.all.count_lo(), vec![0, 4, 0]);
        assert_eq!(d.all.count_hi(), vec![0, 4, 0]);
        assert_eq!(d.all.count_mean(), vec![0.0, 4.0, 0.0]);
        assert_eq!(d.single_peak, d.all);

        // Means of the counts sum to n.
        let d = closure_count_details(3.0, 1.2, 20, 1, 5, 0.05, 0.05).unwrap();
        let total: f64 = d.all.count_mean().iter().sum();
        assert!((total - 20.0).abs() < 1e-9);
    }

    /// Sample sizes where listing is out of reach. Run with
    /// `cargo test --release -- --ignored --nocapture details_timing`.
    #[test]
    #[ignore]
    fn details_timing() {
        for &(mean, sd, n, smin, smax) in &[
            (4.0, 1.5, 100, 1, 7),
            (3.0, 1.2, 300, 1, 5),
            (4.0, 1.8, 300, 1, 7),
            (5.5, 2.5, 100, 1, 10),
        ] {
            let t = std::time::Instant::now();
            let d = closure_count_details(mean, sd, n, smin, smax, 0.005, 0.005).unwrap();
            println!(
                "n = {n} on {smin}..={smax}, mean {mean}, sd {sd}: {} samples, {} single-peaked, {:.2?}",
                d.all.n_samples,
                d.single_peak.n_samples,
                t.elapsed()
            );
        }
    }
}
