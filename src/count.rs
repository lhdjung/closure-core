//! Fast count-in-advance algorithm for CLOSURE.
//!
//! Counts valid sorted combinations matching mean and SD constraints
//! without enumerating them, using dynamic programming over frequency
//! assignments for each scale value.

use std::collections::HashMap;

/// Count valid sorted combinations that CLOSURE would find.
///
/// Returns the number of sorted (non-decreasing) sequences of `n` values
/// from `scale_min..=scale_max` whose sample mean and sample SD match the
/// targets within the given rounding tolerances.
///
/// This is equivalent to `closure_parallel(...).results.sample.len()` but
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
/// The number of valid sorted combinations.
pub fn closure_count(
    mean: f64,
    sd: f64,
    n: i32,
    scale_min: i32,
    scale_max: i32,
    rounding_error_mean: f64,
    rounding_error_sd: f64,
) -> u64 {
    if n < 2 || scale_min > scale_max {
        return 0;
    }

    let n_i64 = n as i64;
    let n_f64 = n as f64;

    // --- Sum bounds (integer, since scale values are integers) ---
    let target_sum = mean * n_f64;
    let sum_tolerance = rounding_error_mean * n_f64;
    let sum_lower = ((target_sum - sum_tolerance).ceil() as i64)
        .max(n_i64 * scale_min as i64);
    let sum_upper = ((target_sum + sum_tolerance).floor() as i64)
        .min(n_i64 * scale_max as i64);
    if sum_lower > sum_upper {
        return 0;
    }

    // --- Variance bounds ---
    // sample_sd = sqrt((sum_sq - sum²/n) / (n-1))
    // We track var_nm1 = sum_sq - sum²/n = sample_variance * (n-1)
    let sd_lo = (sd - rounding_error_sd).max(0.0);
    let sd_hi = sd + rounding_error_sd;
    let var_nm1_lo = sd_lo * sd_lo * (n_f64 - 1.0);
    let var_nm1_hi = sd_hi * sd_hi * (n_f64 - 1.0);

    // Global sum_sq bounds for pruning (across all valid sums).
    // var_nm1 = sum_sq - sum²/n, so sum_sq = var_nm1 + sum²/n.
    let min_abs_sum = if sum_lower <= 0 && sum_upper >= 0 {
        0.0
    } else {
        (sum_lower.unsigned_abs().min(sum_upper.unsigned_abs())) as f64
    };
    let max_abs_sum = (sum_lower.unsigned_abs().max(sum_upper.unsigned_abs())) as f64;
    let global_sq_lo = (var_nm1_lo + min_abs_sum * min_abs_sum / n_f64 - 1.0) as i64;
    let global_sq_hi = (var_nm1_hi + max_abs_sum * max_abs_sum / n_f64 + 1.0) as i64;

    // --- DP ---
    let scale_max_i64 = scale_max as i64;
    let scale_max_sq = scale_max_i64 * scale_max_i64;

    // State: (remaining_items, sum, sum_sq) -> count of frequency assignments
    let mut dp: HashMap<(u32, i64, i64), u64> = HashMap::new();
    dp.insert((n as u32, 0, 0), 1);

    // Accumulate valid complete states here (remaining == 0 and constraints met)
    let mut total: u64 = 0;

    for v in scale_min..=scale_max {
        let v_i64 = v as i64;
        let v_sq = v_i64 * v_i64;
        let is_last = v == scale_max;
        let next_v = (v + 1) as i64;
        let next_v_sq = next_v * next_v;

        let mut next_dp: HashMap<(u32, i64, i64), u64> =
            HashMap::with_capacity(dp.len());

        for (&(remaining, sum, sum_sq), &count) in &dp {
            if remaining == 0 {
                // Complete state; check constraints and accumulate
                accumulate_if_valid(
                    &mut total,
                    count,
                    sum,
                    sum_sq,
                    n_i64,
                    n_f64,
                    sum_lower,
                    sum_upper,
                    var_nm1_lo,
                    var_nm1_hi,
                );
                continue; // Don't propagate — it would just pass through unchanged
            }

            let rem = remaining as usize;

            if is_last {
                // All remaining items must take this value
                let f = rem as i64;
                *next_dp
                    .entry((0, sum + f * v_i64, sum_sq + f * v_sq))
                    .or_insert(0) += count;
                continue;
            }

            // Try each frequency f for value v
            for f in 0..=rem {
                let f_i64 = f as i64;
                let new_rem = (rem - f) as u32;
                let new_sum = sum + f_i64 * v_i64;
                let new_sq = sum_sq + f_i64 * v_sq;

                if new_rem > 0 {
                    let nr = new_rem as i64;

                    // Sum pruning.
                    // As f increases: min_total_sum and max_total_sum both decrease
                    // (since v < next_v <= scale_max for non-last values).
                    let max_total_sum = new_sum + nr * scale_max_i64;
                    if max_total_sum < sum_lower {
                        break; // Larger f only makes it worse
                    }
                    let min_total_sum = new_sum + nr * next_v;
                    if min_total_sum > sum_upper {
                        continue; // Larger f will decrease min_total_sum
                    }

                    // Sum-of-squares pruning (same monotonicity as sum pruning)
                    let max_total_sq = new_sq + nr * scale_max_sq;
                    if max_total_sq < global_sq_lo {
                        break;
                    }
                    let min_total_sq = new_sq + nr * next_v_sq;
                    if min_total_sq > global_sq_hi {
                        continue;
                    }
                }

                *next_dp.entry((new_rem, new_sum, new_sq)).or_insert(0) += count;
            }
        }

        dp = next_dp;
    }

    // Process remaining complete states from the final DP layer
    for (&(remaining, sum, sum_sq), &count) in &dp {
        debug_assert_eq!(remaining, 0, "All items should be placed after the last scale value");
        if remaining != 0 {
            continue;
        }
        accumulate_if_valid(
            &mut total,
            count,
            sum,
            sum_sq,
            n_i64,
            n_f64,
            sum_lower,
            sum_upper,
            var_nm1_lo,
            var_nm1_hi,
        );
    }

    total
}

/// Check sum and variance constraints; if valid, add `count` to `total`.
#[inline]
fn accumulate_if_valid(
    total: &mut u64,
    count: u64,
    sum: i64,
    sum_sq: i64,
    n_i64: i64,
    n_f64: f64,
    sum_lower: i64,
    sum_upper: i64,
    var_nm1_lo: f64,
    var_nm1_hi: f64,
) {
    if sum < sum_lower || sum > sum_upper {
        return;
    }
    // Use n*sum_sq - sum² to avoid division for the integer part,
    // then compare against n * var_nm1 bounds.
    let n_times_var_nm1 = n_i64 * sum_sq - sum * sum;
    let n_times_var_nm1_f = n_times_var_nm1 as f64;
    let threshold_lo = n_f64 * var_nm1_lo;
    let threshold_hi = n_f64 * var_nm1_hi;
    if n_times_var_nm1_f < threshold_lo - 1e-6 || n_times_var_nm1_f > threshold_hi + 1e-6 {
        return;
    }
    *total += count;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_single_solution() {
        // n=3, scale=[1,3], mean=2.0, sd=1.0
        // Only {1,2,3}: sum=6, sum_sq=14, var_nm1=14-36/3=2, sd=sqrt(2/2)=1.0
        assert_eq!(closure_count(2.0, 1.0, 3, 1, 3, 0.0, 0.0), 1);
    }

    #[test]
    fn test_count_all_same() {
        // n=4, scale=[1,3], mean=2.0, sd=0.0
        // Only {2,2,2,2}
        assert_eq!(closure_count(2.0, 0.0, 4, 1, 3, 0.0, 0.0), 1);
    }

    #[test]
    fn test_count_no_solutions() {
        // mean=1.5 with n=3 requires sum=4.5, not an integer → 0 solutions
        assert_eq!(closure_count(1.5, 0.5, 3, 1, 3, 0.0, 0.0), 0);
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
        let count = closure_count(2.0, 1.0, 4, 1, 3, 0.5, 100.0);
        assert_eq!(count, 11);
    }

    #[test]
    fn test_count_n2() {
        // n=2, scale=[1,5], mean=3.0, sd=0
        // Only {3,3}
        assert_eq!(closure_count(3.0, 0.0, 2, 1, 5, 0.0, 0.0), 1);

        // n=2, scale=[1,5], mean=3.0, sd=sqrt(2) ≈ 1.4142
        // Need sum=6, var_nm1 = sum_sq - 36/2 = sum_sq - 18 = 2
        // So sum_sq = 20 → pairs with sum=6 and sum_sq=20:
        //   {2,4}: sum_sq=4+16=20 ✓
        // That's 1 solution
        let sd = (2.0_f64).sqrt();
        assert_eq!(closure_count(3.0, sd, 2, 1, 5, 0.0, 0.0), 1);
    }

    #[test]
    fn test_count_edge_cases() {
        assert_eq!(closure_count(1.0, 0.0, 0, 1, 5, 0.0, 0.0), 0); // n=0
        assert_eq!(closure_count(1.0, 0.0, 1, 1, 5, 0.0, 0.0), 0); // n=1
        assert_eq!(closure_count(1.0, 0.0, 5, 3, 2, 0.0, 0.0), 0); // invalid scale
    }

    #[test]
    fn test_count_matches_closure_parallel() {
        // Cross-validate with closure_parallel on a small case.
        // Use small rounding tolerances to avoid f64 boundary mismatches
        // between CLOSURE's floating-point DFS and our integer DP.
        use crate::closure_parallel;

        let mean = 3.0_f64;
        let sd = 1.0_f64;
        let n = 5_i32;
        let scale_min = 1_i32;
        let scale_max = 5_i32;
        let re_mean = 0.05;
        let re_sd = 0.05;

        let results = closure_parallel::<f64, i32>(
            mean, sd, n, scale_min, scale_max, re_mean, re_sd, 1, None, None,
        )
        .unwrap();
        let expected = results.results.sample.len() as u64;

        let counted = closure_count(mean, sd, n, scale_min, scale_max, re_mean, re_sd);
        assert_eq!(counted, expected, "count={counted} but closure_parallel found {expected}");
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
            mean, sd, n, scale_min, scale_max, re_mean, re_sd, 1, None, None,
        )
        .unwrap();
        let expected = results.results.sample.len() as u64;

        let counted = closure_count(mean, sd, n, scale_min, scale_max, re_mean, re_sd);
        assert_eq!(counted, expected, "count={counted} but closure_parallel found {expected}");
    }

    #[test]
    fn test_count_moderate_case() {
        // n=10, scale=[1,7], mean=3.0, sd=2.0 — the benchmark from the plan
        // We just verify it completes quickly and returns a nonzero count.
        let count = closure_count(3.0, 2.0, 10, 1, 7, 0.0, 0.0);
        assert!(count > 0, "Expected nonzero count for n=10 benchmark case");
    }
}
