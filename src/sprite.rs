//! Creating a space to experiment with a Rust translation of SPRITE

use crate::sample_counts::MAX_ABS_SCALE;
use crate::sprite_types::{OccurrenceConstraints, RestrictionsMinimum, RestrictionsOption};
use crate::{
    counts_to_result_list, FloatType, IntegerType, ParameterError, ParquetConfig,
    ProminenceThresholds, ResultListFromMeanSdN, SampleCounts, SampleFormat, SearchBounds,
    StreamingConfig, StreamingFrequencyState, StreamingResult, ValueGrid, MAX_GRID_WIDTH,
};
use num::NumCast;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::mpsc::channel;
use std::thread;

const MAX_DELTA_LOOPS_LOWER: u32 = 20_000;
/// The SD loop limit's floor for its upper end; see [`loop_limits`].
const MAX_DELTA_LOOPS_UPPER: u32 = 1_000_000;
/// Attempts per sample asked for, under `stop_after`.
const MAX_DUP_LOOPS: u32 = 20;
/// Most attempts one search makes, in memory or streaming.
const MAX_ATTEMPTS: usize = 100_000_000;
/// Float slack for the tests' own tolerance checks.
#[cfg(test)]
const DUST: f64 = 1e-12;
/// Stop once this share of all draws are duplicates. Under uniform sampling
/// that share estimates the fraction of the solution space already found, but
/// SPRITE's random walk is not uniform, so it only says that the part of the
/// space SPRITE tends to reach looks exhausted.
const LP_COVERAGE_THRESHOLD: f64 = 0.99;
/// Fewest successful draws before the duplicate rate may stop a search. On a
/// small space the rate passes the threshold after a few hundred draws: with
/// two samples SPRITE reaches once in 200 draws, one run in five missed one.
///
/// On a small space of expensive draws this floor sets the run time: mean 3,
/// sd 2, n 1000 on 1..=5 with tolerances 0.005 and 0.001 has 23 samples,
/// SPRITE finds 21 within its first few hundred draws, and the remaining
/// draws take most of its 4–5 s. With tolerances of 0.05 the same mean and
/// SD admit about a million samples, and a run without `stop_after` lasts
/// until the duplicate rate or [`attempt_cap`] ends it, hours rather than
/// seconds.
// ponytail: a sample rarer than about 1 in 1,000 draws can still be missed;
// raise this if such samples matter more than the time on small spaces.
const MIN_DRAWS: u64 = 5_000;

/// Most attempts a search makes, failures included, the same in memory and
/// streaming: `MAX_DUP_LOOPS` per sample asked for, but never more than
/// `MAX_ATTEMPTS`, and with no limit `MAX_ATTEMPTS` itself. The in-memory
/// search used to saturate at `usize::MAX` without a limit (no practical
/// bound) while streaming stopped at `MAX_ATTEMPTS`.
///
/// Never fewer than `MIN_DRAWS` either, so a small `stop_after` leaves the
/// duplicate rate its chance to end the search. That holds only while
/// attempts succeed: the cap counts failures too, `MIN_DRAWS` only successful
/// draws, so where many attempts fail the cap ends a limited search first.
fn attempt_cap(stop_after: Option<usize>) -> usize {
    stop_after.map_or(MAX_ATTEMPTS, |limit| {
        limit
            .saturating_mul(MAX_DUP_LOOPS as usize)
            .clamp(MIN_DRAWS as usize, MAX_ATTEMPTS)
    })
}

// Internal struct to hold the final, validated parameters (generic version)
#[derive(Debug)]
struct SpriteParams<U>
where
    U: IntegerType,
{
    n: U, // Changed from n_obs: u32 to match CLOSURE
    /// Admissible totals in internal units, fixed responses included: those
    /// within the mean tolerance that the free values can also reach.
    sum_lo: i64,
    sum_hi: i64,
    /// Bounds on `n·Σx² − (Σx)²` in internal units, from the SD and its
    /// tolerance: CLOSURE's integer-exact acceptance rule.
    m2n_lo: i64,
    m2n_hi: i64,
    /// Why no attempt can succeed, when that is known before searching.
    infeasible: Option<&'static str>,
    /// Sum and sum of squares of the fixed responses.
    fixed_sum: i64,
    fixed_sum_sq: i64,
    /// `scale_min` in internal units (`items` per scale point, so every grid
    /// value `scale_min + j / items` is the whole number
    /// `scale_min * items + j`): a value `v` sits at grid position
    /// `v - scale_min_scaled`.
    scale_min_scaled: i64,
    /// Scaled possible values (integers)
    possible_values_scaled: Vec<U>,
    /// Scaled fixed responses (integers)
    fixed_responses_scaled: Vec<U>,
    /// The number of fixed responses
    n_fixed: usize,
    /// Maps each scaled value (as i64) to its index in `possible_values_scaled` (O(1) lookup)
    value_to_index: HashMap<i64, usize>,
    /// The smallest spacing between neighbouring possible values; larger
    /// only where an exact restriction left a gap.
    min_step: i64,
}

/// Internal function to build and validate SPRITE parameters
#[allow(clippy::too_many_arguments)]
fn build_sprite_params<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    items: u32,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
) -> Result<SpriteParams<U>, ParameterError>
where
    T: FloatType,
    U: IntegerType,
{
    // Convert scale values to i32 for range checking and compatibility with existing logic
    let (Some(scale_min_i32), Some(scale_max_i32)) = (U::to_i32(&scale_min), U::to_i32(&scale_max))
    else {
        return Err(ParameterError::InputValidation(
            "scale_min and scale_max must fit in an i32".to_string(),
        ));
    };

    if scale_min_i32 >= scale_max_i32 {
        return Err(ParameterError::InputValidation(
            "scale_max must be greater than scale_min".to_string(),
        ));
    }
    // The output grid keys values in hundredths as an i32; past this a
    // `U` wider than i32 passed every check below and the grid panicked.
    if scale_min_i32 < -MAX_ABS_SCALE || scale_max_i32 > MAX_ABS_SCALE {
        return Err(ParameterError::InputValidation(
            "scale values must lie within ±21,474,836".to_string(),
        ));
    }

    // The rounding errors are the acceptance tolerances; the internal grid
    // resolution is only derived from them.
    let re_mean = T::to_f64(&rounding_error_mean).unwrap_or(f64::NAN);
    let re_sd = T::to_f64(&rounding_error_sd).unwrap_or(f64::NAN);
    let sd_f64 = T::to_f64(&sd).unwrap_or(f64::NAN);
    // `>= 0.0` alone let an infinite SD or tolerance through.
    if ![re_mean, re_sd, sd_f64]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
    {
        return Err(ParameterError::InputValidation(
            "sd and the rounding errors must be finite and not negative".to_string(),
        ));
    }
    // Grid values are keyed and reported in hundredths of a scale point, so
    // more than 100 items would put neighbouring values on the same key.
    if !(1..=100).contains(&items) {
        return Err(ParameterError::InputValidation(
            "items must be between 1 and 100".to_string(),
        ));
    }
    let k = (scale_max_i32 as i64 - scale_min_i32 as i64) * items as i64 + 1;
    if k > MAX_GRID_WIDTH as i64 {
        return Err(ParameterError::InputValidation(format!(
            "The scale has {k} grid values, (scale_max - scale_min) * items + 1, but SPRITE \
             allows at most {MAX_GRID_WIDTH}: it stores every sample as one count per grid \
             value, 400 KB per sample at that limit"
        )));
    }
    // The internal grid is `1 / items` apart, so it represents every value a
    // multi-item mean can take exactly. Deriving it from the rounding errors
    // instead rounded 1.333 to 1.3 at `items = 3`, and samples accepted on the
    // rounded values then missed the reported mean and SD on the real ones.
    let scale_factor = items;

    // Internal values run from `scale_min * items` to `scale_max * items`, and
    // samples are reported in hundredths; `items` is at most 100, so if both
    // ends fit in `U` times 100, everything does. Internal values that did not
    // fit used to be dropped from the grid without a word, so samples could
    // never use the top of the scale.
    let fits_u = |v: i32| <U as NumCast>::from(v as i64 * 100).is_some();
    if !fits_u(scale_min_i32) || !fits_u(scale_max_i32) {
        return Err(ParameterError::InputValidation(
            "scale values times 100 must fit in the sample's integer type".to_string(),
        ));
    }

    // Check mean is in range
    let mean_f64 = T::to_f64(&mean).unwrap();
    let scale_min_f64 = U::to_f64(&scale_min).unwrap();
    let scale_max_f64 = U::to_f64(&scale_max).unwrap();
    // Within the rounding error, as CLOSURE accepts: 5.004 ± 0.005 on 1..=5
    // admits the sample of all 5s.
    if !(mean_f64 + re_mean >= scale_min_f64 && mean_f64 - re_mean <= scale_max_f64) {
        return Err(ParameterError::InputValidation(
            "Mean is outside the [scale_min, scale_max] range, even allowing for its \
             rounding error."
                .to_string(),
        ));
    }

    // Handle restrictions
    let restrictions_exact = restrictions_exact.unwrap_or_default();
    let restrictions_minimum = match restrictions_minimum {
        RestrictionsOption::Default => {
            Some(RestrictionsMinimum::from_range(scale_min_i32, scale_max_i32).extract())
        }
        RestrictionsOption::Opt(opt_map) => opt_map.map(|rm| rm.extract()),
        RestrictionsOption::Null => None,
    };

    if let Some(ref min_map) = restrictions_minimum {
        let constraints = OccurrenceConstraints::new(restrictions_exact.clone(), min_map.clone());
        if constraints.check_conflicts() {
            let exact_keys: HashSet<_> = constraints.exact.keys().collect();
            let min_keys: HashSet<_> = constraints.minimum.keys().collect();
            let conflict_keys: Vec<_> = exact_keys
                .intersection(&min_keys)
                .map(|&&k| k as f64 / 100.0)
                .collect();
            return Err(ParameterError::Conflict(format!(
                "Value(s) in both exact and minimum restrictions: {conflict_keys:?}"
            )));
        }
    }

    let n_usize = U::to_usize(&n).unwrap_or(0);
    if n_usize < 2 {
        return Err(ParameterError::InputValidation(
            "n must be at least 2".to_string(),
        ));
    }
    // Every attempt holds the n values of its sample, so n near `i32::MAX`
    // asked for gigabytes per thread and aborted when the allocation failed.
    if n_usize > crate::MAX_LISTING_N {
        return Err(ParameterError::InputValidation(format!(
            "n is {n_usize}, but SPRITE lists samples for n up to {}: each attempt holds \
             all n values. closure_count has no such limit",
            crate::MAX_LISTING_N
        )));
    }
    // Counted before the fixed responses are built: a count of `usize::MAX`
    // overflowed the vector's capacity, and one of 2^40 asked for terabytes.
    let n_fixed = restrictions_exact
        .values()
        .chain(restrictions_minimum.iter().flat_map(|m| m.values()))
        .try_fold(0usize, |sum, &count| sum.checked_add(count));
    if !matches!(n_fixed, Some(f) if f <= n_usize) {
        return Err(ParameterError::InputValidation(format!(
            "The restrictions fix more responses than n ({n_usize})"
        )));
    }

    // Internal values are the whole numbers from `scale_min * items` to
    // `scale_max * items`, the one at grid position `j` being
    // `scale_min_scaled + j`. Restriction keys are read off the output grid's
    // own labels, so a key names the value it labels in the results. Keys
    // computed here separately rounded values ending in exactly half a
    // hundredth the other way: at 40 items the grid labels 1.025 as 102,
    // but SPRITE rejected 102 and took 103 for it.
    let scale_min_scaled = scale_min_i32 as i64 * scale_factor as i64;
    let grid = <Sprite as SampleFormat<U>>::value_grid(scale_min, scale_max, items);
    let value_of_key: HashMap<i32, i64> = grid
        .values_hundredths()
        .iter()
        .enumerate()
        .map(|(j, &key)| (key, scale_min_scaled + j as i64))
        .collect();
    let to_u = |v: i64| -> U { NumCast::from(v).expect("checked to fit") };

    // Process fixed responses
    let mut fixed_responses_scaled: Vec<U> = Vec::new();
    let mut fixed_values: HashSet<i64> = HashSet::new();

    for (&key, &count) in &restrictions_exact {
        let Some(&v) = value_of_key.get(&key) else {
            return Err(ParameterError::InputValidation(format!(
                "Invalid key in restrictions_exact: {}",
                key as f64 / 100.0
            )));
        };
        fixed_values.insert(v);
        fixed_responses_scaled.extend(std::iter::repeat_n(to_u(v), count));
    }

    if let Some(min_map) = restrictions_minimum {
        for (&key, &count) in &min_map {
            let Some(&v) = value_of_key.get(&key) else {
                return Err(ParameterError::InputValidation(format!(
                    "Invalid key in restrictions_minimum: {}",
                    key as f64 / 100.0
                )));
            };
            fixed_responses_scaled.extend(std::iter::repeat_n(to_u(v), count));
        }
    }

    // The free values: every grid value but those exactly restricted.
    let final_possible_values_scaled: Vec<U> = (scale_min_scaled..scale_min_scaled + k)
        .filter(|v| !fixed_values.contains(v))
        .map(to_u)
        .collect();

    let n_fixed = fixed_responses_scaled.len();

    let value_to_index: HashMap<i64, usize> = final_possible_values_scaled
        .iter()
        .enumerate()
        .map(|(i, v)| (U::to_i64(v).unwrap(), i))
        .collect();

    let min_step = final_possible_values_scaled
        .windows(2)
        .map(|w| U::to_i64(&w[1]).unwrap() - U::to_i64(&w[0]).unwrap())
        .min()
        .unwrap_or(1);

    // The search runs on an integer sum and sum of squares in internal units.
    // Both `n·Σx²` and `(Σx)²` are at most `(n·max|x|)²`, so if that fits in
    // an i64 nothing overflows, and acceptance is exact. In f64, an offset
    // of 1e7 already returned samples whose SD missed the tolerance.
    let peak = scale_min_i32
        .unsigned_abs()
        .max(scale_max_i32.unsigned_abs()) as i128
        * items as i128;
    let fits = (n_usize as i128)
        .checked_mul(peak)
        .and_then(|x| x.checked_mul(x))
        .is_some_and(|x| x <= i64::MAX as i128);
    if !fits {
        return Err(ParameterError::InputValidation(
            "n and the scale values are too large for exact integer arithmetic".to_string(),
        ));
    }

    let to_i64 = |v: &U| U::to_i64(v).unwrap();
    let fixed_sum: i64 = fixed_responses_scaled.iter().map(to_i64).sum();
    let fixed_sum_sq: i64 = fixed_responses_scaled
        .iter()
        .map(|v| to_i64(v).pow(2))
        .sum();
    let mean_f64 = T::to_f64(&mean).unwrap_or(f64::NAN);
    let units = scale_factor as f64;
    let (re_mean, re_sd) = crate::widened_for_precision::<T>(mean_f64, sd_f64, re_mean, re_sd);
    let bounds = SearchBounds::new(
        mean_f64 * units,
        sd_f64 * units,
        n_usize,
        re_mean * units,
        re_sd * units,
    );
    let (sum_lo, sum_hi, infeasible) = feasibility(
        &bounds,
        n_usize,
        &final_possible_values_scaled,
        &fixed_responses_scaled,
        fixed_sum,
    );

    Ok(SpriteParams {
        n, // Changed from n_obs
        sum_lo,
        sum_hi,
        m2n_lo: bounds.m2n_lo,
        m2n_hi: bounds.m2n_hi,
        infeasible,
        fixed_sum,
        fixed_sum_sq,
        scale_min_scaled,
        possible_values_scaled: final_possible_values_scaled,
        fixed_responses_scaled,
        n_fixed,
        value_to_index,
        min_step,
    })
}

/// The admissible totals, and whether any attempt can succeed at all.
///
/// Every total within the mean tolerance that the free values can reach is
/// admissible, and the SD phase never changes the total, so each attempt aims
/// at one of them at random. Aiming every attempt at `round(mean * n)` left
/// all the other admissible totals unreachable.
///
/// An SD outside what any sample with an admissible total can reach used to
/// run every attempt to its loop limit, 16 s to return nothing for mean 3.5,
/// sd 0.1, n 20 on 1..=5. Both limits are exact in integers:
///
/// - Most spread: the `m` free values, whole numbers in `[lo, hi]` with free
///   total `T`, have the largest `Σx²` with as many as possible at `hi`, the
///   rest at `lo`, and at most one in between; the fixed responses add their
///   own squares. Values excluded by restrictions can only lower it, so this
///   is a valid bound. Within each run of `hi − lo` totals it is convex in
///   `S`, so every admissible total is checked, up to a million of them;
///   past that the Bhatia–Davis bound `(n·hi − S)(S − n·lo)` stands in, which
///   whole numbers reach only where every value can sit at an end. It alone
///   let mean 3.20, sd 2.15, n 5 on 1..=5 search for 87 ms to return nothing,
///   though the widest such sample has an SD of 2.05.
/// - Least spread: internal values are whole numbers, so at best the `m`
///   free values split between two neighbours, `T mod m` of them one above
///   the rest, for free total `T`; the fixed responses add their own squares.
///   Within each run of `m` totals that is concave in `S`, so its least lies
///   at an end of the range or where `T` is a multiple of `m`, and across
///   those points it is convex, lowest where the free values average what
///   the fixed ones do. Without fixed responses it is `r (n − r)` for
///   `r = S mod n`. Ignoring them let mean 3, sd 0.005, n 20,000 on 1..=5
///   with the default restrictions search for 6.6 s, though a 1 and a 5
///   among 3s already have an SD of 0.020.
fn feasibility<U: IntegerType>(
    bounds: &SearchBounds,
    n: usize,
    pv: &[U],
    fixed: &[U],
    fixed_sum: i64,
) -> (i64, i64, Option<&'static str>) {
    let r_n = (n - fixed.len()) as i64;
    let to_i64 = |v: &U| U::to_i64(v).unwrap();
    let (pv_min, pv_max) = match (pv.first(), pv.last()) {
        (Some(lo), Some(hi)) => (to_i64(lo), to_i64(hi)),
        _ if r_n > 0 => return (0, 0, Some("No possible values to sample from.")),
        _ => (0, 0),
    };
    let sum_lo = bounds.sum_lo.max(fixed_sum + r_n * pv_min);
    let sum_hi = bounds.sum_hi.min(fixed_sum + r_n * pv_max);
    if sum_lo > sum_hi {
        return (
            0,
            0,
            Some("No total within the mean tolerance is reachable."),
        );
    }

    let n = n as i128;
    let (s_lo, s_hi) = (sum_lo as i128, sum_hi as i128);
    let fixed_sum_sq: i128 = fixed.iter().map(|v| (to_i64(v) as i128).pow(2)).sum();
    let (fs, m) = (fixed_sum as i128, r_n as i128);
    let (plo, phi) = (pv_min as i128, pv_max as i128);
    let most_at = |s: i128| {
        let free_sq = if m == 0 {
            0
        } else {
            // `k` free values at the top, one at `plo + v` if `v > 0`, the
            // rest at the bottom. `s` is reachable, so `extra` is in
            // `0..=m·(phi − plo)` and `k <= m`.
            let (d, extra) = (phi - plo, s - fs - m * plo);
            let (k, v) = if d == 0 {
                (0, 0)
            } else {
                (extra / d, extra % d)
            };
            let between = (v > 0) as i128;
            k * phi * phi + (m - k - between) * plo * plo + between * (plo + v).pow(2)
        };
        n * (fixed_sum_sq + free_sq) - s * s
    };
    let most = if s_hi - s_lo <= 1_000_000 {
        (s_lo..=s_hi).map(most_at).max().unwrap()
    } else {
        let values = || pv.iter().chain(fixed).map(to_i64);
        let lo = values().min().unwrap_or(0) as i128;
        let hi = values().max().unwrap_or(0) as i128;
        let vertex = (n * (lo + hi)).div_euclid(2);
        [s_lo, s_hi, vertex, vertex + 1]
            .into_iter()
            .map(|s| s.clamp(s_lo, s_hi))
            .map(|s| (n * hi - s) * (s - n * lo))
            .max()
            .unwrap()
    };
    if (bounds.m2n_lo as i128) > most {
        return (0, 0, Some("No sample on the scale reaches the SD."));
    }
    let least_at = |s: i128| {
        let free_sq = if m == 0 {
            0
        } else {
            let (q, r) = ((s - fs).div_euclid(m), (s - fs).rem_euclid(m));
            m * q * q + r * (2 * q + 1)
        };
        n * (fixed_sum_sq + free_sq) - s * s
    };
    // Totals whose free part is a multiple of `m`, at or below `s`.
    let split = |s: i128| fs + (s - fs).div_euclid(m.max(1)) * m;
    let mut candidates = vec![s_lo, s_hi, split(s_lo + m - 1), split(s_hi)];
    if m > 0 && n > m {
        let even = split(fs + (fs * m).div_euclid(n - m));
        candidates.extend([even, even + m]);
    }
    let least = candidates
        .into_iter()
        .filter(|s| (s_lo..=s_hi).contains(s))
        .map(least_at)
        .min()
        .unwrap();
    if (bounds.m2n_hi as i128) < least {
        return (0, 0, Some("No sample on the scale is as narrow as the SD."));
    }
    (sum_lo, sum_hi, None)
}

impl<U: IntegerType> SpriteParams<U> {
    /// Whether no single-peaked sample meets the bounds, on a grid `k` wide;
    /// see [`crate::beyond_single_peak`]. Internal units are grid steps.
    fn beyond_single_peak(&self, k: usize) -> bool {
        // Infeasible inputs carry placeholder sums, not the admissible range.
        if self.infeasible.is_some() {
            return false;
        }
        let n = U::to_usize(&self.n).expect("n was validated");
        let shift = n as i64 * self.scale_min_scaled;
        crate::beyond_single_peak(self.m2n_lo, self.sum_lo - shift, self.sum_hi - shift, n, k)
    }

    /// Whether some sum of squares puts `n·Σx² − total²` inside the SD
    /// bounds. Since `x² ≡ x (mod 2)`, `Σx² ≡ total (mod 2)`, so at a fixed
    /// total that statistic moves in steps of `2n`, and a band narrower than
    /// that misses some totals altogether. An attempt aimed at one of them
    /// used to bounce across the band until the loop limit: mean 5000.01, sd
    /// 0.1, n 1000 on 0..=10000 took 25 s for 4 samples, against 0.2 s on
    /// 4995..=5005.
    fn total_can_hit_sd(&self, total: i64) -> bool {
        let n = U::to_i64(&self.n).expect("n was validated") as i128;
        let (total, s2) = (total as i128, (total as i128).pow(2));
        let q_lo = -(-(self.m2n_lo as i128 + s2)).div_euclid(n);
        let q_hi = (self.m2n_hi as i128 + s2).div_euclid(n);
        q_lo + (q_lo - total).rem_euclid(2) <= q_hi
    }

    /// Tabulate a sample in internal units onto a value grid `k` wide.
    ///
    /// Exact: internal units are grid steps, so a value's position is plain
    /// integer arithmetic with no rounding through hundredths.
    fn grid_counts(&self, sample: &[U], k: usize) -> Vec<u32> {
        let mut counts = vec![0u32; k];
        for v in sample {
            let idx = U::to_i64(v).unwrap() - self.scale_min_scaled;
            counts[usize::try_from(idx).expect("SPRITE values sit on the grid")] += 1;
        }
        counts
    }
}

/// SPRITE technique: sample parameter reconstruction via iterative techniques
pub struct Sprite {
    pub restrictions_exact: Option<HashMap<i32, usize>>,
    pub restrictions_minimum: RestrictionsOption,
}

impl<U: IntegerType> crate::SampleFormat<U> for Sprite {
    const TECHNIQUE: &'static str = "sprite";
    /// SPRITE emits values already scaled to hundredths of a scale point.
    const SAMPLE_SCALE_FACTOR: i32 = 1;
}

impl<T: FloatType, U: IntegerType + 'static> crate::Technique<T, U> for Sprite {
    fn run(
        &mut self,
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        items: u32,
        parquet_config: Option<ParquetConfig>,
        stop_after: Option<usize>,
        prominence: Option<ProminenceThresholds>,
    ) -> Result<ResultListFromMeanSdN<U>, ParameterError> {
        sprite_parallel(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            items,
            self.restrictions_exact.clone(),
            self.restrictions_minimum.clone(),
            parquet_config,
            stop_after,
            prominence,
        )
    }

    fn run_streaming(
        &mut self,
        mean: T,
        sd: T,
        n: U,
        scale_min: U,
        scale_max: U,
        rounding_error_mean: T,
        rounding_error_sd: T,
        items: u32,
        config: StreamingConfig,
        stop_after: Option<usize>,
        prominence: Option<ProminenceThresholds>,
    ) -> Result<StreamingResult, ParameterError> {
        sprite_parallel_streaming(
            mean,
            sd,
            n,
            scale_min,
            scale_max,
            rounding_error_mean,
            rounding_error_sd,
            items,
            self.restrictions_exact.clone(),
            self.restrictions_minimum.clone(),
            config,
            stop_after,
            prominence,
        )
    }
}

/// Main SPRITE API: Generate valid distributions matching summary statistics at random
///
/// This is the SPRITE equivalent of `closure_parallel()`. It finds multiple possible
/// distributions of raw data that match the given mean and standard deviation.
///
/// # Arguments (matching CLOSURE parameter order)
/// * `mean` - The target mean
/// * `sd` - The target standard deviation
/// * `n` - The number of observations (generic integer type)
/// * `scale_min` - The minimum value on the scale (generic integer type)
/// * `scale_max` - The maximum value on the scale (generic integer type).
///   The scale may have at most 100,000 grid values, counting
///   `(scale_max - scale_min) * items + 1`, because every sample is stored as
///   one count per grid value
/// * `rounding_error_mean` - A sample is accepted when its mean is within this
///   distance of `mean`, inclusive — the same rule CLOSURE applies
/// * `rounding_error_sd` - Likewise for the sample SD
/// * `items` - Number of items averaged (default 1)
/// * `restrictions_exact` - Optional exact count requirements for specific
///   values, keyed in hundredths of a scale point (`300` is the value 3)
/// * `restrictions_minimum` - Minimum count requirements, keyed the same way.
///   [`RestrictionsOption::Default`] requires at least one response at each end
///   of the scale; pass [`RestrictionsOption::Null`] for no minimum
/// * `parquet_config` - Optional configuration for writing results to Parquet files
/// * `stop_after` - Optional maximum number of distributions to find
/// * `prominence` - Cutoffs for the shape classification, as fractions of `n`.
///   `None` uses the defaults, which are conventions; see
///   [`ProminenceThresholds`]
///
/// # Returns
/// A `ResultListFromMeanSdN<U>` containing all distributions and comprehensive statistics
/// including horns metrics and frequency distributions. Samples are reported
/// in hundredths of a scale point, which round multi-item values when `items`
/// does not divide 100; see [`crate::ResultsTable::sample`].
///
/// # Example
/// ```no_run
/// use closure_core::{sprite_parallel, RestrictionsOption};
///
/// let results = sprite_parallel(
///     2.2_f64, 1.3_f64, 20, 1, 5,
///     0.05, 0.05, 1,
///     None, RestrictionsOption::Default,
///     None, Some(5), None,
/// ).unwrap();
/// ```
#[allow(clippy::too_many_arguments)]
pub fn sprite_parallel<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    items: u32,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    parquet_config: Option<ParquetConfig>,
    stop_after: Option<usize>,
    prominence: Option<ProminenceThresholds>,
) -> Result<ResultListFromMeanSdN<U>, ParameterError>
where
    T: FloatType,
    U: IntegerType + 'static,
{
    let thresholds = crate::checked_thresholds(prominence)?;
    // Build and validate parameters
    let params = build_sprite_params(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
        items,
        restrictions_exact,
        restrictions_minimum,
    )?;

    // Tabulate onto the shared value grid. SPRITE searches on its own
    // `scale_factor` lattice; the grid is the same lattice expressed in
    // hundredths of a scale point, which is the unit SPRITE reports in.
    //
    // Counting straight onto the grid also drops the old round-trip through
    // integer-binned values: horns and the frequency tables are now computed at
    // the grid's real resolution, which for a multi-item scale is finer than a
    // whole scale point.
    let grid = <Sprite as SampleFormat<U>>::value_grid(scale_min, scale_max, items);
    let beyond_single_peak = params.beyond_single_peak(grid.len());
    let counts = find_distributions_all_internal(&params, grid, stop_after);

    // Calculate all statistics using the shared function from lib.rs SPRITE
    // searches the solution space at random rather than enumerating it, so a
    // shape it never generated has not been ruled out.
    let sprite_results: ResultListFromMeanSdN<U> =
        counts_to_result_list(counts, false, &thresholds, beyond_single_peak);

    // Write to Parquet if configured
    if let Some(config) = parquet_config {
        <Sprite as SampleFormat<U>>::write_result_list(
            &crate::normalize_base_path(&config.file_path),
            &sprite_results,
            &config,
        )
        .map_err(crate::output_error)?;
    }

    Ok(sprite_results)
}

/// SPRITE streaming API: Generate distributions and stream to Parquet files
///
/// This is the SPRITE equivalent of `closure_parallel_streaming()`. It finds distributions
/// matching the given mean and SD and streams them directly to Parquet files without
/// keeping all results in memory.
///
/// Use this when:
/// - Result sets are very large (> 1GB in memory)
/// - You only need file output, not in-memory processing
/// - Memory efficiency is critical
///
/// # Arguments (matching CLOSURE parameter order)
/// * `mean` - The target mean
/// * `sd` - The target standard deviation
/// * `n` - The number of observations (generic integer type)
/// * `scale_min` - The minimum value on the scale (generic integer type)
/// * `scale_max` - The maximum value on the scale (generic integer type).
///   The scale may have at most 100,000 grid values, counting
///   `(scale_max - scale_min) * items + 1`, because every sample is stored as
///   one count per grid value
/// * `rounding_error_mean` - A sample is accepted when its mean is within this
///   distance of `mean`, inclusive — the same rule CLOSURE applies
/// * `rounding_error_sd` - Likewise for the sample SD
/// * `items` - Number of items averaged (default 1)
/// * `restrictions_exact` - Optional exact count requirements for specific
///   values, keyed in hundredths of a scale point (`300` is the value 3)
/// * `restrictions_minimum` - Minimum count requirements, keyed the same way.
///   [`RestrictionsOption::Default`] requires at least one response at each end
///   of the scale; pass [`RestrictionsOption::Null`] for no minimum
/// * `config` - Streaming configuration (file path, batch size, progress reporting)
/// * `stop_after` - Optional limit on number of distributions to find
/// * `prominence` - Cutoffs for the shape classification, as fractions of `n`.
///   `None` uses the defaults, which are conventions; see
///   [`ProminenceThresholds`]
///
/// # Returns
/// A `Result<StreamingResult, ParameterError>` with the total count and file path.
#[allow(clippy::too_many_arguments)]
pub fn sprite_parallel_streaming<T, U>(
    mean: T,
    sd: T,
    n: U,
    scale_min: U,
    scale_max: U,
    rounding_error_mean: T,
    rounding_error_sd: T,
    items: u32,
    restrictions_exact: Option<HashMap<i32, usize>>,
    restrictions_minimum: RestrictionsOption,
    config: StreamingConfig,
    stop_after: Option<usize>,
    prominence: Option<ProminenceThresholds>,
) -> Result<StreamingResult, ParameterError>
where
    T: FloatType,
    U: IntegerType + 'static,
{
    let thresholds = crate::checked_thresholds(prominence)?;
    use crate::write_streaming_statistics;
    use std::collections::HashMap;

    // Build and validate parameters
    let params = build_sprite_params(
        mean,
        sd,
        n,
        scale_min,
        scale_max,
        rounding_error_mean,
        rounding_error_sd,
        items,
        restrictions_exact,
        restrictions_minimum,
    )?;

    let n_usize = U::to_usize(&params.n).unwrap();

    // Setup channels for streaming results
    let (tx_results, rx_results) = channel::<Vec<(Vec<u32>, f64)>>();
    let (tx_stats, rx_stats) = channel::<(Vec<f64>, HashMap<i32, i64>)>();

    // Add a flag to track writer thread status
    let writer_failed = Arc::new(AtomicUsize::new(0)); // 0 = ok, 1 = failed
    let writer_failed_for_compute = writer_failed.clone();
    let writer_failed_for_thread = writer_failed.clone();

    // Counter for total distributions found
    let total_counter = Arc::new(AtomicUsize::new(0));

    // The value grid every result in this run is indexed on.
    let grid = <Sprite as SampleFormat<U>>::value_grid(scale_min, scale_max, items);

    // Shared state for tracking min/max horns frequencies
    let freq_state = Arc::new(Mutex::new(StreamingFrequencyState::new(
        grid.len(),
        n_usize,
        &thresholds,
    )));
    let freq_state_for_thread = freq_state.clone();

    // One directory per run, as in memory mode.
    let base_path = crate::prepare_output_dir(&config.file_path)?;

    // Spawn the shared writer thread. It owns the output format; this path
    // only feeds it (counts, horns) pairs.
    let writer_handle = <Sprite as SampleFormat<U>>::spawn_streaming_writer(
        base_path.clone(),
        grid.clone(),
        n_usize,
        config.batch_size,
        config.format,
        config.show_progress.then_some((1000, "distributions")),
        rx_results,
        writer_failed_for_thread,
    );

    // Spawn statistics collector thread
    let freq_state_for_stats = freq_state.clone();
    let stats_handle = thread::spawn(move || {
        let mut all_horns = Vec::new();

        while let Ok((horns_batch, _)) = rx_stats.recv() {
            all_horns.extend(horns_batch);
        }

        (all_horns, freq_state_for_stats)
    });

    // Find distributions using streaming approach
    find_distributions_all_streaming(
        &params,
        &grid,
        stop_after,
        tx_results,
        tx_stats,
        &total_counter,
        &freq_state_for_thread,
        &writer_failed_for_compute,
        &config,
    );

    // Wait for threads to complete
    let total_written = crate::join_writer(writer_handle)?;

    let (all_horns, final_freq_state) = stats_handle
        .join()
        .map_err(|_| crate::output_error("the statistics thread panicked"))?;

    // Write statistics files, even when nothing was found, so a reader always
    // finds a complete result set. SPRITE never enumerates the space
    // exhaustively.
    write_streaming_statistics(
        &base_path,
        &all_horns,
        n_usize,
        &grid,
        final_freq_state,
        false,
        &thresholds,
        params.beyond_single_peak(grid.len()),
    )?;

    Ok(StreamingResult {
        total_combinations: total_written,
        file_path: config.file_path,
    })
}

/// Find distributions using streaming approach (parallelized with rayon)
#[allow(clippy::too_many_arguments)]
fn find_distributions_all_streaming<U>(
    params: &SpriteParams<U>,
    grid: &ValueGrid,
    stop_after: Option<usize>,
    tx_results: std::sync::mpsc::Sender<Vec<(Vec<u32>, f64)>>,
    tx_stats: std::sync::mpsc::Sender<(Vec<f64>, HashMap<i32, i64>)>,
    total_counter: &Arc<AtomicUsize>,
    freq_state: &Arc<Mutex<StreamingFrequencyState>>,
    writer_failed: &Arc<AtomicUsize>,
    config: &StreamingConfig,
) where
    U: IntegerType + 'static,
{
    use crate::horns_from_counts;

    let should_stop = Arc::new(AtomicBool::new(false));
    // Keyed by grid counts, as in `find_distributions_all_internal`.
    let unique_distributions = Arc::new(Mutex::new(HashSet::<Box<[u32]>>::new()));
    let total_failures = Arc::new(AtomicU32::new(0));
    let total_duplicates = Arc::new(AtomicU32::new(0));
    let cumulative_duplicates = Arc::new(AtomicU64::new(0));

    let batch_size = 100;
    let max_iterations = attempt_cap(stop_after);

    for batch_start in (0..max_iterations).step_by(batch_size) {
        if should_stop.load(Ordering::Relaxed) || writer_failed.load(Ordering::Relaxed) == 1 {
            break;
        }

        {
            let unique = unique_distributions.lock().unwrap();
            if let Some(limit) = stop_after {
                if unique.len() >= limit {
                    break;
                }
            }
        }

        let batch_end = (batch_start + batch_size).min(max_iterations);
        let batch_range = batch_start..batch_end;

        batch_range.into_par_iter().for_each(|_| {
            if should_stop.load(Ordering::Relaxed) {
                return;
            }

            if let Some(limit) = stop_after {
                if total_counter.load(Ordering::Relaxed) >= limit {
                    should_stop.store(true, Ordering::Relaxed);
                    return;
                }
            }

            let mut thread_rng = rand::rng();

            match find_distribution_internal(params, &mut thread_rng) {
                Ok(distribution) => {
                    // Tabulate onto the shared grid. The counts vector is the
                    // deduplication key, the frequency vector horns needs, and
                    // the row that goes to disk, so it is built once.
                    let counts = params.grid_counts(&distribution, grid.len());

                    // Checked under the same lock as the insert, so no more
                    // than `limit` samples are ever admitted, however many
                    // attempts in the batch succeed at once.
                    let mut unique = unique_distributions.lock().unwrap();
                    if stop_after.is_some_and(|limit| unique.len() >= limit) {
                        should_stop.store(true, Ordering::Relaxed);
                        return;
                    }
                    if unique.insert(counts.clone().into_boxed_slice()) {
                        drop(unique);

                        let freqs: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
                        let horns = horns_from_counts(&freqs);

                        freq_state.lock().unwrap().update(&counts, horns);

                        // Send to writer
                        if tx_results.send(vec![(counts, horns)]).is_ok() {
                            let _ = tx_stats.send((vec![horns], HashMap::new()));
                            total_counter.fetch_add(1, Ordering::Relaxed);
                        }

                        total_failures.store(0, Ordering::Relaxed);
                        total_duplicates.store(0, Ordering::Relaxed);
                    } else {
                        total_duplicates.fetch_add(1, Ordering::Relaxed);
                        cumulative_duplicates.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(_) => {
                    total_failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        // Check early stopping conditions
        let failures = total_failures.load(Ordering::Relaxed);
        let cum_dup = cumulative_duplicates.load(Ordering::Relaxed);

        if failures >= 1000 {
            if config.show_progress {
                eprintln!(
                    "Warning: Too many failed attempts ({}). Stopping search.",
                    failures
                );
            }
            should_stop.store(true, Ordering::Relaxed);
            break;
        }

        // Lincoln-Petersen coverage estimate (same formula as non-streaming path):
        // duplicate_rate = c_dup / total_draws converges to n_found / N_total under
        // uniform sampling, so it directly estimates the fraction of the space covered.
        let n_found = unique_distributions.lock().unwrap().len() as f64;
        let total_draws = n_found + cum_dup as f64;
        if n_found > 0.0 && total_draws >= MIN_DRAWS as f64 {
            let duplicate_rate = cum_dup as f64 / total_draws;
            if duplicate_rate >= LP_COVERAGE_THRESHOLD {
                if config.show_progress {
                    eprintln!(
                        "Warning: Duplicate rate {:.1}%: the distributions SPRITE reaches look exhausted ({} unique). Stopping search.",
                        duplicate_rate * 100.0,
                        n_found as usize
                    );
                }
                should_stop.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
}

/// Internal function to find multiple distributions (parallelized with rayon)
fn find_distributions_all_internal<U>(
    params: &SpriteParams<U>,
    grid: ValueGrid,
    stop_after: Option<usize>,
) -> SampleCounts
where
    U: IntegerType,
{
    let k = grid.len();
    // All state lives on the main thread; only the early-exit signal is shared.
    // Distributions are deduplicated by their counts on the grid, not as
    // sorted samples: the two identify a distribution equally, but counts take
    // `k` values rather than `n`.
    let mut unique_distributions = HashSet::<Box<[u32]>>::new();
    let mut results = SampleCounts::new(grid, U::to_usize(&params.n).unwrap());
    let mut total_failures: u32 = 0;
    // Cumulative duplicates across all batches (never reset); used by LP estimator.
    let mut cumulative_duplicates: u64 = 0;
    let should_stop = Arc::new(AtomicBool::new(false));

    // Process in batches for better control and early termination
    let batch_size = 100;
    let n_distributions = stop_after.unwrap_or(usize::MAX);
    let max_iterations = attempt_cap(stop_after);

    for batch_start in (0..max_iterations).step_by(batch_size) {
        if should_stop.load(Ordering::Relaxed) || results.nrow() >= n_distributions {
            break;
        }

        let batch_end = (batch_start + batch_size).min(max_iterations);

        // Parallel phase: pure computation, zero shared-state locking.
        // Items where should_stop is already set are skipped.
        let batch_results: Vec<Result<Vec<U>, String>> = (batch_start..batch_end)
            .into_par_iter()
            .filter(|_| !should_stop.load(Ordering::Relaxed))
            .map(|_| {
                let mut thread_rng = rand::rng();
                find_distribution_internal(params, &mut thread_rng)
            })
            .collect();

        // Serial phase: merge results into local state — no locking needed.
        for result in batch_results {
            match result {
                Ok(distribution) => {
                    let counts = params.grid_counts(&distribution, k).into_boxed_slice();
                    if unique_distributions.insert(counts.clone()) {
                        results.push_row(&counts);
                        total_failures = 0;
                    } else {
                        cumulative_duplicates += 1;
                    }
                }
                Err(_) => {
                    total_failures += 1;
                }
            }
            if results.nrow() >= n_distributions {
                should_stop.store(true, Ordering::Relaxed);
                break;
            }
        }

        // Check early stopping conditions after each batch
        if total_failures >= 1000 {
            break;
        }

        // Lincoln-Petersen coverage estimate:
        // Under uniform sampling, the duplicate rate c_dup / total_draws converges to
        // n_found / N_total, so it directly estimates the fraction of the space covered.
        // Stop when duplicate_rate >= LP_COVERAGE_THRESHOLD (default 0.99).
        let n_found = unique_distributions.len() as f64;
        let total_draws = n_found + cumulative_duplicates as f64;
        if n_found > 0.0
            && total_draws >= MIN_DRAWS as f64
            && cumulative_duplicates as f64 / total_draws >= LP_COVERAGE_THRESHOLD
        {
            break;
        }
    }

    results
}

/// Most steps one attempt takes to adjust the mean, and most shifts toward
/// the SD, for `n` values on `k` free grid values.
///
/// Each shift moves two values one grid step, so walking the sample across
/// the grid takes on the order of `n·k` shifts. The SD limit is `n·k²`
/// between 20,000 and the larger of 1,000,000 and `n·k`. A fixed 1,000,000
/// defeated fine, wide grids: mean 16, sd 13, n 1000 on 1..=31 at 100 items
/// needs over a million shifts from its direct start, and no attempt got
/// there. Typical Likert inputs stay at 20,000.
///
/// Computed in u64 and saturated to u32: `n·k²` in u32 wrapped on wide
/// multi-item grids (n = 50,000 on 0..=100 at items = 3 is 4.5e9).
fn loop_limits(n: usize, k: usize) -> (u32, u32) {
    let (n, k) = (n as u64, k as u64);
    let n_k = n.saturating_mul(k);
    let upper = n_k.max(MAX_DELTA_LOOPS_UPPER as u64);
    let sd = n_k
        .saturating_mul(k)
        .clamp(MAX_DELTA_LOOPS_LOWER as u64, upper);
    let to_u32 = |x: u64| x.min(u32::MAX as u64) as u32;
    (to_u32(n_k), to_u32(sd))
}

/// Internal function to find a single distribution
fn find_distribution_internal<U>(
    params: &SpriteParams<U>,
    rng: &mut impl Rng,
) -> Result<Vec<U>, String>
where
    U: IntegerType,
{
    if let Some(reason) = params.infeasible {
        return Err(reason.to_string());
    }
    let n_usize = U::to_usize(&params.n).unwrap();
    let r_n = n_usize - params.n_fixed;
    let pv = &params.possible_values_scaled;

    // Draw among the totals the SD band can be reached from. Each check is
    // O(1); an attempt that misses 1,000 times fails like a doomed one did.
    let total_target_sum = (0..1_000)
        .map(|_| rng.random_range(params.sum_lo..=params.sum_hi))
        .find(|&total| params.total_can_hit_sd(total))
        .ok_or("No admissible total can reach the SD")?;
    let vec_target_sum = total_target_sum - params.fixed_sum;

    // Direct mean initialization (O(n)): fill the r_n free slots with the
    // grid value just below their target mean and raise exactly enough of them
    // to the next one that the total is hit exactly. One attempt in ten
    // starts from a random fill instead. For a given total the direct start
    // is always the same, and the shifts below cannot reach every sample
    // from it: with 1.67 and 2 fixed on a three-item 1..=3 scale, free
    // values `2.33 2.33 2.33` never become `1.33 2.67 3`. Not more often:
    // walking a random fill to the SD costs up to ~50 direct starts on
    // large samples (mean 3, sd 0.3, n 1000).
    let slot_mean = vec_target_sum as f64 / r_n.max(1) as f64;
    let floor_idx = pv
        .partition_point(|v| (U::to_i64(v).unwrap() as f64) <= slot_mean + 1e-9)
        .saturating_sub(1);

    let mut vec: Vec<U> = if !pv.is_empty() && r_n > 0 {
        let floor_i64 = U::to_i64(&pv[floor_idx]).unwrap();
        let direct: Option<Vec<U>> = if rng.random_bool(0.1) {
            None
        } else if floor_idx + 1 < pv.len() {
            let ceil_i64 = U::to_i64(&pv[floor_idx + 1]).unwrap();
            let step = ceil_i64 - floor_i64;
            let deficit = vec_target_sum - r_n as i64 * floor_i64;
            if step > 0 && deficit >= 0 && deficit % step == 0 {
                let n_ceil = (deficit / step) as usize;
                if n_ceil <= r_n {
                    let ceil_val = pv[floor_idx + 1];
                    let floor_val = pv[floor_idx];
                    let mut v: Vec<U> = (0..r_n)
                        .map(|k| if k < n_ceil { ceil_val } else { floor_val })
                        .collect();
                    v.shuffle(rng);
                    Some(v)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            // Mean is at the maximum; all elements must equal floor_val
            if vec_target_sum == r_n as i64 * floor_i64 {
                Some(vec![pv[floor_idx]; r_n])
            } else {
                None
            }
        };
        direct.unwrap_or_else(|| {
            // Random fill: also the fallback for edge cases from restrictions
            (0..r_n).map(|_| *pv.choose(rng).unwrap()).collect()
        })
    } else {
        Vec::new()
    };

    // Mean correction: no-op on the direct path (sum is exact); brings the
    // random fill to an admissible total.
    let (max_loops_mean, max_loops_sd) = loop_limits(n_usize, pv.len());
    adjust_mean_internal(&mut vec, params, max_loops_mean, rng)?;

    // Running sums for the mutable part of the distribution. Shifts keep the
    // total, so only the sum of squares runs.
    let total = vec.iter().map(|v| U::to_i64(v).unwrap()).sum::<i64>() + params.fixed_sum;
    // A random fill's mean correction can land on any admissible total.
    if !params.total_can_hit_sd(total) {
        return Err("The total reached cannot reach the SD".to_string());
    }
    let mut running_sum_sq: i64 = vec.iter().map(|v| U::to_i64(v).unwrap().pow(2)).sum();
    let n_i64 = n_usize as i64;

    // Once `2 n` random tries find no move, useful moves are rare, and the
    // attempt switches to a scan over every pair of values for good: it finds
    // a move whenever one exists, and costs less than random tries that
    // mostly miss. Giving up after a few fruitless random shifts instead lost
    // samples near the widest SD, where the last useful move can be one of
    // about `2 n²`: mean 3, sd 2, n 1000 on 1..=5 found none of its 23.
    let mut scanning = false;
    // Shifts since the distance to the SD band last shrank. With rare moves,
    // a total that cannot hit the band makes the attempt bounce across it:
    // mean 3, sd 0.08, n 300 on 1..=5 took minutes to find its one sample.
    let mut best_distance = i64::MAX;
    let mut stuck = 0;
    for _ in 1..=max_loops_sd {
        // `n·Σx² − (Σx)²` is exact, and `build_sprite_params` made sure it
        // fits in an i64.
        let m2n = n_i64 * (running_sum_sq + params.fixed_sum_sq) - total * total;
        if m2n >= params.m2n_lo && m2n <= params.m2n_hi {
            // Success - combine vec with fixed responses
            let mut full_vec = vec;
            full_vec.extend_from_slice(&params.fixed_responses_scaled);
            return Ok(full_vec);
        }
        // Shifting needs two free values; with fewer the SD cannot move.
        if vec.len() < 2 {
            break;
        }
        let increase_sd = m2n < params.m2n_lo;
        let distance = if increase_sd {
            params.m2n_lo - m2n
        } else {
            m2n - params.m2n_hi
        };
        if distance < best_distance {
            best_distance = distance;
            stuck = 0;
        } else if scanning {
            stuck += 1;
            if stuck == 10 {
                break;
            }
        }

        // Shift values to adjust SD (updates running sums in O(1)). Every
        // shift keeps the total, so the mean needs no correction afterwards.
        let moved = (!scanning
            && shift_values_internal(&mut vec, params, increase_sd, &mut running_sum_sq, rng))
            || {
                scanning = true;
                shift_values_exhaustive(&mut vec, params, increase_sd, &mut running_sum_sq, rng)
            };
        if !moved {
            break;
        }
    }

    // Failed to find a solution
    Err("Could not find a matching distribution within iteration limit".to_string())
}

/// Adjust mean of scaled integer values
fn adjust_mean_internal<U>(
    vec: &mut [U],
    params: &SpriteParams<U>,
    max_iter: u32,
    rng: &mut impl Rng,
) -> Result<(), String>
where
    U: IntegerType,
{
    // With no free values `vec` is empty too, and the loop below reports
    // whether the fixed responses alone already hit the mean. Failing here
    // first rejected every sample whose values are all exactly restricted.
    if params.possible_values_scaled.is_empty() && !vec.is_empty() {
        return Err("Cannot adjust mean with no possible values.".to_string());
    }

    let mut total: i64 = vec.iter().map(|v| U::to_i64(v).unwrap()).sum::<i64>() + params.fixed_sum;
    let mean_ok = |total: i64| (params.sum_lo..=params.sum_hi).contains(&total);

    for _ in 0..max_iter {
        if mean_ok(total) {
            return Ok(());
        }
        // Every response is fixed by the restrictions: nothing can move.
        if vec.is_empty() {
            break;
        }

        let increase_mean = total < params.sum_lo;
        let min_poss_val = params.possible_values_scaled[0];
        let max_poss_val = params.possible_values_scaled[params.possible_values_scaled.len() - 1];

        let max_attempts = vec.len().max(1) * 4;
        let mut changed = false;

        for _ in 0..max_attempts {
            let index_to_try = rng.random_range(0..vec.len());
            let current_val = vec[index_to_try];

            let is_valid_to_bump = if increase_mean {
                U::to_i64(&current_val).unwrap() < U::to_i64(&max_poss_val).unwrap()
            } else {
                U::to_i64(&current_val).unwrap() > U::to_i64(&min_poss_val).unwrap()
            };

            if is_valid_to_bump {
                if let Some(&pos) = params.value_to_index.get(&U::to_i64(&current_val).unwrap()) {
                    let new_pos = if increase_mean {
                        pos + 1
                    } else {
                        pos.saturating_sub(1)
                    };
                    if let Some(&new_val) = params.possible_values_scaled.get(new_pos) {
                        let old_i64 = U::to_i64(&current_val).unwrap();
                        let new_i64 = U::to_i64(&new_val).unwrap();
                        total += new_i64 - old_i64;
                        vec[index_to_try] = new_val;
                        changed = true;
                        break;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
    // The loop checks before each bump, so the last bump is checked here, as
    // is a vector with nothing free, whose limit `n * k` is 0.
    if mean_ok(total) {
        return Ok(());
    }

    let err_msg = if !params.fixed_responses_scaled.is_empty() {
        "Couldn't initialize data with correct mean. This *might* be because the restrictions cannot be satisfied."
    } else {
        "Couldn't initialize data with correct mean. This might indicate a coding error if the mean is in range."
    };
    Err(err_msg.to_string())
}

/// Shift values to adjust standard deviation
fn shift_values_internal<U>(
    vec: &mut [U],
    params: &SpriteParams<U>,
    increase_sd: bool,
    running_sum_sq: &mut i64,
    rng: &mut impl Rng,
) -> bool
where
    U: IntegerType,
{
    let max_attempts = vec.len() * 2;
    let pv = &params.possible_values_scaled;
    let index_of = |v: i64| params.value_to_index.get(&v).copied();
    // Every move keeps the total, so the SD goes the way the sum of squares
    // does, and that is exact in integers.
    let helps = |d_sq: i64| if increase_sd { d_sq > 0 } else { d_sq < 0 };

    for _ in 0..max_attempts {
        let i = rng.random_range(0..vec.len());
        let val1_i64 = U::to_i64(&vec[i]).unwrap();
        let Some(val1_idx) = index_of(val1_i64) else {
            continue;
        };

        // Which way val1 moves to its neighbour on the grid; the compensating
        // moves go the other way. Picked at random, as rsprite2 does with
        // `incFirst`. Always moving val1 up meant a gap left by an exact
        // restriction could only be crossed downward together with an upward
        // move just as wide, which is the swap back, so the value below the
        // gap never filled.
        let up = rng.random_bool(0.5);
        let neighbour = if up {
            pv.get(val1_idx + 1)
        } else {
            val1_idx.checked_sub(1).and_then(|k| pv.get(k))
        };
        let Some(&val1_new) = neighbour else {
            continue;
        };
        let val1_new_i64 = U::to_i64(&val1_new).unwrap();
        let step = val1_new_i64 - val1_i64;
        let d1_sq = val1_new_i64 * val1_new_i64 - val1_i64 * val1_i64;

        // Stage 1: another value makes the whole compensating move, to any
        // grid value however many grid steps away. On an even grid this is
        // the plain mean-preserving swap. Only moving between neighbours, a
        // gap from a restriction trapped the search: with 2 excluded on 1..=5,
        // `3 3 5` could not become `1 5 5`.
        let k = rng.random_range(0..vec.len() - 1);
        let k = if k >= i { k + 1 } else { k };
        let from = U::to_i64(&vec[k]).unwrap();
        let to = from - step;
        if let Some(to_idx) = index_of(to) {
            let d_sq = d1_sq + to * to - from * from;
            if helps(d_sq) {
                vec[i] = val1_new;
                vec[k] = pv[to_idx];
                *running_sum_sq += d_sq;
                return true;
            }
        }
        if step.abs() < 2 * params.min_step {
            continue; // too small to split
        }

        // Stage 2: val1 crossed a gap. Split the compensation over `parts`
        // other values, each moving `step / parts` the other way, scanning
        // from a random position so no part of `vec` is favoured.
        let start = rng.random_range(0..vec.len());
        for parts in 2..=(step.abs() / params.min_step).min(10) {
            if step % parts != 0 {
                continue;
            }
            let sub = step / parts;
            let mut moves: Vec<(usize, usize)> = Vec::with_capacity(parts as usize);
            let mut d_sq = d1_sq;
            for k in (start..vec.len()).chain(0..start) {
                if k == i {
                    continue;
                }
                let from = U::to_i64(&vec[k]).unwrap();
                let to = from - sub;
                if let Some(to_idx) = index_of(to) {
                    moves.push((k, to_idx));
                    d_sq += to * to - from * from;
                    if moves.len() == parts as usize {
                        break;
                    }
                }
            }
            if moves.len() < parts as usize || !helps(d_sq) {
                continue;
            }
            vec[i] = val1_new;
            for (k, to_idx) in moves {
                vec[k] = pv[to_idx];
            }
            // The total is unchanged: `step` one way, `parts` times `sub` back.
            *running_sum_sq += d_sq;
            return true;
        }
    }

    false
}

/// Find a move [`shift_values_internal`] missed, or establish that none is
/// left.
///
/// The same moves, taken over the values present rather than at random
/// positions: a stage 1 move chosen at random among all that help, failing
/// that the stage 2 split whose parts help most. With `P` distinct values
/// present this costs `O(n + P²)`.
// ponytail: quadratic in distinct values, which is at most `min(n, k)`; fine
// on Likert grids, revisit if wide multi-item grids with huge n get slow.
fn shift_values_exhaustive<U>(
    vec: &mut [U],
    params: &SpriteParams<U>,
    increase_sd: bool,
    running_sum_sq: &mut i64,
    rng: &mut impl Rng,
) -> bool
where
    U: IntegerType,
{
    let pv = &params.possible_values_scaled;
    let val = |j: usize| U::to_i64(&pv[j]).unwrap();
    let index_of = |v: i64| params.value_to_index.get(&v).copied();
    let helps = |d_sq: i64| if increase_sd { d_sq > 0 } else { d_sq < 0 };
    // Grid neighbours of `a`, in both directions.
    let neighbours = |a: usize| [a.checked_sub(1), Some(a + 1).filter(|&j| j < pv.len())];

    let mut count = vec![0usize; pv.len()];
    for v in vec.iter() {
        count[index_of(U::to_i64(v).unwrap()).expect("free values sit on the grid")] += 1;
    }
    let present: Vec<usize> = (0..pv.len()).filter(|&j| count[j] > 0).collect();

    // Stage 1: one value to a neighbour, another the same distance back.
    let mut chosen = None;
    let mut seen = 0u64;
    for &a in &present {
        for a_new in neighbours(a).into_iter().flatten() {
            let step = val(a_new) - val(a);
            let d1 = val(a_new).pow(2) - val(a).pow(2);
            for &b in &present {
                if b == a && count[a] < 2 {
                    continue;
                }
                let Some(b_new) = index_of(val(b) - step) else {
                    continue;
                };
                let d_sq = d1 + val(b_new).pow(2) - val(b).pow(2);
                if helps(d_sq) {
                    // Reservoir sampling: each helpful move equally likely.
                    seen += 1;
                    if rng.random_range(0..seen) == 0 {
                        chosen = Some((a, a_new, b, b_new, d_sq));
                    }
                }
            }
        }
    }
    if let Some((a, a_new, b, b_new, d_sq)) = chosen {
        apply_move(vec, pv, (a, a_new), &[(b, b_new, 1)]);
        *running_sum_sq += d_sq;
        return true;
    }

    // Stage 2: a step across a gap, compensated in `parts` smaller steps by
    // the copies whose moves help most.
    for &a in &present {
        for a_new in neighbours(a).into_iter().flatten() {
            let step = val(a_new) - val(a);
            if step.abs() < 2 * params.min_step {
                continue;
            }
            let d1 = val(a_new).pow(2) - val(a).pow(2);
            for parts in 2..=(step.abs() / params.min_step).min(10) {
                if step % parts != 0 {
                    continue;
                }
                let sub = step / parts;
                let mut candidates: Vec<(i64, usize, usize, usize)> = present
                    .iter()
                    .filter_map(|&j| {
                        let copies = count[j] - (j == a) as usize;
                        let j_new = index_of(val(j) - sub)?;
                        (copies > 0).then(|| (val(j_new).pow(2) - val(j).pow(2), j, j_new, copies))
                    })
                    .collect();
                if increase_sd {
                    candidates.sort_by_key(|c| std::cmp::Reverse(c.0));
                } else {
                    candidates.sort_by_key(|c| c.0);
                }
                let mut need = parts as usize;
                let mut d_sq = d1;
                let mut moves = Vec::new();
                for (gain, j, j_new, copies) in candidates {
                    let take = copies.min(need);
                    d_sq += gain * take as i64;
                    moves.push((j, j_new, take));
                    need -= take;
                    if need == 0 {
                        break;
                    }
                }
                if need == 0 && helps(d_sq) {
                    apply_move(vec, pv, (a, a_new), &moves);
                    *running_sum_sq += d_sq;
                    return true;
                }
            }
        }
    }
    false
}

/// Move one copy of `pv[first.0]` to `pv[first.1]`, then for each
/// `(j, j_new, copies)` that many other values from `pv[j]` to `pv[j_new]`.
/// A value moves at most once, so the first move is never undone.
fn apply_move<U: IntegerType>(
    vec: &mut [U],
    pv: &[U],
    first: (usize, usize),
    moves: &[(usize, usize, usize)],
) {
    let mut moved = vec![false; vec.len()];
    let i = vec
        .iter()
        .position(|v| *v == pv[first.0])
        .expect("the value is present");
    vec[i] = pv[first.1];
    moved[i] = true;
    for &(j, j_new, copies) in moves {
        let mut left = copies;
        for (k, v) in vec.iter_mut().enumerate() {
            if left == 0 {
                break;
            }
            if !moved[k] && *v == pv[j] {
                *v = pv[j_new];
                moved[k] = true;
                left -= 1;
            }
        }
    }
}

/// Calculate mean of a slice
#[cfg(test)]
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation of a slice
#[cfg(test)]
fn std_dev(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }
    let data_mean = mean(data);
    let variance =
        data.iter().map(|&v| (v - data_mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    Some(variance.sqrt())
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::OutputFormat;

    /// The acceptance rule, as a test would state it: inclusive, with the
    /// same float slack the search uses.
    fn assert_within(actual: f64, target: f64, tolerance: f64, what: &str) {
        assert!(
            (actual - target).abs() <= tolerance + DUST,
            "{what} {actual} is not within {tolerance} of {target}"
        );
    }

    /// Helper to convert scaled integers to floats
    fn unscale_distribution(scaled_values: &[i32], scale_factor: u32) -> Vec<f64> {
        scaled_values
            .iter()
            .map(|&v| v as f64 / scale_factor as f64)
            .collect()
    }

    #[test]
    fn sprite_test_mean() {
        let results = sprite_parallel(
            2.2_f64, // mean
            1.3_f64, // sd
            20,      // n (generic integer, inferred as i32)
            1,       // scale_min
            5,       // scale_max
            0.05,    // rounding_error_mean (precision 1: 0.1^1/2 = 0.05)
            0.05,    // rounding_error_sd (precision 1)
            1,       // items
            None,    // restrictions_exact
            RestrictionsOption::Default,
            None, // parquet_config
            Some(5),
            None, // stop_after (was n_distributions)
        )
        .unwrap();

        // Convert first result to floats and check mean
        // Results are in 100x scale (standard for CLOSURE/SPRITE statistics)
        let first_dist = unscale_distribution(&results.results.sample(0), 100);
        let computed_mean = mean(&first_dist);
        assert_within(computed_mean, 2.2, 0.05, "mean");
    }

    #[test]
    fn sprite_test_sd() {
        let results = sprite_parallel(
            2.2_f64, // mean
            1.3_f64, // sd
            20,      // n
            1,       // scale_min
            5,       // scale_max
            0.05,    // rounding_error_mean (precision 1)
            0.05,    // rounding_error_sd (precision 1)
            1,       // items
            None,    // restrictions_exact
            RestrictionsOption::Default,
            None, // parquet_config
            Some(5),
            None, // stop_after
        )
        .unwrap();

        // Check all distributions
        // Results are in 100x scale (standard for CLOSURE/SPRITE statistics)
        for i in 0..results.results.len() {
            let dist_scaled = &results.results.sample(i);
            let dist = unscale_distribution(dist_scaled, 100);
            let computed_sd = std_dev(&dist).unwrap();
            assert_within(computed_sd, 1.3, 0.05, "sd");
        }
    }

    #[test]
    fn sprite_test_big() {
        // What is the target mean and SD?
        let test_mean = 26.281;
        let test_sd = 14.6339;

        // To how many decimal places were they reported?
        let test_mean_digits = 3;
        let test_sd_digits = 4;

        // Convert precision to rounding errors
        let rounding_error_mean = 0.1_f64.powi(test_mean_digits) / 2.0; // 0.0005
        let rounding_error_sd = 0.1_f64.powi(test_sd_digits) / 2.0; // 0.00005

        // What is the target sample size?
        let test_n = 2000;

        // How many distributions should SPRITE generate?
        let target_runs = 5000;

        let results = sprite_parallel(
            test_mean,
            test_sd,
            test_n,
            1,
            50,
            rounding_error_mean,
            rounding_error_sd,
            1,
            None,
            RestrictionsOption::Default,
            None, // parquet_config
            Some(target_runs),
            None, // stop_after
        )
        .unwrap();

        println!("Number of target results: {:?}\n", results.results.len());
        println!("First result (scaled):\n");
        println!("{:?}", &results.results.sample(0)[..10]); // Print first 10 values

        // Results are in 100x scale (standard for CLOSURE/SPRITE statistics)
        let scale_factor = 100;

        // Go through the SPRITE results and check if they conform to the
        // input mean and SD. If any result sample doesn't, throw an error.
        for i in 0..results.results.len() {
            let dist_scaled = &results.results.sample(i);
            let dist = unscale_distribution(dist_scaled, scale_factor);
            let computed_mean = mean(&dist);
            let computed_sd = std_dev(&dist).unwrap();
            assert_within(computed_mean, test_mean, rounding_error_mean, "mean");
            assert_within(computed_sd, test_sd, rounding_error_sd, "sd");
        }
    }

    #[test]
    fn test_sprite_parallel_streaming() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        // Test streaming mode with separate files
        // `Both` so this test covers the counts layout and the legacy
        // per-position one in the same run.
        let config = StreamingConfig {
            file_path: "test_sprite_streaming/".to_string(),
            batch_size: 100,
            show_progress: false,
            format: OutputFormat::Both,
        };

        let _ = std::fs::create_dir("test_sprite_streaming");

        let expected_mean = 2.2;
        let expected_sd = 1.3;
        let expected_n = 20;
        let rounding_error_mean = 0.05;
        let rounding_error_sd = 0.05;

        let result = sprite_parallel_streaming::<f64, i32>(
            expected_mean,
            expected_sd,
            expected_n,
            1, // scale_min
            5, // scale_max
            rounding_error_mean,
            rounding_error_sd,
            1,    // items
            None, // restrictions_exact
            RestrictionsOption::Default,
            config,
            Some(3),
            None, // stop_after - only generate 3 samples for faster testing
        )
        .unwrap();

        assert!(result.total_combinations > 0);
        assert_eq!(result.file_path, "test_sprite_streaming/");

        // Check that files were created, in both layouts
        for name in &[
            "counts",
            "scale_values",
            "format",
            "sample",
            "horns",
            "metrics_main",
            "metrics_horns",
            "frequency",
        ] {
            assert!(
                std::path::Path::new(&format!("test_sprite_streaming/{name}.parquet")).exists(),
                "{name}.parquet was not written"
            );
        }

        // Read and validate the samples from the parquet file
        let file = File::open("test_sprite_streaming/sample.parquet")
            .expect("Failed to open sample.parquet");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to create parquet reader builder");
        let reader = builder.build().expect("Failed to build parquet reader");

        let mut total_samples_checked = 0;
        const MAX_SAMPLES_TO_CHECK: usize = 3; // Only check first 3 samples for speed

        // Process each batch (but only check a few samples)
        'outer: for maybe_batch in reader {
            let batch = maybe_batch.expect("Failed to read batch");
            let num_rows = batch.num_rows();
            let num_columns = batch.num_columns();

            // Each row is a sample, each column is a position in the sample
            for row_idx in 0..num_rows {
                if total_samples_checked >= MAX_SAMPLES_TO_CHECK {
                    break 'outer; // Stop after checking enough samples
                }

                // Extract the scaled sample values for this row
                let mut scaled_values: Vec<i32> = Vec::with_capacity(num_columns);

                for col_idx in 0..num_columns {
                    let column = batch.column(col_idx);
                    let int_array = column
                        .as_any()
                        .downcast_ref::<arrow::array::Int32Array>()
                        .expect("Column is not Int32Array");
                    scaled_values.push(int_array.value(row_idx));
                }

                // Verify sample size
                assert_eq!(
                    scaled_values.len(),
                    expected_n as usize,
                    "Sample size mismatch: expected {}, got {}",
                    expected_n,
                    scaled_values.len()
                );

                // SPRITE reports values in hundredths of a scale point, in
                // both layouts.
                let sample_values: Vec<f64> =
                    scaled_values.iter().map(|&v| v as f64 / 100.0).collect();

                // Calculate mean and SD for this sample
                let computed_mean = mean(&sample_values);
                let computed_sd = std_dev(&sample_values).unwrap();

                // Inclusive, with the search's float slack: a sample whose
                // total sits on the edge of the tolerance is admissible.
                assert_within(computed_mean, expected_mean, rounding_error_mean, "mean");
                assert_within(computed_sd, expected_sd, rounding_error_sd, "sd");

                total_samples_checked += 1;
            }
        }

        // Verify we checked the expected number of samples
        assert_eq!(
            total_samples_checked, MAX_SAMPLES_TO_CHECK,
            "Expected to check {} samples, but only checked {}",
            MAX_SAMPLES_TO_CHECK, total_samples_checked
        );

        // Clean up test files
        let _ = std::fs::remove_dir_all("test_sprite_streaming");
    }

    #[test]
    fn test_sprite_parallel_with_parquet() {
        // Test in-memory mode with optional Parquet output
        let config = ParquetConfig {
            file_path: "test_sprite_parquet/".to_string(),
            batch_size: 100,
            format: OutputFormat::default(),
        };

        let _ = std::fs::create_dir("test_sprite_parquet");

        let results = sprite_parallel::<f64, i32>(
            2.2_f64, // mean
            1.3_f64, // sd
            20,      // n
            1,       // scale_min
            5,       // scale_max
            0.05,    // rounding_error_mean (precision 1)
            0.05,    // rounding_error_sd (precision 1)
            1,       // items
            None,    // restrictions_exact
            RestrictionsOption::Default,
            Some(config), // parquet_config
            Some(5),
            None, // stop_after
        )
        .unwrap();

        // Check that results were returned
        assert!(!results.results.is_empty());

        // Check that files were created
        for name in &[
            "counts",
            "scale_values",
            "format",
            "metrics_main",
            "metrics_horns",
            "frequency",
        ] {
            assert!(
                std::path::Path::new(&format!("test_sprite_parquet/{name}.parquet")).exists(),
                "{name}.parquet was not written"
            );
        }

        // Clean up test files
        let _ = std::fs::remove_dir_all("test_sprite_parquet");
    }

    /// Regression test for the LP stopping criterion formula.
    ///
    /// The old formula `n_found / (n_found + c_dup) >= 0.99` is equivalent to checking that
    /// 99% of all draws have been *unique*, which is what the early phase of any search looks
    /// like. It fires at c_dup = 1 the moment n_found >= 99, i.e., on the very first duplicate
    /// once 99 unique distributions have been found — regardless of how many total valid
    /// distributions exist.
    ///
    /// The corrected formula `c_dup / (n_found + c_dup) >= 0.99` checks the *duplicate rate*,
    /// which only reaches 0.99 once the accessible neighborhood is truly saturated.
    ///
    /// In practice SPRITE's random walk only reaches a subset of all valid distributions
    /// (Cause 2, sampling bias), so LP fires when that subset is saturated rather than when
    /// the full space is covered. The number of unique distributions found is therefore bounded
    /// by the accessible neighborhood (~1,400–1,500 for these inputs in typical runs). This test
    /// simply verifies that SPRITE finds at least 100 distributions — more than the old
    /// formula's hard floor of 99 — and that all returned distributions are valid.
    #[test]
    fn test_lp_stopping_criterion_not_premature() {
        let results = sprite_parallel(
            3.0_f64,
            1.0_f64,
            100_i32,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            RestrictionsOption::Default,
            None,
            Some(10_000),
            None,
        )
        .unwrap();

        let n_found = results.results.len();

        // All returned distributions must have correct mean and SD.
        for i in 0..results.results.len() {
            let dist_scaled = &results.results.sample(i);
            let dist = unscale_distribution(dist_scaled, 100);
            assert_within(mean(&dist), 3.0, 0.05, "mean");
            assert_within(std_dev(&dist).unwrap(), 1.0, 0.05, "sd");
        }

        // SPRITE should find at least 100 distributions (above the old LP floor of 99).
        assert!(
            n_found >= 100,
            "SPRITE found only {n_found} distributions with stop_after=10000 \
             (CLOSURE finds 24,907 for these inputs). \
             LP stopping criterion may be broken."
        );
    }

    #[test]
    fn the_attempt_cap_is_bounded_without_a_limit_and_never_wraps() {
        // `stop_after = None` used to become `usize::MAX`, then times 20:
        // a panic in debug builds, a silent wrap in release. Just past
        // `usize::MAX / 20` a wrapped product falls below 20.
        assert_eq!(attempt_cap(None), MAX_ATTEMPTS);
        assert_eq!(attempt_cap(Some(usize::MAX / 20 + 1)), MAX_ATTEMPTS);
        assert_eq!(attempt_cap(Some(usize::MAX)), MAX_ATTEMPTS);
        assert_eq!(attempt_cap(Some(1)), MIN_DRAWS as usize);
        assert_eq!(attempt_cap(Some(1_000)), 20_000);
    }

    #[test]
    fn the_widest_whole_number_sample_rules_out_an_sd_up_front() {
        // On 1..=5 with n = 5 and mean 3.20 (total 16), the widest sample is
        // 1, 1, 4, 5, 5: SD 2.049. Bhatia-Davis allows 2.23, so SD 2.15 used
        // to pass the precheck and search to no avail.
        let params = |sd: f64| {
            build_sprite_params::<f64, i32>(
                3.2,
                sd,
                5,
                1,
                5,
                0.005,
                0.005,
                1,
                None,
                RestrictionsOption::Null,
            )
            .unwrap()
        };
        assert!(params(2.15).infeasible.is_some());
        assert!(params(2.05).infeasible.is_none());
        // Fixed responses are counted as fixed: three 3s among n = 5 leave two
        // free values with total 7, widest as 2 and 5, so the widest sample
        // is 2, 3, 3, 3, 5: SD 1.095. Bhatia-Davis over all five values
        // allowed 2.23.
        let mut exact = std::collections::HashMap::new();
        exact.insert(300, 3);
        let fixed = |sd: f64| {
            build_sprite_params::<f64, i32>(
                3.2,
                sd,
                5,
                1,
                5,
                0.005,
                0.005,
                1,
                Some(exact.clone()),
                RestrictionsOption::Null,
            )
            .unwrap()
        };
        assert!(fixed(1.10).infeasible.is_none());
        assert!(fixed(1.11).infeasible.is_some());
    }

    #[test]
    fn the_exhaustive_scan_finds_the_one_move_left() {
        // Near the widest SD on 1..=5, `2 4` among 499 1s and 499 5s has one
        // move that raises the SD: the 2 to 1 and the 4 to 5. Random tries
        // hit it about once in n shifts, and attempts used to give up first.
        let params = build_sprite_params::<f64, i32>(
            3.0,
            2.0,
            1000,
            1,
            5,
            0.005,
            0.001,
            1,
            None,
            RestrictionsOption::Null,
        )
        .unwrap();
        let mut rng = rand::rng();
        let mut vec: Vec<i32> = [vec![1; 499], vec![5; 499], vec![2, 4]].concat();
        let sum_sq = |v: &[i32]| v.iter().map(|&x| (x * x) as i64).sum::<i64>();
        let mut running = sum_sq(&vec);
        assert!(shift_values_exhaustive(
            &mut vec,
            &params,
            true,
            &mut running,
            &mut rng
        ));
        assert!(vec.iter().all(|&v| v == 1 || v == 5));
        assert_eq!(running, sum_sq(&vec));
        // And then none is left, which ends the attempt.
        assert!(!shift_values_exhaustive(
            &mut vec,
            &params,
            true,
            &mut running,
            &mut rng
        ));
    }

    #[test]
    fn one_value_moving_several_grid_steps_can_compensate() {
        // With 4 fixed once on 1..=5, the free values `3 3 5 5 5` have one
        // move that raises the SD: a 3 to 1 and the other 3 to 5, two grid
        // steps. Compensating only between neighbouring grid values, SPRITE
        // had no move at all here.
        let params = build_sprite_params::<f64, i32>(
            4.17,
            1.6,
            6,
            1,
            5,
            0.005,
            0.005,
            1,
            Some([(400, 1)].into_iter().collect()),
            RestrictionsOption::Null,
        )
        .unwrap();
        let mut rng = rand::rng();
        let mut vec = vec![3, 3, 5, 5, 5];
        let mut sum_sq: i64 = vec.iter().map(|v| (v * v) as i64).sum();
        let mut moved = false;
        for _ in 0..100 {
            if shift_values_internal(&mut vec, &params, true, &mut sum_sq, &mut rng) {
                moved = true;
                break;
            }
        }
        assert!(moved, "no move found");
        vec.sort();
        assert_eq!(vec, vec![1, 5, 5, 5, 5]);
        assert_eq!(sum_sq, vec.iter().map(|v| (v * v) as i64).sum::<i64>());
    }

    #[test]
    fn loop_limits_grow_with_the_grid_and_never_wrap() {
        // Likert inputs keep the old floor.
        assert_eq!(loop_limits(100, 5), (500, 20_000));
        // Mean 16, sd 13, n 1000 on 1..=31 at 100 items needs over a million
        // shifts; a fixed cap of 1,000,000 failed every direct start.
        assert_eq!(loop_limits(1_000, 3_001), (3_001_000, 3_001_000));
        // `n·k²` used to be a u32: here 2^32, which wrapped to 0 and then
        // clamped to the floor of 20,000.
        assert_eq!(loop_limits(65_536, 256), (16_777_216, 16_777_216));
        assert_eq!(loop_limits(usize::MAX, 100_000), (u32::MAX, u32::MAX));
    }

    #[test]
    fn the_sd_parity_check_keeps_every_total_a_sample_reaches() {
        // Every sequence of 6 values on 1..=5, by total and `n·Σx² − total²`.
        let n = 6usize;
        let samples: Vec<(i64, i64)> = (0..5usize.pow(n as u32))
            .map(|mut code| {
                let (mut s, mut q) = (0i64, 0i64);
                for _ in 0..n {
                    let x = (code % 5) as i64 + 1;
                    code /= 5;
                    s += x;
                    q += x * x;
                }
                (s, n as i64 * q - s * s)
            })
            .collect();
        let mut ruled_out = 0;
        for mean_x100 in (150..=450).step_by(7) {
            for sd_x100 in (20..=200).step_by(9) {
                let Ok(params) = build_sprite_params::<f64, i32>(
                    mean_x100 as f64 / 100.0,
                    sd_x100 as f64 / 100.0,
                    n as i32,
                    1,
                    5,
                    0.05,
                    0.005,
                    1,
                    None,
                    RestrictionsOption::Null,
                ) else {
                    continue;
                };
                for total in params.sum_lo..=params.sum_hi {
                    let reached = samples.iter().any(|&(s, m2n)| {
                        s == total && (params.m2n_lo..=params.m2n_hi).contains(&m2n)
                    });
                    let kept = params.total_can_hit_sd(total);
                    assert!(
                        kept || !reached,
                        "mean {mean_x100}, sd {sd_x100}, total {total}"
                    );
                    ruled_out += (!kept) as usize;
                }
            }
        }
        // The check must also bite, or it guards nothing.
        assert!(ruled_out > 0);
    }
}
