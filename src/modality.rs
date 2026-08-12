//! Shape analysis of reconstructed samples.
//!
//! Every sample a technique in this crate produces is a count vector over the
//! shared [`ValueGrid`](crate::sample_counts::ValueGrid). This module answers,
//! for each such vector, *what shape is it* — and then aggregates those answers
//! across a whole result set.
//!
//! # Why two axes rather than one
//!
//! "Unimodal" answers how many modes a histogram has and says nothing about
//! where they are. But `(33, 31, 24, 7, 5)` and `(5, 24, 31, 24, 5)` are both
//! unimodal in the textbook sense while telling completely different stories: a
//! floor effect and a centred distribution. Forensically the difference is the
//! whole point, so [`ShapeClass`] crosses mode *count* with mode *location*:
//!
//! | | interior peak | peak at a scale boundary |
//! |---|---|---|
//! | one mode | [`OneModeInterior`] ("bell") | [`OneModeLowEdge`] / [`OneModeHighEdge`] (J) |
//! | two or more | [`TwoModes`] | [`ThreeOrMoreModes`] |
//! | no unique mode | [`Flat`] | — |
//!
//! [`OneModeInterior`]: ShapeClass::OneModeInterior
//! [`OneModeLowEdge`]: ShapeClass::OneModeLowEdge
//! [`OneModeHighEdge`]: ShapeClass::OneModeHighEdge
//! [`TwoModes`]: ShapeClass::TwoModes
//! [`ThreeOrMoreModes`]: ShapeClass::ThreeOrMoreModes
//! [`Flat`]: ShapeClass::Flat
//!
//! The classes partition: every sample lands in exactly one, so the per-class
//! counts always sum to the number of samples scanned.
//!
//! # Why modes need a prominence threshold
//!
//! Modality of an integer count vector is not well defined on its own. Ties are
//! common and one count of difference creates or destroys a "strict local
//! maximum": `(30, 29, 30, 29, 32)` has three of them and is, by any reasonable
//! reading, flat. Mode detection here therefore requires *topographic
//! prominence* — a candidate peak must rise by at least
//! [`ShapeAccumulator::min_prominence`] of the sample above the higher of the
//! two valleys separating it from any taller peak. The threshold is a fraction
//! of `n`, so it does not silently change meaning with the length of the scale.
//!
//! # The graded companion
//!
//! Because any threshold is a judgement call, [`unimodality_deficit`] reports
//! the same question without one: the number of observations that would have to
//! be removed before the histogram is single-peaked. It is zero exactly when the
//! vector is weakly unimodal, and needs no tuning constant.

use strum_macros::{EnumCount as EnumCountMacro, EnumIter, IntoStaticStr};

/// Default minimum prominence for a local maximum to count as a mode,
/// as a fraction of the sample size.
///
/// A peak must rise 5% of the sample above its key col. The value is a
/// judgement call, exposed here so it is visible and overridable rather than
/// buried in a predicate; [`unimodality_deficit`] is the threshold-free
/// alternative.
pub const DEFAULT_MODE_PROMINENCE: f64 = 0.05;

/// Where a sample's mass sits, crossing mode count with mode location.
///
/// The variants partition the space of count vectors: [`classify`] returns
/// exactly one for any input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCountMacro, EnumIter, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ShapeClass {
    /// Every grid position holds the same count. No unique mode.
    Flat,
    /// One mode, away from both ends of the scale — the "bell" case.
    OneModeInterior,
    /// One mode, at the lowest scale value: a floor effect (J-shape).
    OneModeLowEdge,
    /// One mode, at the highest scale value: a ceiling effect (J-shape).
    OneModeHighEdge,
    /// Two modes.
    TwoModes,
    /// Three or more modes.
    ThreeOrMoreModes,
}

impl ShapeClass {
    /// snake_case name, as written to Parquet.
    pub fn as_str(&self) -> &'static str {
        self.into()
    }

    /// All variants in declaration order.
    pub fn all() -> impl Iterator<Item = Self> + Clone {
        <Self as strum::IntoEnumIterator>::iter()
    }

    /// True for the three classes with exactly one mode, wherever it sits.
    ///
    /// This is unimodality in the textbook sense. [`ShapeClass::is_bell`] is the
    /// narrower property most shape claims in papers actually assert.
    pub fn is_unimodal(&self) -> bool {
        matches!(
            self,
            Self::OneModeInterior | Self::OneModeLowEdge | Self::OneModeHighEdge
        )
    }

    /// True only for a single mode away from both scale boundaries.
    pub fn is_bell(&self) -> bool {
        matches!(self, Self::OneModeInterior)
    }

    /// True for a single mode sitting at one end of the scale.
    pub fn is_j_shape(&self) -> bool {
        matches!(self, Self::OneModeLowEdge | Self::OneModeHighEdge)
    }

    /// True for two or more modes.
    pub fn is_multimodal(&self) -> bool {
        matches!(self, Self::TwoModes | Self::ThreeOrMoreModes)
    }
}

/// A maximal run of equal counts: `[start, end]` inclusive, at height `height`.
struct Run {
    start: usize,
    end: usize,
    height: u32,
}

/// Split a count vector into maximal runs of equal value.
fn runs_of(counts: &[u32]) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (i, &c) in counts.iter().enumerate() {
        match runs.last_mut() {
            Some(last) if last.height == c => last.end = i,
            _ => runs.push(Run {
                start: i,
                end: i,
                height: c,
            }),
        }
    }
    runs
}

/// Grid positions of the modes of `counts`, ascending.
///
/// A mode is a maximal run of equal counts that is strictly higher than the runs
/// on either side of it (a missing neighbour counts as lower, so a peak may sit
/// at either end of the scale), and whose topographic prominence is at least
/// `min_prominence_counts` observations.
///
/// Prominence is the run's height minus the higher of its two key cols, where a
/// key col is the lowest point between the run and the nearest strictly taller
/// run on that side. With no taller run on a side, that side's col is 0, so the
/// tallest run of a non-empty sample is always a mode.
///
/// Each mode is reported by the first position of its run. A vector whose counts
/// are all equal has no unique mode and yields an empty result.
pub fn modes(counts: &[u32], min_prominence_counts: u32) -> Vec<usize> {
    let runs = runs_of(counts);
    // A single run means every count is equal: flat, no unique mode.
    if runs.len() < 2 {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        let higher_left = i > 0 && runs[i - 1].height > run.height;
        let higher_right = i + 1 < runs.len() && runs[i + 1].height > run.height;
        if higher_left || higher_right {
            continue; // not a local maximum
        }

        // Key col to the left: lowest run between here and the nearest taller
        // run. Zero if no taller run exists on that side.
        let mut left_col = 0;
        for j in (0..i).rev() {
            if runs[j].height > run.height {
                left_col = runs[j + 1..i]
                    .iter()
                    .map(|r| r.height)
                    .min()
                    .unwrap_or(run.height);
                break;
            }
        }
        let mut right_col = 0;
        for (j, taller) in runs.iter().enumerate().skip(i + 1) {
            if taller.height > run.height {
                right_col = runs[i + 1..j]
                    .iter()
                    .map(|r| r.height)
                    .min()
                    .unwrap_or(run.height);
                break;
            }
        }

        let prominence = run.height - left_col.max(right_col);
        if prominence >= min_prominence_counts {
            out.push(run.start);
        }
    }
    out
}

/// Classify a count vector into exactly one [`ShapeClass`].
///
/// `min_prominence_counts` is the prominence a local maximum needs to count as a
/// mode; see [`modes`].
pub fn classify(counts: &[u32], min_prominence_counts: u32) -> ShapeClass {
    let peaks = modes(counts, min_prominence_counts);
    match peaks.len() {
        // No qualifying peak means every count is equal, or the sample is empty.
        0 => ShapeClass::Flat,
        1 => {
            let runs = runs_of(counts);
            let run = runs
                .iter()
                .find(|r| r.start == peaks[0])
                .expect("a mode is always the start of a run");
            if run.start == 0 {
                ShapeClass::OneModeLowEdge
            } else if run.end == counts.len() - 1 {
                ShapeClass::OneModeHighEdge
            } else {
                ShapeClass::OneModeInterior
            }
        }
        2 => ShapeClass::TwoModes,
        _ => ShapeClass::ThreeOrMoreModes,
    }
}

/// The number of observations that must be removed before `counts` is
/// single-peaked, minimised over every possible peak position.
///
/// Zero exactly when `counts` is already weakly unimodal (non-decreasing to a
/// maximum, then non-increasing). This is the discrete counterpart of an excess-
/// mass statistic: it grades how far a histogram is from unimodal without
/// needing a threshold, so it says something useful in the cases where mode
/// counting is genuinely ambiguous.
///
/// For a fixed peak position `p`, the largest weakly-unimodal vector lying under
/// `counts` with its peak at `p` is the running minimum outward from `p`; the
/// deficit is the mass that running minimum leaves behind. Runs in `O(k²)`.
pub fn unimodality_deficit(counts: &[u32]) -> u32 {
    let k = counts.len();
    if k == 0 {
        return 0;
    }
    let total: u32 = counts.iter().sum();

    let mut best = u32::MAX;
    for p in 0..k {
        let mut kept = counts[p];
        let mut running = counts[p];
        for i in (0..p).rev() {
            running = running.min(counts[i]);
            kept += running;
        }
        running = counts[p];
        for &c in counts.iter().skip(p + 1) {
            running = running.min(c);
            kept += running;
        }
        best = best.min(total - kept);
        if best == 0 {
            break;
        }
    }
    best
}

/// Per-value count bounds over the members of one [`ShapeClass`].
///
/// This is the conditional-bounds table: `count_lo` / `count_hi` are the minimum
/// and maximum count of each grid value **among samples of this class alone**.
///
/// Read a row as a conditional, not as a statement about the data: *if* the raw
/// data had this shape, then the count of each value lies in this range. When
/// the underlying search was exhaustive that conditional is deductive, which
/// makes it the one form of shape claim a single atypical sample cannot defeat —
/// it quantifies over the whole class rather than asserting that some member
/// exists.
#[derive(Clone, Debug)]
pub struct ShapeBounds {
    pub class: ShapeClass,
    /// How many scanned samples fell into this class.
    pub n_samples: u64,
    /// Per-grid-position minimum count within the class; empty when `n_samples` is 0.
    pub count_lo: Vec<i32>,
    /// Per-grid-position maximum count within the class; empty when `n_samples` is 0.
    pub count_hi: Vec<i32>,
}

/// Shape of a whole result set: how many samples of each class, what each class
/// requires, and how far from unimodal the set is.
///
/// Every field here is produced by an actual scan over samples — there is no
/// default value and no placeholder, which is what keeps a path that forgot to
/// compute the shapes from silently reporting the strongest possible claim.
#[derive(Clone, Debug)]
pub struct ModalityShapes {
    /// True when every sample satisfying the reported statistics was scanned.
    ///
    /// False for a truncated search (`stop_after`) and for SPRITE, which samples
    /// the solution space at random rather than enumerating it. When false, a
    /// class with zero members proves nothing: see [`ModalityShapes::can_be`].
    pub exhaustive: bool,
    /// Number of samples scanned.
    pub n_scanned: u64,
    /// One entry per [`ShapeClass`], in declaration order.
    pub bounds: Vec<ShapeBounds>,
    /// Smallest [`unimodality_deficit`] over the scanned samples.
    pub deficit_min: u32,
    /// Mean [`unimodality_deficit`] over the scanned samples.
    pub deficit_mean: f64,
    /// Largest [`unimodality_deficit`] over the scanned samples.
    pub deficit_max: u32,
    /// Prominence threshold used for mode detection, as a fraction of `n`.
    pub min_prominence: f64,
}

impl ModalityShapes {
    /// Number of scanned samples in `class`.
    pub fn n_of(&self, class: ShapeClass) -> u64 {
        self.bounds
            .iter()
            .find(|b| b.class == class)
            .map_or(0, |b| b.n_samples)
    }

    /// Proportion of scanned samples in `class`.
    ///
    /// Weights every scanned sample equally. For an exhaustive CLOSURE run that
    /// is a uniform weighting over admissible datasets — the same weighting the
    /// horns summaries already use — and not a posterior probability.
    pub fn proportion_of(&self, class: ShapeClass) -> f64 {
        if self.n_scanned == 0 {
            return f64::NAN;
        }
        self.n_of(class) as f64 / self.n_scanned as f64
    }

    /// Whether the raw data could have had a shape satisfying `predicate`.
    ///
    /// - `Some(true)`: a scanned sample has such a shape, so it is possible.
    ///   Sound whether or not the search was exhaustive — a witness is a witness.
    /// - `Some(false)`: **no** admissible sample has such a shape, so the raw
    ///   data did not either. Only ever returned for an exhaustive search.
    /// - `None`: none were found, but the search was partial, so nothing follows.
    ///
    /// The `None` case is the whole reason this returns an `Option`. A truncated
    /// CLOSURE run and a SPRITE run both fail to see most of the space; reporting
    /// their silence as `false` would turn "we did not look" into "we proved it
    /// impossible".
    pub fn can_be(&self, predicate: impl Fn(ShapeClass) -> bool) -> Option<bool> {
        let found = self
            .bounds
            .iter()
            .any(|b| b.n_samples > 0 && predicate(b.class));
        match (found, self.exhaustive) {
            (true, _) => Some(true),
            (false, true) => Some(false),
            (false, false) => None,
        }
    }

    /// Could the raw data have had exactly one mode, anywhere on the scale?
    pub fn can_be_unimodal(&self) -> Option<bool> {
        self.can_be(|c| c.is_unimodal())
    }

    /// Could the raw data have had exactly one mode away from both scale ends?
    ///
    /// This is the claim papers make when they describe responses as bell-shaped
    /// or approximately normal, and it is strictly stronger than
    /// [`ModalityShapes::can_be_unimodal`], which a floor or ceiling effect also
    /// satisfies.
    pub fn can_be_bell_shaped(&self) -> Option<bool> {
        self.can_be(|c| c.is_bell())
    }

    /// Could the raw data have had two or more modes?
    pub fn can_be_multimodal(&self) -> Option<bool> {
        self.can_be(|c| c.is_multimodal())
    }

    /// Could the raw data have piled up at the bottom of the scale?
    pub fn can_be_j_shape_low(&self) -> Option<bool> {
        self.can_be(|c| c == ShapeClass::OneModeLowEdge)
    }

    /// Could the raw data have piled up at the top of the scale?
    pub fn can_be_j_shape_high(&self) -> Option<bool> {
        self.can_be(|c| c == ShapeClass::OneModeHighEdge)
    }
}

/// Running shape statistics over a stream of count vectors.
///
/// Shared by the in-memory and streaming paths so both report the same thing;
/// it needs one sample at a time and never the whole result set.
pub struct ShapeAccumulator {
    k: usize,
    /// Prominence threshold in observations, derived once from `n`.
    min_prominence_counts: u32,
    min_prominence: f64,
    n_scanned: u64,
    per_class: Vec<ShapeBounds>,
    deficit_min: u32,
    deficit_max: u32,
    deficit_sum: u64,
}

impl ShapeAccumulator {
    /// Start accumulating over count vectors of width `k` from samples of size
    /// `n`, requiring `min_prominence` (a fraction of `n`) for a mode.
    pub fn new(k: usize, n: usize, min_prominence: f64) -> Self {
        // At least one observation, so a threshold of 0 does not admit ties as
        // separate modes.
        let min_prominence_counts = ((min_prominence * n as f64).ceil() as u32).max(1);
        Self {
            k,
            min_prominence_counts,
            min_prominence,
            n_scanned: 0,
            per_class: ShapeClass::all()
                .map(|class| ShapeBounds {
                    class,
                    n_samples: 0,
                    count_lo: Vec::new(),
                    count_hi: Vec::new(),
                })
                .collect(),
            deficit_min: u32::MAX,
            deficit_max: 0,
            deficit_sum: 0,
        }
    }

    /// Fold one sample's count vector into the running statistics.
    pub fn update(&mut self, counts: &[u32]) {
        debug_assert_eq!(counts.len(), self.k);
        let class = classify(counts, self.min_prominence_counts);
        let slot = &mut self.per_class[class as usize];

        if slot.n_samples == 0 {
            slot.count_lo = counts.iter().map(|&c| c as i32).collect();
            slot.count_hi = slot.count_lo.clone();
        } else {
            for (i, &c) in counts.iter().enumerate() {
                let c = c as i32;
                if c < slot.count_lo[i] {
                    slot.count_lo[i] = c;
                }
                if c > slot.count_hi[i] {
                    slot.count_hi[i] = c;
                }
            }
        }
        slot.n_samples += 1;

        let deficit = unimodality_deficit(counts);
        self.deficit_min = self.deficit_min.min(deficit);
        self.deficit_max = self.deficit_max.max(deficit);
        self.deficit_sum += deficit as u64;
        self.n_scanned += 1;
    }

    /// Finish, declaring whether the scan covered the whole solution space.
    ///
    /// Pass `false` for a truncated search or for SPRITE. That does not weaken
    /// the counts or the conditional bounds — it only stops
    /// [`ModalityShapes::can_be`] from reporting an absence as a proof.
    pub fn finish(self, exhaustive: bool) -> ModalityShapes {
        let n = self.n_scanned;
        ModalityShapes {
            exhaustive,
            n_scanned: n,
            bounds: self.per_class,
            deficit_min: if n == 0 { 0 } else { self.deficit_min },
            deficit_mean: if n == 0 {
                f64::NAN
            } else {
                self.deficit_sum as f64 / n as f64
            },
            deficit_max: self.deficit_max,
            min_prominence: self.min_prominence,
        }
    }
}

/// The shapes of a result set with nothing in it.
///
/// `exhaustive` is false: an empty scan has seen nothing, so every `can_be`
/// query is `None` rather than a claim of impossibility.
pub fn empty_shapes(min_prominence: f64) -> ModalityShapes {
    ShapeAccumulator::new(0, 0, min_prominence).finish(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prominence threshold of 1 observation: the loosest setting that still
    /// refuses to treat a tie as two modes. Used where the test is about the
    /// shape logic rather than about thresholding.
    const LOOSE: u32 = 1;

    #[test]
    fn flat_vectors_have_no_mode() {
        assert_eq!(modes(&[20, 20, 20, 20, 20], LOOSE), Vec::<usize>::new());
        assert_eq!(classify(&[20, 20, 20, 20, 20], LOOSE), ShapeClass::Flat);
        // A flat vector is not a witness that the data could be bell-shaped.
        assert!(!ShapeClass::Flat.is_unimodal());
    }

    #[test]
    fn one_interior_peak_is_a_bell() {
        assert_eq!(
            classify(&[5, 24, 42, 24, 5], LOOSE),
            ShapeClass::OneModeInterior
        );
        // Plateaus at the top are still one mode.
        assert_eq!(
            classify(&[5, 20, 20, 5, 5], LOOSE),
            ShapeClass::OneModeInterior
        );
    }

    #[test]
    fn monotone_vectors_are_unimodal_but_not_bells() {
        // The example from the assessment: textbook-unimodal, floor effect.
        let c = classify(&[33, 31, 24, 7, 5], LOOSE);
        assert_eq!(c, ShapeClass::OneModeLowEdge);
        assert!(c.is_unimodal(), "one mode, so unimodal in the usual sense");
        assert!(!c.is_bell(), "but the mode sits at the scale boundary");
        assert!(c.is_j_shape());

        assert_eq!(
            classify(&[15, 15, 15, 15, 40], LOOSE),
            ShapeClass::OneModeHighEdge
        );
    }

    #[test]
    fn two_separated_peaks_are_bimodal() {
        assert_eq!(classify(&[40, 5, 5, 5, 45], LOOSE), ShapeClass::TwoModes);
        assert_eq!(classify(&[0, 49, 0, 2, 49], LOOSE), ShapeClass::TwoModes);
    }

    #[test]
    fn a_bump_between_two_peaks_is_a_mode_only_if_prominent_enough() {
        // The centre bump rises 5 above the valleys either side of it. Whether
        // that is a third mode is exactly the judgement the threshold encodes,
        // so pin down both answers rather than pretending there is only one.
        let w = [40, 5, 10, 5, 40];
        assert_eq!(classify(&w, 5), ShapeClass::ThreeOrMoreModes);
        assert_eq!(classify(&w, 6), ShapeClass::TwoModes);
    }

    #[test]
    fn prominence_suppresses_noise_peaks() {
        // Near-flat with three strict local maxima. Any threshold above the
        // 1-3 count wobble must call this flat-ish rather than trimodal.
        let noisy = [30, 29, 30, 29, 32];
        assert_eq!(classify(&noisy, LOOSE), ShapeClass::ThreeOrMoreModes);
        // 5% of n=150 is 8 observations, far above the wobble.
        assert_eq!(classify(&noisy, 8), ShapeClass::OneModeHighEdge);
    }

    #[test]
    fn prominence_threshold_scales_with_n_not_with_k() {
        // The same shape on a 5-point and an 11-point scale must classify the
        // same way. The old `total / k` rule did not have this property.
        let short = [5, 40, 5, 40, 10];
        let long = [5, 40, 5, 40, 10, 0, 0, 0, 0, 0, 0];
        let thresh = ((DEFAULT_MODE_PROMINENCE * 100.0).ceil() as u32).max(1);
        assert_eq!(classify(&short, thresh), ShapeClass::TwoModes);
        assert_eq!(classify(&long, thresh), ShapeClass::TwoModes);
    }

    #[test]
    fn classes_partition_the_space() {
        // Exhaustive over all count vectors of 8 observations on a 4-point
        // scale: every one lands in exactly one class, and the unimodal classes
        // are exactly the one-mode ones.
        let mut seen = 0;
        for a in 0..=8u32 {
            for b in 0..=(8 - a) {
                for c in 0..=(8 - a - b) {
                    let d = 8 - a - b - c;
                    let v = [a, b, c, d];
                    let class = classify(&v, 1);
                    assert_eq!(
                        class.is_unimodal(),
                        modes(&v, 1).len() == 1,
                        "class/mode-count disagreement on {v:?}"
                    );
                    assert!(!(class.is_unimodal() && class.is_multimodal()));
                    seen += 1;
                }
            }
        }
        assert_eq!(seen, 165, "C(8+3, 3) count vectors");
    }

    #[test]
    fn deficit_is_zero_exactly_for_weakly_unimodal_vectors() {
        assert_eq!(unimodality_deficit(&[5, 24, 42, 24, 5]), 0);
        assert_eq!(unimodality_deficit(&[33, 31, 24, 7, 5]), 0);
        assert_eq!(unimodality_deficit(&[20, 20, 20, 20, 20]), 0);
        assert_eq!(unimodality_deficit(&[5, 20, 20, 5, 5]), 0);
        // A deep valley. The best single peak keeps one 40 and then no more
        // than 5 at every later position: 40 + 5*4 = 60 of the 100 survive, so
        // 40 observations have to go before the shape is single-peaked.
        assert_eq!(unimodality_deficit(&[40, 5, 10, 5, 40]), 40);
    }

    #[test]
    fn deficit_agrees_with_a_brute_force_unimodality_check() {
        // Over every count vector of 7 observations on a 5-point scale, a zero
        // deficit must coincide with the textbook definition of weak unimodality.
        fn weakly_unimodal(f: &[u32]) -> bool {
            let mut descending = false;
            for i in 1..f.len() {
                if f[i] > f[i - 1] {
                    if descending {
                        return false;
                    }
                } else if f[i] < f[i - 1] {
                    descending = true;
                }
            }
            true
        }
        let mut checked = 0;
        for a in 0..=7u32 {
            for b in 0..=(7 - a) {
                for c in 0..=(7 - a - b) {
                    for d in 0..=(7 - a - b - c) {
                        let e = 7 - a - b - c - d;
                        let v = [a, b, c, d, e];
                        assert_eq!(
                            unimodality_deficit(&v) == 0,
                            weakly_unimodal(&v),
                            "deficit disagrees with the definition on {v:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 330, "C(7+4, 4) count vectors");
    }

    #[test]
    fn accumulator_counts_and_bounds_are_conditional_on_the_class() {
        let mut acc = ShapeAccumulator::new(5, 100, DEFAULT_MODE_PROMINENCE);
        acc.update(&[15, 15, 15, 15, 40]); // high edge
        acc.update(&[15, 15, 15, 16, 39]); // high edge
        acc.update(&[0, 49, 0, 2, 49]); // two modes
        let shapes = acc.finish(true);

        assert_eq!(shapes.n_scanned, 3);
        assert_eq!(shapes.n_of(ShapeClass::OneModeHighEdge), 2);
        assert_eq!(shapes.n_of(ShapeClass::TwoModes), 1);
        assert_eq!(shapes.n_of(ShapeClass::OneModeInterior), 0);

        // Bounds over the high-edge class alone, not over all three samples.
        let hi = shapes
            .bounds
            .iter()
            .find(|b| b.class == ShapeClass::OneModeHighEdge)
            .unwrap();
        assert_eq!(hi.count_lo, vec![15, 15, 15, 15, 39]);
        assert_eq!(hi.count_hi, vec![15, 15, 15, 16, 40]);

        // No sample has an interior mode, and the search saw everything.
        assert_eq!(shapes.can_be_bell_shaped(), Some(false));
        assert_eq!(shapes.can_be_unimodal(), Some(true));
        assert_eq!(shapes.can_be_multimodal(), Some(true));
    }

    #[test]
    fn a_partial_search_reports_absence_as_unknown() {
        let mut acc = ShapeAccumulator::new(5, 100, DEFAULT_MODE_PROMINENCE);
        acc.update(&[15, 15, 15, 15, 40]);
        let shapes = acc.finish(false);

        // Found: still sound, a witness is a witness even in a partial scan.
        assert_eq!(shapes.can_be_unimodal(), Some(true));
        // Not found, and we did not look everywhere: not a proof.
        assert_eq!(shapes.can_be_bell_shaped(), None);
        assert_eq!(shapes.can_be_multimodal(), None);
    }

    #[test]
    fn class_counts_always_sum_to_the_number_scanned() {
        let mut acc = ShapeAccumulator::new(4, 9, DEFAULT_MODE_PROMINENCE);
        for a in 0..=9u32 {
            for b in 0..=(9 - a) {
                for c in 0..=(9 - a - b) {
                    acc.update(&[a, b, c, 9 - a - b - c]);
                }
            }
        }
        let shapes = acc.finish(true);
        let summed: u64 = shapes.bounds.iter().map(|b| b.n_samples).sum();
        assert_eq!(summed, shapes.n_scanned);
        assert_eq!(shapes.n_scanned, 220, "C(9+3, 3) count vectors");
    }

    #[test]
    fn empty_scan_claims_nothing() {
        let shapes = empty_shapes(DEFAULT_MODE_PROMINENCE);
        assert_eq!(shapes.n_scanned, 0);
        assert_eq!(shapes.can_be_unimodal(), None);
        assert_eq!(shapes.can_be_bell_shaped(), None);
        assert!(shapes.proportion_of(ShapeClass::Flat).is_nan());
    }
}
