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
//! [`DEFAULT_MODE_PROMINENCE`] of the sample above the higher of the two
//! valleys separating it from any taller peak. The threshold is a fraction of
//! `n`, so it does not silently change meaning with the length of the scale.
//! The tallest runs of a sample have no taller peak to be measured against, so
//! they are measured against each other: tied tallest runs separated by a
//! qualifying valley are distinct modes, and ones separated only by
//! sub-threshold dips merge into one broad mode. A unique tallest run is
//! therefore a mode at every threshold, which keeps a textbook-unimodal
//! histogram out of [`ShapeClass::Flat`] even on a grid too wide for any single
//! bar to clear the threshold outright.
//!
//! # Why one threshold is not enough
//!
//! No value of that threshold is validated, and the crate's strongest output —
//! an empty shape class, i.e. a proof that the raw data did not have that shape
//! — would otherwise rest entirely on it, and in both directions: lowering the
//! threshold splits modes and manufactures proofs that the data could not have
//! been unimodal, raising it merges modes and manufactures proofs that the data
//! could not have been multimodal. Every sample is therefore classified at
//! each threshold in [`DEFAULT_PROMINENCE_LADDER`], and
//! [`ModalityShapes::can_be`] answers `Some(false)` only when the class is empty
//! across all of them. [`ModalityShapes::min_prominence_admitting`] reports the
//! margin.
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

/// The band of prominence thresholds every result set is classified over.
///
/// A shape class being empty is the strongest thing this crate says — it is the
/// deductive claim that the raw data did not have that shape. That claim must
/// not rest on one tuning constant, so [`ShapeAccumulator`] classifies each
/// sample at every threshold in this band as well as at the configured one, and
/// [`ModalityShapes::can_be`] only answers `Some(false)` when the class is
/// empty across all of them. A reader sees the whole envelope in
/// `modality_prominence.parquet` rather than one point from it.
///
/// # Why the band ends where it does
///
/// The lower end is a noise floor: below about 2% of `n` a one- or two-count
/// wobble creates a mode, which is the failure a prominence rule exists to
/// prevent.
///
/// The upper end is set by where the classification stops meaning what its name
/// says. Prominence suppresses a peak by comparing it to a *taller* peak, and it
/// does not care how much of the sample the suppressed peak holds. At 15% of `n`
/// on a 1-5 scale, `(20, 7, 8, 33, 32)` classifies as [`OneModeInterior`] — the
/// floor spike of 20 responses is erased because it stands only 13 above the
/// valley beside it, and a distribution with a fifth of the sample at the bottom
/// of the scale is then reported as a candidate bell. Past roughly 10% the rule
/// is no longer measuring modality, so extending the band further would weaken
/// every `Some(false)` for no gain in honesty.
///
/// [`OneModeInterior`]: ShapeClass::OneModeInterior
pub const DEFAULT_PROMINENCE_LADDER: [f64; 3] = [0.02, 0.05, 0.10];

/// Where a sample's mass sits, crossing mode count with mode location.
///
/// The variants partition the space of count vectors: [`classify`] returns
/// exactly one for any input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumCountMacro, EnumIter, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum ShapeClass {
    /// No distinguishable mode at the threshold: every count is the same, or
    /// the tied tallest runs and the sub-threshold dips between them span the
    /// entire scale. See [`modes`].
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
/// at either end of the scale), and that stands out from the rest of the sample:
///
/// - A run *below* the tallest needs topographic prominence of at least
///   `min_prominence_counts` observations: its height minus the higher of its
///   two key cols, where a key col is the lowest point between the run and the
///   nearest strictly taller run on that side (a side with no taller run
///   contributes a col of 0).
/// - Runs *at* the tallest height have no taller peak to be measured against,
///   so they are measured against each other: consecutive tied tallest runs
///   merge into one broad mode when the valley between them dips less than
///   `min_prominence_counts` below the summit, and stay distinct modes when it
///   dips at least that far. A unique tallest run is thus a mode at every
///   threshold — which keeps a textbook-unimodal histogram on a wide grid,
///   whose tallest bar is only about `n / k`, from being reported as having no
///   mode. Wide grids arise for real: a multi-item SPRITE grid is
///   `(scale_max - scale_min) * items + 1` positions across.
///
/// Merging (rather than exempting each tied tallest run from the threshold) is
/// what stops a one-count wobble around a tied maximum, like
/// `(30, 29, 30, 29, 30)`, from reading as multimodal at every threshold.
///
/// A sample therefore has no mode in exactly two cases, and both classify as
/// [`ShapeClass::Flat`]: every count is equal, or the tied tallest runs and
/// every dip between them merge into a single region spanning the entire scale
/// — flat at this resolution.
///
/// Each mode is reported by the first position of its (first) run.
pub fn modes(counts: &[u32], min_prominence_counts: u32) -> Vec<usize> {
    mode_extents(&runs_of(counts), min_prominence_counts)
        .into_iter()
        .map(|(start, _)| start)
        .collect()
}

/// The modes of a sample as grid-position extents `(start, end)`, inclusive,
/// ascending — the engine behind [`modes`], on a run decomposition computed
/// once by the caller.
///
/// Classifying one sample at several thresholds shares the decomposition rather
/// than rebuilding it per threshold. Extents matter because merged tied tallest
/// runs form one broad mode; [`classify`] needs to know whether that mode
/// touches a scale boundary.
fn mode_extents(runs: &[Run], min_prominence_counts: u32) -> Vec<(usize, usize)> {
    // A single run means every count is equal: flat, no mode.
    if runs.len() < 2 {
        return Vec::new();
    }
    let summit = runs
        .iter()
        .map(|r| r.height)
        .max()
        .expect("runs is non-empty");

    // Group the summit runs: consecutive tied tallest runs merge when the
    // valley between them dips less than the threshold below the summit — at
    // this resolution the dip is texture, not a separation. Merging is
    // transitive by construction and keeps the result mirror-symmetric, which
    // designating one tied run as "the" maximum would not.
    let mut groups: Vec<(usize, usize)> = Vec::new(); // (first, last) run index
    for (i, run) in runs.iter().enumerate() {
        if run.height < summit {
            continue;
        }
        let merges = groups.last().is_some_and(|&(_, last)| {
            // Two summit runs cannot be adjacent (adjacent runs differ in
            // height), so there is at least one run between them.
            let valley = runs[last + 1..i]
                .iter()
                .map(|r| r.height)
                .min()
                .expect("summit runs are separated by at least one run");
            summit - valley < min_prominence_counts
        });
        match groups.last_mut() {
            Some(group) if merges => group.1 = i,
            _ => groups.push((i, i)),
        }
    }

    // A single group reaching from the first run to the last means every dip
    // in the vector is below the threshold: the sample is flat at this
    // resolution and has no distinguishable mode, exactly like a constant
    // vector. No lower run can qualify either — any local maximum sits between
    // two merged summit runs, so its key cols are at least the sub-threshold
    // valley floor and its prominence falls short of the threshold.
    if groups == [(0, runs.len() - 1)] {
        return Vec::new();
    }

    let mut out: Vec<(usize, usize)> = groups
        .iter()
        .map(|&(first, last)| (runs[first].start, runs[last].end))
        .collect();

    for (i, run) in runs.iter().enumerate() {
        if run.height == summit {
            continue; // handled by the grouping above
        }
        let higher_left = i > 0 && runs[i - 1].height > run.height;
        let higher_right = i + 1 < runs.len() && runs[i + 1].height > run.height;
        if higher_left || higher_right {
            continue; // not a local maximum
        }

        // Key col on each side: the lowest run between here and the nearest
        // strictly taller run. The summit is strictly taller than this run, so
        // at least one side has one; a side without one contributes 0.
        let left_col = runs[..i]
            .iter()
            .rposition(|r| r.height > run.height)
            .map(|j| {
                runs[j + 1..i]
                    .iter()
                    .map(|r| r.height)
                    .min()
                    .unwrap_or(run.height)
            });
        let right_col = runs[i + 1..]
            .iter()
            .position(|r| r.height > run.height)
            .map(|offset| {
                let j = i + 1 + offset;
                runs[i + 1..j]
                    .iter()
                    .map(|r| r.height)
                    .min()
                    .unwrap_or(run.height)
            });

        let col = left_col.unwrap_or(0).max(right_col.unwrap_or(0));
        if run.height - col >= min_prominence_counts {
            out.push((run.start, run.end));
        }
    }
    out.sort_unstable();
    out
}

/// Classify a count vector into exactly one [`ShapeClass`].
///
/// `min_prominence_counts` is the prominence a local maximum needs to count as a
/// mode; see [`modes`].
pub fn classify(counts: &[u32], min_prominence_counts: u32) -> ShapeClass {
    classify_from_runs(&runs_of(counts), counts.len(), min_prominence_counts)
}

/// [`classify`], on a run decomposition computed once by the caller.
fn classify_from_runs(runs: &[Run], k: usize, min_prominence_counts: u32) -> ShapeClass {
    let peaks = mode_extents(runs, min_prominence_counts);
    match peaks[..] {
        // No distinguishable mode: every count is equal, or nothing but
        // sub-threshold texture across the whole scale. See [`modes`].
        [] => ShapeClass::Flat,
        // A lone mode spanning both edges would have classified as `Flat`
        // above, so the edge tests below are mutually exclusive.
        [(start, end)] => {
            if start == 0 {
                ShapeClass::OneModeLowEdge
            } else if end == k - 1 {
                ShapeClass::OneModeHighEdge
            } else {
                ShapeClass::OneModeInterior
            }
        }
        [_, _] => ShapeClass::TwoModes,
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

/// Per-class sample counts at one prominence threshold.
///
/// One rung of the envelope described on [`DEFAULT_PROMINENCE_LADDER`]. The
/// conditional bounds in [`ModalityShapes::bounds`] are computed only at the
/// primary rung; these counts exist so a reader can see how far the threshold
/// would have to move before a class stops being empty.
#[derive(Clone, Debug)]
pub struct LadderRung {
    /// Threshold as a fraction of `n`.
    pub min_prominence: f64,
    /// The same threshold in observations, after rounding up and flooring at 1.
    /// Two rungs can share this value at small `n`, which is itself worth
    /// seeing: it means the envelope is narrower than it looks.
    pub min_prominence_counts: u32,
    /// True for the rung that [`ModalityShapes::bounds`] was computed at.
    pub primary: bool,
    /// Samples per class, indexed in [`ShapeClass`] declaration order.
    pub n_per_class: Vec<u64>,
}

impl LadderRung {
    /// Samples of `class` at this threshold.
    pub fn n_of(&self, class: ShapeClass) -> u64 {
        self.n_per_class.get(class as usize).copied().unwrap_or(0)
    }
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
    /// One entry per [`ShapeClass`], in declaration order. Counts and bounds
    /// here are at the primary prominence threshold, [`Self::min_prominence`].
    pub bounds: Vec<ShapeBounds>,
    /// Per-class counts at every threshold in the envelope, ascending.
    /// See [`DEFAULT_PROMINENCE_LADDER`].
    pub ladder: Vec<LadderRung>,
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
    /// Number of scanned samples in `class`, at the primary threshold.
    pub fn n_of(&self, class: ShapeClass) -> u64 {
        self.bounds
            .iter()
            .find(|b| b.class == class)
            .map_or(0, |b| b.n_samples)
    }

    /// The lowest threshold in the envelope at which `class` has any member,
    /// or `None` when it is empty throughout.
    ///
    /// This is the robustness number behind a `Some(false)`: it says how far the
    /// prominence rule would have to be relaxed or tightened before the shape
    /// became admissible at all. `None` means no threshold in the defensible
    /// band admits it.
    pub fn min_prominence_admitting(&self, class: ShapeClass) -> Option<f64> {
        self.ladder
            .iter()
            .filter(|rung| rung.n_of(class) > 0)
            .map(|rung| rung.min_prominence)
            .fold(None, |acc: Option<f64>, p| {
                Some(acc.map_or(p, |a| a.min(p)))
            })
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
    /// - `Some(true)`: a scanned sample has such a shape at *some* threshold in
    ///   the prominence envelope, so it is possible. Sound whether or not the
    ///   search was exhaustive — a witness is a witness.
    /// - `Some(false)`: **no** admissible sample has such a shape at *any*
    ///   threshold in the envelope, so the raw data did not either. Only ever
    ///   returned for an exhaustive search.
    /// - `None`: none were found, but the search was partial, so nothing follows.
    ///
    /// The `None` case is why this returns an `Option`. A truncated CLOSURE run
    /// and a SPRITE run both fail to see most of the space; reporting their
    /// silence as `false` would turn "we did not look" into "we proved it
    /// impossible".
    ///
    /// # Why this quantifies over the envelope
    ///
    /// Mode detection needs a prominence threshold and no single value of it is
    /// validated. Answering from one threshold would make the crate's strongest
    /// output — a proof of impossibility — contingent on that constant, in both
    /// directions: lowering the threshold splits modes and manufactures
    /// `Some(false)` for unimodal shapes, raising it merges modes and
    /// manufactures `Some(false)` for multimodal ones. Requiring the class to be
    /// empty across the whole band of [`DEFAULT_PROMINENCE_LADDER`] makes the
    /// answer conservative in its free parameter. Use
    /// [`Self::min_prominence_admitting`] to see the margin, and
    /// [`Self::can_be_at_primary`] for the single-threshold answer.
    pub fn can_be(&self, predicate: impl Fn(ShapeClass) -> bool) -> Option<bool> {
        let found = self
            .ladder
            .iter()
            .any(|rung| ShapeClass::all().any(|c| predicate(c) && rung.n_of(c) > 0));
        match (found, self.exhaustive) {
            (true, _) => Some(true),
            (false, true) => Some(false),
            (false, false) => None,
        }
    }

    /// [`Self::can_be`], answered from the primary threshold alone.
    ///
    /// Reported for continuity with the per-class counts and conditional bounds,
    /// which are also computed at that threshold. Prefer [`Self::can_be`] for
    /// any claim that leaves the process: a `Some(false)` here is only as good
    /// as [`Self::min_prominence`].
    pub fn can_be_at_primary(&self, predicate: impl Fn(ShapeClass) -> bool) -> Option<bool> {
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
    min_prominence: f64,
    /// Every threshold in the envelope, ascending, as (fraction, observations).
    /// Always contains `min_prominence`; see [`DEFAULT_PROMINENCE_LADDER`].
    ladder: Vec<(f64, u32)>,
    /// Index into `ladder` of the primary threshold.
    primary: usize,
    /// `[rung][class]` sample counts.
    ladder_counts: Vec<Vec<u64>>,
    n_scanned: u64,
    per_class: Vec<ShapeBounds>,
    deficit_min: u32,
    deficit_max: u32,
    deficit_sum: u64,
}

impl ShapeAccumulator {
    /// Start accumulating over count vectors of width `k` from samples of size
    /// `n`, requiring `min_prominence` (a fraction of `n`) for a mode.
    ///
    /// Per-class counts and conditional bounds are reported at
    /// `min_prominence`. Class counts are additionally tracked across
    /// [`DEFAULT_PROMINENCE_LADDER`], which is what lets
    /// [`ModalityShapes::can_be`] be conservative in the threshold.
    pub fn new(k: usize, n: usize, min_prominence: f64) -> Self {
        // At least one observation, so a threshold of 0 does not admit ties as
        // separate modes.
        let to_counts = |p: f64| ((p * n as f64).ceil() as u32).max(1);

        let mut fractions: Vec<f64> = DEFAULT_PROMINENCE_LADDER.to_vec();
        // The configured threshold is always a rung, so `can_be` can never
        // disagree with `can_be_at_primary` about a witness.
        if !fractions.contains(&min_prominence) {
            fractions.push(min_prominence);
        }
        fractions.sort_by(|a, b| a.partial_cmp(b).expect("prominences are finite"));
        let ladder: Vec<(f64, u32)> = fractions.iter().map(|&p| (p, to_counts(p))).collect();
        let primary = ladder
            .iter()
            .position(|&(p, _)| p == min_prominence)
            .expect("the configured threshold was just inserted");
        let n_classes = ShapeClass::all().count();

        Self {
            k,
            min_prominence,
            ladder_counts: vec![vec![0u64; n_classes]; ladder.len()],
            ladder,
            primary,
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
        // One run decomposition serves every threshold in the envelope.
        let runs = runs_of(counts);
        let mut primary_class = None;
        for (rung, &(_, threshold)) in self.ladder.iter().enumerate() {
            let class = classify_from_runs(&runs, counts.len(), threshold);
            self.ladder_counts[rung][class as usize] += 1;
            if rung == self.primary {
                primary_class = Some(class);
            }
        }

        let class = primary_class.expect("the primary threshold is always a rung");
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

    /// Fold another accumulator over the same grid and thresholds into this
    /// one, so chunks of a result set can be scanned in parallel.
    pub fn merge(&mut self, other: Self) {
        debug_assert_eq!(self.k, other.k);
        debug_assert_eq!(self.ladder, other.ladder);
        for (mine, theirs) in self.ladder_counts.iter_mut().zip(other.ladder_counts) {
            for (a, b) in mine.iter_mut().zip(theirs) {
                *a += b;
            }
        }
        for (mine, theirs) in self.per_class.iter_mut().zip(other.per_class) {
            if theirs.n_samples == 0 {
                continue;
            }
            if mine.n_samples == 0 {
                *mine = theirs;
                continue;
            }
            for (lo, &other_lo) in mine.count_lo.iter_mut().zip(&theirs.count_lo) {
                *lo = (*lo).min(other_lo);
            }
            for (hi, &other_hi) in mine.count_hi.iter_mut().zip(&theirs.count_hi) {
                *hi = (*hi).max(other_hi);
            }
            mine.n_samples += theirs.n_samples;
        }
        self.deficit_min = self.deficit_min.min(other.deficit_min);
        self.deficit_max = self.deficit_max.max(other.deficit_max);
        self.deficit_sum += other.deficit_sum;
        self.n_scanned += other.n_scanned;
    }

    /// Finish, declaring whether the scan covered the whole solution space.
    ///
    /// Pass `false` for a truncated search or for SPRITE. That does not weaken
    /// the counts or the conditional bounds — it only stops
    /// [`ModalityShapes::can_be`] from reporting an absence as a proof.
    pub fn finish(self, exhaustive: bool) -> ModalityShapes {
        let n = self.n_scanned;
        let primary = self.primary;
        let ladder = self
            .ladder
            .iter()
            .zip(self.ladder_counts)
            .enumerate()
            .map(|(i, (&(min_prominence, counts), n_per_class))| LadderRung {
                min_prominence,
                min_prominence_counts: counts,
                primary: i == primary,
                n_per_class,
            })
            .collect();
        ModalityShapes {
            exhaustive,
            n_scanned: n,
            bounds: self.per_class,
            ladder,
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
    fn the_tallest_bar_is_a_mode_however_wide_the_grid() {
        // A textbook-unimodal histogram spread over 41 grid positions: 10
        // everywhere, 11 at the centre. The tallest bar holds 11 of 411
        // observations, so it cannot clear a 5%-of-n threshold (21) on its own.
        //
        // Treating "no bar clears the threshold" as flat would put this in
        // `Flat`, which is not unimodal, and an exhaustive scan over samples
        // like it would then answer `Some(false)` to *every* `can_be` query at
        // once — proving the data had no shape at all. A unique tallest run is
        // therefore always a mode.
        let mut hump = vec![10u32; 41];
        hump[20] = 11;
        let n: u32 = hump.iter().sum();
        assert_eq!(n, 411);
        let thresh = ((DEFAULT_MODE_PROMINENCE * n as f64).ceil() as u32).max(1);
        assert!(thresh > 11, "the premise: no bar reaches the threshold");

        assert_eq!(modes(&hump, thresh), vec![20]);
        assert_eq!(classify(&hump, thresh), ShapeClass::OneModeInterior);
        // The threshold-free measure agrees it is unimodal, which is what makes
        // a `Flat` verdict here demonstrably wrong rather than a judgement call.
        assert_eq!(unimodality_deficit(&hump), 0);

        let mut acc = ShapeAccumulator::new(41, n as usize, DEFAULT_MODE_PROMINENCE);
        acc.update(&hump);
        let shapes = acc.finish(true);
        assert_eq!(shapes.can_be_unimodal(), Some(true));
        assert_eq!(shapes.can_be_bell_shaped(), Some(true));
    }

    #[test]
    fn tied_maxima_do_not_bypass_the_threshold() {
        // The module doc's motivating example with its unique maximum removed:
        // a one-count wobble around a tied maximum. Exempting every tallest run
        // from the threshold would make this `ThreeOrMoreModes` at *any*
        // threshold — a multimodality witness, and a threshold-immune proof
        // that the data could not have been unimodal, both manufactured by a
        // ±1-count wobble that no rung of the prominence band could veto.
        let wobble = [30u32, 29, 30, 29, 30];
        let n: u32 = wobble.iter().sum();
        let thresh = ((DEFAULT_MODE_PROMINENCE * n as f64).ceil() as u32).max(1);
        assert_eq!(modes(&wobble, thresh), Vec::<usize>::new());
        assert_eq!(classify(&wobble, thresh), ShapeClass::Flat);

        // At a resolution finer than the dips the same vector has three modes:
        // `Flat` is a statement at a threshold, not an absolute.
        assert_eq!(classify(&wobble, 1), ShapeClass::ThreeOrMoreModes);

        // Through the accumulator it witnesses nothing anywhere in the band.
        let mut acc = ShapeAccumulator::new(5, n as usize, DEFAULT_MODE_PROMINENCE);
        acc.update(&wobble);
        let shapes = acc.finish(true);
        assert_eq!(shapes.can_be_multimodal(), Some(false));
        assert_eq!(shapes.can_be_unimodal(), Some(false));
    }

    #[test]
    fn tied_maxima_with_a_qualifying_valley_are_separate_modes() {
        // A genuine U: the valley dips far below the tied rims, so merging
        // must not kick in and the sample stays bimodal.
        let u = [40u32, 10, 5, 10, 40];
        let n: u32 = u.iter().sum();
        let thresh = ((DEFAULT_MODE_PROMINENCE * n as f64).ceil() as u32).max(1);
        assert_eq!(modes(&u, thresh), vec![0, 4]);
        assert_eq!(classify(&u, thresh), ShapeClass::TwoModes);
    }

    #[test]
    fn merged_tied_maxima_are_one_broad_mode_and_mirror_cleanly() {
        // Tied maxima at positions 0 and 2 with a one-count dip between them:
        // one broad mode spanning 0..=2, which touches the low edge.
        let low = [30u32, 29, 30, 20, 10];
        let n: u32 = low.iter().sum();
        let thresh = ((DEFAULT_MODE_PROMINENCE * n as f64).ceil() as u32).max(1);
        assert_eq!(modes(&low, thresh), vec![0]);
        assert_eq!(classify(&low, thresh), ShapeClass::OneModeLowEdge);

        // Its mirror image must classify as the mirrored class — a rule that
        // designated one tied run as "the" maximum would break this.
        let mut high = low;
        high.reverse();
        assert_eq!(classify(&high, thresh), ShapeClass::OneModeHighEdge);

        // Interior ties merge to one interior mode.
        let mid = [10u32, 30, 29, 30, 10];
        let n: u32 = mid.iter().sum();
        let thresh = ((DEFAULT_MODE_PROMINENCE * n as f64).ceil() as u32).max(1);
        assert_eq!(classify(&mid, thresh), ShapeClass::OneModeInterior);
    }

    #[test]
    fn flat_is_no_distinguishable_mode_at_the_threshold() {
        // Constant vectors are flat at every threshold.
        assert_eq!(classify(&[7, 7, 7, 7, 7], 1), ShapeClass::Flat);
        assert_eq!(classify(&[7, 7, 7, 7, 7], 1000), ShapeClass::Flat);
        assert_eq!(classify(&[], 1), ShapeClass::Flat);

        // A shallow comb: 21 tied maxima, every dip one count deep. At a 5%
        // threshold the dips are texture, the whole scale is one summit, and
        // the sample is flat at this resolution — in particular it is not a
        // multimodality witness. At a threshold the dips clear, the modes are
        // real and the same vector reads as multimodal.
        let comb: Vec<u32> = (0..41).map(|i| if i % 2 == 0 { 3 } else { 2 }).collect();
        let n: u32 = comb.iter().sum();
        let thresh = ((DEFAULT_MODE_PROMINENCE * n as f64).ceil() as u32).max(1);
        assert!(thresh > 1, "the premise: the dips sit below the threshold");
        assert_eq!(classify(&comb, thresh), ShapeClass::Flat);
        assert_eq!(classify(&comb, 1), ShapeClass::ThreeOrMoreModes);
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
    fn a_proof_of_impossibility_must_survive_the_whole_prominence_band() {
        // `[10, 40, 30, 38, 10]`: the second peak stands 8 above the valley
        // separating it from the taller one. That clears the 0.02 and 0.05
        // rungs (3 and 7 observations at n=128) but not the 0.10 rung (13), so
        // the sample reads as two modes at the primary threshold and as one
        // interior mode at the top of the band.
        let mut acc = ShapeAccumulator::new(5, 128, DEFAULT_MODE_PROMINENCE);
        acc.update(&[10, 40, 30, 38, 10]);
        let shapes = acc.finish(true);

        assert_eq!(shapes.n_of(ShapeClass::TwoModes), 1, "primary threshold");
        assert_eq!(shapes.n_of(ShapeClass::OneModeInterior), 0);

        // Answering from the primary threshold alone would call a bell shape
        // impossible. It is not — it is impossible *at that threshold*, and the
        // threshold is a judgement call, so the deductive claim is not available.
        assert_eq!(shapes.can_be_at_primary(|c| c.is_bell()), Some(false));
        assert_eq!(shapes.can_be_bell_shaped(), Some(true));
        assert_eq!(
            shapes.min_prominence_admitting(ShapeClass::OneModeInterior),
            Some(0.10)
        );

        // The same guard in the other direction: a shape visible only at the
        // bottom of the band is still a witness.
        let mut acc = ShapeAccumulator::new(5, 133, DEFAULT_MODE_PROMINENCE);
        acc.update(&[10, 40, 35, 38, 10]);
        let shapes = acc.finish(true);
        assert_eq!(shapes.n_of(ShapeClass::OneModeInterior), 1, "primary");
        assert_eq!(shapes.can_be_at_primary(|c| c.is_multimodal()), Some(false));
        assert_eq!(shapes.can_be_multimodal(), Some(true));
        assert_eq!(
            shapes.min_prominence_admitting(ShapeClass::TwoModes),
            Some(0.02)
        );
    }

    #[test]
    fn the_band_always_contains_the_configured_threshold() {
        // Otherwise `can_be` could miss a witness that `can_be_at_primary` sees.
        for prominence in [0.01, 0.05, 0.07, 0.25] {
            let acc = ShapeAccumulator::new(5, 100, prominence);
            let shapes = acc.finish(true);
            let primary: Vec<&LadderRung> = shapes.ladder.iter().filter(|r| r.primary).collect();
            assert_eq!(primary.len(), 1, "exactly one primary rung");
            assert_eq!(primary[0].min_prominence, prominence);
            assert_eq!(shapes.min_prominence, prominence);
            // Rungs are ascending and each is reported in observations too, so a
            // reader can see when two of them collapse onto the same threshold.
            assert!(shapes
                .ladder
                .windows(2)
                .all(|w| w[0].min_prominence < w[1].min_prominence));
        }
    }

    #[test]
    fn the_band_collapses_visibly_at_small_n() {
        // At n = 20 the 0.02 and 0.05 rungs both round to a single observation,
        // so the envelope is narrower than its three entries suggest. The
        // per-rung observation count is what makes that visible rather than
        // implied.
        let acc = ShapeAccumulator::new(5, 20, DEFAULT_MODE_PROMINENCE);
        let shapes = acc.finish(true);
        let counts: Vec<u32> = shapes
            .ladder
            .iter()
            .map(|r| r.min_prominence_counts)
            .collect();
        assert_eq!(counts, vec![1, 1, 2]);
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
    fn merging_accumulators_matches_one_sequential_scan() {
        let rows: Vec<[u32; 4]> = (0..=9u32)
            .flat_map(|a| {
                (0..=(9 - a))
                    .flat_map(move |b| (0..=(9 - a - b)).map(move |c| [a, b, c, 9 - a - b - c]))
            })
            .collect();
        let mut whole = ShapeAccumulator::new(4, 9, DEFAULT_MODE_PROMINENCE);
        for row in &rows {
            whole.update(row);
        }
        let mut merged = ShapeAccumulator::new(4, 9, DEFAULT_MODE_PROMINENCE);
        for chunk in rows.chunks(37) {
            let mut part = ShapeAccumulator::new(4, 9, DEFAULT_MODE_PROMINENCE);
            for row in chunk {
                part.update(row);
            }
            merged.merge(part);
        }
        // An empty accumulator is an identity.
        merged.merge(ShapeAccumulator::new(4, 9, DEFAULT_MODE_PROMINENCE));

        let (whole, merged) = (whole.finish(true), merged.finish(true));
        assert_eq!(whole.n_scanned, merged.n_scanned);
        assert_eq!(whole.deficit_min, merged.deficit_min);
        assert_eq!(whole.deficit_max, merged.deficit_max);
        assert_eq!(whole.deficit_mean, merged.deficit_mean);
        for (a, b) in whole.bounds.iter().zip(&merged.bounds) {
            assert_eq!(
                (a.class, a.n_samples, &a.count_lo, &a.count_hi),
                (b.class, b.n_samples, &b.count_lo, &b.count_hi)
            );
        }
        for (a, b) in whole.ladder.iter().zip(&merged.ladder) {
            assert_eq!(a.n_per_class, b.n_per_class);
        }
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
