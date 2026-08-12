//! Exhaustive cross-checks of CLOSURE's enumeration and its shape analysis.
//!
//! Rather than pinning a table of expected proportions — which would need
//! rewriting every time a shape definition is tuned, and those definitions are
//! the part most likely to change — these tests assert properties that must
//! hold whatever the definitions are:
//!
//! 1. an independent brute-force enumeration finds exactly the same result set;
//! 2. the DP counter in `count.rs`, which uses integer arithmetic, agrees with
//!    the float DFS on how many samples exist;
//! 3. shape classes partition the result set;
//! 4. `can_be_*` really is the disjunction over scanned samples, on every code
//!    path, so a path that forgets to compute shapes cannot report a proof;
//! 5. widening the rounding tolerance only ever grows the result set, so the
//!    `can_be_*` answers are monotone in it.
//!
//! The full grid runs in about four seconds under `cargo test`. Anything larger
//! is marked `#[ignore]`.

use closure_core::modality::{ShapeClass, DEFAULT_MODE_PROMINENCE};
use closure_core::{closure_count, closure_parallel};

/// The grid the assessment swept. 156 of these 280 cells are non-empty.
const NS: [i32; 4] = [20, 30, 52, 100];
const MEANS: [f64; 5] = [2.2, 3.0, 3.24, 3.5, 4.1];
const SDS: [f64; 6] = [0.6, 0.8, 1.0, 1.13, 1.3, 1.5];
const REMS: [f64; 2] = [0.005, 0.01];

const SCALE_MIN: i32 = 1;
const SCALE_MAX: i32 = 5;

fn run(mean: f64, sd: f64, n: i32, rem: f64) -> closure_core::ResultListFromMeanSdN<i32> {
    closure_parallel(mean, sd, n, SCALE_MIN, SCALE_MAX, rem, rem, 1, None, None)
        .expect("valid parameters")
}

/// Every count vector of `n` observations over the scale whose mean and sample
/// SD land inside the tolerance, found without using any CLOSURE code.
///
/// Deliberately naive: it walks all `C(n + k - 1, k - 1)` compositions and
/// filters. That is the point — it shares no pruning logic with the DFS under
/// test, so agreement between them is evidence about the DFS rather than about
/// a shared assumption.
fn brute_force(mean: f64, sd: f64, n: i32, rem: f64) -> Vec<Vec<u32>> {
    let k = (SCALE_MAX - SCALE_MIN + 1) as usize;
    let n_u = n as u32;
    let mut out = Vec::new();
    let mut current = vec![0u32; k];

    fn walk(
        i: usize,
        left: u32,
        current: &mut Vec<u32>,
        out: &mut Vec<Vec<u32>>,
        n: i32,
        mean: f64,
        sd: f64,
        rem: f64,
    ) {
        let k = current.len();
        if i == k - 1 {
            current[i] = left;
            let sum: i64 = current
                .iter()
                .enumerate()
                .map(|(j, &c)| c as i64 * (SCALE_MIN as i64 + j as i64))
                .sum();
            let sum_sq: i64 = current
                .iter()
                .enumerate()
                .map(|(j, &c)| {
                    let v = SCALE_MIN as i64 + j as i64;
                    c as i64 * v * v
                })
                .sum();
            let n_f = n as f64;
            let var = (sum_sq as f64 - (sum as f64) * (sum as f64) / n_f) / (n_f - 1.0);
            if var >= 0.0
                && (sum as f64 / n_f - mean).abs() <= rem + 1e-9
                && (var.sqrt() - sd).abs() <= rem + 1e-9
            {
                out.push(current.clone());
            }
            return;
        }
        for c in 0..=left {
            current[i] = c;
            walk(i + 1, left - c, current, out, n, mean, sd, rem);
        }
        current[i] = 0;
    }

    walk(0, n_u, &mut current, &mut out, n, mean, sd, rem);
    out.sort();
    out
}

fn sorted_rows(results: &closure_core::ResultListFromMeanSdN<i32>) -> Vec<Vec<u32>> {
    let mut rows: Vec<Vec<u32>> = results.results.counts.rows().map(|r| r.to_vec()).collect();
    rows.sort();
    rows
}

#[test]
fn closure_finds_exactly_what_brute_force_finds() {
    // Small n only: brute force is C(n+4, 4) vectors, so n=30 is already 46k.
    let mut checked = 0;
    for n in [12, 18, 25] {
        for mean in MEANS {
            for sd in [0.8, 1.0, 1.3] {
                let expected = brute_force(mean, sd, n, 0.005);
                let got = sorted_rows(&run(mean, sd, n, 0.005));
                assert_eq!(
                    got, expected,
                    "CLOSURE and brute force disagree at mean={mean} sd={sd} n={n}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 45);
}

#[test]
fn the_dp_counter_agrees_with_the_enumeration() {
    // `closure_count` reaches the same answer through integer arithmetic on
    // (sum, sum_sq) while the DFS accumulates in f64. Disagreement here is the
    // signature of a float boundary case being included by one and not the
    // other, which nothing else in the suite would catch.
    let mut compared = 0;
    for n in NS {
        for mean in MEANS {
            for sd in SDS {
                for rem in REMS {
                    let enumerated = run(mean, sd, n, rem).results.counts.nrow() as u64;
                    let counted = closure_count(mean, sd, n, SCALE_MIN, SCALE_MAX, rem, rem);
                    assert_eq!(
                        enumerated, counted,
                        "enumeration and DP count disagree at mean={mean} sd={sd} n={n} rem={rem}"
                    );
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, NS.len() * MEANS.len() * SDS.len() * REMS.len());
}

#[test]
fn shape_classes_partition_every_result_set() {
    let mut non_empty = 0;
    for n in NS {
        for mean in MEANS {
            for sd in SDS {
                let results = run(mean, sd, n, 0.005);
                let shapes = &results.modality_shapes;
                let total = results.results.counts.nrow() as u64;
                if total == 0 {
                    continue;
                }
                non_empty += 1;

                assert_eq!(shapes.n_scanned, total, "every sample must be classified");
                let summed: u64 = shapes.bounds.iter().map(|b| b.n_samples).sum();
                assert_eq!(
                    summed, total,
                    "class counts must sum to the number of samples at mean={mean} sd={sd} n={n}"
                );

                // Conditional bounds must be inside the unconditional ones: a
                // class is a subset of the result set, so it cannot range wider.
                for bounds in &shapes.bounds {
                    if bounds.n_samples == 0 {
                        continue;
                    }
                    for (i, (&lo, &hi)) in bounds
                        .count_lo
                        .iter()
                        .zip(bounds.count_hi.iter())
                        .enumerate()
                    {
                        assert!(lo <= hi);
                        assert!(
                            lo >= results.modality_counts.count_lo[i]
                                && hi <= results.modality_counts.count_hi[i],
                            "class {:?} bounds escape the overall range at value index {i}",
                            bounds.class
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        non_empty, 44,
        "the grid's live cells, pinned so a change is noticed"
    );
}

#[test]
fn can_be_is_the_disjunction_over_scanned_samples() {
    // Re-derives the flags from the raw count vectors and compares. This is the
    // assertion that catches a code path which forgot to run the scan: a
    // hardcoded `false` would pass every other test in this file.
    let prominence = ((DEFAULT_MODE_PROMINENCE * 100.0).ceil() as u32).max(1);
    for (mean, sd, n) in [
        (3.0, 1.13, 100),
        (3.5, 1.5, 100),
        (3.0, 0.8, 100),
        (2.2, 1.3, 100),
    ] {
        let results = run(mean, sd, n, 0.005);
        if results.results.counts.is_empty() {
            continue;
        }
        let shapes = &results.modality_shapes;
        assert!(shapes.exhaustive, "an unlimited run enumerates everything");

        let mut any_bell = false;
        let mut any_unimodal = false;
        for row in results.results.counts.rows() {
            let class = closure_core::modality::classify(row, prominence);
            any_bell |= class.is_bell();
            any_unimodal |= class.is_unimodal();
        }
        assert_eq!(shapes.can_be_bell_shaped(), Some(any_bell));
        assert_eq!(shapes.can_be_unimodal(), Some(any_unimodal));
    }
}

#[test]
fn a_truncated_search_never_reports_a_proof() {
    // `stop_after` sees part of the space, so an unseen shape is unknown rather
    // than impossible. A found shape is still sound: a witness is a witness.
    let results = closure_parallel(
        3.0,
        1.13,
        100,
        SCALE_MIN,
        SCALE_MAX,
        0.005,
        0.005,
        1,
        None,
        Some(5),
    )
    .unwrap();
    let shapes = &results.modality_shapes;
    assert!(!shapes.exhaustive);
    for answer in [
        shapes.can_be_bell_shaped(),
        shapes.can_be_unimodal(),
        shapes.can_be_multimodal(),
        shapes.can_be_j_shape_low(),
        shapes.can_be_j_shape_high(),
    ] {
        assert_ne!(
            answer,
            Some(false),
            "a partial scan must answer None, never a proof of impossibility"
        );
    }
}

#[test]
fn widening_the_tolerance_only_ever_adds_samples() {
    // S(0.005) is a subset of S(0.01), so every count of a shape class is
    // monotone in the tolerance, and so is each `can_be_*` answer. This is the
    // executable form of the instability that motivated reporting proportions:
    // the boolean can flip from false to true purely on the tolerance, and the
    // direction of that flip is fixed.
    for n in [30, 52, 100] {
        for mean in MEANS {
            for sd in [1.0, 1.13, 1.3, 1.5] {
                let tight = run(mean, sd, n, 0.005);
                let loose = run(mean, sd, n, 0.01);

                let tight_rows = sorted_rows(&tight);
                let loose_rows = sorted_rows(&loose);
                for row in &tight_rows {
                    assert!(
                        loose_rows.binary_search(row).is_ok(),
                        "a sample admissible at 0.005 must stay admissible at 0.01"
                    );
                }

                for class in ShapeClass::all() {
                    assert!(
                        tight.modality_shapes.n_of(class) <= loose.modality_shapes.n_of(class),
                        "class {class:?} shrank when the tolerance widened \
                         at mean={mean} sd={sd} n={n}"
                    );
                }
                if tight.modality_shapes.can_be_bell_shaped() == Some(true) {
                    assert_eq!(loose.modality_shapes.can_be_bell_shaped(), Some(true));
                }
            }
        }
    }
}

#[test]
fn a_ceiling_effect_is_not_reported_as_a_bell() {
    // The worked example from the assessment: M=3.50, SD=1.50, n=100, 1-5.
    // Every admissible dataset with a single mode has it at the ceiling, so a
    // paper describing these responses as bell-shaped is contradicted outright.
    //
    // Separating "how many modes" from "where" is what makes that sayable. The
    // one-axis predicate this replaced could only report unimodality, and its
    // answer flipped with the tolerance; `can_be_bell_shaped` does not.
    for rem in [0.005, 0.01] {
        let results = run(3.5, 1.5, 100, rem);
        let shapes = &results.modality_shapes;
        assert_eq!(shapes.n_of(ShapeClass::OneModeInterior), 0);
        assert_eq!(
            shapes.can_be_bell_shaped(),
            Some(false),
            "no interior mode is admissible at tolerance {rem}"
        );
        assert_eq!(
            shapes.can_be_unimodal(),
            Some(true),
            "but ceiling-peaked datasets are, and those have one mode"
        );
        assert!(shapes.n_of(ShapeClass::OneModeHighEdge) > 0);
        assert_eq!(shapes.n_of(ShapeClass::OneModeLowEdge), 0);
    }

    // Conditional bounds are what make the "our data were one of those" reply
    // checkable: within the ceiling-peaked class the whole distribution is
    // pinned far tighter than the reported mean and SD pin it on their own.
    let results = run(3.5, 1.5, 100, 0.01);
    let hi = results
        .modality_shapes
        .bounds
        .iter()
        .find(|b| b.class == ShapeClass::OneModeHighEdge)
        .unwrap();
    assert_eq!(hi.n_samples, 50);
    assert_eq!(hi.count_lo, vec![14, 12, 12, 13, 37]);
    assert_eq!(hi.count_hi, vec![16, 17, 18, 22, 41]);
    // Compare with the range the reported statistics allow on their own.
    assert_eq!(results.modality_counts.count_lo, vec![0, 0, 0, 0, 24]);
    assert_eq!(results.modality_counts.count_hi, vec![25, 50, 38, 51, 50]);
}

#[test]
fn the_representative_is_an_actual_sample_and_the_expectation_need_not_be() {
    let results = run(3.0, 1.13, 100, 0.005);
    let k = 5;
    let rep: Vec<u32> = results.frequency.f_representative()[..k]
        .iter()
        .map(|&f| f as u32)
        .collect();

    let rows: Vec<Vec<u32>> = sorted_rows(&results);
    assert!(
        rows.binary_search(&rep).is_ok(),
        "the representative must be a real member of the result set"
    );
    assert_eq!(rep.iter().sum::<u32>(), 100);

    // f_expected sums to n as well, but is generally fractional — which is
    // exactly why it must not be presented as a reconstruction.
    let expected = &results.frequency.f_expected()[..k];
    assert!((expected.iter().sum::<f64>() - 100.0).abs() < 1e-9);
    assert!(expected.iter().any(|f| (f - f.round()).abs() > 1e-9));
}

#[test]
#[ignore = "slow: n=400 takes ~7s in release and much longer in debug"]
fn large_n_still_agrees_with_the_dp_counter() {
    for n in [200, 400] {
        let enumerated = run(3.0, 1.3, n, 0.005).results.counts.nrow() as u64;
        let counted = closure_count(3.0, 1.3, n, SCALE_MIN, SCALE_MAX, 0.005, 0.005);
        assert_eq!(enumerated, counted, "disagreement at n={n}");
    }
}

/// Read the one-row `modality_summary.parquet` a run writes, as (column, value)
/// pairs rendered to strings.
fn read_summary(base: &str) -> std::collections::HashMap<String, String> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let path = format!("{base}modality_summary.parquet");
    let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().expect("one row group").unwrap();
    let schema = batch.schema();
    (0..batch.num_columns())
        .map(|i| {
            let name = schema.field(i).name().clone();
            let value = arrow::util::display::array_value_to_string(batch.column(i), 0).unwrap();
            (name, value)
        })
        .collect()
}

#[test]
fn streaming_reports_the_same_shapes_as_memory_mode() {
    // The regression this exists for: the streaming path used to emit a
    // hardcoded "no sample can be bell-shaped" because it never ran the scan.
    // A placeholder that spells "unknown" the same way as the strongest claim
    // the field can make is worse than no field at all, so pin the agreement.
    let dir = std::env::temp_dir().join("closure_streaming_shapes");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = format!("{}/", dir.display());

    closure_core::closure_parallel_streaming(
        3.5,
        1.5,
        100,
        SCALE_MIN,
        SCALE_MAX,
        0.01,
        0.01,
        1,
        closure_core::StreamingConfig::new(base.clone(), 1000, false),
        None,
    )
    .unwrap();

    let summary = read_summary(&base);
    let memory = run(3.5, 1.5, 100, 0.01);
    let shapes = &memory.modality_shapes;

    assert_eq!(summary["n_scanned"], shapes.n_scanned.to_string());
    assert_eq!(summary["exhaustive"], "true");
    for class in ShapeClass::all() {
        assert_eq!(
            summary[&format!("n_{}", class.as_str())],
            shapes.n_of(class).to_string(),
            "streaming and memory disagree on class {class:?}"
        );
    }
    // Specifically: the class that used to be reported as impossible by default.
    assert_eq!(summary["n_one_mode_high_edge"], "50");
    assert_eq!(summary["n_one_mode_interior"], "0");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn streaming_marks_a_truncated_run_as_not_exhaustive() {
    let dir = std::env::temp_dir().join("closure_streaming_partial");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = format!("{}/", dir.display());

    closure_core::closure_parallel_streaming(
        3.5,
        1.5,
        100,
        SCALE_MIN,
        SCALE_MAX,
        0.01,
        0.01,
        1,
        closure_core::StreamingConfig::new(base.clone(), 100, false),
        Some(20),
    )
    .unwrap();

    assert_eq!(read_summary(&base)["exhaustive"], "false");
    let _ = std::fs::remove_dir_all(&dir);
}
