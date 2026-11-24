//! Regression tests for correctness bugs found in the CLOSURE search, the DP
//! counter and the streaming statistics. Each test names the failure it pins.

use closure_core::{closure_count, closure_parallel, closure_parallel_streaming, StreamingConfig};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

fn f64_cell(path: &str, column: &str) -> f64 {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    let idx = batch.schema().index_of(column).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap()
        .value(0)
}

fn bool_cell(path: &str, column: &str) -> bool {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let batch = reader.next().unwrap().unwrap();
    let idx = batch.schema().index_of(column).unwrap();
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .unwrap()
        .value(0)
}

#[test]
fn a_sample_size_equal_to_the_seed_depth_is_still_checked() {
    // n = 2 is exactly the DFS seed depth, so the search never enters the loop
    // that checks the sum and upper SD bounds. The terminal check has to.
    let results =
        closure_parallel::<f64, i32>(3.0, 0.0, 2, 1, 5, 0.0, 0.0, 1, None, None, None).unwrap();
    let samples: Vec<Vec<i32>> = (0..results.results.len())
        .map(|i| results.results.sample(i))
        .collect();
    assert_eq!(samples, vec![vec![3, 3]]);
    assert_eq!(closure_count(3.0, 0.0, 2, 1, 5, 0.0, 0.0).unwrap(), 1);
}

#[test]
fn stop_after_is_an_upper_bound_in_every_path() {
    for limit in [5, 150] {
        let results = closure_parallel::<f64, i32>(
            3.0,
            1.0,
            40,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            Some(limit),
            None,
        )
        .unwrap();
        assert_eq!(results.results.len(), limit, "stop_after = {limit}");
    }
}

#[test]
fn a_sample_exactly_on_a_bound_is_found() {
    // 3.24 * 25 is 81.00000000000001 in f64, so with no tolerance a sum of
    // exactly 81 used to fall outside the target interval in both the DFS and
    // the DP counter.
    let results =
        closure_parallel::<f64, i32>(3.24, 1.0, 25, 1, 5, 0.0, 0.05, 1, None, None, None).unwrap();
    assert!(!results.results.is_empty());
    assert_eq!(
        results.results.len() as u64,
        closure_count(3.24, 1.0, 25, 1, 5, 0.0, 0.05).unwrap()
    );
}

#[test]
fn the_counter_agrees_with_the_search_on_negative_scales() {
    for (mean, sd, n, lo, hi) in [(0.0, 1.5, 10, -3, 3), (-1.0, 1.0, 8, -3, 1)] {
        let results =
            closure_parallel::<f64, i32>(mean, sd, n, lo, hi, 0.05, 0.05, 1, None, None, None)
                .unwrap();
        assert_eq!(
            closure_count(mean, sd, n, lo, hi, 0.05, 0.05).unwrap(),
            results.results.len() as u64,
            "mean={mean} sd={sd} n={n} scale=[{lo},{hi}]"
        );
    }
}

#[test]
fn streaming_statistics_describe_the_samples_written() {
    let dir = std::env::temp_dir().join("closure_regression_streaming");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = format!("{}/", dir.display());

    // A limit above 100 takes the parallel path, which used to fold every
    // sample a branch produced into the statistics before truncating what was
    // written.
    let result = closure_parallel_streaming::<f64, i32>(
        3.0,
        1.0,
        40,
        1,
        5,
        0.05,
        0.05,
        1,
        StreamingConfig::new(base.clone(), 50, false),
        Some(150),
        None,
    )
    .unwrap();
    assert_eq!(result.total_combinations, 150);
    assert_eq!(
        f64_cell(&format!("{base}metrics_main.parquet"), "samples_all"),
        150.0
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_streaming_run_still_writes_its_statistics() {
    let dir = std::env::temp_dir().join("closure_regression_empty_streaming");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let base = format!("{}/", dir.display());

    // Impossible SD: no sample exists. The parallel path used to return early
    // without writing the statistics files that the sequential path writes.
    let result = closure_parallel_streaming::<f64, i32>(
        3.0,
        5.0,
        40,
        1,
        5,
        0.05,
        0.05,
        1,
        StreamingConfig::new(base.clone(), 50, false),
        None,
        None,
    )
    .unwrap();
    assert_eq!(result.total_combinations, 0);
    assert_eq!(
        f64_cell(&format!("{base}metrics_main.parquet"), "samples_all"),
        0.0
    );
    // The search was complete; it found nothing. That used to be written as
    // `exhaustive = false`.
    assert!(bool_cell(
        &format!("{base}modality_summary.parquet"),
        "exhaustive"
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_complete_search_that_finds_nothing_is_exhaustive() {
    // Mean 1.5 with n = 3 needs a sum of 4.5: no sample exists. The empty
    // result used to say `exhaustive = false` and list every cutoff as one
    // observation, as if n were 0.
    let results =
        closure_parallel::<f64, i32>(1.5, 0.5, 3, 1, 3, 0.0, 0.0, 1, None, None, None).unwrap();
    assert_eq!(results.results.len(), 0);
    let shapes = &results.modality_shapes;
    assert!(shapes.exhaustive);
    assert_eq!(
        shapes.can_be_unimodal(),
        None,
        "no samples, no shape inference"
    );

    let results =
        closure_parallel::<f64, i32>(3.0, 5.0, 100, 1, 5, 0.05, 0.05, 1, None, None, None).unwrap();
    let rungs: Vec<u32> = results
        .modality_shapes
        .ladder
        .iter()
        .map(|r| r.min_prominence_counts)
        .collect();
    assert_eq!(rungs, vec![2, 5, 10]);
}

#[test]
fn invalid_parameters_are_errors_not_panics() {
    assert!(
        closure_parallel::<f64, i32>(3.0, 1.0, 1, 1, 5, 0.0, 0.0, 1, None, None, None).is_err()
    );
    assert!(
        closure_parallel::<f64, i32>(3.0, 1.0, 0, 1, 5, 0.0, 0.0, 1, None, None, None).is_err()
    );
    assert!(
        closure_parallel::<f64, i32>(3.0, 1.0, 10, 5, 1, 0.0, 0.0, 1, None, None, None).is_err()
    );
    assert!(
        closure_parallel::<f64, i32>(3.0, -1.0, 10, 1, 5, 0.0, 0.0, 1, None, None, None).is_err()
    );
    // A scale outside the output grid's range: this used to finish the search
    // and then panic converting the bounds to i32.
    let far = 3_000_000_000i64;
    assert!(
        closure_parallel::<f64, i64>(3e9, 1.0, 5, far, far + 4, 0.0, 0.0, 1, None, None, None)
            .is_err()
    );
    // Within the grid's range, but n·Σx² would overflow i64.
    assert!(closure_parallel::<f64, i64>(
        2e7, 1.0, 1000, 20_000_000, 20_000_004, 0.0, 0.0, 1, None, None, None
    )
    .is_err());
}

#[test]
fn a_cutoff_given_as_a_percentage_is_an_error() {
    // 5.0 meant as 5% is 5n observations: every sample read as flat, and the
    // run reported that the data must have been flat.
    use closure_core::ProminenceThresholds;
    let run = |primary: f64, band: Vec<f64>| {
        closure_parallel::<f64, i32>(
            3.0,
            1.13,
            100,
            1,
            5,
            0.005,
            0.005,
            1,
            None,
            None,
            Some(ProminenceThresholds { primary, band }),
        )
    };
    assert!(run(5.0, vec![2.0, 5.0, 10.0]).is_err());
    assert!(run(0.05, vec![0.02, 1.0]).is_err());
    // Fractions are fine, and are what the run reports at.
    let shapes = run(0.04, vec![0.03, 0.06]).unwrap().modality_shapes;
    assert_eq!(shapes.min_prominence, 0.04);
    let rungs: Vec<u32> = shapes
        .ladder
        .iter()
        .map(|r| r.min_prominence_counts)
        .collect();
    assert_eq!(rungs, vec![3, 4, 6]);
    assert_eq!(
        shapes.must_be(|c| c == closure_core::modality::ShapeClass::Flat),
        Some(false)
    );
}

#[test]
fn a_deep_search_does_not_overflow_the_stack() {
    // The search recursed once per value, so from about n = 20,000 it
    // overflowed the thread's stack and aborted the whole process, which
    // nothing can catch. Deep branches now run on an explicit stack.
    let n = 20_000;
    let results =
        closure_parallel::<f64, i32>(0.5, 0.5, n, 0, 1, 0.005, 0.005, 1, None, None, None).unwrap();
    assert_eq!(
        results.results.len() as u64,
        closure_count(0.5, 0.5, n, 0, 1, 0.005, 0.005).unwrap()
    );
    assert_eq!(results.results.len(), 201);
}

#[test]
fn a_huge_stop_after_means_no_limit() {
    // `limit * k` wrapped: 2^62 times 4 values is 0 in release builds, so the
    // result was truncated to nothing; debug builds panicked.
    let all = closure_parallel::<f64, i32>(2.5, 1.0, 10, 1, 4, 0.05, 0.05, 1, None, None, None)
        .unwrap()
        .results
        .len();
    assert!(all > 0);
    for limit in [1usize << 62, usize::MAX] {
        let results = closure_parallel::<f64, i32>(
            2.5,
            1.0,
            10,
            1,
            4,
            0.05,
            0.05,
            1,
            None,
            Some(limit),
            None,
        )
        .unwrap();
        assert_eq!(results.results.len(), all, "stop_after = {limit}");
        assert!(results.modality_shapes.exhaustive);
    }
}

#[test]
fn the_counter_rejects_what_the_search_rejects() {
    // `closure_count` skipped validation: sd = -1 with a 0.05 tolerance was
    // read as "SD between 0 and 0.95" and counted 10 samples. It now returns
    // the error `closure_parallel` does.
    for (mean, sd, re_mean, re_sd) in [
        (3.0, -1.0, 0.05, 0.05),
        (3.0, 1.0, -0.05, 0.05),
        (3.0, 1.0, 0.05, -0.05),
        (f64::NAN, 1.0, 0.05, 0.05),
        (3.0, f64::INFINITY, 0.05, 0.05),
    ] {
        assert!(closure_parallel::<f64, i32>(
            mean, sd, 10, 1, 5, re_mean, re_sd, 1, None, None, None
        )
        .is_err());
        assert!(closure_count(mean, sd, 10, 1, 5, re_mean, re_sd).is_err());
    }
}

#[test]
fn a_sample_is_found_from_its_own_exact_statistics() {
    // Feed the search the full-precision mean and SD of a known sample with
    // zero tolerance. The sample sits exactly on every bound, which is where
    // float thresholds used to lose it.
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..200 {
        let n = rng.random_range(3..=40);
        let (lo, hi) = if rng.random_bool(0.2) {
            (-3, 3)
        } else {
            (1, 7)
        };
        let mut sample: Vec<i32> = (0..n).map(|_| rng.random_range(lo..=hi)).collect();
        sample.sort_unstable();
        let sum: i64 = sample.iter().map(|&v| v as i64).sum();
        let ssq: i64 = sample.iter().map(|&v| (v as i64) * (v as i64)).sum();
        let n_f = n as f64;
        let mean = sum as f64 / n_f;
        let sd = ((n as i64 * ssq - sum * sum) as f64 / (n_f * (n_f - 1.0))).sqrt();

        let results =
            closure_parallel::<f64, i32>(mean, sd, n, lo, hi, 0.0, 0.0, 1, None, None, None)
                .unwrap();
        let found = (0..results.results.len()).any(|i| results.results.sample(i) == sample);
        assert!(found, "{sample:?} not found from mean={mean} sd={sd}");
        assert_eq!(
            closure_count(mean, sd, n, lo, hi, 0.0, 0.0).unwrap(),
            results.results.len() as u64
        );
    }
}

#[test]
fn sprite_takes_its_tolerances_literally() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // A tolerance of 0.02 used to be reinterpreted as "one decimal place",
    // i.e. widened to 0.05; 0.01 on the mean was narrowed to 0.005. Both are
    // now the inclusive half-widths CLOSURE uses.
    let results = sprite_parallel::<f64, i32>(
        2.205,
        1.3,
        20,
        1,
        5,
        0.01,
        0.02,
        1,
        None,
        RestrictionsOption::Default,
        None,
        Some(200),
        None,
    )
    .unwrap();
    assert!(!results.results.is_empty());
    for i in 0..results.results.len() {
        let sample: Vec<f64> = results
            .results
            .sample(i)
            .iter()
            .map(|&v| v as f64 / 100.0)
            .collect();
        let mean = sample.iter().sum::<f64>() / 20.0;
        let sd = (sample.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 19.0).sqrt();
        assert!(
            (mean - 2.205).abs() <= 0.01 + 1e-9,
            "sample {i}: mean {mean}"
        );
        assert!((sd - 1.3).abs() <= 0.02 + 1e-9, "sample {i}: sd {sd}");
    }
}

/// Mean and SD of a result row, computed on the real grid values.
fn row_mean_sd(row: &[u32], values: &[f64]) -> (f64, f64) {
    let n: f64 = row.iter().map(|&c| c as f64).sum();
    let mean = row
        .iter()
        .zip(values)
        .map(|(&c, &v)| c as f64 * v)
        .sum::<f64>()
        / n;
    let ss: f64 = row
        .iter()
        .zip(values)
        .map(|(&c, &v)| c as f64 * (v - mean).powi(2))
        .sum();
    (mean, (ss / (n - 1.0)).sqrt())
}

#[test]
fn multi_item_sprite_samples_match_on_the_real_grid() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // SPRITE's internal grid used to be 10^decimals wide, which cannot hold
    // 1/3 or 1/4 steps: 1.333 became 1.3, samples were accepted on the rounded
    // values, and then missed the reported mean and SD on the real ones. At a
    // tolerance of 0.5 the grid was whole numbers and half values never
    // appeared at all.
    for (items, re) in [(2, 0.5), (3, 0.05), (4, 0.05)] {
        let results = sprite_parallel::<f64, i32>(
            2.2,
            1.3,
            20,
            1,
            5,
            re,
            re,
            items,
            None,
            RestrictionsOption::Default,
            None,
            Some(200),
            None,
        )
        .unwrap();
        assert!(!results.results.is_empty(), "items={items}");
        let values = results.results.counts.grid().values().to_vec();
        let mut off_integer = 0;
        for row in results.results.counts.rows() {
            let (mean, sd) = row_mean_sd(row, &values);
            assert!(
                (mean - 2.2).abs() <= re + 1e-9,
                "items={items}: mean {mean}"
            );
            assert!((sd - 1.3).abs() <= re + 1e-9, "items={items}: sd {sd}");
            off_integer += row
                .iter()
                .zip(&values)
                .filter(|&(&c, &v)| c > 0 && v.fract() != 0.0)
                .count();
        }
        assert!(
            off_integer > 0,
            "items={items}: no fractional value ever used"
        );
    }

    // An exact restriction on 1 1/3 is keyed as 133 hundredths.
    let restricted = sprite_parallel::<f64, i32>(
        2.2,
        1.3,
        20,
        1,
        5,
        0.05,
        0.05,
        3,
        Some([(133, 2)].into_iter().collect()),
        RestrictionsOption::Default,
        None,
        Some(20),
        None,
    )
    .unwrap();
    assert!(!restricted.results.is_empty());
    for row in restricted.results.counts.rows() {
        assert_eq!(row[1], 2, "1 1/3 must appear exactly twice");
    }
}

#[test]
fn sprite_reaches_every_total_the_mean_tolerance_allows() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // Mean 3.0 ± 0.05 at n = 100 admits totals 295..=305. Every attempt used
    // to aim at exactly 300, and no step of the search changes the total, so
    // all 170-odd samples SPRITE found had the same sum.
    let results = sprite_parallel::<f64, i32>(
        3.0,
        1.0,
        100,
        1,
        5,
        0.05,
        0.05,
        1,
        None,
        RestrictionsOption::Default,
        None,
        Some(2000),
        None,
    )
    .unwrap();
    let values = results.results.counts.grid().values().to_vec();
    let mut totals = std::collections::BTreeSet::new();
    for row in results.results.counts.rows() {
        let (mean, sd) = row_mean_sd(row, &values);
        assert!((mean - 3.0).abs() <= 0.05 + 1e-9, "mean {mean}");
        assert!((sd - 1.0).abs() <= 0.05 + 1e-9, "sd {sd}");
        totals.insert((mean * 100.0).round() as i64);
    }
    assert_eq!(totals, (295..=305).collect());
}

#[test]
fn sprite_with_every_response_fixed_does_not_panic() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // The default restrictions fix one response at each end of the scale, so
    // at n = 2 nothing is left free. A mean other than 3 then sent the mean
    // adjustment into `random_range(0..0)`.
    let run = |mean: f64| {
        sprite_parallel::<f64, i32>(
            mean,
            2.83,
            2,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            RestrictionsOption::Default,
            None,
            Some(1),
            None,
        )
        .unwrap()
        .results
        .len()
    };
    assert_eq!(run(2.5), 0);
    assert_eq!(run(3.0), 1);
}

#[test]
fn streaming_writes_into_the_named_directory_and_creates_it() {
    use closure_core::{sprite_parallel_streaming, RestrictionsOption};

    // Streaming used to read `out` (not an existing directory) as a file-name
    // prefix, writing `out_counts.parquet` where memory mode wrote
    // `out/counts.parquet`, and never created a missing `out/`: that run
    // returned Ok with zero samples and wrote nothing.
    let root = std::env::temp_dir().join("closure_regression_stream_dirs");
    let _ = std::fs::remove_dir_all(&root);
    for (i, suffix) in ["", "/"].iter().enumerate() {
        let closure_dir = root.join(format!("closure{i}/nested"));
        let path = format!("{}{suffix}", closure_dir.display());
        let result = closure_parallel_streaming::<f64, i32>(
            3.0,
            1.0,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            StreamingConfig::new(path, 100, false),
            None,
            None,
        )
        .unwrap();
        assert!(result.total_combinations > 0);
        assert!(closure_dir.join("counts.parquet").exists(), "{suffix:?}");
        assert!(
            closure_dir.join("metrics_main.parquet").exists(),
            "{suffix:?}"
        );

        let sprite_dir = root.join(format!("sprite{i}/nested"));
        let path = format!("{}{suffix}", sprite_dir.display());
        sprite_parallel_streaming::<f64, i32>(
            2.2,
            1.3,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            RestrictionsOption::Default,
            StreamingConfig::new(path, 100, false),
            Some(5),
            None,
        )
        .unwrap();
        assert!(sprite_dir.join("counts.parquet").exists(), "{suffix:?}");
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_failed_write_is_an_error_not_an_empty_result() {
    use closure_core::{
        closure_parallel, sprite_parallel_streaming, ParameterError, ParquetConfig,
        RestrictionsOption,
    };

    // A directory squatting on `counts.parquet` makes that file impossible to
    // create. Streaming used to print the error and return Ok with zero
    // samples; memory mode ignored every write error and returned Ok.
    let dir = std::env::temp_dir().join("closure_regression_failed_write");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("counts.parquet")).unwrap();
    let path = dir.display().to_string();

    let streamed = closure_parallel_streaming::<f64, i32>(
        3.0,
        1.0,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        StreamingConfig::new(path.clone(), 100, false),
        None,
        None,
    );
    assert!(matches!(streamed, Err(ParameterError::Output(_))));

    let limited = closure_parallel_streaming::<f64, i32>(
        3.0,
        1.0,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        StreamingConfig::new(path.clone(), 100, false),
        Some(5),
        None,
    );
    assert!(matches!(limited, Err(ParameterError::Output(_))));

    let sprite = sprite_parallel_streaming::<f64, i32>(
        2.2,
        1.3,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        None,
        RestrictionsOption::Default,
        StreamingConfig::new(path.clone(), 100, false),
        Some(5),
        None,
    );
    assert!(matches!(sprite, Err(ParameterError::Output(_))));

    let in_memory = closure_parallel::<f64, i32>(
        3.0,
        1.0,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        Some(ParquetConfig::new(path, 100)),
        None,
        None,
    );
    assert!(matches!(in_memory, Err(ParameterError::Output(_))));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sprite_streaming_honours_stop_after() {
    use closure_core::{sprite_parallel_streaming, RestrictionsOption};

    // The limit was checked before each attempt in a parallel batch, so every
    // attempt already in flight still wrote its sample: `Some(3)` wrote 6 to
    // 10.
    let dir = std::env::temp_dir().join("closure_regression_sprite_stop_after");
    for _ in 0..5 {
        let _ = std::fs::remove_dir_all(&dir);
        let result = sprite_parallel_streaming::<f64, i32>(
            2.2,
            1.3,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            RestrictionsOption::Default,
            StreamingConfig::new(dir.display().to_string(), 100, false),
            Some(3),
            None,
        )
        .unwrap();
        assert_eq!(result.total_combinations, 3);
        let base = format!("{}/", dir.display());
        assert_eq!(
            f64_cell(&format!("{base}metrics_main.parquet"), "samples_all"),
            3.0
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sprite_without_a_limit_does_not_overflow() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // `stop_after = None` became `usize::MAX`, which was then multiplied by
    // 20: a panic in debug builds. A small space lets the search end on its
    // own duplicate-rate rule.
    let results = sprite_parallel::<f64, i32>(
        2.0,
        0.9,
        6,
        1,
        3,
        0.05,
        0.05,
        1,
        None,
        RestrictionsOption::Null,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(!results.results.is_empty());
}

#[test]
fn sprite_rejects_a_scale_its_integer_type_cannot_hold() {
    use closure_core::{sprite_parallel, ParameterError, RestrictionsOption};

    // At items = 1000, a 1..=50 scale runs to 50,000 internal units, past i16.
    // Values that did not fit used to be dropped from the grid silently.
    let run = |scale_max: i16, items: u32| {
        sprite_parallel::<f64, i16>(
            25.0,
            10.0,
            20,
            1,
            scale_max,
            0.05,
            0.05,
            items,
            None,
            RestrictionsOption::Default,
            None,
            Some(5),
            None,
        )
    };
    assert!(matches!(
        run(50, 1000),
        Err(ParameterError::InputValidation(_))
    ));
    // Samples are reported in hundredths, which at 1..=500 reach 50,000.
    // Reading one back used to panic instead.
    assert!(matches!(
        run(500, 1),
        Err(ParameterError::InputValidation(_))
    ));
    assert!(run(50, 3).is_ok());
}

#[test]
fn a_sprite_keeps_its_restrictions_across_runs() {
    use closure_core::{RestrictionsOption, Sprite, Technique};

    // `run` used to `take()` the exact restrictions and reset the minimum
    // ones to `Default`, so a second run on the same `Sprite` quietly ran
    // unrestricted.
    let mut sprite = Sprite {
        restrictions_exact: Some([(300, 4)].into_iter().collect()),
        restrictions_minimum: RestrictionsOption::Null,
    };
    for _ in 0..2 {
        let results = <Sprite as Technique<f64, i32>>::run(
            &mut sprite,
            2.2,
            1.3,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            Some(20),
            None,
        )
        .unwrap();
        assert!(!results.results.is_empty());
        for row in results.results.counts.rows() {
            assert_eq!(row[2], 4, "the value 3 must appear exactly four times");
        }
    }
}

#[test]
fn from_range_takes_scale_values() {
    use closure_core::{sprite_parallel, RestrictionsMinimum, RestrictionsOption};

    // `from_range(1, 5)` used to store keys 1 and 5, which as hundredths are
    // 0.01 and 0.05, and SPRITE rejected them as invalid.
    let results = sprite_parallel::<f64, i32>(
        2.2,
        1.3,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        None,
        RestrictionsOption::new(RestrictionsMinimum::from_range(1, 5)),
        None,
        Some(20),
        None,
    )
    .unwrap();
    assert!(!results.results.is_empty());
    for row in results.results.counts.rows() {
        assert!(row[0] >= 1 && row[4] >= 1, "both scale ends must appear");
    }
}

#[test]
fn sprite_fills_the_value_below_an_exactly_restricted_one() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // Fixing the value 2 leaves a gap from 1 to 3. Every shift moved its
    // first value up, so a 3 could only become a 1 alongside a 1 becoming a
    // 3, the swap back. The 1s the start phase never placed stayed out of
    // reach. CLOSURE and rsprite2 both find this sample.
    let results = sprite_parallel::<f64, i32>(
        3.25,
        1.83,
        8,
        1,
        6,
        0.005,
        0.005,
        1,
        Some([(200, 1)].into_iter().collect()),
        RestrictionsOption::Null,
        None,
        Some(50),
        None,
    )
    .unwrap();
    let samples: Vec<Vec<i32>> = (0..results.results.len())
        .map(|i| results.results.sample(i))
        .collect();
    assert_eq!(samples, vec![vec![100, 100, 200, 300, 400, 400, 500, 600]]);
}

#[test]
fn sprite_finds_a_sample_whose_every_value_is_fixed() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // With no value left free, the mean adjustment failed before checking
    // whether the fixed responses already hit the mean.
    let results = sprite_parallel::<f64, i32>(
        1.5,
        std::f64::consts::FRAC_1_SQRT_2,
        2,
        1,
        2,
        0.005,
        0.005,
        1,
        Some([(100, 1), (200, 1)].into_iter().collect()),
        RestrictionsOption::Null,
        None,
        Some(5),
        None,
    )
    .unwrap();
    assert_eq!(results.results.len(), 1);
    assert_eq!(results.results.sample(0), vec![100, 200]);
}

#[test]
fn sprite_gives_up_quickly_on_an_impossible_sd() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // No sample on 1..=5 with mean 3 has an SD near 3.5; the most is about
    // 2.05. Every attempt used to run its full loop limit first, about ten
    // seconds without `stop_after`.
    let start = std::time::Instant::now();
    let results = sprite_parallel::<f64, i32>(
        3.0,
        3.5,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        None,
        RestrictionsOption::Null,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(results.results.is_empty());
    assert!(start.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn degenerate_grids_are_input_errors() {
    use closure_core::{sprite_parallel, ParameterError, RestrictionsOption};

    // A one-value scale made every CLOSURE sample `Flat`, so an exhaustive
    // run "proved" the data was not unimodal. SPRITE already rejected it.
    assert!(matches!(
        closure_parallel::<f64, i32>(3.0, 0.0, 5, 3, 3, 0.05, 0.05, 1, None, None, None),
        Err(ParameterError::InputValidation(_))
    ));
    assert!(closure_count(3.0, 0.0, 5, 3, 3, 0.05, 0.05).is_err());

    // Past 100 items, neighbouring grid values share a key in hundredths:
    // duplicate column names, and samples that did not round-trip.
    assert!(matches!(
        sprite_parallel::<f64, i32>(
            1.5,
            0.3,
            10,
            1,
            2,
            0.05,
            0.05,
            200,
            None,
            RestrictionsOption::Null,
            None,
            Some(3),
            None,
        ),
        Err(ParameterError::InputValidation(_))
    ));
}

#[test]
fn sprite_gives_up_quickly_on_an_sd_too_narrow_for_the_mean() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // n 20 with mean 3.5 needs half the values above the other half, an SD of
    // at least about 0.51. Every attempt used to run its full loop limit
    // first, about 16 seconds without `stop_after`.
    let start = std::time::Instant::now();
    let results = sprite_parallel::<f64, i32>(
        3.5,
        0.1,
        20,
        1,
        5,
        0.005,
        0.005,
        1,
        None,
        RestrictionsOption::Null,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(results.results.is_empty());
    assert!(start.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn sprite_reaches_samples_beyond_a_restriction_gap() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // Each case has a single sample, and every attempt used to fail on it.
    // In the first two, reaching it from SPRITE's start takes one value moving
    // two grid steps to balance another's single step across the gap; in the
    // third, no shift reaches it from that start at all, so it takes a
    // different one.
    struct Case {
        mean: f64,
        sd: f64,
        n: i32,
        scale_max: i32,
        items: u32,
        exact: Vec<(i32, usize)>,
        expected: Vec<i32>,
    }
    let cases = [
        Case {
            mean: 4.17,
            sd: 1.6,
            n: 6,
            scale_max: 5,
            items: 1,
            exact: vec![(400, 1)],
            expected: vec![100, 400, 500, 500, 500, 500],
        },
        Case {
            mean: 1.5,
            sd: 1.0,
            n: 4,
            scale_max: 5,
            items: 2,
            exact: vec![(150, 0)],
            expected: vec![100, 100, 100, 300],
        },
        Case {
            mean: 2.0,
            sd: 0.61,
            n: 7,
            scale_max: 3,
            items: 3,
            exact: vec![(167, 3), (200, 1)],
            expected: vec![133, 167, 167, 167, 200, 267, 300],
        },
    ];
    for c in cases {
        let results = sprite_parallel::<f64, i32>(
            c.mean,
            c.sd,
            c.n,
            1,
            c.scale_max,
            0.005,
            0.005,
            c.items,
            Some(c.exact.into_iter().collect()),
            RestrictionsOption::Null,
            None,
            Some(10),
            None,
        )
        .unwrap();
        let samples: Vec<Vec<i32>> = (0..results.results.len())
            .map(|i| results.results.sample(i))
            .collect();
        assert_eq!(samples, vec![c.expected], "mean {}, sd {}", c.mean, c.sd);
    }
}

#[test]
fn sprite_keeps_drawing_long_enough_to_find_rare_samples() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // Two of these five samples come up about once in 200 draws. The
    // duplicate rate passed 99% after a few hundred, so one run in five
    // missed one of them.
    for stop_after in [None, Some(15)] {
        let results = sprite_parallel::<f64, i32>(
            3.0,
            1.5,
            9,
            1,
            5,
            0.005,
            0.005,
            1,
            None,
            RestrictionsOption::Null,
            None,
            stop_after,
            None,
        )
        .unwrap();
        assert_eq!(
            results.results.len() as u64,
            closure_count(3.0, 1.5, 9, 1, 5, 0.005, 0.005).unwrap(),
            "stop_after {stop_after:?}"
        );
    }
}

#[test]
fn the_counter_agrees_with_the_search_far_from_zero() {
    // Scale values near 2.1e7 are within the accepted range, but the counter
    // bounded Σx² in f64, where its ±1 slack vanished past 2^53, and pruned
    // the one valid state: 0 against the search's 1.
    let (n, low) = (141_i64, 21_000_000_i64);
    let mean = ((n - 1) * low + low + 1) as f64 / n as f64;
    let sd = (1.0 / n as f64).sqrt();
    let found = closure_parallel::<f64, i32>(
        mean, sd, 141, 21_000_000, 21_000_001, 0.0, 0.0, 1, None, None, None,
    )
    .unwrap()
    .results
    .len();
    assert_eq!(found, 1);
    assert_eq!(
        closure_count(mean, sd, 141, 21_000_000, 21_000_001, 0.0, 0.0).unwrap(),
        1
    );
}

#[test]
fn a_bound_hit_exactly_at_a_large_n_is_found() {
    // 187,110 ones in a million have mean 0.18711 and SD 0.39 exactly. The
    // SD bound `n·(n−1)·sd²` is about 1.5e11, where f64 error passed the
    // fixed 1e-6 slack and the bound rounded one unit past the sample.
    assert_eq!(
        closure_count(0.18711, 0.39, 1_000_000, 0, 1, 0.0, 0.0).unwrap(),
        1
    );
}

#[test]
fn sprite_rejects_a_scale_too_wide_for_the_grid() {
    use closure_core::{sprite_parallel, ParameterError, RestrictionsOption};

    // An i64 sample type holds 3e7 easily, but the output grid keys values
    // in hundredths as an i32. This used to panic building the grid.
    let result = sprite_parallel::<f64, i64>(
        15e6,
        1.0,
        10,
        0,
        30_000_000,
        0.05,
        0.05,
        1,
        None,
        RestrictionsOption::Null,
        None,
        Some(1),
        None,
    );
    assert!(matches!(result, Err(ParameterError::InputValidation(_))));

    // Near the limit at 100 items, the sum of squares would overflow an i64.
    // Release builds wrapped and returned no samples.
    let result = sprite_parallel::<f64, i32>(
        21_000_000.5,
        0.5,
        4,
        21_000_000,
        21_000_001,
        0.05,
        0.05,
        100,
        None,
        RestrictionsOption::Null,
        None,
        Some(5),
        None,
    );
    assert!(matches!(result, Err(ParameterError::InputValidation(_))));
}

#[test]
fn sprite_samples_far_from_zero_match_their_sd() {
    use closure_core::{sprite_parallel, RestrictionsOption};

    // With values near 1e7, `n·Σx² − (Σx)²` in f64 lost its low digits, and
    // about half the samples returned missed the SD tolerance.
    let results = sprite_parallel::<f64, i64>(
        1e7 + 2.5,
        1.5,
        200,
        10_000_000,
        10_000_005,
        0.005,
        0.005,
        1,
        None,
        RestrictionsOption::Null,
        None,
        Some(50),
        None,
    )
    .unwrap()
    .results;
    assert!(!results.is_empty());
    let grid = results.counts.grid().values();
    for row in results.counts.rows() {
        let values: Vec<f64> = row
            .iter()
            .enumerate()
            .flat_map(|(j, &c)| std::iter::repeat_n(grid[j] - 1e7, c as usize))
            .collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        assert!((mean - 2.5).abs() <= 0.005 + 1e-9, "mean {mean}");
        assert!((sd - 1.5).abs() <= 0.005 + 1e-9, "sd {sd}");
    }
}

#[test]
fn sprite_rejects_an_infinite_sd_or_tolerance() {
    use closure_core::{sprite_parallel, ParameterError, RestrictionsOption};

    // The check said "must be finite" but only tested `>= 0.0`, which
    // infinity passes.
    let inf = f64::INFINITY;
    for (sd, re_mean, re_sd) in [(inf, 0.05, 0.05), (1.3, inf, 0.05), (1.3, 0.05, inf)] {
        let result = sprite_parallel::<f64, i32>(
            2.2,
            sd,
            20,
            1,
            5,
            re_mean,
            re_sd,
            1,
            None,
            RestrictionsOption::Null,
            None,
            Some(5),
            None,
        );
        assert!(
            matches!(result, Err(ParameterError::InputValidation(_))),
            "sd {sd}, tolerances {re_mean} and {re_sd}"
        );
    }
}

#[test]
fn sprite_rejects_huge_restriction_counts_without_building_them() {
    use closure_core::{sprite_parallel, ParameterError, RestrictionsMinimum, RestrictionsOption};

    // The fixed responses were built before their number was checked
    // against n: a minimum count of `usize::MAX` panicked on capacity
    // overflow, and an exact count of 2^40 asked for about 4 TB.
    let run = |exact: Option<usize>, minimum: Option<usize>| {
        sprite_parallel::<f64, i32>(
            3.0,
            1.0,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            exact.map(|c| [(300, c)].into_iter().collect()),
            minimum.map_or(RestrictionsOption::Null, |c| {
                RestrictionsOption::new(RestrictionsMinimum::new([(300, c)].into_iter().collect()))
            }),
            None,
            Some(1),
            None,
        )
    };
    for result in [run(None, Some(usize::MAX)), run(Some(1 << 40), None)] {
        assert!(matches!(result, Err(ParameterError::InputValidation(_))));
    }
    // Counts whose sum overflows a usize.
    let both = sprite_parallel::<f64, i32>(
        3.0,
        1.0,
        20,
        1,
        5,
        0.05,
        0.05,
        1,
        Some([(300, usize::MAX)].into_iter().collect()),
        RestrictionsOption::new(RestrictionsMinimum::new([(100, 2)].into_iter().collect())),
        None,
        Some(1),
        None,
    );
    assert!(matches!(both, Err(ParameterError::InputValidation(_))));
}

#[test]
fn sprite_rejects_a_grid_too_wide_to_store() {
    use closure_core::{sprite_parallel, ParameterError, RestrictionsOption};

    // 0..=10,000,000 fits the output grid's keys, so it passed, and every
    // sample row held ten million counts, 40 MB, even at n = 2.
    // An SD no sample reaches, so an accepted grid returns at once, without
    // shape statistics over 100,000 grid values.
    let run = |scale_max: i32, items: u32| {
        sprite_parallel::<f64, i32>(
            scale_max as f64 / 2.0,
            1e9,
            2,
            0,
            scale_max,
            0.05,
            0.05,
            items,
            None,
            RestrictionsOption::Null,
            None,
            Some(1),
            None,
        )
    };
    match run(10_000_000, 1) {
        Err(ParameterError::InputValidation(msg)) => {
            assert!(msg.contains("at most 100000"), "{msg}")
        }
        other => panic!("expected an input error, got {:?}", other.map(|r| r.results.len())),
    }
    // The limit counts grid values, items included: 99,999 at one item fits,
    // 1,000 at 100 items does not.
    assert!(run(99_998, 1).is_ok());
    assert!(run(1_000, 100).is_err());
}

#[test]
fn a_limited_run_that_saw_everything_is_exhaustive() {
    // `stop_after` used to mark every limited run as partial, so `can_be`
    // answered `None` even when the limit was never reached.
    let total = closure_count(3.0, 1.2, 20, 1, 5, 0.05, 0.05).unwrap() as usize;
    assert!(total > 0);
    for limit in [total + 1, 1_000_000] {
        let results = closure_parallel::<f64, i32>(
            3.0,
            1.2,
            20,
            1,
            5,
            0.05,
            0.05,
            1,
            None,
            Some(limit),
            None,
        )
        .unwrap();
        assert_eq!(results.results.len(), total);
        assert!(results.modality_shapes.exhaustive, "stop_after = {limit}");
    }
    let results =
        closure_parallel::<f64, i32>(3.0, 1.2, 20, 1, 5, 0.05, 0.05, 1, None, Some(total), None)
            .unwrap();
    assert!(
        !results.modality_shapes.exhaustive,
        "reaching the limit proves nothing"
    );
}
