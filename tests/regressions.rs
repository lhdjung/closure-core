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

#[test]
fn a_sample_size_equal_to_the_seed_depth_is_still_checked() {
    // n = 2 is exactly the DFS seed depth, so the search never enters the loop
    // that checks the sum and upper SD bounds. The terminal check has to.
    let results = closure_parallel::<f64, i32>(3.0, 0.0, 2, 1, 5, 0.0, 0.0, 1, None, None).unwrap();
    let samples: Vec<Vec<i32>> = (0..results.results.len())
        .map(|i| results.results.sample(i))
        .collect();
    assert_eq!(samples, vec![vec![3, 3]]);
    assert_eq!(closure_count(3.0, 0.0, 2, 1, 5, 0.0, 0.0), 1);
}

#[test]
fn stop_after_is_an_upper_bound_in_every_path() {
    for limit in [5, 150] {
        let results =
            closure_parallel::<f64, i32>(3.0, 1.0, 40, 1, 5, 0.05, 0.05, 1, None, Some(limit))
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
        closure_parallel::<f64, i32>(3.24, 1.0, 25, 1, 5, 0.0, 0.05, 1, None, None).unwrap();
    assert!(!results.results.is_empty());
    assert_eq!(
        results.results.len() as u64,
        closure_count(3.24, 1.0, 25, 1, 5, 0.0, 0.05)
    );
}

#[test]
fn the_counter_agrees_with_the_search_on_negative_scales() {
    for (mean, sd, n, lo, hi) in [(0.0, 1.5, 10, -3, 3), (-1.0, 1.0, 8, -3, 1)] {
        let results =
            closure_parallel::<f64, i32>(mean, sd, n, lo, hi, 0.05, 0.05, 1, None, None).unwrap();
        assert_eq!(
            closure_count(mean, sd, n, lo, hi, 0.05, 0.05),
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
    )
    .unwrap();
    assert_eq!(result.total_combinations, 0);
    assert_eq!(
        f64_cell(&format!("{base}metrics_main.parquet"), "samples_all"),
        0.0
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_parameters_are_errors_not_panics() {
    assert!(closure_parallel::<f64, i32>(3.0, 1.0, 1, 1, 5, 0.0, 0.0, 1, None, None).is_err());
    assert!(closure_parallel::<f64, i32>(3.0, 1.0, 0, 1, 5, 0.0, 0.0, 1, None, None).is_err());
    assert!(closure_parallel::<f64, i32>(3.0, 1.0, 10, 5, 1, 0.0, 0.0, 1, None, None).is_err());
    assert!(closure_parallel::<f64, i32>(3.0, -1.0, 10, 1, 5, 0.0, 0.0, 1, None, None).is_err());
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
            closure_parallel::<f64, i32>(mean, sd, n, lo, hi, 0.0, 0.0, 1, None, None).unwrap();
        let found = (0..results.results.len()).any(|i| results.results.sample(i) == sample);
        assert!(found, "{sample:?} not found from mean={mean} sd={sd}");
        assert_eq!(
            closure_count(mean, sd, n, lo, hi, 0.0, 0.0),
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
    );
    assert!(matches!(in_memory, Err(ParameterError::Output(_))));

    let _ = std::fs::remove_dir_all(&dir);
}
