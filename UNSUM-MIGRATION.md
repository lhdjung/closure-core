# Adapting unsum to the current closure-core

This guide lists every change in closure-core that affects the unsum R package,
and says exactly what to change in unsum for each one.

- **From:** closure-core `1fe8f95`. unsum's `src/rust/Cargo.lock` pins this
  commit (branch `frequency-details`), and `src/rust/vendor/closure-core` is
  byte-identical to it.
- **To:** branch `optimize` as of `9448030`, 31 commits later.
- **unsum state:** line numbers refer to unsum's working tree on 2026-09-25,
  including its uncommitted changes.

Everything marked *verified* was run. A scratch copy of unsum was built with
`R CMD INSTALL` against the new closure-core, using the binding patch in
[Appendix A](#appendix-a-full-binding-patch), and checked from R.

---

## Contents

1. [The short version](#1-the-short-version)
2. [Step 1: point unsum at the new closure-core](#2-step-1-point-unsum-at-the-new-closure-core)
3. [Step 2: the Rust binding](#3-step-2-the-rust-binding-srcrustsrclibrs)
4. [Step 3: the R code](#4-step-3-the-r-code)
5. [Results that change numerically](#5-results-that-change-numerically)
6. [New errors](#6-new-errors)
7. [Optional: switch to the counts layout](#7-optional-switch-to-the-counts-layout)
8. [File reference](#8-file-reference)
9. [Commit index](#9-commit-index)
- [Appendix A: full binding patch](#appendix-a-full-binding-patch)

---

## 1. The short version

Work through the rows in order. **Blocks** says what happens if the row is
skipped.

| # | What changed in closure-core | Blocks | What to change in unsum | Where |
|---|---|---|---|---|
| 1 | The vendored copy is 31 commits behind | everything | Re-pin and re-vendor | `src/rust/Cargo.toml`, `vendor.tar.xz` |
| 2 | `ParquetConfig` / `StreamingConfig` gained a `format` field | compile | Set `format: OutputFormat::Samples` | `lib.rs` config wrappers |
| 3 | `ModalityConclusion` was removed; it is now `ModalityShapes` | compile | Build `modality_conclusion` from `rl.modality_shapes` | `lib.rs` `modality_conclusion_to_robj` |
| 4 | `FrequencyTable::f_count()` was removed | compile | Use `f_expected()` and `f_representative()` | `lib.rs` `frequency_table_to_robj` |
| 5 | `ResultsTable.sample` field was removed | compile | Use `results.sample(i)` | `lib.rs` `results_table_to_robj` |
| 6 | Error messages come through `Display` | readable errors | `format!("{}", e)`, not `{:?}` | `lib.rs`, 3 places |
| 7 | `closure_count` returns `Result<u64, ParameterError>` | nothing at compile time; see 3.6 | Match on the result | `lib.rs` `count_closure_combinations`, `R/count.R` |
| 8 | `frequency` columns: `f_count` → `f_expected` + `f_representative`; `value` is now double | every generator call | Update the schema check and every `f_count` use | `R/utils.R`, plots, predicates, docs |
| 9 | `value` columns of `frequency_dist`, `modality_counts`, `modality_pairs` are now double | every generator call | Update the schema check | `R/utils.R` |
| 10 | `modality_conclusion` can be `NA`, and its columns mean different things | results, docs | Handle `NA` and update the docs | `R/s7-result.R`, docs |
| 11 | Streaming writes 5 more files (`modality_*.parquet`) | `path = ` runs | Extend `FILES_EXPECTED`, read the new files | `R/constants.R`, `R/read-write-basic.R` |
| 12 | Multi-item SPRITE tables are one row per grid value, not per integer | SPRITE `items > 1` | Use the grid length, not `scale_max - scale_min + 1` | `R/utils.R` |
| 13 | Empty streaming runs write every file | empty results | Retire the empty-folder special case | `R/read-write-basic.R` |
| 14 | Numbers differ: CLOSURE boundaries, SPRITE coverage, and more | tests, NEWS | Update snapshots and NEWS | `tests/`, `NEWS.md` |

Rows 1–7 are all in the binding patch in [Appendix A](#appendix-a-full-binding-patch),
which compiles and is verified end to end. Rows 8–13 are R changes, set out
in [section 4](#4-step-3-the-r-code).

---

## 2. Step 1: point unsum at the new closure-core

`src/rust/Cargo.toml`:

```toml
# before
closure-core = { git = "https://github.com/lhdjung/closure-core.git", branch = "frequency-details" }
# after (or `branch = "master"` once `optimize` is merged)
closure-core = { git = "https://github.com/lhdjung/closure-core.git", branch = "optimize" }
```

Then run `cargo update -p closure-core` in `src/rust`, and regenerate
`src/rust/vendor.tar.xz` and `vendor-config.toml` (e.g.
`rextendr::vendor_pkgs()`). The offline build in `src/Makevars.in` unpacks
that tarball, so the old copy would otherwise keep being compiled.

The crates closure-core depends on (arrow/parquet 55, rand 0.9, rayon,
strum 0.26, thiserror 2) were already in the vendored set. The new version
adds no dependencies.

---

## 3. Step 2: the Rust binding (`src/rust/src/lib.rs`)

Before the patch, compiling the binding against the new closure-core fails with
exactly six errors:

```
error[E0063]: missing field `format` in initializer of `ParquetConfig`
error[E0063]: missing field `format` in initializer of `StreamingConfig`
error[E0432]: unresolved import `closure_core::ModalityConclusion`
error[E0599]: no method named `f_count` found for reference `&FrequencyTable`
error[E0609]: no field `modality_conclusion` on type `&ResultListFromMeanSdN<i32>`
error[E0615]: attempted to take value of method `sample` on type `&ResultsTable<i32>`
```

Sections 3.1–3.4 fix them. Sections 3.5 and 3.6 cover changes that compile
unchanged but behave differently. The complete patch is in
[Appendix A](#appendix-a-full-binding-patch). It compiles without warnings and
is what the verification ran on.

### 3.1 Config structs: choose an output layout

Both config structs gained `format: OutputFormat`.

- `Counts` (the default of `::new`) writes a new layout: `counts.parquet`, one
  column per scale value.
- `Samples` writes the old layout: `sample.parquet` with columns
  `pos1..posN`, plus `horns.parquet`.
- `Both` writes both.

To keep unsum's reader working with the fewest changes, use `Samples`:

```rust
Ok(StreamingConfigR(StreamingConfig {
    file_path,
    batch_size,
    show_progress,
    format: OutputFormat::Samples,
}))
```

Do the same in `ParquetConfigR`. unsum never uses that one, because
`create_combinations()` always passes `None` for memory mode, so deleting it
works just as well.

[Section 7](#7-optional-switch-to-the-counts-layout) explains how to move to
the counts layout later.

### 3.2 `modality_conclusion`: now three-valued, with new meanings

`ResultListFromMeanSdN.modality_conclusion` (four `bool`s) is gone. Its
replacement is `modality_shapes: closure_core::modality::ModalityShapes`, which
classifies every sample into one of six shape classes and answers
`Option<bool>`:

- `Some(true)`: some sample has that shape. This holds whether or not the
  search was exhaustive.
- `Some(false)`: no admissible sample has that shape at any mode threshold in
  the prominence band. This is only returned when the search was
  **exhaustive**: CLOSURE without `stop_after`.
- `None`: none was found, but the search was partial. That covers SPRITE
  always, and CLOSURE with `stop_after`. Absence is not proof here.

The patch keeps the four column names so that R-side code stays stable, and
maps `None` to `NA`:

| Column | Was (at `1fe8f95`) | Is now | Method |
|---|---|---|---|
| `can_be_unimodal` | some sample is weakly unimodal. That included flat and monotone samples. | some sample has exactly one mode, anywhere. Flat samples no longer count. | `can_be_unimodal()` |
| `can_be_bimodal` | some sample has two or more qualifying peaks, with the mean between them | some sample has two or more modes | `can_be_multimodal()` |
| `j_shape_low` | heuristic from the per-value count box: count at value 2 can exceed count at value 1 | some sample has its single mode at the lowest value | `can_be_j_shape_low()` |
| `j_shape_high` | mirror of `j_shape_low` | mirror | `can_be_j_shape_high()` |

In the verification, SPRITE with `stop_after = 30` gave
`TRUE TRUE TRUE NA`: `j_shape_high` was not found and not ruled out.

`ModalityShapes` also answers `can_be_bell_shaped()` (one mode away from both
ends). That is the claim papers usually make, and a candidate fifth column. For
exactly two modes, use `can_be(|c| c == ShapeClass::TwoModes)`.

The patch also exports three new tibbles, each identical to the file of the
same name that streaming writes (see [section 8](#8-file-reference)):
`modality_shapes`, `modality_summary` and `modality_prominence`. Exporting them
from memory mode is what lets `write_basic()` produce the same files as
streaming.

### 3.3 `frequency`: `f_count` is replaced by two columns

| Column | Was | Is now |
|---|---|---|
| `value` | integer | **double**. Multi-item SPRITE grids have fractional values; for CLOSURE they are still whole numbers. |
| `f_count` | memory mode: counts in the sample nearest the group's mean profile; streaming: the group's mean counts. The two modes disagreed. | **removed** |
| `f_expected` | – | mean count per value across the group, identical in memory and streaming |
| `f_representative` | – | counts in the group's medoid: the actual sample with the smallest summed distance (EMD) to all others. **`NaN` in streaming mode**, which cannot find a medoid in one pass. |
| `f_relative` | memory: `f_count / n` of the nearest sample; streaming: mean / `n` | `f_expected / n` in both modes |

The patch writes the columns in this order: `samples, value, f_expected,
f_representative, f_relative`.

### 3.4 `results`: samples are stored as counts

`ResultsTable` no longer has a `sample: Vec<Vec<U>>` field. It holds
`counts: SampleCounts`, one count per scale value per sample. That is lossless,
since every sample is sorted. `results.sample(i)` rebuilds one sample and
`results.samples()` rebuilds all of them. The patch expands one sample at a
time, so the R-side `results` tibble (`id`, `sample` list-column, `horns`)
looks exactly as before:

```rust
let samples_robjs: Vec<Robj> = (0..results_table.len())
    .map(|i| results_table.sample(i).into_robj())
    .collect();
```

SPRITE samples are still in **hundredths** (`150` is 1.5), as before. CLOSURE
samples are whole scale values.

### 3.5 Error messages

`ParameterError` is still `InputValidation | Consistency | Conflict`, plus a
new `Output(String)` for failed writes. The binding formatted errors with
`{:?}`, which prints the variant name (`InputValidation("…")`). With `{}`, R
gets just the message. *Verified:*

```
"CLOSURE error: sd and the rounding errors must not be negative"
"Streaming error: failed to write results: failed to create the sample file under '…/bad/': Is a directory (os error 21)"
```

### 3.6 `closure_count` returns a `Result`

`closure_count(...)` now returns `Result<u64, ParameterError>`. It gives the
same error `closure_parallel` gives for the same invalid input (negative SD or
tolerance, non-finite values, `n < 2`, `scale_min > scale_max`, scale out of
range). It used to return `0` for those, and before that it counted them.

**This does not break the build.** `Robj::from(count)` still compiles, because
extendr converts any `Result` into an `Robj`. With extendr's default
`result_panic` feature, an `Err` then becomes a Rust panic, which reaches R as
an unhelpful error. Match on it instead, as the patch does:

```rust
match closure_count(mean, sd, n, scale_min, scale_max, rounding_error_mean, rounding_error_sd) {
    Ok(count) => Robj::from(count),
    Err(e) => Robj::from(format!("CLOSURE error: {}", e)),
}
```

*Verified* from R with the patch:

```
count_closure_combinations(3.0,  1.0, 20L, 1L, 5L, 0.05, 0.05)  # 48
count_closure_combinations(3.0, -1.0, 20L, 1L, 5L, 0.05, 0.05)  # "CLOSURE error: sd and the rounding errors must not be negative"
count_closure_combinations(3.0,  1.0,  1L, 1L, 5L, 0.05, 0.05)  # "CLOSURE error: n must be at least 2"
```

On the R side, `closure_count_all()` (`R/count.R`) should then treat a
character result the way `generate-basic.R` does: abort with the message.
Normally unsum's own argument checks catch invalid input before it gets that
far.

---

## 4. Step 3: the R code

With the patched binding and the current R code, `closure_generate()` fails
immediately, in memory and on disk. *Verified:*

```
CLOSURE data must not be changed before passing them to other `closure_*()` functions.
! Specifically, `frequency` must be a tibble with:
• 15 rows and 4 columns
• These column names and types: "samples" (character), "value" (integer),
  "f_count" (double), and "f_relative" (double)
```

The following changes clear that and the next few failures.

### 4.1 `R/constants.R`

```r
TIBBLE_NAMES <- c(
  "inputs", "metrics_main", "metrics_horns",
  "modality_counts", "modality_pairs", "modality_conclusion",
  "modality_shapes", "modality_summary", "modality_prominence",   # new
  "frequency", "frequency_dist", "results"
)
```

Position these to match the order of the S7 properties. The binding returns
them after `modality_conclusion`.

Streaming now also writes the modality tables, so the "not persisted" comments
no longer hold. `modality_conclusion` itself is still derived and never
written; section 4.3 shows how to rebuild it from `modality_summary`.

`FILES_EXPECTED` for `OutputFormat::Samples`. This is the exact list closure-core
writes, *verified* for CLOSURE and SPRITE, for found and for empty results, with
unsum's own two files added:

```r
FILES_EXPECTED <- c(
  "info.md",                      # written by unsum
  "inputs.parquet",               # written by unsum
  "metrics_main.parquet",
  "metrics_horns.parquet",
  "frequency.parquet",
  "frequency_dist.parquet",
  "modality_counts.parquet",      # new
  "modality_pairs.parquet",       # new
  "modality_shapes.parquet",      # new
  "modality_summary.parquet",     # new
  "modality_prominence.parquet",  # new
  "horns.parquet",
  "sample.parquet"
)
```

`read_basic()` rejects any folder whose files are not exactly this set, so
this list has to match the output format precisely.
[Section 7](#7-optional-switch-to-the-counts-layout) gives the list for
`Counts`.

### 4.2 `check_generator_output()` in `R/utils.R`

| Line | Now | Change to |
|---|---|---|
| 217 | `scale_length <- scale_max - scale_min + 1` | Grid length: `(scale_max - scale_min) * items + 1`. For multi-item SPRITE the tables have one row per grid value. *Verified:* 1–7 at `items = 5` gives `frequency` 93 rows (3 × 31) and `modality_counts` 31. For CLOSURE, `items` is 1. |
| 220–229 | empty check: `scale_length == nrow(frequency) && all(is.nan(f_count))` | `all(is.nan(data$frequency$f_expected))`. Empty results have 3 × k rows (15 for 1–5, *verified*), so the row test never matched. That was already true at `1fe8f95`, so this check was dead before this update. |
| 231–242 | `frequency`: 4 columns, `value` integer, `f_count` | 5 columns: `samples` character, `value` **double**, `f_expected`, `f_representative`, `f_relative` double |
| 244–256 | `frequency_dist$value` integer | **double** (`count`, `n_samples` stay integer) |
| 258–297 | modality tibbles, "only present for in-memory results" | `modality_counts$value` and `modality_pairs$value_a`/`value_b` are **double**, and their row counts use the grid length too. `modality_conclusion` is still logical but may be `NA`. The block now applies to data read from disk as well. Add checks for the three new tibbles (schemas in [section 8](#8-file-reference)). |

### 4.3 `R/read-write-basic.R`

**Reading (`read_basic()`)**

- **Read the new files** alongside the other small ones:
  `modality_counts`, `modality_pairs`, `modality_shapes`, `modality_summary`,
  `modality_prominence`.
- **Rebuild `modality_conclusion` from `modality_summary`** so that data read
  from disk has the same columns as data in memory. This mirrors
  `ModalityShapes::can_be` exactly. *Verified:* identical to the in-memory
  columns for five CLOSURE cases (including all-`NA` empty results) and a
  `stop_after` run.

  ```r
  conclusion_from_summary <- function(s) {
    can_be <- function(classes) {
      if (sum(unlist(s[paste0("band_n_", classes)])) > 0) TRUE
      else if (isTRUE(s$exhaustive)) FALSE
      else NA
    }
    tibble::tibble(
      can_be_unimodal = can_be(c("one_mode_interior", "one_mode_low_edge", "one_mode_high_edge")),
      can_be_bimodal  = can_be(c("two_modes", "three_or_more_modes")),
      j_shape_low     = can_be("one_mode_low_edge"),
      j_shape_high    = can_be("one_mode_high_edge")
    )
  }
  ```

  Use the `band_n_*` columns, not `n_*`. The `n_*` counts are taken at one
  mode threshold, 5% of `n`. `band_n_*` counts a sample for every class it takes
  at any threshold between 2% and 10% of `n`, and that is what the proof
  semantics rest on.
- **The empty-results escape hatch (lines 157–163 and 239–260) is obsolete.**
  An empty streaming run now writes every file. `sample.parquet` and
  `horns.parquet` have zero rows, `metrics_main$samples_all` is 0, and
  `frequency` is all `NaN` (*verified*). The folder therefore counts as
  complete, and the normal read path must cope with zero-row files. The
  `create_empty_results()` call there can go.

**Writing (`write_basic()`)**

It writes `intersect(names(data), names_on_disk)`, where `names_on_disk` comes
from `FILES_EXPECTED`. Once memory mode returns `modality_shapes`,
`modality_summary` and `modality_prominence` (from the patch) and
`FILES_EXPECTED` lists their files, it writes them automatically, and folders
written from R match folders streamed from Rust.

### 4.4 Every use of `f_count`

| File | Lines | Suggested replacement |
|---|---|---|
| `R/plot-bar-basic.R` | 64, 249, 269, 273, 284, 287 | `f_expected` |
| `R/plot-ecdf-basic.R` | 207, 218, 314, 324, 327, 346, 355 | `f_expected` |
| `R/predicates-basic.R` | 22 (`metric = c("f_count", "f_relative")`) | `c("f_expected", "f_representative", "f_relative")` |
| `R/doc-helpers.R` | 29, 68 | describe both columns (see 3.3) |

Default to `f_expected`. It is defined in both modes. `f_representative` is
`NaN` for anything written by streaming, and `closure_generate(path = …)`
always streams. Offer `f_representative` where a real sample is wanted (e.g.
a bar plot of the medoid), and fall back or warn when it is `NaN`.

### 4.5 `modality_conclusion` consumers

- `R/s7-result.R:214` says "flagging modality and J-shape status". Say that
  `NA` means "not found, but the search was partial (SPRITE, or `stop_after`)".
- Any `if (x$modality_conclusion$can_be_unimodal)` needs `isTRUE()` or explicit
  `NA` handling.
- The column meanings in the table in 3.2 changed. Update the user
  documentation, and consider renaming `can_be_bimodal` to
  `can_be_multimodal` while breaking anyway.

### 4.6 `count_closure_combinations()` / `R/count.R`

- It now agrees exactly with the number of samples `closure_generate()` finds,
  including samples sitting exactly on a bound and negative scales.
- Input that `closure_generate()` would reject (negative SD or tolerance,
  non-finite values, `n < 2`, `scale_min > scale_max`) is now an error; see
  3.6 for handling it. unsum validates these in R first, so it should not
  happen in practice.
- Counts above about 1.8e19 saturate at `u64::MAX` instead of wrapping to a
  small number. R receives a double (*verified*: `numeric double 48`), so
  `1.844674e+19` means "at least this many".

---

## 5. Results that change numerically

None of these changes need code in unsum. Some will break snapshot tests, and
all of them belong in `NEWS.md`.

**CLOSURE**
- Samples lying exactly on a mean or SD bound were sometimes dropped because
  of float error (e.g. mean 3.24, n = 25). They are now found, so counts can
  **rise**. (`784373a`)
- With `n = 2`, every pair was returned regardless of the mean. (`784373a`)
- In-memory `stop_after` above 100 could return more samples than asked
  (426 for 150); it is now an exact upper bound. (`784373a`)
- Horns values are unchanged.

**SPRITE**
- Rounding errors are taken literally as inclusive half-widths. They used to
  be turned into a number of decimal places: 0.02 became 0.05 and 0.01 became
  0.005. A tolerance like 0.005 already mapped to two decimal places, so for
  unsum's usual settings the accepted interval is essentially the same; other
  rounding settings can change. (`784373a`)
- **Multi-item scales are exact.** 1.333 used to be stored as 1.3. Some
  samples missed the reported mean/SD on the real values: 16 of 200 at
  `items = 3`, 83 of 200 at `items = 4`. `items = 2` never produced half
  values at some tolerances. (`6cfe417`)
- **Frequency tables and horns for `items > 1` are on the real grid.** They
  used to be computed after rounding every value to a whole number. `frequency`
  and `frequency_dist` now have one row per grid value, e.g. 1, 1.5, 2, …, 5 at
  `items = 2` (*verified*). Horns, and `metrics_horns$uniform`, which refers to
  a uniform distribution over the grid, are different numbers. (`0d41544`)
- **Many more distinct samples.** Every attempt aimed at the single total
  `round(mean * n)`, so SPRITE never produced any other admissible total.
  For mean 3.0, SD 1.0, n = 100 it found ~170 samples, all with total 300. It
  now finds ~1,450, about 130 for each total from 295 to 305. The horns range
  and the extreme-horns rows widen accordingly. (`5883909`)
- Streaming now stops exactly at `stop_after`; `Some(3)` used to write 5–10.
  (`745928f`)

**Modality**
- `modality_conclusion` is derived per sample now (see 3.2). The old
  `j_shape_*` flags came from a heuristic on per-value count ranges.
  (`e51484c` and following)

---

## 6. New errors

All of these arrive in R as the character string the binding returns. unsum
already turns that into `abort_in_export()` and deletes the new folder. The
input errors also come from `count_closure_combinations()` once it is patched
as in 3.6.

| Condition | Error | Before |
|---|---|---|
| Output folder cannot be created, or any Parquet write fails | `Output`: "failed to write results: …" | Streaming printed the error and returned `total_combinations = 0`. unsum then warned "Summary statistics are internally inconsistent", which was false. |
| `n < 2`, `scale_min > scale_max`, negative or non-finite SD / tolerance | `InputValidation` | panics or wrong results |
| CLOSURE: scale values beyond ±21,474,836, or `(n · max|x|)²` beyond `i64` | `InputValidation` | panic after the search |
| SPRITE: `items = 0` | `InputValidation` | not rejected |
| SPRITE: `scale_max × 100` or `scale_max × items` outside `i32` | `InputValidation` | values silently dropped, or a panic when reading samples |

---

## 7. Optional: switch to the counts layout

`OutputFormat::Counts` is closure-core's default. It stores one row per sample
and one **count column per scale value** instead of one column per position.
There is no `id` column: the row number is the id. For a 235k-sample run on
1–7 with n = 60 that is about 6.6 MB in memory instead of ~62 MB. On disk it is
about 1.3× smaller.

With `format: OutputFormat::Counts`, the folder holds (*verified*):

```r
FILES_EXPECTED <- c(
  "info.md", "inputs.parquet",
  "counts.parquet", "scale_values.parquet", "format.parquet",
  "metrics_main.parquet", "metrics_horns.parquet",
  "frequency.parquet", "frequency_dist.parquet",
  "modality_counts.parquet", "modality_pairs.parquet",
  "modality_shapes.parquet", "modality_summary.parquet",
  "modality_prominence.parquet"
)
```

There is no `sample.parquet` or `horns.parquet`; horns is the last column of
`counts.parquet`. To rebuild the old `results$sample` list-column in R
(*verified*: identical to `sample.parquet` and `horns.parquet` from the same
run, for CLOSURE and SPRITE):

```r
counts <- nanoparquet::read_parquet(file.path(dir, "counts.parquet"))
key    <- nanoparquet::read_parquet(file.path(dir, "scale_values.parquet"))
m      <- as.matrix(counts[key$column])        # one count column per value
# CLOSURE: whole values; SPRITE: hundredths, as the old layout had them
unit   <- if (technique == "CLOSURE") key$value else key$value_hundredths
sample <- lapply(seq_len(nrow(m)), \(i) rep(unit, m[i, ]))
horns  <- counts$horns
```

The count columns are unsigned integers: `UInt8` for `n < 255`, `UInt16` for
`n < 65535`, otherwise `UInt32`. Check how your reader maps `UInt32`. Column
names are readable (`v1`, `v1_5`, `vn2` for −2), but `scale_values.parquet` is
the authoritative mapping. `format.parquet` has one row: `format = "counts"`,
`version = 2`, `technique`, `n`, `k`, `items`, `scale_min`, `scale_max`,
`sample_scale_factor`.

Reading counts directly, e.g. for bar plots, avoids expanding the samples at
all. Memory mode could also hand `results.counts.as_flat()` to R as an integer
matrix instead of a list of vectors.

---

## 8. File reference

Types are as nanoparquet reads them (*verified*). "Both" means the column holds
the same values in memory mode and in streaming.

| File | Columns |
|---|---|
| `metrics_main` | `samples_all` dbl, `values_all` dbl. Unchanged. |
| `metrics_horns` | `mean`, `uniform`, `sd`, `cv`, `mad`, `min`, `median`, `max`, `range`, all dbl. Unchanged. |
| `frequency` | `samples` chr (`all` / `horns_min` / `horns_max`, each × k), `value` **dbl**, `f_expected` dbl (both), `f_representative` dbl (memory only, `NaN` when streamed), `f_relative` dbl (both) |
| `frequency_dist` | `value` **dbl**, `count` int, `n_samples` int |
| `modality_counts` | `value` dbl, `count_lo` int, `count_hi` int: per-value count range over all samples |
| `modality_pairs` | `value_a` dbl, `value_b` dbl, `resolved` lgl, `a_greater` lgl: whether each adjacent ordering is fixed |
| `modality_shapes` | `class` chr, `n_samples` dbl, `value` dbl, `count_lo` int, `count_hi` int: per-value count range **within each shape class**. Classes with no samples have no rows. |
| `modality_summary` | one row: `exhaustive` lgl, `n_scanned` dbl, `min_prominence` dbl, `deficit_min` int, `deficit_mean` dbl, `deficit_max` int, `n_<class>` dbl ×6 (at the primary threshold), `band_n_<class>` dbl ×6 (anywhere in the band) |
| `modality_prominence` | `min_prominence` dbl, `min_prominence_counts` int, `primary` lgl, `class` chr, `n_samples` dbl: class counts at each rung of the threshold band (0.02, 0.05, 0.10 of n) |
| `sample` (Samples layout) | `pos1..posN` int. Unchanged; SPRITE values in hundredths. |
| `horns` (Samples layout) | `horns` dbl. Unchanged. |
| `counts`, `scale_values`, `format` (Counts layout) | see [section 7](#7-optional-switch-to-the-counts-layout) |

The six classes are `flat`, `one_mode_interior`, `one_mode_low_edge`,
`one_mode_high_edge`, `two_modes` and `three_or_more_modes`. Every sample falls
into exactly one of them at a given threshold.

Folder paths: `file_path` is always treated as a directory and created if
missing. unsum passes a folder it just created, so nothing changes. Before,
streaming to a path that was not an existing directory wrote
`<path>_counts.parquet`-style files next to it instead. (`af4236e`)

---

## 9. Commit index

Commits since `1fe8f95` that affect unsum; the rest are tests, docs or tooling.

| Commit | Effect on unsum |
|---|---|
| `0d41544` | Results stored as counts; `OutputFormat`; `value` columns double; multi-item SPRITE tables on the real grid |
| `e51484c`–`d6ac82e` | `ModalityShapes` replaces `ModalityConclusion`; `f_expected` / `f_representative` replace `f_count`; `modality_*.parquet` files |
| `784373a` | CLOSURE boundary samples, `n = 2`, exact `stop_after`; SPRITE literal tolerances; input errors instead of panics; empty runs write statistics |
| `6d62dd3` | `can_be` checks the whole threshold band; `band_n_*` columns in `modality_summary` |
| `6cfe417` | Multi-item SPRITE exact |
| `5883909` | SPRITE reaches every admissible total |
| `6fcadce` | SPRITE `n = 2` with default restrictions no longer panics |
| `af4236e`, `57b36b1` | Output folder semantics; write failures are errors (`ParameterError::Output`) |
| `b6e7a47`, `9b939b5` | `closure_count` validation and saturation |
| "Return an error from closure_count on invalid input" | `closure_count` returns `Result<u64, ParameterError>` (3.6) |
| `745928f` | SPRITE streaming honours `stop_after` |
| `f799da3`, `31c452a`, `0c1c877` | Overflow and range checks turned into errors |
| `0c74d37`, `be5a3f7` | Restriction docs. unsum passes `NULL`, i.e. `RestrictionsOption::Null`, so neither the default "both scale ends" rule nor the hundredths keys affect it. If `restrict_exact` / `restrict_min` are ever exposed, their names must be hundredths (`"300"` for the value 3), which is what `parse_restrict_*` in the binding already passes through. |
| "Take scale values in RestrictionsMinimum::from_range" | `from_range(1, 5)` now means the scale values 1 and 5; it used to need `from_range(100, 500)`. unsum does not call it. |

---

## Appendix A: full binding patch

This is a patch against unsum's `src/rust/src/lib.rs`. It compiles against
closure-core `9448030` without warnings. The verification in this document was
run on a package built with exactly this file.

```diff
--- a/src/rust/src/lib.rs
+++ b/src/rust/src/lib.rs
@@ -1,6 +1,7 @@
+use closure_core::modality::{ModalityShapes, ShapeClass};
 use closure_core::{
     closure_count, closure_parallel, closure_parallel_streaming, sprite_parallel,
-    sprite_parallel_streaming, FrequencyDist, ModalityConclusion, ModalityCounts, ModalityPairs,
+    sprite_parallel_streaming, FrequencyDist, ModalityCounts, ModalityPairs, OutputFormat,
     ParquetConfig, RestrictionsMinimum, RestrictionsOption, ResultListFromMeanSdN, StreamingConfig,
 };
 /// This is part of unsum, an R package that uses extendr for Rust integration
@@ -34,6 +35,7 @@
         Ok(ParquetConfigR(ParquetConfig {
             file_path,
             batch_size,
+            format: OutputFormat::Samples,
         }))
     }
 }
@@ -69,6 +71,7 @@
             file_path,
             batch_size,
             show_progress,
+            format: OutputFormat::Samples,
         }))
     }
 }
@@ -105,25 +108,99 @@
     .into()
 }
 
-/// (can_be_unimodal, can_be_bimodal, j_shape_low, j_shape_high) — exactly one row
-fn modality_conclusion_to_robj(mc: &ModalityConclusion) -> Robj {
+/// `None` ("not ruled out, but the search was partial") becomes `NA`.
+fn to_rbool(x: Option<bool>) -> Rbool {
+    x.map_or(Rbool::na(), Rbool::from)
+}
+
+/// Same four columns as before, now three-valued; see the migration guide.
+fn modality_conclusion_to_robj(ms: &ModalityShapes) -> Robj {
     data_frame!(
-        can_be_unimodal = vec![mc.can_be_unimodal],
-        can_be_bimodal  = vec![mc.can_be_bimodal],
-        j_shape_low     = vec![mc.j_shape_low],
-        j_shape_high    = vec![mc.j_shape_high]
+        can_be_unimodal = vec![to_rbool(ms.can_be_unimodal())],
+        can_be_bimodal  = vec![to_rbool(ms.can_be_multimodal())],
+        j_shape_low     = vec![to_rbool(ms.can_be_j_shape_low())],
+        j_shape_high    = vec![to_rbool(ms.can_be_j_shape_high())]
     )
     .into()
 }
 
+/// Long format, one row per (class, grid value), like `modality_shapes.parquet`.
+fn modality_shapes_to_robj(ms: &ModalityShapes, values: &[f64]) -> Robj {
+    let (mut class, mut n_samples, mut value, mut count_lo, mut count_hi) =
+        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
+    for b in &ms.bounds {
+        for (i, (&lo, &hi)) in b.count_lo.iter().zip(&b.count_hi).enumerate() {
+            class.push(b.class.as_str());
+            n_samples.push(b.n_samples as f64);
+            value.push(values[i]);
+            count_lo.push(lo);
+            count_hi.push(hi);
+        }
+    }
+    data_frame!(
+        class = class,
+        n_samples = n_samples,
+        value = value,
+        count_lo = count_lo,
+        count_hi = count_hi
+    )
+    .into()
+}
+
+/// One row, same columns as `modality_summary.parquet`.
+fn modality_summary_to_robj(ms: &ModalityShapes) -> Robj {
+    let mut pairs: Vec<(String, Robj)> = vec![
+        ("exhaustive".into(), ms.exhaustive.into()),
+        ("n_scanned".into(), (ms.n_scanned as f64).into()),
+        ("min_prominence".into(), ms.min_prominence.into()),
+        ("deficit_min".into(), (ms.deficit_min as f64).into()),
+        ("deficit_mean".into(), ms.deficit_mean.into()),
+        ("deficit_max".into(), (ms.deficit_max as f64).into()),
+    ];
+    for class in ShapeClass::all() {
+        pairs.push((format!("n_{}", class.as_str()), (ms.n_of(class) as f64).into()));
+    }
+    for class in ShapeClass::all() {
+        let n = ms.band_n_per_class[class as usize] as f64;
+        pairs.push((format!("band_n_{}", class.as_str()), n.into()));
+    }
+    let mut df: Robj = List::from_pairs(pairs).into();
+    df.set_attrib("class", "data.frame").unwrap();
+    df.set_attrib("row.names", vec![1i32]).unwrap();
+    df
+}
+
+/// One row per (rung, class), same columns as `modality_prominence.parquet`.
+fn modality_prominence_to_robj(ms: &ModalityShapes) -> Robj {
+    let (mut prom, mut counts, mut primary, mut class, mut n_samples) =
+        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
+    for rung in &ms.ladder {
+        for c in ShapeClass::all() {
+            prom.push(rung.min_prominence);
+            counts.push(rung.min_prominence_counts as i32);
+            primary.push(rung.primary);
+            class.push(c.as_str());
+            n_samples.push(rung.n_of(c) as f64);
+        }
+    }
+    data_frame!(
+        min_prominence = prom,
+        min_prominence_counts = counts,
+        primary = primary,
+        class = class,
+        n_samples = n_samples
+    )
+    .into()
+}
+
 /// Helper function to convert FrequencyTable to R data frame
 /// The frequency table now includes a 'samples' column as the first column
 fn frequency_table_to_robj(freq_table: &closure_core::FrequencyTable) -> Robj {
-    // Create a data frame with columns: samples, value, f_count, f_relative
     let df = data_frame!(
         samples = freq_table.samples_group().to_vec(),
         value = freq_table.value().to_vec(),
-        f_count = freq_table.f_count().to_vec(),
+        f_expected = freq_table.f_expected().to_vec(),
+        f_representative = freq_table.f_representative().to_vec(),
         f_relative = freq_table.f_relative().to_vec()
     );
 
@@ -231,7 +308,9 @@
     let frequency_dist = frequency_dist_to_robj(&rl.frequency_dist);
     let modality_counts = modality_counts_to_robj(&rl.modality_counts);
     let modality_pairs = modality_pairs_to_robj(&rl.modality_pairs);
-    let modality_conclusion = modality_conclusion_to_robj(&rl.modality_conclusion);
+    let modality_conclusion = modality_conclusion_to_robj(&rl.modality_shapes);
+    let modality_shapes =
+        modality_shapes_to_robj(&rl.modality_shapes, rl.results.counts.grid().values());
     let results = results_table_to_robj(&rl.results);
 
     vec![
@@ -242,6 +321,9 @@
         ("modality_counts", modality_counts),
         ("modality_pairs", modality_pairs),
         ("modality_conclusion", modality_conclusion),
+        ("modality_shapes", modality_shapes),
+        ("modality_summary", modality_summary_to_robj(&rl.modality_shapes)),
+        ("modality_prominence", modality_prominence_to_robj(&rl.modality_shapes)),
         ("results", results),
     ]
 }
@@ -252,15 +334,9 @@
     // Clone the id vector (Vec<f64>) for R compatibility
     let id_vec: Vec<f64> = results_table.id.clone();
 
-    // Convert each sample to an R integer vector and collect into a list
-    let samples_robjs: Vec<Robj> = results_table
-        .sample
-        .iter()
-        .map(|sample| {
-            // Each sample becomes an R integer vector
-            let sample_clone: Vec<i32> = sample.clone();
-            sample_clone.into_robj()
-        })
+    // Samples are stored as count vectors now; expand one at a time.
+    let samples_robjs: Vec<Robj> = (0..results_table.len())
+        .map(|i| results_table.sample(i).into_robj())
         .collect();
 
     // Create a list of samples (each element is a vector)
@@ -289,7 +365,9 @@
     rounding_error_mean: f64,
     rounding_error_sd: f64,
 ) -> Robj {
-    let count = closure_count(
+    // Invalid input is an error now, not a count of 0; report it the way
+    // `create_combinations()` does.
+    match closure_count(
         mean,
         sd,
         n,
@@ -297,9 +375,10 @@
         scale_max,
         rounding_error_mean,
         rounding_error_sd,
-    );
-
-    Robj::from(count)
+    ) {
+        Ok(count) => Robj::from(count),
+        Err(e) => Robj::from(format!("CLOSURE error: {}", e)),
+    }
 }
 
 #[extendr]
@@ -406,7 +485,7 @@
         let result = match result {
             Ok(r) => r,
             Err(e) => {
-                return Robj::from(format!("Streaming error: {:?}", e));
+                return Robj::from(format!("Streaming error: {}", e));
             }
         };
         let result_list = list!(
@@ -435,7 +514,7 @@
             ) {
                 Ok(results) => results,
                 Err(e) => {
-                    return Robj::from(format!("CLOSURE error: {:?}", e));
+                    return Robj::from(format!("CLOSURE error: {}", e));
                 }
             }
         }
@@ -463,7 +542,7 @@
             ) {
                 Ok(results) => results,
                 Err(e) => {
-                    return Robj::from(format!("SPRITE error: {:?}", e));
+                    return Robj::from(format!("SPRITE error: {}", e));
                 }
             }
         }
```
