# Adapting unsum to the current closure-core

This guide lists every change in closure-core since the version unsum uses
that affects unsum, and says exactly what to change in unsum for each one.

- **From:** unsum `6d83a70` ("Adopt current closure-core and store samples as
  counts"). Its `src/rust/Cargo.lock` pins closure-core `optimize#5d63b62`,
  and it writes the counts layout (`OutputFormat::Counts`).
- **To:** closure-core `7bf2a52` on `optimize`.
- **Line numbers** refer to unsum's master at `9dc590d`.

The earlier migration from closure-core `1fe8f95`, which a previous version of
this guide set out, is done in unsum `6d83a70`. That version is still in the
history: `git show 21d63b9:UNSUM-MIGRATION.md`.

Everything marked *verified* was run. The binding patch in section 3 was
compiled with `cargo check` against `7bf2a52`, without warnings. The R changes
in section 4 were not run.

---

## Contents

1. [The short version](#1-the-short-version)
2. [Step 1: get the new closure-core](#2-step-1-get-the-new-closure-core)
3. [Step 2: the Rust binding](#3-step-2-the-rust-binding-srcrustsrclibrs)
4. [Step 3: the R code](#4-step-3-the-r-code)
5. [Results that change](#5-results-that-change)
6. [New errors](#6-new-errors)
7. [Multi-item SPRITE samples are rounded to hundredths](#7-multi-item-sprite-samples-are-rounded-to-hundredths)
8. [Commit index](#8-commit-index)

---

## 1. The short version

Work through the rows in order. **Blocks** says what happens if the row is
skipped.

| # | What changed in closure-core | Blocks | What to change in unsum | Where |
|---|---|---|---|---|
| 1 | The commits after `7054029` are not on GitHub yet | everything | Push `optimize`, then update and re-vendor | closure-core, then `src/rust/Cargo.lock`, `vendor.tar.xz` |
| 2 | The four entry points take a last `prominence` argument | compile | Pass `None` | `src/rust/src/lib.rs`, the calls at lines 421, 434, 468 and 481 |
| 3 | `modality_summary` has eight new columns: `bar_gap_*`, `second_peak_*`, `edge_drop_*` | the schema check, in memory and on disk | Export them, and expect 26 columns | `src/rust/src/lib.rs` 145, `R/constants.R` 69, `R/utils.R` 327 |
| 3a | `metrics_horns` has two new columns: `single_peak_ceiling`, `beyond_single_peak` | the schema check, on disk at once and in memory once exported | Export them, and expect 11 columns | `src/rust/src/lib.rs` 287, `R/utils.R` 196–210 |
| 4 | Shape classes and SPRITE coverage changed | snapshot tests | Update snapshots and `NEWS.md` (section 5) | `tests/`, `NEWS.md` |
| 5 | New input errors | nothing; they arrive as strings | Mention in `NEWS.md` (section 6) | |
| 6 | Multi-item SPRITE samples are in rounded hundredths | exact statistics at `items` = 3, 6, 7, … | Use the grid's exact values where it matters (section 7) | wherever samples are recomputed |

---

## 2. Step 1: get the new closure-core

unsum already follows `branch = "optimize"`. But GitHub's `optimize` is at
`7054029`, and everything after it, including all the changes below, exists
only in the local clone of closure-core. Cargo can only fetch what is pushed,
and at `7054029` the patch in section 3 does not compile: there is no
`prominence` argument and none of the new summary fields. So:

1. In closure-core: `git push origin optimize`.
2. In unsum's `src/rust`: `cargo update -p closure-core`, and check that
   `Cargo.lock` now names `7bf2a52` or later.
3. Regenerate `src/rust/vendor.tar.xz` and `vendor-config.toml` (e.g.
   `rextendr::vendor_pkgs()`). The offline build in `src/Makevars.in` unpacks
   that tarball, so the old copy would otherwise keep being compiled.

closure-core's dependencies are unchanged since `5d63b62`.

---

## 3. Step 2: the Rust binding (`src/rust/src/lib.rs`)

Against the new closure-core, the binding fails with four errors
(*verified*):

```
error[E0061]: this function takes 11 arguments but 10 arguments were supplied   (×2, CLOSURE)
error[E0061]: this function takes 13 arguments but 12 arguments were supplied   (×2, SPRITE)
```

This patch fixes them and exports the new columns of `modality_summary` and
`metrics_horns`. *Verified:* it compiles
without warnings.

```diff
@@ -143,6 +143,14 @@
         ("deficit_min".into(), (ms.deficit_min as i32).into()),
         ("deficit_mean".into(), ms.deficit_mean.into()),
         ("deficit_max".into(), (ms.deficit_max as i32).into()),
+        ("bar_gap_min".into(), (ms.bar_gap_min as i32).into()),
+        ("bar_gap_max".into(), (ms.bar_gap_max as i32).into()),
+        ("bar_gap_min_share".into(), ms.bar_gap_min_share.into()),
+        ("bar_gap_max_share".into(), ms.bar_gap_max_share.into()),
+        ("second_peak_min".into(), (ms.second_peak_min as i32).into()),
+        ("second_peak_max".into(), (ms.second_peak_max as i32).into()),
+        ("edge_drop_min".into(), (ms.edge_drop_min as i32).into()),
+        ("edge_drop_max".into(), (ms.edge_drop_max as i32).into()),
     ];
     for class in ShapeClass::all() {
         pairs.push((format!("n_{}", class.as_str()), (ms.n_of(class) as f64).into()));
@@ -284,7 +292,9 @@
         min = rl.metrics_horns.min,
         median = rl.metrics_horns.median,
         max = rl.metrics_horns.max,
-        range = rl.metrics_horns.range
+        range = rl.metrics_horns.range,
+        single_peak_ceiling = rl.metrics_horns.single_peak_ceiling,
+        beyond_single_peak = rl.metrics_horns.beyond_single_peak
     )
     .into();
 
@@ -429,6 +439,7 @@
                 items_val,
                 streaming_config,
                 stop_after,
+                None, // prominence: the default cutoffs
             )
         } else {
             sprite_parallel_streaming(
@@ -444,6 +455,7 @@
                 restrict_min_parsed,
                 streaming_config,
                 stop_after,
+                None, // prominence: the default cutoffs
             )
         };
 
@@ -476,6 +488,7 @@
             items_val,
             None, // No parquet config - just return results in memory
             stop_after,
+            None, // prominence: the default cutoffs
         )
     } else {
         sprite_parallel(
@@ -491,6 +504,7 @@
             restrict_min_parsed,
             None, // No parquet config - just return results in memory
             stop_after,
+            None, // prominence: the default cutoffs
         )
     };
 
```

**The counts must be `as i32`, not `as f64`.** Streaming writes `bar_gap_min`,
`bar_gap_max`, `second_peak_*` and `edge_drop_*` as unsigned integers, which
nanoparquet reads as R integers, and the schema check compares `typeof`
exactly. The `deficit_*` columns already follow this rule.

**`prominence`.** `None` keeps the default cutoffs for the shape
classification: a mode must stand 5% of `n` above its surroundings for the
per-class counts, and the shape queries consider every whole-observation
cutoff from 2% to 10% of `n`. Both are conventions, not derived values. To
expose them to R, build `ProminenceThresholds { primary, band }` from two R
arguments. Every cutoff must be a fraction of `n` above 0 and below 1 (`0.05`,
not `5`); anything else is an `InputValidation` error. A `primary` outside the
band widens the band to reach it.

**What the new columns mean.** Each verdict now comes with the number it
rests on, in observations, as its smallest and largest value over the samples
(0 for the smallest when there are no samples):

- `bar_gap_*`: the gap between a sample's tallest and shortest bar; the
  number behind *flat*. If `bar_gap_max` is 2, every sample's bars lie within
  2 observations of each other. The `_share` columns divide by the average
  bar, `n / k`, and are `NaN` when there are no samples.
- `second_peak_*`: the largest cutoff at which a sample still has two or more
  modes, 0 if it never does; the number behind *multimodal*. If
  `second_peak_min` is 7, every sample's second peak stands at least 7
  observations above its dip, so every sample is multimodal at any cutoff up
  to 7. If `second_peak_max` is 0, no sample has two modes at any cutoff.
- `edge_drop_*`: how far a sample's tallest bar stands above the lowest bar
  between it and the nearer end of the scale, 0 if it sits at an end; the
  number behind *bell* versus *floor or ceiling effect*. A single-peaked
  sample is a bell at any cutoff up to its edge drop. If `edge_drop_min` is 8,
  every single-peaked sample is a bell at any cutoff up to 8.

Each is exactly where the class changes, so a reader can weigh these numbers
against `n` without adopting closure-core's cutoffs.

In `metrics_horns`, `single_peak_ceiling` is the highest horns index any
single-peaked histogram on the scale can have, `(2k − 1)² / (9 (k − 1)²)` for
`k` grid values: 0.5625 on 1–5. `beyond_single_peak` is `TRUE` when no
single-peaked histogram meets the reported statistics: at every mean the
rounding allows, even the lowest SD it allows is beyond the largest a
single-peaked histogram with that mean can have (Navard, Seaman and Young's
`B_μ`, whose maximum over the mean is the ceiling). Then every histogram with
these statistics has a dip, so, if they are right, the original data was not
single-peaked. It follows from the reported statistics alone, so it holds for
SPRITE and for runs with `stop_after` too. On 1–5 the line is a population SD
of 1.5 at means 2.5 and 3.5, and 1.41 at mean 3; at mean 3 and `n = 50`, a
reported SD of 1.44 is flagged. The derivation is in closure-core's
`protocols/horns-and-shapes.qmd`.

---

## 4. Step 3: the R code

`R/constants.R`, line 69: add the eight columns right after `deficit_max`.
`check_component_tibble()` compares names with `identical()`, so the order
must match the binding and the file.

```r
MODALITY_SUMMARY_TYPES <- c(
  list(
    "exhaustive" = "logical",
    "n_scanned" = "double",
    "min_prominence" = "double",
    "deficit_min" = "integer",
    "deficit_mean" = "double",
    "deficit_max" = "integer",
    "bar_gap_min" = "integer",
    "bar_gap_max" = "integer",
    "bar_gap_min_share" = "double",
    "bar_gap_max_share" = "double",
    "second_peak_min" = "integer",
    "second_peak_max" = "integer",
    "edge_drop_min" = "integer",
    "edge_drop_max" = "integer"
  ),
  `names<-`(as.list(rep("double", 6L)), paste0("n_", SHAPE_CLASSES)),
  `names<-`(as.list(rep("double", 6L)), paste0("band_n_", SHAPE_CLASSES))
)
```

`R/utils.R`, line 327: `dims = c(1L, 18L)` becomes `dims = c(1L, 26L)`.

`R/utils.R`, lines 196–210, the `metrics_horns` check: `dims = c(1L, 9L)`
becomes `dims = c(1L, 11L)`, and two types follow `"range" = "double"`:

```r
      "range" = "double",
      "single_peak_ceiling" = "double",
      "beyond_single_peak" = "logical"
```

Streaming writes the two columns already, so without this change, results
read from disk fail the check even before the binding exports them.

Nothing else in R needs to change. `modality_conclusion` and its rebuild from
`modality_summary` work as before.

---

## 5. Results that change

None of these need code in unsum. Some will break snapshot tests, and all of
them belong in `NEWS.md`.

**CLOSURE**

- **Large samples no longer crash R.** The search recursed once per value, so
  from about `n = 20,000` it overflowed its stack and aborted the process.
  Deep branches now run on an explicit stack. (`ae301d4`)
- At large `n` (from about 18,000), samples exactly on an SD bound were
  dropped, because the bounds' fixed float slack no longer covered f64 error
  there. Counts can **rise**. (`72d2f41`)
- A run with `stop_after` that ends below its limit has seen every sample and
  is marked exhaustive, so `modality_conclusion` can be `TRUE`/`FALSE` where it
  was `NA`. (`93cf367`)
- A complete search that finds no sample now says `exhaustive = TRUE` in
  `modality_summary`. `modality_conclusion` stays `NA`: no samples is the
  consistency finding, not a shape one. `modality_prominence` lists the
  cutoffs in observations of the real `n` (2, 5, 10 at `n = 100`), not 1 each.
  (`ce0783d`)
- Wide scales are fast: a scale of 0–600 at `n = 3` took 11 s and 1.7 GB, and
  now takes under a millisecond. (`bc29dec`)

**SPRITE**

- **The search walks differently**, so sample sets differ from before, and
  coverage rises. Every shift moves its first value up or down at random,
  which reaches samples behind an exactly restricted value (`4f2b079`); a
  compensating value may move several grid steps
  (`cf5c659`), and one attempt in ten starts from a random fill (`7054029`).
  On fine grids (`items` near 100 on a wide scale), the per-attempt step limit
  now grows with the grid, so high SDs are reached where every attempt used to
  fail (`cbdd3b7`).
- **Acceptance is CLOSURE's integer rule.** A sample is kept when its exact
  integer sum and `n·Σx² − (Σx)²` lie within CLOSURE's bounds, not by comparing
  a float SD. Results differ only for samples sitting on a bound, and on scale
  values in the millions. (`15b08ab`)
- **Small solution spaces take longer and miss fewer samples.** The duplicate
  rate may stop a search only after 5,000 successful draws; tiny cases take
  about 5 ms instead of 0.15 ms. Where draws are expensive, this floor sets the
  run time: mean 3, SD 2, `n = 1000` on 1–5 with tolerances 0.005 and 0.001
  has 23 samples, and SPRITE finds about 21 of them in 4–5 s (it used to find
  none). With tolerances of 0.05 the same mean and SD admit about a million
  samples, so a run without `stop_after` goes on until the duplicate rate or
  the attempt cap ends it. (`33bb164`, `38adc9d`, `7223cc7`)
- **One attempt cap for both modes.** A search makes at most 20 attempts per
  sample asked for, between 5,000 and 100 million, and 100 million without
  `stop_after`. In memory, a run without `stop_after` used to have no
  practical bound; streaming with `stop_after` used to try up to 100 million
  times. (`533ea9a`)
- **Hopeless SDs return at once with no samples**, including when fixed
  responses (restrictions) rule the SD out. (`15d2f6c`, `72bcf94`)
- **Near the narrowest possible SD, runs finish in milliseconds.** An attempt
  gives up when no move is left, or when ten shifts in a row bring it no closer
  to the SD. Mean 3, SD 0.08, `n = 300` on 1–5 took 66 s and now takes about
  35 ms. (`abae5c7`, `38adc9d`)
- Memory mode no longer prints warnings to stderr when it stops early.
  (`df52c88`)
- Restriction keys are the output grid's own labels. At `items` = 40, 56 and
  80, a value ending in exactly half a hundredth used to need a different key
  from the one it is labelled with in the results. Matters only if
  `restrict_exact` or `restrict_min` is used. (`f2e83a4`)

**Shapes**

- **Near-uniform samples are flat.** A sample is flat when its tallest bar
  exceeds its shortest by less than the cutoff, so `(30, 30, 30, 30, 29)` is
  flat. The tallest bar used to be a mode at every cutoff, so that sample read
  as a floor effect and `(29, 30, 30, 30, 29)` as a bell. `n_flat` and
  `band_n_flat` rise; the one-mode classes fall. On scales with many values and
  a small `n` many more samples are flat. (`f862887`)
- **A single peak reaches a scale end when the bars between are within the
  cutoff of it.** Whether a peak was a bell or a floor effect used to turn on
  the exact position of the tallest bar: at a cutoff of 5,
  `(31, 29, 30, 5, 5)` was a floor effect and `(30, 29, 31, 5, 5)` a bell.
  Both are floor effects now, as is `(19, 21, 2, 2, 6)`. `j_shape_low` and
  `j_shape_high` can become `TRUE` where they were `FALSE`, and
  `n_one_mode_interior` falls: for mean 2.1, SD 1.3, `n = 50` on 1–5, from 3
  to 1. (`d3749b6`)
- Equal lower peaks separated by a dip below the cutoff merge into one mode,
  as tied tallest runs already did: `(8, 7, 8, 2, 20)` has two modes, not
  three, matching its mirror image. (`dc2cf02`)
- The unimodality deficit has the same values and is computed in linear time,
  which matters only on very wide grids. (`9c357f3`)

---

## 6. New errors

All of these arrive in R as the character string the binding returns. unsum
already turns that into `abort_in_export()`.

| Condition | Error | Before |
|---|---|---|
| More than 100,000 scale values (CLOSURE), or grid values `(scale_max − scale_min) · items + 1` (SPRITE) | `InputValidation` | CLOSURE could list billions of seeds; SPRITE spent 40 MB per sample. `count_closure_combinations()` has no such limit. |
| CLOSURE: `scale_min == scale_max` | `InputValidation` | every sample `Flat`, so an exhaustive run "proved" the data not unimodal |
| SPRITE: `items > 100` | `InputValidation` | neighbouring grid values shared a key in hundredths |
| SPRITE: scale values beyond ±21,474,836 | `InputValidation` | panic building the output grid |
| SPRITE: an infinite SD or tolerance | `InputValidation` | an empty result |
| SPRITE: restrictions fixing more responses than `n`, however large the counts | `InputValidation` | a huge count panicked or tried to allocate terabytes |
| A prominence cutoff not above 0 and below 1 (only once cutoffs are exposed) | `InputValidation` | a cutoff of 5 meant as 5% classified every sample as flat |

---

## 7. Multi-item SPRITE samples are rounded to hundredths

SPRITE reports samples in hundredths of a scale point: `results.sample(i)`
and the `value_hundredths` column of `scale_values.parquet` give 150 for 1.5.
When `items` does not divide 100 (3, 6, 7, 9, …) those are **rounded**: at
`items = 3` the grid value 4/3 is 133. The sample itself met the mean and SD
on the exact values, but statistics recomputed from rounded hundredths can
miss a tight tolerance. (`e030a5b`)

For anything computed from multi-item samples, use the exact values:
`scale_values.parquet`'s `value` column, or `counts.grid().values()` in the
binding, with the per-value counts. For `items` = 1, 2, 4, 5, 10, 20, 25, 50
and 100 the hundredths are exact.

---

## 8. Commit index

Commits since `5d63b62` that affect unsum; the rest are tests, docs, tooling
and refactors.

| Commit | Effect on unsum |
|---|---|
| `490a6c3` | `prominence` argument on the four entry points (3) |
| `f862887` | Near-uniform samples are flat; `bar_gap_*` columns (3, 4, 5) |
| `a248020` | `second_peak_*` and `edge_drop_*` columns (3) |
| `7bf2a52` | `single_peak_ceiling` and `beyond_single_peak` in `metrics_horns` (3) |
| `d3749b6` | A single peak's edge is decided by the bars within the cutoff (5) |
| `dc2cf02` | Equal lower peaks merge (5) |
| `47784a3` | Cutoffs must lie between 0 and 1; a cutoff of 0 reads as 1 (6) |
| `ce0783d` | Complete empty runs are exhaustive; real `n` in `modality_prominence` (5) |
| `ae301d4` | No stack overflow at large `n` (5) |
| `bc29dec` | Wide CLOSURE scales are fast; at most 100,000 scale values (5, 6) |
| `72d2f41` | Exact bounds at large `n` (5) |
| `93cf367` | Limited runs below their limit are exhaustive (5) |
| `4f2b079`, `cf5c659`, `7054029`, `cbdd3b7` | SPRITE reaches more samples (5) |
| `15b08ab` | SPRITE accepts by CLOSURE's integer rule (5) |
| `33bb164`, `38adc9d`, `abae5c7`, `7223cc7` | SPRITE run times on small and hard cases (5) |
| `533ea9a` | One SPRITE attempt cap for memory and streaming (5) |
| `15d2f6c`, `72bcf94` | SPRITE returns at once on hopeless SDs (5) |
| `f2e83a4` | SPRITE restriction keys follow the grid labels (5) |
| `4f2b079`, `236c6cc`, `6a1290e`, `67af72f`, `f55bf11` | New input errors (6) |
| `e030a5b` | Multi-item SPRITE hundredths are rounded (7) |
| `df52c88` | No stderr warnings from in-memory SPRITE (5) |
| `03a04da` | f32 inputs keep samples on a bound; unsum passes f64, so nothing changes |
| `c718a69` | `file_path = ""` means the current directory; unsum always passes a folder |
