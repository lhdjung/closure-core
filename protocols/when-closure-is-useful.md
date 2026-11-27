# When is CLOSURE still useful?

2026-11-24. Written after the single-peak ceiling and the per-sample shape
numbers were added (see `protocols/horns-and-shapes.qmd`). The question: now
that some shape conclusions follow from the reported statistics alone, what is CLOSURE's
exhaustive listing of datasets still needed for?

## Short answer

CLOSURE moves from being the method to being the exact tool for close looks.
Screening should rely on faster checks. CLOSURE remains the only way to make
exact statements about the counts themselves, to show the datasets a set of
statistics allows, and to check new analytic bounds.

## What else can now answer CLOSURE's questions

| Question | Fastest answer today |
|---|---|
| Do the reported statistics fit together at all? | `closure_count` exactly, by dynamic programming, without listing datasets. GRIM, GRIMMER and BRIMMER are quicker screens, but they test necessary conditions only |
| Could the data have been strictly single-peaked? | Often the reported mean and SD alone: `beyond_single_peak`. Otherwise CLOSURE's `deficit_min` |
| Must the data have had several peaks? | Almost always the reported statistics alone (see the sweep). Otherwise CLOSURE |
| Could it have been a bell rather than a floor or ceiling effect? Flat? | CLOSURE, with `edge_drop_*`, `bar_gap_*` and the band queries |
| What must the counts have been? ("at least 12 people answered 1"; "if it was a bell, the count at 4 was 18 to 25") | CLOSURE only: `modality_counts` and `modality_shapes` |
| What do the datasets look like? | CLOSURE (all of them); SPRITE (examples, at any n) |

## The sweep

`examples/shape_conclusion_sweep.rs` runs CLOSURE over a grid of reported
statistics: means from 1.2 to the scale maximum minus 0.2 in steps of 0.1, SDs
from 0.3 to 2.6 in steps of 0.1, both rounded to two decimals (rounding error
0.005). For every consistent case it records which shape conclusions CLOSURE
reaches, at every cutoff in the default band, and whether checks that use only
the reported statistics reach them too: the mean-free single-peak ceiling, and
`beyond_single_peak`, which uses the largest SD of a single-peaked
histogram of the actual $n$ values at every admissible mean, computed exactly
(tides_article, Appendix L). "Middle third" means a mean in the middle third
of the scale, where a floor or ceiling effect is not obvious from the mean.

| | 1–5, n = 20 | 1–5, n = 50 | 1–7, n = 30 |
|---|---|---|---|
| consistent cases | 97 | 204 | 443 |
| CLOSURE: must have several peaks | 22 | 52 | 87 |
| … caught by the mean-free ceiling | 20 | 48 | 74 |
| … caught by `beyond_single_peak` | 22 | 52 | 87 |
| statistics only: beyond the mean-free ceiling | 20 | 52 | 86 |
| statistics only: `beyond_single_peak` | 32 | 85 | 167 |
| CLOSURE: every sample has a dip | 46 | 96 | 199 |
| … caught by the mean-free ceiling | 20 | 52 | 86 |
| … caught by `beyond_single_peak` | 32 | 85 | 167 |
| CLOSURE: cannot be a bell | 54 | 123 | 193 |
| middle-third means | 53 | 106 | 205 |
| … cannot be a bell, and no `beyond_single_peak` | 2 | 6 | 2 |

How to read it:

- **"Must have several peaks" is free here.** `beyond_single_peak` catches
  all of CLOSURE's verdicts (22 of 22, 52 of 52, 87 of 87), instantly. The
  mean-free ceiling alone catches fewer (20, 48, 74).
- **"Every sample has a dip" is mostly free.** `beyond_single_peak` catches
  about two thirds to nine tenths of these verdicts (32 of 46, 85 of 96, 167
  of 199). Before it was computed exactly at the actual n and mean, it caught
  24, 79 and 126. It sets the flag when even the lowest admissible SD is past
  the largest any single-peaked histogram of the actual n values with an
  admissible mean can have. What remains are cases where some single-peaked
  histogram with an admissible mean is at least as wide as the lowest
  admissible SD, but none has an SD in the admissible range.
  `closure_count_details` settles those exactly without listing samples.
- **"Cannot be a bell" is where CLOSURE still adds the most, but less than it
  seems.** Most of these verdicts have a mean near an end of the scale, where
  a floor or ceiling effect is visible from the mean itself, or a flagged SD.
  With the mean in the middle third and no flag, they are 1–6% of cases.
  These verdicts are taken at the cutoff, so read them with `edge_drop_*` and
  `second_peak_*` rather than on their own.
- **At typical SDs with a central mean, samples of every shape fit.** There is
  nothing to infer, and no method could do better.

Caveats: the grid is not a sample of published statistics, the scales are
two, and the sample sizes small. The pattern is clear enough to act on, but
its proportions should not be quoted as rates in the literature.

## What only CLOSURE gives

1. **Exact statements about the counts at the actual n.** Per-value count
   ranges across all datasets (`modality_counts`), and within each shape class
   (`modality_shapes`): "every dataset has at least 12 responses of 1", "if
   the data was a bell, the count at 4 was between 18 and 25". These are often
   the most persuasive findings in an assessment, and no formula gives them
   yet.
2. **The datasets themselves.** Showing an author, editor or reader the
   complete set of datasets their statistics allow, or its medoid and its
   extremes, carries more weight than a bound.
3. **Ground truth.** Every analytic result in `horns-and-shapes.qmd` was
   checked against exact enumeration, and the sweep above needs it. New bounds
   should keep being checked this way.

## When not to use it

- **To check consistency only.** `closure_count` answers whether any dataset
  exists, exactly and much faster. GRIM, GRIMMER and BRIMMER screen
  analytically, but passing them does not prove a dataset exists: mean 3,
  SD 1.41, n = 3 on 1–5 passes GRIM, the parity of the sum of squares and the
  largest-SD bound, yet no dataset has it.
- **To screen many papers.** Run the fast checks first, and CLOSURE only on
  what they leave open.
- **At large n or on wide scales.** Listing explodes: for mean 4, SD 1.5,
  n = 100 on 1–7, about 76,000 datasets at two decimals (rounding error
  0.005), and about 6.1 million at one decimal (0.05). There, SPRITE gives examples,
  and absences of shapes can come only from `beyond_single_peak`.

## Suggested order

1. Consistency: GRIM, GRIMMER and BRIMMER as quick screens, then
   `closure_count`, which is exact. If no dataset exists, that is the
   finding; there is no shape to infer.
2. Shape from the reported statistics alone: `beyond_single_peak`. If set,
   the data was not single-peaked, for any n and with either technique.
3. If questions remain and n is moderate: CLOSURE, for exact statements about
   shapes and counts, with the numbers behind each verdict (`bar_gap_*`,
   `second_peak_*`, `edge_drop_*`), and for figures.

## Where this could go

- **Done: an n-sharp line at the reported mean.** `beyond_single_peak` uses
  the largest single-peaked SD at the actual n and mean, computed exactly by
  the recursion of tides_article, Appendix L, up to a size limit (on 7
  values, n up to 528), and $B_\mu$ beyond it.
- **Done: a dynamic program for counts and shapes.** `closure_count_details`
  gives per-value count ranges and the existence of a single-peaked dataset
  exactly, without listing datasets. Count ranges within the other shape
  classes still need CLOSURE.

## Reproduce

```
cargo run --release --example shape_conclusion_sweep
```

It takes a second or two in release mode.
