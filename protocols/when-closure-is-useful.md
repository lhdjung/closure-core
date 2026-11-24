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

| Question | Fastest exact answer today |
|---|---|
| Do the reported statistics fit together at all? | GRIM, GRIMMER and BRIMMER analytically; `closure_count` exactly, by dynamic programming, without listing datasets |
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
`beyond_single_peak`, which uses the bound at the reported mean (Navard et
al.'s $B_\mu$, proved in `horns-and-shapes.qmd`). "Middle third" means a mean in the middle third of the scale, where
a floor or ceiling effect is not obvious from the mean.

| | 1–5, n = 20 | 1–5, n = 50 | 1–7, n = 30 |
|---|---|---|---|
| consistent cases | 97 | 204 | 443 |
| CLOSURE: must have several peaks | 22 | 52 | 87 |
| statistics only: beyond the mean-free ceiling | 20 | 52 | 86 |
| statistics only: `beyond_single_peak` | 24 | 79 | 126 |
| CLOSURE: every sample has a dip | 46 | 96 | 199 |
| … caught by the mean-free ceiling | 20 | 52 | 86 |
| … caught by `beyond_single_peak` | 24 | 79 | 126 |
| CLOSURE: cannot be a bell | 54 | 123 | 193 |
| middle-third means | 53 | 106 | 205 |
| … cannot be a bell, and no `beyond_single_peak` | 6 | 10 | 8 |

How to read it:

- **"Must have several peaks" is nearly free.** Even the mean-free ceiling
  reproduces almost every one of CLOSURE's verdicts, instantly and at any n.
- **"Every sample has a dip" is mostly free.** `beyond_single_peak` catches
  about half to two thirds of these verdicts (24 of 46, 79 of 96, 126 of
  199). It sets the flag exactly when no single-peaked histogram with free
  proportions fits. What remains comes from whole-number effects at the
  actual n, which only an exact method sees.
- **"Cannot be a bell" is where CLOSURE still adds the most, but less than it
  seems.** Most of these verdicts have a mean near an end of the scale, where
  a floor or ceiling effect is visible from the mean itself, or a flagged SD.
  With the mean in the middle third and no flag, they are 4–11% of cases.
  They sit just below the line, with margins of 0–3 observations, so they
  depend on the cutoff.
- **At typical SDs with a central mean, CLOSURE finds every shape possible.**
  There is nothing to infer, and no method could do better.

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
  exists, exactly and much faster; GRIM, GRIMMER and BRIMMER do it
  analytically.
- **To screen many papers.** Run the fast checks first, and CLOSURE only on
  what they leave open.
- **At large n or on wide scales.** Listing explodes: about 6.1 million
  datasets for mean 4, SD 1.5, n = 100 on 1–7. There, SPRITE gives examples,
  and absences of shapes can come only from `beyond_single_peak`.

## Suggested order

1. Consistency: GRIM, GRIMMER, BRIMMER, or `closure_count`. If no dataset
   exists, that is the finding; there is no shape to infer.
2. Shape from the reported statistics alone: `beyond_single_peak`. If set,
   the data was not single-peaked, for any n and with either technique.
3. If questions remain and n is moderate: CLOSURE, for exact statements about
   shapes and counts, with the numbers behind each verdict (`bar_gap_*`,
   `second_peak_*`, `edge_drop_*`), and for figures.

## Where this could go

- **An n-sharp ceiling**, like Popoviciu's odd-n correction for the range-only
  bound, would catch more small samples.
- **A dynamic program for counts and shapes.** `closure_count` already counts
  datasets without listing them. The same approach could plausibly give
  per-value count ranges and the existence of a single-peaked dataset exactly,
  at larger n. That is a guess; it has not been built. If it works, CLOSURE's
  only unique product is the list of datasets itself.

## Reproduce

```
cargo run --release --example shape_conclusion_sweep
```

It takes under a second in release mode.
