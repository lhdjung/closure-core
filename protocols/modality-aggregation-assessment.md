# Assessment: aggregating result sets into modality claims

> **Status: implemented.** Everything in Part 5 has been built; see
> [Part 8](#part-8--what-was-implemented) for what changed and for the two
> numbers in this document that the implementation revised.

Scope: the five commits preceding `0d41544`, i.e.

| commit | subject |
| --- | --- |
| `2e24383` | Use `can_be_unimodal` and `can_be_bimodal` in `ModalityConclusion` |
| `d556ef2` | Assess modality per sample |
| `6a03fc4` | Add `is_unimodal()` and `is_bimodal_mean_between()` |
| `00686e2` | Require qualifying maxima for bimodality |
| `1fe8f95` | Modify `FrequencyTable` using Earth Mover's Distance (EMD) |

Line references are to the tree at `0d41544`, which is where these five landed.

**Verdict in one line:** the *direction* of all five commits is right — per-sample
shape assessment beats per-value bounds, and EMD beats coordinatewise averaging —
but the summaries they emit are the wrong *functionals*. Two booleans over an
existential quantifier and one arbitrarily-chosen representative sample throw away
almost all of the information the loop already computed, and in three code paths
they are silently wrong.

The most consequential single gap is identified in Part 6: an existential summary
is not merely noisy, it is strategically void, because a challenged author need only
claim their data were the one atypical member. The fix is *conditional bounds* —
per-value count ranges computed over each shape class — which are prior-free,
immune to that defence, and nearly free to compute in the loop that already exists.

All empirical claims below come from an independent Python re-enumeration of the
CLOSURE solution space (`enumerate_counts()`, brute-force validated) whose result
sets and flag values were then checked against the actual `closure_parallel()`
output — set sizes 313, 348, 162, 99, 200, 70, 9, 1539, 2792 and every flag value
agree exactly. Sweep: scale 1–5, `n ∈ {20, 30, 52, 100}`,
`mean ∈ {2.2, 3.0, 3.24, 3.5, 4.1}`, `sd ∈ {0.6, 0.8, 1.0, 1.13, 1.3, 1.5, 1.8}`,
rounding error 0.005 on both. 52 of 140 cells are non-empty.

---

## Part 1 — Does aggregating like this make sense?

### 1.1 The core logic is sound, and it is the good kind of sound

The CLOSURE result set `S` is, by construction, *every* length-`n` integer vector
over the scale whose mean and SD round to the reported values. The real raw data,
if the reported statistics are accurate, is a member of `S`. That gives a genuinely
deductive inference rule, and commit `d556ef2` is the commit that finally uses it:

> If **no** element of `S` has property *P*, then the raw data does not have
> property *P*.

This is falsification, not estimation, and it is exactly the right ambition for a
forensic tool. It needs no prior, no sampling model, and no appeal to typicality.
When `can_be_unimodal == false`, the tool has *proved* that the underlying data
were not bell-shaped, given only the reported mean, SD, `n`, and scale bounds. That
is a strong claim of a kind very few methods can make.

The commit sequence is also right that this must be computed **per sample**.
The pre-`d556ef2` version derived the flags from `count_lo`/`count_hi` —
per-value minima and maxima taken independently across samples. That reasons about
a *box* that contains `S`, not about `S`. A box corner such as
"the centre value at its maximum while both extremes sit at their minima" is
generally not a real sample: those extrema are attained by different members of
`S` and are jointly incompatible with the sum and sum-of-squares constraints.
The old flag could therefore say "unimodal shapes are possible" when no admissible
dataset is unimodal. Replacing an outer-bound relaxation with an exact scan over
`S` is a correctness fix, not a refinement, and it costs nothing: it rides along in
the horns loop (`src/lib.rs:1180-1185`).

Two smaller things are also right:

- **`is_unimodal()` on the count vector, not the raw values** (`src/lib.rs:721`).
  Shape is a property of the histogram; the frequency representation is the natural
  domain. Allowing plateaus, and allowing the peak anywhere, is the correct weak
  definition — a strict definition would reject `[5,20,20,5,5]` for no good reason.
- **EMD as the distance on frequency vectors** (`src/lib.rs:846`). Likert values are
  ordinal, so moving mass from 4 to 5 must cost less than moving it from 1 to 5.
  L1/L2 on raw count vectors treat all categories as exchangeable and are simply
  the wrong geometry. The 1-D reduction to the L1 distance between CDFs is exact
  and O(k). This one is unambiguously an improvement.

### 1.2 …but a valid inference rule does not make a valid *report*

Everything above concerns the `false` branch. The problem is what the code does with
the other branch, and how often the other branch is taken.

`can_be_unimodal == true` means only "at least one of the |S| admissible datasets
happens to be unimodal." It licenses no conclusion about the raw data whatsoever.
And in the sweep it is the usual outcome:

- both flags true simultaneously — i.e. **zero** information conveyed — in
  **23 of 52 (44%)** non-empty configurations;
- `j_shape_low` true in 43/52 (83%), `j_shape_high` in 39/52 (75%).

So most of the time `ModalityConclusion` is four `true`s. That is not a conclusion.

---

## Part 2 — Where the status quo breaks

### 2.1 The boolean is the wrong functional, and it is unstable

This is the central methodological problem.

The loop at `src/lib.rs:1180-1185` visits every sample and evaluates
`is_unimodal()` on each — the proportion is *already there* and is thrown away by
`any_unimodal |= …`. That proportion is a far better statistic, and not just
because it is more granular. The boolean is an existential over a set whose
cardinality swings by orders of magnitude with the reported precision; the
proportion is nearly invariant to it.

Target mean 3.5, SD 1.5, n = 100, scale 1–5 — confirmed against the real
`closure_parallel()`:

| rounding error | \|S\| | p(unimodal) | `can_be_unimodal` |
| --- | --- | --- | --- |
| 0.005 | 162 | 0.0000 | **false** |
| 0.01 | 1539 | 0.0013 | **true** |
| 0.05 | 28154 | 0.0021 | **true** |

The flag flips from "proof that the data were not unimodal" to "unimodality is
possible" purely because the caller widened a nuisance parameter — and the `true`
is carried by 2 samples out of 1539. Nothing about the study changed. Meanwhile
the proportion moves 0.0000 → 0.0013 → 0.0021, which is an honest and readable
account of the same fact.

The same stability shows up where the flag is uninformative. Mean 3.0, SD 1.3,
n = 100: p(unimodal) = 0.0489 / 0.0469 / 0.0479 across the three tolerances, while
`can_be_unimodal` is `true` in all three and says nothing.

And the proportions are genuinely discriminating where the flags are not — all six
rows below report `can_be_unimodal = true, can_be_bimodal = true`:

| n | mean | sd | \|S\| | p(uni) | p(bim) |
| --- | --- | --- | --- | --- | --- |
| 100 | 3.00 | 1.13 | 313 | 0.224 | 0.326 |
| 100 | 3.00 | 1.30 | 348 | 0.049 | 0.589 |
| 100 | 3.24 | 1.30 | 647 | 0.042 | 0.685 |
| 100 | 3.50 | 1.30 | 245 | 0.045 | 0.612 |
| 52 | 4.10 | 1.30 | 23 | 0.043 | 0.304 |
| 100 | 2.20 | 1.30 | 166 | 0.054 | 0.512 |

"22% of admissible datasets are unimodal" and "4% are" are very different states of
knowledge, reported identically.

One caveat that should be stated wherever such a proportion is: counting members of
`S` equally is a uniform prior over admissible datasets, and it is not a posterior.
But this is not a new commitment — `metrics_horns.mean`, `.sd`, `.median` and
`.mad` already average over `S` with exactly that weighting. A shape proportion is
no more of a modelling claim than the horns statistics the crate already ships.

### 2.2 The two predicates are not complements, are not comparable, and leave a hole

`is_unimodal()` is a **weak shape** test (no valley anywhere, plateaus fine, peak
anywhere). `is_bimodal_mean_between()` is a **strict, thresholded, location-aware**
test (≥ 2 strict local maxima, each above `total/k`, with the sample mean strictly
between the outermost qualifying peaks). These are not two halves of a partition —
they are two unrelated criteria with wildly different strictness, presented
side by side as if they were exhaustive.

The consequence is a large unnamed remainder. Fraction of `S` classified as
*neither*:

| n | mean | sd | \|S\| | neither |
| --- | --- | --- | --- | --- |
| 100 | 4.10 | 1.30 | 85 | 79% |
| 100 | 2.20 | 1.50 | 200 | 68% |
| 52 | 4.10 | 1.50 | 9 | **100%** |

In that last row both flags are `false`. A reader reasonably parses that as
"provably not unimodal *and* provably not bimodal," which sounds like a
contradiction, when what actually happened is that every admissible dataset has a
shape the taxonomy declines to name. That happens in 2/52 configurations.

### 2.3 `is_unimodal()` is correct; the *concept* it implements is the wrong one

An earlier draft of this document claimed `(33, 31, 24, 7, 5)` was "not unimodal in
any useful sense." That was wrong and is withdrawn. Under the standard definition —
non-decreasing up to a maximum, then non-increasing — that vector is unimodal. It
has exactly one mode. `is_unimodal()` (`src/lib.rs:721`) implements the textbook
definition faithfully and there is no bug in it.

The real problem is that **modality is the wrong single axis**, and the code half
knows this. There are two orthogonal properties here:

| | peak in the interior | peak at a scale boundary |
| --- | --- | --- |
| **one mode** | bell / centred | J-shaped (floor or ceiling effect) |
| **two or more modes** | genuine bimodality | polarised with an edge spike |
| **no unique mode** | flat / uniform | — |

`is_unimodal()` collapses the first column into the second: it answers "how many
modes" and discards "where." `j_shape_low` / `j_shape_high` answer "where" — but
they are computed from an entirely different mechanism (per-value `count_lo`/
`count_hi` bounds, `src/lib.rs:1028-1030`, i.e. the box relaxation that `d556ef2`
replaced everywhere else) and are not tied to any actual sample. So the two axes
are present in the output but are never crossed, and one of them is computed with
the method the commit series abandoned.

The measurable consequence: at mean 2.2, SD 1.13, n = 100, **13 of the 26 unimodal
samples (50%) peak at a scale endpoint**. Half the witnesses for `can_be_unimodal`
are floor/ceiling shapes. That is not a definitional error — they really are
unimodal — but the doc comment at `src/lib.rs:496-498` promises something else:

> `can_be_unimodal`: … so a **bell-shaped** sample cannot be ruled out.

"Bell-shaped" is `one mode AND interior peak`. The code checks only the first
conjunct, so the field cannot support the sentence documenting it. §6.2 shows this
gap is not academic: it is exactly the crack an adversarial author escapes through.

**Should "J-shaped" be a distinct category?** Not a distinct *modality* category —
it is a location category, and treating it as a third value of a one-dimensional
"shape" enum is what produced the current muddle. It should be a distinct
*reported* category, because a floor/ceiling effect and a bell curve are different
data-generating stories and a reader needs to know which ones remain admissible.
Crossing the two axes gives that for free, replaces both `is_unimodal` and
`j_shape_*` with one per-sample computation, and removes the last user of the box
relaxation.

**One case where the criticism does survive on definitional grounds.**
`[20, 20, 20, 20, 20]` returns `true`: it never decreases, so it never
decreases-then-increases. Weak unimodality admits the uniform distribution; strict
unimodality (unique maximum) does not. Both conventions are defensible and the code
has silently chosen the weak one — but "a flat histogram is a witness that the data
may be bell-shaped" is a claim no reader will accept, and nothing in the docs
records the choice.

### 2.3b Modality of a discrete histogram is ill-posed, which is why §2.4 exists

Worth stating explicitly, because it explains the thresholding problem rather than
just describing it. For continuous densities, modality has a clean definition. For
an integer count vector over `k` categories it does not: ties are common, and a
single count of difference creates or destroys a "strict local maximum." A vector
like `(30, 29, 30, 29, 32)` has three strict local maxima and is, by any reasonable
reading, flat.

This is why `00686e2` had to invent a threshold — the predicate is trying to
extract a robust property from a definition that has none. The principled route is
a **graded** measure rather than a predicate: the distance from the observed count
vector to the nearest weakly-unimodal count vector. That is the discrete analogue
of Hartigan's dip statistic / excess mass, it is computable exactly by isotonic
regression (PAVA run for each of the `k` candidate peak positions, so O(k²) per
sample, negligible at `k ≤ 11`), and it makes "unimodal" a threshold on a reported
continuous quantity instead of a hidden constant inside a private function.
Reporting the *distribution* of that distance across `S` would say far more than
any boolean.

### 2.4 The `avg` threshold (`00686e2`) is an unvalidated tuning constant

`00686e2` added `freqs[i] > total / k` as a peak qualification (`src/lib.rs:761`,
`768`). The motivation in the doc comment is real — a single count at a far extreme
should not create a mode — but the constant is doing a lot of work:

| n | mean | sd | \|S\| | bimodal with threshold | without |
| --- | --- | --- | --- | --- | --- |
| 100 | 3.0 | 1.13 | 313 | 102 | 159 |
| 100 | 3.5 | 1.00 | 207 | 60 | 126 |
| 100 | 2.2 | 1.30 | 166 | 85 | 151 |
| 52 | 3.5 | 1.13 | 70 | 31 | 52 |

It reclassifies 35–52% of bimodal samples. It is also **scale-length dependent** in
a way nobody chose: on a 5-point scale a peak must hold > 20% of the mass, on a
7-point scale > 14.3%, on an 11-point scale > 9.1%. So the same underlying shape
changes classification with `k`. And in a set where *every* sample is near-uniform,
no bar clears `total/k`, so `can_be_bimodal` is `false` for a reason unrelated to
bimodality. A threshold this consequential needs to be a documented, named,
overridable parameter with a stated rationale, not a literal inside a private
function.

### 2.5 The flags largely restate the SD

Across the sweep the pattern is nearly deterministic: `can_be_unimodal` is `false`
once SD ≳ 1.4 on a 1–5 scale, `can_be_bimodal` is `false` once SD ≲ 0.9, and in the
band between them both are `true`. That is unsurprising — SD *is* a dispersion
statistic and horns is an explicitly rescaled variance (`src/lib.rs:790-819`) — but
it means the two booleans add little beyond `metrics_horns`, which is already
reported with nine summaries. The proportions, by contrast, vary meaningfully
*within* the SD band (0.224 vs 0.042 at SD 1.13 vs 1.30 in §2.1) and are worth their
place.

### 2.6 Three paths where the flags are wrong, not merely weak

**(a) Streaming always reports both flags `false`.** `6a03fc4` hardcoded
`can_be_unimodal = false; can_be_bimodal = false` inside `compute_modality()`
(`src/lib.rs:1022-1025`) on the stated assumption that the caller overrides them.
`counts_to_result_list()` does (`src/lib.rs:1254-1255`).
`write_streaming_statistics()` does not (`src/lib.rs:2090-2091`) — nothing between
that call and the struct construction at 2113 touches the flags. So every streaming
run emits `can_be_unimodal = false`, which by its own documentation means "proved
that no sample can be bell-shaped." It is asserting a proof it never attempted. A
placeholder that means "unknown" must not be spelled the same as the strongest
claim the field can make; had it defaulted to `true` the failure would have been
merely uninformative rather than actively false.

**(b) `stop_after` invalidates the `false` branch, silently.** With a limit,
`closure_parallel()` enumerates a truncated subset and passes it to the same
`counts_to_result_list()` (`src/lib.rs:1806-1811`). Over a subset, `true` survives
— a witness is a witness — but `false` no longer proves anything; it means only
"not found among the ones we kept." The API exposes no way to tell the two apart,
and the truncation itself is nondeterministic (the `found_count` atomic at
`src/lib.rs:1742-1776` races), so the flag can differ between runs on identical
input.

**(c) SPRITE reuses the field with an incompatible meaning.** `sprite.rs:438` feeds
SPRITE output through `counts_to_result_list()`, so SPRITE result lists carry
`can_be_unimodal` / `can_be_bimodal` too. But SPRITE is a randomized, unseeded,
non-exhaustive search (`rand::rng()` at `sprite.rs:680`, `sprite.rs:802`). For
SPRITE the flag means "none of the distributions we happened to generate was
unimodal" — not a proof, and not reproducible across runs. One struct field now
carries "deductively proved" for exhaustive CLOSURE and "not observed in a random
draw" for SPRITE, with nothing in the type to distinguish them. Under the
`false`-placeholder bug in (a), SPRITE-via-streaming manages both problems at once.

### 2.7 Two consistency and coverage gaps

- **Degenerate cases disagree.** `compute_modality()`'s empty branch returns
  `can_be_unimodal: true` (`src/lib.rs:987`); `empty_result_list()` returns
  `can_be_unimodal: false` (`src/lib.rs:1141`). Same situation, opposite answers.
- **The modality tables are never persisted.** `create_stats_writers()`
  (`src/lib.rs:1393`) opens writers for `metrics_main`, `metrics_horns` and
  `frequency` only, and `write_statistics_files()` (`src/lib.rs:1839-1900`) writes
  those three plus `frequency_dist`. `ModalityCounts`, `ModalityPairs` and
  `ModalityConclusion` reach no Parquet file. Five commits of inferential machinery
  are visible only to in-process Rust callers holding the returned struct — not to
  the R package, which is the consumer this is for.
- **No tests.** `src/lib.rs` has 16 `#[test]` functions and not one references
  `is_unimodal`, `is_bimodal_mean_between`, `can_be_unimodal`, or `can_be_bimodal`.
  Each of the four defects in §2.6–2.7 would be caught by a three-line test.

---

## Part 3 — The medoid frequency table (`1fe8f95`)

### 3.1 The diagnosis was right

Replacing `f_average` with an actual sample fixes a real problem: a coordinatewise
mean of count vectors is generally not itself a valid sample. It need not be
integral, and it will not in general satisfy the mean/SD constraints that define
`S` — so the row a user plots as "the CLOSURE result" was a shape CLOSURE had just
finished proving impossible. Choosing a real member of `S` is the right instinct,
and choosing it under EMD rather than L1/L2 is the right metric (§1.1).

### 3.2 But it is not a medoid

`compute_frequency_rows()` (`src/lib.rs:866-915`) computes
`argmin_{x ∈ S} EMD(x, centroid)`. A medoid is `argmin_{x ∈ S} Σ_{y ∈ S} d(x, y)`.
These differ: under the 1-D reduction the former minimises distance to the *mean*
CDF, the latter to something median-like in CDF space. They disagree in **4 of the
8** configurations tested:

| n | mean | sd | \|S\| | ties at min | matches true medoid | mean EMD to set | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 3.00 | 1.13 | 313 | 2 | yes | 25.3 | 48 |
| 100 | 3.24 | 1.30 | 647 | 1 | **no** | 27.2 | 64 |
| 100 | 3.50 | 1.00 | 207 | 1 | **no** | 25.3 | 58 |
| 52 | 3.50 | 1.13 | 70 | 1 | yes | 13.3 | 28 |
| 100 | 2.20 | 1.50 | 200 | 1 | yes | 15.9 | 36 |
| 100 | 3.00 | 0.80 | 99 | 2 | **no** | 15.0 | 36 |
| 52 | 3.00 | 1.30 | 95 | 2 | **no** | 14.7 | 30 |
| 100 | 4.10 | 1.13 | 87 | 1 | yes | 15.1 | 30 |

The irony is that the commit's own stated motivation — avoid a fictional average —
argues for the true medoid, which is the robust choice. Anchoring on the centroid
inherits the mean's sensitivity to the (typically skewed) shape of `S` while the
name promises otherwise. Either compute the real medoid (O(N²) naively, but 1-D EMD
admits an O(N·k) route via the componentwise median of the CDFs, then a nearest-real-
sample projection) or rename the column to something like `representative` and say
in the docs that it is the sample nearest the mean profile.

### 3.3 A single representative badly misrepresents the set

`mean EMD to set ≈ 25` at n = 100 means the average admissible dataset differs from
the reported representative by moving a quarter of all observations one full scale
point; the worst differs by 0.5–0.64 points per observation. Handing a user one row
labelled with the group name and no dispersion invites the reading "here is what
the data looked like" — and unlike a single number, that reading is not obviously
wrong, which is what makes it dangerous. CLOSURE *is* meant to support inference
about the shape of the unknown data (see Part 6); the objection is not to shape
claims but to a point estimate standing in for a set whose spread is this large.
The representative's own shape is also arbitrary with respect to the set: at
n = 100, mean 3.0, SD 1.13 the selected sample is unimodal while 33% of `S` is
bimodal.

Two smaller defects:

- **Ties are broken by enumeration order.** The comparison is strict `<`
  (`src/lib.rs:905`), so the first minimiser wins. Ties occur in 3 of the 8 cases
  above. Order is deterministic for a full `closure_parallel()` run, but not under
  `stop_after`, and not for SPRITE.
- **`f_relative` is now redundant.** It is exactly `f_count / n`
  (`src/lib.rs:912`), a constant rescale of its neighbour. The old pair carried two
  genuinely different quantities.

### 3.4 The `f_count` schema collision is the most dangerous item in the five commits

The column name is `f_count` in both writers, the Parquet schema is identical
(`src/lib.rs:1435`), the values are on the same scale — both sum to `n` — and the
meaning is different:

| path | `f_count` means |
| --- | --- |
| in-memory (`counts_to_result_list`, `src/lib.rs:911`) | raw count of each value **in one selected sample** |
| streaming (`write_streaming_statistics`, `src/lib.rs:2063`) | **mean** count per sample across the group — i.e. the old `f_average` |

The comment at `src/lib.rs:2044-2046` records the discrepancy honestly, but a
comment is not a guard. Two runs of the same study with different memory settings
produce byte-comparable files with incompatible semantics, and no downstream
consumer can detect which it has. Either give the streaming path a different column
name, or add a column recording which estimator produced the row, or (best) carry
both quantities as the pre-`1fe8f95` schema did.

### 3.5 Dropping `f_average` lost something real

`f_average[v] = E[count of v]` over `S` is a well-defined quantity — the mean
admissible histogram under the same uniform-over-`S` weighting the horns statistics
already use. It is not a valid sample, but it was never claimed to be, and it is the
right thing to plot as a central tendency band. The fix for "users mistake it for a
dataset" is a rename (`f_expected`) and a docstring, not deletion. A frequency table
carrying `f_expected`, `f_representative`, and per-value `[lo, hi]` from
`ModalityCounts` would communicate the shape *and* the uncertainty in one object;
today those live in three unconnected structs, one of which is never written to
disk.

---

## Part 4 — Strengths, stated plainly

1. **The falsification frame is correct and valuable.** Reasoning "no admissible
   dataset has this shape ⟹ the data did not have this shape" is the strongest kind
   of claim available here and is exactly what a forensic tool should output.
2. **`d556ef2` fixed a real unsoundness.** Per-value bounds describe a box around
   `S`, not `S`; the box has corners that are not admissible samples. Moving to an
   exact per-sample scan removed a class of false "possible" verdicts.
3. **It is free.** The scan piggybacks on the existing horns loop; both predicates
   short-circuit once satisfied (`src/lib.rs:1180-1185`).
4. **EMD is the right distance for ordinal data**, and the 1-D CDF reduction is
   exact and cheap.
5. **The commits are honest about their seams.** `6a03fc4` documents the placeholder,
   `1fe8f95` documents the streaming/in-memory divergence. The problem is that the
   comments were left to do work that the code should be doing.
6. **The naming improved.** `can_be_unimodal` over `unimodal` (`2e24383`) correctly
   signals the existential reading, even though the report shape then fails to
   exploit it.

---

## Part 5 — Avenues for improvement

### P0 — correctness, small and mechanical

1. Make `ModalityConclusion`'s shape fields non-defaultable. Return
   `Option<bool>` (or an explicit `Unknown` variant) from `compute_modality()` and
   have each caller supply a value or `None`. That converts the streaming bug in
   §2.6(a) into a compile error rather than a false proof, and gives SPRITE and
   `stop_after` somewhere honest to put "we did not enumerate everything."
2. Either compute the flags in the streaming path — `is_unimodal()` and
   `is_bimodal_mean_between()` only need the per-sample count vector, which
   `StreamingFrequencyState` already sees — or set them to `None`. The former is
   cheap and preferable; there is no reason streaming should be less informative.
3. Reconcile the two empty-case defaults (`src/lib.rs:987` vs `src/lib.rs:1141`).
4. Write the modality tables to Parquet, or delete them. Right now they are
   unreachable from the consumer they were built for.
5. Add tests: the two predicates on hand-built vectors (uniform, monotone, plateau,
   twin-peaked, single-spike-at-extreme), and one end-to-end assertion per code path
   that the flags are not silently defaulted. See Part 7 for a concrete plan — the
   full 280-cell exhaustive sweep runs in 4 s under `cargo test`.

### P1 — report proportions, and report exactly what was searched

6. Replace the two booleans with counts already available in the loop:
   `n_unimodal`, `n_bimodal`, `n_samples_scanned`, plus an `exhaustive: bool`.
   Keep `can_be_*` as derived accessors (`n_unimodal > 0`) if the R side depends on
   them, but make the counts the primary output. This single change fixes §2.1,
   §2.2's unnamed remainder (`n_scanned − n_uni − n_bim` becomes visible), §2.6(b)
   and §2.6(c) at once, and costs one `usize` increment per sample.
7. Document the uniform-over-`S` weighting wherever the proportions surface, with
   the note that `metrics_horns` already carries the same weighting.

### P2 — fix the shape taxonomy

8. Separate the two axes: **number of modes** × **mode location** (§2.3). Classify
   each sample once, per-sample, on both. This makes `can_be_bell_shaped` (one mode,
   interior) expressible — which is what the existing doc comment already claims to
   report — retires `j_shape_low` / `j_shape_high` as separately-computed fields,
   and removes the last consumer of the `count_lo`/`count_hi` box relaxation that
   `d556ef2` replaced everywhere else.
9. Emit a taxonomy that partitions, with counts per class:
   `{flat, one_mode_interior, one_mode_low_edge, one_mode_high_edge,
   two_modes, three_or_more}`. That turns the current pair of booleans into a shape
   histogram over `S`, eliminates the unnamed remainder of §2.2, and is what the
   loop is already positioned to produce.
10. Make the peak threshold explicit and scale-invariant, or replace it. `total/k`
    reclassifies up to half the set (§2.4) and varies with `k` for no chosen reason.
    Better than tuning it: report the graded unimodality distance of §2.3b
    (isotonic regression to the nearest weakly-unimodal vector, O(k²) per sample),
    which makes the threshold a visible parameter over a reported quantity rather
    than a constant inside a private function.
10b. **Add conditional bounds — the single highest-value addition here.** For each
    shape class, accumulate `count_lo`/`count_hi` over the members of that class
    (§6.2). This yields prior-free, adversary-proof statements of the form "if the
    data were bell-shaped, at least 39 responses were at the ceiling," which is the
    only tier of claim that survives the single-outlier defence. One extra pair of
    accumulators per class in the existing loop at `src/lib.rs:1174-1186`.

### P3 — fix the frequency table

11. Rename the streaming column or add an estimator tag. §3.4 is a silent
    data-integrity bug and should be fixed before anything else in this section.
12. Restore an `f_expected` column alongside the representative. Both are
    meaningful; only one was ever confusing, and the fix for that is documentation.
13. Compute the real medoid, or rename the current column. Given the commit's own
    motivation, computing it is the better half of that choice.
14. Ship dispersion with the representative: per-value `[lo, hi]` (already in
    `ModalityCounts`), or a small set of extremal representatives. The min/max-horns
    groups are a first step in this direction — generalising them beats reporting a
    single centre with no spread.
15. Break ties deterministically (e.g. lexicographic smallest among minimisers) so
    the reported representative is reproducible under `stop_after` and SPRITE.

### P4 — the framing question worth settling

16. All of the above is about the *sample* histogram. "No admissible dataset has a
    unimodal histogram" is a claim about the observed data, not about the latent
    population; with n = 100 on 5 categories the step from one to the other is
    usually short but is not free. Whatever the shape output ends up being, it
    should say which of the two it is talking about. This is a documentation
    decision, but it determines how the numbers get read in a paper.

---

## Part 6 — The single-outlier defence, and why proportions are not enough

### 6.1 The adversarial structure

The worry that motivates this section: if the tool reports "at least one admissible
dataset is unimodal," an author whose data are being questioned simply replies
"ours was that one." An existential summary is worthless to the accuser and free to
the accused. Any single atypical member of `S` defeats the inference.

This is correct, and it is a stronger argument against the boolean than the
stability argument in §2.1 — that one says the flag is *noisy*, this one says the
flag is *strategically void*. It also shows why proportions alone are only a partial
answer. "Only 0.13% of admissible datasets are unimodal" is much better, but a
determined defender has two replies: *"we were in the 0.13%"*, and — more awkward —
*"your 0.13% weights all admissible datasets equally, which is not a model of
anything."* The second objection is fair (§2.1) and cannot be argued away.

### 6.2 The defence is not blocked — it is made checkable

To be explicit up front, because this is the part that is easy to misread: **the
author can still claim their data were that one sample, and CLOSURE cannot stop
them.** Nothing here proves which dataset was real. That is not the mechanism. The
mechanism is that a *vague* claim costs nothing and a *specific* claim can be
checked against evidence outside CLOSURE.

Worked example. A paper reports satisfaction on a 1–5 scale, M = 3.50, SD = 1.50,
n = 100, and describes the responses as roughly bell-shaped. Allow a generous
rounding tolerance of 0.01 (at the 0.005 implied by two-decimal reporting there are
**no** unimodal datasets at all, and the finding is already tier 1). CLOSURE
enumerates 1539 admissible datasets. Two of them are unimodal:
`(15, 15, 15, 15, 40)` and `(15, 15, 15, 16, 39)`. A representative member of the
other 1537 looks like `(0, 49, 0, 2, 49)`.

The current output is `can_be_unimodal = true`, the author replies "ours was one of
those," and the exchange is over with nothing gained.

Now condition on the shape instead. The table below is not a bound on what the data
*are* — it reads: *if* the data were unimodal, here is what else must be true.

| quantity | over all 1539 admissible datasets | over the 2 unimodal ones |
| --- | --- | --- |
| median | 3, 3.5, or 4 | **exactly 4** |
| modal category | 2, 4, or 5 | **exactly 5** |
| % choosing 5 | 24–50 | **39–40** |
| % choosing 4 or 5 | 43–75 | **exactly 55** |
| % choosing 1 or 2 | 19–50 | **exactly 30** |
| count of each value | 0–25 / 0–50 / 0–38 / 0–51 / 24–50 | 15 / 15 / 15 / 15–16 / 39–40 |

The left column is why the mean and SD alone constrain almost nothing. The right
column is the point: "unimodal" is not a free adjective here. It is equivalent to
asserting a median of 4, a mode of 5, exactly 55% in the top two categories and
exactly 30% in the bottom two. Every one of those is the kind of number papers
routinely report — in a table, in a figure, in a "% agreeing" sentence, in a
subsequent paper on the same dataset. If any of them disagrees, the unimodality
claim is dead, and it is dead by arithmetic rather than by argument.

The analogy is an alibi. "I was somewhere else" cannot be checked. "I was at the
petrol station on Route 9 at 20:47" can still be true — but now it can be checked,
and the person has to keep it consistent with everything else known. Forcing the
move from the first to the second is the whole of what a tool like this can do.

Three things make the narrowing stronger than one table suggests:

1. **It usually forecloses the claim actually made.** Both witnesses peak at the
   ceiling, so the honest description is "responses piled up at the top of the
   scale," not "bell-shaped." A paper claiming approximate normality is already
   contradicted — deductively, tier 1, with no counterexample to point at. This is
   the §2.3 taxonomy gap doing real damage: the current field cannot express
   "bell-shaped is impossible but a ceiling effect is not," which is precisely the
   finding.
2. **It compounds across variables.** One variable requiring a knife-edge
   15/15/15/15/40 split is a curiosity. A paper with twenty such variables, each
   requiring its own knife-edge distribution to sustain the described shape, is a
   different matter entirely. Vague existential flags do not compound; specific
   commitments do.
3. **The narrowing is not an artefact of this cell.** Mean 2.2, SD 1.3, n = 100:
   the count of "1" ranges 28–52 across all of `S`, but 39–44 across the unimodal
   subset.

What does *not* help: horns ranges over `S` and over the unimodal subset are
0.552–0.563 and 0.557–0.562 — indistinguishable. The nine horns summaries already
shipped cannot see any of this. Shape information is genuinely additional, which is
the strongest argument for continuing this line of work.

**Honest limit.** If a paper reports only a mean, an SD, and an n, there may be
nothing to check the specific claim against, and the defence survives intact.
CLOSURE never settles a case by itself. It narrows the set of survivable stories
until one of them collides with something else in the record — and an existential
boolean does no narrowing at all.

### 6.3 Three tiers of claim, in decreasing strength

1. **Unconditional deductive** — "no admissible dataset is bell-shaped." Survives a
   maximally adversarial defender because there is no counterexample to point at.
   Available whenever SD is high enough, which on a 1–5 scale is common.
2. **Conditional deductive** — "*if* the data were bell-shaped, then at least 39
   responses were at the ceiling." Prior-free, immune to the single-outlier
   defence (it quantifies over the whole shape class, not one member), and it
   converts a shape question into an arithmetic one the reader can check against
   the paper. This tier is missing from the codebase and is the largest gap
   identified in this document.
3. **Measure-based** — "0.13% of admissible datasets are bell-shaped." Useful
   context, but carries the uniform-weighting commitment and can be argued with.

Report 1 and 2 as findings; report 3 as context. The current code reports none of
the three — it reports the existential, which is tier 3 collapsed to one bit.

The implementation cost is low. `ModalityCounts` already computes `count_lo` /
`count_hi` over all of `S` in a single pass; maintaining one additional pair of
accumulators per shape class inside the existing loop at `src/lib.rs:1174-1186`
yields tier 2 for every class at essentially no cost, and the per-class sample
counts yield tier 3.

One caveat that ties back to §2.6(b–c): conditional bounds are deductive **only if
the enumeration was exhaustive**. Under `stop_after`, or for SPRITE, they become
"bounds over the members we happened to generate," which is a much weaker and
easily misread statement. Whatever carries these numbers must also carry the
`exhaustive` flag recommended in P1.

---

## Part 7 — Can the exhaustive sweep be a Rust test?

Yes, and it should be. All findings in this document came from a Python
re-enumeration cross-checked against `closure_parallel()`; there is no reason that
cross-check cannot live in the repository.

### 7.1 It is fast enough

Measured on this machine, `closure_parallel()` over the full 280-cell grid
(`n ∈ {20,30,52,100} × mean ∈ {2.2,3.0,3.24,3.5,4.1} × sd ∈ {0.6,…,1.8} ×
rem ∈ {0.005,0.01}`, 156 non-empty cells, 36 083 samples total):

| build | sweep | n=200 single cell | n=400 | n=800 |
| --- | --- | --- | --- | --- |
| release | 0.55 s | 0.27 s | 6.6 s | 172 s |
| debug (`cargo test` default) | 4.0 s | 1.8 s | — | — |

Four seconds in debug for the whole grid is an ordinary test. Wider scales are also
cheap (1–7 at n=60: 75 ms release; 0–10 at n=60: 4.5 s). Keep `n ≤ 200` in the
default test and put anything larger behind `#[ignore]`.

### 7.2 Where it goes

`is_unimodal` and `is_bimodal_mean_between` are private, so unit tests belong in the
existing `mod tests` inside `src/lib.rs` (16 `#[test]` functions today, none
touching any of this). The end-to-end sweep can go in `tests/`.

### 7.3 What to assert — invariants, not golden numbers

A table of 156 expected proportions would need rewriting every time a definition
changes, and the definitions are exactly what is in flux. Properties are cheaper to
maintain and catch more:

1. **`can_be_unimodal == S.iter().any(is_unimodal)`**, asserted on every cell and
   on *every code path* (in-memory, streaming, `stop_after`, SPRITE). This single
   assertion catches §2.6(a) — the hardcoded `false` — immediately.
2. **Mutual exclusivity per sample.** A weakly unimodal vector has at most one
   strict local maximum, so no sample should satisfy both predicates. Assert over
   all 36 083 samples. If it ever fails, one of the definitions has drifted.
3. **Monotonicity in tolerance.** `S(0.005) ⊆ S(0.01) ⊆ S(0.05)`, so every
   `can_be_*` flag must be monotone non-decreasing in the rounding error, and every
   `count_lo`/`count_hi` interval must be nested. This turns the instability in
   §2.1 into an executable property, and would catch pruning bugs in the DFS.
4. **Shift invariance.** `is_bimodal_mean_between(f, s)` must not depend on `s`
   (both the mean and the peak values shift by the same constant). This is what
   makes the literal `0` at `src/lib.rs:1183` safe; today nothing records that.
5. **Cross-check `|S|` against the DP counter.** `src/count.rs` counts solutions
   with integer `i64` arithmetic on `(remaining, sum, sum_sq)` while the DFS uses
   `f64`. Asserting the two agree is the only test that would catch f64 boundary
   misses in the enumeration — everything else in this document assumes `S` is
   complete. Expect to need a small tolerance on the SD comparison, since the two
   paths round differently.
6. **Brute-force reference for small cells.** For `k = 5, n ≤ 25` there are only
   `C(n+4, 4)` count vectors (23 751 at n = 25) — enumerate all of them, filter by
   mean and SD directly, and assert set equality with CLOSURE's output. This
   validates the search itself, not just the predicates, and is the check that
   underwrites every other number.

Items 1–4 are a few lines each. Items 5–6 are the ones with real diagnostic value,
and neither exists today.

---

## Part 8 — What was implemented

All of Part 5 landed. New module `src/modality.rs`; new integration suite
`tests/shape_sweep.rs`. Full suite: 68 tests, green.

### The shape output

`ModalityConclusion`'s four booleans are gone, replaced by `ModalityShapes`:
per-class sample counts, per-class conditional count bounds, an `exhaustive`
flag, and the unimodality-deficit spread. `ShapeClass` crosses mode count with
mode location exactly as §2.3 proposed (`Flat`, `OneModeInterior`,
`OneModeLowEdge`, `OneModeHighEdge`, `TwoModes`, `ThreeOrMoreModes`), and the
classes partition — asserted over every result set in the sweep.

`can_be_*` are now methods returning `Option<bool>`: `Some(true)` on a witness,
`Some(false)` only when the search was exhaustive, `None` when a partial scan
found nothing. That is the P0-1 fix — an absence can no longer be spelled the
same way as a proof — and it makes the truncated-search and SPRITE cases
(§2.6 b–c) honest by construction rather than by convention.

`j_shape_low` / `j_shape_high` now come from the per-sample scan
(`can_be_j_shape_low` / `_high`) instead of the `count_lo`/`count_hi` box, which
removes the last consumer of the relaxation `d556ef2` replaced everywhere else.

Mode detection uses topographic prominence with an explicit threshold —
`DEFAULT_MODE_PROMINENCE = 0.05` of `n`, scale-length-invariant, unlike the old
`total/k`. Alongside it, `unimodality_deficit` grades the same question with no
threshold at all: the observations that must be removed before a histogram is
single-peaked, zero exactly for weakly unimodal vectors, cross-checked against
the textbook definition over all 330 count vectors of 7 observations on a 5-point
scale.

### The two numbers this document had wrong

Rebuilding shape detection on prominence changed the worked example in §2.1 and
§6.2, in a way that strengthens the conclusion.

The old `is_unimodal` rejected any wobble at all. At M = 3.50, SD = 1.50, n = 100
it therefore found **no** unimodal dataset at tolerance 0.005 and two at 0.01 —
the instability §2.1 is built on. But `(16, 13, 14, 19, 38)` is a ceiling effect
with three counts of noise at the bottom, not a multimodal distribution, and the
prominence rule classifies it as what it is. The corrected picture:

| tolerance | \|S\| | interior mode | ceiling mode | `can_be_bell_shaped` |
| --- | --- | --- | --- | --- |
| 0.005 | 162 | 0 | 5 | `Some(false)` |
| 0.01 | 1539 | 0 | 50 | `Some(false)` |

So the instability was itself partly an artefact of the fragile predicate. The
forensically load-bearing claim — *no admissible dataset has an interior mode* —
is identical at both tolerances, and is now stated at the tier that survives
(§6.3 tier 1) rather than as a unimodality flag that flips. §2.1's argument for
reporting proportions over booleans stands unchanged; its illustration was
sharper than the underlying measurement deserved.

The conditional bounds of §6.2 loosen correspondingly and remain decisive. At
tolerance 0.01, for the ceiling-peaked class versus all of `S`:

| quantity | all 1539 admissible datasets | the 50 ceiling-peaked ones |
| --- | --- | --- |
| count of each value | 0–25 / 0–50 / 0–38 / 0–51 / 24–50 | 14–16 / 12–17 / 12–18 / 13–22 / 37–41 |
| median | 3, 3.5, or 4 | **exactly 4** |
| % choosing 5 | 24–50 | 37–41 |
| % choosing 4 or 5 | 43–75 | 53–59 |

Both tables are pinned in `tests/shape_sweep.rs`.

### The frequency table

`f_count` is gone. Both paths now emit `f_expected` (mean count across the
group), `f_representative` (the medoid's count), and `f_relative`
(`f_expected / n`). Streaming writes `NaN` for `f_representative`, which it
cannot compute in one pass — visibly absent rather than silently redefined,
which closes the §3.4 schema collision. `f_expected` restores the quantity
§3.5 argued was lost.

The medoid is now the real one, `argmin Σ_y EMD(x, y)`, not the sample nearest
the centroid (§3.2). The naive form is `O(m²k)`; since 1-D EMD is the L1 distance
between cumulative counts and every cumulative count is an integer in `0..=n`,
one histogram pass reduces it to `O(mk + kn)`. The shortcut is tested against
the definition on real result sets. Ties break lexicographically, so the reported
sample no longer depends on enumeration order (§3.3).

### Reaching disk, and being tested

`modality_counts`, `modality_pairs`, `modality_shapes` and `modality_summary`
are now written to Parquet. Before this they were computed and dropped —
unreachable from the R package the analysis exists for.

`tests/shape_sweep.rs` implements Part 7. The whole grid runs in **5.9 s** under
`cargo test`; the `n = 400` case is `#[ignore]`d. Two of its checks found nothing
wrong but are the ones worth having:

- **Brute force.** For 45 configurations at `n ≤ 25`, an independent enumerator
  that walks every composition and filters on mean and SD returns exactly
  CLOSURE's result set. It shares no pruning logic with the DFS, so this is
  evidence about the search rather than about a shared assumption.
- **DP cross-check.** Over all 240 grid cells, `closure_count`'s integer-arithmetic
  DP agrees with the float DFS on `|S|` — the check that would catch an f64
  boundary case, and the one every other number in this document depends on.

Plus: class partitioning and conditional bounds nested inside unconditional ones
over the whole sweep; `can_be_*` re-derived from raw count vectors and compared;
tolerance monotonicity (`S(0.005) ⊆ S(0.01)`, so no class may shrink); a
truncated run never returning `Some(false)`; and an end-to-end streaming test
asserting the emitted `modality_summary.parquet` matches memory mode class for
class — the direct regression test for §2.6(a).

### Not done

Nothing from Part 5 was skipped. Two judgement calls worth flagging:

- `DEFAULT_MODE_PROMINENCE = 0.05` is a documented default, not a validated one.
  §2.4 asked for the threshold to be explicit, named and scale-invariant, which
  it now is; picking its value against a labelled set of shapes is still open,
  and `unimodality_deficit` is the threshold-free reading in the meantime.
- Removing `ModalityConclusion` and renaming `f_count` are breaking changes for
  the R side.

---

## What I would keep unchanged

The per-sample scan, its placement inside the horns loop, the short-circuiting,
EMD as the distance, the 1-D CDF reduction, and the `can_be_` naming convention.
The five commits identified the right problems and reached for the right tools.
What is missing is at the output boundary: the loop computes a rich picture of `S`
and then reports two bits of it, in three places incorrectly, to a file format that
never receives them.
