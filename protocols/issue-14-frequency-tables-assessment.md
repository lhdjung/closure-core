# Issue #14: Collect frequency tables instead of samples?

Assessment of <https://github.com/lhdjung/closure-core/issues/14> and of the R
representation proposed in the issue comment.

**Verdict in one line:** the idea is sound and the issue actually *understates*
the case — but the R mockup should be adopted only as the storage/wire format,
not as the user-facing shape of `data$results`, and it needs three fixes before
it is used as a template.

---

## Part 1 — Does the overall idea make sense?

Yes. Every load-bearing claim in the issue checks out against the code, and two
of the issue's hedges can be upgraded to certainties.

### 1.1 "No information loss" is stronger than the issue claims

The issue argues the change is lossless *because* order is deliberately ignored.
For CLOSURE-Rust the argument is stronger than that: there is no order
information to lose in the first place.

`closure_branch_recurse()` (`src/lib.rs:2736`) iterates
`for vi in min_value_idx..scale_range` and passes `vi` as the child's
`min_value_idx` (`src/lib.rs:2788`). Every emitted combination is therefore
non-decreasing **by construction**. Confirmed empirically in
`parallel_results.csv`:

```
2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,...,4
```

52 integers encoding what is really three numbers. The current sample
representation is not a richer encoding that we are agreeing to discard — it is a
strictly redundant unary encoding of the frequency vector. The map
sorted-sample → frequency-vector is a bijection.

This also disposes of the issue's worry that "the Rust and Python
implementations will sometimes arrange the integers within a sample in different
order." Rust's output is always sorted; the
[Python implementation](https://github.com/larigaldie-n/CLOSURE-Python) returns
the same samples but not necessarily in sorted order, which is exactly what
unsum's `identical_sorted_cols()` helper exists to paper over. Under the
frequency representation the two implementations become byte-comparable, the
discrepancy disappears at the source, and `identical_sorted_cols()` can be
retired in favour of plain equality.

### 1.2 The same holds for SPRITE — and more decisively

The issue says "I think the same logic would apply to SPRITE." It does, and for a
sharper reason: SPRITE **already sorts every distribution** before emitting it
(`src/sprite.rs:925`, `src/sprite.rs:1089`) and uses the sorted vector as the
hash key for its uniqueness check (`unique.insert(hashable_values)`). SPRITE's
notion of "distinct result" is *already defined on the multiset*, i.e. on the
frequency vector. Representing results as frequency tables makes the output
format agree with the deduplication semantics the algorithm already uses.

⚠️ **But the size argument does not transfer to SPRITE unconditionally.** See
§1.6.

### 1.3 The size win is exactly n/k, and it grows with n

Output per sample drops from `n` integers to `k` integers, where
`k = scale_max - scale_min + 1`. The win is `n/k`, and since `k` is fixed by the
scale while `n` is the thing that blows up, the win grows precisely where the
OOM problem lives. The issue's `n = 60`, 1–7 example gives 8.6×; the
`parallel_results.csv` case (n = 52) gives ~7.4×.

Two further wins the issue does not mention:

- **Narrower dtype.** Counts are bounded by `n`, so `n < 256` fits in `u8` and
  `n < 65536` in `u16`, against the current `Int32`. That is another 2–4× on top,
  giving ~30× for the n = 60 / 1–7 case.
- **Better Parquet compression.** `k` count columns are far more homogeneous
  column-wise than `n` position columns, which currently hold long runs that
  shift position from row to row.

### 1.4 The speculated hot-path overhead is not merely unproven — it is backwards

The issue calls the overhead "speculation ... would likely just shift work around."
It is better than that: **the frequency vector is already computed for every
single sample, then thrown away.**

| Location | What it does per sample |
|---|---|
| `src/lib.rs:1040-1046` | builds `freqs` for horns + modality checks |
| `src/lib.rs:2328-2335` | builds `freqs` **and** a `sample_freq` HashMap (streaming) |
| `src/sprite.rs:934-949` | builds `freqs` **and** a `sample_freq` HashMap |
| unsum `R/plot-bar-basic.R:173` | `tabulate(samp - scale_min_val + 1L, ...)` — recomputes in R what Rust already computed and discarded |

So the change **removes** work. The streaming path currently builds two
redundant frequency structures per sample; both would collapse into the single
representation that is now the output.

### 1.5 It can be maintained incrementally in the DFS for free

Better still, the frequency vector need not be computed *after* the fact. The
recursion can carry a `counts: [u32; k]` array, mirroring the existing
push/pop discipline:

```rust
counts[vi] += 1;
closure_branch_recurse(counts, next_sum, next_m2, vi, ctx, results);
counts[vi] -= 1;
```

`combo` is only ever read at the emission site (`combo.clone()`,
`src/lib.rs:2753`) — `running_sum` and `running_m2` are threaded separately — so
the `Vec<U>` can be replaced outright rather than maintained alongside. The one
allocation in the hot leaf then clones `k` values instead of `n`.

**Net effect on the hot path: cheaper by a factor of n/k, not more expensive.**

### 1.6 The one precondition: `k < n`

The size win is `n/k`, so it inverts whenever the value grid is wider than the
sample. This is a property of the grid, **not** of which technique is running —
the right way to state it is:

> Store counts when `k < n`; otherwise store samples.

What differs between the two techniques is only whether they can *reach* the bad
regime.

**CLOSURE cannot, today.** It hard-rejects multi-item scales at both entry
points (`src/lib.rs:1546`, `src/lib.rs:2053`):

```rust
if items != 1 {
    return Err(ParameterError::InputValidation(
        "CLOSURE requires items == 1".to_string(),
    ));
}
```

`items` is present in the signature only for parity with the shared `Technique`
trait (`src/lib.rs:75`); `Closure::run` threads it through to be rejected. The
restriction is structural rather than incidental: the DFS walks a fixed
consecutive-integer grid (`value_as_u[vi]`, stepped by `U::one()`), and the
`count.rs` DP assumes the same integer lattice for its `(remaining, sum, sum_sq)`
state. A multi-item mean grid is rational, not integer, so supporting it would
mean reworking both. So for CLOSURE, `k = scale_max - scale_min + 1` always, and
`k < n` holds in every realistic case.

**SPRITE can.** `src/sprite.rs:173-192` builds possible values as
`scale_min + (i-1)/items` across the range, giving

```
k_sprite = (scale_max - scale_min) * items + 1
```

- `items = 1` (plain Likert), 1–7 scale → k = 7. Same win as CLOSURE.
- `items = 5`, 1–7 scale → k = 31. Win only if n > 31.
- `items = 20`, 1–7 scale → k = 121. **A pessimization** for typical n.

So the issue's "the same logic would apply to SPRITE" is right about
information and right about CLOSURE-equivalent (`items = 1`) SPRITE runs; it just
needs the `k < n` guard for multi-item runs.

**Forward-looking:** if CLOSURE ever gains multi-item support, it inherits this
caveat unchanged. Better to implement the guard as a grid-width check computed
from `scale_min`, `scale_max`, and `items` in shared code than to special-case it
per technique — that way CLOSURE is covered for free if the `items == 1`
restriction is ever lifted.

(Separately, SPRITE's internal `freqs` vector is indexed on a fixed 0.01 grid —
`src/sprite.rs:200`, `:244`, `:880` — which is wider than `k_sprite`. If frequency
output is added to SPRITE, index it on the actual `poss_values` grid, not the
0.01 grid.)

### 1.7 Real costs

- **Breaking format change** in the Parquet output and in `data$results`. unsum
  hard-checks `identical(names(data$results), c("id", "sample", "horns"))` at
  `R/read-write-basic.R:28`. Needs a format-version marker plus either migration
  or a compat reader.
- **Cross-implementation interchange.** `R/utils.R:811-816` documents that the
  "n"-column wide format is shared with "closure-core's test harness or the
  original Python implementation." Changing it is a coordinated version bump
  across closure-core, unsum, and the Python implementation, not a unilateral
  one. Keep a documented converter in both directions so cross-validation tests
  survive.
- **`n` becomes implicit.** Currently `n == length(sample)` is self-evident;
  with counts it is `sum(counts)`. Harmless — `n` is already in `data$inputs` —
  but worth asserting on read.
- **Requires a dense, consecutive-integer scale.** Already true for CLOSURE.

**Recommendation: do it for CLOSURE.** The technical case is one-sided; the real
work is format-compatibility coordination, not algorithmics.

---

## Part 2 — Should unsum represent the tables like the R mockup?

**The mockup's target shape is right. Its construction path is wrong, and it
should not become the user-facing shape of `data$results`.**

### 2.1 What the mockup gets right

The final shape — one row per sample, one column per scale value — is correct:

```
#> # A tibble: 3 × 7
#>      v1    v2    v3    v4    v5    v6    v7
#>   <int> <int> <int> <int> <int> <int> <int>
#> 1     7    20    20    16    11    22     4
```

It is rectangular (so dplyr/tidyr/`write.csv` work naturally), it drops the
list-column overhead of the current `sample` representation, and it is
row-appendable, which the streaming writer needs. It is also directly analogous
to the existing per-position layout in `create_samples_writer()`
(`src/lib.rs:1196`), so the writer changes are mechanical.

### 2.2 Three fixes needed before using it as a template

**(a) `table()` is a latent bug — use `tabulate()`.**

`as.integer(table(round(runif(100, 1, 7))))` returns one element per *observed*
distinct value. The mockup yields length-7 vectors only by luck: with 100 uniform
draws over 7 values, all 7 happen to appear. Real CLOSURE samples routinely have
zero counts — the first row of `parallel_results.csv` contains only 2s, 3s and
4s. With `table()`, such samples silently produce short vectors that either error
in `tibble()` or, worse, get recycled.

The correct primitive is the one unsum already uses at `R/plot-bar-basic.R:173`:

```r
tabulate(samp - scale_min + 1L, nbins = n_scale_vals)
```

**(b) The `t()` round-trip must not exist.**

The mockup builds columns-as-samples and then transposes. This is the single most
important thing *not* to carry over. unsum already pays this cost today:
`R/read-write-basic.R:293-322` transposes on read via `t()`, wrapped in a
`tryCatch` whose error message tells users to fall back to
`include = "stats_and_horns"` or `"stats_only"` when memory runs out. `t()`
materializes a full dense copy, so it is a documented memory hazard.

Rust should emit **rows-as-samples directly** — one Arrow column per scale value,
appended row by row — so unsum reads with no transposition at all. Reproducing
the mockup's transpose would forfeit much of the benefit the issue is chasing.
`as_wide_n_tibble()` (`R/utils.R:818`) then collapses to something far simpler
and faster.

**(c) Name columns by scale value, not position.**

`paste0("v", seq_along(x))` gives `v1..v7`, which coincides with the scale only
when `scale_min == 1`. On a 0–10 or 2–6 scale the labels are actively
misleading. Either name by actual value (`v0`…`v10`) or keep positional names and
document them as positional, leaning on `scale_min`/`scale_max` in
`data$inputs`. Value-based names are self-describing and cost nothing —
recommended.

### 2.3 The deeper question: should users see this shape?

Adopting counts as the user-facing `data$results` would replace the `sample`
list-column with `id` + `k` count columns + `horns`. Arguments both ways:

**For:** rectangular and tidyverse-friendly; much leaner than an R list of `m`
integer vectors; and it is *already* what the plotting code wants —
`plot-bar-basic.R` tabulates each sample back into counts anyway.

**Against:** `data$results$sample` is user-facing and documented; the package
Description promises to reconstruct "raw data", and a count matrix reads as a
summary of a sample even though it is information-equivalent; and the ECDF path
(`R/plot-ecdf-basic.R:164`) currently just `unlist()`s samples into a flat value
vector.

**Recommended split:**

1. **Storage/wire format (Rust → Parquet → unsum): adopt the count format.**
   This is where the OOM and disk wins are, and it is internal, so the
   compatibility cost is bounded by a format-version bump.
2. **User-facing `data$results`: keep `sample` available, reconstructed on
   demand.** Reconstruction is exact and cheap — `rep(scale_vals, counts)` per
   row yields the sorted sample, which is the only sample the algorithm ever
   produced. Expose the choice via a `closure_read()` / `closure_generate()`
   argument (a `format = c("counts", "samples")` switch, or a new `include`
   variant), defaulting to counts for large result sets.

This keeps the memory win where it actually bites — on disk, and in the `t()`
hazard on read — without breaking anyone's `data$results$sample` code.

3. **Change the two internal consumers to read counts directly**, otherwise they
   will expand and then re-summarize:
   - `plot-bar-basic.R:173` — stop calling `tabulate()`; the counts are the input.
   - `plot-ecdf-basic.R:164` — build the ECDF from cumulative counts. An ECDF
     *is* a cumulative frequency table, so this is strictly better than
     expanding to `values_all` elements first.

---

## Summary

| Question | Answer |
|---|---|
| Does the idea make sense? | **Yes** — losslessness is provable, not conventional (§1.1–1.2) |
| Is the hot-path overhead real? | **No** — frequency vectors are already computed and discarded; the change removes work (§1.4–1.5) |
| Size win | `n/k`, ~30× for n=60 / 1–7 with dtype narrowing (§1.3) |
| Precondition | `k < n`. Always true for CLOSURE (`items == 1` is enforced); can fail for multi-item SPRITE (§1.6) |
| Main cost | Format compatibility with unsum + the Python implementation (§1.7) |
| Is the R mockup's shape right? | **Yes** for storage; fix `table()`→`tabulate()`, drop `t()`, name by scale value (§2.1–2.2) |
| Should it be the user-facing shape? | **Not by default** — store counts, reconstruct `sample` on demand (§2.3) |

### Verification notes

Claims above were checked against `src/lib.rs`, `src/sprite.rs`,
`parallel_results.csv`, and unsum at `/Users/lukasjung/r_projects/packages/unsum`
(v0.2.0.9000). The behaviour of the Python implementation (same samples, not
necessarily sorted) is per the package author rather than inspected here. Not
verified: whether `data$results$sample` appears in unsum's vignettes or NEWS as a
compatibility commitment beyond its use in `R/` source.

---

## Addendum — what the implementation found (2026-08-12)

Implemented in closure-core as format version 2. Two of the assessment's claims
did not survive contact with the code.

### The disk win is ~1.3×, not 8.6× or 30×

§1.3 reasoned from raw element counts: `n` integers per sample become `k`, so
disk should shrink by `n/k`, more with a narrower dtype. Measured on
`mean = 3.9, sd = 1.4, n = 60`, scale 1–7 — 234,860 samples, the assessment's own
example case:

| Layout | Bytes |
|---|---|
| `counts.parquet` (7 count columns + horns) | 758,038 |
| `sample.parquet` + `horns.parquet` (60 position columns + horns) | 1,006,916 |

Counts alone against positions alone is 520 KB vs 768 KB, so **1.5× on the
sample data and 1.3× on the directory**.

The reasoning missed that Parquet never stored `n` integers per sample in the
first place. CLOSURE emits samples in sorted DFS order, so each `posN` column is
nearly constant down long runs, and dictionary + RLE encoding had already
squeezed out most of the redundancy the count representation removes explicitly.
§1.3's "better Parquet compression" bullet has it backwards: the *position*
layout is the one that compresses well, precisely because of the sortedness that
makes the counts representation lossless.

Enabling a compression codec would shrink both layouts; the writers currently
use none.

### The memory win is real and is the actual payoff

`m` heap-allocated `Vec<U>` of length `n` become one flat `Vec<u32>` of `m * k`.
For the run above that is 234,860 × (24-byte header + 240 bytes) ≈ **62 MB**
against 234,860 × 28 ≈ **6.6 MB**, a factor of 9.4 (arithmetic, not measured).
That is where the OOM problem in the issue lives, so the change is worth making
— just for a different reason than §1.3 gave.

### An `id` column would have cost more than the counts saved

The first cut wrote `id` as `Float64` into `counts.parquet`, matching the old
`results.parquet`. At 234,860 rows that one monotone column was 2.15 MB — 74 % of
the file, and on its own more than the entire position layout. It is pure
redundancy with row order, so it is not written. Rows are ids.

### Deviations from the assessment's recommendations

- **§1.6's `k < n` guard is not implemented.** SPRITE stores counts
  unconditionally, including multi-item grids where `k > n`. SPRITE result sets
  are small because SPRITE is not exhaustive, so the pessimization is bounded,
  and one unconditional format is worth more than a per-run branch. Enforced by
  `SampleFormat`, whose format-touching methods are all provided methods.
- **Grid resolution changed for multi-item SPRITE.** Memory mode used to bin to
  whole scale points and streaming mode to a fixed 0.01 grid, so the two
  disagreed. Both now use the real `poss_values` grid, as §1.6's parenthetical
  recommended. Horns is unchanged at `items == 1` and more accurate above it.
- **`value` columns are `Float64`, not `Int32`**, in `frequency`,
  `frequency_dist`, `modality_counts` and `modality_pairs` — a multi-item grid
  has fractional values.
- **Column names follow §2.2(c)**: `v1`…`v7` by scale value, `v1_2` for
  fractions, `vn2` for negatives, with `scale_values.parquet` as the
  authoritative key so nothing has to parse a name.
