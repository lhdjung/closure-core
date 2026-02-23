⏺ Now I have the full picture. Here's the analysis:

  The depth formula on master:
  let depth = (n_usize / 10).clamp(2, 15.min(n_usize - 1));

  For your benchmark (n=60, scale 1–10, range_size=10):
  - Old (hardcoded depth=2): C(11, 2) = 55 initial work items
  - New (depth=6): C(15, 6) = 5,005 initial work items

  That's a 91x increase in initial combinations. Crucially, generate_initial_combinations generates all sorted combinations at that depth without any pruning — it doesn't check mean/SD feasibility. The
   Python version this was ported from likely pruned during generation, making deeper depth beneficial. Without that pruning, deeper depth just creates massive overhead.

  Proposed fix — revert to depth = 2 in both closure_parallel and closure_parallel_streaming. With range_size=10 and depth=2, you get 55 parallel work items, which is plenty to saturate typical core
  counts. The two lines to change in closure-core/src/lib.rs:

  // Line ~1173 (closure_parallel) and ~1653 (closure_parallel_streaming)
  // FROM:
  let depth = (n_usize / 10).clamp(2, 15.min(n_usize - 1));
  // TO:
  let depth = 2;

  If you want to keep the adaptive behavior for future use (once pruning during generation is added), you could also leave the formula but cap it at 2 for now with a comment explaining why.

  Want me to clone closure-core and make this change directly?
