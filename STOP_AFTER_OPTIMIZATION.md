# stop_after Optimization

## Problem

When using `stop_after` with small limits (especially `stop_after = 1`), performance was unexpectedly poor:
- Only 50% time reduction compared to full search
- In some benchmarks with many iterations, actually 3x **slower** than full search

## Root Causes

1. **Parallel overhead without benefit**: Rayon spins up worker threads and processes multiple initial combinations in parallel, but most work is discarded when the limit is reached
2. **Atomic counter contention**: Frequent atomic loads/stores in hot paths cause cache line ping-ponging between CPU cores
3. **No branch-level early exit**: `dfs_branch()` computes ALL valid combinations from each initial pair, even if only 1 is needed
4. **Post-computation overhead**: Statistics computation (horns calculations, min/max search, frequency tables) runs on all results regardless of size

## Solution: Fast Path for Small Limits

Added a sequential fast path that activates when `stop_after ≤ 100`:

### In `dfs_parallel()`:
```rust
if let Some(limit) = stop_after {
    if limit <= 100 {
        // Sequential processing - no parallel overhead
        let mut found = Vec::with_capacity(limit);
        for (combo, running_sum, running_m2) in combinations {
            if found.len() >= limit {
                break;  // Early exit
            }
            let branch_results = dfs_branch(/*...*/);
            let remaining = limit - found.len();
            found.extend(branch_results.into_iter().take(remaining));
        }
        return found;
    }
}
// ... parallel path for larger limits or unlimited search
```

### Benefits:
- No thread spawning overhead
- No atomic counter contention
- Early termination at the outer loop level
- Simpler code path with less synchronization

### Benchmark Results (n=100, ~25K total samples):

**After sequential fast path only:**
- `stop_after=1`: 1.11x faster than full search (153ms → 138ms)
- `stop_after=10`: 1.13x faster than full search (153ms → 135ms)
- `stop_after=100`: 1.12x faster than full search (153ms → 137ms)

**After adding branch-level early exit:**
- `stop_after=1`: **133.95x faster** than full search (153ms → 1.1ms) 🚀
- `stop_after=10`: **110.03x faster** than full search (153ms → 1.4ms) 🚀
- `stop_after=100`: **88.53x faster** than full search (153ms → 1.7ms) 🚀
- `stop_after=1000`: **23.95x faster** than full search (153ms → 6.4ms) 🚀

The dramatic improvement comes from `dfs_branch()` stopping immediately when it finds enough results, rather than exhaustively searching each branch.

## Implementation Details

### 1. Sequential Fast Path (stop_after ≤ 100)
Both `dfs_parallel()` and `dfs_parallel_streaming()` use sequential processing for small limits to avoid parallel overhead.

### 2. Branch-Level Early Exit
`dfs_branch()` now accepts a `stop_after` parameter:
```rust
fn dfs_branch(..., stop_after: Option<usize>) -> Vec<Vec<U>> {
    let limit = stop_after.unwrap_or(usize::MAX);

    while let Some(current) = stack.pop_back() {
        if current.values.len() >= n {
            if current_std >= sd_lower {
                results.push(current.values);
                if results.len() >= limit {
                    return results;  // Immediate exit!
                }
            }
        }
        // ...
    }
}
```

This eliminates the wasteful computation of results that would be discarded anyway.

## When to Use stop_after

- **Tiny problems**: If full search completes in < 1ms, `stop_after` adds overhead without benefit
- **Small/medium problems**: 10-100x speedup when you only need a few samples
- **Large problems**: Even with high limits (1000+), still 20-30x faster than full search
