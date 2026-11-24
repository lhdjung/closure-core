#!/usr/bin/env bash
# Time CLOSURE and SPRITE on two git revisions and print the speedup and the
# change in peak heap use per case.
#
# Usage: scripts/compare_versions.sh <old_ref> [<new_ref>]   # new_ref defaults to HEAD
#
# Both revisions are measured with the benchmark source from the *working copy*,
# so an older revision only has to be API-compatible with today's harness, not
# to contain the benchmark itself. Before 490a6c3 the entry points took no
# trailing `prominence` argument; for such revisions the lines of
# examples/benchmark_baseline.rs ending in `// prominence` are dropped.
set -euo pipefail

old=${1:?usage: $0 <old_ref> [<new_ref>]}
new=${2:-HEAD}
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'git -C "$repo" worktree remove --force "$tmp/wt" 2>/dev/null || true; rm -rf "$tmp"; git -C "$repo" worktree prune' EXIT

run() { # <ref> <outfile>
    echo "==> benchmarking $1" >&2
    git -C "$repo" worktree add --detach --force "$tmp/wt" "$1" >/dev/null
    rm -rf "$tmp/wt/examples"
    cp -R "$repo/examples" "$tmp/wt/examples"
    if ! grep -rq 'prominence: Option<ProminenceThresholds>' "$tmp/wt/src"; then
        bench="$tmp/wt/examples/benchmark_baseline.rs"
        grep -v '// prominence$' "$bench" >"$bench.tmp"
        mv "$bench.tmp" "$bench"
    fi
    # Shared target dir so the dependencies are only built once.
    CARGO_TARGET_DIR="$tmp/target" cargo run --release --quiet --manifest-path "$tmp/wt/Cargo.toml" \
        --example benchmark_baseline >"$2"
    git -C "$repo" worktree remove --force "$tmp/wt"
    # Labels are the join key between the two runs.
    dups=$(cut -f1 "$2" | sort | uniq -d)
    if [ -n "$dups" ]; then
        echo "error: duplicate case labels in examples/common/mod.rs:" >&2
        echo "$dups" >&2
        exit 1
    fi
}

run "$old" "$tmp/old.tsv"
run "$new" "$tmp/new.tsv"

# One table per metric: <closure column> <sprite column> <decimals>.
table() {
    awk -F'\t' -v old="$(git rev-parse --short "$old")" -v new="$(git rev-parse --short "$new")" \
        -v cc="$1" -v sc="$2" -v d="$3" '
        function ratio(a, b) { return b == 0 ? sprintf("%9s", "-") : sprintf("%8.2fx", a / b) }
        NR == FNR { if (FNR > 1) { c[$1] = $cc; s[$1] = $sc } ; next }
        FNR == 1 {
            printf "%-15s %11s %11s %9s %11s %11s %9s\n", \
                "Case", "clo " old, "clo " new, "ratio", "spr " old, "spr " new, "ratio"
            next
        }
        {
            printf "%-15s %11." d "f %11." d "f %s %11." d "f %11." d "f %s\n", \
                $1, c[$1], $cc, ratio(c[$1], $cc), s[$1], $sc, ratio(s[$1], $sc)
        }
    ' "$tmp/old.tsv" "$tmp/new.tsv"
}

echo "Time (ms, median of 3)"
table 3 5 3
echo
echo "Peak heap (KiB above what was live before the call, median of 3)"
table 6 7 1
echo "(ratio = old / new, so > 1 means $new is faster or leaner; - where new is 0)"
