#!/usr/bin/env bash
# Time CLOSURE and SPRITE on two git revisions and print the speedup per case.
#
# Usage: scripts/compare_versions.sh <old_ref> [<new_ref>]   # new_ref defaults to HEAD
#
# Both revisions are measured with the benchmark source from the *working copy*,
# so an older revision only has to be API-compatible with today's harness, not
# to contain the benchmark itself.
set -euo pipefail

old=${1:?usage: $0 <old_ref> [<new_ref>]}
new=${2:-HEAD}
repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'git -C "$repo" worktree remove --force "$tmp/wt" 2>/dev/null || true; rm -rf "$tmp"' EXIT

run() { # <ref> <outfile>
    echo "==> benchmarking $1" >&2
    git -C "$repo" worktree add --detach --force "$tmp/wt" "$1" >/dev/null
    rm -rf "$tmp/wt/examples"
    cp -R "$repo/examples" "$tmp/wt/examples"
    # Shared target dir so the dependencies are only built once.
    CARGO_TARGET_DIR="$tmp/target" cargo run --release --quiet --manifest-path "$tmp/wt/Cargo.toml" \
        --example benchmark_baseline >"$2"
    git -C "$repo" worktree remove --force "$tmp/wt"
}

run "$old" "$tmp/old.tsv"
run "$new" "$tmp/new.tsv"

awk -F'\t' -v old="$(git rev-parse --short "$old")" -v new="$(git rev-parse --short "$new")" '
    NR == FNR { if (FNR > 1) { c[$1] = $3; s[$1] = $5 } ; next }
    FNR == 1 {
        printf "%-15s %11s %11s %9s %11s %11s %9s\n", \
            "Case", "clo " old, "clo " new, "speedup", "spr " old, "spr " new, "speedup"
        next
    }
    {
        printf "%-15s %11.3f %11.3f %8.2fx %11.3f %11.3f %8.2fx\n", \
            $1, c[$1], $3, c[$1] / $3, s[$1], $5, s[$1] / $5
    }
' "$tmp/old.tsv" "$tmp/new.tsv"
echo "(times in ms, median of 3; speedup > 1 means $new is faster)"
