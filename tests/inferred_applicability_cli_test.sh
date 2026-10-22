#!/usr/bin/env bash
set -euo pipefail
tepl="$1"
source="$2"
explicit_source="$3"
work="${TEST_TMPDIR:?}/inferred-applicability"
mkdir -p "$work"

"$tepl" check "$source" --semantic-policy=algebraic --infer-applicability \
    --rule remove_output_permutation > "$work/check.txt"
grep -Fq 'explicit where feasibility constraints' "$work/check.txt"
grep -Fq 'suggested-rule:' "$work/check.txt"
grep -Fq 'subsequence(' "$work/check.txt"
grep -Fq 'inference-program: synthesize' "$work/check.txt"
"$tepl" compile "$source" --semantic-policy=algebraic --infer-applicability \
    -o "$work/program.json"
"$tepl" validate "$work/program.json" > "$work/validate.txt"
"$tepl" rewrite "$source" --semantic-policy=algebraic --infer-applicability \
    --graph feasible -o "$work/direct.tepl" 2> "$work/direct.log"
"$tepl" rewrite "$work/program.json" --graph feasible \
    -o "$work/loaded.tepl" 2> "$work/loaded.log"
cmp "$work/direct.tepl" "$work/loaded.tepl"
grep -q 'Rewrites: 1; stop: fixed_point' "$work/loaded.log"
"$tepl" rewrite "$work/program.json" --graph infeasible \
    -o "$work/skipped.tepl" 2> "$work/skipped.log"
grep -q 'Rewrites: 0; stop: fixed_point' "$work/skipped.log"

if "$tepl" check "$source" --semantic-policy=algebraic > "$work/ordinary.log" 2>&1; then
    exit 1
fi
grep -q 'explicit where condition' "$work/ordinary.log"

# The user-visible example follows the explicit where path without a flag.
"$tepl" compile "$explicit_source" --semantic-policy=algebraic \
    -o "$work/explicit.json"
"$tepl" rewrite "$work/explicit.json" --graph feasible \
    -o "$work/explicit.tepl" 2> "$work/explicit.log"
grep -q 'Rewrites: 1; stop: fixed_point' "$work/explicit.log"
"$tepl" rewrite "$work/explicit.json" --graph infeasible \
    -o "$work/explicit-skipped.tepl" 2> "$work/explicit-skipped.log"
grep -q 'Rewrites: 0; stop: fixed_point' "$work/explicit-skipped.log"
