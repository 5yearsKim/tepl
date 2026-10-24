#!/usr/bin/env bash
set -euo pipefail
tepl="$1"
source="$2"
explicit_source="$3"
work="${TEST_TMPDIR:?}/where-suggestions"
mkdir -p "$work"
# Suggestions do not make an unproved rule acceptable.
if "$tepl" check "$source" --semantic-policy=algebraic --suggest-where \
    --rule remove_output_permutation > "$work/check.txt"; then
    exit 1
fi
grep -Fq 'status=unknown' "$work/check.txt"
grep -Fq 'suggestion: verified sufficient restriction' "$work/check.txt"
grep -Fq 'range(len(source.attrs.permutation))' "$work/check.txt"
if "$tepl" compile "$source" --semantic-policy=algebraic -o "$work/program.json" \
    > "$work/compile.log" 2>&1; then exit 1; fi
# The removed flag cannot silently insert conditions into a rule.
for command in check compile rewrite; do
    if "$tepl" "$command" "$source" --semantic-policy=algebraic --infer-applicability \
        > "$work/retired.log" 2>&1; then exit 1; fi
done
# Materialize the reviewable source suggestion, then use ordinary compilation.
dialect="$(realpath "$(dirname "$source")/dialects/tensor.tepl")"
cat > "$work/suggested.tepl" <<TEPL
from "$dialect" import TensorLang as t;
use t::{dot, transpose};
TEPL
awk '/^rule / {printing=1} printing {print} printing && /^}/ {exit}' \
    "$work/check.txt" >> "$work/suggested.tepl"
"$tepl" compile "$work/suggested.tepl" --semantic-policy=algebraic -o "$work/program.json"
"$tepl" validate "$work/program.json"
# Explicit feasibility formulas still require every written precondition.
"$tepl" compile "$explicit_source" --semantic-policy=algebraic -o "$work/explicit.json"
"$tepl" rewrite "$work/explicit.json" --graph feasible -o "$work/feasible.tepl" 2> "$work/feasible.log"
grep -q 'Rewrites: 1; stop: fixed_point' "$work/feasible.log"
"$tepl" rewrite "$work/explicit.json" --graph infeasible -o "$work/infeasible.tepl" 2> "$work/infeasible.log"
grep -q 'Rewrites: 0; stop: fixed_point' "$work/infeasible.log"
