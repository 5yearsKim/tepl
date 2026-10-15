#!/usr/bin/env bash
set -euo pipefail
tepl="$1"
source="$2"
schema="$3"
project="$(dirname "$(dirname "$source")")"
rules="$project/rules"
work="${TEST_TMPDIR:-/tmp}/tepl-rewrite-cli"
mkdir -p "$work"
"$tepl" compile "$source" --rules "$rules" -o "$work/program.json"
"$tepl" validate "$work/program.json" > "$work/validation.txt"
"$tepl" schema -o "$work/schema.json"
cmp "$schema" "$work/schema.json"
"$tepl" rewrite "$source" --rules "$rules" --trace -o "$work/direct.tepl" 2> "$work/direct.log"
"$tepl" rewrite "$work/program.json" --trace -o "$work/loaded.tepl" 2> "$work/loaded.log"
cmp "$work/direct.tepl" "$work/loaded.tepl"
grep -Fq "Successfully applied rule 'compose_transposes'." "$work/loaded.log"
grep -Fq "Successfully applied rule 'remove_identity'." "$work/loaded.log"
grep -Fq "Rewrite completed. Result saved to $work/loaded.tepl." "$work/loaded.log"
grep -q 'Rewrites: 2; stop: fixed_point' "$work/loaded.log"
grep -q 'let shared = (d0.add X X);' "$work/loaded.tepl"
"$tepl" check "$work/loaded.tepl" > "$work/checked.txt"
set +e
"$tepl" rewrite "$work/program.json" --max-steps 0 -o "$work/limited.tepl" 2> "$work/limited.log"
status=$?
set -e
test "$status" -eq 3
grep -q 'Rewrites: 0; stop: step_limit' "$work/limited.log"
grep -Fq "Rewrite stopped (step_limit). Result saved to $work/limited.tepl." "$work/limited.log"
"$tepl" check "$work/limited.tepl" > "$work/limited-check.txt"

# File, recursive rule-directory, and project inputs preserve the same scopes.
"$tepl" rewrite "$source" --rules "$rules/transpose.tepl" -o "$work/file.tepl" 2> "$work/file.log"
"$tepl" rewrite "$project" -o "$work/project.tepl" 2> "$work/project.log"
"$tepl" rewrite "$project/graphs" --rules "$rules" -o "$work/graphs.tepl" 2> "$work/graphs.log"
cmp "$work/direct.tepl" "$work/file.tepl"
cmp "$work/direct.tepl" "$work/project.tepl"
cmp "$work/direct.tepl" "$work/graphs.tepl"
"$tepl" rewrite "$source" -o "$work/no-rules.tepl" 2> "$work/no-rules.log"
grep -q 'Rewrites: 0; stop: fixed_point' "$work/no-rules.log"

# Rule selection works for source and compiled JSON, and does not run siblings.
"$tepl" rewrite "$source" --rules "$rules" --rule compose_transposes --trace -o "$work/one-rule.tepl" 2> "$work/one-rule.log"
grep -q 'Rewrites: 1; stop: fixed_point' "$work/one-rule.log"
"$tepl" rewrite "$work/program.json" --rule compose_transposes -o "$work/one-loaded.tepl" 2> "$work/one-loaded.log"
cmp "$work/one-rule.tepl" "$work/one-loaded.tepl"
"$tepl" check "$work/one-rule.tepl" > "$work/one-rule-check.txt"
"$tepl" rewrite "$source" --rules "$rules" --rule remove_identity -o "$work/identity-only.tepl" 2> "$work/identity-only.log"
grep -q 'Rewrites: 0; stop: fixed_point' "$work/identity-only.log"
if grep -q 'Successfully applied rule' "$work/identity-only.log"; then
  exit 1
fi

"$tepl" rewrite "$source" --rules "$rules" --trace > "$work/stdout.tepl" 2> "$work/stdout.log"
cmp "$work/direct.tepl" "$work/stdout.tepl"
grep -Fq 'Rewrite completed. Result written to stdout.' "$work/stdout.log"

if "$tepl" rewrite "$source" --rules "$rules" --rule missing -o "$work/missing.tepl" 2> "$work/missing.log"; then
  exit 1
fi
grep -q "unknown rule 'missing'" "$work/missing.log"
if "$tepl" rewrite "$work/program.json" --rules "$rules" -o "$work/invalid.tepl" 2> "$work/invalid.log"; then
  exit 1
fi
grep -q -- '--rules requires TEPL source input' "$work/invalid.log"
mkdir -p "$work/empty"
if "$tepl" rewrite "$source" --rules "$work/empty" -o "$work/empty.tepl" 2> "$work/empty.log"; then
  exit 1
fi
grep -q 'no .tepl files found' "$work/empty.log"

# Explicit --rules overrides the project's default rule directory selection.
cp -R "$project" "$work/selected-project"
printf 'invalid syntax\n' > "$work/selected-project/rules/unselected.tepl"
"$tepl" rewrite "$work/selected-project" --rules "$work/selected-project/rules/transpose.tepl" -o "$work/selected.tepl" 2> "$work/selected.log"
grep -q 'Rewrites: 2; stop: fixed_point' "$work/selected.log"
"$tepl" check "$work/selected.tepl" > "$work/selected-check.txt"
