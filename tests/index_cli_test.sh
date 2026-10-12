#!/usr/bin/env bash
set -euo pipefail
compiler="$1"
rules="$2"
symbolic="$($compiler check "$rules" --semantic-policy=algebraic --index-rule distribute)"
[[ "$symbolic" == *"status=proved"* ]]
[[ "$symbolic" == *"lhs_contracting_dimensions = @d.lhs_contracting_dimensions"* ]]
[[ "$symbolic" == *"additional-guard: true"* ]]
[[ "$symbolic" == *"lhs-normalized: Reduce"* ]]
if "$compiler" check "$rules" --semantic-policy=algebraic --index-rule associate > /dev/null; then
  exit 1
fi
if "$compiler" check "$rules" --index-rule distribute > /dev/null; then
  exit 1
fi
if "$compiler" check "$rules" --index-rule missing > /dev/null; then
  exit 1
fi
ast="$($compiler parse "$rules" --ast)"
[[ "$ast" == *"bind K"* ]]

# Reproduce bazel run's changed working directory and invocation-directory env.
invocation_directory="$PWD"
compiler_absolute="$(realpath "$compiler")"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT
(
  cd "$temporary_directory"
  BUILD_WORKING_DIRECTORY="$invocation_directory" "$compiler_absolute" \
    parse "$rules" --ast > /dev/null
)
