#!/usr/bin/env bash
set -euo pipefail
compiler="$1"
rules="$2"
symbolic="$($compiler check "$rules" --semantic-policy=algebraic --index-rule distribute)"
[[ "$symbolic" == *"status=proved"* ]]
[[ "$symbolic" == *"lhs_contracting_dimensions = @d.lhs_contracting_dimensions"* ]]
[[ "$symbolic" == *"additional-guard: true"* ]]
[[ "$symbolic" == *"lhs-normalized: Reduce"* ]]
guarded="$($compiler check "$rules" --semantic-policy=algebraic --index-rule associate)"
[[ "$guarded" == *"status=proved"* ]]
[[ "$guarded" == *"declared-conditions:"* ]]
[[ "$guarded" == *"additional-guard: true"* ]]
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

# Removing declarations must fail with an actionable diagnostic. Other selected
# valid rules in the same file must still be independently checkable.
dialect_absolute="$(realpath "$(dirname "$rules")/../dialects/tensor.tepl")"
cat > "$temporary_directory/unguarded.tepl" <<EOF
from "$dialect_absolute" import TensorLang as t;
use t::{dot, add};
rule bad { (dot[@d] (dot[@inner] A B) C) => (dot[auto] A (dot[auto] B C)) }
rule good { (dot[@d] (add A B) C) => (add (dot[auto] A C) (dot[auto] B C)) }
EOF
if "$compiler" check "$temporary_directory/unguarded.tepl" --semantic-policy=algebraic > "$temporary_directory/error" 2>&1; then
  exit 1
fi
failure="$(cat "$temporary_directory/error")"
[[ "$failure" == *"cannot verify rule 'bad'"* ]]
[[ "$failure" == *"explicit where condition"* ]]
"$compiler" check "$temporary_directory/unguarded.tepl" --semantic-policy=algebraic --index-rule good > /dev/null
