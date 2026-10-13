#!/usr/bin/env bash
set -euo pipefail
compiler="$1"
rules="$2"
symbolic="$($compiler check "$rules" --semantic-policy=algebraic --rule distribute)"
[[ "$symbolic" == *"status=proved"* ]]
[[ "$symbolic" == *"lhs_contracting_dimensions = @d.lhs_contracting_dimensions"* ]]
[[ "$symbolic" == *"additional-guard: true"* ]]
[[ "$symbolic" == *"lhs-normalized: Reduce"* ]]
guarded="$($compiler check "$rules" --semantic-policy=algebraic --rule associate)"
[[ "$guarded" == *"status=proved"* ]]
[[ "$guarded" == *"declared-conditions:"* ]]
[[ "$guarded" == *"additional-guard: true"* ]]
if "$compiler" check "$rules" --rule distribute > /dev/null; then
  exit 1
fi
if "$compiler" check "$rules" --rule missing > /dev/null; then
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
"$compiler" check "$temporary_directory/unguarded.tepl" --semantic-policy=algebraic --rule good > /dev/null

# Retired flags must not silently configure a nonexistent solver path.
for option in --solver-timeout-ms=1000 --verify-small --index-rule=distribute; do
    if "$compiler" check "$rules" "$option" > /dev/null 2>&1; then
        exit 1
    fi
done

# Semantics must have an axis interface; auto cannot fall back to search.
cat > "$temporary_directory/no_axes.tepl" <<'TEPL'
dialect D { op identity(x:tensor)->tensor { semantics {yield x;} } }
TEPL
if "$compiler" check "$temporary_directory/no_axes.tepl" > "$temporary_directory/error" 2>&1; then exit 1; fi
grep -Fq 'semantics requires an axes block' "$temporary_directory/error"
cat > "$temporary_directory/no_semantics.tepl" <<'TEPL'
dialect D { op identity(x:tensor)->tensor { attrs {p:index[];} } }
rule unsupported { (identity[@p] X) => (identity[auto] X) }
TEPL
if "$compiler" check "$temporary_directory/no_semantics.tepl" > "$temporary_directory/error" 2>&1; then exit 1; fi
grep -Fq 'auto requires operations with axes and semantics' "$temporary_directory/error"

# New semantics is compiler-only until codegen learns its constructors.
if "$compiler" generate "$rules" --semantic-policy=algebraic --out "$temporary_directory/generated" > "$temporary_directory/error" 2>&1; then exit 1; fi
grep -Fq 'code generation is not implemented' "$temporary_directory/error"
test ! -e "$temporary_directory/generated"
