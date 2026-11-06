#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ $# == 0 ]]; then
  bazel build //:tepl //tests:codegen_semantic_reference
  compiler="$project_root/bazel-bin/tepl"
  reference="$project_root/bazel-bin/tests/codegen_semantic_reference"
else
  compiler="$1"
  reference="${2:-$project_root/bazel-bin/tests/codegen_semantic_reference}"
fi
if [[ ! -x "$reference" ]]; then bazel build //tests:codegen_semantic_reference; fi
work_directory="$(mktemp -d "${TEST_TMPDIR:-${TMPDIR:-/tmp}}/tepl-cpp-codegen.XXXXXX")"
trap 'rm -rf "$work_directory"' EXIT
if [[ $# -ge 3 ]]; then
  eggc_include="$(dirname -- "$(dirname -- "$3")")"
elif [[ -n "${EGGC_SOURCE_DIR:-}" ]]; then
  eggc_include="$EGGC_SOURCE_DIR/include"
else
  curl -fsSL --retry 2 https://codeload.github.com/5yearsKim/egg-c/tar.gz/0c28bd5050b85ed10e915b5348c27f760ed31ae5 -o "$work_directory/eggc.tar.gz"
  printf '%s  %s\n' '89d2ffbd7999edb65ae3c180d1c9caf72117fb1e8f1f39bf90107e4bb8d8b4f9' "$work_directory/eggc.tar.gz" | sha256sum -c -
  tar -xzf "$work_directory/eggc.tar.gz" -C "$work_directory"
  eggc_include="$work_directory/egg-c-0c28bd5050b85ed10e915b5348c27f760ed31ae5/include"
fi
"$compiler" generate tests/codegen/rust/expressions.tepl --target cpp --cpp-namespace expressions --out "$work_directory/expressions"
"$compiler" compile tests/codegen/rust/expressions.tepl --out "$work_directory/expressions.json"
"$compiler" generate "$work_directory/expressions.json" --target cpp --cpp-namespace expressions --out "$work_directory/loaded"
diff -ru "$work_directory/expressions" "$work_directory/loaded"
"$compiler" generate "$work_directory/expressions.json" --target cpp --cpp-namespace expressions --out "$work_directory/expressions" --check
"$compiler" generate tests/codegen/rust/empty.tepl --target cpp --cpp-namespace empty --out "$work_directory/empty"
"$compiler" compile tests/codegen/rust/current.tepl --semantic-policy algebraic --out "$work_directory/current.json"
"$compiler" generate "$work_directory/current.json" --target cpp --out "$work_directory/current"
"$compiler" generate tests/codegen/rust/current.tepl --semantic-policy algebraic --target cpp --out "$work_directory/current" --check
"$reference" "$work_directory/current.json" --cpp > "$work_directory/reference.cc"
# Preserve arbitrary string bytes in both graph attributes and rule expressions.
printf 'dialect N {attrs A {note:string } op copy(x:tensor)->tensor {attrs:A dtype(t){yield t }shape(s){yield s }}}\nrule identity {(node:N.copy X)=>X where node.attrs.note=="a\000b" }\ngraph g {input X:f32[2] yield (N.copy[note="a\000b"] X) }\n' > "$work_directory/strings.tepl"
"$compiler" generate "$work_directory/strings.tepl" --target cpp --cpp-namespace strings --out "$work_directory/strings"
"$compiler" generate tests/codegen/cpp/witnesses.tepl --target cpp --cpp-namespace witnesses --out "$work_directory/witnesses"
"$compiler" generate examples/semantic --semantic-policy algebraic --target cpp --cpp-namespace semantic --out "$work_directory/semantic" --no-format
"$compiler" generate examples/sample --target cpp --cpp-namespace sample --out "$work_directory/sample" --no-format
# Relocation, nested keyword modules, and separate generated namespaces.
mkdir -p "$work_directory/names/rules/class" "$work_directory/components"
cp tests/codegen/rust/names.tepl "$work_directory/names/rules/class/while.tepl"
"$compiler" generate "$work_directory/names" --target cpp --cpp-namespace app::optimizer --out "$work_directory/components/generated"
cp tests/codegen/cpp/integration.cc "$work_directory/integration.cc"
cp tests/codegen/cpp/odr.cc "$work_directory/odr.cc"
cp tests/codegen/cpp/strings.cc "$work_directory/strings.cc"
cp tests/codegen/cpp/witnesses.cc "$work_directory/witnesses.cc"
if [[ -n "${CXX:-}" ]]; then
  compilers=("$CXX")
else
  compilers=(g++)
  if command -v clang++ >/dev/null; then compilers+=(clang++); fi
fi
for cxx in "${compilers[@]}"; do
  flags=(-std=c++20 -Wall -Wextra -Werror -I "$project_root/runtime/cpp/include" -I "$eggc_include" -I "$work_directory")
  for optimization in -O0 -O2; do
    printf 'Checking C++ generation with %s %s\n' "$cxx" "$optimization"
    extra=()
    if [[ "$optimization" == -O2 ]]; then extra=(-fsanitize=undefined -fno-sanitize-recover=undefined); fi
    "$cxx" "${flags[@]}" "$optimization" "${extra[@]}" "$work_directory/integration.cc" "$work_directory/odr.cc" -o "$work_directory/integration"
    "$work_directory/integration"
    "$cxx" "${flags[@]}" "$optimization" "${extra[@]}" "$work_directory/reference.cc" -o "$work_directory/reference"
    "$work_directory/reference"
    "$cxx" "${flags[@]}" "$optimization" "${extra[@]}" runtime/cpp/tests/values.cc -o "$work_directory/values"
    "$work_directory/values"
    for fixture in strings witnesses; do
      "$cxx" "${flags[@]}" "$optimization" "${extra[@]}" "$work_directory/$fixture.cc" -o "$work_directory/$fixture-test"
      "$work_directory/$fixture-test"
    done
  done
  # Each public runtime header must supply its own dependencies.
  for header in runtime/cpp/include/tepl/runtime/*.h; do
    printf '#include <%s>\n' "${header#runtime/cpp/include/}" > "$work_directory/header.cc"
    "$cxx" "${flags[@]}" -fsyntax-only "$work_directory/header.cc"
  done
  for header in "$work_directory/current"/*.h "$work_directory/current"/dialects/*.h "$work_directory/current"/rules/*.h "$work_directory/current"/graphs/*.h; do
    printf '#include "%s"\n' "$header" > "$work_directory/header.cc"
    "$cxx" "${flags[@]}" -fsyntax-only "$work_directory/header.cc"
  done
  printf 'C++ code generation passed with %s\n' "$cxx"
done
# Check mode must leave edited output alone; regeneration owns only its manifest.
printf '\n// edited\n' >> "$work_directory/expressions/program.h"
cp "$work_directory/expressions/program.h" "$work_directory/edited.h"
if "$compiler" generate "$work_directory/expressions.json" --target cpp --cpp-namespace expressions --out "$work_directory/expressions" --check; then
  echo 'Expected drift check to fail' >&2; exit 1
fi
cmp "$work_directory/edited.h" "$work_directory/expressions/program.h"
printf 'keep me\n' > "$work_directory/expressions/unrelated.txt"
"$compiler" generate tests/codegen/rust/empty.tepl --target cpp --out "$work_directory/expressions"
test ! -e "$work_directory/expressions/dialects/d.h"
grep -Fxq 'keep me' "$work_directory/expressions/unrelated.txt"
