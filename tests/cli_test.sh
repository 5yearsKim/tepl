#!/usr/bin/env bash
set -euo pipefail

tepl="${TEST_SRCDIR}/${TEST_WORKSPACE}/$1"
example="${TEST_SRCDIR}/${TEST_WORKSPACE}/$2"
output="${TEST_TMPDIR}/output"

check_exit() {
  local expected="$1"
  shift
  local actual=0
  "$tepl" "$@" >"$output" 2>&1 || actual=$?
  if [[ "$actual" != "$expected" ]]; then
    cat "$output" >&2
    echo "Expected exit $expected, got $actual for: $*" >&2
    exit 1
  fi
}

check_exit 0 --help
grep -Fq 'parse' "$output"
check_exit 0 -h
check_exit 0 parse --help
grep -Fq -- '--tree' "$output"
grep -Fq -- '--ast' "$output"
check_exit 0 parse -h

check_exit 0 parse "$example"
grep -Fxq 'Parsed 1 rule(s).' "$output"
check_exit 0 parse "$example" --tree
grep -Fq '(program' "$output"
cp "$output" "${TEST_TMPDIR}/tree"
check_exit 0 parse --tree "$example"
cmp "$output" "${TEST_TMPDIR}/tree"
check_exit 0 parse "$example" --ast
grep -Fq '  rule commute_add' "$output"
grep -Fq '      operator add' "$output"
grep -Fq '  from "../dialects/tensor.tepl" import TensorLang as t' "$output"
grep -Fq '  use t::{add}' "$output"
grep -Fq '  dialect TensorLang' "$output"
cp "$output" "${TEST_TMPDIR}/ast"
check_exit 0 parse --ast "$example"
cmp "$output" "${TEST_TMPDIR}/ast"

check_exit 2
check_exit 2 parse
check_exit 2 unknown "$example"
check_exit 2 host-template "$example"
check_exit 2 parse "$example" --impl
check_exit 2 parse "$example" --unknown
check_exit 2 parse "$example" --tree --ast
check_exit 2 parse "$example" extra
check_exit 2 parse "${TEST_TMPDIR}/missing.tepl"
grep -Fq 'cannot open file' "$output"

invalid="${TEST_TMPDIR}/invalid.tepl"
printf 'rule r { X => }\n' >"$invalid"
check_exit 1 parse "$invalid"
grep -Fq "${invalid}:1:" "$output"

missing_import="${TEST_TMPDIR}/missing_import.tepl"
printf 'import "missing.tepl"; rule r { X => X }\n' >"$missing_import"
check_exit 1 parse "$missing_import"
grep -Fq "cannot open import 'missing.tepl'" "$output"

bad_arity="${TEST_TMPDIR}/bad_arity.tepl"
printf 'import "%s"; rule r { (add X) => X }\n' "$example" >"$bad_arity"
# An import must contain dialect declarations only.
check_exit 1 parse "$bad_arity"
grep -Fq 'contains rules' "$output"

dialect="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/dialects/tensor.tepl"
printf 'from "%s" import TensorLang as t; use t; rule r { (multiply X Y) => (multiply Y X) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t; rule r { (t.add X Y) => (t.add Y X) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t; rule r { (t.dot[@d] X Y) => (t.dot[@d] X Y) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

# Parsing preserves unresolved operation names; check resolves them through core.
printf 'from "%s" import TensorLang as t; rule r { (missing.add X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity" --ast
grep -Fq 'operator missing.add' "$output"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'missing.add'" "$output"

printf 'from "%s" import TensorLang; use TensorLang::{add}; rule r { (add X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t; use t::{add}; rule r { (multiply X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'multiply'" "$output"

printf 'from "%s" import TensorLang as t; use t::{missing}; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'missing' in dialect 't'" "$output"

printf 'from "%s" import Missing as t; use t; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "dialect 'Missing' is not defined" "$output"

printf 'from "%s" import TensorLang as t; use missing; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown dialect alias 'missing'" "$output"

printf 'import "%s"; rule r { (add X) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq 'expects exactly 2 operands' "$output"

other="${TEST_TMPDIR}/other.tepl"
printf 'dialect Other { op add(lhs: tensor, rhs: tensor) -> tensor; }\n' >"$other"
printf 'from "%s" import TensorLang as t; from "other.tepl" import Other as o; use t::{add}; use o::{add}; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "ambiguous operation 'add'" "$output"

same_name="${TEST_TMPDIR}/same_name.tepl"
printf 'dialect TensorLang { op add(lhs: tensor, rhs: tensor) -> tensor; }\n' >"$same_name"
printf 'from "%s" import TensorLang as t; from "same_name.tepl" import TensorLang as other; rule r { (t.add X Y) => (other.add X Y) }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

duplicate="${TEST_TMPDIR}/duplicate.tepl"
printf 'dialect Duplicate {} dialect Duplicate {}\n' >"$duplicate"
printf 'from "duplicate.tepl" import Duplicate as d; rule r { X => X }\n' >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "duplicate dialect 'Duplicate'" "$output"

printf 'import "%s"; rule r { (dot X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq 'requires an attribute descriptor' "$output"

printf 'import "%s"; rule r { (missing X) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown operation 'missing'" "$output"

printf 'dialect t { op bad(x: tensor) -> tensor { attrs { axis: mystery; } } }\n' >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown attribute type 'mystery'" "$output"

printf 'dialect t { op bad(x: tensor) -> tensor { attrs { kind: symbol; } } }\n' >"$bad_arity"
check_exit 1 check "$bad_arity"
grep -Fq "unknown attribute type 'symbol'" "$output"

cycle_a="${TEST_TMPDIR}/a.tepl"
cycle_b="${TEST_TMPDIR}/b.tepl"
printf 'import "b.tepl"; rule r { X => X }\n' >"$cycle_a"
printf 'import "a.tepl";\n' >"$cycle_b"
check_exit 1 parse "$cycle_a"
grep -Fq 'cyclic import' "$output"

abstract="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/rules/abstract.tepl"
inherited="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/rules/inherited.tepl"
check_exit 0 parse "$abstract"
grep -Fxq 'Parsed 3 rule(s).' "$output"
check_exit 0 parse "$inherited"
grep -Fxq 'Parsed 7 rule(s).' "$output"
check_exit 0 parse "$inherited" --ast
grep -Fq 'parameter F: op<(tensor, tensor) -> tensor>' "$output"
grep -Fq 'extends associate_right' "$output"

printf 'from "%s" import {missing};\n' "$abstract" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "rule 'missing'" "$output"

printf 'from "%s" import {commute_add};\n' "$example" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq 'must be abstract' "$output"

template="${TEST_TMPDIR}/template.tepl"
printf 'abstract rule a() { X => X } abstract rule b() { Y => Y }\n' >"$template"
printf 'from "template.tepl" import {a}; from "template.tepl" import {b};\n' >"$bad_arity"
check_exit 0 parse "$bad_arity" --ast
grep -Fq 'abstract rule a' "$output"
grep -Fq 'abstract rule b' "$output"

printf 'from "template.tepl" import {a};\n' >"$bad_arity"
check_exit 0 parse "$bad_arity" --ast
grep -Fq 'abstract rule a' "$output"
if grep -Fq 'abstract rule b' "$output"; then
  echo 'Unselected template leaked into the AST' >&2
  exit 1
fi

printf 'abstract rule a() { X => X } abstract rule a() { Y => Y }\n' >"$template"
check_exit 1 parse "$bad_arity"
grep -Fq "duplicate rule 'a'" "$output"

printf 'from "b.tepl" import {b}; abstract rule a() { X => X }\n' >"$cycle_a"
printf 'from "a.tepl" import {a}; abstract rule b() { X => X }\n' >"$cycle_b"
check_exit 1 parse "$cycle_a"
grep -Fq 'cyclic import' "$output"

# Parse preserves dtype annotations; core validates them through check.
basic="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/rules/basic.tepl"
check_exit 0 parse "$basic" --ast
grep -Fq 'tensor X f32[N]' "$output"
grep -Fq 'float 1.0:f32' "$output"
grep -Fq 'integer 1:i32' "$output"
check_exit 0 check "$basic"
grep -Fq 'same_dtype' "$output"

bad_dtype="${TEST_TMPDIR}/bad_dtype.tepl"
printf 'rule r { X: float32[N] X => X }\n' >"$bad_dtype"
check_exit 0 parse "$bad_dtype" --ast
grep -Fq 'tensor X float32[N]' "$output"
check_exit 1 check "$bad_dtype"
grep -Fq "${bad_dtype}:1:10: unknown dtype 'float32'" "$output"
printf 'rule r { X => 256:u8 }\n' >"$bad_dtype"
check_exit 0 parse "$bad_dtype"
check_exit 1 check "$bad_dtype"
grep -Fq "invalid for dtype 'u8'" "$output"

# Semantic analysis emits the checked IR and rejects invalid rules.
check_exit 0 check --help
check_exit 2 check
check_exit 2 check "${TEST_TMPDIR}/missing.tepl"
check_exit 0 check "$example"
grep -Fq 'CheckedProgram' "$output"
grep -Fq 'MatchOp(TensorLang.add' "$output"
grep -Fq 'BuildOp(TensorLang.add' "$output"
check_exit 0 check "$inherited"
grep -Fq 'commute_small_vectors' "$output"
grep -Fq 'expanded at' "$output"
check_exit 1 check "$bad_dtype"
grep -Fq "invalid for dtype 'u8'" "$output"
printf 'rule r { X => missing }\n' >"$invalid"
check_exit 1 check "$invalid"
grep -Fq "unknown RHS capture 'missing'" "$output"
grep -Fq "${invalid}:1:" "$output"

# Generation checks semantics before creating output and diagnoses unsupported targets.
check_exit 0 generate --help
grep -Fq -- '--target' "$output"
check_exit 2 generate "$example"
check_exit 2 generate "$example" --target unknown --out "${TEST_TMPDIR}/invalid_target"
check_exit 0 generate "$example" --target cpp --no-format --out "${TEST_TMPDIR}/cpp"
test -f "${TEST_TMPDIR}/cpp/generated.h"
test -f "${TEST_TMPDIR}/cpp/rules/simple.h"
check_exit 0 generate "$example" --target cpp --no-format --out "${TEST_TMPDIR}/cpp" --check
check_exit 1 generate "$example" --target cpp --no-format --cpp-namespace class --out "${TEST_TMPDIR}/cpp_invalid"
test ! -e "${TEST_TMPDIR}/cpp_invalid"
check_exit 1 generate "$example" --target python --out "${TEST_TMPDIR}/unimplemented"
grep -Fq 'python code generation is not implemented' "$output"
check_exit 1 generate "$invalid" --out "${TEST_TMPDIR}/invalid_program"
test ! -e "${TEST_TMPDIR}/invalid_program"
# Removed crate-generation options are usage errors.
check_exit 2 generate "$example" --out "${TEST_TMPDIR}/bad_package" --package-name example_rules
check_exit 2 generate "$example" --out "${TEST_TMPDIR}/old_layout" --layout standalone
check_exit 0 generate "$example" --target rust --out "${TEST_TMPDIR}/generated"
test -f "${TEST_TMPDIR}/generated/mod.rs"
test -f "${TEST_TMPDIR}/generated/dialects/tensor_lang.rs"
test -f "${TEST_TMPDIR}/generated/rules/simple.rs"
test ! -e "${TEST_TMPDIR}/generated/Cargo.toml"
test ! -e "${TEST_TMPDIR}/generated/src"
grep -Fq 'pub mod rule_commute_add' "${TEST_TMPDIR}/generated/rules/simple.rs"
# Formatting defaults to on, can be disabled, and uses the same policy in check mode.
raw_out="${TEST_TMPDIR}/raw_output"
check_exit 0 generate "$example" --no-format --out "$raw_out"
if cmp -s "${TEST_TMPDIR}/generated/mod.rs" "$raw_out/mod.rs"; then
  echo 'Default output unexpectedly matches unformatted output' >&2
  exit 1
fi
check_exit 0 generate "$example" --no-format --out "$raw_out" --check
check_exit 1 generate "$example" --out "$raw_out" --check
check_exit 0 generate "$example" --format --out "$raw_out"
cmp "${TEST_TMPDIR}/generated/mod.rs" "$raw_out/mod.rs"
# Missing rustfmt fails before output writes; opting out needs no formatter.
PATH="${TEST_TMPDIR}/no_tools" check_exit 2 generate "$example" --out "${TEST_TMPDIR}/missing_formatter"
grep -Fq 'rustfmt failed' "$output"
test ! -e "${TEST_TMPDIR}/missing_formatter"
PATH="${TEST_TMPDIR}/no_tools" check_exit 0 generate "$example" --no-format --out "${TEST_TMPDIR}/without_formatter"
PATH="${TEST_TMPDIR}/no_tools" check_exit 2 generate "$example" --target cpp --out "${TEST_TMPDIR}/missing_cpp_formatter"
grep -Fq 'clang-format failed' "$output"
test ! -e "${TEST_TMPDIR}/missing_cpp_formatter"
PATH="${TEST_TMPDIR}/no_tools" check_exit 0 generate "$example" --target cpp --no-format --out "${TEST_TMPDIR}/without_cpp_formatter"
printf 'not a directory\n' > "${TEST_TMPDIR}/blocked_output"
check_exit 2 generate "$example" --out "${TEST_TMPDIR}/blocked_output"

# Directory loading preserves file scope, relative paths, and instance ownership.
project="${TEST_TMPDIR}/project"
mkdir -p "$project/rules/nested" "$project/dialects"
printf 'dialect Test { op add(x: tensor, y: tensor) -> tensor; }\n' > "$project/dialects/test.tepl"
printf 'from "../dialects/test.tepl" import Test as t; rule same { (t.add X Y) => (t.add Y X) where { $allowed(X); } }\n' > "$project/rules/one.tepl"
printf 'from "../../dialects/test.tepl" import Test as t; rule same { X: [N] (t.add X Y) => (t.add Y X) where { $allowed(N, N); } }\n' > "$project/rules/nested/two.tepl"
check_exit 0 generate "$project" --out "${TEST_TMPDIR}/project_out"
test -f "${TEST_TMPDIR}/project_out/dialects/test.rs"
test -f "${TEST_TMPDIR}/project_out/rules/nested/two.rs"
grep -Fq '"one::same"' "${TEST_TMPDIR}/project_out/rules/one.rs"
grep -Fq '"nested::two::same"' "${TEST_TMPDIR}/project_out/rules/nested/two.rs"
check_exit 0 generate "$inherited" --out "${TEST_TMPDIR}/inherited_out"
grep -Fq 'pub mod rule_commute_small_vectors' "${TEST_TMPDIR}/inherited_out/rules/inherited.rs"
test ! -f "${TEST_TMPDIR}/inherited_out/rules/abstract.rs"
# Rust normalization collisions produce a diagnostic, never numeric suffixes.
printf 'dialect Test { op subtract(x: tensor, y: tensor) -> tensor; }\n' > "$project/dialects/duplicate.tepl"
check_exit 1 generate "$project" --out "${TEST_TMPDIR}/collision_out"
grep -Fq 'dialect module name collision' "$output"
test ! -e "${TEST_TMPDIR}/collision_out"

# Invalid target names report the TEPL location before invoking the formatter.
invalid_name="${TEST_TMPDIR}/invalid_name.tepl"
printf 'dialect self_ { op copy(x: tensor) -> tensor; }\n' > "$invalid_name"
check_exit 1 generate "$invalid_name" --out "${TEST_TMPDIR}/invalid_name_out"
grep -Fq "${invalid_name}:1:" "$output"
grep -Fq "cannot generate Rust identifier 'Self'" "$output"
test ! -e "${TEST_TMPDIR}/invalid_name_out"
PATH="${TEST_TMPDIR}/no_tools" check_exit 1 generate "$invalid_name" --out "${TEST_TMPDIR}/invalid_name_out"
grep -Fq "does not allow r#Self" "$output"
test ! -e "${TEST_TMPDIR}/invalid_name_out"

# Module output is suitable for an existing crate; check mode never repairs drift.
module_out="${TEST_TMPDIR}/any_name"
check_exit 0 generate "$example" --out "$module_out"
test -f "$module_out/op_node.rs"
test -f "$module_out/.tepl-generated-files"
test ! -e "$module_out/Cargo.toml"
check_exit 0 generate "$example" --out "$module_out" --check
printf '// handwritten host\n' > "$module_out/host.rs"
printf '// edited generated file\n' > "$module_out/op_node.rs"
cp "$module_out/op_node.rs" "${TEST_TMPDIR}/edited_node"
check_exit 1 generate "$example" --out "$module_out" --check
cmp "$module_out/op_node.rs" "${TEST_TMPDIR}/edited_node"
check_exit 0 generate "$example" --out "$module_out"
# Switching the source removes stale generated rule files but keeps the host file.
check_exit 0 generate "$inherited" --out "$module_out"
test ! -f "$module_out/rules/simple.rs"
test -f "$module_out/rules/inherited.rs"
test -f "$module_out/host.rs"
check_exit 0 generate "$inherited" --out "$module_out" --check
