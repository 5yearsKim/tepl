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
grep -Fq '  from "dialects/tensor.tepl" import TensorLang as t' "$output"
grep -Fq '  use t::{add}' "$output"
grep -Fq '  dialect TensorLang' "$output"
cp "$output" "${TEST_TMPDIR}/ast"
check_exit 0 parse --ast "$example"
cmp "$output" "${TEST_TMPDIR}/ast"

check_exit 2
check_exit 2 parse
check_exit 2 unknown "$example"
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

printf 'from "%s" import TensorLang as t; rule r { (missing.add X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown dialect alias 'missing'" "$output"

printf 'from "%s" import TensorLang; use TensorLang::{add}; rule r { (add X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 0 parse "$bad_arity"

printf 'from "%s" import TensorLang as t; use t::{add}; rule r { (multiply X Y) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown operation 'multiply'" "$output"

printf 'from "%s" import TensorLang as t; use t::{missing}; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown operation 'missing' in dialect alias 't'" "$output"

printf 'from "%s" import Missing as t; use t; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "dialect 'Missing' is not defined" "$output"

printf 'from "%s" import TensorLang as t; use missing; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown dialect alias 'missing'" "$output"

printf 'import "%s"; rule r { (add X) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq 'expects exactly 2 operands' "$output"

other="${TEST_TMPDIR}/other.tepl"
printf 'dialect Other { op add(lhs: tensor, rhs: tensor) -> tensor; }\n' >"$other"
printf 'from "%s" import TensorLang as t; from "other.tepl" import Other as o; use t::{add}; use o::{add}; rule r { X => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
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
check_exit 1 parse "$bad_arity"
grep -Fq 'requires an attribute descriptor' "$output"

printf 'import "%s"; rule r { (missing X) => X }\n' "$dialect" >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown operation 'missing'" "$output"

printf 'dialect t { op bad(x: tensor) -> tensor { attrs { axis: mystery; } } }\n' >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown attribute type 'mystery'" "$output"

printf 'dialect t { op bad(x: tensor) -> tensor { attrs { kind: symbol; } } }\n' >"$bad_arity"
check_exit 1 parse "$bad_arity"
grep -Fq "unknown attribute type 'symbol'" "$output"

cycle_a="${TEST_TMPDIR}/a.tepl"
cycle_b="${TEST_TMPDIR}/b.tepl"
printf 'import "b.tepl"; rule r { X => X }\n' >"$cycle_a"
printf 'import "a.tepl";\n' >"$cycle_b"
check_exit 1 parse "$cycle_a"
grep -Fq 'cyclic import' "$output"

abstract="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/abstract.tepl"
inherited="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/inherited.tepl"
check_exit 0 parse "$abstract"
grep -Fxq 'Parsed 3 rule(s).' "$output"
check_exit 0 parse "$inherited"
grep -Fxq 'Parsed 7 rule(s).' "$output"
check_exit 0 parse "$inherited" --ast
grep -Fq 'parameter F: op<(tensor, tensor) -> tensor>' "$output"
grep -Fq 'extends associate_right' "$output"
check_exit 1 host-template "$inherited"
grep -Fq 'inheritance expansion' "$output"

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

# Dtype validation runs for both parsing and host generation.
basic="${TEST_SRCDIR}/${TEST_WORKSPACE}/examples/basic.tepl"
check_exit 0 parse "$basic" --ast
grep -Fq 'tensor X f32[N]' "$output"
grep -Fq 'float 1.0:f32' "$output"
grep -Fq 'integer 1:i32' "$output"
check_exit 0 host-template "$basic"
grep -Fq 'fn same_dtype(&self, arg0: &TensorInfo, arg1: &TensorInfo)' "$output"

bad_dtype="${TEST_TMPDIR}/bad_dtype.tepl"
printf 'rule r { X: float32[N] X => X }\n' >"$bad_dtype"
check_exit 1 parse "$bad_dtype"
grep -Fq "${bad_dtype}:1:13: unknown dtype 'float32'" "$output"
check_exit 1 host-template "$bad_dtype"
grep -Fq "unknown dtype 'float32'" "$output"
printf 'rule r { X => 256:u8 }\n' >"$bad_dtype"
check_exit 1 parse "$bad_dtype"
grep -Fq "invalid for dtype 'u8'" "$output"
