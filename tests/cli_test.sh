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
check_exit 0 parse -h

check_exit 0 parse "$example"
grep -Fxq 'Parsed 1 rule(s).' "$output"
check_exit 0 parse "$example" --tree
grep -Fq '(program' "$output"
cp "$output" "${TEST_TMPDIR}/tree"
check_exit 0 parse --tree "$example"
cmp "$output" "${TEST_TMPDIR}/tree"

check_exit 2
check_exit 2 parse
check_exit 2 unknown "$example"
check_exit 2 parse "$example" --unknown
check_exit 2 parse "$example" extra
check_exit 2 parse "${TEST_TMPDIR}/missing.tepl"
grep -Fq 'cannot open file' "$output"

invalid="${TEST_TMPDIR}/invalid.tepl"
printf 'rule r { X => }\n' >"$invalid"
check_exit 1 parse "$invalid"
grep -Fq "${invalid}:1:" "$output"
