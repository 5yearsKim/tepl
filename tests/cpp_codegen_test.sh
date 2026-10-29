#!/usr/bin/env bash
set -euo pipefail
root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
"$root/tools/test_codegen_cpp.sh" "$root/$1" "$root/$2" "$root/$3"
