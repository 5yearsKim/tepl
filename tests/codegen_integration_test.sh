#!/usr/bin/env bash
set -euo pipefail
root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
exec "$root/tools/test_codegen.sh" "$root/$1"
