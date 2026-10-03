#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ -z "${TEPL_COMPILER:-}" ]]; then
    bazel build //:tepl
    compiler="$project_root/bazel-bin/tepl"
else
    compiler="$TEPL_COMPILER"
fi
"$compiler" generate "$project_root/examples" --target rust \
    --out "$project_root/labs/rust-egg/src/ir" "$@"
