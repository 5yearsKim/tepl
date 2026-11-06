#!/usr/bin/env bash
set -euo pipefail
"$1" --check "$2"
"$1" --internal --check "$3"
