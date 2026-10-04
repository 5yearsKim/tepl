#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--check]\n' "$0"
}

mode=format
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case "${1:-}" in
    '') ;;
    --check) mode=check ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

formatter="${CLANG_FORMAT:-clang-format}"
if ! command -v "$formatter" >/dev/null 2>&1; then
    printf 'Cannot find %s. Install clang-format or set CLANG_FORMAT to its executable path.\n' "$formatter" >&2
    exit 127
fi

if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
    project_root="$BUILD_WORKSPACE_DIRECTORY"
else
    project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd -- "$project_root"

# Only first-party C++ sources: exclude downloaded and generated ANTLR code.
shopt -s globstar nullglob
files=(src/**/*.{cc,cpp,cxx,h,hpp,hxx} tests/**/*.{cc,cpp,cxx,h,hpp,hxx} tools/**/*.{cc,cpp,cxx,h,hpp,hxx})
if (( ${#files[@]} == 0 )); then
    printf 'No C++ sources found.\n'
    exit 0
fi

if [[ "$mode" == check ]]; then
    "$formatter" --style=file --dry-run --Werror "${files[@]}"
    printf 'Formatting check passed for %s C++ files.\n' "${#files[@]}"
else
    "$formatter" --style=file -i "${files[@]}"
    printf 'Formatted %s C++ files.\n' "${#files[@]}"
fi
