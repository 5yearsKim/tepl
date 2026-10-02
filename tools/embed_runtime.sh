#!/usr/bin/env bash
set -euo pipefail
output="$1"
shift
{
    printf '#pragma once\n#include "src/codegen/output.h"\nnamespace tepl::codegen::rust {\ninline std::vector<GeneratedFile> runtimeFiles() { return {\n'
    for input in "$@"; do
        relative="${input#runtime/rust/}"
        printf '{"%s", R"TEPL_RUNTIME(' "$relative"
        cat "$input"
        printf ')TEPL_RUNTIME"},\n'
    done
    printf '}; }\n}\n'
} > "$output"
