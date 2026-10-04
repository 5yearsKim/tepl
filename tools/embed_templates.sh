#!/usr/bin/env bash
set -euo pipefail
output="$1"
shift
{
    printf '#pragma once\n#include "src/codegen/output.h"\nnamespace tepl::codegen::rust {\ninline std::vector<GeneratedFile> templateFiles() { return {\n'
    for input in "$@"; do
        relative="${input#templates/rust/}"
        printf '{"%s", R"TEPL_TEMPLATE(' "$relative"
        cat "$input"
        printf ')TEPL_TEMPLATE"},\n'
    done
    printf '}; }\n}\n'
} > "$output"
