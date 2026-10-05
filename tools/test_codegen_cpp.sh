#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ $# == 0 ]]; then
    bazel build //:tepl
    compiler="$project_root/bazel-bin/tepl"
else
    compiler="$1"
fi
work_directory="$(mktemp -d "${TMPDIR:-/tmp}/tepl-codegen-cpp.XXXXXX")"
trap 'rm -rf "$work_directory"' EXIT
revision=0c28bd5050b85ed10e915b5348c27f760ed31ae5
eggc_source="${EGGC_SOURCE_DIR:-}"
if [[ -z "$eggc_source" ]]; then
    eggc_source="$work_directory/egg-c"
    git clone --quiet --no-checkout https://github.com/5yearsKim/egg-c.git "$eggc_source"
    git -C "$eggc_source" checkout --quiet "$revision"
fi
eggc_source="$(cd "$eggc_source" && pwd)"
test -f "$eggc_source/include/eggc/all.hpp"
cxx="${CXX:-c++}"
fixtures=(dtype analysis builtins early_where runtime literal_root custom names hygiene dialect_only examples empty)
# Optional fixture arguments allow focused reruns after a failing integration.
if [[ $# -gt 1 ]]; then fixtures=("${@:2}"); fi
for fixture in "${fixtures[@]}"; do
    application="$work_directory/$fixture"
    mkdir -p "$application"
    input="$project_root/tests/codegen/$fixture.tepl"
    case "$fixture" in
        runtime|hygiene|dialect_only) input="$project_root/tests/codegen/cpp/$fixture.tepl" ;;
        examples) input="$project_root/examples" ;;
        custom)
            # A Rust host named `signed` is a C++ keyword. Diagnose the original
            # and use the same fixture with an explicitly renamed host.
            if "$compiler" generate "$input" --target cpp --no-format --out "$application/invalid" > "$application/name_error" 2>&1; then
                echo 'C++ keyword host unexpectedly generated' >&2
                exit 1
            fi
            grep -Fq "reserved C++ identifier 'signed'" "$application/name_error"
            sed 's/\$signed(/\$signed_value(/g' "$input" > "$application/custom.tepl"
            cp "$project_root/tests/codegen/other.tepl" "$project_root/tests/codegen/mirror.tepl" "$application/"
            input="$application/custom.tepl"
            ;;
        names)
            mkdir -p "$application/rules/type"
            cp "$input" "$application/rules/type/match.tepl"
            input="$application"
            ;;
    esac
    "$compiler" generate "$input" --target cpp --no-format --out "$application/ir"
    "$compiler" generate "$input" --target cpp --no-format --out "$application/ir" --check
    if [[ "$fixture" == empty ]]; then
        printf '#include "ir/generated.h"\nint main() {}\n' > "$application/test.cc"
        "$cxx" -std=c++20 -I"$application" -I"$eggc_source/include" "$application/test.cc" -o "$application/test"
        "$application/test"
        continue
    fi
    sources=("$project_root/tests/codegen/cpp/$fixture.cc")
    if [[ "$fixture" == hygiene ]]; then
        "$compiler" generate "$input" --target cpp --cpp-namespace app::eggc --no-format --out "$application/nested_eggc"
        "$compiler" generate "$input" --target cpp --cpp-namespace app::std --no-format --out "$application/nested_std"
        # GCC's pragma-once identity heuristic also considers timestamps. Force
        # equal timestamps so missing namespace identity is caught reliably.
        while IFS= read -r header; do
            touch -r "$application/ir/generated.h" "$header"
        done < <(rg --files "$application/ir" "$application/nested_eggc" "$application/nested_std" -g '*.h')
        # Every public header must compile without an umbrella or egg-c header
        # being included first. Separate translation units expose include-order
        # dependencies that ordinary integration programs can mask.
        while IFS= read -r header; do
            printf '#include "%s"\n' "${header#"$application/"}" > "$application/header.cc"
            "$cxx" -std=c++20 -I"$application" -I"$eggc_source/include" -fsyntax-only "$application/header.cc"
        done < <(rg --files "$application/ir" -g '*.h' | sort)
    fi
    if [[ "$fixture" == names ]]; then
        "$compiler" generate "$input" --target cpp --cpp-namespace other::generated --out "$application/second"
        "$compiler" generate "$input" --target cpp --cpp-namespace other::generated --out "$application/second" --check
        sources+=("$project_root/tests/codegen/cpp/names_odr.cc")
        if "$cxx" -std=c++20 -I"$application" -I"$eggc_source/include" "$project_root/tests/codegen/cpp/wrong_attrs.cc" -o "$application/wrong" > "$application/type_error" 2>&1; then
            echo 'Cross-dialect attributes unexpectedly compiled' >&2
            exit 1
        fi
    fi
    for mode in debug release; do
        flags=(-O0 -g)
        if [[ "$mode" == release ]]; then flags=(-O2); fi
        "$cxx" -std=c++20 "${flags[@]}" -I"$application" -I"$eggc_source/include" "${sources[@]}" -o "$application/test-$mode"
        "$application/test-$mode"
    done
    case "$fixture" in
        dtype|runtime|analysis|builtins)
            "$cxx" -std=c++20 -O1 -g -fsanitize=undefined -fno-sanitize-recover=all -I"$application" -I"$eggc_source/include" "${sources[@]}" -o "$application/test-ubsan"
            "$application/test-ubsan"
            ;;
    esac
done
echo 'Generated C++ integration tests passed.'
