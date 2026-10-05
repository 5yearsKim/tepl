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
work_directory="$(mktemp -d "${TMPDIR:-/tmp}/tepl-codegen.XXXXXX")"
trap 'rm -rf "$work_directory"' EXIT
# Retain Cargo's build cache across runs.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$project_root/labs/rust-egg/target}"

# Test handwritten lab code independently of compiler output.
lab="$project_root/labs/rust-egg"
cargo test --manifest-path "$lab/Cargo.toml" --locked --all-targets

# Reuse the lab's tests and host helpers against freshly generated IR in isolation.
# Keep the directory layout for tests that include the example TEPL sources.
generated_lab="$work_directory/labs/rust-egg"
mkdir -p "$generated_lab/src"
cp "$lab/Cargo.toml" "$lab/Cargo.lock" "$generated_lab/"
cp "$lab/src/lib.rs" "$generated_lab/src/"
cp -R "$lab/src/host" "$generated_lab/src/"
cp -R "$lab/tests" "$lab/examples" "$generated_lab/"
cp -R "$project_root/examples" "$work_directory/examples"
"$compiler" generate "$project_root/examples" --target rust --out "$generated_lab/src/ir"
cargo test --manifest-path "$generated_lab/Cargo.toml" --locked --all-targets
cargo test --manifest-path "$generated_lab/Cargo.toml" --locked --release --test runtime shape_builtins

# Only definitions absent from the example project need isolated generated crates.
prepare_application() {
    local application="$1"
    mkdir -p "$application/src"
    cat > "$application/Cargo.toml" <<'TOML'
[package]
name = "tepl_generated"
version = "0.1.0"
edition = "2024"

[dependencies]
egg = "0.11.0"
TOML
    cp "$lab/Cargo.lock" "$application/Cargo.lock"
    printf 'pub mod ir;\n' > "$application/src/lib.rs"
}

for source in dtype early_where names builtins analysis custom empty dialect_only literal_root; do
    crate="$work_directory/$source"
    prepare_application "$crate"
    input="$project_root/tests/codegen/$source.tepl"
    module_out="$crate/src/ir"
    if [[ "$source" == names ]]; then
        # One fixture covers nested output modules and Rust keyword source paths.
        mkdir -p "$crate/rules/type" "$crate/src/components"
        cp "$input" "$crate/rules/type/match.tepl"
        input="$crate"
        module_out="$crate/src/components/generated"
        printf 'pub mod components;\n' > "$crate/src/lib.rs"
        printf 'pub mod generated;\n' > "$crate/src/components/mod.rs"
    fi
    "$compiler" generate "$input" --target rust --out "$module_out"
    if [[ "$source" == empty ]]; then
        cargo check --offline --manifest-path "$crate/Cargo.toml"
        continue
    fi
    mkdir -p "$crate/tests"
    cp "$project_root/tests/codegen/$source.rs" "$crate/tests/generated.rs"
    cargo test --offline --manifest-path "$crate/Cargo.toml" --test generated

    # Check only behavior that could accidentally depend on debug overflow checks.
    case "$source" in
        dtype)
            cargo test --offline --manifest-path "$crate/Cargo.toml" --release --test generated
            release_test=
            ;;
        custom) release_test=overflow_and_division_by_zero ;;
        analysis) release_test=checked_boundaries ;;
        builtins) release_test=failures_reject_candidates ;;
        *) release_test= ;;
    esac
    if [[ -n "$release_test" ]]; then
        cargo test --offline --manifest-path "$crate/Cargo.toml" --release --test generated "$release_test"
    fi
    if [[ "$source" == names ]]; then
        mkdir -p "$crate/examples"
        cp "$project_root/tests/codegen/wrong_attrs.rs" "$crate/examples/wrong_attrs.rs"
        if cargo check --offline --manifest-path "$crate/Cargo.toml" --example wrong_attrs > "$work_directory/type_error" 2>&1; then
            echo 'Cross-dialect attributes unexpectedly typechecked' >&2
            exit 1
        fi
        grep -Eq 'error\[E0(271|308)\]' "$work_directory/type_error"
    fi
done
