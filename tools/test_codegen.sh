#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ $# == 0 ]]; then
    bazel build //:tepl //tests:codegen_reference //tests:codegen_semantic_reference
    compiler="$project_root/bazel-bin/tepl"
else
    compiler="$1"
fi
reference="${2:-$project_root/bazel-bin/tests/codegen_reference}"
semantic_reference="${3:-$project_root/bazel-bin/tests/codegen_semantic_reference}"
if [[ ! -x "$reference" ]]; then bazel build //tests:codegen_reference; fi
if [[ ! -x "$semantic_reference" ]]; then bazel build //tests:codegen_semantic_reference; fi
work_directory="$(mktemp -d "${TEST_TMPDIR:-${TMPDIR:-/tmp}}/tepl-rust-codegen.XXXXXX")"
trap 'rm -rf "$work_directory"' EXIT
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$work_directory/target}"
mkdir -p "$work_directory/src" "$work_directory/tests"
cat > "$work_directory/Cargo.toml" <<TOML
[package]
name = "tepl-generated-tests"
version = "0.1.0"
edition = "2024"
[dependencies]
tepl-runtime = { path = "$project_root/runtime/rust" }
TOML
"$compiler" generate tests/codegen/rust/expressions.tepl --out "$work_directory/src/expressions"
"$compiler" compile tests/codegen/rust/expressions.tepl --out "$work_directory/compiled.json"
"$compiler" generate "$work_directory/compiled.json" --out "$work_directory/src/loaded"
diff -ru "$work_directory/src/expressions" "$work_directory/src/loaded"
"$compiler" generate "$work_directory/compiled.json" --out "$work_directory/src/expressions" --check
"$compiler" generate tests/codegen/rust/empty.tepl --out "$work_directory/src/empty"
"$compiler" compile tests/codegen/rust/current.tepl --semantic-policy algebraic --out "$work_directory/current.json"
"$compiler" generate "$work_directory/current.json" --out "$work_directory/src/current"
"$compiler" generate tests/codegen/rust/current.tepl --semantic-policy algebraic --out "$work_directory/src/current" --check
"$compiler" generate examples/semantic --semantic-policy algebraic --out "$work_directory/src/semantic"
"$compiler" generate examples/sample --out "$work_directory/src/sample"
# Nested Rust keyword modules also work after relocating the generated module.
mkdir -p "$work_directory/names/rules/type" "$work_directory/src/components"
cp tests/codegen/rust/names.tepl "$work_directory/names/rules/type/match.tepl"
"$compiler" generate "$work_directory/names" --out "$work_directory/src/components/generated"
"$compiler" compile "$work_directory/names" --out "$work_directory/names.json"
"$compiler" generate "$work_directory/names.json" --out "$work_directory/src/components/generated" --check
printf 'pub mod generated;\n' > "$work_directory/src/components/mod.rs"
cat > "$work_directory/src/lib.rs" <<'RUST'
pub mod expressions;
pub mod loaded;
pub mod semantic;
pub mod empty;
pub mod sample;
pub mod components;
pub mod current;
RUST
cp tests/codegen/rust/generated.rs "$work_directory/tests/generated.rs"
"$reference" > "$work_directory/tests/reference.rs"
"$semantic_reference" "$work_directory/current.json" > "$work_directory/tests/semantic_reference.rs"
cargo test --offline --manifest-path "$work_directory/Cargo.toml" --all-targets
# Overflow and error handling must not depend on debug overflow checks.
cargo test --offline --release --manifest-path "$work_directory/Cargo.toml" --tests
cargo test --offline --locked --manifest-path "$project_root/runtime/rust/Cargo.toml"
# Check mode reports drift without rewriting, and synchronization removes only owned files.
printf '\n// changed\n' >> "$work_directory/src/expressions/mod.rs"
cp "$work_directory/src/expressions/mod.rs" "$work_directory/changed.rs"
if "$compiler" generate "$work_directory/compiled.json" --out "$work_directory/src/expressions" --check; then
    echo 'Expected drift check to fail' >&2
    exit 1
fi
cmp "$work_directory/src/expressions/mod.rs" "$work_directory/changed.rs"
printf 'keep me\n' > "$work_directory/src/expressions/unrelated.txt"
"$compiler" generate tests/codegen/rust/empty.tepl --out "$work_directory/src/expressions"
test ! -e "$work_directory/src/expressions/dialects/d.rs"
grep -Fxq 'keep me' "$work_directory/src/expressions/unrelated.txt"
