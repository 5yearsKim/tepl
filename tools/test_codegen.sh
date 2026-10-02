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
export CARGO_TARGET_DIR="$work_directory/target"

for source in custom empty dialect_only literal_root lora basic binders inherited simple; do
    if [[ -f "$project_root/tests/codegen/$source.tepl" ]]; then
        input="$project_root/tests/codegen/$source.tepl"
    else
        input="$project_root/examples/rules/$source.tepl"
    fi
    crate="$work_directory/$source"
    "$compiler" generate "$input" --target rust --out "$crate"
    if [[ -f "$project_root/tests/codegen/$source.rs" ]]; then
        mkdir -p "$crate/tests"
        cp "$project_root/tests/codegen/$source.rs" "$crate/tests/generated.rs"
    fi
    cargo test --manifest-path "$crate/Cargo.toml"
    if [[ "$source" == custom ]]; then
        cargo test --manifest-path "$crate/Cargo.toml" --release
    fi
done

# Generate the whole example project and execute cross-dialect rewrites.
crate="$work_directory/project"
"$compiler" generate "$project_root/examples" --target rust --out "$crate"
mkdir -p "$crate/tests"
cp "$project_root/tests/codegen/project.rs" "$crate/tests/generated.rs"
cargo test --manifest-path "$crate/Cargo.toml"

# Repeated generation uses the same filenames and contents.
cp -R "$crate/src" "$work_directory/first_src"
"$compiler" generate "$project_root/examples" --target rust --out "$crate"
diff -r "$work_directory/first_src" "$crate/src"

# The public typed constructor rejects attributes from another dialect.
mkdir -p "$crate/examples"
cat > "$crate/examples/wrong_attrs.rs" <<'RS'
use tepl_generated::ir::{OpNode, dialects::{scalar, tensor_lang}};
fn main() {
    let _ = OpNode::new(scalar::Op::Add, tensor_lang::OpAttrs::None,
                        vec![egg::Id::from(0); 2]);
}
RS
if cargo check --manifest-path "$crate/Cargo.toml" --example wrong_attrs > "$work_directory/type_error" 2>&1; then
    echo 'Cross-dialect attributes unexpectedly typechecked' >&2
    exit 1
fi
grep -Eq 'error\[E0(271|308)\]' "$work_directory/type_error"
