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

# The consuming application owns its Cargo package and module declaration.
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
    printf 'pub mod ir;\n' > "$application/src/lib.rs"
}

for source in custom empty dialect_only literal_root lora basic binders inherited simple; do
    if [[ -f "$project_root/tests/codegen/$source.tepl" ]]; then
        input="$project_root/tests/codegen/$source.tepl"
    else
        input="$project_root/examples/rules/$source.tepl"
    fi
    crate="$work_directory/$source"
    prepare_application "$crate"
    "$compiler" generate "$input" --target rust --out "$crate/src/ir"
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
prepare_application "$crate"
"$compiler" generate "$project_root/examples" --target rust --out "$crate/src/ir"
mkdir -p "$crate/tests"
cp "$project_root/tests/codegen/project.rs" "$crate/tests/generated.rs"
cargo test --manifest-path "$crate/Cargo.toml"

# Repeated generation uses the same filenames and contents.
cp -R "$crate/src" "$work_directory/first_src"
"$compiler" generate "$project_root/examples" --target rust --out "$crate/src/ir"
diff -r "$work_directory/first_src" "$crate/src"

# Existing-crate output is formatted deterministically and check detects drift.
module_out="$work_directory/ir"
"$compiler" generate "$project_root/examples" --out "$module_out"
"$compiler" generate "$project_root/examples" --out "$module_out" --check
printf '// changed\n' >> "$module_out/op_node.rs"
if "$compiler" generate "$project_root/examples" --out "$module_out" --check; then
    echo 'Edited generated output unexpectedly passed --check' >&2
    exit 1
fi

# A renamed module works under another application module, including deeply
# nested source rules. Relocate the same output to another name without editing it.
relocated="$work_directory/relocated"
prepare_application "$relocated"
printf 'pub mod components;\n' > "$relocated/src/lib.rs"
mkdir -p "$relocated/src/components" "$work_directory/nested_project"
printf 'pub mod any_name;\n' > "$relocated/src/components/mod.rs"
cp -R "$project_root/examples/dialects" "$project_root/examples/rules" "$work_directory/nested_project/"
mkdir -p "$work_directory/nested_project/rules/nested/deeper"
cat > "$work_directory/nested_project/rules/nested/deeper/commute.tepl" <<'TEPL'
from "../../../dialects/scalar.tepl" import Scalar as s;
rule commute_add { (s.add X Y) => (s.add Y X) }
TEPL
"$compiler" generate "$work_directory/nested_project" --out "$relocated/src/components/any_name"
test ! -e "$relocated/src/components/any_name/src"
test ! -e "$relocated/src/components/any_name/Cargo.toml"
mkdir -p "$relocated/tests"
sed 's/tepl_generated::ir/tepl_generated::components::any_name/g' \
    "$project_root/tests/codegen/project.rs" > "$relocated/tests/generated.rs"
cat >> "$relocated/tests/generated.rs" <<'RS'

#[test]
fn deeply_nested_rule_executes_in_renamed_module() {
    use tepl_generated::components::any_name::rules::nested::deeper::commute::rule_commute_add;
    let mut graph = EGraph::<OpNode, ()>::default();
    let x = graph.add(OpNode::literal("1", DType::I32).unwrap());
    let y = graph.add(OpNode::literal("2", DType::I32).unwrap());
    let root = graph.add(OpNode::new(s::Op::Add, s::OpAttrs::None, vec![x, y]).unwrap());
    graph.rebuild();
    let rule = rule_commute_add::build_rewrite(metadata, Inference(true), ()).unwrap();
    let matches = rule.search(&graph);
    assert!(!rule.apply(&mut graph, &matches).is_empty());
    graph.rebuild();
    let swapped = graph.lookup(OpNode::new(s::Op::Add, s::OpAttrs::None, vec![y, x]).unwrap()).unwrap();
    assert_eq!(graph.find(root), graph.find(swapped));
    assert_eq!(rule.name.to_string(), "nested::deeper::commute::commute_add");
}
RS
cargo test --manifest-path "$relocated/Cargo.toml"
mv "$relocated/src/components/any_name" "$relocated/src/components/generated"
printf 'pub mod generated;\n' > "$relocated/src/components/mod.rs"
sed -i 's/components::any_name/components::generated/g' "$relocated/tests/generated.rs"
cargo test --manifest-path "$relocated/Cargo.toml"

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
