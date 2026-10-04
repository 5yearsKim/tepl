// Exercise the same public runtime APIs in the lab and freshly generated crates.
extern crate rust_egg as tepl_generated;

#[path = "../../../tests/codegen/shape_builtins.rs"]
mod shape_builtins;
#[path = "../../../tests/codegen/shape_patterns.rs"]
mod shape_patterns;

#[path = "../../../tests/codegen/match_checks.rs"]
mod match_checks;
