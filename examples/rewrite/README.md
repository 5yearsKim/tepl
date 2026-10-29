# Compile and rewrite graphs

This project connects portable executable programs, the greedy engine, and
Rust/C++ generated optimizers.

| File | Demonstrates |
| --- | --- |
| [simple.tepl](simple.tepl) | Remove nested copies from a concrete graph |
| [auto.tepl](auto.tepl) | Synthesize transpose attributes using the semantic dialect |
| [rules/transpose.tepl](rules/transpose.tepl) | Explicit permutation composition and an identity guard |
| [graphs/transpose.tepl](graphs/transpose.tepl) | Shared tensor nodes used by generated applications |

Run from the repository root:

```sh
bazel build //:tepl
bazel-bin/tepl compile examples/rewrite/simple.tepl --out /tmp/simple.json
bazel-bin/tepl validate /tmp/simple.json
bazel-bin/tepl rewrite /tmp/simple.json --out /tmp/rewritten.tepl
bazel-bin/tepl generate examples/rewrite --out /tmp/app/src/generated
```

`tepl.toml` defines the project root. `auto.tepl` explicitly imports the neighboring
semantic dialect; the other rule/graph directories share the local tensor dialect.
The [Rust application](../../labs/rust-egg/README.md) consumes the generated
transpose graph and rules. Use `--target cpp` with the
[C++ runtime](../../runtime/cpp/README.md) for egg-c output.
