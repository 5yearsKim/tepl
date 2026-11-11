# Labs

Labs contain reproducible compiler experiments and application demos. Runnable
TEPL language examples are indexed in [examples](../examples/README.md);
mathematical arguments and finite theorem checks are in
[research](../research/README.md).

| Lab | Purpose and status |
| --- | --- |
| [einsum_benchmark](einsum_benchmark/README.md) | 16-input Einsum contraction-path rewrites and inferred-axis measurements |
| [rust-egg](rust-egg/README.md) | Maintained Rust/egg application with generated transpose rules |
| [tutorial_cpp](tutorial_cpp/README.md) | Maintained C++20/egg-c CMake application |
| [tutorial_rust](tutorial_rust/README.md) | Historical tutorial fixtures using retired descriptor syntax |
| [attribute_inferences](attribute_inferences/README.md) | Attribute-inference validation, scalability, XLA integration, and workload analysis |

An experiment README should identify scope, reproduction commands, primary
reports, and result locations. Detailed measurements belong in `reports/` with
raw `results/`, inputs, scripts, and tests kept beside them. Use each recorded
revision, compiler/source hashes, and protocol when interpreting outcomes.

Generated catalogs and reports are maintained through their generators.
Historical measurements are not current support promises; source changes or
relocation can invalidate a frozen run's integrity check. Preserve recorded
numbers and hashes rather than rewriting them to match a newer checkout.
