# Tensor analysis

Generated C++ programs provide the same independent shape/dtype facts and checked
rewrite behavior as Rust. `TensorAnalysis` evaluates generated metadata functions;
explicit shape programs take precedence over logical-axis contracts.

For manual construction, register facts before inserting inputs:

```cpp
ir::EGraph graph;
graph.analysis.register_input("X",
    tepl_runtime::TensorFacts::known({2, 3}, ir::DType::F32)).value();
auto x = graph.add(ir::OpNode::input("X"));
```

Each fact is unknown, known, or conflicting. Unknown facts can be refined by known
facts; conflicting known shapes or dtypes stop checked execution. Equal extents
do not make logical axes identical.

Conditions with missing required metadata skip their candidate. Invalid guard or
replacement evaluation records a rule rejection without stopping unrelated rules.
The runtime validates the whole RHS and rejects known output shape/dtype changes
before insertion. Source conditions run before descriptor derivations; conditions
using derived descriptors run afterward.

Inspect `graph.analysis.rejections()` for rule failures and use
`tepl_runtime::runner::run` to propagate analysis conflicts and resource errors.
The [runtime reference](../../../../runtime/cpp/README.md) describes the full contract.
Application host functions and custom analysis adapters are not part of this API.
