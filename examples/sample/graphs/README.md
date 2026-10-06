# 🧩 Concrete graphs

These declarations build initial computation graphs. TEPL discovers `graphs/`
alongside `dialects/` and `rules/`, and generates Rust and C++ constructors.

| Example | Syntax shown | Output type |
| --- | --- | --- |
| [direct_yield.tepl](direct_yield.tepl) | Typed input, nested expressions, direct `yield` | `f32[3, 2]` |
| [bindings.tepl](bindings.tepl) | Multiple inputs, sequential `let` bindings, shared values, named output | `f32[3, 2]` |
| [attributes.tepl](attributes.tepl) | Multiple attribute fields, lists, variadic operands | `f32[8, 18]` |
| [literals.tepl](literals.tepl) | Rank-zero inputs, typed literals, operation imports, literal output | `f32[]` and `i32[]` |
| [untyped_inputs.tepl](untyped_inputs.tepl) | Inputs with host-supplied metadata | Depends on inputs |

## Grammar conventions

- `graph name { ... }` declares an initial computation graph.
- `input X: f32[2, 3];` declares an external input with concrete metadata.
  `input X;` leaves its shape and dtype to the host.
- `let name = expression;` constructs and names a value. References must name
  an input or an earlier binding; repeated references share the same value.
- `(t.op[field = value, ...] operands...)` constructs an operation with explicit
  attributes. Field order is irrelevant. Unknown or duplicate fields are errors;
  required fields must be supplied unless the dialect declares a default.
- Exactly one final `yield expression;` selects the output. It accepts a name,
  literal, or nested operation expression.

Operations use the existing dialect definitions. Each operation expression maps
to an `OpNode`; input shape/dtype annotations become separate analysis metadata.
The host supplies tensor values, selects rewrite rules, and runs the optimizer.

## 🚀 Build and insert

Generate the sample project from the repository root:

```sh
bazel-bin/tepl generate examples/sample --target rust --out my_app/src/generated
```

Each graph exposes `graphs::FILE::graph_NAME::build()`. In Rust, typed inputs
configure the generated tensor analysis before any nodes are inserted:

```rust
let definition = generated::graphs::bindings::graph_shared_result::build()?;
let (egraph, built) = definition.into_egraph()?;
let root = built.root;
let x = built.get("X").unwrap();
```

For inputs without annotations, insert into an existing e-graph of any analysis
type:

```rust
let definition = generated::graphs::untyped_inputs::graph_host_typed_add::build()?;
let built = definition.insert_into(&mut egraph)?;
```

Supply host metadata before creating a tensor-analysis graph:

```rust
use generated::analysis::TensorInfo;
use generated::DType;

let definition = generated::graphs::untyped_inputs::graph_host_typed_add::build()?
    .with_input_info("X", TensorInfo { shape: vec![2, 3], dtype: DType::F32 })?
    .with_input_info("Y", TensorInfo { shape: vec![2, 3], dtype: DType::F32 })?;
let (egraph, built) = definition.into_egraph()?;
```

`into_egraph_with(factory)` passes input bindings to a custom analysis factory.
For example, `into_egraph_with(TensorAnalysis::new)` uses the generated analysis.
The host still chooses and applies rewrite rules.

C++ exposes the same helpers under `generated::graphs::FILE::graph_NAME`:

```cpp
auto definition = tepl_generated::graphs::bindings::graph_shared_result::build();
auto [egraph, built] = definition.into_egraph();
auto root = built.root;
auto x = built.get("X").value();
```

## 🔎 Validation and metadata

`tepl check` validates names, references, arity, literal types, and attribute
schemas. `build()` also evaluates available shape and dtype programs. Invalid
known metadata fails before insertion; absent inference remains unknown.
Rust reports `NodeError` through `Result`; C++ throws `NodeError`.

`nodes()`, `inputs()`, `get(name)`, and `root()` expose the prepared definition.

Input dtype and shape are optional independently: `input X: f32;` or
`input X: [2, 3];`. The tensor analysis registers complete shape/dtype pairs.
Supply missing facts with `with_input_info`; conflicting declared facts are
errors. `insert_into` accepts unannotated inputs; use `into_egraph` or
`into_egraph_with` for inputs carrying metadata.

Bindings preserve sharing. All declared nodes, including unused bindings, are
inserted. Prepared child IDs are local indices; insertion remaps them to the
host e-graph. After later unions, use `egraph.find(id)` to canonicalize handles.

Attributes support `index`, `i64`, `bool`, `string`, `dtype`, `precision`, nested
lists, and optional values (`none`). Missing optional fields become `None`;
empty-list defaults become empty lists. Precision values are `default`, `high`,
and `highest`. Attribute order does not affect the generated node.

Opaque payloads such as regions and elements need host construction. Graph
syntax currently accepts identifier input names and one output; it does not
serialize arbitrary `OpNode` values or cyclic graphs.
