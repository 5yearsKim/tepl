# Concrete graph examples

Graphs declare input metadata, operation attributes, shared values, and one
result. Both backends generate a constructor for each graph.

| File | Demonstrates |
| --- | --- |
| [direct_yield.tepl](direct_yield.tepl) | Typed input and nested result |
| [bindings.tepl](bindings.tepl) | Sequential `let` bindings and sharing |
| [attributes.tepl](attributes.tepl) | Lists, multiple fields, variadic operands |
| [literals.tepl](literals.tepl) | Rank-zero inputs and typed literal results |
| [untyped_inputs.tepl](untyped_inputs.tepl) | Inputs with unknown metadata |

`input X: f32[2,3]` declares facts; shape and dtype may also be omitted
independently. References name inputs or earlier bindings. Operation attributes
use `field=value`; required fields need values unless defaults apply. Exactly
one final `yield` selects a name, literal, or nested operation.

## Generated construction

Generate the sample project as described in its [README](../README.md).
`graphs::FILE::graph_NAME::build()` returns a built graph, root, and binding map.
Input facts are registered before insertion. For Rust:

```rust
let built = generated::graphs::bindings::graph_shared_result::build()?;
let root = built.root;
let x = built.bindings["X"];
let graph = built.graph;
```

For C++:

```cpp
auto built = sample_ir::graphs::bindings::graph_shared_result::build().value();
auto root = built.root;
auto x = built.bindings.at("X");
```

After unions, canonicalize handles with the e-graph's `find`. Declared bindings
preserve sharing, including nodes unused by the final result.

## Metadata and errors

Shape and dtype are independent unknown/known/conflicting facts. Missing contracts
or metadata produce unknown facts; invalid inference or conflicts are errors.
Register external facts before inserting inputs. For dynamically supplied facts,
construct through the runtime node/analysis API rather than updating already
inserted unknown inputs. The retired prepared-definition and `with_input_info`
interfaces are not part of the current generated API.

The [Rust](../../../../runtime/rust/README.md) and
[C++](../../../../runtime/cpp/README.md) guides cover manual construction, checked
runners, typed attributes, and error handling. Opaque region/element payloads
require application-provided typed values.
