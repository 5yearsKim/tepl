# TEPL C++ runtime

Shared C++20 execution support for generated egg-c programs. Generated headers
contain program-specific operation types, attribute schemas, metadata functions,
rules, and graph builders. This library contains reusable execution algorithms.
It does not depend on the TEPL compiler, generated programs, or compiled JSON.

GCC and Clang with signed `__int128` are supported. MSVC is deferred. The tested
egg-c revision is `0c28bd5050b85ed10e915b5348c27f760ed31ae5`, pinned with a download
checksum in `MODULE.bazel` and the CMake tutorial.

## Generate and consume

```sh
bazel build //:tepl
bazel-bin/tepl generate examples/semantic --semantic-policy algebraic \
  --target cpp --cpp-namespace my_ir --out /tmp/my_app/generated
```

Add egg-c and `runtime/cpp/include` to the application's include paths. With
CMake, create the `eggc::eggc` target first, then:

```cmake
add_subdirectory(/path/to/tepl/runtime/cpp tepl-runtime)
target_link_libraries(my_app PRIVATE tepl::runtime)
```

Bazel consumers can depend on `//runtime/cpp:runtime`. The compiler itself does
not depend on egg-c.

```cpp
#include "generated/generated.h"

// Graph/rule paths follow their source module paths.
auto built = my_ir::graphs::transpose::graph_example::build().value();
auto rules = my_ir::rules::all().value();
auto result = tepl_runtime::runner::run(built.graph, rules);
if (!result) {
  // result.error().kind and .what() describe the failure.
}
```

See the fully executable [tutorial](../../labs/tutorial_cpp/README.md).

## Contracts

`tepl_runtime::Program` is the sole generated/runtime interface. It defines
operation IDs/names, arity, attribute validation, and shape/dtype inference.
Generated `Program` types satisfy this concept at compile time. Runtime ABI 1 is
checked with `static_assert`; regenerate output when changing runtime versions.
C++ and Rust ABI versions are independent.

`EvalResult<T>` represents public fallible results. `value()` returns the value
or throws the typed `EvalError`. Internal evaluators use the same error to
propagate failures to result boundaries; arbitrary application or allocation
exceptions are not swallowed. Error kinds match Rust: `Unavailable`, `Invalid`,
`Conflict`, and `Limit`.

`Value` supports checked signed 128-bit expression arithmetic, finite floating
values, lists, logical axes, tensors, typed descriptors, and opaque payloads.
Schema fields retain concrete C++ types. C++ keywords receive a trailing
underscore; collisions with existing spellings are diagnosed before writing.
Generated namespaces must already be valid C++ names.

`TensorFacts` contains independent shape and dtype facts. Unknown is no evidence;
known facts refine unknown facts; differing known facts produce a conflict.
Register input facts with `graph.analysis.register_input(...)` before inserting
input nodes. A graph owns a stable shared analysis handle and is movable, not
copyable. Input metadata cannot change after insertion.

The e-graph wrapper derives from egg-c's `EGraph`. Explicit egg-c template
arguments or a cast to `tepl_runtime::BaseGraph<Program>&` expose it to APIs whose
template deduction requires the base type. Analysis data carries a diagnostic
handle because egg-c does not expose its analysis object to callbacks.

Attribute-aware matching keeps concrete node witnesses paired with their tensor
captures. Search emits distinct tensor substitutions and honors cancellation and
sink rejection. Each application batch rebuilds and rematches its root once;
applications recheck metadata and guards against the current graph. Source
conditions run before derivations, derivation locals have their own scope, and
dependent conditions retain the original `where` locals.

The entire replacement is prepared and inferred before insertion. A known root
shape/dtype disagreement rejects it. Missing required metadata skips a candidate;
invalid evaluation records one diagnostic per rule in `graph.analysis.rejections()`.
Conflicts and resource limits stop checked execution. Use `runner::run` for typed
error propagation. The runner accepts egg-c `RunOptions`; extraction and cost
models remain application choices. Internal witness unions and repairs can be
absent from egg-c per-rule counters; saturation still tracks graph revisions.
Validation before insertion does not provide
rollback for allocation failures during insertion.

## Validate

```sh
bazel test //tests:codegen_cpp_test //tests:cpp_codegen_test //runtime/cpp:values_test
./tools/test_codegen_cpp.sh
```

The integration script uses GCC and available Clang by default; set `CXX` to
select one compiler and `EGGC_SOURCE_DIR` for a local egg-c checkout. It compares
source/JSON output, builds sample/semantic projects, checks standalone headers and
multiple translation units, and compares current synthesis fixtures with the
independent native evaluator. Debug and optimized UBSan builds exercise both
generated programs and runtime numeric boundaries.
