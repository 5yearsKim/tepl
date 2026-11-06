# TEPL C++ runtime

Shared C++20 execution support for generated egg-c programs. GCC and Clang need
signed `__int128` support; MSVC is deferred. The tested egg-c revision is
`0c28bd5050b85ed10e915b5348c27f760ed31ae5`, pinned with a checksum in
`MODULE.bazel` and the CMake tutorial. The compiler itself does not depend on egg-c.

## Generate and consume

```sh
bazel build //:tepl
bazel-bin/tepl generate examples/projects/transpose --target cpp \
  --cpp-namespace my_ir --out /path/to/app/generated
```

Create the `eggc::eggc` CMake target, then add the shared runtime:

```cmake
add_subdirectory(/path/to/tepl/runtime/cpp tepl-runtime)
target_link_libraries(my_app PRIVATE tepl::runtime)
```

Direct consumers need egg-c and `runtime/cpp/include` in their include paths.
Bazel consumers can depend on `//runtime/cpp:runtime`.

```cpp
#include "generated/generated.h"
auto built = my_ir::graphs::transpose::graph_example::build().value();
auto rules = my_ir::rules::all().value();
auto result = tepl_runtime::runner::run(built.graph, rules);
if (!result) {
  // result.error().kind and .what() describe the failure.
}
```

See the [CMake application](../../labs/tutorial_cpp/README.md) for a complete demo.

## Integration API

`Program` is the generated/runtime concept for typed operations, attributes, and
metadata inference. Public fallible APIs return `EvalResult<T>`; `value()` returns
the value or throws `EvalError`. Internal evaluators propagate that typed error;
arbitrary application/allocation exceptions remain visible.

`graph.analysis.register_input(...)` registers facts before node insertion.
The graph is movable, not copyable. Its stable analysis handle exposes
`check()` and `rejections()`. The e-graph wrapper derives from egg-c's graph;
explicit egg-c template arguments or a `BaseGraph<Program>&` cast work with APIs
whose template deduction requires the base type. Diagnostic handles in analysis
data let callbacks report errors without a global registry.

`runner::run` returns checked results and accepts egg-c `RunOptions`. Custom
witness unions and repairs may be absent from per-rule counters; saturation
still tracks graph revisions. Extraction and cost models are application choices.

The [shared execution contract](../../design/code-generation.md#shared-execution-contract)
defines fact merging, evaluation order, witness matching, and validation before
insertion. Runtime ABI 1 is checked with `static_assert`; regenerate output when
updating it. C++ and Rust ABI versions are independent.

## Validate

```sh
bazel test //tests:codegen_cpp_test //tests:cpp_codegen_test //runtime/cpp:values_test
./tools/test_codegen_cpp.sh
```

The script uses GCC and available Clang; `CXX` selects one compiler and
`EGGC_SOURCE_DIR` selects a local egg-c checkout. Coverage includes debug and
optimized UBSan programs, independent native comparisons, source/JSON parity,
standalone headers, namespaces, and multiple translation units.
