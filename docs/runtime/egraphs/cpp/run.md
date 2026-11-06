# Run generated rules with egg-c

C++ generation consumes the same verified executable program as Rust. The
application depends on the shared C++20 runtime and egg-c.

After [setting up the dependencies](setup.md), run the complete example from the repository root:

```sh
bazel build //:tepl
cmake -S labs/tutorial_cpp -B /tmp/tepl-tutorial-cpp
cmake --build /tmp/tepl-tutorial-cpp
ctest --test-dir /tmp/tepl-tutorial-cpp --output-on-failure
```

The [tutorial source](../../../../labs/tutorial_cpp/program.tepl) declares transpose
metadata and axis contracts, composes two transposes with synthesized attributes,
and removes identity transposes. CMake generates headers in its build directory
with `--target cpp --cpp-namespace tutorial_ir`.

The optimized expression is the original input `X`: composing two swaps gives
an identity transpose, which the second rule removes. The e-graph retains the
original form too; extraction chooses the cheaper expression.

The [application](../../../../labs/tutorial_cpp/src/main.cpp) uses the generated API:

```cpp
#include "generated/generated.h"
namespace ir = tutorial_ir;
auto built = ir::graphs::program::graph_sample::build().value();
auto rules = ir::rules::all().value();
auto report = tepl_runtime::runner::run(built.graph, rules).value();
```

`build()` registers input metadata before node insertion. `rules::all()` returns
egg-c rewrites; individual rules expose `definition().build()`. Fallible APIs
return `EvalResult<T>`; inspect the result or use `value()` to propagate a typed
`EvalError`. Pass egg-c `RunOptions` to the checked runner to configure limits.

Generated headers use relative includes and a configurable namespace. Add the
shared runtime with CMake's `tepl::runtime` target or the runtime and egg-c include
paths. See the [runtime reference](../../../../runtime/cpp/README.md) for manual
construction, error handling, and the generated/runtime contract.

Next: [supply tensor metadata](analysis.md) for manually constructed graphs.
