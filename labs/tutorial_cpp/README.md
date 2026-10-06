# TEPL C++ tutorial

This lab mirrors [`tutorial_rust`](../tutorial_rust) using the C++20
[egg-c](https://github.com/5yearsKim/egg-c) library. The TEPL files are identical;
the application and shared printer are written in C++.

| Binary | TEPL project | Application |
| --- | --- | --- |
| `pattern_basic` | [`pattern_basic`](pattern_basic) | [`src/basic.cpp`](src/basic.cpp) |
| `pattern_analyze` | [`pattern_analyze`](pattern_analyze) | [`src/analyze.cpp`](src/analyze.cpp) |

## Build and run

You need Bazel, CMake 3.20 or newer, Git, and a C++20 compiler. Generated code
requires GCC or Clang with `__int128` support; MSVC is currently unsupported.

From the repository root:

```sh
bazel build //:tepl
cmake -S labs/tutorial_cpp -B labs/tutorial_cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build labs/tutorial_cpp/build -j2
labs/tutorial_cpp/build/pattern_basic
labs/tutorial_cpp/build/pattern_analyze
```

CMake fetches egg-c at the tested revision
`0c28bd5050b85ed10e915b5348c27f760ed31ae5`. The first configuration needs
network access. To use an existing checkout instead:

```sh
cmake -S labs/tutorial_cpp -B labs/tutorial_cpp/build \
  -DEGGC_SOURCE_DIR=/path/to/egg-c
```

Use `-DTEPL_EXECUTABLE=/path/to/tepl` if your compiler is elsewhere.

The build generates C++ headers automatically and regenerates them when TEPL
files change. Generated sources live in `src/tepl_pattern_basic/` and
`src/tepl_pattern_analyze/`; both directories and build outputs are ignored
by Git. Each generated project has its own namespace.

Both binaries check their expected results. Run them together with:

```sh
ctest --test-dir labs/tutorial_cpp/build --output-on-failure
```

## Basic example

[`my_dialect.tepl`](pattern_basic/dialects/my_dialect.tepl) defines `add` and
`dot`, with an `axis` attribute for dot.
[`add_dot.tepl`](pattern_basic/rules/add_dot.tepl) defines addition
reassociation, addition commutativity, and a guarded dot swap.
[`add_dot_sample.tepl`](pattern_basic/graphs/add_dot_sample.tepl) builds:

```text
add(add(dot(x, y), dot(x, z)), dot(y, z))
```

`Host::supports_axis` implements `$supports_axis` and permits only axis `0`.
Returning `false` or `std::nullopt` rejects a candidate. The graph also contains
an unused axis-`1` dot to exercise that condition.

The application creates an `EGraph` with `eggc::NoAnalysis<tepl::OpNode>` and
builds rules with `build<Analysis>()`. C++ rule builders default to
`TensorAnalysis`, so the basic example selects structural analysis explicitly.

`eggc::run(graph, rules)` modifies the graph and reports why it stopped. After
saturation, the program prints the e-graph and extracts a minimum-depth
expression with `eggc::ast_depth_cost<tepl::OpNode>()`.

The minimum AST depth is **4**, counting leaves as depth `1`. Equivalent
arrangements tie at this depth, so extraction may retain the original order.
AST depth measures expression structure, rather than execution speed.

## Tensor analysis example

[`my_dialect.tepl`](pattern_analyze/dialects/my_dialect.tepl) adds shape and
dtype functions. Addition requires equal shapes; matrix dot maps `[M, K]` and
`[K, N]` to `[M, N]`. Both require matching numeric dtypes.

[`add_dot_sample.tepl`](pattern_analyze/graphs/add_dot_sample.tepl) declares
`x: i32[2, 3]` and `y, z: i32[3, 4]`, then yields
`add(dot(x, y), dot(x, z))`. The application configures analysis before
inserting nodes:

```cpp
auto graph_def = tepl::graphs::add_dot_sample::graph_add_dot_sample::build();
auto bindings = graph_def.input_bindings();
auto analysis = tepl::TensorAnalysis(std::move(bindings));
eggc::EGraph<tepl::OpNode, tepl::TensorAnalysis> graph{std::move(analysis)};
auto built = graph_def.insert_nodes(graph);
```

[`add_dot.tepl`](pattern_analyze/rules/add_dot.tepl) keeps all three basic
rules and adds an abstract factoring rule. Its tensor declarations bind the
shared dtype `T` and dimensions `M`, `K`, and `N`. The concrete
`factor_small_dot` rule inherits that pattern and adds `K <= 8`.

The host rejects axis-`1` swaps because matrix multiplication generally does
not commute. Factoring uses matching dot axes and preserves the output type.
This example assumes wrapping `i32` arithmetic.

The binary prints input and intermediate metadata, the saturated graph, and
the expressions before and after minimum-size extraction:

```text
  Factored add(y, z): i32[3, 4]
Before: add(dot[axis=1](x, y), dot[axis=1](x, z)) (AST size 7)
After:  dot[axis=1](x, add(y, z)) (AST size 5)
Output: i32[2, 4]
```

AST size counts expression nodes, including repeated inputs. Factoring reduces
two dots to one while preserving `i32[2, 4]`. Either order of `y` and `z` may
be extracted. The program rewrites expressions and infers types; it does not
evaluate tensor values.

## Reading the graph

Both binaries use [`src/utils.h`](src/utils.h). Each aliases its generated
namespace as `tepl` before including the shared printer. The printer lists
equivalent nodes on separate lines and marks the output with `(root)`.
The analysis example also shows each e-class's tensor type:

```text
e5: i32[2, 4] (root)
  add(e3, e4)
  add(e4, e3)
  dot[axis=1](e0, e7)
```

Each `eN` refers to a child e-class. The three alternatives above represent
the same output. E-class IDs and equally optimal expressions can differ
between egg-c and Rust egg.
