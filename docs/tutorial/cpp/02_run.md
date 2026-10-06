# 🚀 Run your first TEPL code

Let's build a small optimizer for addition and vector dot products. You will
write operations, rewrite rules, and a starting graph in TEPL, then run the
generated C++ code with egg-c. This guide covers structural rewriting;
[the next guide](03_analyse.md) adds tensor shape and dtype analysis.

The TEPL source is the same as in the [Rust tutorial](../rust/02_run.md).
The C++ application uses CMake and egg-c headers; generated declarations live
in a C++ namespace.

## 🔧 Make `tepl` available

Follow [Build TEPL](01_build.md) first. From the TEPL repository root:

```sh
export PATH="$PWD/bazel-bin:$PATH"
tepl --help
```

This makes `tepl` available from any directory in this terminal. Use the
absolute build path in your shell configuration to keep it available later.

## Set up a C++ project

You can try the finished example in
[`labs/tutorial_cpp`](../../../labs/tutorial_cpp/README.md). From the repository root:

```sh
cmake -S labs/tutorial_cpp -B labs/tutorial_cpp/build -DCMAKE_BUILD_TYPE=Debug
cmake --build labs/tutorial_cpp/build --target pattern_basic -j2
labs/tutorial_cpp/build/pattern_basic
```

The lab generates headers automatically. The steps below assume you are
starting from scratch, with TEPL, CMake, Git, and a C++20 compiler available:

```sh
mkdir -p my_optimizer/src
cd my_optimizer
```

All remaining commands run from your new project root.

## 📁 Add the TEPL files

Initialize a TEPL project, rename its starter files, and add a graph directory:

```sh
tepl init pattern_basic
mv pattern_basic/dialects/your_dialect.tepl pattern_basic/dialects/my_dialect.tepl
mv pattern_basic/rules/your_rule.tepl pattern_basic/rules/add_dot.tepl
mkdir -p pattern_basic/graphs
```

Your TEPL project will contain:

```text
pattern_basic/
├── dialects/
│   └── my_dialect.tepl
├── rules/
│   └── add_dot.tepl
└── graphs/
    └── add_dot_sample.tepl
```

### 1. Define the operations

Replace `pattern_basic/dialects/my_dialect.tepl` with:

```javascript
dialect MyDialect {
    op add(lhs: tensor, rhs: tensor) -> tensor;

    op dot(lhs: tensor, rhs: tensor) -> tensor {
        attrs {
            axis: index;
        }
    }
}
```

The dialect names the operations and their operands. Both take two tensors and
return a tensor. `dot` also carries an `axis` attribute of type `index`.
This basic dialect has no shape or dtype inference. These declarations describe
nodes; they do not implement numerical execution.

### 2. Write the rewrite rules

Replace `pattern_basic/rules/add_dot.tepl` with:

```javascript
from "../dialects/my_dialect.tepl" import MyDialect as d;

rule associate_add {
    (d.add (d.add X Y) Z) => (d.add X (d.add Y Z))
}

rule swap_add {
    (d.add X Y) => (d.add Y X)
}

rule swap_dot {
    (d.dot[@dot] X Y) => (d.dot[@dot] Y X)

    where {
        $supports_axis(@dot.axis);
    }
}
```

The import gives the dialect the short name `d`. Expressions put the operation
first: `(d.add X Y)` means `X + Y`. Capitalized names capture matched tensors,
and `=>` adds an equivalent expression to the e-graph.

The first two rules regroup addition and swap its operands. The last rule
captures the dot attributes as `@dot` and reuses them on the right. Its `where`
block calls a C++ host function, `$supports_axis`, to decide whether swapping
is allowed. This demo assumes semantics where these rewrites are valid.

### 3. Build the starting graph

Create `pattern_basic/graphs/add_dot_sample.tepl`:

```javascript
from "../dialects/my_dialect.tepl" import MyDialect as d;

graph add_dot_sample {
    input x;
    input y;
    input z;

    let a = (d.dot[axis = 0] x y);
    let b = (d.dot[axis = 0] x z);
    let c = (d.dot[axis = 0] y z);
    // An unused node lets us see the host condition reject axis 1.
    let unsupported = (d.dot[axis = 1] x y);
    yield (d.add (d.add a b) c);
}
```

`input` declares external tensors, `let` names nodes, and `yield` selects the
output. The initial expression is `(dot(x, y) + dot(x, z)) + dot(y, z)`.
Inputs have no shape or dtype annotations here. All declared nodes are inserted,
including the unused `unsupported` node.

## ⚙️ Generate C++

Check the TEPL project and generate its headers:

```sh
tepl check pattern_basic
tepl generate pattern_basic --target cpp \
  --cpp-namespace tepl_pattern_basic \
  --out src/tepl_pattern_basic --no-format
```

`check` prints `CheckedProgram`. Generation writes operation types, rule
builders, graph builders, and the `generated.h` entry point.

The target defaults to Rust, so pass `--target cpp`. `--cpp-namespace` names
this generated project's namespace; its default is `tepl_generated`.
The output directory and namespace are separate settings.

`--no-format` lets you generate without installing `clang-format`. Omit that
flag if you want TEPL to format the headers. Run generation again after editing
TEPL files. Add these entries to `.gitignore`:

```gitignore
/build/
/src/tepl_pattern_basic/
```

## Run the optimizer

Download [utils.h](../../../labs/tutorial_cpp/src/utils.h "Download source") and save it as
`src/utils.h` for readable graph and expression output. Create `src/basic.cpp`:

```cpp
#include <eggc/all.hpp>

#include "tepl_pattern_basic/generated.h"
namespace tepl = tepl_pattern_basic;
#include "utils.h"

namespace r = tepl::rules::add_dot;
using Analysis = eggc::NoAnalysis<tepl::OpNode>;

struct Host {
  // Implements TEPL's $supports_axis.
  std::optional<bool> supports_axis(std::uint64_t axis) const {
    return axis == 0;
  }
};

int main() {
  // Build the starting graph without tensor metadata analysis.
  auto graph_def = tepl::graphs::add_dot_sample::graph_add_dot_sample::build();
  eggc::EGraph<tepl::OpNode, Analysis> graph;
  auto built = graph_def.insert_nodes(graph);

  // Build rules; swap_dot uses the host callback.
  std::vector rules{
      r::rule_associate_add::build<Analysis>(),
      r::rule_swap_add::build<Analysis>(),
      r::rule_swap_dot::build<Analysis>(Host{}),
  };

  // Run rewrites to saturation.
  auto report = eggc::run(graph, rules);
  utils::require(report.reason == eggc::StopReason::Saturated,
                 "Rewrites did not saturate");
  std::cout << "Stop reason: Saturated\n";
  utils::print_egraph(graph, built.root);

  // Extract a minimum-depth expression.
  auto [depth, best] =
      eggc::Extractor(graph, eggc::ast_depth_cost<tepl::OpNode>())
          .find_best(built.root);
  std::cout << "Minimum AST depth: " << depth << "\nBest expression:\n"
            << utils::expression(best) << '\n';
  utils::require(depth == 4, "Expected minimum AST depth 4");
}
```

`tepl` is an alias for the generated namespace, rather than another dependency.
Declare it before including `utils.h`, which uses that alias.

`Host` implements `$supports_axis` as a `const` method returning
`std::optional<bool>`. Returning `axis == 0` wraps that boolean as a present
result. `false` and `std::nullopt` both reject a rewrite candidate. The generated
`HostFunctions` concept checks the method signature at compile time; you do not
need to inherit a base class or write a Rust-style trait implementation.

`eggc::NoAnalysis<tepl::OpNode>` corresponds to Rust's unit analysis (`()`).
C++ rule builders default to `TensorAnalysis`, so every basic rule explicitly
uses `build<Analysis>()`. Rules without callbacks need no host argument.

`eggc::run(graph, rules)` modifies `graph` and returns a report. The graph
stays in the application, ready for printing and extraction. The extractor
uses `ast_depth_cost` to select an expression from the yielded root's e-class.
Generated graph builders return values directly; invalid graph or node
construction throws `NodeError`, rather than returning a Rust `Result`.

## Build with CMake

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_optimizer LANGUAGES CXX)

# Fetch only the egg-c header library.
include(FetchContent)
set(BUILD_TESTING OFF)
set(EGGC_BUILD_EXAMPLES OFF)
set(EGGC_BUILD_BENCHMARKS OFF)
FetchContent_Declare(eggc
  GIT_REPOSITORY https://github.com/5yearsKim/egg-c.git
  GIT_TAG 0c28bd5050b85ed10e915b5348c27f760ed31ae5)
FetchContent_MakeAvailable(eggc)

add_executable(pattern_basic src/basic.cpp)
target_link_libraries(pattern_basic PRIVATE eggc::eggc)
target_compile_features(pattern_basic PRIVATE cxx_std_20)
set_target_properties(pattern_basic PROPERTIES CXX_EXTENSIONS OFF)

enable_testing()
add_test(NAME pattern_basic COMMAND pattern_basic)
```

The `eggc::eggc` target supplies its include path. It represents a header
library, so `target_link_libraries` adds no compiled egg-c binary.

Configure, compile, and run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j2
./build/pattern_basic
```

The first configuration fetches the pinned dependency. For an existing egg-c
checkout, add `-DFETCHCONTENT_SOURCE_DIR_EGGC=/path/to/egg-c` to the configure
command. To use Clang, add `-DCMAKE_CXX_COMPILER=clang++` in a fresh build
directory.

This project generates TEPL headers with the earlier CLI command. The reference
lab's [CMakeLists.txt](../../../labs/tutorial_cpp/CMakeLists.txt) also regenerates
headers when TEPL files change.

## 🔎 Read the result

One run produces this excerpt; e-class IDs and tied extraction choices may vary:

```text
Stop reason: Saturated

Saturated e-graph (nodes in each e-class are equivalent):
...
e3
  dot[axis=0](e0, e1)
  dot[axis=0](e1, e0)
...
e6
  dot[axis=1](e0, e1)
...
e8 (root)
  add(e16, e4)
  add(e3, e9)
  add(e4, e16)
  add(e5, e7)
  add(e7, e5)
  add(e9, e3)
...

Minimum AST depth: 4
Best expression:
add(add(dot[axis=0](x, y), dot[axis=0](x, z)), dot[axis=0](y, z))
```

`Saturated` means the rules found no more new equivalences. Each `e` block is
an e-class: its nodes are equivalent, and child IDs refer to other e-classes.
Here, `e0`, `e1`, and `e2` contain inputs `x`, `y`, and `z`. The two nodes in
`e3` show an accepted dot swap. The axis-`1` node in `e6` has no swapped
alternative because the host rejected it. The root class contains several
addition orders.

Input leaves have depth `1`, dots have depth `2`, and two addition levels give
minimum depth **4**. All arrangements of these three dot products tie at that
depth, so retaining the original arrangement is valid.

The shared printer formats expressions directly, including dot attributes.
The Rust basic example uses egg's `RecExpr::pretty`; the display and e-class
IDs can differ while representing the same computation. This example explores
equivalent forms; it does not evaluate tensors or measure execution speed.

Next: [add tensor shape and dtype analysis](03_analyse.md).
