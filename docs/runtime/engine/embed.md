# Embed the TEPL engine

The C++ API loads a compiled program, rewrites one concrete graph, and returns
its result and statistics. Compile source with the CLI first:

```sh
bazel-bin/tepl compile examples/projects/transpose -o /tmp/program.json
```

In an application linked to the repository's `//src/rewrite` Bazel target:

```cpp
#include "src/executable/program.h"
#include "src/rewrite/engine.h"
#include "src/rewrite/printer.h"

// json_text contains the compiled file's contents.
auto program = tepl::executable::load(json_text);
tepl::rewrite::Options options;
options.max_steps = 100;
options.trace = true;
auto result = tepl::rewrite::run(program, program.ir().graphs.at(0), options);
auto source = tepl::rewrite::printGraph(program.ir(), result.graph);
```

`run` copies its input. The result contains the graph, rewrite count, stop reason,
optional trace, and resource-limit detail. Stop reasons are `fixed_point`,
`step_limit`, `resource_limit`, and `cancelled`.

Set `options.rule` to execute one named rule. The API also exposes expression-work
and list-size limits, plus a cancellation callback checked between candidate
attempts. Temporary RHS nodes count toward the node limit. Trace entries identify
the rule and node before replacement; node IDs are not stable across rewrites.

The engine depends on the executable loader and shared IR, without depending on
the parser or semantic proof search. To compile source in-process, compiler
lowering is a separate dependency: `//src/core/lowering`.

See [execution behavior](run.md#understand-execution), the
[engine API](../../../src/rewrite/engine.h), and the
[compiled program reference](../../references/program-format.md).
