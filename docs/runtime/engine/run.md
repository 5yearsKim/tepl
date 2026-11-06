# Run rewrites

The TEPL engine applies rules directly to a concrete graph and exports the
result as readable TEPL. For a guided example, start with
[your first rewrite](../../getting-started/first-rewrite.md).

Run these commands from the repository root after [building TEPL](../../getting-started/install.md).

## Run a project

```sh
bazel-bin/tepl rewrite examples/projects/transpose --trace -o /tmp/result.tepl
bazel-bin/tepl check /tmp/result.tepl
```

The project loads `dialects/`, `rules/`, and `graphs/`. This example composes two
transposes and removes the resulting identity transpose.

## Select rules and graphs

A graph does not import its rules. You can select them separately:

```sh
bazel-bin/tepl rewrite examples/projects/transpose/graphs/transpose.tepl \
  --rules examples/projects/transpose/rules --rule compose_transposes --trace
```

`--rules` accepts a file or directory. `--rule` selects one named rule from the
loaded rules; omitting it executes all of them. All loaded source rules are
checked. Use `--graph <name>` when the input contains several graphs.

## Reuse compiled rules

```sh
bazel-bin/tepl compile examples/projects/transpose -o /tmp/program.json
bazel-bin/tepl validate /tmp/program.json
bazel-bin/tepl rewrite /tmp/program.json --trace -o /tmp/result.tepl
```

The JSON contains the rules and operation contracts. Execution does not parse
source or repeat attribute synthesis. See the [format reference](../../references/program-format.md).

## Understand execution

Each iteration visits reachable nodes in node-ID order and rules in compiled
order. The first effective replacement is committed, then scanning restarts.
Repeated captures require the same graph node. Replacing a shared node updates
all its uses; unaffected sharing is preserved.

Candidates must pass capture constraints, conditions, derivations, and replacement
metadata checks. Failed candidates leave the graph unchanged. Known root shape
and dtype must be preserved; missing metadata can prevent constrained matches.

The engine removes dead operations but retains input declarations. It does not
compare alternatives using a cost model. Use an [e-graph](../egraphs/overview.md)
for that workflow.

## Limit a run

```sh
bazel-bin/tepl rewrite examples/projects/transpose --max-steps 100 --max-nodes 10000
```

Defaults are 1,000 rewrites and 1,000,000 nodes. Opposing rules can cycle, so rule
selection and limits matter. Exit code `0` means a fixed point; `3` means a limit
or cancellation stopped execution with the current graph exported.

The CLI writes the graph to stdout or `-o`, and statistics and trace to stderr.
Exported source can contain absolute dialect imports; rechecking it requires
those dialect files. The compiled JSON remains executable without them.

Next: [embed the engine](embed.md), or look up [CLI options](../../references/cli.md#rewrite).
