# Your first rewrite

Swap a matrix's axes twice, then simplify the graph back to its input.
You need the [TEPL CLI](install.md); run these commands from the repository root.

## Start with a graph

The [tutorial project](../../examples/projects/transpose/README.md) contains three parts:

```text
examples/projects/transpose/
├── tepl.toml
├── dialects/tensor.tepl
├── rules/transpose.tepl
└── graphs/transpose.tepl
```

`tepl.toml` marks the project root. Imports such as `"dialects/tensor.tepl"`
resolve from that root.

The graph in `graphs/transpose.tepl` is:

```tepl
from "dialects/tensor.tepl" import Tensor as t

graph example {
    input X: f32[2, 3]
    let swapped = (t.transpose[permutation = [1, 0]] X)
    yield (t.transpose[permutation = [1, 0]] swapped)
}
```

`[1, 0]` swaps the two axes: the intermediate shape is `[3, 2]`, and swapping
again returns the original arrangement. `yield` selects the graph's output.

## Read the rules

`rules/transpose.tepl` imports the same dialect and declares:

```tepl
from "dialects/tensor.tepl" import Tensor as t

rule compose_transposes {
    (t.transpose (t.transpose X)) => (t.transpose X)
}

rule remove_identity {
    X: [Dims...]
    (node: t.transpose X) => X
    where node.attrs.permutation == range(len(Dims))
}
```

The first rule combines two transposes. TEPL infers the replacement permutation
from the dialect's semantics; here it becomes `[0, 1]`.

The second removes a transpose only when its permutation leaves every axis in
place. `Dims...` captures the input dimensions, `node.attrs` reads the matched
operation's attributes, and `where` restricts the match.

## Check and run

```sh
bazel-bin/tepl check examples/projects/transpose
bazel-bin/tepl rewrite examples/projects/transpose --trace -o /tmp/result.tepl
bazel-bin/tepl check /tmp/result.tepl
```

`check` prints a `CheckedProgram`, including semantic verification results.
`rewrite --trace` reports the two successful rule applications:

```text
Successfully applied rule 'compose_transposes'.
Successfully applied rule 'remove_identity'.
```

The output file contains:

```tepl
graph example {
    input X: f32[2, 3]

    yield X
}
```

Both transposes are gone. TEPL changed the computation graph without evaluating
any tensor elements.

Next: learn [dialects, rules, and graphs](../language/core-concepts.md), or see
[why these rules are valid](../language/semantics.md).
