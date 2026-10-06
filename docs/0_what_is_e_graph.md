# 🤔 What is an e-graph?

An **e-graph (equivalence graph)** compactly stores many equivalent expressions.
An **e-class** groups expressions known to have the same value. An **e-node**
represents an operation whose children refer to e-classes, allowing expressions
to share subexpressions.

## 🔄 Why use an e-graph?

<p align="center">
  <img src="../../misc/images/why_e_graph.png" alt="An e-graph keeps a * 4 + a, a * 5, and (a &lt;&lt; 2) + a as equivalent alternatives for a cost model to choose from later" width="720">
</p>

Rewriting can expose several valid alternatives before we know which is cheapest.
For example, consider `a * 4 + a`. For nonnegative integers with exact arithmetic,
both rewrites are valid:

```text
a * 4 + a  →  a * 5
a * 4 + a  →  (a << 2) + a
```

Here, `<< 2` shifts left by two bits, multiplying by four. Fixed-width arithmetic
requires rules that respect its overflow and shift semantics.

A greedy rewrite system may choose one form immediately and discard the other.
But the better choice can depend on the target architecture or later optimizations.

An e-graph keeps the discovered alternatives together:

```text
Equivalent expressions in one e-class:
    a * 4 + a
    a * 5
    (a << 2) + a
```

These expressions share structure in the graph; they are not stored as separate
copies of complete expression trees. Further rewrites can explore any of them.
A **cost model** assigns costs, and an **extractor** selects a low-cost expression
from the represented alternatives.

| Approach | When it chooses a form |
| --- | --- |
| Greedy rewriting | Commits to a replacement at each step. |
| Equality saturation | Keeps equivalent alternatives and extracts a form after search. |

**Equality saturation** repeatedly applies rewrites to an e-graph until no new
equivalences are found or a resource limit is reached. The result depends on the
rules, search limits, and cost model; it need not be globally optimal.

## 🧠 What is e-class analysis?

An expression also has properties that are useful during optimization.
**E-class analysis** computes and stores facts about the value represented by
an e-class. Examples include:

```text
definitely_nonzero = true
sign               = positive
range              = [0, 255]
constant           = 5
```

These facts can determine whether a rewrite is legal. For example, under exact
rational arithmetic:

```text
x / x  →  1    only if x != 0
```

An analysis can record a fact about `x`:

```text
e-class for x:
    definitely_nonzero = true
```

The rewrite checks this fact before applying. If nonzero cannot be proven, it
skips the match. Other numerical systems may need additional conditions.

Analysis computes facts from operations and their operands, then combines those
facts when e-classes merge. Rewrite rules add equivalent expressions; analysis
maintains facts about their shared value.

Constant propagation, range analysis, sign analysis, and tensor shape/dtype
inference all fit this model. For tensor rewrites, metadata helps validate
replacements and reject incompatible matches early.

## 🥚 What is `egg`?

[`egg`](https://github.com/egraphs-good/egg) is a Rust library for e-graphs and
equality saturation. It provides pattern matching, rewriting, e-class analysis,
and extraction so applications can build their own optimizers.

[TEPL](../1_what_is_tepl.md) builds on this workflow with a language for tensor
rewrites. It generates rules and tensor analysis support for egg in Rust or
egg-c in C++.
