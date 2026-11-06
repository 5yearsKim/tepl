# Design philosophy

TEPL centers the rewrite: its graph structure, applicability, and replacement
should be readable together.

## Keep the transformation visible

Patterns and replacements use the same expression syntax. Tensor constraints,
conditions, and attribute derivations stay close to the rule. Use
[reusable rules](language/reusable-rules.md) when they clarify a shared transformation.

## Make meaning explicit

Dialects define the operation vocabulary and its contracts. Shape/dtype inference
supports metadata checks; axis and value contracts enable supported
[semantic verification](language/semantics.md). Applications choose a numerical
policy consistent with their execution semantics.

## Automate bookkeeping

Pure builtins and descriptor constructors express metadata calculations.
[Axis attribute inference](language/axis-inference.md) constructs supported
replacement mappings without moving the mathematical rule into host code.

## Keep execution a choice

A checked program can run in the [TEPL engine](runtime/engine/run.md) or generate
code for [e-graph integration](runtime/egraphs/overview.md). Host applications
choose rules, resource limits, and, for e-graphs, extraction costs.
