# Internal compiler operations

Internal operations describe proof expressions and compiler-generated metadata.
They have their own `Intrinsic` identities and arity catalog. Adding an entry
here never makes it callable from TEPL source.

Public expression functions live in [the built-in catalog](../catalog.def).
Public axes functions live in [the axes catalog](../axes.cc). An axes
function may explicitly target an internal operation; its source availability
and signature still come from the axes catalog.

Symbolic IR retains textual operation names for formatting, structural keys,
and proof laws. Its validator resolves those names through these catalogs.
The native execution boundary resolves them to typed IDs before dispatch.

## Implementation map

| Responsibility | Location |
| --- | --- |
| Identities, names, arities | `catalog.def`, `catalog.h`, `catalog.cc` |
| Native execution | `src/rewrite/intrinsics.cc` |
| Shared list/integer helpers | `src/rewrite/builtins/` |
| Symbolic construction and simplification | `src/core/semantics/symbolic/expr.cc` |
| Proof obligations and validation | `src/core/semantics/context.cc` |
| Axis normalization and set reasoning | `src/core/semantics/index_maps/axis_normalize.cc`, `axis_set_proof.cc` |
| Generated proof operations | `templates/{cpp,rust}/src/builtins/semantic.*` and backend verification emitters |

`axis_partition` requires an injective, complete position sequence.
`inverse` inverts such a permutation. `positions` requires distinct inputs and
requires every requested axis to exist. `membership_positions` and
`matched_positions` select corresponding positions on opposite sides of an
intersection. Axis identity includes owner, group, and member; equal extents do
not imply equal axes.

Ordered set operations preserve input traversal order. `replace_axes` requires
an injective mapping of equal-length source and target sequences. `place_axes`
inserts the left sequence at valid unique positions and fills the remaining
positions from the right sequence.

`concat` permits zero or more operands in structural IR. The public source
function requires at least two. `if`, `&&`, and `||` are evaluated lazily by the
evaluator before ordinary argument evaluation. Dtype constants use the
validated `dtype_<name>` form; operators use the shared language operator IDs.
Neither category is resolved as a public function.

<!-- BEGIN BUILTIN CATALOG -->

## Internal operation inventory

Generated from `catalog.def`. These spellings identify symbolic IR operations;
they do not expose source functions. Source axes functions may explicitly lower
to these operations through the axes catalog.

| Operation | Argument count |
| --- | --- |
| `concat` | 0 or more |
| `index` | 2 |
| `if` | 3 |
| `empty_axes` | 0 |
| `axis_partition` | 2 |
| `inverse` | 1 |
| `order_positions` | 1 |
| `offset_positions` | 2 |
| `offset_extents` | 2 |
| `scale_extents` | 2 |
| `extent_add` | 2 |
| `extent_sub` | 2 |
| `extent_mul` | 2 |
| `extent_floor_div` | 2 |
| `extent_le` | 2 |
| `is_valid_extents` | 1 |
| `is_positive_extents` | 1 |
| `distinct_axes` | 1 |
| `axis_extents` | 1 |
| `positions` | 2 |
| `membership_positions` | 2 |
| `matched_positions` | 2 |
| `union_axes` | 0 or more |
| `unique_axes` | 1 |
| `ordered_complement` | 2 |
| `filter_axes` | 2 |
| `intersection_axes` | 2 |
| `difference_axes` | 2 |
| `same_set` | 2 |
| `disjoint` | 2 |
| `subset` | 2 |
| `replace_axes` | 3 |
| `place_axes` | 3 |

<!-- END BUILTIN CATALOG -->
