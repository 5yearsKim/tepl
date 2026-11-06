#!/usr/bin/env python3
"""Exhaustive small-model check of the draft ordered-contraction theorem.

This is a standalone research experiment, not TEPL's inference implementation.
The oracle enumerates dot axis pairings, batch/contract choices, and batch orders.
It compares exact index partitions (read-once tensors, all dimensions >= 2).
No random numerical testing or third-party packages are used.
"""

import argparse
import json
from functools import lru_cache
from itertools import combinations, permutations, product


def is_transpose(tree):
    return not isinstance(tree, int) and tree[0] == "transpose"


def leaf_network(ports):
    return tuple((p,) for p in ports), tuple(ports)


def apply_dot(left, right, pairs, batch_order):
    """Execute concrete positional attributes on an index partition.

    pairs contains (left axis, right axis, is_batch); batch_order is the
    ordered list of left batch axes. Contracted classes remain in the partition
    but disappear from the output list.
    """
    lp, lo = left
    rp, ro = right
    parent = {p: p for block in lp + rp for p in block}

    def root(p):
        while parent[p] != p:
            p = parent[p]
        return p

    def merge(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    for block in lp + rp:
        for p in block:
            merge(block[0], p)
    for i, j, _ in pairs:
        merge(lo[i], ro[j])
    blocks = {}
    for p in parent:
        blocks.setdefault(root(p), []).append(p)
    partition = tuple(sorted(tuple(sorted(v)) for v in blocks.values()))
    used_left = {i for i, _, _ in pairs}
    used_right = {j for _, j, _ in pairs}
    output = (
        tuple(root(lo[i]) for i in batch_order)
        + tuple(root(p) for i, p in enumerate(lo) if i not in used_left)
        + tuple(root(p) for j, p in enumerate(ro) if j not in used_right)
    )
    return partition, output


def all_joins(left, right):
    """All legal dot_general attributes, modulo contracting-list order.

    Contracting-list reorderings have identical partitions and exact arithmetic
    semantics. Paired dimensions are all two, so every pairing is shape-legal.
    """
    lo, ro = left[1], right[1]

    def match(i, used, pairs):
        if i == len(lo):
            yield pairs
            return
        yield from match(i + 1, used, pairs)
        for j in range(len(ro)):
            if j not in used:
                for is_batch in (False, True):
                    yield from match(
                        i + 1, used | {j}, pairs + [(i, j, is_batch)]
                    )

    for pairs in match(0, set(), []):
        batches = [i for i, _, b in pairs if b]
        for order in permutations(batches):
            yield apply_dot(left, right, pairs, order)


@lru_cache(None)
def all_networks(tree, leaves):
    if isinstance(tree, int):
        return frozenset([leaf_network(leaves[tree])])
    if is_transpose(tree):
        return frozenset(
            (partition, order)
            for partition, output in all_networks(tree[1], leaves)
            for order in permutations(output)
        )
    return frozenset(
        result
        for left in all_networks(tree[0], leaves)
        for right in all_networks(tree[1], leaves)
        for result in all_joins(left, right)
    )


def construct(tree, leaves, target):
    """Boundary/order solver; returns a concrete reconstructed network or None.

    The target's partition determines logical axes. At a transpose, no output
    ordering obligation needs to propagate to its child. At a dot, required
    axes must follow batch, left-free, right-free order.
    """
    partition, target_output = target
    logical = {p: block[0] for block in partition for p in block}
    leaf_axes = [tuple(logical[p] for p in leaf) for leaf in leaves]
    support = {
        block[0]: {i for i, axes in enumerate(leaf_axes) if block[0] in axes}
        for block in partition
    }
    final = set(target_output)

    @lru_cache(None)
    def interface(t):
        if isinstance(t, int):
            return frozenset([t]), frozenset(leaf_axes[t])
        if is_transpose(t):
            return interface(t[1])
        inside = interface(t[0])[0] | interface(t[1])[0]
        boundary = frozenset(
            a for a, uses in support.items()
            if uses & inside and (uses - inside or a in final)
        )
        return inside, boundary

    def solve(t, required):
        if isinstance(t, int):
            relevant = set(required)
            if tuple(a for a in leaf_axes[t] if a in relevant) != required:
                return None
            return leaf_network(leaves[t]), leaf_axes[t]
        if is_transpose(t):
            child = solve(t[1], ())
            if child is None:
                return None
            net, layout = child
            order = required + tuple(a for a in layout if a not in required)
            actual = tuple(net[1][layout.index(a)] for a in order)
            return (net[0], actual), order

        left_axes = interface(t[0])[1]
        right_axes = interface(t[1])[1]
        boundary = interface(t)[1]
        shared = left_axes & right_axes
        batch = shared & boundary
        contract = shared - boundary
        left_free = left_axes - shared
        right_free = right_axes - shared

        def role(a):
            return 0 if a in batch else 1 if a in left_free else 2

        tags = [role(a) for a in required]
        if tags != sorted(tags):
            return None
        left = solve(t[0], tuple(a for a in required if a in left_free))
        right = solve(t[1], tuple(a for a in required if a in right_free))
        if left is None or right is None:
            return None
        ln, ll = left
        rn, rl = right
        prefix = tuple(a for a in required if a in batch)
        batch_order = prefix + tuple(sorted(batch - set(prefix)))
        pairs = [
            (ll.index(a), rl.index(a), a in batch) for a in sorted(shared)
        ]
        assert {ll[i] for i, _, b in pairs if not b} == contract
        net = apply_dot(ln, rn, pairs, tuple(ll.index(a) for a in batch_order))
        layout = (
            batch_order
            + tuple(a for a in ll if a in left_free)
            + tuple(a for a in rl if a in right_free)
        )
        return net, layout

    solution = solve(tree, target_output)
    return None if solution is None else solution[0]


def trees_in_order(order):
    if len(order) == 1:
        yield order[0]
    for split in range(1, len(order)):
        for left in trees_in_order(order[:split]):
            for right in trees_in_order(order[split:]):
                yield left, right


def restrict(leaves, target, axes):
    partition, output = target
    blocks = tuple(block for block in partition if block[0] in axes)
    ports = {p for block in blocks for p in block}
    smaller = tuple(tuple(p for p in leaf if p in ports) for leaf in leaves)
    return smaller, (blocks, tuple(a for a in output if a in axes))


def run(inputs, max_rank, max_total_axes, transpose_cases):
    trees = [
        tree for order in permutations(range(inputs)) for tree in trees_in_order(order)
    ]
    candidates = set(trees)
    if transpose_cases:
        for tree in trees:
            candidates.add(("transpose", tree))
            if not isinstance(tree, int):
                candidates.add((("transpose", tree[0]), tree[1]))
                candidates.add((tree[0], ("transpose", tree[1])))
    counts = dict(configurations=0, decisions=0, impossible=0, pair_witnesses=0,
                  constructed=0, root_transpose_repairs=0)
    for ranks in product(range(max_rank + 1), repeat=inputs):
        if sum(ranks) > max_total_axes:
            continue
        counts["configurations"] += 1
        offset, leaves = 0, []
        for rank in ranks:
            leaves.append(tuple(range(offset, offset + rank)))
            offset += rank
        leaves = tuple(leaves)
        sources = trees + ([("transpose", tree) for tree in trees]
                           if transpose_cases else [])
        targets = set().union(*(all_networks(tree, leaves) for tree in sources))
        for tree in sorted(candidates, key=repr):
            reachable = all_networks(tree, leaves)
            for target in targets:
                solution = construct(tree, leaves, target)
                counts["decisions"] += 1
                assert (solution is not None) == (target in reachable), (
                    ranks, tree, target, solution
                )
                if solution is not None:
                    assert solution == target, (ranks, tree, target, solution)
                    counts["constructed"] += 1
                    continue
                counts["impossible"] += 1
                for axes in combinations(target[1], 2):
                    smaller, small_target = restrict(leaves, target, axes)
                    if small_target not in all_networks(tree, smaller):
                        assert max(map(len, smaller), default=0) <= 2
                        # The restricted source must remain expressible too.
                        assert any(
                            small_target in all_networks(source, smaller)
                            for source in sources
                        )
                        counts["pair_witnesses"] += 1
                        break
                else:
                    raise AssertionError(("no pair obstruction", ranks, tree, target))
                repaired = construct(("transpose", tree), leaves, target)
                assert repaired == target
                counts["root_transpose_repairs"] += 1
        # Keep the exhaustive experiment's memory bounded between rank vectors.
        all_networks.cache_clear()
    return {
        "inputs": inputs, "max_input_rank": max_rank,
        "max_total_input_axes": max_total_axes, "rhs_skeletons": len(candidates),
        "transpose_cases": transpose_cases, "counts": counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=int, default=3)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--max-total-axes", type=int, default=6)
    parser.add_argument("--transpose-cases", action="store_true")
    args = parser.parse_args()
    if args.inputs < 1 or args.max_rank < 0 or args.max_total_axes < 0:
        parser.error("inputs must be positive; rank/axis limits must be nonnegative")
    print(json.dumps(run(args.inputs, args.max_rank, args.max_total_axes,
                         args.transpose_cases), indent=2))
