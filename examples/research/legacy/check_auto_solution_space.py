#!/usr/bin/env python3
"""Check the solution-count theorem against exhaustive positional attributes.

Standalone research experiment, not a production inference pass. The oracle
enumerates all four ordered dot lists and all transpose permutations. It shares
the network evaluator with check_ordered_contractions, whose semantic exactness
is a separate lemma; this experiment does not independently prove that lemma.
"""

import argparse
import json
from collections import Counter
from functools import lru_cache
from itertools import permutations, product
from math import factorial

from examples.research.check_ordered_contractions import (
    apply_dot,
    construct,
    is_transpose,
    leaf_network,
    trees_in_order,
)


@lru_cache(None)
def descriptors(nleft, nright):
    """Every legal four-list descriptor, including reduction reorderings."""
    result = []
    for nc in range(min(nleft, nright) + 1):
        for lc in permutations(range(nleft), nc):
            for rc in permutations(range(nright), nc):
                left = tuple(i for i in range(nleft) if i not in lc)
                right = tuple(i for i in range(nright) if i not in rc)
                for nb in range(min(len(left), len(right)) + 1):
                    for lb in permutations(left, nb):
                        for rb in permutations(right, nb):
                            result.append((lc, rc, lb, rb))
    return tuple(result)


@lru_cache(None)
def exhaustive_counts(tree, leaves):
    if isinstance(tree, int):
        return Counter({leaf_network(leaves[tree]): 1})
    if is_transpose(tree):
        result = Counter()
        for (partition, layout), count in exhaustive_counts(tree[1], leaves).items():
            for order in permutations(layout):
                result[partition, order] += count
        return result
    result = Counter()
    for left, nl in exhaustive_counts(tree[0], leaves).items():
        for right, nr in exhaustive_counts(tree[1], leaves).items():
            for lc, rc, lb, rb in descriptors(len(left[1]), len(right[1])):
                pairs = [(i, j, False) for i, j in zip(lc, rc)]
                pairs += [(i, j, True) for i, j in zip(lb, rb)]
                result[apply_dot(left, right, pairs, lb)] += nl * nr
    return result


def solution_count(tree, leaves, target):
    """Structural product formula; no enumeration of positional attributes."""
    partition, final = target
    logical = {p: block[0] for block in partition for p in block}
    axes = [tuple(logical[p] for p in leaf) for leaf in leaves]
    support = {
        block[0]: frozenset(i for i, row in enumerate(axes) if block[0] in row)
        for block in partition
    }

    @lru_cache(None)
    def inputs(t):
        if isinstance(t, int):
            return frozenset([t])
        if is_transpose(t):
            return inputs(t[1])
        return inputs(t[0]) | inputs(t[1])

    @lru_cache(None)
    def boundary(t):
        inside = inputs(t)
        return frozenset(
            a for a, uses in support.items()
            if uses & inside and (uses - inside or a in final)
        )

    def count(t, required):
        if isinstance(t, int):
            return int(tuple(a for a in axes[t] if a in required) == required)
        if is_transpose(t):
            return (
                factorial(len(boundary(t))) // factorial(len(required))
                * count(t[1], ())
            )
        left, right, parent = boundary(t[0]), boundary(t[1]), boundary(t)
        batch = left & right & parent
        contract = (left & right) - parent
        lf, rf = left - right, right - left
        roles = [0 if a in batch else 1 if a in lf else 2 for a in required]
        if roles != sorted(roles):
            return 0
        qb = tuple(a for a in required if a in batch)
        ql = tuple(a for a in required if a in lf)
        qr = tuple(a for a in required if a in rf)
        local = factorial(len(contract)) * factorial(len(batch)) // factorial(len(qb))
        return local * count(t[0], ql) * count(t[1], qr)

    return count(tree, final)


def check_examples():
    """Paper examples, including two distinct forms of harmless ambiguity."""
    vectors = ((0,), (1,), (2,))
    partition = ((0,), (1, 2))
    cases = [
        ("associate_obstruction", (0, (1, 2)), vectors, (partition, (1, 0)), 0),
        ("reverse_and_lora_obstruction", ((0, 1), 2), vectors, (partition, (0, 1)), 0),
        (
            "matrix_association",
            (0, (1, 2)),
            ((0, 1), (2, 3), (4, 5)),
            (((0,), (1, 2), (3, 4), (5,)), (0, 5)),
            1,
        ),
        (
            "two_reduction_orders",
            ((0, 1), 2),
            ((0, 1), (2, 3), ()),
            (((0, 2), (1, 3)), ()),
            2,
        ),
        (
            "inverse_transposes",
            ("transpose", ("transpose", 0)),
            ((0, 1, 2),),
            leaf_network((0, 1, 2)),
            6,
        ),
    ]
    for name, tree, leaves, target, expected in cases:
        assert solution_count(tree, leaves, target) == expected, name
        assert exhaustive_counts(tree, leaves)[target] == expected, name
    exhaustive_counts.cache_clear()
    return len(cases)


def run(max_rank, max_total_axes):
    trees = [t for order in permutations(range(3)) for t in trees_in_order(order)]
    candidates = set(trees)
    for tree in trees:
        candidates.add(("transpose", tree))
        candidates.add((("transpose", tree[0]), tree[1]))
        candidates.add((tree[0], ("transpose", tree[1])))
    stats = Counter(configurations=0, decisions=0, impossible=0, unique=0, ambiguous=0)
    largest_count = 0
    for ranks in product(range(max_rank + 1), repeat=3):
        if sum(ranks) > max_total_axes:
            continue
        offset, leaves = 0, []
        for rank in ranks:
            leaves.append(tuple(range(offset, offset + rank)))
            offset += rank
        leaves = tuple(leaves)
        stats["configurations"] += 1
        targets = set().union(*(exhaustive_counts(t, leaves) for t in candidates))
        for tree in sorted(candidates, key=repr):
            oracle = exhaustive_counts(tree, leaves)
            for target in targets:
                predicted = solution_count(tree, leaves, target)
                actual = oracle[target]
                assert predicted == actual, (ranks, tree, target, predicted, actual)
                witness = construct(tree, leaves, target)
                assert witness == (target if actual else None)
                stats["decisions"] += 1
                stats["impossible" if actual == 0 else "unique" if actual == 1 else "ambiguous"] += 1
                largest_count = max(largest_count, actual)
        exhaustive_counts.cache_clear()
    return dict(
        stats,
        largest_solution_count=largest_count,
        rhs_skeletons=len(candidates),
        named_examples=check_examples(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rank", type=int, default=2)
    parser.add_argument("--max-total-axes", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.max_rank, args.max_total_axes), indent=2, sort_keys=True))
