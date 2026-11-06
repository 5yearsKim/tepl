#!/usr/bin/env python3
"""Check aligned weighted-product scalar synthesis on finite instances.

The proposed procedure uses only signed partitions and metadata.
An independent oracle enumerates concrete positional descriptors and computes
exact tensor-entry polynomials. This is a research experiment, not TEPL's
production synthesizer or a machine-checked proof. Run from the repository root.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from math import prod
from functools import lru_cache
from itertools import permutations, product
import json

from check_oriented_axis_synthesis import (
    Tensor, coords, dot_descriptors, polynomial_product, polynomial_sum, subsets,
)


@dataclass(frozen=True)
class State:
    shape: tuple
    # Blocks contain ((kind, capture/position, axis), orientation).
    # kind 'i' denotes a physical input port and 'o' a current output port.
    blocks: tuple
    coefficient: Fraction = Fraction(1)


@dataclass(frozen=True)
class Witness:
    descriptor: object
    children: tuple = ()


def children(tree):
    if isinstance(tree, int):
        return ()
    return tree[2] if tree[0] == 'h' else tree[1:]


def leaf_state(c, shapes):
    blocks = tuple((((('i', c, i), 0), (('o', i, 0), 0)))
                   for i in range(len(shapes[c])))
    return State(shapes[c], blocks)


def transition(child_states, shape, links, shapes):
    """Forget private output placeholders while retaining every physical port."""
    graph, dimensions = {}, {}

    def vertex(q, n):
        if q in dimensions:
            assert dimensions[q] == n
        dimensions[q] = n
        graph.setdefault(q, [])

    def edge(a, b, bit):
        assert dimensions[a] == dimensions[b]
        graph[a].append((b, bit))
        graph[b].append((a, bit))

    for side, state in enumerate(child_states):
        for block in state.blocks:
            renamed = []
            for q, bit in block:
                if q[0] == 'o':
                    name, extent = ('tmp', side, q[1]), state.shape[q[1]]
                else:
                    name, extent = q, shapes[q[1]][q[2]]
                    assert name not in graph, 'inputs must be read once'
                vertex(name, extent)
                renamed.append((name, bit))
            for name, bit in renamed[1:]:
                edge(renamed[0][0], name, renamed[0][1] ^ bit)
    for i, n in enumerate(shape):
        vertex(('o', i, 0), n)
    for a, b, bit in links:
        edge(a, b, bit)
    visited, blocks = set(), []
    for start in sorted(graph):
        if start in visited:
            continue
        bits, stack = {start: 0}, [start]
        while stack:
            q = stack.pop()
            visited.add(q)
            for other, bit in graph[q]:
                expected = bits[q] ^ bit
                if other in bits:
                    assert bits[other] == expected, 'reflection cycle'
                else:
                    bits[other] = expected
                    stack.append(other)
        kept = sorted(q for q in bits if q[0] != 'tmp')
        assert kept and any(q[0] == 'i' for q in kept)
        blocks.append(tuple((q, bits[q] ^ bits[kept[0]]) for q in kept))
    return State(tuple(shape), tuple(sorted(blocks)),
                 prod((s.coefficient for s in child_states), start=Fraction(1)))


def pairing_choices(ls, rs):
    """Enumerate role-labelled partial matchings, not four-list permutations."""
    def visit(i, used, batch, reduced):
        if i == len(ls):
            for ordered in permutations(batch):
                yield (tuple(x for x, _ in ordered), tuple(y for _, y in ordered),
                       tuple(x for x, _ in reduced), tuple(y for _, y in reduced))
            return
        yield from visit(i + 1, used, batch, reduced)
        for j, extent in enumerate(rs):
            if j not in used and ls[i] == extent:
                yield from visit(i + 1, used | {j}, batch + ((i, j),), reduced)
                yield from visit(i + 1, used | {j}, batch, reduced + ((i, j),))
    yield from visit(0, set(), (), ())


def choices(tree, child_shapes):
    op = tree[0]
    shape = child_shapes[0]
    if op == 'd':
        yield from pairing_choices(*child_shapes)
    elif op in ('m', 'h'):
        if op == 'h':
            assert child_shapes and tree[1] != 0
        if all(s == shape for s in child_shapes):
            yield None
    elif op == 'p':
        yield from permutations(range(len(shape)))
    else:
        assert op in ('r', 's')
        yield from subsets(range(len(shape)))


def state_step(tree, states, descriptor, shapes):
    op = tree[0]
    shape = states[0].shape
    links = []

    def attach(j, side, i, bit=0):
        links.append((('o', j, 0), ('tmp', side, i), bit))

    if op == 'h':
        assert tree[1] != 0 and all(s.shape == shape for s in states)
        for i in range(len(shape)):
            attach(i, 0, i)
            for side in range(1, len(states)):
                links.append((('tmp', 0, i), ('tmp', side, i), 0))
        result = transition(states, shape, links, shapes)
        return replace(result, coefficient=tree[1] * result.coefficient)
    if op == 'p':
        shape = tuple(shape[i] for i in descriptor)
        for j, i in enumerate(descriptor):
            attach(j, 0, i)
    elif op == 'r':
        for i in range(len(shape)):
            attach(i, 0, i, int(i in descriptor))
    elif op == 's':
        kept = tuple(i for i in range(len(shape)) if i not in descriptor)
        shape = tuple(shape[i] for i in kept)
        for j, i in enumerate(kept):
            attach(j, 0, i)
    else:
        if op == 'm':
            assert states[0].shape == states[1].shape
            axes = tuple(range(len(shape)))
            lb, rb, lk, rk = axes, axes, (), ()
        else:
            assert op == 'd'
            lb, rb, lk, rk = descriptor
        ls, rs = states[0].shape, states[1].shape
        lf = tuple(i for i in range(len(ls)) if i not in lb + lk)
        rf = tuple(i for i in range(len(rs)) if i not in rb + rk)
        for i, j in zip(lb + lk, rb + rk):
            links.append((('tmp', 0, i), ('tmp', 1, j), 0))
        output = tuple((0, i) for i in lb + lf) + tuple((1, j) for j in rf)
        shape = tuple(states[side].shape[i] for side, i in output)
        for j, (side, i) in enumerate(output):
            attach(j, side, i)
    return transition(states, shape, links, shapes)


@lru_cache(None)
def reachable(tree, shapes):
    if isinstance(tree, int):
        return ((leaf_state(tree, shapes), Witness(tree)),)
    table = {}
    for entries in product(*(reachable(c, shapes) for c in children(tree))):
        states, witnesses = zip(*entries)
        for descriptor in choices(tree, tuple(s.shape for s in states)):
            state = state_step(tree, states, descriptor, shapes)
            table.setdefault(state, Witness(descriptor, witnesses))
    return tuple(table.items())


# Independent oracle: exact entry polynomials and concrete positional operators.
# No signed partitions, signed replay, or state-table choices are consulted.
def tensor_leaf(c, shapes):
    return Tensor(shapes[c], tuple(((((c, u),), 1),) for u in coords(shapes[c])))


def tensor_step(tree, tensors, descriptor):
    op = tree[0]
    first = tensors[0]
    entries = first.entries()
    if op == 'h':
        assert tree[1] != 0 and all(t.shape == first.shape for t in tensors)
        values = []
        for polys in zip(*(t.values for t in tensors)):
            result = (((), tree[1]),)
            for poly in polys:
                result = polynomial_product(result, poly)
            values.append(result)
        return Tensor(first.shape, tuple(values))
    if op == 'p':
        shape = tuple(first.shape[i] for i in descriptor)
        result = []
        for y in coords(shape):
            x = [0] * len(shape)
            for j, i in enumerate(descriptor):
                x[i] = y[j]
            result.append(entries[tuple(x)])
        return Tensor(shape, tuple(result))
    if op == 'r':
        return Tensor(first.shape, tuple(entries[tuple(
            n - 1 - y[i] if i in descriptor else y[i] for i, n in enumerate(first.shape))]
            for y in coords(first.shape)))
    if op == 's':
        kept = tuple(i for i in range(len(first.shape)) if i not in descriptor)
        shape = tuple(first.shape[i] for i in kept)
        return Tensor(shape, tuple(polynomial_sum(poly for x, poly in entries.items()
                            if tuple(x[i] for i in kept) == y) for y in coords(shape)))
    second = tensors[1]
    if op == 'm':
        assert first.shape == second.shape
        return Tensor(first.shape, tuple(polynomial_product(a, b)
                                         for a, b in zip(first.values, second.values)))
    assert op == 'd'
    lb, rb, lk, rk = descriptor
    lf = tuple(i for i in range(len(first.shape)) if i not in lb + lk)
    rf = tuple(i for i in range(len(second.shape)) if i not in rb + rk)
    shape = tuple(first.shape[i] for i in lb + lf) + tuple(second.shape[j] for j in rf)
    terms = {y: [] for y in coords(shape)}
    for x, a in entries.items():
        for z, b in second.entries().items():
            if all(x[i] == z[j] for i, j in zip(lb + lk, rb + rk)):
                y = tuple(x[i] for i in lb + lf) + tuple(z[j] for j in rf)
                terms[y].append(polynomial_product(a, b))
    return Tensor(shape, tuple(polynomial_sum(terms[y]) for y in coords(shape)))


def oracle_choices(tree, tensors):
    op = tree[0]
    shape = tensors[0].shape
    if op == 'd':
        yield from dot_descriptors(shape, tensors[1].shape)
    elif op in ('m', 'h'):
        if op == 'h':
            assert tensors and tree[1] != 0
        if all(t.shape == shape for t in tensors):
            yield None
    elif op == 'p':
        yield from permutations(range(len(shape)))
    else:
        # Include every list order, unlike the canonical subset constructor.
        for selected in subsets(range(len(shape))):
            yield from permutations(selected)


@lru_cache(None)
def oracle(tree, shapes):
    if isinstance(tree, int):
        return frozenset((tensor_leaf(tree, shapes),))
    result = set()
    for tensors in product(*(oracle(c, shapes) for c in children(tree))):
        for descriptor in oracle_choices(tree, tensors):
            result.add(tensor_step(tree, tensors, descriptor))
    return frozenset(result)


def evaluate_witness(tree, shapes, witness):
    if isinstance(tree, int):
        assert witness.descriptor == tree and not witness.children
        return tensor_leaf(tree, shapes)
    tensors = tuple(evaluate_witness(c, shapes, w) for c, w in zip(children(tree), witness.children))
    assert len(witness.children) == len(children(tree))
    # Check the returned descriptor is legal in the independent enumerator.
    assert witness.descriptor in tuple(oracle_choices(tree, tensors))
    return tensor_step(tree, tensors, witness.descriptor)


def base_templates(count):
    if count == 1:
        return tuple([0] + [(a, 0) for a in 'prs']
                     + [(a, (b, 0)) for a in 'prs' for b in 'prs'])
    bases = [('d', 0, 1), ('d', 1, 0), ('m', 0, 1), ('m', 1, 0)] if count == 2 else [
        (a, (b, i, j), k) for a, b in product(('d', 'm'), repeat=2)
        for i, j, k in permutations(range(3))]
    if count == 3:
        bases += [(a, i, (b, j, k)) for a, b in product(('d', 'm'), repeat=2)
                  for i, j, k in permutations(range(3))]
    trees = set(bases)
    for base in bases:
        trees.update((op, base) for op in 'prs')
        trees.update(('s', (op, base)) for op in 'prs')
    if count == 2:
        for op in ('d', 'm'):
            for a, b in product('prs', repeat=2):
                base = (op, (a, 0), (b, 1))
                trees.update((base, ('s', base), ('r', base)))
    return tuple(sorted(trees, key=repr))


def h(q, *operands):
    return 'h', Fraction(q), tuple(operands)


def templates(count):
    base = base_templates(count)
    trees = set(base)
    for tree in base:
        trees.update((h(-1, tree), h(Fraction(2, 3), tree)))
    if count == 1:
        trees.update((h(-1, h(-1, 0)), h(2, h(Fraction(1, 2), 0)),
                      ('s', h(-1, 0)), h(-1, ('s', 0)),
                      ('r', h(Fraction(-3, 5), 0))))
    elif count == 2:
        for q in (-1, Fraction(2, 3)):
            trees.update((h(q, 0, 1), ('s', h(q, 0, 1)),
                          ('d', h(q, 0), 1), h(q, ('d', 0, 1)),
                          h(1, h(q, 0), 1)))
    else:
        trees.update((h(1, 0, 1, 2), h(Fraction(-3, 5), 0, 1, 2),
                      ('s', h(Fraction(-3, 5), 0, 1, 2)),
                      h(Fraction(-3, 5), ('m', 0, 1), 2)))
    return tuple(sorted(trees, key=repr))


def targeted_checks():
    shapes = ((2,), (2,))
    states = dict(reachable(('m', 0, 1), shapes))
    assert len(states) == 1 and all(s.shape == (2,) for s in states)
    assert any(s.shape == () for s, _ in reachable(('s', ('m', 0, 1)), shapes))
    assert not reachable(('m', 0, 1), ((2,), (3,)))
    assert not reachable(('m', 0, 1), ((), (2,)))
    assert len(reachable(('m', 0, 1), ((), ()))) == 1
    # Reflection can move between operands, but cannot disappear from a product.
    left = set(dict(reachable(('s', ('m', ('r', 0), 1)), shapes)))
    right = set(dict(reachable(('s', ('m', 0, ('r', 1))), shapes)))
    bare = set(dict(reachable(('s', ('m', 0, 1)), shapes)))
    assert {s for s in left if not s.shape} == {s for s in right if not s.shape}
    assert {s for s in bare if not s.shape} < {s for s in left if not s.shape}


    one = ((2,),)
    assert set(dict(reachable(h(-1, h(-1, 0)), one))) == set(dict(reachable(0, one)))
    assert set(dict(reachable(h(2, h(Fraction(1, 2), 0)), one))) == set(dict(reachable(0, one)))
    assert not (set(dict(reachable(h(-1, 0), one))) & set(dict(reachable(0, one))))
    assert set(dict(reachable(('s', h(-1, 0)), one))) == set(dict(reachable(h(-1, ('s', 0)), one)))
    assert not reachable(h(Fraction(2, 3), 0, 1), ((2,), (3,)))
    three = ((2,), (2,), (2,))
    assert set(dict(reachable(h(Fraction(2, 3), 0, 1, 2), three))) == set(dict(reachable(
        h(Fraction(2, 3), ('m', 0, 1), 2), three)))


def main():
    targeted_checks()
    families = [((),), ((2, 3),), ((), ()), ((2,), (2,)),
                ((3,), (3,)), ((2,), (3,)), ((), (2,)),
                ((2, 2), (2,)), ((2,), (2,), (2,))]
    reports = []
    for shapes in families:
        trees = templates(len(shapes))
        normal_to_tensor, tensor_to_normal = {}, {}
        by_tree = {}
        for tree in trees:
            table = dict(reachable(tree, shapes))
            reference = oracle(tree, shapes)
            recovered = set()
            for state, witness in table.items():
                tensor = evaluate_witness(tree, shapes, witness)
                assert tensor in reference, (shapes, tree, state)
                if state in normal_to_tensor:
                    assert normal_to_tensor[state] == tensor, 'normal form is not sufficient'
                if tensor in tensor_to_normal:
                    assert tensor_to_normal[tensor] == state, 'normal form is not necessary'
                normal_to_tensor[state], tensor_to_normal[tensor] = tensor, state
                recovered.add(tensor)
            assert recovered == reference, ('reachable-state completeness', shapes, tree)
            by_tree[tree] = (table, reference)
        feasible = 0
        for table, reference in by_tree.values():
            for state, tensor in normal_to_tensor.items():
                assert (state in table) == (tensor in reference)
                feasible += state in table
        checks = len(trees) * len(normal_to_tensor)
        reports.append(dict(shapes=shapes, skeletons=len(trees), targets=len(normal_to_tensor),
                            checks=checks, feasible=feasible, infeasible=checks-feasible))
        reachable.cache_clear()
        oracle.cache_clear()
    print(json.dumps(dict(families=reports,
                          checks=sum(r['checks'] for r in reports),
                          feasible=sum(r['feasible'] for r in reports),
                          infeasible=sum(r['infeasible'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
