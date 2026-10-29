#!/usr/bin/env python3
"""Finite check of the signed-axis completeness proof, not a TEPL compiler pass.

Enumerate positional descriptors for selected small read-once skeletons. Compare
existence with the canonical-set/order/orientation constructor, and evaluate every
constructed descriptor. An independent exact polynomial evaluator checks signed
network equality against tensor semantics, including odd and unequal extents.
No random tests, floating-point arithmetic, or third-party packages are used.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
import json


def subsets(xs):
    xs = tuple(xs)
    return tuple(c for k in range(len(xs) + 1) for c in combinations(xs, k))


def coords(shape):
    return tuple(product(*(range(n) for n in shape)))


@dataclass(frozen=True)
class Network:
    # A block consists of (physical port, orientation relative to least port).
    blocks: tuple
    output: tuple  # (least physical port of class, orientation)


@dataclass(frozen=True)
class Tensor:
    shape: tuple
    # Each output entry is a sorted polynomial: ((monomial, coefficient), ...).
    # A monomial is a tuple of (capture, physical coordinate tuple) factors.
    values: tuple

    def entries(self):
        return dict(zip(coords(self.shape), self.values))


def polynomial_sum(polys):
    result = {}
    for poly in polys:
        for monomial, coefficient in poly:
            result[monomial] = result.get(monomial, 0) + coefficient
    return tuple(sorted(result.items()))


def polynomial_product(left, right):
    return polynomial_sum(
        ((tuple(sorted(a + b)), ca * cb),)
        for a, ca in left for b, cb in right
    )


def normalize(blocks, output, joins=()):
    """Signed replay by an explicit parity graph, independent of synthesis."""
    graph = {q: [] for block in blocks for q, _ in block}
    edges = [(block[0][0], q, bit) for block in blocks for q, bit in block[1:]]
    edges += list(joins)
    for a, b, bit in edges:
        graph[a].append((b, bit))
        graph[b].append((a, bit))
    representatives, bits, normalized = {}, {}, []
    for root in sorted(graph):
        if root in bits:
            continue
        bits[root] = 0
        stack, block = [root], []
        while stack:
            q = stack.pop()
            representatives[q] = root
            block.append((q, bits[q]))
            for other, bit in graph[q]:
                expected = bits[q] ^ bit
                if other not in bits:
                    bits[other] = expected
                    stack.append(other)
                else:
                    assert bits[other] == expected, 'unexpected reflection cycle'
        normalized.append(tuple(sorted(block)))
    return Network(tuple(normalized), tuple(
        (representatives[q], bit ^ bits[q]) for q, bit in output))


def leaf(c, shapes):
    ports = tuple((c, i) for i in range(len(shapes[c])))
    network = Network(tuple(((q, 0),) for q in ports), tuple((q, 0) for q in ports))
    tensor = Tensor(shapes[c], tuple(((((c, u),), 1),) for u in coords(shapes[c])))
    return network, tensor


def unary(op, state, descriptor):
    network, tensor = state
    selected = set(descriptor)
    rank = len(tensor.shape)
    assert len(selected) == len(descriptor) and selected <= set(range(rank))
    values = tensor.entries()
    if op == 'p':
        assert len(descriptor) == rank
        output = tuple(network.output[i] for i in descriptor)
        shape = tuple(tensor.shape[i] for i in descriptor)
        result = []
        for y in coords(shape):
            x = [0] * rank
            for j, i in enumerate(descriptor):
                x[i] = y[j]
            result.append(values[tuple(x)])
    elif op == 'r':
        output = tuple((q, bit ^ (i in selected))
                       for i, (q, bit) in enumerate(network.output))
        shape = tensor.shape
        result = [values[tuple(n - 1 - y[i] if i in selected else y[i]
                              for i, n in enumerate(shape))] for y in coords(shape)]
    else:
        assert op == 's'
        retained = tuple(i for i in range(rank) if i not in selected)
        output = tuple(network.output[i] for i in retained)
        shape = tuple(tensor.shape[i] for i in retained)
        result = [polynomial_sum(poly for x, poly in values.items()
                                 if tuple(x[i] for i in retained) == y)
                  for y in coords(shape)]
    return Network(network.blocks, output), Tensor(shape, tuple(result))


def dot(left, right, descriptor):
    ln, lt = left
    rn, rt = right
    lb, rb, lk, rk = descriptor
    assert len(lb) == len(rb) and len(lk) == len(rk)
    assert len(set(lb + lk)) == len(lb + lk)
    assert len(set(rb + rk)) == len(rb + rk)
    assert all(0 <= i < len(lt.shape) for i in lb + lk)
    assert all(0 <= i < len(rt.shape) for i in rb + rk)
    assert all(lt.shape[i] == rt.shape[j] for i, j in zip(lb + lk, rb + rk))
    lf = tuple(i for i in range(len(lt.shape)) if i not in lb + lk)
    rf = tuple(i for i in range(len(rt.shape)) if i not in rb + rk)
    joins = [(ln.output[i][0], rn.output[j][0], ln.output[i][1] ^ rn.output[j][1])
             for i, j in zip(lb + lk, rb + rk)]
    output = tuple(ln.output[i] for i in lb + lf) + tuple(rn.output[j] for j in rf)
    network = normalize(ln.blocks + rn.blocks, output, joins)
    shape = tuple(lt.shape[i] for i in lb + lf) + tuple(rt.shape[j] for j in rf)
    # Direct semantics: multiply all operand entries satisfying the positional
    # coordinate equalities, then accumulate by the output coordinate tuple.
    accum = {y: [] for y in coords(shape)}
    for x, lpoly in lt.entries().items():
        for z, rpoly in rt.entries().items():
            if all(x[i] == z[j] for i, j in zip(lb + lk, rb + rk)):
                y = tuple(x[i] for i in lb + lf) + tuple(z[j] for j in rf)
                accum[y].append(polynomial_product(lpoly, rpoly))
    return network, Tensor(shape, tuple(polynomial_sum(accum[y]) for y in coords(shape)))


@lru_cache(None)
def dot_descriptors(ls, rs):
    """All four ordered lists, including every contraction-list order."""
    result = []
    for k in range(min(len(ls), len(rs)) + 1):
        for lk in permutations(range(len(ls)), k):
            for rk in permutations(range(len(rs)), k):
                if any(ls[i] != rs[j] for i, j in zip(lk, rk)):
                    continue
                lf = tuple(i for i in range(len(ls)) if i not in lk)
                rf = tuple(j for j in range(len(rs)) if j not in rk)
                for b in range(min(len(lf), len(rf)) + 1):
                    for lb in permutations(lf, b):
                        for rb in permutations(rf, b):
                            if all(ls[i] == rs[j] for i, j in zip(lb, rb)):
                                result.append((lb, rb, lk, rk))
    return tuple(result)


@lru_cache(None)
def all_states(tree, shapes):
    if isinstance(tree, int):
        return (leaf(tree, shapes),)
    op = tree[0]
    states = {}

    def record(state):
        network, tensor = state
        if network in states:
            assert states[network] == tensor, 'signed representation is not sufficient'
        states[network] = tensor

    if op == 'd':
        for left in all_states(tree[1], shapes):
            for right in all_states(tree[2], shapes):
                for descriptor in dot_descriptors(left[1].shape, right[1].shape):
                    record(dot(left, right, descriptor))
    else:
        for child in all_states(tree[1], shapes):
            axes = range(len(child[1].shape))
            descriptors = permutations(axes) if op == 'p' else subsets(axes)
            for descriptor in descriptors:
                record(unary(op, child, descriptor))
    return tuple(states.items())


def construct(tree, shapes, target):
    """Canonical live sets, order construction, then graph orientation solving."""
    logical, epsilon, support = {}, {}, {}
    for block in target.blocks:
        a = block[0][0]
        support[a] = {q[0] for q, _ in block}
        for q, bit in block:
            logical[q], epsilon[q] = a, bit
    final = tuple(a for a, _ in target.output)
    nodes = []

    def analyze(t):
        index = len(nodes)
        node = {'tree': t, 'id': index}
        nodes.append(node)
        if isinstance(t, int):
            node['inputs'] = {t}
            node['layout'] = tuple(logical[t, i] for i in range(len(shapes[t])))
            node['live'] = set(node['layout'])
            return node
        node['children'] = [analyze(child) for child in t[1:]]
        node['inputs'] = set().union(*(c['inputs'] for c in node['children']))
        mandatory = {a for a, uses in support.items() if uses & node['inputs']
                     and (a in final or not uses <= node['inputs'])}
        child = node['children'][0]
        if t[0] == 'd':
            ls, rs = (c['live'] for c in node['children'])
            node['b'], node['k'] = ls & rs & mandatory, (ls & rs) - mandatory
            node['lf'], node['rf'] = ls - rs, rs - ls
            node['live'] = node['b'] | node['lf'] | node['rf']
        else:
            node['live'] = child['live'] & mandatory if t[0] == 's' else child['live']
        return node

    root = analyze(tree)
    if root['live'] != set(final):
        return None

    def order(node, required):
        node['required'] = required
        if isinstance(node['tree'], int):
            return tuple(a for a in node['layout'] if a in required) == required
        op = node['tree'][0]
        children = node['children']
        if op == 'd':
            blocks = [tuple(a for a in required if a in node[key])
                      for key in ('b', 'lf', 'rf')]
            return (sum(blocks, ()) == required and order(children[0], blocks[1])
                    and order(children[1], blocks[2]))
        return order(children[0], () if op == 'p' else required)

    if not order(root, final):
        return None
    graph = {(n['id'], a): [] for n in nodes for a in n['live']}
    pins = []

    def equal(n, m, a):
        x, y = (n['id'], a), (m['id'], a)
        graph[x].append(y)
        graph[y].append(x)

    for n in nodes:
        t = n['tree']
        if isinstance(t, int):
            pins.extend(((n['id'], logical[t, i]), epsilon[t, i])
                        for i in range(len(shapes[t])))
        elif t[0] == 'd':
            left, right = n['children']
            for a in n['b'] | n['k']:
                equal(left, right, a)
            for a in n['b'] | n['lf']:
                equal(n, left, a)
            for a in n['rf']:
                equal(n, right, a)
        elif t[0] != 'r':
            for a in n['live']:
                equal(n, n['children'][0], a)
    pins.extend(((root['id'], a), bit) for a, bit in target.output)
    pin_values = {}
    for vertex, bit in pins:
        pin_values.setdefault(vertex, set()).add(bit)
    values = {}
    for vertex in graph:
        if vertex in values:
            continue
        component, stack, bits = {vertex}, [vertex], set()
        while stack:
            current = stack.pop()
            bits |= pin_values.get(current, set())
            for other in graph[current]:
                if other not in component:
                    component.add(other)
                    stack.append(other)
        if len(bits) > 1:
            return None
        chosen = next(iter(bits), 0)
        values.update((x, chosen) for x in component)

    def build(n):
        t = n['tree']
        if isinstance(t, int):
            return leaf(t, shapes)
        children = n['children']
        results = [build(c) for c in children]
        layouts = [c['layout'] for c in children]
        if t[0] == 'd':
            batch = tuple(a for a in n['required'] if a in n['b'])
            batch += tuple(sorted(n['b'] - set(batch)))
            contract = tuple(sorted(n['k']))
            descriptor = (tuple(layouts[0].index(a) for a in batch),
                          tuple(layouts[1].index(a) for a in batch),
                          tuple(layouts[0].index(a) for a in contract),
                          tuple(layouts[1].index(a) for a in contract))
            n['layout'] = (batch + tuple(a for a in layouts[0] if a in n['lf'])
                           + tuple(a for a in layouts[1] if a in n['rf']))
            return dot(*results, descriptor)
        layout = layouts[0]
        if t[0] == 'p':
            n['layout'] = n['required'] + tuple(a for a in layout if a not in n['required'])
            descriptor = tuple(layout.index(a) for a in n['layout'])
        elif t[0] == 's':
            n['layout'] = tuple(a for a in layout if a in n['live'])
            descriptor = tuple(i for i, a in enumerate(layout) if a not in n['live'])
        else:
            n['layout'] = layout
            descriptor = tuple(i for i, a in enumerate(layout)
                               if values[n['id'], a] != values[children[0]['id'], a])
        return unary(t[0], results[0], descriptor)

    return build(root)


def replace_at(tree, path, op):
    if not path:
        return op, tree
    result = list(tree)
    result[path[0]] = replace_at(result[path[0]], path[1:], op)
    return tuple(result)


def paths(tree):
    yield ()
    if not isinstance(tree, int):
        for i in range(1, len(tree)):
            for path in paths(tree[i]):
                yield (i,) + path


def skeletons(count):
    if count == 1:
        return tuple([0] + [(a, 0) for a in 'prs']
                     + [(a, (b, 0)) for a in 'prs' for b in 'prs']
                     + [(a, (b, (c, 0))) for a in 'prs' for b in 'prs' for c in 'prs'])
    bases = [('d', 0, 1), ('d', 1, 0)] if count == 2 else [
        ('d', ('d', a, b), c) for a, b, c in permutations(range(3))]
    if count == 3:
        bases += [('d', a, ('d', b, c)) for a, b, c in permutations(range(3))]
    trees = set(bases)
    for base in bases:
        for path in paths(base):
            for op in 'prs':
                wrapped = replace_at(base, path, op)
                trees.add(wrapped)
                # Exercise reversals before/after sums and permutations.
                trees.add(replace_at(wrapped, path, 'r'))
        trees.add(('s', ('r', base)))
        trees.add(('r', ('s', base)))
    if count == 2:
        for a, b in product('prs', repeat=2):
            trees.add(('d', (a, 0), (b, 1)))
            trees.add(('s', ('d', (a, 0), (b, 1))))
    return tuple(sorted(trees, key=repr))


def main():
    families = [((2, 2),), ((2, 3),), ((),), ((2,), (2,)),
                ((3,), (3,)), ((2,), (3,)), ((), (2,)),
                ((2, 2), (2,)), ((2,), (2,), (2,))]
    reports = []
    for shapes in families:
        trees = skeletons(len(shapes))
        targets, semantic_keys = {}, {}
        rhs_semantics = {}
        for tree in trees:
            states = all_states(tree, shapes)
            rhs_semantics[tree] = {tensor for _, tensor in states}
            for network, tensor in states:
                if network in targets:
                    assert targets[network] == tensor
                if tensor in semantic_keys:
                    assert semantic_keys[tensor] == network, 'signed exactness failed'
                targets[network] = tensor
                semantic_keys[tensor] = network
        checks = successes = 0
        for tree in trees:
            for network, tensor in targets.items():
                actual = construct(tree, shapes, network)
                expected = tensor in rhs_semantics[tree]
                assert (actual is not None) == expected, (shapes, tree, network, expected)
                if actual is not None:
                    assert actual == (network, tensor), (shapes, tree, network, actual)
                    successes += 1
                checks += 1
        reports.append(dict(shapes=shapes, skeletons=len(trees), targets=len(targets),
                            checks=checks, feasible=successes, infeasible=checks-successes))
        all_states.cache_clear()
    print(json.dumps(dict(families=reports,
                          checks=sum(r['checks'] for r in reports),
                          feasible=sum(r['feasible'] for r in reports),
                          infeasible=sum(r['infeasible'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
