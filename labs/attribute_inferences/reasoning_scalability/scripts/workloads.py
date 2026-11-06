"""Automatic, rank-polymorphic workloads for the main scaling experiment."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
DIALECT = ROOT / "examples/semantic/dialects/tensor.tepl"
FIELDS = ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
          "lhs_batching_dimensions", "rhs_batching_dimensions")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class Node:
    name: str
    axes: tuple
    children: tuple = ()
    attrs: dict | None = None


def dot(lhs, rhs, label, name):
    assert set(lhs.axes) & set(rhs.axes) == {"b", label}
    attrs = dict(zip(FIELDS, ([lhs.axes.index(label)], [rhs.axes.index(label)],
                              [lhs.axes.index("b")], [rhs.axes.index("b")])))
    axes = ("b",) + tuple(a for a in lhs.axes if a not in ("b", label))
    axes += tuple(a for a in rhs.axes if a not in ("b", label))
    assert len(axes) == 3 and len(set(axes)) == 3
    return Node(name, axes, (lhs, rhs), attrs)


def chain(n):
    leaves = [Node(f"A{i}", ("b", f"d{i-1}", f"d{i}"))
              for i in range(1, n + 1)]
    lhs = leaves[0]
    for i in range(1, n):
        lhs = dot(lhs, leaves[i], f"d{i}", f"l{i}")
    rhs = leaves[-1]
    for i in range(n - 2, -1, -1):
        rhs = dot(leaves[i], rhs, f"d{i+1}", f"r{i+1}")
    assert lhs.axes == rhs.axes == ("b", "d0", f"d{n}")
    return leaves, lhs, rhs


def internal(node):
    if node.children:
        yield from internal(node.children[0])
        yield from internal(node.children[1])
        yield node


def expression(node, automatic=False):
    if not node.children:
        return node.name
    label = "" if automatic else node.name + ": "
    return f"({label}dot {expression(node.children[0], automatic)} {expression(node.children[1], automatic)})"


def capture_shape(leaf):
    # "b" denotes an entire leading batch sequence, not one concrete axis.
    return ','.join(['Batch...'] + ['D' + axis[1:] for axis in leaf.axes if axis != 'b'])


def field_expression(field):
    return {
        'lhs_contracting_dimensions': '[len(Batch)+1]',
        'rhs_contracting_dimensions': '[len(Batch)]',
        'lhs_batching_dimensions': 'range(len(Batch))',
        'rhs_batching_dimensions': 'range(len(Batch))',
    }[field]


def positive_extents(dimensions):
    return ['all([d>0 for d in Batch])'] + [f'D{i}>0' for i in range(dimensions)]


def dot_source(size):
    n = size + 1
    leaves, lhs, rhs = chain(n)
    name = f"chain_automatic_{size}"
    lines = [f"rule {name} {{"]
    for leaf in leaves:
        lines.append(f"    {leaf.name}: f32[{capture_shape(leaf)}]")
    lines += ["", "    " + expression(lhs), "    =>", "    " + expression(rhs, True), "    where {"]
    conditions = positive_extents(n + 1)
    for node in internal(lhs):
        conditions += [f"{node.name}.attrs.{field} == {field_expression(field)}" for field in FIELDS]
    lines.append("        " + "\n        && ".join(conditions))
    lines.append("    }")
    lines.append("}")
    return name, "\n".join(lines) + "\n"


def additive_source(terms, dots_per_term=2):
    """Disjoint read-once chains; shared output dimensions make sums legal."""
    captures, conditions, left, right = [], [], [], []
    for term in range(terms):
        leaves, lhs, rhs = chain(dots_per_term + 1)
        # Both trees share leaves, so rename each object exactly once.
        seen = set()
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            node.name = f"t{term}_" + node.name
            for child in node.children:
                visit(child)
        visit(lhs)
        visit(rhs)
        for leaf in leaves:
            captures.append(f"{leaf.name}:f32[{capture_shape(leaf)}]")
        for node in internal(lhs):
            conditions.extend(f"{node.name}.attrs.{f}=={field_expression(f)}" for f in FIELDS)
        left.append(expression(lhs))
        right.append(expression(rhs, True))
    def balanced(xs, op):
        if len(xs) == 1:
            return xs[0]
        mid = len(xs) // 2
        return f"({op} {balanced(xs[:mid], op)} {balanced(xs[mid:], op)})"
    def conjunction(xs):
        if len(xs) == 1:
            return xs[0]
        mid = len(xs) // 2
        return f"({conjunction(xs[:mid])} && {conjunction(xs[mid:])})"
    conditions += positive_extents(dots_per_term + 2)
    name = f"terms_automatic_{terms}"
    return name, (f"rule {name} {{\n" + '\n'.join(captures) + '\n' +
                  balanced(left, 'add') + ' => ' + balanced(right, 'add') +
                  '\nwhere {' + conjunction(conditions) + '}\n}\n')
