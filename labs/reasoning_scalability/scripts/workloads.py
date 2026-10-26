"""Contraction workloads and shared paths for the bounded inference benchmark."""

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DIALECT = ROOT / "examples/semantic/dialects/tensor.tepl"
PERMUTATIONS = tuple(itertools.permutations(range(3)))
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


def chain(n, layout):
    leaves = []
    for i in range(1, n + 1):
        axes = ("b", f"d{i-1}", f"d{i}")
        order = (0, 1, 2) if layout == "canonical" else PERMUTATIONS[(i - 1) % 6]
        leaves.append(Node(f"A{i}", tuple(axes[j] for j in order)))
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
    inferred = automatic(node) if callable(automatic) else automatic
    label = "" if inferred else node.name + ": "
    return f"({label}dot {expression(node.children[0], automatic)} {expression(node.children[1], automatic)})"


def dot_source(size, layout, mode):
    n = size + 1
    leaves, lhs, rhs = chain(n, layout)
    name = f"chain_{layout}_{mode}_{size}"
    inferred = lambda node: mode == "automatic" or (mode == "partial" and int(node.name[1:]) % 2 == 1)
    lines = [f"rule {name} {{"]
    for leaf in leaves:
        dims = ["B" if a == "b" else "D" + a[1:] for a in leaf.axes]
        lines.append(f"    {leaf.name}: f32[{', '.join(dims)}]")
    lines += ["", "    " + expression(lhs), "    =>", "    " + expression(rhs, inferred), "    where {"]
    conditions = [f"{d} > 0" for d in ["B"] + [f"D{i}" for i in range(n + 1)]]
    for node in internal(lhs):
        conditions += [f"{node.name}.attrs.{field} == {node.attrs[field]}" for field in FIELDS]
    lines.append("        " + "\n        && ".join(conditions))
    lines.append("    }")
    derived = [node for node in internal(rhs) if not inferred(node)]
    if derived:
        lines.append("    derive {")
        for node in derived:
            fields = ", ".join(f"{f}: {node.attrs[f]}" for f in FIELDS)
            lines.append(f"        {node.name}.attrs = {{ {fields} }};")
        lines.append("    }")
    lines.append("}")
    return name, "\n".join(lines) + "\n"
