"""Parse corpus graph structure and classify field-level axis inference tasks."""

from dataclasses import dataclass, field
import re

if __package__:
    from . import hlo_support
    from .paths import HERE, ROOT
else:
    import hlo_support
    from paths import HERE, ROOT

TOKEN = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|=>|\.\.\.|\d+\.\d+(?:[eE][+-]?\d+)?|\d+|[A-Za-z_]\w*|[^\s]'
)


@dataclass
class Node:
    name: str
    kind: str = "operation"
    label: str = ""
    attribute: tuple | None = None
    children: list = field(default_factory=list)
    span: tuple | None = None

    def operations(self):
        return ([self] if self.kind == "operation" else []) + [
            op for child in self.children for op in child.operations()
        ]


class Rule:
    def __init__(self, source):
        self.source = source
        self.tokens = [
            m for m in TOKEN.finditer(source) if not m[0].startswith(("//", "/*"))
        ]
        self.values = [t[0] for t in self.tokens]
        self.name = self.values[self.values.index("rule") + 1]
        arrow = self.values.index("=>")
        opening = self.values.index("{", self.values.index("rule"))
        self.lhs = None
        for start in range(opening + 1, arrow):
            try:
                lhs, end = self.graph(start)
                if end == arrow:
                    self.lhs = lhs
                    break
            except (ValueError, IndexError):
                pass
        if self.lhs is None:
            raise ValueError("cannot find LHS graph: " + self.name)
        self.rhs, self.rhs_end = self.graph(arrow + 1)
        self.derive = (
            self.values.index("derive", self.rhs_end)
            if "derive" in self.values[self.rhs_end :]
            else None
        )

    def closing(self, index, left, right):
        depth = 0
        for i in range(index, len(self.values)):
            depth += self.values[i] == left
            depth -= self.values[i] == right
            if depth == 0:
                return i
        raise ValueError("unbalanced " + left)

    def graph(self, start):
        v = self.values
        i = start
        if v[i] != "(":
            if v[i] in {"+", "-"}:
                i += 1
            if not re.fullmatch(r"\w+(?:\.\d+)?", v[i]):
                raise ValueError("not a graph leaf")
            literal = v[i][0].isdigit()
            node = Node(v[i], "literal" if literal else "capture")
            i += 1
            if literal and v[i] == ":":
                i += 2
            node.span = (self.tokens[start].start(), self.tokens[i - 1].end())
            return node, i
        i += 1
        if v[i] == "let":
            name = v[i + 1]
            if v[i + 2] != "=":
                raise ValueError("invalid graph binding")
            child, end = self.graph(i + 3)
            if v[end] != ")":
                raise ValueError("unterminated binding")
            return (
                Node(
                    name,
                    "binding",
                    children=[child],
                    span=(self.tokens[start].start(), self.tokens[end].end()),
                ),
                end + 1,
            )
        label = ""
        if v[i + 1] == ":":
            label, i = v[i], i + 2
        name = v[i]
        i += 1
        if v[i] == ".":
            name += "." + v[i + 1]
            i += 2
        attribute = None
        if v[i] == "[":
            end = self.closing(i, "[", "]")
            attribute = (self.tokens[i].start(), self.tokens[end].end())
            i = end + 1
        node = Node(name, label=label, attribute=attribute)
        while v[i] != ")":
            child, i = self.graph(i)
            node.children.append(child)
        node.span = (self.tokens[start].start(), self.tokens[i].end())
        return node, i + 1

    def statements(self):
        if self.derive is None:
            return []
        start = self.derive + 2
        end = self.closing(self.derive + 1, "{", "}")
        depth = 0
        rows = []
        for i in range(start, end):
            # Statement starts are explicit even when several share a line.
            assignment = self.values[i : i + 4]
            if i > start and depth == 0 and (
                self.values[i] == "let"
                or (len(assignment) == 4 and assignment[1:] == [".", "attrs", "="])
            ):
                tokens = self.values[start:i]
                text = self.source[self.tokens[start].start() : self.tokens[i - 1].end()]
                rows.append((tokens, text))
                start = i
            value = self.values[i]
            if value in {"{", "[", "("}:
                depth += 1
            if value in {"}", "]", ")"}:
                depth -= 1
        if start < end:
            rows.append((
                self.values[start:end],
                self.source[self.tokens[start].start() : self.tokens[end - 1].end()],
            ))
        return rows

    def condition_span(self):
        if "where" not in self.values[self.rhs_end :]:
            return None
        start = self.values.index("where", self.rhs_end) + 1
        if self.values[start] == "{":
            end = self.closing(start, "{", "}")
            start = self.values.index("yield", start + 1) + 1
        else:
            end = self.derive if self.derive is not None else len(self.values) - 1
        return self.tokens[start].start(), self.tokens[end - 1].end()

    def attribute_sites(self):
        inventory, aliases = operation_inventory()
        return [
            op
            for op in self.rhs.operations()
            if inventory[aliases[op.name.split(".")[-1]]]["attributes"]
        ]


def operation_inventory():
    inventory = hlo_support.inventory()
    views = HERE.parent / "tepl/dialects/axis_views.tepl"
    if views.exists():
        inventory.update(hlo_support.declarations(views))
    for op in inventory.values():
        op["attributes"] = bool(re.search(r"\battrs\s*{", op["text"]))
    aliases = {
        alias: name
        for name, op in inventory.items()
        for alias in [name, *op["aliases"]]
    }
    return inventory, aliases


def eligibility(rule, aliases):
    from scripts.axis_fields import eligibility as check

    return check(rule, aliases)


def classify():
    from scripts.axis_fields import classify as build

    return build()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    result = classify()
    print("Classified", len(result["cases"]), "axis-only cases.")
