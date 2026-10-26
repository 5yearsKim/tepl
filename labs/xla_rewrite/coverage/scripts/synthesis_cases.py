"""Parse corpus graph structure, classify cases, and erase RHS descriptors."""

from dataclasses import dataclass, field
import itertools
import hashlib
import json
import re

if __package__:
    from . import hlo_support
    from .paths import HERE, ROOT
else:
    import hlo_support
    from paths import HERE, ROOT

TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|=>|\.\.\.|\d+\.\d+(?:[eE][+-]?\d+)?|\d+|[A-Za-z_]\w*|[^\s]')
THEOREM_OPERATIONS = {"dot_general", "transpose", "reduce_sum", "add"}


@dataclass
class Node:
    name: str
    kind: str = "operation"
    label: str = ""
    attribute: tuple | None = None
    children: list = field(default_factory=list)

    def operations(self):
        return ([self] if self.kind == "operation" else []) + [
            op for child in self.children for op in child.operations()]


class Rule:
    def __init__(self, source):
        self.source = source
        self.tokens = [m for m in TOKEN.finditer(source)
                       if not m[0].startswith(("//", "/*"))]
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
        self.derive = (self.values.index("derive", self.rhs_end)
                       if "derive" in self.values[self.rhs_end:] else None)

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
            return node, i
        i += 1
        if v[i] == "let":
            name = v[i + 1]
            if v[i + 2] != "=":
                raise ValueError("invalid graph binding")
            child, end = self.graph(i + 3)
            if v[end] != ")":
                raise ValueError("unterminated binding")
            return Node(name, "binding", children=[child]), end + 1
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
        return node, i + 1

    def statements(self):
        if self.derive is None:
            return []
        start = self.derive + 2
        end = self.closing(self.derive + 1, "{", "}")
        depth = 0
        rows = []
        for i in range(start, end):
            value = self.values[i]
            if value in {"{", "[", "("}:
                depth += 1
            if value in {"}", "]", ")"}:
                depth -= 1
            if value == ";" and depth == 0:
                tokens = self.values[start:i]
                text = self.source[self.tokens[start].start():self.tokens[i].end()]
                rows.append((tokens, text))
                start = i + 1
        return rows

    def auto_source(self, selected):
        sites = self.attribute_sites()
        selected = set(selected)
        edits = [(*sites[i].attribute, "") for i in selected
                 if sites[i].attribute is not None]
        labels = {sites[i].label for i in selected if sites[i].label}
        if self.derive is not None:
            rows = [(tokens, text) for tokens, text in self.statements()
                    if not (len(tokens) > 3 and tokens[1:3] == [".", "attrs"]
                            and tokens[0] in labels)]
            # Drop unused local metadata formulas belonging to erased descriptors.
            changed = True
            while changed:
                changed = False
                for index, (tokens, text) in enumerate(rows):
                    if tokens[:1] != ["let"]:
                        continue
                    name = tokens[1]
                    others = [token for j, (ts, _) in enumerate(rows) if j != index
                              for token in ts]
                    if name not in others:
                        rows.pop(index)
                        changed = True
                        break
            end = self.closing(self.derive + 1, "{", "}")
            replacement = ("derive {\n        " + "\n        ".join(text for _, text in rows)
                           + "\n    }") if rows else ""
            edits.append((self.tokens[self.derive].start(), self.tokens[end].end(), replacement))
        source = self.source
        for start, end, replacement in sorted(edits, reverse=True):
            source = source[:start] + replacement + source[end:]
        # Inferred descriptors cannot remain dependencies of explicit providers.
        clean = re.sub(r"//[^\n]*", "", source)
        for label in labels:
            if re.search(r"\b" + re.escape(label) + r"\s*\.\s*attrs\b", clean):
                raise ValueError("explicit code still depends on inferred " + label + ".attrs")
        return source

    def attribute_sites(self):
        inventory, aliases = operation_inventory()
        return [op for op in self.rhs.operations()
                if inventory[aliases[op.name.split(".")[-1]]]["attributes"]]


def operation_inventory():
    inventory = hlo_support.inventory()
    for op in inventory.values():
        op["attributes"] = bool(re.search(r"\battrs\s*{", op["text"]))
    aliases = {alias: name for name, op in inventory.items()
               for alias in [name, *op["aliases"]]}
    return inventory, aliases


def eligibility(rule, aliases):
    def terms(node, rhs=False):
        if node.kind == "capture":
            return [frozenset([node.name])]
        if node.kind != "operation":
            raise ValueError("literal or bound-value pattern is outside the theorem fragment")
        op = aliases[node.name.split(".")[-1]]
        if op not in THEOREM_OPERATIONS:
            raise ValueError(op + " is outside the theorem operations")
        children = [terms(child, rhs) for child in node.children]
        if op == "add":
            return children[0] + children[1]
        if op != "dot_general":
            if rhs and len(children[0]) > 1:
                raise ValueError("RHS components must be an explicit outer sum")
            return children[0]
        if rhs and any(len(child) != 1 for child in children):
            raise ValueError("RHS contraction contains an unexpanded addition")
        result = []
        for left, right in itertools.product(*children):
            if left & right:
                raise ValueError("a component reads a capture more than once")
            result.append(left | right)
        return result
    try:
        lhs, rhs = terms(rule.lhs), terms(rule.rhs, True)
        if len(lhs) != len(set(lhs)) or len(rhs) != len(set(rhs)):
            raise ValueError("additive components share the same input set")
        if set(lhs) != set(rhs):
            raise ValueError("source and replacement components use different inputs")
        return True, "Read-once components and matching distinct input sets; theorem comparison is restricted to extents >= 2 and ideal arithmetic."
    except ValueError as e:
        return False, str(e)


def classify():
    inventory, aliases = operation_inventory()
    manifest = json.loads((HERE / "data/verification_cases.json").read_text())
    rows = []
    for case in manifest["cases"]:
        if case["control"]:
            continue
        rule = Rule((HERE / case["file"]).read_text())
        required = sorted({aliases[n.name.split(".")[-1]]
                           for n in rule.lhs.operations() + rule.rhs.operations()})
        missing = [name for name in required if not inventory[name]["semantic"]]
        eligible, reason = eligibility(rule, aliases)
        category = 3 if missing else 1 if eligible else 2
        rows.append(dict(case=case["name"], catalog_id=case["catalog_id"],
                         rule_name=rule.name,
                         file_sha256=hashlib.sha256((HERE / case["file"]).read_bytes()).hexdigest(),
                         category=category, theorem_eligible=eligible and not missing,
                         ordered_fragment=eligible and set(required) <= {"dot_general", "transpose"},
                         required_operations=required, missing_semantics=missing,
                         rhs_descriptors=len(rule.attribute_sites()),
                         reason=("Missing local semantics: " + ", ".join(missing)) if missing else reason,
                         scope=case["scope"]))
    # Every catalog schema is accounted for. Unported schemas retain a separate
    # porting state; semantic support is assessed from their mathematical family.
    unported_supported = {
        "elementwise-square-sqrt", "elementwise-square-rsqrt",
        "elementwise-max-min-clamp", "elementwise-min-max-clamp",
        "elementwise-power-half", "elementwise-rsqrt-negative-power",
        "elementwise-rsqrt-reciprocal", "elementwise-select-constant",
    }
    catalog = json.loads((HERE.parent / "catalog/rules.json").read_text())["rules"]
    schemas = []
    for schema in catalog:
        cases = [row for row in rows if row["catalog_id"] == schema["id"]]
        category = min(row["category"] for row in cases) if cases else (
            2 if schema["id"] in unported_supported else 3)
        schemas.append(dict(catalog_id=schema["id"], category=category,
                            ported=bool(cases), cases=[r["case"] for r in cases],
                            reason=("Classification of the scoped concrete algebraic cases; see per-case requirements."
                                    if cases else "No concrete case; scalar equation is representable but not measured."
                                    if category == 2 else "No concrete case; requires unsupported literal/range facts, general indexing/reducers, nonlocal matching, or variable-arity construction.")))
    result = dict(schema_version=1,
                  theorem_sha256=hashlib.sha256((ROOT / "examples/research/complete_attribute_synthesis_proof_kr.md").read_bytes()).hexdigest(),
                  policy="Classify before execution. Categories refer to the declared algebraic evaluation model; source matching adaptations remain in each case scope. No result status is used.",
                  categories={"1": "Theorem fragment", "2": "TEPL semantics beyond theorem", "3": "Unsupported required capabilities"},
                  cases=rows, schemas=schemas)
    path = HERE / "data/synthesis_classification.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    from collections import Counter
    result = classify()
    print("Case categories:", dict(Counter(row["category"] for row in result["cases"])))
    print("Schema categories:", dict(Counter(row["category"] for row in result["schemas"])))
