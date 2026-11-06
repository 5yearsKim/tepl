"""One canonical field-level axis task, with conservative unit-view normalization."""

import copy
import hashlib
import itertools
import json
import re

from .paths import HERE, ROOT, TEPL
from .synthesis_cases import Rule, operation_inventory

EXCLUDED = {
    "slice",
    "convolution",
    "general_convolution",
    "concatenate",
    "pad",
    "nonnegative_pad",
    "dynamic_slice",
    "dynamic_update_slice",
    "gather",
    "scatter",
    "scatter_add",
    "rotate",
    "iota",
    "reduce_window",
    "reduce_window_sum",
    "reduce_window_min",
    "reduce_window_max",
}
WIRE = {
    "dot_general",
    "transpose",
    "reverse",
    "reduce_sum",
    "multiply",
    "negate",
    "broadcast_in_dim",
    "remove_unit_axes",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_fields(text):
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char in ",;" and depth == 0:
            if text[start:i].strip():
                parts.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def conjuncts(text):
    """Only positive top-level conjuncts provide normalization premises."""
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                break
        if i != len(text) - 1:
            break
        text = text[1:-1].strip()
    depth, start, parts = 0, 0, []
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text[i : i + 2] == "&&":
            parts.extend(conjuncts(text[start:i]))
            start = i + 2
    if parts:
        return parts + conjuncts(text[start:])
    return [text]


def fields(inventory):
    result = {}
    for op, data in inventory.items():
        match = re.search(r"\battrs\s*{([^}]+)}", data["text"])
        result[op] = []
        if match:
            for item in split_fields(match[1]):
                m = re.fullmatch(
                    r"(\w+)\s*:\s*(\w+)(\[\])?(\?)?(?:\s*=\s*(.*))?", item, re.S
                )
                if not m:
                    raise ValueError("unsupported schema field: " + item)
                result[op].append(
                    dict(
                        name=m[1],
                        role=m[2],
                        list=bool(m[3]),
                        optional=bool(m[4]),
                        default=m[5],
                    )
                )
    return result


class AxisTask:
    def __init__(self, source):
        self.rule = Rule(source)
        self.inventory, self.aliases = operation_inventory()
        self.schemas = fields(self.inventory)
        self.targets, self.normalizations = [], []
        self.derived, self.normalization_conditions = [], []
        self.counter = 0
        self.rhs = {n.label: n for n in self.rule.rhs.operations() if n.label}
        self.providers, self.locals, self.remaining = {}, {}, []
        for tokens, text in self.rule.statements():
            m = re.fullmatch(r"\s*(\w+)\.attrs\s*=\s*(.*);\s*", text, re.S)
            if m:
                self.providers[m[1]] = m[2]
            else:
                self.remaining.append(text)
                m = re.fullmatch(r"\s*let\s+(\w+)\s*=\s*(.*);\s*", text, re.S)
                if m:
                    self.locals[m[1]] = m[2]
        self.guards = {}
        clean = re.sub(r"//[^\n]*", "", source)
        where = re.search(r"\bwhere\s*{(.*?)}", clean, re.S)
        self.conditions = where[1] if where else ""
        self.units = set()
        for atom in conjuncts(self.conditions):
            m = re.fullmatch(r"(\w+)\.attrs\.(\w+)\s*==\s*(\[[^\]]*\])", atom)
            if m:
                self.guards[m[1], m[2]] = m[3]
            m = re.fullmatch(r"(\w+)\s*==\s*1", atom)
            if m:
                self.units.add(m[1])
        prefix = source[: self.rule.lhs.span[0]]
        self.input_shapes = {
            name: self.list_value("[" + dims + "]")
            for name, dims in re.findall(r"\b(\w+)\s*:\s*\[([^\]]*)\]", prefix)
        }
        # Broadcast legality equates mapping length with operand rank.
        # A captured empty injection therefore also establishes scalar rank.
        for node in self.rule.lhs.operations():
            if (
                self.opname(node) == "broadcast_in_dim"
                and self.guards.get((node.label, "dimensions")) == "[]"
                and node.children[0].kind == "capture"
            ):
                name = node.children[0].name
                if self.input_shapes.get(name) is None:
                    self.input_shapes[name] = []
        self.shape_substitutions = {}

    def list_value(self, expression):
        if expression is None:
            return None
        expression = expression.strip()
        if not (expression.startswith("[") and expression.endswith("]")):
            return None
        items = split_fields(expression[1:-1])
        if any(not re.fullmatch(r"\w+", item) for item in items):
            return None
        return ["1" if x in getattr(self, "units", ()) else x for x in items]

    def opname(self, node):
        return self.aliases[node.name.split(".")[-1]]

    def provider(self, node):
        if node.attribute:
            return self.rule.source[node.attribute[0] + 1 : node.attribute[1] - 1]
        return self.providers.get(node.label)

    def field(self, node, name, seen=()):
        key = (node.label or str(node.span), name)
        if key in seen:
            raise ValueError("cyclic RHS attribute dependency")
        provider = self.provider(node)

        provider_stack = set()

        def read(value):
            if value is None:
                return None
            value = value.strip()
            if value in self.locals:
                if value in provider_stack:
                    raise ValueError("cyclic descriptor helper: " + value)
                provider_stack.add(value)
                result = read(self.locals[value])
                provider_stack.remove(value)
                return result
            if value.startswith("{") and value.endswith("}"):
                fallback = None
                for item in split_fields(value[1:-1]):
                    if item.startswith(".."):
                        fallback = item[2:].strip()
                        continue
                    m = re.match(r"(\w+)\s*[:=]\s*(.*)", item, re.S)
                    if m and m[1] == name:
                        return m[2]
                return read(fallback)
            if re.fullmatch(r"\w+\.attrs", value):
                return value + "." + name
            raise ValueError("unsupported descriptor provider: " + value)

        value = read(provider)
        if value is None:
            spec = next(f for f in self.schemas[self.opname(node)] if f["name"] == name)
            value = spec["default"]
        if value is None:
            raise ValueError("missing supplied field " + name)

        def replace(m):
            label, field = m[1], m[2]
            if label not in self.rhs:
                return m[0]
            other = self.rhs[label]
            spec = next(
                f for f in self.schemas[self.opname(other)] if f["name"] == field
            )
            if spec["role"] == "axis":
                raise ValueError(
                    "supplied field depends on automatic RHS axis: " + m[0]
                )
            return "(" + self.field(other, field, seen + (key,)) + ")"

        def check_helpers(expression, seen_helpers=frozenset()):
            # Keep helpers as supplied programs, but inspect their dependencies.
            for match in re.finditer(r"\b(\w+)\.attrs\.(\w+)", expression):
                replace(match)
            for name in set(re.findall(r"\b\w+\b", expression)) & self.locals.keys():
                if name in seen_helpers:
                    raise ValueError("cyclic supplied helper: " + name)
                check_helpers(self.locals[name], seen_helpers | {name})

        check_helpers(value)
        return re.sub(r"\b(\w+)\.attrs\.(\w+)", replace, value)

    def known(self, value):
        if value is None:
            return None
        m = re.fullmatch(r"(\w+)\.attrs\.(\w+)", value.strip())
        return self.guards.get((m[1], m[2]), value) if m else value

    def label(self):
        while True:
            name = "axis_eval_" + str(self.counter)
            self.counter += 1
            if name not in self.rule.values:
                return name

    def normalized_op(self, name, values, child, rhs):
        label = self.label()
        if rhs:
            self.derived.append(
                label
                + ".attrs = {"
                + ", ".join(k + ": " + v for k, v in values.items())
                + "};"
            )
        else:
            self.normalization_conditions.extend(
                label + ".attrs." + k + " == " + v for k, v in values.items()
            )
        return "(" + label + ": " + name + " " + child + ")"

    def render(self, node, rhs=False):
        source = self.rule.source
        if node.kind != "operation":
            if node.kind == "binding":
                raise ValueError("bound graph outside axis migration")
            return source[node.span[0] : node.span[1]], (
                [] if node.kind == "literal" else self.input_shapes.get(node.name)
            )
        op = self.opname(node)
        children = [self.render(c, rhs) for c in node.children]
        schema = self.schemas[op]
        values = {}
        if rhs:
            for f in schema:
                if f["role"] == "axis" and not f["optional"]:
                    values[f["name"]] = "auto"
                    self.targets.append(
                        dict(
                            node=node.label or "rhs@" + str(node.span[0]),
                            operation=op,
                            field=f["name"],
                            role="axis",
                            list=f["list"],
                            forced=(op == "broadcast_in_dim" and children[0][1] == []),
                        )
                    )
                else:
                    values[f["name"]] = self.field(node, f["name"])
        else:
            for f in schema:
                values[f["name"]] = self.guards.get(
                    (node.label, f["name"]), node.label + ".attrs." + f["name"]
                )
        shape = (
            children[0][1]
            if children
            and op in {"reverse", "negate", "exp", "log", "sqrt", "rsqrt", "abs"}
            else None
        )
        if (
            op
            in {"add", "multiply", "subtract", "divide", "power", "maximum", "minimum"}
            and children
            and children[0][1] is not None
            and all(child[1] == children[0][1] for child in children)
        ):
            shape = children[0][1]
        if op == "reshape":
            target = self.list_value(self.known(values["shape"]))
            incoming = children[0][1]
            if (
                incoming is None
                or target is None
                or [x for x in incoming if x != "1"] != [x for x in target if x != "1"]
            ):
                raise ValueError(
                    "reshape is not provably unit-only for every automatic axis assignment"
                )
            drop = [i for i, x in enumerate(incoming) if x == "1"]
            insert = [i for i, x in enumerate(target) if x == "1"]
            text = children[0][0]
            if drop:
                text = self.normalized_op(
                    "axis_view.remove_unit_axes", {"dimensions": str(drop)}, text, rhs
                )
            if insert:
                dimensions = [i for i, x in enumerate(target) if x != "1"]
                text = self.normalized_op(
                    "hlo.broadcast",
                    {
                        "dimensions": str(dimensions),
                        "new_extents": str([1] * len(insert)),
                    },
                    text,
                    rhs,
                )
            self.normalizations.append(
                dict(
                    kind="unit_only_reshape",
                    side="rhs" if rhs else "lhs",
                    node=node.label,
                    input_shape=incoming,
                    target_shape=target,
                    dropped_axes=drop,
                    inserted_axes=insert,
                )
            )
            if node.label:
                self.shape_substitutions[node.label + ".attrs.shape"] = (
                    "[" + ",".join(target) + "]"
                )
            return text, target
        if op == "broadcast_in_dim":
            dims = self.list_value(self.known(values["dimensions"]))
            new = self.list_value(self.known(values["new_extents"]))
            if (
                dims is not None
                and new is not None
                and children[0][1] is not None
                and all(x.isdigit() for x in dims)
            ):
                shape = [None] * (len(dims) + len(new))
                for i, d in enumerate(dims):
                    if int(d) < len(shape) and i < len(children[0][1]):
                        shape[int(d)] = children[0][1][i]
                rest = iter(new)
                shape = [next(rest) if x is None else x for x in shape]
        if rhs and schema:
            label = node.label or self.label()
            self.derived.append(
                label
                + ".attrs = {"
                + ", ".join(f["name"] + ": " + values[f["name"]] for f in schema)
                + "};"
            )
            attrs = ""
        else:
            label = node.label
            attrs = (
                source[node.attribute[0] : node.attribute[1]] if node.attribute else ""
            )
        label = label + ": " if label else ""
        return (
            "("
            + label
            + node.name
            + attrs
            + " "
            + " ".join(c[0] for c in children)
            + ")",
            shape,
        )

    def generate(self):
        lhs, _ = self.render(self.rule.lhs)
        rhs, _ = self.render(self.rule.rhs, True)
        tail = self.rule.source[self.rule.rhs.span[1] :]
        if self.rule.derive is not None:
            end = self.rule.closing(self.rule.derive + 1, "{", "}")
            a = self.rule.tokens[self.rule.derive].start() - self.rule.rhs.span[1]
            b = self.rule.tokens[end].end() - self.rule.rhs.span[1]
            tail = tail[:a] + tail[b:]
        text = self.rule.source[: self.rule.lhs.span[0]] + lhs + " => " + rhs + tail
        if self.normalization_conditions:
            extra = " && ".join("(" + x + ")" for x in self.normalization_conditions)
            original = self.conditions
            if original:
                text = text.replace(
                    "where {" + original + "}",
                    "where {" + original + " && " + extra + "}",
                )
            else:
                end = text.rfind("}")
                text = text[:end] + "where { " + extra + " }\n" + text[end:]
        remaining = list(self.remaining)
        while True:
            kept = []
            for statement in remaining:
                m = re.match(r"\s*let\s+(\w+)\b", statement)
                others = (
                    text
                    + "\n"
                    + "\n".join(self.derived)
                    + "\n"
                    + "\n".join(s for s in remaining if s is not statement)
                )
                if m and not re.search(r"\b" + re.escape(m[1]) + r"\b", others):
                    continue
                kept.append(statement)
            if kept == remaining:
                break
            remaining = kept
        remaining += self.derived
        if remaining:
            end = text.rfind("}")
            text = (
                text[:end]
                + "derive {\n "
                + "\n ".join(remaining)
                + "\n}\n"
                + text[end:]
            )
        for old, new in self.shape_substitutions.items():
            text = re.sub(r"\b" + re.escape(old) + r"\b", lambda _: new, text)
        return (
            "// Canonical axis-only evaluation task; no descriptor or non-axis inference.\n"
            + (
                'from "dialects/axis_views.tepl" import AxisViews as axis_view;\n'
                if self.normalizations
                else ""
            )
            + text
        )


def eligibility(rule, aliases):
    scalar_captures = set(
        re.findall(r"\b(\w+)\s*:\s*\[\s*\]", rule.source[: rule.lhs.span[0]])
    )
    reciprocal_inputs = set()

    def normalize(node):
        node = copy.deepcopy(node)
        if node.kind != "operation":
            return node
        op = aliases[node.name.split(".")[-1]]
        if op == "divide":
            numerator, denominator = node.children
            if (
                denominator.kind == "capture"
                and denominator.name in scalar_captures
                and numerator.kind == "literal"
                and numerator.name == "1"
            ):
                reciprocal_inputs.add(denominator.name)
                denominator.name = "$reciprocal_" + denominator.name
                return denominator
            if (
                denominator.kind == "operation"
                and aliases[denominator.name.split(".")[-1]] == "broadcast_in_dim"
            ):
                scalar = denominator.children[0]
                if scalar.kind == "capture" and scalar.name in scalar_captures:
                    reciprocal_inputs.add(scalar.name)
                    scalar.name = "$reciprocal_" + scalar.name
                    node.name = "multiply"
                    node.children = [normalize(numerator), denominator]
                    return node
            raise ValueError(
                "division is not an independent scalar reciprocal normalization"
            )
        node.children = [normalize(c) for c in node.children]
        return node

    def captures(node):
        return ({node.name} if node.kind == "capture" else set()).union(
            *(captures(c) for c in node.children)
        )

    def terms(node, rhs=False):
        if node.kind == "capture":
            return [frozenset([node.name])]
        if node.kind != "operation":
            raise ValueError("literal outside the structural fragment")
        op = aliases[node.name.split(".")[-1]]
        if op not in WIRE | {"add"}:
            raise ValueError(op + " requires scalar normalization or another adapter")
        children = [terms(c, rhs) for c in node.children]
        if op == "add":
            return children[0] + children[1]
        if len(children) == 1:
            return children[0]
        if rhs and any(len(c) > 1 for c in children):
            raise ValueError(
                "product of additive families outside the joint unary extension"
            )
        out = []
        for parts in itertools.product(*children):
            if sum(map(len, parts)) != len(frozenset.union(*parts)):
                raise ValueError("repeated capture within a component")
            out.append(frozenset.union(*parts))
        return out

    try:
        source, replacement = normalize(rule.lhs), normalize(rule.rhs)
        if len(reciprocal_inputs) > 1 or reciprocal_inputs & (
            captures(source) | captures(replacement)
        ):
            raise ValueError(
                "reciprocal scalar is correlated with another value occurrence"
            )
        lhs, rhs = terms(source), terms(replacement, True)
        if len(lhs) != len(set(lhs)) or len(rhs) != len(set(rhs)):
            raise ValueError("duplicate additive capture sets")
        if set(lhs) != set(rhs):
            raise ValueError("different capture sets")
        return (
            True,
            "Structural admission candidate; positive-extent domain and supplied-program obligations require audit.",
        )
    except ValueError as e:
        return False, str(e)


def classify():
    inventory, aliases = operation_inventory()
    manifest = json.loads((HERE / "data/verification_cases.json").read_text())
    output = TEPL / "rules/axis"
    output.mkdir(exist_ok=True)
    for old in output.glob("*.tepl"):
        if not old.read_text().startswith("// Canonical axis-only evaluation task;"):
            raise ValueError(
                "refusing to replace a non-generated axis task: " + str(old)
            )
        old.unlink()
    rows = []
    for case in manifest["cases"]:
        source_path = HERE / case["file"]
        source = source_path.read_text()
        rule = Rule(source)
        required = sorted(
            {
                aliases[n.name.split(".")[-1]]
                for n in rule.lhs.operations() + rule.rhs.operations()
            }
        )
        missing = [
            op for op in required if not inventory[op]["semantic"] and op != "reshape"
        ]
        row = dict(
            case=case["name"],
            catalog_id=case["catalog_id"],
            rule_name=rule.name,
            control=case["control"],
            source_file=case["file"],
            source_sha256=sha(source_path),
            required_operations=required,
            scope=case["scope"],
            in_scope=False,
            axis_fields=[],
            normalizations=[],
        )
        try:
            blocked = sorted(set(required) & EXCLUDED)
            if blocked:
                raise ValueError("excluded operations: " + ", ".join(blocked))
            if missing:
                raise ValueError("missing semantics: " + ", ".join(missing))
            task = AxisTask(source)
            generated = task.generate()
            path = output / (case["name"] + ".tepl")
            path.write_text(generated)
            parsed = Rule(generated)
            candidate, reason = eligibility(parsed, aliases)
            axis_count = len(task.targets)
            has_attrs = bool(rule.attribute_sites())
            row.update(
                in_scope=True,
                file=str(path.relative_to(ROOT)),
                file_sha256=sha(path),
                axis_fields=task.targets,
                normalizations=task.normalizations,
                task=(
                    "axis_inference"
                    if axis_count
                    else "supplied_non_axis" if has_attrs else "attribute_free"
                ),
                admission_status=(
                    (
                        "domain_restricted_candidate"
                        if "divide" in required
                        else "candidate"
                    )
                    if candidate and axis_count
                    else "outside" if axis_count else "not_applicable"
                ),
                admission_reason=reason,
                admission_domain=(
                    "positive extents; exact reals; independent read-once components; nonzero scalar divisor for reciprocal normalization"
                    if "divide" in required
                    else "positive extents; exact reals; independent read-once components"
                ),
            )
        except ValueError as e:
            row.update(
                task="out_of_scope",
                admission_status="not_audited",
                exclusion_reason=str(e),
            )
        rows.append(row)
    result = dict(
        schema_version=2,
        protocol="axis_only_v1",
        policy="One canonical axis-only rule per in-scope case; original conditions retained; no explicit-baseline or partial-descriptor experiment.",
        dependencies={
            str(path.relative_to(ROOT)): sha(path)
            for path in [
                HERE / "scripts/axis_fields.py",
                HERE / "scripts/synthesis_cases.py",
                HERE / "scripts/hlo_support.py",
                HERE / "scripts/paths.py",
                HERE / "data/verification_cases.json",
                TEPL / "dialects/hlo.tepl",
                TEPL / "dialects/axis_views.tepl",
                TEPL / "tepl.toml",
            ]
        },
        cases=rows,
    )
    (HERE / "data/synthesis_classification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
