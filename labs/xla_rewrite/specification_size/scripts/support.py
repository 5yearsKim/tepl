"""Source preparation, lexical scanning, and proof checks for the token experiment."""

from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
LAB = HERE.parent
ROOT = LAB.parents[1]
EVALUATION = LAB / "coverage/results/broadcast_update.json"
sys.path.insert(0, str(LAB / "coverage"))
from scripts.axis_fields import AxisTask
from scripts.synthesis_cases import Rule

# Maximal-munch lexical units shared by TEPL and C++; whitespace/comments cost zero.
LEX = re.compile(
    r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)")'
    r'|(?P<string>(?:u8|u|U|L)?"(?:\\[\s\S]|[^"\\])*"'
    r"|(?:u8|u|U|L)?'(?:\\[\s\S]|[^'\\])*')"
    r'|(?P<number>(?:0[xX][0-9a-fA-F\']+(?:\.[0-9a-fA-F\']*)?(?:[pP][+-]?\d+)?'
    r'|(?:\d[\d\']*(?:\.\d[\d\']*)?|\.\d+)(?:[eE][+-]?\d+)?)[a-zA-Z0-9_]*)'
    r'|(?P<identifier>[A-Za-z_]\w*)'
    r'|(?P<operator><=>|>>=|<<=|->\*|\.\.\.|\.\.|::|->|=>|\+\+|--|&&|\|\||<=|>=|==|!='
    r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|\.\*|##|[^\s])'
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(source):
    return [m for m in LEX.finditer(source) if m.lastgroup != "comment"]


def values(source):
    return [m[0] for m in tokens(source)]


def rule_text(source):
    rule = Rule(source)
    start = next(t.start() for t in tokens(source) if t[0] == "rule")
    return source[start : rule.tokens[-1].end()]


def assignments(source):
    """Return top-level field-value spans, including their descriptor labels."""
    rule = Rule(source)
    if rule.derive is None:
        return []
    ts = tokens(source)
    start = rule.tokens[rule.derive + 1].end()
    end = rule.tokens[rule.closing(rule.derive + 1, "{", "}")].start()
    ts = [t for t in ts if start <= t.start() < end]
    out, i = [], 0
    while i < len(ts):
        j, depth = i, 0
        while j < len(ts):
            v = ts[j][0]
            depth += v in {"{", "[", "("}
            depth -= v in {"}", "]", ")"}
            if v == ";" and depth == 0:
                break
            j += 1
        statement = ts[i : j + 1]
        if not statement or j == len(ts):
            raise ValueError("unterminated derive statement")
        row = dict(start=statement[0].start(), end=statement[-1].end(), fields=[])
        if statement[0][0] == "let":
            row.update(kind="helper", name=statement[1][0])
        elif [t[0] for t in statement[1:5]] == [".", "attrs", "=", "{"]:
            row.update(kind="descriptor", node=statement[0][0])
            k, first, level = 5, 5, 0
            while k < len(statement):
                v = statement[k][0]
                if level == 0 and v in {",", "}"}:
                    part = statement[first:k]
                    if part:
                        if len(part) < 3 or part[1][0] != ":":
                            raise ValueError("expected canonical named field")
                        row["fields"].append(dict(
                            field=part[0][0], start=part[2].start(), end=part[-1].end()
                        ))
                    first = k + 1
                    if v == "}":
                        break
                level += v in {"{", "[", "("}
                level -= v in {"}", "]", ")"}
                k += 1
        else:
            raise ValueError("unsupported derive statement in size experiment")
        out.append(row)
        i = j + 1
    return out


def matched_pair(source):
    """Keep the axis-only migration, but restore handwritten target expressions."""
    task = AxisTask(source)
    inferred = task.generate()
    nodes = {n.label or "rhs@" + str(n.span[0]): n for n in task.rule.rhs.operations()}
    holes = [
        (a, f) for a in assignments(inferred) for f in a["fields"]
        if inferred[f["start"] : f["end"]] == "auto"
    ]
    if len(holes) != len(task.targets):
        raise ValueError("migration target order changed")
    edits, field_rows = [], []
    for target, (assignment, field) in zip(task.targets, holes):
        if target["field"] != field["field"]:
            raise ValueError("migration field order changed")
        expression = task.field(nodes[target["node"]], target["field"])
        edits.append((field["start"], field["end"], expression))
        field_rows.append(dict(**target, canonical_node=assignment["node"]))
    explicit = inferred
    for a, b, text in reversed(edits):
        explicit = explicit[:a] + text + explicit[b:]
    # The migration removes helpers used only by automatic fields. Restore their
    # transitive dependency closure, not every old local or a folded old answer.
    stmts = assignments(explicit)
    for statement in reversed(stmts):
        if statement["kind"] == "helper":
            explicit = explicit[:statement["start"]] + explicit[statement["end"]:]
    needed = set(values(explicit)) & task.locals.keys()
    while True:
        expanded = needed | {
            name for helper in needed for name in values(task.locals[helper])
            if name in task.locals
        }
        if expanded == needed:
            break
        needed = expanded
    helpers = []
    for ts, text in task.rule.statements():
        if ts[0] == "let" and ts[1] in needed:
            helpers.append(text)
    parsed = Rule(explicit)
    insertion = parsed.tokens[parsed.derive + 1].end()
    explicit = explicit[:insertion] + "\n" + "\n".join(helpers) + explicit[insertion:]
    target_keys = {(r["canonical_node"], r["field"]) for r in field_rows}
    def supplied_tokens(text):
        excluded = []
        for stmt in assignments(text):
            if stmt["kind"] == "helper":
                excluded.append((stmt["start"], stmt["end"]))
            else:
                excluded.extend((f["start"], f["end"]) for f in stmt["fields"]
                                if (stmt["node"], f["field"]) in target_keys)
        return [t[0] for t in tokens(text)
                if not any(a <= t.start() and t.end() <= b for a, b in excluded)]
    if supplied_tokens(explicit) != supplied_tokens(inferred):
        raise ValueError("matched pair changed non-axis source")
    return dict(explicit=explicit, inferred=inferred, fields=field_rows)


def edit(source, changes):
    last = len(source) + 1
    for a, b, text in sorted(changes, reverse=True):
        if not 0 <= a <= b <= len(source) or b > last:
            raise ValueError("overlapping source edits")
        source = source[:a] + text + source[b:]
        last = a
    return source


def compact_source(source):
    """Drop auto fields without altering supplied fields or default precedence."""
    rule = Rule(source)
    task = AxisTask(source)
    nodes = {n.label: n for n in rule.rhs.operations() if n.label}
    compact_edits = []
    for stmt in assignments(source):
        if stmt["kind"] != "descriptor":
            continue
        node = nodes[stmt["node"]]
        schema = {f["name"]: f for f in task.schemas[task.opname(node)]}
        kept = []
        for field in stmt["fields"]:
            expression = source[field["start"] : field["end"]]
            spec = schema[field["field"]]
            # An explicit auto overrides a default; omission must not restore it.
            if expression == "auto" and spec["role"] == "axis" and not spec["optional"] and spec["default"] is None:
                continue
            kept.append(field["field"] + ": " + expression)
        statement = stmt["node"] + ".attrs = {" + ", ".join(kept) + "};"
        compact_edits.append((stmt["start"], stmt["end"], statement if kept else ""))
    compact = edit(source, compact_edits)
    rule = Rule(compact)
    if rule.derive is not None and not assignments(compact):
        a = rule.tokens[rule.derive].start()
        b = rule.tokens[rule.closing(rule.derive + 1, "{", "}")].end()
        compact = compact[:a] + compact[b:]
    rule = Rule(compact)
    ts = tokens(rule_text(compact))
    names = Counter(t[0] for t in ts)
    changes = []
    for node in rule.lhs.operations() + rule.rhs.operations():
        if node.label and names[node.label] == 1:
            label = next(t for t in rule.tokens if t.start() >= node.span[0] and t[0] == node.label)
            colon = next(t for t in rule.tokens if t.start() >= label.end())
            if colon[0] != ":":
                raise ValueError("expected node label")
            changes.append((label.start(), colon.end(), ""))
    compact = edit(compact, changes)
    def invariant(text):
        parsed = Rule(text)
        def graph(node):
            return node.kind, node.name, tuple(graph(c) for c in node.children)
        where = []
        if "where" in parsed.values:
            start = parsed.values.index("where")
            end = parsed.closing(start + 1, "{", "}")
            where = parsed.values[start : end + 1]
        return graph(parsed.lhs), graph(parsed.rhs), where
    if invariant(compact) != invariant(source):
        raise ValueError("compaction changed patterns or conditions")
    # Layout and spaces are immaterial, but the full original sources are saved.
    return compact


def xla_parts(manifest, cases):
    blocks = {r["id"]: r for r in manifest["blocks"]}
    requested = set()
    for row in manifest["cases"]:
        if row["case"] in cases:
            if row["status"] != "reviewed" or not row["note"].strip():
                raise ValueError("main cohort needs reviewed XLA boundaries")
            requested.update(row["blocks"])
    line_map = {}
    for name in sorted(requested):
        block = blocks[name]
        path = LAB / "catalog/upstream" / block["file"]
        if path.name != block["file"] or sha(path) != manifest["source_hashes"][block["file"]]:
            raise ValueError("XLA snapshot differs")
        lines = path.read_text().splitlines()
        for a, b, category in block["spans"]:
            if category not in {"matching", "legality", "attributes", "construction", "management", "policy", "logging"} or not 1 <= a <= b <= len(lines):
                raise ValueError("invalid annotated source span")
            for line in range(a, b + 1):
                key = (block["file"], line)
                if key in line_map and line_map[key] != category:
                    raise ValueError("conflicting XLA attribution: " + str(key))
                line_map[key] = category
    counts = Counter()
    listing = []
    for filename in sorted({name for name, line in line_map}):
        text = (LAB / "catalog/upstream" / filename).read_text()
        breaks = [v.start() for v in re.finditer("\n", text)]
        for token in tokens(text):
            line = bisect_right(breaks, token.start()) + 1
            if (filename, line) in line_map:
                counts[line_map[filename, line]] += 1
        listing.append(f"{filename} SHA-256 {manifest['source_hashes'][filename]}")
        previous = 0
        for line, text in enumerate(text.splitlines(), 1):
            if (filename, line) not in line_map:
                continue
            if line != previous + 1:
                listing.append("...")
            listing.append(f"{line:05d} [{line_map[filename, line]}] {text}")
            previous = line
    return dict(counts), "\n".join(listing) + "\n"


def command(args, timeout, work):
    start = time.perf_counter()
    try:
        r = subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        result = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    except subprocess.TimeoutExpired:
        result = dict(returncode=None, stdout="", stderr="command timed out")
    result["command"] = [str(a).replace(str(work), "<work>") for a in args]
    for key in ["stdout", "stderr"]:
        result[key] = result[key].replace(str(work), "<work>")
    result["seconds"] = round(time.perf_counter() - start, 6)
    return result


def verify(source, name, variant, compiler, timeout, work, targets):
    directory = work / name
    directory.mkdir(exist_ok=True)
    path = directory / (variant + ".tepl")
    path.write_text(source)
    check = command([compiler, "check", path, "--semantic-policy=algebraic", "--rule", Rule(source).name], timeout, work)
    proof = check["stdout"]
    result = dict(accepted=False, check=check)
    if check["returncode"] != 0 or proof.count("status=proved") != 1 or "additional-guard: true" not in proof:
        return result
    expected = len({f["node"] for f in targets}) if variant == "inferred" else 0
    if proof.count("constructor auto#") != expected:
        raise ValueError("inferred descriptor inventory changed: " + name)
    artifact = directory / (variant + ".json")
    compiled = command([compiler, "compile", path, "--semantic-policy=algebraic", "-o", artifact], timeout, work)
    result["compile"] = compiled
    if compiled["returncode"] != 0:
        return result
    result["validation"] = command([compiler, "validate", artifact], timeout, work)
    completed = json.loads(artifact.read_text())

    def holes(value):
        if isinstance(value, dict):
            return value.get("kind") in {"auto", "auto_field", "hole"} or any(holes(v) for v in value.values())
        return isinstance(value, list) and any(holes(v) for v in value)

    if holes(completed):
        raise ValueError("completed executable retains a hole")
    result.update(artifact_sha256=sha(artifact), accepted=result["validation"]["returncode"] == 0)
    return result


def verified(result, expected_descriptors):
    check = result["check"]
    proof = check["stdout"]
    return (
        check["returncode"] == 0
        and proof.count("status=proved") == 1
        and "additional-guard: true" in proof
        and proof.count("constructor auto#") == expected_descriptors
        and result.get("compile", {}).get("returncode") == 0
        and result.get("validation", {}).get("returncode") == 0
        and bool(re.fullmatch(r"[0-9a-f]{64}", result.get("artifact_sha256", "")))
    )


def accepted_task(row):
    check = row.get("check", {})
    return (
        row.get("task") == "axis_inference"
        and not row.get("control", False)
        and row.get("outcome") == "inferred"
        and row.get("requested_axis_fields", 0) > 0
        and row.get("validated_json") is True
        and check.get("returncode") == 0
        and check.get("stdout", "").count("status=proved") == 1
        and "additional-guard: true" in check.get("stdout", "")
        and row.get("compile", {}).get("returncode") == 0
        and row.get("validation", {}).get("returncode") == 0
        and bool(row.get("artifact_sha256"))
    )


def load_successful_tasks():
    """Select the 31 proof/compile/validation successes without a size proxy."""
    evaluation = json.loads(EVALUATION.read_text())
    axis = [r for r in evaluation["results"]
            if r["task"] == "axis_inference" and not r.get("control", False)]
    selected = [r for r in axis if accepted_task(r)]
    if len(axis) != 32 or len(selected) != 31 or len({r["case"] for r in selected}) != 31:
        raise ValueError("expected 31 distinct successes in the recorded 32-task run")
    for row in selected:
        if sha(ROOT / row["file"]) != row["file_sha256"]:
            raise ValueError("successful task source changed: " + row["case"])
    return evaluation, selected
