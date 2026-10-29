#!/usr/bin/env python3
"""Measure matched explicit/axis-auto rules without changing frozen evaluations."""

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parents[1]
LAB = HERE.parent
ROOT = LAB.parents[1]
sys.path.insert(0, str(LAB / "coverage"))
from scripts.axis_fields import AxisTask
from scripts.synthesis_cases import Rule

PROTOCOL = "matched_axis_specification_size_v1"
# Maximal-munch lexical units, rather than model/subword tokens. The same
# operator inventory is used for both languages; comments and spaces cost zero.
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


def sizes(source):
    ts = tokens(source)
    newlines = [m.start() for m in re.finditer("\n", source)]
    code_lines = set()
    layout_lines, pending = 0, False
    for t in ts:
        a, b = bisect_right(newlines, t.start()), bisect_right(newlines, t.end() - 1)
        code_lines.update(range(a, b + 1))
        # A single layout for both variants removes hand-formatting effects.
        if t[0] == "}" and pending:
            layout_lines += 1
            pending = False
        pending = True
        if t[0] in {"{", "}", ";", "=>"}:
            layout_lines += 1
            pending = False
    return dict(tokens=len(ts), code_lines=len(code_lines), layout_lines=layout_lines + pending)


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
        vs = values(expression)
        form = (
            "field_copy" if re.fullmatch(r"\w+\.attrs\.\w+", expression.strip())
            else "empty_list" if vs == ["[", "]"]
            else "literal" if all(v in {"[", "]", ",", "+", "-", "true", "false"} or v.isdigit() for v in vs)
            else "helper_reference" if expression.strip() in task.locals
            else "computed_expression"
        )
        field_rows.append(dict(
            **target, canonical_node=assignment["node"], expression=expression,
            form=form, explicit_tokens=len(vs), inferred_tokens=1,
        ))
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
    before = components(explicit, target_keys)
    after = components(inferred, target_keys)
    for category in {"lhs_pattern", "rhs_pattern", "tensor_constraints", "guard", "supplied_values", "descriptor_syntax", "normalization_adapter", "rule_syntax", "derivation_syntax"}:
        if before["lexemes"].get(category, []) != after["lexemes"].get(category, []):
            raise ValueError("non-axis change in matched pair: " + category)
    return dict(explicit=explicit, inferred=inferred, fields=field_rows,
                normalizations=task.normalizations, explicit_parts=before["counts"],
                inferred_parts=after["counts"])


def components(source, targets):
    rule = Rule(source)
    ts = tokens(source)
    start = next(i for i, t in enumerate(ts) if t[0] == "rule")
    ts = ts[start:]
    labels = ["rule_syntax"] * len(ts)

    def mark(a, b, label):
        for i, t in enumerate(ts):
            if a <= t.start() and t.end() <= b:
                labels[i] = label

    opening = next(t for t in ts if t[0] == "{")
    mark(opening.end(), rule.lhs.span[0], "tensor_constraints")
    mark(*rule.lhs.span, "lhs_pattern")
    mark(*rule.rhs.span, "rhs_pattern")
    if "where" in rule.values:
        i = rule.values.index("where")
        end = rule.closing(i + 1, "{", "}")
        mark(rule.tokens[i].start(), rule.tokens[end].end(), "guard")
    if rule.derive is not None:
        i = rule.derive
        end = rule.closing(i + 1, "{", "}")
        mark(rule.tokens[i].start(), rule.tokens[end].end(), "derivation_syntax")
    for stmt in assignments(source):
        if stmt["kind"] == "helper":
            mark(stmt["start"], stmt["end"], "local_helpers")
            continue
        if stmt["node"].startswith("axis_eval_") and not any((stmt["node"], f["field"]) in targets for f in stmt["fields"]):
            mark(stmt["start"], stmt["end"], "normalization_adapter")
            continue
        mark(stmt["start"], stmt["end"], "descriptor_syntax")
        for field in stmt["fields"]:
            label = "axis_values" if (stmt["node"], field["field"]) in targets else "supplied_values"
            mark(field["start"], field["end"], label)
    lexemes = {}
    for label, token in zip(labels, ts):
        lexemes.setdefault(label, []).append(token[0])
    counts = {k: len(v) for k, v in lexemes.items()}
    if sum(counts.values()) != sizes(rule_text(source))["tokens"]:
        raise ValueError("token attribution is not exhaustive")
    return dict(counts=counts, lexemes=lexemes)


def union_ranges(ranges):
    result = []
    for a, b in sorted(ranges):
        if not (1 <= a <= b):
            raise ValueError("invalid source range")
        if result and a <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def xla_size(manifest, names, key):
    by_file = {}
    for row in manifest["cases"]:
        if row["case"] not in names:
            continue
        for span in row[key]:
            by_file.setdefault(span["file"], []).append([span["line"], span["end_line"]])
    totals = Counter()
    for name, spans in by_file.items():
        source = (LAB / "catalog/upstream" / name).read_text()
        lines = source.splitlines(keepends=True)
        groups = union_ranges(spans)
        selected = {i for a, b in groups for i in range(a, b + 1)}
        if selected and max(selected) > len(lines):
            raise ValueError("source range exceeds file: " + name)
        newlines = [m.start() for m in re.finditer("\n", source)]
        code_lines = set()
        for t in tokens(source):
            a = bisect_right(newlines, t.start()) + 1
            b = bisect_right(newlines, t.end() - 1) + 1
            if a in selected:
                totals["tokens"] += 1
                code_lines.update(set(range(a, b + 1)) & selected)
        totals["code_lines"] += len(code_lines)
        totals["physical_lines"] += len(selected)
    return dict(totals)


def init_manifest():
    path = HERE / "data/xla_excerpts.json"
    if path.exists():
        raise ValueError("refusing to overwrite excerpt annotations")
    classification = json.loads((LAB / "coverage/data/synthesis_classification.json").read_text())
    catalog = {r["id"]: r for r in json.loads((LAB / "catalog/rules.json").read_text())["rules"]}
    cases = []
    for r in classification["cases"]:
        if r["control"] or r["task"] != "axis_inference":
            continue
        entry = catalog[r["catalog_id"]]
        cases.append(dict(case=r["case"], catalog_id=r["catalog_id"],
                          category=entry["category"], review_status="context_only",
                          tepl_scope=r["scope"], xla_conditions=entry["conditions"],
                          xla_policy=entry["policy"],
                          contexts=entry["sources"], excerpts=[], review_note=""))
    result = dict(schema_version=1, policy="Catalog ranges are context only; audited excerpts require a case-specific boundary and review note.", cases=cases)
    path.write_text(json.dumps(result, indent=2) + "\n")


def validate_manifest(manifest):
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported excerpt manifest")
    catalog = {r["id"]: r for r in json.loads((LAB / "catalog/rules.json").read_text())["rules"]}
    for row in manifest["cases"]:
        entry = catalog[row["catalog_id"]]
        if row["category"] != entry["category"] or row["contexts"] != entry["sources"]:
            raise ValueError("excerpt context differs from pinned catalog")
        if row["review_status"] not in {"context_only", "reviewed"}:
            raise ValueError("unknown excerpt review status")
        if row["review_status"] == "reviewed" and (not row["excerpts"] or not row["review_note"].strip()):
            raise ValueError("reviewed excerpts need ranges and a case-specific note")
        for span in row["contexts"] + row["excerpts"]:
            name = span["file"]
            if Path(name).name != name:
                raise ValueError("excerpt must use a pinned source filename")
            path = LAB / "catalog/upstream" / name
            if path.suffix not in {".cc", ".h"} or not path.is_file():
                raise ValueError("excerpt is not an XLA snapshot source")
            if not (1 <= span["line"] <= span["end_line"] <= len(path.read_text().splitlines())):
                raise ValueError("excerpt range is outside the pinned source")


def dependencies():
    files = list((ROOT / "src").rglob("*"))
    files += list((LAB / "coverage/scripts").glob("*.py"))
    files += list((LAB / "tepl/dialects").glob("*.tepl"))
    files += list((LAB / "tepl/rules/semantic").glob("*.tepl"))
    files += list((LAB / "tepl/rules/axis").glob("*.tepl"))
    files += list((LAB / "catalog/upstream").glob("*"))
    files += [LAB / "coverage/data/synthesis_classification.json",
              LAB / "coverage/results/synthesis.json", LAB / "coverage/data/verification_cases.json",
              LAB / "catalog/rules.json", LAB / "catalog/source_manifest.json",
              LAB / "tepl/tepl.toml", HERE / "data/xla_excerpts.json",
              HERE / "scripts/measure.py", HERE / "protocol.md",
              ROOT / "research/attribute-synthesis/proofs/completeness.md"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files)) if p.is_file()}


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


def run(compiler, timeout, jobs):
    frozen = json.loads((LAB / "coverage/data/synthesis_classification.json").read_text())
    prior = json.loads((LAB / "coverage/results/synthesis.json").read_text())
    previous = {r["case"]: r["outcome"] for r in prior["results"]}
    rows = [r for r in frozen["cases"] if r["task"] == "axis_inference" and not r["control"]]
    if len(rows) != 32:
        raise ValueError("changed cohort requires a new protocol version")
    manifest = json.loads((HERE / "data/xla_excerpts.json").read_text())
    validate_manifest(manifest)
    annotations = {r["case"]: r for r in manifest["cases"]}
    if set(annotations) != {r["case"] for r in rows} or len(annotations) != len(manifest["cases"]):
        raise ValueError("excerpt manifest must cover the complete preregistered cohort once")
    hashes, compiler_hash = dependencies(), sha(compiler)
    with tempfile.TemporaryDirectory(prefix="tepl-specification-size-") as temporary:
        work = Path(temporary)
        shutil.copytree(LAB / "tepl/dialects", work / "dialects")
        shutil.copyfile(LAB / "tepl/tepl.toml", work / "tepl.toml")

        def worker(row):
            original_path = LAB / "coverage" / row["source_file"]
            if sha(original_path) != row["source_sha256"] or sha(ROOT / row["file"]) != row["file_sha256"]:
                raise ValueError("selected fixture differs from recorded cohort: " + row["case"])
            original = original_path.read_text()
            pair = matched_pair(original)
            if values(pair["inferred"]) != values((ROOT / row["file"]).read_text()) or pair["normalizations"] != row["normalizations"]:
                raise ValueError("migration differs from recorded cohort: " + row["case"])
            record = dict(case=row["case"], catalog_id=row["catalog_id"],
                          category=annotations[row["case"]]["category"], scope=row["scope"],
                          forced_only=all(f["forced"] for f in row["axis_fields"]),
                          prior_outcome=previous[row["case"]], **pair)
            record["variants"] = {}
            for variant, source in [("original", original), ("explicit", pair["explicit"]), ("inferred", pair["inferred"])]:
                record["variants"][variant] = dict(
                    rule=sizes(rule_text(source)), imports=sizes(source[:source.index("rule ")]),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    **verify(source, row["case"], variant, compiler, timeout, work, pair["fields"]),
                )
            record["paired"] = all(record["variants"][v]["accepted"] for v in ["explicit", "inferred"])
            print(row["case"] + ": " + ", ".join(v + "=" + str(r["accepted"]) for v, r in record["variants"].items()), flush=True)
            return record

        negative = next(r for r in frozen["cases"] if r["control"])
        source = (LAB / "coverage" / negative["source_file"]).read_text()
        control = verify(source, negative["case"], "original", compiler, timeout, work, [])
        if control["accepted"] or control["check"]["returncode"] != 1 or "cannot verify rule" not in control["check"]["stdout"] + control["check"]["stderr"]:
            raise ValueError("negative control must fail semantic verification")
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            records = list(pool.map(worker, rows))
    if hashes != dependencies() or sha(compiler) != compiler_hash:
        raise ValueError("measurement inputs changed during the run")
    setup = [LAB / "tepl/dialects/hlo.tepl"]
    if any(r["normalizations"] for r in records):
        setup.append(LAB / "tepl/dialects/axis_views.tepl")
    result = dict(schema_version=1, protocol=PROTOCOL,
                  run_at_utc=datetime.now(timezone.utc).isoformat(),
                  tepl_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  compiler_sha256=compiler_hash, environment=platform.platform(),
                  python_version=platform.python_version(), timeout_seconds_per_command=timeout,
                  jobs=jobs, dependencies=hashes, negative_control=control, cases=records,
                  shared_dialects={str(p.relative_to(ROOT)): sizes(p.read_text()) for p in sorted(setup)})
    (HERE / "results/measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summary(rows):
    result = dict(cases=len(rows))
    for variant in ["original", "explicit", "inferred"]:
        result[variant] = dict(sum_metrics(rows, variant))
    reductions = [100 * (r["variants"]["explicit"]["rule"]["tokens"] - r["variants"]["inferred"]["rule"]["tokens"]) / r["variants"]["explicit"]["rule"]["tokens"] for r in rows]
    result["median_reduction_percent"] = statistics.median(reductions) if rows else 0
    return result


def failure_reason(result):
    for phase in ["check", "compile", "validation"]:
        evidence = result.get(phase)
        if not evidence:
            continue
        text = evidence["stdout"] + "\n" + evidence["stderr"]
        reason = re.search(r"^\s*reason:\s*(.*)", text, re.M)
        if reason:
            detail = reason[1]
            if "declared conditions and LHS legality do not establish legality obligation" in detail:
                return "The current checker cannot establish the handwritten attribute's legality from the LHS and conditions."
            return detail.replace("|", "\\|")
        if evidence["returncode"] != 0:
            lines = [line for line in text.splitlines() if line.strip()]
            return (lines[0] if lines else phase + " failed").replace("|", "\\|")
    return "proof or artifact acceptance requirement not met"


def sum_metrics(rows, variant):
    total = Counter()
    for row in rows:
        total.update(row["variants"][variant]["rule"])
    return total


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


def validate_measurement(data):
    """Recompute source counts and acceptance, not only report formatting."""
    if data["protocol"] != PROTOCOL:
        raise ValueError("wrong measurement protocol")
    for path, value in data["dependencies"].items():
        if sha(ROOT / path) != value:
            raise ValueError("stale size measurement dependency: " + path)
    frozen = json.loads((LAB / "coverage/data/synthesis_classification.json").read_text())
    expected = {r["case"]: r for r in frozen["cases"] if not r["control"] and r["task"] == "axis_inference"}
    if len(data["cases"]) != len(expected) or {r["case"] for r in data["cases"]} != set(expected):
        raise ValueError("measurement must cover every selected case once")
    for row in data["cases"]:
        original = (LAB / "coverage" / expected[row["case"]]["source_file"]).read_text()
        pair = matched_pair(original)
        for key, value in pair.items():
            if row[key] != value:
                raise ValueError("stale source or attribution: " + row["case"] + ":" + key)
        for variant, source in [("original", original), ("explicit", pair["explicit"]), ("inferred", pair["inferred"])]:
            evidence = row["variants"][variant]
            targets = len({f["node"] for f in row["fields"]}) if variant == "inferred" else 0
            if (
                evidence["rule"] != sizes(rule_text(source))
                or evidence["imports"] != sizes(source[:source.index("rule ")])
                or evidence["source_sha256"] != hashlib.sha256(source.encode()).hexdigest()
                or evidence["accepted"] != verified(evidence, targets)
            ):
                raise ValueError("invalid source metric or verification evidence: " + row["case"])
        if row["paired"] != all(row["variants"][v]["accepted"] for v in ["explicit", "inferred"]):
            raise ValueError("invalid paired cohort membership")
    control = data["negative_control"]
    if control["accepted"] or control["check"]["returncode"] != 1 or "cannot verify rule" not in control["check"]["stdout"] + control["check"]["stderr"]:
        raise ValueError("missing failed semantic negative control")


def render(data):
    validate_measurement(data)
    rows = data["cases"]
    paired = [r for r in rows if r["paired"]]
    s = summary(paired)
    total_saved = s["explicit"]["tokens"] - s["inferred"]["tokens"]
    percent = 100 * total_saved / s["explicit"]["tokens"] if paired else 0
    original_paired = [r for r in paired if r["variants"]["original"]["accepted"]]
    original_metrics = summary(original_paired)
    original_a = original_metrics["original"].get("tokens", 0)
    original_b = original_metrics["inferred"].get("tokens", 0)
    original_note = (
        f"The matched baseline uses named fields in both variants. In the **{len(original_paired)}** cases where the original compact handwritten rules also verify, "
        f"the original rules have **{original_a:,} tokens**, compared with **{original_b:,}** for the current inferred form. "
        f"The inferred form changes this by **{100*(original_b-original_a)/original_a:+.1f}%**. "
        "Inference shortens field expressions, but the current migration expands compact descriptor copies; "
        "the controlled reduction is not a claim that it shortens the original files."
        if original_a else "No original/explicit/inferred triple passes all checks; an original-source size claim is unavailable."
    )
    manifest = json.loads((HERE / "data/xla_excerpts.json").read_text())
    validate_manifest(manifest)
    audited = {r["case"] for r in manifest["cases"] if r["review_status"] == "reviewed" and r["excerpts"] and r["review_note"].strip()}
    ready = bool(paired) and {r["case"] for r in paired} <= audited
    context = xla_size(manifest, {r["case"] for r in paired}, "contexts")
    excerpt = xla_size(manifest, {r["case"] for r in paired}, "excerpts") if ready else None
    out = ["# XLA specification size: matched axis-only rules", "",
           f"A fresh run checks all **{len(rows)}** axis-inference cases from the frozen axis-only cohort. "
           f"**{sum(r['variants']['inferred']['accepted'] for r in rows)}** inferred rules verify, compile, and validate; "
           f"**{len(paired)}** cases pass the same checks for both matched variants and form the primary size cohort.", "",
           f"For these paired rules ({sum(len(r['fields']) for r in paired)} axis fields), lexical tokens decrease from **{s['explicit']['tokens']:,} to {s['inferred']['tokens']:,}** "
           f"(**{percent:.1f}%**). The median per-rule reduction is **{s['median_reduction_percent']:.1f}%**. "
           "This measures specification size, not usability, runtime speed, or equivalence to unrestricted XLA behavior.", "",
           original_note, "",
           "## Paired specification size", "",
           "| Representation | Layout lines | Lexical tokens |", "|---|---:|---:|"]
    if ready:
        out.append(f"| Audited corresponding XLA excerpts | — | {excerpt.get('tokens', 0):,} |")
    else:
        out.append("| Corresponding XLA implementation excerpts | Pending excerpt audit | Pending excerpt audit |")
    for variant, label in [("explicit", "TEPL with explicit RHS axis fields"), ("inferred", "TEPL with inferred RHS axis fields")]:
        out.append(f"| {label} | {s[variant]['layout_lines']:,} | {s[variant]['tokens']:,} |")
    out += ["", "Layout lines use the same brace/semicolon/arrow line-break policy for both TEPL variants. "
            "They exclude comments and blank lines. Lexical tokens, rather than lines, are the primary metric. "
            "Both totals include the complete rule: constraints, patterns, guards, descriptor syntax, and derivations. "
            "Imports and shared dialect definitions are reported separately. See [the protocol](../protocol.md).", "",
            "## Where the savings come from", "",
            "Every rule token belongs to exactly one component; the differences sum to the whole-rule reduction. "
            "Positive savings mean fewer tokens in the inferred variant.", "",
            "| Component | Explicit tokens | Inferred tokens | Saved |", "|---|---:|---:|---:|"]
    before, after = Counter(), Counter()
    for row in paired:
        before.update(row["explicit_parts"])
        after.update(row["inferred_parts"])
    labels = {"axis_values": "Axis field expressions", "local_helpers": "Local helper computations",
              "supplied_values": "Supplied non-axis field expressions", "descriptor_syntax": "Descriptor labels, field names, and punctuation",
              "normalization_adapter": "Fixed normalization adapter assignments", "tensor_constraints": "Tensor shape/dtype constraints",
              "lhs_pattern": "LHS pattern", "rhs_pattern": "RHS pattern", "guard": "Conditions and where locals",
              "rule_syntax": "Rule declaration and outer syntax", "derivation_syntax": "Derive block syntax"}
    for key, label in labels.items():
        out.append(f"| {label} | {before[key]:,} | {after[key]:,} | {before[key]-after[key]:,} |")
    if sum(before.values()) - sum(after.values()) != total_saved:
        raise ValueError("component savings do not reconcile")
    out += [f"| **Total** | **{sum(before.values()):,}** | **{sum(after.values()):,}** | **{total_saved:,}** |", "",
            "### Axis-expression detail", "",
            "This table counts only field-value expressions. Any removed helper statements appear in the component table, not again here.", "",
            "| Handwritten expression form | Fields | Explicit tokens | Auto tokens | Saved | Share of total savings |", "|---|---:|---:|---:|---:|---:|"]
    for form in ["field_copy", "empty_list", "literal", "helper_reference", "computed_expression"]:
        fs = [f for r in paired for f in r["fields"] if f["form"] == form]
        a, b = sum(f["explicit_tokens"] for f in fs), len(fs)
        share = 100*(a-b)/total_saved if total_saved else 0
        out.append(f"| {form.replace('_', ' ')} | {len(fs)} | {a:,} | {b:,} | {a-b:,} | {share:.1f}% |")
    out += ["", "## Breakdown by family", "",
            "| Family | Paired cases | Explicit tokens | Inferred tokens | Saved (%) |", "|---|---:|---:|---:|---:|"]
    for family in sorted({r["category"] for r in rows}):
        group = [r for r in paired if r["category"] == family]
        a, b = sum_metrics(group, "explicit")["tokens"], sum_metrics(group, "inferred")["tokens"]
        out.append(f"| {family} | {len(group)} | {a:,} | {b:,} | {100*(a-b)/a:.1f}% |" if a else f"| {family} | 0 | — | — | — |")
    out += ["", "## Sensitivity checks", "",
            "| Cohort | Cases | Explicit tokens | Inferred tokens | Saved (%) |", "|---|---:|---:|---:|---:|"]
    groups = [("All 32 attempted pairs (includes unproved rules)", rows),
              ("Paired verified rules", paired),
              ("Not marked scalar-forced", [r for r in paired if not r["forced_only"]]),
              ("All targets marked scalar-forced", [r for r in paired if r["forced_only"]]),
              ("Without unit-view normalization", [r for r in paired if not r["normalizations"]]),
              ("With unit-view normalization", [r for r in paired if r["normalizations"]])]
    for label, group in groups:
        a, b = sum_metrics(group, "explicit")["tokens"], sum_metrics(group, "inferred")["tokens"]
        out.append(f"| {label} | {len(group)} | {a:,} | {b:,} | {100*(a-b)/a:.1f}% |" if a else f"| {label} | 0 | — | — | — |")
    out += ["", "The frozen classifier marks a field forced only for a rank-zero broadcast's empty mapping. "
            "It does not establish that other tasks have multiple possible axis assignments. "
            "A general uniqueness or intended-optimization audit is a separate experiment."]
    original_tokens = sum_metrics(paired, "original")["tokens"]
    out += ["", "### Comparison with the original source", "",
            "This table uses only cases where all three TEPL variants verify, compile, and validate. "
            "It retains the original descriptor-copy idioms rather than expanding them away.", "",
            "| Representation | Cases | Layout lines | Lexical tokens |", "|---|---:|---:|---:|"]
    for variant, label in [("original", "Original handwritten explicit rules"), ("explicit", "Matched named-field explicit rules"), ("inferred", "Current inferred rules")]:
        metric = original_metrics[variant]
        out.append(f"| {label} | {len(original_paired)} | {metric.get('layout_lines', 0):,} | {metric.get('tokens', 0):,} |")
    out += ["", f"The original handwritten files have **{original_tokens:,} rule tokens** over the primary cohort; "
            f"**{len(original_paired)}/{len(paired)}** of those original rules pass current proof/compile/validation. "
            "Their difference from the matched explicit baseline reflects descriptor expansion, normalization, and dead-helper cleanup; "
            "it is not attributed to inference. All per-rule original sizes and outcomes are retained in the result JSON.", "",
            "## Per-case results", "",
            "O/E/I mean original handwritten / matched explicit / inferred. Pass means proof, compilation, and JSON validation all succeed. "
            "Savings for excluded pairs are descriptive only.", "",
            "| Case | O/E/I pass | Primary cohort | Explicit tokens | Inferred tokens | Saved |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        a, b = row["variants"]["explicit"]["rule"]["tokens"], row["variants"]["inferred"]["rule"]["tokens"]
        passes = "/".join("yes" if row["variants"][v]["accepted"] else "no" for v in ["original", "explicit", "inferred"])
        out.append(f"| {row['case']} | {passes} | {'yes' if row['paired'] else 'no'} | {a} | {b} | {a-b} |")
    out += ["", "### Unproved matched variants", "",
            "An unproved explicit baseline is not evidence that its handwritten assignment is wrong; "
            "it can expose a metadata-definedness or proof-support limit. These cases are excluded from the primary paired claim.", "",
            "| Case | Variant | Reason |", "|---|---|---|"]
    for row in rows:
        for variant in ["explicit", "inferred"]:
            if not row["variants"][variant]["accepted"]:
                out.append(f"| {row['case']} | {variant} | {failure_reason(row['variants'][variant])} |")
    out += ["", "## XLA excerpt audit", "",
            f"The catalog's enclosing source ranges contain **{context.get('code_lines', 0):,} nonblank code lines** and "
            f"**{context.get('tokens', 0):,} lexical tokens** for this cohort after file/range deduplication. "
            "These contexts include unrelated branches and source-policy checks. They are not corresponding implementation excerpts "
            "and must not be used as an XLA-to-TEPL compression ratio.", "",
            f"**{len({r['case'] for r in paired} & audited)}/{len(paired)}** paired cases have reviewed excerpt boundaries. "
            "The three-way table stays incomplete until every paired case is reviewed. "
            "The [excerpt manifest](../data/xla_excerpts.json) records contexts, selected ranges, and review notes. "
            "Count overlapping excerpt lines once globally, and separately document shared helpers and policy checks.", "",
            "## Shared setup and evidence", "",
            "| Shared dialect file (whole file; not a minimal dependency slice) | Code lines | Tokens |", "|---|---:|---:|"]
    for path, metric in data["shared_dialects"].items():
        out.append(f"| `{Path(path).name}` | {metric['code_lines']:,} | {metric['tokens']:,} |")
    imports = {v: sum(r["variants"][v]["imports"]["tokens"] for r in paired) for v in ["explicit", "inferred"]}
    out += ["", f"Per-file imports add **{imports['explicit']} explicit / {imports['inferred']} inferred tokens** before deduplication. "
            "Shared dialects are identical across the paired variants and are not multiplied by the number of rules. "
            "The negative control fails semantic verification as required.", "",
            f"Run UTC: `{data['run_at_utc']}`. Compiler SHA-256: `{data['compiler_sha256']}`. "
            f"Repository revision: `{data['tepl_revision']}`; input hashes include the experiment's working-tree sources. "
            f"Arithmetic policy: `algebraic`; {data['timeout_seconds_per_command']:g} seconds per command; {data['jobs']} workers.", "",
            "[Raw measurements](../results/measurement.json) include complete measured paired sources, per-field expressions, "
            "disjoint component counts, compiler commands and diagnostics, artifact hashes, and input hashes. "
            "Historical inference results remain unchanged. This is a source-size experiment on a fixed cohort; "
            "it does not establish that the generated rules preserve every XLA numerical, layout, or profitability policy.", ""]
    tex = [r"% Generated by specification_size/scripts/measure.py; see protocol.md.",
           r"% Controlled named-field layout, not the original compact descriptor-copy baseline.",
           f"% Original-source sensitivity: {len(original_paired)} verified triples; original {original_a:,} / inferred {original_b:,} lexical tokens.",
           r"\begin{tabular}{lrr}", r"\toprule", r"Representation & Layout lines & Lexical tokens \\", r"\midrule"]
    tex.append(r"Corresponding XLA excerpts & \multicolumn{2}{c}{Pending excerpt audit} \\" if not ready else f"Audited XLA excerpts & --- & {excerpt.get('tokens', 0):,} " + r"\\")
    for variant, label in [("explicit", "TEPL with explicit RHS axis fields"), ("inferred", "TEPL with inferred RHS axis fields")]:
        tex.append(f"{label} & {s[variant]['layout_lines']:,} & {s[variant]['tokens']:,} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    return {HERE / "reports/specification_size.md": "\n".join(out), HERE / "reports/specification_size.tex": "\n".join(tex)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init-manifest", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.timeout <= 0 or args.jobs <= 0:
        parser.error("timeout and jobs must be positive")
    if args.init_manifest:
        init_manifest()
        return
    data = run(args.compiler.resolve(strict=True), args.timeout, args.jobs) if args.run else json.loads((HERE / "results/measurement.json").read_text())
    for path, text in render(data).items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise ValueError("stale generated report: " + str(path.relative_to(ROOT)))
        else:
            path.write_text(text)


if __name__ == "__main__":
    main()
