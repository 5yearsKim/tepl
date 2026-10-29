#!/usr/bin/env python3
"""Compare reviewed XLA excerpts with explicit and compact inferred TEPL."""

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure as m

HERE, LAB, ROOT = m.HERE, m.LAB, m.ROOT
PROTOCOL = "xla_responsibility_implicit_axis_v1"
LABELS = {
    "matching": "Matching",
    "legality": "Legality checks",
    "attributes": "Attribute reasoning and syntax",
    "construction": "Replacement construction",
    "management": "Graph-management code",
}


def edit(source, changes):
    last = len(source) + 1
    for a, b, text in sorted(changes, reverse=True):
        if not 0 <= a <= b <= len(source) or b > last:
            raise ValueError("overlapping source edits")
        source = source[:a] + text + source[b:]
        last = a
    return source


def compact_sources(source):
    """Drop auto fields without altering supplied fields or default precedence."""
    rule = m.Rule(source)
    task = m.AxisTask(source)
    nodes = {n.label: n for n in rule.rhs.operations() if n.label}
    implicit_edits, compact_edits = [], []
    for stmt in m.assignments(source):
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
        implicit_edits.append((stmt["start"], stmt["end"], statement))
        compact_edits.append((stmt["start"], stmt["end"], statement if kept else ""))
    implicit = edit(source, implicit_edits)
    compact = edit(source, compact_edits)
    rule = m.Rule(compact)
    if rule.derive is not None and not m.assignments(compact):
        a = rule.tokens[rule.derive].start()
        b = rule.tokens[rule.closing(rule.derive + 1, "{", "}")].end()
        compact = compact[:a] + compact[b:]
    rule = m.Rule(compact)
    ts = m.tokens(m.rule_text(compact))
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
        parsed = m.Rule(text)
        def graph(node):
            return node.kind, node.name, tuple(graph(c) for c in node.children)
        where = []
        if "where" in parsed.values:
            start = parsed.values.index("where")
            end = parsed.closing(start + 1, "{", "}")
            where = parsed.values[start : end + 1]
        return graph(parsed.lhs), graph(parsed.rhs), where
    for text in [implicit, compact]:
        if invariant(text) != invariant(source):
            raise ValueError("compaction changed patterns or conditions")
    # Layout and spaces are immaterial, but the full original sources are saved.
    return dict(implicit=implicit, compact=compact)


def tepl_parts(source):
    rule = m.Rule(source)
    ts = m.tokens(source)
    start = next(i for i, t in enumerate(ts) if t[0] == "rule")
    ts = ts[start:]
    categories = ["management"] * len(ts)

    def mark(a, b, category):
        for i, token in enumerate(ts):
            if a <= token.start() and token.end() <= b:
                categories[i] = category

    opening = next(t for t in ts if t[0] == "{")
    mark(opening.end(), rule.lhs.span[0], "legality")
    mark(*rule.lhs.span, "matching")
    mark(*rule.rhs.span, "construction")
    for node in rule.rhs.operations():
        if node.attribute:
            mark(*node.attribute, "attributes")
    for keyword, category in [("where", "legality"), ("derive", "attributes")]:
        if keyword in rule.values:
            i = rule.values.index(keyword)
            end = rule.closing(i + 1, "{", "}")
            mark(rule.tokens[i].start(), rule.tokens[end].end(), category)
    result = Counter(categories)
    if sum(result.values()) != m.sizes(m.rule_text(source))["tokens"]:
        raise ValueError("TEPL attribution is not exhaustive")
    return dict(result)


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
        if path.name != block["file"] or m.sha(path) != manifest["source_hashes"][block["file"]]:
            raise ValueError("XLA snapshot differs")
        lines = path.read_text().splitlines()
        for a, b, category in block["spans"]:
            if category not in {*LABELS, "policy", "logging"} or not 1 <= a <= b <= len(lines):
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
        for token in m.tokens(text):
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


def dependencies():
    files = [HERE / "scripts/responsibility.py", HERE / "responsibility_protocol.md",
             HERE / "data/responsibility_spans.json", HERE / "results/measurement.json"]
    return dict(m.dependencies(), **{str(p.relative_to(ROOT)): m.sha(p) for p in files})


def run(compiler, jobs, timeout):
    baseline = json.loads((HERE / "results/measurement.json").read_text())
    m.validate_measurement(baseline)
    compiler_hash = m.sha(compiler)
    if compiler_hash != baseline["compiler_sha256"]:
        raise ValueError("rerun measure.py with this compiler first")
    manifest = json.loads((HERE / "data/responsibility_spans.json").read_text())
    audit = {r["case"]: r for r in manifest["cases"]}
    if len(audit) != len(manifest["cases"]) or set(audit) != {r["case"] for r in baseline["cases"]}:
        raise ValueError("XLA audit must cover all attempted cases exactly once")
    hashes = dependencies()
    with tempfile.TemporaryDirectory(prefix="tepl-size-implicit-") as directory:
        work = Path(directory)
        shutil.copytree(LAB / "tepl/dialects", work / "dialects")
        shutil.copyfile(LAB / "tepl/tepl.toml", work / "tepl.toml")

        def worker(row):
            sources = compact_sources(row["inferred"])
            result = dict(case=row["case"], category=row["category"], sources=sources, variants={})
            # Both omission-only and full compaction request the same holes as
            # field-auto. Descriptor count checks also cover implicit holes.
            for variant, source in sources.items():
                verified = m.verify(source, row["case"] + "_" + variant, "inferred", compiler, timeout, work, row["fields"])
                result["variants"][variant] = dict(verified, rule=m.sizes(m.rule_text(source)), parts=tepl_parts(source))
            result["comparable_tepl"] = all(r["accepted"] for r in row["variants"].values()) and all(r["accepted"] for r in result["variants"].values())
            result["primary"] = result["comparable_tepl"] and audit[row["case"]]["status"] == "reviewed"
            print(row["case"] + ": " + ", ".join(v + "=" + str(r["accepted"]) for v, r in result["variants"].items()), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            records = list(pool.map(worker, baseline["cases"]))
    if hashes != dependencies() or m.sha(compiler) != compiler_hash:
        raise ValueError("measurement inputs changed during execution")
    selected = {r["case"] for r in records if r["primary"]}
    xla, listing = xla_parts(manifest, selected)
    result = dict(protocol=PROTOCOL, run_at_utc=datetime.now(timezone.utc).isoformat(),
                  compiler_sha256=compiler_hash, dependencies=hashes, timeout_seconds=timeout,
                  jobs=jobs, cases=records, xla=xla)
    (HERE / "results/responsibility.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def render(data):
    if data["protocol"] != PROTOCOL:
        raise ValueError("wrong responsibility protocol")
    for path, value in data["dependencies"].items():
        if m.sha(ROOT / path) != value:
            raise ValueError("stale responsibility input: " + path)
    baseline = json.loads((HERE / "results/measurement.json").read_text())
    m.validate_measurement(baseline)
    before = {r["case"]: r for r in baseline["cases"]}
    manifest = json.loads((HERE / "data/responsibility_spans.json").read_text())
    audit = {r["case"]: r for r in manifest["cases"]}
    if len(data["cases"]) != len(before) or {r["case"] for r in data["cases"]} != set(before):
        raise ValueError("missing attempted case")
    for row in data["cases"]:
        if row["sources"] != compact_sources(before[row["case"]]["inferred"]):
            raise ValueError("changed compact source")
        for variant, evidence in row["variants"].items():
            source = row["sources"][variant]
            expected = len({f["node"] for f in before[row["case"]]["fields"]})
            if evidence["accepted"] != m.verified(evidence, expected) or evidence["rule"] != m.sizes(m.rule_text(source)) or evidence["parts"] != tepl_parts(source):
                raise ValueError("invalid compact measurement")
        paired = all(v["accepted"] for v in before[row["case"]]["variants"].values()) and all(v["accepted"] for v in row["variants"].values())
        if paired != row["comparable_tepl"] or row["primary"] != (paired and audit[row["case"]]["status"] == "reviewed"):
            raise ValueError("invalid common cohort")
    rows = [r for r in data["cases"] if r["primary"]]
    selected = {r["case"] for r in rows}
    xla, listing = xla_parts(manifest, selected)
    if xla != data["xla"]:
        raise ValueError("XLA responsibility counts changed")
    explicit, compact = Counter(), Counter()
    for row in rows:
        original = LAB / "coverage" / next(r for r in json.loads((LAB / "coverage/data/synthesis_classification.json").read_text())["cases"] if r["case"] == row["case"])["source_file"]
        explicit.update(tepl_parts(original.read_text()))
        compact.update(row["variants"]["compact"]["parts"])
    xtotal, etotal, ctotal = sum(xla.get(k, 0) for k in LABELS), sum(explicit.values()), sum(compact.values())
    if not xtotal or not etotal:
        raise ValueError("no common audited cohort")
    out = ["# XLA and TEPL specification size by responsibility", "",
           f"The primary table uses **{len(rows)} common cases** with reviewed XLA branch mappings and successful "
           "proof, compilation, and JSON validation for the original explicit, omitted-field, and compact inferred TEPL variants. "
           "XLA source is measured from the pinned snapshot; XLA itself is not compiled or executed in this experiment.", "",
           "| Responsibility | XLA | Explicit TEPL | Compact inferred TEPL | Saved from XLA |",
           "|---|---:|---:|---:|---:|"]
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Responsibility & XLA & Explicit TEPL & Compact inferred TEPL & Saved from XLA \\", r"\midrule"]
    for key, label in LABELS.items():
        a, b, c = xla.get(key, 0), explicit[key], compact[key]
        out.append(f"| {label} | {a:,} | {b:,} | {c:,} | {a-c:,} |")
        tex.append(f"{label} & {a:,} & {b:,} & {c:,} & {a-c:,} " + r"\\")
    out += [f"| **Total** | **{xtotal:,}** | **{etotal:,}** | **{ctotal:,}** | **{xtotal-ctotal:,}** |", "",
            f"Compact inferred TEPL has **{xtotal-ctotal:,} fewer lexical tokens ({100*(xtotal-ctotal)/xtotal:.1f}%)** than the selected XLA transformation excerpts "
            f"and **{etotal-ctotal:,} fewer ({100*(etotal-ctotal)/etotal:.1f}%)** than the original explicit TEPL rules.", "",
            "These are descriptive source-footprint differences. XLA branches often handle more ranks, operand orders, and cases than the TEPL specializations. "
            "The table does not establish equivalent support breadth or assign the entire reduction causally to the language. "
            "Source policy and logging are excluded rather than credited as savings; library implementations and shared dialect setup are also outside this rule/excerpt table. "
            "The five responsibilities partition every included token exactly once.", "",
            "## What omission and compaction contribute", "",
            "All rows below use the same primary cohort. The original idiomatic baseline keeps descriptor copies and spreads. "
            "The expanded baseline controls their expansion cost.", "",
            "| Representation | Tokens |", "|---|---:|"]
    totals = dict(original=etotal,
                  expanded=sum(before[r["case"]]["variants"]["explicit"]["rule"]["tokens"] for r in rows),
                  auto=sum(before[r["case"]]["variants"]["inferred"]["rule"]["tokens"] for r in rows),
                  implicit=sum(r["variants"]["implicit"]["rule"]["tokens"] for r in rows), compact=ctotal)
    for key, label in [("original", "Original explicit TEPL"), ("expanded", "Expanded explicit fields"), ("auto", "Explicit field-level auto"), ("implicit", "Omitted axis fields; labels and assignments retained"), ("compact", "Omitted fields plus unused-label/empty-block cleanup")]:
        out.append(f"| {label} | {totals[key]:,} |")
    out += ["", f"Expanded expressions → `auto` saves **{totals['expanded']-totals['auto']:,}** tokens. "
            f"Omitting inferable fields saves another **{totals['auto']-totals['implicit']:,}**. "
            f"Removing unused labels, empty assignments, and empty derive blocks saves another **{totals['implicit']-ctotal:,}**. "
            f"Expansion itself costs **{totals['expanded']-etotal:,}** tokens relative to the idiomatic original; "
            "it must be subtracted when claiming net practical savings.", "",
            "## Coverage and exclusions", "",
            f"All **{len(data['cases'])}** preregistered axis cases were attempted. "
            f"Omitted-field inference succeeds on **{sum(r['variants']['implicit']['accepted'] for r in data['cases'])}**; "
            f"compact inference succeeds on **{sum(r['variants']['compact']['accepted'] for r in data['cases'])}**. "
            f"**{sum(r['comparable_tepl'] for r in data['cases'])}** have accepted original explicit counterparts. "
            "The inherited negative control fails semantic verification.", "",
            "| Case | Explicit / omitted / compact accepted | In main table | Audit note |", "|---|---|---|---|"]
    for row in data["cases"]:
        passes = [before[row["case"]]["variants"]["original"]["accepted"], row["variants"]["implicit"]["accepted"], row["variants"]["compact"]["accepted"]]
        note = audit[row["case"]]["note"].replace("|", "\\|")
        out.append(f"| {row['case']} | {' / '.join('yes' if p else 'no' for p in passes)} | {'yes' if row['primary'] else 'no'} | {note} |")
    out += ["", "## XLA attribution and limits", "",
            f"File/line ranges are deduplicated before counting. The selected blocks also contain **{xla.get('policy', 0):,} policy tokens** "
            f"and **{xla.get('logging', 0):,} logging tokens**, recorded separately and excluded from the primary total. "
            "Annotations use each line's main responsibility; control syntax belongs to the block it controls. "
            "Different reasonable assignments can change individual rows, though they do not change the included-source total.", "",
            "Transformation-specific local helpers are included once. General XLA HLO/matcher/shape APIs and TEPL compiler/runtime/builtins are library dependencies, "
            "not expanded into these source counts. Whole shared TEPL dialect sizes appear in the [field-auto report](specification_size.md#shared-setup-and-evidence). "
            "An independent boundary/category review is recommended before publication.", "",
            "The algebraic policy does not reproduce all IEEE behavior or backend profitability policies. "
            "Successful inference constructs a valid witness, which can differ from the handwritten or intended optimization. "
            "No claim about runtime speed or authoring effort follows from token counts.", "",
            "## Evidence", "",
            "[Protocol](../responsibility_protocol.md), [annotated source listing](xla_excerpts.txt), "
            "[span ledger](../data/responsibility_spans.json), [measurements](../results/responsibility.json), "
            "and [LaTeX table](responsibility.tex).", "",
            f"Run UTC: `{data['run_at_utc']}`. Compiler SHA-256: `{data['compiler_sha256']}`. "
            f"{data['jobs']} workers; {data['timeout_seconds']:g} seconds per compiler command. "
            "Source, tool, protocol, and baseline hashes are included in the measurement JSON.", ""]
    tex += [f"Total & {xtotal:,} & {etotal:,} & {ctotal:,} & {xtotal-ctotal:,} " + r"\\", r"\bottomrule", r"\end{tabular}", ""]
    return {HERE / "reports/responsibility.md": "\n".join(out),
            HERE / "reports/responsibility.tex": "\n".join(tex),
            HERE / "reports/xla_excerpts.txt": listing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    data = run(args.compiler.resolve(strict=True), args.jobs, args.timeout) if args.run else json.loads((HERE / "results/responsibility.json").read_text())
    for path, text in render(data).items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise ValueError("stale responsibility report: " + str(path))
        else:
            path.write_text(text)


if __name__ == "__main__":
    main()
