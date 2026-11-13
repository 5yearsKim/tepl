#!/usr/bin/env python3
"""Recount frozen author-written sources and optionally revalidate both forms."""
import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys

import run as artifact

sys.path.insert(0, str(artifact.ROOT / "labs/attribute_inferences/xla_rewrite/coverage"))
from scripts.synthesis_cases import Rule

# Maximal-munch lexical units from the retained measurement protocol.
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
LABELS = dict(matching="Matching (LHS)", legality="Legality checks",
              attributes="Attribute reasoning", construction="Replacement construction",
              other="Other syntax")


def tokens(text):
    return [m for m in LEX.finditer(text) if m.lastgroup != "comment"]


def sections(source):
    start = next(m.start() for m in tokens(source) if m[0] == "rule")
    text = source[start:]
    rule = Rule(text)
    spans = [(rule.lhs.span, "matching"), (rule.rhs.span, "construction")]
    for keyword, category in (("where", "legality"), ("derive", "attributes")):
        if keyword in rule.values:
            index = rule.values.index(keyword)
            end = rule.closing(index + 1, "{", "}")
            spans.append(((rule.tokens[index].start(), rule.tokens[end].end()), category))
    result = Counter(dict.fromkeys(LABELS, 0))
    for token in tokens(text):
        categories = [category for (a, b), category in spans if a <= token.start() and token.end() <= b]
        if len(categories) > 1:
            raise ValueError("overlapping TEPL source sections")
        result[categories[0] if categories else "other"] += 1
    return dict(result)


def xla_counts(manifest, selected):
    blocks = {r["id"]: r for r in manifest["blocks"]}
    requested = set()
    for row in manifest["cases"]:
        if row["case"] in selected:
            if row["status"] != "reviewed" or not row["note"].strip():
                raise ValueError("missing reviewed XLA mapping")
            requested.update(row["blocks"])
    source_root = artifact.ROOT / "labs/attribute_inferences/xla_rewrite/catalog/upstream"
    text = {}
    for name, digest in manifest["source_hashes"].items():
        path = source_root / name
        if path.name != name or artifact.sha(path) != digest:
            raise ValueError("XLA source snapshot changed")
        text[name] = path.read_text()
    line_map = {}
    for name in requested:
        block = blocks[name]
        for a, b, category in block["spans"]:
            if category not in {*LABELS, "management", "policy", "logging"} or not 1 <= a <= b <= len(text[block["file"]].splitlines()):
                raise ValueError("invalid XLA responsibility span")
            for line in range(a, b + 1):
                key = block["file"], line
                if key in line_map and line_map[key] != category:
                    raise ValueError("conflicting XLA responsibility")
                line_map[key] = category
    result = Counter()
    for name, source in text.items():
        breaks = [m.start() for m in re.finditer("\n", source)]
        for token in tokens(source):
            category = line_map.get((name, bisect_right(breaks, token.start()) + 1))
            if category:
                result["other" if category == "management" else category] += 1
    return dict(result)


def measure():
    artifact.verify_inputs()
    corpus = json.loads((artifact.FIXTURES / "source-size.json").read_text())
    ledger = json.loads((artifact.FIXTURES / "xla-spans.json").read_text())
    rows = [dict(case=r["case"], explicit=sections(r["explicit"]), inferred=sections(r["inferred"]),
                 xla_mapping=r["xla_mapping"]) for r in corpus["cases"]]
    _, coverage = artifact.verify_inputs()
    if {r["case"] for r in rows} != {r["case"] for r in coverage if r["axis_fields"] and r["paper_accepted"]}:
        raise ValueError("source-size cohort differs from 31 successful axis rules")
    matched = [r for r in rows if r["xla_mapping"]["status"] == "reviewed"]
    def total(group, variant):
        result = Counter(dict.fromkeys(LABELS, 0))
        for row in group:
            result.update(row[variant])
        return dict(result)
    xla = xla_counts(ledger, {r["case"] for r in matched})
    return dict(protocol=corpus["protocol"], matched_cases=len(matched), all_tepl_cases=len(rows),
                measurement_kind="static lexical recount of frozen source variants",
                dependencies={str(p.relative_to(artifact.ROOT)): artifact.sha(p) for p in
                              (Path(__file__).resolve(), artifact.HERE / "run.py",
                               artifact.FIXTURES / "source-size.json", artifact.FIXTURES / "xla-spans.json")},
                matched=dict(xla={key: xla.get(key, 0) for key in LABELS},
                             explicit=total(matched, "explicit"), inferred=total(matched, "inferred")),
                all_tepl=dict(explicit=total(rows, "explicit"), inferred=total(rows, "inferred")),
                cases=rows, xla_excluded={k: xla.get(k, 0) for k in ("policy", "logging")}), corpus


def render(result):
    text = ["# Rewrite source sizes", "", "Counts use lexical tokens, excluding comments, whitespace, imports, and shared declarations.", "",
            "The XLA comparison covers 30 reviewed mappings. All 31 inference-successful TEPL cases appear in the separate total.", "",
            "| Section | XLA (30) | Explicit TEPL (30) | Inferred TEPL (30) |", "|---|---:|---:|---:|"]
    for key, label in LABELS.items():
        text.append("| " + label + " | " + " | ".join(str(result["matched"][v][key]) for v in ("xla", "explicit", "inferred")) + " |")
    text.append("| Total | " + " | ".join(str(sum(result["matched"][v].values())) for v in ("xla", "explicit", "inferred")) + " |")
    text += ["", "All 31 TEPL rules: " + ", ".join(f"{v} = {sum(parts.values())}" for v, parts in result["all_tepl"].items()) + ".", "",
             "The dot scalar multiplier identity has no corresponding XLA dot excerpt in the pinned snapshot. Shared XLA tokens are deduplicated; annotations are retained in the fixture ledger.", "",
             "Counting frozen source bytes is a static measurement. Optional current proof/compilation outcomes are recorded separately in `verification.json`.", ""]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--compiler", type=Path, default=artifact.ROOT / "bazel-bin/tepl")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("choose a fresh output directory")
    result, corpus = measure()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact.save(out / "results.json", result)
    (out / "report.md").write_text(render(result))
    if args.verify:
        for name in ("inputs", "programs", "logs", "bin"):
            (out / name).mkdir()
        shutil.copytree(artifact.DIALECTS, out / "inputs/dialects")
        (out / "inputs/tepl.toml").write_text('tepl-version = "0.2.0"\n')
        binary = out / "bin/tepl"
        shutil.copy2(args.compiler.resolve(strict=True), binary)
        rows = []
        for row in corpus["cases"]:
            for variant in ("explicit", "inferred"):
                name = row["case"] + "." + variant
                source = out / "inputs" / (name + ".tepl")
                source.write_text(artifact.migrate(row[variant]))
                measured = artifact.attempt(dict(case=name, rule_name=Rule(source.read_text()).name), binary, out, "default", 30)
                rows.append(dict(variant=variant, **measured))
                print(name + ": " + measured["outcome"], flush=True)
        artifact.save(out / "verification.json", dict(compiler_sha256=artifact.sha(binary), records=rows))
        result["verification"] = {variant: dict(accepted=sum(r["accepted"] for r in rows if r["variant"] == variant),
                                                tasks=sum(r["variant"] == variant for r in rows))
                                  for variant in ("explicit", "inferred")}
        artifact.save(out / "results.json", result)
        (out / "report.md").write_text(render(result) + "\nCurrent verification: " +
                                      ", ".join(f"{v} {r['accepted']}/{r['tasks']}" for v, r in result["verification"].items()) + ".\n")
    print(render(result))


if __name__ == "__main__":
    main()
