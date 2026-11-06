#!/usr/bin/env python3
"""Validate the pinned XLA catalog and generate its Markdown reports."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request


ROOT = Path(__file__).resolve().parent
CATEGORIES = {
    "elementwise": "Elementwise",
    "transpose-reshape": "Transpose / reshape",
    "broadcast": "Broadcast",
    "reduction": "Reduction",
    "dot": "Dot",
    "convolution": "Convolution",
    "indexing": "Indexing",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(name):
    return json.loads((ROOT / name).read_text())


def visitor_methods(source):
    """Locate definitions for the review checklist, not individual rewrites.

    This small lexical scan handles the pinned file's comments and strings. It
    deliberately does not attempt C++ pattern extraction or guard inference.
    """
    stripped = re.sub(
        r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda match: "".join("\n" if c == "\n" else " " for c in match[0]),
        source,
    )
    methods = []
    for match in re.finditer(r"AlgebraicSimplifierVisitor::(\w+)\s*\(", stripped):
        start = stripped.find("{", match.end())
        require(start >= 0, f"Missing body for {match[1]}")
        depth, end = 1, start + 1
        while depth and end < len(stripped):
            depth += (stripped[end] == "{") - (stripped[end] == "}")
            end += 1
        require(depth == 0, f"Unclosed body for {match[1]}")
        methods.append((
            match[1],
            source.count("\n", 0, match.start()) + 1,
            source.count("\n", 0, end) + 1,
        ))
    return methods


def source_url(manifest, ref):
    item = next(f for f in manifest["files"] if Path(f["path"]).name == ref["file"])
    anchor = f"#L{ref['line']}"
    if "end_line" in ref:
        anchor += f"-L{ref['end_line']}"
    return f"{manifest['repository']}/blob/{manifest['revision']}/{item['path']}{anchor}"


def validate(manifest, catalog, review):
    require(re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]), "Invalid revision")
    require(manifest["repository"] == "https://github.com/openxla/xla", "Unexpected repository")
    require(catalog["schema_version"] == review["schema_version"] == 1, "Unknown schema")
    sources = {}
    for item in manifest["files"]:
        local = (ROOT / item["local_path"]).resolve()
        require(local.is_relative_to(ROOT / "upstream"), "Source path escapes upstream/")
        data = local.read_bytes()
        require(len(data) == item["bytes"], f"Changed size: {local.name}")
        require(hashlib.sha256(data).hexdigest() == item["sha256"], f"Changed hash: {local.name}")
        expected = f"https://raw.githubusercontent.com/openxla/xla/{manifest['revision']}/{item['path']}"
        require(item["url"] == expected, f"Unpinned URL: {local.name}")
        require(local.name not in sources, f"Duplicate source: {local.name}")
        sources[local.name] = data.decode().splitlines()

    def validate_ref(ref, owner):
        require(ref["file"] in sources, f"Unknown source in {owner}")
        lines = sources[ref["file"]]
        lo, hi = ref["line"], ref.get("end_line", ref["line"])
        require(type(lo) is int and type(hi) is int and 1 <= lo <= hi <= len(lines),
                f"Invalid source range in {owner}")
        return "\n".join(lines[lo - 1:hi])

    ids, equations = set(), set()
    by_symbol = {}
    for rule in catalog["rules"]:
        key = rule["id"]
        require(re.fullmatch(r"[a-z][a-z0-9-]*", key), f"Invalid ID: {key}")
        require(key not in ids, f"Duplicate ID: {key}")
        ids.add(key)
        require(rule["category"] in CATEGORIES, f"Unknown category: {key}")
        require(key.startswith(rule["category"] + "-"), f"ID/category mismatch: {key}")
        for field in ["rewrite", "conditions", "rhs_attributes", "policy"]:
            require(isinstance(rule[field], str) and rule[field].strip(), f"Empty {field}: {key}")
        require("->" in rule["rewrite"], f"Missing rewrite direction: {key}")
        fingerprint = (rule["category"], rule["rewrite"], rule["conditions"])
        require(fingerprint not in equations, f"Duplicate schema: {key}")
        equations.add(fingerprint)
        require(rule["sources"], f"Missing source: {key}")
        for ref in rule["sources"]:
            body = validate_ref(ref, key)
            require(re.search(r"\b" + re.escape(ref["symbol"]) + r"\s*\(", body),
                    f"Source range omits symbol {ref['symbol']}: {key}")
            if ref["file"] == "algebraic_simplifier.cc":
                by_symbol.setdefault(ref["symbol"], set()).add(key)
        require(rule["test_status"] == ("linked" if rule["tests"] else "source-only"),
                f"Inconsistent test status: {key}")
        for ref in rule["tests"]:
            suite, name = ref["name"].split(".", 1)
            lines = sources[ref["file"]]
            validate_ref(ref, key)
            declaration = "\n".join(lines[ref["line"] - 1:ref["line"] + 3])
            pattern = r"TEST(?:_F|_P)?\s*\(\s*" + re.escape(suite) + r"\s*,\s*" + re.escape(name) + r"\s*\)"
            require(re.search(pattern, declaration), f"Missing test {ref['name']}: {key}")

    actual = set(visitor_methods("\n".join(sources["algebraic_simplifier.cc"]) + "\n"))
    recorded = set()
    for method in review["methods"]:
        signature = (method["symbol"], method["line"], method["end_line"])
        require(signature not in recorded, f"Duplicate method review: {signature}")
        recorded.add(signature)
        require(method["status"] in {"cataloged", "support", "excluded"},
                f"Unreviewed method: {signature}")
        require(method["reason"].strip(), f"Missing review reason: {signature}")
        require(set(method["rule_ids"]) <= ids, f"Unknown rule in {signature}")
        expected = by_symbol.get(method["symbol"], set())
        require(set(method["rule_ids"]) == expected, f"Stale method links: {signature}")
        require((method["status"] == "cataloged") == bool(expected), f"Review status mismatch: {signature}")
    require(actual == recorded, f"Method inventory mismatch; missing={actual-recorded}, extra={recorded-actual}")
    for branch in review["excluded_branches"]:
        require(branch["symbol"] in {m[0] for m in actual} and branch["reason"].strip(),
                "Invalid branch exclusion")


def render_catalog(manifest, catalog):
    rules = catalog["rules"]
    counts = Counter(r["category"] for r in rules)
    missing_tests = sum(not r["tests"] for r in rules)
    lines = [
        "# XLA Algebraic Rewrite Catalog", "",
        "Generated from `rules.json` by `summarize_catalog.py`. See [README.md](README.md) for scope, notation, and counting policy.", "",
        f"XLA revision: `{manifest['revision']}`. These are source summaries, not TEPL verification results.", "",
        "## Category counts", "", "| Category | Schemas |", "|---|---:|",
    ]
    lines += [f"| {title} | {counts[key]} |" for key, title in CATEGORIES.items()]
    lines += [f"| **Total** | **{len(rules)}** |", "",
              f"{len(rules)-missing_tests} entries link related upstream tests; {missing_tests} are source-only. Links may include negative tests and do not establish that a schema was executed or proved.", ""]
    for category, title in CATEGORIES.items():
        lines += [f"## {title}", ""]
        for rule in rules:
            if rule["category"] != category:
                continue
            lines += [f"### {rule['id']}", "", f"```text\n{rule['rewrite']}\n```", "",
                      f"- **Conditions:** {rule['conditions']}",
                      f"- **RHS attributes:** {rule['rhs_attributes']}"]
            if rule["policy"] != "No additional local policy check.":
                lines.append(f"- **Source policy:** {rule['policy']}")
            refs = ", ".join(f"[{ref['symbol']}]({source_url(manifest, ref)})" for ref in rule["sources"])
            lines.append(f"- **Source:** {refs}.")
            if rule["tests"]:
                tests = ", ".join(f"[{ref['name']}]({source_url(manifest, ref)})" for ref in rule["tests"])
                lines.append(f"- **Related tests:** {tests}.")
            else:
                lines.append("- **Related tests:** No dedicated test identified in the captured file.")
            if rule["notes"]:
                lines.append(f"- **Note:** {rule['notes']}")
            lines.append("")
    return "\n".join(lines)


def render_review(manifest, review):
    counts = Counter(m["status"] for m in review["methods"])
    lines = ["# XLA Source Review", "", "Generated from `source_review.json` by `summarize_catalog.py`.", "",
             f"Revision: `{manifest['revision']}`. {review['coverage_unit']}", "",
             f"{len(review['methods'])} method definitions: {counts['cataloged']} linked to catalog entries, {counts['support']} support methods, {counts['excluded']} excluded. No pending methods in this inventory.", "",
             "This checklist tracks source review, not a mechanically established count of every possible rewrite. Schema boundaries and exclusions are manual decisions.", "",
             "## Method checklist", "", "| Source method | Status | Linked schemas | Reason |", "|---|---|---:|---|"]
    for method in review["methods"]:
        ref = dict(file="algebraic_simplifier.cc", line=method["line"], end_line=method["end_line"])
        lines.append(f"| [{method['symbol']}]({source_url(manifest, ref)}) | {method['status']} | {len(method['rule_ids'])} | {method['reason']} |")
    lines += ["", "## Excluded branches inside reviewed methods", "", "| Method | Branch | Reason |", "|---|---|---|"]
    lines += [f"| {b['symbol']} | {b['branch']} | {b['reason']} |" for b in review["excluded_branches"]]
    lines += ["", "## Shared helpers", "", "| Helper | Role |", "|---|---|"]
    lines += [f"| {h['symbol']} | {h['reason']} |" for h in review["shared_helpers"]]
    lines.append("")
    return "\n".join(lines)


def fetch_sources(manifest):
    for item in manifest["files"]:
        with urllib.request.urlopen(item["url"], timeout=60) as response:
            data = response.read()
        require(hashlib.sha256(data).hexdigest() == item["sha256"], f"Downloaded hash mismatch: {item['path']}")
        path = (ROOT / item["local_path"]).resolve()
        require(path.is_relative_to(ROOT / "upstream"), "Source path escapes upstream/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate and check generated files without writing")
    parser.add_argument("--fetch", action="store_true", help="Restore the exact source files recorded in the manifest")
    args = parser.parse_args()
    manifest, catalog, review = (read_json(name) for name in (
        "source_manifest.json", "rules.json", "source_review.json"))
    if args.fetch:
        fetch_sources(manifest)
    validate(manifest, catalog, review)
    reports = {"catalog.md": render_catalog(manifest, catalog),
               "source_review.md": render_review(manifest, review)}
    for name, content in reports.items():
        path = ROOT / name
        if args.check:
            require(path.exists() and path.read_text() == content, f"Regenerate {name}")
        else:
            path.write_text(content)
    print(f"Validated {len(catalog['rules'])} schemas and {len(review['methods'])} method reviews.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError) as error:
        print(f"Catalog error: {error}", file=sys.stderr)
        sys.exit(1)
