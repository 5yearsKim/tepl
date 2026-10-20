#!/usr/bin/env python3
"""Prepare and check only the five manifest-listed inference pilot cases."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
MANIFEST = HERE / "data/pilot_cases.json"
RESULT = HERE / "results/pilot.json"
REPORT = HERE / "reports/pilot.md"
DIALECT = ROOT / "labs/xla_rewrite/tepl/dialects/hlo.tepl"
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|@?[A-Za-z_]\w*|\d+|=>|==|!=|<=|>=|&&|\|\||\.\.|\+\+|[^\s]')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomment(source):
    return re.sub(r"//[^\n]*", lambda m: " " * len(m[0]), source)


def tokens(source):
    return TOKEN.findall(uncomment(source))


def block(source, name):
    """Return the span and contents of a single named rule block."""
    text = uncomment(source)
    match = re.search(r"\b" + name + r"\s*\{", text)
    if not match:
        return None
    start, cursor, depth = match.start(), match.end(), 1
    body_start = cursor
    while depth:
        require(cursor < len(text), "unterminated " + name)
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return start, cursor, source[body_start:cursor - 1]


def rhs_span(source):
    start = source.index("=>") + 2
    ends = [b[0] for name in ("where", "derive") if (b := block(source, name))]
    return start, min(ends) if ends else source.rfind("}")


def annotations(rhs):
    spans, cursor = [], 0
    while cursor < len(rhs):
        if rhs[cursor] != "[":
            cursor += 1
            continue
        start, depth = cursor, 1
        cursor += 1
        while depth:
            require(cursor < len(rhs), "unterminated RHS annotation")
            depth += (rhs[cursor] == "[") - (rhs[cursor] == "]")
            cursor += 1
        spans.append((start, cursor))
    return spans


def invariant(source):
    start, end = rhs_span(source)
    rhs = uncomment(source[start:end])
    for left, right in reversed(annotations(rhs)):
        rhs = rhs[:left] + rhs[right:]
    guard = block(source, "where")
    prefix = source[:start]
    prefix = re.sub(r'from "[^"]+"', 'from "DIALECT"', prefix)
    return tokens(prefix), tokens(rhs), tokens(guard[2] if guard else "")


def make_variant(source, case, spec):
    names = set(case["rhs_descriptors"])
    inferred = set(spec["infer"])
    pins = spec.get("pins", {})
    require(inferred <= names and set(pins) <= inferred, "unknown inference target")
    start, end = rhs_span(source)
    rhs = source[start:end]
    found = re.findall(r"\[@(\w+)\]", uncomment(rhs))
    require(set(found) == names, "manifest RHS descriptors differ from source")
    require(len(found) == len(names), "pilot expects each RHS descriptor once")
    for name in inferred:
        fields = pins.get(name, {})
        require(set(fields) <= set(case["rhs_descriptors"][name]), "unknown pinned field")
        annotation = "[auto" + "".join(f", {k} = {v}" for k, v in fields.items()) + "]"
        rhs = rhs.replace("[@" + name + "]", annotation)
    result = source[:start] + rhs + source[end:]
    derive = block(result, "derive")
    keep = set(spec.get("keep_derived", []))
    if derive:
        if keep:
            # Pilot assistance keeps only independent constant constructors.
            # Reject locals/dependencies rather than silently dropping them.
            assignments = re.findall(r"@(\w+)\s*=\s*(\{[^{}]*\})\s*;", derive[2])
            require(len(assignments) == len(re.findall(r";", derive[2])),
                    "assisted derivation needs a more general dependency slicer")
            retained = [(n, body) for n, body in assignments if n in keep]
            require({n for n, _ in retained} == keep, "missing retained derivation")
            require(all("@" not in body for _, body in retained),
                    "pilot cannot drop dependencies of retained derivations")
            replacement = "derive {\n" + "\n".join(f"        @{n} = {body};" for n, body in retained) + "\n    }"
        else:
            replacement = ""
        result = result[:derive[0]] + replacement + result[derive[1]:]
    else:
        require(not keep, "no derivations to retain")
    require(invariant(result) == invariant(source), "pair changes structure or conditions")
    requested = sum(len(case["rhs_descriptors"][n]) - len(pins.get(n, {})) for n in inferred)
    require(requested > 0, "assistance supplies every target field")
    return result, requested


def code_metrics(source):
    start, end = rhs_span(source)
    rhs = uncomment(source[start:end])
    attribute_code = " ".join(rhs[a:b] for a, b in annotations(rhs))
    derive, guard = block(source, "derive"), block(source, "where")
    if derive:
        attribute_code += " " + source[derive[0]:derive[1]]
    return {"attribute_tokens": len(tokens(attribute_code)),
            "whole_rule_tokens": len(tokens(source[source.index("rule "):])),
            "guard_tokens": len(tokens(guard[2])) if guard else 0}


def prepare(manifest):
    jobs = []
    for case in manifest["cases"]:
        source = (ROOT / case["source"]).read_text()
        specs = [{"mode": "explicit"},
                 {"mode": "full_auto", "infer": list(case["rhs_descriptors"])}] + case["assisted"]
        for spec in specs:
            mode = spec["mode"]
            text, requested = (source, 0) if mode == "explicit" else make_variant(source, case, spec)
            path = HERE / "tepl/rules" / case["name"] / (mode + ".tepl")
            path.parent.mkdir(parents=True, exist_ok=True)
            relative_dialect = Path(os.path.relpath(DIALECT, path.parent)).as_posix()
            require(text.count('"dialects/hlo.tepl"') == 1, "unexpected dialect import")
            text = text.replace('"dialects/hlo.tepl"', '"' + relative_dialect + '"')
            text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"
            path.write_text(text)
            jobs.append({"case": case["name"], "mode": mode, "file": str(path.relative_to(ROOT)),
                         "policy": manifest["policy"], "control": False,
                         "requested_fields": requested, "metrics": code_metrics(text),
                         "expected_constructor_fields": [field for name in spec.get("infer", [])
                                                         for field in case["rhs_descriptors"][name]]})
    control = manifest["control"]
    original = next(j for j in jobs if j["case"] == control["source_case"] and j["mode"] == "explicit")
    text = (ROOT / original["file"]).read_text()
    require(text.count(control["replace"]) == 1, "control replacement is ambiguous")
    text = text.replace(control["replace"], control["with"])
    path = HERE / "tepl/rules" / control["source_case"] / "incorrect_control.tepl"
    path.write_text(text)
    jobs.append({"case": control["source_case"], "mode": "incorrect_control", "file": str(path.relative_to(ROOT)),
                 "policy": manifest["policy"], "control": True, "requested_fields": 0,
                 "metrics": code_metrics(text), "expected_constructor_fields": []})
    return jobs


def constructor_fields(stdout):
    fields, active = [], False
    for line in stdout.splitlines():
        if re.fullmatch(r"\s*constructor auto#\d+:", line):
            active = True
            continue
        field = re.fullmatch(r"        (\w+) = (.+)", line)
        if active and field:
            fields.append(field.groups())
        else:
            active = False
    return fields


def run_job(compiler, job, timeout):
    command = [str(compiler), "check", job["file"], "--semantic-policy=" + job["policy"]]
    started = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
        constructors = constructor_fields(stdout)
        complete = (Counter(f for f, _ in constructors) == Counter(job["expected_constructor_fields"])
                    and not any(re.search(r"auto#\d+\.", expr) for _, expr in constructors))
        proved = (returncode == 0 and stdout.count("status=proved") == 1
                  and re.search(r"^\s*additional-guard: true\s*$", stdout, re.MULTILINE)
                  and complete)
        status = "proved" if proved else "not_proved" if returncode == 1 and "cannot verify rule" in stderr else "error"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
        status, returncode, constructors = "timeout", None, []
    return {**job, "command": command, "status": status, "returncode": returncode,
            "seconds": round(time.perf_counter() - started, 6), "stdout": stdout, "stderr": stderr,
            "constructors": [{"field": f, "expression": e} for f, e in constructors]}


def report(data):
    rows = data["results"]
    by_case = {(r["case"], r["mode"]): r for r in rows}
    regular = [r for r in rows if not r["control"]]
    full = [r for r in regular if r["mode"] == "full_auto"]
    assisted = [r for r in regular if r["mode"].startswith("assisted")]
    recovered = {r["case"] for r in assisted if r["status"] == "proved"
                 and by_case[r["case"], "full_auto"]["status"] != "proved"}
    covered = {r["case"] for r in full + assisted if r["status"] == "proved"}
    explicit_proved = sum(r["status"] == "proved" for r in regular if r["mode"] == "explicit")
    lines = ["# Attribute inference: small pilot", "",
             "Generated by `scripts/pilot.py` from `results/pilot.json`.", "",
             "Purpose-selected examples; these fractions are not XLA corpus coverage estimates.", "",
             f"- Explicit baselines proved: **{explicit_proved}/{len(full)}**.",
             f"- Full-auto versions proved: **{sum(r['status'] == 'proved' for r in full)}/{len(full)}**.",
             f"- Assisted variants proved: **{sum(r['status'] == 'proved' for r in assisted)}/{len(assisted)}**; these overlap with full-auto successes.",
             f"- Additional cases recovered by assistance: **{len(recovered)}**.",
             f"- Cases proved with full or assisted inference: **{len(covered)}/{len(full)}**.",
             "- Incorrect constructor control: **" + next(r["status"] for r in rows if r["control"]) + "** (expected: not_proved).", "",
             "## Per-variant evidence", "",
             "Attribute tokens include RHS annotations and explicit derivations. Inferred-field counts exclude pins.",
             "Savings apply only to proved variants. Shared semantic definitions and guards are retained.", "",
             "| Case | Variant | Result | Fields requested | Attribute tokens | Reduction vs explicit |",
             "|---|---|---|---:|---:|---:|"]
    for r in regular:
        baseline = by_case[r["case"], "explicit"]["metrics"]["attribute_tokens"]
        count = r["metrics"]["attribute_tokens"]
        reduction = f"{baseline - count} ({100 * (baseline-count)/baseline:.1f}%)" if r["status"] == "proved" else "unproved"
        lines.append(f"| {r['case']} | {r['mode']} | {r['status']} | {r['requested_fields']} | {count} | {reduction} |")
    lines += ["", "## Synthesized expressions", ""]
    for r in regular:
        if r["status"] == "proved" and r["constructors"]:
            lines += [f"### {r['case']}: {r['mode']}", "", "```text"]
            lines += [f"{v['field']} = {v['expression']}" for v in r["constructors"]]
            lines += ["```", ""]
    lines += ["## Unproved variants and control", ""]
    for r in rows:
        if r["status"] != "proved":
            lines += [f"### {r['case']}: {r['mode']}", "", "```text", r["stderr"].strip() or r["status"], "```", ""]
    lines += ["## Interpretation and checkpoint", "",
              "Full and assisted inputs preserve the original LHS, RHS operations, tensor restrictions, guards and policy.",
              "Dot swapping retains rank-two restrictions; synthesized constants do not demonstrate general dot remapping.",
              "The slice example retains its original unit-stride guards. Transpose composition computes a metadata-dependent permutation.",
              "Assisted operation and assisted field results are distinct. Pinned fields printed in constructors are supplied, not inferred.",
              "A failed proof is not an impossibility result. The local algebraic model is not complete IEEE/XLA equivalence.",
              "Single-run CLI times are retained only as execution evidence, not performance measurements.",
              "Independent numerical validation, a corpus sweep, generality additions and scaling remain pending.",
              "**Stop here and obtain user confirmation before the next stage.**", ""]
    return "\n".join(lines)


def dependency_paths(jobs, manifest):
    return sorted({MANIFEST, Path(__file__).resolve(), DIALECT,
                   ROOT / "labs/xla_rewrite/tepl/tepl.toml",
                   ROOT / "labs/xla_rewrite/coverage/data/verification_cases.json",
                   ROOT / "BUILD.bazel", ROOT / "MODULE.bazel", ROOT / "MODULE.bazel.lock"}
                  | {ROOT / c["source"] for c in manifest["cases"]}
                  | {ROOT / j["file"] for j in jobs}
                  | {p for p in (ROOT / "src").rglob("*") if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--check", action="store_true", help="validate saved hashes/report without executing TEPL")
    args = parser.parse_args()
    require(args.timeout > 0, "timeout must be positive")
    compiler = args.compiler.resolve(strict=True)
    if args.check:
        data = json.loads(RESULT.read_text())
        require(digest(compiler) == data["compiler_sha256"], "compiler changed; rerun pilot")
        for name, expected in data["dependencies"].items():
            require(digest(ROOT / name) == expected, "stale input: " + name)
        require(REPORT.read_text() == report(data), "report differs from saved results")
        print("Pilot hashes and report are current; no TEPL checks executed.")
        return
    manifest = json.loads(MANIFEST.read_text())
    require(manifest["schema_version"] == 1 and len(manifest["cases"]) == 5, "this runner is limited to the five-case pilot")
    require(len({c['name'] for c in manifest['cases']}) == 5, "duplicate cases")
    jobs = prepare(manifest)
    dependencies = {str(p.relative_to(ROOT)): digest(p) for p in dependency_paths(jobs, manifest)}
    compiler_hash = digest(compiler)
    rows = []
    for job in jobs:
        row = run_job(compiler, job, args.timeout)
        rows.append(row)
        print(f"{row['case']} / {row['mode']}: {row['status']}", flush=True)
    require(digest(compiler) == compiler_hash and all(digest(ROOT / p) == h for p, h in dependencies.items()),
            "compiler or inputs changed during measurement")
    data = {"schema_version": 1, "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "tepl_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines(),
            "untracked_files": subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines(),
            "compiler": str(compiler), "compiler_sha256": compiler_hash,
            "environment": platform.platform(), "python": platform.python_version(),
            "timeout_seconds_per_variant": args.timeout, "selection": manifest["selection"],
            "dependencies": dependencies, "results": rows}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(data, indent=2) + "\n")
    REPORT.write_text(report(data))
    require(all(r["status"] == "proved" for r in rows if r["mode"] == "explicit"), "explicit baseline failed; inspect results")
    require(all(r["status"] not in ("error", "timeout") for r in rows), "pilot infrastructure error or timeout; inspect results")
    require(all(r["status"] == "not_proved" for r in rows if r["control"]), "incorrect constructor control was not semantically rejected")
    print("Saved " + str(REPORT.relative_to(ROOT)) + "; stop for user review.")


if __name__ == "__main__":
    main()
