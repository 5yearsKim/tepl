#!/usr/bin/env python3
"""Measure descriptor inference under the original XLA case conditions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

if __package__:
    from .paths import HERE, ROOT, TEPL
    from .synthesis_cases import Rule
else:
    from paths import HERE, ROOT, TEPL
    from synthesis_cases import Rule


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_variant(compiler, path, rule_name, expected_holes, timeout):
    command = [str(compiler), "check", str(path.relative_to(ROOT)),
               "--semantic-policy=algebraic", "--rule", rule_name]
    started = time.perf_counter()
    record = dict(file=str(path.relative_to(ROOT)), file_sha256=sha(path),
                  command=command, expected_inferred_descriptors=expected_holes)
    try:
        checked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                 timeout=timeout)
        proof = (checked.returncode == 0 and checked.stdout.count("status=proved") == 1
                 and "additional-guard: true" in checked.stdout)
        constructors = checked.stdout.count("constructor auto#")
        record.update(returncode=checked.returncode, stdout=checked.stdout,
                      stderr=checked.stderr, constructors=constructors,
                      ordered_path="law: ordered-requirements-construction" in checked.stdout,
                      status="not_proved", validated_json=False)
        if proof and constructors != expected_holes:
            raise ValueError(f"{path}: expected {expected_holes} inferred descriptors, got {constructors}")
        if proof:
            with tempfile.TemporaryDirectory(prefix="tepl-xla-auto-") as directory:
                artifact = Path(directory) / "program.json"
                compiled = subprocess.run(
                    [str(compiler), "compile", str(path.relative_to(ROOT)),
                     "--semantic-policy=algebraic", "-o", str(artifact)],
                    cwd=ROOT, text=True, capture_output=True, timeout=timeout)
                record["compile_returncode"] = compiled.returncode
                record["compile_diagnostics"] = compiled.stderr
                if compiled.returncode == 0:
                    valid = subprocess.run([str(compiler), "validate", str(artifact)],
                                           cwd=ROOT, text=True, capture_output=True,
                                           timeout=timeout)
                    record["validated_json"] = valid.returncode == 0
                    record["artifact_sha256"] = sha(artifact)
                    record["validation_returncode"] = valid.returncode
                    record["validation_diagnostics"] = valid.stderr
            record["status"] = "success" if record["validated_json"] else "not_compiled"
    except subprocess.TimeoutExpired:
        record.update(status="timeout", validated_json=False)
    record["seconds"] = round(time.perf_counter() - started, 6)
    return record


def measure_case(row, case, explicit, compiler, timeout):
    result = dict(row, explicit_status=explicit["status"], attempts=[])
    count = row["rhs_descriptors"]
    if row["category"] == 3:
        result["outcome"] = "unsupported"
        return result
    if not count:
        result["outcome"] = ("verified_without_attributes" if explicit["status"] == "proved"
                             else "not_proved_without_attributes")
        return result
    source = (HERE / case["file"]).read_text()
    rule = Rule(source)
    for inferred_count in range(count, 0, -1):
        for selected in itertools.combinations(range(count), inferred_count):
            mode = "full_auto" if inferred_count == count else "partial_auto"
            name = case["name"] + ("" if mode == "full_auto" else
                                    "__auto_" + "_".join(map(str, selected)))
            path = TEPL / "rules" / mode / (name + ".tepl")
            try:
                generated = rule.auto_source(selected)
            except ValueError as e:
                result["attempts"].append(dict(mode=mode, selected=list(selected),
                                               status="dependent_descriptor", reason=str(e)))
                continue
            path.parent.mkdir(exist_ok=True)
            path.write_text("// Generated inference variant; original graph and conditions preserved.\n"
                            + generated)
            attempt = check_variant(compiler, path, rule.name, len(selected), timeout)
            attempt.update(mode=mode, selected=list(selected))
            result["attempts"].append(attempt)
            if attempt["status"] == "success":
                if "outcome" not in result:
                    result.update(outcome=mode, inferred_descriptors=len(selected),
                                  explicit_descriptors=count - len(selected),
                                  successful_variant=attempt["file"])
        # Full and partial modes are measured independently. For partial mode,
        # stop after the first successful cardinality, which maximizes inference.
        if inferred_count < count and any(a["mode"] == "partial_auto" and
                                         a["status"] == "success" for a in result["attempts"]):
            break
    result.setdefault("outcome", "not_proved")
    return result


def dependencies():
    paths = [*sorted((ROOT / "src").rglob("*")),
             *sorted(TEPL.rglob("*.tepl")), TEPL / "tepl.toml",
             HERE / "data/synthesis_classification.json",
             HERE / "data/syntax_migration.json",
             HERE / "data/verification_cases.json",
             HERE / "results/verification.json", HERE / "results/compilation.json",
             HERE / "scripts/run_synthesis.py", HERE / "scripts/synthesis_cases.py",
             HERE / "scripts/migrate_syntax.py", HERE / "scripts/hlo_support.py",
             HERE.parent / "catalog/rules.json",
             ROOT / "examples/research/complete_attribute_synthesis_proof_kr.md"]
    paths += [HERE / "baselines/pre_axis_views" / name
              for name in ("verification.json", "synthesis.json")]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths if path.is_file()}


def run(compiler, timeout, jobs):
    classification = json.loads((HERE / "data/synthesis_classification.json").read_text())
    cases = {c["name"]: c for c in json.loads(
        (HERE / "data/verification_cases.json").read_text())["cases"]}
    explicit = {r["case"]: r for r in json.loads(
        (HERE / "results/verification.json").read_text())["results"]}
    for row in classification["cases"]:
        path = HERE / cases[row["case"]]["file"]
        if sha(path) != row["file_sha256"] or sha(path) != explicit[row["case"]]["file_sha256"]:
            raise ValueError("stale classification or explicit baseline: " + row["case"])
    compiler_hash = sha(compiler)
    source_hashes = {str(p.relative_to(ROOT)): sha(p)
                     for p in sorted((ROOT / "src").rglob("*")) if p.is_file()}
    rows = classification["cases"]
    def worker(row):
        result = measure_case(row, cases[row["case"]], explicit[row["case"]], compiler, timeout)
        print(f"{row['case']}: category {row['category']}, {result['outcome']}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        records = list(pool.map(worker, rows))
    if sha(compiler) != compiler_hash or any(sha(ROOT / p) != h for p, h in source_hashes.items()):
        raise ValueError("compiler changed during measurement")
    result = dict(schema_version=1,
                  run_at_utc=datetime.now(timezone.utc).isoformat(),
                  tepl_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  compiler_sha256=compiler_hash,
                  tracked_code_changes=subprocess.check_output(
                      ["git", "diff", "HEAD", "--name-only", "--", "src", "tests", "labs/xla_rewrite"], cwd=ROOT, text=True).splitlines(),
                  environment=platform.platform(), timeout_seconds_per_command=timeout,
                  jobs=jobs,
                  selection_policy="Full auto and partial auto measured independently. Partial subsets are tried in descending inferred count and lexicographic node order, stopping after the first successful cardinality. Full/partial overlap is counted once. No where conditions are added.",
                  dependencies=dependencies(), results=records)
    (HERE / "results/synthesis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.timeout <= 0 or args.jobs <= 0:
        parser.error("timeout and jobs must be positive")
    run(args.compiler.resolve(strict=True), args.timeout, args.jobs)


if __name__ == "__main__":
    main()
