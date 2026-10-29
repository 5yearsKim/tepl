#!/usr/bin/env python3
"""Run the single canonical axis-only evaluation task for each in-scope case."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time

if __package__:
    from .paths import HERE, ROOT
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.paths import HERE, ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command, timeout):
    started = time.perf_counter()
    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=timeout
    )
    return dict(
        command=[str(x) for x in command],
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        seconds=round(time.perf_counter() - started, 6),
    )


def measure_case(row, compiler, timeout, artifact_dir):
    result = dict(row)
    if not row["in_scope"]:
        return dict(result, outcome="outside_scope")
    path = ROOT / row["file"]
    if sha(path) != row["file_sha256"]:
        raise ValueError("stale axis task: " + row["case"])
    source = path.read_text()
    requested = len(row["axis_fields"])
    if len(re.findall(r":\s*auto\b", source)) != requested:
        raise ValueError("axis hole inventory does not match task: " + row["case"])
    result.update(
        outcome="not_proved", requested_axis_fields=requested, validated_json=False
    )
    started = time.perf_counter()
    try:
        check = run_command(
            [
                str(compiler),
                "check",
                row["file"],
                "--semantic-policy=algebraic",
                "--rule",
                row["rule_name"],
            ],
            timeout,
        )
        result["check"] = check
        proved = (
            check["returncode"] == 0
            and check["stdout"].count("status=proved") == 1
            and "additional-guard: true" in check["stdout"]
        )
        result["search_path"] = [
            line.strip()
            for line in check["stdout"].splitlines()
            if line.strip().startswith("law:")
        ]
        result["inferred_descriptors"] = check["stdout"].count("constructor auto#")
        expected = len({f["node"] for f in row["axis_fields"]})
        if proved and result["inferred_descriptors"] != expected:
            raise ValueError(
                "requested descriptors not all synthesized: " + row["case"]
            )
        if not proved:
            if "semantic-inference:" not in check["stdout"]:
                result["outcome"] = "invalid_task"
            return result
        artifact = artifact_dir / (row["case"] + ".json")
        compiled = run_command(
            [
                str(compiler),
                "compile",
                row["file"],
                "--semantic-policy=algebraic",
                "-o",
                str(artifact),
            ],
            timeout,
        )
        result["compile"] = compiled
        if compiled["returncode"]:
            result["outcome"] = "compile_failed"
            return result
        validated = run_command([str(compiler), "validate", str(artifact)], timeout)
        result["validation"] = validated
        result["validated_json"] = validated["returncode"] == 0
        result["artifact_sha256"] = sha(artifact)

        # Completed executable IR cannot retain an AutoField or symbolic hole.
        def unresolved(value):
            if isinstance(value, dict):
                if value.get("kind") in {"auto", "auto_field", "hole"}:
                    return True
                return any(unresolved(v) for v in value.values())
            return isinstance(value, list) and any(unresolved(v) for v in value)

        if unresolved(json.loads(artifact.read_text())):
            raise ValueError("unresolved field in completed artifact: " + row["case"])
        result["outcome"] = (
            ("inferred" if requested else "verified")
            if result["validated_json"]
            else "validation_failed"
        )
    except subprocess.TimeoutExpired:
        result["outcome"] = "timeout"
    finally:
        result["seconds"] = round(time.perf_counter() - started, 6)
    return result


def dependencies():
    files = [p for p in (ROOT / "src").rglob("*") if p.is_file()]
    files += [
        HERE / "scripts" / name
        for name in [
            "axis_fields.py",
            "synthesis_cases.py",
            "run_synthesis.py",
            "synthesis_report.py",
            "hlo_support.py",
            "paths.py",
        ]
    ]
    files += list((HERE.parent / "tepl/dialects").glob("*.tepl"))
    files += list((HERE.parent / "tepl/rules/axis").glob("*.tepl"))
    files += [
        HERE / c["file"]
        for c in json.loads((HERE / "data/verification_cases.json").read_text())[
            "cases"
        ]
    ]
    files += [HERE.parent / "tepl/tepl.toml"]
    files += [
        HERE / "data/synthesis_classification.json",
        HERE / "data/verification_cases.json",
        ROOT / "research/attribute-synthesis/proofs/completeness.md",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def validate_classification(classification):
    if classification.get("protocol") != "axis_only_v1":
        raise ValueError("regenerate axis-only classification first")
    for path, value in classification["dependencies"].items():
        if sha(ROOT / path) != value:
            raise ValueError("stale classification dependency: " + path)
    for row in classification["cases"]:
        if sha(HERE / row["source_file"]) != row["source_sha256"]:
            raise ValueError("stale original source: " + row["case"])


def run(compiler, timeout, jobs):
    classification = json.loads(
        (HERE / "data/synthesis_classification.json").read_text()
    )
    validate_classification(classification)
    hashes = dependencies()
    compiler_hash = sha(compiler)
    with tempfile.TemporaryDirectory(prefix="tepl-axis-evaluation-") as directory:

        def worker(row):
            result = measure_case(row, compiler, timeout, Path(directory))
            print(row["case"] + ": " + result["outcome"], flush=True)
            return result

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            records = list(pool.map(worker, classification["cases"]))
    if sha(compiler) != compiler_hash or hashes != dependencies():
        raise ValueError("evaluation inputs changed during run")
    controls = [r for r in records if r["control"]]
    if any(r["outcome"] in {"verified", "inferred"} for r in controls):
        raise ValueError("negative control accepted")
    result = dict(
        schema_version=2,
        protocol="axis_only_v1",
        run_at_utc=datetime.now(timezone.utc).isoformat(),
        tepl_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        compiler_sha256=compiler_hash,
        environment=platform.platform(),
        timeout_seconds_per_command=timeout,
        jobs=jobs,
        dependencies=hashes,
        selection_policy="One migrated task per case; every eligible original RHS axis field is auto; no descriptor erasure, partial variants, explicit-baseline selection, or added guards.",
        results=records,
    )
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
