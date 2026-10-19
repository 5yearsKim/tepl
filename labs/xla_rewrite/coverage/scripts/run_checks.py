#!/usr/bin/env python3
"""Run the explicit TEPL cases independently and preserve unsuccessful checks."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import hlo_support
    from .paths import HERE, ROOT, TEPL
else:
    import hlo_support
    from paths import HERE, ROOT, TEPL


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(compiler, timeout):
    hlo_support.validate_support()
    operations = hlo_support.inventory()
    manifest = json.loads((HERE / "data/verification_cases.json").read_text())
    compiler_digest = digest(compiler)
    compiler_sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((ROOT / "src/core/semantics").rglob("*")) if p.is_file()
    } | {"src/core/checking/operations/semantics.cc":
         digest(ROOT / "src/core/checking/operations/semantics.cc")}
    records = []
    for case in manifest["cases"]:
        path = (HERE / case["file"]).resolve()
        if re.search(r"\bauto\b", path.read_text()):
            raise ValueError(f"explicit case contains auto: {path}")
        command = [str(compiler), "check", str(path.relative_to(ROOT)),
                   "--semantic-policy=" + case["policy"]]
        required = hlo_support.required_operations(path)
        missing = [name for name in required if not operations[name]["semantic"]]
        started = time.perf_counter()
        proof_evidence = []
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                    text=True, timeout=timeout)
            proved = (result.returncode == 0
                      and result.stdout.count("status=proved") == 1
                      and "additional-guard: true" in result.stdout)
            if proved and missing:
                raise ValueError("proof reported for operations without semantics: " + str(missing))
            status = "proved" if proved else "missing_semantics" if missing else "not_proved"
            diagnostics = result.stderr.strip()
            returncode = result.returncode
            if proved:
                proof_evidence = [line.strip() for line in result.stdout.splitlines()
                                  if "status=proved" in line or "additional-guard:" in line]
        except subprocess.TimeoutExpired:
            status, diagnostics, returncode = "timeout", "wall-clock timeout", None
        record = {"case": case["name"], "catalog_id": case["catalog_id"],
                  "status": status, "returncode": returncode,
                  "seconds": round(time.perf_counter() - started, 6),
                  "file_sha256": digest(path), "diagnostics": diagnostics,
                  "required_operations": required, "missing_semantics": missing,
                  "proof_evidence": proof_evidence}
        records.append(record)
        print(f"{case['name']}: {status}", flush=True)
    dependencies = {str(p.relative_to(ROOT)): digest(p) for p in (
        HERE / "data/verification_cases.json", HERE / "scripts/run_checks.py", HERE / "baselines/reasoning_baseline.json",
        TEPL / "tepl.toml", HERE / "scripts/hlo_support.py", HERE / "data/hlo_support.json",
        HERE.parent / "catalog/rules.json",
        TEPL / "dialects/hlo.tepl", HERE / "scripts/paths.py",
        HERE / "baselines/layout_migration.json",
        HERE / "baselines/pre_layout/verification_cases.json",
        HERE / "baselines/pre_layout/compilation_cases.json")}
    if digest(compiler) != compiler_digest or any(
            digest(ROOT / path) != expected for path, expected in compiler_sources.items()):
        raise ValueError("compiler or semantic sources changed during measurement; rerun on a stable build")
    result = {
        "schema_version": 1,
        "tepl_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_code_changes": subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only", "--", "src", "tests",
             "BUILD.bazel", "MODULE.bazel"],
            cwd=ROOT, text=True).splitlines(),
        "untracked_code_files": subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "src", "tests"],
            cwd=ROOT, text=True).splitlines(),
        "compiler_source_hashes": compiler_sources,
        "compiler_sha256": compiler_digest, "dependencies": dependencies,
        "environment": platform.platform(), "python": platform.python_version(),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds_per_case": timeout, "results": records,
    }
    (HERE / "results/verification.json").write_text(json.dumps(result, indent=2) + "\n")
    controls = {c["name"] for c in manifest["cases"] if c["control"]}
    return all(r["status"] == "not_proved" for r in records
               if r["case"] in controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not run(args.compiler.resolve(strict=True), args.timeout):
        parser.error("negative control unexpectedly passed or timed out")


if __name__ == "__main__":
    main()
