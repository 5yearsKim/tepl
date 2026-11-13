#!/usr/bin/env python3
"""Reproduce paper coverage from bundled inputs, preserving every outcome."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = HERE / "fixtures/paper"
DIALECTS = ROOT / "labs/attribute_inferences/xla_rewrite/tepl/dialects"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot_checkout(out):
    """Preserve tracked source plus new implementation files, including dirty edits."""
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names.extend(str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file())
    for directory in (ROOT / "src", HERE / "fixtures", ROOT / "labs/attribute_inferences/xla_algebraic_simplifier"):
        names.extend(str(p.relative_to(ROOT)) for p in directory.rglob("*")
                     if p.is_file() and not p.is_relative_to(out) and "output" not in p.relative_to(directory).parts
                     and "__pycache__" not in p.parts)
    paths = sorted({n for n in names if n and (ROOT / n).is_file()})
    hashes = {n: sha(ROOT / n) for n in paths}
    with tarfile.open(out / "checkout.tar.gz", "w:gz") as archive:
        for name in paths:
            archive.add(ROOT / name, arcname=name, recursive=False)
    save(out / "checkout-files.json", hashes)
    if any(sha(ROOT / n) != h for n, h in hashes.items()):
        raise ValueError("checkout changed while snapshotting")
    return sha(out / "checkout.tar.gz")


def verify_inputs():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (FIXTURES / name).resolve()
        if not path.is_relative_to(FIXTURES.resolve()) or sha(path) != digest:
            raise ValueError("paper input hash mismatch: " + name)
    cases = json.loads((FIXTURES / "cases.json").read_text())["cases"]
    if len(cases) != 123 or len({r["case"] for r in cases}) != 123:
        raise ValueError("expected 123 unique modeled cases")
    if sum(bool(r["axis_fields"]) for r in cases) != 38:
        raise ValueError("expected 38 axis-inference cases")
    for row in cases:
        if sha(FIXTURES / row["source"]) != row["source_sha256"]:
            raise ValueError("case source mismatch: " + row["case"])
    return manifest, cases


def migrate(source):
    """Adapt retired separators and where-block syntax, retaining expressions."""
    # Reuse the compiler corpus lexer so strings and comments stay intact.
    sys.path.insert(0, str(ROOT / "labs/attribute_inferences/xla_rewrite/coverage"))
    from scripts.synthesis_cases import TOKEN

    tokens = [m for m in TOKEN.finditer(source)
              if not m[0].startswith(("//", "/*"))]
    edits = [(m.start(), m.end(), "") for m in tokens if m[0] == ";"]
    for i, token in enumerate(tokens):
        if token[0] != "where" or tokens[i + 1][0] != "{":
            continue
        depth = 0
        for closing in tokens[i + 1:]:
            depth += closing[0] == "{"
            depth -= closing[0] == "}"
            if depth == 0:
                edits.extend([(tokens[i + 1].start(), tokens[i + 1].end(), ""),
                              (closing.start(), closing.end(), "")])
                break
        else:
            raise ValueError("unclosed where block")
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def command(argv, directory, timeout):
    started = time.monotonic()
    try:
        process = subprocess.run(list(map(str, argv)), cwd=ROOT, capture_output=True,
                                 text=True, timeout=timeout)
        result = dict(returncode=process.returncode, stdout=process.stdout,
                      stderr=process.stderr)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result = dict(returncode="timeout", stdout=decode(error.stdout), stderr=decode(error.stderr))
    result.update(argv=list(map(str, argv)), seconds=time.monotonic() - started)
    save(directory, result)
    return result


def has_hole(value):
    if isinstance(value, dict):
        return value.get("kind") in {"auto", "auto_field", "hole"} or any(map(has_hole, value.values()))
    return isinstance(value, list) and any(map(has_hole, value))


def attempt(row, binary, out, backend, timeout):
    name = row["case"]
    source = out / "inputs" / (name + ".tepl")
    artifact = out / "programs" / (name + ".json")
    flags = ["--core"] if backend == "core" else []
    record = dict(row, accepted=False, commands={}, source_sha256=sha(source))
    phases = [
        ("check", [binary, "check", source, "--semantic-policy", "algebraic", "--rule", row["rule_name"], *flags]),
        ("compile", [binary, "compile", source, "--semantic-policy", "algebraic", *flags, "-o", artifact]),
        ("validate", [binary, "validate", artifact]),
    ]
    for phase, argv in phases:
        log = out / "logs" / f"{name}.{phase}.json"
        result = command(argv, log, timeout)
        record["commands"][phase] = str(log.relative_to(out))
        proved = phase != "check" or (result["stdout"].count("status=proved") == 1
                                             and "additional-guard: true" in result["stdout"])
        if result["returncode"] != 0 or not proved:
            record.update(outcome="timeout" if result["returncode"] == "timeout" else phase + "_unresolved")
            return record
    if has_hole(json.loads(artifact.read_text())):
        raise ValueError("unresolved executable field: " + name)
    record.update(accepted=True, outcome="accepted", artifact=str(artifact.relative_to(out)),
                  artifact_sha256=sha(artifact))
    return record


def coverage(args):
    manifest, cases = verify_inputs()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("choose a new or empty output directory")
    for name in ("inputs", "logs", "programs", "bin"):
        (out / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(DIALECTS, out / "inputs/dialects")
    (out / "inputs/tepl.toml").write_text('tepl-version = "0.2.0"\n')
    compiler = args.compiler.resolve(strict=True)
    binary = out / "bin/tepl"
    shutil.copy2(compiler, binary)
    binary_hash = sha(binary)
    for row in cases:
        (out / "inputs" / (row["case"] + ".tepl")).write_text(migrate((FIXTURES / row["source"]).read_text()))
    controls = [dict(case="negative_subtract_to_add", rule_name="negative_subtract_to_add",
                     axis_fields=[], control=True)]
    control = ROOT / "labs/attribute_inferences/xla_rewrite/tepl/rules/semantic/negative_subtract_to_add.tepl"
    (out / "inputs/negative_subtract_to_add.tepl").write_bytes(control.read_bytes())
    input_hashes = {str(p.relative_to(out)): sha(p) for p in sorted((out / "inputs").rglob("*")) if p.is_file()}
    provenance = dict(protocol="paper_123_current_syntax_v1", backend=args.backend,
                      run_at_utc=datetime.now(timezone.utc).isoformat(),
                      revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      platform=platform.platform(), compiler_sha256=binary_hash,
                      input_hashes=input_hashes, fixture_manifest_sha256=sha(FIXTURES / "manifest.json"),
                      runner_sha256=sha(__file__), timeout=args.timeout, jobs=args.jobs,
                      syntax_migration="remove separator tokens and where-block braces; preserve expressions",
                      numerical_execution=False)
    provenance["checkout_archive_sha256"] = snapshot_checkout(out)
    provenance["dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    (out / "implementation.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    save(out / "provenance.json", provenance)
    def work(row):
        result = attempt(row, binary, out, args.backend, args.timeout)
        print(row["case"] + ": " + result["outcome"], flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(work, cases + controls))
    if sha(binary) != binary_hash or any(sha(out / n) != h for n, h in input_hashes.items()):
        raise ValueError("inputs or compiler changed during measurement")
    save(out / "results.json", dict(provenance=provenance, records=rows))
    validate_run(out)
    selected = [r for r in rows if not r.get("control")]
    axis = [r for r in selected if r["axis_fields"]]
    summary = dict(cases=len(selected), axis_tasks=len(axis),
                   axis_accepted=sum(r["accepted"] for r in axis),
                   accepted=sum(r["accepted"] for r in selected),
                   outcomes=dict(Counter(r["outcome"] for r in selected)),
                   changed=[r["case"] for r in selected if r["accepted"] != r["paper_accepted"]],
                   paper_reference=manifest["targets"]["coverage"])
    save(out / "summary.json", summary)
    (out / "report.md").write_text(
        "# XLA rewrite coverage\n\n"
        f"Backend: `{args.backend}`; arithmetic: algebraic.\n\n"
        f"Accepted: **{summary['accepted']}/123** modeled tasks. "
        f"Axis inference: **{summary['axis_accepted']}/38**.\n\n"
        "Acceptance requires proof under the written conditions, compilation, "
        "and portable validation. Unresolved cases and all command logs are retained.\n\n"
        "See `summary.json`, `results.json`, and `provenance.json`.\n")
    print(json.dumps(summary, indent=2))


def validate_run(out):
    data = json.loads((out / "results.json").read_text())
    provenance = data["provenance"]
    for name, digest in provenance["input_hashes"].items():
        if sha(out / name) != digest:
            raise ValueError("changed run input: " + name)
    if sha(out / "bin/tepl") != provenance["compiler_sha256"]:
        raise ValueError("changed run compiler")
    if "checkout_archive_sha256" in provenance and sha(out / "checkout.tar.gz") != provenance["checkout_archive_sha256"]:
        raise ValueError("changed checkout snapshot")
    _, expected = verify_inputs()
    expected = {r["case"]: r for r in expected}
    rows = data["records"]
    if len(rows) != 124 or {r["case"] for r in rows} != set(expected) | {"negative_subtract_to_add"}:
        raise ValueError("missing or duplicate coverage cases")
    for row in rows:
        if row["case"] in expected:
            for key, value in expected[row["case"]].items():
                if key != "source_sha256" and row.get(key) != value:
                    raise ValueError("case metadata changed: " + row["case"])
            source = out / "inputs" / (row["case"] + ".tepl")
            frozen = FIXTURES / expected[row["case"]]["source"]
            if source.read_text() != migrate(frozen.read_text()) or sha(source) != row["source_sha256"]:
                raise ValueError("case expressions changed: " + row["case"])
        elif row["accepted"] or not row.get("control"):
            raise ValueError("negative control accepted or mislabeled")
        if not row["accepted"]:
            continue
        if row["outcome"] != "accepted" or set(row["commands"]) != {"check", "compile", "validate"}:
            raise ValueError("incomplete success evidence")
        for phase, name in row["commands"].items():
            command_result = json.loads((out / name).read_text())
            if command_result["returncode"] != 0:
                raise ValueError("successful case has failed command")
            if phase == "check" and (command_result["stdout"].count("status=proved") != 1
                                       or "additional-guard: true" not in command_result["stdout"]):
                raise ValueError("missing unconditional proof")
        artifact = out / row["artifact"]
        if sha(artifact) != row["artifact_sha256"] or has_hole(json.loads(artifact.read_text())):
            raise ValueError("invalid completed artifact")
    return data


def reproduce(args):
    """Run compiler measurements serially; isolate scalability from other work."""
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("choose a fresh suite output directory")
    out.mkdir(parents=True, exist_ok=True)
    compiler = args.compiler.resolve(strict=True)
    def run(argv):
        subprocess.run(list(map(str, argv)), cwd=ROOT, check=True)
    run([sys.executable, HERE / "run.py", "coverage", "--compiler", compiler,
         "--out", out / "coverage"])
    run([sys.executable, HERE / "source_size.py", "--compiler", compiler,
         "--out", out / "source-size", "--verify"])
    einsum = ROOT / "labs/einsum_benchmark"
    run([sys.executable, einsum / "benchmark.py", "run", "--binary", compiler,
         "--out", out / "einsum", "--compress-artifacts"])
    run([sys.executable, einsum / "axis_report.py", "--run", out / "einsum",
         "--out", out / "einsum-report"])
    if args.scaling_binary:
        scaling = ROOT / "labs/attribute_inferences/reasoning_scalability/scripts/benchmark.py"
        run([sys.executable, scaling, "--binary", args.scaling_binary.resolve(strict=True),
             "--out", out / "scaling"])
        run([sys.executable, scaling, "--binary", args.scaling_binary.resolve(strict=True),
             "--out", out / "scaling", "--check"])
    save(out / "suite.json", dict(completed=["coverage", "source-size", "einsum"] + (["scaling"] if args.scaling_binary else []),
                                  compiler_sha256=sha(compiler), xla_integration_measured=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check-inputs")
    sub.add_parser("quick")
    run = sub.add_parser("coverage")
    run.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--backend", choices=["default", "core"], default="default")
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--timeout", type=float, default=30)
    check = sub.add_parser("check-run")
    check.add_argument("out", type=Path)
    suite = sub.add_parser("reproduce")
    suite.add_argument("--out", type=Path, required=True)
    suite.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    suite.add_argument("--scaling-binary", type=Path)
    args = parser.parse_args()
    if args.action == "check-inputs":
        verify_inputs()
        print("Validated 123 bundled paper tasks (38 axis, 85 verification).")
    elif args.action == "quick":
        verify_inputs()
        suites = [(HERE, "test_*.py"),
                  (ROOT / "labs/einsum_benchmark/tests", "test_*.py"),
                  (ROOT / "labs/attribute_inferences/xla_algebraic_simplifier", "test_prepare.py")]
        for directory, pattern in suites:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", pattern], cwd=ROOT, check=True)
    elif args.action == "check-run":
        validate_run(args.out.resolve())
        print("Validated saved coverage run.")
    elif args.action == "reproduce":
        reproduce(args)
    else:
        if args.jobs < 1 or args.timeout <= 0:
            parser.error("jobs and timeout must be positive")
        coverage(args)


if __name__ == "__main__":
    main()
