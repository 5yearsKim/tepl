#!/usr/bin/env python3
"""Paired, strict TEPL/native full CPU compilation and output comparisons."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def host_cpu():
    cpu = dict(logical_cpus=os.cpu_count(), processor=platform.processor())
    path = Path("/proc/cpuinfo")
    if path.exists():
        first = dict(line.split(":", 1) for line in path.read_text().split("\n\n", 1)[0].splitlines() if ":" in line)
        first = {key.strip(): value.strip() for key, value in first.items()}
        cpu.update(model=first.get("model name"), flags=first.get("flags", "").split())
    return cpu


def archive_graph(path):
    """Keep the complete graph losslessly, without accumulating huge text files."""
    digest = sha(path)
    target = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, target.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    with gzip.open(target, "rb") as restored:
        actual = hashlib.file_digest(restored, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("lossless graph archive check failed")
    path.unlink()
    return digest


def check_result(result, strict):
    if not result["final_graphs_equal"] or len(result["outputs"]) != 3 or not all(r["equal"] for r in result["outputs"]):
        raise ValueError("missing graph/output agreement")
    if not result["rewrite_counts_equal"] or result["baseline_pass_invocations"] != result["integrated_pass_invocations"]:
        raise ValueError("native/integrated rewrite traversal differs")
    if result["total_applications"] <= 0 or result["tepl_applications"] <= 0:
        raise ValueError("missing actual rewrite/delegation evidence")
    if len({r["original_line"] for r in result["rules"]}) != len(result["rules"]):
        raise ValueError("duplicate rule identities")
    if len(result["rules"]) != result["distinct_variants"]:
        raise ValueError("variant count mismatch")
    if sum(r["applications"] for r in result["rules"]) != result["total_applications"]:
        raise ValueError("application count mismatch")
    if sum(r["tepl_applications"] for r in result["rules"]) != result["tepl_applications"]:
        raise ValueError("TEPL application count mismatch")
    missing = sorted(r["original_line"] for r in result["rules"] if r["applications"] != r["tepl_applications"])
    if any(not 0 <= r["tepl_applications"] <= r["applications"] for r in result["rules"]):
        raise ValueError("invalid delegation counts")
    if missing != sorted(result["unported_sites"]) or result["all_observed_variants_delegated"] != (not missing):
        raise ValueError("strict closure evidence mismatch")
    if strict and missing:
        raise ValueError("strict integration incomplete: " + str(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xla-root", type=Path, required=True)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--library", type=Path, default=ROOT / "bazel-bin/labs/attribute_inferences/xla_algebraic_simplifier/tepl_algebraic.so")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", choices=["dlrm", "gpt2", "vit", "resnet"], default=["dlrm", "gpt2", "vit", "resnet"])
    parser.add_argument("--allow-native", action="store_true", help="Explicit hybrid diagnostic run; never certifies full delegation")
    parser.add_argument("--shortform-constants", choices=["finite", "original"], default="finite",
                        help="Initialize elided floats to bounded values and integer literals to index sequences, or retain XLA's synthetic bit patterns")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("choose a fresh output directory")
    manifest = json.loads((HERE / "workloads/manifest.json").read_text())
    xla = args.xla_root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=xla, text=True).strip()
    if revision != manifest["xla_revision"]:
        raise ValueError("XLA revision differs from the pinned workload revision")
    prepared = json.loads((xla / "research/tepl_algebraic/prepared.json").read_text())
    source = xla / "xla/hlo/transforms/simplifiers/algebraic_simplifier.cc"
    if sha(source) != prepared["source_sha256"]:
        raise ValueError("prepared AlgebraicSimplifier source changed")
    if prepared["xla_revision"] != revision:
        raise ValueError("prepared XLA revision changed")
    for name, digest in prepared["files"].items():
        if sha(xla / "research/tepl_algebraic" / name) != digest:
            raise ValueError("prepared adapter source changed: " + name)
    for row in manifest["workloads"]:
        if sha(HERE / "workloads" / row["file"]) != row["sha256"]:
            raise ValueError("workload input changed: " + row["name"])
    originals = {"driver": args.driver or xla / "bazel-bin/research/tepl_algebraic/pipeline",
                 "library": args.library, "compiler": args.compiler}
    for path in originals.values():
        if not path.is_file():
            raise ValueError("build required binary explicitly: " + str(path))
    (out / "bin").mkdir(parents=True)
    for name, path in originals.items():
        shutil.copy2(path.resolve(), out / "bin" / name)
    binary_hashes = {name: sha(out / "bin" / name) for name in originals}
    files = [p for p in HERE.rglob("*") if p.is_file() and not p.is_relative_to(out)
             and p.suffix in {".cc", ".h", ".tepl", ".py", ".bazel", ".json", ".hlo"}
             and "__pycache__" not in p.parts and "output" not in p.parts]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    def ignore(directory, names):
        return [name for name in names if name in {"__pycache__", "output"} or (Path(directory) / name).resolve().is_relative_to(out)]
    shutil.copytree(HERE, out / "source/xla_algebraic_simplifier", ignore=ignore)
    shutil.copytree(HERE.parent / "xla_rewrite/tepl/dialects", out / "source/xla_rewrite/tepl/dialects")
    shutil.copytree(HERE.parent / "xla_workload_count/counter", out / "source/xla_workload_count/counter", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(HERE.parent / "xla_workload_count/support-registry.json", out / "source/xla_workload_count/support-registry.json")
    shutil.copyfile(ROOT / "LICENSE", out / "source/LICENSE")
    shutil.copyfile(source, out / "source/algebraic_simplifier.patched.cc")
    (out / "source/tepl.toml").write_text('tepl-version = "0.2.0"\n')
    (out / "checkout.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT))
    xla_patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=xla)
    (out / "xla-checkout.patch").write_bytes(xla_patch)
    commands = []
    compiler = out / "bin/compiler"
    for name, flags in (("rules", ["--core", "--semantic-policy", "algebraic"]), ("branch_rules", [])):
        argv = [str(compiler), "compile", str(out / "source/xla_algebraic_simplifier" / (name + ".tepl")), *flags, "-o", str(out / (name + ".json"))]
        process = subprocess.run(argv, capture_output=True, text=True)
        save(out / (name + ".compile.json"), dict(argv=argv, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr))
        process.check_returncode()
        subprocess.run([str(compiler), "validate", str(out / (name + ".json"))], check=True)
        commands.append(argv)
    provenance = dict(schema_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                      xla_revision=revision, tepl_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      xla_preparation=prepared, binaries=binary_hashes, sources=hashes,
                      protocol="paired_normal_cpu_pipeline_v1", mode="hybrid" if args.allow_native else "strict",
                      platform=platform.platform(), input_manifest_sha256=sha(HERE / "workloads/manifest.json"),
                      cpu=host_cpu(), python=sys.version,
                      checkout_patch_sha256=sha(out / "checkout.patch"),
                      xla_checkout_patch_sha256=sha(out / "xla-checkout.patch"),
                      program_sha256={name: sha(out / (name + ".json")) for name in ("rules", "branch_rules")},
                      input_sets=3, atol=1.e-5, rtol=1.e-5, timeout_seconds=args.timeout,
                      shortform_constants=args.shortform_constants,
                      timing_claim=False, compile_commands=commands)
    save(out / "provenance.json", provenance)
    results = []
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu" + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    for name in dict.fromkeys(args.workloads):
        row = next(r for r in manifest["workloads"] if r["name"] == name)
        directory = out / name
        directory.mkdir()
        shutil.copyfile(HERE / "workloads" / row["file"], directory / "input.hlo")
        argv = [str(out / "bin/driver"), str(out / "bin/library"), str(out / "rules.json"),
                str(out / "branch_rules.json"), str(directory / "input.hlo"), str(directory),
                "hybrid" if args.allow_native else "strict", args.shortform_constants]
        print(name + ": compiling baseline and integrated CPU pipelines", flush=True)
        with (directory / "driver.stdout").open("w") as stdout, (directory / "driver.stderr").open("w") as stderr:
            try:
                process = subprocess.run(argv, env=environment, stdout=stdout, stderr=stderr, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                save(directory / "failure.json", dict(reason="timeout", argv=argv, seconds=args.timeout,
                     retained_graph_sha256={p.name: archive_graph(p) for p in directory.glob("*.final.hlo")}))
                raise
        if process.returncode:
            save(directory / "failure.json", dict(reason="driver_failure", argv=argv, returncode=process.returncode,
                 retained_graph_sha256={p.name: archive_graph(p) for p in directory.glob("*.final.hlo")}))
            raise RuntimeError(f"{name} failed ({process.returncode}); raw logs preserved in {directory}")
        result = json.loads((directory / "results.json").read_text())
        check_result(result, strict=not args.allow_native)
        expected_policy = "finite_inputs_v2" if args.shortform_constants == "finite" else "xla_parser_default"
        expected_filled = row["elided_floating_literals"] if args.shortform_constants == "finite" else 0
        expected_indices = row["elided_integer_literals"] if args.shortform_constants == "finite" else 0
        if result["shortform_constants"] != dict(policy=expected_policy, placeholders=row["elided_literals"],
                                                 floating_literals_initialized=expected_filled,
                                                 integer_literals_initialized=expected_indices):
            raise ValueError("shortform initialization evidence differs from the recorded input policy")
        result.update(workload=name, input_sha256=row["sha256"],
                      final_hlo_sha256={variant: archive_graph(directory / (variant + ".final.hlo")) for variant in ("baseline", "tepl")},
                      paper_reference=row["reference_counts"])
        result["paper_reference_counts_match"] = (result["distinct_variants"] == row["reference_counts"]["rules"]
                                                   and result["total_applications"] == row["reference_counts"]["firings"])
        results.append(result)
        save(out / "results.json", dict(provenance=provenance, workloads=results))
        print(f"{name}: {result['tepl_applications']}/{result['total_applications']} applications delegated; graph/output checks passed", flush=True)
    if binary_hashes != {name: sha(out / "bin" / name) for name in originals} or hashes != {str(p.relative_to(ROOT)): sha(p) for p in files}:
        raise ValueError("experiment binaries or source changed during run")
    if xla_patch != subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=xla):
        raise ValueError("XLA source changed during run")
    text = ["# XLA branch integration", "", "| Workload | Variants | Applications | TEPL applications | Final graph | Output sets |", "|---|---:|---:|---:|---|---:|"]
    for result in results:
        text.append(f"| {result['workload']} | {result['distinct_variants']} | {result['total_applications']} | {result['tepl_applications']} | Equal | 3 |")
    text += ["", "Mode: " + provenance["mode"] + ". Rules outside the observed port cohort remain upstream XLA implementations. Strict mode requires every observed non-bookkeeping application to be delegated.", ""]
    (out / "report.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
