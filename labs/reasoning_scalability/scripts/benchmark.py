"""Bounded synthetic attribute-inference measurements and three result figures."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

import numpy as np

import workloads as b

CHAIN_SIZES = tuple(range(1, 11))
RANKS = (2, 4, 8, 16, 32)
SEED = 20261026
OUTPUT = b.HERE / "output"
DIRECTORY = OUTPUT / "inputs"
MANIFEST = DIRECTORY / "manifest.json"
RESULT = OUTPUT / "performance.json"
REPORT = OUTPUT / "performance.md"


def rank_source(rank, mode):
    name = f"rank_{rank}_{mode}"
    lhs, permutation = "A", "p1.attrs.permutation"
    for i in range(1, 4):
        lhs = f"(p{i}: transpose {lhs})"
        if i > 1:
            permutation = f"gather({permutation}, p{i}.attrs.permutation)"
    rhs = "(transpose A)" if mode == "automatic" else "(out: transpose A)"
    dims = ",".join(f"D{i}" for i in range(rank))
    text = (f"rule {name} {{ A:f32[{dims}]\n {lhs} => {rhs}\n")
    if mode == "explicit":
        text += f" derive {{out.attrs={{permutation:{permutation}}};}}\n"
    return name, text + "}\n"


def validate_workloads():
    """Exact integer reference checks; no TEPL normalizer or inferred fields."""
    checked = 0
    for size in CHAIN_SIZES:
        leaves, lhs, rhs = b.chain(size + 1)
        rng = np.random.default_rng(SEED + size)
        dims = {"b": 2, **{f"d{i}": 2 + i % 2 for i in range(size + 2)}}
        values = {v.name: rng.integers(-2, 3, tuple(dims[a] for a in v.axes)) for v in leaves}

        def evaluate(node):
            if not node.children:
                return values[node.name]
            a, c = map(evaluate, node.children)
            lc, rc, lb, rb = (node.attrs[f][0] for f in b.FIELDS)
            lf = next(i for i in range(3) if i not in (lc, lb))
            rf = next(i for i in range(3) if i not in (rc, rb))
            return np.matmul(a.transpose(lb, lf, lc), c.transpose(rb, rc, rf))

        oracle = values[leaves[0].name]
        for leaf in leaves[1:]:
            oracle = np.matmul(oracle, values[leaf.name])
        np.testing.assert_array_equal(evaluate(lhs), oracle)
        np.testing.assert_array_equal(evaluate(rhs), oracle)
        checked += 1
    for rank in RANKS:
        rng = np.random.default_rng(SEED + rank)
        for _ in range(10):
            # Exercise all axis positions without an exponentially large array.
            shape = [1] * rank
            for axis, extent in zip(rng.choice(rank, min(rank, 3), replace=False), (2, 3, 4)):
                shape[axis] = extent
            tensor = np.arange(math.prod(shape)).reshape(shape)
            actual, combined = tensor, list(range(rank))
            for _ in range(3):
                permutation = rng.permutation(rank).tolist()
                actual = actual.transpose(permutation)
                combined = [combined[i] for i in permutation]
            np.testing.assert_array_equal(actual, tensor.transpose(combined))
            checked += 1
    return checked


def prepare():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    previous = json.loads(MANIFEST.read_text())["cases"] if MANIFEST.exists() else []
    (DIRECTORY / "tepl.toml").write_text('tepl-version = "0.1.0"\n')
    relative = Path(os.path.relpath(b.DIALECT, DIRECTORY)).as_posix()
    header = f'from "{relative}" import TensorLang as t;\nuse t::{{dot,transpose}};\n'
    cases = []

    def add(name, source, family, size, mode, constructors):
        path = DIRECTORY / f"{name}.tepl"
        source = header + source
        path.write_text(source)
        cases.append(dict(case_id=name, family=family, size=size,
                          mode=mode, rank=3 if family == "chain" else size,
                          expected_constructors=constructors,
                          file=path.relative_to(b.ROOT).as_posix(), input_hash=b.digest(path)))

    for size in CHAIN_SIZES:
        for mode in ("explicit", "partial", "automatic"):
            if size == 1 and mode == "partial":
                continue  # One descriptor cannot be partially automatic.
            name, source = b.dot_source(size, mode)
            count = size if mode == "automatic" else (size + 1) // 2 if mode == "partial" else 0
            add(name, source, "chain", size, mode, count)
    for rank in RANKS:
        for mode in ("explicit", "automatic"):
            name, source = rank_source(rank, mode)
            add(name, source, "rank", rank, mode, int(mode == "automatic"))
    manifest = dict(schema_version=2, seed=SEED, chain_sizes=list(CHAIN_SIZES), ranks=list(RANKS), cases=cases,
                    dense_reference_checks=validate_workloads())
    current = {c["file"] for c in cases}
    for old in previous:
        path = b.ROOT / old["file"]
        if old["file"] not in current and path.parent == DIRECTORY and path.suffix == ".tepl":
            path.unlink(missing_ok=True)
    b.save_json(MANIFEST, manifest)
    return manifest


def measurement_script_hash():
    """Hash executable experiment logic independently of reporting and docs."""
    import ast
    import hashlib

    tree = ast.parse(Path(__file__).read_text())
    reporting = {"render", "provenance", "validate", "measurement_script_hash"}
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name in reporting)]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def provenance(binary):
    paths = [p for p in (b.ROOT / "src").rglob("*") if p.is_file()]
    paths += [Path(__file__), b.HERE / "scripts/workloads.py", b.HERE / "inference_benchmark.cc",
              b.HERE / "BUILD.bazel", b.HERE / "README.md", b.HERE / "requirements.txt", b.DIALECT, MANIFEST,
              DIRECTORY / "tepl.toml", b.ROOT / ".bazelrc", b.ROOT / "MODULE.bazel"]
    return dict(run_at_utc=datetime.now(timezone.utc).isoformat(),
                revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=b.ROOT, text=True).strip(),
                binary=str(binary), binary_hash=b.digest(binary),
                measurement_script_hash=measurement_script_hash(),
                build_command="bazel build -c opt //labs/reasoning_scalability:inference_benchmark",
                dependencies={p.relative_to(b.ROOT).as_posix(): b.digest(p) for p in paths},
                cpu=next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                          if s.startswith("model name")), platform.processor()),
                platform=platform.platform(), logical_cpus=os.cpu_count(),
                python=platform.python_version(), numpy=np.__version__, matplotlib=version("matplotlib"))


def validate(data):
    if data.get("schema_version") != 2:
        raise ValueError("results use an older experiment; rerun the benchmark")
    if b.digest(data["provenance"]["binary"]) != data["provenance"]["binary_hash"]:
        raise ValueError("benchmark binary changed")
    for path, expected in data["provenance"]["dependencies"].items():
        if path == (b.HERE / "README.md").relative_to(b.ROOT).as_posix():
            continue  # Documentation does not affect measured calls.
        if path == Path(__file__).relative_to(b.ROOT).as_posix() and "measurement_script_hash" in data["provenance"]:
            if measurement_script_hash() != data["provenance"]["measurement_script_hash"]:
                raise ValueError("measurement script changed; rerun the benchmark")
            continue
        if b.digest(b.ROOT / path) != expected:
            raise ValueError("stale dependency: " + path)
    ids = {c["case_id"] for c in data["manifest"]["cases"]}
    if (len(ids) != len(data["manifest"]["cases"]) or
            len(data["records"]) != len(ids) * (data["warmups"] + data["repetitions"]) or
            any(r["case_id"] not in ids for r in data["records"])):
        raise ValueError("unexpected cases or record count")
    for case in data["manifest"]["cases"]:
        if b.digest(b.ROOT / case["file"]) != case["input_hash"]:
            raise ValueError("case changed: " + case["file"])
        rows = [r for r in data["records"] if r["case_id"] == case["case_id"]]
        if Counter(r["stage"] for r in rows) != Counter(warmup=data["warmups"], measured=data["repetitions"]):
            raise ValueError("incomplete repetitions: " + case["case_id"])
        for r in rows:
            if r["status"] not in ("proved", "time_limit", "budget_exhausted", "unresolved"):
                raise ValueError("unknown proof outcome")
            if r["status"] == "proved" and r["constructors"] != case["expected_constructors"]:
                raise ValueError("missing constructors")
            if any(r[f] < 0 for f in ("total_ms", "construction_ms", "verification_ms", "other_ms")):
                raise ValueError("negative timing")


def summarize(data):
    summaries = []
    for case in data["manifest"]["cases"]:
        rows = [r for r in data["records"] if r["case_id"] == case["case_id"] and r["stage"] == "measured"]
        statuses = {r["status"] for r in rows}
        result = dict(case, status=next(iter(statuses)) if len(statuses) == 1 else "mixed",
                      reason="; ".join(sorted({r["reason"] for r in rows})),
                      proved_repetitions=sum(r["status"] == "proved" for r in rows))
        for field in ("total_ms", "construction_ms", "verification_ms", "other_ms"):
            values = [r[field] for r in rows]
            for name, function in (("median", statistics.median), ("mean", statistics.mean), ("min", min), ("max", max)):
                result[f"{name}_{field}"] = function(values)
        summaries.append(result)
    return summaries


def render(data):
    from figures import render_figures

    rows = summarize(data)
    out = OUTPUT / "figures"
    out.mkdir(exist_ok=True)
    render_figures(rows, out, CHAIN_SIZES, RANKS)
    labels = {"explicit": "Explicit", "partial": "Partial auto", "automatic": "Full auto"}
    lines = ["# Attribute inference cost and scalability", "",
             "Chain lengths: 1–10 dots per side. Tensor ranks: 2, 4, 8, 16, 32 (log₂ axis).", "",
             f"{sum(r['status']=='proved' for r in rows)}/{len(rows)} cases prove in every measured repetition with complete constructors. "
             f"{data['manifest']['dense_reference_checks']} independent exact integer reference checks pass.", "",
             "## Measurement", "",
             "Optimized C++ build; serial execution; deterministic shuffled case order each round. "
             "Parsing, imports, type checking and initial plan preparation are excluded. "
             "Each timed proveRule call uses fresh contexts and includes all attempted synthesis strategies "
             "and independent candidate verification. Diagnostic printing is outside timed calls.", "",
             f"Warmups: {data['warmups']}; measured repetitions: {data['repetitions']}; "
             f"time limit per call: {data['timeout_seconds']:g} seconds, shared by synthesis, fallback and verification. "
             "The benchmark removes the global work-count cap; local search bounds remain. "
             "The deadline is checked cooperatively, so small overruns are possible. "
             "Curves show per-case medians with min/max whiskers. "
             "Stacked bars use means so phase totals add to the mean total; dispatch/accounting is shown separately.", "",
             "Cross markers identify unsuccessful attempts at their observed latency; they are not connected "
             "to successful-case curves. A mixed case has different outcomes across measured repetitions. "
             "Hatched phase bars identify cases that did not prove in every measured repetition. "
             "Their timings measure attempts rather than successful inference.", "",
             "Chain size means dot operations on each side (size + 1 distinct input tensors), at fixed rank 3. "
             "Partial auto infers odd-numbered RHS nodes and retains the other oracle-generated descriptors. "
             "Size 1 has no proper partial mode and tests identity copying. Rank scaling holds the LHS at "
             "three transposes and the RHS at one transpose, with independent symbolic permutations. "
             "These are metadata/proof timings, not tensor execution times.", "",
             f"CPU: {data['provenance']['cpu']}; platform: {data['provenance']['platform']}. "
             "The only varied quantities are the declared workload factors; arithmetic policy is algebraic.", "",
             "![Inference scalability](figures/scalability.png)", "",
             "[Download the publication figure (PDF)](figures/scalability.pdf).", "",
             "**Figure.** (a) Chain scaling at fixed tensor rank 3. "
             "(b) Rank scaling for three symbolic source transposes composed into one replacement transpose; "
             "the rank axis is logarithmic (base 2). "
             "Panels (a) and (b) show median pipeline time with min-max whiskers. "
             "(c) Mean time by phase for full-auto chains, including all strategy attempts. "
             "Partial auto starts at chain length 2.", "",
             "## Proof outcomes by workload", "",
             "These counts describe controlled synthetic variants. Limit points measure failed "
             "attempt cost; they do not establish a successful inference-time trend beyond that boundary.", "",
             "| Family | Mode | Proved sizes | Limited / unresolved / mixed sizes |",
             "|---|---|---|---|"]
    for family, modes in (("chain", labels), ("rank", ("explicit", "automatic"))):
        for mode in modes:
            group = [r for r in rows if r["family"] == family and r["mode"] == mode]
            good = ", ".join(str(r["size"]) for r in group if r["status"] == "proved") or "—"
            bad = ", ".join(str(r["size"]) for r in group if r["status"] != "proved") or "—"
            lines.append(f"| {family} | {mode} | {good} | {bad} |")
    lines += ["",
             "## Raw case summaries", "",
             "| Family | Size | Mode | Status | Proved repetitions | Median total ms | Construction ms | Verification ms |",
             "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['size']} | {r['mode']} | {r['status']} | {r['proved_repetitions']}/{data['repetitions']} | "
                     f"{r['median_total_ms']:.4f} | {r['median_construction_ms']:.4f} | {r['median_verification_ms']:.4f} |")
    lines += ["", "Raw repetitions and provenance: [performance.json](performance.json). "
              "Figures are also available as SVG and PDF in `figures/`.", ""]
    REPORT.write_text("\n".join(lines))
    b.save_json(OUTPUT / "performance_summary.json", dict(cases=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--binary", type=Path, default=b.ROOT / "bazel-bin/labs/reasoning_scalability/inference_benchmark")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=10,
                        help="time limit per inference call, including synthesis and verification (default: 10)")
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0.001:
        parser.error("invalid measurement settings")
    args.timeout_seconds = round(args.timeout_seconds * 1000) / 1000
    if args.check or args.report:
        data = json.loads(RESULT.read_text())
        validate(data)
        if args.report:
            render(data)
        print(f"Validated {len(data['records'])} API measurements and all input/source hashes.")
        return
    manifest = prepare()
    if args.prepare:
        print(f"Prepared {len(manifest['cases'])} cases; {manifest['dense_reference_checks']} dense checks.")
        return
    binary = args.binary.resolve(strict=True)
    metadata = provenance(binary)
    raw = OUTPUT / "performance_raw.json"
    raw.parent.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop("TEPL_INDEX_PROFILE", None)
    # Allow every call its own deadline, plus preparation and cleanup overhead.
    run_timeout = 300 + len(manifest["cases"]) * (args.warmups + args.repetitions) * (args.timeout_seconds + 1)
    subprocess.run([str(binary), str(MANIFEST), str(raw), str(args.warmups), str(args.repetitions),
                    str(round(args.timeout_seconds * 1000))],
                   cwd=b.ROOT, env=env, check=True, timeout=run_timeout)
    data = dict(schema_version=2, manifest=manifest, provenance=metadata,
                warmups=args.warmups, repetitions=args.repetitions, timeout_seconds=args.timeout_seconds,
                records=json.loads(raw.read_text())["records"])
    validate(data)
    b.save_json(RESULT, data)
    raw.unlink()
    render(data)
    print(f"Measured {len(manifest['cases'])} cases. Report: {REPORT.relative_to(b.ROOT)}")


if __name__ == "__main__":
    main()
