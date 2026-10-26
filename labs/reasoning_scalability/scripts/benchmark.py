"""Bounded synthetic attribute-inference measurements and three result figures."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

import numpy as np

import workloads as b

SIZES = tuple(range(1, 7))
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
    for size in SIZES:
        for layout in ("canonical", "permuted"):
            leaves, lhs, rhs = b.chain(size + 1, layout)
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

            oracle = values[leaves[0].name].transpose(tuple(leaves[0].axes.index(a) for a in ("b", "d0", "d1")))
            for i, leaf in enumerate(leaves[1:], 1):
                oracle = np.matmul(oracle, values[leaf.name].transpose(tuple(leaf.axes.index(a) for a in ("b", f"d{i}", f"d{i+1}"))))
            np.testing.assert_array_equal(evaluate(lhs), oracle)
            np.testing.assert_array_equal(evaluate(rhs), oracle)
            checked += 1
    for rank in SIZES:
        shape = tuple(2 + i % 2 for i in range(rank))
        tensor = np.arange(np.prod(shape)).reshape(shape)
        rng = np.random.default_rng(SEED + rank)
        for _ in range(10):
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

    def add(name, source, family, size, layout, mode, constructors):
        path = DIRECTORY / f"{name}.tepl"
        source = header + source
        path.write_text(source)
        cases.append(dict(case_id=name, family=family, size=size, layout=layout,
                          mode=mode, rank=3 if family == "chain" else size,
                          expected_constructors=constructors,
                          file=path.relative_to(b.ROOT).as_posix(), input_hash=b.digest(path)))

    for size in SIZES:
        for layout in ("canonical", "permuted"):
            for mode in ("explicit", "partial", "automatic"):
                if size == 1 and mode == "partial":
                    continue  # One descriptor cannot be partially automatic.
                name, source = b.dot_source(size, layout, mode)
                count = size if mode == "automatic" else (size + 1) // 2 if mode == "partial" else 0
                add(name, source, "chain", size, layout, mode, count)
    for rank in SIZES:
        for mode in ("explicit", "automatic"):
            name, source = rank_source(rank, mode)
            add(name, source, "rank", rank, "symbolic", mode, int(mode == "automatic"))
    manifest = dict(schema_version=1, seed=SEED, sizes=list(SIZES), cases=cases,
                    dense_reference_checks=validate_workloads())
    current = {c["file"] for c in cases}
    for old in previous:
        path = b.ROOT / old["file"]
        if old["file"] not in current and path.parent == DIRECTORY and path.suffix == ".tepl":
            path.unlink(missing_ok=True)
    b.save_json(MANIFEST, manifest)
    return manifest


def provenance(binary):
    paths = [p for p in (b.ROOT / "src").rglob("*") if p.is_file()]
    paths += [Path(__file__), b.HERE / "scripts/workloads.py", b.HERE / "inference_benchmark.cc",
              b.HERE / "BUILD.bazel", b.HERE / "README.md", b.HERE / "requirements.txt", b.DIALECT, MANIFEST,
              DIRECTORY / "tepl.toml", b.ROOT / ".bazelrc", b.ROOT / "MODULE.bazel"]
    return dict(run_at_utc=datetime.now(timezone.utc).isoformat(),
                revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=b.ROOT, text=True).strip(),
                binary=str(binary), binary_hash=b.digest(binary),
                build_command="bazel build -c opt //labs/reasoning_scalability:inference_benchmark",
                dependencies={p.relative_to(b.ROOT).as_posix(): b.digest(p) for p in paths},
                cpu=next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                          if s.startswith("model name")), platform.processor()),
                platform=platform.platform(), logical_cpus=os.cpu_count(),
                python=platform.python_version(), numpy=np.__version__, matplotlib=version("matplotlib"))


def validate(data):
    if b.digest(data["provenance"]["binary"]) != data["provenance"]["binary_hash"]:
        raise ValueError("benchmark binary changed")
    for path, expected in data["provenance"]["dependencies"].items():
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
        if len({r["status"] for r in rows}) != 1:
            raise ValueError("unstable proof outcome: " + case["case_id"])
        for r in rows:
            if r["status"] == "proved" and r["constructors"] != case["expected_constructors"]:
                raise ValueError("missing constructors")
            if any(r[f] < 0 for f in ("total_ms", "construction_ms", "verification_ms", "other_ms")):
                raise ValueError("negative timing")


def summarize(data):
    summaries = []
    for case in data["manifest"]["cases"]:
        rows = [r for r in data["records"] if r["case_id"] == case["case_id"] and r["stage"] == "measured"]
        result = dict(case, status=rows[0]["status"], reason=rows[0]["reason"])
        for field in ("total_ms", "construction_ms", "verification_ms", "other_ms"):
            values = [r[field] for r in rows]
            for name, function in (("median", statistics.median), ("mean", statistics.mean), ("min", min), ("max", max)):
                result[f"{name}_{field}"] = function(values)
        result["construction_attempts"] = rows[0]["construction_attempts"]
        summaries.append(result)
    return summaries


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summarize(data)
    out = OUTPUT / "figures"
    out.mkdir(exist_ok=True)
    colors = {"explicit": "#3274a1", "partial": "#329b72", "automatic": "#df8434"}
    labels = {"explicit": "Explicit", "partial": "Partial auto", "automatic": "Full auto"}

    def curves(ax, group):
        for mode, label in labels.items():
            series = sorted((r for r in group if r["mode"] == mode), key=lambda r: r["size"])
            successful = [r for r in series if r["status"] == "proved"]
            if not series:
                continue
            ax.errorbar([r["size"] for r in successful], [r["median_total_ms"] for r in successful],
                        yerr=[[r["median_total_ms"] - r["min_total_ms"] for r in successful],
                              [r["max_total_ms"] - r["median_total_ms"] for r in successful]],
                        color=colors[mode], marker="o", label=label, capsize=3)
            for r in series:
                if r["status"] != "proved":
                    ax.scatter(r["size"], r["median_total_ms"], marker="x", color=colors[mode])
        if any(r["status"] != "proved" for r in group):
            ax.plot([], [], linestyle="", marker="x", color="black", label="Unproved / work limit")
        ax.set_xticks(SIZES)
        ax.set_ylabel("Inference pipeline time (ms)")
        ax.grid(alpha=.2)
        ax.legend()

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, layout in zip(axes, ("canonical", "permuted")):
        curves(ax, [r for r in rows if r["family"] == "chain" and r["layout"] == layout])
        ax.set_title(layout.capitalize() + " axes")
        ax.set_xlabel("Dot operations per side")
    save(fig, "chain_scaling")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    curves(ax, [r for r in rows if r["family"] == "rank"])
    ax.set_xlabel("Tensor rank")
    ax.set_title("Three symbolic transposes; one RHS descriptor")
    save(fig, "rank_scaling")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    full = sorted((r for r in rows if r["family"] == "chain" and r["layout"] == "permuted" and r["mode"] == "automatic"), key=lambda r: r["size"])
    bottom = np.zeros(len(full))
    for field, label, color in (("construction_ms", "Construction", "#3274a1"), ("verification_ms", "Verification", "#df8434"), ("other_ms", "Dispatch / accounting", "#a5abb3")):
        values = [r["mean_" + field] for r in full]
        bars = ax.bar([r["size"] for r in full], values, bottom=bottom, label=label, color=color)
        for bar, row in zip(bars, full):
            if row["status"] != "proved":
                bar.set_hatch("//")
        bottom += values
    ax.set(xlabel="Dot operations per side", ylabel="Mean time (ms)", xticks=SIZES,
           title="Full auto, permuted axes: all strategy attempts")
    if any(r["status"] != "proved" for r in full):
        from matplotlib.patches import Patch
        handles, names = ax.get_legend_handles_labels()
        ax.legend(handles + [Patch(facecolor="white", edgecolor="gray", hatch="//")], names + ["Work limit"])
    else:
        ax.legend()
    save(fig, "phase_breakdown")
    lines = ["# Attribute inference cost and scalability", "",
             "Synthetic workloads only; no XLA rules. Sizes and ranks are 1–6.", "",
             f"{sum(r['status']=='proved' for r in rows)}/{len(rows)} cases prove with complete constructors. "
             f"{data['manifest']['dense_reference_checks']} independent exact integer reference checks pass.", "",
             "## Measurement", "",
             "Optimized C++ build; serial execution; deterministic shuffled case order each round. "
             "Parsing, imports, type checking and initial plan preparation are excluded. "
             "Each timed proveRule call uses fresh contexts and includes all attempted synthesis strategies "
             "and independent candidate verification. Diagnostic printing is outside timed calls.", "",
             f"Warmups: {data['warmups']}; measured repetitions: {data['repetitions']}; "
             f"work budget per attempt: {data['budget']}. Curves show per-case medians with min/max whiskers. "
             "Stacked bars use means so phase totals add to the mean total; dispatch/accounting is shown separately.", "",
             "Cross markers identify unsuccessful attempts at their observed latency; they are not connected "
             "to successful-case curves. Hatched phase bars show attempt cost up to a work limit. "
             "Timing ratios are meaningful only for pairs that both prove the same equation.", "",
             "Chain size means dot operations on each side (size + 1 distinct input tensors), at fixed rank 3. "
             "Partial auto infers odd-numbered RHS nodes and retains the other oracle-generated descriptors. "
             "Size 1 has no proper partial mode and tests identity copying. Rank scaling holds the LHS at "
             "three transposes and the RHS at one transpose, with independent symbolic permutations. "
             "Rank 1 is a degenerate permutation domain. These are metadata/proof timings, not tensor execution times.", "",
             f"CPU: {data['provenance']['cpu']}; platform: {data['provenance']['platform']}. "
             "The only varied quantities are the declared workload factors; arithmetic policy is algebraic.", "",
             "![Graph-size scaling](figures/chain_scaling.png)", "",
             "![Rank scaling](figures/rank_scaling.png)", "",
             "![Construction and verification](figures/phase_breakdown.png)", "",
             "## Proof outcomes by workload", "",
             "These counts describe controlled synthetic variants. Work-limit points measure failed "
             "attempt cost; they do not establish a successful inference-time trend beyond that boundary.", "",
             "| Family | Layout | Mode | Proved sizes | Work-limit / unresolved sizes |",
             "|---|---|---|---|---|"]
    for family, layout, modes in (("chain", "canonical", labels), ("chain", "permuted", labels),
                                  ("rank", "symbolic", ("explicit", "automatic"))):
        for mode in modes:
            group = [r for r in rows if r["family"] == family and r["layout"] == layout and r["mode"] == mode]
            good = ", ".join(str(r["size"]) for r in group if r["status"] == "proved") or "—"
            bad = ", ".join(str(r["size"]) for r in group if r["status"] != "proved") or "—"
            lines.append(f"| {family} | {layout} | {mode} | {good} | {bad} |")
    lines += ["",
             "## Raw case summaries", "",
             "| Family | Layout | Size | Mode | Status | Median total ms | Construction ms | Verification ms |",
             "|---|---|---:|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['layout']} | {r['size']} | {r['mode']} | {r['status']} | "
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
    parser.add_argument("--budget", type=int, default=1000000)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1 or args.budget < 1:
        parser.error("invalid measurement settings")
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
    before = provenance(binary)
    raw = OUTPUT / "performance_raw.json"
    raw.parent.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop("TEPL_INDEX_PROFILE", None)
    subprocess.run([str(binary), str(MANIFEST), str(raw), str(args.warmups), str(args.repetitions), str(args.budget)],
                   cwd=b.ROOT, env=env, check=True, timeout=300)
    data = dict(schema_version=1, manifest=manifest, provenance=before,
                warmups=args.warmups, repetitions=args.repetitions, budget=args.budget,
                records=json.loads(raw.read_text())["records"])
    validate(data)
    b.save_json(RESULT, data)
    raw.unlink()
    render(data)
    print(f"Measured {len(manifest['cases'])} cases. Report: {REPORT.relative_to(b.ROOT)}")


if __name__ == "__main__":
    main()
