#!/usr/bin/env python3
"""Generate, validate, and measure the frozen reasoning scalability workloads."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import re
import select
import signal
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
DIALECT = ROOT / "examples/semantic/dialects/tensor.tepl"
SIZES = (3, 4, 6, 8, 12, 16, 24, 32)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
FIELDS = ("lhs_contracting_dimensions", "rhs_contracting_dimensions",
          "lhs_batching_dimensions", "rhs_batching_dimensions")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class Node:
    name: str
    axes: tuple
    children: tuple = ()
    attrs: dict | None = None


def dot(lhs, rhs, label, name):
    assert set(lhs.axes) & set(rhs.axes) == {"b", label}
    attrs = dict(zip(FIELDS, ([lhs.axes.index(label)], [rhs.axes.index(label)],
                              [lhs.axes.index("b")], [rhs.axes.index("b")])))
    axes = ("b",) + tuple(a for a in lhs.axes if a not in ("b", label))
    axes += tuple(a for a in rhs.axes if a not in ("b", label))
    assert len(axes) == 3 and len(set(axes)) == 3
    return Node(name, axes, (lhs, rhs), attrs)


def chain(n, layout):
    leaves = []
    for i in range(1, n + 1):
        axes = ("b", f"d{i-1}", f"d{i}")
        order = (0, 1, 2) if layout == "canonical" else PERMUTATIONS[(i - 1) % 6]
        leaves.append(Node(f"A{i}", tuple(axes[j] for j in order)))
    lhs = leaves[0]
    for i in range(1, n):
        lhs = dot(lhs, leaves[i], f"d{i}", f"l{i}")
    rhs = leaves[-1]
    for i in range(n - 2, -1, -1):
        rhs = dot(leaves[i], rhs, f"d{i+1}", f"r{i+1}")
    assert lhs.axes == rhs.axes == ("b", "d0", f"d{n}")
    return leaves, lhs, rhs


def internal(node):
    if node.children:
        yield from internal(node.children[0])
        yield from internal(node.children[1])
        yield node


def expression(node, automatic=False):
    if not node.children:
        return node.name
    desc = "auto" if automatic else "@" + node.name
    return f"(dot[{desc}] {expression(node.children[0], automatic)} {expression(node.children[1], automatic)})"


def source_header():
    dialect = os.path.relpath(DIALECT, HERE / "inputs")
    return f'from "{dialect}" import TensorLang as t;\nuse t::{{dot, add, mul}};\n\n'


def dot_source(n, layout, mode):
    leaves, lhs, rhs = chain(n, layout)
    name = f"dot_{layout}_{mode}_{n:02}"
    lines = [source_header() + f"rule {name} {{"]
    for leaf in leaves:
        dims = ["B" if a == "b" else "D" + a[1:] for a in leaf.axes]
        lines.append(f"    {leaf.name}: f32[{', '.join(dims)}]")
    lines += ["", "    " + expression(lhs), "    =>",
              "    " + expression(rhs, mode == "automatic"), "    where {"]
    conditions = [f"{d} > 0" for d in ["B"] + [f"D{i}" for i in range(n + 1)]]
    for node in internal(lhs):
        conditions += [f"@{node.name}.{field} == {node.attrs[field]}" for field in FIELDS]
    lines.append("        " + "\n        && ".join(conditions))
    lines.append("    }")
    if mode == "explicit":
        lines.append("    derive {")
        for node in internal(rhs):
            fields = ", ".join(f"{f}: {node.attrs[f]}" for f in FIELDS)
            lines.append(f"        @{node.name} = {{ {fields} }};")
        lines.append("    }")
    lines.append("}")
    return name, "\n".join(lines) + "\n"


def product_source(k):
    def product(order):
        factors = [f"(add A{i} B{i})" for i in order]
        result = factors[0]
        for factor in factors[1:]:
            result = f"(mul {result} {factor})"
        return result
    name = f"product_{k:02}"
    lines = [source_header() + f"rule {name} {{"]
    for i in range(1, k + 1):
        lines += [f"    A{i}: f32[M, N]", f"    B{i}: f32[M, N]"]
    lines += ["", "    " + product(range(1, k + 1)), "    =>",
              "    " + product(range(k, 0, -1)), "    where { M > 0 && N > 0 }", "}"]
    return name, "\n".join(lines) + "\n"


def reference_validate():
    """Check descriptor-selected reads against an independent label-based oracle."""
    rng = np.random.default_rng(20261020)
    validated = 0
    for n in SIZES:
        for layout in ("canonical", "permuted"):
            leaves, lhs, rhs = chain(n, layout)
            dims = {"b": 2, **{f"d{i}": 2 + (i % 2) for i in range(n + 1)}}
            values = {leaf.name: rng.normal(0, .25, tuple(dims[a] for a in leaf.axes))
                      for leaf in leaves}

            def evaluate(node):
                if not node.children:
                    return values[node.name]
                a, b = map(evaluate, node.children)
                lc, rc, lb, rb = (node.attrs[f][0] for f in FIELDS)
                # Interpret positions, not node logical labels.
                af = next(i for i in range(3) if i not in (lc, lb))
                bf = next(i for i in range(3) if i not in (rc, rb))
                return np.matmul(np.transpose(a, (lb, af, lc)),
                                 np.transpose(b, (rb, rc, bf)))

            # Separate oracle: normalize physical leaves using their labels,
            # then multiply canonical matrices without consulting descriptors.
            canonical = [np.transpose(values[v.name],
                         [v.axes.index(a) for a in ("b", f"d{i}", f"d{i+1}")])
                         for i, v in enumerate(leaves)]
            oracle = canonical[0]
            for tensor in canonical[1:]:
                oracle = np.matmul(oracle, tensor)
            np.testing.assert_allclose(evaluate(lhs), oracle, rtol=1e-10, atol=1e-15)
            np.testing.assert_allclose(evaluate(rhs), oracle, rtol=1e-10, atol=1e-15)
            assert len(list(internal(lhs))) == len(list(internal(rhs))) == n - 1
            validated += 1
    for k in range(2, 11):
        factors = rng.normal(size=(k, 2, 3, 4)).sum(axis=1)
        a = factors[0]
        for factor in factors[1:]:
            a = a * factor
        b = factors[-1]
        for factor in factors[-2::-1]:
            b = b * factor
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        validated += 1
    return {"seed": 20261020, "logical_cases_checked": validated,
            "method": "position-based descriptor interpreter versus canonical-label matrix-chain oracle; product reversal"}


def prepare():
    (HERE / "inputs").mkdir(parents=True, exist_ok=True)
    (HERE / "inputs/tepl.toml").write_text('tepl-version = "0.1.0"\n')
    cases = []
    for n in SIZES:
        for layout in ("canonical", "permuted"):
            for mode in ("explicit", "automatic"):
                name, source = dot_source(n, layout, mode)
                cases.append({"case_id": name, "family": "dot", "size": n,
                              "layout": layout, "attribute_mode": mode,
                              "input_nodes": 2 * (n - 1), "expected_expanded_terms": 1,
                              "requested_fields": 4 * (n - 1) if mode == "automatic" else 0,
                              "source": source})
    for k in range(2, 11):
        name, source = product_source(k)
        cases.append({"case_id": name, "family": "product", "size": k,
                      "layout": "same_shape", "attribute_mode": "none",
                      "input_nodes": 2 * (2 * k - 1), "expected_expanded_terms": 2 ** k,
                      "requested_fields": 0, "source": source})
    for case in cases:
        path = HERE / "inputs" / (case["case_id"] + ".tepl")
        source = case.pop("source")
        path.write_text(source)
        case.update(file=str(path.relative_to(ROOT)), input_hash=digest(path),
                    input_bytes=len(source.encode()))
    manifest = {"schema_version": 1, "seed": 20261020, "repetitions": 5,
                "timeout_seconds": 30, "cases": cases,
                "validation": reference_validate()}
    save_json(HERE / "inputs/manifest.json", manifest)
    return manifest


def classify(returncode, stdout, stderr, timeout=False):
    text = stdout + "\n" + stderr
    if timeout:
        return "timeout"
    if returncode is not None and returncode < 0:
        return "error"
    if returncode == 0 and stdout.count("status=proved") == 1:
        return "proved"
    if "budget exhausted:" in text:
        return "budget_exhausted"
    if returncode == 1 and "cannot verify rule" in text:
        return "unresolved"
    return "error"


def constructor_check(stdout, case):
    blocks = re.findall(r"^\s*constructor auto#\d+:\n((?:        \w+ = [^\n]*\n)+)",
                        stdout, re.MULTILINE)
    fields = [m for block in blocks for m in re.findall(r"^        (\w+) = (.*)$", block, re.MULTILINE)]
    expected = Counter({f: case["size"] - 1 for f in FIELDS}) if case["attribute_mode"] == "automatic" else Counter()
    complete = (Counter(f for f, _ in fields) == expected
                and not any(re.search(r"auto#\d+\.", expr) for _, expr in fields)
                and bool(re.search(r"^\s*additional-guard: true\s*$", stdout, re.MULTILINE)))
    return {"complete": complete, "constructor_count": len(blocks),
            "field_count": len(fields)}


def run_case(case, compiler, stage, repetition, timeout):
    directory = HERE / "results/logs" / stage
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"{case['case_id']}_{repetition}"
    stdout_path, stderr_path, rss_path = (Path(str(stem) + suffix)
                                         for suffix in (".stdout", ".stderr", ".rss"))
    command = [str(compiler), "check", case["file"], "--semantic-policy=algebraic",
               "--rule=" + case["case_id"]]
    if stage != "diagnostic":
        command = ["/usr/bin/time", "--quiet", "-f", "%M", "-o", str(rss_path), "--"] + command
    env = dict(os.environ)
    env.pop("TEPL_INDEX_PROFILE", None)
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout_file,
                                   stderr=stderr_file, start_new_session=True)
        pidfd = os.pidfd_open(process.pid)
        ready, _, _ = select.select([pidfd], [], [], timeout)
        os.close(pidfd)
        if ready:
            process.wait()
        else:
            timed_out = True
            # Keep GNU time alive long enough to reap its terminated child and
            # write that child's peak RSS; then clean up the process group.
            if stage != "diagnostic":
                children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
                for pid in children.read_text().split() if children.exists() else []:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    elapsed = time.perf_counter() - started
    stdout, stderr = stdout_path.read_text(), stderr_path.read_text()
    rss = None
    if rss_path.exists():
        numbers = re.findall(r"^\d+$", rss_path.read_text(), re.MULTILINE)
        if numbers:
            rss = int(numbers[-1]) / 1024
    outcome = classify(process.returncode, stdout, stderr, timed_out)
    detail = constructor_check(stdout, case) if outcome == "proved" else None
    if outcome == "proved" and not detail["complete"]:
        outcome = "error"
    reason = re.search(r"cannot verify rule '[^']+': ([^\n]*)", stdout + "\n" + stderr)
    return {**case, "stage": stage, "repetition": repetition, "command": command,
            "elapsed_seconds": elapsed, "peak_rss_mib": rss, "outcome": outcome,
            "exit_code": process.returncode, "diagnostic": reason[1] if reason else "",
            "constructors": detail,
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT))}


def provenance(compiler, manifest):
    paths = [p for p in (ROOT / "src").rglob("*") if p.is_file()]
    paths += [DIALECT, Path(__file__), HERE / "README.md", HERE / "inputs/manifest.json",
              HERE / "inputs/tepl.toml", ROOT / "BUILD.bazel", ROOT / ".bazelrc",
              ROOT / "MODULE.bazel", ROOT / "MODULE.bazel.lock"]
    paths += [ROOT / c["file"] for c in manifest["cases"]]
    limits = dict((k, int(v)) for k, v in re.findall(
        r"static constexpr std::size_t (\w+) = (\d+);",
        (ROOT / "src/core/semantics/proof/proof_policy.h").read_text()))
    return {"run_at_utc": datetime.now(timezone.utc).isoformat(),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines(),
            "compiler": str(compiler), "compiler_hash": digest(compiler),
            "dependencies": {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))},
            "proof_limits": limits, "platform": platform.platform(),
            "cpu": next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name')), platform.processor()),
            "logical_cpus": os.cpu_count(), "load_average_at_start": os.getloadavg(),
            "python": platform.python_version(), "numpy": np.__version__,
            "build_command": "bazel build //:tepl",
            "measurement_note": "CLI has no quiet semantic-report mode; --rule report formatting and redirected output are included in all timed invocations. GNU time wrapper overhead is included."}


def validate_saved(data):
    assert digest(Path(data["provenance"]["compiler"])) == data["provenance"]["compiler_hash"], "compiler changed"
    for path, expected in data["provenance"]["dependencies"].items():
        assert digest(ROOT / path) == expected, f"dependency changed: {path}"
    if data["completed"]:
        assert len(data["records"]) == 41 * 7
        for case in data["manifest"]["cases"]:
            rows = [r for r in data["records"] if r["case_id"] == case["case_id"]]
            assert Counter(r["stage"] for r in rows) == {"warmup": 1, "measured": 5, "diagnostic": 1}
            assert len({r["outcome"] for r in rows}) == 1, f"mixed outcomes: {case['case_id']}"
            for row in rows:
                assert (ROOT / row["stdout_file"]).exists() and (ROOT / row["stderr_file"]).exists()
                if row["stage"] == "measured":
                    assert row["peak_rss_mib"] is not None, f"missing RSS: {case['case_id']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = HERE / "results/run.json"
    if args.check:
        data = json.loads(result.read_text())
        validate_saved(data)
        assert data["completed"], "run is not complete"
        print("Validated compiler/source/input hashes, 287 records, per-case outcomes, and logs.")
        return
    if args.prepare:
        manifest = prepare()
        print(f"Prepared {len(manifest['cases'])} cases; independently validated {manifest['validation']['logical_cases_checked']} logical cases.")
        return
    compiler = args.compiler.resolve(strict=True)
    if args.resume:
        data = json.loads(result.read_text())
        validate_saved(data)
    else:
        if result.exists():
            raise SystemExit("Results already exist; use --resume or preserve them before a new run.")
        manifest = prepare()
        data = {"schema_version": 1, "manifest": manifest,
                "provenance": provenance(compiler, manifest), "records": [], "completed": False}
        save_json(result, data)
    manifest = data["manifest"]
    jobs = []
    for stage, repetitions in (("warmup", 1), ("measured", 5), ("diagnostic", 1)):
        for repetition in range(repetitions):
            cases = list(manifest["cases"])
            random.Random(manifest["seed"] + repetition).shuffle(cases)
            jobs += [(stage, repetition, case) for case in cases]
    done = {(r["stage"], r["repetition"], r["case_id"]) for r in data["records"]}
    for stage, repetition, case in jobs:
        if (stage, repetition, case["case_id"]) in done:
            continue
        row = run_case(case, compiler, stage, repetition, manifest["timeout_seconds"])
        data["records"].append(row)
        save_json(result, data)
        print(f"[{len(data['records'])}/{len(jobs)}] {stage} {repetition} {case['case_id']}: "
              f"{row['outcome']} {row['elapsed_seconds']:.3f}s RSS={row['peak_rss_mib']}", flush=True)
    data["completed"] = True
    data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    validate_saved(data)
    save_json(result, data)
    print("Completed and validated all 287 invocations.")


if __name__ == "__main__":
    main()
