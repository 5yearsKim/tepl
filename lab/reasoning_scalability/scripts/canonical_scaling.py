#!/usr/bin/env python3
"""Measure canonical dot-chain checking, preserving separately named runs."""

import argparse
import json
import os
from pathlib import Path
import random
import re

import numpy as np

import benchmark as b

SIZES = (3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32)
MANIFEST = b.HERE / "inputs/canonical_scaling/manifest.json"


def run_name(value):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise argparse.ArgumentTypeError("use lowercase letters, digits, and underscores")
    return value


def result_path(name):
    return b.HERE / "results" / f"{run_name(name)}.json"


def validate(n):
    leaves, lhs, rhs = b.chain(n, "canonical")
    rng = np.random.default_rng(20261020 + n)
    values = {v.name: rng.normal(0, .25, (2, 2 + i % 2, 2 + (i + 1) % 2))
              for i, v in enumerate(leaves)}

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
    np.testing.assert_allclose(evaluate(lhs), oracle, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(evaluate(rhs), oracle, rtol=1e-10, atol=1e-15)


def prepare():
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (MANIFEST.parent / "tepl.toml").write_text('tepl-version = "0.1.0"\n')
    cases = []
    for n in SIZES:
        validate(n)
        name, source = b.dot_source(n, "canonical", "explicit")
        lines = source.splitlines()
        relative = Path(os.path.relpath(b.DIALECT, MANIFEST.parent)).as_posix()
        lines[0] = f'from "{relative}" import TensorLang as t;'
        path = MANIFEST.parent / f"{name}.tepl"
        path.write_text("\n".join(lines) + "\n")
        cases.append({"case_id": name, "family": "dot", "size": n,
                      "layout": "canonical", "attribute_mode": "explicit",
                      "file": str(path.relative_to(b.ROOT)), "input_hash": b.digest(path),
                      "input_nodes": 2 * (n - 1), "input_bytes": path.stat().st_size,
                      "requested_fields": 0})
    manifest = {"cases": cases, "repetitions": 5, "timeout_seconds": 30,
                "seed": 20261020,
                "validation": "All 11 generated chains numerically checked against an independent matrix-chain oracle."}
    b.save_json(MANIFEST, manifest)
    return manifest


def check(data):
    p = data["provenance"]
    assert b.digest(p["compiler"]) == p["compiler_hash"], "compiler changed"
    for path, expected in p["dependencies"].items():
        assert b.digest(b.ROOT / path) == expected, "dependency changed: " + path
    if data["completed"]:
        assert len(data["records"]) == len(SIZES) * 6
        for case in data["manifest"]["cases"]:
            rows = [r for r in data["records"] if r["case_id"] == case["case_id"]]
            measured = data.get("run_name", "canonical_scaling") + "_measured"
            assert len([r for r in rows if r["stage"] == measured]) == 5
            for row in rows:
                stdout = (b.ROOT / row["stdout_file"]).read_text()
                stderr = (b.ROOT / row["stderr_file"]).read_text()
                outcome = b.classify(row["exit_code"], stdout, stderr, row["outcome"] == "timeout")
                if outcome == "proved" and not b.constructor_check(stdout, row)["complete"]:
                    outcome = "error"
                assert outcome == row["outcome"], row["case_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", type=run_name, default="canonical_scaling",
                        help="name for results and logs; existing runs are preserved")
    parser.add_argument("--description", default="after the shape equality fix",
                        help="description of the checker version used in this run")
    args = parser.parse_args()
    result = result_path(args.run_name)
    compiler = (b.ROOT / "bazel-bin/tepl").resolve(strict=True)
    if args.check or args.resume:
        data = json.loads(result.read_text())
        check(data)
        if args.check:
            assert data["completed"], "measurement is incomplete"
            print("Validated all 66 invocations, compiler/input hashes, and proof outcomes.")
            return
    else:
        if result.exists():
            raise SystemExit("Saved measurements exist; use --resume or preserve them before a new run.")
        manifest = prepare()
        provenance = b.provenance(compiler, manifest)
        for path in (Path(__file__).resolve(), MANIFEST, MANIFEST.parent / "tepl.toml"):
            provenance["dependencies"][str(path.relative_to(b.ROOT))] = b.digest(path)
        data = {"manifest": manifest, "provenance": provenance,
                "run_name": args.run_name, "description": args.description,
                "records": [], "completed": False}
        b.save_json(result, data)
    done = {(r["stage"], r["repetition"], r["case_id"]) for r in data["records"]}
    for stage, repeats in ((args.run_name + "_warmup", 1), (args.run_name + "_measured", 5)):
        for repeat in range(repeats):
            cases = list(data["manifest"]["cases"])
            random.Random(20261020 + repeat).shuffle(cases)
            for case in cases:
                if (stage, repeat, case["case_id"]) in done:
                    continue
                row = b.run_case(case, compiler, stage, repeat, 30)
                data["records"].append(row)
                b.save_json(result, data)
                print(f"[{len(data['records'])}/66] {stage} n={case['size']} "
                      f"{row['outcome']} {row['elapsed_seconds']:.3f}s", flush=True)
    data["completed"] = True
    check(data)
    b.save_json(result, data)
    print("Finished canonical dot-chain scalability measurements.")


if __name__ == "__main__":
    main()
