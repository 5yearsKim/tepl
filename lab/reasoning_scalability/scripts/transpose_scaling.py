#!/usr/bin/env python3
"""Check whole symbolic transpose chains against one composed transpose."""

import argparse
from collections import Counter
import csv
from itertools import permutations, product
import json
import os
from pathlib import Path
import random
import re
import statistics

import numpy as np

import benchmark as b

SIZES = (2, 4, 8, 16, 32, 64, 128)
MODES = ("explicit", "automatic")
REPETITIONS = 5
TIMEOUT = 30
NAME = "transpose_scaling"
DIRECTORY = b.HERE / "inputs" / NAME
RESULT = b.HERE / "results" / f"{NAME}.json"


def compose(sequence):
    result = list(range(3))
    for p in sequence:
        result = [result[i] for i in p]
    return result


def validate():
    # Dense transpose execution and coordinate-list composition are independent
    # evaluators. Unequal extents make incorrect permutation orders visible.
    tensor = np.arange(24).reshape(2, 3, 4)
    choices = list(permutations(range(3)))
    checked = 0
    sequences = list(product(choices, repeat=2))
    for n in SIZES:
        rng = random.Random(20261020 + n)
        sequences += [tuple(rng.choice(choices) for _ in range(n)) for _ in range(16)]
    for sequence in sequences:
        actual = tensor
        for p in sequence:
            actual = actual.transpose(p)
        np.testing.assert_array_equal(actual, tensor.transpose(compose(sequence)))
        checked += 1
    return checked


def source(n, mode):
    name = f"transpose_{mode}_{n:03}"
    lhs = "A"
    combined = "@p1.permutation"
    for i in range(1, n + 1):
        lhs = f"(transpose[@p{i}] {lhs})"
        if i > 1:
            combined = f"gather({combined}, @p{i}.permutation)"
    rhs = "(transpose[auto] A)" if mode == "automatic" else "(transpose[@combined] A)"
    relative = Path(os.path.relpath(b.DIALECT, DIRECTORY)).as_posix()
    text = (f'from "{relative}" import TensorLang as t;\nuse t::{{transpose}};\n\n'
            f"rule {name} {{\n    A: f32[D0,D1,D2]\n    {lhs}\n    =>\n    {rhs}\n"
            "    where { D0 > 0 && D1 > 0 && D2 > 0 }\n")
    if mode == "explicit":
        text += f"    derive {{ @combined = {{ permutation: {combined} }}; }}\n"
    return name, text + "}\n"


def prepare():
    checked = validate()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / "tepl.toml").write_text('tepl-version = "0.1.0"\n')
    cases = []
    for n in SIZES:
        for mode in MODES:
            name, text = source(n, mode)
            path = DIRECTORY / f"{name}.tepl"
            path.write_text(text)
            cases.append({"case_id": name, "family": "transpose", "size": n,
                          "attribute_mode": mode, "file": str(path.relative_to(b.ROOT)),
                          "input_hash": b.digest(path), "input_nodes": n + 1,
                          "input_bytes": path.stat().st_size,
                          "requested_fields": int(mode == "automatic")})
    manifest = {"cases": cases, "sizes": list(SIZES), "rank": 3,
                "permutations": "independent symbolic permutations at every stage",
                "semantic_policy": "algebraic", "repetitions": REPETITIONS,
                "timeout_seconds": TIMEOUT, "seed": 20261020,
                "independent_dense_checks": checked}
    b.save_json(DIRECTORY / "manifest.json", manifest)
    return manifest


def constructor_check(stdout, case):
    blocks = re.findall(r"^\s*constructor auto#\d+:\n((?:        \w+ = [^\n]*\n)+)",
                        stdout, re.MULTILINE)
    fields = [m for block in blocks for m in re.findall(r"^        (\w+) = (.*)$", block, re.MULTILINE)]
    expected = Counter({"permutation": 1}) if case["attribute_mode"] == "automatic" else Counter()
    complete = (Counter(f for f, _ in fields) == expected
                and not any(re.search(r"auto#\d+\.", expr) for _, expr in fields)
                and bool(re.search(r"^\s*additional-guard: true\s*$", stdout, re.MULTILINE)))
    return {"complete": complete, "constructor_count": len(blocks), "field_count": len(fields)}


def classify(row):
    stdout = (b.ROOT / row["stdout_file"]).read_text()
    stderr = (b.ROOT / row["stderr_file"]).read_text()
    outcome = b.classify(row["exit_code"], stdout, stderr, row["outcome"] == "timeout")
    details = constructor_check(stdout, row) if outcome == "proved" else None
    if details is not None and not details["complete"]:
        outcome = "error"
    return outcome, details


def check(data):
    provenance = data["provenance"]
    assert b.digest(provenance["compiler"]) == provenance["compiler_hash"], "compiler changed"
    for path, expected in provenance["dependencies"].items():
        assert b.digest(b.ROOT / path) == expected, "dependency changed: " + path
    for row in data["records"]:
        outcome, details = classify(row)
        assert outcome == row["outcome"], row["case_id"]
        assert details == row["constructors"], row["case_id"]
    if data["completed"]:
        assert len(data["records"]) == len(data["manifest"]["cases"]) * (REPETITIONS + 1)
        for case in data["manifest"]["cases"]:
            rows = [r for r in data["records"] if r["case_id"] == case["case_id"]]
            assert Counter(r["stage"] for r in rows) == {NAME + "_warmup": 1,
                                                         NAME + "_measured": REPETITIONS}
            assert all(r["peak_rss_mib"] is not None for r in rows), case["case_id"]


def summarize(data):
    rows = []
    for case in data["manifest"]["cases"]:
        samples = [r for r in data["records"] if r["case_id"] == case["case_id"]
                   and r["stage"] == NAME + "_measured"]
        outcomes = Counter(r["outcome"] for r in samples)
        row = {"n": case["size"], "mode": case["attribute_mode"],
               "outcome": next(iter(outcomes)) if len(outcomes) == 1 else "mixed",
               "diagnostic": samples[0]["diagnostic"]}
        for field, label in (("elapsed_seconds", "seconds"), ("peak_rss_mib", "rss_mib")):
            values = [r[field] for r in samples]
            for statistic, func in (("median", statistics.median), ("min", min), ("max", max)):
                row[statistic + "_" + label] = func(values)
        rows.append(row)
    b.save_json(b.HERE / f"results/{NAME}_summary.json", rows)
    with (b.HERE / f"results/{NAME}_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def analyze(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = summarize(data)
    colors = {"explicit": "#246fa8", "automatic": "#24855b"}
    labels = {"explicit": "Explicit composed permutation", "automatic": "Inferred permutation (auto)"}
    markers = {"proved": "o", "unresolved": "^", "budget_exhausted": "X",
               "timeout": "s", "error": "x", "mixed": "D"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, measure, label in zip(axes, ("seconds", "rss_mib"),
                                 ("Checking time (seconds)", "Peak process RSS (MiB)")):
        for mode in MODES:
            selected = [r for r in rows if r["mode"] == mode]
            ax.plot([r["n"] for r in selected], [r["median_" + measure] for r in selected],
                    color=colors[mode], alpha=.5, linewidth=1.4)
            for row in selected:
                value = row["median_" + measure]
                ax.errorbar(row["n"], value,
                            yerr=[[value - row["min_" + measure]], [row["max_" + measure] - value]],
                            color=colors[mode], marker=markers[row["outcome"]],
                            markersize=7, capsize=3, linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=.23)
    axes[0].set_yscale("log")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(SIZES, labels=[str(n) for n in SIZES])
    axes[1].set_xlabel("Transpose operations on the LHS (RHS: one transpose)")
    handles = [Line2D([], [], color=colors[m], linewidth=2, label=labels[m]) for m in MODES]
    handles += [Line2D([], [], color="#444444", linestyle="none", marker=markers[s],
                       label={"proved": "Verified", "unresolved": "Unresolved",
                              "budget_exhausted": "Proof budget exhausted", "timeout": "Timeout",
                              "error": "Error", "mixed": "Mixed outcomes"}[s])
                for s in markers if any(r["outcome"] == s for r in rows)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .91),
               ncol=2, frameon=False)
    fig.suptitle("TEPL scalability: transpose composition", fontsize=16, y=.99)
    fig.text(.55, .94, "Rank 3 · independent symbolic permutations · fixed proof budgets", ha="center")
    note = "Median of 5 runs; whiskers show min–max. Timing includes CLI report output. Default Bazel fastbuild.\n"
    note += "Unsuccessful checks are included; lower cost can reflect earlier proof-budget exhaustion."
    fig.text(.03, .018, note, fontsize=9, va="bottom")
    fig.subplots_adjust(left=.12, right=.98, top=.78, bottom=.14, hspace=.16)
    directory = b.HERE / "reports"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{NAME}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    report = ["# Transpose-chain scalability", "",
              f"![Checking time and memory]({NAME}.png)", "",
              "Each rule collapses a chain of n transposes to one transpose. The LHS has independent "
              "symbolic permutation attributes at every stage. Inputs have rank three with symbolic "
              "positive extents. The explicit RHS uses the derived composition of all n permutations; "
              "the auto RHS asks TEPL to infer that permutation. Each whole rule is checked independently.", "",
              "The algebraic policy, default proof budgets (1,000,000 work units per synthesis/verification "
              "context), compiler, and timing method match the corrected dot-chain experiment. "
              "There is one warm-up and five measured fresh-process invocations per case, with a "
              "30-second timeout. Transpose values do not add reductions or arithmetic expansion.", "",
              f"The generator passed {data['manifest']['independent_dense_checks']} independent dense checks, "
              "including every pair of rank-three permutations and random longer compositions.", "",
              "| Transposes | RHS attributes | Outcome | Median time | Median peak RSS |",
              "| ---: | --- | --- | ---: | ---: |"]
    for row in rows:
        report.append(f"| {row['n']} | {row['mode']} | {row['outcome']} | "
                      f"{row['median_seconds']:.3f} s | {row['median_rss_mib']:.2f} MiB |")
    report += ["", "Verified sampled lengths:"]
    for mode in MODES:
        verified = [str(r["n"]) for r in rows if r["mode"] == mode and r["outcome"] == "proved"]
        report.append(f"- {mode}: {', '.join(verified) or 'none'}.")
    report += ["", "The sampled lengths do not establish the exact boundary between them. These results "
               "characterize rank-three symbolic permutation composition, not arbitrary tensor graphs. "
               "Failed checks are retained in the figure and are not completed proofs.", "",
               f"[Raw records and source hashes](../results/{NAME}.json) · "
               f"[Summary CSV](../results/{NAME}_summary.csv) · "
               f"[SVG]({NAME}.svg) · [PDF]({NAME}.pdf)", "",
               "Validation: python3 lab/reasoning_scalability/scripts/transpose_scaling.py --check", ""]
    (directory / f"{NAME}.md").write_text("\n".join(report))
    b.save_json(b.HERE / f"results/{NAME}_plot_environment.json",
                {"matplotlib": matplotlib.__version__, "script_hash": b.digest(Path(__file__)),
                 "raw_results_hash": b.digest(RESULT)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare()
        print(f"Prepared {len(manifest['cases'])} cases; {manifest['independent_dense_checks']} dense checks passed.")
        return
    compiler = (b.ROOT / "bazel-bin/tepl").resolve(strict=True)
    if args.check or args.analyze or args.resume:
        data = json.loads(RESULT.read_text())
        check(data)
        if args.check or args.analyze:
            assert data["completed"], "measurement is incomplete"
            if args.analyze:
                analyze(data)
            print(f"Validated all {len(data['records'])} invocations, input/source hashes, outcomes, and constructors.")
            return
    else:
        if RESULT.exists():
            raise SystemExit("Measurements already exist; use --resume or preserve them before a new run.")
        manifest = prepare()
        provenance = b.provenance(compiler, manifest)
        for path in (Path(__file__).resolve(), DIRECTORY / "manifest.json", DIRECTORY / "tepl.toml"):
            provenance["dependencies"][str(path.relative_to(b.ROOT))] = b.digest(path)
        provenance["active_work_budgets"] = {}
        for name, path in (("synthesis", "src/core/attr_synth/synthesize.h"),
                           ("independent_verification", "src/core/semantics/proof/verify.h")):
            provenance["active_work_budgets"][name] = int(re.search(r"max_work = (\d+)",
                                                                 (b.ROOT / path).read_text())[1])
        data = {"manifest": manifest, "provenance": provenance, "records": [], "completed": False}
        b.save_json(RESULT, data)
    done = {(r["stage"], r["repetition"], r["case_id"]) for r in data["records"]}
    total = len(data["manifest"]["cases"]) * (REPETITIONS + 1)
    for stage, repeats in ((NAME + "_warmup", 1), (NAME + "_measured", REPETITIONS)):
        for repeat in range(repeats):
            cases = list(data["manifest"]["cases"])
            random.Random(20261020 + repeat).shuffle(cases)
            for case in cases:
                if (stage, repeat, case["case_id"]) in done:
                    continue
                row = b.run_case(case, compiler, stage, repeat, TIMEOUT)
                # Reuse the per-process timer; replace its dot-specific
                # constructor classification with this family's single field.
                row["outcome"], row["constructors"] = classify(row)
                data["records"].append(row)
                b.save_json(RESULT, data)
                print(f"[{len(data['records'])}/{total}] {stage} n={case['size']} "
                      f"{case['attribute_mode']} {row['outcome']} {row['elapsed_seconds']:.3f}s", flush=True)
    data["completed"] = True
    check(data)
    b.save_json(RESULT, data)
    summarize(data)
    print("Finished transpose-chain scalability measurements.")


if __name__ == "__main__":
    main()
