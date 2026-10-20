#!/usr/bin/env python3
"""Plot post-fix canonical dot-chain checking costs, including proof failures."""

from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import benchmark as b
import canonical_scaling as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=run.run_name, default="canonical_scaling")
    args = parser.parse_args()
    name = args.run_name
    result = run.result_path(name)
    data = json.loads(result.read_text())
    run.check(data)
    assert data["completed"], "finish the measurement sweep first"
    rows = []
    for case in data["manifest"]["cases"]:
        samples = [r for r in data["records"] if r["case_id"] == case["case_id"]
                   and r["stage"] == name + "_measured"]
        counts = Counter(s["outcome"] for s in samples)
        row = {"n": case["size"], "dots_per_side": case["size"] - 1,
               "outcome": next(iter(counts)) if len(counts) == 1 else "mixed",
               "diagnostic": samples[0]["diagnostic"]}
        for field, label in (("elapsed_seconds", "seconds"), ("peak_rss_mib", "rss_mib")):
            values = [s[field] for s in samples if s[field] is not None]
            for statistic, func in (("median", statistics.median), ("min", min), ("max", max)):
                row[statistic + "_" + label] = func(values) if values else None
        rows.append(row)
    b.save_json(b.HERE / f"results/{name}_summary.json", rows)
    with (b.HERE / f"results/{name}_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    styles = {"proved": ("#246fa8", "o", "Verified"),
              "unresolved": ("#df8c24", "^", "Unresolved"),
              "budget_exhausted": ("#c53e3e", "X", "Work budget exhausted"),
              "timeout": ("#7b5bb0", "s", "Timeout"),
              "error": ("#333333", "x", "Error"),
              "mixed": ("#7b5bb0", "D", "Mixed outcomes")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    xs = [r["n"] for r in rows]
    for ax, measure, label in zip(axes, ("seconds", "rss_mib"),
                                 ("Checking time (seconds)", "Peak process RSS (MiB)")):
        values = [r["median_" + measure] for r in rows]
        ax.plot(xs, values, color="#a4acb3", linewidth=1.5, zorder=1)
        for row in rows:
            value = row["median_" + measure]
            if value is None:
                continue
            color, marker, _ = styles[row["outcome"]]
            ax.errorbar(row["n"], value,
                        yerr=[[value - row["min_" + measure]],
                              [row["max_" + measure] - value]],
                        color=color, marker=marker, markersize=7,
                        capsize=3, linewidth=1.3, zorder=3)
        ax.set_ylabel(label)
        ax.grid(alpha=.23)
    axes[0].set_yscale("log")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xticks(xs)
    axes[1].set_xlabel("Input tensors n (composition depth = n − 1 dot operations)")
    axes[1].set_xlim(2.5, 32.8)
    present = [status for status in styles if any(r["outcome"] == status for r in rows)]
    handles = [Line2D([], [], linestyle="none", color=styles[s][0],
                      marker=styles[s][1], markersize=7, label=styles[s][2]) for s in present]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .91),
               ncol=min(3, len(handles)), frameon=False)
    fig.suptitle("TEPL scalability: dot-chain checking attempts", fontsize=16, y=.99)
    fig.text(.55, .935, "Canonical axes · explicit RHS attributes · fixed proof budgets",
             ha="center", fontsize=11, color="#444444")
    verified = [r["n"] for r in rows if r["outcome"] == "proved"]
    note = ("Verified chain sizes: " + (", ".join(map(str, verified)) or "none") +
            ". Other markers show unsuccessful checks, not completed proofs.\n"
            "Median of 5 runs; whiskers show min–max. Timing includes CLI report output. Default Bazel fastbuild.\n"
            "Lower cost at larger sizes can reflect earlier budget exhaustion.")
    fig.text(.03, .018, note, fontsize=9, va="bottom")
    fig.subplots_adjust(left=.12, right=.98, top=.82, bottom=.15, hspace=.15)
    directory = b.HERE / "reports"
    directory.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    description = data.get("description", "after the shape equality fix")
    text = [f"# Canonical dot chain scalability {description}", "",
            f"![Checking time and memory versus dot-chain length]({name}.png)", "",
            "This graph measures the cost of checking each composite reassociation rule, including unsuccessful attempts. "
            "Markers distinguish completed proofs, unresolved obligations, and exhausted budgets. "
            "It does not establish successful-proof scalability beyond the verified sizes.", "",
            "Eleven chain lengths, symbolic positive extents, rank-three inputs, explicit RHS attributes, "
            "algebraic policy, one warm-up and five measured runs, fixed budgets, and a 30-second timeout. "
            "All generated chains passed independent numerical validation. This run uses the checker "
            f"{description}; earlier measurements remain separate.", "",
            "| Input tensors | Outcome | Median time | Median peak RSS |",
            "| ---: | --- | ---: | ---: |"]
    for row in rows:
        text.append(f"| {row['n']} | {row['outcome']} | {row['median_seconds']:.3f} s | "
                    f"{row['median_rss_mib']:.2f} MiB |")
    text += ["", "The connecting line tracks all checking attempts. Drops in time or memory can reflect "
             "earlier budget exhaustion, rather than improved proof scalability.", "",
             f"[Raw measurements and source hashes](../results/{name}.json) · "
             f"[Summary CSV](../results/{name}_summary.csv) · "
             f"[SVG figure]({name}.svg) · [PDF figure]({name}.pdf)", "",
             "```sh", f"python3 lab/reasoning_scalability/scripts/canonical_scaling.py --run-name {name} --check",
             f"python3 lab/reasoning_scalability/scripts/plot_canonical_scaling.py --run-name {name}", "```", ""]
    (directory / f"{name}.md").write_text("\n".join(text))
    b.save_json(b.HERE / f"results/{name}_plot_environment.json", {
        "matplotlib": matplotlib.__version__, "plot_script_sha256": b.digest(Path(__file__)),
        "raw_results_sha256": b.digest(result)})
    print(f"Saved reports/{name}.png, SVG/PDF exports, CSV, and figure notes.")


if __name__ == "__main__":
    main()
