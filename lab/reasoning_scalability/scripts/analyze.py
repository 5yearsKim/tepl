#!/usr/bin/env python3
"""Validate saved measurements and write summaries, figures, and a report."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import benchmark as b

HERE = b.HERE
COLORS = {"explicit": "#2878b5", "automatic": "#d47716", "none": "#2878b5"}
STATUS = {"proved": "P", "budget_exhausted": "B", "unresolved": "U",
          "timeout": "T", "error": "E", "mixed": "M"}


def summarize(data):
    summaries = []
    for case in data["manifest"]["cases"]:
        rows = [r for r in data["records"] if r["case_id"] == case["case_id"]
                and r["stage"] == "measured"]
        assert len(rows) == 5
        counts = Counter(r["outcome"] for r in rows)
        summary = {**case, "outcome": next(iter(counts)) if len(counts) == 1 else "mixed",
                   "outcome_counts": dict(counts), "diagnostic": rows[0]["diagnostic"]}
        for field, name in (("elapsed_seconds", "seconds"), ("peak_rss_mib", "rss_mib")):
            values = [r[field] for r in rows if r[field] is not None]
            for stat, func in (("median", statistics.median), ("min", min), ("max", max)):
                summary[f"{stat}_{name}"] = func(values) if values else None
        summaries.append(summary)
    return summaries


def verify_logs(data):
    for row in data["records"]:
        stdout = (b.ROOT / row["stdout_file"]).read_text()
        stderr = (b.ROOT / row["stderr_file"]).read_text()
        outcome = b.classify(row["exit_code"], stdout, stderr, row["outcome"] == "timeout")
        if outcome == "proved" and not b.constructor_check(stdout, row)["complete"]:
            outcome = "error"
        assert row["outcome"] == outcome, row["stdout_file"]


def groups(summaries, family):
    if family == "dot":
        return [(f"{layout.title()} / {mode}",
                 sorted([s for s in summaries if s["family"] == "dot"
                         and s["layout"] == layout and s["attribute_mode"] == mode],
                        key=lambda s: s["size"]), mode, layout)
                for layout in ("canonical", "permuted")
                for mode in ("explicit", "automatic")]
    return [("Products of sums", [s for s in summaries if s["family"] == "product"], "none", "canonical")]


def figures(summaries, directory, term_cap):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    dot_proofs = any(s["outcome"] == "proved" for s in summaries if s["family"] == "dot")
    for family, filename in (("dot", "dot_chain_cost"), ("product", "expression_growth")):
        entries = groups(summaries, family)
        fig, axs = plt.subplots(3, 1, figsize=(10, 9),
                                gridspec_kw={"height_ratios": [2, 2, 1.1], "hspace": .40})
        xs = sorted({s["size"] for _, rows, _, _ in entries for s in rows})
        attempt_cost = family == "dot" and not dot_proofs
        for label, rows, mode, layout in entries:
            color = COLORS[mode]
            style = "-" if layout == "canonical" else "--"
            marker = "o" if layout == "canonical" else "s"
            for ax, measure in zip(axs[:2], ("seconds", "rss_mib")):
                ys = [s[f"median_{measure}"] if attempt_cost or s["outcome"] == "proved" else np.nan
                      for s in rows]
                errors = np.array([[s[f"median_{measure}"] - s[f"min_{measure}"] for s in rows],
                                   [s[f"max_{measure}"] - s[f"median_{measure}"] for s in rows]])
                ax.errorbar([s["size"] for s in rows], ys, yerr=errors, color=color,
                            linestyle=style, marker=marker, markersize=5, capsize=3,
                            linewidth=1.7, label=label)
        axs[0].set_yscale("log")
        axs[0].set_ylabel("Median checking time (seconds)")
        axs[1].set_ylabel("Median peak process RSS (MiB)")
        axs[1].set_ylim(bottom=0)
        xlabel = "Input tensors n (each side has n - 1 dots)" if family == "dot" else "Sum factors k"
        for ax in axs[:2]:
            ax.grid(True, alpha=.22)
            ax.set_xticks(xs)
            ax.set_xlim(min(xs) - .4, max(xs) + .4)
            ax.set_xlabel(xlabel)
        axs[0].legend(loc="lower right", fontsize=9, ncol=2 if family == "dot" else 1)
        outcome_ax = axs[2]
        outcome_ax.set_xlim(min(xs) - .4, max(xs) + .4)
        outcome_ax.set_xticks(xs)
        outcome_ax.set_yticks(range(len(entries)), [entry[0] for entry in entries])
        outcome_ax.set_ylim(len(entries) - .5, -.5)
        for i, (_, rows, mode, _) in enumerate(entries):
            for row in rows:
                outcome_ax.text(row["size"], i, STATUS[row["outcome"]], ha="center", va="center",
                                color=COLORS[mode], fontweight="bold", fontsize=10)
        outcome_ax.set_title("Outcomes at every tested size", fontsize=10, loc="left")
        outcome_ax.set_xlabel(xlabel)
        outcome_ax.spines["left"].set_visible(False)
        outcome_ax.spines["bottom"].set_visible(False)
        outcome_ax.tick_params(length=0)
        if family == "product":
            boundary = np.log2(term_cap) + .5
            for ax in axs[:2]:
                ax.axvline(boundary, linestyle=":", color="#555555", linewidth=1.2)
            axs[0].text(boundary + .05, .95, f"Expected term-cap boundary\n(cap = {term_cap})",
                        transform=axs[0].get_xaxis_transform(), va="top", fontsize=8)
            top = axs[0].secondary_xaxis("top")
            top.set_xticks(xs, [str(2 ** k) for k in xs])
            top.set_xlabel("Analytical expanded terms 2^k")
            title = "Products of sums: successful checking cost and all outcomes"
        elif attempt_cost:
            title = "Dot-chain checking attempts: no dot-chain case proved"
        else:
            title = "Dot-chain reassociation: successful checking cost and all outcomes"
        fig.suptitle(title, fontsize=13, y=.985)
        note = ("All attempts shown above; these times do not measure successful proof scalability.\n"
                if attempt_cost else "Only consistently proved cases appear in cost curves; failures remain in the outcome strip.\n")
        note += ("P = proved; B = internal budget; U = unresolved; T = timeout; E = error; M = mixed.\n"
                 "Five runs; min/max whiskers; algebraic policy; fixed budgets; 30 s timeout. CLI report output included.")
        fig.text(.02, .015, note, fontsize=8, va="bottom")
        fig.subplots_adjust(left=.24, right=.96, top=.88, bottom=.15)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(directory / f"{filename}.{ext}", dpi=180, facecolor="white")
        plt.close(fig)
    return not dot_proofs


def format_seconds(value):
    return f"{value:.3f} s"


def render_report(data, summaries, attempt_cost):
    measured = [r for r in data["records"] if r["stage"] == "measured"]
    outcomes = Counter(s["outcome"] for s in summaries)
    p = data["provenance"]
    lines = ["# Reasoning cost and scalability results", "",
             f"The frozen 41-case experiment completed on {data['finished_at_utc'][:10]} using five measured runs per case, "
             "one warm-up, and one diagnostic run. All 287 invocations and input/compiler hashes were validated.", "",
             f"Across the 41 cases, **{outcomes['proved']} consistently proved**, "
             f"**{outcomes['budget_exhausted']} exhausted an internal budget**, and "
             f"**{outcomes['unresolved']} remained unresolved**. "
             f"There were {outcomes['timeout']} consistent timeouts, {outcomes['error']} error cases, "
             f"and {outcomes['mixed']} cases with mixed measured outcomes.", "",
             "The generator's 16 contraction networks and nine product workloads passed independent numerical checks. "
             "These checks validate the benchmark construction, not TEPL's semantic proofs.", "",
             "## Dot chain results", "",
             "All cases retain symbolic extents and fixed rank-three inputs. Canonical and permuted layouts use "
             "the same contraction network; explicit and automatic variants retain identical LHS attributes and conditions.", "",
             "| Layout | Attributes | Proved / 8 | Largest proved n | First failed n | First failure phase |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for _, rows, mode, layout in groups(summaries, "dot"):
        proved = [s for s in rows if s["outcome"] == "proved"]
        failures = [s for s in rows if s["outcome"] != "proved"]
        first = failures[0] if failures else None
        phase = first["diagnostic"].split(":")[0] if first else "none"
        lines.append(f"| {layout} | {mode} | {len(proved)}/8 | "
                     f"{max(s['size'] for s in proved) if proved else 'none'} | "
                     f"{first['size'] if first else 'none'} | {phase} |")
    dots = [s for s in summaries if s["family"] == "dot"]
    if not any(s["outcome"] == "proved" for s in dots):
        lines += ["", "**No dot-chain case was proved, including the smallest three-input explicit baselines.** "
                  "Small cases stop at unestablished shape-equality obligations; larger cases can exhaust the work budget "
                  "during metadata definedness. This run therefore does not establish successful dot-proof or "
                  "attribute-inference scalability. No automatic/explicit success-time ratio is defined.", "",
                  "The figure shows the time and memory consumed by checking attempts. Increasing attempt cost "
                  "is an observation about this checking path; failures can occur before RHS construction or verification.", "",
                  "A flat or falling curve among failures is not improved scalability. A larger rule may exhaust its "
                  "budget in an earlier phase and return faster than a smaller rule that progresses farther."]
    lines += ["", "![Dot chain checking costs and outcomes](dot_chain_cost.png)", ""]
    if attempt_cost:
        lines += ["Figure adaptation: the plan called for successful-proof cost curves. Because there are no successful "
                  "dot cases, Figure 1 instead shows all attempt costs and explicitly labels that distinction. "
                  "The benchmark cases, size grid, conditions, budgets, and repetition count were unchanged.", ""]
    lines += ["## Expression expansion results", "",
              "Source operation count grows linearly with k. The term column is the analytical full expansion, "
              "not an observed allocation. Failed checks do not appear on successful-cost curves.", "",
              "![Expression expansion costs and outcomes](expression_growth.png)", "",
              "| k | Analytical terms | Outcome | Median time | Time min–max | Median peak RSS | Failure phase |",
              "| ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    products = [s for s in summaries if s["family"] == "product"]
    for s in products:
        lines.append(f"| {s['size']} | {s['expected_expanded_terms']} | {s['outcome']} | "
                     f"{format_seconds(s['median_seconds'])} | {s['min_seconds']:.3f}–{s['max_seconds']:.3f} s | "
                     f"{s['median_rss_mib']:.2f} MiB | {s['diagnostic'].split(':')[0] or '—'} |")
    successes = [s for s in products if s["outcome"] == "proved"]
    failures = [s for s in products if s["outcome"] != "proved"]
    if successes:
        biggest = max(successes, key=lambda s: s["size"])
        lines += ["", f"The largest tested consistently proved expression has **k={biggest['size']}** "
                  f"({biggest['expected_expanded_terms']} analytical terms), with median checking time "
                  f"**{format_seconds(biggest['median_seconds'])}** and median peak RSS "
                  f"**{biggest['median_rss_mib']:.2f} MiB**."]
    if failures:
        first = failures[0]
        lines += ["", f"The first tested failure is **k={first['size']}**, classified as `{first['outcome']}`. "
                  f"Its diagnostic is: {first['diagnostic']}"]
    lines += ["", f"The recorded normalization cap is {p['proof_limits']['normalization_terms']} terms; "
              "the first analytical expansion exceeding it is k=9. A general-work or definedness failure at a "
              "smaller k must not be described as reaching the normalization term cap. The observed boundary "
              "is a boundary of the complete checking pipeline at fixed budgets. This sweep does not isolate "
              "exponential normalizer cost when earlier source-lowering or construction work prevents completion.", "",
              "## Every dot case", "",
              "Failed-case costs are retained here even though they cannot establish successful proof scaling.", "",
              "| Layout | Attributes | n | Outcome | Median time | Time min–max | Median peak RSS | Failure phase |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for s in dots:
        lines.append(f"| {s['layout']} | {s['attribute_mode']} | {s['size']} | {s['outcome']} | "
                     f"{format_seconds(s['median_seconds'])} | {s['min_seconds']:.3f}–{s['max_seconds']:.3f} s | "
                     f"{s['median_rss_mib']:.2f} MiB | {s['diagnostic'].split(':')[0] or '—'} |")
    lines += ["", "## Diagnostics and scope", "",
              "Diagnostic runs agreed with measured outcomes. Complete compiler output, including each unresolved "
              "obligation, is retained in the per-invocation logs. Representative first failures:", ""]
    for _, rows, _, _ in groups(summaries, "dot"):
        fail = next((s for s in rows if s["outcome"] != "proved"), None)
        if fail:
            lines += [f"- `{fail['case_id']}`: {fail['diagnostic']}"]
    lines += ["", "Each benchmark is one composite rewrite, not repeated application of a local optimization. "
              "The study does not establish asymptotic complexity, arbitrary overlapping-axis scalability, "
              "floating-point equivalence, or tensor execution speed. Fixed physical layouts also do not "
              "demonstrate arbitrary-rank constructor synthesis.", "",
              "## Measurement environment", "",
              f"- CPU: {p['cpu']}; {p['logical_cpus']} logical CPUs; serial benchmark execution.",
              f"- Platform: `{p['platform']}`.",
              f"- Build: `{p['build_command']}` using the existing default Bazel configuration "
              "(`k8-fastbuild`); this is not a release-optimized performance run.",
              f"- Revision: `{p['revision']}` with the recorded working-tree changes and per-file hashes.",
              f"- Compiler SHA-256: `{p['compiler_hash']}`.",
              f"- Start load averages: {p['load_average_at_start']}.",
              "- One warm-up and five measured fresh processes per case; deterministic shuffled order; 30 s timeout.",
              "- Peak RSS comes from GNU time for each compiler child; median and min/max summarize repetitions.",
              "- Timing includes parsing, reasoning, report serialization, redirected output, and wrapper overhead. "
              "The current CLI does not provide a quiet semantic-report mode.",
              "- No proof limits or compiler implementation were changed for this experiment.", "",
              "Recorded proof limits:", "", "```json", json.dumps(p["proof_limits"], indent=2), "```", "",
              "## Reproduction and saved evidence", "",
              "See [the plan](../README.md) for generation and measurement commands. Run analysis with NumPy and Matplotlib:", "",
              "```sh", "# In a Python environment containing requirements.txt dependencies:",
              "python3 lab/reasoning_scalability/scripts/benchmark.py --check",
              "python3 lab/reasoning_scalability/scripts/analyze.py", "```", "",
              "- [Frozen cases and validation manifest](../inputs/manifest.json)",
              "- [Raw run records and source provenance](../results/run.json)",
              "- [Case summary JSON](../results/summary.json)",
              "- [Case summary CSV](../results/summary.csv)",
              "- [All invocation records as CSV](../results/invocations.csv)",
              "- [Analysis package versions and script hash](../results/analysis_environment.json)",
              "- [Measured stdout and stderr logs](../results/logs/measured/)",
              "- Figures are provided as PNG, SVG, and PDF beside this report.", "",
              "The 41 input rules were not strengthened or filtered after the preflight failures. "
              "Preflight logs are retained separately and excluded from all measured summaries.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = json.loads((HERE / "results/run.json").read_text())
    b.validate_saved(data)
    assert data["completed"]
    verify_logs(data)
    summaries = summarize(data)
    report_dir = HERE / "reports"
    report_dir.mkdir(exist_ok=True)
    if args.check:
        assert json.loads((HERE / "results/summary.json").read_text()) == summaries
        attempt_cost = not any(s["outcome"] == "proved" for s in summaries if s["family"] == "dot")
        assert (report_dir / "results.md").read_text() == render_report(data, summaries, attempt_cost)
        print("Summary and report match raw data; all log outcomes and provenance validated.")
        return
    b.save_json(HERE / "results/summary.json", summaries)
    b.save_json(HERE / "results/analysis_environment.json", {
        "python": __import__("platform").python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "analysis_script_hash": b.digest(Path(__file__)),
    })
    with (HERE / "results/summary.csv").open("w") as stream:
        fields = [k for k in summaries[0] if k not in ("outcome_counts",)]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    with (HERE / "results/invocations.csv").open("w") as stream:
        fields = ["case_id", "stage", "repetition", "family", "size", "layout", "attribute_mode",
                  "input_nodes", "input_bytes", "elapsed_seconds", "peak_rss_mib", "outcome",
                  "exit_code", "diagnostic", "input_hash", "stdout_file", "stderr_file"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["records"])
    attempt_cost = figures(summaries, report_dir, data["provenance"]["proof_limits"]["normalization_terms"])
    (report_dir / "results.md").write_text(render_report(data, summaries, attempt_cost))
    print("Saved summaries, CSV exports, two figures (PNG/SVG/PDF), and reports/results.md.")


if __name__ == "__main__":
    main()
