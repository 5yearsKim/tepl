#!/usr/bin/env python3
"""Compare XLA and TEPL in five source sections over 30 matched rules."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import support as m

HERE, LAB, ROOT = m.HERE, m.LAB, m.ROOT
RESULT = HERE / "results/section_comparison.json"
REPORT = HERE / "reports/section_comparison.md"
VERIFICATION = HERE / "results/section_verification.json"
LABELS = {
    "matching": "Matching (LHS)",
    "legality": "Legality checks (where block)",
    "attributes": "Attribute reasoning (derive block)",
    "construction": "Replacement construction (RHS)",
    "other": "Other syntax",
}
REPRESENTATIONS = ["XLA", "Explicit TEPL", "Axis inferred TEPL"]


def sections(source):
    """Count literal TEPL sections; constraints and outer syntax remain Other."""
    text = m.rule_text(source)
    rule = m.Rule(text)
    spans = [(rule.lhs.span, "matching"), (rule.rhs.span, "construction")]
    for keyword, category in [("where", "legality"), ("derive", "attributes")]:
        if keyword in rule.values:
            start = rule.values.index(keyword)
            end = rule.closing(start + 1, "{", "}")
            spans.append(((rule.tokens[start].start(), rule.tokens[end].end()), category))
    result = Counter({key: 0 for key in LABELS})
    for token in m.tokens(text):
        labels = [category for (start, end), category in spans
                  if start <= token.start() and token.end() <= end]
        if len(labels) > 1:
            raise ValueError("overlapping TEPL sections")
        result[labels[0] if labels else "other"] += 1
    if sum(result.values()) != len(m.tokens(text)):
        raise ValueError("incomplete TEPL section accounting")
    return dict(result)


def add_parts(rows, variant):
    result = Counter({key: 0 for key in LABELS})
    for row in rows:
        result.update(row[variant]["parts"])
    return dict(result)


def representations(original, canonical, name):
    # The companion supplies its shape with a supported expression; restore its
    # handwritten axis mapping using the already supplied local d.
    if name == "catalog_broadcast_transpose_broadcast":
        original = canonical.replace("dimensions: auto", "dimensions: d")
    pair = m.matched_pair(original)
    if m.values(pair["inferred"]) != m.values(canonical):
        raise ValueError("matched migration differs from successful task: " + name)
    compact = m.compact_source(canonical)
    if "auto" in m.values(compact):
        raise ValueError("explicit auto remains in compact inferred source")
    if sections(pair["explicit"])["legality"] != sections(compact)["legality"]:
        raise ValueError("compaction changed conditions")
    return pair["explicit"], compact, pair["fields"]


def collect():
    evaluation, selected = m.load_successful_tasks()
    manifest_path = HERE / "data/xla_spans.json"
    manifest = json.loads(manifest_path.read_text())
    audit = {r["case"]: r for r in manifest["cases"]}
    if len(audit) != 31 or set(audit) != {r["case"] for r in selected}:
        raise ValueError("XLA ledger must cover the 31 selected cases exactly once")
    cases = []
    for row in selected:
        name = row["case"]
        source_path = (LAB / "coverage" / row["source_file"]).resolve()
        if m.sha(source_path) != row["source_sha256"]:
            raise ValueError("original explicit source changed: " + name)
        source = source_path.read_text()
        canonical = (ROOT / row["file"]).read_text()
        explicit_source, inferred_source, fields = representations(source, canonical, name)
        inferred = sections(inferred_source)
        if inferred["legality"] != sections(canonical)["legality"]:
            raise ValueError("where subtotal differs")
        cases.append(dict(case=name,
                          explicit=dict(original_file=str(source_path.relative_to(ROOT)),
                                        original_file_sha256=m.sha(source_path), source=explicit_source,
                                        parts=sections(explicit_source)),
                          inferred=dict(input_file=row["file"], input_file_sha256=row["file_sha256"],
                                        source=inferred_source, parts=inferred), fields=fields,
                          xla_mapping=audit[name]))
    matched = [r for r in cases if r["xla_mapping"]["status"] == "reviewed"]
    if len(cases) != 31 or len(matched) != 30:
        raise ValueError("expected 31 TEPL rules and 30 corresponding XLA mappings")
    xla, listing = m.xla_parts(manifest, {r["case"] for r in matched})
    xla_primary = {key: xla.get("management" if key == "other" else key, 0) for key in LABELS}
    dependencies = [Path(__file__).resolve(), HERE / "protocol.md", manifest_path,
                    m.EVALUATION, HERE / "scripts/support.py",
                    LAB / "coverage/scripts/synthesis_cases.py",
                    LAB / "coverage/scripts/axis_fields.py",
                    LAB / "coverage/scripts/hlo_support.py",
                    LAB / "coverage/scripts/paths.py"]
    dependencies += list((LAB / "tepl/dialects").glob("*.tepl"))
    dependencies += [LAB / "catalog/upstream" / name for name in manifest["source_hashes"]]
    return dict(protocol="xla_tepl_literal_sections_v2", cases=cases,
                matched_cases=[r["case"] for r in matched],
                matched=dict(xla=xla_primary, explicit=add_parts(matched, "explicit"),
                             inferred=add_parts(matched, "inferred")),
                all_tepl=dict(explicit=add_parts(cases, "explicit"), inferred=add_parts(cases, "inferred")),
                xla_excluded=dict(policy=xla.get("policy", 0), logging=xla.get("logging", 0)),
                evaluation_run_at_utc=evaluation["run_at_utc"],
                compiler_sha256=evaluation["compiler_sha256"],
                dependencies={str(p.relative_to(ROOT)): m.sha(p) for p in dependencies}), listing


def source_digest(data):
    sources = {r["case"]: {v: r[v]["source"] for v in ["explicit", "inferred"]} for r in data["cases"]}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def verify_representations(data, compiler, jobs, timeout):
    compiler = compiler.resolve(strict=True)
    compiler_hash = m.sha(compiler)
    dialects = {str(p.relative_to(ROOT)): m.sha(p) for p in (LAB / "tepl/dialects").glob("*.tepl")}
    with tempfile.TemporaryDirectory(prefix="tepl-section-size-") as directory:
        work = Path(directory)
        shutil.copytree(LAB / "tepl/dialects", work / "dialects")
        shutil.copyfile(LAB / "tepl/tepl.toml", work / "tepl.toml")
        def worker(row):
            variants = {v: m.verify(row[v]["source"], row["case"], v, compiler, timeout, work, row["fields"])
                        for v in ["explicit", "inferred"]}
            print(row["case"] + ": " + ", ".join(v + "=" + str(r["accepted"]) for v, r in variants.items()), flush=True)
            return dict(case=row["case"], variants=variants)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(worker, data["cases"]))
    if m.sha(compiler) != compiler_hash or any(m.sha(ROOT / p) != sha for p, sha in dialects.items()):
        raise ValueError("verification inputs changed")
    evidence = dict(source_digest=source_digest(data), run_at_utc=datetime.now(timezone.utc).isoformat(),
                    compiler_sha256=compiler_hash, dialect_hashes=dialects,
                    jobs=jobs, timeout_seconds=timeout, cases=results)
    VERIFICATION.write_text(json.dumps(evidence, indent=2) + "\n")


def attach_verification(data):
    evidence = json.loads(VERIFICATION.read_text())
    if evidence["source_digest"] != source_digest(data):
        raise ValueError("verification does not match measured representations; use --run")
    expected = {r["case"]: r for r in data["cases"]}
    if {r["case"] for r in evidence["cases"]} != set(expected):
        raise ValueError("verification case inventory differs")
    for row in evidence["cases"]:
        for variant, result in row["variants"].items():
            count = len({f["node"] for f in expected[row["case"]]["fields"]}) if variant == "inferred" else 0
            if result["accepted"] != m.verified(result, count):
                raise ValueError("invalid verification evidence")
    inferred_passes = sum(r["variants"]["inferred"]["accepted"] for r in evidence["cases"])
    if inferred_passes != 31:
        raise ValueError("not all compact inferred representations verify")
    data["verification"] = dict(run_at_utc=evidence["run_at_utc"], compiler_sha256=evidence["compiler_sha256"],
                                explicit_accepted=sum(r["variants"]["explicit"]["accepted"] for r in evidence["cases"]),
                                inferred_accepted=inferred_passes)
    data["dependencies"][str(VERIFICATION.relative_to(ROOT))] = m.sha(VERIFICATION)


def table(columns, data):
    out = ["| Section | " + " | ".join(columns.values()) + " |",
           "|---|" + "---:|" * len(columns)]
    for key, label in LABELS.items():
        out.append("| " + label + " | " + " | ".join(f"{data[c][key]:,}" for c in columns) + " |")
    out.append("| **Total** | " + " | ".join(f"**{sum(data[c].values()):,}**" for c in columns) + " |")
    return out


def render(data):
    out = ["# XLA, explicit TEPL, and axis inferred TEPL by source section", "",
           "The graph uses **30 matched rules** from the 31 successful synthesis cases. "
           "The additional `catalog_dot_scalar_multipliers` rule has no corresponding XLA dot implementation "
           "in the pinned snapshot and appears only in the separate **31-rule TEPL table** below. "
           "Explicit TEPL spells out the RHS axis transformations in `derive`. Axis inferred TEPL omits "
           "inferable axis fields, removes unused node labels, and removes empty assignments/derive blocks. "
           "Supplied non-axis attributes and their required labels/helpers remain. Both use the same "
           "normalized graph and `where` conditions (including the successful broadcast–transpose companion).", "",
           "![Token comparison for 30 matched rules](section_comparison.png)", "",
           "## Thirty-rule comparison", ""]
    out += table(dict(zip(["xla", "explicit", "inferred"], REPRESENTATIONS)), data["matched"])
    out += ["", "## All 31 successful cases: TEPL source sizes", ""]
    out += table({"explicit": "Explicit TEPL", "inferred": "Axis inferred TEPL"}, data["all_tepl"])
    out += ["", "## Counting rules", "",
            "TEPL rows follow literal source sections: complete LHS, complete `where` block, complete `derive` "
            "block, complete RHS, and all remaining rule tokens. Block keywords/braces and all local computations "
            "are included. Inline descriptors stay with their containing graph; tensor shape/type declarations, "
            "the arrow, and outer rule syntax are in **Other syntax**. No token is counted twice. "
            "Imports/use declarations and shared dialect implementations are outside whole-rule totals; "
            "comments and whitespace count zero. These are lexical tokens, not LLM tokens.", "",
            "XLA has no LHS/where/derive/RHS blocks. Its selected source lines are assigned the corresponding "
            "responsibilities by inspection: matching, legality, attribute computations, replacement construction, "
            "and other code (including graph management). Control syntax follows the dominant responsibility "
            "of its source line. Shared excerpts/helpers are counted once across rules. "
            f"Policy ({data['xla_excluded']['policy']:,} tokens) and logging "
            f"({data['xla_excluded']['logging']:,} tokens) are recorded separately and excluded.", "",
            "## Interpretation and evidence", "",
            f"A fresh check of the measured representations accepts **{data['verification']['inferred_accepted']}/31 "
            f"compact inferred rules** and **{data['verification']['explicit_accepted']}/31 explicit rules** "
            "through proof, compilation, and JSON validation. Explicit rules with unresolved proofs remain "
            "countable source and are included, with diagnostics in the verification JSON. "
            "Explicit fields are restored from handwritten transformations after the same unit-view normalization; "
            "the companion uses its existing local `d` as its explicit broadcast mapping. "
            "Omission, label removal, and empty-block removal all contribute to the measured reduction. XLA often handles "
            "more ranks and operand orders; equal case counts do not establish equal behavioral breadth. "
            "The annotations have one analyst's review and no independent review.", "",
            "[SVG graph](section_comparison.svg), [LaTeX tables](section_comparison.tex), "
            "[measurements and source snapshots](../results/section_comparison.json), "
            "[new representation verification](../results/section_verification.json), "
            "[annotated XLA excerpts](section_xla_excerpts.txt), "
            "[source annotation ledger](../data/xla_spans.json), and [protocol](../protocol.md).", "",
            f"Recorded synthesis run UTC: `{data['evaluation_run_at_utc']}`. "
            f"Original evaluation compiler SHA-256: `{data['compiler_sha256']}`. "
            f"Representation verification UTC: `{data['verification']['run_at_utc']}`; "
            f"compiler SHA-256: `{data['verification']['compiler_sha256']}`. Frozen evaluation results remain unchanged.", ""]
    tex = []
    for caption, columns, counts in [
            ("30 matched rules", dict(zip(["xla", "explicit", "inferred"], REPRESENTATIONS)), data["matched"]),
            ("All 31 successful cases", {"explicit": "Explicit TEPL", "inferred": "Axis inferred TEPL"}, data["all_tepl"])]:
        tex += ["% " + caption, "\\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule",
                "Section & " + " & ".join(columns.values()) + r" \\", r"\midrule"]
        for key, label in LABELS.items():
            tex.append(label + " & " + " & ".join(f"{counts[c][key]:,}" for c in columns) + r" \\")
        tex += [r"\midrule", "Total & " + " & ".join(f"{sum(counts[c].values()):,}" for c in columns) + r" \\",
                r"\bottomrule", r"\end{tabular}", ""]
    return {REPORT: "\n".join(out), HERE / "reports/section_comparison.tex": "\n".join(tex)}


def graph(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#4477AA", "#228833", "#EEAA33", "#AA3377", "#AAB3BB"]
    counts = data["matched"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(12, 9.2), facecolor="white")
        ax = fig.add_axes([.10, .46, .57, .43])
        bottom = [0, 0, 0]
        columns = ["xla", "explicit", "inferred"]
        for (key, label), color in zip(LABELS.items(), colors):
            values = [counts[c][key] for c in columns]
            ax.bar(range(3), values, bottom=bottom, color=color, width=.56,
                   label=label, edgecolor="white", linewidth=.6)
            for i, value in enumerate(values):
                if value >= 350:
                    ax.text(i, bottom[i] + value / 2, f"{value:,}", ha="center", va="center", fontsize=10,
                            color="white" if key in {"matching", "legality", "construction"} else "#17212B")
            bottom = [a + b for a, b in zip(bottom, values)]
        for i, total in enumerate(bottom):
            ax.text(i, total + max(bottom) * .025, f"{total:,}", ha="center", weight="bold", fontsize=13)
        ax.set_xticks(range(3), REPRESENTATIONS)
        ax.set_ylabel("Lexical tokens")
        ax.set_ylim(0, max(bottom) * 1.14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E2E6EA", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, .98), frameon=False, fontsize=10)
        fig.text(.10, .955, "Rewrite source size by section", fontsize=21, weight="bold", color="#17212B")
        fig.text(.10, .918, "30 matched rules · complete rule tokens, including where and derive", fontsize=12, color="#536171")
        table_ax = fig.add_axes([.06, .12, .90, .27])
        table_ax.axis("off")
        cells = [[label] + [f"{counts[c][key]:,}" for c in columns] for key, label in LABELS.items()]
        cells += [["Total"] + [f"{sum(counts[c].values()):,}" for c in columns]]
        tab = table_ax.table(cellText=cells, colLabels=["Section"] + REPRESENTATIONS,
                             colWidths=[.46, .14, .19, .21], cellLoc="right", loc="center", bbox=[0, 0, 1, 1])
        tab.auto_set_font_size(False)
        tab.set_fontsize(11)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#DEE3E8")
            cell.set_linewidth(.6)
            cell.PAD = .08
            if c == 0:
                cell.set_text_props(ha="left")
            if r == 0:
                cell.set_facecolor("#EDF1F5")
                cell.set_text_props(weight="bold")
            elif r == 6:
                cell.set_facecolor("#F2F5F8")
                cell.set_text_props(weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#FAFBFC")
        fig.text(.06, .068, "Explicit TEPL: axis transformations written in derive. Inferred TEPL: axis fields and unused labels/blocks omitted.",
                 fontsize=9, color="#536171")
        fig.text(.06, .043, "XLA: shared excerpts counted once; policy/logging excluded; broader behavior possible. All 31 TEPL rules are tabulated in the report.",
                 fontsize=9, color="#536171")
        for suffix in ["png", "svg"]:
            fig.savefig(HERE / ("reports/section_comparison." + suffix), dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check counts, hashes, and text reports without writing")
    mode.add_argument("--run", action="store_true", help="verify the explicit and compact inferred representations")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.jobs <= 0 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    data, listing = collect()
    if args.run:
        verify_representations(data, args.compiler, args.jobs, args.timeout)
    attach_verification(data)
    outputs = render(data)
    outputs[RESULT] = json.dumps(data, indent=2) + "\n"
    outputs[HERE / "reports/section_xla_excerpts.txt"] = listing
    for path, text in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise ValueError("stale section comparison: " + str(path))
        else:
            path.write_text(text)
    if not args.check:
        graph(data)
    print("30 matched rules and all 31 TEPL rules: section counts checked")


if __name__ == "__main__":
    main()
