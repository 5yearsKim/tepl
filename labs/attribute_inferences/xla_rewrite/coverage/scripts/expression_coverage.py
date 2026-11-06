#!/usr/bin/env python3
"""Check explicit rule expression and report language coverage separately from proof."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import hlo_support
    from .paths import HERE, ROOT, TEPL, repo_path
else:
    import hlo_support
    from paths import HERE, ROOT, TEPL, repo_path
STATUSES = ("expressible", "needs_extension", "partial", "nonlocal")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(compiler, timeout):
    cases = read(HERE / "data/compilation_cases.json")["cases"]
    results = []
    for case in cases:
        path = (HERE / case["file"]).resolve()
        require(
            not re.search(r"\bauto\b", path.read_text()),
            f"explicit case contains auto: {case['name']}",
        )
        started = time.perf_counter()
        artifact_hash, validated = None, False
        try:
            with tempfile.TemporaryDirectory(prefix="tepl-xla-coverage-") as directory:
                artifact = Path(directory) / "program.json"
                result = subprocess.run(
                    [
                        str(compiler),
                        "compile",
                        str(path.relative_to(ROOT)),
                        "-o",
                        str(artifact),
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    artifact_hash = sha(artifact)
                    remaining = timeout - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(
                            "compiled JSON validation", timeout
                        )
                    result = subprocess.run(
                        [str(compiler), "validate", str(artifact)],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        timeout=remaining,
                    )
                    validated = result.returncode == 0
                status = "compiled" if validated else "not_compiled"
                returncode, diagnostics = result.returncode, result.stderr.strip()
        except subprocess.TimeoutExpired:
            status, returncode, diagnostics = "timeout", None, "wall-clock timeout"
        results.append(
            dict(
                case=case["name"],
                catalog_id=case["catalog_id"],
                status=status,
                returncode=returncode,
                seconds=round(time.perf_counter() - started, 6),
                file_sha256=sha(path),
                artifact_sha256=artifact_hash,
                validated_json=validated,
                diagnostics=diagnostics,
            )
        )
        print(f"{case['name']}: {status}", flush=True)
    dependencies = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            HERE / "data/compilation_cases.json",
            HERE / "scripts/expression_coverage.py",
            TEPL / "tepl.toml",
            TEPL / "dialects/hlo_structural.tepl",
            HERE.parent / "catalog/rules.json",
        )
    }
    data = dict(
        schema_version=1,
        tepl_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        compiler_sha256=sha(compiler),
        dependencies=dependencies,
        environment=platform.platform(),
        run_at_utc=datetime.now(timezone.utc).isoformat(),
        timeout_seconds_per_case=timeout,
        results=results,
    )
    (HERE / "results/compilation.json").write_text(json.dumps(data, indent=2) + "\n")


def validate(catalog, assessment, manifest, measured):
    hlo_support.validate_support()
    ids = {r["id"] for r in catalog}
    assessed = [r["id"] for r in assessment["rules"]]
    require(
        len(assessed) == len(set(assessed)) and set(assessed) == ids,
        "language assessment must cover each catalog schema once",
    )
    require(
        assessment["xla_revision"]
        == read(HERE.parent / "catalog/source_manifest.json")["revision"],
        "XLA revision mismatch",
    )
    for row in assessment["rules"]:
        require(
            row["status"] in STATUSES and row["reason"],
            "language assessment needs a known status and reason",
        )
        if row["status"] == "needs_extension":
            require(
                row["integration"],
                "extension assessment needs an integration requirement",
            )
    for evidence in assessment["evidence"]:
        require(repo_path(evidence).is_file(), f"missing evidence: {evidence}")
    cases = {c["name"]: c for c in manifest["cases"]}
    require(len(cases) == len(manifest["cases"]), "duplicate expression cases")
    require(
        {(HERE / c["file"]).resolve() for c in cases.values()}
        == {p.resolve() for p in (TEPL / "rules/structural").glob("*.tepl")},
        "unlisted expression case files",
    )
    for case in cases.values():
        require(
            case["catalog_id"] in ids and case["scope"],
            "expression case needs catalog ID and scope",
        )
        require(
            not re.search(r"\bauto\b", (HERE / case["file"]).read_text()),
            "expression cases must supply attributes explicitly",
        )
        require(
            'from "dialects/hlo_structural.tepl" import HLO as hlo'
            in (HERE / case["file"]).read_text(),
            "expression cases must use the local structural HLO profile",
        )
    target_set = manifest.get("designated_targets")
    if target_set:
        target_ids = target_set["target_ids"]
        require(
            len(target_ids) == len(set(target_ids)), "duplicate designated target IDs"
        )
        require(
            set(target_ids) == set(target_set["cases"]),
            "designated targets must implement every target schema",
        )
        require(set(target_ids) <= ids, "unknown designated target schema")
        for cid, name in target_set["cases"].items():
            require(
                name in cases and cases[name]["catalog_id"] == cid,
                "designated case must implement its catalog schema",
            )
    results = measured["results"]
    names = [r["case"] for r in results]
    require(
        len(names) == len(set(names)) and set(names) == set(cases),
        "expression measurements must cover each case once",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", measured["tepl_revision"]) is not None,
        "invalid measured TEPL revision",
    )
    for result in results:
        case = cases[result["case"]]
        require(result["catalog_id"] == case["catalog_id"], "case ID mismatch")
        require(
            result["file_sha256"] == sha(HERE / case["file"]),
            "stale expression measurement; rerun expression_coverage.py --run",
        )
        require(
            result["status"] in {"compiled", "not_compiled", "timeout"},
            "expression acceptance must not be recorded as semantic proof",
        )
        if result["status"] == "compiled":
            require(result["returncode"] == 0, "compiled case has unsuccessful exit")
            require(
                result.get("validated_json") and result.get("artifact_sha256"),
                "compiled case must have a validated executable JSON artifact",
            )
    for path, expected in measured["dependencies"].items():
        require(sha(repo_path(path)) == expected, "stale expression dependency: " + path)


def clean(text):
    return text.replace("|", "\\|").replace("\n", " ")


def render(catalog, assessment, manifest, measured):
    by_id = {r["id"]: r for r in assessment["rules"]}
    counts = Counter(r["status"] for r in assessment["rules"])
    results = measured["results"]
    compiled = [r for r in results if r["status"] == "compiled"]
    compiled_ids = {r["catalog_id"] for r in compiled}
    targets = manifest.get("designated_targets", {"target_ids": [], "cases": {}})
    compiled_names = {r["case"] for r in compiled}
    targets_passed = sum(name in compiled_names for name in targets["cases"].values())
    out = [
        "# XLA Rewrite Coverage in TEPL",
        "",
        "This report evaluates whether XLA's algebraic rewrites can be written "
        "using TEPL dialect declarations, graph patterns, conditions and explicit "
        "RHS attributes. Automatic value-equivalence checking is reported separately.",
        "",
        "## Result",
        "",
        f"The language assessment covers **{len(catalog)} catalog schemas**:",
        "",
        f"- **{counts['expressible']}** have algebraic structures and metadata "
        "constructions expressible through ordinary rules and dialect declarations.",
        f"- **{counts['needs_extension']}** require analysis facts or tensor-literal "
        "handling not currently exposed to rules. These are extension requirements, "
        "not supported general schemas. Some concrete cases can instead use "
        "readable constant attributes, literal patterns or dtype conditions; "
        "see the [host-dependency audit](host_dependency_audit.md).",
        f"- **{counts['partial']}** are expressible as concrete arities, extents "
        "or single-result cases; their fully general schemas need additional adaptation.",
        f"- **{counts['nonlocal']}** reuse a separately existing graph value and "
        "need graph lookup or nonlocal matching support that is not currently available.",
        "",
        "These are manual expressibility assessments. "
        "They identify which language mechanisms can represent a schema. "
        "Concrete implementations are measured below.",
        "",
        f"The structural corpus includes a designated rule for each of "
        f"the **{len(targets['target_ids'])} originally expressible schemas**. "
        f"**{targets_passed}/{len(targets['target_ids'])} designated rules compile**. "
        "Each rule has an explicit catalog ID, RHS metadata and scope. "
        "Some implementations select a fixed rank, opcode or padding domain; "
        "their compilation establishes those written cases.",
        "",
        f"Including retained pilot variants, compilation accepted **{len(compiled)} of {len(results)} "
        f"explicit cases**, spanning **{len(compiled_ids)} catalog schemas**. "
        "Each case supplies its RHS attributes. "
        "Value-equivalence results are in the separate "
        "[axis-only evaluation](attribute_synthesis.md).",
        "",
        "Both measurements use the self-contained HLO project configured by "
        "[tepl.toml](../../tepl/tepl.toml). Compilation uses "
        "[HLO declarations without value semantics](../../tepl/dialects/hlo_structural.tepl); "
        "axis-only verification uses migrated rules with the local "
        "[semantic HLO profile](../../tepl/dialects/hlo.tepl). Every compilation case has "
        "a source counterpart; scope exclusions and unresolved cases are "
        "reported separately. Imports resolve from the project root.",
        "",
        "## What counts as coverage",
        "",
        "A missing built-in operation or semantic lowering does not prevent "
        "an algebraic rule from being expressed. For example, a dialect can "
        "declare divide, exp, reshape, pad or dynamic slice; a rule can match "
        "and rebuild those operations without the semantic normalizer evaluating them.",
        "",
        "The evaluation follows the catalog's algebraic scope. XLA visitor "
        "plumbing, graph-mutation APIs, backend profitability and target-specific "
        "numerical policy do not reduce language coverage. Conditions that "
        "determine when the algebraic equation applies remain part of the rule "
        "specification. Host calls and host `fn` parameters are unsupported; "
        "source facts or literal handling that need those capabilities are recorded "
        "as extension requirements, not assumed to be available.",
        "",
        "Three outcomes are recorded independently:",
        "",
        "1. **Expressible:** the rule, its metadata construction and relevant "
        "source requirements have a language-level representation.",
        "2. **Compiled:** a concrete explicit rule passes `tepl compile` and its "
        "executable JSON passes `tepl validate`. Some cases specialize ranks, literals or arities; "
        "their scope is recorded in the [case manifest](../data/compilation_cases.json). "
        "Compilation checks syntax, types and construction, "
        "and does not establish that every source condition has been ported.",
        "3. **Verified:** the optional semantic checking path additionally proves "
        "the written value/coordinate equivalence under its arithmetic policy.",
        "",
        "## Coverage by category",
        "",
        "| Category | Catalog | Rule + dialect | Needs extension | Concrete cases / partial | Nonlocal | Designated compiled / targeted | Compiled cases | Schemas with a compiled case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in dict.fromkeys(r["category"] for r in catalog):
        ids = {r["id"] for r in catalog if r["category"] == category}
        c = Counter(by_id[cid]["status"] for cid in ids)
        tested = [r for r in compiled if r["catalog_id"] in ids]
        target_names = [name for cid, name in targets["cases"].items() if cid in ids]
        out.append(
            f"| {category} | {len(ids)} | "
            + " | ".join(str(c[s]) for s in STATUSES)
            + f" | {sum(name in compiled_names for name in target_names)}/{len(target_names)}"
            + f" | {len(tested)} | {len({r['catalog_id'] for r in tested})}"
            + " |"
        )
    out.extend(
        [
            "",
            "## Examples demonstrated by compilation",
            "",
            "The pilot includes polynomial rules and explicit dot attributes, "
            "division/remainder identities, logical operations, exp/log, min/max, "
            "complex projections, reshape composition, concatenate, dynamic "
            "update identities, padding, scatter/gather, reduction windows, "
            "and convolution attribute transformations.",
            "",
            "It also includes general metadata constructions for reverse "
            "symmetric difference, unit-stride slice composition, reflected "
            "slice starts and lifted reduction dimensions. These use existing "
            "lists, indexing, comprehensions and descriptor constructors.",
            "",
            "Per-case scope, catalog IDs and source files are recorded in the "
            "[compilation manifest](../data/compilation_cases.json); "
            "[compilation.json](../results/compilation.json) records every outcome.",
        ]
    )
    failed = [r for r in results if r["status"] != "compiled"]
    if failed:
        out.extend(["", "Unsuccessful expression checks:", ""])
        out.extend(f"- **{r['case']}:** {clean(r['diagnostics'])}" for r in failed)
    out.extend(
        [
            "",
            "## Execution checks",
            "",
            "[test_rule_execution.py](../tests/test_rule_execution.py) runs eight "
            "representative compiled rewrites and compares their inputs "
            "and outputs using independent dense array meanings. The cases "
            "cover reverse composition, slice composition, signed padding "
            "converted to a crop, symbolic replication counts, pointwise "
            "convolution to dot, convolution operand swapping with reversal, "
            "a splat exponent matched through attributes, and unsigned absolute value.",
            "",
            "The tests require the selected rule to fire. They then compile "
            "the exported replacement graph and compare its shape and values "
            "with the original graph. These concrete checks supplement the "
            "compilation measurements; universal proofs remain in the separate "
            "axis-only attribute-inference report.",
            "",
            "### Dialect and implementation scope",
            "",
            "Dynamic slices/updates use scalar start operands; fixed-rank "
            "cases spell out those operands. Reduction regions are forwarded "
            "as handles. The extended convolution schema exposes dimension "
            "numbers, grouping, window parameters, precision and output shape.",
            "",
            "Padding arithmetic uses a nonnegative-edge padding declaration "
            "where it combines with index metadata. A separate signed-pad "
            "rule demonstrates negative padding. Window and extended "
            "convolution cases use nonnegative edge metadata. Constant "
            "construction supplies payload, element type and shape explicitly.",
            "",
            "The [case manifest](../data/compilation_cases.json) records each targets's scope. "
            "Fixed-rank/opcode/padding instances demonstrate a catalog schema "
            "without asserting that every conditional variant is implemented.",
        ]
    )
    out.extend(
        [
            "",
            "## Evidence and reproduction",
            "",
            "[language_assessments.json](../data/language_assessments.json) retains "
            "all per-schema assessments and integration requirements. Representative "
            "gaps include nonnegativity analysis, opaque constant payloads, variable "
            "arities and nonlocal graph-value reuse; see the "
            "[host-dependency audit](host_dependency_audit.md).",
            "",
            f"TEPL revision: `{measured['tepl_revision']}`. "
            f"XLA revision: `{assessment['xla_revision']}`. "
            "[compilation.json](../results/compilation.json) records input, dialect, "
            "catalog and runner hashes. Follow the "
            "[reproduction commands](../README.md#reproduce) to rerun the evaluation.",
            "",
        ]
    )
    return "\n".join(out)


def generate(check=False):
    catalog = read(HERE.parent / "catalog/rules.json")["rules"]
    assessment = read(HERE / "data/language_assessments.json")
    manifest = read(HERE / "data/compilation_cases.json")
    measured = read(HERE / "results/compilation.json")
    validate(catalog, assessment, manifest, measured)
    report = render(catalog, assessment, manifest, measured)
    destination = HERE / "reports/compilation.md"
    if check:
        require(destination.read_text() == report, "language coverage report is stale")
    else:
        destination.write_text(report)
    print(
        f"Validated {len(catalog)} language assessments and "
        f"{len(manifest['cases'])} expression cases."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compiler", type=Path, default=ROOT / "bazel-bin/tepl")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    try:
        if args.run:
            run(args.compiler.resolve(strict=True), args.timeout)
        generate(args.check)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
