"""Make coverage helpers available during unittest discovery from any directory."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scripts import hlo_support, paths

TEPL = paths.TEPL


def measurement_fixture():
    """Small synthetic measurement for validator tests; no recorded run needed."""
    import copy
    import os
    from scripts import run_synthesis

    source = TEPL / "rules/axis/transpose_compose.tepl"
    source_hash = run_synthesis.sha(source)
    dependency = HERE / "scripts/run_synthesis.py"
    dependencies = {str(dependency.relative_to(paths.ROOT)): run_synthesis.sha(dependency)}
    case = dict(
        case="synthetic_transpose", control=False, in_scope=True,
        task="axis_inference", source_file=os.path.relpath(source, HERE),
        source_sha256=source_hash, file=str(source.relative_to(paths.ROOT)),
        file_sha256=source_hash,
        axis_fields=[dict(node="out", field="permutation", forced=False)],
        normalizations=[],
    )
    outside = dict(case, case="synthetic_outside", in_scope=False,
                   task="outside_scope", axis_fields=[])
    classification = dict(protocol="axis_only_v1", dependencies=dependencies,
                          cases=[case, outside])
    success = dict(copy.deepcopy(case), outcome="inferred",
                   check=dict(returncode=0, stdout="status=proved\nadditional-guard: true\n"),
                   compile=dict(returncode=0), validation=dict(returncode=0),
                   validated_json=True, artifact_sha256="a" * 64,
                   inferred_descriptors=1, requested_axis_fields=1)
    data = dict(protocol="axis_only_v1", dependencies=dependencies,
                results=[success, dict(copy.deepcopy(outside), outcome="outside_scope")],
                run_at_utc="synthetic", compiler_sha256="0" * 64,
                timeout_seconds_per_command=1, jobs=1)
    return data, classification
