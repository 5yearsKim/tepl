#!/usr/bin/env python3
"""Install opt-in hooks at the pinned workload's AlgebraicSimplifier branches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import branch_hooks

HERE = Path(__file__).resolve().parent
SOURCE = Path("xla/hlo/transforms/simplifiers/algebraic_simplifier.cc")
MARKER = "// TEPL algebraic adapter v1"
GENERATED = ("driver.cc", "bridge.h", "branch_builder.cc", "branch_builder.h", "pipeline_driver.cc", "xla.BUILD.bazel")


def instrumented_base():
    counter = HERE.parent / "xla_workload_count/counter"
    with tempfile.TemporaryDirectory(prefix="tepl-xla-source-") as directory:
        root = Path(directory)
        target = root / SOURCE
        target.parent.mkdir(parents=True)
        shutil.copyfile(counter / "algebraic_simplifier.original.cc", target)
        research = root / "research/rewrite_count"
        research.mkdir(parents=True)
        shutil.copyfile(counter / "instrument.py", research / "instrument.py")
        subprocess.run([sys.executable, str(research / "instrument.py")], check=True, capture_output=True)
        return target.read_text()


def patch_source(source):
    if MARKER in source:
        return source
    anchor = "absl::Status AlgebraicSimplifierVisitor::HandleTranspose("
    if source.count(anchor) != 1:
        raise ValueError("unexpected HandleTranspose definition")
    block = '''// TEPL algebraic adapter v1
using TeplAlgebraicHook = absl::StatusOr<HloInstruction*> (*)(HloInstruction*, int);
namespace { TeplAlgebraicHook tepl_algebraic_hook = nullptr; }
void SetTeplAlgebraicHook(TeplAlgebraicHook hook) { tepl_algebraic_hook = hook; }

'''
    source = source.replace(anchor, block + anchor, 1)
    for site in (10114, 10118):
        anchor = f"    return ObserveAlgebraicRewrite({site}, __func__, [&]() {{ return "
        if source.count(anchor) != 1:
            raise ValueError(f"missing instrumented rewrite branch {site}")
        hook = f'''    if (tepl_algebraic_hook) {{
      ABSL_ASSIGN_OR_RETURN(HloInstruction* replacement,
                            tepl_algebraic_hook(transpose, {site}));
      if (replacement != nullptr) {{
        return ObserveAlgebraicRewrite({site}, __func__, [&]() {{
          return ReplaceInstruction(transpose, replacement);
        }});
      }}
    }}
'''
        source = source.replace(anchor, hook + anchor, 1)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xla_root", type=Path)
    args = parser.parse_args()
    path = args.xla_root / SOURCE
    original = path.read_text()
    registry = json.loads((HERE.parent / "xla_workload_count/support-registry.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.xla_root, text=True).strip()
    if revision != registry["xla_revision"]:
        raise ValueError("XLA revision differs from the pinned integration revision")
    out = args.xla_root / "research/tepl_algebraic"
    stamp = out / "prepared.json"
    previous = json.loads(stamp.read_text()) if stamp.exists() else {}
    expected = registry["instrumented_source_sha256"]
    # Reconstruct the exact original on an idempotent invocation too. Local
    # edits outside the owned hook are never overwritten.
    if MARKER not in original and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("source differs from pinned instrumented XLA; preserve local changes")
    if MARKER in original and hashlib.sha256(path.read_bytes()).hexdigest() != previous.get("source_sha256"):
        raise ValueError("prepared XLA source changed; preserve local changes")
    baseline = instrumented_base()
    if hashlib.sha256(baseline.encode()).hexdigest() != expected:
        raise ValueError("instrumentation generator differs from the pinned source")
    patched = branch_hooks.apply(patch_source(baseline))
    for name in GENERATED:
        dest = out / ("BUILD.bazel" if name == "xla.BUILD.bazel" else name)
        data = (HERE / name).read_bytes()
        if (dest.exists() and dest.read_bytes() != data
                and hashlib.sha256(dest.read_bytes()).hexdigest() != previous.get("files", {}).get(dest.name)):
            raise ValueError("conflicting generated file: " + str(dest))
    # Validate all inputs before changing either checkout.
    out.mkdir(parents=True, exist_ok=True)
    for name in GENERATED:
        shutil.copyfile(HERE / name, out / ("BUILD.bazel" if name == "xla.BUILD.bazel" else name))
    path.write_text(patched)
    stamp.write_text(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    xla_revision=revision,
                                    files={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                           for name in ("driver.cc", "bridge.h", "branch_builder.cc", "branch_builder.h", "pipeline_driver.cc", "BUILD.bazel")}), indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
