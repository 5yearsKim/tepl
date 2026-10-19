"""Project paths and validation of the historical corpus after relocation."""

import copy
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
TEPL = HERE.parent / "tepl"
MIGRATION = HERE / "baselines/layout_migration.json"


def relocate(name, migration):
    """Translate a repository-relative historical path to the current layout."""
    for old, new in sorted(migration["paths"].items(), key=lambda item: -len(item[0])):
        if name == old or (old.endswith("/") and name.startswith(old)):
            return new + name[len(old):]
    return name


def relocated_manifest(original, migration):
    """Change only case file paths, preserving conditions, scope and case IDs."""
    result = copy.deepcopy(original)
    prefix = HERE.relative_to(ROOT).as_posix() + "/"
    for case in result["cases"]:
        case["file"] = Path(os.path.relpath(ROOT / relocate(prefix + case["file"], migration), HERE)).as_posix()
    return result


def validate_frozen_corpus(baseline):
    migration = json.loads(MIGRATION.read_text())
    snapshots = migration["manifest_snapshots"]
    for name, expected in baseline["corpus_hashes"].items():
        path = ROOT / snapshots.get(name, relocate(name, migration))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen reasoning corpus changed: " + name)
        if name in snapshots:
            original = json.loads(path.read_text())
            current = json.loads((ROOT / relocate(name, migration)).read_text())
            if current != relocated_manifest(original, migration):
                raise ValueError("frozen manifest changed beyond file paths: " + name)
