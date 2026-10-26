#!/usr/bin/env python3
"""Migrate legacy descriptor bindings without changing formulas or conditions."""

import argparse
import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .paths import HERE, ROOT, TEPL
else:
    from paths import HERE, ROOT, TEPL

RECORD = HERE / "data/syntax_migration.json"
OPERATOR = re.compile(r"\((\w+(?:\.\w+)?)\s*\[\s*@(\w+)\s*\]")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def migrate(source):
    if not re.search(r"@\w+", re.sub(r"//[^\n]*", "", source)):
        return source
    arrow = source.index("=>")
    labels, replacements = {}, []
    for match in OPERATOR.finditer(source):
        operation, descriptor = match.groups()
        if descriptor not in labels:
            label = "attr_" + descriptor
            while re.search(r"\b" + re.escape(label) + r"\b", source):
                label += "_"
            labels[descriptor] = label
            replacement = f"({label}: {operation}"
        else:
            replacement = f"({operation}[{labels[descriptor]}.attrs]"
        replacements.append((match.start(), match.end(), replacement))
    # All first occurrences are source captures or explicit RHS providers.
    for start, end, replacement in reversed(replacements):
        source = source[:start] + replacement + source[end:]
    def descriptor(match):
        name = match[1]
        if name not in labels:
            raise ValueError("descriptor has no operation provider: @" + name)
        return labels[name] + ".attrs"
    source = re.sub(r"@(\w+)", descriptor, source)
    return source


def validate():
    record = json.loads(RECORD.read_text())
    for row in record["files"]:
        original = (ROOT / row["snapshot"]).read_bytes()
        current = (ROOT / row["file"]).read_bytes()
        if digest(original) != row["original_sha256"]:
            raise ValueError("modified migration snapshot: " + row["snapshot"])
        if digest(current) != row["migrated_sha256"]:
            raise ValueError("modified migrated input: " + row["file"])
        if migrate(original.decode()).encode() != current:
            raise ValueError("migration changed more than descriptor syntax: " + row["file"])
    return record


def run():
    if RECORD.exists():
        return validate()
    rows = []
    for profile in ("semantic", "structural"):
        for path in sorted((TEPL / "rules" / profile).glob("*.tepl")):
            original = path.read_bytes()
            converted = migrate(original.decode()).encode()
            if original == converted:
                continue
            snapshot = HERE / "baselines/pre_syntax" / profile / (path.name + ".txt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(original)
            path.write_bytes(converted)
            rows.append(dict(file=str(path.relative_to(ROOT)),
                             snapshot=str(snapshot.relative_to(ROOT)),
                             original_sha256=digest(original),
                             migrated_sha256=digest(converted)))
    record = dict(schema_version=1,
                  policy="Node labels replace legacy @ bindings; formulas, conditions, and graph structure are unchanged.",
                  files=rows)
    RECORD.write_text(json.dumps(record, indent=2) + "\n")
    return validate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = validate() if args.check else run()
    print(f"Validated {len(record['files'])} syntax migrations (semantic and structural counterparts).")


if __name__ == "__main__":
    main()
