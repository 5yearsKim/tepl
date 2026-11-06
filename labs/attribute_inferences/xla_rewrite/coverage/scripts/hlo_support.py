"""Inventory the local HLO declarations and their semantic coverage."""

import json
import re
from pathlib import Path

if __package__:
    from .paths import HERE, TEPL
else:
    from paths import HERE, TEPL


def declarations(path):
    source = re.sub(r"//[^\n]*", "", path.read_text())
    result = {}
    for match in re.finditer(r"\bop\s+(\w+)\s*\(", source):
        cursor, depth, opened = match.end(), 0, False
        while cursor < len(source):
            char = source[cursor]
            if char == "{":
                depth, opened = depth + 1, True
            elif char == "}":
                depth -= 1
                if opened and depth == 0:
                    cursor += 1
                    break
            elif char == ";" and not opened:
                cursor += 1
                break
            cursor += 1
        text = source[match.start():cursor]
        result[match[1]] = dict(
            text=text,
            aliases=re.findall(r"\balias:\s*(\w+)\s*;", text),
            semantic=bool(re.search(r"\bsemantics\s*{", text)),
        )
    return result


def inventory():
    return declarations(TEPL / "dialects/hlo.tepl")


def required_operations(path):
    # Metadata functions such as gather(...) are not HLO graph operations.
    # Parse graph applications so node labels and metadata calls cannot skew
    # the semantic-support denominator.
    if __package__:
        from .synthesis_cases import Rule
    else:
        from synthesis_cases import Rule
    operations = inventory()
    aliases = {alias: name for name, op in operations.items()
               for alias in [name, *op["aliases"]]}
    rule = Rule(path.read_text())
    return sorted({aliases[op.name.split(".")[-1]]
                   for op in rule.lhs.operations() + rule.rhs.operations()})


def validate_support():
    declared = inventory()
    structural = declarations(TEPL / "dialects/hlo_structural.tepl")
    support = json.loads((HERE / "data/hlo_support.json").read_text())
    if set(declared) != set(structural) or set(declared) != set(support["operations"]):
        raise ValueError("HLO declarations and support inventory must cover the same operations")
    for name, op in declared.items():
        row = support["operations"][name]
        if op["semantic"] != row["semantic"] or not row["reason"]:
            raise ValueError("stale HLO semantic support entry: " + name)
        if structural[name]["semantic"]:
            raise ValueError("structural HLO profile must not contain value semantics")
        text = op["text"]
        for block in ("axes", "semantics"):
            match = re.search(r"\b" + block + r"\s*{", text)
            if not match:
                continue
            cursor, depth = match.end(), 1
            while depth:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1
            text = text[:match.start()] + text[cursor:]
        if re.sub(r"\s+", "", text) != re.sub(r"\s+", "", structural[name]["text"]):
            raise ValueError("HLO profiles must share signatures, attributes and dtype contracts: " + name)
    return support
