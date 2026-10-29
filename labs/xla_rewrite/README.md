# XLA Rewrite Evaluation

This experiment measures how many cataloged XLA algebraic rewrites TEPL can
express, compile and verify. 

- [Catalog](catalog/README.md): pinned XLA sources, equations, conditions and counting policy.
- [TEPL project](tepl/README.md): local HLO dialects, explicit rules and input graphs.
- [Coverage](coverage/README.md): experiment inputs, tools and reproduction commands.
- [Attribute synthesis report](coverage/reports/attribute_synthesis.md): field-level axis inference, verification-only outcomes, and theorem admission candidates.
- [Compilation report](coverage/reports/compilation.md): language coverage and executable compilation.
- [Specification size](specification_size/README.md): matched explicit/inferred rule sizes, token attribution, and XLA excerpt review.

`catalog/` contains the source-based catalog. `tepl/` contains the executable
TEPL project. `coverage/` separates case data (`data/`), measurements (`results/`),
readable reports (`reports/`), tools
(`scripts/`) and Python tests (`tests/`). Compilation and verification are
separate measurements. Structural compilation uses declarations without value
semantics; synthesis successes require semantic verification and compilation.

Recorded reports/results identify their measured source and compiler revisions.
Their numbers are not refreshed by documentation changes. Reproduce a frozen run
at its recorded revision; active scripts track the current research proof path
for future runs. Broader mathematical arguments are indexed in
[research](../../research/attribute-synthesis/README.md).
