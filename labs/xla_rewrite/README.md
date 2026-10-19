# XLA Rewrite Evaluation

This experiment measures how many cataloged XLA algebraic rewrites TEPL can
express, compile and verify. 

- [Catalog](catalog/README.md): pinned XLA sources, equations, conditions and counting policy.
- [TEPL project](tepl/README.md): local HLO dialects, explicit rules and input graphs.
- [Coverage](coverage/README.md): experiment inputs, tools and reproduction commands.
- [Compilation report](coverage/reports/compilation.md): language coverage and executable compilation.
- [Verification report](coverage/reports/verification.md): semantic proofs, missing semantics and unresolved obligations.

`catalog/` contains the source-based catalog. `tepl/` contains the executable
TEPL project. `coverage/` separates case data (`data/`), historical baselines
(`baselines/`), measurements (`results/`), readable reports (`reports/`), tools
(`scripts/`) and Python tests (`tests/`). Compilation and verification are
separate measurements; a compiled rule is not necessarily proved.
