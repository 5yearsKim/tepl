# XLA Rewrite Catalog

This directory contains the source-based catalog of XLA algebraic rewrites. TEPL compilation and verification results are reported in [coverage/](../coverage/README.md). The [attribute synthesis report](../coverage/reports/attribute_synthesis.md) describes the axis-only evaluation setup, field-level inference, and verification-only results.

## Files

| File | Purpose |
|---|---|
| [catalog.md](catalog.md) | Readable equations, conditions, RHS attribute calculations, source references, tests, and category counts |
| [rules.json](rules.json) | Manually curated catalog entries |
| [source_review.md](source_review.md) | Method review checklist and exclusions |
| [source_review.json](source_review.json) | Structured checklist, including overloaded methods and links to rule IDs |
| [source_manifest.json](source_manifest.json) | XLA commit, source paths, download URLs, sizes, and SHA-256 hashes |
| `upstream/` | Unmodified source snapshot, tests, delegated convolution-swap helper, and upstream license |
| [summarize_catalog.py](summarize_catalog.py) | Validate the data and generate Markdown reports |

## Scope and counting

The catalog covers logical tensor algebra and indexing in the pinned `algebraic_simplifier.cc`, including the delegated convolution operand-swap implementation. Selection does not depend on TEPL support.

Each entry counts one conditional rewrite schema. Symmetric operand matches, literal/broadcast variants of the same calculation, and overlapping implementations are merged where the transformation is the same. Different replacements or materially different conditions have separate entries. Explicit opcode parameters, such as an elementwise function or comparison direction, remain one schema when they share the same construction. Counts therefore describe this catalog's declared granularity, rather than an official XLA rule total.

Exclude visitor plumbing, ownership, logging, graph mutation mechanics, tuple/computation representation changes, physical copy/bitcast transformations, standalone dtype conversion, distributed collectives, and control-flow or RNG transformations. The review ledger records exclusions. Retain logical reshape/transpose rewrites, numerical assumptions, shape/indexing conditions, and attribute formulas. Record backend, layout, configuration, and profitability checks separately as source policy.

Entries summarize source behavior; they are not formal specifications or proofs of exact numerical equality. XLA includes floating-point transformations with rounding or exceptional-value differences. Their gates and assumptions must be considered when constructing explicit TEPL rules. Any restriction to TEPL's arithmetic model must be reported as an adaptation. Related tests include positive and negative cases; they were inspected as source evidence, not executed here.

## Notation

- `X`, `Y`, and `Z` are tensors; `C` is a literal unless stated otherwise. Scalar constants may be broadcast to the required shape.
- Axes use zero-based positions. `gather(p,q)` selects entries of sequence `p` at positions `q`, preserving order.
- `transpose(X,p)` uses output-to-input positions: output axis `i` is input axis `p[i]`.
- `broadcast(X,d)` maps input axis `i` to output position `d[i]`; its output shape is supplied by the rule.
- `slice(X,a,b,s)` uses inclusive starts, exclusive limits, and positive strides.
- `slice_dynamic(X,start,size)` and `update(X,Y,start)` clamp starts to valid bounds. `update` means HLO dynamic-update-slice.
- `reduce(X,init,f,r)` folds with combiner `f` over axes `r`. Identity-initializer assumptions are retained where a source transformation relies on them.
- `dot` means HLO generalized dot. Its attributes specify paired batch and contraction positions; remaining axes form its result.
- Names such as `reshape_units`, `mixed_radix_iotas`, and `transpose_if_needed` abbreviate constructions explained in the RHS-attributes field. Equations are descriptive notation, not TEPL syntax.

## Reproduce the reports

Python 3.9 or newer is sufficient; no XLA build is required.

```sh
python3 labs/attribute_inferences/xla_rewrite/catalog/summarize_catalog.py
python3 labs/attribute_inferences/xla_rewrite/catalog/summarize_catalog.py --check
python3 -m unittest discover -s labs/attribute_inferences/xla_rewrite/catalog -p 'test_*.py'
```

The validator checks snapshot hashes, unique IDs, categories, source ranges and symbols, test declarations, and complete method-checklist coverage. It also detects stale generated reports. It does not extract mathematical rules or prove their conditions automatically.

To restore the captured sources from the pinned URLs:

```sh
python3 labs/attribute_inferences/xla_rewrite/catalog/summarize_catalog.py --fetch --check
```

Edit `rules.json` or `source_review.json`, then regenerate the reports. Changing the upstream revision requires a new source review and updated manifest; do not mix source versions.
