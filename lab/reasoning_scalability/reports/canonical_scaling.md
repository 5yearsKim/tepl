# Canonical dot chain scalability after the shape equality fix

![Checking time and memory versus dot-chain length](canonical_scaling.png)

This graph measures the cost of checking each composite reassociation rule, including unsuccessful attempts. Markers distinguish completed proofs, unresolved obligations, and exhausted budgets. It does not establish successful-proof scalability beyond the verified sizes.

Eleven chain lengths, symbolic positive extents, rank-three inputs, explicit RHS attributes, algebraic policy, one warm-up and five measured runs, fixed budgets, and a 30-second timeout. All generated chains passed independent numerical validation. This run uses the checker after the shape-equality fix; the original 41-case measurements remain separate.

| Input tensors | Outcome | Median time | Median peak RSS |
| ---: | --- | ---: | ---: |
| 3 | proved | 0.470 s | 16.09 MiB |
| 4 | unresolved | 0.983 s | 18.72 MiB |
| 5 | unresolved | 1.628 s | 21.61 MiB |
| 6 | budget_exhausted | 1.859 s | 23.57 MiB |
| 7 | budget_exhausted | 1.854 s | 25.78 MiB |
| 8 | budget_exhausted | 1.958 s | 28.71 MiB |
| 10 | budget_exhausted | 2.136 s | 34.73 MiB |
| 12 | budget_exhausted | 2.329 s | 42.96 MiB |
| 16 | budget_exhausted | 3.050 s | 61.69 MiB |
| 24 | budget_exhausted | 5.228 s | 88.84 MiB |
| 32 | budget_exhausted | 4.721 s | 80.95 MiB |

The connecting line tracks all checking attempts. Drops in time or memory can reflect earlier budget exhaustion, rather than improved proof scalability.

[Raw measurements and source hashes](../results/canonical_scaling.json) · [Summary CSV](../results/canonical_scaling_summary.csv) · [SVG figure](canonical_scaling.svg) · [PDF figure](canonical_scaling.pdf)

```sh
python3 lab/reasoning_scalability/scripts/canonical_scaling.py --check
python3 lab/reasoning_scalability/scripts/plot_canonical_scaling.py
```
