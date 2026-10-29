# Attribute synthesis research

This collection studies soundness, feasibility, and constructive axis-attribute
synthesis. It was moved from `examples/research/` so runnable TEPL examples and
research arguments have separate entry points.

## Primary documents

| Document | Purpose and status |
| --- | --- |
| [Paper](paper.md) | Concise theorem/paper formulation; mathematical draft |
| [Completeness proof](proofs/completeness.md) | Full argument, admitted fragments, extensions, and finite-check references |
| [Soundness proof](proofs/soundness.md) | Verification-discipline obligations; does not formally verify the compiler |
| [Korean draft](drafts/completeness.ko.md) | Independently authored working draft; preserve its text and development separately |

The signed-axis specialization and broader supplied-metadata construction have
different admission assumptions. Consult each theorem's actual statement before
claiming coverage. The [implementation design](../../design/attribute-synthesis.md)
describes what production TEPL currently implements; only axis fields are
synthesized, with other fields supplied/defaulted.

## Supporting material

| Location | Status |
| --- | --- |
| [notes/auto-realizability.md](notes/auto-realizability.md) | Earlier existence/ambiguity proposal, with historical compiler references |
| [notes/axis-domain-propagation.md](notes/axis-domain-propagation.md) | Earlier concrete propagation design; not the current production contract |
| [scripts/](scripts/) | Finite checks supporting current mathematical arguments |
| [archive/](archive/README.md) | Earlier fragments, proposals, and their validation scripts |

Run finite checks from the repository root:

```sh
python3 research/attribute-synthesis/scripts/check_oriented_axis_synthesis.py
python3 research/attribute-synthesis/scripts/check_pointwise_axis_synthesis.py
python3 research/attribute-synthesis/scripts/check_supplied_axis_synthesis.py
```

These scripts use independent small-instance oracles. They are not production
synthesis, benchmark measurements, or formal proof checkers. Reproduction of
compiler measurements belongs to the [lab index](../../labs/README.md).

Historical results may contain the former proof path and its SHA-256. Keep those
records intact and use their recorded revision for reproduction. Active lab
scripts use the new proof location for future classifications and runs.
