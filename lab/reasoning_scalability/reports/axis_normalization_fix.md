# Registered axis view equality fix

The dot chain is lowered and normalized as a whole. Repeated use of a previously
verified reassociation rule is unnecessary. The failed checks exposed a gap in
axis equality and position lookup, rather than a failure of dot reassociation.

## Cause

TEPL preserves registered axis views during canonicalization so their origin
maps remain available. This also preserves nested identity selections such as
`gather(gather(B, [0]), [0])` when `B` contains one axis. After more dots are
composed, those views occur inside filters and concatenations. The equality
checker did not simplify their sequence meaning recursively.

For example, the four-input case needed an equality equivalent to this one,
where `B`, `I`, and `J` are disjoint coordinate groups:

```text
concat(filter_axes(concat(B, I, J), B),
       filter_axes(concat(B, I, J), I))
    == concat(B, I)
```

Preserved identity selections obscured the matching groups. A related gap
affected `auto`: the checker could fail to recover the identity position map
between a filtered view and its retained group, even when that group was known
to occur in increasing physical positions.

## Change

Axis equality now compares the sequence meaning of nested views without
changing the registered mapping graph. It composes selectors, removes identity
selections, projects literal selectors into known concatenation blocks, and
simplifies physical-order filters structurally. If this does not establish
equality, the existing proof procedure remains available.

Position lookup also recognizes the identity map for an ordered retained group.
It requires established increasing positions, a distinct parent frame, and a
valid selector. A reversed group does not satisfy this shortcut.

Equal numerical extents do not identify axes. The new regression tests reject
swapped or substituted equal-sized axes and ensure that filtering a permuted
frame preserves that frame's physical order.

## Results

The default synthesis and independent verification contexts each have a work
budget of 1,000,000. This fix leaves those budgets unchanged.

- Canonical four- and five-input chains prove with both explicit RHS attributes
  and `auto`, without additional conditions. Synthesis includes independent
  verification of the completed rule.
- The original permuted three-input cases prove in both modes. The permuted
  four-input case proves with explicit attributes; its `auto` variant reaches
  the work budget.
- Sampled canonical explicit chains with six or more inputs still reach the
  unchanged work budget. These outcomes do not establish that their proofs
  would complete with a larger budget.

[Functional recheck, inputs, diagnostics, and hashes](../results/axis_normalization_fix.json)
records individual CLI checks. It is separate from repeated timing measurements.
The refreshed [scalability report](canonical_scaling_axis_fix.md) records timing
and memory under the same fixed budgets; earlier measurements remain preserved.

Validation commands:

```sh
bazel test //tests:all --test_output=errors
python3 -m unittest discover -s lab/reasoning_scalability/tests
```

All 27 C++ test targets and all six experiment tests passed. Formatting checks
passed for the four C++ files touched by this fix.
