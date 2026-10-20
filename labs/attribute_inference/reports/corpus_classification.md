# XLA corpus classification for attribute inference

Static classification only. Verification statuses come from the saved explicit sweep, not a new check.

Cases, not catalog schemas, are the counting unit. The negative control is listed in JSON and excluded below.

| RHS inference work | Recorded proved | Recorded unproved | Missing semantics | Total |
|---|---:|---:|---:|---:|
| No RHS attributes | 63 | 13 | 22 | 98 |
| Has RHS attributes | 23 | 19 | 69 | 111 |

Of **111** attribute-bearing cases, **42** have semantics for all required operations.
Of these, **38** have only axis/extent (`index[]`) RHS fields; **4** request other or mixed fields.
This type-based eligibility criterion identifies experiments to attempt; it does not promise inference support.
Comparison direction is a scalar setting, reported separately from axis/extent inference.

| Attribute scope with semantics | Recorded proved | Recorded unproved | Total |
|---|---:|---:|---:|
| axis_extent | 20 | 18 | 38 |
| other_attributes | 3 | 1 | 4 |
| mixed_attributes | 0 | 0 | 0 |

## Next batch

Five predeclared family strata: first remaining axis/extent case with all operation semantics in broadcast, reduction, dot, indexing, convolution, in original manifest order. Reduction and indexing strata require a recorded unproved explicit baseline to probe synthesis beyond baseline-proved rules. Exclude pilot cases. Not representative sampling.

| Case | Family | Recorded explicit result |
|---|---|---|
| broadcast_add | broadcast | proved |
| reduce_compose | reduction | not_proved |
| dot_associate | dot | proved |
| reverse_add_scalar | indexing | not_proved |
| expr_convolution_scale_filter_bias | convolution | not_proved |

Source files differing from their recorded explicit sweep hashes: none.

The next batch is a five-case mechanism sample, not an estimate of full-corpus coverage.
Only manifest-listed cases will execute. The rest remain unmeasured for automatic inference.
