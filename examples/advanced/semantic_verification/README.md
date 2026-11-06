# Semantic verification

Declare what operations do, then check whether a rewrite preserves that meaning.
The dialect defines ordered axes and scalar values for transpose, addition, and
contraction. The project includes an identity-transpose rule and algebraic
contraction transformations.

## Run

```sh
bazel-bin/tepl check examples/advanced/semantic_verification
bazel-bin/tepl rewrite examples/advanced/semantic_verification --trace -o /tmp/verified.tepl
```

The square input's identity transpose disappears, leaving `yield X`.
Change its permutation to `[1, 0]`: the rule no longer matches, even though the
output shape is still `[2, 2]`.

## Compare policies

This project defaults to `algebraic` for its contraction rules. Check the identity
rule alone under the strict policy:

```sh
bazel-bin/tepl check examples/advanced/semantic_verification/rules/remove_identity.tepl \
  --semantic-policy strict
bazel-bin/tepl check examples/advanced/semantic_verification/rules/contraction_algebra.tepl \
  --rule associate
```

Both report proved rules. The contraction example needs the supported ideal
arithmetic laws; a strict override rejects reassociation. These proofs concern
the declared operation meaning, not IEEE rounding or overflow behavior.

Read [semantic verification](../../../docs/language/semantics.md).
