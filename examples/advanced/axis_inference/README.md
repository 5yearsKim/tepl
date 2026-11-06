# Automatic axis attributes

Compare two rules for composing transposes. `compose_explicit` calculates the
permutation using `gather`; `compose_inferred` leaves it for TEPL to construct
from the operation's axis and value contracts.

## Run

```sh
bazel-bin/tepl check examples/advanced/axis_inference
bazel-bin/tepl rewrite examples/advanced/axis_inference --rule compose_inferred \
  --trace -o /tmp/inferred.tepl
bazel-bin/tepl rewrite examples/advanced/axis_inference --rule compose_explicit \
  --trace -o /tmp/explicit.tepl
```

Both outputs contain one transpose with permutation `[0, 1]`. Compare the files:

```sh
cmp /tmp/inferred.tepl /tmp/explicit.tepl
```

This project deliberately stops at the composed transpose. The
[complete transpose project](../../projects/transpose/README.md) also removes
identity transposes. Both use the strict arithmetic policy.

Read [automatic axis attributes](../../../docs/language/axis-inference.md).
