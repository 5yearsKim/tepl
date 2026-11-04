# Core correctness tests

These tests check compilation, portable-program integrity, native rewriting,
conditional applicability, and supported extension schemas. A separate dense
integer interpreter checks source/replacement values. The fixed chain fixtures
exercise longer constructor programs; they produce no timing measurements.

```sh
bazel test -c opt //labs/accepted_core:acceptance_test //labs/accepted_core:extensions_test
```

For compilation and runtime scaling measurements, use the
[main scaling experiment](../reasoning_scalability/README.md).
