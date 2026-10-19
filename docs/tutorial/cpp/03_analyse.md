# Tensor analysis

The previous C++ host-function API is no longer supported. Rules cannot call
application functions to obtain analysis facts.

Current generated Rust programs provide shape and dtype analysis through the
shared runtime. See [tensor analysis](../../advanced/custom_analysis.md) and the
[Rust tutorial](../rust/03_analyse.md).
