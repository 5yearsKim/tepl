# TEPL builtins

`catalog.*` owns builtin IDs, names, signatures, semantic domains, variadic
arity, and expression-section availability. Metadata and rule checkers use this
catalog; successful IR stores a resolved builtin ID, never an unresolved name.

An unprefixed call resolves to an explicitly bound template `fn` parameter
first, then a builtin. `$name(...)` always selects a host function, including
when its name is also a builtin. Builtins do not add entries to the host-function
table or methods to generated `HostFunctions` traits. Unknown names, wrong arity,
incompatible types, and unavailable sections produce source diagnostics.

Semantic domain and section availability are independent: `broadcast_shape`
is a shape-domain function available in both shapes and rules. Catalog context
flags distinguish operation `shape` and `dtype`, rule `where`, and rule `derive`, so future
functions can restrict their sections without separate name-resolution tables.

## Signatures and availability

Operation metadata programs use checked signed `i128` Integer values, Bool, DType, and homogeneous
nested lists. In the shape signatures below, `T` is a single shared element type
within one call and may itself be a list. Rule values keep their existing types:
`index` is unsigned `u64`, `i64` is signed, and `index_list` contains indices.
`L` below denotes `index_list`; `I` is one shared `index` or `i64` type.
There is no implicit signed/unsigned conversion or floating builtin overload.

| Builtin | Shape signature | Rule signature | Domain | Sections |
| --- | --- | --- | --- | --- |
| `len` | `List<T> → Integer` | `L → index` | Common | shape, dtype, where, derive |
| `range` | `Integer → List<Integer>` | `index → L` | Common | shape, dtype, where, derive |
| `concat` | Two or more `List<T> → List<T>` | Two or more `L → L` | Common | shape, dtype, where, derive |
| `gather` | `(List<T>, List<Integer>) → List<T>` | `(L, L) → L` | Common | shape, dtype, where, derive |
| `exclude` | `(List<T>, List<T>) → List<T>` | `(L, L) → L` | Common | shape, dtype, where, derive |
| `slice` | `(List<T>, Integer, Integer) → List<T>` | `(L, index, index) → L` | Common | shape, dtype, where, derive |
| `replace` | `(List<T>, Integer, T) → List<T>` | `(L, index, index) → L` | Common | shape, dtype, where, derive |
| `sum`, `product` | `List<Integer> → Integer` | `L → index` | Common | shape, dtype, where, derive |
| `all`, `any` | `List<Bool> → Bool` | Unavailable: rules have no Boolean-list type | Common | shape, dtype |
| `contains` | `(List<T>, T) → Bool` | `(L, index) → bool` | Common | shape, dtype, where, derive |
| `is_disjoint` | `(List<T>, List<T>) → Bool` | `(L, L) → bool` | Common | shape, dtype, where, derive |
| `is_valid_axis_list` | `(List<Integer>, Integer) → Bool` | `(L, index) → bool` | Shape | shape, dtype, where, derive |
| `broadcast_shape` | `(List<Integer>, List<Integer>) → List<Integer>` | `(L, L) → L` | Shape | shape, dtype, where, derive |
| `min`, `max` | `(Integer, Integer) → Integer` | `(I, I) → I` | Common | shape, dtype, where, derive |
| `floor_div`, `ceil_div` | `(Integer, Integer) → Integer` | `(I, I) → I` | Common | shape, dtype, where, derive |

`is_float`, `is_integer`, `is_signed_integer`, `is_unsigned_integer`, and
`is_numeric` have signature `DType → Bool` in dtype programs and
`dtype → bool` in rules. They belong to the DType domain and are available in
`dtype`, `where`, and `derive`. Bool is neither integer nor numeric.

Rule expressions retain their existing syntax. Dimension sequences and
`index_list` host results supply lists; list literals, indexing, and
comprehensions remain operation metadata expression forms. `derive` assigns
attribute descriptors, so builtins usually appear inside descriptor-producing
host calls, rather than as the whole assignment:

```tepl
where {
    len(Batch) > 0;
    product(Batch) <= 4096;
    is_valid_axis_list(range(len(Batch)), len(Batch));
}
derive {
    @out = $infer_attrs(@in, concat(Batch, range(2)));
}
```

## Runtime behavior

Rust implementations live in `templates/rust/src/builtins/` and are copied into
`builtins/` in each generated module. `common` contains generic list/integer
primitives; `shape` contains broadcasting, axis validation, and tensor dimension
conversions; `dtype` contains the five dtype classification predicates.
`error` supplies `BuiltinError` and `BuiltinResult`.

Shape and rule emitters share builtin-call lowering. Integer operators also
call the checked runtime helpers. Shape errors become invalid inference; rule
errors reject a candidate before insertion or subsequent host calls. Boolean
operators retain short-circuit evaluation, and arguments execute once in order.
The builtin/operator prefix of `where` also runs during matching, in source
order as its bindings become available, and is rechecked before application.
False or a builtin error prunes the branch. The first host-containing condition
ends this prefix, including nested and short-circuited host calls. That condition
and all subsequent conditions, plus `derive`, run during application. Tensor
shape/dtype declarations bind dimensions through the same match-checking plan.

Overflow, invalid indices/ranges, nonpositive rounded divisors, and incompatible
broadcasting are failures. `/` truncates toward zero; rounded division requires
a strictly positive divisor. Empty sum is zero, empty product is one, and any
zero makes a product zero even if an earlier prefix would overflow. Rules check
reductions against `u64`; operation shapes check against `i128` and validate
final dimensions against `u64`. No silent widening changes rule arithmetic.

Rust callers use `builtins::common` and `builtins::shape` directly. Errors and
results use `builtins::{BuiltinError, BuiltinResult}`; integer helpers use the
sealed `builtins::common::Integer` trait.

Compiler tests live in `tests/core_test.cc` and `tests/shape_core_test.cc`.
`tests/codegen/builtins.{tepl,rs}` exercises rule builtins in freshly generated
Rust in debug and release, alongside shared runtime tests in
`labs/rust-egg/tests/runtime/shape_builtins.rs`. Generated code contains no test implementation.
