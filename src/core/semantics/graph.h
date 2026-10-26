#pragma once
#include "src/core/semantics/context.h"
#include "src/core/semantics/symbolic/index_expr.h"

namespace tepl::core::semantics {
struct Expr;
using E = std::shared_ptr<const Expr>;
struct Expr {
  enum class Kind { Read, Add, Product, Reduce, Const, Call, Number } kind;
  std::size_t capture = 0, occurrence = 0;
  S axes;  // Read: physical-dimension order. Reduce: bound coordinates.
  std::vector<E> args;
  std::int64_t constant = 0;
  I access;  // Optional calculated read; axes remains the capture interface.
  ScalarFunction function = ScalarFunction::Invalid;
  std::vector<S> parameters;
  S dtype;
  std::string number;
};
struct TensorMeaning {
  S axes, shape, dtype;
  E value;
  TensorAxisInterface interface;
  // Capture interfaces survive value cancellation and shape-only uses.
  std::map<std::size_t, S> occurrence_axes;
  bool may_have_independent_axes = false;
};
struct Lowerer {
  Context& context;
  // In verification, leaves are expressed in a source-derived coordinate frame.
  const std::map<std::size_t, S>* anchors = nullptr;
  std::vector<S> obligations;
  const std::map<std::size_t, S>* interfaces = nullptr;
  S expected_dtype;
  S metadata(const Meta&, const Instance&, const std::vector<TensorMeaning>&,
             const std::vector<S>&);
  S dtype(const Instance&, const std::vector<TensorMeaning>&);
  E rename(const E&, const S&, const S&);
  E evaluateAt(const E&, const S& frame, const I& access);
  I indexValue(const IndexValue&, const Instance&,
               const std::vector<TensorMeaning>&, const std::vector<S>&);
  E value(const Value&, const Instance&, const std::vector<TensorMeaning>&,
          const std::vector<S>&);
  std::map<std::size_t, S> outputPositions(const Meta&, const Instance&,
                                           const std::vector<TensorMeaning>&,
                                           const std::vector<S>&);
  TensorMeaning operation(std::size_t, std::vector<TensorMeaning>);
  TensorMeaning lower(std::size_t);
};
// Compose the source again after learning legality across the complete source,
// including shapes shared through pointwise consumers. sourceScope must be set.
TensorMeaning lowerSource(Context&);
E expression(Expr::Kind, std::vector<E>, S = {});
E constant(std::int64_t);
E scalarCall(ScalarFunction, std::vector<E>, std::vector<S> = {}, S = {});
E number(std::string, S = {});
std::vector<E> tensorReads(const E&);
S freeAxes(const E&);
std::string show(const E&);
E arithmeticSkeleton(Context&, std::size_t);
}  // namespace tepl::core::semantics
