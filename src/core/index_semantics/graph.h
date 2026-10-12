#pragma once
#include "src/core/index_semantics/expr.h"

namespace tepl::core::index_semantics {
struct Expr;
using E = std::shared_ptr<const Expr>;
struct Expr {
  enum class Kind { Read, Add, Product, Reduce } kind;
  std::size_t capture = 0, occurrence = 0;
  S axes;  // Read: physical-dimension order. Reduce: bound coordinates.
  std::vector<E> args;
};
struct TensorMeaning {
  S axes, shape, dtype;
  E value;
};
struct Lowerer {
  Context& context;
  // In verification, leaves are expressed in a source-derived coordinate frame.
  const std::map<std::size_t, S>* anchors = nullptr;
  std::vector<S> obligations;
  S metadata(const Meta&, const Instance&, const std::vector<TensorMeaning>&,
             const std::vector<S>&);
  S dtype(const Instance&, const std::vector<TensorMeaning>&);
  E rename(const E&, const S&, const S&);
  E value(const Value&, const std::vector<TensorMeaning>&,
          const std::vector<S>&);
  TensorMeaning operation(std::size_t, std::vector<TensorMeaning>);
  TensorMeaning lower(std::size_t);
};
E expression(Expr::Kind, std::vector<E>, S = {});
S freeAxes(const E&);
std::string show(const E&);
E arithmeticSkeleton(const Plan&, std::size_t);
}  // namespace tepl::core::index_semantics
