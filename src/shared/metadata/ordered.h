#pragma once

#include <optional>

#include "src/shared/metadata/axes.h"

namespace tepl::core::axis {
struct OrderedContract {
  enum class Kind { Contraction, Permutation, Summation, Addition, Reverse, Broadcast, Multiplication, Negation, UnitRemoval } kind;
  std::size_t batch_group = 0, contract_group = 0;
  std::size_t left_free_group = 0, right_free_group = 0;
  std::size_t left_batch = 0, right_batch = 0;
  std::size_t left_contract = 0, right_contract = 0;
  std::size_t permutation = 0, dimensions = 0;
};
// Shape/axis template only. The compiler additionally checks scalar semantics.
std::optional<OrderedContract> recognizeOrdered(const ShapeContract&);
// Sound schema extension templates; excluded from the complete constructor.
std::optional<OrderedContract> recognizeCoreShape(const ShapeContract&);
}  // namespace tepl::core::axis
