#pragma once

#include <optional>

#include "src/legacy_semantics/solver.h"

namespace tepl::core::semantics {
// A certificate extracted from primitive IR, never from operation/field names.
// The recognized shape program is exactly the unrestricted legality contract.
struct OrderedContract {
  enum class Kind { Permutation, Contraction } kind;
  std::size_t left = 0, right = 1;
  std::size_t permutation = 0;
  std::size_t left_contract = 0, right_contract = 0;
  std::size_t left_batch = 0, right_batch = 0;
};
std::optional<OrderedContract> extractOrderedContract(
    const inference::SemanticNode&);

struct OrderedObstruction {
  std::size_t node = 0, first = 0, second = 0;
};

// nullopt means the theorem does not apply; the caller may use the generic
// solver. An admitted instance returns Found, certified Invalid, or Unknown
// on resource exhaustion. Its dimensions are >= 2, inputs read once, RHS
// descriptors independent and unrestricted, and arithmetic algebraic.
std::optional<Solution> solveOrdered(const inference::SemanticPlan&,
                                     const std::vector<List>& shapes,
                                     const std::vector<List>& known,
                                     const std::vector<List>& pinned,
                                     std::size_t budget = 100000);

// Independent certificate checks, also exposed for differential tests.
bool checkOrderedWitness(const inference::SemanticPlan&,
                         const std::vector<List>& shapes,
                         const std::vector<List>& known,
                         const std::vector<List>& fields);
bool checkOrderedObstruction(const inference::SemanticPlan&,
                             const std::vector<List>& shapes,
                             const std::vector<List>& known,
                             const OrderedObstruction&);
}  // namespace tepl::core::semantics
