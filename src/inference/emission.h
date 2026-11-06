#pragma once
#include <variant>

#include "src/inference/evaluate.h"
#include "src/inference/timing.h"

namespace tepl::inference {
// A rank-independent, producer-first expression DAG. Input/Output read only
// the replayed SOURCE network. No expression invokes synthesis or entailment.
enum class Op {
  Input,
  Output,
  Empty,
  Union,
  Intersect,
  Difference,
  Filter,
  Concat,
  Positions,
  Equal,
  SameSet,
  Distinct,
  Subset,
  SourceField, SourceExtents, SourceShape, Length, Range, Plus, Gather,
  Inverse, SymmetricDifference, Complement, Pullback, Remap, SourceField2, SourceField3, Singleton, BroadcastTransposeExtents
};
struct Expression {
  Op op;
  std::vector<std::size_t> args;
  std::size_t capture = 0;
  bool operator==(const Expression&) const = default;
};
struct EmissionBlock {
  Program source;
  std::vector<Expression> expressions;
  std::vector<std::size_t> checks;
  std::map<std::size_t, std::vector<std::size_t>> attributes;
  std::map<std::size_t, std::size_t> layouts;
  bool operator==(const EmissionBlock&) const = default;
};
struct Emission {
  Program original;
  std::vector<EmissionBlock> blocks;
  bool operator==(const Emission&) const = default;
};
// Total structural emission on validated synthesis programs, modulo resources.
// Never reads concrete ranks/fields, a rule guard, or a proof context.
Emission emit(const Program&);
// Replay source once, evaluate the total predicate, then evaluate builders only
// on success. Impossible denotes a false predicate within the stated domain.
Result predicate(const Emission&, const Metadata&, const Limits& = {});
Result execute(const Emission&, const Metadata&, const Limits& = {});
// Soundness-only execution allows zero/singleton extents. A false canonical
// predicate here is Inapplicable, never mathematical infeasibility.
Result executeSound(const Emission&, const Metadata&, const Limits& = {},
                    ExecutionTiming* = nullptr);
struct Portable {
  Program program;
  std::vector<std::vector<Expression>> nodes;
  std::vector<std::vector<std::size_t>> checks;
  std::vector<std::vector<std::vector<std::size_t>>> attributes;
  bool operator==(const Portable&) const = default;
};
Portable portable(const Emission&);
bool validatePortable(const Portable&);
}  // namespace tepl::inference
