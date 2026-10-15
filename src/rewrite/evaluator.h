#pragma once
#include <map>

#include "src/program/program.h"
#include "src/rewrite/value.h"

namespace tepl::rewrite {
struct Bindings {
  std::map<std::size_t, std::size_t> captures;
  std::map<std::size_t, DescriptorValue> descriptors;
  std::map<std::size_t, Value> dimensions, dtypes, locals;
};
class Evaluator {
 public:
  Evaluator(const program::Program& program, const core::ConcreteGraph& graph,
            std::size_t rule, Bindings& bindings, Budget& budget)
      : program_(program),
        graph_(graph),
        rule_(rule),
        bindings_(bindings),
        budget_(budget) {}
  Value eval(const core::TypedExpr&);
  Value symbolic(const core::index_semantics::Symbol&);

 private:
  const program::Program& program_;
  const core::ConcreteGraph& graph_;
  std::size_t rule_;
  Bindings& bindings_;
  Budget& budget_;
  std::map<std::pair<std::size_t, std::size_t>, Value> axis_cache_;
  std::set<std::pair<std::size_t, std::size_t>> active_axes_;
};
// Infer and validate metadata from operation contracts. Missing input metadata
// stays unknown; a violated available contract is an error.
void infer(const core::Program&, core::ConcreteGraph&, std::size_t node,
           Budget&);
}  // namespace tepl::rewrite
