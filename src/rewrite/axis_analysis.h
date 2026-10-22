#pragma once

#include <optional>

#include "src/inference/evaluate.h"
#include "src/rewrite/value.h"

namespace tepl::rewrite {
struct Bindings;
// Per-match metadata context shared by where queries and construction. It
// computes axis facts only; feasibility is evaluated by ordinary where ASTs.
class AxisAnalysis {
 public:
  AxisAnalysis(const executable::ProgramData& p, const executable::Rule& r,
               const core::ConcreteGraph& g, Bindings& b, Budget& budget)
      : program_(p), rule_(r), graph_(g), bindings_(b), budget_(budget) {}
  Value query(core::builtins::Builtin, const std::vector<Value>&);
  void constructAttributes();

 private:
  void metadata(bool replacement);
  const inference::Result& source();
  const inference::Result& replacement();
  void charge(const inference::Result&);
  Value axes(const inference::AxisSequence&);
  const executable::ProgramData& program_;
  const executable::Rule& rule_;
  const core::ConcreteGraph& graph_;
  Bindings& bindings_;
  Budget& budget_;
  inference::Metadata metadata_;
  std::map<std::size_t, const inference::Instruction*> nodes_;
  std::optional<inference::Result> source_, replacement_;
};
}  // namespace tepl::rewrite
