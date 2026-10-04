#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
int main() {
  namespace ir = tepl_generated;
  using A = ir::analysis::TensorAnalysis;
  eggc::EGraph<ir::OpNode, A> graph;
  auto root = graph.add(ir::OpNode::literal("1", ir::DType::I32));
  auto report = eggc::run(
      graph,
      std::vector{ir::rules::literal_root::rule_integer_root::build_rewrite()});
  auto two = graph.lookup(ir::OpNode::literal("2", ir::DType::I32));
  assert(two && graph.find(root) == graph.find(*two));
  assert(report.reason == eggc::StopReason::Saturated);
  assert((eggc::Extractor<ir::OpNode, A>(graph).find_best(root).first == 1));
  bool invalid = false;
  try {
    ir::OpNode::literal("128", ir::DType::I8);
  } catch (const ir::NodeError&) {
    invalid = true;
  }
  assert(invalid);
  assert(ir::accepts_literal(ir::DType::I64, "-9223372036854775808"));
  assert(!ir::accepts_literal(ir::DType::U64, "-0"));
}
