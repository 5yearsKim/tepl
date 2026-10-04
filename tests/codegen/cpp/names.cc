#include <cassert>
#include <eggc/all.hpp>

#include "ir/generated.h"
#include "second/generated.h"

namespace ir = tepl_generated;
struct Host {
  std::optional<bool> type(const ir::TensorInfo&) const { return true; }
  std::optional<bool> match(const ir::TensorInfo&) const { return true; }
  std::optional<bool> gen(const ir::TensorInfo&) const { return true; }
};
int odr_check();
int main() {
  ir::analysis::TensorBindingTable inputs;
  inputs.insert("x", {{2}, ir::DType::F32});
  eggc::EGraph<ir::OpNode, ir::analysis::TensorAnalysis> graph{
      ir::analysis::TensorAnalysis(inputs)};
  auto x = graph.add(ir::OpNode::input("x"));
  auto b = graph.add(ir::OpNode::make(ir::dialects::b::Op::BCopy, {}, {x}));
  auto egg =
      graph.add(ir::OpNode::make(ir::dialects::egg::Op::EggCopy, {}, {b}));
  auto root =
      graph.add(ir::OpNode::make(ir::dialects::std::Op::StdCopy, {}, {egg}));
  auto rule = ir::rules::type::match::rule_match::build_rewrite(Host{});
  eggc::run(graph, std::vector{rule});
  assert(graph.find(root) == graph.find(x));
  assert(odr_check() == 13);
  auto node =
      other::generated::OpNode::literal("1", other::generated::DType::I32);
  assert(node.children().empty());
  auto typed = ir::OpNode::make(ir::dialects::type::Op::Type,
                                ir::dialects::type::Match{2, 1, true, 2}, {x});
  assert(typed.attrs().schema_id());
}
