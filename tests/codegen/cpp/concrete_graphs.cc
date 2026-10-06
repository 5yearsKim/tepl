#include <cassert>
#include <eggc/all.hpp>
#include <limits>

#include "ir/generated.h"
namespace ir = tepl_generated;
namespace graphs = ir::graphs::concrete_graphs;
using namespace graphs;
using ir::DType;
using ir::TensorInfo;
TensorInfo info() { return {{2, 3}, DType::F32}; }
template <class F>
void rejects(F fn) {
  bool rejected = false;
  try {
    fn();
  } catch (const ir::NodeError&) {
    rejected = true;
  }
  assert(rejected);
}
int main() {
#ifdef TEPL_TEST_NESTED_GRAPHS
  auto [nested, nested_result] =
      ir::graphs::type::match::graph_nested::build().into_egraph();
  assert((nested.analysis_data(nested_result.root).info() ==
          TensorInfo{{}, DType::I32}));
#endif
  ir::GraphDefinition def = graph_shared::build();
  assert(def.nodes().size() == 4);
  const ir::InputSpec& input_spec = def.inputs()[0];
  assert((input_spec.shape == std::vector<std::uint64_t>{2, 3}));
  assert(def.get("N") == def.get("shared_node"));
  const auto& children = def.nodes()[def.root()].children();
  assert(children[0] == children[1]);
  const auto& attrs = std::get<ir::dialects::graph_ops::ConfigAttrs>(
      std::get<ir::dialects::graph_ops::OpAttrs>(
          def.nodes()[*def.get("unused")].attrs().value)
          .value);
  assert(attrs.axis == 1 &&
         attrs.offset == std::numeric_limits<std::int64_t>::min());
  assert(attrs.regions && attrs.regions->empty());
  assert(attrs.label == "a\"b\\c" && !attrs.note && attrs.flags.empty());
  assert((attrs.axes == std::vector<std::uint64_t>{0, 1}));
  auto [typed, built] = def.into_egraph();
  assert(built.get("N") == built.get("shared_node") && built.get("unused"));
  assert(typed.analysis_data(built.root).info() == info());
  auto defaults = graph_defaults::build();
  const auto& default_attrs = std::get<ir::dialects::graph_ops::ConfigAttrs>(
      std::get<ir::dialects::graph_ops::OpAttrs>(
          defaults.nodes()[defaults.root()].attrs().value)
          .value);
  assert(default_attrs.axis == std::numeric_limits<std::uint64_t>::max());
  assert(default_attrs.offset == 1 && !default_attrs.axes &&
         default_attrs.note == "text");
  eggc::EGraph<ir::OpNode> graph;
  auto existing = graph.add(ir::OpNode::input("already"));
  ir::BuiltGraph structural = graph_structural::build().insert_into(graph);
  assert(structural.get("X") != existing);
  auto structural_node = graph.nodes(structural.root).front();
  assert((structural_node.children() ==
          std::vector<eggc::Id>{*structural.get("X"), *structural.get("X")}));
  assert(structural.get("unused") && structural.get("Y") &&
         !structural.get("missing"));
  rejects([&] { def.insert_into(graph); });
  auto input = graph_input_output::build().insert_into(graph);
  assert(input.root == input.get("X"));
  auto literal = graph_literal::build();
  assert(std::get<ir::LiteralAttrs>(literal.nodes()[0].attrs().value).value ==
         "+01");
  auto [scalars, scalar] = literal.into_egraph();
  assert((scalars.analysis_data(scalar.root).info() ==
          TensorInfo{{}, DType::I32}));
  auto [unknown, unknown_scalar] = graph_untyped_literal::build().into_egraph();
  assert(!unknown.analysis_data(unknown_scalar.root).info());
  auto supplied = graph_partial::build()
                      .with_input_info("X", info())
                      .with_input_info("Y", info());
  auto [host, result] =
      supplied.into_egraph_with([](ir::TensorBindingTable bindings) {
        return ir::TensorAnalysis(std::move(bindings));
      });
  assert(host.analysis_data(result.root).info() == info());
  rejects(
      [] { graph_partial::build().with_input_info("X", {{9}, DType::F32}); });
  rejects(
      [] { graph_shared::build().with_input_info("X", {{2, 3}, DType::I32}); });
  rejects([] { graph_structural::build().with_input_info("missing", info()); });
  rejects([] { graph_invalid_shape::build(); });
  rejects([] { graph_invalid_dtype::build(); });
  rejects([] { graph_invalid_partial_shape::build(); });
  rejects([] { graph_invalid_partial_dtype::build(); });
  rejects(
      [] { ir::graphs::GraphDefinition({ir::OpNode::input("X")}, {}, {}, 0); });
  auto [untyped, untyped_result] = graph_structural::build().into_egraph();
  assert(!untyped.analysis_data(untyped_result.root).info());
}
