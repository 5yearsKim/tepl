#include <cassert>

#include "witnesses/generated.h"
namespace ir = witnesses;
namespace rt = tepl_runtime;
int main() {
  auto b = ir::graphs::witnesses::graph_g::build().value();
  auto second =
      b.graph.add(ir::OpNode::operation(ir::dialects::d::Op::Source,
                                        ir::OpAttrs(ir::dialects::d::Size{3}),
                                        {b.bindings.at("X")})
                      .value());
  b.graph.merge(b.root, second);
  b.graph.rebuild();
  assert(!b.graph.analysis_data(b.root).shape.is_known());
  auto count = b.graph.node_count();
  auto result = rt::runner::run(b.graph, ir::rules::all().value());
  assert(result);
  const auto& shape = b.graph.analysis_data(b.root).shape.get("shape");
  assert(shape == std::vector<std::uint64_t>{2} ||
         shape == std::vector<std::uint64_t>{3});
  assert(b.graph.node_count() == count + 1);
}
