#include <cassert>
#include <string>

#include "src/core/pipeline.h"
#include "src/core/syntax/parse.h"

namespace {
const std::string dialect = R"(
dialect D {
  op input(x: tensor) -> tensor
  op graph(x: tensor) -> tensor
  op add(x: tensor, y: tensor) -> tensor
  op config(x: tensor) -> tensor {
    attrs { axis: u64 flags: bool[] = [] note: string? }
  }
  op many(x: tensor...) -> tensor
  op opaque() -> tensor { attrs { value: elements } }
}
)";
tepl::core::semantics::InspectionData check(const std::string& source) {
  auto parsed = tepl::parse(dialect + source);
  assert(parsed.ok());
  auto checked = tepl::core::inspect(*parsed.program);
  assert(checked.ok());
  return std::move(*checked.program);
}
}  // namespace
int main() {
  const auto program = check(R"(
    graph example {
      input X: f32[2, 3]
      let N = (D.input X)
      let same = N
      let unused = (D.config[axis = +001] N)
      yield (D.add same (D.graph N))
    }
    graph scalar { yield -128:i8 }
    graph plain { input X yield X }
    graph variadic { yield (D.many) }
    graph partial { input X: [2] input Y: f32 yield (D.add X Y) }
  )");
  assert(program.graphs.size() == 5);
  const auto& graph = program.graphs.front();
  assert(graph.nodes.size() == 5 && graph.root == 4);
  assert(graph.bindings[1].second == graph.bindings[2].second);
  assert(graph.nodes[0].shape->size() == 2);
  assert(graph.nodes[2].attributes[0].text == "1");
  assert(graph.nodes[2].attributes[1].kind ==
         tepl::core::AttributeValue::Kind::kList);
  assert(graph.nodes[2].attributes[2].kind ==
         tepl::core::AttributeValue::Kind::kNone);
  assert(program.graphs.back().nodes[0].shape &&
         !program.graphs.back().nodes[0].dtype);
}
