#include <cassert>
#include <string>

#include "src/codegen/generate.h"
#include "src/core/analyze.h"
#include "src/parse.h"

namespace {
const std::string dialect = R"(
dialect D {
  op input(x: tensor) -> tensor;
  op graph(x: tensor) -> tensor;
  op add(x: tensor, y: tensor) -> tensor;
  op config(x: tensor) -> tensor {
    attrs { axis: index; flags: bool[] = []; note: string?; }
  }
  op many(x: tensor...) -> tensor;
  op opaque() -> tensor { attrs { value: elements; } }
}
)";
tepl::core::Program check(const std::string& source) {
  auto parsed = tepl::parse(dialect + source);
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(checked.ok());
  return std::move(*checked.program);
}
void reject(const std::string& source, const std::string& message) {
  auto parsed = tepl::parse(dialect + source);
  assert(parsed.ok());
  auto checked = tepl::core::analyze(*parsed.program);
  assert(!checked.ok());
  bool found = false;
  for (const auto& diagnostic : checked.diagnostics)
    found |= diagnostic.message.find(message) != std::string::npos;
  assert(found);
}
}  // namespace
int main() {
  const auto program = check(R"(
    graph example {
      input X: f32[2, 3];
      let N = (D.input X);
      let same = N;
      let unused = (D.config[axis = +001] N);
      yield (D.add same (D.graph N));
    }
    graph scalar { yield -128:i8; }
    graph plain { input X; yield X; }
    graph variadic { yield (D.many); }
    graph partial { input X: [2]; input Y: f32; yield (D.add X Y); }
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
  for (auto target :
       {tepl::codegen::Target::kRust, tepl::codegen::Target::kCpp}) {
    tepl::codegen::Options options;
    options.target = target;
    auto generated = tepl::codegen::generate(program, options);
    assert(generated.ok());
    bool found = false;
    for (const auto& file : generated.files)
      found |= file.contents.find("graph_example") != std::string::npos;
    assert(found);
  }
  reject("graph bad { yield missing; }", "unknown or forward");
  reject("graph bad { let A = B; let B = 1; yield A; }", "unknown or forward");
  reject("graph bad { input X; let X = 1; yield X; }", "duplicate graph value");
  reject("graph bad { yield (D.add 1); }", "operand count");
  reject("graph bad { yield (D.config 1); }", "missing graph attribute");
  reject("graph bad { yield (D.config[axis=0, axis=1] 1); }",
         "duplicate graph attribute");
  reject("graph bad { yield (D.config[axis=0, extra=1] 1); }",
         "unknown graph attribute");
  reject("graph bad { yield (D.config[axis=-1] 1); }", "invalid value");
  reject("graph bad { yield (D.config[axis=18446744073709551616] 1); }",
         "invalid value");
  reject("graph bad { yield (D.config[axis=0, flags=[1]] 1); }",
         "invalid value");
  reject("graph bad { yield (D.input[axis=0] 1); }", "has no attributes");
  reject("graph bad { yield (D.opaque[value=1]); }",
         "no concrete value syntax");
  reject("graph bad { input X: unknown[]; yield X; }", "unknown dtype");
  reject("graph bad { input X: [18446744073709551616]; yield X; }",
         "index range");
  reject("graph bad { yield 128:i8; }", "invalid for its dtype");
  reject("graph bad { yield 1; } graph bad { yield 2; }", "duplicate graph");
  for (const auto& source :
       {"graph bad { input X; }", "graph bad { yield 1; yield 2; }",
        "graph bad { input X: [-1]; yield X; }",
        "graph bad { let X = 1; input Y; yield X; }"})
    assert(!tepl::parse(source).ok());
}
