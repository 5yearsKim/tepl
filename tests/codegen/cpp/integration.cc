#include <cassert>

#include "components/generated/generated.h"
#include "current/generated.h"
#include "empty/generated.h"
#include "expressions/generated.h"
#include "sample/generated.h"
#include "semantic/generated.h"
namespace rt = tepl_runtime;
namespace e = expressions;
namespace d = e::dialects::d;
namespace c = tepl_generated;
template <class P>
eggc::Id input(rt::EGraph<P>& graph, std::string name,
               std::vector<std::uint64_t> shape) {
  graph.analysis
      .register_input(name, rt::TensorFacts::known(shape, rt::DType::F32))
      .value();
  return graph.add(rt::Node<P>::input(name));
}
std::size_t other_rule_count();
int main() {
  assert(other_rule_count() == e::rules::all().value().size());
  assert(empty::rules::all().value().empty());
  assert(semantic::rules::all().value().size() > 0);
  assert(sample::rules::all().value().size() > 0);
  assert(std::string(semantic::SEMANTIC_POLICY) == "algebraic");
  {
    auto b = e::graphs::expressions::graph_composed::build().value();
    auto x = b.bindings.at("X");
    std::vector<e::Rewrite> rules{
        e::rules::expressions::rule_compose::definition().build().value()};
    auto result = rt::runner::run(b.graph, rules);
    assert(result);
    bool found = false;
    for (const auto& n : b.graph.nodes(b.root))
      if (n.children() == std::vector<eggc::Id>{b.graph.find(x)}) {
        auto attrs = n.attribute_value().get<rt::Descriptor>("attributes");
        if (attrs.fields[0] == rt::shape({0, 1}) &&
            attrs.fields[1] == rt::Value(std::string("derived")) &&
            attrs.fields[4] == rt::integer("-1"))
          found = true;
      }
    assert(found);
    rules = {
        e::rules::expressions::rule_identity::definition().build().value()};
    assert(rt::runner::run(b.graph, rules));
    assert(b.graph.find(b.root) == b.graph.find(x));
  }
  assert((e::graphs::expressions::graph_variadic::build()
              .value()
              .graph.analysis_data(1)
              .shape.get("shape") == std::vector<std::uint64_t>{2, 3}));
  {
    auto b = e::graphs::expressions::graph_literal::build().value();
    assert((b.graph.analysis_data(b.root) ==
            rt::TensorFacts::known({}, rt::DType::I32)));
  }
  for (auto def :
       {e::rules::expressions::rule_false_guard::definition(),
        e::rules::expressions::rule_invalid_guard::definition(),
        e::rules::expressions::rule_invalid_rhs::definition(),
        e::rules::expressions::rule_incompatible_rhs::definition()}) {
    e::EGraph graph;
    auto x = input(graph, "X", {2, 3});
    auto root = graph.add(e::OpNode::operation(d::Op::Sink, {}, {x}).value());
    graph.rebuild();
    auto count = graph.node_count();
    std::vector<e::Rewrite> rules{def.build().value()};
    assert(rt::runner::run(graph, rules));
    assert(graph.node_count() == count);
    assert(graph.find(root) != graph.find(x));
    if (def.name.ends_with("invalid_guard") ||
        def.name.ends_with("invalid_rhs"))
      assert(graph.analysis.rejections().size() == 1);
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2, 3});
    auto root = graph.add(e::OpNode::operation(d::Op::Sink, {}, {x}).value());
    std::vector<e::Rewrite> rules{
        e::rules::expressions::rule_invalid_guard::definition().build().value(),
        e::rules::expressions::rule_valid_sink::definition().build().value()};
    assert(rt::runner::run(graph, rules));
    assert(graph.find(root) == graph.find(x));
    assert(graph.analysis.rejections().size() == 1);
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2, 3});
    auto root = graph.add(e::OpNode::operation(d::Op::Copy, {}, {x}).value());
    std::vector<e::Rewrite> rules{
        e::rules::expressions::rule_short_circuit::definition()
            .build()
            .value()};
    assert(rt::runner::run(graph, rules));
    assert(graph.find(root) == graph.find(x));
    assert(!graph.analysis.register_input(
        "X", rt::TensorFacts::known({3, 2}, rt::DType::F32)));
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2, 3});
    auto y = input(graph, "Y", {3, 2});
    graph.merge(x, y);
    graph.rebuild();
    assert(!graph.analysis.check());
    assert(!rt::runner::run(graph, std::vector<e::Rewrite>{}));
  }
  {
    auto b = c::graphs::current::graph_configured::build().value();
    std::vector<c::Rewrite> rules{
        c::rules::current::rule_partial::definition().build().value()};
    assert(rt::runner::run(b.graph, rules));
    bool identity = false;
    for (const auto& node : b.graph.nodes(b.root))
      if (node.kind() == rt::NodeKind::Operation &&
          node.children() ==
              std::vector<eggc::Id>{b.graph.find(b.bindings.at("X"))}) {
        auto a = node.attribute_value().get<rt::Descriptor>("attrs");
        if (a.fields[0] == rt::shape({0, 1}) &&
            a.fields[1] == rt::Value(false) && a.fields[2].none())
          identity = true;
      }
    assert(identity);
  }
  // Typed schemas reject incorrect attributes and arity before graph insertion.
  assert(!e::OpNode::operation(d::Op::Copy, {}, {}));
  assert(!e::OpNode::operation(d::Op::Transpose, {}, {0}));
  assert(!e::OpNode::literal("256", rt::DType::U8));
  assert(!e::OpNode::literal("-1", rt::DType::U64));
  {
    e::EGraph graph;
    auto a = input(graph, "A", {2, 2}), b = input(graph, "B", {2, 2});
    auto attrs = [](std::vector<std::uint64_t> p) {
      return e::OpAttrs(d::Shared{p, {}, true, rt::Precision::Default, 0});
    };
    auto left = graph.add(
        e::OpNode::operation(d::Op::Transpose, attrs({1, 0}), {a}).value());
    auto right = graph.add(
        e::OpNode::operation(d::Op::Transpose, attrs({0, 1}), {b}).value());
    graph.merge(left, right);
    graph.rebuild();
    std::vector<e::Rewrite> rules{
        e::rules::expressions::rule_identity::definition().build().value()};
    assert(rt::runner::run(graph, rules));
    assert(graph.find(left) == graph.find(b));
    assert(graph.find(left) != graph.find(a));
  }
  for (auto def : {e::rules::expressions::rule_sequence::definition(),
                   e::rules::expressions::rule_dtype_variable::definition(),
                   e::rules::expressions::rule_float_guard::definition()}) {
    e::EGraph graph;
    auto x = input(graph, "X", {2, 3, 4});
    auto root = graph.add(e::OpNode::operation(d::Op::Copy, {}, {x}).value());
    assert(
        rt::runner::run(graph, std::vector<e::Rewrite>{def.build().value()}));
    assert(graph.find(root) == graph.find(x));
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2}), y = input(graph, "Y", {2});
    auto root = graph.add(e::OpNode::operation(d::Op::Add, {}, {x, y}).value());
    std::vector<e::Rewrite> rules{
        e::rules::expressions::rule_repeated::definition().build().value()};
    assert(rt::runner::run(graph, rules));
    assert(graph.find(root) != graph.find(x));
    graph.merge(x, y);
    assert(rt::runner::run(graph, rules));
    assert(graph.find(root) == graph.find(x));
  }
  {
    e::EGraph graph;
    auto x = graph.add(e::OpNode::input("unknown"));
    auto root =
        graph.add(e::OpNode::operation(d::Op::Unknown, {}, {x}).value());
    assert(rt::runner::run(
        graph, std::vector<e::Rewrite>{
                   e::rules::expressions::rule_require_shape::definition()
                       .build()
                       .value()}));
    assert(graph.find(root) != graph.find(x));
    assert(graph.analysis.rejections().empty());
    assert(!graph.analysis.register_input(
        "unknown", rt::TensorFacts::known({2}, rt::DType::F32)));
  }
  {
    e::EGraph graph;
    graph.analysis
        .register_input("X",
                        {rt::Fact<std::vector<std::uint64_t>>::known({2}), {}})
        .value();
    auto x = graph.add(e::OpNode::input("X"));
    auto copy = graph.add(e::OpNode::operation(d::Op::Copy, {}, {x}).value());
    auto unknown = graph.add(e::OpNode::input("unknown"));
    auto parent =
        graph.add(e::OpNode::operation(d::Op::Copy, {}, {unknown}).value());
    graph.rebuild();
    assert(graph.analysis_data(copy).shape.is_known());
    assert(!graph.analysis_data(copy).dtype.is_known());
    assert(!graph.analysis_data(parent).shape.is_known());
    graph.merge(unknown, x);
    graph.rebuild();
    assert(graph.analysis_data(parent).shape.is_known());
  }
  {
    e::EGraph graph;
    graph.analysis
        .register_input("X", rt::TensorFacts::known({}, rt::DType::I32))
        .value();
    auto x = graph.add(e::OpNode::input("X"));
    auto one = graph.add(e::OpNode::literal("01", rt::DType::I32).value());
    auto root =
        graph.add(e::OpNode::operation(d::Op::Add, {}, {x, one}).value());
    assert(rt::runner::run(
        graph, std::vector<e::Rewrite>{
                   e::rules::expressions::rule_match_literal::definition()
                       .build()
                       .value()}));
    assert(graph.find(root) == graph.find(x));
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {});
    auto root = graph.add(e::OpNode::operation(d::Op::Copy, {}, {x}).value());
    assert(rt::runner::run(graph,
                           std::vector<e::Rewrite>{
                               e::rules::expressions::rule_literal::definition()
                                   .build()
                                   .value()}));
    bool found = false;
    for (const auto& n : graph.nodes(root))
      if (n.kind() == rt::NodeKind::Literal && n.text() == "0" &&
          n.dtype() == rt::DType::F32)
        found = true;
    assert(found);
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2});
    auto inner = graph.add(e::OpNode::operation(d::Op::Copy, {}, {x}).value());
    auto root =
        graph.add(e::OpNode::operation(d::Op::Copy, {}, {inner}).value());
    assert(rt::runner::run(
        graph,
        std::vector<e::Rewrite>{
            e::rules::expressions::rule_bind::definition().build().value()}));
    assert(graph.find(root) == graph.find(inner));
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2, 2});
    auto attrs = [](std::vector<std::uint64_t> p) {
      return e::OpAttrs(d::Shared{p, {}, true, rt::Precision::Default, 0});
    };
    auto a = graph.add(
        e::OpNode::operation(d::Op::Transpose, attrs({1, 0}), {x}).value());
    auto b = graph.add(
        e::OpNode::operation(d::Op::Transpose, attrs({0, 1}), {x}).value());
    graph.merge(a, b);
    graph.rebuild();
    auto root = graph.add(
        e::OpNode::operation(d::Op::Transpose, attrs({1, 0}), {a}).value());
    assert(rt::runner::run(graph,
                           std::vector<e::Rewrite>{
                               e::rules::expressions::rule_compose::definition()
                                   .build()
                                   .value()}));
    bool identity = false, swap = false;
    for (const auto& node : graph.nodes(root))
      if (node.children() == std::vector<eggc::Id>{graph.find(x)}) {
        auto p = node.attribute_value().field(0);
        identity = identity || p == rt::shape({0, 1});
        swap = swap || p == rt::shape({1, 0});
      }
    assert(identity && swap);
  }
  auto logical = [](rt::graphs::BuiltGraph<c::Program> built) {
    auto facts = static_cast<const rt::TensorFacts&>(
        built.graph.analysis_data(built.root));
    assert(rt::runner::run(
        built.graph, std::vector<c::Rewrite>{
                         c::rules::current::rule_logical_axes::definition()
                             .build()
                             .value()}));
    assert(built.graph.analysis_data(built.root) == facts);
    bool found = false;
    for (const auto& node : built.graph.nodes(built.root))
      if (node.children() ==
          std::vector<eggc::Id>{built.graph.find(built.bindings.at("X"))})
        found = true;
    assert(found);
    assert(built.graph.analysis.rejections().empty());
  };
  logical(c::graphs::current::graph_identity::build().value());
  logical(c::graphs::current::graph_empty::build().value());
  logical(c::graphs::current::graph_zero_singleton::build().value());
  {
    auto built = c::graphs::current::graph_nonidentity::build().value();
    auto count = built.graph.node_count();
    assert(rt::runner::run(
        built.graph, std::vector<c::Rewrite>{
                         c::rules::current::rule_logical_axes::definition()
                             .build()
                             .value()}));
    assert(built.graph.node_count() == count);
  }
  {
    auto built = e::graphs::expressions::graph_composed::build().value();
    auto rule =
        e::rules::expressions::rule_identity::definition().build().value();
    std::size_t count = 0;
    bool complete = rule.custom_search(built.graph,
                                       [&](auto) {
                                         ++count;
                                         return false;
                                       },
                                       {});
    assert(!complete && count == 1);
    count = 0;
    complete = rule.custom_search(
        built.graph,
        [&](auto) {
          ++count;
          return true;
        },
        [] { return true; });
    assert(!complete && count == 0);
  }
  {
    e::EGraph graph;
    auto x = input(graph, "X", {2}), y = input(graph, "Y", {2});
    rt::matching::Bindings bindings;
    bindings.captures = {{0, x}, {1, y}};
    graph.merge(x, y);
    graph.rebuild();
    rt::RuleContext<e::Program> ctx(graph, x, std::move(bindings));
    assert(ctx.capture(0) == ctx.capture(1));
  }
}
