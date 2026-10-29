#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "src/codegen/rust/names.h"
#include "src/rewrite/builder.h"
#include "src/rewrite/matcher.h"

// Produce expectations with the native evaluator, independently of the Rust
// emitter/runtime. Compare one match at a time, before either engine rewrites.
namespace {
using namespace tepl;
using namespace tepl::rewrite;
using codegen::rust::quote;
std::string emit(const Value& value) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>)
          return "Value::None";
        else if constexpr (std::is_same_v<T, Integer>)
          return "Value::Integer(" + integerText(v) + "i128)";
        else if constexpr (std::is_same_v<T, bool>)
          return v ? "Value::Boolean(true)" : "Value::Boolean(false)";
        else if constexpr (std::is_same_v<T, std::string>)
          return "Value::String(" + quote(v) + ".into())";
        else if constexpr (std::is_same_v<T, List>) {
          std::string out = "Value::List(vec![";
          for (const auto& x : v) out += emit(x) + ",";
          return out + "])";
        } else if constexpr (std::is_same_v<T, DescriptorValue>) {
          std::string out =
              "Value::Descriptor {schema:" + std::to_string(v.schema) +
              ",fields:vec![";
          for (const auto& x : v.fields) out += emit(x) + ",";
          return out + "]}";
        } else
          throw std::runtime_error("unexpected reference value");
      },
      value.data);
}
std::string tree(const executable::ProgramData& p, const core::ConcreteGraph& g,
                 std::size_t id) {
  const auto& n = g.nodes.at(id);
  if (n.kind == core::ConcreteNode::Kind::kInput)
    return "Expected::Input(" + quote(n.text) + ")";
  if (n.kind != core::ConcreteNode::Kind::kOperation)
    throw std::runtime_error(
        "reference fixture requires input/operation nodes");
  Value attrs;
  if (auto schema = p.operations.at(n.operation->value).attributes) {
    DescriptorValue d{schema->value, {}};
    for (const auto& x : n.attributes) d.fields.push_back(attributeValue(x));
    attrs = d;
  }
  std::string out = "Expected::Op(" + std::to_string(n.operation->value) + "," +
                    emit(attrs) + ",vec![";
  for (auto child : n.operands) out += tree(p, g, child) + ",";
  return out + "])";
}
std::string facts(const core::ConcreteNode& n) {
  std::string shape = "rt::Fact::Unknown";
  if (n.shape) {
    shape = "rt::Fact::Known(vec![";
    for (auto x : *n.shape) shape += std::to_string(x) + "u64,";
    shape += "])";
  }
  return "rt::TensorFacts {shape:" + shape + ",dtype:" +
         (n.dtype
              ? "rt::Fact::Known(rt::" + codegen::rust::dtype(*n.dtype) + ")"
              : "rt::Fact::Unknown") +
         "}";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("expected compiled fixture path");
    std::ifstream input(argv[1]);
    auto program = executable::load(
        std::string(std::istreambuf_iterator<char>(input), {}));
    const auto& p = program.ir();
    std::cout << R"(use tepl_generated_tests::current as c;
use tepl_runtime::{self as rt, Value, Program as _};
enum Expected { Input(&'static str), Op(usize, Value, Vec<Expected>) }
fn check_node(node: &c::OpNode, children: &[rt::PreparedReplacement<c::Program>],
              graph: &c::EGraph, expected: &Expected) {
    match expected {
        Expected::Input(name) => assert!(matches!(node.kind(), rt::NodeKind::Input(actual) if actual == name)),
        Expected::Op(op, attrs, expected_children) => {
            assert!(matches!(node.kind(), rt::NodeKind::Operation(actual) if c::Program::operation_id(*actual) == *op));
            assert_eq!(node.attribute_value(), *attrs);
            assert_eq!(children.len(), expected_children.len());
            for (child, expected) in children.iter().zip(expected_children) { check(child, graph, expected); }
        }
    }
}
fn check(rhs: &rt::PreparedReplacement<c::Program>, graph: &c::EGraph, expected: &Expected) {
    match rhs {
        rt::PreparedReplacement::Existing {id,..} => {
            let node = &graph[*id].nodes[0];
            let children: Vec<_> = node.operands().iter().map(|id| rt::PreparedReplacement::Existing {id:*id,facts:graph[*id].data.clone()}).collect();
            check_node(node, &children, graph, expected);
        }
        rt::PreparedReplacement::New {node,children,..} => check_node(node, children, graph, expected),
    }
}
)";
    std::size_t accepted = 0, rejected = 0;
    for (const auto& original : p.graphs) {
      auto graph = original;
      Budget inference;
      for (std::size_t i = 0; i < graph.nodes.size(); ++i)
        infer(p, graph, i, inference);
      for (const auto& rule : p.rules) {
        Budget budget;
        Bindings bindings;
        bool allowed = false;
        auto candidate = graph;
        std::size_t root = graph.root;
        try {
          if (match(p, rule, graph, root, bindings, budget)) {
            Evaluator evaluator(program, graph, rule.id.value, bindings,
                                budget);
            allowed = evaluator.evaluateWhere();
            if (allowed)
              root = build(p, *rule.rhs, candidate, bindings, budget, 10000);
          }
        } catch (const EvaluationError&) {
          allowed = false;
        } catch (const ConditionUnavailable&) {
          allowed = false;
        }
        allowed ? ++accepted : ++rejected;
        std::cout
            << "#[test]\nfn " << rule.name << "_" << graph.name << "() {\n"
            << "let built=c::graphs::current::graph_" << graph.name
            << "::build().unwrap();\n"
            << "let definition=c::rules::current::rule_" << rule.name
            << "::definition();\n"
            << "let "
               "witnesses=rt::matching::matches(&definition.lhs,built.root,&"
               "built.graph,&mut rt::Budget::default()).unwrap();\n"
            << "let mut prepared=vec![];\nfor mut bindings in witnesses {\n"
            << "if !rt::matching::constraints(&definition.constraints,&mut "
               "bindings,&built.graph).unwrap() {continue;}\n"
            << "let mut "
               "ctx=rt::RuleContext::new(&built.graph,built.root,bindings);\n"
            << "if let Some(rhs)=(definition.prepare)(&mut ctx).unwrap() {\n";
        if (allowed) {
          for (const auto& derivation : rule.derivations)
            if (derivation.target)
              std::cout << "assert_eq!(ctx.descriptor("
                        << derivation.target->value << ").unwrap(),"
                        << emit(Value(bindings.descriptors.at(
                               derivation.target->value)))
                        << ");\n";
          std::cout << "assert_eq!(*rhs.facts(),"
                    << facts(candidate.nodes.at(root)) << ");\n"
                    << "check(&rhs,&built.graph,&" << tree(p, candidate, root)
                    << ");\n";
        }
        std::cout << "prepared.push(rhs);\n}\n}\nassert_eq!(prepared.len(),"
                  << (allowed ? 1 : 0) << ");\n}\n";
      }
    }
    if (!accepted || !rejected)
      throw std::runtime_error(
          "reference fixture needs both accepted and rejected matches");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
