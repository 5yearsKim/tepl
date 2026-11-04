#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "src/codegen/cpp/names.h"
#include "src/codegen/rust/names.h"
#include "src/rewrite/builder.h"
#include "src/rewrite/matcher.h"

// Produce expectations with the native evaluator, independently of generated
// emitters and runtimes. Compare one match at a time, before either engine
// rewrites.
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
std::string emitCpp(const Value& value) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::monostate>)
          return "rt::Value()";
        else if constexpr (std::is_same_v<T, Integer>)
          return "rt::integer(" + codegen::cpp::quote(integerText(v)) + ")";
        else if constexpr (std::is_same_v<T, bool>)
          return v ? "rt::Value(true)" : "rt::Value(false)";
        else if constexpr (std::is_same_v<T, std::string>)
          return "rt::Value(std::string(" + codegen::cpp::quote(v) + "," +
                 std::to_string(v.size()) + "))";
        else if constexpr (std::is_same_v<T, List>) {
          std::string out = "rt::Value(rt::Value::List{";
          for (const auto& x : v) out += emitCpp(x) + ",";
          return out + "})";
        } else if constexpr (std::is_same_v<T, DescriptorValue>) {
          std::string out =
              "rt::Value(rt::Descriptor{" + std::to_string(v.schema) + ",{";
          for (const auto& x : v.fields) out += emitCpp(x) + ",";
          return out + "}})";
        } else
          throw std::runtime_error("unexpected reference value");
      },
      value.data);
}
std::string treeCpp(const executable::ProgramData& p,
                    const core::ConcreteGraph& g, std::size_t id) {
  const auto& n = g.nodes.at(id);
  if (n.kind == core::ConcreteNode::Kind::kInput)
    return "Expected{true," + codegen::cpp::quote(n.text) + ",0,{}, {}}";
  if (n.kind != core::ConcreteNode::Kind::kOperation)
    throw std::runtime_error("reference requires input/operation nodes");
  Value attrs;
  if (auto schema = p.operations.at(n.operation->value).attributes) {
    DescriptorValue d{schema->value, {}};
    for (const auto& x : n.attributes) d.fields.push_back(attributeValue(x));
    attrs = d;
  }
  std::string out = "Expected{false,{}," + std::to_string(n.operation->value) +
                    "," + emitCpp(attrs) + ",{";
  for (auto child : n.operands) out += treeCpp(p, g, child) + ",";
  return out + "}}";
}
std::string factsCpp(const core::ConcreteNode& n) {
  std::string shape = "{}";
  if (n.shape) {
    shape = "rt::Fact<std::vector<std::uint64_t>>::known({";
    for (auto x : *n.shape) shape += std::to_string(x) + "ULL,";
    shape += "})";
  }
  return "rt::TensorFacts{" + shape + "," +
         (n.dtype ? "rt::Fact<rt::DType>::known(rt::" +
                        codegen::rust::dtype(*n.dtype) + ")"
                  : "{}") +
         "}";
}
void cppPrelude() {
  std::cout << R"(#include "current/generated.h"
#include <cassert>
namespace c = tepl_generated;
namespace rt = tepl_runtime;
struct Expected {bool input; std::string name; std::size_t op; rt::Value attrs; std::vector<Expected> children;};
void check(const rt::PreparedReplacement<c::Program>& rhs,const c::EGraph& graph,const Expected& expected) {
  auto node=rhs.existing?graph.nodes(*rhs.existing).front():rhs.node;
  if(expected.input) {assert(node.kind()==rt::NodeKind::Input && node.text()==expected.name);return;}
  assert(node.kind()==rt::NodeKind::Operation && c::Program::operation_id(node.op())==expected.op);
  assert(node.attribute_value()==expected.attrs);
  assert(node.children().size()==expected.children.size());
  for(std::size_t i=0;i<expected.children.size();++i) {
    auto child=rhs.existing?rt::PreparedReplacement<c::Program>{node.children()[i],{}, {},graph.analysis_data(node.children()[i])}:rhs.children[i];
    check(child,graph,expected.children[i]);
  }
}
int main() {
)";
}
void cppCase(const executable::ProgramData& p, const core::ConcreteGraph& graph,
             const executable::Rule& rule, bool allowed, bool invalid,
             const Bindings& bindings, const core::ConcreteGraph& candidate,
             std::size_t root) {
  std::cout << "{ auto built=c::graphs::current::graph_" << graph.name
            << "::build().value();\n"
            << "auto definition=c::rules::current::rule_" << rule.name
            << "::definition();\n"
            << "rt::Budget budget; auto "
               "witnesses=rt::matching::matches<c::Program>(definition.lhs,"
               "built.root,built.graph,budget);\n"
            << "std::size_t prepared=0; for(auto binding:witnesses) {\n"
            << "if(!rt::matching::constraints<c::Program>(definition."
               "constraints,binding,built.graph))continue;\n"
            << "rt::RuleContext<c::Program> "
               "ctx(built.graph,built.root,std::move(binding));\n"
            << "auto result=definition.prepare(ctx);\n";
  if (invalid) {
    std::cout << "assert(!result); ++prepared; } assert(prepared==1); }\n";
    return;
  }
  std::cout << "auto rhs=result.value(); if(rhs) {\n";
  if (allowed) {
    for (const auto& d : rule.derivations)
      if (d.target)
        std::cout << "assert(ctx.descriptor(" << d.target->value << ")=="
                  << emitCpp(Value(bindings.descriptors.at(d.target->value)))
                  << ");\n";
    std::cout << "assert((rhs->facts==" << factsCpp(candidate.nodes.at(root))
              << "));\ncheck(*rhs,built.graph," << treeCpp(p, candidate, root)
              << ");\n";
  }
  std::cout << "++prepared; }} assert(prepared==" << (allowed ? 1 : 0)
            << "); }\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2 && argc != 3)
      throw std::runtime_error(
          "expected compiled fixture path and optional --cpp");
    const bool cpp = argc == 3 && std::string(argv[2]) == "--cpp";
    if (argc == 3 && !cpp) throw std::runtime_error("unknown reference target");
    std::ifstream input(argv[1]);
    auto program = executable::load(
        std::string(std::istreambuf_iterator<char>(input), {}));
    const auto& p = program.ir();
    if (cpp)
      cppPrelude();
    else
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
        bool allowed = false, invalid = false;
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
          invalid = true;
        } catch (const ConditionUnavailable&) {
          allowed = false;
        }
        allowed ? ++accepted : ++rejected;
        if (cpp) {
          cppCase(p, graph, rule, allowed, invalid, bindings, candidate, root);
          continue;
        }
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
            << "let result=(definition.prepare)(&mut ctx);\n";
        if (invalid) {
          std::cout << "assert!(result.is_err()); prepared.push(()); } "
                       "assert_eq!(prepared.len(),1); }\n";
          continue;
        }
        std::cout << "if let Some(rhs)=result.unwrap() {\n";
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
    if (cpp) std::cout << "}\n";
    if (!accepted || !rejected)
      throw std::runtime_error(
          "reference fixture needs both accepted and rejected matches");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
