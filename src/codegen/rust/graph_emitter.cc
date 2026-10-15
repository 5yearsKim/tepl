#include "src/codegen/rust/graph_emitter.h"

#include <map>
#include <set>

#include "src/codegen/rust/code_writer.h"
#include "src/codegen/rust/expression_emitter.h"
namespace tepl::codegen::rust {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string facts(const core::ConcreteNode& node) {
  std::string shape = "rt::Fact::Unknown";
  if (node.shape) {
    shape = "rt::Fact::Known(vec![";
    for (auto d : *node.shape) shape += num(d) + "u64,";
    shape += "])";
  }
  return "rt::TensorFacts { shape: " + shape + ", dtype: " +
         (node.dtype ? "rt::Fact::Known(rt::" + dtype(*node.dtype) + ")"
                     : "rt::Fact::Unknown") +
         " }";
}
}  // namespace
void emitGraphs(const executable::ProgramData& p, const Names&,
                const ProjectPlan& project, GenerationResult& result) {
  std::map<std::string, std::map<std::string, std::string>> indexes;
  std::map<std::string, std::string> identities;
  std::set<std::string> files, directories;
  std::map<std::string, SourceOrigin> origins;
  indexes[""];
  for (const auto& source : project.graph_modules) {
    const auto& origin = p.graphs.at(source.graphs.front().value).origin;
    std::string path, raw_path;
    std::size_t depth = 0;
    for (const auto& part : source.path) {
      auto id = Identifier::make(part, NameKind::kModule, origin);
      if (id.key == "mod") throw NameError(origin, "reserved graph module mod");
      auto target = path.empty() ? id.key : path + "/" + id.key;
      raw_path += "/" + part;
      auto [it, inserted] = identities.emplace(target, raw_path);
      if (!inserted && it->second != raw_path)
        throw NameError(origin, "graph module collision: " + part);
      indexes[path][id.key] = id.token;
      if (++depth < source.path.size()) directories.insert(target);
      path = target;
    }
    if (!files.insert(path).second)
      throw NameError(origin, "duplicate graph module: " + path);
    origins[path] = origin;
    std::string root = "super::super::super";
    for (std::size_t i = 1; i < source.path.size(); ++i)
      root = "super::" + root;
    CodeWriter out;
    std::set<std::string> graph_names;
    for (auto id : source.graphs) {
      const auto& graph = p.graphs.at(id.value);
      auto name = Identifier::make("graph_" + graph.name, NameKind::kModule,
                                   graph.origin);
      if (!graph_names.insert(name.key).second)
        throw NameError(graph.origin, "graph name collision");
      out.open("pub mod " + name.token);
      out.line(
          "use ::tepl_runtime::{self as rt, Value, EvalResult, Program as _};");
      out.line("use rt::values::AttributeValue;");
      out.line("use " + root +
               "::{Program, OpNode, OpAttrs, EGraph, TensorAnalysis};");
      out.open("pub fn build() -> EvalResult<rt::graphs::BuiltGraph<Program>>");
      out.line("let mut graph = EGraph::new(TensorAnalysis::default());");
      for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::string expr;
        if (node.kind == core::ConcreteNode::Kind::kInput) {
          out.line("graph.analysis.register_input(" + quote(node.text) + ", " +
                   facts(node) + ")?;");
          expr = "OpNode::input(" + quote(node.text) + ")";
        } else if (node.kind == core::ConcreteNode::Kind::kLiteral)
          expr =
              "OpNode::literal(" + quote(node.text) + ", " +
              (node.dtype ? "Some(rt::" + dtype(*node.dtype) + ")" : "None") +
              ")?";
        else {
          std::string attrs = "None";
          const auto& op = p.operations.at(node.operation->value);
          if (op.attributes) {
            std::string v =
                "Value::Descriptor { schema: " + num(op.attributes->value) +
                ", fields: vec![";
            for (const auto& a : node.attributes) v += value(a) + ",";
            attrs = "Some(OpAttrs::from_value(&(" + v + "] }))?)";
          }
          expr = "OpNode::operation(Program::operation(" +
                 num(node.operation->value) + "), " + attrs + ", vec![";
          for (auto child : node.operands) expr += "n" + num(child) + ",";
          expr += "])?";
        }
        out.line("let n" + num(i) + " = graph.add(" + expr + ");");
      }
      out.line("graph.analysis.check()?;");
      out.line("let bindings = ::std::collections::BTreeMap::from([");
      for (const auto& [name, id] : graph.bindings)
        out.line("(" + quote(name) + ".to_string(), n" + num(id) + "),");
      out.line("]);");
      out.line("Ok(rt::graphs::BuiltGraph { graph, root: n" + num(graph.root) +
               ", bindings })");
      out.close();
      out.close();
    }
    result.files.push_back({"graphs/" + path + ".rs", out.str()});
  }
  for (const auto& path : files)
    if (directories.contains(path))
      throw NameError(origins.at(path),
                      "graph file/directory module collision: " + path);
  for (const auto& [path, children] : indexes) {
    std::string contents;
    for (const auto& [key, token] : children)
      contents += "pub mod " + token + ";\n";
    result.files.push_back(
        {"graphs/" + (path.empty() ? "" : path + "/") + "mod.rs", contents});
  }
}
}  // namespace tepl::codegen::rust
