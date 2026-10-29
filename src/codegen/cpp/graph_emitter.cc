#include "src/codegen/cpp/graph_emitter.h"

#include <map>
#include <set>

#include "src/codegen/cpp/code_writer.h"
#include "src/codegen/cpp/expression_emitter.h"
#include "src/codegen/cpp/paths.h"
namespace tepl::codegen::cpp {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string dtypeOption(std::optional<core::DType> d) {
  return d ? "::std::optional<rt::DType>(rt::" + dtype(*d) + ")"
           : "::std::nullopt";
}
std::string facts(const core::ConcreteNode& node) {
  std::string shape = "{}";
  if (node.shape) {
    shape = "rt::Fact<::std::vector<::std::uint64_t>>::known({";
    for (auto d : *node.shape) shape += num(d) + "ULL,";
    shape += "})";
  }
  return "rt::TensorFacts{" + shape + "," +
         (node.dtype
              ? "rt::Fact<rt::DType>::known(rt::" + dtype(*node.dtype) + ")"
              : "{}") +
         "}";
}
}  // namespace
void emitGraphs(const executable::ProgramData& p, const Names& n,
                const ProjectPlan& project, GenerationResult& result) {
  std::map<std::string, std::set<std::string>> indexes;
  std::map<std::string, std::string> identities;
  std::set<std::string> files, dirs;
  std::map<std::string, SourceOrigin> origins;
  indexes[""];
  for (const auto& source : project.graph_modules) {
    const auto& origin = p.graphs.at(source.graphs.front().value).origin;
    std::string path, raw, qualified;
    for (std::size_t i = 0; i < source.path.size(); ++i) {
      auto id = Identifier::make(source.path[i], NameKind::kModule, origin);
      if (id.key == "graphs")
        throw NameError(origin, "reserved graph module graphs");
      auto next = path.empty() ? id.key : path + "/" + id.key;
      raw += "/" + source.path[i];
      auto [it, inserted] = identities.emplace(next, raw);
      if (!inserted && it->second != raw)
        throw NameError(origin, "graph module collision");
      indexes[path].insert(id.key);
      if (i + 1 < source.path.size()) dirs.insert(next);
      path = next;
      qualified += (qualified.empty() ? "" : "::") + id.key;
    }
    if (!files.insert(path).second)
      throw NameError(origin, "duplicate graph module");
    origins[path] = origin;
    CodeWriter out;
    out.line("#pragma once\n#include \"" + parentInclude(path) + "program.h\"");
    std::set<std::string> names;
    for (auto id : source.graphs) {
      const auto& graph = p.graphs.at(id.value);
      auto name = Identifier::make("graph_" + graph.name, NameKind::kModule,
                                   graph.origin);
      if (!names.insert(name.key).second)
        throw NameError(graph.origin, "graph name collision");
      out.open("namespace " + n.root.substr(2) + "::graphs::" + qualified +
               "::" + name.token);
      out.line("namespace rt = ::tepl_runtime;");
      out.open("inline rt::EvalResult<rt::graphs::BuiltGraph<" + n.root +
               "::Program>> build()");
      out.open("return rt::evaluate([&]() -> rt::graphs::BuiltGraph<" + n.root +
               "::Program>");
      out.line(n.root + "::EGraph graph;");
      for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::string expr;
        if (node.kind == core::ConcreteNode::Kind::kInput) {
          out.line("graph.analysis.register_input(" + quote(node.text) + "," +
                   facts(node) + ").value();");
          expr = n.root + "::OpNode::input(" + quote(node.text) + ")";
        } else if (node.kind == core::ConcreteNode::Kind::kLiteral)
          expr = n.root + "::OpNode::literal(" + quote(node.text) + "," +
                 dtypeOption(node.dtype) + ").value()";
        else {
          std::string attrs = "::std::nullopt";
          const auto& op = p.operations.at(node.operation->value);
          if (op.attributes) {
            std::string v =
                "rt::Value(rt::Descriptor{" + num(op.attributes->value) + ",{";
            for (const auto& a : node.attributes) v += value(a) + ",";
            v += "}})";
            attrs = n.root + "::OpAttrs::from_value(" + v + ").value()";
          }
          expr = n.root + "::OpNode::operation(" + n.root +
                 "::Program::operation(" + num(node.operation->value) + ")," +
                 attrs + ",{";
          for (auto child : node.operands) expr += "n" + num(child) + ",";
          expr += "}).value()";
        }
        out.line("[[maybe_unused]] auto n" + num(i) + " = graph.add(" + expr +
                 ");");
      }
      out.line("graph.rebuild(); graph.analysis.check().value();");
      std::string bindings = "{";
      for (const auto& [name, id] : graph.bindings)
        bindings += "{" + quote(name) + ",n" + num(id) + "},";
      out.line("return {::std::move(graph),n" + num(graph.root) + "," +
               bindings + "}};");
      out.close(");");
      out.close();
      out.close();
    }
    result.files.push_back({"graphs/" + path + ".h", out.str()});
  }
  for (const auto& path : files)
    if (dirs.contains(path))
      throw NameError(origins.at(path),
                      "graph file/directory module collision");
  for (const auto& [path, children] : indexes) {
    std::string contents = "#pragma once\n";
    for (const auto& child : children) {
      auto next = path.empty() ? child : path + "/" + child;
      contents += "#include \"" + child +
                  (dirs.contains(next) ? "/graphs.h" : ".h") + "\"\n";
    }
    result.files.push_back(
        {"graphs/" + (path.empty() ? "" : path + "/") + "graphs.h", contents});
  }
}
}  // namespace tepl::codegen::cpp
