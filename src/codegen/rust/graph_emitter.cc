#include "src/codegen/rust/graph_emitter.h"

#include <map>
#include <set>
#include <sstream>

#include "src/codegen/rust/dialect_emitter.h"

namespace tepl::codegen::rust {
namespace {
std::string attribute(const core::AttributeValue& value,
                      const core::AttributeField& field, std::size_t depth,
                      bool optional, std::set<std::string>& imports) {
  using K = core::AttributeValue::Kind;
  if (value.kind == K::kNone) return "None";
  if (optional)
    return std::string("Some(") +
           attribute(value, field, depth, false, imports) + ")";
  if (depth) {
    std::string result = "vec![";
    for (std::size_t i = 0; i < value.elements.size(); ++i) {
      if (i) result += ", ";
      result += attribute(value.elements[i], field, depth - 1, false, imports);
    }
    return result + "]";
  }
  if (value.kind == K::kBool) return value.text;
  if (value.kind == K::kString) return quote(value.text) + ".to_string()";
  if (value.kind == K::kEnum) {
    if (field.type == "dtype") {
      imports.insert("DType");
      return dtype(*core::resolveDType(value.text));
    }
    auto variant = value.text == "default" ? "Default"
                   : value.text == "high"  ? "High"
                                           : "Highest";
    imports.insert("types");
    return std::string("types::Precision::") + variant;
  }
  return value.text + (field.type == "index" ? "u64" : "i64");
}
}  // namespace
void emitGraphs(const core::Program& program, const Names& names,
                const ProjectPlan& project, GenerationResult& output) {
  std::map<std::string, std::map<std::string, std::string>> indexes;
  std::map<std::string, std::string> identities;
  std::set<std::string> files, directories;
  std::map<std::string, SourceOrigin> origins;
  indexes[""];
  for (const auto& source : project.graph_modules) {
    const auto& origin = program.graphs.at(source.graphs.front().value).origin;
    std::string path, raw_path;
    std::size_t depth = 0;
    for (const auto& part : source.path) {
      auto id = Identifier::make(part, NameKind::kModule, origin);
      if (id.key == "mod" || (path.empty() && id.key == "definition"))
        throw NameError(origin, "reserved graph output module '" + part + "'");
      auto target = path.empty() ? id.key : path + "/" + id.key;
      raw_path += "/" + part;
      auto [found, inserted] = identities.emplace(target, raw_path);
      if (!inserted && found->second != raw_path)
        throw NameError(origin, "graph module name collision: " + part);
      indexes[path][id.key] = id.token;
      if (++depth < source.path.size()) directories.insert(target);
      path = target;
    }
    if (!files.insert(path).second)
      throw NameError(origin, "duplicate graph module: " + path);
    origins.emplace(path, origin);
    std::string root = "super::super::super";
    for (std::size_t i = 1; i < source.path.size(); ++i)
      root = "super::" + root;
    std::ostringstream out;
    out << "// Generated graph constructors.\n";
    std::set<std::string> graph_names;
    for (auto id : source.graphs) {
      const auto& graph = program.graphs.at(id.value);
      auto graph_name = Identifier::make("graph_" + graph.name,
                                         NameKind::kModule, graph.origin);
      if (!graph_names.insert(graph_name.key).second)
        throw NameError(graph.origin, "graph name collision: " + graph.name);
      std::set<std::string> imports = {"GraphDefinition", "NodeError",
                                       "OpNode"};
      bool uses_ids = false;
      std::ostringstream body;
      body << "pub fn build() -> Result<GraphDefinition, NodeError> {\n"
           << "let mut nodes = Vec::new();\n";
      for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::string expression;
        if (node.kind == core::ConcreteNode::Kind::kInput) {
          expression = "OpNode::input(" + quote(node.text) + ")";
        } else if (node.kind == core::ConcreteNode::Kind::kLiteral) {
          imports.insert("Op");
          imports.insert("OpAttrs");
          if (node.dtype) imports.insert("DType");
          expression =
              "OpNode::from_parts(Op::Literal, vec![], "
              "OpAttrs::Literal { value: " +
              quote(node.text) + ".to_string(), dtype: " +
              (node.dtype ? "Some(" + dtype(*node.dtype) + ")" : "None") +
              " })?";
        } else {
          const auto& op = program.operations.at(node.operation->value);
          imports.insert("dialects");
          auto operation = names.operation(op.id, root);
          operation.erase(
              0, root.size() + 2);  // Use the imported dialects module.
          auto prefix = operation.substr(0, operation.rfind("::Op::"));
          std::string attrs = prefix + "::OpAttrs::None";
          if (op.attributes) {
            const auto& schema =
                program.attribute_schemas.at(op.attributes->value);
            attrs =
                prefix + "::OpAttrs::" + names.schema(*op.attributes) + " { ";
            for (std::size_t j = 0; j < schema.fields.size(); ++j) {
              if (j) attrs += ", ";
              attrs += names.field(*op.attributes, j) + ": ";
              attrs += attribute(node.attributes.at(j), schema.fields[j],
                                 schema.fields[j].list_depth,
                                 schema.fields[j].optional, imports);
            }
            attrs += " }";
          }
          std::string children = "vec![";
          uses_ids |= !node.operands.empty();
          for (auto child : node.operands)
            children += "Id::from(" + std::to_string(child) + "usize),";
          children += "]";
          expression = "OpNode::new(" + operation + ", " + attrs + ", " +
                       children + ")?";
        }
        body << "nodes.push(" << expression << ");\n";
      }
      body << "let inputs = vec![\n";
      for (const auto& input : graph.nodes) {
        if (input.kind != core::ConcreteNode::Kind::kInput) continue;
        auto index = &input - graph.nodes.data();
        std::string shape = "None";
        if (input.shape) {
          shape = "Some(vec![";
          for (auto dim : *input.shape) shape += std::to_string(dim) + "u64,";
          shape += "])";
        }
        imports.insert("InputSpec");
        if (input.dtype) imports.insert("DType");
        auto input_dtype =
            input.dtype ? "Some(" + dtype(*input.dtype) + ")" : "None";
        body << "InputSpec { node: " << index << ", name: " << quote(input.text)
             << ".to_string(), shape: " << shape << ", dtype: " << input_dtype
             << " },\n";
      }
      body << "];\nlet bindings = BTreeMap::from([\n";
      for (const auto& [name, node] : graph.bindings) {
        body << "(" << quote(name) << ".to_string(), " << node << "usize),\n";
      }
      body << "]);\nGraphDefinition::new(nodes, inputs, bindings, "
           << graph.root << ")\n}\n";
      out << "pub mod " << graph_name.token << " {\n"
          << "use " << root << "::{";
      bool first_import = true;
      for (const auto& name : imports) {
        if (!first_import) out << ", ";
        out << name;
        first_import = false;
      }
      out << "};\nuse ::std::collections::BTreeMap;\n";
      if (uses_ids) out << "use ::egg::Id;\n";
      out << body.str() << "}\n";
    }
    output.files.push_back({"graphs/" + path + ".rs", out.str()});
  }
  for (const auto& file : files)
    if (directories.contains(file))
      throw NameError(origins.at(file),
                      "graph file/directory module collision: " + file);
  for (const auto& [path, children] : indexes) {
    std::string contents;
    if (path.empty())
      contents +=
          "mod definition;\npub use definition::{BuiltGraph, GraphDefinition, "
          "InputSpec};\n";
    for (const auto& [key, token] : children) {
      contents += "pub mod " + token + ";\n";
    }
    output.files.push_back(
        {"graphs/" + (path.empty() ? "" : path + "/") + "mod.rs", contents});
  }
}
}  // namespace tepl::codegen::rust
