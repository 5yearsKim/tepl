#include "src/codegen/cpp/graph_emitter.h"

#include <map>
#include <set>
#include <sstream>

#include "src/codegen/cpp/dialect_emitter.h"

namespace tepl::codegen::cpp {
namespace {
std::string attribute(const core::AttributeValue& value,
                      const core::AttributeField& field, std::size_t depth,
                      bool optional, std::set<std::string>& imports) {
  using K = core::AttributeValue::Kind;
  if (value.kind == K::kNone) return "::std::nullopt";
  if (optional) {
    static const std::map<std::string, std::string> types = {
        {"index", "::std::uint64_t"},
        {"i64", "::std::int64_t"},
        {"bool", "bool"},
        {"string", "::std::string"},
        {"dtype", "DType"},
        {"precision", "Precision"},
        {"region", "Region"},
        {"elements", "Elements"},
        {"replica_groups", "ReplicaGroups"},
        {"dot_algorithm", "DotAlgorithm"}};
    auto type = types.at(field.type);
    if (type != "bool" && !type.starts_with("::std::")) imports.insert(type);
    for (std::size_t i = 0; i < depth; ++i)
      type = "::std::vector<" + type + ">";
    auto contents = attribute(value, field, depth, false, imports);
    return "::std::optional<" + type + ">(" + type +
           (depth ? contents : "{" + contents + "}") + ")";
  }
  if (depth) {
    std::string result = "{";
    for (std::size_t i = 0; i < value.elements.size(); ++i) {
      if (i) result += ", ";
      result += attribute(value.elements[i], field, depth - 1, false, imports);
    }
    return result + "}";
  }
  if (value.kind == K::kBool) return value.text;
  if (value.kind == K::kString)
    return std::string("::std::string(") + quote(value.text) + ")";
  if (value.kind == K::kEnum) {
    if (field.type == "dtype") {
      imports.insert("DType");
      return dtype(*core::resolveDType(value.text));
    }
    auto variant = value.text == "default" ? "Default"
                   : value.text == "high"  ? "High"
                                           : "Highest";
    imports.insert("Precision");
    return std::string("Precision::") + variant;
  }
  if (field.type == "i64" && value.text == "-9223372036854775808")
    return "(-9223372036854775807LL - 1LL)";
  return value.text + (field.type == "index" ? "ULL" : "LL");
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
      if (id.key == "graphs" || (path.empty() && id.key == "definition"))
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
    auto root = names.root;
    std::string namespace_path;
    for (const auto& part : source.path)
      namespace_path +=
          "::" + Identifier::make(part, NameKind::kModule, origin).token;
    std::ostringstream out;
    std::string include;
    for (std::size_t i = 0; i < source.path.size(); ++i) include += "../";
    out << "#pragma once\n#include " << quote(include + "graphs/definition.h")
        << "\n";
    std::set<std::string> graph_names;
    for (auto id : source.graphs) {
      const auto& graph = program.graphs.at(id.value);
      auto graph_name = Identifier::make("graph_" + graph.name,
                                         NameKind::kModule, graph.origin);
      if (!graph_names.insert(graph_name.key).second)
        throw NameError(graph.origin, "graph name collision: " + graph.name);
      std::set<std::string> imports = {"GraphDefinition", "InputSpec",
                                       "OpNode"};
      bool uses_dialects = false;
      std::ostringstream body;
      body << "inline GraphDefinition build() {\n"
           << "::std::vector<OpNode> nodes;\n";
      for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::string expression;
        if (node.kind == core::ConcreteNode::Kind::kInput) {
          expression = "OpNode::input(" + quote(node.text) + ")";
        } else if (node.kind == core::ConcreteNode::Kind::kLiteral) {
          imports.insert("Op");
          imports.insert("LiteralAttrs");
          if (node.dtype) imports.insert("DType");
          expression =
              "OpNode::from_parts(Op::Literal, {}, LiteralAttrs { " +
              quote(node.text) + ", " +
              (node.dtype ? "::std::optional(" + dtype(*node.dtype) + ")"
                          : "::std::nullopt") +
              " })";
        } else {
          const auto& op = program.operations.at(node.operation->value);
          uses_dialects = true;
          auto operation = names.operation(op.id, root);
          operation.erase(0, root.size() + 2);  // Use the local dialects alias.
          auto prefix = operation.substr(0, operation.rfind("::Op::"));
          std::string attrs = prefix + "::OpAttrs{}";
          if (op.attributes) {
            const auto& schema =
                program.attribute_schemas.at(op.attributes->value);
            attrs = schemaType(names, *op.attributes);
            attrs.erase(0, root.size() + 2);
            attrs += " { ";
            for (std::size_t j = 0; j < schema.fields.size(); ++j) {
              if (j) attrs += ", ";
              attrs += attribute(node.attributes.at(j), schema.fields[j],
                                 schema.fields[j].list_depth,
                                 schema.fields[j].optional, imports);
            }
            attrs += " }";
          }
          std::string children = "{";
          for (auto child : node.operands)
            children += std::to_string(child) + ",";
          children += "}";
          expression = "OpNode::make(" + operation + ", " + attrs + ", " +
                       children + ")";
        }
        body << "nodes.push_back(" << expression << ");\n";
      }
      body << "::std::vector<InputSpec> inputs{\n";
      for (const auto& input : graph.nodes) {
        if (input.kind != core::ConcreteNode::Kind::kInput) continue;
        auto index = &input - graph.nodes.data();
        std::string shape = "::std::nullopt";
        if (input.shape) {
          shape = "::std::vector<::std::uint64_t>{";
          for (auto dim : *input.shape) shape += std::to_string(dim) + "ULL,";
          shape += "}";
        }
        if (input.dtype) imports.insert("DType");
        auto input_dtype = input.dtype
                               ? "::std::optional(" + dtype(*input.dtype) + ")"
                               : "::std::nullopt";
        body << "{ " << index << ", " << quote(input.text) << ", " << shape
             << ", " << input_dtype << " },\n";
      }
      body << "};\n::std::map<::std::string, ::std::size_t, ::std::less<>> "
              "bindings{\n";
      for (const auto& [name, node] : graph.bindings) {
        body << "{" << quote(name) << ", " << node << "},\n";
      }
      body << "};\nreturn GraphDefinition(::std::move(nodes), "
              "::std::move(inputs), ::std::move(bindings), "
           << graph.root << ");\n}\n";
      out << "namespace " << root.substr(2) << "::graphs" << namespace_path
          << "::" << graph_name.token << " {\n";
      for (const auto& name : imports)
        out << "using " << root << "::" << name << ";\n";
      if (uses_dialects)
        out << "namespace dialects = " << root << "::dialects;\n";
      out << body.str() << "}\n";
    }
    output.files.push_back({"graphs/" + path + ".h", out.str()});
  }
  for (const auto& file : files)
    if (directories.contains(file))
      throw NameError(origins.at(file),
                      "graph file/directory module collision: " + file);
  for (const auto& [path, children] : indexes) {
    std::string contents = std::string("#pragma once\n");
    if (path.empty()) contents += "#include \"definition.h\"\n";
    for (const auto& [key, token] : children) {
      auto target = path.empty() ? key : path + "/" + key;
      contents += "#include \"" + key +
                  (indexes.contains(target) ? "/graphs.h" : ".h") + "\"\n";
    }
    output.files.push_back(
        {"graphs/" + (path.empty() ? "" : path + "/") + "graphs.h", contents});
  }
}
}  // namespace tepl::codegen::cpp
