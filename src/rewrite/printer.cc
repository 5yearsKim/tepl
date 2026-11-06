#include "src/rewrite/printer.h"

#include <filesystem>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace tepl::rewrite {
namespace {
std::string quote(const std::string& s) {
  if (s.find_first_of("\r\n") != std::string::npos)
    throw std::invalid_argument(
        "TEPL string syntax cannot represent a newline");
  std::ostringstream out;
  out << std::quoted(s);
  return out.str();
}
std::string attribute(const core::AttributeValue& v) {
  using K = core::AttributeValue::Kind;
  if (v.kind == K::kNone) return "none";
  if (v.kind == K::kString) return quote(v.text);
  if (v.kind != K::kList) return v.text;
  std::string s = "[";
  for (const auto& a : v.elements) {
    if (s.size() > 1) s += ", ";
    s += attribute(a);
  }
  return s + "]";
}
}  // namespace
std::string printGraph(const executable::ProgramData& p,
                       const core::ConcreteGraph& graph, bool imports) {
  std::set<std::size_t> live;
  std::vector<std::size_t> pending{graph.root};
  while (!pending.empty()) {
    auto id = pending.back();
    pending.pop_back();
    if (!live.insert(id).second) continue;
    for (auto a : graph.nodes.at(id).operands) pending.push_back(a);
  }
  using Dialect = std::pair<std::string, std::string>;
  std::map<Dialect, std::string> dialects;
  for (auto i : live) {
    const auto& n = graph.nodes[i];
    if (n.operation) {
      const auto& op = p.operations.at(n.operation->value);
      auto key = Dialect{op.origin.definition.source_name, op.dialect};
      if (!dialects.contains(key))
        dialects[key] = "d" + std::to_string(dialects.size());
    }
  }
  std::ostringstream out;
  if (imports) {
    for (const auto& [key, alias] : dialects) {
      if (key.first.empty())
        throw std::invalid_argument(
            "cannot print dialect import without its definition source");
      out << "from "
          << quote(std::filesystem::absolute(key.first)
                       .lexically_normal()
                       .string())
          << " import " << key.second << " as " << alias << "\n";
    }
    if (!dialects.empty()) out << '\n';
  }
  std::map<std::size_t, std::string> names;
  std::set<std::string> taken;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i)
    if (graph.nodes[i].kind == core::ConcreteNode::Kind::kInput) {
      names[i] = graph.nodes[i].text;
      taken.insert(names[i]);
    }
  for (const auto& [name, id] : graph.bindings)
    if (live.contains(id) && !names.contains(id) && taken.insert(name).second)
      names[id] = name;
  for (auto i : live)
    if (!names.contains(i)) {
      auto name = "v" + std::to_string(i);
      while (!taken.insert(name).second) name += "_";
      names[i] = name;
    }
  out << "graph " << graph.name << " {\n";
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto& n = graph.nodes[i];
    if (n.kind != core::ConcreteNode::Kind::kInput) continue;
    out << "    input " << names.at(i);
    if (n.dtype || n.shape) {
      out << ": ";
      if (n.dtype) out << core::dtypeName(*n.dtype);
      if (n.shape) {
        out << '[';
        for (std::size_t k = 0; k < n.shape->size(); ++k) {
          if (k) out << ", ";
          out << n.shape->at(k);
        }
        out << ']';
      }
    }
    out << "\n";
  }
  out << '\n';
  for (auto i : live) {
    const auto& n = graph.nodes[i];
    if (n.kind == core::ConcreteNode::Kind::kInput) continue;
    out << "    let " << names.at(i) << " = ";
    if (n.kind == core::ConcreteNode::Kind::kLiteral) {
      out << n.text;
      if (n.dtype) out << ':' << core::dtypeName(*n.dtype);
    } else {
      const auto& op = p.operations.at(n.operation->value);
      out << '('
          << (imports
                  ? dialects.at({op.origin.definition.source_name, op.dialect})
                  : op.dialect)
          << '.' << op.name;
      if (!n.attributes.empty()) {
        out << '[';
        const auto& fields =
            p.attribute_schemas.at(op.attributes->value).fields;
        for (std::size_t a = 0; a < n.attributes.size(); ++a) {
          if (a) out << ", ";
          out << fields[a].name << " = " << attribute(n.attributes[a]);
        }
        out << ']';
      }
      for (auto a : n.operands) out << ' ' << names.at(a);
      out << ')';
    }
    out << "\n";
  }
  out << "    yield " << names.at(graph.root) << "\n}\n";
  return out.str();
}

}  // namespace tepl::rewrite
