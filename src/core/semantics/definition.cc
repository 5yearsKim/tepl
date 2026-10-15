#include "src/core/semantics/definition.h"

#include <functional>
#include <sstream>
namespace tepl::core::semantics {
std::string format(const Meta& m) {
  if (m.kind == Meta::Kind::Attribute) return "attrs." + m.name;
  if (m.kind == Meta::Kind::List) {
    std::string s = "[";
    for (std::size_t i = 0; i < m.args.size(); ++i) {
      if (i) s += ",";
      s += format(*m.args[i]);
    }
    return s + "]";
  }
  if (m.kind != Meta::Kind::Call) return m.name;
  std::string s = m.name + "(";
  for (std::size_t i = 0; i < m.args.size(); ++i) {
    if (i) s += ", ";
    s += format(*m.args[i]);
  }
  return s + ")";
}
std::string format(const Value& v, const Definition& d) {
  if (v.kind == Value::Kind::Const)
    return "Const(" + std::to_string(v.constant) + ")";
  if (v.kind == Value::Kind::Operand) return d.operands.at(v.index);
  if (v.kind == Value::Kind::Read) {
    std::function<std::string(const IndexValue&)> index =
        [&](const IndexValue& a) {
          if (a.kind == IndexValue::Kind::Coordinates)
            return "Coords(" + d.groups.at(a.group).name + ")";
          if (a.kind == IndexValue::Kind::Scale ||
              a.kind == IndexValue::Kind::Offset)
            return std::string(a.kind == IndexValue::Kind::Scale
                                   ? "IndexScale("
                                   : "IndexOffset(") +
                   format(*a.parameter) + ", " + index(*a.args[0]) + ")";
          std::string text =
              a.kind == IndexValue::Kind::Add ? "IndexAdd(" : "Concat(";
          for (std::size_t i = 0; i < a.args.size(); ++i) {
            if (i) text += ", ";
            text += index(*a.args[i]);
          }
          return text + ")";
        };
    return "Read(" + d.operands.at(v.index) + ", " + index(*v.access) + ")";
  }
  std::string s = v.kind == Value::Kind::Add       ? "Add("
                  : v.kind == Value::Kind::Product ? "Product("
                                                   : "Reduce(";
  if (v.kind == Value::Kind::Reduce) s += d.groups.at(v.index).name + ", ";
  for (std::size_t i = 0; i < v.args.size(); ++i) {
    if (i) s += ", ";
    s += format(*v.args[i], d);
  }
  return s + ")";
}
std::string format(const Definition& d) {
  std::ostringstream out;
  for (const auto& g : d.groups) {
    if (g.domain) {
      out << "    bind " << g.name << " = group(" << format(*g.domain) << ")\n";
      continue;
    }
    out << "    bind " << g.name << " { ";
    for (std::size_t i = 0; i < g.bindings.size(); ++i) {
      if (i) out << ", ";
      const auto& b = g.bindings[i];
      out << d.operands.at(b.operand) << ": " << format(*b.positions);
    }
    out << " }\n";
  }
  for (const auto& a : d.assertions) out << "    assert " << format(*a) << '\n';
  out << "    output-axes: " << format(*d.output) << '\n';
  out << "    semantics [index]: " << format(*d.value, d) << '\n';
  return out.str();
}
}  // namespace tepl::core::semantics
