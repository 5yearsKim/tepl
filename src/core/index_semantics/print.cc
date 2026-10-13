#include <functional>
#include <sstream>

#include "src/core/index_semantics/ir.h"
namespace tepl::core::index_semantics {
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
  if (v.kind == Value::Kind::Operand) return d.operands.at(v.index);
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
std::string format(const Symbol& s) {
  if (s.kind == Symbol::Kind::AxisGroup) return s.name;
  if (s.kind == Symbol::Kind::Field || s.kind == Symbol::Kind::Hole)
    return std::string(s.kind == Symbol::Kind::Hole ? "auto#" : "@#") +
           std::to_string(s.owner) + "." + s.name;
  if (s.kind == Symbol::Kind::Shape || s.kind == Symbol::Kind::DType)
    return std::string(s.kind == Symbol::Kind::Shape ? "shape" : "dtype") +
           "(Input(#" + std::to_string(s.owner) + "))";
  if (s.kind == Symbol::Kind::Integer || s.kind == Symbol::Kind::Boolean)
    return s.name;
  std::string out = s.kind == Symbol::Kind::List ? "[" : s.name + "(";
  for (std::size_t i = 0; i < s.args.size(); ++i) {
    if (i) out += ", ";
    out += format(*s.args[i]);
  }
  return out + (s.kind == Symbol::Kind::List ? "]" : ")");
}
std::string format(const Plan& p) {
  auto label = [&](const Symbol& s) {
    auto text = format(s);
    for (const auto& [id, name] : p.descriptor_names) {
      const auto from = "@#" + std::to_string(id) + ".";
      const auto to = "@" + name + ".";
      std::size_t pos = 0;
      while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
      }
    }
    return text;
  };
  std::map<std::string, std::string> aliases;
  std::vector<std::pair<std::string, std::string>> declarations;
  std::function<std::string(const Symbol&)> render =
      [&](const Symbol& s) -> std::string {
    if (s.kind != Symbol::Kind::Call && s.kind != Symbol::Kind::List)
      return label(s);
    auto id = s.structuralKey();
    if (!id.empty() && aliases.contains(id)) return aliases.at(id);
    std::string text = s.kind == Symbol::Kind::List ? "[" : s.name + "(";
    for (std::size_t i = 0; i < s.args.size(); ++i) {
      if (i) text += ", ";
      text += render(*s.args[i]);
    }
    text += s.kind == Symbol::Kind::List ? "]" : ")";
    if (s.kind == Symbol::Kind::List || text.size() < 90) return text;
    auto name = "m" + std::to_string(declarations.size());
    declarations.emplace_back(name, text);
    if (!id.empty()) aliases[id] = name;
    return name;
  };
  std::function<std::string(std::size_t)> graph = [&](std::size_t n) {
    const auto& node = p.nodes.at(n);
    if (node.capture)
      return "Input(#" + std::to_string(node.capture->value) + ")";
    std::function<std::string(const Value&)> value = [&](const Value& v) {
      if (v.kind == Value::Kind::Operand)
        return "align[n" + std::to_string(n) + ":" +
               node.definition->operands.at(v.index) + "](" +
               graph(node.inputs.at(v.index)) + ")";
      std::string s = v.kind == Value::Kind::Add       ? "Add("
                      : v.kind == Value::Kind::Product ? "Product("
                                                       : "Reduce(";
      if (v.kind == Value::Kind::Reduce)
        s += "n" + std::to_string(n) + "." +
             node.definition->groups.at(v.index).name + ", ";
      for (std::size_t i = 0; i < v.args.size(); ++i) {
        if (i) s += ", ";
        s += value(*v.args[i]);
      }
      return s + ")";
    };
    return value(*node.definition->value);
  };
  std::ostringstream out;
  out << "    index-inference: policy="
      << (p.algebraic ? "algebraic" : "strict") << " status="
      << (p.proof.status == ProofStatus::Proved ? "proved" : "unknown") << '\n';
  out << "      lhs-graph: " << graph(p.lhs) << '\n';
  out << "      rhs-graph: " << graph(p.rhs) << '\n';
  for (const auto& step : p.proof.steps) out << "      " << step << '\n';
  if (!p.proof.lhs_normalized.empty()) {
    out << "      lhs-normalized: " << p.proof.lhs_normalized << '\n';
    out << "      rhs-normalized: " << p.proof.rhs_normalized << '\n';
  }
  for (const auto& [id, fields] : p.proof.constructors) {
    const Definition* definition = nullptr;
    for (const auto& n : p.nodes)
      if (n.descriptor && n.descriptor->value == id && n.definition)
        definition = n.definition.get();
    out << "      constructor auto#" << id << ":\n";
    for (std::size_t f = 0; f < fields.size(); ++f)
      out << "        " << definition->fields.at(f) << " = "
          << render(*fields[f]) << '\n';
  }
  out << "      declared-conditions:" << (p.assumptions.empty() ? " true" : "")
      << '\n';
  for (const auto& a : p.assumptions) out << "        " << render(*a) << '\n';
  out << "      additional-guard: "
      << (p.proof.status == ProofStatus::Unknown ? "unresolved"
          : p.proof.guard.empty()                ? "true"
                                                 : "")
      << '\n';
  for (const auto& g : p.proof.guard) out << "        " << render(*g) << '\n';
  if (!p.proof.guard_origins.empty()) {
    out << "      guard-origins (unresolved obligations):\n";
    for (const auto& g : p.proof.guard_origins)
      out << "        [" << guardKindName(g.kind) << "] "
          << render(*g.condition) << '\n';
  }
  out << "      reason: " << p.proof.reason << '\n';
  std::ostringstream bindings;
  if (!p.proof.axis_definitions.empty())
    bindings << "      coordinate frame (fresh identities; domain lengths "
                "remain symbolic):\n";
  for (const auto& g : p.proof.axis_definitions)
    if (g->kind == Symbol::Kind::AxisGroup)
      bindings << "        " << g->name
               << " = group(extents=" << render(*g->args[0]) << ")\n";
  for (const auto& [id, axes] : p.proof.accesses)
    bindings << "        access@" << id << " = " << render(*axes) << '\n';
  if (p.proof.lhs_axes)
    bindings << "        lhs-output = " << render(*p.proof.lhs_axes) << '\n'
             << "        rhs-output = " << render(*p.proof.rhs_axes) << '\n';
  for (const auto& [name, value] : declarations)
    out << "      let " << name << " = " << value << '\n';
  out << bindings.str();
  return out.str();
}
}  // namespace tepl::core::index_semantics
