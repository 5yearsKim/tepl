#include <functional>
#include <sstream>

#include "src/core/semantics/ir.h"
#include "src/shared/catalog/axis_conditions.h"
namespace tepl::core::semantics {
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
      std::string to = "@" + name + ".";
      for (const auto& node : p.nodes)
        if (node.descriptor && node.descriptor->value == id &&
            !node.label.empty()) {
          to = node.label + ".attrs.";
          break;
        }
      std::size_t pos = 0;
      while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
      }
    }
    return text;
  };
  std::map<symbolic::Key, std::string> aliases;
  std::map<symbolic::Key, std::pair<bool, bool>> position_matches;
  std::function<void(const Symbol&)> collect_matches = [&](const Symbol& s) {
    if (s.kind == Symbol::Kind::Call && s.args.size() == 2 &&
        (s.name == "membership_positions" || s.name == "matched_positions") &&
        !s.args[0]->structuralKey().empty() &&
        !s.args[1]->structuralKey().empty()) {
      auto match_key =
          s.args[0]->structuralKey() + ":" + s.args[1]->structuralKey();
      auto& found = position_matches[match_key];
      if (s.name == "membership_positions")
        found.first = true;
      else
        found.second = true;
    }
    for (const auto& arg : s.args) collect_matches(*arg);
  };
  for (const auto& [_, fields] : p.proof.constructors)
    for (const auto& field : fields) collect_matches(*field);
  for (auto it = position_matches.begin(); it != position_matches.end();) {
    if (!it->second.first || !it->second.second)
      it = position_matches.erase(it);
    else
      ++it;
  }
  std::map<symbolic::Key, std::string> match_names;
  std::vector<std::pair<std::string, std::string>> declarations;
  std::function<std::string(const Symbol&)> render =
      [&](const Symbol& s) -> std::string {
    if (s.kind != Symbol::Kind::Call && s.kind != Symbol::Kind::List)
      return label(s);
    auto id = s.structuralKey();
    if (!id.empty() && aliases.contains(id)) return aliases.at(id);
    if (s.kind == Symbol::Kind::Call && s.args.size() == 2 &&
        (s.name == "membership_positions" || s.name == "matched_positions") &&
        !s.args[0]->structuralKey().empty() &&
        !s.args[1]->structuralKey().empty()) {
      auto match_key =
          s.args[0]->structuralKey() + ":" + s.args[1]->structuralKey();
      if (position_matches.contains(match_key)) {
        auto found = match_names.find(match_key);
        if (found == match_names.end()) {
          auto name = "p" + std::to_string(match_names.size());
          match_names.emplace(match_key, name);
          declarations.emplace_back(name, "match_positions(" +
                                              render(*s.args[0]) + ", " +
                                              render(*s.args[1]) + ")");
          found = match_names.find(match_key);
        }
        return found->second + (s.name == "membership_positions"
                                    ? ".left_positions"
                                    : ".right_positions");
      }
    }
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
    if (node.literal) return std::string("Literal(") + *node.literal + ")";
    if (node.capture)
      return "Input(#" + std::to_string(node.capture->value) + ")";
    std::function<std::string(const Value&)> value = [&](const Value& v) {
      if (v.kind == Value::Kind::Const)
        return "Const(" + std::to_string(v.constant) + ")";
      if (v.kind == Value::Kind::Read)
        return format(v, *node.definition) + " over " +
               graph(node.inputs.at(v.index));
      if (v.kind == Value::Kind::Operand)
        return "align[n" + std::to_string(n) + ":" +
               node.definition->operands.at(v.index) + "](" +
               graph(node.inputs.at(v.index)) + ")";
      std::string s = v.kind == Value::Kind::Add       ? "Add("
                      : v.kind == Value::Kind::Product ? "Product("
                                                       : "Reduce(";
      if (v.kind == Value::Kind::Call)
        s = std::string(scalarSignature(v.function).name) + "(";
      if (v.kind == Value::Kind::Reduce)
        s += "n" + std::to_string(n) + "." +
             node.definition->groups.at(v.index).name + ", ";
      for (std::size_t i = 0; i < v.args.size(); ++i) {
        if (i) s += ", ";
        s += value(*v.args[i]);
      }
      for (const auto& parameter : v.parameters) s += ", " + format(*parameter);
      return s + ")";
    };
    return value(*node.definition->value);
  };
  std::ostringstream out;
  out << "    semantic-inference: policy="
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
      << (p.proof.status == ProofStatus::Unknown ? "unresolved" : "true")
      << '\n';
  if (!p.suggested_rule.empty())
    out << "      suggested-rule:\n" << p.suggested_rule << "\n";
  for (const auto& g : p.proof.guard) out << "        " << render(*g) << '\n';
  if (!p.proof.guard_origins.empty()) {
    out << "      guard-origins (unresolved obligations):\n";
    for (const auto& g : p.proof.guard_origins)
      out << "        [" << guardKindName(g.kind) << "] "
          << render(*g.condition) << '\n';
  }
  if (!p.suggested_rule.empty())
    out << "      suggestion: verified sufficient restriction; bounded search, "
           "not necessarily weakest\n";
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
}  // namespace tepl::core::semantics
