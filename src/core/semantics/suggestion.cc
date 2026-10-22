#include "src/core/semantics/suggestion.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "src/shared/metadata/ordered.h"

namespace tepl::core::semantics {
namespace {
using B = builtins::Builtin;
class Suggestion {
 public:
  Suggestion(const Plan& plan, const Rule& rule, const ProgramTables& tables)
      : p(plan), r(rule), tables(tables) {
    for (const auto& c : r.captures) used.insert(c.name);
    for (const auto& l : r.locals) used.insert(l.name);
    for (const auto& d : r.dimensions) used.insert(d.name);
    for (const auto& d : r.dtypes) used.insert(d.name);
    for (const auto& n : p.nodes)
      if (!n.label.empty()) used.insert(n.label);
    std::set<std::string> locals;
    for (const auto& l : r.locals)
      local_names[l.id.value] =
          locals.insert(l.name).second ? l.name : fresh(l.name);
    for (std::size_t i = 0; i < p.nodes.size(); ++i)
      if (p.nodes[i].definition) {
        auto name = p.nodes[i].label;
        if (name.empty())
          name = fresh(i == p.lhs ? "source" : i == p.rhs ? "result" : "node");
        names[i] = name;
      }
  }
  std::string run() {
    auto left = graph(p.lhs, false);
    auto right = graph(p.rhs, true);
    std::vector<std::string> checks{"axis_domain()"};
    if (p.proof.inference->mode == inference::Mode::Synthesize) {
      propagate(p.rhs, names.at(p.lhs) + ".axes.order", checks);
    } else {
      checks.push_back(names.at(p.rhs) + ".legal");
      checks.push_back(names.at(p.lhs) + ".axes.partition == " +
                       names.at(p.rhs) + ".axes.partition");
      checks.push_back(names.at(p.lhs) + ".axes.order == " + names.at(p.rhs) +
                       ".axes.order");
    }
    std::ostringstream out;
    out << "rule " << r.name << " {\n";
    for (const auto& dtype : r.dtypes) out << "  dtype " << dtype.name << ";\n";
    for (const auto& c : r.constraints) {
      out << "  " << capture(c.capture.value) << ": ";
      if (c.dtype)
        std::visit(
            [&](auto d) {
              using T = decltype(d);
              if constexpr (std::is_same_v<T, DType>)
                out << dtypeName(d);
              else
                out << r.dtypes.at(d.value).name;
            },
            *c.dtype);
      out << "[";
      for (std::size_t i = 0; i < c.shape.size(); ++i) {
        if (i) out << ", ";
        auto e = c.shape[i];
        if (e.symbol)
          out << r.dimensions.at(e.symbol->value).name;
        else if (e.kind == ShapeElement::Kind::kWildcard)
          out << "_";
        if (e.kind == ShapeElement::Kind::kSequence) out << "...";
      }
      out << "]\n";
    }
    out << "  " << left << "\n  =>\n  " << right << "\n  where {\n";
    // Preserve the user's restrictions and where-local scope.
    std::vector<std::string> original_checks;
    for (const auto& c : r.conditions) {
      if (auto* l = std::get_if<LocalLet>(&c->value))
        out << "    let " << local_names.at(l->local.value) << " = "
            << expression(*l->value) << ";\n";
      else
        original_checks.push_back(expression(*c));
    }
    checks.insert(checks.begin(), original_checks.begin(),
                  original_checks.end());
    for (const auto& binding : bindings) out << "    " << binding << "\n";
    for (std::size_t i = 0; i < checks.size(); ++i)
      out << "    (" << checks[i] << ")"
          << (i + 1 < checks.size() ? " &&\n" : "\n");
    out << "  }\n";
    bool derives = !r.derivations.empty();
    if (derives) {
      out << "  derive {\n";
      for (const auto& d : r.derivations) {
        if (!d.target) {
          const auto& l = std::get<LocalLet>(d.value->value);
          out << "    let " << local_names.at(l.local.value) << " = "
              << expression(*l.value) << ";\n";
        } else {
          for (const auto& [id, node] : names)
            if (isRhs(id) && p.nodes[id].descriptor == d.target &&
                !sourceDescriptors.contains(d.target->value))
              out << "    " << node << ".attrs = " << expression(*d.value)
                  << ";\n";
        }
      }
      out << "  }\n";
    }
    out << "}";
    return out.str();
  }

 private:
  const Plan& p;
  const Rule& r;
  const ProgramTables& tables;
  std::map<std::size_t, std::string> names, sourceDescriptors, rhsDescriptors;
  std::map<std::size_t, std::string> required, local_names;
  std::set<std::string> used;
  std::vector<std::string> bindings;
  std::string fresh(std::string base) {
    for (auto& c : base)
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    auto result = base;
    for (std::size_t i = 2; !used.insert(result).second; ++i)
      result = base + "_" + std::to_string(i);
    return result;
  }
  std::string capture(std::size_t c) const { return r.captures.at(c).name; }
  bool isRhs(std::size_t id) const {
    for (const auto& n : p.proof.inference->replacement)
      if (n.node == id) return true;
    return false;
  }
  std::string graph(std::size_t id, bool rhs) {
    const auto& n = p.nodes.at(id);
    if (n.capture) return capture(n.capture->value);
    if (n.literal) return *n.literal;
    std::string result = "(" + names.at(id) + ": " + n.operation_name;
    if (n.descriptor) {
      auto d = n.descriptor->value;
      if (!rhs) {
        if (sourceDescriptors.contains(d))
          result += "[" + sourceDescriptors.at(d) + ".attrs]";
        else
          sourceDescriptors[d] = names.at(id);
      } else {
        rhsDescriptors.try_emplace(d, names.at(id));
        if (sourceDescriptors.contains(d))
          result += "[" + sourceDescriptors.at(d) + ".attrs]";
      }
    }
    for (auto child : n.inputs) result += " " + graph(child, rhs);
    return result + ")";
  }
  std::string member(std::size_t id, std::size_t role) const {
    const auto& n = p.nodes.at(id);
    auto c = *axis::recognizeOrdered(*n.definition);
    auto group = c.kind == axis::OrderedContract::Kind::Permutation ? 0
                 : role == 0 ? c.batch_group
                 : role == 1 ? c.contract_group
                 : role == 2 ? c.left_free_group
                             : c.right_free_group;
    return names.at(id) + ".axes." + n.definition->groups.at(group).name +
           ".members";
  }
  std::string bind(const std::string& base, const std::string& value) {
    auto name = fresh(base);
    bindings.push_back("let " + name + " = " + value + ";");
    return name;
  }
  void propagate(std::size_t id, const std::string& q,
                 std::vector<std::string>& checks) {
    const auto& n = p.nodes[id];
    required[id] = q;
    if (n.capture) {
      checks.push_back("subsequence(" + q + ", " + capture(n.capture->value) +
                       ".axes.order)");
      return;
    }
    auto c = *axis::recognizeOrdered(*n.definition);
    if (c.kind == axis::OrderedContract::Kind::Permutation) {
      propagate(n.inputs[0], "[]", checks);
      return;
    }
    auto wanted = bind(id == p.rhs ? "wanted" : names.at(id) + "_wanted", q);
    required[id] = wanted;
    auto prefix = id == p.rhs ? std::string{} : names.at(id) + "_";
    auto batches = bind(prefix + "batches",
                        "filter(" + wanted + ", " + member(id, 0) + ")");
    auto leftName = p.nodes[n.inputs[0]].capture
                        ? "from" + capture(p.nodes[n.inputs[0]].capture->value)
                        : "from_left";
    auto rightName = p.nodes[n.inputs[1]].capture
                         ? "from" + capture(p.nodes[n.inputs[1]].capture->value)
                         : "from_right";
    auto left = bind(prefix + leftName,
                     "filter(" + wanted + ", " + member(id, 2) + ")");
    auto right = bind(prefix + rightName,
                      "filter(" + wanted + ", " + member(id, 3) + ")");
    checks.push_back(wanted + " == concat(" + batches + ", " + left + ", " +
                     right + ")");
    propagate(n.inputs[0], left, checks);
    propagate(n.inputs[1], right, checks);
  }
  std::string descriptor(std::size_t id) {
    if (sourceDescriptors.contains(id))
      return sourceDescriptors.at(id) + ".attrs";
    if (rhsDescriptors.contains(id)) return rhsDescriptors.at(id) + ".attrs";
    throw std::runtime_error("descriptor has no printable node binding");
  }
  std::string query(const BuiltinCall& v) {
    auto integer = [&](std::size_t i) {
      return static_cast<std::size_t>(std::stoull(
          std::get<NumericConstant>(v.arguments.at(i)->value).spelling));
    };
    if (v.builtin == B::kAxisSourceOrder)
      return names.at(p.lhs) + ".axes.order";
    if (v.builtin == B::kAxisRhsOrder) return names.at(p.rhs) + ".axes.order";
    if (v.builtin == B::kAxisRhsLegal) return names.at(p.rhs) + ".legal";
    if (v.builtin == B::kAxisSourcePartition)
      return names.at(p.lhs) + ".axes.partition";
    if (v.builtin == B::kAxisRhsPartition)
      return names.at(p.rhs) + ".axes.partition";
    if (v.builtin == B::kAxisCaptureOrder)
      return capture(integer(0)) + ".axes.order";
    if (v.builtin == B::kAxisSourceNodeOrder ||
        v.builtin == B::kAxisRhsNodeOrder)
      return names.at(integer(0)) + ".axes.order";
    if (v.builtin == B::kAxisRequired) {
      auto id = integer(0);
      if (p.nodes.at(id).capture) return required.at(id);
      return names.at(id) + ".required_order";
    }
    if (v.builtin == B::kAxisInputOrder)
      return capture(p.nodes.at(integer(0)).capture->value) + ".axes.order";
    if (v.builtin == B::kAxisBatch || v.builtin == B::kAxisContract ||
        v.builtin == B::kAxisLeftFree || v.builtin == B::kAxisRightFree ||
        v.builtin == B::kAxisLive)
      return member(integer(0), v.builtin == B::kAxisBatch       ? 0
                                : v.builtin == B::kAxisContract  ? 1
                                : v.builtin == B::kAxisLeftFree  ? 2
                                : v.builtin == B::kAxisRightFree ? 3
                                                                 : 4);
    if (v.builtin == B::kAxisSourceGroupOrder ||
        v.builtin == B::kAxisSourceGroupMembers ||
        v.builtin == B::kAxisRhsGroupOrder ||
        v.builtin == B::kAxisRhsGroupMembers ||
        v.builtin == B::kAxisRhsGroupPositions) {
      auto base = member(integer(0), integer(1));
      base.resize(base.size() - std::string("members").size());
      return base + (v.builtin == B::kAxisSourceGroupMembers ||
                             v.builtin == B::kAxisRhsGroupMembers
                         ? "members"
                     : v.builtin == B::kAxisRhsGroupPositions ? "positions"
                                                              : "order");
    }
    std::string result(builtins::builtinName(v.builtin));
    result += "(";
    for (std::size_t i = 0; i < v.arguments.size(); ++i) {
      if (i) result += ", ";
      result += expression(*v.arguments[i]);
    }
    return result + ")";
  }
  std::string expression(const TypedExpr& e) {
    return std::visit(
        [&](const auto& v) -> std::string {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, NumericConstant>)
            return v.spelling;
          else if constexpr (std::is_same_v<T, bool>)
            return v ? "true" : "false";
          else if constexpr (std::is_same_v<T, CaptureRef>)
            return capture(v.capture.value);
          else if constexpr (std::is_same_v<T, DimensionRef>)
            return r.dimensions.at(v.dimension.value).name;
          else if constexpr (std::is_same_v<T, DTypeVariableRef>)
            return r.dtypes.at(v.variable.value).name;
          else if constexpr (std::is_same_v<T, DType>)
            return std::string(dtypeName(v));
          else if constexpr (std::is_same_v<T, DescriptorRef>)
            return descriptor(v.descriptor.value);
          else if constexpr (std::is_same_v<T, DescriptorFieldRef>) {
            return descriptor(v.descriptor.value) + "." +
                   tables.attribute_schemas.at(v.schema.value)
                       .fields.at(v.field)
                       .name;
          } else if constexpr (std::is_same_v<T, OutputAxisGroupRef>)
            return v.node + ".axes." + v.group + ".positions";
          else if constexpr (std::is_same_v<T, BuiltinCall>)
            return query(v);
          else if constexpr (std::is_same_v<T, BinaryExpr>)
            return "(" + expression(*v.lhs) + " " +
                   std::string(spelling(v.op)) + " " + expression(*v.rhs) + ")";
          else if constexpr (std::is_same_v<T, UnaryExpr>)
            return std::string(spelling(v.op)) + "(" + expression(*v.operand) +
                   ")";
          else if constexpr (std::is_same_v<T, LocalRef>)
            return local_names.at(v.local.value);
          else if constexpr (std::is_same_v<T, ListExpr>) {
            std::string result = "[";
            for (std::size_t i = 0; i < v.elements.size(); ++i) {
              if (i) result += ", ";
              result += expression(*v.elements[i]);
            }
            return result + "]";
          } else if constexpr (std::is_same_v<T, DescriptorConstruct>) {
            std::string result = "{";
            for (std::size_t i = 0; i < v.fields.size(); ++i) {
              if (i) result += ", ";
              result += tables.attribute_schemas.at(v.schema.value)
                            .fields.at(i)
                            .name +
                        ": " + expression(*v.fields[i]);
            }
            return result + "}";
          } else if constexpr (std::is_same_v<T, IndexExpr>)
            return expression(*v.value) + "[" + expression(*v.index) + "]";
          else if constexpr (std::is_same_v<T, ConditionalExpr>)
            return "if " + expression(*v.condition) + " then " +
                   expression(*v.then_value) + " else " +
                   expression(*v.else_value);
          else if constexpr (std::is_same_v<T, ComprehensionExpr>)
            return "[" + expression(*v.element) + " for " +
                   local_names.at(v.variable.value) + " in " +
                   expression(*v.iterable) + "]";
          else if constexpr (std::is_same_v<T, StringConstant>) {
            std::ostringstream out;
            out << std::quoted(v.value);
            return out.str();
          } else if constexpr (std::is_same_v<T, EnumConstant>)
            return v.value;
          else if constexpr (std::is_same_v<T, NoneConstant>)
            return "none";
          else if constexpr (std::is_same_v<T, OptionalValue>)
            return expression(*v.value);
          else
            throw std::runtime_error(
                "unprintable source condition in generated suggestion");
        },
        e.value);
  }
};
}  // namespace
std::string suggestRule(const Plan& p, const Rule& r,
                        const ProgramTables& tables) {
  return Suggestion(p, r, tables).run();
}
}  // namespace tepl::core::semantics
