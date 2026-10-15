#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/index_semantics/context.h"
#include "src/core/index_semantics/graph.h"
#include "src/program/program.h"

namespace tepl::program {
Program compile(const core::Program& checked) {
  Program output;
  output.ir_ = checked;
  output.axes_.resize(checked.rules.size());
  auto typeId = [&](core::Type type) {
    for (std::size_t i = 0; i < output.ir_.types.size(); ++i)
      if (output.ir_.types[i] == type) return core::TypeId{i};
    output.ir_.types.push_back(type);
    return core::TypeId{output.ir_.types.size() - 1};
  };
  for (std::size_t i = 0; i < output.ir_.rules.size(); ++i) {
    auto& r = output.ir_.rules[i];
    const auto plan = r.semantic_plan;
    if (plan) {
      if (plan->proof.status != core::index_semantics::ProofStatus::Proved)
        throw std::invalid_argument("rule '" + r.name +
                                    "' has no verified semantic plan");
      if (plan->algebraic) output.policy_ = "algebraic";
      output.axes_[i] = plan->proof.axis_definitions;
    }
    std::unique_ptr<core::index_semantics::Context> coordinates;
    std::function<core::TypedExprPtr(const core::TypedExprPtr&)> lower;
    lower = [&](const core::TypedExprPtr& e) -> core::TypedExprPtr {
      auto copy = std::make_shared<core::TypedExpr>(*e);
      std::visit(
          [&](auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, core::OutputAxisGroupRef>) {
              if (!plan)
                throw std::invalid_argument(
                    "output group requires semantic plan");
              if (!coordinates) {
                coordinates = std::make_unique<core::index_semantics::Context>(
                    *plan, 1000000);
                coordinates->sourceScope(plan->lhs);
                core::index_semantics::Lowerer source{*coordinates};
                source.lower(plan->lhs);
              }
            } else if constexpr (std::is_same_v<T, core::BuiltinCall>) {
              for (auto& a : v.arguments) a = lower(a);
            } else if constexpr (std::is_same_v<T, core::UnaryExpr>)
              v.operand = lower(v.operand);
            else if constexpr (std::is_same_v<T, core::BinaryExpr>) {
              v.lhs = lower(v.lhs);
              v.rhs = lower(v.rhs);
            } else if constexpr (std::is_same_v<T, core::LocalLet> ||
                                 std::is_same_v<T, core::OptionalValue>)
              v.value = lower(v.value);
            else if constexpr (std::is_same_v<T, core::ListExpr>) {
              for (auto& a : v.elements) a = lower(a);
            } else if constexpr (std::is_same_v<T, core::IndexExpr>) {
              v.value = lower(v.value);
              v.index = lower(v.index);
            } else if constexpr (std::is_same_v<T, core::ConditionalExpr>) {
              v.condition = lower(v.condition);
              v.then_value = lower(v.then_value);
              v.else_value = lower(v.else_value);
            } else if constexpr (std::is_same_v<T, core::ComprehensionExpr>) {
              v.element = lower(v.element);
              v.iterable = lower(v.iterable);
            } else if constexpr (std::is_same_v<T, core::DescriptorConstruct>) {
              for (auto& a : v.fields) a = lower(a);
            }
          },
          copy->value);
      if (const auto* ref = std::get_if<core::OutputAxisGroupRef>(&e->value)) {
        std::optional<std::size_t> node;
        for (auto id : coordinates->source().nodes)
          if (plan->nodes[id].descriptor == ref->descriptor) {
            if (node) throw std::invalid_argument("ambiguous output group");
            node = id;
          }
        if (!node)
          throw std::invalid_argument("missing output group operation");
        const auto& def = *plan->nodes[*node].definition;
        bool found = false;
        for (std::size_t g = 0; g < def.groups.size(); ++g)
          if (def.groups[g].name == ref->group) {
            copy->value = core::SymbolicExpr{coordinates->positions(
                coordinates->source().tensor_axes.at(*node),
                coordinates->source().tensor_groups.at(*node).at(g))};
            found = true;
            break;
          }
        if (!found) throw std::invalid_argument("unknown output group");
      }
      return copy;
    };
    for (auto& e : r.conditions) e = lower(e);
    if (plan)
      for (const auto& guard : plan->proof.guard)
        r.conditions.push_back(std::make_shared<core::TypedExpr>(
            core::TypedExpr{r.origin, typeId(core::Type{core::TypeKind::kBool}),
                            core::SymbolicExpr{guard}}));
    for (auto& d : r.derivations) d.value = lower(d.value);
    for (const auto& hole : r.attribute_holes) {
      if (!plan || !plan->proof.constructors.contains(hole.descriptor.value))
        throw std::invalid_argument("auto constructor is unresolved in rule '" +
                                    r.name + "'");
      const auto& schema = checked.attribute_schemas.at(hole.schema.value);
      const auto& values = plan->proof.constructors.at(hole.descriptor.value);
      if (values.size() != schema.fields.size())
        throw std::invalid_argument("incomplete auto constructor");
      core::DescriptorConstruct ctor{hole.schema, {}};
      for (std::size_t f = 0; f < values.size(); ++f) {
        const auto& field = schema.fields[f];
        auto type = core::resolveType(field.type)
                        .value_or(core::Type{core::TypeKind::kOpaque});
        if (type.kind == core::TypeKind::kOpaque) type.opaque_name = field.type;
        for (std::size_t d = 0; d < field.list_depth; ++d)
          type = core::listType(type);
        type.optional = field.optional;
        ctor.fields.push_back(std::make_shared<core::TypedExpr>(core::TypedExpr{
            r.origin, typeId(type), core::SymbolicExpr{values[f]}}));
      }
      auto& descriptor = r.descriptors.at(hole.descriptor.value);
      descriptor.kind = core::Descriptor::Kind::kDerived;
      r.derivations.push_back(
          {hole.descriptor,
           std::make_shared<core::TypedExpr>(
               core::TypedExpr{r.origin, descriptor.type, std::move(ctor)}),
           r.origin});
    }
    if (coordinates)
      for (const auto& [key, value] : coordinates->source().groups)
        output.axes_[i].push_back(value);
    auto& axes = output.axes_[i];
    axes.erase(std::remove_if(
                   axes.begin(), axes.end(),
                   [](const auto& s) {
                     return !s ||
                            s->kind !=
                                core::index_semantics::Symbol::Kind::AxisGroup;
                   }),
               axes.end());
    r.attribute_holes.clear();
    r.semantic_plan.reset();
    r.inference.reset();
  }
  validate(output);
  return output;
}
}  // namespace tepl::program
