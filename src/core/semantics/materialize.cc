#include "src/core/semantics/materialize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/semantics/completion.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/graph.h"
#include "src/shared/catalog/axis_conditions.h"
#include "src/shared/catalog/core_guard.h"
#include "src/inference/extensions.h"
namespace tepl::core::semantics {
Materialized materialize(InspectionData checked) {
  Materialized output;
  static_cast<ProgramTables&>(output.ir) =
      std::move(static_cast<ProgramTables&>(checked));
  output.axes.resize(checked.rules.size());
  for (std::size_t i = 0; i < checked.rules.size(); ++i) {
    auto& r = checked.rules[i];
    const auto plan = r.semantic_plan;
    std::vector<symbolic::SymbolPtr> axes;
    output.kinds.push_back(plan ? VerificationKind::Semantic
                                : VerificationKind::Structural);
    if (plan) {
      if (plan->proof.status != core::semantics::ProofStatus::Proved)
        throw std::invalid_argument("rule '" + r.name +
                                    "' has no verified semantic plan");
      axes = plan->proof.axis_definitions;
    }
    std::unique_ptr<core::semantics::Context> coordinates;
    Completion completion(
        [&](const OutputAxisGroupRef& ref) -> symbolic::SymbolPtr {
          if (!plan)
            throw std::invalid_argument("output group requires semantic plan");
          if (!coordinates) {
            coordinates = std::make_unique<Context>(*plan, 1000000);
            coordinates->sourceScope(plan->lhs);
            Lowerer source{*coordinates};
            source.lower(plan->lhs);
          }
          std::optional<std::size_t> node;
          for (auto id : coordinates->source().nodes)
            if (plan->nodes[id].label == ref.node) {
              if (node) throw std::invalid_argument("ambiguous output group");
              node = id;
            }
          if (!node)
            throw std::invalid_argument("missing output group operation");
          const auto& definition = *plan->nodes[*node].definition;
          for (std::size_t g = 0; g < definition.groups.size(); ++g)
            if (definition.groups[g].name == ref.group)
              return coordinates->positions(
                  coordinates->source().tensor_axes.at(*node),
                  coordinates->source().tensor_groups.at(*node).at(g));
          throw std::invalid_argument("unknown output group");
        });
    if (plan && !plan->proof.guard.empty())
      throw std::invalid_argument(
          "unresolved obligations cannot become runtime guards");
    if (plan && plan->proof.core) {
      for (auto& d : r.descriptors)
        if (d.kind == core::Descriptor::Kind::kInferred)
          d.kind = core::Descriptor::Kind::kDerived;
      r.derivations.erase(
          std::remove_if(r.derivations.begin(), r.derivations.end(),
                         [&](const auto& d) {
                           if (!d.target) return inference::hasExtensions(*plan->proof.core);
                           return d.target &&
                                  std::any_of(
                                      plan->proof.core->replacement.begin(),
                                      plan->proof.core->replacement.end(),
                                      [&](const auto& n) {
                                        return n.descriptor == d.target->value;
                                      });
                         }),
          r.derivations.end());
    }
    for (auto& derivation : r.derivations) {
      if (!derivation.target) continue;
      auto& descriptor = r.descriptors.at(derivation.target->value);
      if (descriptor.kind != core::Descriptor::Kind::kInferred) continue;
      if (!plan || !plan->proof.constructors.contains(descriptor.id.value))
        throw std::invalid_argument("auto constructor is unresolved in rule '" +
                                    r.name + "'");
      auto ctor = std::get<core::DescriptorConstruct>(derivation.value->value);
      const auto& values = plan->proof.constructors.at(descriptor.id.value);
      if (values.size() != ctor.fields.size())
        throw std::invalid_argument("incomplete auto constructor");
      for (std::size_t f = 0; f < ctor.fields.size(); ++f)
        if (std::holds_alternative<core::AutoField>(ctor.fields[f]->value))
          ctor.fields[f] = std::make_shared<core::TypedExpr>(core::TypedExpr{
              r.origin, ctor.fields[f]->type, core::SymbolicExpr{values[f]}});
      descriptor.kind = core::Descriptor::Kind::kDerived;
      derivation.value = std::make_shared<core::TypedExpr>(
          core::TypedExpr{r.origin, descriptor.type, std::move(ctor)});
    }
    auto completed_rule = completion.rule(std::move(static_cast<RuleData&>(r)));
    if (plan && plan->proof.core) {
      completed_rule.core =
          inference::portable(inference::emit(*plan->proof.core));
      if (!completed::coreGuard(completed_rule))
        throw std::invalid_argument(
            "core guard certificate does not match completed conditions");
    }
    completed::validateAxisConditions(completed_rule);
    if (coordinates)
      for (const auto& [key, value] : coordinates->source().groups)
        axes.push_back(value);
    axes.erase(std::remove_if(
                   axes.begin(), axes.end(),
                   [](const auto& s) {
                     return !s ||
                            s->kind != core::semantics::Symbol::Kind::AxisGroup;
                   }),
               axes.end());
    for (const auto& axis : axes)
      output.axes[i].push_back(completion.symbol(axis));
    output.ir.rules.push_back(std::move(completed_rule));
  }
  return output;
}
}  // namespace tepl::core::semantics
