#include "src/core/semantics/materialize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/semantics/completion.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/graph.h"
namespace tepl::core::semantics {
Materialized materialize(InspectionData checked) {
  Materialized output;
  static_cast<ProgramTables&>(output.ir) =
      std::move(static_cast<ProgramTables&>(checked));
  output.axes.resize(checked.rules.size());
  auto typeId = [&](core::Type type) {
    for (std::size_t i = 0; i < output.ir.types.size(); ++i)
      if (output.ir.types[i] == type) return core::TypeId{i};
    output.ir.types.push_back(type);
    return core::TypeId{output.ir.types.size() - 1};
  };
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
            if (plan->nodes[id].descriptor == ref.descriptor) {
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
    if (plan)
      for (const auto& guard : plan->proof.guard)
        r.conditions.push_back(std::make_shared<core::TypedExpr>(
            core::TypedExpr{r.origin, typeId(core::Type{core::TypeKind::kBool}),
                            core::SymbolicExpr{guard}}));
    for (const auto& hole : r.attribute_holes) {
      if (!plan || !plan->proof.constructors.contains(hole.descriptor.value))
        throw std::invalid_argument("auto constructor is unresolved in rule '" +
                                    r.name + "'");
      const auto& schema = output.ir.attribute_schemas.at(hole.schema.value);
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
    r.attribute_holes.clear();
    auto completed_rule = completion.rule(std::move(static_cast<RuleData&>(r)));
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
