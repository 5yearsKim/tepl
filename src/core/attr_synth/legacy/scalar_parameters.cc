// Legacy non-axis attribute inference, retained for reference.
// Not built or invoked by production axis auto synthesis; no general
// completeness guarantee.
#include "src/core/attr_synth/legacy/scalar_parameters.h"

#include "src/core/semantics/symbolic/tree.h"

namespace tepl::core::attr_synth {
using namespace semantics;
void inferScalarParameters(Context& c, const Terms& source) {
  using Slot = std::pair<ScalarFunction, std::size_t>;
  std::map<Slot, std::map<Key, S>> parameters;
  auto collect = [&](const E& value) {
    c.tick();
    if (value->kind == Expr::Kind::Call)
      for (std::size_t p = 0; p < value->parameters.size(); ++p) {
        auto parameter = c.canonical(value->parameters[p]);
        parameters[{value->function, p}].emplace(c.key(parameter), parameter);
      }
  };
  for (const auto& term : source)
    for (const auto& factor : term.factors) walkTree(factor, collect);
  for (const auto& node : c.plan.nodes) {
    if (!node.automatic) continue;
    walkTree(node.definition->value, [&](const ValuePtr& raw) {
      const auto& value = *raw;
      c.tick();
      if (value.kind == Value::Kind::Call)
        for (std::size_t p = 0; p < value.parameters.size(); ++p) {
          const auto& parameter = *value.parameters[p];
          if (parameter.kind != Meta::Kind::Attribute) continue;
          auto field = c.field(node, parameter.index);
          if (field->kind != Symbol::Kind::Hole) continue;
          auto found = parameters.find({value.function, p});
          if (found == parameters.end() || found->second.size() != 1) continue;
          auto candidate = found->second.begin()->second;
          c.validateSource(candidate);
          c.assignAttribute(field->owner, field->index, candidate);
          c.note("scalar-parameter-correspondence");
        }
    });
  }
}
}  // namespace tepl::core::attr_synth
