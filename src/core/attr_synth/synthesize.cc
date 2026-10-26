#include "src/core/attr_synth/synthesize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "src/core/attr_synth/construct.h"
#include "src/core/attr_synth/ordered/construct.h"
#include "src/core/attr_synth/scalar_parameters.h"

namespace tepl::core::attr_synth {
using namespace semantics;
namespace {
Result run(const Plan& plan, std::size_t max_work,
           const ordered::Admission* admitted = nullptr) {
  Context c{plan, max_work};
  Result result;
  auto& candidate = result.candidate;
  std::vector<RuleProof::RewriteStep> rewrites;
  std::string phase = "source lowering";
  try {
    c.sourceScope(plan.lhs);
    auto lhs = lowerSource(c);
    phase = "metadata definedness";
    for (const auto& value : plan.derived_metadata) c.checkMetadata(value);
    for (const auto& [value, type] : plan.constructed_metadata)
      c.checkMetadataType(value, type);
    auto lt = normalize(c, lhs.value, &rewrites);
    phase = "scalar parameter construction";
    inferScalarParameters(c, lt);
    phase = "coordinate correspondence";
    auto rt = normalize(c, arithmeticSkeleton(c, plan.rhs));
    auto match = correspond(c, lhs, lt, rt);
    phase = "attribute construction";
    std::vector<S> all;
    for (const auto& t : lt) all.push_back(t.reductions);
    Lowerer target{c, &match.anchors};
    target.interfaces = &match.interfaces;
    target.expected_dtype = lhs.dtype;
    auto reductions = call("unique_axes", {call("concat", std::move(all))});
    auto rhs = admitted ? ordered::construct(c, target, *admitted,
                                             ordered::deriveRequirements(
                                                 c, *admitted, match, lhs.axes),
                                             reductions)
                        : construct(c, target, match, lhs, reductions, lt);
    auto checked = normalize(c, rhs.value, &rewrites);
    auto simplified_lhs = collectTerms(c, lt, &rewrites);
    auto simplified_rhs = collectTerms(c, checked, &rewrites);
    equivalentTerms(c, simplified_lhs, simplified_rhs);
    c.require(call("==", {lhs.axes, rhs.axes}), GuardKind::CoordinateOrder);
    c.require(call("==", {lhs.shape, rhs.shape}));
    c.require(call("==", {lhs.dtype, rhs.dtype}));
    for (const auto& n : plan.nodes)
      if (n.automatic) {
        auto& fields = candidate.constructors[n.descriptor->value];
        for (std::size_t f = 0; f < n.definition->fields.size(); ++f) {
          auto value = c.canonical(c.field(n, f));
          c.checkConstructor(n, f, value);
          fields.push_back(value);
        }
      }
    candidate.guard = c.guards();
    result.constructed = true;
    for (const auto* law :
         {"index-coefficient-matching", "scalar-parameter-correspondence"}) {
      const auto step = std::string("law: ") + law;
      if (std::find(c.steps().begin(), c.steps().end(), step) !=
          c.steps().end())
        result.steps.push_back(step);
    }
    result.rewrites = std::move(rewrites);
    if (admitted)
      result.steps.push_back("law: ordered-requirements-construction");
  } catch (const ProofFailure& e) {
    result.candidate = {};
    result.candidate.guard = c.guards();
    result.guard_origins = c.guardOrigins();
    result.reason = phase + ": " + e.describe();
    result.steps = c.steps();
  }
  return result;
}
Result identity(const Plan& plan, std::size_t max_work) {
  Result result;
  try {
    Context c{plan, max_work};
    c.sourceScope(plan.lhs);
    Lowerer source{c};
    source.lower(plan.lhs);
    auto& candidate = result.candidate;
    std::function<bool(std::size_t, std::size_t)> copy = [&](std::size_t a,
                                                             std::size_t b) {
      c.tick();
      const auto& left = plan.nodes.at(a);
      const auto& right = plan.nodes.at(b);
      if (left.capture || right.capture) return left.capture == right.capture;
      if (left.literal || right.literal)
        return left.literal == right.literal &&
               left.literal_dtype == right.literal_dtype;
      if (left.operation != right.operation ||
          left.inputs.size() != right.inputs.size())
        return false;
      for (std::size_t i = 0; i < left.inputs.size(); ++i)
        if (!copy(left.inputs[i], right.inputs[i])) return false;
      if (right.automatic) {
        std::vector<S> fields;
        for (std::size_t f = 0; f < right.definition->fields.size(); ++f)
          fields.push_back(c.field(left, f));
        auto [found, added] =
            candidate.constructors.emplace(right.descriptor->value, fields);
        if (!added) {
          if (found->second.size() != fields.size()) return false;
          for (std::size_t f = 0; f < fields.size(); ++f)
            if (!equal(found->second[f], fields[f])) return false;
        }
      }
      return true;
    };
    result.constructed = copy(plan.lhs, plan.rhs);
    if (!result.constructed)
      result.reason = "operation trees do not match for descriptor copying";
  } catch (const ProofFailure& e) {
    result.reason = e.describe();
  }
  if (!result.constructed) result.candidate = {};
  return result;
}

}  // namespace

Result synthesize(const Plan& plan, Strategy strategy, std::size_t max_work) {
  switch (strategy) {
    case Strategy::Identity:
      return identity(plan, max_work);
    case Strategy::Ordered: {
      auto admitted = ordered::admit(plan);
      if (admitted.admitted()) return run(plan, max_work, &admitted);
      Result result;
      result.reason = admitted.reason;
      return result;
    }
    case Strategy::General:
      return run(plan, max_work);
  }
  throw std::invalid_argument("unknown attribute synthesis strategy");
}
}  // namespace tepl::core::attr_synth
