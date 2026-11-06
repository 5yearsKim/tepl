#include "src/core/attr_synth/synthesize.h"

#include <algorithm>
#include <stdexcept>

#include "src/core/attr_synth/construct.h"
#include "src/core/attr_synth/ordered/construct.h"
#include "src/core/attr_synth/ordered/contract.h"

namespace tepl::core::attr_synth {
using namespace semantics;
namespace {
Result run(const Plan& plan, std::size_t max_work,
           std::chrono::steady_clock::time_point deadline,
           const ordered::Admission* admitted = nullptr,
           const Plan* whole = nullptr,
           const inference::EmissionBlock* metadata_program = nullptr) {
  // The exact check/build program must describe every legal metadata instance,
  // including ones outside the written guard. Specialization belongs to the
  // independent verifier, not to generation of the feasibility predicate.
  std::optional<Plan> metadata_plan;
  if (admitted) {
    metadata_plan = plan;
    metadata_plan->assumptions.clear();
  }
  Context c{metadata_plan ? *metadata_plan : plan, max_work, deadline};
  Result result;
  auto& candidate = result.candidate;
  std::string phase = "source lowering";
  try {
    if (whole) {
      std::function<void(std::size_t)> scope = [&](std::size_t id) {
        c.sourceScope(id, false);
        for (auto child : whole->nodes.at(id).inputs) scope(child);
      };
      scope(whole->lhs);
    }
    c.sourceScope(plan.lhs);
    auto lhs = lowerSource(c);
    result.output_shape = lhs.shape;
    phase = "metadata definedness";
    if (!admitted) {
      for (const auto& value : plan.derived_metadata) c.checkMetadata(value);
      for (const auto& [value, type] : plan.constructed_metadata)
        c.checkMetadataType(value, type);
    }
    auto lt = normalize(c, lhs.value);
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
    if (admitted) {
      if (!metadata_program)
        throw std::invalid_argument(
            "ordered specialization requires metadata IR");
      candidate =
          ordered::construct(c, *metadata_program, match, lhs.axes, reductions)
              .candidate;
    } else {
      construct(c, target, match, lhs, reductions, lt);
    }
    for (const auto& n : plan.nodes)
      if (plan.needsSynthesis(n)) {
        if (admitted && !candidate.constructors.contains(n.descriptor->value))
          continue;
        auto& fields = candidate.constructors[n.descriptor->value];
        for (std::size_t f = 0; f < n.definition->fields.size(); ++f) {
          if (admitted) {
            c.checkConstructor(n, f, fields.at(f));
            continue;
          }
          auto value = plan.attributes(n).isAuto(f)
                           ? c.canonical(c.field(n, f))
                           : plan.attributes(n).fields.at(f);
          c.checkConstructor(n, f, value);
          fields.push_back(value);
        }
      }
    candidate.guard = c.guards();
    result.constructed = true;
    for (const auto* law : {"axis-selector-matching"}) {
      const auto step = std::string("law: ") + law;
      if (std::find(c.steps().begin(), c.steps().end(), step) !=
          c.steps().end())
        result.steps.push_back(step);
    }
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
Result identity(const Plan& plan, std::size_t max_work,
                std::chrono::steady_clock::time_point deadline) {
  Result result;
  try {
    Context c{plan, max_work, deadline};
    if (!matchOperationTree(c)) {
      result.reason = "operation trees do not match for descriptor copying";
      return result;
    }
    c.sourceScope(plan.lhs);
    Lowerer source{c};
    source.lower(plan.lhs);
    auto& candidate = result.candidate;
    result.constructed =
        matchOperationTree(c, [&](const Instance& left, const Instance& right) {
          if (plan.needsSynthesis(right)) {
            std::vector<S> fields;
            for (std::size_t f = 0; f < right.definition->fields.size(); ++f)
              fields.push_back(plan.attributes(right).isAuto(f)
                                   ? c.field(left, f)
                                   : plan.attributes(right).fields.at(f));
            auto [found, added] =
                candidate.constructors.emplace(right.descriptor->value, fields);
            if (!added) {
              if (found->second.size() != fields.size()) return false;
              for (std::size_t f = 0; f < fields.size(); ++f)
                if (!equal(found->second[f], fields[f])) return false;
            }
          }
          return true;
        });
    if (!result.constructed)
      result.reason = "operation trees do not match for descriptor copying";
  } catch (const ProofFailure& e) {
    result.reason = e.describe();
  }
  if (!result.constructed) result.candidate = {};
  return result;
}

}  // namespace

Result synthesize(const Plan& plan, Strategy strategy, std::size_t max_work,
                  std::chrono::steady_clock::time_point deadline) {
  switch (strategy) {
    case Strategy::Identity:
      return identity(plan, max_work, deadline);
    case Strategy::Ordered: {
      auto admitted = ordered::admit(plan);
      Result result;
      if (!admitted.admitted()) {
        result.reason = admitted.reason;
        return result;
      }
      // Structural emission precedes ALL proof contexts and normalization.
      // Native execution and symbolic specialization consume this same IR.
      auto metadata_program = inference::emit(admitted.program);
      if (admitted.program.components.empty()) {
        auto part = run(plan, max_work, deadline, &admitted, nullptr,
                        &metadata_program.blocks.at(0));
        part.metadata_program = std::move(metadata_program);
        return part;
      }
      std::size_t block = 0;
      for (const auto& pair : admitted.program.components) {
        auto component = ordered::componentPlan(plan, pair);
        auto single = ordered::admit(component);
        auto part = run(component, max_work, deadline, &single, &plan,
                        &metadata_program.blocks.at(block++));
        if (!part.constructed) {
          part.metadata_program = std::move(metadata_program);
          return part;
        }
        if (result.output_shape)
          result.candidate.feasibility.push_back(
              call("==", {result.output_shape, part.output_shape}));
        else
          result.output_shape = part.output_shape;
        result.candidate.constructors.insert(
            part.candidate.constructors.begin(),
            part.candidate.constructors.end());
        result.candidate.feasibility.insert(result.candidate.feasibility.end(),
                                            part.candidate.feasibility.begin(),
                                            part.candidate.feasibility.end());
      }
      result.constructed = true;
      result.metadata_program = std::move(metadata_program);
      result.steps.push_back("law: ordered-requirements-construction");
      result.steps.push_back("law: separated-additive-components");
      return result;
    }
    case Strategy::General:
      return run(plan, max_work, deadline);
  }
  throw std::invalid_argument("unknown attribute synthesis strategy");
}
}  // namespace tepl::core::attr_synth
