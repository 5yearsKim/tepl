#include "src/core/rule/declarations.h"

#include <map>
#include <set>
#include <utility>

#include "src/core/rule/check_context.h"

namespace tepl::core::detail {

namespace {

// A sequence contributes zero or more dimensions. Without a sequence the
// minimum is also the exact rank, including rank zero for empty shapes.
struct RankRestriction {
  std::size_t minimum;
  bool fixed;
  SourceOrigin origin;

  bool accepts(std::size_t rank) const {
    return fixed ? rank == minimum : rank >= minimum;
  }

  bool compatibleWith(const RankRestriction& other) const {
    if (fixed) return other.accepts(minimum);
    if (other.fixed) return accepts(other.minimum);
    return true;
  }
};

std::optional<DimensionId> dimension(RuleCheckContext& context,
                                     const std::string& name, bool sequence,
                                     const SourceOrigin& at) {
  if (context.scope.captures.contains(name)) {
    context.analysis.report(
        at, "dimension '" + name + "' conflicts with a tensor capture");
    return std::nullopt;
  }
  if (const auto found = context.scope.dimensions.find(name);
      found != context.scope.dimensions.end()) {
    const auto& symbol = context.rule.dimensions[found->second.value];
    if (symbol.sequence != sequence)
      context.analysis.report(
          at,
          "dimension '" + name + "' is used as both an index and a sequence",
          {symbol.origin.definition});
    return found->second;
  }
  const DimensionId id{context.rule.dimensions.size()};
  const auto type = context.analysis.types.concrete(
      {sequence ? TypeKind::kIndexList : TypeKind::kIndex}, at);
  context.rule.dimensions.push_back({id, name, sequence, type, at});
  context.scope.dimensions.emplace(name, id);
  return id;
}

}  // namespace

void checkDeclarations(RuleCheckContext& context) {
  std::set<std::pair<std::size_t, std::string>> declared;
  std::map<std::size_t, std::pair<DType, SourceOrigin>> dtypes;
  std::map<std::size_t, RankRestriction> ranks;
  for (const auto& located : context.input.declarations) {
    const auto& declaration = located.value;
    const auto& at = located.origin;
    const auto& name = declaration.name;
    if (!declared.emplace(located.layer, name).second)
      context.analysis.report(at,
                              "duplicate tensor declaration '" + name + "'");
    const auto found = context.scope.captures.find(name);
    if (found == context.scope.captures.end()) {
      context.analysis.report(at, "tensor declaration '" + name +
                                      "' does not refer to an LHS capture");
      continue;
    }
    CaptureConstraint constraint{found->second, std::nullopt, {}, at};
    bool sequence = false;
    std::size_t minimum_rank = 0;
    if (declaration.dtype) {
      constraint.dtype = resolveDType(declaration.dtype->name);
      if (!constraint.dtype)
        context.analysis.report(
            at, "unknown dtype '" + declaration.dtype->name + "'");
    }
    for (const auto& shape : declaration.shape) {
      auto location = at;
      location.definition.span = shape.span;
      if (const auto* named = std::get_if<ast::NamedDimension>(&shape.value)) {
        ++minimum_rank;
        constraint.shape.push_back(
            {ShapeElement::Kind::kDimension,
             dimension(context, named->name, false, location), location});
      } else if (const auto* rest =
                     std::get_if<ast::SequenceDimension>(&shape.value)) {
        if (sequence)
          context.analysis.report(location,
                                  "a shape can contain at most one sequence");
        sequence = true;
        constraint.shape.push_back(
            {ShapeElement::Kind::kSequence,
             rest->name ? dimension(context, *rest->name, true, location)
                        : std::nullopt,
             location});
      } else {
        ++minimum_rank;
        constraint.shape.push_back(
            {ShapeElement::Kind::kWildcard, std::nullopt, location});
      }
    }
    if (constraint.dtype) {
      const auto [previous, inserted] = dtypes.emplace(
          found->second.value, std::make_pair(*constraint.dtype, at));
      if (!inserted && previous->second.first != *constraint.dtype)
        context.analysis.report(
            at, "conflicting dtype restrictions for '" + name + "'",
            {previous->second.second.definition});
    }
    const RankRestriction rank{minimum_rank, !sequence, at};
    const auto [previous, inserted] = ranks.emplace(found->second.value, rank);
    if (!inserted) {
      auto& restriction = previous->second;
      if (!restriction.compatibleWith(rank))
        context.analysis.report(
            at, "conflicting rank restrictions for '" + name + "'",
            {restriction.origin.definition});
      else if (!restriction.fixed &&
               (rank.fixed || rank.minimum > restriction.minimum))
        restriction = rank;
    }
    context.rule.constraints.push_back(std::move(constraint));
  }
}

}  // namespace tepl::core::detail
