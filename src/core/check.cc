#include <map>
#include <set>
#include <utility>

#include "src/core/analysis_context.h"
#include "src/core/check_expr.h"
#include "src/core/literal.h"
#include "src/core/rule_scope.h"

namespace tepl::core::detail {
namespace {

// A sequence contributes zero or more dimensions. Without a sequence the
// minimum is also the exact rank, including rank zero for scalar declarations.
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

class RuleChecker {
 public:
  RuleChecker(AnalysisContext& ctx, const ExpandedRule& input)
      : ctx_(ctx), input_(input) {
    rule_.id = RuleId{ctx.output.rules.size()};
    rule_.name = input.name;
    rule_.origin = input.origin;
    tensor_ = ctx_.types.concrete({TypeKind::kTensor}, input.origin);
  }

  std::optional<Rule> run() {
    const auto diagnostics_before = ctx_.diagnostics.size();
    collectCaptures(*input_.lhs);
    declarations();
    rule_.lhs = pattern(*input_.lhs);
    registerDerivations();
    rule_.rhs = build(*input_.rhs);
    ExpressionChecker expressions(ctx_, input_, rule_, scope_);
    checkConditions(expressions);
    checkDerivations(expressions);
    if (ctx_.diagnostics.size() != diagnostics_before) return std::nullopt;
    ctx_.types.unify(rule_.lhs->type, rule_.rhs->type, input_.origin);
    if (ctx_.diagnostics.size() != diagnostics_before) return std::nullopt;
    return std::move(rule_);
  }

 private:
  void registerDerivations() {
    // RHS descriptor uses determine schemas before expressions are checked;
    // availability still advances in source order after each assignment.
    for (const auto& derivation : input_.derivations) {
      if (scope_.descriptors.contains(derivation.target)) {
        ctx_.report(derivation.origin, "duplicate descriptor definition '@" +
                                           derivation.target + "'");
        continue;
      }
      addDescriptor(derivation.target, Descriptor::Kind::kDerived,
                    derivation.origin);
    }
  }

  void checkConditions(ExpressionChecker& expressions) {
    for (const auto& condition : input_.conditions) {
      const auto& at = input_.expr_origins.at(condition.get());
      if (auto checked = expressions.check(
              *condition, ctx_.types.concrete({TypeKind::kBool}, at)))
        rule_.conditions.push_back(std::move(checked));
    }
  }

  void checkDerivations(ExpressionChecker& expressions) {
    for (const auto& derivation : input_.derivations) {
      const auto id = scope_.descriptors.at(derivation.target);
      const auto& descriptor = rule_.descriptors.at(id.value);
      if (descriptor.kind != Descriptor::Kind::kDerived ||
          scope_.available_descriptors.contains(id.value))
        continue;
      if (auto checked = expressions.check(*derivation.value, descriptor.type))
        rule_.derivations.push_back(
            {id, std::move(checked), derivation.origin});
      scope_.available_descriptors.insert(id.value);
    }
  }

  const SourceOrigin& graphOrigin(const ast::GraphExpr& node) const {
    return input_.graphs.at(&node).origin;
  }

  CaptureId capture(const std::string& name, const SourceOrigin& at) {
    if (const auto found = scope_.captures.find(name);
        found != scope_.captures.end())
      return found->second;
    const CaptureId id{rule_.captures.size()};
    rule_.captures.push_back({id, name, tensor_, at});
    scope_.captures.emplace(name, id);
    return id;
  }

  void collectCaptures(const ast::GraphExpr& node) {
    const auto& at = graphOrigin(node);
    if (const auto* name = std::get_if<ast::NameRef>(&node.value)) {
      capture(name->name, at);
    } else if (const auto* op = std::get_if<ast::Operator>(&node.value)) {
      for (const auto& operand : op->operands) collectCaptures(*operand);
    } else if (const auto* binding = std::get_if<ast::Binding>(&node.value)) {
      collectCaptures(*binding->expression);
      if (scope_.captures.contains(binding->binder.name))
        ctx_.report(at, "binding '" + binding->binder.name +
                            "' conflicts with an existing LHS capture");
      capture(binding->binder.name, at);
    }
  }

  std::optional<DimensionId> dimension(const std::string& name, bool sequence,
                                       const SourceOrigin& at) {
    if (scope_.captures.contains(name)) {
      ctx_.report(at,
                  "dimension '" + name + "' conflicts with a tensor capture");
      return std::nullopt;
    }
    if (const auto found = scope_.dimensions.find(name);
        found != scope_.dimensions.end()) {
      const auto& symbol = rule_.dimensions[found->second.value];
      if (symbol.sequence != sequence)
        ctx_.report(
            at,
            "dimension '" + name + "' is used as both an index and a sequence",
            {symbol.origin.definition});
      return found->second;
    }
    const DimensionId id{rule_.dimensions.size()};
    const auto type = ctx_.types.concrete(
        {sequence ? TypeKind::kIndexList : TypeKind::kIndex}, at);
    rule_.dimensions.push_back({id, name, sequence, type, at});
    scope_.dimensions.emplace(name, id);
    return id;
  }

  void declarations() {
    std::set<std::pair<std::size_t, std::string>> declared;
    std::map<std::size_t, std::pair<DType, SourceOrigin>> dtypes;
    std::map<std::size_t, RankRestriction> ranks;
    for (const auto& located : input_.declarations) {
      const auto& declaration = located.value;
      const auto& at = located.origin;
      const auto name = std::visit([](const auto& value) { return value.name; },
                                   declaration.value);
      if (!declared.emplace(located.layer, name).second)
        ctx_.report(at, "duplicate tensor declaration '" + name + "'");
      const auto found = scope_.captures.find(name);
      if (found == scope_.captures.end()) {
        ctx_.report(at, "tensor declaration '" + name +
                            "' does not refer to an LHS capture");
        continue;
      }
      CaptureConstraint constraint{found->second, std::nullopt, {}, at};
      bool sequence = false;
      std::size_t minimum_rank = 0;
      if (const auto* tensor =
              std::get_if<ast::TensorDecl>(&declaration.value)) {
        if (tensor->dtype) {
          constraint.dtype = resolveDType(tensor->dtype->name);
          if (!constraint.dtype)
            ctx_.report(at, "unknown dtype '" + tensor->dtype->name + "'");
        }
        for (const auto& shape : tensor->shape) {
          auto location = at;
          location.definition.span = shape.span;
          if (const auto* named =
                  std::get_if<ast::NamedDimension>(&shape.value)) {
            ++minimum_rank;
            constraint.shape.push_back({ShapeElement::Kind::kDimension,
                                        dimension(named->name, false, location),
                                        location});
          } else if (const auto* rest =
                         std::get_if<ast::SequenceDimension>(&shape.value)) {
            if (sequence)
              ctx_.report(location, "a shape can contain at most one sequence");
            sequence = true;
            constraint.shape.push_back(
                {ShapeElement::Kind::kSequence,
                 rest->name ? dimension(*rest->name, true, location)
                            : std::nullopt,
                 location});
          } else {
            ++minimum_rank;
            constraint.shape.push_back(
                {ShapeElement::Kind::kWildcard, std::nullopt, location});
          }
        }
      }
      if (constraint.dtype) {
        const auto [previous, inserted] = dtypes.emplace(
            found->second.value, std::make_pair(*constraint.dtype, at));
        if (!inserted && previous->second.first != *constraint.dtype)
          ctx_.report(at, "conflicting dtype restrictions for '" + name + "'",
                      {previous->second.second.definition});
      }
      const RankRestriction rank{minimum_rank, !sequence, at};
      const auto [previous, inserted] =
          ranks.emplace(found->second.value, rank);
      if (!inserted) {
        auto& restriction = previous->second;
        if (!restriction.compatibleWith(rank))
          ctx_.report(at, "conflicting rank restrictions for '" + name + "'",
                      {restriction.origin.definition});
        else if (!restriction.fixed &&
                 (rank.fixed || rank.minimum > restriction.minimum))
          restriction = rank;
      }
      rule_.constraints.push_back(std::move(constraint));
    }
  }

  DescriptorId addDescriptor(const std::string& name, Descriptor::Kind kind,
                             const SourceOrigin& at) {
    const DescriptorId id{rule_.descriptors.size()};
    rule_.descriptors.push_back(
        {id, name, kind, ctx_.types.concrete({TypeKind::kDescriptor}, at), at});
    scope_.descriptors.emplace(name, id);
    return id;
  }

  std::optional<DescriptorId> descriptor(const ast::Operator& op,
                                         OpId operation, bool lhs,
                                         const SourceOrigin& at) {
    const auto schema = ctx_.output.operations[operation.value].attributes;
    if (!op.attribute) {
      if (schema)
        ctx_.report(
            at, "operation '" + op.name + "' requires an attribute descriptor");
      return std::nullopt;
    }
    if (!schema) {
      ctx_.report(at, "operation '" + op.name + "' has no attributes");
      return std::nullopt;
    }
    const auto& name = op.attribute->name;
    auto found = scope_.descriptors.find(name);
    if (found == scope_.descriptors.end()) {
      if (!lhs) {
        ctx_.report(at, "unknown RHS descriptor '@" + name + "'");
        return std::nullopt;
      }
      const auto id = addDescriptor(name, Descriptor::Kind::kCaptured, at);
      scope_.available_descriptors.insert(id.value);
      found = scope_.descriptors.find(name);
    }
    const auto id = found->second;
    ctx_.types.unify(rule_.descriptors[id.value].type,
                     ctx_.types.concrete({TypeKind::kDescriptor, schema}, at),
                     at);
    return id;
  }

  OpId graphOperation(const ast::GraphExpr& node, const ast::Operator& op) {
    const auto id = input_.graphs.at(&node).operation.value();
    const auto& operation = ctx_.output.operations[id.value];
    const bool variadic =
        !operation.operands.empty() && operation.operands.back().variadic;
    const auto minimum = operation.operands.size() - (variadic ? 1 : 0);
    if (op.operands.size() < minimum ||
        (!variadic && op.operands.size() != minimum))
      ctx_.report(graphOrigin(node), "operation '" + op.name + "' expects " +
                                         (variadic ? "at least " : "exactly ") +
                                         std::to_string(minimum) +
                                         " operands, got " +
                                         std::to_string(op.operands.size()));
    return id;
  }

  template <typename Literal>
  GraphLiteral literal(const Literal& input, const SourceOrigin& at,
                       bool decimal) {
    GraphLiteral result{
        decimal ? GraphLiteral::Kind::kDecimal : GraphLiteral::Kind::kInteger,
        input.digits, std::nullopt};
    if (!input.dtype) return result;
    result.dtype = resolveDType(input.dtype->name);
    if (!result.dtype) {
      ctx_.report(at, "unknown dtype '" + input.dtype->name + "'");
      return result;
    }
    if (!validGraphLiteral(input.digits, *result.dtype)) {
      auto annotation = at;
      annotation.definition.span = input.dtype->span;
      ctx_.report(annotation, "literal '" + input.digits +
                                  "' is invalid for dtype '" +
                                  input.dtype->name + "'");
    }
    return result;
  }

  PatternPtr pattern(const ast::GraphExpr& node) {
    const auto& at = graphOrigin(node);
    Pattern result{at, tensor_, CapturePattern{}};
    if (const auto* name = std::get_if<ast::NameRef>(&node.value)) {
      result.value = CapturePattern{scope_.captures.at(name->name)};
    } else if (const auto* integer =
                   std::get_if<ast::IntegerLiteral>(&node.value)) {
      result.value = literal(*integer, at, false);
    } else if (const auto* decimal =
                   std::get_if<ast::FloatLiteral>(&node.value)) {
      result.value = literal(*decimal, at, true);
    } else if (const auto* op = std::get_if<ast::Operator>(&node.value)) {
      const auto id = graphOperation(node, *op);
      MatchOperation match{id, descriptor(*op, id, true, at), {}};
      for (const auto& child : op->operands) {
        auto checked = pattern(*child);
        if (!checked) return nullptr;
        match.operands.push_back(std::move(checked));
      }
      result.value = std::move(match);
      result.type = ctx_.output.operations[id.value].result;
    } else if (const auto* binding = std::get_if<ast::Binding>(&node.value)) {
      auto checked = pattern(*binding->expression);
      if (!checked) return nullptr;
      result.value = BindPattern{scope_.captures.at(binding->binder.name),
                                 std::move(checked)};
    } else {
      ctx_.report(at, "unsupported LHS expression");
      return nullptr;
    }
    return std::make_shared<Pattern>(std::move(result));
  }

  BuildExprPtr build(const ast::GraphExpr& node) {
    const auto& at = graphOrigin(node);
    BuildExpr result{at, tensor_, CaptureRef{}};
    if (const auto* name = std::get_if<ast::NameRef>(&node.value)) {
      const auto found = scope_.captures.find(name->name);
      if (found == scope_.captures.end()) {
        ctx_.report(at, "unknown RHS capture '" + name->name + "'");
        return nullptr;
      }
      result.value = CaptureRef{found->second};
    } else if (const auto* integer =
                   std::get_if<ast::IntegerLiteral>(&node.value)) {
      result.value = literal(*integer, at, false);
    } else if (const auto* decimal =
                   std::get_if<ast::FloatLiteral>(&node.value)) {
      result.value = literal(*decimal, at, true);
    } else if (const auto* op = std::get_if<ast::Operator>(&node.value)) {
      const auto id = graphOperation(node, *op);
      BuildOperation construction{id, descriptor(*op, id, false, at), {}};
      for (const auto& child : op->operands) {
        auto checked = build(*child);
        if (!checked) return nullptr;
        construction.operands.push_back(std::move(checked));
      }
      result.value = std::move(construction);
      result.type = ctx_.output.operations[id.value].result;
    } else {
      ctx_.report(at, "RHS bindings and projections are not supported");
      return nullptr;
    }
    return std::make_shared<BuildExpr>(std::move(result));
  }

  AnalysisContext& ctx_;
  const ExpandedRule& input_;
  Rule rule_;
  TypeId tensor_;
  RuleScope scope_;
};

}  // namespace

std::optional<Rule> check(AnalysisContext& context, const ExpandedRule& input) {
  return RuleChecker(context, input).run();
}

}  // namespace tepl::core::detail
