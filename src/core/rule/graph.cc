#include "src/core/rule/graph.h"

#include <utility>

#include "src/core/rule/check_context.h"
#include "src/core/rule/descriptors.h"
#include "src/core/type/literal.h"

namespace tepl::core::detail {

namespace {

const SourceOrigin& graphOrigin(const RuleCheckContext& context,
                                const ast::GraphExpr& node) {
  return context.input.graphs.at(&node).origin;
}

CaptureId capture(RuleCheckContext& context, const std::string& name,
                  const SourceOrigin& at) {
  if (const auto found = context.scope.captures.find(name);
      found != context.scope.captures.end())
    return found->second;
  if (resolveDType(name))
    context.analysis.report(
        at, "tensor capture '" + name + "' conflicts with a dtype constant");
  const CaptureId id{context.rule.captures.size()};
  context.rule.captures.push_back({id, name, context.tensor, at});
  context.scope.captures.emplace(name, id);
  return id;
}

OpId graphOperation(RuleCheckContext& context, const ast::GraphExpr& node,
                    const ast::Operator& op) {
  const auto id = context.input.graphs.at(&node).operation.value();
  const auto& operation = context.analysis.output.operations[id.value];
  const bool variadic =
      !operation.operands.empty() && operation.operands.back().variadic;
  const auto minimum = operation.operands.size() - (variadic ? 1 : 0);
  if (op.operands.size() < minimum ||
      (!variadic && op.operands.size() != minimum))
    context.analysis.report(graphOrigin(context, node),
                            "operation '" + op.name + "' expects " +
                                (variadic ? "at least " : "exactly ") +
                                std::to_string(minimum) + " operands, got " +
                                std::to_string(op.operands.size()));
  return id;
}

template <typename Literal>
GraphLiteral literal(RuleCheckContext& context, const Literal& input,
                     const SourceOrigin& at, bool decimal) {
  GraphLiteral result{
      decimal ? GraphLiteral::Kind::kDecimal : GraphLiteral::Kind::kInteger,
      input.digits, std::nullopt};
  if (!input.dtype) return result;
  result.dtype = resolveDType(input.dtype->name);
  if (!result.dtype) {
    context.analysis.report(at, "unknown dtype '" + input.dtype->name + "'");
    return result;
  }
  if (!validGraphLiteral(input.digits, *result.dtype)) {
    auto annotation = at;
    annotation.definition.span = input.dtype->span;
    context.analysis.report(annotation, "literal '" + input.digits +
                                            "' is invalid for dtype '" +
                                            input.dtype->name + "'");
  }
  return result;
}

}  // namespace

void collectCaptures(RuleCheckContext& context, const ast::GraphExpr& node) {
  const auto& at = graphOrigin(context, node);
  if (const auto* name = std::get_if<ast::NameRef>(&node.value)) {
    capture(context, name->name, at);
  } else if (const auto* op = std::get_if<ast::Operator>(&node.value)) {
    for (const auto& operand : op->operands) collectCaptures(context, *operand);
  } else if (const auto* binding = std::get_if<ast::Binding>(&node.value)) {
    collectCaptures(context, *binding->expression);
    if (context.scope.captures.contains(binding->binder.name))
      context.analysis.report(at,
                              "binding '" + binding->binder.name +
                                  "' conflicts with an existing LHS capture");
    capture(context, binding->binder.name, at);
  }
}

PatternPtr checkPattern(RuleCheckContext& context, const ast::GraphExpr& node) {
  const auto& at = graphOrigin(context, node);
  Pattern result{at, context.tensor, CapturePattern{}};
  if (const auto* name = std::get_if<ast::NameRef>(&node.value)) {
    result.value = CapturePattern{context.scope.captures.at(name->name)};
  } else if (const auto* integer =
                 std::get_if<ast::IntegerLiteral>(&node.value)) {
    result.value = literal(context, *integer, at, false);
  } else if (const auto* decimal =
                 std::get_if<ast::FloatLiteral>(&node.value)) {
    result.value = literal(context, *decimal, at, true);
  } else if (const auto* op = std::get_if<ast::Operator>(&node.value)) {
    const auto id = graphOperation(context, node, *op);
    MatchOperation match{id, checkDescriptor(context, *op, id, true, at), {}};
    for (const auto& child : op->operands) {
      auto checked = checkPattern(context, *child);
      if (!checked) return nullptr;
      match.operands.push_back(std::move(checked));
    }
    result.value = std::move(match);
    result.type = context.analysis.output.operations[id.value].result;
  } else if (const auto* binding = std::get_if<ast::Binding>(&node.value)) {
    auto checked = checkPattern(context, *binding->expression);
    if (!checked) return nullptr;
    result.value = BindPattern{context.scope.captures.at(binding->binder.name),
                               std::move(checked)};
  } else {
    context.analysis.report(at, "unsupported LHS expression");
    return nullptr;
  }
  return std::make_shared<Pattern>(std::move(result));
}

BuildExprPtr checkBuild(RuleCheckContext& context, const ast::GraphExpr& node) {
  const auto& at = graphOrigin(context, node);
  BuildExpr result{at, context.tensor, CaptureRef{}};
  if (const auto* name = std::get_if<ast::NameRef>(&node.value)) {
    const auto found = context.scope.captures.find(name->name);
    if (found == context.scope.captures.end()) {
      context.analysis.report(at, "unknown RHS capture '" + name->name + "'");
      return nullptr;
    }
    result.value = CaptureRef{found->second};
  } else if (const auto* integer =
                 std::get_if<ast::IntegerLiteral>(&node.value)) {
    result.value = literal(context, *integer, at, false);
  } else if (const auto* decimal =
                 std::get_if<ast::FloatLiteral>(&node.value)) {
    result.value = literal(context, *decimal, at, true);
  } else if (const auto* op = std::get_if<ast::Operator>(&node.value)) {
    const auto id = graphOperation(context, node, *op);
    BuildOperation construction{
        id, checkDescriptor(context, *op, id, false, at), {}};
    for (const auto& child : op->operands) {
      auto checked = checkBuild(context, *child);
      if (!checked) return nullptr;
      construction.operands.push_back(std::move(checked));
    }
    result.value = std::move(construction);
    result.type = context.analysis.output.operations[id.value].result;
  } else {
    context.analysis.report(at,
                            "RHS bindings and projections are not supported");
    return nullptr;
  }
  return std::make_shared<BuildExpr>(std::move(result));
}

}  // namespace tepl::core::detail
