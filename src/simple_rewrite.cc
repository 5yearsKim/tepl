#include "src/simple_rewrite.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "src/semantic.h"

namespace tepl {
namespace {

using Captures = std::unordered_map<std::string, const ast::GraphExpr*>;

bool equalGraph(const ast::GraphExpr& lhs, const ast::GraphExpr& rhs) {
  if (lhs.value.index() != rhs.value.index()) return false;
  if (const auto* name = std::get_if<ast::NameRef>(&lhs.value)) {
    return name->name == std::get<ast::NameRef>(rhs.value).name;
  }
  if (const auto* binder = std::get_if<ast::BinderRef>(&lhs.value)) {
    return binder->name == std::get<ast::BinderRef>(rhs.value).name;
  }
  if (const auto* op = std::get_if<ast::Operator>(&lhs.value)) {
    const auto& other = std::get<ast::Operator>(rhs.value);
    if (op->name != other.name ||
        op->operands.size() != other.operands.size() ||
        op->attribute.has_value() != other.attribute.has_value()) {
      return false;
    }
    if (op->attribute && op->attribute->name != other.attribute->name) {
      return false;
    }
    for (std::size_t index = 0; index < op->operands.size(); ++index) {
      if (!equalGraph(*op->operands[index], *other.operands[index]))
        return false;
    }
    return true;
  }
  if (const auto* binding = std::get_if<ast::Binding>(&lhs.value)) {
    const auto& other = std::get<ast::Binding>(rhs.value);
    return binding->binder.name == other.binder.name &&
           equalGraph(*binding->expression, *other.expression);
  }
  const auto& projection = std::get<ast::Projection>(lhs.value);
  const auto& other = std::get<ast::Projection>(rhs.value);
  return projection.index.digits == other.index.digits &&
         equalGraph(*projection.tuple, *other.tuple);
}

bool match(const ast::GraphExpr& pattern, const ast::GraphExpr& input,
           Captures& captures) {
  if (const auto* name = std::get_if<ast::NameRef>(&pattern.value)) {
    auto [it, inserted] = captures.emplace(name->name, &input);
    return inserted || equalGraph(*it->second, input);
  }

  const auto* op = std::get_if<ast::Operator>(&pattern.value);
  const auto* actual = std::get_if<ast::Operator>(&input.value);
  if (!op || !actual || op->name != actual->name ||
      op->operands.size() != actual->operands.size() || actual->attribute) {
    return false;
  }
  for (std::size_t index = 0; index < op->operands.size(); ++index) {
    if (!match(*op->operands[index], *actual->operands[index], captures)) {
      return false;
    }
  }
  return true;
}

ast::GraphExprPtr clone(const ast::GraphExpr& source) {
  auto result = std::make_shared<ast::GraphExpr>(source);
  if (auto* op = std::get_if<ast::Operator>(&result->value)) {
    for (auto& operand : op->operands) operand = clone(*operand);
  } else if (auto* binding = std::get_if<ast::Binding>(&result->value)) {
    binding->expression = clone(*binding->expression);
  } else if (auto* projection = std::get_if<ast::Projection>(&result->value)) {
    projection->tuple = clone(*projection->tuple);
  }
  return result;
}

ast::GraphExprPtr substitute(const ast::GraphExpr& rhs,
                             const Captures& captures) {
  if (const auto* name = std::get_if<ast::NameRef>(&rhs.value)) {
    return clone(*captures.at(name->name));
  }
  const auto& op = std::get<ast::Operator>(rhs.value);
  ast::Operator replacement{op.name, std::nullopt, {}};
  replacement.operands.reserve(op.operands.size());
  for (const auto& operand : op.operands) {
    replacement.operands.push_back(substitute(*operand, captures));
  }
  return std::make_shared<ast::GraphExpr>(
      ast::GraphExpr{rhs.span, std::move(replacement)});
}

}  // namespace

std::optional<ast::GraphExprPtr> rewriteOnce(const ast::Rule& rule,
                                             const ast::GraphExpr& input) {
  const auto diagnostics = validateSimpleRule(rule);
  if (!diagnostics.empty()) {
    throw std::invalid_argument("invalid simple rule: " +
                                diagnostics.front().message);
  }

  Captures captures;
  if (!match(*rule.lhs, input, captures)) return std::nullopt;
  return substitute(*rule.rhs, captures);
}

}  // namespace tepl
