#include "src/semantic.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace tepl {
namespace {

using Names = std::unordered_set<std::string>;

void checkGraph(const ast::GraphExpr& expression, bool lhs, Names& captures,
                std::vector<SemanticDiagnostic>& diagnostics) {
  if (const auto* name = std::get_if<ast::NameRef>(&expression.value)) {
    if (lhs) {
      captures.insert(name->name);
    } else if (!captures.contains(name->name)) {
      diagnostics.push_back(
          {expression.span, "unknown RHS variable '" + name->name + "'"});
    }
    return;
  }

  if (std::holds_alternative<ast::IntegerLiteral>(expression.value) ||
      std::holds_alternative<ast::FloatLiteral>(expression.value)) {
    return;
  }

  if (const auto* op = std::get_if<ast::Operator>(&expression.value)) {
    if (op->attribute) {
      diagnostics.push_back(
          {expression.span,
           "operator attributes are not supported by simple rules"});
    }
    for (const auto& operand : op->operands) {
      checkGraph(*operand, lhs, captures, diagnostics);
    }
    return;
  }

  diagnostics.push_back(
      {expression.span,
       "binders and projections are not supported by simple rules"});
}

}  // namespace

std::optional<DType> resolveDType(std::string_view name) {
  static constexpr std::pair<std::string_view, DType> types[] = {
      {"bool", DType::kBool}, {"i8", DType::kI8},     {"i16", DType::kI16},
      {"i32", DType::kI32},   {"i64", DType::kI64},   {"u8", DType::kU8},
      {"u16", DType::kU16},   {"u32", DType::kU32},   {"u64", DType::kU64},
      {"f16", DType::kF16},   {"bf16", DType::kBF16}, {"f32", DType::kF32},
      {"f64", DType::kF64},
  };
  for (auto [spelling, dtype] : types) {
    if (name == spelling) return dtype;
  }
  return std::nullopt;
}

std::vector<SemanticDiagnostic> validateTensorTypes(const ast::Rule& rule) {
  std::vector<SemanticDiagnostic> diagnostics;
  Names captures;
  const auto annotation = [&](const ast::DTypeAnnotation& value) {
    auto dtype = resolveDType(value.name);
    if (!dtype) {
      diagnostics.push_back({value.span, "unknown dtype '" + value.name + "'"});
    }
    return dtype;
  };
  const auto literal = [&](const auto& value) {
    if (!value.dtype) return;
    auto dtype = annotation(*value.dtype);
    if (!dtype) return;
    const bool floating = *dtype == DType::kF16 || *dtype == DType::kBF16 ||
                          *dtype == DType::kF32 || *dtype == DType::kF64;
    if (floating) return;
    auto digits = std::string_view(value.digits);
    const bool negative = digits.starts_with('-');
    if (digits.starts_with('-') || digits.starts_with('+'))
      digits.remove_prefix(1);
    std::string_view limit;
    switch (*dtype) {
      case DType::kBool:
        limit = "1";
        break;
      case DType::kI8:
        limit = negative ? "128" : "127";
        break;
      case DType::kI16:
        limit = negative ? "32768" : "32767";
        break;
      case DType::kI32:
        limit = negative ? "2147483648" : "2147483647";
        break;
      case DType::kI64:
        limit = negative ? "9223372036854775808" : "9223372036854775807";
        break;
      case DType::kU8:
        limit = "255";
        break;
      case DType::kU16:
        limit = "65535";
        break;
      case DType::kU32:
        limit = "4294967295";
        break;
      case DType::kU64:
        limit = "18446744073709551615";
        break;
      default:
        break;
    }
    while (digits.size() > 1 && digits.front() == '0') digits.remove_prefix(1);
    const bool unsigned_type = *dtype == DType::kBool || *dtype == DType::kU8 ||
                               *dtype == DType::kU16 || *dtype == DType::kU32 ||
                               *dtype == DType::kU64;
    if (digits.find('.') != std::string_view::npos ||
        (unsigned_type && negative) || digits.size() > limit.size() ||
        (digits.size() == limit.size() && digits > limit)) {
      diagnostics.push_back({value.dtype->span, "literal '" + value.digits +
                                                    "' is invalid for dtype '" +
                                                    value.dtype->name + "'"});
    }
  };
  std::function<void(const ast::GraphExpr&, bool)> graph =
      [&](const ast::GraphExpr& expression, bool lhs) {
        if (const auto* name = std::get_if<ast::NameRef>(&expression.value)) {
          if (lhs) captures.insert(name->name);
        } else if (const auto* number =
                       std::get_if<ast::IntegerLiteral>(&expression.value)) {
          literal(*number);
        } else if (const auto* number =
                       std::get_if<ast::FloatLiteral>(&expression.value)) {
          literal(*number);
        } else if (const auto* op =
                       std::get_if<ast::Operator>(&expression.value)) {
          for (const auto& child : op->operands) graph(*child, lhs);
        } else if (const auto* binding =
                       std::get_if<ast::Binding>(&expression.value)) {
          if (lhs) captures.insert(binding->binder.name);
          graph(*binding->expression, lhs);
        } else if (const auto* projection =
                       std::get_if<ast::Projection>(&expression.value)) {
          graph(*projection->tuple, lhs);
        }
      };
  if (rule.lhs) graph(*rule.lhs, true);
  if (rule.rhs) graph(*rule.rhs, false);
  Names declared;
  for (const auto& declaration : rule.declarations) {
    const auto name = std::visit([](const auto& value) { return value.name; },
                                 declaration.value);
    if (const auto* tensor = std::get_if<ast::TensorDecl>(&declaration.value)) {
      if (tensor->dtype) annotation(*tensor->dtype);
    }
    if (!declared.insert(name).second) {
      diagnostics.push_back(
          {declaration.span, "duplicate tensor declaration '" + name + "'"});
    }
    if (rule.lhs && !captures.contains(name)) {
      diagnostics.push_back(
          {declaration.span, "tensor declaration '" + name +
                                 "' does not refer to an LHS capture"});
    }
  }
  return diagnostics;
}

std::vector<SemanticDiagnostic> validateSimpleRule(const ast::Rule& rule) {
  auto diagnostics = validateTensorTypes(rule);
  if (rule.is_abstract || rule.inheritance) {
    diagnostics.push_back(
        {rule.span,
         "simple rules require concrete rules after inheritance expansion"});
    return diagnostics;
  }
  for (const auto& declaration : rule.declarations) {
    diagnostics.push_back(
        {declaration.span,
         "shape declarations are not supported by simple rules"});
  }
  for (const auto& condition : rule.conditions) {
    diagnostics.push_back(
        {condition->span,
         "where conditions are not supported by simple rules"});
  }
  for (const auto& derivation : rule.derivations) {
    diagnostics.push_back(
        {derivation.span, "derive values are not supported by simple rules"});
  }

  Names captures;
  if (rule.lhs) {
    checkGraph(*rule.lhs, true, captures, diagnostics);
  } else {
    diagnostics.push_back({rule.span, "rule is missing its LHS"});
  }
  if (rule.rhs) {
    checkGraph(*rule.rhs, false, captures, diagnostics);
  } else {
    diagnostics.push_back({rule.span, "rule is missing its RHS"});
  }
  return diagnostics;
}

}  // namespace tepl
