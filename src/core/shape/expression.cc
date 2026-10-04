#include "src/core/shape/expression.h"

#include <algorithm>

namespace tepl::core::shape::detail {

ExprPtr ExpressionChecker::check(const ast::ShapeExpr& expression,
                                 std::optional<TypeId> expected) {
  const auto origin = context_.origin(expression.span);
  auto result =
      std::visit([&](const auto& value) { return lower(value, origin); },
                 expression.value);
  if (result && expected) context_.types.unify(result->type, *expected, origin);
  return result;
}

ExprPtr ExpressionChecker::lower(const ast::NameRef& name,
                                 const SourceOrigin& origin) {
  const auto found = context_.scope.find(name.name);
  if (found == context_.scope.end()) {
    context_.report(origin, "unknown shape name '" + name.name + "'");
    return nullptr;
  }
  const auto& symbol = context_.program.symbols[found->second.value];
  return make(origin, symbol.type, SymbolRef{symbol.id});
}

ExprPtr ExpressionChecker::lower(const ast::IntegerLiteral& value,
                                 const SourceOrigin& origin) {
  if (!validIntegerLiteral(value.digits)) {
    context_.report(origin,
                    "shape integer literal is outside signed 128-bit range");
    return nullptr;
  }
  return make(origin, concrete(Type::Kind::kInteger, origin),
              Integer{value.digits});
}

ExprPtr ExpressionChecker::lower(const ast::BooleanLiteral& value,
                                 const SourceOrigin& origin) {
  return make(origin, concrete(Type::Kind::kBoolean, origin), value.value);
}

ExprPtr ExpressionChecker::lower(const ast::ShapeAttrs&,
                                 const SourceOrigin& origin) {
  context_.report(origin, "attrs requires a field access in shape expressions");
  return nullptr;
}

ExprPtr ExpressionChecker::lower(const ast::ShapeField& field,
                                 const SourceOrigin& origin) {
  if (!std::holds_alternative<ast::ShapeAttrs>(field.value->value)) {
    context_.report(origin, "shape field access requires attrs.field");
    return nullptr;
  }
  if (!context_.operation.attributes) {
    context_.report(
        origin, "operation has no attribute schema for attrs." + field.field);
    return nullptr;
  }
  const auto schema_id = *context_.operation.attributes;
  const auto& fields =
      context_.analysis.output.attribute_schemas[schema_id.value].fields;
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const auto& attribute = fields[i];
    if (attribute.name != field.field) continue;
    if (attribute.optional ||
        (attribute.type != "index" && attribute.type != "i64" &&
         attribute.type != "bool")) {
      context_.report(origin,
                      "attribute '" + field.field +
                          "' is optional or opaque to shape expressions");
      return nullptr;
    }
    const auto kind =
        attribute.type == "bool" ? Type::Kind::kBoolean : Type::Kind::kInteger;
    return make(origin, concrete(kind, origin, attribute.list_depth),
                AttributeRef{schema_id, i});
  }
  context_.report(origin,
                  "unknown shape attribute field '" + field.field + "'");
  return nullptr;
}

ExprPtr ExpressionChecker::lower(const ast::ShapeCall& call,
                                 const SourceOrigin& origin) {
  const auto* signature = builtins::resolveBuiltin(call.callee);
  if (!signature) {
    context_.report(origin, "unknown shape builtin '" + call.callee + "'");
    return nullptr;
  }
  if (!signature->availableIn(builtins::Context::kShape)) {
    context_.report(origin, "builtin '" + call.callee +
                                "' is not available in shape expressions");
    return nullptr;
  }
  const auto count = signature->arguments.size();
  if (call.arguments.size() < count ||
      (!signature->variadic && call.arguments.size() != count)) {
    context_.report(origin,
                    "shape builtin '" + call.callee + "' expects " +
                        (signature->variadic ? "at least " : "exactly ") +
                        std::to_string(count) + " arguments");
    return nullptr;
  }
  std::optional<TypeId> element;
  std::optional<TypeId> list;
  const auto type = [&](builtins::SignatureType pattern) -> TypeId {
    switch (pattern) {
      case builtins::SignatureType::kInteger:
        return concrete(Type::Kind::kInteger, origin);
      case builtins::SignatureType::kBoolean:
        return concrete(Type::Kind::kBoolean, origin);
      case builtins::SignatureType::kIntegerList:
        return concrete(Type::Kind::kInteger, origin, 1);
      case builtins::SignatureType::kBooleanList:
        return concrete(Type::Kind::kBoolean, origin, 1);
      case builtins::SignatureType::kT:
      case builtins::SignatureType::kListT:
        if (!element) element = context_.types.variable(origin);
        if (pattern == builtins::SignatureType::kT) return *element;
        if (!list) list = context_.types.list(*element, origin);
        return *list;
    }
    return context_.types.variable(origin);
  };
  Call result{signature->builtin, {}};
  for (std::size_t i = 0; i < call.arguments.size(); ++i) {
    const auto expected = type(signature->arguments[std::min(i, count - 1)]);
    auto argument = check(*call.arguments[i], expected);
    if (!argument) return nullptr;
    result.arguments.push_back(std::move(argument));
  }
  return make(origin, type(signature->result), std::move(result));
}

ExprPtr ExpressionChecker::lower(const ast::ShapeUnary& unary,
                                 const SourceOrigin& origin) {
  // Keep the minimum signed value representable without first checking its
  // positive magnitude as a separate (out-of-range) literal.
  if (unary.op != UnaryOp::kLogicalNot) {
    if (const auto* integer =
            std::get_if<ast::IntegerLiteral>(&unary.operand->value))
      return lower(ast::IntegerLiteral{std::string(spelling(unary.op)) +
                                       integer->digits},
                   origin);
  }
  const auto type =
      concrete(unary.op == UnaryOp::kLogicalNot ? Type::Kind::kBoolean
                                                : Type::Kind::kInteger,
               origin);
  auto operand = check(*unary.operand, type);
  if (!operand) return nullptr;
  return make(origin, type, Unary{unary.op, std::move(operand)});
}

ExprPtr ExpressionChecker::lower(const ast::ShapeBinary& binary,
                                 const SourceOrigin& origin) {
  const auto kind = category(binary.op);
  const auto operand_type =
      kind == BinaryCategory::kEquality
          ? context_.types.variable(origin)
          : concrete(kind == BinaryCategory::kLogical ? Type::Kind::kBoolean
                                                      : Type::Kind::kInteger,
                     origin);
  auto lhs = check(*binary.lhs, operand_type);
  auto rhs = check(*binary.rhs, operand_type);
  if (!lhs || !rhs) return nullptr;
  const auto result_type = kind == BinaryCategory::kArithmetic
                               ? operand_type
                               : concrete(Type::Kind::kBoolean, origin);
  return make(origin, result_type,
              Binary{binary.op, std::move(lhs), std::move(rhs)});
}

ExprPtr ExpressionChecker::lower(const ast::ShapeList& list,
                                 const SourceOrigin& origin) {
  const auto element_type = context_.types.variable(origin);
  List result;
  for (const auto& element : list.elements) {
    auto checked = check(*element, element_type);
    if (!checked) return nullptr;
    result.elements.push_back(std::move(checked));
  }
  return make(origin, context_.types.list(element_type, origin),
              std::move(result));
}

ExprPtr ExpressionChecker::lower(const ast::ShapeIndex& index,
                                 const SourceOrigin& origin) {
  const auto element_type = context_.types.variable(origin);
  auto value = check(*index.value, context_.types.list(element_type, origin));
  auto position = check(*index.index, concrete(Type::Kind::kInteger, origin));
  if (!value || !position) return nullptr;
  return make(origin, element_type,
              Index{std::move(value), std::move(position)});
}

ExprPtr ExpressionChecker::lower(const ast::ShapeConditional& conditional,
                                 const SourceOrigin& origin) {
  auto condition =
      check(*conditional.condition, concrete(Type::Kind::kBoolean, origin));
  const auto type = context_.types.variable(origin);
  auto then_value = check(*conditional.then_value, type);
  auto else_value = check(*conditional.else_value, type);
  if (!condition || !then_value || !else_value) return nullptr;
  return make(origin, type,
              Conditional{std::move(condition), std::move(then_value),
                          std::move(else_value)});
}

ExprPtr ExpressionChecker::lower(const ast::ShapeComprehension& comprehension,
                                 const SourceOrigin& origin) {
  const auto element_type = context_.types.variable(origin);
  // The iterable is checked in the outer scope.
  auto iterable =
      check(*comprehension.iterable, context_.types.list(element_type, origin));
  if (!iterable) return nullptr;
  if (comprehension.variable == "attrs") {
    context_.report(origin, "shape binding cannot use reserved name 'attrs'");
    return nullptr;
  }
  const auto previous = context_.scope.find(comprehension.variable);
  const auto saved = previous == context_.scope.end()
                         ? std::optional<SymbolId>{}
                         : previous->second;
  const auto symbol =
      context_.symbol(comprehension.variable, element_type,
                      context_.origin(comprehension.variable_span));
  context_.scope[comprehension.variable] = symbol;
  auto element = check(*comprehension.element);
  if (saved)
    context_.scope[comprehension.variable] = *saved;
  else
    context_.scope.erase(comprehension.variable);
  if (!element) return nullptr;
  const auto result_type = context_.types.list(element->type, origin);
  return make(origin, result_type,
              Comprehension{std::move(element), symbol, std::move(iterable)});
}

}  // namespace tepl::core::shape::detail
