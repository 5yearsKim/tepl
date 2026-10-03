#include "src/core/rule/expression.h"

namespace tepl::core::detail {

TypedExprPtr ExpressionChecker::check(const ast::ConstraintExpr& expression,
                                      std::optional<TypeId> expected) {
  const auto& origin = input_.expr_origins.at(&expression);
  return std::visit(
      [&](const auto& value) { return lower(value, origin, expected); },
      expression.value);
}

TypedExprPtr ExpressionChecker::lower(const ast::NameRef& name,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  if (const auto found = scope_.captures.find(name.name);
      found != scope_.captures.end()) {
    const auto& capture = rule_.captures.at(found->second.value);
    return make(origin, capture.type, CaptureRef{capture.id}, expected);
  }
  if (const auto found = scope_.dimensions.find(name.name);
      found != scope_.dimensions.end()) {
    const auto& dimension = rule_.dimensions.at(found->second.value);
    return make(origin, dimension.type, DimensionRef{dimension.id}, expected);
  }
  context_.report(origin, "unknown constraint name '" + name.name + "'");
  return nullptr;
}

TypedExprPtr ExpressionChecker::lower(const ast::AttributeRef& reference,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  const auto found = scope_.descriptors.find(reference.name);
  if (found == scope_.descriptors.end() ||
      !scope_.available_descriptors.contains(found->second.value)) {
    context_.report(
        origin, "descriptor '@" + reference.name + "' is not available here");
    return nullptr;
  }
  const auto& descriptor = rule_.descriptors.at(found->second.value);
  return make(origin, descriptor.type, DescriptorRef{descriptor.id}, expected);
}

TypedExprPtr ExpressionChecker::number(std::string spelling, bool decimal,
                                       const SourceOrigin& origin,
                                       std::optional<TypeId> expected) {
  auto& types = context_.types;
  const auto known = expected ? types.known(*expected) : std::nullopt;
  TypeId type;
  if (decimal) {
    type = types.concrete({TypeKind::kF64}, origin);
  } else if (known && !isNumeric(known->kind)) {
    // Keep the diagnostic a type mismatch for a literal used as (e.g.) Bool.
    type = types.concrete(
        {spelling.starts_with('-') ? TypeKind::kI64 : TypeKind::kIndex},
        origin);
  } else {
    // Integer defaulting is deferred until every rule has contributed types.
    type = types.variable(origin);
  }
  types.require(type, TypeRequirement::kNumeric, origin);
  types.literal(type, spelling, origin);
  return make(origin, type, NumericConstant{std::move(spelling), decimal},
              expected);
}

TypedExprPtr ExpressionChecker::lower(const ast::IntegerLiteral& literal,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return number(literal.digits, false, origin, expected);
}

TypedExprPtr ExpressionChecker::lower(const ast::FloatLiteral& literal,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return number(literal.digits, true, origin, expected);
}

TypedExprPtr ExpressionChecker::lower(const ast::BooleanLiteral& literal,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return make(origin, context_.types.concrete({TypeKind::kBool}, origin),
              literal.value, expected);
}

TypedExprPtr ExpressionChecker::lower(const ast::Call& call,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  const auto bound = input_.bound_functions.find(&call);
  if (call.kind == ast::CallKind::kNative &&
      bound == input_.bound_functions.end()) {
    context_.report(origin, "unknown native function '" + call.callee +
                                "'; use '$" + call.callee +
                                "(...)' to call a host function");
    return nullptr;
  }
  const auto id =
      bound != input_.bound_functions.end()
          ? bound->second
          : context_.host(call.callee, call.arguments.size(), origin);
  // Nested calls can grow the host table, so retain a copy of the signature.
  const auto signature = context_.output.host_functions.at(id.value).signature;
  if (bound != input_.bound_functions.end() &&
      signature.arguments.size() != call.arguments.size())
    context_.report(origin, "conflicting arity for bound host function '" +
                                call.callee + "'");
  if (expected) context_.types.unify(signature.result, *expected, origin);
  HostCall result{id, {}};
  bool complete = true;
  for (std::size_t i = 0; i < call.arguments.size(); ++i) {
    const auto type = i < signature.arguments.size()
                          ? std::optional<TypeId>(signature.arguments[i])
                          : std::nullopt;
    auto argument = check(*call.arguments[i], type);
    complete &= argument != nullptr;
    result.arguments.push_back(std::move(argument));
  }
  if (!complete) return nullptr;
  return make(origin, signature.result, std::move(result));
}

TypedExprPtr ExpressionChecker::lower(const ast::UnaryExpr& unary,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  const bool logical = unary.op == UnaryOp::kLogicalNot;
  if (!logical) {
    // Preserve signed spelling, including the minimum signed integer whose
    // unsigned magnitude would not fit a positive I64 value.
    const auto sign = std::string(spelling(unary.op));
    if (const auto* integer =
            std::get_if<ast::IntegerLiteral>(&unary.operand->value))
      return number(sign + integer->digits, false, origin, expected);
    if (const auto* decimal =
            std::get_if<ast::FloatLiteral>(&unary.operand->value))
      return number(sign + decimal->digits, true, origin, expected);
  }
  auto& types = context_.types;
  const auto type = logical    ? types.concrete({TypeKind::kBool}, origin)
                    : expected ? *expected
                               : types.variable(origin);
  auto operand = check(*unary.operand, type);
  if (!operand) return nullptr;
  if (!logical)
    types.require(type,
                  unary.op == UnaryOp::kNegate ? TypeRequirement::kSignedNumeric
                                               : TypeRequirement::kNumeric,
                  origin);
  return make(origin, type, UnaryExpr{unary.op, std::move(operand)}, expected);
}

TypedExprPtr ExpressionChecker::lower(const ast::BinaryExpr& binary,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto& types = context_.types;
  const auto kind = category(binary.op);
  const bool logical = kind == BinaryCategory::kLogical;
  const auto operand_type = logical ? types.concrete({TypeKind::kBool}, origin)
                                    : types.variable(origin);
  const auto result_type = kind == BinaryCategory::kArithmetic || logical
                               ? operand_type
                               : types.concrete({TypeKind::kBool}, origin);
  if (expected) types.unify(result_type, *expected, origin);
  if (!logical) {
    const auto requirement =
        kind == BinaryCategory::kEquality   ? TypeRequirement::kEquality
        : binary.op == BinaryOp::kRemainder ? TypeRequirement::kInteger
                                            : TypeRequirement::kNumeric;
    types.require(operand_type, requirement, origin);
  }
  // Both operands contribute equations to the same type variable. No separate
  // AST "hint" traversal or speculative type choices are needed.
  auto lhs = check(*binary.lhs, operand_type);
  auto rhs = check(*binary.rhs, operand_type);
  if (!lhs || !rhs) return nullptr;
  return make(origin, result_type,
              BinaryExpr{binary.op, std::move(lhs), std::move(rhs)});
}

}  // namespace tepl::core::detail
