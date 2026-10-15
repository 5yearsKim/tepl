#include "src/core/rule/expression.h"

#include <algorithm>

namespace tepl::core::detail {

TypedExprPtr ExpressionChecker::check(const ast::ConstraintExpr& expression,
                                      std::optional<TypeId> expected) {
  const auto& origin = input_.expr_origins.at(&expression);
  if (expected) {
    if (auto type = context_.types.known(*expected); type && type->optional) {
      if (auto conditional =
              std::get_if<ast::ConditionalExpr>(&expression.value))
        return lower(*conditional, origin, expected);
      if (auto name = std::get_if<ast::NameRef>(&expression.value);
          name && name->name == "none")
        return make(origin, *expected, NoneConstant{});
      auto value =
          std::visit([&](const auto& v) { return lower(v, origin, {}); },
                     expression.value);
      if (!value) return nullptr;
      if (auto actual = context_.types.known(value->type);
          actual && actual->optional) {
        context_.types.unify(value->type, *expected, origin);
        return value;
      }
      type->optional = false;
      context_.types.unify(value->type, context_.types.concrete(*type, origin),
                           origin);
      return make(origin, *expected, OptionalValue{value});
    }
  }
  return std::visit(
      [&](const auto& value) { return lower(value, origin, expected); },
      expression.value);
}

TypedExprPtr ExpressionChecker::lower(const ast::NameRef& name,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  if (auto found = locals_.find(name.name); found != locals_.end()) {
    const auto& local = rule_.locals.at(found->second.value);
    return make(origin, local.type, LocalRef{local.id}, expected);
  }
  if (name.name == "default" || name.name == "high" || name.name == "highest")
    return make(origin, context_.types.concrete({TypeKind::kPrecision}, origin),
                EnumConstant{name.name}, expected);
  if (name.name == "none") {
    context_.report(origin, "none requires an optional attribute context");
    return nullptr;
  }
  if (auto dtype = resolveDType(name.name))
    return make(origin, context_.types.concrete({TypeKind::kDType}, origin),
                *dtype, expected);
  if (auto variable = scope_.dtypes.find(name.name);
      variable != scope_.dtypes.end()) {
    const auto& symbol = rule_.dtypes.at(variable->second.value);
    return make(origin, symbol.type, DTypeVariableRef{symbol.id}, expected);
  }
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

TypedExprPtr ExpressionChecker::lower(const ast::DescriptorField& field,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto descriptor = lower(field.descriptor, origin, {});
  if (!descriptor) return nullptr;
  if (field.output_group) {
    if (section_ != builtins::Context::kWhere) {
      context_.report(origin,
                      "output-axis group references are only supported in "
                      "where conditions");
      return nullptr;
    }
    return make(
        origin, context_.types.concrete({TypeKind::kIndexList}, origin),
        OutputAxisGroupRef{
            std::get<DescriptorRef>(descriptor->value).descriptor, field.field},
        expected);
  }
  auto type = context_.types.known(descriptor->type);
  if (!type || !type->schema) {
    context_.report(
        origin, "descriptor field access requires a known attribute schema");
    return nullptr;
  }
  const auto& schema =
      context_.output.attribute_schemas.at(type->schema->value);
  for (std::size_t i = 0; i < schema.fields.size(); ++i) {
    const auto& attribute = schema.fields[i];
    if (attribute.name != field.field) continue;
    auto field_type =
        resolveType(attribute.type)
            .value_or(Type{TypeKind::kOpaque, {}, 0, false, attribute.type});
    for (std::size_t depth = 0; depth < attribute.list_depth; ++depth)
      field_type = listType(field_type);
    field_type.optional = attribute.optional;
    return make(origin, context_.types.concrete(field_type, origin),
                DescriptorFieldRef{
                    std::get<DescriptorRef>(descriptor->value).descriptor,
                    schema.id, i},
                expected);
  }
  context_.report(origin, "unknown descriptor field '" + field.field + "'");
  return nullptr;
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
    // A nonnumeric expected type still produces a type-mismatch diagnostic.
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
  return builtin(call, origin, expected);
}

TypedExprPtr ExpressionChecker::builtin(const ast::Call& call,
                                        const SourceOrigin& origin,
                                        std::optional<TypeId> expected) {
  const auto* signature = builtins::resolveBuiltin(call.callee);
  if (!signature) {
    context_.report(origin, "unknown builtin '" + call.callee + "'");
    return nullptr;
  }
  if (!signature->availableIn(section_)) {
    context_.report(origin, "builtin '" + call.callee +
                                "' is not available in " +
                                std::string(builtins::contextName(section_)) +
                                " expressions");
    return nullptr;
  }
  const auto count = signature->arguments.size();
  if (!signature->acceptsArity(call.arguments.size())) {
    context_.report(origin,
                    "builtin '" + call.callee + "' expects " +
                        (signature->variadic ? "at least " : "exactly ") +
                        std::to_string(count) + " arguments");
    return nullptr;
  }
  auto& types = context_.types;
  std::optional<TypeId> element, integer;
  auto genericElement = [&]() {
    if (!element) element = types.variable(origin);
    return *element;
  };
  auto integerPair = [&]() {
    if (!integer) {
      integer = types.variable(origin);
      types.require(*integer, TypeRequirement::kInteger, origin);
    }
    return *integer;
  };
  const auto type = [&](builtins::SignatureType pattern) -> TypeId {
    using S = builtins::SignatureType;
    switch (pattern) {
      case S::kDType:
        return types.concrete({TypeKind::kDType}, origin);
      case S::kInteger:
        return signature->rule_integer_pair
                   ? integerPair()
                   : types.concrete({TypeKind::kIndex}, origin);
      case S::kT:
        return genericElement();
      case S::kIntegerList:
        return types.concrete({TypeKind::kIndexList}, origin);
      case S::kListT:
        return types.list(genericElement(), origin);
      case S::kBoolean:
        return types.concrete({TypeKind::kBool}, origin);
      case S::kBooleanList:
        return types.concrete({TypeKind::kBool, {}, 1}, origin);
    }
    return types.variable(origin);
  };
  // Only instantiate generic variables that occur in this signature.
  const auto result_type = type(signature->result);
  if (expected) types.unify(result_type, *expected, origin);
  BuiltinCall result{signature->builtin, {}};
  bool complete = true;
  for (std::size_t i = 0; i < call.arguments.size(); ++i) {
    auto argument = check(*call.arguments[i],
                          type(signature->arguments[std::min(i, count - 1)]));
    complete &= argument != nullptr;
    result.arguments.push_back(std::move(argument));
  }
  if (!complete) return nullptr;
  return make(origin, result_type, std::move(result));
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
  // Both operands constrain the same type variable.
  TypedExprPtr lhs, rhs;
  const auto* left_name = std::get_if<ast::NameRef>(&binary.lhs->value);
  if (kind == BinaryCategory::kEquality && left_name &&
      left_name->name == "none") {
    rhs = check(*binary.rhs, operand_type);
    lhs = check(*binary.lhs, operand_type);
  } else {
    lhs = check(*binary.lhs, operand_type);
    rhs = check(*binary.rhs, operand_type);
  }
  if (!lhs || !rhs) return nullptr;
  return make(origin, result_type,
              BinaryExpr{binary.op, std::move(lhs), std::move(rhs)});
}

TypedExprPtr ExpressionChecker::local(const std::string& name,
                                      const ast::ConstraintExpr& value,
                                      const SourceOrigin& origin) {
  if (locals_.contains(name) || scope_.captures.contains(name) ||
      scope_.dimensions.contains(name) || scope_.dtypes.contains(name) ||
      resolveDType(name) || name == "none" || name == "default" ||
      name == "high" || name == "highest") {
    context_.report(origin,
                    "duplicate or conflicting local name '" + name + "'");
    return nullptr;
  }
  auto checked = check(value);
  if (!checked) return nullptr;
  LocalId id{rule_.locals.size()};
  rule_.locals.push_back({id, name, checked->type});
  locals_.emplace(name, id);
  return make(origin, checked->type, LocalLet{id, checked});
}
TypedExprPtr ExpressionChecker::lower(const ast::LocalBinding& binding,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId>) {
  return local(binding.name, *binding.value, origin);
}
TypedExprPtr ExpressionChecker::lower(const ast::StringLiteral& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return make(origin, context_.types.concrete({TypeKind::kString}, origin),
              StringConstant{v.value}, expected);
}
TypedExprPtr ExpressionChecker::lower(const ast::ListExpr& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto& types = context_.types;
  auto element = types.variable(origin);
  auto list = types.list(element, origin);
  if (expected) {
    types.unify(list, *expected, origin);
    if (auto t = types.known(*expected))
      if (auto e = elementType(*t))
        types.unify(element, types.concrete(*e, origin), origin);
  }
  ListExpr result;
  for (const auto& e : v.elements) {
    auto checked = check(*e, element);
    if (!checked) return nullptr;
    result.elements.push_back(checked);
  }
  return make(origin, list, result);
}
TypedExprPtr ExpressionChecker::lower(const ast::IndexExpr& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto list = check(*v.value);
  auto index =
      check(*v.index, context_.types.concrete({TypeKind::kIndex}, origin));
  if (!list || !index) return nullptr;
  auto element = context_.types.element(list->type, origin);
  return make(origin, element, IndexExpr{list, index}, expected);
}
TypedExprPtr ExpressionChecker::lower(const ast::ConditionalExpr& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto condition =
      check(*v.condition, context_.types.concrete({TypeKind::kBool}, origin));
  auto type = expected ? *expected : context_.types.variable(origin);
  auto yes = check(*v.then_value, type), no = check(*v.else_value, type);
  if (!condition || !yes || !no) return nullptr;
  return make(origin, type, ConditionalExpr{condition, yes, no});
}
TypedExprPtr ExpressionChecker::lower(const ast::ComprehensionExpr& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto iterable = check(*v.iterable);
  if (!iterable) return nullptr;
  auto item = context_.types.element(iterable->type, origin);
  const auto previous = locals_;
  LocalId id{rule_.locals.size()};
  rule_.locals.push_back({id, v.variable, item});
  locals_[v.variable] = id;
  std::optional<TypeId> result_element;
  if (expected) result_element = context_.types.element(*expected, origin);
  auto element = check(*v.element, result_element);
  locals_ = previous;
  if (!element) return nullptr;
  auto result = context_.types.list(element->type, origin);
  return make(origin, result, ComprehensionExpr{element, id, iterable},
              expected);
}
TypedExprPtr ExpressionChecker::lower(const ast::DescriptorConstructor& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  TypedExprPtr spread;
  if (v.spread) spread = lower(*v.spread, origin, expected);
  auto type =
      expected ? context_.types.known(*expected) : std::optional<Type>{};
  if ((!type || !type->schema) && spread)
    type = context_.types.known(spread->type);
  if (!type || type->kind != TypeKind::kDescriptor || !type->schema) {
    context_.report(
        origin,
        "descriptor constructor requires a schema from RHS usage or a spread");
    return nullptr;
  }
  auto schema = *type->schema;
  const auto& fields =
      context_.output.attribute_schemas.at(schema.value).fields;
  std::map<std::string, const ast::DescriptorConstructor::Field*>
      explicit_fields;
  for (const auto& f : v.fields) {
    if (!explicit_fields.emplace(f.name, &f).second)
      context_.report(origin, "duplicate constructor field '" + f.name + "'");
    if (std::none_of(fields.begin(), fields.end(),
                     [&](const auto& a) { return a.name == f.name; }))
      context_.report(origin, "unknown constructor field '" + f.name + "'");
  }
  DescriptorConstruct result{schema, {}};
  for (std::size_t i = 0; i < fields.size(); ++i) {
    const auto& f = fields[i];
    auto field_type = resolveType(f.type).value_or(
        Type{TypeKind::kOpaque, {}, 0, false, f.type});
    for (std::size_t depth = 0; depth < f.list_depth; ++depth)
      field_type = listType(field_type);
    field_type.optional = f.optional;
    auto id = context_.types.concrete(field_type, origin);
    TypedExprPtr value;
    if (auto found = explicit_fields.find(f.name);
        found != explicit_fields.end()) {
      value = check(*found->second->value, id);
    } else if (spread) {
      // Spread is restricted to a descriptor reference, so field reads retain
      // identity.
      value = make(
          origin, id,
          DescriptorFieldRef{std::get<DescriptorRef>(spread->value).descriptor,
                             schema, i});
    } else if (f.empty_default) {
      auto underlying = field_type;
      underlying.optional = false;
      value =
          make(origin, context_.types.concrete(underlying, origin), ListExpr{});
      if (f.optional) value = make(origin, id, OptionalValue{value});
    } else if (f.optional)
      value = make(origin, id, NoneConstant{});
    else
      context_.report(origin,
                      "missing required constructor field '" + f.name + "'");
    if (!value) return nullptr;
    result.fields.push_back(value);
  }
  return make(origin, context_.types.concrete(*type, origin), result, expected);
}

}  // namespace tepl::core::detail
