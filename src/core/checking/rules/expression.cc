#include "src/core/checking/rules/expression.h"

#include <algorithm>

namespace tepl::core::detail {

TypedExprPtr ExpressionChecker::check(const syntax::ConstraintExpr& expression,
                                      std::optional<TypeId> expected) {
  const auto& origin = input_.expr_origins.at(expression.resolution_id);
  if (expected) {
    if (auto type = context_.types.known(*expected); type && type->optional) {
      if (auto conditional =
              std::get_if<syntax::ConditionalExpr>(&expression.value))
        return lower(*conditional, origin, expected);
      if (auto name = std::get_if<syntax::NameRef>(&expression.value);
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

TypedExprPtr ExpressionChecker::lower(const syntax::NameRef& name,
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

TypedExprPtr ExpressionChecker::lower(const syntax::NodeProperty& property,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto node = scope_.nodes.find(property.node);
  const auto& path = property.path;
  if (node != scope_.nodes.end() && !path.empty() && path[0] == "attrs") {
    if (!node->second.descriptor) {
      context_.report(origin, "node has no attribute schema");
      return {};
    }
    auto id = *node->second.descriptor;
    const auto& descriptor = rule_.descriptors.at(id.value);
    if (descriptor.kind == Descriptor::Kind::kInferred) {
      context_.report(
          origin,
          "automatic RHS attributes are unavailable before construction");
      return {};
    }
    // Explicit RHS fields may be read in where after their derivations.
    if (!scope_.available_descriptors.contains(id.value) &&
        section_ != builtins::Context::kWhere) {
      context_.report(origin,
                      "derived attributes are not available yet (forward "
                      "reference or cycle)");
      return {};
    }
    if (path.size() == 1)
      return make(origin, descriptor.type, DescriptorRef{id}, expected);
    if (path.size() == 2) {
      auto type = context_.types.known(descriptor.type);
      if (type && type->schema) {
        const auto& schema =
            context_.output.attribute_schemas.at(type->schema->value);
        for (std::size_t f = 0; f < schema.fields.size(); ++f) {
          const auto& field = schema.fields[f];
          if (field.name != path[1]) continue;
          auto t =
              resolveType(field.type)
                  .value_or(Type{TypeKind::kOpaque, {}, 0, false, field.type});
          for (std::size_t d = 0; d < field.list_depth; ++d) t = listType(t);
          t.optional = field.optional;
          return make(origin, context_.types.concrete(t, origin),
                      DescriptorFieldRef{id, schema.id, f}, expected);
        }
      }
    }
    context_.report(origin, "unknown node attribute property");
    return {};
  }
  if (section_ != builtins::Context::kWhere) {
    context_.report(
        origin, "logical axis properties are available in where conditions");
    return {};
  }
  NodeAxisRef ref;
  ref.node = property.node;
  if (node == scope_.nodes.end()) {
    auto capture = scope_.captures.find(property.node);
    if (capture == scope_.captures.end() ||
        path != std::vector<std::string>{"axes", "order"}) {
      context_.report(origin, "unknown node or unsupported input property '" +
                                  property.node + "'");
      return {};
    }
    ref.capture = capture->second;
    ref.projection = "order";
  } else {
    ref.rhs = node->second.rhs;
    if ((path == std::vector<std::string>{"axes", "partition"}) ||
        (path == std::vector<std::string>{"legal"} && ref.rhs)) {
      ref.projection = path.back();
    } else if (path == std::vector<std::string>{"required_order"} && ref.rhs) {
      ref.projection = "required_order";
    } else if (path == std::vector<std::string>{"axes", "order"}) {
      ref.projection = "order";
    } else if (path.size() == 3 && path[0] == "axes") {
      ref.group = path[1];
      ref.projection = path[2];
      const auto& op =
          context_.output.operations.at(node->second.operation.value);
      if (!op.semantic_definition ||
          std::none_of(op.semantic_definition->groups.begin(),
                       op.semantic_definition->groups.end(),
                       [&](const auto& g) { return g.name == ref.group; })) {
        context_.report(origin,
                        "unknown semantic axis group '" + ref.group + "'");
        return {};
      }
      if (ref.projection != "members" && ref.projection != "order" &&
          ref.projection != "positions") {
        context_.report(origin, "unknown axis group projection");
        return {};
      }
      if (ref.projection == "positions") {
        auto group =
            std::find_if(op.semantic_definition->groups.begin(),
                         op.semantic_definition->groups.end(),
                         [&](const auto& g) { return g.name == ref.group; });
        const auto id = static_cast<std::size_t>(
            group - op.semantic_definition->groups.begin());
        bool exposed = false;
        auto visit = [&](auto&& self, const semantics::Meta& m) -> void {
          if (m.kind == semantics::Meta::Kind::Group && m.index == id)
            exposed = true;
          for (const auto& arg : m.args) self(self, *arg);
        };
        visit(visit, *op.semantic_definition->output);
        if (!exposed) {
          context_.report(origin,
                          "semantic group is absent from the operation output");
          return {};
        }
      }
      if (!ref.rhs && ref.projection == "positions")
        return make(origin,
                    context_.types.concrete({TypeKind::kIndexList}, origin),
                    OutputAxisGroupRef{property.node, ref.group}, expected);
    } else {
      context_.report(origin, "unknown semantic node property");
      return {};
    }
    if (ref.rhs && ref.projection != "members" &&
        ref.projection != "required_order" && node->second.descriptor &&
        rule_.descriptors.at(node->second.descriptor->value).kind ==
            Descriptor::Kind::kInferred) {
      context_.report(origin,
                      "actual automatic RHS order and positions are "
                      "unavailable before construction; use group.members");
      return {};
    }
  }
  auto t = ref.projection == "legal" ? Type{TypeKind::kBool}
           : ref.projection == "positions" || ref.projection == "partition"
               ? Type{TypeKind::kIndexList}
           : ref.projection == "members" ? logicalAxisSetType()
                                         : listType(logicalAxisType());
  return make(origin, context_.types.concrete(t, origin), ref, expected);
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

TypedExprPtr ExpressionChecker::lower(const syntax::IntegerLiteral& literal,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return number(literal.digits, false, origin, expected);
}

TypedExprPtr ExpressionChecker::lower(const syntax::FloatLiteral& literal,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return number(literal.digits, true, origin, expected);
}

TypedExprPtr ExpressionChecker::lower(const syntax::BooleanLiteral& literal,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return make(origin, context_.types.concrete({TypeKind::kBool}, origin),
              literal.value, expected);
}

TypedExprPtr ExpressionChecker::lower(const syntax::Call& call,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return builtin(call, origin, expected);
}

TypedExprPtr ExpressionChecker::builtin(const syntax::Call& call,
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
      case S::kAxisSet:
        return types.concrete(logicalAxisSetType(), origin);
      case S::kAxisList:
        return types.concrete(listType(logicalAxisType()), origin);
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

TypedExprPtr ExpressionChecker::lower(const syntax::UnaryExpr& unary,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  const bool logical = unary.op == UnaryOp::kLogicalNot;
  if (!logical) {
    // Preserve signed spelling, including the minimum signed integer whose
    // unsigned magnitude would not fit a positive I64 value.
    const auto sign = std::string(spelling(unary.op));
    if (const auto* integer =
            std::get_if<syntax::IntegerLiteral>(&unary.operand->value))
      return number(sign + integer->digits, false, origin, expected);
    if (const auto* decimal =
            std::get_if<syntax::FloatLiteral>(&unary.operand->value))
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

TypedExprPtr ExpressionChecker::lower(const syntax::BinaryExpr& binary,
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
  const auto* left_name = std::get_if<syntax::NameRef>(&binary.lhs->value);
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
                                      const syntax::ConstraintExpr& value,
                                      const SourceOrigin& origin) {
  if (scope_.nodes.contains(name) || locals_.contains(name) ||
      scope_.captures.contains(name) || scope_.dimensions.contains(name) ||
      scope_.dtypes.contains(name) || resolveDType(name) || name == "none" ||
      name == "default" || name == "high" || name == "highest") {
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
TypedExprPtr ExpressionChecker::lower(const syntax::LocalBinding& binding,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId>) {
  return local(binding.name, *binding.value, origin);
}
TypedExprPtr ExpressionChecker::lower(const syntax::StringLiteral& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  return make(origin, context_.types.concrete({TypeKind::kString}, origin),
              StringConstant{v.value}, expected);
}
TypedExprPtr ExpressionChecker::lower(const syntax::ListExpr& v,
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
TypedExprPtr ExpressionChecker::lower(const syntax::IndexExpr& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto list = check(*v.value);
  auto index =
      check(*v.index, context_.types.concrete({TypeKind::kIndex}, origin));
  if (!list || !index) return nullptr;
  auto element = context_.types.element(list->type, origin);
  return make(origin, element, IndexExpr{list, index}, expected);
}
TypedExprPtr ExpressionChecker::lower(const syntax::ConditionalExpr& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  auto condition =
      check(*v.condition, context_.types.concrete({TypeKind::kBool}, origin));
  auto type = expected ? *expected : context_.types.variable(origin);
  auto yes = check(*v.then_value, type), no = check(*v.else_value, type);
  if (!condition || !yes || !no) return nullptr;
  return make(origin, type, ConditionalExpr{condition, yes, no});
}
TypedExprPtr ExpressionChecker::lower(const syntax::ComprehensionExpr& v,
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
TypedExprPtr ExpressionChecker::lower(const syntax::DescriptorConstructor& v,
                                      const SourceOrigin& origin,
                                      std::optional<TypeId> expected) {
  TypedExprPtr spread;
  if (v.spread)
    spread =
        lower(syntax::NodeProperty{*v.spread, {"attrs"}}, origin, expected);
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
  std::map<std::string, const syntax::DescriptorConstructor::Field*>
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
