#include <functional>
#include <map>
#include <set>
#include <stdexcept>

#include "src/core/type/literal.h"
#include "src/program/program.h"

namespace tepl::program {
namespace {
void require(bool b, const std::string& s) {
  if (!b) throw std::invalid_argument("compiled program: " + s);
}
// Portable axis expressions have a closed instruction set.
void callArity(const std::string& name, std::size_t count) {
  if (name == "concat" || name == "union_axes") {
    require(count > 0, "empty variadic metadata call");
    return;
  }
  if (name.starts_with("dtype_")) {
    require(count == 0 && core::resolveDType(name.substr(6)).has_value(),
            "invalid dtype instruction");
    return;
  }
  if (const auto* b = core::builtins::resolveBuiltin(name)) {
    require(b->variadic ? count >= b->arguments.size()
                        : count == b->arguments.size(),
            "metadata builtin arity mismatch");
    return;
  }
  static const std::map<std::string, std::size_t> arities = {
      {"+", 2},
      {"-", 2},
      {"*", 2},
      {"/", 2},
      {"%", 2},
      {"==", 2},
      {"!=", 2},
      {"<", 2},
      {"<=", 2},
      {">", 2},
      {">=", 2},
      {"&&", 2},
      {"||", 2},
      {"!", 1},
      {"if", 3},
      {"index", 2},
      {"rank", 1},
      {"shape", 1},
      {"all_axes", 1},
      {"empty_axes", 0},
      {"permute", 2},
      {"axis_partition", 2},
      {"inverse", 1},
      {"order_positions", 1},
      {"offset_positions", 2},
      {"offset_extents", 2},
      {"scale_extents", 2},
      {"extent_add", 2},
      {"extent_sub", 2},
      {"extent_mul", 2},
      {"extent_floor_div", 2},
      {"extent_le", 2},
      {"is_valid_extents", 1},
      {"is_positive_extents", 1},
      {"distinct_axes", 1},
      {"axis_extents", 1},
      {"positions", 2},
      {"membership_positions", 2},
      {"matched_positions", 2},
      {"unique_axes", 1},
      {"ordered_complement", 2},
      {"filter_axes", 2},
      {"intersection_axes", 2},
      {"difference_axes", 2},
      {"same_set", 2},
      {"disjoint", 2},
      {"subset", 2},
      {"replace_axes", 3},
      {"place", 3},
      {"place_axes", 3}};
  auto it = arities.find(name);
  require(it != arities.end() && it->second == count,
          "unsupported metadata instruction or arity: " + name);
}
template <class V>
void ids(const V& v) {
  for (std::size_t i = 0; i < v.size(); ++i)
    require(v[i].id.value == i, "table IDs must be consecutive");
}
}  // namespace
void validate(const Program& program) {
  using namespace core;
  const auto& p = program.ir();
  require(program.axes_.size() == p.rules.size(),
          "axis definition table size mismatch");
  auto fieldType = [](const AttributeField& f) {
    auto t = resolveType(f.type).value_or(Type{TypeKind::kOpaque});
    if (t.kind == TypeKind::kOpaque) t.opaque_name = f.type;
    for (std::size_t i = 0; i < f.list_depth; ++i) t = listType(t);
    t.optional = f.optional;
    return t;
  };
  ids(p.operations);
  ids(p.attribute_schemas);
  ids(p.rules);
  ids(p.graphs);
  auto type = [&](TypeId id) -> const Type& {
    require(id.value < p.types.size(), "unknown type ID");
    return p.types[id.value];
  };
  for (const auto& t : p.types)
    if (t.schema)
      require(t.kind == TypeKind::kDescriptor &&
                  t.schema->value < p.attribute_schemas.size(),
              "invalid descriptor schema");
  auto arity = [&](OpId id, std::size_t count) {
    require(id.value < p.operations.size(), "unknown operation ID");
    const auto& op = p.operations[id.value];
    bool variadic = !op.operands.empty() && op.operands.back().variadic;
    require(variadic ? count >= op.operands.size() - 1
                     : count == op.operands.size(),
            "incorrect operation arity");
  };
  for (const auto& op : p.operations) {
    type(op.result);
    for (const auto& arg : op.operands) type(arg.type);
    if (op.attributes)
      require(op.attributes->value < p.attribute_schemas.size(),
              "unknown operation schema");
    for (const auto* meta : {&op.shape, &op.dtype})
      if (*meta) {
        const auto& m = **meta;
        ids(m.symbols);
        auto mt = [&](metadata::TypeId t) -> metadata::Type {
          require(t.value < m.types.size(), "unknown metadata type");
          return m.types[t.value];
        };
        using MT = metadata::Type;
        using MK = MT::Kind;
        require(m.kind == (meta == &op.shape ? metadata::ProgramKind::Shape
                                             : metadata::ProgramKind::DType),
                "metadata program kind mismatch");
        std::set<std::size_t> bound;
        for (const auto& s : m.symbols) mt(s.type);
        for (const auto& a : m.parameters) {
          require(a.symbol.value < m.symbols.size() &&
                      a.operand < op.operands.size(),
                  "invalid metadata parameter");
          require(a.variadic == op.operands[a.operand].variadic,
                  "metadata parameter variadic mismatch");
          MT expected{m.kind == metadata::ProgramKind::Shape ? MK::kInteger
                                                             : MK::kDType,
                      (m.kind == metadata::ProgramKind::Shape ? 1u : 0u) +
                          (a.variadic ? 1u : 0u)};
          require(mt(m.symbols[a.symbol.value].type) == expected,
                  "metadata parameter type mismatch");
          require(bound.insert(a.symbol.value).second,
                  "duplicate metadata parameter");
        }
        std::function<void(const metadata::ExprPtr&)> expr =
            [&](const metadata::ExprPtr& e) {
              require(bool(e), "null metadata expression");
              mt(e->type);
              std::visit(
                  [&](const auto& v) {
                    using T = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<T, metadata::SymbolRef>) {
                      require(bound.contains(v.symbol.value),
                              "unbound metadata symbol");
                      require(mt(e->type) == mt(m.symbols[v.symbol.value].type),
                              "metadata symbol type mismatch");
                    } else if constexpr (std::is_same_v<T, metadata::Integer>)
                      require(mt(e->type) == MT{MK::kInteger} &&
                                  metadata::validIntegerLiteral(v.spelling),
                              "invalid metadata integer");
                    else if constexpr (std::is_same_v<T, bool>)
                      require(mt(e->type) == MT{MK::kBoolean},
                              "metadata bool type mismatch");
                    else if constexpr (std::is_same_v<T, DType>)
                      require(mt(e->type) == MT{MK::kDType},
                              "metadata dtype type mismatch");
                    else if constexpr (std::is_same_v<T,
                                                      metadata::AttributeRef>) {
                      require(op.attributes == v.schema &&
                                  v.field < p.attribute_schemas[v.schema.value]
                                                .fields.size(),
                              "invalid metadata attribute");
                      const auto& f =
                          p.attribute_schemas[v.schema.value].fields[v.field];
                      require(!f.optional &&
                                  (f.type == "index" || f.type == "i64" ||
                                   f.type == "bool" || f.type == "dtype"),
                              "unsupported metadata attribute type");
                      require(
                          mt(e->type) == MT{f.type == "bool"    ? MK::kBoolean
                                            : f.type == "dtype" ? MK::kDType
                                                                : MK::kInteger,
                                            f.list_depth},
                          "metadata attribute type mismatch");
                    } else if constexpr (std::is_same_v<T, metadata::Call>) {
                      for (const auto& a : v.arguments) expr(a);
                      callArity(std::string(builtins::builtinName(v.builtin)),
                                v.arguments.size());
                      const auto* b = builtins::resolveBuiltin(
                          builtins::builtinName(v.builtin));
                      std::optional<MT> generic;
                      auto accept = [&](builtins::SignatureType signature,
                                        MT t) {
                        using S = builtins::SignatureType;
                        if (signature == S::kT || signature == S::kListT) {
                          if (signature == S::kListT) {
                            require(t.list_depth > 0,
                                    "metadata builtin requires list");
                            --t.list_depth;
                          }
                          require(!generic || *generic == t,
                                  "metadata generic type mismatch");
                          generic = t;
                        } else {
                          MT expected{signature == S::kDType ? MK::kDType
                                      : (signature == S::kBoolean ||
                                         signature == S::kBooleanList)
                                          ? MK::kBoolean
                                          : MK::kInteger,
                                      (signature == S::kIntegerList ||
                                       signature == S::kBooleanList)
                                          ? 1u
                                          : 0u};
                          require(t == expected,
                                  "metadata builtin type mismatch");
                        }
                      };
                      for (std::size_t i = 0; i < v.arguments.size(); ++i)
                        accept(
                            b->arguments[std::min(i, b->arguments.size() - 1)],
                            mt(v.arguments[i]->type));
                      accept(b->result, mt(e->type));
                    } else if constexpr (std::is_same_v<T, metadata::Unary>) {
                      expr(v.operand);
                      require(mt(v.operand->type) == mt(e->type) &&
                                  mt(e->type) == MT{v.op == UnaryOp::kLogicalNot
                                                        ? MK::kBoolean
                                                        : MK::kInteger},
                              "metadata unary type mismatch");
                    } else if constexpr (std::is_same_v<T, metadata::Binary>) {
                      expr(v.lhs);
                      expr(v.rhs);
                      auto t = mt(v.lhs->type);
                      require(t == mt(v.rhs->type),
                              "metadata operand types differ");
                      auto c = category(v.op);
                      require(mt(e->type) == (c == BinaryCategory::kArithmetic
                                                  ? t
                                                  : MT{MK::kBoolean}),
                              "metadata binary result type mismatch");
                      if (c == BinaryCategory::kArithmetic ||
                          c == BinaryCategory::kComparison)
                        require(t == MT{MK::kInteger},
                                "metadata operator requires integer");
                      if (c == BinaryCategory::kLogical)
                        require(t == MT{MK::kBoolean},
                                "metadata operator requires bool");
                    } else if constexpr (std::is_same_v<T, metadata::List>) {
                      auto element = mt(e->type);
                      require(element.list_depth > 0,
                              "metadata list type mismatch");
                      --element.list_depth;
                      for (const auto& a : v.elements) {
                        expr(a);
                        require(mt(a->type) == element,
                                "metadata list element type mismatch");
                      }
                    } else if constexpr (std::is_same_v<T, metadata::Index>) {
                      expr(v.value);
                      expr(v.index);
                      auto element = mt(v.value->type);
                      require(element.list_depth > 0 &&
                                  mt(v.index->type) == MT{MK::kInteger},
                              "invalid metadata index");
                      --element.list_depth;
                      require(mt(e->type) == element,
                              "metadata index result type mismatch");
                    } else if constexpr (std::is_same_v<
                                             T, metadata::Conditional>) {
                      expr(v.condition);
                      expr(v.then_value);
                      expr(v.else_value);
                      require(mt(v.condition->type) == MT{MK::kBoolean} &&
                                  mt(v.then_value->type) == mt(e->type) &&
                                  mt(v.else_value->type) == mt(e->type),
                              "metadata conditional type mismatch");
                    } else if constexpr (std::is_same_v<
                                             T, metadata::Comprehension>) {
                      expr(v.iterable);
                      require(v.variable.value < m.symbols.size(),
                              "unknown comprehension symbol");
                      auto element = mt(v.iterable->type);
                      require(element.list_depth > 0,
                              "metadata comprehension requires list");
                      --element.list_depth;
                      require(element == mt(m.symbols[v.variable.value].type),
                              "metadata comprehension variable type mismatch");
                      auto saved = bound;
                      bound.insert(v.variable.value);
                      expr(v.element);
                      auto result = mt(v.element->type);
                      ++result.list_depth;
                      require(result == mt(e->type),
                              "metadata comprehension result type mismatch");
                      bound = std::move(saved);
                    }
                  },
                  e->value);
            };
        for (const auto& s : m.statements)
          std::visit(
              [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, metadata::Let>) {
                  expr(v.value);
                  require(v.symbol.value < m.symbols.size(),
                          "unknown metadata local");
                  require(
                      mt(m.symbols[v.symbol.value].type) == mt(v.value->type),
                      "metadata local type mismatch");
                  require(bound.insert(v.symbol.value).second,
                          "metadata local assigned twice");
                } else {
                  expr(v.condition);
                  require(mt(v.condition->type) == MT{MK::kBoolean},
                          "metadata assertion requires bool");
                }
              },
              s.value);
        expr(m.result.value);
        require(
            mt(m.result.value->type) == (m.kind == metadata::ProgramKind::Shape
                                             ? MT{MK::kInteger, 1}
                                             : MT{MK::kDType}),
            "metadata return type mismatch");
      }
  }
  for (const auto& op : p.operations) {
    if (!op.semantic_definition) continue;
    namespace ix = index_semantics;
    const auto& d = *op.semantic_definition;
    require(d.operands.size() == op.operands.size(),
            "axis operand table mismatch");
    const auto fieldCount =
        op.attributes ? p.attribute_schemas[op.attributes->value].fields.size()
                      : 0;
    require(d.fields.size() == fieldCount, "axis field table mismatch");
    for (auto f : d.extent_fields)
      require(f < fieldCount, "unknown extent field");
    std::size_t available = 0;
    std::function<void(const ix::MetaPtr&)> meta = [&](const auto& e) {
      require(bool(e), "null axis metadata expression");
      using K = ix::Meta::Kind;
      if (e->kind == K::Operand)
        require(e->index < op.operands.size(), "unknown axis operand");
      if (e->kind == K::Attribute)
        require(e->index < fieldCount, "unknown axis field");
      if (e->kind == K::Group)
        require(e->index < available, "forward or cyclic axis group reference");
      if (e->kind == K::Integer)
        require(metadata::validIntegerLiteral(e->name), "invalid axis integer");
      if (e->kind == K::Boolean)
        require(e->name == "true" || e->name == "false",
                "invalid axis boolean");
      if (e->kind == K::Call)
        callArity(e->name, e->args.size());
      else if (e->kind != K::List)
        require(e->args.empty(), "axis metadata leaf has arguments");
      for (const auto& a : e->args) meta(a);
    };
    for (const auto& g : d.groups) {
      require(!g.bindings.empty() || g.domain, "axis group has no domain");
      for (const auto& b : g.bindings) {
        require(b.operand < op.operands.size(), "unknown bound operand");
        meta(b.positions);
        require(b.positions->type == ix::Type::Positions,
                "axis binding requires positions");
      }
      if (g.domain) meta(g.domain);
      ++available;
    }
    for (const auto& a : d.assertions) {
      meta(a);
      require(a->type == ix::Type::Boolean, "axis assertion requires bool");
    }
    meta(d.output);
    require(d.output->type == ix::Type::Axes, "axis output requires axes");
    std::function<void(const ix::IndexValuePtr&)> index = [&](const auto& e) {
      require(bool(e), "null coordinate expression");
      using K = ix::IndexValue::Kind;
      if (e->kind == K::Coordinates)
        require(e->group < available && e->args.empty(),
                "invalid coordinate group");
      else if (e->kind == K::Scale || e->kind == K::Offset) {
        require(e->args.size() == 1, "invalid coordinate arity");
        meta(e->parameter);
      } else
        require(e->args.size() >= 2, "invalid coordinate arity");
      for (const auto& a : e->args) index(a);
    };
    std::function<void(const ix::ValuePtr&)> value = [&](const auto& e) {
      require(bool(e), "null scalar expression");
      using K = ix::Value::Kind;
      if (e->kind == K::Operand || e->kind == K::Read)
        require(e->index < op.operands.size() && e->args.empty(),
                "invalid scalar operand");
      if (e->kind == K::Read) index(e->access);
      if (e->kind == K::Reduce)
        require(e->index < available && e->args.size() == 1,
                "invalid reduction");
      if (e->kind == K::Const)
        require(e->args.empty(), "constant has operands");
      if (e->kind == K::Add || e->kind == K::Product)
        require(e->args.size() >= 2, "invalid scalar arity");
      for (const auto& a : e->args) value(a);
    };
    value(d.value);
  }
  for (const auto& r : p.rules) {
    ids(r.captures);
    ids(r.dimensions);
    ids(r.descriptors);
    ids(r.locals);
    ids(r.dtypes);
    for (const auto& c : r.captures) type(c.type);
    for (const auto& d : r.descriptors)
      require(type(d.type).schema.has_value(), "descriptor without schema");
    for (const auto& d : r.dimensions) type(d.type);
    for (const auto& d : r.locals) type(d.type);
    for (const auto& d : r.dtypes) type(d.type);
    std::set<std::size_t> captured, descriptors, locals, dimensions, dtypes;
    std::function<void(const PatternPtr&)> pattern = [&](const PatternPtr& e) {
      require(bool(e), "null pattern");
      type(e->type);
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, CapturePattern> ||
                          std::is_same_v<T, BindPattern>) {
              require(v.capture.value < r.captures.size(), "unknown capture");
              captured.insert(v.capture.value);
              if constexpr (std::is_same_v<T, BindPattern>)
                pattern(v.expression);
            } else if constexpr (std::is_same_v<T, MatchOperation>) {
              arity(v.operation, v.operands.size());
              if (v.descriptor) {
                require(v.descriptor->value < r.descriptors.size(),
                        "unknown descriptor");
                require(type(r.descriptors[v.descriptor->value].type).schema ==
                            p.operations[v.operation.value].attributes,
                        "pattern attribute schema mismatch");
                descriptors.insert(v.descriptor->value);
              }
              for (const auto& a : v.operands) pattern(a);
            }
          },
          e->value);
    };
    pattern(r.lhs);
    for (const auto& c : r.constraints) {
      require(captured.contains(c.capture.value),
              "unbound constrained capture");
      for (const auto& e : c.shape)
        if (e.symbol) {
          require(e.symbol->value < r.dimensions.size(), "unknown dimension");
          dimensions.insert(e.symbol->value);
        }
      if (c.dtype)
        if (const auto* id = std::get_if<DTypeVariableId>(&*c.dtype)) {
          require(id->value < r.dtypes.size(), "unknown dtype variable");
          dtypes.insert(id->value);
        }
    }
    std::set<std::pair<std::size_t, std::size_t>> groups;
    for (const auto& s : program.axes(r.id.value)) {
      require(bool(s) && s->kind == index_semantics::Symbol::Kind::AxisGroup,
              "invalid axis definition");
      groups.insert({s->owner, s->index});
    }
    std::function<void(const index_semantics::SymbolPtr&)> sym =
        [&](const auto& s) {
          require(bool(s), "null symbolic expression");
          using K = index_semantics::Symbol::Kind;
          if (s->kind == K::Shape || s->kind == K::DType)
            require(captured.contains(s->owner), "unknown symbolic capture");
          if (s->kind == K::Field) {
            require(descriptors.contains(s->owner),
                    "unknown symbolic descriptor");
            require(s->index <
                        p.attribute_schemas[type(r.descriptors[s->owner].type)
                                                .schema->value]
                            .fields.size(),
                    "unknown symbolic field");
          }
          if (s->kind == K::AxisGroup)
            require(
                groups.contains({s->owner, s->index}) && s->args.size() == 1,
                "undefined axis group");
          require(s->kind != K::Hole, "unresolved symbolic hole");
          if (s->kind == K::Call)
            callArity(s->name, s->args.size());
          else if (s->kind != K::AxisGroup && s->kind != K::List)
            require(s->args.empty(), "symbolic leaf has arguments");
          if (s->kind == K::Integer)
            require(metadata::validIntegerLiteral(s->name),
                    "invalid symbolic integer");
          if (s->kind == K::Boolean)
            require(s->name == "true" || s->name == "false",
                    "invalid symbolic bool");
          for (const auto& a : s->args) sym(a);
        };
    for (const auto& s : program.axes(r.id.value)) sym(s);
    std::function<void(const TypedExprPtr&)> expr = [&](const TypedExprPtr& e) {
      require(bool(e), "null expression");
      const auto& t = type(e->type);
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, CaptureRef>) {
              require(captured.contains(v.capture.value), "unbound capture");
              require(t == type(r.captures[v.capture.value].type),
                      "capture type mismatch");
            } else if constexpr (std::is_same_v<T, DimensionRef>) {
              require(dimensions.contains(v.dimension.value),
                      "unbound dimension");
              require(t == type(r.dimensions[v.dimension.value].type),
                      "dimension type mismatch");
            } else if constexpr (std::is_same_v<T, DTypeVariableRef>) {
              require(dtypes.contains(v.variable.value),
                      "unbound dtype variable");
              require(t == Type{TypeKind::kDType}, "dtype type mismatch");
            } else if constexpr (std::is_same_v<T, DescriptorRef>) {
              require(descriptors.contains(v.descriptor.value),
                      "unbound descriptor");
              require(t == type(r.descriptors[v.descriptor.value].type),
                      "descriptor type mismatch");
            } else if constexpr (std::is_same_v<T, DescriptorFieldRef>) {
              require(descriptors.contains(v.descriptor.value),
                      "unbound descriptor");
              require(
                  type(r.descriptors[v.descriptor.value].type).schema ==
                          v.schema &&
                      v.field <
                          p.attribute_schemas.at(v.schema.value).fields.size(),
                  "invalid field reference");
              require(t == fieldType(p.attribute_schemas.at(v.schema.value)
                                         .fields.at(v.field)),
                      "field type mismatch");
            } else if constexpr (std::is_same_v<T, LocalRef>) {
              require(locals.contains(v.local.value), "unbound local");
              require(t == type(r.locals.at(v.local.value).type),
                      "local reference type mismatch");
            } else if constexpr (std::is_same_v<T, OutputAxisGroupRef>)
              require(false, "unresolved output-axis reference");
            else if constexpr (std::is_same_v<T, SymbolicExpr>)
              sym(v.value);
            else if constexpr (std::is_same_v<T, LocalLet>) {
              expr(v.value);
              require(v.local.value < r.locals.size(), "unknown local");
              require(type(r.locals[v.local.value].type) == type(v.value->type),
                      "local type mismatch");
              locals.insert(v.local.value);
            } else if constexpr (std::is_same_v<T, ComprehensionExpr>) {
              expr(v.iterable);
              require(v.variable.value < r.locals.size(),
                      "unknown comprehension variable");
              require(bool(v.element), "null comprehension element");
              require(elementType(type(v.iterable->type)) ==
                              type(r.locals[v.variable.value].type) &&
                          elementType(t) == type(v.element->type),
                      "comprehension type mismatch");
              auto saved = locals;
              locals.insert(v.variable.value);
              expr(v.element);
              locals = std::move(saved);
            } else {
              forEachExpressionChild(*e, [&](const TypedExpr& a) {
                expr(std::make_shared<TypedExpr>(a));
              });
              if constexpr (std::is_same_v<T, bool>)
                require(t == Type{TypeKind::kBool}, "boolean type mismatch");
              else if constexpr (std::is_same_v<T, NumericConstant>)
                require(validScalarLiteral(v.spelling, t.kind),
                        "invalid numeric constant");
              else if constexpr (std::is_same_v<T, BinaryExpr>) {
                auto a = type(v.lhs->type), b = type(v.rhs->type);
                require(a == b, "binary operand type mismatch");
                if (category(v.op) == BinaryCategory::kLogical)
                  require(a == Type{TypeKind::kBool} && t == a,
                          "logical expression must be bool");
                if (category(v.op) == BinaryCategory::kEquality ||
                    category(v.op) == BinaryCategory::kComparison)
                  require(t == Type{TypeKind::kBool},
                          "comparison must return bool");
                if (category(v.op) == BinaryCategory::kArithmetic ||
                    category(v.op) == BinaryCategory::kComparison)
                  require(!a.optional && !a.list_depth && isNumeric(a.kind),
                          "numeric operator type mismatch");
                if (category(v.op) == BinaryCategory::kArithmetic)
                  require(t == a, "arithmetic result type mismatch");
                if (v.op == BinaryOp::kRemainder)
                  require(isInteger(a.kind), "remainder requires integer");
              } else if constexpr (std::is_same_v<T, ConditionalExpr>)
                require(type(v.condition->type) == Type{TypeKind::kBool} &&
                            type(v.then_value->type) == t &&
                            type(v.else_value->type) == t,
                        "invalid conditional types");
              else if constexpr (std::is_same_v<T, DescriptorConstruct>) {
                require(
                    t.schema == v.schema &&
                        v.schema.value < p.attribute_schemas.size() &&
                        v.fields.size() ==
                            p.attribute_schemas[v.schema.value].fields.size(),
                    "invalid descriptor constructor");
                for (std::size_t f = 0; f < v.fields.size(); ++f)
                  require(
                      type(v.fields[f]->type) ==
                          fieldType(
                              p.attribute_schemas[v.schema.value].fields[f]),
                      "constructor field type mismatch");
              } else if constexpr (std::is_same_v<T, BuiltinCall>) {
                const auto* b =
                    builtins::resolveBuiltin(builtins::builtinName(v.builtin));
                require(b && (b->variadic
                                  ? v.arguments.size() >= b->arguments.size()
                                  : v.arguments.size() == b->arguments.size()),
                        "builtin arity mismatch");
                std::optional<Type> generic, integer;
                auto bindType = [&](std::optional<Type>& slot, Type value) {
                  if (slot)
                    require(*slot == value, "builtin argument type mismatch");
                  else
                    slot = value;
                };
                auto accept = [&](builtins::SignatureType signature,
                                  Type value) {
                  using S = builtins::SignatureType;
                  switch (signature) {
                    case S::kT:
                      bindType(generic, value);
                      break;
                    case S::kListT: {
                      auto e = elementType(value);
                      require(e.has_value(), "builtin expects list");
                      bindType(generic, *e);
                      break;
                    }
                    case S::kInteger:
                      if (b->rule_integer_pair) {
                        require(!value.optional && !value.list_depth &&
                                    isInteger(value.kind),
                                "builtin expects integer");
                        bindType(integer, value);
                      } else
                        require(value == Type{TypeKind::kIndex},
                                "builtin expects index");
                      break;
                    case S::kIntegerList:
                      require(value == Type{TypeKind::kIndexList},
                              "builtin expects index list");
                      break;
                    case S::kBooleanList:
                      require(value == listType(Type{TypeKind::kBool}),
                              "builtin expects bool list");
                      break;
                    case S::kBoolean:
                      require(value == Type{TypeKind::kBool},
                              "builtin expects bool");
                      break;
                    case S::kDType:
                      require(value == Type{TypeKind::kDType},
                              "builtin expects dtype");
                      break;
                  }
                };
                for (std::size_t i = 0; i < v.arguments.size(); ++i)
                  accept(b->arguments[std::min(i, b->arguments.size() - 1)],
                         type(v.arguments[i]->type));
                accept(b->result, t);
              } else if constexpr (std::is_same_v<T, ListExpr>) {
                auto element = elementType(t);
                require(element.has_value(), "list expression type mismatch");
                for (const auto& a : v.elements)
                  require(type(a->type) == *element,
                          "list element type mismatch");
              } else if constexpr (std::is_same_v<T, IndexExpr>)
                require(elementType(type(v.value->type)) == t &&
                            type(v.index->type) == Type{TypeKind::kIndex},
                        "index expression type mismatch");
              else if constexpr (std::is_same_v<T, UnaryExpr>) {
                require(type(v.operand->type) == t, "unary type mismatch");
                if (v.op == UnaryOp::kLogicalNot)
                  require(t == Type{TypeKind::kBool}, "not requires bool");
                else
                  require(!t.optional && !t.list_depth && isNumeric(t.kind),
                          "unary requires number");
              } else if constexpr (std::is_same_v<T, DType>)
                require(t == Type{TypeKind::kDType}, "dtype type mismatch");
              else if constexpr (std::is_same_v<T, StringConstant>)
                require(t == Type{TypeKind::kString}, "string type mismatch");
              else if constexpr (std::is_same_v<T, EnumConstant>)
                require(t == Type{TypeKind::kPrecision} &&
                            (v.value == "default" || v.value == "high" ||
                             v.value == "highest"),
                        "invalid enum constant");
              else if constexpr (std::is_same_v<T, NoneConstant>)
                require(t.optional, "none requires optional type");
              else if constexpr (std::is_same_v<T, OptionalValue>) {
                auto inner = t;
                inner.optional = false;
                require(t.optional && type(v.value->type) == inner,
                        "optional type mismatch");
              }
            }
          },
          e->value);
    };
    for (const auto& e : r.conditions) {
      expr(e);
      if (!std::holds_alternative<LocalLet>(e->value))
        require(type(e->type) == Type{TypeKind::kBool},
                "where result must be bool");
    }
    locals.clear();
    for (const auto& d : r.derivations) {
      expr(d.value);
      if (d.target) {
        require(d.target->value < r.descriptors.size(),
                "unknown derived descriptor");
        require(
            type(d.value->type) == type(r.descriptors[d.target->value].type),
            "derived descriptor type mismatch");
        require(!descriptors.contains(d.target->value),
                "descriptor assigned twice");
        descriptors.insert(d.target->value);
      }
    }
    std::function<void(const BuildExprPtr&)> build = [&](const auto& e) {
      require(bool(e), "null replacement");
      type(e->type);
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, CaptureRef>)
              require(captured.contains(v.capture.value),
                      "unbound RHS capture");
            else if constexpr (std::is_same_v<T, BuildOperation>) {
              arity(v.operation, v.operands.size());
              if (v.descriptor) {
                require(descriptors.contains(v.descriptor->value),
                        "unbound RHS descriptor");
                require(type(r.descriptors[v.descriptor->value].type).schema ==
                            p.operations[v.operation.value].attributes,
                        "RHS schema mismatch");
              }
              for (const auto& a : v.operands) build(a);
            }
          },
          e->value);
    };
    build(r.rhs);
  }
  for (const auto& g : p.graphs) validateGraph(p, g);
}
void validateGraph(const core::Program& p, const core::ConcreteGraph& g) {
  using namespace core;
  require(g.root < g.nodes.size(), "invalid graph root");
  std::function<void(const AttributeValue&, AttributeField)> attr =
      [&](const AttributeValue& a, AttributeField f) {
        using K = AttributeValue::Kind;
        if (f.optional && a.kind == K::kNone) return;
        if (f.list_depth) {
          require(a.kind == K::kList, "expected list attribute");
          --f.list_depth;
          f.optional = false;
          for (const auto& e : a.elements) attr(e, f);
          return;
        }
        require(a.elements.empty(), "scalar attribute has elements");
        if (f.type == "index" || f.type == "i64")
          require(a.kind == K::kInteger &&
                      validScalarLiteral(a.text, f.type == "index"
                                                     ? TypeKind::kIndex
                                                     : TypeKind::kI64),
                  "invalid integer attribute");
        else if (f.type == "bool")
          require(a.kind == K::kBool && (a.text == "true" || a.text == "false"),
                  "invalid bool attribute");
        else if (f.type == "string")
          require(a.kind == K::kString, "invalid string attribute");
        else if (f.type == "dtype")
          require(a.kind == K::kEnum && resolveDType(a.text).has_value(),
                  "invalid dtype attribute");
        else if (f.type == "precision")
          require(
              a.kind == K::kEnum && (a.text == "default" || a.text == "high" ||
                                     a.text == "highest"),
              "invalid precision attribute");
        else
          require(false, "unsupported concrete attribute type");
      };
  std::set<std::string> inputs, names;
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const auto& n = g.nodes[i];
    for (auto a : n.operands)
      require(a < i, "graph must be topologically ordered");
    if (n.kind == ConcreteNode::Kind::kOperation) {
      require(n.operation && n.operation->value < p.operations.size(),
              "invalid graph operation");
      const auto& op = p.operations[n.operation->value];
      bool variadic = !op.operands.empty() && op.operands.back().variadic;
      require(variadic ? n.operands.size() >= op.operands.size() - 1
                       : n.operands.size() == op.operands.size(),
              "graph operation arity mismatch");
      require(
          n.attributes.size() ==
              (op.attributes
                   ? p.attribute_schemas.at(op.attributes->value).fields.size()
                   : 0),
          "graph attribute count mismatch");
      for (std::size_t f = 0; f < n.attributes.size(); ++f)
        attr(n.attributes[f],
             p.attribute_schemas[op.attributes->value].fields[f]);
    } else {
      require(!n.operation && n.operands.empty() && n.attributes.empty(),
              "input or literal has operation fields");
      if (n.kind == ConcreteNode::Kind::kInput)
        require(!n.text.empty() && inputs.insert(n.text).second,
                "duplicate or empty input name");
      else if (n.dtype)
        require(validGraphLiteral(n.text, *n.dtype),
                "invalid typed graph literal");
      else
        require(validScalarLiteral(n.text, TypeKind::kI64) ||
                    validScalarLiteral(n.text, TypeKind::kIndex) ||
                    validScalarLiteral(n.text, TypeKind::kF64),
                "invalid graph literal");
    }
  }
  for (const auto& [name, id] : g.bindings)
    require(id < g.nodes.size() && !name.empty() && names.insert(name).second,
            "invalid graph binding");
}
}  // namespace tepl::program
