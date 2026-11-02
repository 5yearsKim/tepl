#include <functional>
#include <map>
#include <set>
#include <stdexcept>

#include "src/executable/program.h"
#include "src/inference/evaluate.h"
#include "src/inference/extensions.h"
#include "src/shared/builtins/axes.h"
#include "src/shared/builtins/intrinsics/catalog.h"
#include "src/shared/catalog/axis_conditions.h"
#include "src/shared/catalog/core_guard.h"
#include "src/shared/metadata/ordered.h"
#include "src/shared/types/literal.h"

namespace tepl::executable {
namespace {
void require(bool b, const std::string& s) {
  if (!b) throw std::invalid_argument("compiled program: " + s);
}

// Bind completed instructions to the actual rule and operation axis contracts.
// Runtime validation cannot trust a serialized schedule's claimed roles.
void validateInference(const ProgramData& p, const Rule& r) {
  using namespace core;
  if (!r.inference) return;
  const auto& program = *r.inference;
  std::map<std::size_t, PatternPtr> bound;
  auto instruction = [&](const std::vector<inference::Instruction>& code,
                         std::size_t& next, std::optional<CaptureId> capture,
                         std::optional<OpId> operation,
                         std::optional<DescriptorId> descriptor,
                         std::vector<std::size_t> inputs) {
    require(next < code.size(), "truncated inference schedule");
    const auto id = next++;
    const auto& n = code[id];
    require(n.inputs == inputs, "inference operand schedule mismatch");
    if (capture) {
      require(n.kind == inference::Kind::Input && n.capture == capture->value,
              "inference capture mismatch");
      return id;
    }
    require(operation && operation->value < p.operations.size() &&
                n.operation == operation->value &&
                n.descriptor ==
                    (descriptor ? std::optional<std::size_t>{descriptor->value}
                                : std::nullopt),
            "inference operation/descriptor mismatch");
    const auto& op = p.operations[operation->value];
    require(bool(op.axis_contract), "inference operation lacks axis contract");
    auto c = inference::hasExtensions(program) ? core::axis::recognizeCoreShape(*op.axis_contract) : core::axis::recognizeOrdered(*op.axis_contract);
    require(c.has_value(), "inference operation is outside core templates: " + op.name);
    if (c->kind == core::axis::OrderedContract::Kind::Negation) {
      require(n.kind == inference::Kind::Negation && n.fields.empty() && !descriptor && !op.attributes, "invalid core negation");
      return id;
    }
    if (c->kind == core::axis::OrderedContract::Kind::Addition) {
      require((n.kind == inference::Kind::Addition ||
               (inference::hasExtensions(program) && n.kind == inference::Kind::Multiplication)) && n.fields.empty() &&
                  !descriptor && !op.attributes,
              "inference addition must be attribute-free");
      return id;
    }
    auto kind = c->kind == core::axis::OrderedContract::Kind::UnitRemoval
                    ? inference::Kind::UnitRemoval
                : c->kind == core::axis::OrderedContract::Kind::Reverse
                    ? inference::Kind::Reverse
                : c->kind == core::axis::OrderedContract::Kind::Broadcast
                    ? inference::Kind::Broadcast
                : c->kind == core::axis::OrderedContract::Kind::Permutation
                    ? inference::Kind::Permutation
                : c->kind == core::axis::OrderedContract::Kind::Summation
                    ? inference::Kind::Summation
                    : inference::Kind::Contraction;
    auto fields =
        (kind == inference::Kind::Reverse || kind == inference::Kind::UnitRemoval) ? std::vector<std::size_t>{0}
        : kind == inference::Kind::Broadcast ? std::vector<std::size_t>{0, 1}
        : kind == inference::Kind::Permutation
            ? std::vector<std::size_t>{c->permutation}
        : kind == inference::Kind::Summation
            ? std::vector<std::size_t>{c->dimensions}
            : std::vector<std::size_t>{c->left_batch, c->right_batch,
                                       c->left_contract, c->right_contract};
    require(n.kind == kind && n.fields == fields,
            "inference field adapter mismatch");
    require(
        op.attributes && descriptor && descriptor->value < r.descriptors.size(),
        "invalid inference descriptor");
    for (const auto& field :
         p.attribute_schemas.at(op.attributes->value).fields)
      require((field.type == "u64" || field.type == "i64") &&
                  field.list_depth == 1 && !field.optional,
              "inference requires nonoptional position-list fields");
    return id;
  };
  std::function<std::size_t(const PatternPtr&,
                            const std::vector<inference::Instruction>&,
                            std::size_t&)>
      pattern;
  pattern = [&](const auto& e, const auto& code, auto& next) -> std::size_t {
    if (const auto* b = std::get_if<core::BindPattern>(&e->value)) {
      bound[b->capture.value] = b->expression;
      return pattern(b->expression, code, next);
    }
    if (const auto* c = std::get_if<core::CapturePattern>(&e->value))
      return instruction(code, next, c->capture, {}, {}, {});
    const auto* op = std::get_if<core::MatchOperation>(&e->value);
    require(op != nullptr, "inference does not support literal patterns");
    std::vector<std::size_t> inputs;
    for (const auto& child : op->operands)
      inputs.push_back(pattern(child, code, next));
    return instruction(code, next, {}, op->operation, op->descriptor,
                       std::move(inputs));
  };
  std::size_t next = 0;
  pattern(r.lhs, program.source, next);
  require(next == program.source.size(), "extra source inference instructions");
  next = 0;
  std::function<std::size_t(const core::BuildExprPtr&)> build =
      [&](const auto& e) -> std::size_t {
    if (const auto* c = std::get_if<core::CaptureRef>(&e->value)) {
      if (auto it = bound.find(c->capture.value); it != bound.end())
        return pattern(it->second, program.replacement, next);
      return instruction(program.replacement, next, c->capture, {}, {}, {});
    }
    const auto* op = std::get_if<core::BuildOperation>(&e->value);
    require(op != nullptr, "inference does not support literal replacements");
    std::vector<std::size_t> inputs;
    for (const auto& child : op->operands) inputs.push_back(build(child));
    return instruction(program.replacement, next, {}, op->operation,
                       op->descriptor, std::move(inputs));
  };
  build(r.rhs);
  require(next == program.replacement.size(),
          "extra RHS inference instructions");
}
// Loaded source axes expressions and lowered symbolic instructions use their
// respective catalogs. Internal operations never grant source permissions.
void callArity(const std::string& name, std::size_t count,
               bool axes_source = false) {
  bool accepted = false;
  if (axes_source) {
    if (const auto* entry = core::builtins::resolveAxisBuiltin(name))
      accepted = entry->acceptsArity(count);
  } else if (const auto* entry = core::intrinsics::resolve(name)) {
    accepted = entry->acceptsArity(count);
  } else if (const auto* entry = core::builtins::resolveBuiltin(name)) {
    accepted = entry->acceptsArity(count);
  } else if (name.starts_with("dtype_")) {
    accepted = count == 0 && core::resolveDType(name.substr(6)).has_value();
  }
  if (resolveBinaryOperator(name))
    accepted = count == 2;
  else if (resolveUnaryOperator(name))
    accepted = count == 1;
  require(accepted, "unsupported metadata instruction or arity: " + name);
}
template <class V>
void ids(const V& v) {
  for (std::size_t i = 0; i < v.size(); ++i)
    require(v[i].id.value == i, "table IDs must be consecutive");
}
}  // namespace
void validate(const ExecutableProgram& program) {
  using namespace core;
  const auto& p = program.ir();
  for (const auto& r : p.rules)
    if (r.core)
      require(program.policy() == "algebraic",
              "core DAG requires algebraic semantics");
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
                                  (f.type == "u64" || f.type == "i64" ||
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
                      const auto* b = &builtins::signature(v.builtin);
                      require(b->acceptsArity(v.arguments.size()),
                              "metadata builtin arity mismatch");
                      require(
                          b->availableIn(m.kind == metadata::ProgramKind::Shape
                                             ? builtins::Context::kShape
                                             : builtins::Context::kDType),
                          "metadata builtin unavailable in section");
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
    if (!op.axis_contract) continue;
    namespace ix = axis;
    const auto& d = *op.axis_contract;
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
        callArity(e->name, e->args.size(), true);
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
    for (auto operand : d.implicit_operands)
      require(operand < op.operands.size(), "unknown implicit operand");
  }
  for (const auto& r : p.rules) {
    require(!r.inference,
            "runtime applicability inference is unsupported; compile a rule "
            "with verified source conditions");
    if (r.core) {
      require(!r.inference,
              "core DAG cannot coexist with applicability inference");
      require(inference::validatePortable(*r.core),
              "core DAG differs from structural emitter");
      auto proxy = r;
      proxy.inference = r.core->program;
      validateInference(p, proxy);
    }
    ids(r.captures);
    ids(r.dimensions);
    ids(r.descriptors);
    ids(r.locals);
    ids(r.dtypes);
    for (const auto& c : r.captures) type(c.type);
    for (const auto& d : r.descriptors) {
      require(type(d.type).schema.has_value(), "descriptor without schema");
    }
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
      require(bool(s) && s->kind == completed::Symbol::Kind::AxisGroup,
              "invalid axis definition");
      groups.insert({s->owner, s->index});
    }
    std::function<void(const completed::SymbolPtr&)> sym = [&](const auto& s) {
      require(bool(s), "null symbolic expression");
      using K = completed::Symbol::Kind;
      if (s->kind == K::Shape || s->kind == K::DType)
        require(captured.contains(s->owner), "unknown symbolic capture");
      if (s->kind == K::Field) {
        require(descriptors.contains(s->owner), "unknown symbolic descriptor");
        require(
            s->index < p.attribute_schemas[type(r.descriptors[s->owner].type)
                                               .schema->value]
                           .fields.size(),
            "unknown symbolic field");
      }
      if (s->kind == K::AxisGroup)
        require(groups.contains({s->owner, s->index}) && s->args.size() == 1,
                "undefined axis group");
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
    std::function<void(const completed::TypedExprPtr&)> expr =
        [&](const completed::TypedExprPtr& e) {
          require(bool(e), "null expression");
          const auto& t = type(e->type);
          std::visit(
              [&](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, CaptureRef>) {
                  require(captured.contains(v.capture.value),
                          "unbound capture");
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
                  require(type(r.descriptors[v.descriptor.value].type).schema ==
                                  v.schema &&
                              v.field < p.attribute_schemas.at(v.schema.value)
                                            .fields.size(),
                          "invalid field reference");
                  require(t == fieldType(p.attribute_schemas.at(v.schema.value)
                                             .fields.at(v.field)),
                          "field type mismatch");
                } else if constexpr (std::is_same_v<T, LocalRef>) {
                  require(locals.contains(v.local.value), "unbound local");
                  require(t == type(r.locals.at(v.local.value).type),
                          "local reference type mismatch");
                } else if constexpr (std::is_same_v<T, completed::SymbolicExpr>)
                  sym(v.value);
                else if constexpr (std::is_same_v<T, completed::LocalLet>) {
                  expr(v.value);
                  require(v.local.value < r.locals.size(), "unknown local");
                  require(
                      type(r.locals[v.local.value].type) == type(v.value->type),
                      "local type mismatch");
                  locals.insert(v.local.value);
                } else if constexpr (std::is_same_v<
                                         T, completed::ComprehensionExpr>) {
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
                  completed::forEachExpressionChild(
                      *e, [&](const completed::TypedExpr& a) {
                        expr(std::make_shared<completed::TypedExpr>(a));
                      });
                  if constexpr (std::is_same_v<T, bool>)
                    require(t == Type{TypeKind::kBool},
                            "boolean type mismatch");
                  else if constexpr (std::is_same_v<T, NumericConstant>)
                    require(validScalarLiteral(v.spelling, t.kind),
                            "invalid numeric constant");
                  else if constexpr (std::is_same_v<T, completed::BinaryExpr>) {
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
                  } else if constexpr (std::is_same_v<
                                           T, completed::ConditionalExpr>)
                    require(type(v.condition->type) == Type{TypeKind::kBool} &&
                                type(v.then_value->type) == t &&
                                type(v.else_value->type) == t,
                            "invalid conditional types");
                  else if constexpr (std::is_same_v<
                                         T, completed::DescriptorConstruct>) {
                    require(t.schema == v.schema &&
                                v.schema.value < p.attribute_schemas.size() &&
                                v.fields.size() ==
                                    p.attribute_schemas[v.schema.value]
                                        .fields.size(),
                            "invalid descriptor constructor");
                    for (std::size_t f = 0; f < v.fields.size(); ++f)
                      require(type(v.fields[f]->type) ==
                                  fieldType(p.attribute_schemas[v.schema.value]
                                                .fields[f]),
                              "constructor field type mismatch");
                  } else if constexpr (std::is_same_v<T,
                                                      completed::BuiltinCall>) {
                    const auto* b = &builtins::signature(v.builtin);
                    require(b->acceptsArity(v.arguments.size()),
                            "builtin arity mismatch");
                    std::optional<Type> generic, integer;
                    auto bindType = [&](std::optional<Type>& slot, Type value) {
                      if (slot)
                        require(*slot == value,
                                "builtin argument type mismatch");
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
                        case S::kAnyInteger:
                          require(!value.optional && !value.list_depth &&
                                      isInteger(value.kind),
                                  "builtin expects integer");
                          break;
                        case S::kSignedInteger:
                          require(value == Type{TypeKind::kI64},
                                  "builtin expects i64");
                          break;
                        case S::kInteger:
                          if (b->rule_integer_pair) {
                            require(!value.optional && !value.list_depth &&
                                        isInteger(value.kind),
                                    "builtin expects integer");
                            bindType(integer, value);
                          } else
                            require(value == Type{TypeKind::kU64},
                                    "builtin expects u64");
                          break;
                        case S::kAxisSet:
                          require(value == logicalAxisSetType(),
                                  "builtin expects logical axis set");
                          break;
                        case S::kAxisList:
                          require(value == listType(logicalAxisType()),
                                  "builtin expects logical axis list");
                          break;
                        case S::kIntegerList:
                          require(value == Type{TypeKind::kU64List},
                                  "builtin expects u64 list");
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
                  } else if constexpr (std::is_same_v<T, completed::ListExpr>) {
                    auto element = elementType(t);
                    require(element.has_value(),
                            "list expression type mismatch");
                    for (const auto& a : v.elements)
                      require(type(a->type) == *element,
                              "list element type mismatch");
                  } else if constexpr (std::is_same_v<T, completed::IndexExpr>)
                    require(elementType(type(v.value->type)) == t &&
                                type(v.index->type) == Type{TypeKind::kU64},
                            "index expression type mismatch");
                  else if constexpr (std::is_same_v<T, completed::UnaryExpr>) {
                    require(type(v.operand->type) == t, "unary type mismatch");
                    if (v.op == UnaryOp::kLogicalNot)
                      require(t == Type{TypeKind::kBool}, "not requires bool");
                    else
                      require(!t.optional && !t.list_depth && isNumeric(t.kind),
                              "unary requires number");
                  } else if constexpr (std::is_same_v<T, DType>)
                    require(t == Type{TypeKind::kDType}, "dtype type mismatch");
                  else if constexpr (std::is_same_v<T, StringConstant>)
                    require(t == Type{TypeKind::kString},
                            "string type mismatch");
                  else if constexpr (std::is_same_v<T, EnumConstant>)
                    require(t == Type{TypeKind::kPrecision} &&
                                (v.value == "default" || v.value == "high" ||
                                 v.value == "highest"),
                            "invalid enum constant");
                  else if constexpr (std::is_same_v<T, NoneConstant>)
                    require(t.optional, "none requires optional type");
                  else if constexpr (std::is_same_v<T,
                                                    completed::OptionalValue>) {
                    auto inner = t;
                    inner.optional = false;
                    require(t.optional && type(v.value->type) == inner,
                            "optional type mismatch");
                  }
                }
              },
              e->value);
        };
    auto where_schedule = completed::scheduleWhere(r);
    for (const auto& e : where_schedule.source) {
      expr(e);
      if (!std::holds_alternative<completed::LocalLet>(e->value))
        require(type(e->type) == Type{TypeKind::kBool},
                "where result must be bool");
    }
    auto where_locals = locals;
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
    if (r.core)
      for (const auto& n : r.core->program.replacement)
        if (n.descriptor) {
          require(*n.descriptor < r.descriptors.size() &&
                      r.descriptors[*n.descriptor].kind ==
                          completed::Descriptor::Kind::kDerived,
                  "invalid core descriptor");
          require(descriptors.insert(*n.descriptor).second,
                  "core descriptor already assigned");
        }
    locals = std::move(where_locals);
    for (const auto& e : where_schedule.replacement) {
      expr(e);
      if (!std::holds_alternative<completed::LocalLet>(e->value))
        require(type(e->type) == Type{TypeKind::kBool},
                "where result must be bool");
    }
    if (r.inference && r.inference->mode == inference::Mode::Synthesize)
      for (const auto& n : r.inference->replacement)
        if (n.descriptor) {
          require(*n.descriptor < r.descriptors.size() &&
                      r.descriptors[*n.descriptor].kind ==
                          completed::Descriptor::Kind::kDerived,
                  "inferred RHS descriptor must be derived");
          require(descriptors.insert(*n.descriptor).second,
                  "inferred descriptor already assigned");
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
    validateInference(p, r);
    if (r.core)
      require(core::completed::coreTypes(
                  r.core->program,
                  [&](std::size_t id) -> const core::metadata::Program* {
                    return id < p.operations.size() && p.operations[id].dtype
                               ? &*p.operations[id].dtype
                               : nullptr;
                  }),
              "core dtype contracts are not certified");
    if (r.core)
      require(core::completed::coreGuard(r),
              "written guard does not certify core feasibility");
    completed::validateAxisConditions(r);
  }
  for (const auto& g : p.graphs) validateGraph(p, g);
}
void validateGraph(const ProgramData& p, const core::ConcreteGraph& g) {
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
        if (f.type == "u64" || f.type == "i64")
          require(
              a.kind == K::kInteger &&
                  validScalarLiteral(a.text, f.type == "u64" ? TypeKind::kU64
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
                    validScalarLiteral(n.text, TypeKind::kU64) ||
                    validScalarLiteral(n.text, TypeKind::kF64),
                "invalid graph literal");
    }
  }
  for (const auto& [name, id] : g.bindings)
    require(id < g.nodes.size() && !name.empty() && names.insert(name).second,
            "invalid graph binding");
}
}  // namespace tepl::executable
