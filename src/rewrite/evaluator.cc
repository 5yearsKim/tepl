#include "src/rewrite/evaluator.h"

#include <functional>
#include <limits>

#include "src/rewrite/builtins/dispatch.h"
#include "src/rewrite/intrinsics.h"
#include "src/rewrite/operators.h"
#include "src/shared/builtins/axes.h"
#include "src/shared/catalog/axis_conditions.h"

namespace tepl::rewrite {
namespace {
void ensure(bool b, const char* s) {
  if (!b) throw EvaluationError(s);
}
Value shape(const core::ConcreteNode& n) {
  ensure(n.shape.has_value(), "shape unavailable");
  List xs;
  for (auto d : *n.shape) xs.emplace_back(Integer(d));
  return xs;
}
Value dtype(const core::ConcreteNode& n) {
  ensure(n.dtype.has_value(), "dtype unavailable");
  return std::string(core::dtypeName(*n.dtype));
}
template <class F>
Value binary(BinaryOp op, F eval) {
  auto a = eval(false);
  if (op == BinaryOp::kLogicalAnd) return boolean(a) && boolean(eval(true));
  if (op == BinaryOp::kLogicalOr) return boolean(a) || boolean(eval(true));
  return {};
}
class MetadataEvaluator {
 public:
  MetadataEvaluator(const core::metadata::Program& m,
                    const core::ConcreteNode& n, const core::ConcreteGraph& g,
                    Budget& b)
      : m(m), node(n), graph(g), budget(b) {}
  Value run() {
    for (const auto& p : m.parameters) {
      auto input = [&](std::size_t i) {
        const auto& n = graph.nodes.at(node.operands.at(i));
        return m.kind == core::metadata::ProgramKind::Shape ? shape(n)
                                                            : dtype(n);
      };
      if (p.variadic) {
        List xs;
        for (std::size_t i = p.operand; i < node.operands.size(); ++i)
          xs.push_back(input(i));
        locals[p.symbol.value] = xs;
      } else
        locals[p.symbol.value] = input(p.operand);
    }
    for (const auto& s : m.statements)
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, core::metadata::Let>)
              locals[v.symbol.value] = eval(*v.value);
            else
              ensure(boolean(eval(*v.condition)),
                     "operation metadata assertion failed");
          },
          s.value);
    return eval(*m.result.value);
  }

 private:
  Value eval(const core::metadata::Expr& e) {
    budget.tick();
    return std::visit(
        [&](const auto& v) -> Value {
          using T = std::decay_t<decltype(v)>;
          using namespace core::metadata;
          if constexpr (std::is_same_v<T, SymbolRef>)
            return locals.at(v.symbol.value);
          else if constexpr (std::is_same_v<T, Integer>)
            return parseInteger(v.spelling);
          else if constexpr (std::is_same_v<T, bool>)
            return v;
          else if constexpr (std::is_same_v<T, core::DType>)
            return std::string(core::dtypeName(v));
          else if constexpr (std::is_same_v<T, AttributeRef>)
            return attributeValue(node.attributes.at(v.field));
          else if constexpr (std::is_same_v<T, Call>) {
            std::vector<Value> a;
            for (const auto& x : v.arguments) a.push_back(eval(*x));
            return evaluateBuiltin(v.builtin, a, budget);
          } else if constexpr (std::is_same_v<T, Unary>)
            return evaluateOperator(spelling(v.op), {eval(*v.operand)}, budget);
          else if constexpr (std::is_same_v<T, Binary>) {
            if (category(v.op) == BinaryCategory::kLogical)
              return binary(
                  v.op, [&](bool rhs) { return eval(rhs ? *v.rhs : *v.lhs); });
            auto a = eval(*v.lhs), b = eval(*v.rhs);
            return evaluateOperator(spelling(v.op), {a, b}, budget);
          } else if constexpr (std::is_same_v<T, List>) {
            rewrite::List xs;
            for (const auto& x : v.elements) xs.push_back(eval(*x));
            budget.size(xs.size());
            return xs;
          } else if constexpr (std::is_same_v<T, Index>) {
            auto value = eval(*v.value);
            auto index = eval(*v.index);
            return evaluateIndex(value, index, budget);
          } else if constexpr (std::is_same_v<T, Conditional>)
            return eval(boolean(eval(*v.condition)) ? *v.then_value
                                                    : *v.else_value);
          else if constexpr (std::is_same_v<T, Comprehension>) {
            auto xs = list(eval(*v.iterable));
            rewrite::List out;
            auto saved = locals;
            for (const auto& x : xs) {
              locals[v.variable.value] = x;
              out.push_back(eval(*v.element));
            }
            locals = std::move(saved);
            return out;
          } else
            throw EvaluationError("unsupported metadata expression");
        },
        e.value);
  }
  const core::metadata::Program& m;
  const core::ConcreteNode& node;
  const core::ConcreteGraph& graph;
  Budget& budget;
  std::map<std::size_t, Value> locals;
};
std::vector<std::uint64_t> dimensions(const Value& v) {
  std::vector<std::uint64_t> xs;
  for (const auto& x : list(v)) {
    checkType(x, {core::TypeKind::kU64});
    xs.push_back(static_cast<std::uint64_t>(integer(x)));
  }
  return xs;
}
// Concrete execution of an axes contract. Group members carry fresh identity
// and extents; equal extents alone never identify two axes.
Value axisShape(const core::axis::ShapeContract& def,
                const core::ConcreteNode& node,
                const core::ConcreteGraph& graph, Budget& budget) {
  using M = core::axis::Meta;
  std::vector<Value> groups;
  std::function<Value(const M&)> eval = [&](const M& m) -> Value {
    budget.tick();
    using K = M::Kind;
    if (m.kind == K::Integer) return parseInteger(m.name);
    if (m.kind == K::Boolean) return m.name == "true";
    if (m.kind == K::Attribute)
      return attributeValue(node.attributes.at(m.index));
    if (m.kind == K::Operand)
      return shape(graph.nodes.at(node.operands.at(m.index)));
    if (m.kind == K::Group) return groups.at(m.index);
    if (m.kind == K::Call && m.name == "&&") {
      ensure(m.args.size() == 2, "bad logical arity");
      return boolean(eval(*m.args[0])) && boolean(eval(*m.args[1]));
    }
    if (m.kind == K::Call && m.name == "||") {
      ensure(m.args.size() == 2, "bad logical arity");
      return boolean(eval(*m.args[0])) || boolean(eval(*m.args[1]));
    }
    std::vector<Value> args;
    for (const auto& a : m.args) args.push_back(eval(*a));
    if (m.kind == K::List) return args;
    if (core::builtins::resolveAxisBuiltin(m.name))
      return evaluateAxisCall(m.name, args, budget);
    return evaluateOperator(m.name, args, budget);
  };
  std::vector<std::set<std::size_t>> used(node.operands.size());
  for (std::size_t g = 0; g < def.groups.size(); ++g) {
    const auto& group = def.groups[g];
    std::optional<List> extents;
    for (const auto& bind : group.bindings) {
      auto positions = eval(*bind.positions);
      auto input = shape(graph.nodes.at(node.operands.at(bind.operand)));
      ensure(boolean(evaluateBuiltin(core::builtins::Builtin::kIsValidAxisList,
                                     {positions, Integer(list(input).size())},
                                     budget)),
             "invalid axis binding");
      for (const auto& p : list(positions))
        ensure(used.at(bind.operand).insert(position(p)).second,
               "overlapping axis bindings");
      auto selected = list(evaluateBuiltin(core::builtins::Builtin::kGather,
                                           {input, positions}, budget));
      if (extents)
        ensure(*extents == selected, "axis group extents differ");
      else
        extents = selected;
    }
    if (group.domain) extents = list(eval(*group.domain));
    ensure(extents.has_value(), "axis group has no domain");
    List axes;
    for (std::size_t i = 0; i < extents->size(); ++i) {
      checkType((*extents)[i], {core::TypeKind::kU64});
      axes.emplace_back(Axis{0, g, i, integer((*extents)[i])});
    }
    groups.emplace_back(axes);
  }
  // Implicit operands must cover every input dimension. Explicit Read operands
  // use the declared indexing/assertion contract instead.
  for (auto i : def.implicit_operands)
    ensure(used.at(i).size() ==
               list(shape(graph.nodes.at(node.operands.at(i)))).size(),
           "axis bindings do not cover operand");
  for (const auto& a : def.assertions)
    ensure(boolean(eval(*a)), "axis assertion failed");
  auto output = eval(*def.output);
  ensure(boolean(evaluateIntrinsic(core::intrinsics::Intrinsic::kDistinctAxes,
                                   {output}, budget)),
         "duplicate output axes");
  return evaluateIntrinsic(core::intrinsics::Intrinsic::kAxisExtents, {output},
                           budget);
}
}  // namespace
Value Evaluator::eval(const core::completed::TypedExpr& e) {
  budget_.tick();
  const auto& p = program_.ir();
  auto value = std::visit(
      [&](const auto& v) -> Value {
        using T = std::decay_t<decltype(v)>;
        using namespace core;
        if constexpr (std::is_same_v<T, CaptureRef>)
          return NodeRef{bindings_.captures.at(v.capture.value)};
        else if constexpr (std::is_same_v<T, DimensionRef>)
          return bindings_.dimensions.at(v.dimension.value);
        else if constexpr (std::is_same_v<T, DescriptorRef>)
          return bindings_.descriptors.at(v.descriptor.value);
        else if constexpr (std::is_same_v<T, DescriptorFieldRef>)
          return bindings_.descriptors.at(v.descriptor.value)
              .fields.at(v.field);
        else if constexpr (std::is_same_v<T, DTypeVariableRef>)
          return bindings_.dtypes.at(v.variable.value);
        else if constexpr (std::is_same_v<T, DType>)
          return std::string(dtypeName(v));
        else if constexpr (std::is_same_v<T, NumericConstant>) {
          if (p.types.at(e.type.value).kind == TypeKind::kF64)
            return std::stod(v.spelling);
          return parseInteger(v.spelling);
        } else if constexpr (std::is_same_v<T, bool>)
          return v;
        else if constexpr (std::is_same_v<T, StringConstant> ||
                           std::is_same_v<T, EnumConstant>)
          return v.value;
        else if constexpr (std::is_same_v<T, NoneConstant>)
          return {};
        else if constexpr (std::is_same_v<T, core::completed::OptionalValue>)
          return eval(*v.value);
        else if constexpr (std::is_same_v<T, core::completed::BuiltinCall>) {
          std::vector<Value> a;
          for (const auto& x : v.arguments) a.push_back(eval(*x));
          if (core::builtins::isAxisQuery(v.builtin))
            return axis_analysis_.query(v.builtin, a);
          return evaluateBuiltin(v.builtin, a, budget_);
        } else if constexpr (std::is_same_v<T, core::completed::UnaryExpr>)
          return evaluateOperator(spelling(v.op), {eval(*v.operand)}, budget_);
        else if constexpr (std::is_same_v<T, core::completed::BinaryExpr>) {
          if (category(v.op) == BinaryCategory::kLogical)
            return binary(
                v.op, [&](bool rhs) { return eval(rhs ? *v.rhs : *v.lhs); });
          auto a = eval(*v.lhs), b = eval(*v.rhs);
          return evaluateOperator(spelling(v.op), {a, b}, budget_);
        } else if constexpr (std::is_same_v<T, LocalRef>)
          return bindings_.locals.at(v.local.value);
        else if constexpr (std::is_same_v<T, core::completed::LocalLet>) {
          auto a = eval(*v.value);
          bindings_.locals[v.local.value] = a;
          return a;
        } else if constexpr (std::is_same_v<T, core::completed::ListExpr>) {
          List xs;
          for (const auto& x : v.elements) xs.push_back(eval(*x));
          budget_.size(xs.size());
          return xs;
        } else if constexpr (std::is_same_v<T, core::completed::IndexExpr>) {
          auto value = eval(*v.value);
          auto index = eval(*v.index);
          return evaluateIndex(value, index, budget_);
        } else if constexpr (std::is_same_v<T,
                                            core::completed::ConditionalExpr>)
          return eval(boolean(eval(*v.condition)) ? *v.then_value
                                                  : *v.else_value);
        else if constexpr (std::is_same_v<T,
                                          core::completed::ComprehensionExpr>) {
          auto xs = list(eval(*v.iterable));
          List out;
          auto saved = bindings_.locals;
          for (const auto& x : xs) {
            bindings_.locals[v.variable.value] = x;
            out.push_back(eval(*v.element));
          }
          bindings_.locals = std::move(saved);
          return out;
        } else if constexpr (std::is_same_v<
                                 T, core::completed::DescriptorConstruct>) {
          DescriptorValue out{v.schema.value, {}};
          for (const auto& f : v.fields) out.fields.push_back(eval(*f));
          return out;
        } else if constexpr (std::is_same_v<T, core::completed::SymbolicExpr>)
          return symbolic(*v.value);
        else
          throw EvaluationError("unresolved expression in executable program");
      },
      e.value);
  // A let node's expression type is its initialized value's type.
  return checkType(std::move(value), p.types.at(e.type.value));
}
bool Evaluator::evaluateWhere() {
  const auto& rule = program_.ir().rules.at(rule_);
  bool allowed = true;
  auto schedule = core::completed::scheduleWhere(rule);
  for (const auto& condition : schedule.source) {
    auto value = eval(*condition);
    if (!std::holds_alternative<core::completed::LocalLet>(condition->value) &&
        !boolean(value)) {
      allowed = false;
      break;
    }
  }
  if (!allowed) return false;
  auto where_locals = std::move(bindings_.locals);
  bindings_.locals.clear();
  for (const auto& d : rule.derivations) {
    auto value = eval(*d.value);
    if (d.target)
      bindings_.descriptors[d.target->value] =
          std::get<DescriptorValue>(value.data);
  }
  bindings_.locals = std::move(where_locals);
  for (const auto& condition : schedule.replacement) {
    auto value = eval(*condition);
    if (!std::holds_alternative<core::completed::LocalLet>(condition->value) &&
        !boolean(value)) {
      allowed = false;
      break;
    }
  }
  if (!allowed) return false;
  return true;
}
Value Evaluator::symbolic(const core::completed::Symbol& s) {
  budget_.tick();
  using K = core::completed::Symbol::Kind;
  if (s.kind == K::Integer) return parseInteger(s.name);
  if (s.kind == K::Boolean) return s.name == "true";
  if (s.kind == K::Field)
    return bindings_.descriptors.at(s.owner).fields.at(s.index);
  if (s.kind == K::Shape)
    return shape(graph_.nodes.at(bindings_.captures.at(s.owner)));
  if (s.kind == K::DType)
    return dtype(graph_.nodes.at(bindings_.captures.at(s.owner)));
  if (s.kind == K::AxisGroup) {
    auto key = std::pair{s.owner, s.index};
    if (axis_cache_.contains(key)) return axis_cache_.at(key);
    ensure(active_axes_.insert(key).second, "cyclic axis definition");
    const core::completed::Symbol* definition = nullptr;
    for (const auto& d : program_.axes(rule_))
      if (d->owner == s.owner && d->index == s.index) {
        definition = d.get();
        break;
      }
    ensure(definition && definition->args.size() == 1,
           "missing axis definition");
    auto extents = list(symbolic(*definition->args[0]));
    List axes;
    for (std::size_t i = 0; i < extents.size(); ++i)
      axes.emplace_back(Axis{s.owner, s.index, i, integer(extents[i])});
    active_axes_.erase(key);
    return axis_cache_[key] = axes;
  }
  if (s.kind == K::Call && s.name == "&&")
    return boolean(symbolic(*s.args.at(0))) && boolean(symbolic(*s.args.at(1)));
  if (s.kind == K::Call && s.name == "||")
    return boolean(symbolic(*s.args.at(0))) || boolean(symbolic(*s.args.at(1)));
  if (s.kind == K::Call && s.name == "if")
    return symbolic(*s.args.at(boolean(symbolic(*s.args.at(0))) ? 1 : 2));
  std::vector<Value> args;
  for (const auto& a : s.args) args.push_back(symbolic(*a));
  if (s.kind == K::List) return args;
  ensure(s.kind == K::Call, "unresolved symbolic expression");
  return evaluateSymbolicCall(s.name, args, budget_);
}
void infer(const executable::ProgramData& p, core::ConcreteGraph& graph,
           std::size_t id, Budget& budget) {
  auto& n = graph.nodes.at(id);
  using K = core::ConcreteNode::Kind;
  if (n.kind == K::kInput) return;
  if (n.kind == K::kLiteral) {
    n.shape = std::vector<std::uint64_t>{};
    return;
  }
  const auto& op = p.operations.at(n.operation->value);
  // Cached metadata in a serialized graph is not an inference contract.
  n.shape.reset();
  n.dtype.reset();
  if (op.attributes) {
    const auto& fields = p.attribute_schemas.at(op.attributes->value).fields;
    ensure(fields.size() == n.attributes.size(), "attribute count mismatch");
    for (std::size_t f = 0; f < fields.size(); ++f)
      attribute(attributeValue(n.attributes[f]), fields[f]);
  }
  bool known_shape = true, known_dtype = true;
  for (auto a : n.operands) {
    known_shape &= graph.nodes.at(a).shape.has_value();
    known_dtype &= graph.nodes.at(a).dtype.has_value();
  }
  if (known_shape) {
    if (op.shape)
      n.shape =
          dimensions(MetadataEvaluator(*op.shape, n, graph, budget).run());
    else if (op.axis_contract)
      n.shape = dimensions(axisShape(*op.axis_contract, n, graph, budget));
  }
  if (known_dtype && op.dtype) {
    auto value = MetadataEvaluator(*op.dtype, n, graph, budget).run();
    auto t = core::resolveDType(std::get<std::string>(value.data));
    ensure(t.has_value(), "invalid inferred dtype");
    n.dtype = *t;
  }
}
}  // namespace tepl::rewrite
