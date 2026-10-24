#include <algorithm>
#include <functional>
#include <set>
#include <stdexcept>
#include <type_traits>

#include "src/core/semantics/compilation_context.h"
#include "src/core/semantics/context.h"
#include "src/core/semantics/graph.h"
#include "src/core/semantics/suggestion.h"
#include "src/core/attr_synth/synthesize.h"
#include "src/shared/metadata/ordered.h"

namespace tepl::core::semantics {
namespace {
struct Builder {
  CompilationContext& context;
  Rule& rule;
  Plan plan;
  std::map<std::size_t, const Pattern*> bindings;
  std::set<std::size_t> active;
  void assumptions() {
    std::set<std::size_t> leaves;
    for (const auto& n : plan.nodes)
      if (n.capture) leaves.insert(n.capture->value);
    std::map<std::size_t, S> dims, dtypes;
    for (const auto& constraint : rule.constraints) {
      if (!leaves.contains(constraint.capture.value)) continue;
      auto shape = make(Symbol::Kind::Shape, {}, constraint.capture.value);
      auto rank = call("len", {shape});
      std::size_t fixed = 0;
      bool sequence = false;
      for (const auto& e : constraint.shape)
        if (e.kind == ShapeElement::Kind::kSequence)
          sequence = true;
        else
          ++fixed;
      plan.assumptions.push_back(
          call(sequence ? ">=" : "==", {rank, integer(fixed)}));
      S offset = integer(0);
      for (const auto& e : constraint.shape) {
        S value;
        if (e.kind == ShapeElement::Kind::kSequence) {
          auto length = call("-", {rank, integer(fixed)});
          value = call("slice", {shape, offset, call("+", {offset, length})});
          offset = call("+", {offset, length});
        } else {
          value = call("index", {shape, offset});
          offset = call("+", {offset, integer(1)});
        }
        if (e.symbol) {
          auto [it, inserted] = dims.emplace(e.symbol->value, value);
          if (!inserted)
            plan.assumptions.push_back(call("==", {it->second, value}));
        }
      }
      if (constraint.dtype) {
        auto dtype = make(Symbol::Kind::DType, {}, constraint.capture.value);
        if (const auto* fixed = std::get_if<DType>(&*constraint.dtype))
          plan.assumptions.push_back(call(
              "==",
              {dtype, call("dtype_" + std::string(dtypeName(*fixed)), {})}));
        else {
          auto id = std::get<DTypeVariableId>(*constraint.dtype).value;
          auto [it, inserted] = dtypes.emplace(id, dtype);
          if (!inserted)
            plan.assumptions.push_back(call("==", {it->second, dtype}));
        }
      }
    }
    std::unique_ptr<Context> coordinates;
    std::map<std::size_t, S> capture_axes;
    E source_value;
    auto sourceCoordinates = [&]() -> Context& {
      if (!coordinates) {
        coordinates = std::make_unique<Context>(plan, 1000000);
        coordinates->sourceScope(plan.lhs);
        Lowerer source{*coordinates};
        source_value = source.lower(plan.lhs).value;
      }
      return *coordinates;
    };
    auto sourceQuery = [&](const BuiltinCall& query) -> S {
      using B = builtins::Builtin;
      if (query.builtin != B::kAxisCaptureOrder &&
          query.builtin != B::kAxisSourceNodeOrder &&
          query.builtin != B::kAxisSourceGroupOrder &&
          query.builtin != B::kAxisSourceGroupMembers &&
          query.builtin != B::kAxisSourceOrder)
        return {};
      auto& c = sourceCoordinates();
      auto argument = [&](std::size_t i) {
        return std::stoull(
            std::get<NumericConstant>(query.arguments.at(i)->value).spelling);
      };
      if (query.builtin == B::kAxisSourceOrder)
        return c.source().tensor_axes.at(plan.lhs);
      if (query.builtin == B::kAxisCaptureOrder) {
        if (capture_axes.empty())
          for (const auto& term : normalize(c, source_value))
            for (const auto& read : term.reads) {
              auto [found, added] =
                  capture_axes.try_emplace(read->capture, read->axes);
              if (!added && !c.entails(call("==", {found->second, read->axes})))
                throw std::runtime_error(
                    "logical axis order requires an unambiguous source capture "
                    "occurrence");
            }
        auto found = capture_axes.find(argument(0));
        if (found == capture_axes.end())
          throw std::runtime_error("source capture has no logical axis frame");
        return found->second;
      }
      auto id = argument(0);
      if (query.builtin == B::kAxisSourceNodeOrder)
        return c.source().tensor_axes.at(id);
      auto contract = axis::recognizeOrdered(*plan.nodes.at(id).definition);
      if (!contract) throw std::runtime_error("unsupported source group query");
      auto role = argument(1);
      auto group = contract->kind == axis::OrderedContract::Kind::Permutation
                       ? 0
                   : role == 0 ? contract->batch_group
                   : role == 1 ? contract->contract_group
                   : role == 2 ? contract->left_free_group
                               : contract->right_free_group;
      auto axes = c.source().tensor_groups.at(id).at(group);
      return query.builtin == B::kAxisSourceGroupMembers
                 ? call("unique_axes", {axes})
                 : axes;
    };
    auto outputPositions = [&](const OutputAxisGroupRef& ref) -> S {
      sourceCoordinates();
      std::optional<std::size_t> operation;
      for (auto id : coordinates->source().nodes)
        if (plan.nodes[id].label == ref.node) {
          if (operation)
            throw std::runtime_error("duplicate semantic node label");
          operation = id;
        }
      if (!operation)
        throw std::runtime_error(
            "output-axis reference must name an LHS operation");
      const auto& n = plan.nodes[*operation];
      if (!n.definition)
        throw std::runtime_error(
            "output-axis reference requires axis semantics");
      for (std::size_t g = 0; g < n.definition->groups.size(); ++g) {
        if (n.definition->groups[g].name != ref.group) continue;
        bool present = false;
        std::function<void(const Meta&)> visit = [&](const Meta& m) {
          if (m.kind == Meta::Kind::Group && m.index == g) present = true;
          // Only the sequence operand of permute contributes output groups.
          auto count = m.kind == Meta::Kind::Call && m.name == "permute"
                           ? std::size_t{1}
                           : m.args.size();
          for (std::size_t i = 0; i < count; ++i) {
            if (m.kind == Meta::Kind::Call && m.name == "place" && i == 1)
              continue;
            visit(*m.args[i]);
          }
        };
        visit(*n.definition->output);
        if (!present)
          throw std::runtime_error("axis group '" + ref.group +
                                   "' is absent from the operation output");
        auto result = coordinates->positions(
            coordinates->source().tensor_axes.at(*operation),
            coordinates->source().tensor_groups.at(*operation).at(g));
        coordinates->validateSource(result);
        return result;
      }
      throw std::runtime_error("unknown output-axis group '" + ref.group + "'");
    };
    std::map<std::size_t, S> locals;
    auto rawCall = [](std::string name, std::vector<S> args) {
      return make(Symbol::Kind::Call, std::move(name), 0, 0, std::move(args));
    };
    std::function<S(const TypedExpr&)> lower = [&](const TypedExpr& e) -> S {
      return std::visit(
          [&](const auto& v) -> S {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, OutputAxisGroupRef>) {
              return outputPositions(v);
            } else if constexpr (std::is_same_v<T, SymbolicExpr>) {
              return v.value;
            } else if constexpr (std::is_same_v<T, DescriptorFieldRef>) {
              plan.metadata_types.emplace(
                  std::pair{v.descriptor.value, v.field},
                  context.output.types.at(e.type.value));
              if (auto found = plan.attributes.find(v.descriptor.value);
                  found != plan.attributes.end())
                return found->second.at(v.field);
              if (rule.descriptors.at(v.descriptor.value).kind !=
                  Descriptor::Kind::kCaptured)
                return {};
              for (const auto& n : plan.nodes)
                if (n.descriptor == v.descriptor && n.definition &&
                    !n.automatic)
                  return make(Symbol::Kind::Field,
                              n.definition->fields.at(v.field),
                              v.descriptor.value, v.field);
              return {};
            } else if constexpr (std::is_same_v<T, DimensionRef>)
              return dims.contains(v.dimension.value)
                         ? dims.at(v.dimension.value)
                         : S{};
            else if constexpr (std::is_same_v<T, DTypeVariableRef>)
              return dtypes.contains(v.variable.value)
                         ? dtypes.at(v.variable.value)
                         : S{};
            else if constexpr (std::is_same_v<T, DType>)
              return call("dtype_" + std::string(dtypeName(v)), {});
            else if constexpr (std::is_same_v<T, bool>)
              return boolean(v);
            else if constexpr (std::is_same_v<T, NumericConstant>)
              return v.decimal ? S{} : make(Symbol::Kind::Integer, v.spelling);
            else if constexpr (std::is_same_v<T, BinaryExpr>) {
              auto a = lower(*v.lhs), b = lower(*v.rhs);
              if (!a || !b) return {};
              if ((v.op == BinaryOp::kEqual || v.op == BinaryOp::kNotEqual) &&
                  context.output.types.at(v.lhs->type.value) ==
                      logicalAxisSetType()) {
                auto same = rawCall("same_set", {a, b});
                return v.op == BinaryOp::kEqual ? same : rawCall("!", {same});
              }
              return rawCall(std::string(spelling(v.op)), {a, b});
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
              auto a = lower(*v.operand);
              return a ? rawCall(std::string(spelling(v.op)), {a}) : S{};
            } else if constexpr (std::is_same_v<T, BuiltinCall>) {
              if (auto axes = sourceQuery(v)) return axes;
              std::vector<S> args;
              for (const auto& a : v.arguments) {
                auto x = lower(*a);
                if (!x) return {};
                args.push_back(x);
              }
              return rawCall(
                  v.builtin == builtins::Builtin::kExclude
                      ? "ordered_complement"
                  : v.builtin == builtins::Builtin::kAxisFilter ? "filter_axes"
                  : v.builtin == builtins::Builtin::kAxisDifference
                      ? "difference_axes"
                      : std::string(builtins::builtinName(v.builtin)),
                  std::move(args));
            } else if constexpr (std::is_same_v<T, LocalRef>) {
              auto found = locals.find(v.local.value);
              return found != locals.end() ? found->second : S{};
            } else if constexpr (std::is_same_v<T, ListExpr>) {
              std::vector<S> values;
              for (const auto& a : v.elements) {
                auto value = lower(*a);
                if (!value) return {};
                values.push_back(value);
              }
              return make(Symbol::Kind::List, {}, 0, 0, std::move(values));
            } else if constexpr (std::is_same_v<T, core::IndexExpr>) {
              auto value = lower(*v.value), index = lower(*v.index);
              return value && index ? rawCall("index", {value, index}) : S{};
            } else if constexpr (std::is_same_v<T, ConditionalExpr>) {
              auto condition = lower(*v.condition), yes = lower(*v.then_value),
                   no = lower(*v.else_value);
              return condition && yes && no
                         ? rawCall("if", {condition, yes, no})
                         : S{};
            } else if constexpr (std::is_same_v<T, ComprehensionExpr>) {
              auto iterable = lower(*v.iterable);
              if (!iterable) return {};
              auto variable =
                  call("sequence_variable", {integer(v.variable.value)});
              auto previous = locals.find(v.variable.value);
              std::optional<S> old;
              if (previous != locals.end()) old = previous->second;
              locals[v.variable.value] = variable;
              auto body = lower(*v.element);
              if (old)
                locals[v.variable.value] = *old;
              else
                locals.erase(v.variable.value);
              return body ? rawCall("sequence_map", {iterable, variable, body})
                          : S{};
            } else
              return {};
          },
          e.value);
    };
    for (const auto& condition : rule.conditions) {
      if (const auto* local = std::get_if<LocalLet>(&condition->value)) {
        auto value = lower(*local->value);
        if (!value)
          throw std::runtime_error(
              "unsupported semantic-rule local expression");
        locals[local->local.value] = value;
        continue;
      }
      auto value = lower(*condition);
      if (!value) {
        context.report(condition->origin,
                       "unsupported semantic-rule condition; expected a "
                       "compiler-readable metadata expression");
        throw std::runtime_error("cannot interpret declared condition");
      }
      plan.assumptions.push_back(value);
    }
    std::function<TypedExprPtr(const TypedExprPtr&)> completeQueries =
        [&](const TypedExprPtr& e) -> TypedExprPtr {
      auto out = *e;
      if (auto* comparison = std::get_if<BinaryExpr>(&out.value);
          comparison &&
          (comparison->op == BinaryOp::kEqual ||
           comparison->op == BinaryOp::kNotEqual) &&
          context.output.types.at(comparison->lhs->type.value) ==
              logicalAxisSetType()) {
        out.value = SymbolicExpr{lower(*e)};
        return std::make_shared<TypedExpr>(std::move(out));
      }
      if (auto* query = std::get_if<BuiltinCall>(&out.value)) {
        if (auto axes = sourceQuery(*query)) {
          out.value = SymbolicExpr{axes};
          return std::make_shared<TypedExpr>(std::move(out));
        }
      }
      std::visit(
          [&](auto& v) {
            if constexpr (requires { v.arguments; })
              for (auto& a : v.arguments) a = completeQueries(a);
            if constexpr (requires { v.elements; })
              for (auto& a : v.elements) a = completeQueries(a);
            if constexpr (requires { v.operand; })
              v.operand = completeQueries(v.operand);
            if constexpr (requires { v.lhs; }) {
              v.lhs = completeQueries(v.lhs);
              v.rhs = completeQueries(v.rhs);
            }
            if constexpr (requires { completeQueries(v.value); })
              v.value = completeQueries(v.value);
            if constexpr (requires { v.iterable; }) {
              v.element = completeQueries(v.element);
              v.iterable = completeQueries(v.iterable);
            }
            if constexpr (requires { v.condition; }) {
              v.condition = completeQueries(v.condition);
              v.then_value = completeQueries(v.then_value);
              v.else_value = completeQueries(v.else_value);
            }
            if constexpr (requires { completeQueries(v.index); }) {
              v.value = completeQueries(v.value);
              v.index = completeQueries(v.index);
            }
          },
          out.value);
      return std::make_shared<TypedExpr>(std::move(out));
    };
    for (auto& condition : rule.conditions)
      condition = completeQueries(condition);
    locals.clear();  // where and derive have independent local scopes.
    for (const auto& derivation : rule.derivations) {
      if (!derivation.target) {
        const auto& local = std::get<LocalLet>(derivation.value->value);
        auto value = lower(*local.value);
        if (!value)
          throw std::runtime_error(
              "unsupported semantic-rule local expression");
        locals[local.local.value] = value;
        plan.derived_metadata.push_back(value);
        continue;
      }
      std::vector<S> fields;
      if (const auto* ctor =
              std::get_if<DescriptorConstruct>(&derivation.value->value)) {
        for (const auto& f : ctor->fields) {
          auto value = lower(*f);
          if (!value) {
            context.report(f->origin,
                           "unsupported semantic descriptor field expression");
            throw std::runtime_error(
                "cannot interpret explicit descriptor constructor");
          }
          fields.push_back(value);
          plan.derived_metadata.push_back(value);
          plan.constructed_metadata.emplace_back(
              value, context.output.types.at(f->type.value));
        }
      } else if (const auto* ref =
                     std::get_if<DescriptorRef>(&derivation.value->value)) {
        if (auto found = plan.attributes.find(ref->descriptor.value);
            found != plan.attributes.end())
          fields = found->second;
        else {
          for (const auto& n : plan.nodes) {
            if (n.descriptor != ref->descriptor || !n.definition) continue;
            for (std::size_t f = 0; f < n.definition->fields.size(); ++f)
              fields.push_back(make(Symbol::Kind::Field,
                                    n.definition->fields[f],
                                    ref->descriptor.value, f));
            break;
          }
        }
      } else
        throw std::runtime_error("unsupported semantic descriptor expression");
      plan.attributes.emplace(derivation.target->value, std::move(fields));
    }
  }
  std::size_t capture(CaptureId id, const SourceOrigin& at) {
    if (bindings.contains(id.value)) {
      if (!active.insert(id.value).second)
        throw std::runtime_error("cyclic intermediate capture");
      auto result = source(*bindings.at(id.value));
      active.erase(id.value);
      return result;
    }
    Instance n;
    n.capture = id;
    n.origin = at;
    plan.nodes.push_back(n);
    return plan.nodes.size() - 1;
  }
  std::size_t operation(OpId id, std::optional<DescriptorId> descriptor,
                        std::vector<std::size_t> inputs, const SourceOrigin& at,
                        const std::string& label = "") {
    const auto& op = context.output.operations.at(id.value);
    if (!op.semantic_definition)
      throw std::runtime_error("operation '" + op.name +
                               "' lacks axis semantics; all operations in a "
                               "semantic rule require axes and semantics");
    if (!op.dtype)
      throw std::runtime_error("index operation requires dtype contract");
    Instance n;
    n.operation = id;
    n.label = label;
    n.operation_name = rule.operation_spellings.contains(id.value)
                           ? rule.operation_spellings.at(id.value)
                           : op.alias.value_or(op.name);
    n.definition = op.semantic_definition;
    n.dtype = std::make_shared<metadata::Program>(*op.dtype);
    n.descriptor = descriptor;
    n.inputs = std::move(inputs);
    n.origin = at;
    for (const auto& h : rule.attribute_holes) {
      if (!descriptor || h.descriptor != *descriptor) continue;
      n.automatic = true;
    }
    plan.nodes.push_back(std::move(n));
    return plan.nodes.size() - 1;
  }
  std::size_t literal(const GraphLiteral& value, const SourceOrigin& origin) {
    Instance n;
    n.literal = value.spelling;
    n.literal_dtype = value.dtype;
    n.origin = origin;
    plan.nodes.push_back(std::move(n));
    return plan.nodes.size() - 1;
  }
  std::size_t source(const Pattern& p) {
    if (const auto* l = std::get_if<GraphLiteral>(&p.value))
      return literal(*l, p.origin);
    if (const auto* c = std::get_if<CapturePattern>(&p.value))
      return capture(c->capture, p.origin);
    if (const auto* b = std::get_if<BindPattern>(&p.value))
      return source(*b->expression);
    if (const auto* op = std::get_if<MatchOperation>(&p.value)) {
      std::vector<std::size_t> inputs;
      for (const auto& input : op->operands) inputs.push_back(source(*input));
      return operation(op->operation, op->descriptor, std::move(inputs),
                       p.origin, op->label);
    }
    throw std::runtime_error(
        "semantic graphs support tensor captures and operation "
        "applications");
  }
  std::size_t target(const BuildExpr& p) {
    if (const auto* l = std::get_if<GraphLiteral>(&p.value))
      return literal(*l, p.origin);
    if (const auto* c = std::get_if<CaptureRef>(&p.value))
      return capture(c->capture, p.origin);
    if (const auto* op = std::get_if<BuildOperation>(&p.value)) {
      std::vector<std::size_t> inputs;
      for (const auto& input : op->operands) inputs.push_back(target(*input));
      return operation(op->operation, op->descriptor, std::move(inputs),
                       p.origin, op->label);
    }
    throw std::runtime_error(
        "semantic graphs support captures and operation applications");
  }
};
}  // namespace
bool compile(CompilationContext& context, InspectedRule& rule) {
  bool uses = false;
  std::function<void(const Pattern&)> scan = [&](const Pattern& p) {
    if (const auto* op = std::get_if<MatchOperation>(&p.value)) {
      uses |= bool(context.output.operations.at(op->operation.value)
                       .semantic_definition);
      for (const auto& c : op->operands) scan(*c);
    } else if (const auto* b = std::get_if<BindPattern>(&p.value))
      scan(*b->expression);
  };
  std::function<void(const BuildExpr&)> rhs = [&](const BuildExpr& p) {
    if (const auto* op = std::get_if<BuildOperation>(&p.value)) {
      uses |= bool(context.output.operations.at(op->operation.value)
                       .semantic_definition);
      for (const auto& c : op->operands) rhs(*c);
    }
  };
  scan(*rule.lhs);
  rhs(*rule.rhs);
  if (!uses) return false;
  try {
    Builder b{context, rule};
    std::function<void(const Pattern&)> bind = [&](const Pattern& p) {
      if (const auto* v = std::get_if<BindPattern>(&p.value)) {
        b.bindings[v->capture.value] = v->expression.get();
        bind(*v->expression);
      } else if (const auto* op = std::get_if<MatchOperation>(&p.value))
        for (const auto& c : op->operands) bind(*c);
    };
    bind(*rule.lhs);
    b.plan.algebraic =
        context.options.semantic_policy == SemanticPolicy::kAlgebraic;
    b.plan.lhs = b.source(*rule.lhs);
    b.plan.rhs = b.target(*rule.rhs);
    for (const auto& d : rule.descriptors)
      b.plan.descriptor_names[d.id.value] = d.name;
    // Public properties retain symbolic node names until both trees exist.
    // Resolve them once to the same closed queries used by generated guards.
    auto indexType = [&] {
      core::Type t{TypeKind::kIndex};
      for (std::size_t i = 0; i < context.output.types.size(); ++i)
        if (context.output.types[i] == t) return TypeId{i};
      context.output.types.push_back(t);
      return TypeId{context.output.types.size() - 1};
    }();
    auto number = [&](std::size_t i, const SourceOrigin& at) -> TypedExprPtr {
      return std::make_shared<TypedExpr>(
          TypedExpr{at, indexType, NumericConstant{std::to_string(i), false}});
    };
    std::function<TypedExprPtr(const TypedExprPtr&)> properties =
        [&](const TypedExprPtr& e) -> TypedExprPtr {
      auto out = *e;
      if (const auto* ref = std::get_if<NodeAxisRef>(&e->value)) {
        using B = builtins::Builtin;
        B query;
        std::vector<TypedExprPtr> args;
        if (ref->capture) {
          query = B::kAxisCaptureOrder;
          args.push_back(number(ref->capture->value, e->origin));
        } else {
          auto it =
              std::find_if(b.plan.nodes.begin(), b.plan.nodes.end(),
                           [&](const auto& n) { return n.label == ref->node; });
          if (it == b.plan.nodes.end())
            throw std::runtime_error("unknown semantic node '" + ref->node +
                                     "'");
          auto id = static_cast<std::size_t>(it - b.plan.nodes.begin());
          args.push_back(number(id, e->origin));
          if (ref->projection == "partition" || ref->projection == "legal") {
            if (id != (ref->rhs ? b.plan.rhs : b.plan.lhs))
              throw std::runtime_error(
                  "network partition and legality queries require a rule root");
            query = ref->projection == "legal" ? B::kAxisRhsLegal
                    : ref->rhs                 ? B::kAxisRhsPartition
                                               : B::kAxisSourcePartition;
            args.clear();
          } else if (ref->projection == "required_order")
            query = B::kAxisRequired;
          else if (ref->group.empty())
            query = ref->rhs ? B::kAxisRhsNodeOrder : B::kAxisSourceNodeOrder;
          else {
            auto contract = axis::recognizeOrdered(*it->definition);
            if (!contract)
              throw std::runtime_error(
                  "logical group access is outside the supported operation "
                  "templates");
            auto g = std::find_if(
                it->definition->groups.begin(), it->definition->groups.end(),
                [&](const auto& group) { return group.name == ref->group; });
            auto group =
                static_cast<std::size_t>(g - it->definition->groups.begin());
            std::size_t role = 4;
            if (contract->kind == axis::OrderedContract::Kind::Contraction) {
              if (group == contract->batch_group)
                role = 0;
              else if (group == contract->contract_group)
                role = 1;
              else if (group == contract->left_free_group)
                role = 2;
              else if (group == contract->right_free_group)
                role = 3;
              else
                throw std::runtime_error("unrecognized semantic axis group");
            }
            if (ref->rhs && it->automatic) {
              query = role == 0   ? B::kAxisBatch
                      : role == 1 ? B::kAxisContract
                      : role == 2 ? B::kAxisLeftFree
                      : role == 3 ? B::kAxisRightFree
                                  : B::kAxisLive;
            } else {
              args.push_back(number(role, e->origin));
              if (ref->rhs)
                query = ref->projection == "members" ? B::kAxisRhsGroupMembers
                        : ref->projection == "positions"
                            ? B::kAxisRhsGroupPositions
                            : B::kAxisRhsGroupOrder;
              else
                query = ref->projection == "members"
                            ? B::kAxisSourceGroupMembers
                            : B::kAxisSourceGroupOrder;
            }
          }
        }
        out.value = BuiltinCall{query, std::move(args)};
      } else
        std::visit(
            [&](auto& v) {
              if constexpr (requires { v.arguments; })
                for (auto& a : v.arguments) a = properties(a);
              if constexpr (requires { v.elements; })
                for (auto& a : v.elements) a = properties(a);
              if constexpr (requires { v.fields; })
                for (auto& a : v.fields) a = properties(a);
              if constexpr (requires { v.operand; })
                v.operand = properties(v.operand);
              if constexpr (requires { v.lhs; }) {
                v.lhs = properties(v.lhs);
                v.rhs = properties(v.rhs);
              }
              if constexpr (requires { properties(v.value); })
                v.value = properties(v.value);
              if constexpr (requires { properties(v.index); }) {
                v.value = properties(v.value);
                v.index = properties(v.index);
              }
              if constexpr (requires { v.condition; }) {
                v.condition = properties(v.condition);
                v.then_value = properties(v.then_value);
                v.else_value = properties(v.else_value);
              }
              if constexpr (requires { v.iterable; }) {
                v.iterable = properties(v.iterable);
                v.element = properties(v.element);
              }
            },
            out.value);
      return std::make_shared<TypedExpr>(std::move(out));
    };
    for (auto& c : rule.conditions) c = properties(c);
    const bool selected = context.options.semantic_rule.empty() ||
                          context.options.semantic_rule == rule.name;
    std::function<bool(const TypedExpr&)> axisQueries =
        [&](const TypedExpr& e) {
          bool found = false;
          if (auto* b = std::get_if<BuiltinCall>(&e.value))
            found = builtins::isReplacementAxisQuery(b->builtin) ||
                    b->builtin == builtins::Builtin::kAxisDomain ||
                    b->builtin == builtins::Builtin::kAxisSourcePartition;
          forEachExpressionChild(
              e, [&](const auto& child) { found |= axisQueries(child); });
          return found;
        };
    bool explicit_axis_conditions = false;
    for (const auto& c : rule.conditions)
      explicit_axis_conditions |= axisQueries(*c);
    if (explicit_axis_conditions)
      throw std::runtime_error(
          "runtime feasibility queries are unsupported; use source metadata "
          "restrictions "
          "and check --suggest-where for a verified suggestion");
    const Rule suggestion_source = rule;
    b.assumptions();
    if (selected) {
      b.plan.proof = synthesize(b.plan);
      if (b.plan.proof.status != ProofStatus::Proved &&
          context.options.suggest_conditions)
        b.plan.suggested_rule =
            suggestRestrictions(b.plan, suggestion_source, context.output);
    }
    if (selected && !context.options.inspect_semantic_rules &&
        b.plan.proof.status != ProofStatus::Proved) {
      auto message =
          "cannot verify rule '" + rule.name + "': " + b.plan.proof.reason;
      message +=
          ". Add an explicit where condition establishing the unresolved "
          "requirement, or change the rewrite. Use check --suggest-where to "
          "search for a verified restriction.";
      if (!b.plan.suggested_rule.empty())
        message +=
            "\nVerified sufficient restriction (bounded search; not "
            "necessarily weakest):\n" +
            b.plan.suggested_rule;
      context.report(rule.origin, std::move(message));
    }
    rule.semantic_plan = std::make_shared<Plan>(std::move(b.plan));
  } catch (const std::exception& e) {
    context.report(rule.origin, std::string("semantic inference: ") + e.what());
  }
  return true;
}
}  // namespace tepl::core::semantics
