#include <functional>
#include <set>
#include <stdexcept>
#include <type_traits>

#include "src/core/analysis_context.h"
#include "src/core/index_semantics/check.h"
#include "src/core/index_semantics/graph.h"
#include "src/core/index_semantics/synthesize.h"

namespace tepl::core::index_semantics {
namespace {
struct Builder {
  detail::AnalysisContext& context;
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
    auto outputPositions = [&](const OutputAxisGroupRef& ref) -> S {
      if (!coordinates) {
        coordinates = std::make_unique<Context>(plan, 1000000);
        coordinates->sourceScope(plan.lhs);
        Lowerer source{*coordinates};
        source.lower(plan.lhs);
      }
      std::optional<std::size_t> operation;
      for (auto id : coordinates->source().nodes)
        if (plan.nodes[id].descriptor == ref.descriptor) {
          if (operation)
            throw std::runtime_error(
                "ambiguous output-axis reference: descriptor names multiple "
                "LHS operations");
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
    std::function<S(const TypedExpr&)> lower = [&](const TypedExpr& e) -> S {
      return std::visit(
          [&](const auto& v) -> S {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, OutputAxisGroupRef>) {
              return outputPositions(v);
            } else if constexpr (std::is_same_v<T, DescriptorFieldRef>) {
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
              return a && b ? call(std::string(spelling(v.op)), {a, b}) : S{};
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
              auto a = lower(*v.operand);
              return a ? call(std::string(spelling(v.op)), {a}) : S{};
            } else if constexpr (std::is_same_v<T, BuiltinCall>) {
              std::vector<S> args;
              for (const auto& a : v.arguments) {
                auto x = lower(*a);
                if (!x) return {};
                args.push_back(x);
              }
              return call(v.builtin == builtins::Builtin::kExclude
                              ? "ordered_complement"
                              : std::string(builtins::builtinName(v.builtin)),
                          std::move(args));
            } else
              return {};
          },
          e.value);
    };
    for (const auto& condition : rule.conditions) {
      auto value = lower(*condition);
      if (!value) {
        context.report(condition->origin,
                       "unsupported semantic-rule condition; expected a "
                       "compiler-readable metadata expression");
        throw std::runtime_error("cannot interpret declared condition");
      }
      plan.assumptions.push_back(value);
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
                        std::vector<std::size_t> inputs,
                        const SourceOrigin& at) {
    const auto& op = context.output.operations.at(id.value);
    if (!op.semantic_definition)
      throw std::runtime_error("operation '" + op.name +
                               "' lacks axis semantics; all operations in a "
                               "semantic rule require axes and semantics");
    if (!op.dtype)
      throw std::runtime_error("index operation requires dtype contract");
    Instance n;
    n.definition = op.semantic_definition;
    n.dtype = std::make_shared<metadata::Program>(*op.dtype);
    n.descriptor = descriptor;
    n.inputs = std::move(inputs);
    n.origin = at;
    for (const auto& h : rule.attribute_holes) {
      if (!descriptor || h.descriptor != *descriptor) continue;
      n.automatic = true;
      n.pins.resize(h.overrides.size());
      for (std::size_t f = 0; f < h.overrides.size(); ++f) {
        const auto& v = h.overrides[f];
        if (!v) continue;
        Pin pin;
        if (v->kind == AttributePin::Kind::kField) {
          pin.descriptor = v->descriptor;
          pin.field = v->field;
        } else if (v->kind == AttributePin::Kind::kLiteral &&
                   v->text == "list") {
          List values;
          for (const auto& element : v->arguments) {
            if (element->kind != AttributePin::Kind::kLiteral)
              throw std::runtime_error("unsupported nonliteral auto pin");
            values.push_back(std::stoll(element->text));
          }
          pin.literal = values;
        } else
          throw std::runtime_error(
              "index auto pins must be index lists or known descriptor fields");
        n.pins[f] = pin;
      }
    }
    plan.nodes.push_back(std::move(n));
    return plan.nodes.size() - 1;
  }
  std::size_t source(const Pattern& p) {
    if (const auto* c = std::get_if<CapturePattern>(&p.value))
      return capture(c->capture, p.origin);
    if (const auto* b = std::get_if<BindPattern>(&p.value))
      return source(*b->expression);
    if (const auto* op = std::get_if<MatchOperation>(&p.value)) {
      std::vector<std::size_t> inputs;
      for (const auto& input : op->operands) inputs.push_back(source(*input));
      return operation(op->operation, op->descriptor, std::move(inputs),
                       p.origin);
    }
    throw std::runtime_error(
        "semantic graphs support tensor captures and operation "
        "applications");
  }
  std::size_t target(const BuildExpr& p) {
    if (const auto* c = std::get_if<CaptureRef>(&p.value))
      return capture(c->capture, p.origin);
    if (const auto* op = std::get_if<BuildOperation>(&p.value)) {
      std::vector<std::size_t> inputs;
      for (const auto& input : op->operands) inputs.push_back(target(*input));
      return operation(op->operation, op->descriptor, std::move(inputs),
                       p.origin);
    }
    throw std::runtime_error(
        "semantic graphs support captures and operation applications");
  }
};
}  // namespace
bool compile(detail::AnalysisContext& context, Rule& rule) {
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
    b.assumptions();
    const bool selected = context.options.semantic_rule.empty() ||
                          context.options.semantic_rule == rule.name;
    if (selected) b.plan.proof = synthesize(b.plan);
    if (selected && !context.options.inspect_semantic_rules &&
        b.plan.proof.status != ProofStatus::Proved)
      context.report(rule.origin,
                     "cannot verify rule '" + rule.name +
                         "': " + b.plan.proof.reason +
                         ". Add an explicit where condition establishing the "
                         "unresolved requirement, or change the rewrite.");
    rule.semantic_plan = std::make_shared<Plan>(std::move(b.plan));
  } catch (const std::runtime_error& e) {
    context.report(rule.origin, std::string("semantic inference: ") + e.what());
  }
  return true;
}
}  // namespace tepl::core::index_semantics
