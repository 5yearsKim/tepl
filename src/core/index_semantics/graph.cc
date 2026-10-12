#include "src/core/index_semantics/graph.h"

#include <functional>
#include <stdexcept>
#include <type_traits>
namespace tepl::core::index_semantics {
S Lowerer::metadata(const Meta& m, const Instance& n,
                    const std::vector<TensorMeaning>& inputs,
                    const std::vector<S>& groups) {
  context.tick();
  if (m.kind == Meta::Kind::Integer) return make(Symbol::Kind::Integer, m.name);
  if (m.kind == Meta::Kind::Boolean) return make(Symbol::Kind::Boolean, m.name);
  if (m.kind == Meta::Kind::Attribute)
    return context.canonical(context.field(n, m.index));
  if (m.kind == Meta::Kind::Operand) return inputs.at(m.index).shape;
  if (m.kind == Meta::Kind::Group) return groups.at(m.index);
  std::vector<S> args;
  for (const auto& a : m.args) args.push_back(metadata(*a, n, inputs, groups));
  if (m.kind == Meta::Kind::List)
    return make(Symbol::Kind::List, {}, 0, 0, std::move(args));
  if (m.name == "shape") return args[0];
  if (m.name == "rank") return call("len", args);
  if (m.name == "all_axes") return call("range", {call("len", args)});
  return context.canonical(
      call(m.name == "permute" ? "gather" : m.name, std::move(args)));
}
S Lowerer::dtype(const Instance& n, const std::vector<TensorMeaning>& inputs) {
  if (!n.dtype) throw std::runtime_error("missing symbolic dtype contract");
  std::map<std::size_t, S> symbols;
  for (const auto& p : n.dtype->parameters) {
    if (p.variadic)
      throw std::runtime_error("variadic dtype parameter unsupported");
    symbols[p.symbol.value] = inputs.at(p.operand).dtype;
  }
  std::function<S(const metadata::Expr&)> eval =
      [&](const metadata::Expr& e) -> S {
    context.tick();
    return std::visit(
        [&](const auto& v) -> S {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, metadata::SymbolRef>)
            return symbols.at(v.symbol.value);
          else if constexpr (std::is_same_v<T, metadata::Integer>)
            return make(Symbol::Kind::Integer, v.spelling);
          else if constexpr (std::is_same_v<T, bool>)
            return boolean(v);
          else if constexpr (std::is_same_v<T, core::DType>)
            return call("dtype_" + std::string(dtypeName(v)), {});
          else if constexpr (std::is_same_v<T, metadata::AttributeRef>)
            return context.field(n, v.field);
          else if constexpr (std::is_same_v<T, metadata::Call>) {
            std::vector<S> args;
            for (const auto& a : v.arguments) args.push_back(eval(*a));
            return call(std::string(builtins::builtinName(v.builtin)),
                        std::move(args));
          } else if constexpr (std::is_same_v<T, metadata::Binary>)
            return call(std::string(spelling(v.op)),
                        {eval(*v.lhs), eval(*v.rhs)});
          else if constexpr (std::is_same_v<T, metadata::Unary>)
            return call(std::string(spelling(v.op)), {eval(*v.operand)});
          else if constexpr (std::is_same_v<T, metadata::List>) {
            std::vector<S> args;
            for (const auto& a : v.elements) args.push_back(eval(*a));
            return make(Symbol::Kind::List, {}, 0, 0, std::move(args));
          } else if constexpr (std::is_same_v<T, metadata::Index>)
            return call("index", {eval(*v.value), eval(*v.index)});
          else if constexpr (std::is_same_v<T, metadata::Conditional>)
            return call("if", {eval(*v.condition), eval(*v.then_value),
                               eval(*v.else_value)});
          else
            throw std::runtime_error(
                "dtype comprehension is outside symbolic proof fragment");
        },
        e.value);
  };
  for (const auto& s : n.dtype->statements) {
    if (const auto* l = std::get_if<metadata::Let>(&s.value))
      symbols[l->symbol.value] = eval(*l->value);
    else
      obligations.push_back(
          eval(*std::get<metadata::Assert>(s.value).condition));
  }
  return eval(*n.dtype->result.value);
}

E expression(Expr::Kind k, std::vector<E> args, S axes) {
  return std::make_shared<Expr>(
      Expr{k, 0, 0, std::move(axes), std::move(args)});
}
S freeAxes(const E& e) {
  if (e->kind == Expr::Kind::Read) return e->axes;
  auto a = freeAxes(e->args[0]);
  if (e->kind == Expr::Kind::Reduce)
    return call("difference_axes", {a, e->axes});
  return call("union_axes", {a, freeAxes(e->args[1])});
}
std::string show(const E& e) {
  if (e->kind == Expr::Kind::Read)
    return "Read(#" + std::to_string(e->capture) + ", access@" +
           std::to_string(e->occurrence) + ")";
  std::string out = e->kind == Expr::Kind::Add       ? "Add("
                    : e->kind == Expr::Kind::Product ? "Product("
                                                     : "Reduce(";
  if (e->kind == Expr::Kind::Reduce) out += format(*e->axes) + ", ";
  for (std::size_t i = 0; i < e->args.size(); ++i) {
    if (i) out += ", ";
    out += show(e->args[i]);
  }
  return out + ")";
}
E Lowerer::rename(const E& e, const S& from, const S& to) {
  context.tick();
  if (e->kind == Expr::Kind::Read) {
    auto axes = context.replace(e->axes, from, to);
    auto relations = context.wiring;
    for (const auto& w : relations)
      if (equal(w.interface, e->axes))
        context.bind(axes, context.replace(w.group, from, to), w.positions);
    auto cover = context.covers.find(key(e->axes));
    if (cover != context.covers.end()) {
      auto transported = context.replace(cover->second, from, to);
      if (!equal(transported, axes) || !context.covers.contains(key(axes)))
        context.covers[key(axes)] = transported;
    }
    // These are consequences of legal source coordinate composition.
    obligations.push_back(call("distinct_axes", {axes}));
    obligations.push_back(call(
        "==",
        {context.extents(axes), make(Symbol::Kind::Shape, {}, e->capture)}));
    return std::make_shared<Expr>(
        Expr{e->kind, e->capture, e->occurrence, axes, {}});
  }
  std::vector<E> args;
  for (const auto& a : e->args) args.push_back(rename(a, from, to));
  return expression(e->kind, std::move(args), e->axes);
}
E Lowerer::value(const Value& v, const std::vector<TensorMeaning>& inputs,
                 const std::vector<S>& groups) {
  context.tick();
  if (v.kind == Value::Kind::Operand) return inputs.at(v.index).value;
  std::vector<E> args;
  for (const auto& a : v.args) args.push_back(value(*a, inputs, groups));
  auto k = v.kind == Value::Kind::Add       ? Expr::Kind::Add
           : v.kind == Value::Kind::Product ? Expr::Kind::Product
                                            : Expr::Kind::Reduce;
  return expression(k, std::move(args),
                    v.kind == Value::Kind::Reduce ? groups.at(v.index) : S{});
}
TensorMeaning Lowerer::operation(std::size_t id,
                                 std::vector<TensorMeaning> inputs) {
  context.tick();
  const auto obligation_begin = obligations.size();
  if (!anchors) {
    for (const auto& in : inputs) context.input_shapes[id].push_back(in.shape);
  }
  const auto& n = context.plan.nodes.at(id);
  std::vector<S> groups, domains;
  std::vector<std::vector<S>> positions(inputs.size()),
      coordinates(inputs.size());
  for (std::size_t g = 0; g < n.definition->groups.size(); ++g) {
    S domain, axes;
    for (const auto& b : n.definition->groups[g].bindings) {
      auto p = metadata(*b.positions, n, inputs, groups);
      auto d = call("gather", {inputs[b.operand].shape, p});
      obligations.push_back(call("is_valid_axis_list",
                                 {p, call("len", {inputs[b.operand].shape})}));
      if (domain)
        obligations.push_back(call("==", {domain, d}));
      else
        domain = d;
      if (anchors) {
        auto selection = context.select(inputs[b.operand].axes, p);
        if (axes)
          obligations.push_back(call("==", {axes, selection}));
        else
          axes = selection;
      }
      positions[b.operand].push_back(p);
    }
    if (!domain)
      throw std::runtime_error("logical group has no operand domain");
    if (!anchors)
      axes = make(Symbol::Kind::AxisGroup,
                  "n" + std::to_string(id) + "." + n.definition->groups[g].name,
                  id, g, {domain});
    if (!anchors) context.coordinate_groups[{id, g}] = axes;
    for (const auto& b : n.definition->groups[g].bindings)
      coordinates[b.operand].push_back(axes);
    groups.push_back(axes);
    domains.push_back(domain);
  }
  if (!anchors)
    for (std::size_t i = obligation_begin; i < obligations.size(); ++i)
      context.learn(obligations[i]);
  for (std::size_t o = 0; o < inputs.size(); ++o) {
    auto ps = call("concat", positions[o]), cs = call("concat", coordinates[o]);
    obligations.push_back(
        call("axis_partition", {ps, call("len", {inputs[o].shape})}));
    if (!anchors) {
      auto layout = call("gather", {cs, call("inverse", {ps})});
      for (std::size_t g = 0; g < n.definition->groups.size(); ++g)
        for (const auto& b : n.definition->groups[g].bindings)
          if (b.operand == o)
            context.bind(layout, groups[g],
                         metadata(*b.positions, n, inputs, groups));
      context.covers[key(layout)] = cs;
      inputs[o].value = rename(inputs[o].value, inputs[o].axes, layout);
      inputs[o].axes = layout;
    }
  }
  for (const auto& a : n.definition->assertions)
    obligations.push_back(metadata(*a, n, inputs, groups));
  std::function<void(const Meta&)> outputChecks = [&](const Meta& m) {
    if (m.kind != Meta::Kind::Call) return;
    if (m.name == "permute") {
      auto p = metadata(*m.args[1], n, inputs, groups),
           size = call("len", {metadata(*m.args[0], n, inputs, groups)});
      obligations.push_back(call("is_valid_axis_list", {p, size}));
      obligations.push_back(call("==", {call("len", {p}), size}));
    }
    for (const auto& a : m.args) outputChecks(*a);
  };
  outputChecks(*n.definition->output);
  auto axes = metadata(*n.definition->output, n, inputs, groups);
  auto shape = metadata(*n.definition->output, n, inputs, domains);
  auto dt = dtype(n, inputs);
  auto body = value(*n.definition->value, inputs, groups);
  obligations.push_back(call("distinct_axes", {axes}));
  if (anchors) {
    obligations.push_back(call("same_set", {axes, freeAxes(body)}));
    obligations.push_back(call("==", {context.extents(axes), shape}));
  } else {
    for (std::size_t i = obligation_begin; i < obligations.size(); ++i)
      context.learn(obligations[i]);
    for (const auto& group : groups) {
      auto p = context.positions(axes, group);
      if (!callIs(p, "positions")) context.bind(axes, group, p);
    }
    context.learn(call("==", {context.extents(axes), shape}));
  }
  if (!anchors) {
    context.tensor_axes[id] = axes;
    context.tensor_groups[id] = groups;
  }
  return {axes, shape, dt, body};
}
TensorMeaning Lowerer::lower(std::size_t id) {
  context.tick();
  const auto& n = context.plan.nodes.at(id);
  if (n.capture) {
    auto shape = make(Symbol::Kind::Shape, {}, n.capture->value);
    auto axes = anchors ? anchors->at(id)
                        : make(Symbol::Kind::AxisGroup,
                               "input" + std::to_string(id), id, 0, {shape});
    if (!anchors) context.coordinate_groups[{id, 0}] = axes;
    return {axes, shape, make(Symbol::Kind::DType, {}, n.capture->value),
            std::make_shared<Expr>(
                Expr{Expr::Kind::Read, n.capture->value, id, axes, {}})};
  }
  std::vector<TensorMeaning> inputs;
  for (auto child : n.inputs) inputs.push_back(lower(child));
  return operation(id, std::move(inputs));
}
E arithmeticSkeleton(const Plan& p, std::size_t id) {
  const auto& n = p.nodes.at(id);
  if (n.capture)
    return std::make_shared<Expr>(Expr{
        Expr::Kind::Read, n.capture->value, id, call("empty_axes", {}), {}});
  std::vector<E> inputs;
  for (auto child : n.inputs) inputs.push_back(arithmeticSkeleton(p, child));
  std::function<E(const Value&)> lower = [&](const Value& v) -> E {
    if (v.kind == Value::Kind::Operand) return inputs.at(v.index);
    std::vector<E> args;
    for (const auto& a : v.args) args.push_back(lower(*a));
    return expression(
        v.kind == Value::Kind::Add       ? Expr::Kind::Add
        : v.kind == Value::Kind::Product ? Expr::Kind::Product
                                         : Expr::Kind::Reduce,
        std::move(args),
        v.kind == Value::Kind::Reduce ? call("empty_axes", {}) : S{});
  };
  return lower(*n.definition->value);
}
}  // namespace tepl::core::index_semantics
