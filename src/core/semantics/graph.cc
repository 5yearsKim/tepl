#include "src/core/semantics/graph.h"

#include <charconv>
#include <functional>

#include "src/core/semantics/symbolic/tree.h"

namespace tepl::core::semantics {

E expression(Expr::Kind k, std::vector<E> args, S axes) {
  return std::make_shared<Expr>(
      Expr{k, 0, 0, std::move(axes), std::move(args)});
}
E constant(std::int64_t n) {
  return std::make_shared<Expr>(Expr{Expr::Kind::Const, 0, 0, {}, {}, n});
}
E scalarCall(ScalarFunction function, std::vector<E> args,
             std::vector<S> parameters, S dtype) {
  Expr value{Expr::Kind::Call, 0, 0, {}, std::move(args)};
  value.function = function;
  value.parameters = std::move(parameters);
  value.dtype = std::move(dtype);
  return std::make_shared<Expr>(std::move(value));
}
E number(std::string text, S dtype) {
  auto typedInteger = [&](std::int64_t n) -> E {
    auto value = std::make_shared<Expr>(*constant(n));
    value->dtype = dtype;
    return value;
  };
  if (text.starts_with("+")) text.erase(0, 1);
  std::int64_t integer;
  auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), integer);
  if (ec == std::errc{} && end == text.data() + text.size())
    return typedInteger(integer);
  // Canonical decimal digits and exponent; no host floating-point rounding.
  bool negative = text.starts_with("-");
  if (negative) text.erase(0, 1);
  std::int64_t exponent = 0;
  if (auto e = text.find_first_of("eE"); e != std::string::npos) {
    auto spelling = text.substr(e + 1);
    if (spelling.starts_with("+")) spelling.erase(0, 1);
    auto [last, error] = std::from_chars(
        spelling.data(), spelling.data() + spelling.size(), exponent);
    if (error != std::errc{} || last != spelling.data() + spelling.size())
      throw ProofFailure(
          "numeric literal exponent is outside the supported range");
    text.resize(e);
  }
  if (auto dot = text.find('.'); dot != std::string::npos) {
    auto places = static_cast<std::int64_t>(text.size() - dot - 1);
    if (__builtin_sub_overflow(exponent, places, &exponent))
      throw ProofFailure("numeric literal exponent overflow");
    text.erase(dot, 1);
  }
  auto first = text.find_first_not_of('0');
  if (first == std::string::npos) return typedInteger(0);
  text.erase(0, first);
  while (text.size() > 1 && text.back() == '0') {
    text.pop_back();
    if (__builtin_add_overflow(exponent, std::int64_t{1}, &exponent))
      throw ProofFailure("numeric literal exponent overflow");
  }
  // Integral decimal spellings participate in existing coefficient arithmetic.
  if (exponent >= 0 && exponent <= 18 && text.size() + exponent <= 19) {
    auto digits = (negative ? "-" : "") + text + std::string(exponent, '0');
    auto [last, error] =
        std::from_chars(digits.data(), digits.data() + digits.size(), integer);
    if (error == std::errc{} && last == digits.data() + digits.size())
      return typedInteger(integer);
  }
  Expr value{Expr::Kind::Number};
  value.number = (negative ? "-" : "") + text + "e" + std::to_string(exponent);
  value.dtype = dtype;
  return std::make_shared<Expr>(std::move(value));
}
std::vector<E> tensorReads(const E& value) {
  std::vector<E> out;
  walkTree(value, [&](const E& e) {
    if (e->kind == Expr::Kind::Read) out.push_back(e);
  });
  return out;
}
S freeAxes(const E& e) {
  if (e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Number)
    return call("empty_axes", {});
  if (e->kind == Expr::Kind::Read)
    return e->access ? indexDependencies(e->access) : e->axes;
  if (e->kind == Expr::Kind::Reduce)
    return call("difference_axes", {freeAxes(e->args[0]), e->axes});
  std::vector<S> axes;
  for (const auto& arg : e->args) axes.push_back(freeAxes(arg));
  return call("union_axes", std::move(axes));
}
std::string show(const E& e) {
  if (e->kind == Expr::Kind::Number) return e->number;
  if (e->kind == Expr::Kind::Const)
    return "Const(" + std::to_string(e->constant) + ")";
  if (e->kind == Expr::Kind::Read)
    return "Read(#" + std::to_string(e->capture) + ", access@" +
           std::to_string(e->occurrence) +
           (e->access ? ", " + showIndex(e->access) : "") + ")";
  std::string out = e->kind == Expr::Kind::Add       ? "Add("
                    : e->kind == Expr::Kind::Product ? "Product("
                                                     : "Reduce(";
  if (e->kind == Expr::Kind::Call)
    out = std::string(scalarSignature(e->function).name) + "(";
  if (e->kind == Expr::Kind::Reduce) out += format(*e->axes) + ", ";
  for (std::size_t i = 0; i < e->args.size(); ++i) {
    if (i) out += ", ";
    out += show(e->args[i]);
  }
  for (const auto& p : e->parameters) out += ", parameter=" + format(*p);
  return out + ")";
}
std::map<std::size_t, S> Lowerer::outputPositions(
    const Meta& output, const Instance& n,
    const std::vector<TensorMeaning>& inputs, const std::vector<S>& groups) {
  using OutputMaps = std::pair<S, std::map<std::size_t, S>>;
  std::function<OutputMaps(const Meta&)> outputMaps = [&](const Meta& m) {
    if (m.kind == Meta::Kind::Group) {
      auto size = call("len", {groups[m.index]});
      return OutputMaps{size, {{m.index, call("range", {size})}}};
    }
    if (m.name == "place") {
      auto left = outputMaps(*m.args[0]), right = outputMaps(*m.args[2]);
      auto p = metadata(*m.args[1], n, inputs, groups);
      auto rest =
          call("ordered_complement",
               {call("range", {call("+", {left.first, right.first})}), p});
      for (auto& [g, positions] : left.second)
        positions = equal(positions, call("range", {left.first}))
                        ? p
                        : call("gather", {p, positions});
      for (auto& [g, positions] : right.second)
        positions = equal(positions, call("range", {right.first}))
                        ? rest
                        : call("gather", {rest, positions});
      left.second.insert(right.second.begin(), right.second.end());
      left.first = call("+", {left.first, right.first});
      return left;
    }
    if (m.name == "permute") {
      auto result = outputMaps(*m.args[0]);
      auto p = metadata(*m.args[1], n, inputs, groups);
      for (auto& [g, positions] : result.second)
        positions = call("gather", {call("inverse", {p}), positions});
      return result;
    }
    OutputMaps result{integer(0), {}};
    for (const auto& a : m.args) {
      auto part = outputMaps(*a);
      for (const auto& [g, positions] : part.second)
        result.second[g] = call("offset_positions", {positions, result.first});
      result.first = call("+", {result.first, part.first});
    }
    return result;
  };
  return outputMaps(output).second;
}
TensorMeaning Lowerer::operation(std::size_t id,
                                 std::vector<TensorMeaning> inputs) {
  context.tick();
  const auto obligation_begin = obligations.size();
  if (!anchors) {
    std::vector<S> shapes;
    for (const auto& in : inputs) shapes.push_back(in.shape);
    context.recordInputs(id, std::move(shapes));
  }
  const auto& n = context.plan.nodes.at(id);
  std::set<std::size_t> explicit_reads, implicit_reads;
  walkTree(n.definition->value, [&](const ValuePtr& v) {
    if (v->kind == Value::Kind::Read) explicit_reads.insert(v->index);
    if (v->kind == Value::Kind::Operand) implicit_reads.insert(v->index);
  });
  for (auto o : explicit_reads)
    if (implicit_reads.contains(o))
      throw ProofFailure("an operand cannot mix implicit and calculated reads");
  std::vector<S> groups, domains;
  std::vector<std::vector<S>> positions(inputs.size()),
      coordinates(inputs.size());
  for (std::size_t g = 0; g < n.definition->groups.size(); ++g) {
    S domain, axes;
    if (n.definition->groups[g].domain) {
      domain = metadata(*n.definition->groups[g].domain, n, inputs, groups);
      obligations.push_back(call("is_valid_extents", {domain}));
    }
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
        context.selectionFacts(inputs[b.operand].axes, p, selection);
        if (axes)
          obligations.push_back(call("==", {axes, selection}));
        else
          axes = selection;
      }
      positions[b.operand].push_back(p);
    }
    if (!domain) throw ProofFailure("logical group has no operand domain");
    if (anchors && !axes)
      axes = make(Symbol::Kind::AxisGroup, "target-domain", id, g, {domain});
    if (!anchors) {
      // A consumer selects existing producer coordinates. Keep that view,
      // including partial selections crossing group boundaries, rather than
      // creating a new family and replacing the producer's entire frame.
      bool producer_view = false;
      for (const auto& b : n.definition->groups[g].bindings)
        if (const auto& producer = context.plan.nodes[n.inputs[b.operand]];
            producer.definition &&
            inputs[b.operand].axes->kind != Symbol::Kind::AxisGroup) {
          auto p = metadata(*b.positions, n, inputs, groups);
          axes = context.select(inputs[b.operand].axes, p);
          producer_view = true;
          break;
        }
      // A pure reindexing keeps the producer coordinate identities. Only its
      // ordered output interface changes; there is no new arithmetic binder.
      const auto& bindings = n.definition->groups[g].bindings;
      bool identity_value = n.definition->value->kind == Value::Kind::Operand &&
                            n.definition->value->index == 0 &&
                            inputs.size() == 1 &&
                            n.definition->groups.size() == 1 &&
                            bindings.size() == 1 && bindings[0].operand == 0 &&
                            bindings[0].positions->kind == Meta::Kind::Call &&
                            bindings[0].positions->name == "all_axes";
      axes = producer_view    ? axes
             : identity_value ? inputs[0].axes
                              : make(Symbol::Kind::AxisGroup,
                                     "n" + std::to_string(id) + "." +
                                         n.definition->groups[g].name,
                                     id, g, {domain});
    }
    if (!anchors) context.registerCoordinates(id, g, axes);
    for (const auto& b : n.definition->groups[g].bindings)
      coordinates[b.operand].push_back(axes);
    groups.push_back(axes);
    domains.push_back(domain);
  }
  bool temporary_domain = false;
  if (anchors) {
    bool needs_interface = false;
    for (const auto& g : n.definition->groups)
      needs_interface |= bool(g.domain);
    if (needs_interface) {
      if (!interfaces || !interfaces->contains(id))
        throw ProofFailure(
            "output-only axes require an established output interface");
      auto maps = outputPositions(*n.definition->output, n, inputs, groups);
      for (std::size_t g = 0; g < groups.size(); ++g)
        if (n.definition->groups[g].domain) {
          if (!maps.contains(g))
            throw ProofFailure("output-only group is absent from output");
          auto projected = context.select(interfaces->at(id), maps.at(g));
          // A computed-read producer can have coordinates eliminated by its
          // consumer. Its local domain must not be confused with the final
          // output domain. Fresh local coordinates disappear by substitution.
          if (!explicit_reads.empty() && id != context.plan.rhs &&
              !context.entails(
                  call("==", {context.extents(projected), domains[g]}))) {
            temporary_domain = true;
            auto same_length = call(
                "==", {call("len", {groups[g]}), call("len", {projected})});
            context.require(same_length);
            // Cache an established fact, never a new premise.
            context.learn(same_length);
          } else {
            groups[g] = projected;
            obligations.push_back(
                call("==", {context.extents(groups[g]), domains[g]}));
          }
        }
    }
  }
  if (!anchors)
    for (std::size_t i = obligation_begin; i < obligations.size(); ++i)
      context.learn(obligations[i]);
  for (std::size_t o = 0; o < inputs.size(); ++o) {
    if (explicit_reads.contains(o)) {
      auto bound = call("concat", positions[o]);
      auto rest = context.canonical(
          call("ordered_complement",
               {call("range", {call("len", {inputs[o].shape})}), bound}));
      auto remaining = context.select(inputs[o].axes, rest);
      context.selectionFacts(inputs[o].axes, rest, remaining);
      positions[o].push_back(rest);
      coordinates[o].push_back(remaining);
    }
    auto ps = call("concat", positions[o]), cs = call("concat", coordinates[o]);
    obligations.push_back(
        call("axis_partition", {ps, call("len", {inputs[o].shape})}));
    if (!anchors) {
      context.learn(obligations.back());
      S offset = integer(0);
      for (const auto& coordinate : coordinates[o]) {
        auto size = call("len", {coordinate});
        context.bind(cs, coordinate,
                     context.canonical(call("offset_positions",
                                            {call("range", {size}), offset})));
        offset = call("+", {offset, size});
      }
      auto layout = call("gather", {cs, call("inverse", {ps})});
      bool existing_view = true;
      for (std::size_t g = 0; g < coordinates[o].size(); ++g)
        existing_view &=
            explicit_reads.contains(o)
                ? context.entails(call(
                      "==", {context.select(inputs[o].axes, positions[o][g]),
                             coordinates[o][g]}))
                : equal(context.select(inputs[o].axes, positions[o][g]),
                        coordinates[o][g]);
      if (existing_view) layout = inputs[o].axes;
      context.partition(layout, coordinates[o], positions[o]);
      for (std::size_t g = 0; g < n.definition->groups.size(); ++g)
        for (const auto& b : n.definition->groups[g].bindings)
          if (b.operand == o)
            context.bind(layout, groups[g],
                         metadata(*b.positions, n, inputs, groups));
      context.cover(layout, cs);
      context.transportFrame(inputs[o].axes, layout);
      inputs[o].value = rename(inputs[o].value, inputs[o].axes, layout);
      std::map<std::size_t, S> accesses;
      for (const auto& read : tensorReads(inputs[o].value))
        accesses[read->occurrence] = read->axes;
      for (auto& [occurrence, axes] : inputs[o].occurrence_axes)
        axes = accesses.contains(occurrence)
                   ? accesses.at(occurrence)
                   : context.transportInterface(inputs[o].axes, layout, axes);
      inputs[o].axes = layout;
      inputs[o].interface = context.tensorInterface(layout);
    }
  }
  for (const auto& a : n.definition->assertions)
    obligations.push_back(metadata(*a, n, inputs, groups));
  std::function<void(const Meta&)> outputChecks = [&](const Meta& m) {
    if (m.kind != Meta::Kind::Call) return;
    if (m.name == "place") {
      auto left = metadata(*m.args[0], n, inputs, groups),
           right = metadata(*m.args[2], n, inputs, groups);
      auto p = metadata(*m.args[1], n, inputs, groups);
      obligations.push_back(
          call("is_valid_axis_list",
               {p, call("+", {call("len", {left}), call("len", {right})})}));
      obligations.push_back(
          call("==", {call("len", {p}), call("len", {left})}));
    }
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
  S peer;
  for (const auto& in : inputs)
    if (in.dtype) {
      peer = in.dtype;
      break;
    }
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (!inputs[i].dtype) {
      auto inferred =
          peer ? peer
               : (expected_dtype ? expected_dtype : call("dtype_i64", {}));
      if (n.definition->value->kind == Value::Kind::Call) {
        auto types = scalarSignature(n.definition->value->function).types;
        if (types == ScalarTypes::Boolean ||
            (types == ScalarTypes::Select && i == 0))
          inferred = call("dtype_bool", {});
      }
      inputs[i].dtype = inferred;
      auto literal = std::make_shared<Expr>(*inputs[i].value);
      literal->dtype = inferred;
      inputs[i].value = literal;
    }
  auto dtype_begin = obligations.size();
  auto dt = dtype(n, inputs);
  if (!anchors)
    for (std::size_t i = dtype_begin; i < obligations.size(); ++i)
      context.learn(obligations[i]);
  auto body = value(*n.definition->value, n, inputs, groups);
  if (body->dtype) context.require(call("==", {body->dtype, dt}));
  auto typed_body = std::make_shared<Expr>(*body);
  typed_body->dtype = dt;
  body = typed_body;
  obligations.push_back(call("distinct_axes", {axes}));
  bool independent_output = false;
  walkTree(body, [&](const E& e) {
    independent_output |=
        e->kind == Expr::Kind::Const || e->kind == Expr::Kind::Number;
  });
  for (const auto& g : n.definition->groups)
    independent_output |= bool(g.domain);
  for (const auto& input : inputs)
    independent_output |= input.may_have_independent_axes;
  if (anchors) {
    obligations.push_back(independent_output
                              ? call("subset", {freeAxes(body), axes})
                              : call("same_set", {axes, freeAxes(body)}));
    obligations.push_back(call("==", {context.extents(axes), shape}));
    // Publish geometry only after the current node's legality is established.
    // Strict verification cannot introduce a premise while doing this.
    for (std::size_t i = obligation_begin; i < obligations.size(); ++i)
      context.require(obligations[i]);
    for (std::size_t o = 0; o < inputs.size(); ++o)
      context.partition(inputs[o].axes, coordinates[o], positions[o], false);
  } else {
    for (std::size_t i = obligation_begin; i < obligations.size(); ++i)
      context.learn(obligations[i]);
    for (const auto& group : groups) {
      auto p = context.positions(axes, group);
      if (!callIs(p, "positions")) context.bind(axes, group, p);
    }
    context.learn(call("==", {context.extents(axes), shape}));
  }
  auto output_mapping =
      outputPositions(*n.definition->output, n, inputs, groups);
  std::vector<S> free_groups, output_positions;
  for (std::size_t g = 0; g < groups.size(); ++g)
    if (output_mapping.contains(g)) {
      free_groups.push_back(groups[g]);
      output_positions.push_back(context.canonical(output_mapping.at(g)));
    }
  context.partition(axes, free_groups, output_positions, !anchors);
  if (anchors && interfaces && interfaces->contains(id)) {
    bool output_only = false;
    for (const auto& group : n.definition->groups)
      output_only |= bool(group.domain);
    if (output_only && !temporary_domain) {
      auto required = interfaces->at(id);
      context.require(call("==", {axes, required}), GuardKind::CoordinateOrder);
      context.bind(axes, required, call("range", {call("len", {required})}));
      context.bind(required, axes, call("range", {call("len", {axes})}));
    }
  }
  if (!anchors) {
    context.recordTensor(id, axes, groups);
  }
  context.recordShape(axes, shape);
  std::map<std::size_t, S> occurrence_axes;
  for (const auto& input : inputs)
    occurrence_axes.insert(input.occurrence_axes.begin(),
                           input.occurrence_axes.end());
  return {axes,
          shape,
          dt,
          body,
          context.tensorInterface(axes),
          std::move(occurrence_axes),
          independent_output};
}
TensorMeaning Lowerer::lower(std::size_t id) {
  context.tick();
  const auto& n = context.plan.nodes.at(id);
  if (n.literal) {
    auto axes = call("empty_axes", {}), shape = make(Symbol::Kind::List);
    S dt = n.literal_dtype
               ? call("dtype_" + std::string(dtypeName(*n.literal_dtype)), {})
               : S{};
    if (!dt && id == context.plan.rhs) dt = expected_dtype;
    return {
        axes, shape, dt, number(*n.literal, dt), context.tensorInterface(axes),
        {},   true};
  }

  if (n.capture) {
    auto shape = make(Symbol::Kind::Shape, {}, n.capture->value);
    auto axes = anchors ? anchors->at(id)
                        : make(Symbol::Kind::AxisGroup,
                               "input" + std::to_string(id), id, 0, {shape});
    if (!anchors) context.registerCoordinates(id, 0, axes);
    return {axes,
            shape,
            make(Symbol::Kind::DType, {}, n.capture->value),
            std::make_shared<Expr>(
                Expr{Expr::Kind::Read, n.capture->value, id, axes, {}}),
            context.tensorInterface(axes),
            {{id, axes}}};
  }
  std::vector<TensorMeaning> inputs;
  for (auto child : n.inputs) inputs.push_back(lower(child));
  return operation(id, std::move(inputs));
}
TensorMeaning lowerSource(Context& c) {
  Lowerer source{c};
  source.lower(c.plan.lhs);
  Lowerer composed{c};
  auto lhs = composed.lower(c.plan.lhs);
  c.enableSetReasoning();
  return lhs;
}

E arithmeticSkeleton(Context& c, std::size_t id) {
  c.tick();
  const auto& n = c.plan.nodes.at(id);
  if (n.literal) return number(*n.literal);
  if (n.capture)
    return std::make_shared<Expr>(Expr{
        Expr::Kind::Read, n.capture->value, id, call("empty_axes", {}), {}});
  std::vector<E> inputs;
  for (auto child : n.inputs) inputs.push_back(arithmeticSkeleton(c, child));
  std::function<E(const Value&)> lower = [&](const Value& v) -> E {
    if (v.kind == Value::Kind::Const) return constant(v.constant);
    if (v.kind == Value::Kind::Operand || v.kind == Value::Kind::Read)
      return inputs.at(v.index);
    std::vector<E> args;
    for (const auto& a : v.args) args.push_back(lower(*a));
    if (v.kind == Value::Kind::Call) {
      Lowerer lowerer{c};
      std::vector<S> parameters;
      for (const auto& m : v.parameters)
        parameters.push_back(lowerer.metadata(*m, n, {}, {}));
      return scalarCall(v.function, std::move(args), std::move(parameters));
    }
    return expression(
        v.kind == Value::Kind::Add       ? Expr::Kind::Add
        : v.kind == Value::Kind::Product ? Expr::Kind::Product
                                         : Expr::Kind::Reduce,
        std::move(args),
        v.kind == Value::Kind::Reduce ? call("empty_axes", {}) : S{});
  };
  return lower(*n.definition->value);
}
}  // namespace tepl::core::semantics
