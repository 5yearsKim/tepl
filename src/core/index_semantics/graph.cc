#include "src/core/index_semantics/graph.h"

#include <functional>
#include <stdexcept>
#include <type_traits>

#include "src/core/builtins/axes.h"
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
  if (m.name == "extent_add" || m.name == "extent_sub" ||
      m.name == "extent_le" || m.name == "extent_mul" ||
      m.name == "extent_floor_div")
    obligations.push_back(
        call("==", {call("len", {args[0]}), call("len", {args[1]})}));
  if (m.name == "extent_floor_div")
    obligations.push_back(call("is_positive_extents", {args[1]}));
  const auto* entry = builtins::resolveAxisBuiltin(m.name);
  if (!entry)
    return context.canonical(call(m.name, std::move(args)));  // Operator.
  return std::visit(
      [&](auto target) -> S {
        using T = decltype(target);
        if constexpr (std::is_same_v<T, builtins::Builtin>) {
          // Axis complements retain an explicit symbolic operation for proof
          // laws.
          if (target == builtins::Builtin::kExclude)
            return context.canonical(
                call("ordered_complement", std::move(args)));
          return context.canonical(call(
              std::string(builtins::builtinName(target)), std::move(args)));
        } else if constexpr (std::is_same_v<T, intrinsics::Intrinsic>) {
          if (target == intrinsics::Intrinsic::kPlaceAxes &&
              args[0]->sort != Symbol::Sort::Axes) {
            auto base = call("concat", {args[0], args[2]});
            auto rest = call("ordered_complement",
                             {call("range", {call("len", {base})}), args[1]});
            return context.canonical(call(
                "gather",
                {base, call("inverse", {call("concat", {args[1], rest})})}));
          }
          return context.canonical(
              call(std::string(intrinsics::signature(target).name),
                   std::move(args)));
        } else {
          if (target == builtins::AxisQuery::Shape) return args[0];
          if (target == builtins::AxisQuery::Rank) return call("len", args);
          return call("range", {call("len", args)});
        }
      },
      entry->target);
}
S Lowerer::dtype(const Instance& n, const std::vector<TensorMeaning>& inputs) {
  if (!n.dtype) throw ProofFailure("missing symbolic dtype contract");
  std::map<std::size_t, S> symbols;
  for (const auto& p : n.dtype->parameters) {
    if (p.variadic) throw ProofFailure("variadic dtype parameter unsupported");
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
            throw ProofFailure(
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
E constant(std::int64_t n) {
  return std::make_shared<Expr>(Expr{Expr::Kind::Const, 0, 0, {}, {}, n});
}
S freeAxes(const E& e) {
  if (e->kind == Expr::Kind::Const) return call("empty_axes", {});
  if (e->kind == Expr::Kind::Read)
    return e->access ? indexDependencies(e->access) : e->axes;
  auto a = freeAxes(e->args[0]);
  if (e->kind == Expr::Kind::Reduce)
    return call("difference_axes", {a, e->axes});
  return call("union_axes", {a, freeAxes(e->args[1])});
}
std::string show(const E& e) {
  if (e->kind == Expr::Kind::Const)
    return "Const(" + std::to_string(e->constant) + ")";
  if (e->kind == Expr::Kind::Read)
    return "Read(#" + std::to_string(e->capture) + ", access@" +
           std::to_string(e->occurrence) +
           (e->access ? ", " + showIndex(e->access) : "") + ")";
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
  if (e->kind == Expr::Kind::Const || equal(from, to)) return e;
  context.transportFrame(from, to);
  if (e->kind == Expr::Kind::Read) {
    auto axes = context.transportInterface(from, to, e->axes);
    // These are consequences of legal source coordinate composition.
    obligations.push_back(call("distinct_axes", {axes}));
    obligations.push_back(call(
        "==",
        {context.extents(axes), make(Symbol::Kind::Shape, {}, e->capture)}));
    return std::make_shared<Expr>(
        Expr{e->kind,
             e->capture,
             e->occurrence,
             axes,
             {},
             0,
             e->access ? indexRename(context, e->access, from, to) : I{}});
  }
  std::vector<E> args;
  for (const auto& a : e->args) args.push_back(rename(a, from, to));
  return expression(e->kind, std::move(args), e->axes);
}
I Lowerer::indexValue(const IndexValue& v, const Instance& n,
                      const std::vector<TensorMeaning>& inputs,
                      const std::vector<S>& groups) {
  if (v.kind == IndexValue::Kind::Coordinates)
    return coordinates(groups.at(v.group));
  std::vector<I> args;
  for (const auto& a : v.args)
    args.push_back(indexValue(*a, n, inputs, groups));
  if (v.kind == IndexValue::Kind::Scale || v.kind == IndexValue::Kind::Offset) {
    auto scale = metadata(*v.parameter, n, inputs, groups);
    obligations.push_back(
        call("==", {call("len", {scale}), indexLength(args[0])}));
    bool offset = v.kind == IndexValue::Kind::Offset;
    obligations.push_back(
        call(offset ? "is_valid_extents" : "is_positive_extents", {scale}));
    context.note(offset ? "index-offset-normalization"
                        : "index-scale-normalization");
    return offset ? indexOffset(context, scale, args[0])
                  : indexScale(context, scale, args[0]);
  }
  if (v.kind == IndexValue::Kind::Concat) return indexConcat(args);
  for (std::size_t i = 1; i < args.size(); ++i)
    obligations.push_back(
        call("==", {indexLength(args[0]), indexLength(args[i])}));
  return indexAdd(args);
}
E Lowerer::evaluateAt(const E& e, const S& frame, const I& access) {
  context.tick();
  if (e->kind == Expr::Kind::Const) return e;
  if (e->kind == Expr::Kind::Read) {
    auto composed = indexSubstitute(
        context, e->access ? e->access : coordinates(e->axes), frame, access);
    return std::make_shared<Expr>(
        Expr{e->kind, e->capture, e->occurrence, e->axes, {}, 0, composed});
  }
  if (e->kind == Expr::Kind::Reduce)
    context.require(call("disjoint", {e->axes, indexDependencies(access)}),
                    GuardKind::ReductionDomain);
  std::vector<E> args;
  for (const auto& a : e->args) args.push_back(evaluateAt(a, frame, access));
  return expression(e->kind, args, e->axes);
}
E Lowerer::value(const Value& v, const Instance& n,
                 const std::vector<TensorMeaning>& inputs,
                 const std::vector<S>& groups) {
  context.tick();
  if (v.kind == Value::Kind::Const) return constant(v.constant);
  if (v.kind == Value::Kind::Operand) return inputs.at(v.index).value;
  if (v.kind == Value::Kind::Read) {
    const auto& input = inputs.at(v.index);
    auto access = indexValue(*v.access, n, inputs, groups);
    context.note("index-add-normalization");
    context.note("index-substitution");
    obligations.push_back(
        call("==", {indexLength(access), call("len", {input.shape})}));
    // Coordinates, positive scales, and nonnegative offsets preserve the lower
    // bound. The maximal access must fit the input. This is legality, not
    // equivalence.
    obligations.push_back(call(
        "extent_le", {indexUpperBound(context, access),
                      call("offset_extents",
                           {input.shape, make(Symbol::Kind::Integer, "-1")})}));
    return evaluateAt(input.value, input.axes, access);
  }
  std::vector<E> args;
  for (const auto& a : v.args) args.push_back(value(*a, n, inputs, groups));
  auto k = v.kind == Value::Kind::Add       ? Expr::Kind::Add
           : v.kind == Value::Kind::Product ? Expr::Kind::Product
                                            : Expr::Kind::Reduce;
  return expression(k, std::move(args),
                    v.kind == Value::Kind::Reduce ? groups.at(v.index) : S{});
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
  auto scanUses = [&](auto&& self, const Value& v) -> void {
    if (v.kind == Value::Kind::Read) explicit_reads.insert(v.index);
    if (v.kind == Value::Kind::Operand) implicit_reads.insert(v.index);
    for (const auto& a : v.args) self(self, *a);
  };
  scanUses(scanUses, *n.definition->value);
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
      auto scan = [&](auto&& self, const E& e) -> void {
        if (e->kind == Expr::Kind::Read) accesses[e->occurrence] = e->axes;
        for (const auto& a : e->args) self(self, a);
      };
      scan(scan, inputs[o].value);
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
  auto dt = dtype(n, inputs);
  auto body = value(*n.definition->value, n, inputs, groups);
  obligations.push_back(call("distinct_axes", {axes}));
  auto hasConstant = [&](auto&& self, const E& e) -> bool {
    if (e->kind == Expr::Kind::Const) return true;
    for (const auto& a : e->args)
      if (self(self, a)) return true;
    return false;
  };
  bool independent_output = hasConstant(hasConstant, body);
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
E arithmeticSkeleton(const Plan& p, std::size_t id) {
  const auto& n = p.nodes.at(id);
  if (n.capture)
    return std::make_shared<Expr>(Expr{
        Expr::Kind::Read, n.capture->value, id, call("empty_axes", {}), {}});
  std::vector<E> inputs;
  for (auto child : n.inputs) inputs.push_back(arithmeticSkeleton(p, child));
  std::function<E(const Value&)> lower = [&](const Value& v) -> E {
    if (v.kind == Value::Kind::Const) return constant(v.constant);
    if (v.kind == Value::Kind::Operand || v.kind == Value::Kind::Read)
      return inputs.at(v.index);
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
