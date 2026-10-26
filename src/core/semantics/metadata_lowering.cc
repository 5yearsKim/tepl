#include <functional>
#include <type_traits>

#include "src/core/semantics/graph.h"
#include "src/shared/builtins/axes.h"

namespace tepl::core::semantics {
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
  if (m.name == "gather")
    obligations.push_back(
        call("is_valid_axis_list", {args[1], call("len", {args[0]})}));
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

}  // namespace tepl::core::semantics
