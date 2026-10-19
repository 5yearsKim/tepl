#include "src/core/semantics/symbolic/scalar_normalize.h"

#include <algorithm>
#include <charconv>
#include <functional>
#include <optional>

#include "src/core/semantics/symbolic/normalize.h"

namespace tepl::core::semantics {
namespace {
using F = ScalarFunction;
std::optional<std::int64_t> intValue(const S& s) {
  if (s->kind != Symbol::Kind::Integer) return {};
  std::int64_t n;
  auto [end, error] =
      std::from_chars(s->name.data(), s->name.data() + s->name.size(), n);
  if (error != std::errc{} || end != s->name.data() + s->name.size()) return {};
  return n;
}
bool is(const E& e, F f) {
  return e->kind == Expr::Kind::Call && e->function == f;
}
bool integer(const E& e, std::int64_t n) {
  return e->kind == Expr::Kind::Const && e->constant == n;
}
bool floating(Context& c, const E& e) {
  return e->dtype && c.entails(call("is_float", {e->dtype}));
}
bool nonzero(Context& c, const E& e) {
  if (e->kind == Expr::Kind::Const) return e->constant != 0;
  if (e->kind == Expr::Kind::Number) return true;
  if (is(e, F::Exp) && floating(c, e)) return true;
  if (e->kind == Expr::Kind::Product)
    return nonzero(c, e->args[0]) && nonzero(c, e->args[1]);
  return false;
}
bool positive(Context& c, const E& e) {
  return (e->kind == Expr::Kind::Const && e->constant > 0) ||
         (is(e, F::Exp) && floating(c, e));
}
E binary(Expr::Kind kind, E a, E b) { return expression(kind, {a, b}); }
E negate(E a) { return binary(Expr::Kind::Product, constant(-1), a); }
E fromTerms(const Terms& terms) {
  E out;
  for (const auto& t : terms) {
    E value;
    const auto& factors = t.factors;
    for (const auto& factor : factors)
      value = value ? binary(Expr::Kind::Product, value, factor) : factor;
    if (!value)
      value = constant(t.coefficient);
    else if (t.coefficient != 1)
      value = binary(Expr::Kind::Product, constant(t.coefficient), value);
    if (!callIs(t.reductions, "empty_axes"))
      value = expression(Expr::Kind::Reduce, {value}, t.reductions);
    out = out ? binary(Expr::Kind::Add, out, value) : value;
  }
  return out ? out : constant(0);
}
std::vector<E> flatten(const E& e, Expr::Kind kind) {
  if (e->kind != kind) return {e};
  std::vector<E> out;
  for (const auto& a : e->args) {
    auto part = flatten(a, kind);
    out.insert(out.end(), part.begin(), part.end());
  }
  return out;
}
bool matchValues(Context& c, const std::vector<E>& a, const std::vector<E>& b,
                 bool unordered) {
  if (a.size() != b.size()) return false;
  std::vector<bool> used(b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    bool found = false;
    for (std::size_t j = 0; j < b.size(); ++j) {
      if (used[j] || (!unordered && i != j)) continue;
      if (equivalentValue(c, a[i], b[j])) {
        used[j] = true;
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}
}  // namespace
bool equivalentValue(Context& c, const E& a, const E& b) {
  c.tick();
  if (a->kind != b->kind) return false;
  if (a->kind == Expr::Kind::Const) return a->constant == b->constant;
  if (a->kind == Expr::Kind::Number) return a->number == b->number;
  if (a->kind == Expr::Kind::Read) {
    if (a->capture != b->capture) return false;
    if (!a->axes || !b->axes) return !a->axes && !b->axes;
    return equalIndex(c, a->access ? a->access : coordinates(a->axes),
                      b->access ? b->access : coordinates(b->axes));
  }
  if (a->kind == Expr::Kind::Call) {
    if (a->function != b->function ||
        a->parameters.size() != b->parameters.size())
      return false;
    if (a->dtype && b->dtype && !c.entails(call("==", {a->dtype, b->dtype})))
      return false;
    for (std::size_t i = 0; i < a->parameters.size(); ++i)
      if (!c.entails(call("==", {a->parameters[i], b->parameters[i]})))
        return false;
    return matchValues(
        c, a->args, b->args,
        c.plan.algebraic && scalarSignature(a->function).commutative);
  }
  if (a->kind == Expr::Kind::Reduce)
    return c.entails(call(c.plan.algebraic ? "same_set" : "==",
                          {a->axes, b->axes})) &&
           equivalentValue(c, a->args[0], b->args[0]);
  return matchValues(c, c.plan.algebraic ? flatten(a, a->kind) : a->args,
                     c.plan.algebraic ? flatten(b, b->kind) : b->args,
                     c.plan.algebraic);
}
E normalizeScalar(Context& c, const E& e) {
  c.tick();
  if (!c.plan.algebraic)
    return e;  // Congruence keeps argument evaluation order.
  auto args = e->args;
  for (auto& a : args) {
    auto dtype = a->dtype;
    auto normalized = fromTerms(collectTerms(c, normalize(c, a)));
    auto copy = std::make_shared<Expr>(*normalized);
    copy->dtype = dtype;
    a = copy;
  }
  auto parameters = e->parameters;
  for (auto& p : parameters) p = c.canonical(p);
  E result = scalarCall(e->function, args, parameters, e->dtype);
  const auto& signature = scalarSignature(e->function);
  const auto f = e->function;
  auto eq = [&](const E& a, const E& b) { return equivalentValue(c, a, b); };
  auto apply = [&](F id, std::vector<E> xs) {
    return scalarCall(id, std::move(xs), {}, e->dtype);
  };
  auto law = [&](std::string name, E value) {
    c.note("scalar-" + name);
    return value;
  };
  if (signature.associative) {
    std::vector<E> flat;
    for (const auto& a : args) {
      if (is(a, f))
        flat.insert(flat.end(), a->args.begin(), a->args.end());
      else
        flat.push_back(a);
    }
    args = std::move(flat);
    if (signature.idempotent) {
      std::vector<E> unique;
      for (const auto& a : args)
        if (std::none_of(unique.begin(), unique.end(),
                         [&](const E& b) { return eq(a, b); }))
          unique.push_back(a);
      args = std::move(unique);
    }
    if (f == F::And || f == F::Or) {
      const int identity = f == F::And ? 1 : 0, absorbing = 1 - identity;
      for (const auto& a : args)
        if (integer(a, absorbing))
          return law("boolean-absorbing", constant(absorbing));
      std::erase_if(args, [&](const E& a) { return integer(a, identity); });
      if (args.empty()) return law("boolean-identity", constant(identity));
    }
    if (f == F::Xor) {
      std::vector<E> remaining;
      for (const auto& a : args) {
        if (integer(a, 0)) continue;
        auto it = std::find_if(remaining.begin(), remaining.end(),
                               [&](const E& b) { return eq(a, b); });
        if (it == remaining.end())
          remaining.push_back(a);
        else
          remaining.erase(it);
      }
      args = std::move(remaining);
      if (args.empty()) return law("xor-cancellation", constant(0));
    }
    if (f == F::Minimum || f == F::Maximum) {
      // Absorption in a totally ordered mathematical value domain.
      const auto dual = f == F::Minimum ? F::Maximum : F::Minimum;
      const auto snapshot = args;
      std::erase_if(args, [&](const E& a) {
        if (!is(a, dual)) return false;
        for (const auto& nested : a->args)
          for (const auto& b : snapshot)
            if (a != b && eq(nested, b)) return true;
        return false;
      });
    }
    if (f == F::Minimum || f == F::Maximum) {
      if (e->dtype &&
          c.entails(call("==", {e->dtype, call("dtype_i32", {})}))) {
        auto identity = f == F::Minimum ? 2147483647LL : -2147483648LL;
        // Captured i32 elements lie within their dtype bounds. Ideal-arithmetic
        // intermediates such as X+X need not: never infer their value range
        // from a dtype label alone.
        bool bounded = std::all_of(args.begin(), args.end(), [&](const E& a) {
          return (a->kind == Expr::Kind::Read && a->dtype &&
                  c.entails(call("==", {a->dtype, call("dtype_i32", {})}))) ||
                 (a->kind == Expr::Kind::Const &&
                  a->constant >= -2147483648LL && a->constant <= 2147483647LL);
        });
        if (bounded) {
          std::erase_if(args, [&](const E& a) { return integer(a, identity); });
          if (args.empty())
            return law("ordered-type-bound", constant(identity));
        }
      }
      if (!args.empty() && args[0]->kind == Expr::Kind::Call &&
          scalarSignature(args[0]->function).monotone) {
        auto function = args[0]->function;
        bool all = std::all_of(args.begin(), args.end(), [&](const E& a) {
          return is(a, function) && a->parameters.empty();
        });
        if (all) {
          std::vector<E> inner;
          for (const auto& a : args) inner.push_back(a->args[0]);
          return law("monotone-order", apply(function, {apply(f, inner)}));
        }
      }
    }
    if (args.size() == 1) return law("idempotent-or-identity", args[0]);
    return law("associative-commutative",
               scalarCall(f, args, parameters, e->dtype));
  }
  if (f == F::Not) {
    if (is(args[0], F::Not)) return law("not-involution", args[0]->args[0]);
    if (integer(args[0], 0) || integer(args[0], 1))
      return law("not-constant", constant(1 - args[0]->constant));
  }
  if (f == F::Select) {
    if (eq(args[1], args[2])) return law("select-identical", args[1]);
    if (integer(args[0], 0)) return law("select-false", args[2]);
    if (integer(args[0], 1)) return law("select-true", args[1]);
    if (is(args[0], F::Not))
      return law("select-not",
                 apply(F::Select, {args[0]->args[0], args[2], args[1]}));
    if (e->dtype && c.entails(call("==", {e->dtype, call("dtype_bool", {})}))) {
      if (integer(args[1], 1) && integer(args[2], 0))
        return law("select-predicate", args[0]);
      if (integer(args[1], 0) && integer(args[2], 1))
        return law("select-negated-predicate", apply(F::Not, {args[0]}));
    }
    if (is(args[0], F::Compare) && args[0]->parameters.size() == 1) {
      auto direction = intValue(c.canonical(args[0]->parameters[0]));
      auto a = args[0]->args[0], b = args[0]->args[1];
      if (direction && eq(args[1], a) && eq(args[2], b)) {
        if (*direction == 0) return law("select-equal", b);
        if (*direction == 1) return law("select-unequal", a);
        if (a->dtype &&
            c.entails(call("==", {a->dtype, call("dtype_bool", {})}))) {
          if (*direction == 2 || *direction == 3)
            return law("select-boolean-order", apply(F::Or, {a, b}));
          if (*direction == 4 || *direction == 5)
            return law("select-boolean-order", apply(F::And, {a, b}));
        }
      }
    }
  }
  if (f == F::Compare && parameters.size() == 1) {
    auto direction = intValue(parameters[0]);
    if (!direction)
      for (int i = 0; i < 6; ++i)
        if (c.entails(call(
                "==", {parameters[0], tepl::core::semantics::integer(i)}))) {
          direction = i;
          break;
        }
    if (!direction && eq(args[0], args[1])) {
      auto relation = [&](int i) {
        return call("==", {parameters[0], tepl::core::semantics::integer(i)});
      };
      if (c.entails(call(
              "||", {call("||", {relation(0), relation(2)}), relation(4)})))
        return law("compare-reflexive", constant(1));
    }
    if (direction && *direction == 3) {
      if (is(args[0], F::Maximum) && args[0]->args.size() == 2) {
        for (int i = 0; i < 2; ++i)
          if (eq(args[0]->args[i], args[1]))
            return law("compare-maximum",
                       scalarCall(f, {args[0]->args[1 - i], args[1]},
                                  parameters, e->dtype));
      }
      if (is(args[1], F::Minimum) && args[1]->args.size() == 2) {
        for (int i = 0; i < 2; ++i)
          if (eq(args[0], args[1]->args[i]))
            return law("compare-minimum",
                       scalarCall(f, {args[0], args[1]->args[1 - i]},
                                  parameters, e->dtype));
      }
    }
    if (direction && eq(args[0], args[1]))
      return law(
          "compare-reflexive",
          constant(*direction == 0 || *direction == 2 || *direction == 4));
    if (direction && args[0]->dtype &&
        c.entails(call("==", {args[0]->dtype, call("dtype_bool", {})}))) {
      if ((*direction == 1 && integer(args[1], 0)) ||
          (*direction == 0 && integer(args[1], 1)))
        return law("compare-boolean", args[0]);
    }
  }
  if (f == F::Abs) {
    if (is(args[0], F::Abs)) return law("abs-idempotent", args[0]);
    if (args[0]->dtype &&
        c.entails(call("is_unsigned_integer", {args[0]->dtype})))
      return law("abs-unsigned", args[0]);
  }
  if (f == F::Power) {
    if (integer(args[1], 0)) return law("power-zero", constant(1));
    if (integer(args[1], 1)) return law("power-one", args[0]);
    if (integer(args[1], 2))
      return law("power-two", binary(Expr::Kind::Product, args[0], args[0]));
    if (integer(args[1], 3))
      return law("power-three",
                 binary(Expr::Kind::Product, args[0],
                        binary(Expr::Kind::Product, args[0], args[0])));
    if (integer(args[1], -1) && nonzero(c, args[0]))
      return law("power-reciprocal", apply(F::Divide, {constant(1), args[0]}));
    if (is(args[0], F::Exp) && floating(c, args[0]))
      return law("power-exponential",
                 apply(F::Exp, {binary(Expr::Kind::Product, args[0]->args[0],
                                       args[1])}));
  }
  if (f == F::Log) {
    if (is(args[0], F::Exp) && floating(c, args[0]))
      return law("log-exp", args[0]->args[0]);
    if (is(args[0], F::Sqrt) && positive(c, args[0]->args[0]))
      return law("log-sqrt", binary(Expr::Kind::Product, number("0.5"),
                                    apply(F::Log, {args[0]->args[0]})));
  }
  if (f == F::Sqrt && args[0]->kind == Expr::Kind::Product &&
      eq(args[0]->args[0], args[0]->args[1]))
    return law("sqrt-square", apply(F::Abs, {args[0]->args[0]}));
  if (f == F::Divide) {
    if (integer(args[1], 1)) return law("divide-one", args[0]);
    if (is(args[0], F::Exp) && is(args[1], F::Exp) && floating(c, args[0]) &&
        floating(c, args[1]))
      return law("divide-exponentials",
                 apply(F::Exp, {binary(Expr::Kind::Add, args[0]->args[0],
                                       negate(args[1]->args[0]))}));
    if (is(args[1], F::Exp) && floating(c, args[1]))
      return law("divide-exponential",
                 binary(Expr::Kind::Product, args[0],
                        apply(F::Exp, {negate(args[1]->args[0])})));
  }
  if (f == F::Clamp && is(args[1], F::Clamp) && eq(args[0], args[1]->args[0]) &&
      eq(args[2], args[1]->args[2]))
    return law("clamp-idempotent", args[1]);
  return result;
}
void scalarProductLaws(Context& c, std::vector<E>& factors) {
  if (!c.plan.algebraic) return;
  for (std::size_t i = 0; i < factors.size(); ++i)
    for (std::size_t j = i + 1; j < factors.size(); ++j) {
      auto a = factors[i], b = factors[j];
      if (is(a, F::Exp) && is(b, F::Exp) && floating(c, a) && floating(c, b)) {
        factors[i] = normalizeScalar(
            c, scalarCall(F::Exp,
                          {binary(Expr::Kind::Add, a->args[0], b->args[0])}, {},
                          a->dtype));
        factors.erase(factors.begin() + j--);
        c.note("scalar-exponential-product");
      } else if (is(a, F::Abs) && is(b, F::Abs) &&
                 equivalentValue(c, a->args[0], b->args[0])) {
        factors[i] = a->args[0];
        factors[j] = b->args[0];
        c.note("scalar-square-abs");
      }
    }
}
}  // namespace tepl::core::semantics
