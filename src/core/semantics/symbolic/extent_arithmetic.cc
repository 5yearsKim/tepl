#include <algorithm>
#include <functional>
#include <limits>

#include "src/core/semantics/symbolic/index_expr.h"

namespace tepl::core::semantics {
namespace {
// Vector arithmetic is pointwise. Scoped tuple expressions are atoms;
// floor division stays opaque except for the separately guarded upper bound.
using Monomial = std::vector<Key>;
using Polynomial = std::map<Monomial, std::int64_t>;
std::int64_t add(std::int64_t a, std::int64_t b) {
  std::int64_t n;
  if (__builtin_add_overflow(a, b, &n))
    throw ProofFailure("extent arithmetic overflow");
  return n;
}
std::int64_t mul(std::int64_t a, std::int64_t b) {
  std::int64_t n;
  if (__builtin_mul_overflow(a, b, &n))
    throw ProofFailure("extent arithmetic overflow");
  return n;
}
std::int64_t literal(const S& s) {
  if (s->kind != Symbol::Kind::Integer)
    throw ProofFailure("nonliteral extent offset unsupported");
  try {
    return std::stoll(s->name);
  } catch (...) {
    throw ProofFailure("extent arithmetic overflow");
  }
}
void accumulate(Polynomial& a, const Polynomial& b, std::int64_t factor) {
  for (const auto& [m, n] : b) a[m] = add(a[m], mul(n, factor));
  std::erase_if(a, [](const auto& term) { return term.second == 0; });
  if (a.size() > ProofLimits::normalization_terms)
    throw ProofFailure("extent polynomial exceeds term limit",
                       FailureKind::BudgetExceeded);
}
Polynomial polynomial(Context& c, const S& raw,
                      std::map<Key, S>* atoms = nullptr) {
  c.tick();
  auto x = c.canonical(raw);
  if (callIs(x, "extent_add") || callIs(x, "extent_sub")) {
    auto out = polynomial(c, x->args[0], atoms);
    accumulate(out, polynomial(c, x->args[1], atoms),
               callIs(x, "extent_sub") ? -1 : 1);
    return out;
  }
  if (callIs(x, "offset_extents")) {
    auto out = polynomial(c, x->args[0], atoms);
    accumulate(out, {{{}, literal(x->args[1])}}, 1);
    return out;
  }
  if (callIs(x, "scale_extents")) {
    Polynomial out;
    accumulate(out, polynomial(c, x->args[0], atoms), literal(x->args[1]));
    return out;
  }
  if (callIs(x, "extent_mul")) {
    auto a = polynomial(c, x->args[0], atoms),
         b = polynomial(c, x->args[1], atoms);
    Polynomial out;
    for (const auto& [am, an] : a)
      for (const auto& [bm, bn] : b) {
        c.tick();
        auto m = am;
        m.insert(m.end(), bm.begin(), bm.end());
        std::sort(m.begin(), m.end());
        out[m] = add(out[m], mul(an, bn));
        if (out.size() > ProofLimits::normalization_terms)
          throw ProofFailure("extent polynomial exceeds term limit",
                             FailureKind::BudgetExceeded);
      }
    std::erase_if(out, [](const auto& term) { return term.second == 0; });
    return out;
  }
  if (atoms) (*atoms)[key(x)] = x;
  return {{{key(x)}, 1}};
}
bool query(Context& c, const S& f, QueryMode mode) {
  return mode == QueryMode::Evidence ? c.evidence(f) : c.entails(f);
}
// Produce only an upper bound: rewrite through monotone contexts. In
// particular, do not recurse into subtraction or an unproved signed product.
S upper(Context& c, const S& raw, QueryMode mode) {
  c.tick();
  auto x = c.canonical(raw);
  if (callIs(x, "extent_add"))
    return call("extent_add",
                {upper(c, x->args[0], mode), upper(c, x->args[1], mode)});
  if (callIs(x, "offset_extents"))
    return call("offset_extents", {upper(c, x->args[0], mode), x->args[1]});
  if (callIs(x, "scale_extents") && literal(x->args[1]) > 0)
    return call("scale_extents", {upper(c, x->args[0], mode), x->args[1]});
  if (callIs(x, "extent_mul")) {
    std::vector<S> factors;
    auto collect = [&](auto&& self, const S& y) -> void {
      if (callIs(y, "extent_mul")) {
        self(self, y->args[0]);
        self(self, y->args[1]);
      } else
        factors.push_back(y);
    };
    collect(collect, x);
    for (std::size_t i = 0; i < factors.size(); ++i) {
      auto f = factors[i];
      if (!callIs(f, "extent_floor_div")) continue;
      for (std::size_t j = 0; j < factors.size(); ++j) {
        if (i == j || !equalExtentArithmetic(c, factors[j], f->args[1], mode))
          continue;
        if (!query(c, call("is_valid_extents", {f->args[0]}), mode) ||
            !query(c, call("is_positive_extents", {f->args[1]}), mode))
          continue;
        S out = f->args[0];
        bool positive = true;
        for (std::size_t k = 0; k < factors.size(); ++k)
          if (k != i && k != j) {
            if (!query(c, call("is_valid_extents", {factors[k]}), mode)) {
              positive = false;
              break;
            }
            out = call("extent_mul", {out, factors[k]});
          }
        if (positive) {
          c.note("extent-floor-bound");
          return out;
        }
      }
    }
  }
  return x;
}
// Bounded forward propagation of established vector inequalities. A known
// L<=U permits remainder+m*L <= remainder+m*U when m is nonnegative.
// Factored polynomial subexpressions are recognized without expanding rank.
bool monotoneBound(Context& c, const S& left, const S& right, QueryMode mode) {
  std::map<Key, S> atoms;
  auto first = polynomial(c, left, &atoms), goal = polynomial(c, right, &atoms);
  struct Bound {
    Polynomial lower, upper;
  };
  std::vector<Bound> bounds;
  for (const auto& [l, u] : c.extentBounds()) {
    c.tick();
    if (!query(c, call("==", {call("len", {left}), call("len", {l})}), mode) ||
        !query(c, call("==", {call("len", {l}), call("len", {u})}), mode))
      continue;
    auto lower = polynomial(c, l, &atoms), upper = polynomial(c, u, &atoms);
    if (!lower.empty() && lower != upper)
      bounds.push_back({std::move(lower), std::move(upper)});
  }
  auto scaled = [&](const Polynomial& p, const Monomial& factor,
                    std::int64_t coefficient) {
    Polynomial out;
    for (const auto& [m, n] : p) {
      c.tick();
      auto product = m;
      product.insert(product.end(), factor.begin(), factor.end());
      std::sort(product.begin(), product.end());
      out[product] = add(out[product], mul(n, coefficient));
    }
    return out;
  };
  std::set<Polynomial> visited;
  std::function<bool(const Polynomial&, std::size_t)> search =
      [&](const Polynomial& current, std::size_t depth) {
        c.tick();
        if (current == goal) return true;
        if (depth == 0 || !visited.insert(current).second) return false;
        for (const auto& bound : bounds) {
          const auto anchor =
              std::max_element(bound.lower.begin(), bound.lower.end(),
                               [](const auto& a, const auto& b) {
                                 return a.first.size() < b.first.size();
                               });
          for (const auto& [term, n] : current) {
            c.tick();
            if (!std::includes(term.begin(), term.end(), anchor->first.begin(),
                               anchor->first.end()))
              continue;
            auto d = anchor->second;
            if (d == -1 && n == std::numeric_limits<std::int64_t>::min())
              continue;
            if (n % d != 0 || n / d <= 0) continue;
            auto coefficient = n / d;
            Monomial factor;
            std::set_difference(term.begin(), term.end(), anchor->first.begin(),
                                anchor->first.end(),
                                std::back_inserter(factor));
            bool nonnegative = true;
            for (const auto& atom : factor)
              if (!query(c,
                         call("==", {call("len", {atoms.at(atom)}),
                                     call("len", {left})}),
                         mode) ||
                  !query(c, call("is_valid_extents", {atoms.at(atom)}), mode)) {
                nonnegative = false;
                break;
              }
            if (!nonnegative) continue;
            auto contribution = scaled(bound.lower, factor, coefficient);
            bool present = true;
            for (const auto& [m, value] : contribution) {
              auto it = current.find(m);
              if (it == current.end() ||
                  (value > 0 ? it->second < value : it->second > value)) {
                present = false;
                break;
              }
            }
            if (!present) continue;
            auto next = current;
            accumulate(next, contribution, -1);
            accumulate(next, scaled(bound.upper, factor, coefficient), 1);
            if (next == current) continue;
            if (search(next, depth - 1)) {
              c.note("extent-monotonicity");
              return true;
            }
          }
        }
        return false;
      };
  return search(first, ProofLimits::extent_bound_steps);
}
}  // namespace
bool equalExtentArithmetic(Context& c, const S& left, const S& right,
                           QueryMode mode) {
  if (!query(c, call("==", {call("len", {left}), call("len", {right})}), mode))
    return false;
  return polynomial(c, left) == polynomial(c, right);
}
bool extentBoundArithmetic(Context& c, const S& left, const S& right,
                           QueryMode mode) {
  if (!query(c, call("==", {call("len", {left}), call("len", {right})}), mode))
    return false;
  auto a = c.canonical(left), b = c.canonical(right);
  if (equalExtentArithmetic(c, a, b, mode)) return true;
  if (callIs(a, "concat") && callIs(b, "concat") &&
      a->args.size() == b->args.size()) {
    for (std::size_t i = 0; i < a->args.size(); ++i)
      if (!extentBoundArithmetic(c, a->args[i], b->args[i], mode)) return false;
    return true;
  }
  auto tightened = upper(c, a, mode);
  if (equalExtentArithmetic(c, tightened, b, mode)) return true;
  return monotoneBound(c, tightened, b, mode);
}
}  // namespace tepl::core::semantics
