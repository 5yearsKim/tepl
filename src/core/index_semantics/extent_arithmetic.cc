#include <algorithm>

#include "src/core/index_semantics/index_expr.h"

namespace tepl::core::index_semantics {
namespace {
// Vector arithmetic is pointwise. Scoped tuple expressions are atoms;
// floor division stays opaque except for the separately guarded upper bound.
using Monomial = std::vector<std::string>;
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
Polynomial polynomial(Context& c, const S& raw) {
  c.tick();
  auto x = c.canonical(raw);
  if (callIs(x, "extent_add") || callIs(x, "extent_sub")) {
    auto out = polynomial(c, x->args[0]);
    accumulate(out, polynomial(c, x->args[1]),
               callIs(x, "extent_sub") ? -1 : 1);
    return out;
  }
  if (callIs(x, "offset_extents")) {
    auto out = polynomial(c, x->args[0]);
    accumulate(out, {{{}, literal(x->args[1])}}, 1);
    return out;
  }
  if (callIs(x, "scale_extents")) {
    Polynomial out;
    accumulate(out, polynomial(c, x->args[0]), literal(x->args[1]));
    return out;
  }
  if (callIs(x, "extent_mul")) {
    auto a = polynomial(c, x->args[0]), b = polynomial(c, x->args[1]);
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
  return equalExtentArithmetic(c, upper(c, a, mode), b, mode);
}
}  // namespace tepl::core::index_semantics
