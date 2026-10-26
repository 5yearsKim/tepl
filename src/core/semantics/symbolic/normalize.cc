#include "src/core/semantics/symbolic/normalize.h"

#include <algorithm>

#include "src/core/semantics/symbolic/reduction_reflection.h"
#include "src/core/semantics/symbolic/scalar_normalize.h"

namespace tepl::core::semantics {
namespace {
std::int64_t multiply(std::int64_t a, std::int64_t b) {
  std::int64_t out;
  if (__builtin_mul_overflow(a, b, &out))
    throw ProofFailure("coefficient overflow (signed 64-bit fragment)");
  return out;
}
std::int64_t add(std::int64_t a, std::int64_t b) {
  std::int64_t out;
  if (__builtin_add_overflow(a, b, &out))
    throw ProofFailure("coefficient overflow (signed 64-bit fragment)");
  return out;
}
bool sameMonomial(Context& c, const Term& a, const Term& b) {
  if (footprint(a) != footprint(b)) return false;
  if (!c.entails(call("same_set", {a.reductions, b.reductions}))) return false;
  std::vector<bool> used(b.factors.size());
  if (a.factors.size() != b.factors.size()) return false;
  for (const auto& factor : a.factors) {
    bool found = false;
    for (std::size_t i = 0; i < b.factors.size(); ++i) {
      if (!used[i] && equivalentValue(c, factor, b.factors[i])) {
        used[i] = true;
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}
S readAxes(const Term& t) {
  std::vector<S> xs;
  for (const auto& r : t.reads) xs.push_back(freeAxes(r));
  return call("union_axes", std::move(xs));
}
}  // namespace
Terms normalize(Context& c, const E& e,
                std::vector<RuleProof::RewriteStep>* rewrites) {
  c.tick();
  if (e->kind == Expr::Kind::Read)
    return {{call("empty_axes", {}), {e}, 1, {e}}};
  if (e->kind == Expr::Kind::Call) {
    auto simplified = normalizeScalar(c, e);
    if (simplified->kind != Expr::Kind::Call)
      return normalize(c, simplified, rewrites);
    return {{call("empty_axes", {}), tensorReads(simplified), 1, {simplified}}};
  }
  if (e->kind == Expr::Kind::Number)
    return {{call("empty_axes", {}), {}, 1, {e}}};
  if (e->kind == Expr::Kind::Const)
    return {{call("empty_axes", {}), {}, e->constant}};
  if (!c.plan.algebraic)
    throw ProofFailure("arithmetic normalization requires algebraic policy");
  auto record = [&](const std::string& law, const Terms& out,
                    std::vector<S> conditions = {}) {
    c.note(law);
    if (rewrites)
      rewrites->push_back({law, show(e), show(out), std::move(conditions)});
  };
  auto left = normalize(c, e->args[0], rewrites);
  if (e->kind == Expr::Kind::Reduce) {
    std::vector<S> conditions;
    for (auto& t : left) {
      auto disjoint = call("disjoint", {t.reductions, e->axes});
      c.require(disjoint, GuardKind::ReductionDomain);
      conditions.push_back(disjoint);
      t.reductions = call("concat", {e->axes, t.reductions});
      normalizeReflections(c, t);
    }
    if (left.size() > 1) record("reduction-linearity", left);
    if (e->args[0]->kind == Expr::Kind::Reduce)
      record("reduction-composition", left, conditions);
    return left;
  }
  auto right = normalize(c, e->args[1], rewrites);
  if (e->kind == Expr::Kind::Add) {
    if (left.size() + right.size() > ProofLimits::normalization_terms)
      throw ProofFailure("normalization term limit exceeded",
                         FailureKind::BudgetExceeded);
    left.insert(left.end(), right.begin(), right.end());
    if (e->args[0]->kind == Expr::Kind::Add ||
        e->args[1]->kind == Expr::Kind::Add)
      record("add-associativity", left);
    return left;
  }
  if (left.empty() || right.empty()) return {};
  if (left.size() > ProofLimits::normalization_terms / right.size())
    throw ProofFailure("normalization term limit exceeded",
                       FailureKind::BudgetExceeded);
  Terms out;
  std::vector<S> conditions;
  for (const auto& a : left)
    for (const auto& b : right) {
      c.tick();
      auto q = call("disjoint",
                    {a.reductions, c.difference(readAxes(b), b.reductions)});
      auto r = call("disjoint",
                    {b.reductions, c.difference(readAxes(a), a.reductions)});
      auto disjoint = call("disjoint", {a.reductions, b.reductions});
      c.require(q, GuardKind::ReductionDomain);
      c.require(r, GuardKind::ReductionDomain);
      c.require(disjoint, GuardKind::ReductionDomain);
      conditions.insert(conditions.end(), {q, r, disjoint});
      Term t{call("concat", {a.reductions, b.reductions}), a.reads,
             multiply(a.coefficient, b.coefficient)};
      t.reads.insert(t.reads.end(), b.reads.begin(), b.reads.end());
      t.factors = a.factors;
      t.factors.insert(t.factors.end(), b.factors.begin(), b.factors.end());
      out.push_back(std::move(t));
    }
  if (left.size() > 1 || right.size() > 1)
    record("product-distributivity", out);
  if (!callIs(left[0].reductions, "empty_axes") ||
      !callIs(right[0].reductions, "empty_axes"))
    record("reduction-extraction", out, conditions);
  if (e->args[0]->kind == Expr::Kind::Product ||
      e->args[1]->kind == Expr::Kind::Product)
    record("product-associativity", out);
  if (e->args[0]->kind == Expr::Kind::Const ||
      e->args[1]->kind == Expr::Kind::Const)
    record("constant-arithmetic", out);
  return out;
}
Terms collectTerms(Context& c, const Terms& raw,
                   std::vector<RuleProof::RewriteStep>* rewrites) {
  Terms out;
  for (auto t : raw) {
    c.tick();
    if (t.factors.empty() && !t.reads.empty()) t.factors = t.reads;
    scalarProductLaws(c, t.factors);
    t.reads.clear();
    for (const auto& f : t.factors) {
      auto rs = tensorReads(f);
      t.reads.insert(t.reads.end(), rs.begin(), rs.end());
    }
    if (!t.coefficient) {
      c.note("zero-multiplication");
      if (rewrites)
        rewrites->push_back(
            {"zero-multiplication", show(Terms{t}), "Const(0)", {}});
      continue;
    }
    auto it = std::find_if(out.begin(), out.end(), [&](const Term& u) {
      return sameMonomial(c, t, u);
    });
    if (it == out.end())
      out.push_back(t);
    else {
      auto before = rewrites ? show(Terms{*it, t}) : std::string{};
      it->coefficient = add(it->coefficient, t.coefficient);
      auto law =
          it->coefficient ? "coefficient-combination" : "additive-cancellation";
      c.note(law);
      if (rewrites)
        rewrites->push_back({law,
                             std::move(before),
                             it->coefficient ? show(Terms{*it}) : "Const(0)",
                             {}});
    }
  }
  std::erase_if(out, [](const Term& t) { return !t.coefficient; });
  return out;
}
void equivalentTerms(Context& c, const Terms& left, const Terms& right) {
  if (left.size() != right.size())
    throw ProofFailure("normalized scalar term counts differ");
  std::vector<bool> used(right.size());
  for (const auto& a : left) {
    c.tick();
    bool found = false;
    for (std::size_t i = 0; i < right.size(); ++i) {
      if (!used[i] && a.coefficient == right[i].coefficient &&
          sameMonomial(c, a, right[i])) {
        used[i] = true;
        found = true;
        break;
      }
    }
    if (!found)
      throw ProofFailure(
          "normalized coefficients, tensor accesses, or reduction domains "
          "differ");
  }
}
std::string footprint(const Term& t) {
  std::vector<std::size_t> ids;
  for (const auto& r : t.reads) ids.push_back(r->capture);
  std::sort(ids.begin(), ids.end());
  std::string out;
  for (auto id : ids) out += std::to_string(id) + ",";
  return out;
}
std::string show(const Terms& terms) {
  if (terms.empty()) return "Const(0)";
  std::string out;
  for (const auto& t : terms) {
    if (!out.empty()) out += " + ";
    if (!callIs(t.reductions, "empty_axes"))
      out += "Reduce(" + format(*t.reductions) + ", ";
    if (t.factors.empty())
      out += "Const(" + std::to_string(t.coefficient) + ")";
    else if (t.coefficient != 1)
      out += std::to_string(t.coefficient) + " * ";
    if (t.factors.size() > 1) out += "Product(";
    for (std::size_t i = 0; i < t.factors.size(); ++i) {
      if (i) out += ", ";
      out += show(t.factors[i]);
    }
    if (t.factors.size() > 1) out += ")";
    if (!callIs(t.reductions, "empty_axes")) out += ")";
  }
  return out;
}

}  // namespace tepl::core::semantics
