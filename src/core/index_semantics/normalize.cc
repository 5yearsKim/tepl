#include "src/core/index_semantics/normalize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace tepl::core::index_semantics {
namespace {
S readAxes(const Term& t) {
  std::vector<S> xs;
  for (const auto& r : t.reads) xs.push_back(r->axes);
  return call("union_axes", std::move(xs));
}
}  // namespace
Terms normalize(Context& c, const E& e,
                std::vector<RuleProof::RewriteStep>* rewrites) {
  c.tick();
  if (e->kind == Expr::Kind::Read) return {{call("empty_axes", {}), {e}}};
  if (!c.plan.algebraic)
    throw std::runtime_error(
        "arithmetic normalization requires algebraic policy");
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
      c.require(disjoint);
      conditions.push_back(disjoint);
      t.reductions = call("concat", {e->axes, t.reductions});
    }
    if (left.size() > 1) record("reduction-linearity", left);
    if (e->args[0]->kind == Expr::Kind::Reduce)
      record("reduction-composition", left, conditions);
    return left;
  }
  auto right = normalize(c, e->args[1], rewrites);
  if (e->kind == Expr::Kind::Add) {
    left.insert(left.end(), right.begin(), right.end());
    if (e->args[0]->kind == Expr::Kind::Add ||
        e->args[1]->kind == Expr::Kind::Add)
      record("add-associativity", left);
    return left;
  }
  if (left.size() > 256 / right.size())
    throw std::runtime_error("normalization term limit exceeded");
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
      c.require(q);
      c.require(r);
      c.require(disjoint);
      conditions.insert(conditions.end(), {q, r, disjoint});
      Term t{call("concat", {a.reductions, b.reductions}), a.reads};
      t.reads.insert(t.reads.end(), b.reads.begin(), b.reads.end());
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
  return out;
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
  std::string out;
  for (const auto& t : terms) {
    if (!out.empty()) out += " + ";
    if (!callIs(t.reductions, "empty_axes"))
      out += "Reduce(" + format(*t.reductions) + ", ";
    if (t.reads.size() > 1) out += "Product(";
    for (std::size_t i = 0; i < t.reads.size(); ++i) {
      if (i) out += ", ";
      out += show(t.reads[i]);
    }
    if (t.reads.size() > 1) out += ")";
    if (!callIs(t.reductions, "empty_axes")) out += ")";
  }
  return out;
}
Correspondence correspond(Context& c, const TensorMeaning& lhs, const Terms& lt,
                          const Terms& rt) {
  Correspondence result;
  std::map<std::string, const Term*> left;
  for (const auto& t : lt)
    if (!left.emplace(footprint(t), &t).second)
      throw std::runtime_error(
          "ambiguous additive correspondence is unsupported");
  std::vector<std::pair<const Term*, const Term*>> pairs;
  std::set<std::string> seen;
  for (const auto& t : rt) {
    c.tick();
    auto k = footprint(t);
    if (!left.contains(k) || !seen.insert(k).second)
      throw std::runtime_error("normalized tensor monomials differ");
    std::map<std::size_t, E> reads;
    for (const auto& r : left.at(k)->reads)
      if (!reads.emplace(r->capture, r).second)
        throw std::runtime_error(
            "repeated tensor factors require occurrence matching "
            "(unsupported)");
    for (const auto& r : t.reads) {
      auto axes = reads.at(r->capture)->axes;
      auto [it, inserted] = result.anchors.emplace(r->occurrence, axes);
      if (!inserted) c.require(call("==", {axes, it->second}));
    }
    for (std::size_t i = 0; i < t.reads.size(); ++i)
      if (t.reads[i]->capture != left.at(k)->reads[i]->capture)
        c.note("commutative-matching");
    if (k != footprint(lt[pairs.size()])) c.note("commutative-matching");
    pairs.emplace_back(left.at(k), &t);
  }
  if (seen.size() != left.size())
    throw std::runtime_error("normalized tensor monomial counts differ");
  std::function<std::set<std::size_t>(std::size_t)> visit =
      [&](std::size_t id) {
        c.tick();
        const auto& n = c.plan.nodes[id];
        std::set<std::size_t> leaves;
        if (n.capture)
          leaves.insert(id);
        else
          for (auto child : n.inputs) {
            auto xs = visit(child);
            leaves.insert(xs.begin(), xs.end());
          }
        S boundary;
        for (const auto& [source, target] : pairs) {
          std::vector<S> inside, outside;
          bool active = false;
          for (const auto& r : target->reads) {
            bool local = leaves.contains(r->occurrence);
            active |= local;
            (local ? inside : outside)
                .push_back(result.anchors.at(r->occurrence));
          }
          if (!active) continue;
          auto b = c.intersect(
              call("union_axes", inside),
              call("union_axes", {call("union_axes", outside), lhs.axes}));
          if (boundary)
            c.require(call("same_set", {boundary, b}));
          else
            boundary = b;
        }
        if (!boundary)
          throw std::runtime_error("RHS subtree has no semantic contribution");
        result.boundaries[id] = boundary;
        return leaves;
      };
  visit(c.plan.rhs);
  return result;
}
}  // namespace tepl::core::index_semantics
