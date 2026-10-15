#include "src/core/semantics/symbolic/normalize.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

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
  std::vector<bool> used(b.reads.size());
  for (const auto& r : a.reads) {
    bool found = false;
    for (std::size_t i = 0; i < b.reads.size(); ++i) {
      if (used[i] || r->capture != b.reads[i]->capture) continue;
      auto left_access = r->access ? r->access : coordinates(r->axes);
      auto right_access = b.reads[i]->access ? b.reads[i]->access
                                             : coordinates(b.reads[i]->axes);
      if (equalIndex(c, left_access, right_access)) {
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
  if (e->kind == Expr::Kind::Read) return {{call("empty_axes", {}), {e}}};
  if (!c.plan.algebraic)
    throw ProofFailure("arithmetic normalization requires algebraic policy");
  if (e->kind == Expr::Kind::Const)
    return {{call("empty_axes", {}), {}, e->constant}};
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
  for (const auto& t : raw) {
    c.tick();
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
    if (t.reads.empty())
      out += "Const(" + std::to_string(t.coefficient) + ")";
    else if (t.coefficient != 1)
      out += std::to_string(t.coefficient) + " * ";
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
  // Use raw terms to establish coordinates. Coefficients and cancellation are
  // checked later in the fully lowered graph, not in the axis-free skeleton.
  std::map<std::string, std::vector<const Term*>> left;
  for (const auto& t : lt) left[footprint(t)].push_back(&t);
  std::vector<std::pair<const Term*, const Term*>> pairs;
  auto bindReads = [&](const std::vector<E>& reads, const Term& target) {
    std::map<std::size_t, S> axes;
    for (const auto& r : reads) {
      auto frame = r->access && r->access->kind == IndexExpr::Kind::Coordinates
                       ? r->access->axes
                       : r->axes;
      auto [it, inserted] = axes.emplace(r->capture, frame);
      if (!inserted && !c.entails(call("==", {it->second, frame})))
        throw ProofFailure(
            "ambiguous tensor access correspondence is unsupported");
    }
    for (const auto& r : target.reads) {
      if (!axes.contains(r->capture)) continue;
      auto [it, inserted] =
          result.anchors.emplace(r->occurrence, axes.at(r->capture));
      if (!inserted) c.require(call("==", {it->second, axes.at(r->capture)}));
    }
  };
  for (const auto& t : rt) {
    c.tick();
    auto k = footprint(t);
    if (!left.contains(k)) continue;
    std::vector<E> reads;
    for (const auto* source : left.at(k))
      reads.insert(reads.end(), source->reads.begin(), source->reads.end());
    bindReads(reads, t);
    const auto* source = left.at(k).front();
    for (std::size_t i = 0; i < t.reads.size(); ++i)
      if (t.reads[i]->capture != source->reads[i]->capture)
        c.note("commutative-matching");
    if (pairs.size() < lt.size() && k != footprint(lt[pairs.size()]))
      c.note("commutative-matching");
    pairs.emplace_back(source, &t);
  }
  // Shape-only operands and canceled subtrees still have coordinate interfaces.
  // Fall back to source occurrences only if their access frame is unambiguous.
  std::vector<E> source_reads;
  for (const auto& [id, axes] : lhs.occurrence_axes)
    source_reads.push_back(std::make_shared<Expr>(
        Expr{Expr::Kind::Read, c.plan.nodes[id].capture->value, id, axes, {}}));
  std::function<void(std::size_t)> fill = [&](std::size_t id) {
    const auto& n = c.plan.nodes[id];
    if (n.capture && !result.anchors.contains(id)) {
      std::vector<E> reads;
      for (const auto& r : source_reads)
        if (r->capture == n.capture->value) reads.push_back(r);
      Term target{call("empty_axes", {}),
                  {std::make_shared<Expr>(
                      Expr{Expr::Kind::Read, n.capture->value, id, {}, {}})}};
      bindReads(reads, target);
      if (!result.anchors.contains(id))
        throw ProofFailure(
            "shape-only capture has no established source interface");
    }
    for (auto child : n.inputs) fill(child);
  };
  fill(c.plan.rhs);
  bool has_domains = false;
  for (auto id : c.source().nodes)
    if (const auto& definition = c.plan.nodes[id].definition)
      for (const auto& group : definition->groups)
        has_domains |= bool(group.domain);
  auto independent_axes = has_domains
                              ? c.difference(lhs.axes, freeAxes(lhs.value))
                              : call("empty_axes", {});
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
            S dependencies = result.anchors.at(r->occurrence);
            for (const auto& sr : source->reads)
              if (sr->capture == r->capture && sr->access) {
                dependencies = freeAxes(sr);
                break;
              }
            (local ? inside : outside).push_back(dependencies);
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
        if (!boundary) {
          std::vector<S> interfaces;
          for (auto leaf : leaves)
            interfaces.push_back(result.anchors.at(leaf));
          boundary = call("union_axes", std::move(interfaces));
        }
        if (has_domains && n.definition)
          boundary = call("union_axes", {boundary, independent_axes});
        result.boundaries[id] = boundary;
        std::vector<S> input_frames;
        for (auto leaf : leaves)
          input_frames.push_back(result.anchors.at(leaf));
        result.interfaces[id] =
            id == c.plan.rhs
                ? lhs.axes
                : call("concat",
                       {c.filter(lhs.axes, boundary),
                        c.filter(call("union_axes", std::move(input_frames)),
                                 c.difference(boundary, lhs.axes))});
        return leaves;
      };
  visit(c.plan.rhs);
  return result;
}
}  // namespace tepl::core::semantics
