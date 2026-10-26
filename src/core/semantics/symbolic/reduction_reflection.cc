#include "src/core/semantics/symbolic/reduction_reflection.h"

namespace tepl::core::semantics {
namespace {
std::vector<I> reflections(Context& c, const Term& term) {
  std::vector<I> result;
  std::set<std::pair<Key, Key>> seen;
  auto collect = [&](auto&& self, const I& index) -> void {
    c.tick();
    if (index->kind == IndexExpr::Kind::Reverse &&
        index->args[0]->kind == IndexExpr::Kind::Coordinates &&
        seen.emplace(c.key(index->args[0]->axes), c.key(index->dimensions))
            .second)
      result.push_back(index);
    for (const auto& child : index->args) self(self, child);
  };
  for (const auto& read : term.reads)
    if (read->access) collect(collect, read->access);
  return result;
}
// Reflection is a bijection only for the selected bound coordinates in their
// original finite box. Substitute the entire summand, preserving free axes.
std::optional<Term> reflect(Context& c, const Term& term, const I& reflection) {
  auto frame = reflection->args[0]->axes;
  auto selected = c.select(frame, reflection->dimensions);
  if (!c.entails(call("is_valid_axis_list",
                      {reflection->dimensions, call("len", {frame})})) ||
      !c.entails(call("distinct_axes", {frame})) ||
      !c.entails(call("subset", {selected, term.reductions})) ||
      !equalExtentArithmetic(c, reflection->axes, c.extents(frame)))
    return {};
  auto replacement = indexReverse(c, coordinates(frame), c.extents(frame),
                                  reflection->dimensions);
  auto substitute = [&](auto&& self, const E& value) -> E {
    if (value->kind == Expr::Kind::Reduce &&
        !c.entails(call("disjoint", {value->axes, frame})))
      return {};  // Never cross a nested binder.
    auto result = std::make_shared<Expr>(*value);
    if (value->kind == Expr::Kind::Read)
      result->access = indexSubstitute(
          c, value->access ? value->access : coordinates(value->axes), frame,
          replacement);
    result->args.clear();
    for (const auto& arg : value->args) {
      auto child = self(self, arg);
      if (!child) return {};
      result->args.push_back(child);
    }
    return result;
  };
  Term result = term;
  result.factors.clear();
  result.reads.clear();
  for (const auto& factor : term.factors) {
    auto value = substitute(substitute, factor);
    if (!value) return {};
    result.factors.push_back(value);
    auto reads = tensorReads(value);
    result.reads.insert(result.reads.end(), reads.begin(), reads.end());
  }
  return result;
}
// Prefer plain accesses for earlier captures. This is independent of operation
// names, descriptor names and read occurrences, and avoids searching selectors.
std::vector<std::size_t> score(const Term& term) {
  std::map<std::size_t, std::size_t> counts;
  auto count = [&](auto&& self, const I& index) -> std::size_t {
    std::size_t total = index->kind == IndexExpr::Kind::Reverse;
    for (const auto& child : index->args) total += self(self, child);
    return total;
  };
  for (const auto& read : term.reads)
    counts[read->capture] += read->access ? count(count, read->access) : 0;
  std::vector<std::size_t> result;
  for (const auto& [capture, total] : counts) result.push_back(total);
  return result;
}
}  // namespace
void normalizeReflections(Context& c, Term& term) {
  if (!c.plan.algebraic) return;
  auto best = score(term);
  std::optional<Term> normalized;
  for (const auto& reflection : reflections(c, term))
    if (auto candidate = reflect(c, term, reflection)) {
      auto next = score(*candidate);
      if (next < best) {
        best = std::move(next);
        normalized = std::move(candidate);
      }
    }
  if (normalized) {
    term = std::move(*normalized);
    c.note("reduction-reflection");
  }
}
}  // namespace tepl::core::semantics
