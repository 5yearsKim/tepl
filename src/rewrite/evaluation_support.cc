#include "src/rewrite/evaluation_support.h"

#include <algorithm>
#include <limits>
namespace tepl::rewrite::detail {
void ensure(bool b, const char* text) {
  if (!b) throw EvaluationError(text);
}
Integer arithmetic(std::string_view op, Integer a, Integer b) {
  Integer out = 0;
  if (op == "+")
    ensure(!__builtin_add_overflow(a, b, &out), "integer overflow");
  else if (op == "-")
    ensure(!__builtin_sub_overflow(a, b, &out), "integer overflow");
  else if (op == "*")
    ensure(!__builtin_mul_overflow(a, b, &out), "integer overflow");
  else {
    ensure(b != 0, "division by zero");
    ensure(!(a == std::numeric_limits<Integer>::min() && b == -1),
           "integer overflow");
    out = op == "%" ? a % b : a / b;
  }
  return out;
}
bool contains(const List& xs, const Value& x, Budget& budget) {
  budget.charge(xs.size());
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}
bool distinct(const List& xs, Budget& budget) {
  for (std::size_t i = 0; i < xs.size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      budget.tick();
      if (xs[i] == xs[j]) return false;
    }
  return true;
}
bool valid(const List& xs, Integer rank, Budget& budget) {
  if (rank < 0 || static_cast<unsigned __int128>(rank) >
                      std::numeric_limits<std::uint64_t>::max())
    return false;
  if (!distinct(xs, budget)) return false;
  for (const auto& x : xs)
    if (integer(x) < 0 || integer(x) >= rank) return false;
  return true;
}
void chargeArguments(std::span<const Value> args, Budget& budget) {
  budget.tick();
  for (const auto& v : args)
    if (const auto* xs = std::get_if<List>(&v.data)) {
      budget.size(xs->size());
      budget.charge(xs->size());
    }
}
}  // namespace tepl::rewrite::detail
