#include "src/rewrite/operators.h"

#include <cmath>

#include "src/rewrite/evaluation_support.h"
namespace tepl::rewrite {
namespace {
using detail::arithmetic;
using detail::ensure;
Value numeric(std::string_view op, const std::vector<Value>& a) {
  if (std::holds_alternative<double>(a[0].data)) {
    double l = std::get<double>(a[0].data),
           r = a.size() == 1 ? 0 : std::get<double>(a[1].data), n = 0;
    if (op == "<") return l < r;
    if (op == "<=") return l <= r;
    if (op == ">") return l > r;
    if (op == ">=") return l >= r;
    if (a.size() == 1)
      n = op == "-" ? -l : l;
    else if (op == "+")
      n = l + r;
    else if (op == "-")
      n = l - r;
    else if (op == "*")
      n = l * r;
    else if (op == "/")
      n = l / r;
    else
      throw EvaluationError("unsupported floating operator");
    ensure(std::isfinite(n), "nonfinite floating value");
    return n;
  }
  auto l = integer(a[0]);
  if (a.size() == 1) return op == "-" ? arithmetic("-", 0, l) : l;
  auto r = integer(a[1]);
  if (op == "<") return l < r;
  if (op == "<=") return l <= r;
  if (op == ">") return l > r;
  if (op == ">=") return l >= r;
  return arithmetic(op, l, r);
}
}  // namespace
Value evaluateOperator(std::string_view op, const std::vector<Value>& args,
                       Budget& budget) {
  detail::chargeArguments(args, budget);
  if (op == "==") {
    ensure(args.size() == 2, "invalid operator arity");
    return args[0] == args[1];
  }
  if (op == "!=") {
    ensure(args.size() == 2, "invalid operator arity");
    return args[0] != args[1];
  }
  if (op == "!") {
    ensure(args.size() == 1, "invalid operator arity");
    return !boolean(args[0]);
  }
  ensure(args.size() == 2 || ((op == "+" || op == "-") && args.size() == 1),
         "invalid operator arity");
  ensure(op == "+" || op == "-" || op == "*" || op == "/" || op == "%" ||
             op == "<" || op == "<=" || op == ">" || op == ">=",
         "unknown operator");
  return numeric(op, args);
}
Value evaluateIndex(const Value& value, const Value& index, Budget& budget) {
  const Value args[] = {value, index};
  detail::chargeArguments(args, budget);
  const auto& xs = list(value);
  auto i = position(index);
  detail::ensure(i < xs.size(), "index out of bounds");
  return xs[i];
}
}  // namespace tepl::rewrite
