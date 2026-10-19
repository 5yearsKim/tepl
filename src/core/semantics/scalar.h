#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace tepl::core::semantics {
// Compiler-side scalar functions, independent of executable metadata builtins.
enum class ScalarFunction {
  Invalid,
  Add,
  Multiply,
  Subtract,
  Negate,
  Minimum,
  Maximum,
  Exp,
  Log,
  Sqrt,
  Rsqrt,
  Abs,
  Power,
  Divide,
  Not,
  And,
  Or,
  Xor,
  Select,
  Compare,
  Clamp,
  Tanh,
  Cbrt,
  Erf,
  Logistic,
  Remainder
};
enum class ScalarTypes { Numeric, Floating, Boolean, Select, Compare };
struct ScalarSignature {
  ScalarFunction id;
  std::string_view name;
  std::size_t arity;
  ScalarTypes types;
  std::size_t parameters = 0;
  bool associative = false, commutative = false, idempotent = false;
  bool monotone = false;
};
inline constexpr auto scalarFunctions = std::to_array<ScalarSignature>({
    {ScalarFunction::Add, "add", 2, ScalarTypes::Numeric},
    {ScalarFunction::Multiply, "mul", 2, ScalarTypes::Numeric},
    {ScalarFunction::Subtract, "sub", 2, ScalarTypes::Numeric},
    {ScalarFunction::Negate, "neg", 1, ScalarTypes::Numeric},
    {ScalarFunction::Minimum, "min", 2, ScalarTypes::Numeric, 0, true, true,
     true},
    {ScalarFunction::Maximum, "max", 2, ScalarTypes::Numeric, 0, true, true,
     true},
    {ScalarFunction::Exp, "exp", 1, ScalarTypes::Floating, 0, false, false,
     false, true},
    {ScalarFunction::Log, "log", 1, ScalarTypes::Floating},
    {ScalarFunction::Sqrt, "sqrt", 1, ScalarTypes::Floating},
    {ScalarFunction::Rsqrt, "rsqrt", 1, ScalarTypes::Floating},
    {ScalarFunction::Abs, "abs", 1, ScalarTypes::Numeric},
    {ScalarFunction::Power, "pow", 2, ScalarTypes::Numeric},
    {ScalarFunction::Divide, "div", 2, ScalarTypes::Numeric},
    {ScalarFunction::Not, "not", 1, ScalarTypes::Boolean},
    {ScalarFunction::And, "and", 2, ScalarTypes::Boolean, 0, true, true, true},
    {ScalarFunction::Or, "or", 2, ScalarTypes::Boolean, 0, true, true, true},
    {ScalarFunction::Xor, "xor", 2, ScalarTypes::Boolean, 0, true, true},
    {ScalarFunction::Select, "select", 3, ScalarTypes::Select},
    {ScalarFunction::Compare, "compare", 2, ScalarTypes::Compare, 1},
    {ScalarFunction::Clamp, "clamp", 3, ScalarTypes::Numeric},
    {ScalarFunction::Tanh, "tanh", 1, ScalarTypes::Floating, 0, false, false,
     false, true},
    {ScalarFunction::Cbrt, "cbrt", 1, ScalarTypes::Floating, 0, false, false,
     false, true},
    {ScalarFunction::Erf, "erf", 1, ScalarTypes::Floating},
    {ScalarFunction::Logistic, "logistic", 1, ScalarTypes::Floating},
    {ScalarFunction::Remainder, "rem", 2, ScalarTypes::Numeric},
});
inline const ScalarSignature* scalarSignature(std::string_view name) {
  for (const auto& f : scalarFunctions)
    if (f.name == name) return &f;
  return nullptr;
}
inline const ScalarSignature& scalarSignature(ScalarFunction id) {
  for (const auto& f : scalarFunctions)
    if (f.id == id) return f;
  throw std::logic_error("invalid scalar function");
}
}  // namespace tepl::core::semantics
