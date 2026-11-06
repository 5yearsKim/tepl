#pragma once
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace tepl_runtime {
enum class ErrorKind { Unavailable, Invalid, Conflict, Limit };
struct EvalError : std::runtime_error {
  ErrorKind kind;
  EvalError(ErrorKind kind, std::string message)
      : std::runtime_error(std::move(message)), kind(kind) {}
};
inline void require(bool ok, const char* message) {
  if (!ok) throw EvalError(ErrorKind::Invalid, message);
}
// Public fallible APIs return results. Internal evaluators propagate only typed
// EvalError exceptions; allocation and application exceptions remain visible.
template <class T>
class [[nodiscard]] EvalResult {
  std::variant<T, EvalError> data_;

 public:
  EvalResult(T value) : data_(std::move(value)) {}
  EvalResult(EvalError error) : data_(std::move(error)) {}
  bool ok() const { return std::holds_alternative<T>(data_); }
  explicit operator bool() const { return ok(); }
  T& value() & {
    if (!ok()) throw error();
    return std::get<T>(data_);
  }
  const T& value() const& {
    if (!ok()) throw error();
    return std::get<T>(data_);
  }
  T value() && {
    if (!ok()) throw error();
    return std::get<T>(std::move(data_));
  }
  const EvalError& error() const { return std::get<EvalError>(data_); }
};
template <>
class [[nodiscard]] EvalResult<void> {
  std::variant<std::monostate, EvalError> data_;

 public:
  EvalResult() = default;
  EvalResult(EvalError error) : data_(std::move(error)) {}
  bool ok() const { return data_.index() == 0; }
  explicit operator bool() const { return ok(); }
  void value() const {
    if (!ok()) throw error();
  }
  const EvalError& error() const { return std::get<EvalError>(data_); }
};
template <class F>
auto evaluate(F&& f) -> EvalResult<std::invoke_result_t<F>> {
  using T = std::invoke_result_t<F>;
  try {
    if constexpr (std::is_void_v<T>) {
      f();
      return {};
    } else
      return f();
  } catch (const EvalError& error) {
    return error;
  }
}
inline constexpr unsigned RUNTIME_ABI = 1;
}  // namespace tepl_runtime
