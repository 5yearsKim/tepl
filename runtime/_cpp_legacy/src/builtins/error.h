#pragma once

#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace @TEPL_NAMESPACE@::builtins {
// take() is internal failure propagation. Generated boundaries catch only this
// exception, so expected TEPL failures reject candidates without hiding host
// bugs.
class BuiltinError : public ::std::runtime_error {
 public:
  using ::std::runtime_error::runtime_error;
};
template <class T>
class BuiltinResult {
 public:
  BuiltinResult(T value) : value_(::std::move(value)) {}
  BuiltinResult(BuiltinError error) : value_(::std::move(error)) {}
  explicit operator bool() const { return ::std::holds_alternative<T>(value_); }
  const T& value() const& {
    if (auto error = ::std::get_if<BuiltinError>(&value_)) throw *error;
    return ::std::get<T>(value_);
  }
  T value() && {
    if (auto error = ::std::get_if<BuiltinError>(&value_)) throw *error;
    return ::std::move(::std::get<T>(value_));
  }
  const BuiltinError& error() const { return ::std::get<BuiltinError>(value_); }

 private:
  ::std::variant<T, BuiltinError> value_;
};
template <class T>
T take(BuiltinResult<T> value) {
  return ::std::move(value).value();
}
template <class T>
T require(::std::optional<T> value) {
  if (!value) throw BuiltinError("unavailable value");
  return ::std::move(*value);
}
template <class F>
auto attempt(F&& fn) -> BuiltinResult<decltype(fn())> {
  try {
    return fn();
  } catch (const BuiltinError& error) {
    return error;
  } catch (const ::std::length_error&) {
    return BuiltinError("capacity exceeded");
  } catch (const ::std::bad_alloc&) {
    return BuiltinError("capacity exceeded");
  }
}
}  // namespace @TEPL_NAMESPACE@::builtins
