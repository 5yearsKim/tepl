#pragma once
#include <optional>
#include <string>
#include <utility>

namespace @TEPL_NAMESPACE@::analysis {
template <class T>
struct Inference {
  enum class Kind { Known, Unknown, Invalid };
  Kind kind = Kind::Unknown;
  ::std::optional<T> value;
  ::std::string error;
  static Inference known(T value) {
    return {Kind::Known, ::std::move(value), {}};
  }
  static Inference unknown() { return {}; }
  static Inference invalid(::std::string error) {
    return {Kind::Invalid, {}, ::std::move(error)};
  }
  ::std::optional<T> into_option() const {
    return kind == Kind::Known ? value : ::std::nullopt;
  }
};
}  // namespace @TEPL_NAMESPACE@::analysis
