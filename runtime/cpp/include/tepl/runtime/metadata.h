#pragma once
#include "intrinsics.h"
namespace tepl_runtime {
template <class T>
struct Fact {
  enum class Kind { Unknown, Known, Conflict };
  Kind kind = Kind::Unknown;
  T value{};
  static Fact known(T value) { return {Kind::Known, std::move(value)}; }
  static Fact conflict() { return {Kind::Conflict, {}}; }
  bool operator==(const Fact&) const = default;
  bool is_known() const { return kind == Kind::Known; }
  const T& get(const char* name) const {
    if (kind != Kind::Known)
      throw EvalError(
          kind == Kind::Conflict ? ErrorKind::Conflict : ErrorKind::Unavailable,
          name);
    return value;
  }
  bool merge(const Fact& from) {
    auto old = *this;
    if (kind == Kind::Conflict || from.kind == Kind::Conflict)
      *this = conflict();
    else if (kind == Kind::Unknown)
      *this = from;
    else if (from.is_known() && value != from.value)
      *this = conflict();
    return *this != old;
  }
  bool disagrees(const Fact& other) const {
    return is_known() && other.is_known() && value != other.value;
  }
};
struct TensorFacts {
  Fact<std::vector<std::uint64_t>> shape;
  Fact<DType> dtype;
  static TensorFacts known(std::vector<std::uint64_t> shape, DType dtype) {
    return {Fact<std::vector<std::uint64_t>>::known(std::move(shape)),
            Fact<DType>::known(dtype)};
  }
  bool operator==(const TensorFacts&) const = default;
  bool conflicts() const {
    return shape.kind == decltype(shape)::Kind::Conflict ||
           dtype.kind == decltype(dtype)::Kind::Conflict;
  }
  bool disagrees(const TensorFacts& other) const {
    return shape.disagrees(other.shape) || dtype.disagrees(other.dtype);
  }
  Value shape_value() const { return tepl_runtime::shape(shape.get("shape")); }
  Value dtype_value() const { return to_value(dtype.get("dtype")); }
};
namespace metadata {
struct Context {
  const std::vector<TensorFacts>& operands;
  const Value& attrs;
  Budget& budget;
  std::map<std::size_t, Value> locals;
  Context(const std::vector<TensorFacts>& operands, const Value& attrs,
          Budget& budget)
      : operands(operands), attrs(attrs), budget(budget) {}
  void parameter(std::size_t symbol, std::size_t operand, bool variadic,
                 bool dtype) {
    auto input = [&](std::size_t i) {
      require(i < operands.size(), "missing operand");
      return dtype ? operands[i].dtype_value() : operands[i].shape_value();
    };
    if (variadic) {
      Value::List out;
      for (auto i = operand; i < operands.size(); ++i) out.push_back(input(i));
      locals[symbol] = Value(out);
    } else
      locals[symbol] = input(operand);
  }
  Value local(std::size_t id) const {
    auto i = locals.find(id);
    require(i != locals.end(), "missing metadata local");
    return i->second;
  }
  Value field(std::size_t id) const { return attrs.field(id); }
};
}  // namespace metadata
}  // namespace tepl_runtime
