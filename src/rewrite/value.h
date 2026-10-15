#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "src/core/ir.h"

namespace tepl::rewrite {
using Integer = __int128;
struct EvaluationError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct LimitError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Axis {
  std::size_t owner, group, member;
  Integer extent;
  bool operator==(const Axis&) const = default;
};
struct NodeRef {
  std::size_t id;
  bool operator==(const NodeRef&) const = default;
};
struct Value;
struct DescriptorValue {
  std::size_t schema;
  std::vector<Value> fields;
  bool operator==(const DescriptorValue&) const = default;
};
struct Value {
  using List = std::vector<Value>;
  std::variant<std::monostate, Integer, double, bool, std::string, List, Axis,
               NodeRef, DescriptorValue>
      data;
  Value() = default;
  template <class T>
  Value(T v) : data(std::move(v)) {}
  bool operator==(const Value&) const = default;
};
using List = Value::List;
Integer integer(const Value&);
bool boolean(const Value&);
const List& list(const Value&);
std::size_t position(const Value&);
Integer parseInteger(const std::string&);
std::string integerText(Integer);
Value attributeValue(const core::AttributeValue&);
core::AttributeValue attribute(const Value&, const core::AttributeField&);
struct Budget {
  std::size_t steps = 1000000, max_list = 1000000;
  void charge(std::size_t n) {
    if (n > steps) throw LimitError("expression step limit");
    steps -= n;
  }
  void tick() { charge(1); }
  void size(std::size_t n) const {
    if (n > max_list) throw LimitError("list size limit");
  }
};
Value checkType(Value, const core::Type&);
}  // namespace tepl::rewrite
