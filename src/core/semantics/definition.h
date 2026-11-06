#pragma once
#include "src/core/semantics/scalar.h"
#include "src/shared/metadata/axes.h"
namespace tepl::core::semantics {
using axis::Binding;
using axis::Group;
using axis::Meta;
using axis::MetaPtr;
using axis::Type;
// Definition-time coordinate tuples, separate from scalar tensor values.
struct IndexValue;
using IndexValuePtr = std::shared_ptr<const IndexValue>;
struct IndexValue {
  enum class Kind {
    Coordinates,
    Add,
    Concat,
    Scale,
    Offset,
    Select,
    Reverse
  } kind;
  std::size_t group = 0;
  std::vector<IndexValuePtr> args;
  MetaPtr parameter;  // Coefficients, offsets, or selectors; no tensor values.
};
struct Value;
using ValuePtr = std::shared_ptr<const Value>;
struct Value {
  enum class Kind { Operand, Read, Add, Product, Reduce, Const, Call } kind;
  std::size_t index = 0;  // operand or reduction group
  std::vector<ValuePtr> args;
  SourceOrigin origin;
  std::int64_t constant = 0;
  IndexValuePtr access;
  ScalarFunction function = ScalarFunction::Invalid;
  std::vector<MetaPtr> parameters;
};
struct Definition : axis::ShapeContract {
  ValuePtr value;
};
std::string format(const Meta&);
std::string format(const Value&, const Definition&);
std::string format(const Definition&);
}  // namespace tepl::core::semantics
