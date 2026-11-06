#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "src/shared/catalog/ids.h"
#include "src/shared/metadata/axes.h"
#include "src/shared/metadata/ir.h"
#include "src/shared/source/source.h"
#include "src/shared/types/operators.h"
#include "src/shared/types/types.h"

namespace tepl::core {
struct Capture {
  CaptureId id;
  std::string name;
  TypeId type;
  SourceOrigin origin;
};
struct DimensionSymbol {
  DimensionId id;
  std::string name;
  bool sequence = false;
  TypeId type;
  SourceOrigin origin;
};
struct ShapeElement {
  enum class Kind { kDimension, kWildcard, kSequence } kind;
  // Anonymous sequences and wildcards have no symbol.
  std::optional<DimensionId> symbol;
  SourceOrigin origin;
};
struct DTypeVariable {
  DTypeVariableId id;
  std::string name;
  TypeId type;
  SourceOrigin origin;
};
using DTypeConstraint = std::variant<DType, DTypeVariableId>;
struct CaptureConstraint {
  CaptureId capture;
  std::optional<DTypeConstraint> dtype;
  // Each entry is an additional runtime shape restriction. Inheritance may
  // add restrictions, including different symbolic patterns for one capture.
  std::vector<ShapeElement> shape;
  SourceOrigin origin;
};
struct GraphLiteral {
  enum class Kind { kInteger, kDecimal } kind;
  std::string spelling;
  std::optional<DType> dtype;
};
struct Pattern;
using PatternPtr = std::shared_ptr<const Pattern>;
struct CapturePattern {
  CaptureId capture;
};
struct MatchOperation {
  OpId operation;
  std::optional<DescriptorId> descriptor;
  std::vector<PatternPtr> operands;
  std::string label;
};
struct BindPattern {
  CaptureId capture;
  PatternPtr expression;
};
struct Pattern {
  SourceOrigin origin;
  TypeId type;
  std::variant<CapturePattern, GraphLiteral, MatchOperation, BindPattern> value;
};
struct BuildExpr;
using BuildExprPtr = std::shared_ptr<const BuildExpr>;
struct CaptureRef {
  CaptureId capture;
};
struct BuildOperation {
  OpId operation;
  std::optional<DescriptorId> descriptor;
  std::vector<BuildExprPtr> operands;
  std::string label;
};
struct BuildExpr {
  SourceOrigin origin;
  TypeId type;
  std::variant<CaptureRef, GraphLiteral, BuildOperation> value;
};
struct DimensionRef {
  DimensionId dimension;
};
struct DescriptorRef {
  DescriptorId descriptor;
};
struct DTypeVariableRef {
  DTypeVariableId variable;
};
struct DescriptorFieldRef {
  DescriptorId descriptor;
  AttributeSchemaId schema;
  std::size_t field;
};
struct NumericConstant {
  std::string spelling;
  bool decimal = false;
};
struct LocalRef {
  LocalId local;
};
struct StringConstant {
  std::string value;
};
struct EnumConstant {
  std::string value;
};
struct NoneConstant {};
struct Local {
  LocalId id;
  std::string name;
  TypeId type;
};
}  // namespace tepl::core
