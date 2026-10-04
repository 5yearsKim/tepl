#pragma once

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "src/core/ids.h"
#include "src/core/shape/ir.h"
#include "src/core/type/dtype_policy.h"
#include "src/core/type/types.h"
#include "src/operators.h"
#include "src/source.h"

namespace tepl::core {

struct AttributeField {
  std::string name;
  std::string type;
  std::size_t list_depth = 0;
  bool optional = false;
  bool empty_default = false;
  SourceOrigin origin;
};

struct AttributeSchema {
  AttributeSchemaId id;
  std::string dialect;
  std::string name;
  std::vector<AttributeField> fields;
  SourceOrigin origin;
};

struct Operand {
  std::string name;
  TypeId type;
  bool variadic = false;
  SourceOrigin origin;
};

struct Operation {
  OpId id;
  std::string dialect;
  std::string name;
  std::optional<std::string> alias;
  std::vector<Operand> operands;
  TypeId result;
  std::optional<AttributeSchemaId> attributes;
  SourceOrigin origin;
  std::optional<shape::Program> shape;
  std::optional<DTypePolicy> dtype_policy;
};

struct HostFunction {
  HostFunctionId id;
  std::string name;
  Signature signature;
  // None rejects a match; failure is separate from the successful value type.
  bool fallible = true;
  SourceOrigin origin;
};

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

struct Descriptor {
  DescriptorId id;
  std::string name;
  enum class Kind { kCaptured, kDerived } kind;
  TypeId type;
  SourceOrigin origin;
};

struct ShapeElement {
  enum class Kind { kDimension, kWildcard, kSequence } kind;
  // Anonymous sequences and wildcards have no symbol.
  std::optional<DimensionId> symbol;
  SourceOrigin origin;
};

struct CaptureConstraint {
  CaptureId capture;
  std::optional<DType> dtype;
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
};
struct BuildExpr {
  SourceOrigin origin;
  TypeId type;
  std::variant<CaptureRef, GraphLiteral, BuildOperation> value;
};

using ::tepl::BinaryOp;
using ::tepl::UnaryOp;
struct TypedExpr;
using TypedExprPtr = std::shared_ptr<const TypedExpr>;
struct DimensionRef {
  DimensionId dimension;
};
struct DescriptorRef {
  DescriptorId descriptor;
};
struct NumericConstant {
  std::string spelling;
  bool decimal = false;
};
struct HostCall {
  HostFunctionId function;
  std::vector<TypedExprPtr> arguments;
};
struct BuiltinCall {
  builtins::Builtin builtin;
  std::vector<TypedExprPtr> arguments;
};
struct UnaryExpr {
  UnaryOp op;
  TypedExprPtr operand;
};
struct BinaryExpr {
  BinaryOp op;
  TypedExprPtr lhs;
  TypedExprPtr rhs;
};
struct TypedExpr {
  SourceOrigin origin;
  TypeId type;
  std::variant<CaptureRef, DimensionRef, DescriptorRef, NumericConstant, bool,
               HostCall, BuiltinCall, UnaryExpr, BinaryExpr>
      value;
};

struct Derivation {
  DescriptorId target;
  TypedExprPtr value;
  SourceOrigin origin;
};

struct Rule {
  std::string
      source_name;  // Owning instance module, including inherited rules.
  RuleId id;
  std::string name;
  SourceOrigin origin;
  std::vector<Capture> captures;
  std::vector<DimensionSymbol> dimensions;
  std::vector<Descriptor> descriptors;
  PatternPtr lhs;
  std::vector<CaptureConstraint> constraints;
  std::vector<TypedExprPtr> conditions;
  std::vector<Derivation> derivations;
  BuildExprPtr rhs;
};

// IDs index these owning tables. Rule-local IDs index their rule's tables.
// Types are concrete on successful analysis; tables may contain equal types.
struct Dialect {
  std::string name;
  SourceOrigin origin;
};

struct Program {
  std::vector<Dialect> dialects;
  std::vector<std::string> sources;
  std::vector<Type> types;
  std::vector<Operation> operations;
  std::vector<AttributeSchema> attribute_schemas;
  std::vector<HostFunction> host_functions;
  std::vector<Rule> rules;
};

}  // namespace tepl::core
