#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "src/core/ids.h"
#include "src/core/index_semantics/ir.h"
#include "src/core/metadata/ir.h"
#include "src/core/rule/attribute_holes.h"
#include "src/core/type/types.h"
#include "src/legacy_semantics/inference_ir.h"
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
  std::optional<metadata::Program> shape;
  std::optional<metadata::Program> dtype;
  std::shared_ptr<const index_semantics::Definition> semantic_definition;
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
  enum class Kind { kCaptured, kDerived, kInferred } kind;
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
struct DTypeVariableRef {
  DTypeVariableId variable;
};
struct DescriptorFieldRef {
  DescriptorId descriptor;
  AttributeSchemaId schema;
  std::size_t field;
};

// A reference to an operation occurrence, not merely its attribute schema.
struct OutputAxisGroupRef {
  DescriptorId descriptor;
  std::string group;
};

struct NumericConstant {
  std::string spelling;
  bool decimal = false;
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
struct LocalRef {
  LocalId local;
};
struct LocalLet {
  LocalId local;
  TypedExprPtr value;
};
struct ListExpr {
  std::vector<TypedExprPtr> elements;
};
struct IndexExpr {
  TypedExprPtr value, index;
};
struct ConditionalExpr {
  TypedExprPtr condition, then_value, else_value;
};
struct ComprehensionExpr {
  TypedExprPtr element;
  LocalId variable;
  TypedExprPtr iterable;
};
struct StringConstant {
  std::string value;
};
struct EnumConstant {
  std::string value;
};
struct NoneConstant {};
struct OptionalValue {
  TypedExprPtr value;
};
struct DescriptorConstruct {
  AttributeSchemaId schema;
  // All fields in schema order; spreads/defaults have been expanded.
  std::vector<TypedExprPtr> fields;
};
struct Local {
  LocalId id;
  std::string name;
  TypeId type;
};
struct TypedExpr {
  SourceOrigin origin;
  TypeId type;
  std::variant<CaptureRef, DimensionRef, DescriptorRef, DescriptorFieldRef,
               OutputAxisGroupRef, DTypeVariableRef, DType, NumericConstant,
               bool, BuiltinCall, UnaryExpr, BinaryExpr, LocalRef, LocalLet,
               ListExpr, IndexExpr, ConditionalExpr, ComprehensionExpr,
               StringConstant, EnumConstant, NoneConstant, OptionalValue,
               DescriptorConstruct>
      value;
};

// Includes statement initializers and comprehension bodies; leaf nodes have no
// children.
template <class F>
void forEachExpressionChild(const TypedExpr& expression, F&& visit) {
  std::visit(
      [&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, BuiltinCall>) {
          for (const auto& a : v.arguments) visit(*a);
        } else if constexpr (std::is_same_v<T, UnaryExpr>)
          visit(*v.operand);
        else if constexpr (std::is_same_v<T, BinaryExpr>) {
          visit(*v.lhs);
          visit(*v.rhs);
        } else if constexpr (std::is_same_v<T, LocalLet> ||
                             std::is_same_v<T, OptionalValue>)
          visit(*v.value);
        else if constexpr (std::is_same_v<T, ListExpr>) {
          for (const auto& a : v.elements) visit(*a);
        } else if constexpr (std::is_same_v<T, IndexExpr>) {
          visit(*v.value);
          visit(*v.index);
        } else if constexpr (std::is_same_v<T, ConditionalExpr>) {
          visit(*v.condition);
          visit(*v.then_value);
          visit(*v.else_value);
        } else if constexpr (std::is_same_v<T, ComprehensionExpr>) {
          visit(*v.iterable);
          visit(*v.element);
        } else if constexpr (std::is_same_v<T, DescriptorConstruct>) {
          for (const auto& a : v.fields) visit(*a);
        }
      },
      expression.value);
}

struct Derivation {
  std::optional<DescriptorId> target;
  // A missing target denotes a local let statement.
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
  // Ordered where statements: Boolean guards or LocalLet nodes.
  std::vector<TypedExprPtr> conditions;
  std::vector<Local> locals;
  std::vector<Derivation> derivations;
  BuildExprPtr rhs;
  std::vector<DTypeVariable> dtypes;
  std::vector<AttributeHole> attribute_holes;
  // Frozen codegen compatibility; compiler analysis does not populate this.
  std::optional<inference::Program> inference;
  std::shared_ptr<const index_semantics::Plan> semantic_plan;
};

// Checked concrete graphs are topologically ordered DAGs, not rewrite patterns.
struct AttributeValue {
  enum class Kind { kInteger, kBool, kString, kEnum, kList, kNone } kind;
  std::string text;
  std::vector<AttributeValue> elements;
};
struct ConcreteNode {
  enum class Kind { kInput, kLiteral, kOperation } kind;
  SourceOrigin origin;
  std::string text;
  std::optional<DType> dtype;
  std::optional<std::vector<std::uint64_t>> shape;
  std::optional<OpId> operation;
  std::vector<std::size_t> operands;
  // Schema field order, including declared defaults.
  std::vector<AttributeValue> attributes;
};
struct ConcreteGraph {
  GraphId id;
  std::string name, source_name;
  SourceOrigin origin;
  std::vector<ConcreteNode> nodes;
  std::vector<std::pair<std::string, std::size_t>> bindings;
  std::size_t root = 0;
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
  std::vector<Rule> rules;
  std::vector<ConcreteGraph> graphs;
};

}  // namespace tepl::core
