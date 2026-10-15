#pragma once
#include "src/shared/catalog/rules.h"
#include "src/shared/metadata/instruction.h"
namespace tepl::core::completed {
struct Descriptor {
  DescriptorId id;
  std::string name;
  enum class Kind { kCaptured, kDerived } kind;
  TypeId type;
  SourceOrigin origin;
};
using ::tepl::BinaryOp;
using ::tepl::UnaryOp;
struct TypedExpr;
using TypedExprPtr = std::shared_ptr<const TypedExpr>;

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
struct OptionalValue {
  TypedExprPtr value;
};
struct DescriptorConstruct {
  AttributeSchemaId schema;
  // All fields in schema order; spreads/defaults have been expanded.
  std::vector<TypedExprPtr> fields;
};
// Executable metadata expression produced by compile-time axis synthesis.
struct SymbolicExpr {
  SymbolPtr value;
};
struct TypedExpr {
  SourceOrigin origin;
  TypeId type;
  std::variant<CaptureRef, DimensionRef, DescriptorRef, DescriptorFieldRef,
               DTypeVariableRef, DType, NumericConstant, bool, BuiltinCall,
               UnaryExpr, BinaryExpr, LocalRef, LocalLet, ListExpr, IndexExpr,
               ConditionalExpr, ComprehensionExpr, StringConstant, EnumConstant,
               NoneConstant, OptionalValue, DescriptorConstruct, SymbolicExpr>
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

struct RuleData {
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
};

}  // namespace tepl::core::completed
