#pragma once

#include <string_view>

namespace tepl {

// Shared language operators. AST and core use the same identities, so lowering
// never depends on the numerical order of two independent enumerations.
enum class UnaryOp { kPlus, kNegate, kLogicalNot };
enum class BinaryOp {
  kAdd,
  kSubtract,
  kMultiply,
  kDivide,
  kRemainder,
  kLess,
  kLessEqual,
  kGreater,
  kGreaterEqual,
  kEqual,
  kNotEqual,
  kLogicalAnd,
  kLogicalOr,
};

enum class BinaryCategory { kArithmetic, kComparison, kEquality, kLogical };

constexpr BinaryCategory category(BinaryOp op) {
  switch (op) {
    case BinaryOp::kAdd:
    case BinaryOp::kSubtract:
    case BinaryOp::kMultiply:
    case BinaryOp::kDivide:
    case BinaryOp::kRemainder:
      return BinaryCategory::kArithmetic;
    case BinaryOp::kLess:
    case BinaryOp::kLessEqual:
    case BinaryOp::kGreater:
    case BinaryOp::kGreaterEqual:
      return BinaryCategory::kComparison;
    case BinaryOp::kEqual:
    case BinaryOp::kNotEqual:
      return BinaryCategory::kEquality;
    case BinaryOp::kLogicalAnd:
    case BinaryOp::kLogicalOr:
      return BinaryCategory::kLogical;
  }
  return BinaryCategory::kArithmetic;
}

constexpr std::string_view spelling(UnaryOp op) {
  switch (op) {
    case UnaryOp::kPlus:
      return "+";
    case UnaryOp::kNegate:
      return "-";
    case UnaryOp::kLogicalNot:
      return "!";
  }
  return "<invalid unary operator>";
}

constexpr std::string_view spelling(BinaryOp op) {
  switch (op) {
    case BinaryOp::kAdd:
      return "+";
    case BinaryOp::kSubtract:
      return "-";
    case BinaryOp::kMultiply:
      return "*";
    case BinaryOp::kDivide:
      return "/";
    case BinaryOp::kRemainder:
      return "%";
    case BinaryOp::kLess:
      return "<";
    case BinaryOp::kLessEqual:
      return "<=";
    case BinaryOp::kGreater:
      return ">";
    case BinaryOp::kGreaterEqual:
      return ">=";
    case BinaryOp::kEqual:
      return "==";
    case BinaryOp::kNotEqual:
      return "!=";
    case BinaryOp::kLogicalAnd:
      return "&&";
    case BinaryOp::kLogicalOr:
      return "||";
  }
  return "<invalid binary operator>";
}

}  // namespace tepl
