#pragma once

#include <stdexcept>
#include <string_view>
#include <utility>

#include "antlr4-runtime.h"
#include "src/core/syntax/ast.h"

namespace tepl::detail {

inline syntax::SourcePosition getPosition(const antlr4::Token* token) {
  return {token->getLine(), token->getCharPositionInLine() + 1};
}

inline syntax::SourceSpan getSpan(const antlr4::Token* start,
                                  const antlr4::Token* stop) {
  auto end = getPosition(stop);
  if (stop->getType() != antlr4::Token::EOF) {
    // Token indexes count Unicode code points, as do lexer columns.
    end.column += stop->getStopIndex() - stop->getStartIndex() + 1;
  }
  return {getPosition(start), end};
}

inline syntax::SourceSpan getSpan(const antlr4::ParserRuleContext* context) {
  return getSpan(context->getStart(), context->getStop());
}

inline syntax::BinaryOp binaryOperator(std::string_view spelling) {
  static constexpr std::pair<std::string_view, syntax::BinaryOp> operators[] = {
      {"+", syntax::BinaryOp::kAdd},
      {"-", syntax::BinaryOp::kSubtract},
      {"*", syntax::BinaryOp::kMultiply},
      {"/", syntax::BinaryOp::kDivide},
      {"%", syntax::BinaryOp::kRemainder},
      {"<", syntax::BinaryOp::kLess},
      {"<=", syntax::BinaryOp::kLessEqual},
      {">", syntax::BinaryOp::kGreater},
      {">=", syntax::BinaryOp::kGreaterEqual},
      {"==", syntax::BinaryOp::kEqual},
      {"!=", syntax::BinaryOp::kNotEqual},
      {"&&", syntax::BinaryOp::kLogicalAnd},
      {"||", syntax::BinaryOp::kLogicalOr},
  };
  for (auto [text, op] : operators) {
    if (spelling == text) return op;
  }
  throw std::logic_error("unsupported binary operator in parse tree");
}

inline syntax::UnaryOp unaryOperator(std::string_view spelling) {
  if (spelling == "+") return syntax::UnaryOp::kPlus;
  if (spelling == "-") return syntax::UnaryOp::kNegate;
  if (spelling == "!") return syntax::UnaryOp::kLogicalNot;
  throw std::logic_error("unsupported unary operator in parse tree");
}

}  // namespace tepl::detail
