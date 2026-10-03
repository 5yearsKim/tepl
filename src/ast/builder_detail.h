#pragma once

#include <stdexcept>
#include <string_view>
#include <utility>

#include "antlr4-runtime.h"
#include "src/ast/ast.h"

namespace tepl::detail {

inline ast::SourcePosition getPosition(const antlr4::Token* token) {
  return {token->getLine(), token->getCharPositionInLine() + 1};
}

inline ast::SourceSpan getSpan(const antlr4::Token* start,
                               const antlr4::Token* stop) {
  auto end = getPosition(stop);
  if (stop->getType() != antlr4::Token::EOF) {
    // Token indexes count Unicode code points, as do lexer columns.
    end.column += stop->getStopIndex() - stop->getStartIndex() + 1;
  }
  return {getPosition(start), end};
}

inline ast::SourceSpan getSpan(const antlr4::ParserRuleContext* context) {
  return getSpan(context->getStart(), context->getStop());
}

inline ast::BinaryOp binaryOperator(std::string_view spelling) {
  static constexpr std::pair<std::string_view, ast::BinaryOp> operators[] = {
      {"+", ast::BinaryOp::kAdd},           {"-", ast::BinaryOp::kSubtract},
      {"*", ast::BinaryOp::kMultiply},      {"/", ast::BinaryOp::kDivide},
      {"%", ast::BinaryOp::kRemainder},     {"<", ast::BinaryOp::kLess},
      {"<=", ast::BinaryOp::kLessEqual},    {">", ast::BinaryOp::kGreater},
      {">=", ast::BinaryOp::kGreaterEqual}, {"==", ast::BinaryOp::kEqual},
      {"!=", ast::BinaryOp::kNotEqual},     {"&&", ast::BinaryOp::kLogicalAnd},
      {"||", ast::BinaryOp::kLogicalOr},
  };
  for (auto [text, op] : operators) {
    if (spelling == text) return op;
  }
  throw std::logic_error("unsupported binary operator in parse tree");
}

inline ast::UnaryOp unaryOperator(std::string_view spelling) {
  if (spelling == "+") return ast::UnaryOp::kPlus;
  if (spelling == "-") return ast::UnaryOp::kNegate;
  if (spelling == "!") return ast::UnaryOp::kLogicalNot;
  throw std::logic_error("unsupported unary operator in parse tree");
}

}  // namespace tepl::detail
