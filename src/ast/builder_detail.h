#pragma once

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

}  // namespace tepl::detail
