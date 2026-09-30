#include "src/parse.h"

#include <exception>

#include "antlr4-runtime.h"
#include "grammar/TeplLexer.h"
#include "grammar/TeplParser.h"
#include "src/ast_builder.h"

namespace tepl {
namespace {

class ErrorListener final : public antlr4::BaseErrorListener {
 public:
  explicit ErrorListener(std::vector<Diagnostic>& diagnostics)
      : diagnostics_(diagnostics) {}

  void syntaxError(antlr4::Recognizer*, antlr4::Token*, std::size_t line,
                   std::size_t column, const std::string& message,
                   std::exception_ptr) override {
    diagnostics_.push_back({line, column + 1, message});
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

ParseResult parse(std::string_view source, std::string_view source_name) {
  ParseResult result;
  ErrorListener errors(result.diagnostics);
  antlr4::ANTLRInputStream input{std::string(source)};
  tepl_generated::TeplLexer lexer(&input);
  lexer.removeErrorListeners();
  lexer.addErrorListener(&errors);
  antlr4::CommonTokenStream tokens(&lexer);
  tepl_generated::TeplParser parser(&tokens);
  parser.removeErrorListeners();
  parser.addErrorListener(&errors);

  auto* program = parser.program();
  if (result.ok()) {
    result.rule_count = program->ruleDecl().size();
    result.tree = program->toStringTree(&parser);
    result.program = AstBuilder{}.build(program, std::string(source_name));
  }
  return result;
}

}  // namespace tepl
