#include "src/simple_rewrite.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "src/parse.h"
#include "src/semantic.h"

namespace {

void check(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

tepl::ast::Rule parseRule(std::string_view source) {
  auto result = tepl::parse(source);
  check(result.ok() && result.program && result.program->rules.size() == 1,
        "rule fixture must parse");
  return std::move(result.program->rules.front());
}

tepl::ast::GraphExprPtr parseInput(std::string_view expression) {
  auto source = "rule input { " + std::string(expression) + " => X }";
  return parseRule(source).lhs;
}

void testSwap() {
  auto rule = parseRule("rule swap_add { (add X Y) => (add Y X) }");
  check(tepl::validateSimpleRule(rule).empty(), "swap rule must be valid");
  auto input = parseInput("(add A (mul B C))");
  auto result = tepl::rewriteOnce(rule, *input);
  check(result.has_value(), "swap rule must match");

  const auto& output = std::get<tepl::ast::Operator>((*result)->value);
  check(output.name == "add" && output.operands.size() == 2,
        "replacement must be add");
  const auto& first = std::get<tepl::ast::Operator>(output.operands[0]->value);
  check(
      first.name == "mul" && first.operands.size() == 2 &&
          std::get<tepl::ast::NameRef>(first.operands[0]->value).name == "B" &&
          std::get<tepl::ast::NameRef>(first.operands[1]->value).name == "C" &&
          std::get<tepl::ast::NameRef>(output.operands[1]->value).name == "A",
      "replacement must swap captured subtrees");

  const auto& original = std::get<tepl::ast::Operator>(input->value);
  check(std::get<tepl::ast::NameRef>(original.operands[0]->value).name == "A",
        "rewriting must not mutate the input");
  check(output.operands[0] != original.operands[1],
        "replacement must own a copy of captured subtrees");
}

void testMismatch() {
  auto rule = parseRule("rule swap_add { (add X Y) => (add Y X) }");
  check(!tepl::rewriteOnce(rule, *parseInput("(mul A B)")),
        "different operator must not match");
  check(!tepl::rewriteOnce(rule, *parseInput("(add A B C)")),
        "different operand count must not match");
}

void testRepeatedVariable() {
  auto rule = parseRule("rule deduplicate { (add X X) => X }");
  check(tepl::rewriteOnce(rule, *parseInput("(add (mul A B) (mul A B))"))
            .has_value(),
        "equal subtrees must match repeated variables");
  check(!tepl::rewriteOnce(rule, *parseInput("(add (mul A B) (mul A C))")),
        "unequal subtrees must not match repeated variables");
}

void testInvalidRule() {
  auto rule = parseRule("rule invalid { (add X Y) => (add Y Z) }");
  auto diagnostics = tepl::validateSimpleRule(rule);
  check(diagnostics.size() == 1 &&
            diagnostics.front().message.find("'Z'") != std::string::npos &&
            diagnostics.front().span.begin.line == 1,
        "unknown RHS variable must have a source location");
  try {
    tepl::rewriteOnce(rule, *parseInput("(add A B)"));
    check(false, "invalid rule must be rejected before matching");
  } catch (const std::invalid_argument&) {
  }

  auto unsupported = parseRule("rule shaped { X: [M] X => X }");
  check(!tepl::validateSimpleRule(unsupported).empty(),
        "unsupported shape constraints must not be ignored");

  auto conditional = parseRule("rule conditional { X => X where { true; } }");
  check(!tepl::validateSimpleRule(conditional).empty(),
        "unsupported conditions must not be ignored");
}

}  // namespace

int main() {
  testSwap();
  testMismatch();
  testRepeatedVariable();
  testInvalidRule();
}
