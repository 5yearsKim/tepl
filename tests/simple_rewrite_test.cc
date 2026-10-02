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

  for (const auto* source :
       {"abstract rule r(F: op<(tensor) -> tensor>) { (F X) => X }",
        "rule r extends base(F = t.add);"}) {
    const auto template_rule = parseRule(source);
    const auto unsupported = tepl::validateSimpleRule(template_rule);
    check(unsupported.size() == 1 &&
              unsupported[0].message.find("inheritance expansion") !=
                  std::string::npos,
          "unexpanded rules must be rejected before graph access");
    try {
      tepl::rewriteOnce(template_rule, *parseInput("(add A B)"));
      check(false, "unexpanded rules must not execute as simple rewrites");
    } catch (const std::invalid_argument&) {
    }
  }
}

void testNumericLiterals() {
  auto integer = parseRule("rule r { (add X 1) => (add 2 X) }");
  auto result = tepl::rewriteOnce(integer, *parseInput("(add A 1)"));
  check(result.has_value(), "Integer literal must match its exact value");
  const auto& output = std::get<tepl::ast::Operator>((*result)->value);
  check(std::get<tepl::ast::IntegerLiteral>(output.operands[0]->value).digits ==
                "2" &&
            std::get<tepl::ast::NameRef>(output.operands[1]->value).name == "A",
        "RHS must construct a literal and substitute captured variables");
  for (const auto* input : {"(add A 2)", "(add A 1.0)", "(add A B)"}) {
    check(!tepl::rewriteOnce(integer, *parseInput(input)),
          "A literal must not act as a variable or match a different kind");
  }
  auto decimal = parseRule("rule r { (add X -1.25) => (add +0.5 X) }");
  result = tepl::rewriteOnce(decimal, *parseInput("(add (mul A B) -1.25)"));
  check(result.has_value(), "Signed float literals must match");
  const auto& float_output = std::get<tepl::ast::Operator>((*result)->value);
  check(std::get<tepl::ast::FloatLiteral>(float_output.operands[0]->value)
                .digits == "+0.5",
        "RHS must preserve float spelling and sign");
  check(!tepl::rewriteOnce(decimal, *parseInput("(add A -1.250)")),
        "Structural literal matching must preserve exact spelling");
  auto capture = parseRule("rule r { (add X X) => X }");
  result = tepl::rewriteOnce(capture, *parseInput("(add 1.0 1.0)"));
  check(result &&
            std::get<tepl::ast::FloatLiteral>((*result)->value).digits == "1.0",
        "Variables must capture literal subtrees");
  check(!tepl::rewriteOnce(capture, *parseInput("(add 1 1.0)")),
        "Repeated variables must respect literal kind");
  auto root = parseRule("rule r { 1 => -2.5 }");
  result = tepl::rewriteOnce(root, *parseInput("1"));
  check(result && std::get<tepl::ast::FloatLiteral>((*result)->value).digits ==
                      "-2.5",
        "Literals must work as rewrite roots");
}

}  // namespace

int main() {
  testSwap();
  testMismatch();
  testRepeatedVariable();
  testNumericLiterals();
  testInvalidRule();
}
