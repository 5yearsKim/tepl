#include "src/ast/ast.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include "rules_cc/cc/runfiles/runfiles.h"
#include "src/ast/print.h"
#include "src/parse.h"

namespace {

void check(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Node, typename... Alternatives>
const Node& as(const std::variant<Alternatives...>& value,
               std::string_view message) {
  auto* node = std::get_if<Node>(&value);
  check(node != nullptr, message);
  return *node;
}

void testLora(const std::string& source, const std::string& path) {
  auto parsed = tepl::parse(source, path);
  check(parsed.ok() && parsed.program.has_value(), "LoRA must build an AST");
  check(tepl::resolveImports(*parsed.program).empty(),
        "LoRA dialect import must resolve");
  check(tepl::validateDialectUses(*parsed.program).empty(),
        "LoRA operations must match the dialect");
  const auto& program = *parsed.program;
  check(program.source_name == path && program.rules.size() == 1,
        "LoRA program name or rule count is wrong");
  check(program.imports.size() == 1 &&
            program.imports[0].path == "dialects/tensor.tepl" &&
            program.imports[0].dialect == "TensorLang" &&
            program.imports[0].alias == "t" && program.uses.size() == 1 &&
            program.uses[0].alias == "t" &&
            program.uses[0].operations ==
                std::vector<std::string>({"add", "dot"}),
        "LoRA dialect import is missing");
  check(program.dialects.size() == 1 &&
            program.dialects[0].name == "TensorLang" &&
            program.dialects[0].operations.size() == 33,
        "Imported tensor operations are missing");
  const auto& operations = program.dialects[0].operations;
  const auto dot =
      std::find_if(operations.begin(), operations.end(),
                   [](const auto& op) { return op.name == "dot_general"; });
  check(dot != operations.end() && dot->alias == "dot" &&
            dot->operands.size() == 2 && dot->attrs.size() == 4 &&
            dot->attrs[0].type == "index" && dot->attrs[0].list,
        "Dot signature or attributes were not preserved");
  const auto concat =
      std::find_if(operations.begin(), operations.end(),
                   [](const auto& op) { return op.name == "concatenate"; });
  check(concat != operations.end() && concat->operands.size() == 3 &&
            concat->operands.back().variadic,
        "Variadic concatenation operand was not preserved");
  check(program.dialects[0].schemas.size() == 1 &&
            program.dialects[0].schemas[0].fields[0].type == "string",
        "Shared string attributes were not preserved");

  const auto& rule = program.rules.front();
  check(rule.name == "lora" && rule.declarations.size() == 4,
        "LoRA declarations are missing");
  const auto& x = as<tepl::ast::TensorDecl>(rule.declarations[0].value,
                                            "X should be a tensor");
  check(x.name == "X" && x.shape.size() == 3, "X shape is wrong");
  const auto& batch = as<tepl::ast::SequenceDimension>(
      x.shape[0].value, "X should begin with a dimension sequence");
  check(batch.name == "Batch", "Batch sequence was not preserved");
  check(
      as<tepl::ast::NamedDimension>(x.shape[1].value, "Expected M").name ==
              "M" &&
          as<tepl::ast::NamedDimension>(x.shape[2].value, "Expected K").name ==
              "K",
      "Dimensions must preserve source order");
  check(x.shape[0].span.begin.line == 5 && x.shape[0].span.begin.column == 9 &&
            x.shape[0].span.end.column == 17,
        "Named sequence source span is wrong");

  const auto& lhs = as<tepl::ast::Operator>(rule.lhs->value, "LHS is not dot");
  check(lhs.name == "dot" && lhs.attribute && lhs.attribute->name == "outer" &&
            lhs.operands.size() == 2,
        "LHS dot or descriptor is wrong");
  const auto& rhs = as<tepl::ast::Operator>(rule.rhs->value, "RHS is not add");
  check(rhs.name == "add" && rhs.operands.size() == 2, "RHS addition is wrong");
  const auto& final_dot = as<tepl::ast::Operator>(
      rhs.operands[1]->value, "Second add operand is not dot");
  check(final_dot.name == "dot" && final_dot.attribute &&
            final_dot.attribute->name == "out" &&
            final_dot.operands.size() == 2,
        "Final dot or derived descriptor is wrong");
  const auto& binding = as<tepl::ast::Binding>(final_dot.operands[0]->value,
                                               "Expected RHS binding");
  const auto& bound_dot = as<tepl::ast::Operator>(
      binding.expression->value, "Binding should contain a dot");
  check(binding.binder.name == "XA" && bound_dot.attribute &&
            bound_dot.attribute->name == "xa",
        "Intermediate ?XA was not preserved");
  check(binding.expression->span.begin.line == 17 &&
            final_dot.operands[0]->span.begin.line == 17,
        "Nested expression spans are wrong");

  check(rule.conditions.size() == 2 && rule.derivations.size() == 3,
        "LoRA where/derive counts are wrong");
  const auto& predicate = as<tepl::ast::Call>(rule.conditions[0]->value,
                                              "Expected broadcastable call");
  check(predicate.callee == "broadcastable" && predicate.arguments.size() == 2,
        "Broadcast predicate arguments are wrong");
  check(rule.derivations[2].target.name == "out", "Missing @out derivation");
  const auto& infer = as<tepl::ast::Call>(rule.derivations[2].value->value,
                                          "Expected infer_dot call");
  check(infer.callee == "infer_dot" && infer.arguments.size() == 3 &&
            as<tepl::ast::BinderRef>(infer.arguments[0]->value,
                                     "Expected ?XA in derivation")
                    .name == "XA",
        "Derived @out must reference ?XA");

  auto printed = tepl::formatAst(program);
  check(printed.find("tensor X [Batch..., M, K]") != std::string::npos &&
            printed.find("bind ?XA") != std::string::npos &&
            printed.find("derive @out") != std::string::npos,
        "Formatted AST is missing LoRA structure");
}

void testExpressionsAndSpans() {
  auto parsed = tepl::parse(
      "rule r {\n"
      "  X: [M, Tail..., _]\n"
      "  S: scalar\n"
      "  (get[000] (tuple (?Y = (dot[@d] X S)) ?Y)) => ?Y\n"
      "  where { K + 2 * N >= 128 && !false || true;\n"
      "          -(M + N) < 0; f(X, @d, ?Y, 999999999999999999999999); }\n"
      "}\n",
      "math.tepl");
  check(parsed.ok() && parsed.program.has_value(), "Expression fixture failed");
  const auto& rule = parsed.program->rules.front();
  check(rule.span.begin.line == 1 && rule.span.begin.column == 1 &&
            rule.declarations[1].span.begin.line == 3 &&
            rule.lhs->span.begin.line == 4 && rule.rhs->span.begin.line == 4,
        "Rule or expression source spans are wrong");
  check(as<tepl::ast::ScalarDecl>(rule.declarations[1].value,
                                  "S should be scalar")
                .name == "S",
        "Scalar declaration is wrong");
  const auto& shape = as<tepl::ast::TensorDecl>(rule.declarations[0].value,
                                                "X should be tensor")
                          .shape;
  check(
      shape.size() == 3 &&
          as<tepl::ast::SequenceDimension>(shape[1].value,
                                           "Expected middle sequence")
                  .name == "Tail" &&
          std::holds_alternative<tepl::ast::WildcardDimension>(shape[2].value),
      "Shape dimension order or types are wrong");
  const auto& projection =
      as<tepl::ast::Projection>(rule.lhs->value, "Expected tuple projection");
  check(projection.index.digits == "000", "Projection index spelling changed");
  const auto& tuple = as<tepl::ast::Operator>(projection.tuple->value,
                                              "Expected tuple operation");
  check(tuple.name == "tuple" && tuple.operands.size() == 2 &&
            as<tepl::ast::BinderRef>(tuple.operands[1]->value,
                                     "Expected binder reference")
                    .name == "Y",
        "Tuple or binder reference is wrong");

  const auto& logical_or = as<tepl::ast::BinaryExpr>(
      rule.conditions[0]->value, "Expected outer logical OR");
  check(logical_or.op == tepl::ast::BinaryOp::kLogicalOr,
        "Logical OR must have lowest precedence");
  const auto& logical_and =
      as<tepl::ast::BinaryExpr>(logical_or.lhs->value, "Expected logical AND");
  check(logical_and.op == tepl::ast::BinaryOp::kLogicalAnd,
        "Logical AND precedence is wrong");
  const auto& comparison =
      as<tepl::ast::BinaryExpr>(logical_and.lhs->value, "Expected comparison");
  check(comparison.op == tepl::ast::BinaryOp::kGreaterEqual,
        "Comparison precedence is wrong");
  const auto& addition =
      as<tepl::ast::BinaryExpr>(comparison.lhs->value, "Expected addition");
  check(addition.op == tepl::ast::BinaryOp::kAdd &&
            as<tepl::ast::BinaryExpr>(addition.rhs->value,
                                      "Expected multiplication")
                    .op == tepl::ast::BinaryOp::kMultiply,
        "Multiplication must bind more tightly than addition");
  check(as<tepl::ast::UnaryExpr>(logical_and.rhs->value,
                                 "Expected logical negation")
                    .op == tepl::ast::UnaryOp::kLogicalNot &&
            as<tepl::ast::BooleanLiteral>(logical_or.rhs->value,
                                          "Expected boolean literal")
                .value,
        "Unary or boolean nodes are wrong");

  const auto& second_comparison = as<tepl::ast::BinaryExpr>(
      rule.conditions[1]->value, "Expected less-than comparison");
  const auto& negated = as<tepl::ast::UnaryExpr>(second_comparison.lhs->value,
                                                 "Expected unary minus");
  const auto& grouped = as<tepl::ast::BinaryExpr>(negated.operand->value,
                                                  "Expected grouped addition");
  check(negated.op == tepl::ast::UnaryOp::kNegate &&
            grouped.op == tepl::ast::BinaryOp::kAdd &&
            negated.operand->span.begin.column == 12,
        "Grouped expression or its source span is wrong");
  const auto& call =
      as<tepl::ast::Call>(rule.conditions[2]->value, "Expected host call");
  check(call.callee == "f" && call.arguments.size() == 4 &&
            as<tepl::ast::IntegerLiteral>(call.arguments[3]->value,
                                          "Expected large integer")
                    .digits == "999999999999999999999999",
        "Call arguments or integer spelling are wrong");
}

void testInvalidSource() {
  auto invalid = tepl::parse("rule r { X => }", "bad.tepl");
  check(!invalid.ok() && !invalid.program.has_value(),
        "Invalid syntax must not expose an AST");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 3, "Expected paths to LoRA and dialect fixtures");
    std::string error;
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
        rules_cc::cc::runfiles::Runfiles::CreateForTest(
            BAZEL_CURRENT_REPOSITORY, &error));
    check(runfiles != nullptr, "Cannot load runfiles: " + error);
    const std::string path = runfiles->Rlocation(argv[1]);
    std::ifstream input(path);
    check(static_cast<bool>(input), "Cannot read LoRA fixture");
    std::string source{std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>()};
    testLora(source, path);
    testExpressionsAndSpans();
    testInvalidSource();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
