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
#include "src/imports.h"
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
  const auto& program = *parsed.program;
  check(program.source_name == path && program.rules.size() == 1,
        "LoRA program name or rule count is wrong");
  check(program.imports.size() == 1 &&
            program.imports[0].path == "../dialects/tensor.tepl" &&
            program.imports[0].dialect == "TensorLang" &&
            program.imports[0].alias == "t" && program.uses.size() == 1 &&
            program.uses[0].alias == "t" &&
            program.uses[0].operations ==
                std::vector<std::string>({"add", "dot"}),
        "LoRA dialect import is missing");
  check(program.dialects.size() == 1 &&
            program.dialects[0].name == "TensorLang" &&
            program.dialects[0].operations.size() == 34,
        "Imported tensor operations are missing");
  const auto& operations = program.dialects[0].operations;
  const auto dot =
      std::find_if(operations.begin(), operations.end(),
                   [](const auto& op) { return op.name == "dot_general"; });
  const auto short_name =
      std::find_if(operations.begin(), operations.end(),
                   [](const auto& op) { return op.name == "dot"; });
  const auto* dot_attrs =
      dot != operations.end() && dot->attrs
          ? std::get_if<tepl::ast::InlineAttrs>(&*dot->attrs)
          : nullptr;
  check(dot != operations.end() && dot->alias == "dot" &&
            short_name == operations.end() && dot->operands.size() == 2 &&
            dot_attrs && dot_attrs->fields.size() == 4 &&
            dot_attrs->fields[0].type == "index" && dot_attrs->fields[0].list,
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
  const auto& inner_rhs_dot = as<tepl::ast::Operator>(
      final_dot.operands[0]->value, "Expected nested RHS dot");
  check(inner_rhs_dot.name == "dot" && inner_rhs_dot.attribute &&
            inner_rhs_dot.attribute->name == "xa" &&
            inner_rhs_dot.operands.size() == 2,
        "Nested dot or descriptor was not preserved");
  check(final_dot.operands[0]->span.begin.line == 17,
        "Nested expression span is wrong");

  check(rule.conditions.size() == 2 && rule.derivations.size() == 3,
        "LoRA where/derive counts are wrong");
  const auto& predicate = as<tepl::ast::Call>(rule.conditions[0]->value,
                                              "Expected broadcastable call");
  check(predicate.callee == "broadcastable" && predicate.arguments.size() == 2,
        "Broadcast predicate arguments are wrong");
  check(rule.derivations[2].target.name == "out", "Missing @out derivation");
  const auto& infer = as<tepl::ast::Call>(rule.derivations[2].value->value,
                                          "Expected infer_lora_out call");
  check(infer.callee == "infer_lora_out" && infer.arguments.size() == 5 &&
            as<tepl::ast::NameRef>(infer.arguments[0]->value, "Expected X")
                    .name == "X" &&
            as<tepl::ast::NameRef>(infer.arguments[1]->value, "Expected A")
                    .name == "A" &&
            as<tepl::ast::NameRef>(infer.arguments[2]->value, "Expected B")
                    .name == "B" &&
            as<tepl::ast::AttributeRef>(infer.arguments[3]->value,
                                        "Expected outer descriptor")
                    .name == "outer" &&
            as<tepl::ast::AttributeRef>(infer.arguments[4]->value,
                                        "Expected inner descriptor")
                    .name == "inner",
        "Derived @out must use LHS captures and descriptors");

  auto printed = tepl::formatAst(program);
  check(printed.find("tensor X [Batch..., M, K]") != std::string::npos &&
            printed.find("let XA") == std::string::npos &&
            printed.find("infer_lora_out") != std::string::npos &&
            printed.find("derive @out") != std::string::npos,
        "Formatted AST is missing LoRA structure");
}

void testExpressionsAndSpans() {
  auto parsed = tepl::parse(
      "rule r {\n"
      "  X: [M, Tail..., _]\n"
      "  S: scalar\n"
      "  (get[000] (tuple (let Y = (dot[@d] X S)) Y)) => Y\n"
      "  where { K + 2 * N >= 128 && !false || true;\n"
      "          -(M + N) < 0; f(X, @d, Y, 999999999999999999999999); }\n"
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
            as<tepl::ast::NameRef>(tuple.operands[1]->value,
                                   "Expected bound name reference")
                    .name == "Y",
        "Tuple or bound name reference is wrong");

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

void testNumericLiterals() {
  auto parsed = tepl::parse(
      "rule literals {\n"
      "  (add 001 -1.2500) => (add +2 999999999999999999999999)\n"
      "  where { check(1.00, -2.5); }\n"
      "}");
  check(parsed.ok() && parsed.program, "Numeric literals must parse");
  const auto& rule = parsed.program->rules.front();
  const auto& lhs = as<tepl::ast::Operator>(rule.lhs->value, "Expected add");
  check(
      as<tepl::ast::IntegerLiteral>(lhs.operands[0]->value, "Expected integer")
                  .digits == "001" &&
          as<tepl::ast::FloatLiteral>(lhs.operands[1]->value, "Expected float")
                  .digits == "-1.2500",
      "Graph literals must preserve kind, sign, and spelling");
  check(lhs.operands[0]->span.begin.line == 2 &&
            lhs.operands[0]->span.begin.column == 8 &&
            lhs.operands[0]->span.end.column == 11 &&
            lhs.operands[1]->span.begin.column == 12 &&
            lhs.operands[1]->span.end.column == 19,
        "Numeric literal spans must include the sign and full spelling");
  const auto& rhs =
      as<tepl::ast::Operator>(rule.rhs->value, "Expected RHS add");
  check(as<tepl::ast::IntegerLiteral>(rhs.operands[0]->value,
                                      "Expected signed integer")
                    .digits == "+2" &&
            as<tepl::ast::IntegerLiteral>(rhs.operands[1]->value,
                                          "Expected large integer")
                    .digits == "999999999999999999999999",
        "Parsing literals must not impose machine numeric ranges");
  const auto& call =
      as<tepl::ast::Call>(rule.conditions[0]->value, "Expected host call");
  check(as<tepl::ast::FloatLiteral>(call.arguments[0]->value,
                                    "Expected host float")
                .digits == "1.00",
        "Host expressions must preserve decimal literals");
  const auto& negative = as<tepl::ast::UnaryExpr>(call.arguments[1]->value,
                                                  "Expected host unary minus");
  check(negative.op == tepl::ast::UnaryOp::kNegate &&
            as<tepl::ast::FloatLiteral>(negative.operand->value,
                                        "Expected negative host float")
                    .digits == "2.5",
        "Constraint signs must retain unary expression semantics");
  const auto printed = tepl::formatAst(*parsed.program);
  check(printed.find("integer 001") != std::string::npos &&
            printed.find("float -1.2500") != std::string::npos &&
            printed.find("float 1.00") != std::string::npos,
        "AST printing must include numeric literals");
  auto dialect = tepl::parse(
      "dialect t { op add(lhs: tensor, rhs: tensor) -> tensor; } "
      "rule r { (add X 1.0) => (add 1.0 X) }");
  check(dialect.ok() && dialect.program,
        "Numeric operands must parse with dialect declarations");
}

void testAttributeTypeAndDefault() {
  auto parsed = tepl::parse(
      "dialect t { op test(input: tensor) -> tensor { "
      "attrs { axis: index = []; shape: index[] = []; } } }",
      "attributes.tepl");
  check(parsed.ok() && parsed.program.has_value(),
        "Attribute fixture must build an AST");
  const auto& op = parsed.program->dialects.front().operations.front();
  const auto* attrs = std::get_if<tepl::ast::InlineAttrs>(&*op.attrs);
  check(attrs && attrs->fields.size() == 2 && !attrs->fields[0].list &&
            attrs->fields[0].empty_default && attrs->fields[1].list &&
            attrs->fields[1].empty_default,
        "Attribute list type must be independent of its default");
}

void testAbstractAndInherited(const std::string& abstract_path,
                              const std::string& inherited_path) {
  const auto read = [](const std::string& path) {
    std::ifstream input(path);
    check(static_cast<bool>(input), "Cannot read rule fixture");
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  };
  auto templates = tepl::parse(read(abstract_path), abstract_path);
  check(templates.ok() && templates.rule_count == 3,
        "Abstract fixture must parse all three templates");
  const auto& commute = templates.program->rules.front();
  check(commute.is_abstract && !commute.inheritance && commute.lhs &&
            commute.rhs && commute.parameters.size() == 1 &&
            commute.source_name == abstract_path,
        "Abstract rule structure is wrong");
  const auto& parameter = commute.parameters.front();
  check(parameter.name == "F" &&
            parameter.kind == tepl::ast::RuleParameterKind::kOperation &&
            parameter.operand_types.size() == 2 &&
            parameter.operand_types[0].name == "tensor" &&
            parameter.result_type.name == "tensor" &&
            parameter.span.begin.line == 2 &&
            parameter.span.begin.column == 5 && parameter.span.end.line == 2 &&
            parameter.span.end.column == 38 &&
            parameter.result_type.span.begin.column == 31,
        "Operation parameter signature or spans are wrong");
  const auto& associate = templates.program->rules[1];
  check(associate.parameters.size() == 2 && associate.conditions.size() == 1 &&
            associate.parameters[1].kind ==
                tepl::ast::RuleParameterKind::kHostFunction &&
            associate.parameters[1].operand_types.size() == 3 &&
            associate.parameters[1].result_type.name == "bool" &&
            as<tepl::ast::Call>(associate.conditions[0]->value,
                                "Expected parameter host call")
                    .callee == "allowed",
        "Host function parameter or condition is wrong");

  auto instances = tepl::parse(read(inherited_path), inherited_path);
  check(instances.ok() && instances.rule_count == 7,
        "Inherited fixture must parse all seven instances");
  auto& program = *instances.program;
  check(program.imports.size() == 2 && program.imports[0].rules.size() == 3 &&
            program.imports[0].rules[0].name == "commute" &&
            !program.imports[0].dialect && !program.imports[0].alias,
        "Selected rule imports were not preserved");
  const auto& instance = program.rules.front();
  check(!instance.is_abstract && instance.parameters.empty() && !instance.lhs &&
            !instance.rhs && instance.inheritance &&
            instance.inheritance->base == "commute" &&
            instance.inheritance->bindings.size() == 1 &&
            instance.inheritance->bindings[0].parameter == "F" &&
            instance.inheritance->bindings[0].value == "t.add" &&
            instance.inheritance->span.begin.line == 5 &&
            instance.inheritance->span.begin.column == 18 &&
            instance.inheritance->span.end.column == 44 &&
            instance.inheritance->bindings[0].span.begin.column == 34 &&
            instance.inheritance->bindings[0].span.end.column == 43,
        "Inheritance names, bindings, or spans are wrong");
  const auto& restricted = program.rules[5];
  check(
      restricted.declarations.size() == 3 &&
          restricted.conditions.size() == 1 &&
          restricted.inheritance->bindings[1].value == "can_reassociate_add" &&
          !restricted.lhs && !restricted.rhs,
      "Child restrictions must be preserved without expanding the pattern");
  check(tepl::resolveImports(program).empty() &&
            program.imported_rules.size() == 3 &&
            program.dialects.size() == 1 && program.rules.size() == 7,
        "Rule and dialect imports must load without merging root rules");
  check(tepl::resolveImports(program).empty() &&
            program.imported_rules.size() == 3,
        "Repeated import loading must not duplicate templates");
  const auto printed = tepl::formatAst(program);
  check(
      printed.find("import {commute, associate_right, distribute_left}") !=
              std::string::npos &&
          printed.find("    extends commute\n      F = t.add\n") !=
              std::string::npos &&
          printed.find(
              "parameter allowed: fn<(tensor, tensor, tensor) -> bool>") !=
              std::string::npos,
      "AST output must include imports, inheritance, and parameter signatures");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 5,
          "Expected paths to LoRA, dialect, abstract, and inherited fixtures");
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
    testNumericLiterals();
    testAttributeTypeAndDefault();
    testInvalidSource();
    testAbstractAndInherited(runfiles->Rlocation(argv[3]),
                             runfiles->Rlocation(argv[4]));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
