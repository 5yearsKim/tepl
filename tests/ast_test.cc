#include "src/core/syntax/ast.h"

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
#include "src/core/resolution/loading.h"
#include "src/core/syntax/parse.h"
#include "src/core/syntax/print.h"

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
  auto loaded = tepl::loadImports(*parsed.program);
  check(loaded.ok(), "LoRA dialect import must resolve");
  const auto& program = loaded.program->declarations;
  check(loaded.program->identity == path && program.rules.size() == 1,
        "LoRA program name or rule count is wrong");
  check(parsed.program->imports.size() == 1 &&
            parsed.program->imports[0].path == "dialects/tensor.tepl" &&
            parsed.program->imports[0].dialect == "TensorLang" &&
            parsed.program->imports[0].alias == "t" &&
            parsed.program->uses.size() == 1 &&
            parsed.program->uses[0].alias == "t" &&
            parsed.program->uses[0].operations ==
                std::vector<std::string>({"add", "dot"}),
        "LoRA dialect import is missing");
  check(program.dialects.size() == 1 &&
            program.dialects[0].name == "TensorLang" &&
            program.dialects[0].operations.size() == 25,
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
          ? std::get_if<tepl::syntax::InlineAttrs>(&*dot->attrs)
          : nullptr;
  check(dot != operations.end() && dot->alias == "dot" &&
            short_name == operations.end() && dot->operands.size() == 2 &&
            dot_attrs && dot_attrs->fields.size() == 6 &&
            dot_attrs->fields[0].type == "axis" &&
            dot_attrs->fields[0].list_depth,
        "Dot signature or attributes were not preserved");
  const auto concat =
      std::find_if(operations.begin(), operations.end(),
                   [](const auto& op) { return op.name == "concatenate"; });
  check(concat != operations.end() && concat->operands.size() == 1 &&
            concat->operands.back().variadic,
        "Variadic concatenation operand was not preserved");
  check(program.dialects[0].schemas.size() == 1 &&
            program.dialects[0].schemas[0].fields[0].type == "replica_groups",
        "Shared collective attributes were not preserved");

  const auto& rule = program.rules.front();
  check(rule.name == "lora" && rule.declarations.size() == 4,
        "LoRA declarations are missing");
  const auto& x = rule.declarations[0];
  check(x.name == "X" && x.shape.size() == 3, "X shape is wrong");
  const auto& batch = as<tepl::syntax::SequenceDimension>(
      x.shape[0].value, "X should begin with a dimension sequence");
  check(batch.name == "Batch", "Batch sequence was not preserved");
  for (const auto& declaration : rule.declarations)
    check(as<tepl::syntax::SequenceDimension>(declaration.shape[0].value,
                                              "Expected batch sequence")
                  .name == "Batch",
          "LoRA inputs must share their batch dimensions");
  check(as<tepl::syntax::NamedDimension>(x.shape[1].value, "Expected M").name ==
                "M" &&
            as<tepl::syntax::NamedDimension>(x.shape[2].value, "Expected K")
                    .name == "K",
        "Dimensions must preserve source order");
  check(x.shape[0].span.begin.line == 7 && x.shape[0].span.begin.column == 7 &&
            x.shape[0].span.end.column == 15,
        "Named sequence source span is wrong");

  const auto& lhs =
      as<tepl::syntax::Operator>(rule.lhs->value, "LHS is not dot");
  check(lhs.name == "dot" && lhs.label == "outer" && lhs.operands.size() == 2,
        "LHS dot or descriptor is wrong");
  const auto& rhs =
      as<tepl::syntax::Operator>(rule.rhs->value, "RHS is not add");
  check(rhs.name == "add" && rhs.operands.size() == 2, "RHS addition is wrong");
  const auto& final_dot = as<tepl::syntax::Operator>(
      rhs.operands[1]->value, "Second add operand is not dot");
  check(final_dot.name == "dot" && final_dot.descriptor_node == "inner" &&
            final_dot.operands.size() == 2,
        "Final dot must reuse inner attributes");
  const auto& inner_rhs_dot = as<tepl::syntax::Operator>(
      final_dot.operands[0]->value, "Expected nested RHS dot");
  check(inner_rhs_dot.name == "dot" &&
            inner_rhs_dot.descriptor_node == "outer" &&
            inner_rhs_dot.operands.size() == 2,
        "Nested dot or descriptor was not preserved");
  check(final_dot.operands[0]->span.begin.line == rule.rhs->span.begin.line,
        "Nested expression span is wrong");

  const auto& xw =
      as<tepl::syntax::Operator>(rhs.operands[0]->value, "Expected X W dot");
  check(xw.descriptor_node == "outer", "X W dot must reuse outer attributes");
  check(rule.conditions.size() == 1 && rule.derivations.empty(),
        "LoRA where/derive counts are wrong");
  auto printed = tepl::formatAst(*loaded.program);
  check(printed.find("operator dot [inner.attrs]") != std::string::npos,
        "Formatted AST is missing captured attribute reuse");
}

void testExtendedAttributesAndParsedShapes() {
  auto parsed = tepl::parse(R"(
dialect D {
  op custom(x: tensor) -> tensor {
    attrs { padding: coordinate[][] algorithm: dot_algorithm? flags: bool[] body: region }
    shape(s) { assert not_implemented(s) yield s }
  }
}
)");
  check(parsed.ok(), "Extended dialect metadata must parse");
  const auto& op = parsed.program->dialects[0].operations[0];
  const auto& fields = std::get<tepl::syntax::InlineAttrs>(*op.attrs).fields;
  check(fields[0].type == "coordinate" && fields[0].list_depth == 2 &&
            fields[1].optional && fields[1].type == "dot_algorithm" &&
            fields[2].list_depth == 1 && fields[3].type == "region",
        "Nested, optional and opaque attribute types must be preserved");
  check(op.shape_definition.has_value(), "Shape definition must be parsed");
  const auto& shape = *op.shape_definition;
  check(shape.parameters.size() == 1 && shape.parameters[0].name == "s" &&
            !shape.parameters[0].variadic && shape.statements.size() == 1,
        "Shape parameters and statements must be retained");
  const auto& assertion =
      std::get<tepl::syntax::MetadataAssert>(shape.statements[0].value);
  const auto& call =
      std::get<tepl::syntax::MetadataCall>(assertion.condition->value);
  check(call.callee == "not_implemented" && call.arguments.size() == 1 &&
            std::get<tepl::syntax::NameRef>(shape.result.value->value).name ==
                "s",
        "Shape calls remain unresolved syntax until core checking");
  const auto printed = tepl::formatAst(*parsed.program);
  check(printed.find("padding: coordinate[][]") != std::string::npos &&
            printed.find("algorithm: dot_algorithm?") != std::string::npos,
        "Attribute type modifiers must print without loss");
}

void testRuleConstructors() {
  auto parsed = tepl::parse(
      "rule r { (source: transpose X) => (result: transpose X) where { let "
      "rank = len([1, 2]) yield -rank "
      "< 0 } derive { result.attrs = { ..source.attrs, permutation: [i for i "
      "in range(2)] } } }");
  check(parsed.ok(), "Constructor fixture parses");
  const auto& rule = parsed.program->rules[0];
  check(rule.conditions.size() == 2,
        "Yield separates the local binding from the final condition");
  check(std::holds_alternative<tepl::syntax::LocalBinding>(
            rule.conditions[0]->value),
        "Where preserves let statements");
  const auto& condition = as<tepl::syntax::BinaryExpr>(
      rule.conditions[1]->value, "Expected yielded comparison");
  check(std::holds_alternative<tepl::syntax::UnaryExpr>(condition.lhs->value),
        "A yielded unary expression must not become part of the binding");
  const auto& ctor = as<tepl::syntax::DescriptorConstructor>(
      rule.derivations[0].value->value, "Expected constructor");
  check(ctor.fields.size() == 1 && ctor.spread && *ctor.spread == "source" &&
            rule.derivations[0].target == "result" &&
            std::holds_alternative<tepl::syntax::ComprehensionExpr>(
                ctor.fields[0].value->value),
        "Constructor preserves comprehension");
  const auto printed = tepl::formatAst(*parsed.program);
  check(printed.find("derive result.attrs") != std::string::npos &&
            printed.find("descriptor ..source.attrs") != std::string::npos,
        "Named attribute derivations and spreads print with public syntax");
}

void testExpressionsAndSpans() {
  auto parsed = tepl::parse(
      "rule r {\n  X: [M, Tail..., _]\n  S: []\n  (get[000] (tuple (let Y = "
      "(d: dot X S)) Y)) => Y\n  where (K + 2 * N >= 128 && !false || "
      "true)\n          && -(M + N) < 0 && f(X, d.attrs, Y, "
      "999999999999999999999999)\n}\n",
      "math.tepl");
  check(parsed.ok() && parsed.program.has_value(), "Expression fixture failed");
  const auto& rule = parsed.program->rules.front();
  check(rule.span.begin.line == 1 && rule.span.begin.column == 1 &&
            rule.declarations[1].span.begin.line == 3 &&
            rule.lhs->span.begin.line == 4 && rule.rhs->span.begin.line == 4,
        "Rule or expression source spans are wrong");
  const auto& scalar = rule.declarations[1];
  check(scalar.name == "S" && scalar.shape.empty() && !scalar.dtype,
        "Rank-zero tensor declaration is wrong");
  check(
      tepl::formatAst(*parsed.program).find("tensor S []") != std::string::npos,
      "Rank-zero tensor must print an empty shape");
  const auto& shape = rule.declarations[0].shape;
  check(shape.size() == 3 &&
            as<tepl::syntax::SequenceDimension>(shape[1].value,
                                                "Expected middle sequence")
                    .name == "Tail" &&
            std::holds_alternative<tepl::syntax::WildcardDimension>(
                shape[2].value),
        "Shape dimension order or types are wrong");
  const auto& projection = as<tepl::syntax::Projection>(
      rule.lhs->value, "Expected tuple projection");
  check(projection.index.digits == "000", "Projection index spelling changed");
  const auto& tuple = as<tepl::syntax::Operator>(projection.tuple->value,
                                                 "Expected tuple operation");
  check(tuple.name == "tuple" && tuple.operands.size() == 2 &&
            as<tepl::syntax::NameRef>(tuple.operands[1]->value,
                                      "Expected bound name reference")
                    .name == "Y",
        "Tuple or bound name reference is wrong");

  check(rule.conditions.size() == 1,
        "Where must preserve one final boolean expression");
  const auto& final_and = as<tepl::syntax::BinaryExpr>(
      rule.conditions[0]->value, "Expected final conjunction");
  const auto& first_and = as<tepl::syntax::BinaryExpr>(
      final_and.lhs->value, "Expected first conjunction");
  check(final_and.op == tepl::syntax::BinaryOp::kLogicalAnd &&
            first_and.op == tepl::syntax::BinaryOp::kLogicalAnd,
        "Explicit conjunctions must associate left");
  const auto& logical_or = as<tepl::syntax::BinaryExpr>(
      first_and.lhs->value, "Expected grouped logical OR");
  check(logical_or.op == tepl::syntax::BinaryOp::kLogicalOr,
        "Logical OR must have lowest precedence");
  const auto& logical_and = as<tepl::syntax::BinaryExpr>(
      logical_or.lhs->value, "Expected logical AND");
  check(logical_and.op == tepl::syntax::BinaryOp::kLogicalAnd,
        "Logical AND precedence is wrong");
  const auto& comparison = as<tepl::syntax::BinaryExpr>(logical_and.lhs->value,
                                                        "Expected comparison");
  check(comparison.op == tepl::syntax::BinaryOp::kGreaterEqual,
        "Comparison precedence is wrong");
  const auto& addition =
      as<tepl::syntax::BinaryExpr>(comparison.lhs->value, "Expected addition");
  check(addition.op == tepl::syntax::BinaryOp::kAdd &&
            as<tepl::syntax::BinaryExpr>(addition.rhs->value,
                                         "Expected multiplication")
                    .op == tepl::syntax::BinaryOp::kMultiply,
        "Multiplication must bind more tightly than addition");
  check(as<tepl::syntax::UnaryExpr>(logical_and.rhs->value,
                                    "Expected logical negation")
                    .op == tepl::syntax::UnaryOp::kLogicalNot &&
            as<tepl::syntax::BooleanLiteral>(logical_or.rhs->value,
                                             "Expected boolean literal")
                .value,
        "Unary or boolean nodes are wrong");

  const auto& second_comparison = as<tepl::syntax::BinaryExpr>(
      first_and.rhs->value, "Expected less-than comparison");
  const auto& negated = as<tepl::syntax::UnaryExpr>(
      second_comparison.lhs->value, "Expected unary minus");
  const auto& grouped = as<tepl::syntax::BinaryExpr>(
      negated.operand->value, "Expected grouped addition");
  check(negated.op == tepl::syntax::UnaryOp::kNegate &&
            grouped.op == tepl::syntax::BinaryOp::kAdd &&
            negated.operand->span.begin.column == 15,
        "Grouped expression or its source span is wrong");
  const auto& call =
      as<tepl::syntax::Call>(final_and.rhs->value, "Expected builtin call");
  check(call.callee == "f" && call.arguments.size() == 4 &&
            as<tepl::syntax::IntegerLiteral>(call.arguments[3]->value,
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
      "rule literals {\n  (add 001 -1.2500) => (add +2 "
      "999999999999999999999999)\n  where check(1.00, -2.5)\n}");
  check(parsed.ok() && parsed.program, "Numeric literals must parse");
  const auto& rule = parsed.program->rules.front();
  const auto& lhs = as<tepl::syntax::Operator>(rule.lhs->value, "Expected add");
  check(as<tepl::syntax::IntegerLiteral>(lhs.operands[0]->value,
                                         "Expected integer")
                    .digits == "001" &&
            as<tepl::syntax::FloatLiteral>(lhs.operands[1]->value,
                                           "Expected float")
                    .digits == "-1.2500",
        "Graph literals must preserve kind, sign, and spelling");
  check(lhs.operands[0]->span.begin.line == 2 &&
            lhs.operands[0]->span.begin.column == 8 &&
            lhs.operands[0]->span.end.column == 11 &&
            lhs.operands[1]->span.begin.column == 12 &&
            lhs.operands[1]->span.end.column == 19,
        "Numeric literal spans must include the sign and full spelling");
  const auto& rhs =
      as<tepl::syntax::Operator>(rule.rhs->value, "Expected RHS add");
  check(as<tepl::syntax::IntegerLiteral>(rhs.operands[0]->value,
                                         "Expected signed integer")
                    .digits == "+2" &&
            as<tepl::syntax::IntegerLiteral>(rhs.operands[1]->value,
                                             "Expected large integer")
                    .digits == "999999999999999999999999",
        "Parsing literals must not impose machine numeric ranges");
  const auto& call =
      as<tepl::syntax::Call>(rule.conditions[0]->value, "Expected host call");
  check(as<tepl::syntax::FloatLiteral>(call.arguments[0]->value,
                                       "Expected host float")
                .digits == "1.00",
        "Host expressions must preserve decimal literals");
  const auto& negative = as<tepl::syntax::UnaryExpr>(
      call.arguments[1]->value, "Expected host unary minus");
  check(negative.op == tepl::syntax::UnaryOp::kNegate &&
            as<tepl::syntax::FloatLiteral>(negative.operand->value,
                                           "Expected negative host float")
                    .digits == "2.5",
        "Constraint signs must retain unary expression semantics");
  const auto printed = tepl::formatAst(*parsed.program);
  check(printed.find("integer 001") != std::string::npos &&
            printed.find("float -1.2500") != std::string::npos &&
            printed.find("float 1.00") != std::string::npos,
        "AST printing must include numeric literals");
  auto dialect = tepl::parse(
      "dialect t { op add(lhs: tensor, rhs: tensor) -> tensor } "
      "rule r { (add X 1.0) => (add 1.0 X) }");
  check(dialect.ok() && dialect.program,
        "Numeric operands must parse with dialect declarations");
}

void testAttributeTypeAndDefault() {
  auto parsed = tepl::parse(
      "dialect t { op test(input: tensor) -> tensor { "
      "attrs { axis: u64 = [] shape: extent[] = [] } } }",
      "attributes.tepl");
  check(parsed.ok() && parsed.program.has_value(),
        "Attribute fixture must build an AST");
  const auto& op = parsed.program->dialects.front().operations.front();
  const auto* attrs = std::get_if<tepl::syntax::InlineAttrs>(&*op.attrs);
  check(attrs && attrs->fields.size() == 2 && !attrs->fields[0].list_depth &&
            attrs->fields[0].empty_default && attrs->fields[1].list_depth &&
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
  const auto abstract_source = read(abstract_path);
  const auto span_text = [&](tepl::SourceSpan span) {
    check(span.begin.line == span.end.line, "Expected a single-line span");
    std::size_t start = 0;
    for (std::size_t line = 1; line < span.begin.line; ++line)
      start = abstract_source.find('\n', start) + 1;
    return abstract_source.substr(start + span.begin.column - 1,
                                  span.end.column - span.begin.column);
  };
  auto templates = tepl::parse(abstract_source, abstract_path);
  check(templates.ok() && templates.rule_count == 3,
        "Abstract fixture must parse all three templates");
  const auto& commute = templates.program->rules.front();
  check(commute.is_abstract && !commute.inheritance && commute.lhs &&
            commute.rhs && commute.parameters.size() == 1 &&
            commute.source_name == abstract_path,
        "Abstract rule structure is wrong");
  const auto& parameter = commute.parameters.front();
  check(parameter.name == "F" && parameter.operand_types.size() == 2 &&
            parameter.operand_types[0].name == "tensor" &&
            parameter.result_type.name == "tensor" &&
            span_text(parameter.span) == "F: op<(tensor, tensor) -> tensor>" &&
            span_text(parameter.result_type.span) == "tensor",
        "Operation parameter signature or spans are wrong");
  const auto& associate = templates.program->rules[1];
  check(associate.parameters.size() == 1 && associate.conditions.empty(),
        "Association template must preserve its operation parameter");

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
  check(restricted.declarations.size() == 3 &&
            restricted.conditions.size() == 1 &&
            restricted.inheritance->bindings.size() == 1 && !restricted.lhs &&
            !restricted.rhs,
        "Child restrictions must be preserved without expanding the pattern");
  const auto loaded = tepl::loadImports(program);
  check(loaded.ok() && loaded.program->imported_rules.size() == 3 &&
            loaded.program->declarations.dialects.size() == 1 &&
            loaded.program->declarations.rules.size() == 7,
        "Rule and dialect imports must load without merging root rules");
  const auto again = tepl::loadImports(program);
  check(again.ok() && again.program->imported_rules.size() == 3,
        "Repeated loading from immutable syntax must not duplicate templates");
  const auto printed = tepl::formatAst(*loaded.program);
  check(
      printed.find("import {commute, associate_right, distribute_left}") !=
              std::string::npos &&
          printed.find("    extends commute\n      F = t.add\n") !=
              std::string::npos &&
          printed.find("parameter F: op<(tensor, tensor) -> tensor>") !=
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
    testRuleConstructors();
    testExpressionsAndSpans();
    testExtendedAttributesAndParsedShapes();
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
