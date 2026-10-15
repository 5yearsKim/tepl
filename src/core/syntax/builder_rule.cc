#include <stdexcept>
#include <string_view>
#include <utility>

#include "src/core/syntax/builder.h"
#include "src/core/syntax/builder_detail.h"

namespace tepl {
namespace {
using Parser = tepl_generated::TeplParser;
using detail::getSpan;

template <typename Value>
syntax::GraphExprPtr makeGraph(const antlr4::ParserRuleContext* context,
                               Value value) {
  return std::make_shared<syntax::GraphExpr>(
      syntax::GraphExpr{getSpan(context), std::move(value)});
}

template <typename Value>
syntax::ConstraintExprPtr makeConstraint(
    const antlr4::ParserRuleContext* context, Value value) {
  return std::make_shared<syntax::ConstraintExpr>(
      syntax::ConstraintExpr{getSpan(context), std::move(value)});
}

syntax::AutoField::Literal autoLiteral(Parser::ConcreteValueContext* value) {
  using L = syntax::AutoField::Literal;
  L out{};
  out.text = value->getText();
  if (!value->concreteValue().empty() || out.text == "[]") {
    out.kind = L::Kind::kList;
    for (auto* child : value->concreteValue())
      out.elements.push_back(autoLiteral(child));
  } else if (value->INT())
    out.kind = L::Kind::kInteger;
  else if (value->FLOAT())
    out.kind = L::Kind::kFloat;
  else if (value->TRUE() || value->FALSE())
    out.kind = L::Kind::kBool;
  else if (value->STRING()) {
    out.kind = L::Kind::kString;
    std::string decoded;
    for (std::size_t i = 1; i + 1 < out.text.size(); ++i) {
      if (out.text[i] == '\\') ++i;
      decoded += out.text[i];
    }
    out.text = std::move(decoded);
  } else
    out.kind = L::Kind::kName;
  return out;
}
void graphAttribute(syntax::Operator& op, Parser::AttributeContext* attribute) {
  if (!attribute) return;
  if (attribute->attrRef()) {
    op.attribute = syntax::AttributeRef{attribute->attrRef()->ID()->getText()};
    return;
  }
  syntax::AutoAttribute out{getSpan(attribute), {}};
  for (auto* input : attribute->autoField()) {
    syntax::AutoField field{};
    field.span = getSpan(input);
    field.name = input->ID(0)->getText();
    if (input->attrRef()) {
      field.descriptor =
          syntax::AttributeRef{input->attrRef()->ID()->getText()};
      field.field = input->ID(1)->getText();
    } else
      field.value = autoLiteral(input->concreteValue());
    out.fields.push_back(std::move(field));
  }
  op.automatic = std::move(out);
}

using detail::binaryOperator;
using detail::unaryOperator;

}  // namespace

std::any AstBuilder::visitRuleDecl(Parser::RuleDeclContext* context) {
  syntax::Rule rule;
  rule.span = getSpan(context);
  rule.name = context->ID()->getText();
  rule.source_name = source_name_;
  rule.is_abstract = context->ABSTRACT() != nullptr;
  if (auto* parameters = context->ruleParameters()) {
    for (auto* parameter : parameters->ruleParameter()) {
      syntax::RuleParameter value;
      value.span = getSpan(parameter);
      value.name = parameter->ID()->getText();
      if (auto* types = parameter->signatureTypes()) {
        for (auto* type : types->signatureType()) {
          value.operand_types.push_back({getSpan(type), type->getText()});
        }
      }
      auto* result = parameter->signatureType();
      value.result_type = {getSpan(result), result->getText()};
      rule.parameters.push_back(std::move(value));
    }
  }
  if (auto* clause = context->inheritanceClause()) {
    syntax::RuleInheritance inheritance;
    inheritance.span = getSpan(clause);
    inheritance.base = clause->ID()->getText();
    if (auto* bindings = clause->ruleBindings()) {
      for (auto* binding : bindings->ruleBinding()) {
        inheritance.bindings.push_back({getSpan(binding),
                                        binding->ID()->getText(),
                                        binding->opRef()->getText()});
      }
    }
    rule.inheritance = std::move(inheritance);
  }
  auto* body = context->rewriteBody();
  auto* inherited = context->inheritedBody();
  const auto declarations = body ? body->shapeDecl()
                            : inherited
                                ? inherited->shapeDecl()
                                : std::vector<Parser::ShapeDeclContext*>{};
  for (auto* declaration : declarations) {
    rule.declarations.push_back(
        std::any_cast<syntax::Declaration>(visit(declaration)));
  }
  const auto dtypes = body        ? body->dtypeDecl()
                      : inherited ? inherited->dtypeDecl()
                                  : std::vector<Parser::DtypeDeclContext*>{};
  for (auto* declaration : dtypes)
    rule.dtypes.push_back(
        {getSpan(declaration), declaration->ID(1)->getText()});
  if (body) {
    rule.lhs = std::any_cast<syntax::GraphExprPtr>(visit(body->graphExpr()));
    rule.rhs = std::any_cast<syntax::GraphExprPtr>(visit(body->rhsGraphExpr()));
  }
  if (auto* where = body        ? body->whereBlock()
                    : inherited ? inherited->whereBlock()
                                : nullptr) {
    for (auto* statement : where->whereStatement()) {
      auto value = std::any_cast<syntax::ConstraintExprPtr>(
          visit(statement->constraintExpr()));
      value = makeConstraint(
          statement, syntax::LocalBinding{statement->ID()->getText(), value});
      rule.conditions.push_back(value);
    }
    rule.conditions.push_back(std::any_cast<syntax::ConstraintExprPtr>(
        visit(where->constraintExpr())));
  }
  if (auto* derive = body ? body->deriveBlock() : nullptr) {
    for (auto* statement : derive->deriveStatement()) {
      const bool local = statement->LET() != nullptr;
      rule.derivations.push_back(
          {getSpan(statement),
           syntax::AttributeRef{local ? statement->ID()->getText()
                                      : statement->attrRef()->ID()->getText()},
           std::any_cast<syntax::ConstraintExprPtr>(
               visit(statement->constraintExpr())),
           local});
    }
  }
  return rule;
}

std::any AstBuilder::visitShapeDecl(Parser::ShapeDeclContext* context) {
  syntax::Declaration tensor{getSpan(context), context->ID()->getText(), {}};
  if (auto* dtype = context->dtypeName()) {
    tensor.dtype = syntax::DTypeAnnotation{getSpan(dtype), dtype->getText()};
  }
  if (auto* elements = context->shapeElements()) {
    tensor.shape =
        std::any_cast<std::vector<syntax::ShapeDimension>>(visit(elements));
  }
  return tensor;
}

std::any AstBuilder::visitShapeElements(Parser::ShapeElementsContext* context) {
  std::vector<syntax::ShapeDimension> dimensions;
  for (auto* child : context->children) {
    if (auto* scalar = dynamic_cast<Parser::ScalarDimContext*>(child)) {
      dimensions.push_back(
          std::any_cast<syntax::ShapeDimension>(visit(scalar)));
    } else if (auto* sequence =
                   dynamic_cast<Parser::SequenceDimContext*>(child)) {
      dimensions.push_back(
          std::any_cast<syntax::ShapeDimension>(visit(sequence)));
    }
  }
  return dimensions;
}

std::any AstBuilder::visitScalarDim(Parser::ScalarDimContext* context) {
  if (auto* name = context->ID()) {
    return syntax::ShapeDimension{getSpan(context),
                                  syntax::NamedDimension{name->getText()}};
  }
  return syntax::ShapeDimension{getSpan(context), syntax::WildcardDimension{}};
}

std::any AstBuilder::visitSequenceDim(Parser::SequenceDimContext* context) {
  syntax::SequenceDimension sequence;
  if (auto* name = context->ID()) sequence.name = name->getText();
  return syntax::ShapeDimension{getSpan(context), std::move(sequence)};
}

std::any AstBuilder::visitBareBindingGraph(
    Parser::BareBindingGraphContext* context) {
  return visit(context->binding());
}

std::any AstBuilder::visitVariableGraph(Parser::VariableGraphContext* context) {
  return makeGraph(context, syntax::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitNumberGraph(Parser::NumberGraphContext* context) {
  std::optional<syntax::DTypeAnnotation> dtype;
  auto spelling = context->getText();
  if (auto* annotation = context->dtypeName()) {
    dtype = syntax::DTypeAnnotation{getSpan(annotation), annotation->getText()};
    spelling.resize(spelling.find(':'));
  }
  if (context->FLOAT()) {
    return makeGraph(context, syntax::FloatLiteral{spelling, dtype});
  }
  return makeGraph(context, syntax::IntegerLiteral{spelling, dtype});
}

std::any AstBuilder::visitParenthesizedBindingGraph(
    Parser::ParenthesizedBindingGraphContext* context) {
  auto binding = std::any_cast<syntax::GraphExprPtr>(visit(context->binding()));
  return makeGraph(context, binding->value);
}

std::any AstBuilder::visitGetGraph(Parser::GetGraphContext* context) {
  return makeGraph(
      context,
      syntax::Projection{
          syntax::IntegerLiteral{context->INT()->getText()},
          std::any_cast<syntax::GraphExprPtr>(visit(context->graphExpr()))});
}

std::any AstBuilder::visitOperatorGraph(Parser::OperatorGraphContext* context) {
  syntax::Operator op{context->opRef()->getText(), std::nullopt, {}};
  graphAttribute(op, context->attribute());
  for (auto* operand : context->graphExpr()) {
    op.operands.push_back(std::any_cast<syntax::GraphExprPtr>(visit(operand)));
  }
  return makeGraph(context, std::move(op));
}

std::any AstBuilder::visitBinding(Parser::BindingContext* context) {
  return makeGraph(context,
                   syntax::Binding{syntax::NameRef{context->ID()->getText()},
                                   std::any_cast<syntax::GraphExprPtr>(
                                       visit(context->graphExpr()))});
}

std::any AstBuilder::visitRhsVariableGraph(
    Parser::RhsVariableGraphContext* context) {
  return makeGraph(context, syntax::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitRhsNumberGraph(
    Parser::RhsNumberGraphContext* context) {
  std::optional<syntax::DTypeAnnotation> dtype;
  auto spelling = context->getText();
  if (auto* annotation = context->dtypeName()) {
    dtype = syntax::DTypeAnnotation{getSpan(annotation), annotation->getText()};
    spelling.resize(spelling.find(':'));
  }
  if (context->FLOAT()) {
    return makeGraph(context, syntax::FloatLiteral{spelling, dtype});
  }
  return makeGraph(context, syntax::IntegerLiteral{spelling, dtype});
}

std::any AstBuilder::visitRhsGetGraph(Parser::RhsGetGraphContext* context) {
  return makeGraph(
      context,
      syntax::Projection{
          syntax::IntegerLiteral{context->INT()->getText()},
          std::any_cast<syntax::GraphExprPtr>(visit(context->rhsGraphExpr()))});
}

std::any AstBuilder::visitRhsOperatorGraph(
    Parser::RhsOperatorGraphContext* context) {
  syntax::Operator op{context->opRef()->getText(), std::nullopt, {}};
  graphAttribute(op, context->attribute());
  for (auto* operand : context->rhsGraphExpr()) {
    op.operands.push_back(std::any_cast<syntax::GraphExprPtr>(visit(operand)));
  }
  return makeGraph(context, std::move(op));
}

std::any AstBuilder::visitConstraintExpr(
    Parser::ConstraintExprContext* context) {
  if (context->IF()) {
    const auto e = context->constraintExpr();
    return makeConstraint(
        context, syntax::ConditionalExpr{
                     std::any_cast<syntax::ConstraintExprPtr>(visit(e[0])),
                     std::any_cast<syntax::ConstraintExprPtr>(visit(e[1])),
                     std::any_cast<syntax::ConstraintExprPtr>(visit(e[2]))});
  }
  return visit(context->logicalOr());
}

template <typename Context>
syntax::ConstraintExprPtr AstBuilder::foldBinary(
    antlr4::ParserRuleContext* parent, const std::vector<Context*>& operands) {
  auto lhs = std::any_cast<syntax::ConstraintExprPtr>(visit(operands.front()));
  for (std::size_t index = 1; index < operands.size(); ++index) {
    const auto spelling = parent->children[2 * index - 1]->getText();
    auto rhs = std::any_cast<syntax::ConstraintExprPtr>(visit(operands[index]));
    const syntax::SourceSpan span{lhs->span.begin, rhs->span.end};
    if (spelling == "++")
      lhs = std::make_shared<syntax::ConstraintExpr>(
          syntax::ConstraintExpr{span, syntax::Call{"concat", {lhs, rhs}}});
    else
      lhs = std::make_shared<syntax::ConstraintExpr>(syntax::ConstraintExpr{
          span, syntax::BinaryExpr{binaryOperator(spelling), std::move(lhs),
                                   std::move(rhs)}});
  }
  return lhs;
}

std::any AstBuilder::visitLogicalOr(Parser::LogicalOrContext* context) {
  return foldBinary(context, context->logicalAnd());
}

std::any AstBuilder::visitLogicalAnd(Parser::LogicalAndContext* context) {
  return foldBinary(context, context->equality());
}

std::any AstBuilder::visitEquality(Parser::EqualityContext* context) {
  return foldBinary(context, context->comparison());
}

std::any AstBuilder::visitComparison(Parser::ComparisonContext* context) {
  return foldBinary(context, context->additive());
}

std::any AstBuilder::visitAdditive(Parser::AdditiveContext* context) {
  return foldBinary(context, context->multiplicative());
}

std::any AstBuilder::visitMultiplicative(
    Parser::MultiplicativeContext* context) {
  return foldBinary(context, context->unary());
}

std::any AstBuilder::visitUnary(Parser::UnaryContext* context) {
  if (auto* postfix = context->postfix()) return visit(postfix);
  return makeConstraint(
      context, syntax::UnaryExpr{unaryOperator(context->getStart()->getText()),
                                 std::any_cast<syntax::ConstraintExprPtr>(
                                     visit(context->unary()))});
}

std::any AstBuilder::visitCallPrimary(Parser::CallPrimaryContext* context) {
  syntax::Call call{context->ID()->getText(), {}};
  if (auto* arguments = context->arguments()) {
    for (auto* argument : arguments->constraintExpr()) {
      call.arguments.push_back(
          std::any_cast<syntax::ConstraintExprPtr>(visit(argument)));
    }
  }
  return makeConstraint(context, std::move(call));
}

std::any AstBuilder::visitNamePrimary(Parser::NamePrimaryContext* context) {
  return makeConstraint(context, syntax::NameRef{context->ID()->getText()});
}

std::any AstBuilder::visitAttributePrimary(
    Parser::AttributePrimaryContext* context) {
  const syntax::AttributeRef descriptor{context->attrRef()->ID()->getText()};
  if (context->ID())
    return makeConstraint(
        context,
        syntax::DescriptorField{descriptor, context->ID()->getText(),
                                context->children.at(1)->getText() == "->"});
  return makeConstraint(context, descriptor);
}

std::any AstBuilder::visitIntegerPrimary(
    Parser::IntegerPrimaryContext* context) {
  return makeConstraint(context,
                        syntax::IntegerLiteral{context->INT()->getText()});
}

std::any AstBuilder::visitFloatPrimary(Parser::FloatPrimaryContext* context) {
  return makeConstraint(context,
                        syntax::FloatLiteral{context->FLOAT()->getText()});
}

std::any AstBuilder::visitTruePrimary(Parser::TruePrimaryContext* context) {
  return makeConstraint(context, syntax::BooleanLiteral{true});
}

std::any AstBuilder::visitFalsePrimary(Parser::FalsePrimaryContext* context) {
  return makeConstraint(context, syntax::BooleanLiteral{false});
}

std::any AstBuilder::visitGroupPrimary(Parser::GroupPrimaryContext* context) {
  auto expression = std::any_cast<syntax::ConstraintExprPtr>(
      visit(context->constraintExpr()));
  return makeConstraint(context, expression->value);
}

std::any AstBuilder::visitPostfix(Parser::PostfixContext* context) {
  auto value =
      std::any_cast<syntax::ConstraintExprPtr>(visit(context->primary()));
  for (auto* index : context->constraintExpr())
    value = makeConstraint(
        context,
        syntax::IndexExpr{
            value, std::any_cast<syntax::ConstraintExprPtr>(visit(index))});
  return value;
}
std::any AstBuilder::visitStringPrimary(Parser::StringPrimaryContext* context) {
  auto raw = context->STRING()->getText();
  std::string decoded;
  for (std::size_t i = 1; i + 1 < raw.size(); ++i) {
    if (raw[i] == '\\') ++i;
    decoded += raw[i];
  }
  return makeConstraint(context, syntax::StringLiteral{decoded});
}
std::any AstBuilder::visitListPrimary(Parser::ListPrimaryContext* context) {
  syntax::ListExpr list;
  for (auto* e : context->constraintExpr())
    list.elements.push_back(std::any_cast<syntax::ConstraintExprPtr>(visit(e)));
  return makeConstraint(context, list);
}
std::any AstBuilder::visitComprehensionPrimary(
    Parser::ComprehensionPrimaryContext* context) {
  return makeConstraint(context, syntax::ComprehensionExpr{
                                     std::any_cast<syntax::ConstraintExprPtr>(
                                         visit(context->constraintExpr(0))),
                                     context->ID()->getText(),
                                     std::any_cast<syntax::ConstraintExprPtr>(
                                         visit(context->constraintExpr(1)))});
}
template <class Context>
syntax::ConstraintExprPtr descriptorConstructor(
    AstBuilder& builder, Context* context,
    std::optional<syntax::AttributeRef> spread) {
  syntax::DescriptorConstructor value{spread, {}};
  for (auto* f : context->descriptorField())
    value.fields.push_back({getSpan(f), f->ID()->getText(),
                            std::any_cast<syntax::ConstraintExprPtr>(
                                builder.visit(f->constraintExpr()))});
  return makeConstraint(context, value);
}
std::any AstBuilder::visitDescriptorPrimary(
    Parser::DescriptorPrimaryContext* context) {
  std::optional<syntax::AttributeRef> spread;
  if (context->attrRef())
    spread = syntax::AttributeRef{context->attrRef()->ID()->getText()};
  return descriptorConstructor(*this, context, spread);
}
std::any AstBuilder::visitDescriptorFieldsPrimary(
    Parser::DescriptorFieldsPrimaryContext* context) {
  return descriptorConstructor(*this, context, {});
}

}  // namespace tepl
