#pragma once

#include <any>
#include <string>
#include <vector>

#include "grammar/TeplBaseVisitor.h"
#include "src/ast/ast.h"

namespace tepl {

// Converts a successfully parsed ANTLR tree into an owning AST.
class AstBuilder final : public tepl_generated::TeplBaseVisitor {
 public:
  ast::Program build(tepl_generated::TeplParser::ProgramContext* context,
                     std::string source_name);

 private:
  std::any visitProgram(tepl_generated::TeplParser::ProgramContext*) override;
  std::any visitImportDecl(
      tepl_generated::TeplParser::ImportDeclContext*) override;
  std::any visitUseDecl(tepl_generated::TeplParser::UseDeclContext*) override;
  std::any visitDialectDecl(
      tepl_generated::TeplParser::DialectDeclContext*) override;
  std::any visitAttrsDecl(
      tepl_generated::TeplParser::AttrsDeclContext*) override;
  std::any visitOpDecl(tepl_generated::TeplParser::OpDeclContext*) override;
  std::any visitOperandDecl(
      tepl_generated::TeplParser::OperandDeclContext*) override;
  std::any visitVariadicOperand(
      tepl_generated::TeplParser::VariadicOperandContext*) override;
  std::any visitAttrField(
      tepl_generated::TeplParser::AttrFieldContext*) override;
  std::any visitRuleDecl(tepl_generated::TeplParser::RuleDeclContext*) override;
  std::any visitTensorDecl(
      tepl_generated::TeplParser::TensorDeclContext*) override;
  std::any visitScalarDecl(
      tepl_generated::TeplParser::ScalarDeclContext*) override;
  std::any visitShapeElements(
      tepl_generated::TeplParser::ShapeElementsContext*) override;
  std::any visitScalarDim(
      tepl_generated::TeplParser::ScalarDimContext*) override;
  std::any visitSequenceDim(
      tepl_generated::TeplParser::SequenceDimContext*) override;

  std::any visitBareBindingGraph(
      tepl_generated::TeplParser::BareBindingGraphContext*) override;
  std::any visitVariableGraph(
      tepl_generated::TeplParser::VariableGraphContext*) override;
  std::any visitNumberGraph(
      tepl_generated::TeplParser::NumberGraphContext*) override;
  std::any visitParenthesizedBindingGraph(
      tepl_generated::TeplParser::ParenthesizedBindingGraphContext*) override;
  std::any visitGetGraph(tepl_generated::TeplParser::GetGraphContext*) override;
  std::any visitOperatorGraph(
      tepl_generated::TeplParser::OperatorGraphContext*) override;
  std::any visitBinding(tepl_generated::TeplParser::BindingContext*) override;

  std::any visitConstraintExpr(
      tepl_generated::TeplParser::ConstraintExprContext*) override;
  std::any visitLogicalOr(
      tepl_generated::TeplParser::LogicalOrContext*) override;
  std::any visitLogicalAnd(
      tepl_generated::TeplParser::LogicalAndContext*) override;
  std::any visitEquality(tepl_generated::TeplParser::EqualityContext*) override;
  std::any visitComparison(
      tepl_generated::TeplParser::ComparisonContext*) override;
  std::any visitAdditive(tepl_generated::TeplParser::AdditiveContext*) override;
  std::any visitMultiplicative(
      tepl_generated::TeplParser::MultiplicativeContext*) override;
  std::any visitUnary(tepl_generated::TeplParser::UnaryContext*) override;
  std::any visitCallPrimary(
      tepl_generated::TeplParser::CallPrimaryContext*) override;
  std::any visitNamePrimary(
      tepl_generated::TeplParser::NamePrimaryContext*) override;
  std::any visitAttributePrimary(
      tepl_generated::TeplParser::AttributePrimaryContext*) override;
  std::any visitIntegerPrimary(
      tepl_generated::TeplParser::IntegerPrimaryContext*) override;
  std::any visitFloatPrimary(
      tepl_generated::TeplParser::FloatPrimaryContext*) override;
  std::any visitTruePrimary(
      tepl_generated::TeplParser::TruePrimaryContext*) override;
  std::any visitFalsePrimary(
      tepl_generated::TeplParser::FalsePrimaryContext*) override;
  std::any visitGroupPrimary(
      tepl_generated::TeplParser::GroupPrimaryContext*) override;

  template <typename Context>
  ast::ConstraintExprPtr foldBinary(antlr4::ParserRuleContext* parent,
                                    const std::vector<Context*>& operands);

  std::string source_name_;
};

}  // namespace tepl
