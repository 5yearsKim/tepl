#include "src/codegen/cpp/analysis_emitter.h"

#include "src/codegen/cpp/code_writer.h"
#include "src/codegen/cpp/shape_expression_emitter.h"

namespace tepl::codegen::cpp {
namespace {
void prelude(CodeWriter& out, const Names& names) {
  out.line("#pragma once");
  out.line("#include \"../op_node.h\"");
  out.line("#include \"../builtins/builtins.h\"");
  out.line("#include \"inference.h\"");
  out.open("namespace " + names.root.substr(2) + "::analysis");
}
void check(CodeWriter& out, const std::string& result) {
  out.line(
      "if(!op.arity().accepts(operands.size()) || !op.accepts_attrs(attrs)) "
      "return Inference<" +
      result + ">::invalid(\"invalid operation signature\");");
}
}  // namespace
std::string emitShapeInference(const core::Program& program,
                               const Names& names) {
  CodeWriter out;
  prelude(out, names);
  const auto runtime = names.root + "::builtins";
  for (const auto& op : program.operations) {
    if (!op.shape) continue;
    const auto& shape = *op.shape;
    out.open("inline " + runtime +
             "::BuiltinResult<::std::vector<::std::uint64_t>> shape_" +
             std::to_string(op.id.value) +
             "(::std::span<const ::std::vector<::std::uint64_t>> operands, "
             "[[maybe_unused]] const OpAttrs& attrs)");
    out.open("return " + runtime +
             "::attempt([&]() -> ::std::vector<::std::uint64_t>");
    for (const auto& parameter : shape.parameters) {
      if (parameter.variadic) {
        auto t = shapeType(
            shape.types.at(shape.symbols.at(parameter.symbol.value).type.value),
            names.root);
        out.line(t + " " + shapeSymbol(parameter.symbol) + ";");
        out.line("for(::std::size_t i=" + std::to_string(parameter.operand) +
                 ";i<operands.size();++i) " + shapeSymbol(parameter.symbol) +
                 ".push_back(" + runtime + "::shape::integers(operands[i]));");
      } else
        out.line("[[maybe_unused]] auto " + shapeSymbol(parameter.symbol) +
                 "=" + runtime + "::shape::integers(operands[" +
                 std::to_string(parameter.operand) + "]);");
    }
    if (op.attributes) {
      const auto& schema = program.attribute_schemas.at(op.attributes->value);
      out.line("[[maybe_unused]] auto fields=" + names.root +
               "::attrs_schema_" + std::to_string(schema.id.value) +
               "(attrs);");
      out.line("if(!fields) throw " + runtime +
               "::BuiltinError(\"invalid shape attributes\");");
      for (std::size_t i = 0; i < schema.fields.size(); ++i) {
        const auto& field = schema.fields[i];
        if (field.optional || (field.type != "index" && field.type != "i64" &&
                               field.type != "bool"))
          continue;
        out.line("[[maybe_unused]] auto attribute_" + std::to_string(i) + "=" +
                 runtime + "::shape::attribute(fields->" +
                 names.field(schema.id, i) + ");");
      }
    }
    for (const auto& statement : shape.statements) {
      if (auto let = std::get_if<core::shape::Let>(&statement.value))
        out.line("[[maybe_unused]] auto " + shapeSymbol(let->symbol) + "=" +
                 shapeExpression(shape, *let->value, names.root) + ";");
      else
        out.line(runtime + "::take(" + runtime + "::common::ensure(" +
                 shapeExpression(
                     shape,
                     *std::get<core::shape::Assert>(statement.value).condition,
                     names.root) +
                 "," + quote(op.dialect + "." + op.name + " shape assertion") +
                 "));");
    }
    out.line("return " + runtime + "::take(" + runtime +
             "::shape::dimensions(" +
             shapeExpression(shape, *shape.result.value, names.root) + "));");
    out.close(");");
    out.close();
  }
  out.open(
      "inline Inference<::std::vector<::std::uint64_t>> infer_shape(Op op, "
      "::std::span<const ::std::vector<::std::uint64_t>> operands, const "
      "OpAttrs& attrs)");
  check(out, "::std::vector<::std::uint64_t>");
  out.line(
      "if(op==Op::Literal) return "
      "Inference<::std::vector<::std::uint64_t>>::known({});");
  for (const auto& op : program.operations)
    if (op.shape) {
      out.open("if(op==" + names.operation(op.id, names.root) + ")");
      out.line("auto result=shape_" + std::to_string(op.id.value) +
               "(operands,attrs);");
      out.line(
          "return result ? "
          "Inference<::std::vector<::std::uint64_t>>::known(result.value()) : "
          "Inference<::std::vector<::std::uint64_t>>::invalid(result.error()."
          "what());");
      out.close();
    }
  out.line("return Inference<::std::vector<::std::uint64_t>>::unknown();");
  out.close();
  out.close();
  return out.str();
}
std::string emitDTypeInference(const core::Program& program,
                               const Names& names) {
  CodeWriter out;
  prelude(out, names);
  out.open(
      "inline Inference<DType> infer_dtype(Op op, ::std::span<const DType> "
      "operands, const OpAttrs& attrs)");
  check(out, "DType");
  out.open("if(op==Op::Literal)");
  out.line(
      "auto dtype=::std::get<LiteralAttrs>(attrs.value).dtype; return dtype ? "
      "Inference<DType>::known(*dtype) : Inference<DType>::unknown();");
  out.close();
  for (const auto& op : program.operations)
    if (op.dtype_policy) {
      auto fixed = std::get_if<core::DType>(&*op.dtype_policy);
      auto result =
          fixed ? "Inference<DType>::known(" + dtype(*fixed) + ")"
                : names.root + "::builtins::dtype::" +
                      std::string(core::dtypePolicyName(*op.dtype_policy)) +
                      "(operands)";
      out.line("if(op==" + names.operation(op.id, names.root) + ") return " +
               result + ";");
    }
  out.line("return Inference<DType>::unknown();");
  out.close();
  out.close();
  return out.str();
}
}  // namespace tepl::codegen::cpp
