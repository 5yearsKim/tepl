#include "src/codegen/cpp/analysis_emitter.h"

#include "src/codegen/cpp/code_writer.h"
#include "src/codegen/cpp/metadata_expression_emitter.h"

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
void evaluator(CodeWriter& out, const core::Program& program,
               const Names& names, const core::Operation& op,
               bool dtype_program) {
  const auto& checked = dtype_program ? *op.dtype : *op.shape;
  const auto runtime = names.root + "::builtins";
  const std::string result =
      dtype_program ? "DType" : "::std::vector<::std::uint64_t>";
  out.open("inline " + runtime + "::BuiltinResult<" + result + "> " +
           (dtype_program ? "dtype_" : "shape_") + std::to_string(op.id.value) +
           "(::std::span<const " + result +
           "> operands, [[maybe_unused]] const OpAttrs& attrs)");
  out.open("return " + runtime + "::attempt([&]() -> " + result);
  for (const auto& parameter : checked.parameters) {
    auto symbol = metadataSymbol(parameter.symbol);
    if (parameter.variadic) {
      out.line(metadataType(
                   checked.types.at(
                       checked.symbols.at(parameter.symbol.value).type.value),
                   names.root) +
               " " + symbol + ";");
      out.line("for(::std::size_t i=" + std::to_string(parameter.operand) +
               ";i<operands.size();++i) " + symbol + ".push_back(" +
               (dtype_program ? "operands[i]"
                              : runtime + "::shape::integers(operands[i])") +
               ");");
    } else {
      auto operand = "operands[" + std::to_string(parameter.operand) + "]";
      out.line("[[maybe_unused]] auto " + symbol + "=" +
               (dtype_program
                    ? operand
                    : runtime + "::shape::integers(" + operand + ")") +
               ";");
    }
  }
  if (op.attributes) {
    const auto& schema = program.attribute_schemas.at(op.attributes->value);
    out.line("[[maybe_unused]] auto fields=" + names.root + "::attrs_schema_" +
             std::to_string(schema.id.value) + "(attrs);");
    out.line("if(!fields) throw " + runtime +
             "::BuiltinError(\"invalid metadata attributes\");");
    for (std::size_t i = 0; i < schema.fields.size(); ++i) {
      const auto& field = schema.fields[i];
      if (field.optional || (field.type != "index" && field.type != "i64" &&
                             field.type != "bool" && field.type != "dtype"))
        continue;
      auto value = "fields->" + names.field(schema.id, i);
      out.line("[[maybe_unused]] auto attribute_" + std::to_string(i) + "=" +
               (field.type == "dtype"
                    ? value
                    : runtime + "::shape::attribute(" + value + ")") +
               ";");
    }
  }
  for (const auto& statement : checked.statements) {
    if (auto let = std::get_if<core::metadata::Let>(&statement.value)) {
      out.line("[[maybe_unused]] auto " + metadataSymbol(let->symbol) + "=" +
               metadataExpression(checked, *let->value, names.root) + ";");
    } else {
      const auto& at = statement.origin.definition.span.begin;
      out.line(runtime + "::take(" + runtime + "::common::ensure(" +
               metadataExpression(
                   checked,
                   *std::get<core::metadata::Assert>(statement.value).condition,
                   names.root) +
               "," +
               quote(op.dialect + "." + op.name +
                     (dtype_program ? " dtype assertion at "
                                    : " shape assertion at ") +
                     std::to_string(at.line) + ":" +
                     std::to_string(at.column)) +
               "));");
    }
  }
  auto value = metadataExpression(checked, *checked.result.value, names.root);
  out.line("return " +
           (dtype_program ? value
                          : runtime + "::take(" + runtime +
                                "::shape::dimensions(" + value + "))") +
           ";");
  out.close(");");
  out.close();
}
std::string inference(const core::Program& program, const Names& names,
                      bool dtype_program) {
  CodeWriter out;
  prelude(out, names);
  const std::string result =
      dtype_program ? "DType" : "::std::vector<::std::uint64_t>";
  for (const auto& op : program.operations)
    if (dtype_program ? op.dtype.has_value() : op.shape.has_value())
      evaluator(out, program, names, op, dtype_program);
  out.open("inline Inference<" + result + "> " +
           (dtype_program ? "infer_dtype" : "infer_shape") +
           "(Op op, ::std::span<const " + result +
           "> operands, const OpAttrs& attrs)");
  check(out, result);
  if (dtype_program) {
    out.line(
        "if(op==Op::Literal) { auto "
        "dtype=::std::get<LiteralAttrs>(attrs.value).dtype; return dtype ? "
        "Inference<DType>::known(*dtype) : Inference<DType>::unknown(); }");
  } else
    out.line("if(op==Op::Literal) return Inference<" + result +
             ">::known({});");
  for (const auto& op : program.operations) {
    if (!(dtype_program ? op.dtype.has_value() : op.shape.has_value()))
      continue;
    out.open("if(op==" + names.operation(op.id, names.root) + ")");
    out.line("auto result=" + std::string(dtype_program ? "dtype_" : "shape_") +
             std::to_string(op.id.value) + "(operands,attrs);");
    out.line("return result ? Inference<" + result +
             ">::known(result.value()) : Inference<" + result +
             ">::invalid(result.error().what());");
    out.close();
  }
  out.line("return Inference<" + result + ">::unknown();");
  out.close();
  out.close();
  return out.str();
}
}  // namespace
std::string emitShapeInference(const core::Program& program,
                               const Names& names) {
  return inference(program, names, false);
}
std::string emitDTypeInference(const core::Program& program,
                               const Names& names) {
  return inference(program, names, true);
}
}  // namespace tepl::codegen::cpp
