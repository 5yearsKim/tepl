#include "src/codegen/cpp/metadata_emitter.h"

#include "src/codegen/cpp/code_writer.h"
#include "src/codegen/cpp/expression_emitter.h"
namespace tepl::codegen::cpp {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
void emitMetadataProgram(CodeWriter& out, const Expressions& e,
                         const core::metadata::Program& p, std::size_t op,
                         bool dtype) {
  auto name = std::string(dtype ? "dtype_" : "shape_") + num(op);
  out.open("inline rt::Value " + name +
           "(const ::std::vector<rt::TensorFacts>& operands,const rt::Value& "
           "attrs,rt::Budget& budget)");
  out.line(
      "[[maybe_unused]] rt::metadata::Context ctx(operands,attrs,budget);");
  for (const auto& param : p.parameters)
    out.line("ctx.parameter(" + num(param.symbol.value) + "," +
             num(param.operand) + "," + (param.variadic ? "true" : "false") +
             "," + (dtype ? "true" : "false") + ");");
  for (const auto& stmt : p.statements)
    std::visit(
        [&](const auto& s) {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, core::metadata::Let>)
            out.line("ctx.locals[" + num(s.symbol.value) +
                     "] = " + e.metadata(*s.value) + ";");
          else
            out.line("rt::require((" + e.metadata(*s.condition) +
                     ").boolean(),\"operation metadata assertion failed\");");
        },
        stmt.value);
  out.line("return " + e.metadata(*p.result.value) + ";");
  out.close();
}
}  // namespace
std::string emitMetadata(const executable::ProgramData& p, const Names& n) {
  CodeWriter out;
  Expressions e(p);
  out.line("#pragma once\n#include <tepl/runtime/axes.h>");
  out.open("namespace " + n.root.substr(2) + "::metadata");
  out.line("namespace rt = ::tepl_runtime;");
  for (const auto& op : p.operations) {
    if (op.shape)
      emitMetadataProgram(out, e, *op.shape, op.id.value, false);
    else if (op.axis_contract) {
      const auto& def = *op.axis_contract;
      out.open(
          "inline rt::Value shape_" + num(op.id.value) +
          "(const ::std::vector<rt::TensorFacts>& operands,const rt::Value& "
          "attrs,rt::Budget& budget)");
      out.line(
          "[[maybe_unused]] rt::axes::AxisContext ctx(operands,attrs,budget);");
      for (const auto& group : def.groups) {
        out.line("ctx.begin_group();");
        for (const auto& binding : group.bindings)
          out.line("ctx.bind(" + num(binding.operand) + "," +
                   e.axis(*binding.positions) + ");");
        out.line("ctx.finish_group(" +
                 (group.domain ? "::std::optional<rt::Value>(" +
                                     e.axis(*group.domain) + ")"
                               : "::std::nullopt") +
                 ");");
      }
      for (auto i : def.implicit_operands)
        out.line("ctx.cover(" + num(i) + ");");
      for (const auto& a : def.assertions)
        out.line("rt::require((" + e.axis(*a) +
                 ").boolean(),\"axis assertion failed\");");
      out.line("return rt::shape(ctx.output(" + e.axis(*def.output) + "));");
      out.close();
    }
    if (op.dtype) emitMetadataProgram(out, e, *op.dtype, op.id.value, true);
  }
  for (bool dtype : {false, true}) {
    std::string kind = dtype ? "dtype" : "shape",
                type = dtype ? "rt::DType" : "::std::vector<::std::uint64_t>";
    out.open("inline rt::EvalResult<::std::optional<" + type + ">> infer_" +
             kind +
             "(::std::size_t op,const ::std::vector<rt::TensorFacts>& "
             "operands,[[maybe_unused]] const rt::Value& "
             "attrs,[[maybe_unused]] rt::Budget& budget)");
    out.open("return rt::evaluate([&]() -> ::std::optional<" + type + ">");
    out.line("for(const auto& x:operands) if(x." + kind +
             ".kind == decltype(x." + kind +
             ")::Kind::Unknown) return ::std::nullopt;");
    out.open("switch(op)");
    for (const auto& op : p.operations)
      if (dtype ? op.dtype.has_value()
                : (op.shape.has_value() || bool(op.axis_contract)))
        out.line("case " + num(op.id.value) + ": return " +
                 (dtype ? "rt::from_value<rt::DType>(" : "(") + kind + "_" +
                 num(op.id.value) + "(operands,attrs,budget)" +
                 (dtype ? ")" : ").dimensions()") + ";");
    out.line("default: return ::std::nullopt;");
    out.close();
    out.close(");");
    out.close();
  }
  out.close();
  return out.str();
}
}  // namespace tepl::codegen::cpp
