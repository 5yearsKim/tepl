#include "src/codegen/rust/metadata_emitter.h"

#include "src/codegen/rust/code_writer.h"
#include "src/codegen/rust/expression_emitter.h"
namespace tepl::codegen::rust {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
void emitMetadataProgram(CodeWriter& out, const Expressions& e,
                         const core::metadata::Program& p, std::size_t op,
                         bool dtype) {
  const std::string name = (dtype ? "dtype_" : "shape_") + num(op);
  out.open("fn " + name +
           "(operands: &[TensorFacts], attrs: &Value, budget: &mut rt::Budget) "
           "-> EvalResult<Value>");
  out.line(
      "let mut ctx = rt::metadata::Context::new(operands, attrs, budget);");
  for (const auto& param : p.parameters)
    out.line("ctx.parameter(" + num(param.symbol.value) + ", " +
             num(param.operand) + ", " + (param.variadic ? "true" : "false") +
             ", " + (dtype ? "true" : "false") + ")?;");
  for (const auto& stmt : p.statements)
    std::visit(
        [&](const auto& s) {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, core::metadata::Let>) {
            out.line("let value = " + e.metadata(*s.value) + ";");
            out.line("ctx.locals.insert(" + num(s.symbol.value) + ", value);");
          } else
            out.line(
                "rt::error::require((" + e.metadata(*s.condition) +
                ").boolean()?, \"operation metadata assertion failed\")?;");
        },
        stmt.value);
  out.line("Ok(" + e.metadata(*p.result.value) + ")");
  out.close();
}
}  // namespace
std::string emitMetadata(const executable::ProgramData& p) {
  CodeWriter out;
  Expressions e(p);
  out.line(
      "use ::tepl_runtime::{self as rt, Value, TensorFacts, DType, "
      "EvalResult};");
  out.line("use rt::values::AttributeValue;");
  for (const auto& op : p.operations) {
    if (op.shape)
      emitMetadataProgram(out, e, *op.shape, op.id.value, false);
    else if (op.axis_contract) {
      const auto& def = *op.axis_contract;
      out.open("fn shape_" + num(op.id.value) +
               "(operands: &[TensorFacts], attrs: &Value, budget: &mut "
               "rt::Budget) -> EvalResult<Value>");
      out.line(
          "let mut ctx = rt::axes::AxisContext::new(operands, attrs, budget);");
      for (const auto& group : def.groups) {
        out.line("ctx.begin_group();");
        for (const auto& binding : group.bindings) {
          out.line("let positions = " + e.axis(*binding.positions) + ";");
          out.line("ctx.bind(" + num(binding.operand) + ", positions)?;");
        }
        if (group.domain)
          out.line("let domain = Some(" + e.axis(*group.domain) + ");");
        else
          out.line("let domain = None;");
        out.line("ctx.finish_group(domain)?;");
      }
      for (auto i : def.implicit_operands)
        out.line("ctx.cover(" + num(i) + ")?;");
      for (const auto& a : def.assertions)
        out.line("rt::error::require((" + e.axis(*a) +
                 ").boolean()?, \"axis assertion failed\")?;");
      out.line("let output = " + e.axis(*def.output) + ";");
      out.line("Ok(rt::values::shape(&ctx.output(output)?))");
      out.close();
    }
    if (op.dtype) emitMetadataProgram(out, e, *op.dtype, op.id.value, true);
  }
  for (bool dtype : {false, true}) {
    const std::string kind = dtype ? "dtype" : "shape";
    out.open("pub fn infer_" + kind +
             "(op: usize, operands: &[TensorFacts], attrs: &Value, budget: "
             "&mut rt::Budget) -> EvalResult<Option<" +
             (dtype ? "DType" : "Vec<u64>") + ">>");
    // Preserve the executable evaluator's all-operands availability boundary.
    out.line(std::string("if operands.iter().any(|x| matches!(x.") + kind +
             ", rt::Fact::Unknown)) { return Ok(None); }");
    out.open("match op");
    for (const auto& op : p.operations)
      if (dtype ? op.dtype.has_value()
                : (op.shape.has_value() || bool(op.axis_contract)))
        out.line(num(op.id.value) + " => " +
                 (dtype ? "DType::from_value(&" : "") + kind + "_" +
                 num(op.id.value) + "(operands, attrs, budget)?" +
                 (dtype ? ")" : ".dimensions()") + ".map(Some),");
    out.line("_ => Ok(None),");
    out.close();
    out.close();
  }
  return out.str();
}
}  // namespace tepl::codegen::rust
