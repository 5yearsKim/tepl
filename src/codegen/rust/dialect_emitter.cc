#include "src/codegen/rust/dialect_emitter.h"

#include <map>

#include "src/codegen/rust/code_writer.h"
namespace tepl::codegen::rust {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string attributeType(const core::AttributeField& field) {
  static const std::map<std::string, std::string> types = {
      {"u64", "u64"},
      {"i64", "i64"},
      {"bool", "bool"},
      {"string", "::std::string::String"},
      {"dtype", "rt::DType"},
      {"precision", "rt::types::Precision"},
      {"dot_algorithm", "rt::types::DotAlgorithm"},
      {"replica_groups", "rt::types::ReplicaGroups"},
      {"region", "rt::types::Region"},
      {"elements", "rt::types::Elements"}};
  auto result = types.at(field.type);
  for (std::size_t i = 0; i < field.list_depth; ++i)
    result = "::std::vec::Vec<" + result + ">";
  return field.optional ? "::std::option::Option<" + result + ">" : result;
}
constexpr auto derive =
    "#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]";
constexpr auto derive_copy =
    "#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]";
}  // namespace
std::string emitDialect(const executable::ProgramData& p, const Names& n,
                        const DialectNames& d) {
  CodeWriter out;
  out.line("use ::tepl_runtime::{self as rt, Value, EvalResult};");
  out.line("use rt::values::{AttributeValue, invalid};");
  out.line(derive_copy);
  out.open("pub enum Op");
  for (auto id : d.operations) out.line(n.operationVariant(id) + ",");
  out.close();
  for (auto id : d.schemas) {
    const auto& s = p.attribute_schemas.at(id.value);
    const auto name = n.schema(id);
    out.line(derive);
    out.open("pub struct " + name);
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      out.line("pub " + n.field(id, i) + ": " + attributeType(s.fields[i]) +
               ",");
    out.close();
    out.open("impl AttributeValue for " + name);
    out.open("fn to_value(&self) -> Value");
    std::string fields = "vec![";
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      fields += "self." + n.field(id, i) + ".to_value(),";
    out.line("Value::Descriptor { schema: " + num(id.value) +
             ", fields: " + fields + "] }");
    out.close();
    out.open("fn from_value(value: &Value) -> EvalResult<Self>");
    out.line(
        "let Value::Descriptor { schema, fields } = value else { return "
        "Err(invalid(\"expected attributes\")); };");
    out.line("rt::error::require(*schema == " + num(id.value) +
             " && fields.len() == " + num(s.fields.size()) +
             ", \"attribute schema mismatch\")?;");
    out.open("Ok(Self");
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      out.line(n.field(id, i) + ": AttributeValue::from_value(&fields[" +
               num(i) + "])? ,");
    out.close(")");
    out.close();
    out.close();
  }
  out.line(derive);
  out.open("pub enum OpAttrs");
  for (auto id : d.schemas) out.line(n.schema(id) + "(" + n.schema(id) + "),");
  out.close();
  out.open("impl From<Op> for super::super::Op");
  out.open("fn from(op: Op) -> Self");
  out.line("Self::" + d.variant + "(op)");
  out.close();
  out.close();
  out.open("impl From<OpAttrs> for super::super::OpAttrs");
  out.open("fn from(attrs: OpAttrs) -> Self");
  out.line("Self::" + d.variant + "(attrs)");
  out.close();
  out.close();
  for (auto id : d.schemas) {
    out.open("impl From<" + n.schema(id) + "> for super::super::OpAttrs");
    out.open("fn from(attrs: " + n.schema(id) + ") -> Self");
    out.line("Self::" + d.variant + "(OpAttrs::" + n.schema(id) + "(attrs))");
    out.close();
    out.close();
  }
  return out.str();
}
std::string emitProgram(const executable::ProgramData& p, const Names& n,
                        const std::string& policy) {
  CodeWriter out;
  out.line("use ::tepl_runtime::{self as rt, Value, EvalResult};");
  out.line("use rt::values::{AttributeValue, invalid};");
  out.line("pub use rt::{DType, Fact, TensorFacts};");
  out.line("pub use rt::types;");
  out.line(
      "pub mod dialects;\npub mod metadata;\npub mod rules;\npub mod graphs;");
  out.line("pub const SEMANTIC_POLICY: &str = " + quote(policy) + ";");
  out.line("pub const RUNTIME_ABI: u32 = 2;");
  out.line("const _: () = assert!(rt::RUNTIME_ABI == RUNTIME_ABI);");
  out.line(derive_copy);
  out.line("pub struct Program;");
  out.line(derive_copy);
  out.open("pub enum Op");
  for (const auto& d : n.dialects)
    out.line(d.variant + "(dialects::" + d.module + "::Op),");
  out.close();
  out.line(derive);
  out.open("pub enum OpAttrs");
  for (const auto& d : n.dialects)
    out.line(d.variant + "(dialects::" + d.module + "::OpAttrs),");
  out.close();
  out.open("impl AttributeValue for OpAttrs");
  out.open("fn to_value(&self) -> Value");
  out.open(p.dialects.empty() ? "match *self" : "match self");
  for (const auto& d : n.dialects)
    for (auto id : d.schemas)
      out.line("Self::" + d.variant + "(dialects::" + d.module +
               "::OpAttrs::" + n.schema(id) + "(attrs)) => attrs.to_value(),");
  for (const auto& d : n.dialects)
    if (d.schemas.empty())
      out.line("Self::" + d.variant + "(attrs) => match *attrs {},");
  out.close();
  out.close();
  out.open("fn from_value(value: &Value) -> EvalResult<Self>");
  out.line(
      "let Value::Descriptor { schema, .. } = value else { return "
      "Err(invalid(\"expected attributes\")); };");
  out.open("match schema");
  for (const auto& d : n.dialects)
    for (auto id : d.schemas)
      out.line(num(id.value) + " => Ok(dialects::" + d.module +
               "::" + n.schema(id) + "::from_value(value)?.into()),");
  out.line("_ => Err(invalid(\"unknown attribute schema\")),");
  out.close();
  out.close();
  out.close();
  out.open("impl rt::Program for Program");
  out.line("type Op = Op;\ntype Attrs = OpAttrs;");
  out.open("fn operation(id: usize) -> Op");
  out.open("match id");
  for (const auto& d : n.dialects)
    for (auto id : d.operations)
      out.line(num(id.value) + " => Op::" + d.variant + "(dialects::" +
               d.module + "::Op::" + n.operationVariant(id) + "),");
  out.line("_ => unreachable!(\"validated operation ID\"),");
  out.close();
  out.close();
  out.open("fn operation_id(op: Op) -> usize");
  out.open("match op");
  for (const auto& d : n.dialects)
    for (auto id : d.operations)
      out.line("Op::" + d.variant + "(dialects::" + d.module + "::Op::" +
               n.operationVariant(id) + ") => " + num(id.value) + ",");
  for (const auto& d : n.dialects)
    if (d.operations.empty())
      out.line("Op::" + d.variant + "(op) => match op {},");
  out.close();
  out.close();
  out.open("fn operation_name(op: Op) -> &'static str");
  out.open("match Self::operation_id(op)");
  for (const auto& op : p.operations)
    out.line(num(op.id.value) + " => " + quote(op.dialect + "." + op.name) +
             ",");
  out.line("_ => unreachable!(),");
  out.close();
  out.close();
  out.open("fn arity(op: Op) -> (usize, bool)");
  out.open("match Self::operation_id(op)");
  for (const auto& op : p.operations) {
    bool variadic = !op.operands.empty() && op.operands.back().variadic;
    out.line(num(op.id.value) + " => (" +
             num(op.operands.size() - (variadic ? 1 : 0)) + ", " +
             (variadic ? "true" : "false") + "),");
  }
  out.line("_ => unreachable!(),");
  out.close();
  out.close();
  out.open("fn accepts_attrs(op: Op, attrs: Option<&OpAttrs>) -> bool");
  out.line("let schema = attrs.map(AttributeValue::to_value);");
  out.open("match (Self::operation_id(op), schema)");
  for (const auto& op : p.operations)
    out.line("(" + num(op.id.value) + ", " +
             (op.attributes ? "Some(Value::Descriptor { schema: " +
                                  num(op.attributes->value) + ", .. })"
                            : "None") +
             ") => true,");
  out.line("_ => false,");
  out.close();
  out.close();
  out.open(
      "fn infer_shape(op: Op, operands: &[TensorFacts], attrs: &Value, budget: "
      "&mut rt::Budget) -> EvalResult<Option<Vec<u64>>>");
  out.line(
      "metadata::infer_shape(Self::operation_id(op), operands, attrs, budget)");
  out.close();
  out.open(
      "fn infer_dtype(op: Op, operands: &[TensorFacts], attrs: &Value, budget: "
      "&mut rt::Budget) -> EvalResult<Option<DType>>");
  out.line(
      "metadata::infer_dtype(Self::operation_id(op), operands, attrs, budget)");
  out.close();
  out.close();
  out.line("pub type OpNode = rt::Node<Program>;");
  out.line("pub type TensorAnalysis = rt::TensorAnalysis<Program>;");
  out.line("pub type EGraph = rt::egg::EGraph<OpNode, TensorAnalysis>;");
  out.line("pub type Rewrite = rt::egg::Rewrite<OpNode, TensorAnalysis>;");
  return out.str();
}
}  // namespace tepl::codegen::rust
