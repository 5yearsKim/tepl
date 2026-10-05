#include "src/codegen/cpp/dialect_emitter.h"

#include <map>

#include "src/codegen/cpp/code_writer.h"

namespace tepl::codegen::cpp {
std::string schemaType(const Names& names, core::AttributeSchemaId id) {
  for (const auto& d : names.dialects)
    for (auto schema : d.schemas)
      if (schema == id)
        return names.root + "::dialects::" + d.module + "::" + names.schema(id);
  throw std::invalid_argument("unknown attribute schema");
}
std::string emitDialect(const core::Program& program, const Names& names,
                        const DialectNames& d) {
  CodeWriter out;
  out.line("#pragma once");
  out.line("#include \"../types.h\"");
  out.open("namespace " + names.root.substr(2) + "::dialects::" + d.module);
  out.open("enum class Op");
  for (auto id : d.operations) out.line(names.operationVariant(id) + ", ");
  out.close(";");
  for (auto id : d.schemas) {
    const auto& schema = program.attribute_schemas.at(id.value);
    out.open("struct " + names.schema(id));
    for (std::size_t i = 0; i < schema.fields.size(); ++i) {
      const auto& field = schema.fields[i];
      static const std::map<std::string, std::string> types = {
          {"dtype", "DType"},
          {"index", "::std::uint64_t"},
          {"i64", "::std::int64_t"},
          {"bool", "bool"},
          {"string", "::std::string"},
          {"precision", "Precision"},
          {"dot_algorithm", "DotAlgorithm"},
          {"replica_groups", "ReplicaGroups"},
          {"region", "Region"},
          {"elements", "Elements"}};
      auto token = types.at(field.type);
      if (field.type == "dtype" || field.type == "precision" ||
          field.type == "dot_algorithm" || field.type == "replica_groups" ||
          field.type == "region" || field.type == "elements")
        token = names.root + "::" + token;
      for (std::size_t depth = 0; depth < field.list_depth; ++depth)
        token = "::std::vector<" + token + ">";
      if (field.optional) token = "::std::optional<" + token + ">";
      if (field.empty_default) out.line("// TEPL default: empty list.");
      out.line(token + " " + names.field(id, i) + "{};");
    }
    out.line("bool operator==(const " + names.schema(id) +
             "&) const = default;");
    out.open("::std::size_t hash_value() const");
    out.line("::std::size_t seed = " + std::to_string(id.value) + ";");
    for (std::size_t i = 0; i < schema.fields.size(); ++i)
      out.line(names.root + "::detail::hash_combine(seed," + names.root +
               "::detail::hash_value(this->" + names.field(id, i) + "));");
    out.line("return seed;");
    out.close();
    out.close(";");
  }
  out.open("struct OpAttrs");
  std::string variant = "using Value=::std::variant<::std::monostate";
  for (auto id : d.schemas) variant += ", " + schemaType(names, id);
  out.line(variant + ">;");
  out.line("Value value;");
  out.line("OpAttrs()=default;");
  out.line(
      "template<class T> requires ::std::constructible_from<Value,T> OpAttrs(T "
      "v): value(::std::move(v)) {}");
  out.line("bool operator==(const OpAttrs&) const=default;");
  out.line("::std::size_t hash_value() const { return " + names.root +
           "::detail::hash_value(value); }");
  out.open("::std::optional<::std::size_t> schema_id() const");
  for (auto id : d.schemas)
    out.line("if(::std::holds_alternative<" + schemaType(names, id) +
             ">(value)) return " + std::to_string(id.value) + ";");
  out.line("return {};");
  out.close();
  out.close(";");
  out.open("inline ::std::string_view name(Op op)");
  out.open("switch(op)");
  for (auto id : d.operations)
    out.line("case Op::" + names.operationVariant(id) + ": return " +
             quote(program.operations[id.value].name) + ";");
  out.close();
  out.line("throw ::std::invalid_argument(\"invalid operation\");");
  out.close();
  out.open("inline ::std::optional<Op> from_name(::std::string_view value)");
  for (auto id : d.operations) {
    const auto& op = program.operations[id.value];
    out.line("if(value==" + quote(op.name) +
             (op.alias ? " || value==" + quote(*op.alias) : "") +
             ") return Op::" + names.operationVariant(id) + ";");
  }
  out.line("return {};");
  out.close();
  out.open("inline " + names.root + "::Arity arity(Op op)");
  out.open("switch(op)");
  for (auto id : d.operations) {
    const auto& op = program.operations[id.value];
    bool variadic = !op.operands.empty() && op.operands.back().variadic;
    out.line("case Op::" + names.operationVariant(id) + ": return {" +
             std::to_string(op.operands.size() - variadic) + "," +
             (variadic ? "true" : "false") + "};");
  }
  out.close();
  out.line("throw ::std::invalid_argument(\"invalid operation\");");
  out.close();
  out.open("inline bool accepts_attrs(Op op,const OpAttrs& attrs)");
  out.open("switch(op)");
  for (auto id : d.operations) {
    const auto& op = program.operations[id.value];
    out.line("case Op::" + names.operationVariant(id) +
             ": return ::std::holds_alternative<" +
             (op.attributes ? schemaType(names, *op.attributes)
                            : "::std::monostate") +
             ">(attrs.value);");
  }
  out.close();
  out.line("return false;");
  out.close();
  out.open("inline ::std::size_t op_id(Op op)");
  out.open("switch(op)");
  for (auto id : d.operations)
    out.line("case Op::" + names.operationVariant(id) + ": return " +
             std::to_string(id.value + 2) + ";");
  out.close();
  out.line("throw ::std::invalid_argument(\"invalid operation\");");
  out.close();
  out.close();
  return out.str();
}
std::string emitOpUnion(const core::Program&, const Names& names) {
  CodeWriter out;
  out.line("template<class O> struct DialectOp;");
  for (const auto& d : names.dialects)
    out.line("template<> struct DialectOp<dialects::" + d.module +
             "::Op> { using Attrs=dialects::" + d.module + "::OpAttrs; };");
  out.open("struct OpAttrs");
  std::string variant =
      "using Value=::std::variant<::std::monostate,LiteralAttrs,InputAttrs";
  for (const auto& d : names.dialects)
    variant += ",dialects::" + d.module + "::OpAttrs";
  out.line(variant + ">;");
  out.line("Value value;");
  out.line("OpAttrs()=default;");
  out.line(
      "template<class T> requires ::std::constructible_from<Value,T> OpAttrs(T "
      "v):value(::std::move(v)) { normalize(); }");
  out.open("void normalize()");
  for (const auto& d : names.dialects)
    out.line("if(auto p=::std::get_if<dialects::" + d.module +
             "::OpAttrs>(&value); p && "
             "::std::holds_alternative<::std::monostate>(p->value)) "
             "value=::std::monostate{};");
  out.close();
  out.line("bool operator==(const OpAttrs&) const=default;");
  out.line(
      "::std::size_t hash_value() const { return detail::hash_value(value); }");
  out.open("::std::optional<::std::size_t> schema_id() const");
  for (const auto& d : names.dialects)
    out.line("if(auto p=::std::get_if<dialects::" + d.module +
             "::OpAttrs>(&value)) return p->schema_id();");
  out.line("return {};");
  out.close();
  out.open(
      "::std::optional<OpAttrs> checked_schema(::std::size_t schema) const");
  out.line(
      "return schema_id()==schema ? ::std::optional(*this) : ::std::nullopt;");
  out.close();
  out.close(";");
  for (const auto& d : names.dialects)
    for (auto id : d.schemas) {
      out.open("inline const " + schemaType(names, id) + "* attrs_schema_" +
               std::to_string(id.value) + "(const OpAttrs& attrs)");
      out.line("auto p=::std::get_if<dialects::" + d.module +
               "::OpAttrs>(&attrs.value);");
      out.line("return p ? ::std::get_if<" + schemaType(names, id) +
               ">(&p->value) : nullptr;");
      out.close();
    }
  out.open("struct Op");
  out.line("enum class Runtime { Literal,Input };");
  out.line("static constexpr auto Literal=Runtime::Literal;");
  out.line("static constexpr auto Input=Runtime::Input;");
  variant = "using Value=::std::variant<Runtime";
  for (const auto& d : names.dialects)
    variant += ",dialects::" + d.module + "::Op";
  out.line(variant + ">;");
  out.line("Value value=Runtime::Input;");
  out.line("Op()=default;");
  out.line(
      "template<class T> requires ::std::constructible_from<Value,T> Op(T "
      "v):value(v) {}");
  out.line("bool operator==(const Op&) const=default;");
  out.open("::std::size_t id() const");
  out.line(
      "if(auto p=::std::get_if<Runtime>(&value)) return *p==Literal ? 0 : 1;");
  for (const auto& d : names.dialects)
    out.line("if(auto p=::std::get_if<dialects::" + d.module +
             "::Op>(&value)) return dialects::" + d.module + "::op_id(*p);");
  out.line("throw NodeError(\"invalid operation\");");
  out.close();
  out.open("::std::string_view name() const");
  out.line(
      "if(auto p=::std::get_if<Runtime>(&value)) return *p==Literal ? "
      "\"<literal>\" : \"<input>\";");
  for (const auto& d : names.dialects)
    out.line("if(auto p=::std::get_if<dialects::" + d.module +
             "::Op>(&value)) return dialects::" + d.module + "::name(*p);");
  out.line("throw NodeError(\"invalid operation\");");
  out.close();
  out.open("static ::std::optional<Op> from_name(::std::string_view value)");
  out.line(
      "if(value==\"<literal>\") return Op(Literal); if(value==\"<input>\") "
      "return Op(Input);");
  for (const auto& d : names.dialects)
    out.line("if(value.starts_with(" + quote(d.name + ".") +
             ")) { auto op=dialects::" + d.module +
             "::from_name(value.substr(" + std::to_string(d.name.size() + 1) +
             ")); if(op) return Op(*op); }");
  out.line("return {};");
  out.close();
  out.open("Arity arity() const");
  out.line("if(::std::holds_alternative<Runtime>(value)) return {0,false};");
  for (const auto& d : names.dialects)
    out.line("if(auto p=::std::get_if<dialects::" + d.module +
             "::Op>(&value)) return dialects::" + d.module + "::arity(*p);");
  out.line("throw NodeError(\"invalid operation\");");
  out.close();
  out.open("bool accepts_attrs(const OpAttrs& attrs) const");
  out.open("if(auto p=::std::get_if<Runtime>(&value))");
  out.line(
      "if(*p==Input) return "
      "::std::holds_alternative<InputAttrs>(attrs.value);");
  out.line(
      "auto literal=::std::get_if<LiteralAttrs>(&attrs.value); return literal "
      "&& valid_literal(literal->value) && (!literal->dtype || "
      "accepts_literal(*literal->dtype,literal->value));");
  out.close();
  for (const auto& d : names.dialects) {
    out.open("if(auto op=::std::get_if<dialects::" + d.module +
             "::Op>(&value))");
    out.line(
        "if(::std::holds_alternative<::std::monostate>(attrs.value)) return "
        "dialects::" +
        d.module + "::accepts_attrs(*op,{});");
    out.line("auto p=::std::get_if<dialects::" + d.module +
             "::OpAttrs>(&attrs.value); return p && dialects::" + d.module +
             "::accepts_attrs(*op,*p);");
    out.close();
  }
  out.line("return false;");
  out.close();
  out.close(";");
  return out.str();
}
}  // namespace tepl::codegen::cpp
