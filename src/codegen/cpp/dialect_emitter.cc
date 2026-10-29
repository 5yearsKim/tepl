#include "src/codegen/cpp/dialect_emitter.h"

#include <map>

#include "src/codegen/cpp/code_writer.h"
namespace tepl::codegen::cpp {
namespace {
std::string num(std::size_t n) { return std::to_string(n); }
std::string attributeType(const core::AttributeField& f) {
  static const std::map<std::string, std::string> types = {
      {"u64", "::std::uint64_t"},
      {"i64", "::std::int64_t"},
      {"bool", "bool"},
      {"string", "::std::string"},
      {"dtype", "::tepl_runtime::DType"},
      {"precision", "::tepl_runtime::Precision"},
      {"dot_algorithm", "::tepl_runtime::DotAlgorithm"},
      {"replica_groups", "::tepl_runtime::ReplicaGroups"},
      {"region", "::tepl_runtime::Region"},
      {"elements", "::tepl_runtime::Elements"}};
  auto t = types.at(f.type);
  for (std::size_t i = 0; i < f.list_depth; ++i) t = "::std::vector<" + t + ">";
  return f.optional ? "::std::optional<" + t + ">" : t;
}
void variantMethods(CodeWriter& out, const std::string& name) {
  out.line(name + "() = default;");
  out.line("template<class T> requires ::std::constructible_from<Storage,T> " +
           name + "(T v):value(::std::move(v)) {}");
  out.line("bool operator==(const " + name + "&) const = default;");
  out.line(
      "::std::size_t hash_value() const { return "
      "::tepl_runtime::detail::hash_value(value); }");
}
}  // namespace
std::string schemaType(const Names& n, core::AttributeSchemaId id) {
  for (const auto& d : n.dialects)
    for (auto s : d.schemas)
      if (s == id)
        return n.root + "::dialects::" + d.module + "::" + n.schema(id);
  throw std::invalid_argument("unknown attribute schema");
}
std::string emitDialect(const executable::ProgramData& p, const Names& n,
                        const DialectNames& d) {
  CodeWriter out;
  out.line("#pragma once\n#include <tepl/runtime/values.h>");
  out.open("namespace " + n.root.substr(2) + "::dialects::" + d.module);
  out.open("enum class Op");
  for (auto id : d.operations) out.line(n.operationVariant(id) + ",");
  out.close(";");
  for (auto id : d.schemas) {
    const auto& s = p.attribute_schemas.at(id.value);
    auto name = n.schema(id);
    out.open("struct " + name);
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      out.line(attributeType(s.fields[i]) + " " + n.field(id, i) + "{};");
    out.line("bool operator==(const " + name + "&) const = default;");
    out.open("::std::size_t hash_value() const");
    out.line("::std::size_t seed = " + num(id.value) + ";");
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      out.line(
          "::tepl_runtime::detail::hash_combine(seed,::tepl_runtime::detail::"
          "hash_value(this->" +
          n.field(id, i) + "));");
    out.line("return seed;");
    out.close();
    out.open("::tepl_runtime::Value to_value() const");
    std::string fields;
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      fields += "::tepl_runtime::to_value(this->" + n.field(id, i) + "),";
    out.line("return ::tepl_runtime::Value(::tepl_runtime::Descriptor{" +
             num(id.value) + ",{" + fields + "}});");
    out.close();
    out.open("static ::tepl_runtime::EvalResult<" + name +
             "> from_value(const ::tepl_runtime::Value& value)");
    out.open("return ::tepl_runtime::evaluate([&]() -> " + name);
    out.line(
        "const auto& d = value.get<::tepl_runtime::Descriptor>(\"expected "
        "attributes\");");
    out.line("::tepl_runtime::require(d.schema == " + num(id.value) +
             " && d.fields.size() == " + num(s.fields.size()) +
             ",\"attribute schema mismatch\");");
    std::string fields_in;
    for (std::size_t i = 0; i < s.fields.size(); ++i)
      fields_in += "::tepl_runtime::from_value<" + attributeType(s.fields[i]) +
                   ">(d.fields[" + num(i) + "]),";
    out.line("return " + name + "{" + fields_in + "};");
    out.close(");");
    out.close();
    out.close(";");
  }
  out.open("struct OpAttrs");
  std::string storage = "using Storage=::std::variant<::std::monostate";
  for (auto id : d.schemas) storage += "," + n.schema(id);
  out.line(storage + ">;\nStorage value;");
  variantMethods(out, "OpAttrs");
  out.open("::tepl_runtime::Value to_value() const");
  for (auto id : d.schemas)
    out.line("if(auto p=::std::get_if<" + n.schema(id) +
             ">(&value)) return p->to_value();");
  out.line(
      "throw "
      "::tepl_runtime::EvalError(::tepl_runtime::ErrorKind::Invalid,\"empty "
      "attributes\");");
  out.close();
  out.close(";");
  out.close();
  return out.str();
}
std::string emitProgram(const executable::ProgramData& p, const Names& n,
                        const std::string& policy) {
  CodeWriter out;
  out.line(
      "#pragma once\n#include <tepl/runtime/runtime.h>\n#include "
      "\"dialects/dialects.h\"\n#include \"metadata.h\"");
  out.open("namespace " + n.root.substr(2));
  out.line("namespace rt = ::tepl_runtime;");
  out.line("using rt::DType; using rt::TensorFacts; using rt::Fact;");
  out.line("inline constexpr auto SEMANTIC_POLICY = " + quote(policy) + ";");
  out.line(
      "inline constexpr unsigned RUNTIME_ABI = 1;\nstatic_assert(RUNTIME_ABI "
      "== rt::RUNTIME_ABI, \"regenerate TEPL code for this runtime\");");
  out.open("struct Op");
  std::string storage = "using Storage=::std::variant<::std::monostate";
  for (const auto& d : n.dialects) storage += ",dialects::" + d.module + "::Op";
  out.line(storage + ">;\nStorage value;");
  variantMethods(out, "Op");
  out.close(";");
  out.open("struct OpAttrs");
  storage = "using Storage=::std::variant<::std::monostate";
  for (const auto& d : n.dialects)
    storage += ",dialects::" + d.module + "::OpAttrs";
  out.line(storage + ">;\nStorage value;");
  variantMethods(out, "OpAttrs");
  for (const auto& d : n.dialects)
    for (auto id : d.schemas)
      out.line("OpAttrs(" + schemaType(n, id) + " v):value(dialects::" +
               d.module + "::OpAttrs(::std::move(v))) {}");
  out.open("rt::Value to_value() const");
  for (const auto& d : n.dialects)
    out.line("if(auto p=::std::get_if<dialects::" + d.module +
             "::OpAttrs>(&value)) return p->to_value();");
  out.line("throw rt::EvalError(rt::ErrorKind::Invalid,\"empty attributes\");");
  out.close();
  out.open("static rt::EvalResult<OpAttrs> from_value(const rt::Value& value)");
  out.open("return rt::evaluate([&]() -> OpAttrs");
  out.line(
      "[[maybe_unused]] const auto& d = value.get<rt::Descriptor>(\"expected "
      "attributes\");");
  for (const auto& d : n.dialects)
    for (auto id : d.schemas)
      out.line("if(d.schema == " + num(id.value) + ") return OpAttrs(" +
               schemaType(n, id) + "::from_value(value).value());");
  out.line(
      "throw rt::EvalError(rt::ErrorKind::Invalid,\"unknown attribute "
      "schema\");");
  out.close(");");
  out.close();
  out.close(";");
  out.open("struct Program");
  out.line("using Op = " + n.root + "::Op; using Attrs = " + n.root +
           "::OpAttrs;");
  out.open("static Op operation(::std::size_t id)");
  out.open("switch(id)");
  for (const auto& op : p.operations)
    out.line("case " + num(op.id.value) + ": return " +
             n.operation(op.id, n.root) + ";");
  out.line(
      "default: throw rt::EvalError(rt::ErrorKind::Invalid,\"invalid operation "
      "ID\");");
  out.close();
  out.close();
  out.open("static ::std::size_t operation_id([[maybe_unused]] Op op)");
  for (const auto& op : p.operations)
    out.line("if(op == Op(" + n.operation(op.id, n.root) + ")) return " +
             num(op.id.value) + ";");
  out.line(
      "throw rt::EvalError(rt::ErrorKind::Invalid,\"invalid operation\");");
  out.close();
  out.open("static ::std::string_view operation_name(Op op)");
  out.open("switch(operation_id(op))");
  for (const auto& op : p.operations)
    out.line("case " + num(op.id.value) + ": return " +
             quote(op.dialect + "." + op.name) + ";");
  out.line(
      "default: throw rt::EvalError(rt::ErrorKind::Invalid,\"invalid "
      "operation\");");
  out.close();
  out.close();
  out.open("static rt::Arity arity(Op op)");
  out.open("switch(operation_id(op))");
  for (const auto& op : p.operations) {
    bool variadic = !op.operands.empty() && op.operands.back().variadic;
    out.line("case " + num(op.id.value) + ": return {" +
             num(op.operands.size() - variadic) + "," +
             (variadic ? "true" : "false") + "};");
  }
  out.line(
      "default: throw rt::EvalError(rt::ErrorKind::Invalid,\"invalid "
      "operation\");");
  out.close();
  out.close();
  out.open(
      "static bool accepts_attrs(Op op,[[maybe_unused]] const "
      "::std::optional<Attrs>& attrs)");
  out.open("switch(operation_id(op))");
  for (const auto& op : p.operations)
    out.line("case " + num(op.id.value) + ": return " +
             (op.attributes
                  ? "attrs && attrs->to_value().get<rt::Descriptor>(\"expected "
                    "attributes\").schema == " +
                        num(op.attributes->value)
                  : "!attrs") +
             ";");
  out.line("default: return false;");
  out.close();
  out.close();
  for (bool dtype : {false, true}) {
    std::string kind = dtype ? "dtype" : "shape",
                type = dtype ? "DType" : "::std::vector<::std::uint64_t>";
    out.open(
        "static rt::EvalResult<::std::optional<" + type + ">> infer_" + kind +
        "(Op op,const ::std::vector<TensorFacts>& operands,const rt::Value& "
        "attrs,rt::Budget& budget)");
    out.line("return metadata::infer_" + kind +
             "(operation_id(op),operands,attrs,budget);");
    out.close();
  }
  out.close(";");
  out.line("static_assert(rt::Program<Program>);");
  out.line(
      "using OpNode = rt::Node<Program>;\nusing TensorAnalysis = "
      "rt::TensorAnalysis<Program>;\nusing EGraph = "
      "rt::EGraph<Program>;\nusing Rewrite = rt::Rewrite<Program>;");
  out.close();
  return out.str();
}
}  // namespace tepl::codegen::cpp
