#include "src/codegen/common/inference_emitter.h"

#include <cctype>
#include <map>
#include <sstream>
#include <stdexcept>

#include "src/core/semantics/solver.h"

namespace tepl::codegen {
namespace {
struct Emitter {
  const core::Program& program;
  const InferenceSyntax& syntax;
  std::map<std::string, std::string> cached;
  std::ostringstream pending;
  std::string key(const core::inference::Value& v) {
    std::string result = std::to_string(static_cast<unsigned>(v.kind)) + ":" +
                         v.text + ":" + std::to_string(v.descriptor.value) +
                         ":" + std::to_string(v.schema.value) + ":" +
                         std::to_string(v.field) + ":" +
                         std::to_string(v.capture.value) + ":" +
                         std::to_string(v.string_literal);
    for (const auto& a : v.arguments) result += "(" + key(*a) + ")";
    return result;
  }
  std::string flush() {
    auto result = pending.str();
    pending.str("");
    return result;
  }
  std::string runtime() { return syntax.root + "::builtins"; }
  std::string descriptor(core::DescriptorId d) {
    return syntax.rust ? "descriptor_" + std::to_string(d.value)
                       : runtime() + "::require(ctx.attrs(" +
                             std::to_string(d.value) + "))";
  }
  std::string literal(const core::inference::Value& v,
                      const core::AttributeField& f, std::size_t depth,
                      bool optional) {
    if (v.text == "none" && !v.string_literal)
      return syntax.rust ? "None" : "::std::nullopt";
    if (optional)
      return syntax.rust
                 ? "Some(" + literal(v, f, depth, false) + ")"
                 : "::std::optional(" + literal(v, f, depth, false) + ")";
    if (depth) {
      std::string element = f.type == "index"  ? "::std::uint64_t"
                            : f.type == "i64"  ? "::std::int64_t"
                            : f.type == "bool" ? "bool"
                            : f.type == "precision"
                                ? syntax.root + "::Precision"
                            : f.type == "dtype" ? syntax.root + "::DType"
                                                : "::std::string";
      if (f.type == "region") element = syntax.root + "::Region";
      if (f.type == "elements") element = syntax.root + "::Elements";
      if (f.type == "replica_groups") element = syntax.root + "::ReplicaGroups";
      if (f.type == "dot_algorithm") element = syntax.root + "::DotAlgorithm";
      std::string vector = element;
      for (std::size_t i = 0; i < depth; ++i)
        vector = "::std::vector<" + vector + ">";
      std::string out = syntax.rust ? "vec![" : vector + "{";
      for (auto& a : v.arguments) out += literal(*a, f, depth - 1, false) + ",";
      return out + (syntax.rust ? "]" : "}");
    }
    if (f.type == "bool") return v.text;
    if (f.type == "string") {
      std::string text = "\"";
      for (char c : v.text) {
        if (c == '\\' || c == '"') text += '\\';
        if (c == '\n')
          text += "\\n";
        else if (c == '\r')
          text += "\\r";
        else if (c == '\t')
          text += "\\t";
        else
          text += c;
      }
      text += '"';
      return syntax.rust ? text + ".to_string()"
                         : "::std::string(" + text + ")";
    }
    if (f.type == "precision") {
      auto variant = v.text == "default" ? "Default"
                     : v.text == "high"  ? "High"
                                         : "Highest";
      return syntax.root +
             (syntax.rust ? "::types::Precision::" : "::Precision::") + variant;
    }
    if (f.type == "dtype") {
      auto name = v.text;
      name[0] = static_cast<char>(std::toupper(name[0]));
      if (name == "Bf16") name = "BF16";
      return syntax.root + "::DType::" + name;
    }
    if (syntax.rust)
      return "(\"" + v.text + "\".parse::<" +
             (f.type == "index" ? "u64" : "i64") + ">().ok()?)";
    return "static_cast<" +
           std::string(f.type == "index" ? "::std::uint64_t"
                                         : "::std::int64_t") +
           ">(" + runtime() + "::common::integer(\"" + v.text + "\"))";
  }
  std::string semantic(const core::inference::Program& inference) {
    const auto& plan = *inference.semantic;
    auto vector = [&](const std::string& type,
                      const std::vector<std::string>& elements) {
      std::string out = syntax.rust ? "vec![" : "::std::vector<" + type + ">{";
      for (const auto& x : elements) out += x + ",";
      return out + (syntax.rust ? "]" : "}");
    };
    auto prefix = runtime() + "::semantic::";
    std::vector<std::string> shapes, dtypes, known, pinned;
    for (auto capture : plan.captures) {
      auto shape = std::make_shared<core::inference::Value>(
          core::inference::Value{core::inference::Value::Kind::kShape});
      shape->capture = capture;
      shapes.push_back(eval(shape));
      auto dtype = std::make_shared<core::inference::Value>(
          core::inference::Value{core::inference::Value::Kind::kDType});
      dtype->capture = capture;
      dtypes.push_back(eval(dtype));
    }
    for (auto f : plan.known) {
      auto v = std::make_shared<core::inference::Value>(
          core::inference::Value{core::inference::Value::Kind::kField});
      v->descriptor = f.descriptor;
      v->schema = f.schema;
      v->field = f.field;
      known.push_back(eval(v));
    }
    for (auto v : inference.pinned) pinned.push_back(eval(v));
    auto borrow = [&](std::string x) {
      return syntax.rust ? "&(" + x + ")" : x;
    };
    auto lists = [&](const auto& x) {
      return borrow(vector(prefix + "List", x));
    };
    auto serialized = core::semantics::encodePlan(plan);
    std::string text = "\"";
    for (char c : serialized) {
      if (c == '\\' || c == '"') text += '\\';
      text += c;
    }
    text += '"';
    auto call = prefix + "construct(" + text + "," + lists(shapes) + "," +
                lists(known) + "," + lists(pinned) + ")";
    auto out = flush();
    out += "if (!" + runtime() + "::dtype::is_numeric(" + dtypes.at(0) + ")";
    for (std::size_t i = 1; i < dtypes.size(); ++i)
      out += " || (" + dtypes[i] + ") != (" + dtypes[0] + ")";
    out += ") { return " + std::string(syntax.rust ? "None" : "{}") + "; }\n";
    return out + (syntax.rust ? "let " : "const auto ") +
           "auto_semantic_fields = " +
           (syntax.rust ? call + ".ok()?"
                        : runtime() + "::take(" + call + ")") +
           ";\n";
  }
  std::string eval(const core::inference::ValuePtr& ptr,
                   const core::AttributeField* target = nullptr) {
    if (ptr->kind == core::inference::Value::Kind::kLiteral)
      return raw(ptr, target);
    auto fingerprint = key(*ptr);
    if (auto found = cached.find(fingerprint); found != cached.end())
      return found->second + (syntax.rust ? ".clone()" : "");
    auto expression = raw(ptr, target);
    auto name = "auto_value_" + std::to_string(cached.size());
    cached.emplace(std::move(fingerprint), name);
    pending << (syntax.rust ? "let " : "const auto ") << name << " = "
            << expression << ";\n";
    return name + (syntax.rust ? ".clone()" : "");
  }
  std::string raw(const core::inference::ValuePtr& ptr,
                  const core::AttributeField* target) {
    using K = core::inference::Value::Kind;
    const auto& v = *ptr;
    if (v.kind == K::kSemanticWitness)
      return syntax.rust
                 ? "auto_semantic_fields.get(" + std::to_string(v.field) +
                       ")?.clone()"
                 : "auto_semantic_fields.at(" + std::to_string(v.field) + ")";
    if (v.kind == K::kLiteral) {
      core::AttributeField fallback{"", "index", v.text == "list" ? 1u : 0u};
      const auto& f = target ? *target : fallback;
      return literal(v, f, f.list_depth, f.optional);
    }
    if (v.kind == K::kField) {
      if (syntax.rust)
        return "(match &(" + descriptor(v.descriptor) + ") { " +
               syntax.attrs(v.schema,
                            syntax.field(v.schema, v.field) + ": field, ..") +
               " => Some(field.clone()), _ => None })?";
      return "([&]() { auto attr=" + descriptor(v.descriptor) +
             "; auto fields=" + syntax.root + "::attrs_schema_" +
             std::to_string(v.schema.value) + "(attr); if(!fields) throw " +
             runtime() +
             "::BuiltinError(\"descriptor schema mismatch\"); return fields->" +
             syntax.field(v.schema, v.field) + "; }())";
    }
    if (v.kind == K::kDType)
      return syntax.rust ? "ctx.tensor(\"?c" + std::to_string(v.capture.value) +
                               "\".parse::<Var>().ok()?)?.dtype"
                         : runtime() + "::require(ctx.tensor(" +
                               std::to_string(v.capture.value) + ")).dtype";
    if (v.kind == K::kShape)
      return syntax.rust ? "ctx.tensor(\"?c" + std::to_string(v.capture.value) +
                               "\".parse::<Var>().ok()?)?.shape.clone()"
                         : runtime() + "::require(ctx.tensor(" +
                               std::to_string(v.capture.value) + ")).shape";
    throw std::logic_error("unsupported inference value");
  }
};
}  // namespace
std::string emitAttributeInference(const core::Program& program,
                                   const core::Rule& rule,
                                   const InferenceSyntax& syntax) {
  if (!rule.inference) return {};
  Emitter e{program, syntax};
  std::ostringstream out;
  if (rule.inference->semantic) out << e.semantic(*rule.inference);
  for (auto& a : rule.inference->assignments) {
    const auto& schema = program.attribute_schemas.at(a.schema.value);
    std::string attrs = "";
    for (std::size_t i = 0; i < a.fields.size(); ++i) {
      if (syntax.rust) attrs += syntax.field(a.schema, i) + ": ";
      attrs += e.eval(a.fields[i], &schema.fields[i]) + ",";
    }
    attrs = syntax.attrs(a.schema, attrs);
    out << e.flush();
    if (syntax.rust)
      out << "let descriptor_" << a.descriptor.value << " = " << attrs << ";\n";
    else
      out << "derived.emplace(" << a.descriptor.value << "," << attrs << ");\n";
  }
  return out.str();
}
}  // namespace tepl::codegen
