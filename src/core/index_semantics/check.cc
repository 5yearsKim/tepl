#include "src/core/index_semantics/check.h"

#include <algorithm>
#include <charconv>
#include <functional>
#include <set>
#include <stdexcept>
#include <type_traits>

#include "src/core/analysis_context.h"
#include "src/core/resolution/source.h"

namespace tepl::core::index_semantics {
namespace {
struct Checker {
  detail::AnalysisContext& context;
  const Operation& operation;
  const std::string& source;
  Definition result;
  std::map<std::string, MetaPtr> locals;
  std::map<std::string, std::size_t> groups, operands, fields;
  SourceOrigin at;

  [[noreturn]] void fail(const std::string& text) {
    throw std::runtime_error(text);
  }
  MetaPtr make(Meta::Kind kind, Type type, std::string name,
               std::size_t index = 0, std::vector<MetaPtr> args = {}) {
    return std::make_shared<Meta>(
        Meta{kind, type, std::move(name), index, std::move(args), at});
  }
  void type(const MetaPtr& p, Type expected) {
    if (p->type != expected)
      fail("incorrect axis expression type: " + format(*p));
  }
  MetaPtr call(const std::string& name, std::vector<MetaPtr> args) {
    auto signature = [&](std::initializer_list<Type> inputs, Type output) {
      if (inputs.size() != args.size()) fail("wrong arity for '" + name + "'");
      std::size_t i = 0;
      for (auto t : inputs) type(args[i++], t);
      return make(Meta::Kind::Call, output, name, 0, args);
    };
    if (name == "extent_add" || name == "extent_sub" || name == "extent_mul" ||
        name == "extent_floor_div")
      return signature({Type::Positions, Type::Positions}, Type::Positions);
    if (name == "scale_extents")
      return signature({Type::Positions, Type::Integer}, Type::Positions);
    if (name == "offset_extents")
      return signature({Type::Positions, Type::Integer}, Type::Positions);
    if (name == "extent_le")
      return signature({Type::Positions, Type::Positions}, Type::Boolean);
    if (name == "is_positive_extents")
      return signature({Type::Positions}, Type::Boolean);
    if (name == "all_axes") return signature({Type::Operand}, Type::Positions);
    if (name == "rank") return signature({Type::Operand}, Type::Integer);
    if (name == "shape") return signature({Type::Operand}, Type::Positions);
    if (name == "ordered_complement")
      return signature({Type::Positions, Type::Positions}, Type::Positions);
    if (name == "gather")
      return signature({Type::Positions, Type::Positions}, Type::Positions);
    if (name == "range") return signature({Type::Integer}, Type::Positions);
    if (name == "place")
      return signature({Type::Axes, Type::Positions, Type::Axes}, Type::Axes);
    if (name == "permute")
      return signature({Type::Axes, Type::Positions}, Type::Axes);
    if (name == "is_valid_axis_list")
      return signature({Type::Positions, Type::Integer}, Type::Boolean);
    if (name == "len") {
      if (args.size() != 1 ||
          (args[0]->type != Type::Axes && args[0]->type != Type::Positions))
        fail("len requires an axis or position sequence");
      return make(Meta::Kind::Call, Type::Integer, name, 0, args);
    }
    if (name == "concat") {
      if (args.empty()) fail("concat requires arguments");
      auto t = args[0]->type;
      if (t != Type::Positions && t != Type::Axes)
        fail("concat requires sequences");
      for (auto& arg : args) type(arg, t);
      return make(Meta::Kind::Call, t, name, 0, args);
    }
    if (name == "==" || name == "!=") {
      if (args.size() != 2 || args[0]->type != args[1]->type)
        fail("equality requires matching types");
      return make(Meta::Kind::Call, Type::Boolean, name, 0, args);
    }
    if (name == "&&" || name == "||")
      return signature({Type::Boolean, Type::Boolean}, Type::Boolean);
    if (name == "<" || name == "<=" || name == ">" || name == ">=")
      return signature({Type::Integer, Type::Integer}, Type::Boolean);
    if (name == "+" || name == "-" || name == "*")
      return signature({Type::Integer, Type::Integer}, Type::Integer);
    fail("unsupported axis builtin '" + name + "'");
  }
  MetaPtr metadata(const ast::MetadataExpr& expr) {
    at = detail::origin(source, expr.span);
    return std::visit(
        [&](const auto& v) -> MetaPtr {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, ast::NameRef>) {
            if (locals.contains(v.name)) return locals.at(v.name);
            if (groups.contains(v.name))
              return make(Meta::Kind::Group, Type::Axes, v.name,
                          groups.at(v.name));
            if (operands.contains(v.name))
              return make(Meta::Kind::Operand, Type::Operand, v.name,
                          operands.at(v.name));
            fail("unknown axes name '" + v.name + "'");
          } else if constexpr (std::is_same_v<T, ast::IntegerLiteral>) {
            try {
              std::stoll(v.digits);
            } catch (...) {
              fail("integer out of range");
            }
            return make(Meta::Kind::Integer, Type::Integer, v.digits);
          } else if constexpr (std::is_same_v<T, ast::BooleanLiteral>) {
            return make(Meta::Kind::Boolean, Type::Boolean,
                        v.value ? "true" : "false");
          } else if constexpr (std::is_same_v<T, ast::MetadataField>) {
            if (!std::holds_alternative<ast::MetadataAttrs>(v.value->value) ||
                !fields.contains(v.field))
              fail("unknown axis attribute '" + v.field + "'");
            return make(Meta::Kind::Attribute, Type::Positions, v.field,
                        fields.at(v.field));
          } else if constexpr (std::is_same_v<T, ast::MetadataCall>) {
            std::vector<MetaPtr> args;
            for (const auto& arg : v.arguments) args.push_back(metadata(*arg));
            at = detail::origin(source, expr.span);
            return call(v.callee, std::move(args));
          } else if constexpr (std::is_same_v<T, ast::MetadataUnary>) {
            auto x = metadata(*v.operand);
            type(x, Type::Integer);
            if (std::string(spelling(v.op)) == "-")
              return call("-",
                          {make(Meta::Kind::Integer, Type::Integer, "0"), x});
            fail("unsupported unary axis expression");
          } else if constexpr (std::is_same_v<T, ast::MetadataBinary>) {
            auto l = metadata(*v.lhs), r = metadata(*v.rhs);
            at = detail::origin(source, expr.span);
            return call(std::string(spelling(v.op)), {l, r});
          } else if constexpr (std::is_same_v<T, ast::MetadataList>) {
            std::vector<MetaPtr> args;
            for (const auto& element : v.elements) {
              auto arg = metadata(*element);
              type(arg, Type::Integer);
              args.push_back(arg);
            }
            return make(Meta::Kind::List, Type::Positions, "", 0,
                        std::move(args));
          } else {
            fail("unsupported expression in index axes block");
          }
        },
        expr.value);
  }
  IndexValuePtr indexValue(const ast::SemanticExpr& expr) {
    at = detail::origin(source, expr.span);
    if (expr.kind != ast::SemanticExpr::Kind::kCall)
      fail("read indices require Coords, IndexAdd, or Concat");
    if (expr.text == "IndexScale") {
      if (expr.arguments.size() != 2 ||
          expr.arguments[0]->kind != ast::SemanticExpr::Kind::kAttribute ||
          !fields.contains(expr.arguments[0]->text))
        fail("IndexScale requires an index[] attribute and a coordinate tuple");
      auto name = expr.arguments[0]->text;
      result.extent_fields.insert(fields.at(name));
      auto scale =
          make(Meta::Kind::Attribute, Type::Positions, name, fields.at(name));
      return std::make_shared<IndexValue>(IndexValue{
          IndexValue::Kind::Scale, 0, {indexValue(*expr.arguments[1])}, scale});
    }
    if (expr.text == "Coords") {
      if (expr.arguments.size() != 1 ||
          expr.arguments[0]->kind != ast::SemanticExpr::Kind::kName ||
          !groups.contains(expr.arguments[0]->text))
        fail("Coords requires an exported bind group");
      return std::make_shared<IndexValue>(
          IndexValue{IndexValue::Kind::Coordinates,
                     groups.at(expr.arguments[0]->text),
                     {}});
    }
    if ((expr.text != "IndexAdd" && expr.text != "Concat") ||
        expr.arguments.size() < 2)
      fail("unsupported coordinate primitive or arity: '" + expr.text + "'");
    std::vector<IndexValuePtr> args;
    for (const auto& a : expr.arguments) args.push_back(indexValue(*a));
    return std::make_shared<IndexValue>(
        IndexValue{expr.text == "IndexAdd" ? IndexValue::Kind::Add
                                           : IndexValue::Kind::Concat,
                   0, args});
  }
  std::set<std::size_t> indexGroups(const IndexValue& v) {
    if (v.kind == IndexValue::Kind::Coordinates) return {v.group};
    std::set<std::size_t> out;
    for (const auto& a : v.args) {
      auto part = indexGroups(*a);
      out.insert(part.begin(), part.end());
    }
    return out;
  }
  ValuePtr value(const ast::SemanticExpr& expr) {
    at = detail::origin(source, expr.span);
    auto node = [&](Value::Kind k, std::size_t index,
                    std::vector<ValuePtr> args = {}) {
      return std::make_shared<Value>(Value{k, index, std::move(args), at});
    };
    if (expr.kind == ast::SemanticExpr::Kind::kName) {
      if (!operands.contains(expr.text))
        fail("semantics operand '" + expr.text + "' is not declared");
      return node(Value::Kind::Operand, operands.at(expr.text));
    }
    if (expr.kind != ast::SemanticExpr::Kind::kCall)
      fail("axis semantics expects Operand, Const, Add, Product, or Reduce");
    if (expr.text == "Read") {
      if (expr.arguments.size() != 2 ||
          expr.arguments[0]->kind != ast::SemanticExpr::Kind::kName ||
          !operands.contains(expr.arguments[0]->text))
        fail("Read requires an operand and a coordinate tuple");
      auto access = indexValue(*expr.arguments[1]);
      return std::make_shared<Value>(Value{Value::Kind::Read,
                                           operands.at(expr.arguments[0]->text),
                                           {},
                                           at,
                                           0,
                                           access});
    }
    if (expr.text == "Const") {
      if (expr.arguments.size() != 1 ||
          expr.arguments[0]->kind != ast::SemanticExpr::Kind::kInteger)
        fail("Const requires one signed integer literal");
      auto text = expr.arguments[0]->text;
      if (text.starts_with("+")) text.erase(0, 1);
      std::int64_t number;
      auto [end, ec] =
          std::from_chars(text.data(), text.data() + text.size(), number);
      if (ec != std::errc{} || end != text.data() + text.size())
        fail("Const literal is outside the signed 64-bit range");
      return std::make_shared<Value>(
          Value{Value::Kind::Const, 0, {}, at, number});
    }
    if (expr.text == "Reduce") {
      if (expr.arguments.size() != 2) fail("Reduce requires group and value");
      const auto& g = *expr.arguments[0];
      if (g.kind != ast::SemanticExpr::Kind::kName || !groups.contains(g.text))
        fail(
            "Reduce requires an exported bind group; axes let locals are "
            "private");
      auto body = value(*expr.arguments[1]);
      return node(Value::Kind::Reduce, groups.at(g.text), {body});
    }
    if ((expr.text != "Add" && expr.text != "Product") ||
        expr.arguments.size() != 2)
      fail("unsupported index semantic primitive or arity: '" + expr.text +
           "'");
    auto l = value(*expr.arguments[0]), r = value(*expr.arguments[1]);
    return node(expr.text == "Add" ? Value::Kind::Add : Value::Kind::Product, 0,
                {l, r});
  }
  bool constantValue(const Value& v) {
    if (v.kind == Value::Kind::Const) return true;
    if (v.kind == Value::Kind::Operand || v.kind == Value::Kind::Read)
      return false;
    for (const auto& a : v.args)
      if (!constantValue(*a)) return false;
    return true;
  }
  std::set<std::size_t> free(const Value& v) {
    if (v.kind == Value::Kind::Const) return {};
    if (v.kind == Value::Kind::Read) return indexGroups(*v.access);
    if (v.kind == Value::Kind::Operand) {
      std::set<std::size_t> out;
      for (std::size_t g = 0; g < result.groups.size(); ++g)
        for (const auto& b : result.groups[g].bindings)
          if (b.operand == v.index) out.insert(g);
      return out;
    }
    auto out = free(*v.args[0]);
    if (v.kind == Value::Kind::Reduce) {
      if (!out.erase(v.index) && !constantValue(*v.args[0]))
        fail("Reduce group is not free in its body");
      return out;
    }
    auto right = free(*v.args[1]);
    if (v.kind == Value::Kind::Add && out != right &&
        !constantValue(*v.args[0]) && !constantValue(*v.args[1]))
      fail(
          "Add operands must have identical free logical groups (no "
          "broadcasting)");
    out.insert(right.begin(), right.end());
    return out;
  }
  std::set<std::size_t> outputGroups(const Meta& m) {
    if (m.kind == Meta::Kind::Group) return {m.index};
    std::set<std::size_t> out;
    if (m.kind != Meta::Kind::Call ||
        (m.name != "concat" && m.name != "permute" && m.name != "place"))
      fail("output must be groups, concatenations, or permutations");
    auto count = m.name == "permute" ? std::size_t{1} : m.args.size();
    for (std::size_t i = 0; i < count; ++i) {
      if (m.name == "place" && i == 1) continue;
      auto part = outputGroups(*m.args[i]);
      for (auto g : part)
        if (!out.insert(g).second) fail("output repeats a logical group");
    }
    return out;
  }
  Definition run(const ast::AxisDefinition& axes,
                 const ast::SemanticDefinition& sem) {
    if (operation.shape)
      fail("index axes derive shape; a separate shape block is not allowed");
    for (std::size_t i = 0; i < operation.operands.size(); ++i) {
      if (operation.operands[i].variadic)
        fail("variadic index operands are unsupported");
      operands[operation.operands[i].name] = i;
      result.operands.push_back(operation.operands[i].name);
    }
    if (operation.attributes) {
      const auto& schema =
          context.output.attribute_schemas.at(operation.attributes->value);
      for (std::size_t i = 0; i < schema.fields.size(); ++i) {
        const auto& f = schema.fields[i];
        if (f.type != "index" || f.list_depth != 1 || f.optional)
          fail(
              "initial axis semantics supports required index[] attributes "
              "only");
        fields[f.name] = i;
        result.fields.push_back(f.name);
      }
    }
    for (const auto& statement : axes.statements) {
      if (const auto* m = std::get_if<ast::MetadataStatement>(&statement)) {
        if (const auto* l = std::get_if<ast::MetadataLet>(&m->value)) {
          at = detail::origin(source, m->span);
          if (locals.contains(l->name) || groups.contains(l->name) ||
              operands.contains(l->name))
            fail("duplicate axes name '" + l->name + "'");
          locals[l->name] = metadata(*l->value);
        } else {
          auto a = metadata(*std::get<ast::MetadataAssert>(m->value).condition);
          type(a, Type::Boolean);
          result.assertions.push_back(a);
        }
      } else {
        const auto& b = std::get<ast::AxisBind>(statement);
        at = detail::origin(source, b.span);
        if (locals.contains(b.name) || groups.contains(b.name) ||
            operands.contains(b.name))
          fail("duplicate axes name '" + b.name + "'");
        Group g{b.name, {}, at};
        if (b.domain) {
          const auto* call = std::get_if<ast::MetadataCall>(&b.domain->value);
          if (!call || call->callee != "group" || call->arguments.size() != 1)
            fail("an output-only binding requires group(extents)");
          g.domain = metadata(*call->arguments[0]);
          type(g.domain, Type::Positions);
          if (g.domain->kind == Meta::Kind::Attribute)
            result.extent_fields.insert(g.domain->index);
        }
        std::set<std::size_t> seen;
        for (const auto& binding : b.bindings) {
          at = detail::origin(source, binding.span);
          if (!operands.contains(binding.operand))
            fail("unknown binding operand '" + binding.operand + "'");
          auto o = operands.at(binding.operand);
          if (!seen.insert(o).second) fail("duplicate operand in bind group");
          auto p = metadata(*binding.positions);
          type(p, Type::Positions);
          g.bindings.push_back({o, p});
        }
        groups[b.name] = result.groups.size();
        result.groups.push_back(std::move(g));
      }
    }
    result.output = metadata(*axes.output);
    type(result.output, Type::Axes);
    result.value = value(*sem.result);
    std::set<std::size_t> reduced, reads, implicit;
    auto scan = [&](auto&& self, const Value& v) -> void {
      if (v.kind == Value::Kind::Reduce) reduced.insert(v.index);
      if (v.kind == Value::Kind::Read) reads.insert(v.index);
      if (v.kind == Value::Kind::Operand) implicit.insert(v.index);
      for (const auto& a : v.args) self(self, *a);
    };
    scan(scan, *result.value);
    auto output = outputGroups(*result.output);
    for (auto g : reduced)
      if (output.contains(g))
        fail(
            "output groups must be exactly the free groups of semantics; "
            "output contains a reduced group");
    auto dependencies = free(*result.value);
    for (auto g : dependencies)
      if (!output.contains(g))
        fail(
            "output groups must be exactly the free groups of semantics; "
            "missing a dependent group");
    for (auto g : output)
      if (!dependencies.contains(g) && !constantValue(*result.value) &&
          !result.groups[g].domain)
        fail("unused output groups require explicit extent domains");
    for (std::size_t o = 0; o < operation.operands.size(); ++o) {
      bool bound = false;
      for (const auto& g : result.groups)
        for (const auto& b : g.bindings) bound |= b.operand == o;
      if (reads.contains(o) && implicit.contains(o))
        fail("an operand cannot mix implicit and calculated reads");
      if (!bound && !reads.contains(o))
        fail("operand has no logical axis binding");
    }
    return result;
  }
};
}  // namespace

std::shared_ptr<const Definition> check(
    detail::AnalysisContext& context, const Operation& op,
    const ast::AxisDefinition& axes, const ast::SemanticDefinition& semantics,
    const std::string& source) {
  Checker checker{context, op, source};
  try {
    return std::make_shared<Definition>(checker.run(axes, semantics));
  } catch (const std::runtime_error& e) {
    context.report(checker.at, std::string("axis semantics: ") + e.what());
    return {};
  }
}
}  // namespace tepl::core::index_semantics
