#include "src/core/semantics/check.h"

#include <algorithm>
#include <map>
#include <set>

#include "src/core/analysis_context.h"
#include "src/core/resolution/source.h"

namespace tepl::core::semantics {
namespace {
using T = Type;
using A = ast::SemanticExpr;
ExprPtr make(std::string op, T type, std::vector<ExprPtr> args = {},
             SourceOrigin at = {}) {
  return std::make_shared<Expr>(
      Expr{std::move(op), type, std::move(args), std::move(at)});
}
bool integerConstant(const Expr& e) {
  if (e.type != T::kInteger) return false;
  if (e.arguments.empty())
    return !e.op.empty() &&
           e.op.find_first_not_of("0123456789") == std::string::npos;
  if (e.op != "+" && e.op != "-" && e.op != "*" && e.op != "unary:+" &&
      e.op != "unary:-")
    return false;
  return std::all_of(e.arguments.begin(), e.arguments.end(),
                     [](const auto& a) { return integerConstant(*a); });
}
ExprPtr scalarConstant(ExprPtr e) {
  if (integerConstant(*e))
    return make(e->op, T::kScalar, e->arguments, e->origin);
  return e;
}
struct Checker {
  detail::AnalysisContext& context;
  const Operation& operation;
  std::string source;
  Program program;
  std::size_t binder = 0;
  using Scope = std::map<std::string, ExprPtr>;
  void error(const A& e, const std::string& message) {
    context.report(detail::origin(source, e.span), "semantics: " + message);
  }
  ExprPtr expression(const A& e, const Scope& scope) {
    auto at = detail::origin(source, e.span);
    auto fail = [&](std::string message) {
      error(e, message);
      return make("invalid", T::kOpaque, {}, at);
    };
    if (e.kind == A::Kind::kName) {
      auto i = scope.find(e.text);
      if (i == scope.end()) return fail("unknown name '" + e.text + "'");
      return i->second;
    }
    if (e.kind == A::Kind::kInteger) return make(e.text, T::kInteger, {}, at);
    if (e.kind == A::Kind::kBool) return make(e.text, T::kBool, {}, at);
    if (e.kind == A::Kind::kAttrs) return make("attrs", T::kAttrs, {}, at);
    if (e.kind == A::Kind::kSum || e.kind == A::Kind::kComprehension) {
      auto domain = expression(*e.arguments[0], scope);
      const bool sum = e.kind == A::Kind::kSum;
      if (domain->type != (sum ? T::kDomain : T::kVector))
        return fail("invalid reduction/iteration domain");
      Scope inner = scope;
      auto name = std::string(sum ? "k" : "v") + std::to_string(binder++);
      if (inner.contains(e.text))
        return fail("binder conflicts with existing name '" + e.text + "'");
      inner[e.text] = make(name, sum ? T::kVector : T::kInteger);
      locals(e.locals, inner);
      auto body = expression(*e.arguments[1], inner);
      if (sum) body = scalarConstant(body);
      if (body->type != (sum ? T::kScalar : T::kInteger))
        return fail("reduction/iteration body has the wrong type");
      return make(sum ? "sum:" + name : "map:" + name,
                  sum ? T::kScalar : T::kVector, {domain, body}, at);
    }
    std::vector<ExprPtr> args;
    for (auto& arg : e.arguments) args.push_back(expression(*arg, scope));
    if (e.kind == A::Kind::kField) {
      if (args[0]->type != T::kAttrs || !operation.attributes)
        return fail("attribute field requires an attribute schema");
      const auto& fields =
          context.output.attribute_schemas.at(operation.attributes->value)
              .fields;
      for (std::size_t i = 0; i < fields.size(); ++i)
        if (fields[i].name == e.text) {
          auto& f = fields[i];
          auto type = T::kOpaque;
          if (!f.optional && !f.list_depth &&
              (f.type == "index" || f.type == "i64"))
            type = T::kInteger;
          if (!f.optional && !f.list_depth && f.type == "bool") type = T::kBool;
          if (!f.optional && f.list_depth == 1 &&
              (f.type == "index" || f.type == "i64"))
            type = T::kVector;
          program.used_fields.push_back(i);
          return make("a" + std::to_string(i), type, {}, at);
        }
      return fail("unknown attribute field '" + e.text + "'");
    }
    if (e.kind == A::Kind::kList) {
      for (auto& a : args)
        if (a->type != T::kInteger)
          return fail("index vectors require integer entries");
      return make("list", T::kVector, args, at);
    }
    if (e.kind == A::Kind::kIndex) {
      if (args[0]->type != T::kVector || args[1]->type != T::kInteger)
        return fail("indexing requires an index vector and integer position");
      return make("index", T::kInteger, args, at);
    }
    if (e.kind == A::Kind::kConditional) {
      if (args[0]->type != T::kBool || args[1]->type != args[2]->type)
        return fail("invalid conditional types");
      return make("if", args[1]->type, args, at);
    }
    if (e.kind == A::Kind::kUnary) {
      if (e.text == "!") {
        if (args[0]->type != T::kBool) return fail("! requires Bool");
      } else if (args[0]->type != T::kInteger && args[0]->type != T::kScalar)
        return fail("unary arithmetic requires a number");
      return make("unary:" + e.text, args[0]->type, args, at);
    }
    if (e.kind == A::Kind::kBinary) {
      auto l = args[0]->type, r = args[1]->type;
      if (e.text == "&&" || e.text == "||") {
        if (l != T::kBool || r != T::kBool)
          return fail("logical operation requires Bool operands");
        return make(e.text, T::kBool, args, at);
      }
      if (e.text == "==" || e.text == "!=") {
        if (l != r || l == T::kTensor || l == T::kAttrs || l == T::kOpaque)
          return fail("incompatible equality types");
        return make(e.text, T::kBool, args, at);
      }
      // Integer constant expressions can be exact scalar coefficients. Metadata
      // integers remain distinct; converting arbitrary metadata expressions is
      // deferred.
      auto scalarLiteral = [&](std::size_t i) {
        auto before = args[i]->type;
        args[i] = scalarConstant(args[i]);
        return before == T::kInteger && args[i]->type == T::kScalar;
      };
      if (l == T::kScalar && scalarLiteral(1)) r = T::kScalar;
      if (r == T::kScalar && scalarLiteral(0)) l = T::kScalar;
      if (l != r || (l != T::kInteger && l != T::kScalar))
        return fail("arithmetic requires compatible numeric operands");
      bool compare =
          e.text == "<" || e.text == ">" || e.text == "<=" || e.text == ">=";
      return make(e.text, compare ? T::kBool : l, args, at);
    }
    if (e.kind != A::Kind::kCall) return fail("unsupported expression");
    auto signature = [&](std::initializer_list<T> types, T result) -> ExprPtr {
      if (args.size() != types.size())
        return fail("wrong arity for '" + e.text + "'");
      std::size_t i = 0;
      for (auto t : types)
        if (args[i++]->type != t)
          return fail("wrong argument type for '" + e.text + "'");
      return make(e.text, result, args, at);
    };
    if (e.text == "read")
      return signature({T::kTensor, T::kVector}, T::kScalar);
    if (e.text == "shape") return signature({T::kTensor}, T::kVector);
    if (e.text == "rank") return signature({T::kTensor}, T::kInteger);
    if (e.text == "len") return signature({T::kVector}, T::kInteger);
    if (e.text == "range") return signature({T::kInteger}, T::kVector);
    if (e.text == "gather" || e.text == "exclude")
      return signature({T::kVector, T::kVector}, T::kVector);
    if (e.text == "index_space") return signature({T::kVector}, T::kDomain);
    if (e.text == "take" || e.text == "slice")
      return signature({T::kVector, T::kInteger, T::kInteger}, T::kVector);
    if (e.text == "scatter")
      return signature({T::kInteger, T::kVector, T::kVector}, T::kVector);
    if (e.text == "concat") {
      if (args.size() < 2) return fail("concat requires at least two vectors");
      for (auto& a : args)
        if (a->type != T::kVector) return fail("concat requires vectors");
      return make(e.text, T::kVector, args, at);
    }
    if (e.text == "is_valid_axis_list")
      return signature({T::kVector, T::kInteger}, T::kBool);
    if (e.text == "is_disjoint")
      return signature({T::kVector, T::kVector}, T::kBool);
    return fail("unknown primitive '" + e.text + "'");
  }
  void locals(const std::vector<ast::SemanticLet>& lets, Scope& scope) {
    for (auto& l : lets) {
      auto v = expression(*l.value, scope);
      if (!scope.emplace(l.name, v).second)
        context.report(detail::origin(source, l.span),
                       "semantics: duplicate local '" + l.name + "'");
    }
  }
};
}  // namespace
std::optional<Program> check(detail::AnalysisContext& context,
                             const Operation& op,
                             const ast::SemanticDefinition& def,
                             const std::string& source) {
  auto before = context.diagnostics.size();
  Checker c{context, op, source};
  Checker::Scope scope;
  for (std::size_t i = 0; i < op.operands.size(); ++i) {
    if (op.operands[i].variadic) {
      context.report(op.origin,
                     "semantics: variadic operands are not supported yet");
      continue;
    }
    scope.emplace(op.operands[i].name,
                  make("t" + std::to_string(i), T::kTensor));
  }
  if (!def.index.empty() &&
      !scope.emplace(def.index, make("o", T::kVector)).second)
    context.report(op.origin,
                   "semantics: output index conflicts with operand name");
  c.locals(def.locals, scope);
  c.program.result = scalarConstant(c.expression(*def.result, scope));
  if (c.program.result->type != T::kScalar)
    context.report(detail::origin(source, def.span),
                   "semantics: yield must be a scalar element");
  if (context.diagnostics.size() != before) return std::nullopt;
  std::sort(c.program.used_fields.begin(), c.program.used_fields.end());
  c.program.used_fields.erase(
      std::unique(c.program.used_fields.begin(), c.program.used_fields.end()),
      c.program.used_fields.end());
  return std::move(c.program);
}
}  // namespace tepl::core::semantics
