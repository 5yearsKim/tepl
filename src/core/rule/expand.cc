#include "src/core/rule/expand.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "src/core/analysis_context.h"
#include "src/core/resolution/source.h"

namespace tepl::core::detail {
namespace {

class RuleExpander {
 public:
  RuleExpander(AnalysisContext& context, const ast::Rule& instance)
      : context_(context), instance_(instance), body_(&instance) {
    output_.name = instance.name;
    output_.source_name = instance.source_name;
    output_.origin = origin(instance.source_name, instance.span);
  }

  std::optional<ExpandedRule> run() {
    const auto diagnostic_count = context_.diagnostics.size();
    if (instance_.inheritance) {
      if (!selectTemplate()) return std::nullopt;
      bindParameters();
    }
    // A failed binding must not be resolved again as an ordinary operation.
    if (context_.diagnostics.size() != diagnostic_count) return std::nullopt;
    if (!body_->lhs || !body_->rhs) {
      context_.report(output_.origin,
                      "rule requires both LHS and RHS expressions");
      return std::nullopt;
    }

    output_.lhs = cloneGraph(*body_->lhs);
    output_.rhs = cloneGraph(*body_->rhs);
    appendRestrictions(*body_, 0);
    if (body_ != &instance_) appendRestrictions(instance_, 1);
    for (const auto& derivation : body_->derivations)
      output_.derivations.push_back({derivation.target.name,
                                     cloneExpression(*derivation.value, *body_),
                                     location(*body_, derivation.span)});
    if (context_.diagnostics.size() != diagnostic_count) return std::nullopt;
    return std::move(output_);
  }

 private:
  SourceOrigin location(const ast::Rule& definition, SourceSpan span) const {
    auto result = origin(definition.source_name, span);
    if (&definition != &instance_)
      result.expansions.push_back(
          origin(instance_.source_name, instance_.span).definition);
    return result;
  }

  bool selectTemplate() {
    const auto& inheritance = *instance_.inheritance;
    const auto at = location(instance_, inheritance.span);
    const auto& scope = context_.scopes.at(sourceKey(instance_.source_name));
    const auto found = scope.templates.find(inheritance.base);
    if (found == scope.templates.end()) {
      context_.report(at, "unknown base rule '" + inheritance.base + "'");
      return false;
    }
    body_ = found->second;
    if (!body_->is_abstract || body_->inheritance) {
      context_.report(
          at,
          "base rule '" + body_->name + "' must be an abstract rewrite rule",
          {origin(body_->source_name, body_->span).definition});
      return false;
    }
    output_.origin = location(*body_, body_->span);
    return true;
  }

  void bindParameters() {
    const auto& inheritance = *instance_.inheritance;
    std::map<std::string, const ast::RuleBinding*> bindings;
    for (const auto& binding : inheritance.bindings)
      if (!bindings.emplace(binding.parameter, &binding).second)
        context_.report(
            location(instance_, binding.span),
            "duplicate binding for parameter '" + binding.parameter + "'");

    std::set<std::string> parameters;
    for (const auto& parameter : body_->parameters) {
      parameters.insert(parameter.name);
      const auto binding = bindings.find(parameter.name);
      if (binding == bindings.end()) {
        context_.report(
            location(instance_, inheritance.span),
            "missing binding for parameter '" + parameter.name + "'");
        continue;
      }
      if (parameter.kind == ast::RuleParameterKind::kOperation)
        bindOperation(parameter, *binding->second);
      else
        bindFunction(parameter, *binding->second);
    }
    for (const auto& [name, binding] : bindings)
      if (!parameters.contains(name))
        context_.report(location(instance_, binding->span),
                        "unknown parameter binding '" + name + "'");
  }

  void bindOperation(const ast::RuleParameter& parameter,
                     const ast::RuleBinding& binding) {
    const auto at = location(instance_, binding.span);
    const auto id =
        context_.operation(binding.value, instance_.source_name, at);
    if (!id) return;
    const auto& op = context_.output.operations[id->value];
    const bool variadic = !op.operands.empty() && op.operands.back().variadic;
    bool compatible =
        !variadic && op.operands.size() == parameter.operand_types.size();
    for (std::size_t i = 0; compatible && i < op.operands.size(); ++i)
      compatible = context_.types.known(op.operands[i].type) ==
                   resolveType(parameter.operand_types[i].name);
    compatible = compatible && context_.types.known(op.result) ==
                                   resolveType(parameter.result_type.name);
    if (!compatible)
      context_.report(at,
                      "operation binding '" + binding.value +
                          "' does not match signature of '" + parameter.name +
                          "'",
                      {origin(body_->source_name, parameter.span).definition});
    op_bindings_.emplace(parameter.name, *id);
  }

  void bindFunction(const ast::RuleParameter& parameter,
                    const ast::RuleBinding& binding) {
    const auto at = location(instance_, binding.span);
    const auto& name = binding.value;
    if (name.find('.') != std::string::npos) {
      context_.report(
          at, "qualified host-function bindings are not supported in v1");
      return;
    }
    const auto id = context_.host(name, parameter.operand_types.size(), at);
    // Declaration resolution validates signature type names before expansion.
    const auto& signature = context_.output.host_functions[id.value].signature;
    for (std::size_t i = 0; i < std::min(signature.arguments.size(),
                                         parameter.operand_types.size());
         ++i)
      context_.types.unify(
          signature.arguments[i],
          context_.types.concrete(
              resolveType(parameter.operand_types[i].name).value(), at),
          at);
    context_.types.unify(
        signature.result,
        context_.types.concrete(resolveType(parameter.result_type.name).value(),
                                at),
        at);
    fn_bindings_.emplace(parameter.name, id);
  }

  bool isParameter(const std::string& name) const {
    return op_bindings_.contains(name) || fn_bindings_.contains(name);
  }

  ast::GraphExprPtr cloneGraph(const ast::GraphExpr& input) {
    auto result = std::make_shared<ast::GraphExpr>(input);
    const auto at = location(*body_, input.span);
    GraphMetadata metadata{at, std::nullopt};
    if (auto* op = std::get_if<ast::Operator>(&result->value)) {
      if (op->name == "tuple") {
        context_.report(
            at, "tuple semantics are not supported in the initial analyzer");
      } else if (fn_bindings_.contains(op->name)) {
        context_.report(
            at, "host-function parameter cannot be used as a graph operation");
      } else if (const auto bound = op_bindings_.find(op->name);
                 bound != op_bindings_.end()) {
        metadata.operation = bound->second;
      } else {
        // Unbound names belong to the definition's file, not the caller's.
        metadata.operation =
            context_.operation(op->name, body_->source_name, at);
      }
      for (auto& operand : op->operands) operand = cloneGraph(*operand);
    } else if (auto* bind = std::get_if<ast::Binding>(&result->value)) {
      if (isParameter(bind->binder.name))
        context_.report(at, "tensor binding conflicts with a rule parameter");
      bind->expression = cloneGraph(*bind->expression);
    } else if (auto* projection =
                   std::get_if<ast::Projection>(&result->value)) {
      context_.report(
          at, "projection semantics are not supported in the initial analyzer");
      projection->tuple = cloneGraph(*projection->tuple);
    } else if (const auto* name = std::get_if<ast::NameRef>(&result->value)) {
      if (isParameter(name->name))
        context_.report(at, "rule parameter '" + name->name +
                                "' cannot be a tensor capture");
    }
    output_.graphs.emplace(result.get(), std::move(metadata));
    return result;
  }

  ast::ConstraintExprPtr cloneExpression(const ast::ConstraintExpr& input,
                                         const ast::Rule& definition) {
    auto result = std::make_shared<ast::ConstraintExpr>(input);
    const auto at = location(definition, input.span);
    output_.expr_origins.emplace(result.get(), at);
    if (auto* call = std::get_if<ast::Call>(&result->value)) {
      if (&definition == body_) {
        if (op_bindings_.contains(call->callee))
          context_.report(
              at, "operation parameter cannot be called as a host function");
        if (const auto found = fn_bindings_.find(call->callee);
            found != fn_bindings_.end()) {
          call->callee =
              context_.output.host_functions[found->second.value].name;
          output_.bound_functions.emplace(call, found->second);
        }
      }
      for (auto& argument : call->arguments)
        argument = cloneExpression(*argument, definition);
    } else if (auto* unary = std::get_if<ast::UnaryExpr>(&result->value)) {
      unary->operand = cloneExpression(*unary->operand, definition);
    } else if (auto* binary = std::get_if<ast::BinaryExpr>(&result->value)) {
      binary->lhs = cloneExpression(*binary->lhs, definition);
      binary->rhs = cloneExpression(*binary->rhs, definition);
    }
    return result;
  }

  void appendRestrictions(const ast::Rule& definition, std::size_t layer) {
    for (const auto& declaration : definition.declarations)
      output_.declarations.push_back(
          {declaration, location(definition, declaration.span), layer});
    for (const auto& condition : definition.conditions)
      output_.conditions.push_back(cloneExpression(*condition, definition));
  }

  AnalysisContext& context_;
  const ast::Rule& instance_;
  const ast::Rule* body_;
  ExpandedRule output_;
  std::map<std::string, OpId> op_bindings_;
  std::map<std::string, HostFunctionId> fn_bindings_;
};

}  // namespace

std::optional<ExpandedRule> expand(AnalysisContext& context,
                                   const ast::Rule& rule) {
  return RuleExpander(context, rule).run();
}

}  // namespace tepl::core::detail
