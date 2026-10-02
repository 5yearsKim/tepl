#include "src/core/type_inference.h"

#include <map>
#include <utility>

#include "src/core/literal.h"

namespace tepl::core::detail {

void TypeInference::report(const SourceOrigin& origin, std::string message,
                           std::vector<SourceLocation> related) {
  diagnostics_.push_back({origin, std::move(message), std::move(related)});
}

TypeId TypeInference::variable(SourceOrigin origin) {
  const TypeId id{variables_.size()};
  variables_.push_back({id.value, 0, std::nullopt, std::move(origin)});
  return id;
}

TypeId TypeInference::concrete(Type type, SourceOrigin origin) {
  const auto id = variable(std::move(origin));
  variables_[id.value].type = type;
  return id;
}

std::size_t TypeInference::root(TypeId id) {
  auto result = id.value;
  while (variables_.at(result).parent != result)
    result = variables_[result].parent;
  // Iterative compression avoids recursion on long chains of inferred calls.
  auto current = id.value;
  while (current != result) {
    const auto next = variables_[current].parent;
    variables_[current].parent = result;
    current = next;
  }
  return result;
}

std::optional<Type> TypeInference::known(TypeId id) {
  return variables_[root(id)].type;
}

void TypeInference::unify(TypeId left, TypeId right,
                          const SourceOrigin& origin) {
  auto a = root(left), b = root(right);
  if (a == b) return;
  auto first = variables_[a].type, second = variables_[b].type;
  if (first && second) {
    if (first->kind != second->kind ||
        (first->schema && second->schema && first->schema != second->schema)) {
      report(
          origin,
          "type mismatch: " + typeName(*first) + " versus " + typeName(*second),
          {variables_[a].origin.definition, variables_[b].origin.definition});
      return;
    }
    if (!first->schema) first->schema = second->schema;
  }
  const auto type = first ? first : second;
  const auto at = first ? variables_[a].origin : variables_[b].origin;
  if (variables_[a].rank < variables_[b].rank) std::swap(a, b);
  variables_[b].parent = a;
  if (variables_[a].rank == variables_[b].rank) ++variables_[a].rank;
  variables_[a].type = type;
  variables_[a].origin = at;
}

void TypeInference::require(TypeId type, TypeRequirement requirement,
                            SourceOrigin origin) {
  requirements_.push_back({type, requirement, std::move(origin)});
}

void TypeInference::literal(TypeId type, std::string spelling,
                            SourceOrigin origin) {
  literals_.push_back({type, std::move(spelling), std::move(origin)});
}

void TypeInference::defaultLiterals() {
  std::map<std::size_t, TypeKind> defaults;
  for (const auto& literal : literals_) {
    const auto id = root(literal.type);
    if (variables_[id].type) continue;
    auto [found, inserted] = defaults.emplace(id, TypeKind::kIndex);
    if (literal.spelling.starts_with('-')) found->second = TypeKind::kI64;
  }
  for (const auto& requirement : requirements_) {
    if (requirement.kind == TypeRequirement::kSignedNumeric) {
      const auto found = defaults.find(root(requirement.type));
      if (found != defaults.end()) found->second = TypeKind::kI64;
    }
  }
  for (const auto& [id, kind] : defaults) variables_[id].type = Type{kind};
}

void TypeInference::checkRequirements() {
  for (const auto& requirement : requirements_) {
    const auto type = known(requirement.type);
    if (!type) continue;
    bool valid = false;
    switch (requirement.kind) {
      case TypeRequirement::kNumeric:
        valid = isNumeric(type->kind);
        break;
      case TypeRequirement::kSignedNumeric:
        valid = type->kind == TypeKind::kI64 || type->kind == TypeKind::kF64;
        break;
      case TypeRequirement::kInteger:
        valid = isInteger(type->kind);
        break;
      case TypeRequirement::kEquality:
        valid = isNumeric(type->kind) || type->kind == TypeKind::kBool;
        break;
    }
    if (!valid)
      report(requirement.origin,
             "operator does not support " + typeName(*type));
  }
}

void TypeInference::checkLiterals() {
  for (const auto& literal : literals_) {
    const auto type = known(literal.type);
    // Nonnumeric types are already diagnosed by the expression requirements.
    if (!type || !isNumeric(type->kind)) continue;
    if (!validHostLiteral(literal.spelling, type->kind))
      report(literal.origin, "host literal '" + literal.spelling +
                                 "' is invalid for " + typeName(*type));
  }
}

std::optional<std::vector<Type>> TypeInference::finish() {
  defaultLiterals();
  for (std::size_t i = 0; i < variables_.size(); ++i) {
    if (root(TypeId{i}) == i && !variables_[i].type)
      report(variables_[i].origin,
             "cannot infer expression or host-function type; provide a typed "
             "context or a function parameter signature");
  }
  checkRequirements();
  checkLiterals();
  if (!diagnostics_.empty()) return std::nullopt;
  std::vector<Type> result;
  result.reserve(variables_.size());
  for (std::size_t i = 0; i < variables_.size(); ++i)
    result.push_back(known(TypeId{i}).value());
  return result;
}

}  // namespace tepl::core::detail
