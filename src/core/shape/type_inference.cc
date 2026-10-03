#include "src/core/shape/type_inference.h"

#include <set>
#include <utility>

namespace tepl::core::shape::detail {

TypeId TypeInference::variable(SourceOrigin origin) {
  TypeId id{variables_.size()};
  variables_.push_back({id.value, Kind::kUnknown, {}, std::move(origin)});
  return id;
}

TypeId TypeInference::concrete(Type type, SourceOrigin origin) {
  if (type.list_depth) {
    --type.list_depth;
    auto element = concrete(type, origin);
    return list(element, std::move(origin));
  }
  const auto id = variable(std::move(origin));
  variables_[id.value].kind =
      type.kind == Type::Kind::kInteger ? Kind::kInteger : Kind::kBoolean;
  return id;
}

TypeId TypeInference::list(TypeId element, SourceOrigin origin) {
  const auto id = variable(std::move(origin));
  variables_[id.value].kind = Kind::kList;
  variables_[id.value].element = element;
  return id;
}

std::size_t TypeInference::root(TypeId id) {
  auto& variable = variables_.at(id.value);
  if (variable.parent != id.value)
    variable.parent = root(TypeId{variable.parent});
  return variable.parent;
}

bool TypeInference::contains(TypeId type, std::size_t variable) {
  const auto id = root(type);
  if (id == variable) return true;
  return variables_[id].kind == Kind::kList &&
         contains(variables_[id].element, variable);
}

std::string TypeInference::describe(TypeId type) {
  const auto id = root(type);
  switch (variables_[id].kind) {
    case Kind::kInteger:
      return "Integer";
    case Kind::kBoolean:
      return "Bool";
    case Kind::kList:
      return "List<" + describe(variables_[id].element) + ">";
    case Kind::kUnknown:
      return "?";
  }
  return "?";
}

void TypeInference::unify(TypeId left, TypeId right,
                          const SourceOrigin& origin) {
  auto a = root(left);
  auto b = root(right);
  if (a == b) return;
  if (variables_[b].kind == Kind::kUnknown) std::swap(a, b);
  if (variables_[a].kind == Kind::kUnknown) {
    if (contains(TypeId{b}, a)) {
      diagnostics_.push_back({origin, "shape type would contain itself", {}});
      return;
    }
    variables_[a].parent = b;
    return;
  }
  if (variables_[a].kind != variables_[b].kind) {
    diagnostics_.push_back(
        {origin,
         "shape type mismatch: " + describe(left) + " and " + describe(right),
         {}});
    return;
  }
  if (variables_[a].kind == Kind::kList) {
    const auto before = diagnostics_.size();
    unify(variables_[a].element, variables_[b].element, origin);
    if (diagnostics_.size() != before) return;
  }
  variables_[a].parent = b;
}

std::optional<Type> TypeInference::resolve(TypeId type) {
  const auto id = root(type);
  switch (variables_[id].kind) {
    case Kind::kInteger:
      return Type{Type::Kind::kInteger};
    case Kind::kBoolean:
      return Type{Type::Kind::kBoolean};
    case Kind::kList:
      if (auto element = resolve(variables_[id].element)) {
        ++element->list_depth;
        return element;
      }
      return std::nullopt;
    case Kind::kUnknown:
      return std::nullopt;
  }
  return std::nullopt;
}

std::optional<std::vector<Type>> TypeInference::finish() {
  std::vector<Type> types;
  std::set<std::size_t> reported;
  bool complete = true;
  for (std::size_t i = 0; i < variables_.size(); ++i) {
    if (auto type = resolve(TypeId{i})) {
      types.push_back(*type);
    } else {
      complete = false;
      // Only report the unresolved leaf; outer lists share that cause.
      const auto id = root(TypeId{i});
      if (variables_[id].kind == Kind::kUnknown && reported.insert(id).second)
        diagnostics_.push_back(
            {variables_[id].origin,
             "cannot infer shape list element type; provide a typed use",
             {}});
    }
  }
  if (!complete) return std::nullopt;
  return types;
}

}  // namespace tepl::core::shape::detail
