#pragma once

#include <optional>
#include <vector>

#include "src/core/diagnostic.h"
#include "src/core/shape/ir.h"

namespace tepl::core::shape::detail {

// Resolves homogeneous nested lists, including initially untyped [] literals.
class TypeInference {
 public:
  explicit TypeInference(std::vector<Diagnostic>& diagnostics)
      : diagnostics_(diagnostics) {}
  TypeId variable(SourceOrigin origin);
  TypeId concrete(Type type, SourceOrigin origin);
  TypeId list(TypeId element, SourceOrigin origin);
  void unify(TypeId left, TypeId right, const SourceOrigin& origin);
  std::optional<std::vector<Type>> finish();

 private:
  enum class Kind { kUnknown, kInteger, kBoolean, kList };
  struct Variable {
    std::size_t parent;
    Kind kind = Kind::kUnknown;
    TypeId element;
    SourceOrigin origin;
  };
  std::size_t root(TypeId id);
  bool contains(TypeId type, std::size_t variable);
  std::string describe(TypeId type);
  std::optional<Type> resolve(TypeId type);
  std::vector<Variable> variables_;
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace tepl::core::shape::detail
