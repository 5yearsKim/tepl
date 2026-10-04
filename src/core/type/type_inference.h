#pragma once

#include <optional>
#include <string>
#include <vector>

#include "src/core/diagnostic.h"
#include "src/core/type/types.h"

namespace tepl::core::detail {

enum class TypeRequirement { kNumeric, kSignedNumeric, kInteger, kEquality };

// Collects type equations independently of AST traversal and symbol resolution.
// TypeIds remain valid when finish() materializes the checked program's table.
class TypeInference {
 public:
  explicit TypeInference(std::vector<Diagnostic>& diagnostics)
      : diagnostics_(diagnostics) {}

  TypeId variable(SourceOrigin origin);
  TypeId concrete(Type type, SourceOrigin origin = {});
  std::optional<Type> known(TypeId id);
  void unify(TypeId left, TypeId right, const SourceOrigin& origin);
  void require(TypeId type, TypeRequirement requirement, SourceOrigin origin);
  void literal(TypeId type, std::string spelling, SourceOrigin origin);

  // Defaults integer literals only after all equations have been collected.
  // An unresolved or inconsistent type never becomes a placeholder IR type.
  std::optional<std::vector<Type>> finish();

 private:
  struct Variable {
    std::size_t parent;
    std::size_t rank = 0;
    std::optional<Type> type;
    SourceOrigin origin;
  };
  struct Requirement {
    TypeId type;
    TypeRequirement kind;
    SourceOrigin origin;
  };
  struct Literal {
    TypeId type;
    std::string spelling;
    SourceOrigin origin;
  };

  std::size_t root(TypeId id);
  void defaultLiterals();
  void checkRequirements();
  void checkLiterals();
  void report(const SourceOrigin& origin, std::string message,
              std::vector<SourceLocation> related = {});

  std::vector<Diagnostic>& diagnostics_;
  std::vector<Variable> variables_;
  std::vector<Requirement> requirements_;
  std::vector<Literal> literals_;
};

}  // namespace tepl::core::detail
