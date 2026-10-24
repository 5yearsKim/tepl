#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "src/core/semantics/definition.h"
#include "src/core/semantics/symbolic/symbol.h"
#include "src/shared/catalog/ids.h"
#include "src/shared/metadata/ir.h"
#include "src/shared/source/source.h"
#include "src/shared/types/types.h"

namespace tepl::core::semantics {

using symbolic::Symbol;
using symbolic::SymbolPtr;
struct Instance {
  std::string label, operation_name;
  std::optional<OpId> operation;
  std::optional<CaptureId> capture;
  std::optional<DescriptorId> descriptor;
  std::shared_ptr<const Definition> definition;
  std::vector<std::size_t> inputs;
  bool automatic = false;
  SourceOrigin origin;
  std::shared_ptr<const metadata::Program> dtype;
  std::optional<std::string> literal;
  std::optional<DType> literal_dtype;
};
enum class ProofStatus { Proved, Unknown };
enum class GuardKind { Legality, CoordinateOrder, ReductionDomain };
const char* guardKindName(GuardKind);
struct ConstructorCandidate {
  std::map<std::size_t, std::vector<SymbolPtr>> constructors;
  std::vector<SymbolPtr> guard;
};
struct RuleProof : ConstructorCandidate {
  ProofStatus status = ProofStatus::Unknown;
  std::string reason;
  struct GuardOrigin {
    SymbolPtr condition;
    GuardKind kind = GuardKind::Legality;
  };
  // Diagnostic provenance only; the verifier checks conditions independently.
  std::vector<GuardOrigin> guard_origins;
  std::vector<std::string> steps;
  std::string lhs_normalized, rhs_normalized;
  struct RewriteStep {
    std::string law, before, after;
    std::vector<SymbolPtr> conditions;
  };
  std::vector<RewriteStep> rewrites;
  std::vector<SymbolPtr> axis_definitions;
  std::map<std::size_t, SymbolPtr> accesses;
  SymbolPtr lhs_axes, rhs_axes;
};
struct Plan {
  std::vector<Instance> nodes;
  std::string suggested_rule;
  std::size_t lhs = 0, rhs = 0;
  bool algebraic = false;
  RuleProof proof;
  std::map<std::size_t, std::string> descriptor_names;
  std::vector<SymbolPtr> assumptions;
  // Explicit derived descriptors, lowered entirely to LHS metadata.
  std::map<std::size_t, std::vector<SymbolPtr>> attributes;
  // Original programs survive normalization so invalid indexing or checked
  // arithmetic cannot disappear from a derived descriptor's legality checks.
  std::vector<SymbolPtr> derived_metadata;
  std::vector<std::pair<SymbolPtr, core::Type>> constructed_metadata;
  std::map<std::pair<std::size_t, std::size_t>, core::Type> metadata_types;
};

// Every law has a direction, an arithmetic requirement, and explicit side
// conditions. Normalization uses these laws; the TEPL rule is never an axiom.
struct Law {
  std::string name, equation, conditions;
  bool algebraic;
};
const std::vector<Law>& laws();
std::string format(const Meta&);
std::string format(const Value&, const Definition&);
std::string format(const Definition&);
std::string format(const Plan&);
std::string format(const Symbol&);

using List = std::vector<std::int64_t>;

}  // namespace tepl::core::semantics
