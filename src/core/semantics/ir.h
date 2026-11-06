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
#include "src/shared/metadata/inference.h"
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
  SourceOrigin origin;
  std::shared_ptr<const metadata::Program> dtype;
  std::optional<std::string> literal;
  std::optional<DType> literal_dtype;
};
enum class ProofStatus { Proved, Unknown };
enum class GuardKind { Legality, CoordinateOrder, ReductionDomain };
const char* guardKindName(GuardKind);
struct ConstructorCandidate {
  // Generated feasibility and independently validated attribute builders.
  std::vector<SymbolPtr> feasibility;
  std::map<std::size_t, std::vector<SymbolPtr>> constructors;
  std::vector<SymbolPtr> guard;
};
struct RuleProof : ConstructorCandidate {
  ProofStatus status = ProofStatus::Unknown;
  std::optional<inference::Program> core;
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
// Captured descriptors read runtime source metadata. Constructed descriptors
// have every field in schema order; a direct Hole marks a field for synthesis.
// Defaults have already been expanded by checking and are ordinary supplied
// fields.
struct DescriptorAttributes {
  enum class Origin { Captured, Constructed } origin = Origin::Captured;
  std::vector<SymbolPtr> fields;
  bool isCaptured() const { return origin == Origin::Captured; }
  bool isAuto(std::size_t field) const {
    return !isCaptured() && fields.at(field)->kind == Symbol::Kind::Hole;
  }
  bool needsSynthesis() const;
  bool allAuto() const;
};
struct Plan {
  std::vector<Instance> nodes;
  std::size_t lhs = 0, rhs = 0;
  bool algebraic = false;
  bool core_backend = false;
  RuleProof proof;
  std::map<std::size_t, std::string> descriptor_names;
  std::vector<SymbolPtr> assumptions;
  // One canonical record per descriptor, including supplied and auto fields.
  std::map<std::size_t, DescriptorAttributes> descriptors;
  const DescriptorAttributes& attributes(const Instance& node) const {
    return descriptors.at(node.descriptor.value().value);
  }
  bool needsSynthesis(const Instance& node) const {
    return node.descriptor && attributes(node).needsSynthesis();
  }
  bool isCaptured(const Instance& node) const {
    return node.descriptor && attributes(node).isCaptured();
  }
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
