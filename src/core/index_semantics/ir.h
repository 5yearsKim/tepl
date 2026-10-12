#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "src/core/ids.h"
#include "src/core/metadata/ir.h"
#include "src/source.h"

namespace tepl::core::index_semantics {

// Operation definitions retain arbitrary-length groups and position
// expressions.
enum class Type { Integer, Boolean, Positions, Axes, Operand };
struct Meta;
using MetaPtr = std::shared_ptr<const Meta>;
struct Meta {
  enum class Kind {
    Integer,
    Boolean,
    Operand,
    Attribute,
    Group,
    List,
    Call
  } kind;
  Type type;
  std::string name;
  std::size_t index = 0;
  std::vector<MetaPtr> args;
  SourceOrigin origin;
};
struct Binding {
  std::size_t operand;
  MetaPtr positions;
};
struct Group {
  std::string name;
  std::vector<Binding> bindings;
  SourceOrigin origin;
};
struct Value;
using ValuePtr = std::shared_ptr<const Value>;
struct Value {
  enum class Kind { Operand, Add, Product, Reduce } kind;
  std::size_t index = 0;  // operand or reduction group
  std::vector<ValuePtr> args;
  SourceOrigin origin;
};
struct Definition {
  std::vector<std::string> operands, fields;
  std::vector<Group> groups;
  std::vector<MetaPtr> assertions;
  MetaPtr output;
  ValuePtr value;
};

struct Pin {
  std::optional<std::vector<std::int64_t>> literal;
  std::optional<DescriptorId> descriptor;
  std::size_t field = 0;
};
struct Instance {
  std::optional<CaptureId> capture;
  std::optional<DescriptorId> descriptor;
  std::shared_ptr<const Definition> definition;
  std::vector<std::size_t> inputs;
  bool automatic = false;
  std::vector<std::optional<Pin>> pins;
  SourceOrigin origin;
  std::shared_ptr<const metadata::Program> dtype;
};
// Immutable, rule-scoped metadata expressions. No tensor values or fixed ranks.
struct Symbol;
using SymbolPtr = std::shared_ptr<const Symbol>;
struct Symbol {
  enum class Sort { Integer, Boolean, Positions, Axes, Extents, DType };
  enum class Kind {
    Integer,
    Boolean,
    Shape,
    DType,
    Field,
    Hole,
    AxisGroup,
    List,
    Call
  } kind;
  std::string name;
  std::size_t owner = 0, index = 0;
  std::vector<SymbolPtr> args;
  Sort sort = Sort::Positions;
  Symbol(Kind k, std::string n = {}, std::size_t o = 0, std::size_t i = 0,
         std::vector<SymbolPtr> a = {}, Sort t = Sort::Positions)
      : kind(k),
        name(std::move(n)),
        owner(o),
        index(i),
        args(std::move(a)),
        sort(t) {}
  // Copies are external, editable IR nodes: recompute their structural
  // identity.
  Symbol(const Symbol& s)
      : Symbol(s.kind, s.name, s.owner, s.index, s.args, s.sort) {}
  Symbol(Symbol&&) = default;
  const std::string& structuralKey() const { return structural_key; }

 private:
  friend SymbolPtr make(Kind, std::string, std::size_t, std::size_t,
                        std::vector<SymbolPtr>);
  friend std::string key(const SymbolPtr&);
  std::string structural_key;
};
enum class ProofStatus { Proved, Conditional, Unknown };
struct RuleProof {
  ProofStatus status = ProofStatus::Unknown;
  std::string reason;
  std::map<std::size_t, std::vector<SymbolPtr>> constructors;
  std::vector<SymbolPtr> guard;
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
  std::size_t lhs = 0, rhs = 0;
  bool algebraic = false;
  RuleProof proof;
  std::map<std::size_t, std::string> descriptor_names;
  std::vector<SymbolPtr> assumptions;
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

}  // namespace tepl::core::index_semantics
