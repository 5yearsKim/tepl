#include "src/core/attr_synth/ordered/extensions.h"

#include <charconv>
#include <functional>

#include "src/core/semantics/symbolic/expr.h"
#include "src/inference/insertion.h"
namespace tepl::core::attr_synth::ordered {
using namespace semantics;
namespace {
using Term = inference::MetadataTerm;
bool call(const Term& t, const char* op, std::size_t count) {
  return t.op == op && t.args.size() == count;
}
// Recognize exactly the finite permutation inverse/scatter comprehensions.
// Keep the scatter bound in the term; acceptance must match the generated
// expression whose source legality establishes its permutation domain.
Term comprehension(Term t) {
  if (!call(t, "sequence_map", 3) || !call(t.args[0], "range", 1)) return t;
  const auto& n = t.args[0].args[0];
  const auto& i = t.args[1];
  if (!call(i, "sequence_variable", 1) || !call(t.args[2], "sum", 1)) return t;
  const auto& inner = t.args[2].args[0];
  if (!call(inner, "sequence_map", 3) || inner.args[0] != t.args[0]) return t;
  const auto& j = inner.args[1];
  const auto& body = inner.args[2];
  if (!call(j, "sequence_variable", 1) || i == j || !call(body, "if", 3) ||
      body.args[2] != inference::termNumber(0) || !call(body.args[0], "==", 2))
    return t;
  const auto& eq = body.args[0];
  for (std::size_t side = 0; side < 2; ++side) {
    const auto& at = eq.args[side];
    if (!call(at, "index", 2) || at.args[1] != j || eq.args[1 - side] != i)
      continue;
    const auto& positions = at.args[0];
    const auto& value = body.args[1];
    if (value == j && n == Term{"len", {positions}})
      return {"inverse", {positions}};
    if (call(value, "index", 2) && value.args[1] == j)
      return {"scatter_n", {positions, value.args[0], n}};
  }
  return t;
}
}  // namespace
std::optional<inference::MetadataTerm> metadataTerm(const SymbolPtr& s) {
  using T = inference::MetadataTerm;
  if (s->kind == Symbol::Kind::Field)
    return inference::termField(s->owner, s->index);
  if (s->kind == Symbol::Kind::Shape) return inference::termShape(s->owner);
  if (s->kind == Symbol::Kind::Integer) {
    std::int64_t n;
    auto r =
        std::from_chars(s->name.data(), s->name.data() + s->name.size(), n);
    if (r.ec != std::errc{} || r.ptr != s->name.data() + s->name.size())
      return {};
    return inference::termNumber(n);
  }
  if (s->kind != Symbol::Kind::List && s->kind != Symbol::Kind::Call) return {};
  auto op = s->kind == Symbol::Kind::List ? "list" : s->name;
  if (op == "ordered_complement") op = "exclude";
  if (op != "list" && op != "len" && op != "range" && op != "concat" &&
      op != "gather" && op != "index" && op != "+" && op != "exclude" &&
      op != "sequence_map" && op != "sequence_variable" && op != "sum" &&
      op != "if" && op != "==")
    return {};
  T out{op, {}};
  for (const auto& a : s->args) {
    auto t = metadataTerm(a);
    if (!t) return {};
    out.args.push_back(std::move(*t));
  }
  return comprehension(std::move(out));
}
bool prepareInsertion(const Plan& p, inference::Program& program) {
  if (!inference::insertionCandidate(program)) return false;
  for (const auto& assumption : p.assumptions)
    if (containsHole(assumption)) return false;
  auto prepared = program;
  prepared.supplied.clear();
  prepared.supplied_checks.clear();
  const auto& root = program.replacement.back();
  for (const auto& node : program.replacement) {
    if (!node.descriptor) continue;
    const auto& descriptor = p.descriptors.at(*node.descriptor);
    if (node.node == root.node && node.kind == inference::Kind::Broadcast) {
      if (!descriptor.isAuto(node.fields[0]) ||
          descriptor.isAuto(node.fields[1]))
        return false;
      auto value = metadataTerm(descriptor.fields.at(node.fields[1]));
      if (!value) return false;
      prepared.supplied.push_back({*node.descriptor, node.fields[1], *value});
    } else if (!descriptor.allAuto()) {
      return false;
    }
  }
  for (const auto& raw : p.derived_metadata) {
    auto value = metadataTerm(raw);
    if (!value) return false;
    prepared.supplied_checks.push_back(*value);
  }
  prepared.restricted_insertion = true;
  try {
    inference::emitInsertion(prepared);
  } catch (const std::invalid_argument&) {
    return false;
  }
  program = std::move(prepared);
  return true;
}
bool suppliedExtensionFields(const Plan& p, const inference::Program& program,
                             const inference::MetadataFacts& f) {
  if (program.restricted_insertion) {
    auto expected = program;
    return prepareInsertion(p, expected) &&
           expected.supplied == program.supplied &&
           expected.supplied_checks == program.supplied_checks;
  }
  auto b = inference::emitExtension(program);
  std::set<std::string> safe;
  std::function<void(const inference::MetadataTerm&)> remember =
      [&](const auto& t) {
        safe.insert(inference::termKey(t));
        for (const auto& a : t.args) remember(a);
      };
  // First collect only terms whose domains are justified by the schema.
  // Do not grant definedness to a supplied expression merely because a
  // simplification erased its value (e.g. gather(undefined, [])).
  for (const auto& [id, fields] : b.attributes)
    for (auto field : fields) {
      auto expected = inference::emissionTerm(b, field);
      remember(expected);
      remember(inference::normalizeMetadata(expected, {}));
      remember(inference::normalizeMetadata(expected, f));
    }
  for (const auto& [left, right] : f.equalities) {
    remember(left);
    remember(right);
  }
  std::set<std::string> source_reads;
  for (const auto& n : program.source) {
    if (n.capture)
      source_reads.insert(inference::termKey(inference::termShape(*n.capture)));
    if (n.descriptor)
      for (auto field : n.fields)
        source_reads.insert(
            inference::termKey(inference::termField(*n.descriptor, field)));
  }
  std::function<bool(const inference::MetadataTerm&)> defined =
      [&](const auto& t) {
        for (const auto& child : t.args)
          if (!defined(child)) return false;
        const auto k = inference::termKey(t);
        if (source_reads.contains(k) || (!t.op.empty() && t.op[0] == '#') ||
            t.op == "list" || t.op == "len" || t.op == "concat")
          return true;
        if (safe.contains(k) || safe.contains(inference::termKey(
                                    inference::normalizeMetadata(t, f))))
          return true;
        // Literal-bound indexing may also arise from a fixed capture
        // constraint.
        if ((t.op == "index" || t.op == "gather") && t.args.size() == 2) {
          auto length = inference::normalizeMetadata({"len", {t.args[0]}}, f);
          std::int64_t n;
          if (length.op.empty() || length.op[0] != '#') return false;
          auto r = std::from_chars(length.op.data() + 1,
                                   length.op.data() + length.op.size(), n);
          if (r.ec != std::errc{} || n < 0) return false;
          auto indices = inference::normalizeMetadata(t.args[1], f);
          if (t.op == "index") indices = {"list", {indices}};
          if (indices.op != "list") return false;
          for (const auto& index : indices.args) {
            std::int64_t i;
            if (index.op.empty() || index.op[0] != '#') return false;
            auto v = std::from_chars(index.op.data() + 1,
                                     index.op.data() + index.op.size(), i);
            if (v.ec != std::errc{} || i < 0 || i >= n) return false;
          }
          return true;
        }
        return false;
      };
  for (const auto& [id, fields] : b.attributes) {
    auto it = p.descriptors.find(id);
    if (it == p.descriptors.end() || it->second.fields.size() != fields.size())
      return false;
    for (std::size_t i = 0; i < fields.size(); ++i) {
      if (it->second.isAuto(i)) continue;
      auto actual = metadataTerm(it->second.fields[i]);
      auto expected = inference::emissionTerm(b, fields[i]);
      if (!actual || !defined(*actual) ||
          !inference::metadataEqual(*actual, expected, f))
        return false;
    }
  }
  // Preserve the domain of every supplied metadata expression. Unsupported
  // helpers do not disappear when descriptors are replaced by a schema DAG.
  for (const auto& raw : p.derived_metadata) {
    auto t = metadataTerm(raw);
    if (!t || !defined(*t)) return false;
  }
  return true;
}
}  // namespace tepl::core::attr_synth::ordered
