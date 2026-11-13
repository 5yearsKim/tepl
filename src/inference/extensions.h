#pragma once
#include <string>
#include <utility>

#include "src/inference/emission.h"

namespace tepl::inference {
// A deliberately small metadata equality language. Facts originate only
// in successful source-guard conjunctions. No tensor values or solver queries.
MetadataTerm termField(std::size_t descriptor, std::size_t field);
MetadataTerm termShape(std::size_t capture);
MetadataTerm termNumber(std::int64_t);
MetadataTerm termCall(std::string, std::vector<MetadataTerm>);
std::string termKey(const MetadataTerm&);
using MetadataEquality = std::pair<MetadataTerm, MetadataTerm>;
struct MetadataFacts;
MetadataTerm normalizeMetadata(MetadataTerm, const MetadataFacts&);
bool metadataEqual(const MetadataTerm&, const MetadataTerm&,
                   const MetadataFacts&);
bool hasExtensions(const Program&);
// Pure signed core programs use the generic orientation DAG. Other extension
// operations still require the specialized schema interpreter.
bool usesSpecializedSchema(const Program&);
// Rejects every unrecognized graph pair; never falls back to canonical
// emission.
EmissionBlock emitExtension(const Program&);
MetadataTerm emissionTerm(const EmissionBlock&, std::size_t);
Result executeExtension(const Emission&, const Metadata&, const Limits&,
                        bool build, ExecutionTiming* = nullptr);
struct MetadataResult {
  Status status = Status::Unsupported;
  std::map<std::size_t, List> values;
  Attributes attributes;
  std::string reason;
  std::size_t work = 0;
};
MetadataResult executeMetadata(const EmissionBlock&, const Metadata&,
                               const Limits&, const std::vector<std::size_t>&,
                               std::optional<std::size_t> feasible = {});
}  // namespace tepl::inference
