#pragma once
#include <string>
#include <utility>

#include "src/inference/emission.h"

namespace tepl::inference {
// A deliberately small metadata equality language. Facts originate only
// in successful source-guard conjunctions. No tensor values or solver queries.
struct MetadataTerm {
  std::string op;
  std::vector<MetadataTerm> args;
  bool operator==(const MetadataTerm&) const = default;
};
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
// Rejects every unrecognized graph pair; never falls back to canonical
// emission.
EmissionBlock emitExtension(const Program&);
MetadataTerm emissionTerm(const EmissionBlock&, std::size_t);
Result executeExtension(const Emission&, const Metadata&, const Limits&,
                        bool build, ExecutionTiming* = nullptr);
}  // namespace tepl::inference
