#pragma once
#include "src/core/semantics/ir.h"
#include "src/inference/metadata_facts.h"
namespace tepl::core::attr_synth::ordered {
std::optional<inference::MetadataTerm> metadataTerm(
    const semantics::SymbolPtr&);
bool prepareInsertion(const semantics::Plan&, inference::Program&);
bool suppliedExtensionFields(const semantics::Plan&, const inference::Program&,
                             const inference::MetadataFacts&);
}  // namespace tepl::core::attr_synth::ordered
