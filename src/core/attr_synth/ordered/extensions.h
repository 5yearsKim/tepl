#pragma once
#include "src/core/semantics/ir.h"
#include "src/inference/certificate.h"
namespace tepl::core::attr_synth::ordered {
std::optional<inference::MetadataTerm> metadataTerm(
    const semantics::SymbolPtr&);
bool suppliedExtensionFields(const semantics::Plan&, const inference::Program&,
                             const inference::LayoutFacts&);
}  // namespace tepl::core::attr_synth::ordered
