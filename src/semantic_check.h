#pragma once
#include <ostream>
#include <string>

#include "src/core/ir.h"
namespace tepl {
int checkSemanticRules(const core::Program&, const std::string& rule,
                       std::ostream& output);
}  // namespace tepl
