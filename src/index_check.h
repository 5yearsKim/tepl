#pragma once
#include <ostream>
#include <string>
#include <vector>

#include "src/core/ir.h"
namespace tepl {
int checkIndexRules(const core::Program&, const std::string& rule,
                    std::ostream& output);
}  // namespace tepl
