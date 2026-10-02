#pragma once

#include <string>
#include <vector>

#include "src/source.h"

namespace tepl::core {

struct Diagnostic {
  SourceOrigin origin;
  std::string message;
  std::vector<SourceLocation> related;
};

}  // namespace tepl::core
