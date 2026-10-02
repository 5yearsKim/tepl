#pragma once

#include <string>
#include <vector>

#include "src/source.h"

namespace tepl::codegen {

struct GeneratedFile {
  // Relative to the output directory. Backends never write to the filesystem.
  std::string path;
  std::string contents;
};

struct Diagnostic {
  SourceOrigin origin;
  std::string message;
};

struct GenerationResult {
  std::vector<GeneratedFile> files;
  std::vector<Diagnostic> diagnostics;
  bool ok() const { return diagnostics.empty(); }
};

}  // namespace tepl::codegen
