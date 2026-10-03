#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "src/codegen/options.h"
#include "src/codegen/output.h"

namespace tepl::codegen {
struct WriteOptions {
  bool check = false;
  bool format = true;
  Target target = Target::kRust;
};
struct WriteResult {
  std::vector<std::string> differences;
  bool current() const { return differences.empty(); }
};
// Owns only paths listed in the generated-file manifest. Check mode never
// changes output. Formatting and path validation finish before output writes.
WriteResult synchronize(const std::vector<GeneratedFile>& files,
                        const std::filesystem::path& directory,
                        const WriteOptions& options = {});
}  // namespace tepl::codegen
