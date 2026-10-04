#pragma once

#include <string>
#include <vector>

namespace tepl::support {
struct ProcessResult {
  int exit_code;
  std::string output;
};

// Run an argument vector directly, searching PATH and inheriting the
// environment. Supplies no stdin and collects combined stdout/stderr while the
// child runs. Throws if launch or wait fails; a nonzero child exit is returned
// normally.
ProcessResult runProcess(const std::vector<std::string>& arguments);
}  // namespace tepl::support
