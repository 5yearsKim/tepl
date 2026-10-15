#pragma once
#include <string>

#include "src/executable/program.h"
#include "src/rewrite/engine.h"
namespace tepl::rewrite {
int runCommand(const executable::ExecutableProgram&, const std::string& graph,
               const std::string& output, const Options&);
void writeOutput(const std::string& text, const std::string& output);
}  // namespace tepl::rewrite
