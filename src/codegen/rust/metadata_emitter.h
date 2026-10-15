#pragma once
#include <string>

#include "src/executable/ir.h"
namespace tepl::codegen::rust {
std::string emitMetadata(const executable::ProgramData&);
}
