#pragma once
#include "src/codegen/cpp/names.h"
namespace tepl::codegen::cpp {
std::string emitMetadata(const executable::ProgramData&, const Names&);
}
