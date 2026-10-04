#pragma once
#include "src/codegen/cpp/names.h"
namespace tepl::codegen::cpp {
std::string emitShapeInference(const core::Program&, const Names&);
std::string emitDTypeInference(const core::Program&, const Names&);
}  // namespace tepl::codegen::cpp
