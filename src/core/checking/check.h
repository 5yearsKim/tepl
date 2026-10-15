#pragma once
#include "src/core/checking/program.h"
namespace tepl::core::checking {
Result<TypedProgram> check(const resolution::ResolvedProgram&);
}
