#pragma once

#include <optional>
#include <string>

#include "src/ast/ast.h"
#include "src/core/shape/ir.h"

namespace tepl::core {
struct Operation;
namespace detail {
struct AnalysisContext;
}
}  // namespace tepl::core
namespace tepl::core::shape {

std::optional<Program> check(core::detail::AnalysisContext& analysis,
                             const Operation& operation,
                             const ast::ShapeDefinition& definition,
                             const std::string& source);

}  // namespace tepl::core::shape
