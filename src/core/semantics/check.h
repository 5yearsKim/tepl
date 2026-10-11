#pragma once
#include <optional>
#include <string>

#include "src/ast/ast.h"
#include "src/core/metadata/ir.h"
#include "src/core/semantics/ir.h"
namespace tepl::core {
struct Operation;
}
namespace tepl::core::detail {
struct AnalysisContext;
}
namespace tepl::core::semantics {
std::optional<Program> check(detail::AnalysisContext&, const Operation&,
                             const ast::SemanticDefinition&,
                             const std::string&);
}  // namespace tepl::core::semantics
