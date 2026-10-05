#pragma once

#include <optional>
#include <string>

#include "src/ast/ast.h"
#include "src/core/metadata/ir.h"

namespace tepl::core {
struct Operation;
namespace detail {
struct AnalysisContext;
}
}  // namespace tepl::core
namespace tepl::core::metadata {

std::optional<Program> check(core::detail::AnalysisContext& analysis,
                             const Operation& operation,
                             const ast::MetadataDefinition& definition,
                             const std::string& source,
                             ProgramKind kind = ProgramKind::Shape);

}  // namespace tepl::core::metadata
