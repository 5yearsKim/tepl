#pragma once

#include <optional>
#include <string>

#include "src/core/syntax/ast.h"
#include "src/shared/metadata/ir.h"

namespace tepl::core {
struct Operation;
namespace detail {
struct CheckingContext;
}
}  // namespace tepl::core
namespace tepl::core::metadata {

std::optional<Program> check(core::detail::CheckingContext& analysis,
                             const Operation& operation,
                             const syntax::MetadataDefinition& definition,
                             const std::string& source,
                             ProgramKind kind = ProgramKind::Shape);

}  // namespace tepl::core::metadata
