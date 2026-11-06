#pragma once
#include <memory>
#include <string>

#include "src/core/semantics/definition.h"
#include "src/core/syntax/ast.h"

namespace tepl::core {
struct Operation;
struct Rule;
}  // namespace tepl::core
namespace tepl::core::detail {
struct CheckingContext;
}
namespace tepl::core::semantics {
std::shared_ptr<const Definition> check(detail::CheckingContext&,
                                        const Operation&,
                                        const syntax::AxisDefinition&,
                                        const syntax::SemanticDefinition&,
                                        const std::string&);
}  // namespace tepl::core::semantics
