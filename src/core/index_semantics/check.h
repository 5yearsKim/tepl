#pragma once
#include <memory>
#include <string>

#include "src/ast/ast.h"
#include "src/core/index_semantics/ir.h"

namespace tepl::core {
struct Operation;
struct Rule;
}  // namespace tepl::core
namespace tepl::core::detail {
struct AnalysisContext;
}
namespace tepl::core::index_semantics {
std::shared_ptr<const Definition> check(detail::AnalysisContext&,
                                        const Operation&,
                                        const ast::AxisDefinition&,
                                        const ast::SemanticDefinition&,
                                        const std::string&);
// Returns true when this rule uses the new semantics (including mixed rules,
// for which a diagnostic is emitted), so legacy inference is not invoked.
bool compile(detail::AnalysisContext&, Rule&);
}  // namespace tepl::core::index_semantics
