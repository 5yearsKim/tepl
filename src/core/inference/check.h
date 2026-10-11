#pragma once
namespace tepl::core {
struct Rule;
}
namespace tepl::core::detail {
struct AnalysisContext;
struct RuleCheckContext;
}  // namespace tepl::core::detail
namespace tepl::core::inference {
void infer(detail::AnalysisContext& context, Rule& rule);
}
