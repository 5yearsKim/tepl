#pragma once
#include "src/core/semantics/options.h"
#include "src/core/semantics/verified_ir.h"
#include "src/shared/diagnostics/result.h"
namespace tepl::core::checking {
class TypedProgram;
}
namespace tepl::core {
namespace semantics {
class VerifiedProgram {
 public:
  const VerifiedData& ir() const { return *ir_; }
  SemanticPolicy policy() const { return policy_; }
  VerificationKind verification(RuleId rule) const {
    return kinds_.at(rule.value);
  }
  const auto& axes(std::size_t rule) const { return axes_.at(rule); }

 private:
  VerifiedProgram(VerifiedData ir, SemanticPolicy policy,
                  std::vector<std::vector<completed::SymbolPtr>> axes,
                  std::vector<VerificationKind> kinds)
      : ir_(std::make_shared<const VerifiedData>(std::move(ir))),
        policy_(policy),
        axes_(std::move(axes)),
        kinds_(std::move(kinds)) {}
  std::shared_ptr<const VerifiedData> ir_;
  SemanticPolicy policy_;
  std::vector<std::vector<completed::SymbolPtr>> axes_;
  std::vector<VerificationKind> kinds_;
  friend Result<VerifiedProgram> verify(const checking::TypedProgram&,
                                        const SemanticOptions&);
};
}  // namespace semantics
}  // namespace tepl::core
