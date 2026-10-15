#pragma once
#include "src/core/checking/ir.h"
#include "src/shared/diagnostics/result.h"
namespace tepl::core::resolution {
class ResolvedProgram;
}
namespace tepl::core::checking {
class TypedProgram {
 public:
  const ProgramData& ir() const { return *ir_; }

 private:
  explicit TypedProgram(ProgramData ir)
      : ir_(std::make_shared<const ProgramData>(std::move(ir))) {}
  std::shared_ptr<const ProgramData> ir_;
  friend Result<TypedProgram> check(const resolution::ResolvedProgram&);
};
}  // namespace tepl::core::checking
