#pragma once
#include <string>

#include "src/executable/ir.h"
namespace tepl::codegen::cpp {
class Expressions {
 public:
  explicit Expressions(const executable::ProgramData& program)
      : program_(program) {}
  std::string rule(const core::completed::TypedExpr&) const;
  std::string symbolic(const core::completed::Symbol&) const;
  std::string metadata(const core::metadata::Expr&) const;
  std::string axis(const core::axis::Meta&) const;

 private:
  const executable::ProgramData& program_;
};
std::string value(const core::AttributeValue&);
std::string scalarType(const core::Type&);
}  // namespace tepl::codegen::cpp
