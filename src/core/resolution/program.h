#pragma once
#include <map>
#include <memory>
#include <unordered_map>

#include "src/core/resolution/project.h"
#include "src/core/resolution/rules/expanded_rule.h"
#include "src/core/syntax/ast.h"
#include "src/shared/diagnostics/result.h"
namespace tepl::core::resolution {
struct ResolvedOperation {
  OpId id;
  std::string source, dialect;
  syntax::OpDecl declaration;
};
struct ResolvedData {
  std::vector<ResolvedOperation> declarations;
  std::shared_ptr<const core::resolution::SourceProject> sources;
  std::vector<std::string> source_names;
  std::unordered_map<std::size_t, OpId> concrete_operations;
  std::vector<detail::ExpandedRule> rules;
  std::map<std::string, std::unordered_map<std::string, OpId>> operations;
};
class ResolvedProgram {
 public:
  const ResolvedData& data() const { return *data_; }

 private:
  explicit ResolvedProgram(ResolvedData data)
      : data_(std::make_shared<const ResolvedData>(std::move(data))) {}
  std::shared_ptr<const ResolvedData> data_;
  friend Result<ResolvedProgram> resolve(core::resolution::SourceProject);
};
}  // namespace tepl::core::resolution
