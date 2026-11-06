#pragma once
#include "src/shared/catalog/completed.h"
#include "src/shared/catalog/data.h"
namespace tepl::executable {
struct Operation : core::OperationSignature {
  std::shared_ptr<const core::axis::ShapeContract> axis_contract;
};
struct Rule : core::completed::RuleData {};
struct ProgramData {
  std::vector<core::Dialect> dialects;
  std::vector<std::string> sources;
  std::vector<core::Type> types;
  std::vector<Operation> operations;
  std::vector<core::AttributeSchema> attribute_schemas;
  std::vector<Rule> rules;
  std::vector<core::ConcreteGraph> graphs;
};
}  // namespace tepl::executable
