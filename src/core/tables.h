#pragma once
#include "src/core/semantics/definition.h"
#include "src/shared/catalog/data.h"
namespace tepl::core {
struct Operation : OperationSignature {
  std::shared_ptr<const semantics::Definition> semantic_definition;
};
struct ProgramTables {
  std::vector<Dialect> dialects;
  std::vector<std::string> sources;
  std::vector<Type> types;
  std::vector<Operation> operations;
  std::vector<AttributeSchema> attribute_schemas;
  std::vector<ConcreteGraph> graphs;
};

}  // namespace tepl::core
