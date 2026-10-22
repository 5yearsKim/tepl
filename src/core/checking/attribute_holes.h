#pragma once
#include "src/shared/catalog/ids.h"
namespace tepl::core {
struct AttributeHole {
  DescriptorId descriptor;
  AttributeSchemaId schema;
};
}  // namespace tepl::core
