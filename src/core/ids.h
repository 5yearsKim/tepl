#pragma once

#include <cstddef>

namespace tepl::core {

template <typename Tag>
struct Id {
  std::size_t value = 0;
  bool operator==(const Id&) const = default;
};

using RuleId = Id<struct RuleTag>;
using OpId = Id<struct OpTag>;
using CaptureId = Id<struct CaptureTag>;
using DimensionId = Id<struct DimensionTag>;
using DescriptorId = Id<struct DescriptorTag>;
using AttributeSchemaId = Id<struct AttributeSchemaTag>;
using HostFunctionId = Id<struct HostFunctionTag>;
using TypeId = Id<struct TypeTag>;

}  // namespace tepl::core
