#pragma once
// Compatibility names; builtin identities and signatures live in one catalog.
#include "src/core/builtins/catalog.h"
namespace tepl::core::shape {
using builtins::Builtin;
using builtins::builtinName;
using builtins::BuiltinSignature;
using builtins::resolveBuiltin;
using builtins::SignatureType;
}  // namespace tepl::core::shape
