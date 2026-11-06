#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/shared/builtins/axes.h"
#include "src/shared/builtins/intrinsics/catalog.h"

namespace {
using namespace tepl::core::builtins;
constexpr std::string_view kBegin = "<!-- BEGIN BUILTIN CATALOG -->";
constexpr std::string_view kEnd = "<!-- END BUILTIN CATALOG -->";
std::string typeName(SignatureType type, bool rule, bool pair) {
  using T = SignatureType;
  switch (type) {
    case T::kDType:
      return rule ? "dtype" : "DType";
    case T::kAnyInteger:
      return rule ? "I" : "Integer";
    case T::kSignedInteger:
      return rule ? "i64" : "Integer";
    case T::kInteger:
      return rule ? (pair ? "I" : "u64") : "Integer";
    case T::kBoolean:
      return rule ? "bool" : "Bool";
    case T::kIntegerList:
      return rule ? "u64[]" : "Integer[]";
    case T::kBooleanList:
      return rule ? "bool[]" : "Bool[]";
    case T::kAxisSet:
      return "LogicalAxisSet";
    case T::kAxisList:
      return "LogicalAxis[]";
    case T::kT:
      return "T";
    case T::kListT:
      return "T[]";
  }
  throw std::logic_error("invalid signature type");
}
std::string signatureText(const BuiltinSignature& entry, bool rule) {
  std::string text = "(";
  for (std::size_t i = 0; i < entry.arguments.size(); ++i) {
    if (i) text += ", ";
    text += typeName(entry.arguments[i], rule, entry.rule_integer_pair);
  }
  if (entry.variadic) text += ", ...";
  return text + ") -> " + typeName(entry.result, rule, entry.rule_integer_pair);
}
std::string_view axisTypeName(AxisType type) {
  switch (type) {
    case AxisType::Integer:
      return "Integer";
    case AxisType::Boolean:
      return "Bool";
    case AxisType::Positions:
      return "Positions";
    case AxisType::Axes:
      return "Axes";
    case AxisType::Operand:
      return "Operand";
    case AxisType::Sequence:
      return "S";
  }
  throw std::logic_error("invalid axis type");
}
std::string publicReference() {
  std::ostringstream out;
  out << "### Expression function inventory\n\n"
         "This table is generated from the compiler catalog. Feature groups do "
         "not grant\n"
         "section access. The axes overloads are listed separately below.\n\n"
         "| Function | Group | Metadata signature | Rule signature | Allowed "
         "sections |\n"
         "| --- | --- | --- | --- | --- |\n";
  for (auto feature :
       {Feature::kList, Feature::kInteger, Feature::kShape, Feature::kDType})
    for (const auto& entry : catalog()) {
      if (entry.feature != feature || !isPublicBuiltin(entry.builtin)) continue;
      out << "| `" << entry.name << "` | " << featureName(feature) << " | `"
          << signatureText(entry, false) << "` | `"
          << signatureText(entry, true) << "` | ";
      bool first = true;
      for (auto context : {Context::kShape, Context::kDType, Context::kWhere,
                           Context::kDerive})
        if (entry.availableIn(context)) {
          if (!first) out << ", ";
          out << '`' << contextName(context) << '`';
          first = false;
        }
      out << " |\n";
    }
  out << "\n### Axes function inventory\n\n"
         "These functions are callable in operation `axes` blocks. `Positions` "
         "is an\n"
         "integer sequence, `Axes` is a sequence of axis identities, and "
         "`Operand` is an\n"
         "operation operand. `S` is one shared sequence type: `Positions` or "
         "`Axes`.\n"
         "An operand supplies its shape through `shape`, `rank`, or "
         "`all_axes`.\n\n"
         "| Function | Axes signature |\n| --- | --- |\n";
  for (const auto& entry : axisCatalog()) {
    out << "| `" << entry.name << "` | `(";
    for (std::size_t i = 0; i < entry.arguments.size(); ++i) {
      if (i) out << ", ";
      out << axisTypeName(entry.arguments[i]);
    }
    if (entry.variadic) out << ", ...";
    out << ") -> " << axisTypeName(entry.result) << "` |\n";
  }
  return out.str();
}
std::string internalReference() {
  std::ostringstream out;
  out << "## Internal operation inventory\n\n"
         "Generated from `catalog.def`. These spellings identify symbolic IR "
         "operations;\n"
         "they do not expose source functions. Source axes functions may "
         "explicitly lower\n"
         "to these operations through the axes catalog.\n\n"
         "| Operation | Argument count |\n| --- | --- |\n";
  for (const auto& entry : tepl::core::intrinsics::catalog()) {
    out << "| `" << entry.name << "` | " << entry.min_arity;
    if (entry.max_arity < 0)
      out << " or more";
    else if (entry.min_arity != static_cast<std::size_t>(entry.max_arity))
      out << "–" << entry.max_arity;
    out << " |\n";
  }
  return out.str();
}
}  // namespace
int main(int argc, char** argv) {
  try {
    bool internal = argc > 1 && std::string_view(argv[1]) == "--internal";
    const int offset = internal ? 1 : 0;
    if (argc != 3 + offset ||
        (std::string_view(argv[1 + offset]) != "--check" &&
         std::string_view(argv[1 + offset]) != "--write"))
      throw std::runtime_error(
          "usage: builtin_reference [--internal] --check|--write FILE");
    const auto path = argv[2 + offset];
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read reference");
    const std::string current{std::istreambuf_iterator<char>(input), {}};
    const auto begin = current.find(kBegin), end = current.find(kEnd);
    if (begin == std::string::npos || end == std::string::npos || end < begin)
      throw std::runtime_error("missing catalog markers");
    const auto generated =
        current.substr(0, begin + kBegin.size()) + "\n\n" +
        (internal ? internalReference() : publicReference()) + "\n" +
        current.substr(end);
    if (std::string_view(argv[1 + offset]) == "--check") {
      if (generated != current)
        throw std::runtime_error(
            "builtin reference is stale; run builtin_reference --write for "
            "this file");
    } else {
      std::ofstream output(path);
      if (!output || !(output << generated))
        throw std::runtime_error("cannot write reference");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
