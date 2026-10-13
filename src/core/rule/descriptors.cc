#include "src/core/rule/descriptors.h"

#include <algorithm>
#include <utility>

#include "src/core/rule/check_context.h"
#include "src/core/rule/expression.h"
#include "src/core/type/literal.h"

namespace tepl::core::detail {

namespace {

DescriptorId addDescriptor(RuleCheckContext& context, const std::string& name,
                           Descriptor::Kind kind, const SourceOrigin& at) {
  const DescriptorId id{context.rule.descriptors.size()};
  context.rule.descriptors.push_back(
      {id, name, kind,
       context.analysis.types.concrete({TypeKind::kDescriptor}, at), at});
  context.scope.descriptors.emplace(name, id);
  return id;
}

}  // namespace

void registerDerivations(RuleCheckContext& context) {
  // RHS descriptor uses determine schemas before expressions are checked;
  // availability still advances in source order after each assignment.
  for (const auto& derivation : context.input.derivations) {
    if (context.scope.descriptors.contains(derivation.target)) {
      context.analysis.report(
          derivation.origin,
          "duplicate descriptor definition '@" + derivation.target + "'");
      continue;
    }
    addDescriptor(context, derivation.target, Descriptor::Kind::kDerived,
                  derivation.origin);
  }
}

std::optional<DescriptorId> checkDescriptor(RuleCheckContext& context,
                                            const ast::Operator& op,
                                            OpId operation, bool lhs,
                                            const SourceOrigin& at) {
  const auto schema =
      context.analysis.output.operations[operation.value].attributes;
  if (op.automatic) {
    if (lhs) {
      context.analysis.report(at,
                              "auto attributes are only allowed on the RHS");
      return std::nullopt;
    }
    if (!schema) {
      context.analysis.report(at,
                              "auto requires an operation attribute schema");
      return std::nullopt;
    }
    const auto id = addDescriptor(
        context, "auto#" + std::to_string(context.rule.descriptors.size()),
        Descriptor::Kind::kInferred, at);
    context.analysis.types.unify(
        context.rule.descriptors[id.value].type,
        context.analysis.types.concrete({TypeKind::kDescriptor, schema}, at),
        at);
    const auto& fields =
        context.analysis.output.attribute_schemas[schema->value].fields;
    AttributeHole hole{id, *schema,
                       std::vector<AttributePinPtr>(fields.size())};
    auto literal = [&](auto&& self, const ast::AutoField::Literal& input,
                       const AttributeField& field, std::size_t depth,
                       bool optional) -> AttributePinPtr {
      using L = ast::AutoField::Literal;
      AttributePin value{AttributePin::Kind::kLiteral, input.text};
      value.string_literal = input.kind == L::Kind::kString;
      bool valid = false;
      if (optional && input.kind == L::Kind::kName && input.text == "none")
        valid = true;
      else if (depth && input.kind == L::Kind::kList) {
        valid = true;
        value.text = "list";
        for (const auto& element : input.elements) {
          auto checked = self(self, element, field, depth - 1, false);
          if (!checked) valid = false;
          value.arguments.push_back(checked);
        }
      } else if (!depth) {
        if ((field.type == "index" || field.type == "i64") &&
            input.kind == L::Kind::kInteger) {
          auto type = field.type == "index" ? TypeKind::kIndex : TypeKind::kI64;
          valid = validHostLiteral(input.text, type);
          if (valid && !value.text.empty() && value.text[0] == '+')
            value.text.erase(0, 1);
        } else if (field.type == "bool")
          valid = input.kind == L::Kind::kBool;
        else if (field.type == "string")
          valid = input.kind == L::Kind::kString;
        else if (field.type == "dtype")
          valid = input.kind == L::Kind::kName &&
                  resolveDType(input.text).has_value();
        else if (field.type == "precision")
          valid = input.kind == L::Kind::kName &&
                  (input.text == "default" || input.text == "high" ||
                   input.text == "highest");
      }
      if (!valid) {
        context.analysis.report(
            at, "invalid auto override for field '" + field.name + "'");
        return {};
      }
      return std::make_shared<AttributePin>(std::move(value));
    };
    for (const auto& override : op.automatic->fields) {
      auto field =
          std::find_if(fields.begin(), fields.end(),
                       [&](const auto& f) { return f.name == override.name; });
      if (field == fields.end()) {
        context.analysis.report(
            at, "unknown auto override field '" + override.name + "'");
        continue;
      }
      auto index = static_cast<std::size_t>(field - fields.begin());
      if (hole.overrides[index]) {
        context.analysis.report(
            at, "duplicate auto override field '" + override.name + "'");
        continue;
      }
      if (!override.descriptor)
        hole.overrides[index] = literal(literal, override.value, *field,
                                        field->list_depth, field->optional);
      else {
        auto source = context.scope.descriptors.find(override.descriptor->name);
        if (source == context.scope.descriptors.end()) {
          context.analysis.report(at, "unknown override descriptor '@" +
                                          override.descriptor->name + "'");
          continue;
        }
        auto source_type = context.analysis.types.known(
            context.rule.descriptors[source->second.value].type);
        if (!source_type || !source_type->schema) {
          context.analysis.report(
              at, "override descriptor must have a known schema");
          continue;
        }
        const auto& source_fields =
            context.analysis.output
                .attribute_schemas[source_type->schema->value]
                .fields;
        auto input = std::find_if(
            source_fields.begin(), source_fields.end(),
            [&](const auto& f) { return f.name == override.field; });
        if (input == source_fields.end() || input->type != field->type ||
            input->list_depth != field->list_depth ||
            input->optional != field->optional) {
          context.analysis.report(
              at, "incompatible or unknown auto override field '@" +
                      override.descriptor->name + "." + override.field + "'");
          continue;
        }
        AttributePin value{AttributePin::Kind::kField};
        value.descriptor = source->second;
        value.schema = *source_type->schema;
        value.field = input - source_fields.begin();
        hole.overrides[index] =
            std::make_shared<AttributePin>(std::move(value));
      }
    }
    context.rule.attribute_holes.push_back(std::move(hole));
    return id;
  }
  if (!op.attribute) {
    if (schema)
      context.analysis.report(
          at, "operation '" + op.name + "' requires an attribute descriptor");
    return std::nullopt;
  }
  if (!schema) {
    context.analysis.report(at,
                            "operation '" + op.name + "' has no attributes");
    return std::nullopt;
  }
  const auto& name = op.attribute->name;
  auto found = context.scope.descriptors.find(name);
  if (found == context.scope.descriptors.end()) {
    if (!lhs) {
      context.analysis.report(at, "unknown RHS descriptor '@" + name + "'");
      return std::nullopt;
    }
    const auto id =
        addDescriptor(context, name, Descriptor::Kind::kCaptured, at);
    context.scope.available_descriptors.insert(id.value);
    found = context.scope.descriptors.find(name);
  }
  const auto id = found->second;
  context.analysis.types.unify(
      context.rule.descriptors[id.value].type,
      context.analysis.types.concrete({TypeKind::kDescriptor, schema}, at), at);
  return id;
}

void checkDerivations(RuleCheckContext& context,
                      ExpressionChecker& expressions) {
  for (const auto& derivation : context.input.derivations) {
    const auto id = context.scope.descriptors.at(derivation.target);
    const auto& descriptor = context.rule.descriptors.at(id.value);
    if (descriptor.kind != Descriptor::Kind::kDerived ||
        context.scope.available_descriptors.contains(id.value))
      continue;
    if (auto checked = expressions.check(*derivation.value, descriptor.type))
      context.rule.derivations.push_back(
          {id, std::move(checked), derivation.origin});
    context.scope.available_descriptors.insert(id.value);
  }
}

}  // namespace tepl::core::detail
