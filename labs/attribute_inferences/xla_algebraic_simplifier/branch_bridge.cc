#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

#include "labs/attribute_inferences/xla_algebraic_simplifier/bridge.h"
#include "src/executable/program.h"
#include "src/rewrite/builder.h"
#include "src/rewrite/matcher.h"

namespace {
using Json = nlohmann::json;
using namespace tepl;
struct BranchContext {
  executable::ExecutableProgram program;
  std::string output;
  explicit BranchContext(executable::ExecutableProgram p)
      : program(std::move(p)) {}
};
core::AttributeValue ReadAttribute(const Json& value) {
  using A = core::AttributeValue;
  if (value.is_array()) {
    A result{A::Kind::kList, {}, {}};
    for (const auto& element : value)
      result.elements.push_back(ReadAttribute(element));
    return result;
  }
  if (value.is_boolean())
    return {A::Kind::kBool, value.get<bool>() ? "true" : "false", {}};
  if (value.is_string()) return {A::Kind::kEnum, value.get<std::string>(), {}};
  if (value.is_number_integer() || value.is_number_unsigned())
    return {A::Kind::kInteger, value.dump(), {}};
  throw std::runtime_error("unsupported branch attribute value");
}
Json WriteAttribute(const core::AttributeValue& value) {
  using K = core::AttributeValue::Kind;
  if (value.kind == K::kList) {
    Json result = Json::array();
    for (const auto& element : value.elements)
      result.push_back(WriteAttribute(element));
    return result;
  }
  if (value.kind == K::kBool) return value.text == "true";
  if (value.kind == K::kInteger) return Json::parse(value.text);
  return value.text;
}
void Error(char* out, size_t size, const char* text) {
  if (out && size) std::snprintf(out, size, "%s", text);
}
}  // namespace

extern "C" void* tepl_branch_load(const char* path, char* error, size_t size) {
  try {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("missing branch program");
    return new BranchContext(executable::load(
        std::string(std::istreambuf_iterator<char>(stream), {})));
  } catch (const std::exception& e) {
    Error(error, size, e.what());
    return nullptr;
  }
}
extern "C" void tepl_branch_destroy(void* context) {
  delete static_cast<BranchContext*>(context);
}
extern "C" int tepl_branch_apply(void* opaque, const char* request,
                                 const char** output, char* error,
                                 size_t size) {
  try {
    if (output) *output = nullptr;
    if (!opaque || !request || !output)
      throw std::runtime_error("null branch API argument");
    auto& context = *static_cast<BranchContext*>(opaque);
    const auto& program = context.program.ir();
    const auto input = Json::parse(request);
    if (input.at("version") != 1)
      throw std::runtime_error("unsupported branch protocol");
    const std::string name =
        input.contains("rule")
            ? input.at("rule").get<std::string>()
            : "AS_" + std::to_string(input.at("site").get<int>());
    const executable::Rule* selected = nullptr;
    for (const auto& rule : program.rules)
      if (rule.name == name) selected = &rule;
    if (!selected) return 0;
    core::ConcreteGraph graph;
    graph.name = name;
    const bool graph_request = input.contains("nodes");
    const auto& captures =
        graph_request ? input.at("nodes") : input.at("captures");
    if (!graph_request && captures.size() != 4)
      throw std::runtime_error("branch requires four capture slots");
    for (const auto& capture : captures) {
      core::ConcreteNode node{};
      node.kind = core::ConcreteNode::Kind::kInput;
      node.text = "hlo_capture";
      auto type = core::resolveDType(capture.at("dtype").get<std::string>());
      if (!type) throw std::runtime_error("unsupported capture dtype");
      node.dtype = *type;
      node.shape = capture.at("shape").get<std::vector<uint64_t>>();
      if (capture.contains("operation")) {
        const executable::Operation* operation = nullptr;
        for (const auto& op : program.operations)
          if (op.name == capture.at("operation").get<std::string>())
            operation = &op;
        if (!operation) throw std::runtime_error("unknown imported operation");
        node.kind = core::ConcreteNode::Kind::kOperation;
        node.operation = operation->id;
        node.operands = capture.at("operands").get<std::vector<size_t>>();
        for (auto operand : node.operands)
          if (operand >= graph.nodes.size())
            throw std::runtime_error("non-topological imported graph");
        if (operation->attributes) {
          for (const auto& field :
               program.attribute_schemas.at(operation->attributes->value)
                   .fields)
            node.attributes.push_back(
                ReadAttribute(capture.at("attrs").at(field.name)));
        }
      }
      graph.nodes.push_back(std::move(node));
    }
    if (graph_request) {
      graph.root = input.at("root").get<size_t>();
      if (graph.root >= graph.nodes.size())
        throw std::runtime_error("invalid graph root");
    } else {
      const executable::Operation* candidate = nullptr;
      for (const auto& operation : program.operations)
        if (operation.name == "candidate") candidate = &operation;
      if (!candidate || !candidate->attributes)
        throw std::runtime_error("missing candidate operation");
      core::ConcreteNode root{};
      root.kind = core::ConcreteNode::Kind::kOperation;
      root.operation = candidate->id;
      root.operands = {0, 1, 2, 3};
      const auto& schema =
          program.attribute_schemas.at(candidate->attributes->value);
      for (const auto& field : schema.fields)
        root.attributes.push_back(
            ReadAttribute(input.at("attrs").at(field.name)));
      graph.root = graph.nodes.size();
      graph.nodes.push_back(std::move(root));
    }
    rewrite::Budget budget;
    for (size_t i = 0; i < graph.nodes.size(); ++i)
      rewrite::infer(program, graph, i, budget);
    rewrite::Bindings bindings;
    if (!rewrite::match(program, *selected, graph, graph.root, bindings,
                        budget))
      return 0;
    rewrite::Evaluator evaluator(context.program, graph, selected->id.value,
                                 bindings, budget);
    if (!evaluator.evaluateWhere()) return 0;
    evaluator.constructAttributes();
    const auto replacement =
        rewrite::build(program, *selected->rhs, graph, bindings, budget, 10000);
    Json nodes = Json::array();
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
      const auto& node = graph.nodes[i];
      Json encoded{
          {"shape", node.shape ? Json(*node.shape) : Json()},
          {"dtype", node.dtype ? Json(core::dtypeName(*node.dtype)) : Json()}};
      if (i < captures.size())
        encoded["original_id"] = captures.at(i).at("original_id");
      else if (node.operation) {
        const auto& operation = program.operations.at(node.operation->value);
        encoded["operation"] = operation.name;
        encoded["operands"] = node.operands;
        encoded["attrs"] = Json::object();
        if (operation.attributes) {
          const auto& fields =
              program.attribute_schemas.at(operation.attributes->value).fields;
          for (size_t f = 0; f < fields.size(); ++f)
            encoded["attrs"][fields[f].name] =
                WriteAttribute(node.attributes.at(f));
        }
      }
      nodes.push_back(std::move(encoded));
    }
    context.output = Json{
        {"version", 1},
        {"site", input.at("site")},
        {"root", replacement},
        {"nodes", nodes}}.dump();
    *output = context.output.c_str();
    return 1;
  } catch (const std::exception& e) {
    Error(error, size, e.what());
    return -1;
  }
}
