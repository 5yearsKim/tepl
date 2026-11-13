#include "research/tepl_algebraic/branch_builder.h"

#include <dlfcn.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"

namespace xla {
namespace {
template <typename T>
T Take(absl::StatusOr<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).value();
}
template <typename T>
llvm::json::Array JsonArray(const T& values) {
  llvm::json::Array result;
  for (auto value : values) result.push_back(int64_t(value));
  return result;
}
std::string Type(PrimitiveType type) {
  if (type == PRED) return "bool";
  std::string name = primitive_util::LowercasePrimitiveTypeName(type);
  if (name.size() > 1 && name[0] == 's') name[0] = 'i';
  return name;
}
llvm::json::Object Capture(const HloInstruction* instruction) {
  if (!instruction->shape().IsArray() || instruction->shape().is_dynamic())
    throw std::runtime_error("branch capture requires a static array");
  return llvm::json::Object{
      {"original_id", instruction->unique_id()},
      {"dtype", Type(instruction->shape().element_type())},
      {"shape", JsonArray(instruction->shape().dimensions())}};
}
const llvm::json::Object& Object(const llvm::json::Value& value) {
  const auto* object = value.getAsObject();
  if (!object) throw std::runtime_error("expected JSON object");
  return *object;
}
int64_t Integer(const llvm::json::Object& value, const char* key) {
  auto item = value.getInteger(key);
  if (!item) throw std::runtime_error(std::string("missing integer: ") + key);
  return *item;
}
std::vector<int64_t> Integers(const llvm::json::Object& value,
                              const char* key) {
  const auto* array = value.getArray(key);
  if (!array) throw std::runtime_error(std::string("missing array: ") + key);
  std::vector<int64_t> result;
  for (const auto& item : *array) {
    auto number = item.getAsInteger();
    if (!number || *number < 0)
      throw std::runtime_error("invalid nonnegative integer");
    result.push_back(*number);
  }
  return result;
}
}  // namespace

TeplBranchAdapter::TeplBranchAdapter(const char* library, const char* branches,
                                     const char* automatic, bool strict)
    : strict_(strict) {
  library_ = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!library_) throw std::runtime_error(dlerror());
  using Load = void* (*)(const char*, char*, size_t);
  Load load = reinterpret_cast<Load>(dlsym(library_, "tepl_branch_load"));
  apply_ =
      reinterpret_cast<decltype(apply_)>(dlsym(library_, "tepl_branch_apply"));
  destroy_ = reinterpret_cast<decltype(destroy_)>(
      dlsym(library_, "tepl_branch_destroy"));
  if (!load || !apply_ || !destroy_)
    throw std::runtime_error("incompatible branch ABI");
  char error[4096] = {};
  branches_ = load(branches, error, sizeof(error));
  if (!branches_) throw std::runtime_error(error);
  automatic_ = load(automatic, error, sizeof(error));
  if (!automatic_) throw std::runtime_error(error);
}
TeplBranchAdapter::~TeplBranchAdapter() {
  if (branches_) destroy_(branches_);
  if (automatic_) destroy_(automatic_);
  if (library_) dlclose(library_);
}
bool TeplBranchAdapter::IsPorted(int site) {
  switch (site) {
    case 934:
    case 1405:
    case 2311:
    case 2317:
    case 2478:
    case 2824:
    case 5025:
    case 5056:
    case 5389:
    case 5605:
    case 5690:
    case 5723:
    case 5761:
    case 5767:
    case 6744:
    case 6750:
    case 8661:
    case 9835:
    case 9840:
    case 10114:
    case 10118:
    case 2742:
    case 2744:
    case 3785:
    case 6898:
    case 6945:
    case 8870:
    case 9023:
    case 9050:
    case 10505:
    case 5132:
    case 5165:
    case 2614:
    case 2590:
    case 5645:
    case 5681:
    case 6358:
    case 10187:
      return true;
    default:
      return false;
  }
}
void TeplBranchAdapter::Observe(int site) {
  if (site == 2744 && pending_[2742]) {
    ++committed[2744];
    pending_[2742] = false;
    return;
  }
  if (pending_[site]) {
    ++committed[site];
    pending_[site] = false;
  }
}
absl::StatusOr<HloInstruction*> TeplBranchAdapter::Propose(
    HloInstruction* root, int site, absl::Span<HloInstruction* const> captures,
    absl::Span<const int64_t> first, absl::Span<const int64_t> second) {
  pending_[site] = false;
  try {
    llvm::json::Array inputs;
    for (size_t i = 0; i < 4; ++i)
      inputs.push_back(Capture(
          captures.empty() ? root
                           : captures[std::min(i, captures.size() - 1)]));
    llvm::json::Object attrs{{"site", site},
                             {"shape", JsonArray(root->shape().dimensions())},
                             {"type", Type(root->shape().element_type())},
                             {"template_id", root->unique_id()},
                             {"a", llvm::json::Array{}},
                             {"b", llvm::json::Array{}},
                             {"c", llvm::json::Array{}},
                             {"d", llvm::json::Array{}},
                             {"e", llvm::json::Array{}},
                             {"low", llvm::json::Array{}},
                             {"high", llvm::json::Array{}},
                             {"ids", llvm::json::Array{}},
                             {"index", 0}};
    if (site == 5690 || site == 5723)
      attrs["a"] = JsonArray(root->dimensions());
    if (site == 5690) attrs["b"] = JsonArray(root->operand(0)->dimensions());
    if (site == 2824) {
      attrs["a"] = JsonArray(root->operand(0)->dimensions());
      attrs["b"] = JsonArray(root->dimensions());
    }
    if (site == 5389) {
      llvm::json::Array ids;
      for (const auto* operand : root->operand(0)->operands())
        ids.push_back(operand->unique_id());
      attrs["ids"] = std::move(ids);
      attrs["index"] = root->tuple_index();
    }
    if (site == 5645) {
      attrs["a"] = JsonArray(root->dimensions());
      attrs["b"] = JsonArray(first);
    }
    if (site == 2614) attrs["a"] = JsonArray(root->operand(1)->dimensions());
    if (site == 5165) attrs["a"] = JsonArray(captures[3]->dimensions());
    if (site == 2590) {
      attrs["index"] = captures[1] == captures[2] ? 0 : 1;
      if (captures[1] != captures[2])
        attrs["a"] = JsonArray(captures[2]->dimensions());
    }
    if (site == 6358) {
      attrs["a"] = JsonArray(captures[3]->dimensions());
      attrs["b"] = JsonArray(captures[3]->operand(0)->shape().dimensions());
      attrs["index"] = root->operand_count();
    }
    if (site == 10187 || site == 2742 || site == 2744 || site == 3785) {
      const auto* dot = site == 10187 ? root->operand(0) : root;
      const auto& dims = dot->dot_dimension_numbers();
      attrs["template_id"] = dot->unique_id();
      attrs["a"] = JsonArray(dims.lhs_contracting_dimensions());
      attrs["b"] = JsonArray(dims.rhs_contracting_dimensions());
      attrs["c"] = JsonArray(dims.lhs_batch_dimensions());
      attrs["d"] = JsonArray(dims.rhs_batch_dimensions());
    }
    if (site == 6898) {
      attrs["a"] = JsonArray(root->operand(0)->dimensions());
      attrs["b"] = JsonArray(first);
      attrs["c"] = JsonArray(second);
    }
    if (site == 6945) attrs["a"] = JsonArray(first);
    if (site == 8870 || site == 9023 || site == 9050) {
      attrs["a"] = JsonArray(root->dimensions());
      attrs["b"] = JsonArray(first);
      attrs["c"] = JsonArray(second);
      attrs["d"] = JsonArray(root->operand(0)->shape().dimensions());
    }
    if (site == 10505) {
      attrs["a"] = JsonArray(
          root->convolution_dimension_numbers().input_spatial_dimensions());
      llvm::json::Array low, high, base, pad_low, pad_high, interior;
      for (const auto& window : root->window().dimensions()) {
        low.push_back(window.padding_low());
        high.push_back(window.padding_high());
        base.push_back(window.base_dilation());
      }
      for (const auto& padding :
           root->operand(0)->padding_config().dimensions()) {
        pad_low.push_back(padding.edge_padding_low());
        pad_high.push_back(padding.edge_padding_high());
        interior.push_back(padding.interior_padding());
      }
      attrs["b"] = std::move(pad_low);
      attrs["c"] = std::move(pad_high);
      attrs["d"] = std::move(interior);
      attrs["low"] = std::move(low);
      attrs["high"] = std::move(high);
      attrs["e"] = std::move(base);
    }
    llvm::json::Object request_data{{"version", 1},
                                    {"site", site},
                                    {"captures", std::move(inputs)},
                                    {"attrs", std::move(attrs)}};
    if (site == 6358)
      request_data["rule"] = "AS_6358_" + std::to_string(root->operand_count());
    auto request =
        llvm::formatv("{0}", llvm::json::Value(std::move(request_data))).str();
    return Apply(root, site, request, false);
  } catch (const std::exception& e) {
    if (!strict_) return nullptr;
    return absl::InvalidArgumentError("AS-" + std::to_string(site) + ": " +
                                      e.what());
  }
}
absl::StatusOr<HloInstruction*> TeplBranchAdapter::Transpose(
    HloInstruction* root, int site) {
  pending_[site] = false;
  try {
    llvm::json::Array nodes;
    HloInstruction* operand = root->mutable_operand(0);
    HloInstruction* input =
        site == 10118 ? operand->mutable_operand(0) : operand;
    nodes.push_back(Capture(input));
    auto append = [&](HloInstruction* instruction) {
      auto node = Capture(instruction);
      node["operation"] = "transpose";
      node["operands"] = llvm::json::Array{int64_t(nodes.size() - 1)};
      node["attrs"] = llvm::json::Object{
          {"permutation", JsonArray(instruction->dimensions())}};
      nodes.push_back(std::move(node));
    };
    if (site == 10118) append(operand);
    append(root);
    auto request =
        llvm::formatv("{0}", llvm::json::Value(llvm::json::Object{
                                 {"version", 1},
                                 {"site", site},
                                 {"rule", site == 10118 ? "transpose_compose"
                                                        : "transpose_identity"},
                                 {"root", int64_t(nodes.size() - 1)},
                                 {"nodes", std::move(nodes)}}))
            .str();
    auto replacement = Apply(root, site, request, true);
    if (replacement.ok() && replacement.value() &&
        replacement.value()->parent() == nullptr)
      return root->parent()->AddInstruction(
          std::unique_ptr<HloInstruction>(replacement.value()));
    return replacement;
  } catch (const std::exception& e) {
    if (!strict_) return nullptr;
    return absl::InvalidArgumentError(e.what());
  }
}
absl::StatusOr<HloInstruction*> TeplBranchAdapter::Apply(
    HloInstruction* root, int site, const std::string& request,
    bool automatic) {
  const char* output = nullptr;
  char error[4096] = {};
  int result = apply_(automatic ? automatic_ : branches_, request.c_str(),
                      &output, error, sizeof(error));
  if (result < 0)
    return absl::InternalError("AS-" + std::to_string(site) + ": " + error);
  if (!result) {
    if (strict_)
      return absl::FailedPreconditionError(
          "selected branch has no TEPL replacement: AS-" +
          std::to_string(site));
    return nullptr;
  }
  auto parsed = llvm::json::parse(output);
  if (!parsed) return absl::InternalError("invalid branch response JSON");
  const auto& response = Object(*parsed);
  const auto* nodes = response.getArray("nodes");
  if (!nodes || Integer(response, "version") != 1 ||
      Integer(response, "site") != site)
    return absl::InternalError("invalid branch response identity");
  int64_t target = Integer(response, "root");
  // Conditional views have no HLO node when disabled. Resolve their root
  // before allocation so the actual replacement root remains unowned.
  for (size_t steps = 0; steps <= nodes->size(); ++steps) {
    if (target < 0 || target >= nodes->size())
      return absl::InternalError("invalid replacement root");
    const auto& node = Object((*nodes)[target]);
    const auto* attrs = node.getObject("attrs");
    if (!attrs || attrs->getBoolean("enabled") != false) break;
    const auto operands = Integers(node, "operands");
    if (operands.size() != 1 || operands[0] >= target)
      return absl::InternalError("invalid conditional view");
    target = operands[0];
  }
  std::map<int64_t, HloInstruction*> originals;
  for (HloInstruction* instruction : root->parent()->instructions())
    originals[instruction->unique_id()] = instruction;
  std::map<int64_t, HloInstruction*> built;
  std::function<HloInstruction*(int64_t)> build =
      [&](int64_t id) -> HloInstruction* {
    if (id < 0 || id >= nodes->size())
      throw std::runtime_error("invalid replacement node index");
    const auto& node = Object((*nodes)[id]);
    if (auto original = node.getInteger("original_id"))
      return originals.at(*original);
    if (built.count(id)) return built.at(id);
    auto name = node.getString("operation");
    auto dtype = node.getString("dtype");
    if (!name || !dtype)
      throw std::runtime_error("incomplete replacement operation");
    const auto* attrs = node.getObject("attrs");
    if (!attrs) throw std::runtime_error("missing replacement attributes");
    const auto dims = Integers(node, "shape");
    std::string hlo_type = dtype->str();
    if (hlo_type.size() > 1 && hlo_type[0] == 'i') hlo_type[0] = 's';
    PrimitiveType type =
        *dtype == "bool"
            ? PRED
            : Take(primitive_util::StringToPrimitiveType(hlo_type));
    Shape shape =
        id == target ? root->shape() : ShapeUtil::MakeShape(type, dims);
    if (shape.element_type() != type ||
        !absl::c_equal(shape.dimensions(), dims))
      throw std::runtime_error(
          "TEPL replacement shape/type differs from selected root");
    std::vector<HloInstruction*> operands;
    for (int64_t index : Integers(node, "operands")) {
      if (index >= id) throw std::runtime_error("non-topological replacement");
      operands.push_back(build(index));
    }
    if (*name == "reference") return originals.at(Integer(*attrs, "id"));
    if ((*name == "reshape" || *name == "broadcast" || *name == "transpose" ||
         *name == "convert") &&
        attrs->getBoolean("enabled") == false) {
      if (!ShapeUtil::Compatible(shape, operands.at(0)->shape()))
        throw std::runtime_error("disabled view changes shape");
      return operands.at(0);
    }
    std::unique_ptr<HloInstruction> instruction;
    if (*name == "add" || *name == "multiply" || *name == "divide") {
      const HloOpcode op = *name == "add"        ? HloOpcode::kAdd
                           : *name == "multiply" ? HloOpcode::kMultiply
                                                 : HloOpcode::kDivide;
      instruction = HloInstruction::CreateBinary(shape, op, operands.at(0),
                                                 operands.at(1));
    } else if (*name == "negate")
      instruction = HloInstruction::CreateUnary(shape, HloOpcode::kNegate,
                                                operands.at(0));
    else if (*name == "reshape")
      instruction = HloInstruction::CreateReshape(shape, operands.at(0));
    else if (*name == "broadcast")
      instruction = HloInstruction::CreateBroadcast(
          shape, operands.at(0), Integers(*attrs, "dimensions"));
    else if (*name == "transpose")
      instruction = HloInstruction::CreateTranspose(
          shape, operands.at(0), Integers(*attrs, "permutation"));
    else if (*name == "convert")
      instruction = HloInstruction::CreateConvert(shape, operands.at(0));
    else if (*name == "iota")
      instruction =
          HloInstruction::CreateIota(shape, Integer(*attrs, "dimension"));
    else if (*name == "first_scalar") {
      if (!operands.at(0)->IsConstant())
        throw std::runtime_error("first_scalar requires a literal");
      instruction = HloInstruction::CreateConstant(
          LiteralUtil::GetFirstScalarLiteral(operands.at(0)->literal()));
    } else if (*name == "reciprocal_literal") {
      if (!operands.at(0)->IsConstant())
        throw std::runtime_error("reciprocal_literal requires a literal");
      auto scalar = LiteralUtil::CreateR0(type, 1);
      auto ones = Take(scalar.Broadcast(shape, {}));
      HloModule module("literal_primitive", HloModuleConfig{});
      HloComputation::Builder computation("reciprocal");
      auto* constant = computation.AddInstruction(
          HloInstruction::CreateConstant(std::move(ones)));
      auto* input = computation.AddInstruction(
          HloInstruction::CreateConstant(operands.at(0)->literal().Clone()));
      computation.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kDivide, constant, input));
      module.AddEntryComputation(computation.Build());
      HloEvaluator evaluator;
      auto value = Take(evaluator.Evaluate(module, {}));
      instruction = HloInstruction::CreateConstant(std::move(value));
    } else if (*name == "clone1" || *name == "clone2" || *name == "clone3") {
      instruction = originals.at(Integer(*attrs, "template_id"))
                        ->CloneWithNewOperands(shape, operands);
    } else if (*name == "dot") {
      const auto* original = originals.at(Integer(*attrs, "template_id"));
      DotDimensionNumbers dims;
      auto assign = [&](auto* field, const char* name) {
        const auto values = Integers(*attrs, name);
        field->Assign(values.begin(), values.end());
      };
      assign(dims.mutable_lhs_contracting_dimensions(), "lc");
      assign(dims.mutable_rhs_contracting_dimensions(), "rc");
      assign(dims.mutable_lhs_batch_dimensions(), "lb");
      assign(dims.mutable_rhs_batch_dimensions(), "rb");
      auto precision = original->precision_config();
      const auto order = Integers(*attrs, "precision_order");
      if (precision.operand_precision_size()) {
        if (order.size() != precision.operand_precision_size())
          throw std::runtime_error("invalid precision permutation");
        const auto old = precision;
        for (size_t i = 0; i < order.size(); ++i)
          precision.set_operand_precision(i, old.operand_precision(order[i]));
      }
      instruction = HloInstruction::CreateDot(shape, operands.at(0),
                                              operands.at(1), dims, precision);
    } else if (*name == "reduce") {
      auto* original = originals.at(Integer(*attrs, "template_id"));
      const auto dimensions = Integers(*attrs, "dimensions");
      if (attrs->getBoolean("reuse") == true) {
        if (!absl::c_equal(original->dimensions(), dimensions) ||
            !ShapeUtil::Compatible(original->shape(), shape))
          throw std::runtime_error("in-place reduction changed its contract");
        auto status = original->ReplaceOperandWith(0, operands.at(0));
        if (!status.ok()) throw std::runtime_error(status.ToString());
        status = original->ReplaceOperandWith(1, operands.at(1));
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return original;
      }
      instruction =
          HloInstruction::CreateReduce(shape, operands.at(0), operands.at(1),
                                       dimensions, original->to_apply());
    } else if (*name == "convolution") {
      const auto* original = originals.at(Integer(*attrs, "template_id"));
      auto window = original->window();
      const auto* lows = attrs->getArray("low");
      const auto* highs = attrs->getArray("high");
      const auto base = Integers(*attrs, "base");
      if (!lows || !highs || lows->size() != window.dimensions_size() ||
          highs->size() != lows->size() || base.size() != lows->size())
        throw std::runtime_error("invalid convolution window attributes");
      for (int i = 0; i < window.dimensions_size(); ++i) {
        auto low = (*lows)[i].getAsInteger();
        auto high = (*highs)[i].getAsInteger();
        if (!low || !high) throw std::runtime_error("invalid signed padding");
        window.mutable_dimensions(i)->set_padding_low(*low);
        window.mutable_dimensions(i)->set_padding_high(*high);
        window.mutable_dimensions(i)->set_base_dilation(base[i]);
      }
      instruction = original->CloneWithNewOperands(shape, operands);
      instruction->set_window(window);
    } else if (*name == "coordinate_sum") {
      const auto axes = Integers(*attrs, "axes");
      const auto coefficients = Integers(*attrs, "coefficients");
      if (axes.empty() || axes.size() != coefficients.size())
        throw std::runtime_error("invalid coordinate sum");
      for (size_t i = 0; i < axes.size(); ++i) {
        auto iota = HloInstruction::CreateIota(shape, axes[i]);
        if (!i && coefficients[i] == 1)
          instruction = std::move(iota);
        else {
          auto scalar = HloInstruction::CreateConstant(
              LiteralUtil::CreateR0(type, coefficients[i]));
          auto* constant = root->AddInstruction(std::move(scalar));
          if (!dims.empty())
            constant = root->AddInstruction(
                HloInstruction::CreateBroadcast(shape, constant, {}));
          auto term = HloInstruction::CreateBinary(
              shape, HloOpcode::kMultiply,
              root->AddInstruction(std::move(iota)), constant);
          if (!i)
            instruction = std::move(term);
          else
            instruction = HloInstruction::CreateBinary(
                shape, HloOpcode::kAdd,
                root->AddInstruction(std::move(instruction)),
                root->AddInstruction(std::move(term)));
        }
      }
    } else if (*name == "scalar") {
      auto constant = HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(type, Integer(*attrs, "value")));
      if (dims.empty())
        instruction = std::move(constant);
      else
        instruction = HloInstruction::CreateBroadcast(
            shape, root->AddInstruction(std::move(constant)), {});
    } else
      throw std::runtime_error("unregistered HLO constructor: " + name->str());
    root->SetupDerivedInstruction(instruction.get());
    HloInstruction* value = id == target
                                ? instruction.release()
                                : root->AddInstruction(std::move(instruction));
    built[id] = value;
    return value;
  };
  try {
    HloInstruction* replacement = build(target);
    if (replacement == root)
      return absl::FailedPreconditionError("non-progressing TEPL replacement");
    ++proposals[site];
    pending_[site] = true;
    return replacement;
  } catch (const std::exception& e) {
    return absl::InternalError(e.what());
  }
}
}  // namespace xla
