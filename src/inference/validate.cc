#include <algorithm>
#include <iterator>
#include <stdexcept>

#include "src/inference/evaluate.h"
#include "src/inference/extensions.h"

namespace tepl::inference {
namespace {
void require(bool ok, const char* reason) {
  if (!ok) throw std::invalid_argument(reason);
}
void schedule(const std::vector<Instruction>& code) {
  require(!code.empty(), "empty inference schedule");
  std::set<std::size_t> nodes;
  std::vector<std::size_t> uses(code.size());
  for (std::size_t i = 0; i < code.size(); ++i) {
    const auto& n = code[i];
    require(nodes.insert(n.node).second, "duplicate inference node");
    std::set<std::size_t> support;
    for (auto child : n.inputs) {
      require(child < i, "inference schedule is not producer-first");
      require(++uses[child] == 1, "inference block must be a tree");
      support.insert(code[child].captures.begin(), code[child].captures.end());
    }
    if (n.kind == Kind::Input) {
      require(n.capture && !n.descriptor && !n.operation && n.inputs.empty() &&
                  n.fields.empty(),
              "invalid inference input instruction");
      support.insert(*n.capture);
    } else if (n.kind == Kind::Negation) {
      require(!n.capture && !n.descriptor && n.inputs.size() == 1 &&
                  n.fields.empty(),
              "negation must be attribute-free and unary");
    } else if (n.kind == Kind::Addition || n.kind == Kind::Multiplication) {
      require(!n.capture && !n.descriptor && n.inputs.size() == 2 &&
                  n.fields.empty(),
              "addition must be attribute-free and binary");
    } else {
      require(n.kind == Kind::Permutation || n.kind == Kind::Contraction ||
                  n.kind == Kind::Summation || n.kind == Kind::Reverse ||
                  n.kind == Kind::Broadcast || n.kind == Kind::UnitRemoval,
              "unknown inference instruction");
      auto arity = n.kind == Kind::Contraction ? 2u : 1u;
      auto count = n.kind == Kind::Contraction ? 4u
                   : n.kind == Kind::Broadcast ? 2u
                                               : 1u;
      require(!n.capture && n.descriptor && n.inputs.size() == arity &&
                  n.fields.size() == count,
              "invalid inference operation instruction");
      auto fields = n.fields;
      std::sort(fields.begin(), fields.end());
      for (std::size_t f = 0; f < fields.size(); ++f)
        require(fields[f] == f, "inference fields must form a bijection");
    }
    require(
        n.captures == std::vector<std::size_t>(support.begin(), support.end()),
        "incorrect compiled subtree capture set");
  }
  for (std::size_t i = 0; i + 1 < code.size(); ++i)
    require(uses[i] == 1, "unreachable inference instruction");
}
using Code = std::vector<Instruction>;
using Alternatives = std::vector<Code>;
// Source distribution copies captured fields, never replacement holes.
Alternatives expand(const Code& code, std::size_t root, bool source,
                    bool allow_sum = true) {
  const auto& n = code.at(root);
  if (n.kind == Kind::Addition) {
    require(allow_sum, "RHS must be an explicit sum of eligible components");
    auto left = expand(code, n.inputs[0], source);
    auto right = expand(code, n.inputs[1], source);
    left.insert(left.end(), std::make_move_iterator(right.begin()),
                std::make_move_iterator(right.end()));
    return left;
  }
  if (n.kind == Kind::Input) return {Code{n}};
  auto left = expand(code, n.inputs[0], source, source);
  auto right = n.inputs.size() == 2 ? expand(code, n.inputs[1], source, source)
                                    : Alternatives{Code{}};
  Alternatives result;
  for (const auto& l : left)
    for (const auto& r : right) {
      auto combined = l;
      auto instruction = n;
      instruction.inputs = {l.size() - 1};
      std::set<std::size_t> captures(l.back().captures.begin(),
                                     l.back().captures.end());
      if (!r.empty()) {
        instruction.inputs.push_back(l.size() + r.size() - 1);
        captures.insert(r.back().captures.begin(), r.back().captures.end());
        for (auto i : r) {
          for (auto& input : i.inputs) input += l.size();
          combined.push_back(std::move(i));
        }
      }
      instruction.captures.assign(captures.begin(), captures.end());
      combined.push_back(std::move(instruction));
      result.push_back(std::move(combined));
    }
  return result;
}
std::map<std::vector<std::size_t>, Code> components(Alternatives alternatives) {
  std::map<std::vector<std::size_t>, Code> result;
  for (auto& code : alternatives) {
    std::set<std::size_t> seen;
    for (const auto& n : code)
      if (n.capture)
        require(seen.insert(*n.capture).second,
                "ordered components require each capture exactly once");
    auto captures = code.back().captures;
    require(!captures.empty(),
            "ordered components require a nonempty input set");
    require(result.emplace(captures, std::move(code)).second,
            "additive components must have distinct input sets");
  }
  return result;
}
}  // namespace

std::vector<Component> decompose(const Program& p) {
  schedule(p.source);
  schedule(p.replacement);
  if (hasExtensions(p)) {
    require(p.source.back().captures == p.replacement.back().captures,
            "schema captures must match");
    for (const auto* code : {&p.source, &p.replacement}) {
      std::set<std::size_t> captures;
      for (const auto& n : *code)
        if (n.capture)
          require(captures.insert(*n.capture).second,
                  "schemas require read-once captures");
    }
    return {};
  }
  auto source = components(expand(p.source, p.source.size() - 1, true));
  auto replacement =
      components(expand(p.replacement, p.replacement.size() - 1, false));
  require(source.size() == replacement.size(),
          "component input sets must match");
  std::vector<Component> result;
  for (auto& [captures, left] : source) {
    auto right = replacement.find(captures);
    require(right != replacement.end(), "component input sets must match");
    result.push_back({std::move(left), std::move(right->second)});
  }
  auto has_addition = [](const Code& code) {
    return std::any_of(code.begin(), code.end(),
                       [](const auto& i) { return i.kind == Kind::Addition; });
  };
  if (!has_addition(p.source) && !has_addition(p.replacement)) return {};
  return result;
}

void validate(const Program& p) {
  require(p.mode == Mode::Synthesize || p.mode == Mode::Validate,
          "unknown inference mode");
  auto expected = decompose(p);
  require(p.components == expected,
          "incorrect compiled component decomposition");
  std::set<std::size_t> source_descriptors, replacement_descriptors;
  for (const auto& n : p.source)
    if (n.descriptor) source_descriptors.insert(*n.descriptor);
  if (p.mode == Mode::Synthesize)
    for (const auto& n : p.replacement)
      if (n.descriptor) {
        require(replacement_descriptors.insert(*n.descriptor).second,
                "synthesized descriptors must be independent");
        require(!source_descriptors.contains(*n.descriptor),
                "synthesized descriptor shares source metadata");
      }
}
}  // namespace tepl::inference
