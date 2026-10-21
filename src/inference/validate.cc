#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "src/inference/evaluate.h"

namespace tepl::inference {
void validate(const Program& p) {
  auto require = [](bool ok, const char* reason) {
    if (!ok) throw std::invalid_argument(reason);
  };
  require(p.mode == Mode::Synthesize || p.mode == Mode::Validate,
          "unknown inference mode");
  std::set<std::size_t> source_descriptors, replacement_descriptors;
  auto block = [&](const std::vector<Instruction>& code, bool source) {
    require(!code.empty(), "empty inference schedule");
    std::set<std::size_t> nodes, captures;
    std::vector<std::size_t> uses(code.size());
    for (std::size_t i = 0; i < code.size(); ++i) {
      const auto& n = code[i];
      require(nodes.insert(n.node).second, "duplicate inference node");
      std::set<std::size_t> support;
      for (auto child : n.inputs) {
        require(child < i, "inference schedule is not producer-first");
        require(++uses[child] == 1, "inference block must be a tree");
        support.insert(code[child].captures.begin(),
                       code[child].captures.end());
      }
      if (n.kind == Kind::Input) {
        require(n.capture && !n.descriptor && !n.operation &&
                    n.inputs.empty() && n.fields.empty(),
                "invalid inference input instruction");
        require(captures.insert(*n.capture).second,
                "repeated inference capture");
        support.insert(*n.capture);
      } else {
        require(n.kind == Kind::Permutation || n.kind == Kind::Contraction,
                "unknown inference instruction");
        auto arity = n.kind == Kind::Permutation ? 1u : 2u;
        auto count = n.kind == Kind::Permutation ? 1u : 4u;
        require(!n.capture && n.descriptor && n.inputs.size() == arity &&
                    n.fields.size() == count,
                "invalid inference operation instruction");
        auto fields = n.fields;
        std::sort(fields.begin(), fields.end());
        for (std::size_t f = 0; f < fields.size(); ++f)
          require(fields[f] == f, "inference fields must form a bijection");
        if (source)
          source_descriptors.insert(*n.descriptor);
        else if (p.mode == Mode::Synthesize)
          require(replacement_descriptors.insert(*n.descriptor).second,
                  "synthesized descriptors must be independent");
      }
      require(n.captures ==
                  std::vector<std::size_t>(support.begin(), support.end()),
              "incorrect compiled subtree capture set");
    }
    for (std::size_t i = 0; i + 1 < code.size(); ++i)
      require(uses[i] == 1, "unreachable inference instruction");
    return captures;
  };
  require(block(p.source, true) == block(p.replacement, false),
          "inference blocks must use the same captures");
  for (auto d : replacement_descriptors)
    require(!source_descriptors.contains(d),
            "synthesized descriptor shares source metadata");
}
}  // namespace tepl::inference
