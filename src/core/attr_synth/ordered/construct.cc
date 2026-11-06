#include "src/core/attr_synth/ordered/construct.h"

#include <stdexcept>

namespace tepl::core::attr_synth::ordered {
using namespace semantics;
Construction construct(Context& c, const inference::EmissionBlock& program,
                       const Correspondence& match, const S& source_output,
                       const S& reduction_order) {
  using Op = inference::Op;
  Construction result;
  std::map<std::size_t, S> inputs;
  for (const auto& node : program.source.replacement)
    if (node.kind == inference::Kind::Input)
      inputs.emplace(*node.capture, match.anchors.at(node.node));
  std::set<std::size_t> contractions;
  for (const auto& node : program.source.replacement)
    if (node.kind == inference::Kind::Contraction) {
      auto positions =
          program.attributes.at(*node.descriptor).at(node.fields.at(2));
      contractions.insert(program.expressions.at(positions).args.at(1));
    }

  // Interpret the already constructed IR over symbolic values. The ordinary
  // expression exporter may simplify these values through axis mappings and
  // partitions. It preserves the feasibility test and canonical memberships;
  // paired finite-sum reindexing below is checked by the independent verifier.
  // Each source input retains its declared physical axis order.
  std::vector<S> values;
  for (const auto& expression : program.expressions) {
    c.tick();
    std::vector<S> args;
    for (auto reference : expression.args) args.push_back(values.at(reference));
    S value;
    switch (expression.op) {
      case Op::Input:
        value = inputs.at(expression.capture);
        break;
      case Op::Output:
        value = source_output;
        break;
      case Op::Empty:
        value = call("empty_axes", {});
        break;
      // These IR nodes denote membership sets. Their enumeration order is
      // unobservable: ordered consumers always filter an explicit traversal.
      case Op::Union:
        value = c.support(call("union_axes", std::move(args)));
        break;
      case Op::Intersect:
        value = c.intersect(c.support(args.at(0)), c.support(args.at(1)));
        break;
      case Op::Difference:
        value = c.difference(c.support(args.at(0)), c.support(args.at(1)));
        break;
      case Op::Filter:
        value = c.filter(args.at(0), args.at(1));
        if (contractions.contains(values.size())) {
          // Optional finite-sum reindexing: permute BOTH paired contraction
          // lists together into source reduction order. This does not change
          // membership, feasibility or output layout. The independent
          // verifier still checks the resulting complete candidate.
          auto preferred = c.filter(reduction_order, args.at(1));
          value =
              call("concat",
                   {preferred,
                    c.filter(args.at(0), c.difference(args.at(1), preferred))});
        }
        break;
      case Op::Concat:
        value = call("concat", std::move(args));
        break;
      case Op::Positions:
        value = c.positions(args.at(0), args.at(1));
        break;
      case Op::Equal:
        value = call("==", std::move(args));
        break;
      case Op::SameSet:
        value = call("same_set", std::move(args));
        break;
      case Op::Distinct:
        value = call("distinct_axes", std::move(args));
        break;
      case Op::Subset:
        value = call("subset", std::move(args));
        break;
      default:
        throw std::invalid_argument(
            "noncanonical operation in symbolic IR adapter");
    }
    values.push_back(c.canonical(value));
  }
  for (auto condition : program.checks)
    result.candidate.feasibility.push_back(values.at(condition));
  for (const auto& [descriptor, attributes] : program.attributes) {
    auto& fields = result.candidate.constructors[descriptor];
    for (auto expression : attributes) fields.push_back(values.at(expression));
  }
  for (const auto& [node, expression] : program.layouts)
    result.layouts.emplace(node, values.at(expression));
  return result;
}
}  // namespace tepl::core::attr_synth::ordered
