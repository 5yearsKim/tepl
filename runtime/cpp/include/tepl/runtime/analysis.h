#pragma once
#include <eggc/egraph.hpp>
#include <memory>
#include <mutex>

#include "node.h"
namespace tepl_runtime {
struct RewriteRejection {
  std::string source, rule;
  EvalError error;
};
struct AnalysisState {
  std::map<std::string, TensorFacts> inputs;
  std::set<std::string> inserted_inputs;
  mutable std::mutex mutex;
  std::optional<EvalError> diagnostic;
  std::map<std::pair<std::string, std::string>, EvalError> rejected;
  void report(EvalError e) {
    std::lock_guard lock(mutex);
    if (!diagnostic) diagnostic = std::move(e);
  }
  void check() const {
    std::lock_guard lock(mutex);
    if (diagnostic) throw *diagnostic;
  }
  void reject(const std::string& source, const std::string& rule, EvalError e) {
    std::lock_guard lock(mutex);
    rejected.try_emplace({source, rule}, std::move(e));
  }
};
// egg-c exposes analysis data rather than the analysis object. Keep its shared
// diagnostic handle in Data so callbacks can propagate errors without globals.
struct AnalysisData : TensorFacts {
  std::shared_ptr<AnalysisState> state;
};
template <Program P>
struct TensorAnalysis {
  using Data = AnalysisData;
  std::shared_ptr<AnalysisState> state = std::make_shared<AnalysisState>();
  EvalResult<void> register_input(std::string name, TensorFacts facts) {
    return evaluate([&] {
      if (facts.conflicts())
        throw EvalError(ErrorKind::Conflict, "conflicting input metadata");
      auto old =
          state->inputs.contains(name) ? state->inputs.at(name) : TensorFacts{};
      auto merged = old;
      merged.shape.merge(facts.shape);
      merged.dtype.merge(facts.dtype);
      if (merged.conflicts())
        throw EvalError(ErrorKind::Conflict,
                        "conflicting input metadata: " + name);
      require(!state->inserted_inputs.contains(name) || merged == old,
              "register input metadata before inserting its node");
      state->inputs[name] = std::move(merged);
    });
  }
  EvalResult<void> check() const {
    return evaluate([&] { state->check(); });
  }
  std::vector<RewriteRejection> rejections() const {
    std::lock_guard lock(state->mutex);
    std::vector<RewriteRejection> out;
    for (const auto& [key, error] : state->rejected)
      out.push_back({key.first, key.second, error});
    return out;
  }
  static TensorFacts infer_node(const Node<P>& node,
                                const std::vector<TensorFacts>& operands,
                                Budget& budget) {
    if (node.kind() == NodeKind::Literal)
      return {Fact<std::vector<std::uint64_t>>::known({}),
              node.dtype() ? Fact<DType>::known(*node.dtype()) : Fact<DType>{}};
    require(node.kind() == NodeKind::Operation,
            "cannot infer unregistered input");
    for (const auto& x : operands)
      if (x.conflicts())
        throw EvalError(ErrorKind::Conflict, "conflicting operand metadata");
    TensorFacts out;
    auto attrs = node.attribute_value();
    auto shape = P::infer_shape(node.op(), operands, attrs, budget);
    if (shape) {
      if (shape.value()) out.shape = decltype(out.shape)::known(*shape.value());
    } else if (shape.error().kind != ErrorKind::Unavailable)
      throw shape.error();
    auto dtype = P::infer_dtype(node.op(), operands, attrs, budget);
    if (dtype) {
      if (dtype.value()) out.dtype = decltype(out.dtype)::known(*dtype.value());
    } else if (dtype.error().kind != ErrorKind::Unavailable)
      throw dtype.error();
    return out;
  }
  Data make(const eggc::EGraph<Node<P>, TensorAnalysis>& graph,
            const Node<P>& node) const {
    try {
      TensorFacts facts;
      if (node.kind() == NodeKind::Input) {
        state->inserted_inputs.insert(node.text());
        auto it = state->inputs.find(node.text());
        if (it != state->inputs.end()) facts = it->second;
      } else {
        std::vector<TensorFacts> operands;
        for (auto id : node.children())
          operands.push_back(graph.analysis_data(id));
        Budget budget;
        facts = infer_node(node, operands, budget);
      }
      return {std::move(facts), state};
    } catch (const EvalError& e) {
      state->report(e);
      return {{Fact<std::vector<std::uint64_t>>::conflict(),
               Fact<DType>::conflict()},
              state};
    }
  }
  eggc::AnalysisMerge merge(Data& into, const Data& from) const {
    bool changed = into.shape.merge(from.shape);
    changed = into.dtype.merge(from.dtype) || changed;
    if (into.conflicts())
      state->report(
          EvalError(ErrorKind::Conflict, "e-class shape or dtype conflict"));
    return changed ? eggc::AnalysisMerge::Changed
                   : eggc::AnalysisMerge::Unchanged;
  }
};
template <Program P>
using BaseGraph = eggc::EGraph<Node<P>, TensorAnalysis<P>>;
// Copies of the analysis share the same state; the public handle remains useful
// after egg-c takes ownership of its analysis object.
template <Program P>
class EGraph : public BaseGraph<P> {
 public:
  TensorAnalysis<P> analysis;
  EGraph(const EGraph&) = delete;
  EGraph& operator=(const EGraph&) = delete;
  EGraph(EGraph&&) = default;
  EGraph& operator=(EGraph&&) = default;
  explicit EGraph(TensorAnalysis<P> analysis = {})
      : BaseGraph<P>(analysis), analysis(std::move(analysis)) {}
};
template <Program P>
std::shared_ptr<AnalysisState> analysis_state(const BaseGraph<P>& graph,
                                              eggc::Id root) {
  return graph.analysis_data(root).state;
}
}  // namespace tepl_runtime
