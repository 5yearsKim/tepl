#pragma once
#include <eggc/egraph.hpp>

#include "../analysis/tensor.h"
#include "pattern.h"

namespace @TEPL_NAMESPACE@::pattern {
template <class A>
using TensorMetadata = ::std::function<::std::optional<TensorInfo>(
    const ::eggc::EGraph<OpNode, A>&, ::eggc::Id)>;
template <class A, class M>
struct MatchContext {
  const ::eggc::EGraph<OpNode, A>& graph;
  const TensorMatch& matched;
  const M& metadata;
  const DerivedAttrs* derived = nullptr;
  ::std::optional<TensorInfo> tensor(BindingId id) const {
    auto i = matched.tensors.find(id);
    if (i == matched.tensors.end()) return {};
    return metadata(graph, graph.find(i->second));
  }
  ::std::optional<OpAttrs> attrs(BindingId id) const {
    if (derived) {
      auto i = derived->find(id);
      if (i != derived->end()) return i->second;
    }
    auto i = matched.attrs.find(id);
    if (i == matched.attrs.end()) return {};
    return i->second;
  }
};
}  // namespace @TEPL_NAMESPACE@::pattern
