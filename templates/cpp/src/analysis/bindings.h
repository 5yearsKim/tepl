#pragma once
#include <map>

#include "tensor_info.h"

namespace @TEPL_NAMESPACE@::analysis {
class TensorBindingTable {
 public:
  void register_symbol(::std::string name, TensorInfo info) {
    auto existing = bindings_.find(name);
    if (existing != bindings_.end()) {
      if (existing->second != info)
        throw ::std::invalid_argument("conflicting tensor type for input '" +
                                      name + "'");
      return;
    }
    bindings_.emplace(::std::move(name), ::std::move(info));
  }
  void insert(::std::string name, TensorInfo info) {
    register_symbol(::std::move(name), ::std::move(info));
  }
  ::std::optional<TensorInfo> info(::std::string_view name) const {
    auto found = bindings_.find(name);
    return found == bindings_.end() ? ::std::nullopt
                                    : ::std::optional(found->second);
  }

 private:
  ::std::map<::std::string, TensorInfo, ::std::less<>> bindings_;
};
}  // namespace @TEPL_NAMESPACE@::analysis
