#include "ir/generated.h"
int main() {
  // Type's attribute value cannot be passed to Std's typed constructor.
  tepl_generated::OpNode::make(tepl_generated::dialects::std::Op::StdCopy,
                               tepl_generated::dialects::type::OpAttrs{}, {});
}
