#include <cassert>

#include "strings/generated.h"
int main() {
  auto b = strings::graphs::strings::graph_g::build().value();
  auto note = b.graph.nodes(b.root).front().attribute_value().field(0).string();
  assert(note.size() == 3 && note[1] == '\0' && note[2] == 'b');
  assert(tepl_runtime::runner::run(b.graph, strings::rules::all().value()));
  assert(b.graph.find(b.root) == b.graph.find(b.bindings.at("X")));
}
