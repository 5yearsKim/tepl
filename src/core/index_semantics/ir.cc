#include "src/core/index_semantics/ir.h"

namespace tepl::core::index_semantics {
const char* guardKindName(GuardKind kind) {
  switch (kind) {
    case GuardKind::Legality:
      return "legality obligation";
    case GuardKind::CoordinateOrder:
      return "coordinate order";
    case GuardKind::ReductionDomain:
      return "reduction domain";
  }
  return "unresolved obligation";
}
const std::vector<Law>& laws() {
  static const std::vector<Law> result{
      {"add-associativity", "Add(Add(x,y),z) = Add(x,Add(y,z))",
       "same free-axis interface; exact associative addition", true},
      {"product-associativity",
       "Product(Product(x,y),z) = Product(x,Product(y,z))",
       "coordinate bindings preserved; exact associative multiplication", true},
      {"commutative-matching",
       "Add(x,y) = Add(y,x); Product(x,y) = Product(y,x)",
       "bindings preserved; exact commutative arithmetic", true},
      {"product-distributivity",
       "Product(Add(x,y),z) -> Add(Product(x,z),Product(y,z))",
       "Add operands aligned; bindings preserved; exact distributive "
       "arithmetic",
       true},
      {"reduction-linearity",
       "Reduce(K,Add(x,y)) -> Add(Reduce(K,x),Reduce(K,y))",
       "same finite reduction domains; exact addition", true},
      {"reduction-extraction",
       "Product(Reduce(K,x),y) -> Reduce(K,Product(x,y))",
       "K fresh and absent from free axes of y; finite value-independent "
       "domains",
       true},
      {"reduction-composition", "Reduce(K,Reduce(J,x)) -> Reduce(K++J,x)",
       "disjoint fresh binders; independent finite domains; exact addition",
       true},
      {"axis-alpha-renaming", "e[axes] = rename(e, corresponding_axes)",
       "bijection preserves domains, output positions, and binder scopes",
       false},
      {"permutation-composition", "permute(permute(S,p),q) = permute(S,p[q])",
       "p and q are full valid permutations of the same ordered interface",
       false},
      {"ordered-axis-split",
       "selected=S[p]; remaining=S[complement(range(len(S)),p)]",
       "S distinct; p valid; selected retains p order; remaining retains S "
       "order",
       false},
      {"axis-map-composition", "S[p][q] = S[gather(p,q)]",
       "p and q valid in their respective coordinate spaces", false},
      {"axis-map-restriction", "q[matched_positions(p,q)] = p",
       "p and q distinct valid selectors into one frame; every coordinate "
       "selected by p survives in q",
       false},
      {"ordered-view-equality", "same_set(A,B) implies A = B",
       "A and B have strictly increasing position maps into the same "
       "distinct coordinate frame",
       false},
      {"paired-axis-selection",
       "match_positions(p,q) = {left,right}; p[left] = q[right]",
       "p and q valid distinct selectors into the same interface; pair order "
       "retained",
       false},
      {"ordered-partition-traversal",
       "stable_select(concat(G1,...,Gn),p) = concat(select_each_block(p))",
       "Gi form consecutive disjoint blocks; p follows physical interface "
       "order",
       false},
      {"physical-axis-order",
       "filter_axes(A,G) = gather(G,order_positions(positions(A,G)))",
       "A and G distinct; G subset of A; order_positions is the stable sorting "
       "permutation",
       false}};
  return result;
}
}  // namespace tepl::core::index_semantics
