#include "src/core/semantics/ir.h"

#include <algorithm>

namespace tepl::core::semantics {
bool DescriptorAttributes::needsSynthesis() const {
  return !isCaptured() &&
         std::any_of(fields.begin(), fields.end(), [](const auto& f) {
           return f->kind == Symbol::Kind::Hole;
         });
}
bool DescriptorAttributes::allAuto() const {
  return !isCaptured() && !fields.empty() &&
         std::all_of(fields.begin(), fields.end(), [](const auto& f) {
           return f->kind == Symbol::Kind::Hole;
         });
}
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
      {"index-add-normalization", "(a+b)+c = a+(c+b); a+a = 2*a",
       "integer coordinates; paired tuple lengths; scoped identities", false},
      {"index-scale-normalization", "a*(b*x)=(a*b)*x; a*(x+y)=a*x+a*y",
       "positive per-axis integer scales; equal tuple lengths", false},
      {"extent-floor-bound", "s * floor(q/s) <= q",
       "paired tuple lengths; nonnegative q; positive s", false},
      {"axis-selector-matching", "reverse selector := matching source selector",
       "scoped coordinate identities; supplied offsets and coefficients; "
       "independent legality and equivalence verification",
       false},
      {"index-offset-normalization", "a+(b+x)=(a+b)+x; s*(a+x)=s*a+s*x",
       "paired tuple lengths; metadata offsets; exact integer arithmetic",
       false},
      {"extent-monotonicity", "L<=U implies r+m*L <= r+m*U",
       "established bounds; equal tuple lengths; nonnegative multiplier",
       false},
      {"index-substitution",
       "Read(producer, q) = producer[output coordinates := q]",
       "capture-avoiding substitution; valid access domains", false},
      {"reduction-reflection", "sum(q, f(q)) = sum(q, f(reflect(q)))",
       "reflection selects only bound coordinates; identical finite domains; "
       "the entire summand is substituted; exact associative addition",
       true},
      {"axis-insertion", "place(S,positions(A,S),remaining(A,S)) = A",
       "distinct coordinates; S subset of A; remaining axes retain A order",
       false},
      {"constant-arithmetic",
       "a*(b*m) = (a*b)*m; Const(a)*Const(b) = Const(a*b)",
       "exact integer coefficients; checked overflow", true},
      {"coefficient-combination", "a*m + b*m = (a+b)*m",
       "identical tensor accesses and reduction domains; exact arithmetic",
       true},
      {"additive-cancellation", "a*m + (-a)*m = 0",
       "identical tensor accesses and reduction domains; interface retained",
       true},
      {"zero-multiplication", "0*m = 0",
       "exact arithmetic; interfaces and operation legality retained", true},
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
}  // namespace tepl::core::semantics
