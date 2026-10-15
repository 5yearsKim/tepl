#pragma once

#include "src/core/semantics/context.h"

namespace tepl::core::semantics {
// Immutable coordinate-tuple expressions. A tuple can have symbolic length.
// Coordinates are scoped logical identities, never axis names or extents.
struct IndexExpr;
using I = std::shared_ptr<const IndexExpr>;
struct IndexExpr {
  enum class Kind {
    Coordinates,
    Add,
    Concat,
    Select,
    Scale,
    Weighted,
    Offset
  } kind;
  S axes;  // Coordinates: identities; Select: selector; Weighted: per-axis
           // scales; Offset: per-axis metadata offsets.
  std::vector<I> args;
  std::int64_t coefficient = 1;
};
I coordinates(S);
I indexAdd(std::vector<I>);
I indexScale(Context&, const S&, const I&);
I indexOffset(Context&, const S&, const I&);
I indexConcat(std::vector<I>);
I indexSelect(Context&, const I&, const S&);
I indexSubstitute(Context&, const I&, const S& frame, const I& replacement);
I indexRename(Context&, const I&, const S& from, const S& to);
S indexLength(const I&);
S indexDependencies(const I&);
S indexUpperBound(Context&, const I&);
bool equalIndex(Context&, const I&, const I&);
std::string showIndex(const I&);
// Affine extent-vector arithmetic, used for domains and access intervals.
bool extentBoundArithmetic(Context&, const S&, const S&, QueryMode);
bool equalExtentArithmetic(Context&, const S&, const S&,
                           QueryMode = QueryMode::Entailment);
}  // namespace tepl::core::semantics
