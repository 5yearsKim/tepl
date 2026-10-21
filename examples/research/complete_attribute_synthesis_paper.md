# Exact Feasibility Generation and Attribute Synthesis

## Setting and theorem

Let \(S=(L,R)\) be a rewrite whose source and replacement are finite trees
over the same named, independently variable tensor inputs, each used exactly
once per tree. Operations are full axis permutations and generalized
contractions. A contraction selects disjoint, duplicate-free batching and
contracting position lists on each operand, pairs lists of equal length and
matching extents, and computes the sum of operand products over contracted
coordinates. Its output order is batching-list order, followed by unselected
left axes and unselected right axes, each in operand order. These are the
entire operation legality conditions. Every RHS descriptor is an independent,
unrestricted unknown.

Metadata \(p\) supplies input shapes and legal source descriptors. Ranks are
arbitrary and finite; physical extents are at least two. Scalars and empty
axis lists are permitted. Arithmetic is exact over the reals. Define

\[
\operatorname{Valid}(S,p,\beta)\iff
\operatorname{Legal}_R(p,\beta)
\land\operatorname{Shape}_R(p,\beta)=\operatorname{Shape}_L(p)
\land\forall X:\llbracket R\rrbracket_{p,\beta,X}
                  =\llbracket L\rrbracket_{p,X}.
\]

Here legality includes every intermediate operation, shapes are ordered,
and attributes \(\beta\) depend only on metadata, not tensor values \(X\).

**Theorem (Exact feasibility generation and constructive synthesis).**
There is a terminating generator that, for each schema \(S\) above, produces
finite metadata programs \((\Phi_S,f_S)\). For every legal \(p\) in this
domain, both programs terminate and

\[
\Phi_S(p)\iff\exists\beta\;\operatorname{Valid}(S,p,\beta).
\]

If \(\Phi_S(p)\) holds, \(f_S(p)\) returns a valid assignment; otherwise
it returns `NotApplicable`. The same generated programs handle all finite
ranks for each fixed schema.

## Constraint and program generation

Give every physical input axis a named port. Replay source attributes,
identifying paired batch and contraction ports while retaining reduced
classes. Permutations rearrange interfaces. This yields logical classes
\(\mathcal A\), their extents, each input's ordered class sequence \(I_c\),
and the ordered final interface \(F\). Let \(\sigma(a)\) be the inputs
containing class \(a\), and \(\bar F\) the set underlying \(F\).

For a RHS subtree containing input set \(U\), compute its boundary

\[
\partial U=\{a:\sigma(a)\cap U\ne\varnothing
          \land(a\in\bar F\ \lor\ \sigma(a)\nsubseteq U)\}.
\tag{1}
\]

At a binary node with child boundaries \(A_\ell,A_r\) and parent boundary
\(P\), compute

\[
\begin{aligned}
B&=A_\ell\cap A_r\cap P,& K&=(A_\ell\cap A_r)\setminus P,\\
LF&=A_\ell\setminus A_r,& RF&=A_r\setminus A_\ell.
\end{aligned}
\tag{2}
\]

Write \(q|_A\) for filtering a sequence by a set, \(\Vert\) for
concatenation, and \(\preceq\) for the subsequence relation. Propagate a
required sequence \(Q\) downward, starting with \(Q=F\). A contraction
passes \(Q|_{LF}\) left and \(Q|_{RF}\) right; a permutation passes an
empty requirement to its child. The predicate \(\Phi_S\) conjoins

\[
\begin{cases}
Q\preceq I_c,&\text{at input }c,\\
Q=(Q|_B)\Vert(Q|_{LF})\Vert(Q|_{RF}),&\text{at a contraction}.
\end{cases}
\tag{3}
\]

Permutations impose no additional check. These calculations use source
metadata and RHS structure, without knowing any RHS attribute values.

If (3) holds, construct layouts bottom-up. A leaf retains \(I_c\). A
permutation orders its child's axes with \(Q\) first and the remaining axes
in child order. At a contraction, choose a batch enumeration \(b\) beginning
with \(Q|_B\) and a contraction enumeration \(k\); enumerate remaining
axes by a fixed canonical class order. Given child layouts \(W_\ell,W_r\),
emit

\[
\begin{aligned}
b_\ell&=\operatorname{pos}(W_\ell,b),&
b_r&=\operatorname{pos}(W_r,b),\\
k_\ell&=\operatorname{pos}(W_\ell,k),&
k_r&=\operatorname{pos}(W_r,k),\\
W&=b\Vert(W_\ell|_{LF})\Vert(W_r|_{RF}),
\end{aligned}
\tag{4}
\]

where \(\operatorname{pos}(W,q)\) lists the positions of \(q\)'s axes
in \(W\). Permutation descriptors use the same positional conversion.

The generator emits these traversals as finite programs over metadata reads,
finite partitions, sets, and sequences. Instructions have their mathematical
meanings: merge classes, select positions, filter, concatenate, compare, and
iterate over finite collections. Source replay and boundary construction use
postorder; requirement propagation uses preorder; attribute construction uses
postorder. Axis loops retain runtime bounds. The predicate returns (3); the
constructor executes the same analysis, returns `NotApplicable` if (3) fails,
and otherwise executes (4) and the permutation cases.

**Algorithm 1. Generation and execution of feasibility and attribute programs.**
`emit` translates the routine templates below into metadata instructions: tree
traversals are expanded using the fixed schema, while operations on axis lists
retain their runtime lengths. Its arguments describe code; they do not execute
that code during generation. `ReplaySource` performs the source replay above:
create input ports, select permutation positions, merge each batch/contract
pair, form ordered outputs, and finally canonicalize classes and compute their
supports. `enum` lists a set in canonical class order, and
`extend(q,A,w) = concat(q, filter(w, A \ set(q)))`. The returned analysis state
contains the network, boundaries, roles, requirements, and Boolean result.

```text
GENERATE(S = (L, R)):                         // compile time; no metadata p
    U[v] ← input captures below v, for every RHS node v
    A ← emit ANALYZE(L, R, U; p)              // p is a runtime parameter
    D ← emit CONSTRUCT(R; state)
    emit ΦS(p): state ← A(p); return state.ok
    emit fS(p): state ← A(p)
                if not state.ok: return NotApplicable
                return D(state)
    return (ΦS, fS)

ANALYZE(L, R, U; p):                         // match time
    (classes, support, I, F) ← ReplaySource(L, p)       // pass 1
    for v in postorder(R):                            // pass 2
        P[v] ← {a in classes | support[a] intersects U[v]
                  and (a in set(F) or support[a] is not a subset of U[v])}
        if v = dot(l, r):
            B[v]  ← P[l] ∩ P[r] ∩ P[v]
            K[v]  ← (P[l] ∩ P[r]) \ P[v]
            LF[v] ← P[l] \ P[r]; RF[v] ← P[r] \ P[l]
    Q[root(R)] ← F                                    // pass 3
    for v in preorder(R):
        if v = perm(u): Q[u] ← []
        if v = dot(l, r):
            Q[l] ← filter(Q[v], LF[v]); Q[r] ← filter(Q[v], RF[v])
    ok ← true                                         // pass 4
    for v in nodes(R):
        if v = input(c): ok ← ok and subsequence(Q[v], I[c])
        if v = dot(l, r):
            qb ← filter(Q[v], B[v])
            ql ← filter(Q[v], LF[v]); qr ← filter(Q[v], RF[v])
            ok ← ok and equal(Q[v], concat(qb, ql, qr))
    return {classes, support, I, F, P, B, K, LF, RF, Q, ok}

CONSTRUCT(R; state):                         // match time; requires state.ok
    β ← empty attribute map                           // pass 5
    for v in postorder(R):
        if v = input(c): W[v] ← I[c]
        if v = perm(u):
            W[v] ← extend(Q[v], P[v], W[u])
            β[v] ← pos(W[u], W[v])
        if v = dot(l, r):
            b ← extend(filter(Q[v], B[v]), B[v], enum(B[v]))
            k ← enum(K[v])
            β[v] ← (pos(W[l], b), pos(W[r], b), pos(W[l], k), pos(W[r], k))
            W[v] ← concat(b, filter(W[l], LF[v]), filter(W[r], RF[v]))
    return Solved(β)
```

Here `CONSTRUCT` reads the named entries of `state`, `dot` denotes the specified
generalized contraction, and `β[v]` is stored in that node's independent
descriptor fields. The five passes satisfy these
invariants, established in proof part 4:

| Pass | Invariant on completion | Definition |
| --- | --- | --- |
| 1. Source analysis | Classes and ordered interfaces equal the source network. | Source replay |
| 2. Boundary analysis | Each stored boundary and role set equals its mathematical definition. | (1)–(2) |
| 3. Order propagation | Each `Q[v]` equals the required subsequence at that node. | Root, permutation, and contraction propagation |
| 4. Constraint checking | `ok` equals the conjunction of all feasibility constraints. | (3) |
| 5. Attribute construction | Layouts and attributes equal the mathematical constructor's results. | (4) and permutation construction |

## Proof

**1. Exactness of the axis representation.** Legal pairings merge distinct
components from disjoint input subtrees. Each merged component has at most
one continuing wire; contraction removes that wire. Consequently each final
class contains at most one port per input, and each intermediate exposes at
most one occurrence of that class. Distinct wires of one intermediate cannot
be merged later: at every ancestor they arrive through the same operand.
Every class contains an input port; every reduced class involves at least
two inputs.

Expanding finite contractions gives the network denotation

\[
T(X)[y]=
\sum_{\substack{t_a\in[0,d_a)\ (a\in\mathcal A)\\t_{F_j}=y_j}}
\prod_c X_c[(t_a)_{a\in I_c}],
\tag{5}
\]

with each input tuple ordered by \(I_c\). Thus equal port partitions and
ordered outputs imply tensor equality. Conversely, for equally shaped outputs,
attach a named port for each output position to its class. If these augmented
partitions differ, some pair is identified in one and separated in the other.
Assign the separated classes coordinates 0 and 1, and all remaining classes 0.
Choose each input to be one-hot at its resulting tuple and evaluate at the
resulting output coordinate. The separating network returns 1: every summed
coordinate is fixed by an input port, so precisely one term contributes.
The identifying network returns 0. Both coordinates exist because extents
are at least two. Different output shapes already violate equality. Hence
semantic equality is equivalent to equality of the complete axis representation.

**2. Necessity of the constraints.** In any valid replacement, a class needed
outside a subtree or at the final output must remain exposed. A non-output
class supported entirely inside the subtree must be reduced at the lowest
common ancestor of its supporting inputs. Its two incoming components must
join there; above that node, no outside occurrence remains with which to
contract it. Permutations cannot eliminate it, and pairing another class
would change the representation. Therefore (1) is the exact boundary, forcing
the roles and same-class pairings in (2).

A contraction's output consists of batch, left-free, and right-free blocks.
Any required subsequence must respect this block order, giving (3); free-axis
subsequences must also occur in the respective children. Batch position lists
can select axes in any order and impose no child-order requirement. A full
permutation can reorder its child freely. At a leaf the physical order is
fixed. Induction from the root therefore proves that every valid assignment
satisfies every generated check.

**3. Sufficiency and construction.** From (1)–(2), the parent boundary is
the disjoint union \(B\mathbin{\dot\cup}LF\mathbin{\dot\cup}RF\).
Induct bottom-up: each constructed layout lists exactly its boundary and
contains its required subsequence. The leaf case is (3). A permutation
places its requirement first. At a contraction, the chosen batch order
contains \(Q|_B\); the child induction hypotheses supply the required free
subsequences. Equation (3) makes their concatenation contain \(Q\).

All position lookups are defined and duplicate-free. Batch and contraction
lists are disjoint, and paired positions select the same logical classes,
hence equal extents. The descriptors are therefore legal. The construction
merges precisely the same-class components, reduces non-output classes at
their support's lowest common ancestor, and never merges distinct classes.
Induction preserves the source partition restricted to each subtree's ports.
At the root, the layout contains \(F\) and has exactly its underlying set,
so it equals \(F\). By (5), the resulting replacement is equivalent for
every input value. This proves sufficiency.

**4. Correctness and termination of generation.** Prove Algorithm 1's five
invariants by induction over each emitted traversal. Source instructions
reproduce the replayed partition and interfaces;
after each pairing loop prefix, exactly that prefix's identifications have
been added. Boundary loops accumulate exactly the visited classes satisfying
(1). Preorder assignments reproduce the required sequences, and the Boolean
accumulator conjoins exactly the visited instances of (3). On the successful
branch, postorder construction reproduces (4) and the permutation layouts,
using identical enumeration conventions. Part 3 establishes the preconditions
of every positional lookup. A false check bypasses construction.

Generation visits a finite schema and emits finite templates. Execution
visits finite trees and iterates over finite metadata collections, so both
terminate. Ranks affect loop lengths, not the generated program's structure.
Parts 1–3 give exact feasibility and correct attributes; this traversal
induction transfers those properties to the generated programs. ∎

**Scope.** The theorem concerns the specified mathematical generator. Applying
it to TEPL additionally requires semantics-preserving frontend translation and
compiler/evaluator correspondence; it does not establish completeness of the
current compiler or of reasoning about arbitrary user conditions. An explicit
inferred-condition feature may apply a rule under \(C(p)\land\Phi_S(p)\).
