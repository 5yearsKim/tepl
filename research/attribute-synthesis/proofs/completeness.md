# Complete axis-attribute synthesis for algebraic tensor rewrites

**Mathematical proof draft, revised October 28, 2026.** This is the unified proof
for contraction, permutation, whole-axis reversal, unary summation,
aligned weighted scalar products, and a restricted additive fragment.
Sections 1--7 prove a direct constructor for the signed-axis core.
Section 11 extends exact constructive completeness to aligned weighted scalar
products by coefficient normalization and finite signed-network state tables.
Section 12 generalizes the contract to metadata-weighted signed wiring, including
broadcast with supplied extents, unit-axis operations, keepdims sums, and full-axis
means. Only axis slots are synthesized; extent/coordinate fields remain supplied.
Sections 12.6--12.7 add linear contexts over separated sums and checked
normalizations, including unit-only reshape and restricted scalar division.
The original unsigned theorem and the signed-axis constructor remain special cases.
The [paper version](../paper.md) gives the compact
statement. The proof concerns the specified mathematical generators; current
compiler coverage and remaining implementation obligations are in Section 9,
and the scalar-extension experiment is in Section 11.4. Section 12 uses no
coordinate-grid enumeration and does not assert integration into the compiler.

## 1. Semantics and supported core fragment

All tensors have exact real entries. Input shapes have arbitrary finite ranks
and physical extents at least two; rank-zero tensors are allowed. Tensor
inputs are independently variable, without symmetry, sparsity, or other
value assumptions.

The operation contracts are:

| Operation | Descriptor, legality, and meaning |
| --- | --- |
| Permutation | A full permutation \(\pi\) of input positions; output position \(j\) reads input position \(\pi_j\). |
| Reversal | An in-range duplicate-free list \(d\); retain the ordered shape and read input coordinate \(n_i-1-y_i\) on selected axes \(i\), and \(y_i\) on the others. Any subset, including the empty set, is legal. |
| Unary sum | An in-range duplicate-free list \(d\); sum over precisely those axes, remove them, and retain the other axes in operand order. |
| Contraction | Four ordered lists \((b_\ell,b_r,k_\ell,k_r)\), each in range and duplicate-free. Batch and contraction lists are disjoint on each operand; corresponding lists have equal lengths and paired extents. Multiply operand reads and sum over the paired contraction axes. Output order is batch-list order, followed by unselected left and right axes in operand order. |
| Addition | Elementwise addition of operands with identical ordered shapes; no attributes or broadcasting. |

Empty descriptor lists and scalar outputs are allowed. These are the entire
legality contracts: there are no extra layout or attribute restrictions.
They justify finite distributivity and changes of summation order under
exact arithmetic, not strict floating-point reassociation.

**Attribute boundary of the signed-axis fragment.** In the core theorem and
the scalar extension in Section 11, operation descriptors may only select,
order, or pair whole axes. Attribute-free operations must match the stated contracts. Exclude
an operation on either side if any descriptor field specifies an extent,
coordinate offset, coordinate stride, padding amount, or target shape, even
when that field could be supplied explicitly. Thus broadcast, slice, reshape,
and stride/padding-based convolution do not enter this fragment.

An axis position is allowed: for example, `[1]` may select the second axis.
A coordinate index within an axis, such as a slice start, is excluded. This
distinction is semantic, not a distinction between integer storage types.
Input extents remain metadata for legality and the fixed reflection
\(i\mapsto n-1-i\); neither use introduces an extent-valued descriptor.
This boundary is necessary for admission, not sufficient: an operation must
also match one of the exact semantic contracts and the tree assumptions below.

A rule specification \(\Sigma\) records the source and fixed RHS trees,
named inputs, operation contracts, captured source-field references, and
RHS axis-attribute holes. The source may contain addition anywhere. Expand it by
linearity and bilinearity to \(L=\sum_{j\in J}L_j\). Require:

1. Each \(L_j\) is a contraction/permutation/reversal/unary-sum tree over a nonempty
   input set \(U_j\), using each input once.
2. The sets \(U_j\) are distinct. They may overlap, so an input can occur in
   several additive components.
3. The actual RHS is an explicit sum \(R=\sum_{j\in J}R_j\), where \(R_j\)
   is such a read-once tree over exactly \(U_j\).
4. Every RHS descriptor consists of independent unrestricted `auto` axis fields.
   There are no pins, explicit descriptors, shared unknowns, constants, or other
   operations in this fragment.

These are finite syntactic admission checks, independent of whether
replacement attributes exist. In particular, RHS normalization must not
copy one original unknown into independent branch-local unknowns.
Source expansion may copy *captured* descriptors; all copies retain their
original values. Expansion preserves legality and the source output shape:
linearity preserves values, and substituting an equally shaped summand does
not change an enclosing operation's descriptor legality or output shape.

Metadata \(p\) supplies the input shapes and all captured source attributes,
including any sharing between source occurrences. Define

\[
\begin{aligned}
\operatorname{Valid}_\Sigma(p,\beta)\iff{}&
 \operatorname{Legal}_R(p,\beta)\\
&\land\operatorname{Shape}_R(p,\beta)=\operatorname{Shape}_L(p)\\
&\land\forall X:\ \llbracket R\rrbracket_{p,\beta,X}
                    =\llbracket L\rrbracket_{p,X}.
\end{aligned}
\tag{1}
\]

Legality includes every intermediate operation; shapes are ordered vectors.
Attributes depend on metadata, not tensor contents: the quantifier order is
\(\exists\beta\,\forall X\). All subsequent metadata quantifiers assume legal
source operations and the extent domain above. A single summand gives the
non-additive fragment.

### 1.1 Typed attributes and the axis-only completion problem

TEPL distinguishes the declared attribute role from the integer used to store
it. The public schema vocabulary is:

| Declaration | Meaning | Storage | Eligible for `auto`? |
| --- | --- | --- | --- |
| `axis`, `axis[]` | Positions selecting, ordering, or pairing tensor axes | `u64` | Yes, for non-optional fields |
| `extent`, `extent[]` | Axis sizes or an explicitly supplied target shape | `u64` | No |
| `coordinate`, `coordinate[]` | Coordinate offsets, starts, strides, or padding amounts | `i64` | No |
| `u64`, `i64` and their lists | General integer settings, such as group counts or numeric codes | Declared integer type | No |

`axes` denotes logical axis structure in an operation declaration; `axis` is
the role of a numeric selector field. `coordinate` declares a metadata
parameter, not the tensor-valued contents of an index operand. Signed storage
does not by itself make negative strides or padding legal: the operation's
contract determines legality. The old public `index` type is removed.

The checker expands defaults and represents each request by `AutoField`.
Supplying an entire RHS descriptor explicitly requests no synthesis for it.
Omitting a descriptor may create only eligible axis requests; it must not create
unknown extent, coordinate, or general integer fields. For example:

```tepl
derive {
    out.attrs = {
        dimensions: auto,
        new_extents: source.attrs.new_extents
    };
}
```

This requests a broadcast axis mapping with the new extents fixed. It does
not request a complete broadcast descriptor from unknown output sizes.
Computing a candidate operation's output shape from fixed metadata is also
different from synthesizing an extent-valued attribute.

More generally, write \(s_\Sigma(p)\) for all supplied RHS fields, including
supplied axis fields and schema defaults, and \(\alpha\) for precisely the
`AutoField` slots. Every program in \(s_\Sigma\) depends only on source
metadata, not tensor entries or unresolved \(\alpha\). Let
\(D_\Sigma(p,\alpha)\) insert \(\alpha\) in those slots and preserve every
supplied field. The migrated completion problem is

\[
\exists\alpha\;\Bigl[\operatorname{Defined}(s_\Sigma(p))\ \land
  \operatorname{Valid}_\Sigma(p,D_\Sigma(p,\alpha))\Bigr].
\tag{T}
\]

Definedness includes integer ranges and checked conversions. The constructor
must preserve \(s_\Sigma(p)\), including its evaluation requirements; it cannot
repair a candidate by changing a supplied extent or coordinate parameter.
For Sections 1--11, every unknown descriptor field has role `axis`, and the
whole-axis contracts exclude supplied extent/coordinate fields as well. Thus
\(\beta\) in their existing statements is just the collection of axis choices:
none of those theorems quantifies over extents or coordinates. Section 12
supplies the broader argument for metadata-weighted signed wiring; role annotations alone do not establish admission.

The mathematics uses ideal exact arithmetic and arbitrary finite ranks.
An implementation additionally needs faithful semantics, representable
metadata, and defined supplied programs. A concrete overflow check or a
bounded proof search is not a completeness proof.

## 2. An exact signed axis representation

For an extent \(n\ge 2\), define \(r_n(t)=n-1-t\). Write \(r_n^0(t)=t\)
and \(r_n^1(t)=r_n(t)\); composition adds the exponents modulo two.
Every such map is a bijection of the full coordinate interval. Bits below
are elements of \(\mathbb F_2\), with addition written \(\oplus\).

Analyze one source component. Give every physical input axis a named port
\((c,i)\). A signed class has a reference coordinate \(t_a\) and a bit
\(\epsilon_q\) for each physical port \(q\): its coordinate is
\(r_{d_a}^{\epsilon_q}(t_a)\). Retain classes even after reduction. Write

\[
N=(P,\sim,\epsilon,F,\eta),\qquad
\mathcal A=P/{\sim},\qquad
I_c=(a(c,0),\ldots,a(c,r_c-1)),
\]

where \(F\) is the ordered output-class sequence, \(\eta_i\) its attachment
bits, and \(d_a\) class \(a\)'s extent. Thus output coordinate \(y_i\) equals
\(r_{d_{F_i}}^{\eta_i}(t_{F_i})\). Classes are sets of named input ports,
not dimension sizes.

Changing the reference of one class from \(t_a\) to \(r_{d_a}(t_a)\) toggles
all its port and output bits without changing its meaning. Fix this freedom
by setting the bit of the least physical input port in each class to zero,
using a fixed order of capture identifiers and positions. All comparisons
below use this canonical reference. Every class has an input port, so this
normalization is always defined.

Source replay is constructive. Start with singleton input classes and zero
bits. Track each exposed axis as a class and an orientation bit. Permutations
reorder these pairs; reversals toggle the selected exposed bits; unary sums
remove exposed pairs. At a contraction, paired exposed coordinates are equal:
merge their classes with the relative parity imposed by their exposed bits.
Rebase the merged class to its least input port, updating its port bits and
any retained exposed bit. Keep the common coordinate for a batch pair and
remove it for a contraction pair. Free axes retain their coordinates.
Finally record the exposed classes and bits as \((F,\eta)\). Explicit finite
class rebasing suffices; no attribute search or equality oracle is used.

Read-once trees have a useful wire property: each class has at most one port
per input and at most one live occurrence at any node. Binary pairings join
components from disjoint input subtrees; a removed component cannot reappear.
Two exposed components from the same child can never be joined later.
Unary sums remove wires without merging them; reversal only changes their
orientation. Thus merges never impose a reflected equality on an already
connected class. Every class contains an input port, including classes reduced
from only one input. There are no unobserved summation classes or reflection
cycles restricting a coordinate to the center of an odd-sized axis.

Expanding the finite sums and products gives the network denotation

\[
\mathcal T_N(X)[y]=
\sum_{\substack{t_a\in[0,d_a)\ (a\in\mathcal A)\\
                 r_{d_{F_i}}^{\eta_i}(t_{F_i})=y_i\ \text{for every }i}}
\prod_c X_c[\bigl(r_{d_{a(c,k)}}^{\epsilon_{(c,k)}}
                        (t_{a(c,k)})\bigr)_{k=0}^{r_c-1}].
\tag{2}
\]

**Lemma 1 (Signed axis exactness).** Two legal read-once components over the
same physical inputs are equal for every input value, with the same ordered
output shape, if and only if their canonical signed port partitions and
ordered signed output attachments agree.

**Proof.** Equal representations make (2) identical up to dummy-index
renaming. Reference changes also preserve (2), by bijective substitution in
the full index intervals. Conversely, first reject different output shapes.
Attach a named port to every output position with its bit \(\eta_i\).

If the resulting unsigned partitions differ, choose two ports connected in
one but separated in the other. The connected network requires either equal
or reflected coordinates at those ports. In the separating network their
class references are independent: choose endpoint coordinates that violate
that required relationship. This is possible because the two endpoints
\(0,d_a-1\) are distinct, and reflection swaps them. Set other class
references to endpoints arbitrarily. If partitions agree but canonical bits
differ, some pair of ports has different relative parity. Assign a class
reference to an endpoint in one network; its port coordinates then violate
the relationship required by the other. This also handles unequal output
attachment bits. Each construction satisfies one network's complete signed
relationships and violates the other's.

Set each input to one at its resulting coordinate tuple and zero elsewhere.
At the corresponding output coordinate, the satisfied network returns one:
every summation coordinate is uniquely fixed by an input port and an
invertible reflection, so exactly one term contributes. The other network
returns zero because its required relationship is violated. Scalar inputs
contribute their single entry, chosen as one. Thus different canonical signed
representations cannot agree for every input.
\(\square\)

This lemma concerns the full signed partition, including reduced classes and
output attachments. Equal output-axis sets, shapes, or unsigned partitions
alone are insufficient. With no reversal, all canonical bits are zero, giving
the previous unsigned axis-exactness lemma as a special case.

## 3. Canonical live axes

For a class \(a\), let \(\sigma(a)\) be its supporting inputs. At RHS node
\(v\), let \(U_v\) be the inputs below it. The mandatory axes are

\[
M_v=\{a:\sigma(a)\cap U_v\ne\varnothing
 \land(a\in\operatorname{set}(F)\lor\sigma(a)\nsubseteq U_v)\}.
\tag{3}
\]

A final-output class or one needed outside the subtree cannot disappear.
Other classes may remain exposed until a later unary sum. Therefore
\(M_v\) is not generally the actual live set.

Instead compute canonical live sets \(P_v\) bottom-up:

\[
\begin{array}{ll}
P_v=\operatorname{set}(I_c),&v\text{ is input }c,\\
P_v=P_u,&v\text{ permutes or reverses child }u,\\
P_v=P_u\cap M_v,&v\text{ sums child }u.
\end{array}
\tag{4}
\]

For a contraction with children \(\ell,r\), use

\[
\begin{aligned}
B_v&=P_\ell\cap P_r\cap M_v,&
K_v&=(P_\ell\cap P_r)\setminus M_v,\\
LF_v&=P_\ell\setminus P_r,&
RF_v&=P_r\setminus P_\ell,\\
P_v&=B_v\cup LF_v\cup RF_v.&&
\end{aligned}
\tag{5}
\]

Thus a dot contracts every completed shared non-output class; a unary sum
removes every completed non-output class still present. An unshared class
must remain free at a dot, even if no longer needed. Direct induction gives
\(M_v\subseteq P_v\), with each class in \(P_v\) supported inside \(U_v\).

For any valid assignment, measure the orientation of each exposed coordinate
against the corresponding canonical source reference. Such a bit is uniquely
determined by its relationship to any supporting input port; validity and
Lemma 1 ensure that all supporting ports agree on it.

**Lemma 2 (Signed canonicalization preserves existence).** If a valid assignment
has intermediate layouts \(W_v\) and exposed orientation bits, another valid
assignment has exactly \(W_v|_{P_v}\) at every node, with the same orientation
bits on surviving classes. In particular,
\(P_{\mathrm{root}}=\operatorname{set}(F)\).

Here \(w|_A\) filters a sequence by membership in \(A\).

**Proof.** By Lemma 1, a valid RHS has the source signed partition. Hence every
mandatory class is exposed where needed. We show by induction that
\(P_v\subseteq\operatorname{set}(W_v)\) and the projected layouts can be
realized:

- Inputs are unchanged; a full permutation can realize any ordering of its
  projected child without changing surviving orientations.
- At a reversal, keep precisely the original selected logical classes that
  survive the projection, translating their positions to the projected child.
  This preserves all surviving orientation bits and the child's axis order.
- A unary sum retains the mandatory classes in its child. These also occur
  in the original output; removing other projected axes produces precisely
  \(W_v|_{P_v}\), since sums preserve survivor order.
- At a dot, a class shared by both canonical children must be joined here.
  If left separate, its components would arrive through the same child at
  every later ancestor and could never join. A shared mandatory class must
  be batched in the original assignment. A shared completed class may have
  been batched or contracted; contracting it now removes only an unnecessary
  output axis.

  A canonical left-free class cannot occur in the original right child.
  If it did, support in both children would make it mandatory in the right
  child, hence present in \(P_r\), a contradiction. Likewise for right-free
  classes. Consequently survivors keep their batch/free roles. Filter the
  original batch order to \(B_v\), contract \(K_v\) in any common order,
  and select their positions in the projected children. The output is
  exactly \(W_v|_{P_v}\). Paired child coordinates have equal orientations
  in the original valid assignment; projection preserves this equality,
  whether the pair remains batched or is contracted earlier.

All descriptors are legal: selected axes are present once, paired extents
agree, and batch/reduction sets are disjoint. These changes join no distinct
source classes and eliminate a class only after all its supporting ports
have joined. Eliminating a completed class earlier is sound because no later
merge can involve it in a valid realization: its input support is already
complete. Later dots can only carry it as a free axis and multiply by factors
independent of its coordinate; distributivity moves its sum across those dots.
Permutations transport it, and reversals are bijections of its full summation
interval, so discarding a later reversal of an already eliminated class
preserves the sum. Finite sums over other classes commute under the exact
arithmetic contract. No summation is duplicated: subsequent sum descriptors
omit the class already eliminated. At the root,
\(M_{\mathrm{root}}=\operatorname{set}(F)\) and
\(W_{\mathrm{root}}=F\), so the inclusions force
\(P_{\mathrm{root}}=\operatorname{set}(F)\). The projected expression has the
same signed network and final signed order, hence is equivalent by Lemma 1.
\(\square\)

This is an early-elimination *normal form*, not a claim that every valid
assignment has the same intermediate sets. Without unary sums on either
side, the canonical sets reduce to the original mandatory boundaries.

## 4. Exact order and orientation constraints, and a constructor

### 4.1 Axis sets and order

First require the root-set check
\(P_{\mathrm{root}}=\operatorname{set}(F)\).
Propagate required subsequences, rather than choosing complete layouts:

- Set \(Q_{\mathrm{root}}=F\).
- At a permutation, pass the empty sequence.
- At a unary sum or reversal, pass \(Q_v\) unchanged.
- At a dot, pass \(Q_v|_{LF_v}\) left and \(Q_v|_{RF_v}\) right.

Let \(\preceq\) denote subsequence and \(\Vert\) concatenation. Define

\[
\phi(p)=
[P_{\mathrm{root}}=\operatorname{set}(F)]
\land\bigwedge_{\text{inputs }c}[Q_c\preceq I_c]
\land\bigwedge_{\text{dots }v}
[Q_v=(Q_v|_{B_v})\Vert(Q_v|_{LF_v})\Vert(Q_v|_{RF_v})].
\tag{6}
\]

These are source-metadata computations, not existential intermediate orders.

**Necessity.** Any valid assignment has a canonical realization by Lemma 2.
Its root order is \(F\). At a dot, output has batch/left-free/right-free
blocks, so every required subsequence respects this block order and passes
the appropriate free subsequence to each child. Batch lists can select child
positions in any order, imposing no batch-order requirement on children.
Unary sums preserve the order of surviving axes; reversals preserve the entire
axis order. A permutation imposes no child requirement; an input has its fixed
physical order. Top-down induction
therefore gives every check in (6).

**Unsigned layout construction.** On success, construct layouts bottom-up.
An input retains \(I_c\). A permutation puts \(Q_v\) first and then its
remaining child axes. A reversal retains its child's layout. A unary sum
removes the positions of \(P_u\setminus P_v\).
At a dot, enumerate \(B_v\) by a list \(b\) beginning with \(Q_v|_{B_v}\);
append remaining batch classes in a fixed class order. Enumerate \(K_v\)
by any fixed list \(k\). For child layouts \(W_\ell,W_r\), emit

\[
\begin{aligned}
\beta_v={}&(\operatorname{pos}(W_\ell,b),\operatorname{pos}(W_r,b),
           \operatorname{pos}(W_\ell,k),\operatorname{pos}(W_r,k)),\\
W_v={}&b\Vert(W_\ell|_{LF_v})\Vert(W_r|_{RF_v}).
\end{aligned}
\tag{7}
\]

Position lookup lists the positions of the given classes. A permutation
uses the same lookup to realize its chosen layout; a sum uses it for the
removed classes.

Induction gives \(\operatorname{set}(W_v)=P_v\) and \(Q_v\preceq W_v\):
(6) supplies the required block order, while sums preserve the required
subsequence. Hence all lookups are defined and every descriptor is legal.
Every source class spanning both children is present in both canonical
sets, since it is mandatory there; the dot joins its occurrences. No
distinct source classes merge. Elimination occurs only after support is
complete. Inductively the partial partition is the source partition
restricted to the subtree's ports. At the root, the set check and
\(F\preceq W_{\mathrm{root}}\) force \(W_{\mathrm{root}}=F\); all non-output
classes have disappeared. This establishes the unsigned connections and
ordered output, but signed equivalence additionally requires the following
orientation checks.

### 4.2 Orientation feasibility without descriptor enumeration

Run this stage only when \(\phi(p)\) holds; otherwise define \(\chi(p)=\mathrm{false}\)
without building the graph. In particular, the root-set check establishes that
every root attachment below names an existing vertex. This convention makes
the feasibility predicate total even on structurally impossible RHS instances.

For every node \(v\) and class \(a\in P_v\), create a bit \(e_{v,a}\),
meaning that its exposed coordinate is \(r_{d_a}^{e_{v,a}}(t_a)\). Use the
canonical source signs from Section 2 and impose these constraints:

| Node or interface | Orientation constraints |
| --- | --- |
| Input \(c\), port \((c,i)\) in class \(a\) | \(e_{c,a}=\epsilon_{(c,i)}\). |
| Permutation \(v\) of \(u\) | \(e_{v,a}=e_{u,a}\) for every \(a\in P_v\). |
| Unary sum \(v\) of \(u\) | \(e_{v,a}=e_{u,a}\) for every surviving \(a\in P_v\); no output constraint for eliminated classes. |
| Reversal \(v\) of \(u\) | No equality constraint between \(e_{v,a}\) and \(e_{u,a}\): the independently selectable reversal bit will be their XOR. |
| Contraction \(v\) | \(e_{\ell,a}=e_{r,a}\) for every \(a\in B_v\cup K_v\). For \(a\in B_v\), also \(e_{v,a}=e_{\ell,a}\). For \(a\in LF_v\) or \(RF_v\), copy the corresponding child's bit to \(e_{v,a}\). |
| Root | \(e_{\mathrm{root},F_i}=\eta_i\) for every output position \(i\). |

These are only variable equalities and pins to zero or one. Build an undirected
graph whose vertices are the \((v,a)\) pairs and whose edges are the required
equalities. Mark the input and root pins. Define \(\chi(p)\) to hold iff no
connected component contains both a zero pin and a one pin. A finite graph
traversal decides \(\chi\); assign the pinned bit to each pinned component and
zero to every unpinned component. Thus consistency and a witness are computed
explicitly, without an assumed constraint-solving oracle. A failed check has
a path between contradictory pins as a certificate.

At a reversal node, construct its descriptor from the chosen bits:

\[
d_v=\operatorname{pos}\left(W_u,
       [a\in W_u:\ e_{v,a}\oplus e_{u,a}=1]\right).
\tag{7a}
\]

The bracketed list filters the child layout in its existing order, so this
descriptor is in range and duplicate-free. An empty list is permitted.
This uses only reverse nodes already present in the fixed RHS; it does not
insert nodes or permit reflections at a contraction or permutation.

**Lemma 3 (Exact orientation feasibility and joint construction).** On the
canonical live sets, a valid component assignment exists iff
\(\phi(p)\land\chi(p)\). The layout constructor (7), the unary cases, and
(7a) jointly construct one whenever the conjunction holds.

**Proof.** For necessity, apply Lemma 2 to a valid assignment. Its layouts
satisfy (6). Measure its surviving coordinates relative to the fixed source
references. A leaf coordinate is its physical port coordinate; permutations
and sums copy surviving coordinates; a dot pairs equal coordinates and copies
its batch/free outputs. Root coordinates have the source attachments. These
are exactly the displayed equalities and pins. Reverse nodes can toggle their
coordinates, so imposing no equality there loses no necessary restriction.
The measured bits therefore satisfy the graph, proving \(\chi\).

For sufficiency, first choose the layouts from Section 4.1 and a graph witness.
At a reversal, (7a) makes the child coordinate read at output coordinate
\(r_{d_a}^{e_{v,a}}(t_a)\) equal to
\(r_{d_a}^{e_{u,a}}(t_a)\), as required. At a contraction, equal child bits
make corresponding coordinates agree for each batch and reduction pair.
Each paired full summation coordinate is a bijective image of \(t_a\), so
the contraction introduces precisely one full sum over that class. Unary
sums have the same property. The unsigned construction joins exactly the
source classes, eliminating each only after its support is complete; the
orientation constraints make every input read and root attachment have the
source sign. Induction over the constructed tree therefore gives (2).
All attributes are legal, and Lemma 1 establishes equality.

Finally, the orientation graph is indexed by logical classes, not their
positions. Any layout satisfying Section 4.1 can realize its bits: reverse
descriptors can select any subset of the actual child positions, and all
other constraints only copy or equate class coordinates. Hence solving the
order and orientation problems separately loses no joint solution.
\(\square\)

If both trees omit reversal, source bits are all zero and assigning zero to
every graph vertex always succeeds whenever \(\phi\) holds. Thus
\(\phi\land\chi=\phi\), and Lemma 3 reduces to the previous unsigned
constructor, not a weaker condition.

## 5. Additive decomposition

Scale each independent input \(X_c\) by a real scalar \(z_c\). Read-once
multilinearity, including unary summation and reversal, gives

\[
L_j((z_cX_c)_c)=\left(\prod_{c\in U_j}z_c\right)L_j(X),
\]

and the same equation for \(R_j\). Distinct \(U_j\) yield distinct monomials.
At any output coordinate, uniqueness of polynomial coefficients therefore gives

\[
\forall X:\ L(X)=R(X)
\quad\Longleftrightarrow\quad
\forall j\;\forall X:\ L_j(X)=R_j(X),
\tag{8}
\]

for legal, equally shaped expressions.

For component \(j\), let \(\psi_j=\phi_j\land\chi_j\) and let \(f_j\)
be the joint layout/orientation constructor of Section 4. Define

\[
\Phi_\Sigma(p)=\bigwedge_{j\in J}\psi_j(p),\qquad
f_\Sigma(p)=\bigcup_{j\in J}f_j(p)
\quad\text{on success}.
\tag{9}
\]

The union assigns disjoint descriptor identifiers. Components use separate
signed axis partitions: addition aligns output coordinates but does not identify
input-port classes from different terms. All successful components have the
source shape, so their sum is legal.

**Theorem (Complete constructive attribute synthesis for the signed-axis core).** For every admitted
specification and legal in-domain metadata,

\[
\boxed{
\begin{aligned}
\Phi_\Sigma(p)&\iff\exists\beta\;
                    \operatorname{Valid}_\Sigma(p,\beta),\\
\Phi_\Sigma(p)&\Rightarrow
                    \operatorname{Valid}_\Sigma(p,f_\Sigma(p)).
\end{aligned}
}
\tag{10}
\]

**Proof.** A valid full assignment makes every component valid by (8).
Lemma 3 implies each \(\psi_j\). Conversely, if every check holds, the
component constructors yield valid assignments. Independence permits their
union; equal output shapes make the sum legal, and (8) gives full equality.
The single-term case proves completeness for contraction/permutation/reversal/
sum trees. Omitting reversal recovers the previous contraction/permutation/
sum theorem, including its additive extension; then omitting unary sums and
addition gives the original contraction/permutation case.
\(\square\)

## 6. Uniform generation of metadata programs

The theorem yields finite programs, not an enumerated table of ranks.
Metadata instructions have their mathematical meanings: source-field reads,
finite signed partitions and merges, bit XOR, filtering, set operations,
concatenation, equality, subsequence checks, position lookup, finite graph
traversal, bounded loops, and conditionals.

A syntax-directed generator precomputes subtree input sets and emits:

~~~text
GENERATE(Sigma):
    expand source and match terms by their distinct input sets
    for each component j:
        emit signed source replay and canonical reference normalization
        emit postorder computation of M, P, B, K, LF, RF
        emit root-set check and preorder propagation of Q
        emit the input and dot order checks
        emit short-circuit failure before orientation analysis if these checks fail
        emit orientation equalities and pins; compute connected components
        emit the contradictory-pin check and a component-wise bit assignment
        emit postorder construction using (7), (7a), and the unary cases
    emit Phi(p): return conjunction of component checks
    emit f(p):
        if not Phi(p): return NotApplicable
        return union of the constructed component assignments
~~~

Source replay normalizes all recorded port and output bits through the completed
signed source partition. Reduced classes remain recorded. Static component
scopes prevent one source occurrence copied by expansion from becoming a fresh metadata
choice, or logical classes from different components being merged.

Induction over emitted traversals establishes five invariants: replay equals
the signed source network; bottom-up sets equal (3)--(5); propagated requirements
and the order Boolean equal (6); connected components represent exactly the
orientation equalities and their pins; successful construction equals (7),
(7a), and the unary cases. During graph traversal, every discovered vertex
lies in the current component; exhausting its finite frontier discovers
precisely that component. Aggregating pins therefore computes \(\chi\), and
the chosen component bits satisfy all constraints on success.
Loop-prefix invariants give the corresponding statements for variable-length
axis lists. The successful branch has all lookup preconditions by Section 4;
a false order or orientation check bypasses construction. Thus evaluation
implements (9)--(10).

Generation traverses finite syntax, including finite source expansion.
Execution traverses finite trees and metadata collections. Both terminate;
ranks determine loop lengths, not program structure. Expansion can be
exponential in the original expression size; this is not a general
polynomial-time normalization claim. For a single expanded term the orientation
graph has \(\sum_v|P_v|\) vertices and linearly many edges and pins in that
quantity. Its connected-component computation needs no enumeration of reversal
subsets and does not iterate over tensor elements or dimension sizes.

## 7. From declared TEPL information to a generated condition

The symbolic specification is extracted as

\[
\Sigma=E(T)=(\mathcal C,\mathcal G_L,\mathcal G_R,
             \mathcal D,\mathcal R,\mathcal H),
\]

recording named captures, operand trees, operation contracts, source-field
references, and ownership of independent RHS holes. Variable-length input
axis sequences are references evaluated using \(p\), not fixed-rank lists.

Admission matches declarations structurally against Section 1's templates:

- Permutation: a full-input axis group, one permutation field, permuted output,
  and transported operand reads.
- Reversal: a full-input axis group, one unrestricted valid axis-list field,
  unchanged ordered output shape, and operand reads reflecting precisely the
  selected full coordinates by \(n_i-1-y_i\).
- Unary sum: selected reduction axes, the ordered complement as output,
  one reduction-list field, and a sum of the operand over the selected group.
- Contraction: four distinct fields, paired batch/reduction groups, ordered
  free complements, the three output blocks, and reduction of operand products.
- Addition: aligned full input interfaces, their shared ordered shape, and
  the sum of operand values.

Allow consistent renaming, concatenation reassociation, equivalent ordering
of exclusions in complements, and reversal of scalar product factors.
These finite checks exclude additional assertions or fields and do not call
a general semantic-equivalence oracle. Check read-once components, distinct
input sets, and descriptor ownership after the specified source expansion.

A local field adapter \(\rho_v\) maps declaration fields to canonical roles.
It is a bijection because the template fields are distinct and exhaustive.
Source adapters read the same captured values, retaining all sharing.
The product of RHS adapters is a bijection on complete assignments.
Structural induction over templates preserves legality, ordered shapes,
and values; source expansion preserves the same properties. Thus declaration
validity is equivalent to canonical validity, and no possible attribute
assignment is lost in translation.

For a metadata language providing the exact analysis queries, finite-list
operations, bits, and finite graph traversal, emit (6), the component-pin check,
and (9) as an explicit condition \(W_T\). Interpret every
query in its component and occurrence scope. Queries abbreviate the finite
procedures in Section 6, not calls to an attribute-existence oracle.
Structural induction over the
emitted expressions gives

\[
\operatorname{Eval}(W_T,p)=\Phi_{E(T)}(p).
\tag{11}
\]

Combining (10)--(11) with the assignment adapters proves the surface claim:
the specified translation produces a checker that accepts exactly when
valid attributes exist, and a constructor that returns them on success.

The mathematical feasibility program uses the source replay and RHS structural
requirements above. In executable TEPL, users instead write source metadata
restrictions that the compiler proves sufficient. For example:

~~~tepl
(source: dot (inner: dot A B) C) => (dot A (dot B C))
where {
    source.attrs.lhs_batching_dimensions == inner.axes.B.positions
}
~~~

This is a sufficient restriction, not a presentation of the complete feasibility
formula. Labels identify operation occurrences without adding tensor captures.
Source logical-axis identities differ from numeric positions and extents.
Runtime inferred-RHS queries and `axis_domain()` have been removed from the
language's executable path. The generalized mathematical generator still needs
canonical live/root-set analysis, orientation graphs, and component scopes;
those definitions do not claim existing public query support.

An additional defined, terminating condition \(C(p)\) gives mathematical-solver applicability
\(C(p)\land\Phi_\Sigma(p)\). Guaranteeing that every match satisfying \(C\)
is feasible requires the separate implication \(C\Rightarrow\Phi_\Sigma\)
within the source domain. This proof does not decide arbitrary implications,
nor require every equivalent spelling of a generated condition to be
recognized by the current compiler.

## 8. Examples, obstructions, and restrictions

**Unary summation.** For \(A:[M,K]\), \(B:[K,N]\),

\[
\sum_{i,j}(AB)_{ij}
=\sum_k\left(\sum_i A_{ik}\right)\left(\sum_j B_{kj}\right).
\]

The canonical RHS removes \(M,N\) in its two unary sums and contracts \(K\)
at the dot. All three descriptors are inferred together. Conversely,
a source summing an input axis cannot be replaced by permutations alone:
the unwanted axis remains in \(P_{\mathrm{root}}\), failing the set check.

**Distribution and LoRA.** Distribution has components \(\{A,C\}\) and
\(\{B,C\}\); copying the source contraction descriptor supplies a witness.
LoRA, \(X(W+AB)\to XW+(XA)B\), has components \(\{X,W\}\) and \(\{X,A,B\}\).
The first is always realizable; the second is an exact ordered-reassociation
problem. Some generalized-dot matches fail it, despite the scalar identity.

**Output order.** For independent vectors, a transposed outer product has
order right-free then left-free. A single dot without a permutation must
output left-free then right-free. Equal extents do not repair this mismatch.

**Orientation and reduction.** For independent vectors of the same extent,
\(\sum_i X_{n-1-i}Y_i=\sum_i X_iY_{n-1-i}\). The source has one class
with opposite input-port bits. A RHS dot with a reverse node on either operand
can satisfy its paired-coordinate equality; a bare dot cannot. In contrast,
\(\sum_i X_{n-1-i}=\sum_i X_i\) needs no RHS reverse: a singleton input-port
class with no output attachment has no relative orientation to preserve.
These are instances of the same orientation graph, not special-case rules.

Failures have checkable structural witnesses: an unwanted canonical root
axis, an input-order inversion, or a pair of required axes reversing the
dot's batch/left-free/right-free order, or an orientation-equality path joining
opposite pins. Recomputing the analysis validates
the witness; (10) excludes every assignment in that component and full rule.
This is an impossibility claim only on legal in-domain metadata.

The boundaries of Sections 1--11 are:

| Restriction | Reason |
| --- | --- |
| Independent read-once inputs per term | Exact signed wire representation, no reflection cycles, and one-hot separation. |
| Distinct term input sets | Separates terms by independent capture scaling. Same-degree sums need additional reasoning. |
| Extents at least two | Distinguishes partitions and relative signs using distinct endpoints \(0,n-1\). Empty or singleton extents are outside the theorem. |
| Independent unrestricted RHS descriptors | Canonical projection and branch witnesses can be realized without coupled constraints. |
| Exact operation and arithmetic contracts | Early elimination, finite distribution, and reordered reductions preserve meaning. |
| Whole-axis identity/reflection maps | Orientation bits are sufficient and full-domain reflection preserves summation. Slices and split/merge reshapes need a richer representation. |

For example, \(A+\operatorname{transpose}(B)\) cannot generally be realized as
\(\operatorname{transpose}_p(A)+\operatorname{transpose}_p(B)\) with one shared
unknown \(p\): the components separately require identity and swapping.
Independent component feasibility is not sufficient when descriptors share.
Likewise, two terms with the same input set cannot be paired independently
merely by the scaling argument.

This is a semantic boundary, not admission of every operation taking an axis
list: sorting, cumulative sums, FFT, and other axis-parameterized operations
do not match the contracts above. The core theorem excludes general weighted
scalar products; Section 11 supplies
their constructive completeness extension. Broadcast is excluded from Sections 1--11. Section 12 extends the structural
contract with supplied-domain axis creation and metadata weights. The earlier
theorems do not admit subtraction/cancellation,
repeated inputs within a component, other reducers, keepdims, general scalar
functions outside Section 11's scalar contract, broadcast, slice, reshape,
convolution, or extra backend restrictions.

## 9. Current implementation and evidence

The standalone native exact solver covers the original read-once
contraction/permutation specialization. Executable TEPL uses independently
verified symbolic constructors under written source conditions. The generalized
unary-sum, reversal, and additive complete algorithms above are mathematical
extensions requiring integration; the weighted-scalar state procedure
in Section 11 and the metadata-weighted extension of Section 12 are likewise
not integrated. General proofs of some such rules
do not imply complete attribute search.

The frozen [XLA classification](../../../labs/attribute_inferences/xla_rewrite/coverage/data/synthesis_classification.json)
and recorded measurements predate these mathematical extensions. This proof revision
does not reclassify cases or increase a reported completeness-coverage count;
that requires a fresh audit of the full admission conditions and case domains.
In particular, supplied-extent broadcast in Section 12 is an axis-only completion
claim, not inference of its extent fields.

| Component | Existing implementation and remaining work |
| --- | --- |
| Admission and declaration roles | [contract.cc](../../../src/core/attr_synth/ordered/contract.cc) recognizes contraction/permutation contracts. Extend templates, descriptor adapters, and admission for sums, reversal, and additive components. |
| Runtime analysis and construction | [evaluate.cc](../../../src/inference/evaluate.cc) and [axis_analysis.cc](../../../src/rewrite/axis_analysis.cc) implement the original source replay, forced boundaries, orders, and constructor. Generalization requires signed source replay, (3)--(7a), unary instructions, the root-set check, and the orientation graph. |
| Compilation | [compile.cc](../../../src/core/attr_synth/ordered/compile.cc) emits the original schedules. Add component expansion/scopes and the generalized schedules. |
| Source restrictions | [restrictions.cc](../../../src/core/semantics/restrictions.cc) searches a bounded vocabulary and verifies sufficient source restrictions. Complete generalized feasibility analysis remains a separate extension. |

The general semantic verifier already proves some additive/reduction rules;
that does not establish complete attribute search for them. Fully explicit
RHS validation checks the supplied assignment rather than existential
attribute existence.

Finite tests support implementation correspondence, not arbitrary-rank
correctness. A separate small exhaustive descriptor experiment checked the
generalized live-set analysis on 1,224 source/RHS cases, including delayed
reductions and reassociation: all decisions agreed and all 629 constructed
solutions had the source network. This corroborates canonicalization but
does not replace its proof or independently prove the network semantics.

For the reversal extension,
[`check_oriented_axis_synthesis.py`](../scripts/check_oriented_axis_synthesis.py) is a
standalone finite research check. It enumerates all semantically distinct
positional descriptor possibilities for each selected skeleton (including all
four ordered contraction lists; sum and reverse selection lists are represented
by subsets). It compares exhaustive RHS existence with the canonical-set,
order, and orientation constructor. Separately, a direct exact polynomial
tensor evaluator implements reads, products, and finite sums without relying
on signed partition equality, checking both representation exactness on the
enumerated states and every constructed result.

The experiment covers nine shape families, with scalars, one to three inputs,
input ranks up to two, equal/unequal extents two and three, nested unary nodes,
batching, delayed sums, output reordering, and contraction reassociation.
It checked **73,764 distinct target-network/RHS-skeleton pairs**: **11,794**
feasible pairs with correct constructed attributes and **61,970** infeasible
pairs, with no disagreement. Source networks are deduplicated across the
selected source skeletons; these are not XLA-case counts. The experiment tests
single components, not additive admission or arbitrary ranks. Reproduce from
the repository root with:

```sh
python3 research/attribute-synthesis/scripts/check_oriented_axis_synthesis.py
```

This finite check supports the mathematical construction and catches small
counterexamples; it is not a machine-checked proof of Lemmas 1--3, a proof for
all ranks/extents, or a refinement check of the C++ implementation.

Formal refinement remains necessary for a theorem about the actual C++
frontend, symbolic normalizers, metadata queries, generator, and runtime.
The declared operation contracts must also match the intended backend.

Finally, resource/domain failures must remain distinct from infeasibility.
The mathematical programs terminate over finite metadata with sufficient
resources. Machine executions additionally need representable indices,
checked arithmetic, and sufficient memory/work budget. Resource exhaustion,
unsupported schemas, or outside-domain data do not prove that attributes
cannot exist.

## 10. Practical TEPL acceptance and restriction suggestions

The completeness theorem concerns the specified mathematical generator and its
admitted domain. It is a specialized decision-and-construction procedure, not
a restriction on every semantic rule expressible in TEPL. The practical compiler
also uses general candidate construction and independent symbolic verification.
No completeness claim is made for that extended search.

Let \(L(p)\) denote source legality, \(C(p)\) the written rule conditions, and
\(V_f(p)\) mean that the generated metadata program \(f\) is defined and makes
the replacement legal, equally shaped and typed, and equivalent under the
selected arithmetic policy. A semantic rule is accepted only on a proof of

\[
\forall p:\quad L(p)\land C(p)\Rightarrow V_f(p).
\]

Within the completeness domain, proving \(C\Rightarrow\Phi_\Sigma\) under
source legality is sufficient for the theorem's constructor. Domain assumptions
must also be established; they cannot silently become match-time restrictions.
Outside the applicable mathematical domain, direct verification can establish
the same acceptance contract without a complete feasibility decision. Some
singleton and supplied/automatic cases now fall within Section 12's broader
mathematical domain, but its complete algorithm is not integrated. Zero extents,
shared-hole cases, and repeated-capture identities may still be verified by the
general path without a completeness claim. Failure there is Unknown, not an
impossibility certificate. Strict floating-point equivalence is not
inferred from an ideal-arithmetic proof.

The implementation proposes copied descriptors for identical operation trees
and verifies operation, operand, and attribute identity directly. Other candidates
are checked in a fresh context using source legality and the written conditions.
All constructor definedness and legality obligations must be discharged before
materialization. Unresolved obligations cannot become hidden runtime guards.
The old runtime inferred-applicability compilation path has been removed.

If the original rule cannot be proved, `check --suggest-where` searches a bounded
vocabulary of source-group equalities and identity-permutation predicates.
For a proposed conjunction \(R\), fresh synthesis and independent verification
must establish

\[
\forall p:\quad L(p)\land C(p)\land R(p)\Rightarrow V_f(p).
\]

A separate concrete metadata search must provide a legal source witness for
\(L\land C\land R\), including source dtype legality. This witness excludes
vacuous recommendations; it is not evidence for the universal implication.
The current search tries individual predicates and pairs, followed by deletion
checks. It may fail to find a restriction or witness even when one exists. A
recommendation is verified sufficient; neither global minimality nor weakest
applicability is claimed. Suggestions do not accept the original rule or alter
its executable conditions: the user reviews and adds them to source, then
normal compilation checks the resulting rule again.

The mathematical theorem, the practical acceptance contract, and implementation
correspondence are separate claims. Unit, CLI, constructor-corruption, and
serialization tests provide implementation evidence, not a formal refinement
proof of the compiler or verifier.

## 11. Scalar extension by coefficient normalization

### 11.1 Admission

Extend the core with attribute-free, positionwise operations of fixed positive
finite arity whose declared scalar semantics have the form

\[
h(x_1,\ldots,x_k)=q_h\prod_i x_i,
\qquad q_h\in\mathbb Q\setminus\{0\}.
\tag{S}
\]

All operands must have identical ordered shapes; the output retains that
shape. The coefficient is fixed in the declaration, not a tensor input or an
unknown attribute. A structural template check recognizes the rational literal
and aligned operand product, allowing factor reordering and reassociation;
it does not decide equality of arbitrary scalar programs. This one contract
includes negation, nonzero rational scaling, and aligned products of any
positive finite arity.

Retain Section 1's exact arithmetic, extents at least two, read-once components,
matching distinct nonempty input sets, independent axis descriptors, and an
actual separated RHS sum. Source addition may expand by multilinearity;
RHS expansion must not copy unknown descriptors. There is no implicit
broadcasting, standalone constant tensor, zero scaling, repeated capture within
a component, or extra extent/coordinate attribute. Addition keeps its existing
restricted role; arbitrary nonlinear scalar functions are not admitted.

### 11.2 Normalize coefficients once

For a non-additive component \(E\), let
\(c_E=\prod_{v\text{ scalar node}}q_v\). Erase these coefficients to
obtain \(\bar E\). An aligned product has the semantics of a full-batch dot:
all corresponding axes are paired in their original order, with no reduction
axes. A \(k\)-ary product uses a fixed chain of these dots; a unary product
is identity. These are semantic interpretations with **fixed** lists, not
additional RHS holes.

**Lemma 4 (Coefficient normalization).** For every legal assignment,
\(E=c_E\mathcal T_{N_E}\), where \(N_E\) is the canonical signed network
of \(\bar E\) from Section 2. Two legal, equally shaped components over the
same inputs are universally equal iff their coefficients and networks agree.

**Proof.** Structural induction pulls fixed coefficients through finite sums
and products. The full-batch interpretation preserves legality and the wire
property, so Lemma 1 applies to \(N_E\). If networks differ, its one-hot
witness gives weighted values \(c_E\ne0\) and zero. If networks agree,
a satisfying one-hot input gives network value one and exposes the two
coefficients. The converse follows directly from equal representations.
\(\square\)

Thus coefficients are compile-time constants, compared exactly as reduced
rational pairs. They need not be carried through the metadata search. A
negative coefficient changes values; an orientation bit changes coordinates.

### 11.3 Complete construction and theorem

The fixed full-batch constraints prevent direct use of the core's greedy
constructor: a pointwise product must keep its axes until a later sum removes
them. Instead compute, at each original RHS node \(v\), a table
\(\mathscr S_v(p)\) of reachable **unweighted** canonical signed networks,
including ordered shape, with one realizing assignment per entry. A leaf
contributes its identity network. Combine child entries by the declared
coordinate relationships using these finite choices:

| Operation | Choices |
| --- | --- |
| Permutation | All child-axis orders. |
| Reversal or unary sum | All child-axis subsets. |
| Contraction | Equal-extent partial matchings, batch/reduction roles, and batch orders. |
| Scalar operation (S) | The fixed positionwise alignment if all child shapes agree; otherwise none. |

Replay each choice as in Section 2, retaining reduced physical input ports;
scalar nodes use their fixed full-batch interpretation. Canonicalize the
result and store its local descriptor and child witnesses. Sorted selection
lists and a common canonical reduction-pair order suffice, since those list
orders do not change exact-real semantics. Keep every batch/permutation order.

With \(r_j\) the RHS root for component \(j\), define

\[
\Phi^\times_\Sigma(p)=\bigwedge_j
 \left[c_{L_j}=c_{R_j}\ \land
 N_{L_j}(p)\in\mathscr S_{r_j}(p)\right].
\tag{14}
\]

Recover the retained root witnesses as \(f^\times_\Sigma(p)\) on success,
and return NotApplicable on failure. Validity is Section 1's definition for
the enlarged operation set.

**Theorem (Complete constructive synthesis for weighted multilinear scalar
components).** A terminating generator produces metadata programs satisfying,
for every admitted specification and legal in-domain metadata,

\[
\boxed{
\begin{aligned}
\Phi^\times_\Sigma(p)&\iff\exists\beta\;
                 \operatorname{Valid}^\times_\Sigma(p,\beta),\\
\Phi^\times_\Sigma(p)&\Rightarrow
                 \operatorname{Valid}^\times_\Sigma(p,f^\times_\Sigma(p)).
\end{aligned}
}
\tag{15}
\]

**Proof.** Induct on the RHS tree with the invariant that each table contains
exactly its legally realizable unweighted networks and a correct witness for
each. Every inserted entry has legal children and a legal local choice.
Conversely, an arbitrary legal assignment's child networks occur by induction
and its local choice is considered up to irrelevant list order. Retaining
one witness loses no solution: equal child networks have equal ordered shapes
and values after coefficient normalization; their coefficients are fixed by
the subtree, and all parent legality tests depend only on child shapes and the
local choice. Independent holes allow these witnesses to combine.

Lemma 4 therefore makes each conjunct in (14) exactly component feasibility,
with a valid recovered witness. Multilinearity of (S) preserves Section 5's
separation by distinct input sets. Equal root shapes and independent component
holes let the witnesses combine, proving (15).

Each node has finite local choices at finite child ranks; induction gives
finite tables. Ranks are bounded by the input ranks, child bounds at unary
nodes, their sum at dots, and their maximum at aligned scalar nodes. Extents
come from input shapes. A syntax-directed generator emits source replay,
postorder choice/table loops, signed merges, root lookup, and witness recovery.
The table invariant proves correctness of these finite traversals; exact
coefficient multiplication is performed once during generation. Hence both
generation and evaluation terminate uniformly for all finite ranks, without
reading tensor values or enumerating coordinate grids. \(\square\)

The state search may be combinatorial. Without the added scalar nodes,
coefficients are one and the direct core constructor gives the same feasibility
answer. Broadcast, slice, reshape, zero scaling, same-input-set cancellation,
shared holes, and singleton/zero extents remain excluded. A concrete frontend
still needs faithful lowering and arithmetic support; the theorem does not
assert completeness or arbitrary rational-literal support in the current C++
implementation. Source guards require the separate implication proof of
Sections 7 and 10.

### 11.4 Finite validation

[`check_pointwise_axis_synthesis.py`](../scripts/check_pointwise_axis_synthesis.py)
implements the equivalent coefficient-decorated state procedure: each subtree
coefficient is constant across its table, so keeping it in a key gives the
same deduplication and feasibility result as normalizing it out. A separate oracle enumerates concrete
positional descriptors and evaluates exact tensor-entry polynomials without
using signed-state equality. The state generator uses role-labelled partial
matchings and canonical reduction-list orders; the oracle enumerates all four
ordered contraction lists. Every retained witness is checked for legality and
replayed, and state equality is checked against polynomial equality across
selected source and RHS trees.

The extended experiment checked **486,463 target-state/RHS-skeleton pairs** over
nine shape families: **16,515** feasible pairs with validated witnesses and **469,948**
infeasible pairs, with no disagreements. Families include scalars, one to three
inputs, equal/unequal extents two and three, permutations, reversals, sums,
contractions, and aligned weighted products. Focused checks reject implicit
scalar broadcasting and mismatched shapes, retain the product's output axes,
and verify movement of reflection between operands under full summation.
Additional checks cover double negation, inverse rational scalings, moving
negation through sum, unequal arithmetic coefficients, and equivalence between
a weighted ternary product and its nested binary realization. Exact fraction
arithmetic in the independent polynomial evaluator validates negative and
nonintegral coefficients. As before, additive admission is justified by the
proof rather than exercised by this single-component experiment.

```sh
python3 research/attribute-synthesis/scripts/check_pointwise_axis_synthesis.py
```

These finite checks support the derivation, not arbitrary-rank correctness,
additive admission, compiler refinement, or a new XLA coverage count. The former
supplied-extent broadcast experiment is archived in
[`legacy/check_supplied_broadcast_axis_synthesis.py`](../archive/check_supplied_broadcast_axis_synthesis.py)
and is excluded from the active theorem and its validation counts.

## 12. Completeness for metadata-weighted signed wiring

This section generalizes the operation boundary, rather than adding unrelated
proofs for operation names. Only `axis` fields marked `auto` are unknown.
Extents, coordinates, general settings, defaults, and any supplied axis fields
remain fixed programs of source metadata, as in (T). The extension uses
structural states and metadata arithmetic; it does not enumerate tensor
coordinates, tensor entries, or tensor-entry polynomials.

### 12.1 Operation contract and admission

Retain exact-real arithmetic, independently variable captures, and the
read-once/separated-addition assumptions of Sections 1 and 5. Each expanded
component has a nonempty capture set, and distinct components have distinct
capture sets. Keep the original RHS nodes and independent axis-hole identities;
do not expand an RHS node into independent copies of its unknowns. Standalone
constant tensors, repeated captures within a component, and cancellation
between components with the same capture set remain outside this theorem.
Section 12.6 relaxes separated RHS addition for unary linear contexts by
tracking their components jointly; it does not make copied holes independent.

All physical input and newly introduced axes have **positive** extents.
Unlike Sections 1--11, this extension allows singleton physical inputs as well.
Their coordinates are always zero and are normalized away below, rather than
being separated by distinct endpoints. Zero extents remain excluded throughout
this extension.

An admitted operation has a **metadata-weighted signed-wiring contract**:

1. Its operands have finite ordered live axes. It may introduce fresh axes
   with extents determined by supplied metadata or declaration constants.
2. It may identify equal-extent live axes from different operands, with identity
   or whole-axis reflection. Every resulting class contains at most one live
   axis from each operand. Identifications are represented by a forest;
   no additional equation may constrain a coordinate to a reflection fixed
   point. A single class cannot occupy two output positions.
3. It chooses an ordered output sequence of the remaining classes and sums
   every other live class over its **entire** coordinate interval. Fresh output
   axes can be independent of all physical inputs. Surviving extents are not
   resized, and axes are not split or merged as coordinate sequences.
4. Its value is a nonzero rational weight \(\rho_v\), multiplied by the
   product of operand values under that wiring and those full-domain sums.
   The weight is fixed by the declared semantics for the local axis choice,
   supplied metadata, and ordered child shapes. It can depend on selected
   extents, but it is **not another unknown or an inferred attribute**.
5. Legality and the weight depend only on supplied metadata, child ordered
   shapes, and the local axis choice. Legal axis choices are finitely and
   completely enumerable with rank/metadata-length bounds. All metadata
   procedures terminate using axis/list traversals and exact arithmetic,
   without traversing coordinate intervals. Every legal local choice in the
   admitted metadata domain has \(\rho_v\ne0\).

Each local choice must respect every supplied field. Enumerating all legal
ordered axis lists is sufficient; canonical list-order representatives may be
used only when they also preserve supplied-field restrictions. A fresh axis is
fresh even if its extent equals another extent or reads the same metadata
expression. Equality of sizes never identifies logical coordinates.

This contract has the following instances. The table supplies interpretations
of the same contract, not additional operation-specific completeness claims.

| Operation family | Interpretation |
| --- | --- |
| Permutation, reversal, sum, generalized contraction | Existing signed wiring, weight one, with no fresh axes. |
| Aligned weighted scalar products | Fixed full-axis alignment and the declared nonzero rational weight. |
| Axis-insertion broadcast | Retain each input axis once, and introduce axes with an explicitly supplied extent list; infer only the ordered injection of input positions into output positions. |
| Supplied-target broadcast | The target shape is supplied; infer an injection preserving mapped non-unit input extents. A mapped unit input coordinate is fixed at zero; its output axis may be fresh with the supplied target extent. Unoccupied output positions are also fresh. |
| Unit-dimension insertion | Broadcast with declaration-supplied unit extents and axis-position holes. |
| Unit-dimension removal | Select only axes of extent one; removing them equals summing their one-element domains. |
| Keepdims summation | Sum selected axes and introduce fresh unit axes in their output positions. |
| Full-axis mean, with or without keepdims | The same summation wiring, with \(\rho_v=(\prod_{i\in d}n_i)^{-1}\); the empty product is one. |

An injection is stored in input-axis order and need not be increasing unless
the operation's legality requires that. For axis-insertion broadcast, supplied
new extents follow increasing unoccupied output positions, as in TEPL's
`new_extents` contract. A target-shape broadcast may expand a singleton input:
drop its one-element domain and create the mapped output domain with the supplied
size. This is not a resize of a surviving non-unit coordinate. No non-unit
input axis is resized, and no output-size field is inferred.
General slice, reshape, convolution, padding, cumulative sum, sorting, and
other reducers are not admitted merely because all their non-axis fields are
supplied. A restricted instance may enter only through a proved adapter to
this contract; a general affine access map is not signed whole-axis wiring.

### 12.2 Canonical boundary state and exactness

At a node, start with named physical input ports and one named boundary port
for each ordered output position. Temporary internal interfaces are private.
A class of extent one has the unique coordinate zero, and reflection is identity
there. Delete all its ports and relationships from the canonical partition;
its full-domain sum contributes one. Input shapes remain fixed in metadata,
and the state retains the full ordered output shape, so all omitted positions
are still known to carry coordinate zero. A parent reintroduces temporary unit
interface vertices from that shape whenever its local wiring uses them.
Represent every remaining class by its observable ports and relative reflection
bits. These classes have extents at least two. The wire invariant is:

- a class has at most one physical port per capture and one current output port;
- distinct child subtrees have disjoint captures;
- coordinate identifications introduce no reflection cycle.

The contract preserves this invariant. A class with physical input ports uses
the least such port as its zero-bit reference, as in Section 2. A class with
no physical input port but with an output port has exactly one observable port;
normalize that port's bit to zero. Its value is constant along that coordinate.
When a class has neither kind of observable port, remove it and multiply the
state weight by its extent. Thus write the normalized state as

\[
\mathcal W_v=(\mathbf n_v,\Pi_v,\varepsilon_v,\lambda_v),
\qquad \lambda_v\in\mathbb Q\setminus\{0\},
\tag{16}
\]

where \(\mathbf n_v\) is the full ordered output shape and \(\Pi_v\) is the
partition of **non-unit physical input and current output ports only**.
Unit classes and portless classes have already been eliminated. All scalar
weights and eliminated-class factors are absorbed into \(\lambda_v\). The coefficient
multiplies the product of the captures; it is not a tensor-valued sample.

For an explicit denotation, let \(A_I\) be the classes of \(\Pi_v\)
containing a physical non-unit input port. Each such class has a reference
coordinate \(t_a\in[0,d_a)\). Read coordinate \(i_{c,k}(t)\) is zero for
a unit input position and otherwise is the identity/reflection specified by
its port bit. Let \(C_v(t,y)\) impose the recorded identity/reflection equality
for each non-unit output port connected to a class in \(A_I\). Pure output
axes impose no condition on input coordinates. Then

\[
\mathcal T_{\mathcal W_v}(X)[y]
 = \lambda_v
   \sum_{\substack{t_a\in[0,d_a)\ (a\in A_I)\\ C_v(t,y)}}
      \prod_{c\in U_v}X_c[(i_{c,k}(t))_{k=0}^{r_c-1}].
\tag{16a}
\]

The sum specifies the denotation; the algorithm below does not expand it over
coordinate tuples. Input shapes and capture identities are fixed metadata.

**Lemma 5 (Weighted boundary exactness and compositionality).** For a fixed
capture set and legal metadata in the admitted extent domain, two components
are equal for every capture value, with the same ordered output shape, iff
their canonical states (16) agree. Equal child states are interchangeable in
every admitted parent, preserving legality and the resulting state.

**Proof.** Expand the finite products and full-domain sums semantically, as in
(2). First remove unit coordinate classes: every relation on their coordinates
is automatically satisfied, every reflection is identity, and every such sum
has one term. Deleting unit positions gives a bijection on each capture's array
entries and leaves its arbitrary values independently variable. A non-unit
class without a physical input or output port contributes one independent
summation variable that occurs in no tensor read or output constraint. Its sum
is exactly its extent. Remove it and multiply the coefficient by that extent.
An output-only class contributes a free output coordinate on which no read
depends; reflection of that coordinate has no observable effect. The remaining
classes, their relative signs, and the ordered output shape determine every
read and sum. Consequently equal states have equal meanings.

For necessity, suppose the ordered shapes agree. If the observable partitions
or relative signs differ, the separating-port argument of Lemma 1 applies to
all remaining classes containing a physical port, whose extents are at least
two. Any distinction involving an output port and a physical port also has
such an extent. Pure output classes are necessarily singleton blocks of output ports,
with zero canonical bit; there is no further relation or sign to distinguish
between them. Choose endpoint coordinates violating the differing relation,
and one-hot inputs as in Lemma 1. One component has value \(\lambda\ne0\)
at the selected output and the other has value zero. Thus universal equality
requires the same observable signed partition. With that partition fixed,
a satisfying one-hot choice makes all remaining observable summation variables
unique and exposes exactly \(\lambda\), so the weights must agree. Scalar or all-unit captures contribute their single entry, chosen as one.
Positivity of every domain makes these witnesses available and keeps eliminated
factors nonzero.

For composition, rename child output ports privately, add the local signed
forest and fresh-domain vertices, attach the new output ports, and then forget
the private interfaces. Unit coordinates can be forgotten and reconstructed
from shape because their only value is zero. Every remaining component that
loses all observable ports is multiplied into the weight **once per resulting
class**. Previously eliminated
classes can never reconnect: they have no live output interface. The parent's
state is therefore determined entirely by child states and its local choice.
Its legality and declared weight depend only on their shapes and supplied
metadata. Replacing a child by another realization of the same state preserves
both. This also proves the invariant by structural induction. \(\square\)

The multiplier is per **class**, not per created axis: two broadcast-only axes
identified and then summed contribute \(n\), not \(n^2\). Distinct histories
with products \(2\cdot6\) and \(3\cdot4\) both give weight twelve. They must
be identified when the remaining observable state agrees. No origin label for
a forgotten coordinate is semantically observable.

### 12.3 Complete construction

First establish the positive-extent domain; a zero extent is OutsideDomain,
not an impossibility certificate for a general TEPL rule. Evaluate supplied
fields next. Within the admitted domain, if one is undefined or out of range,
no valid completion exists. Otherwise replay each source component with its captured
axis fields to obtain its state (16). At each original RHS node, build a table
of reachable canonical states with one realizing assignment per entry:

~~~text
REACH(v, p):
    if v is capture c:
        return its physical-port identity state, weight one
    for each combination of child-table entries:
        for each legal local axis choice consistent with supplied fields:
            replay its signed wiring and fresh positive domains
            weight := declared_local_weight * product(child weights)
            forget private child-output ports
            multiply weight by each newly closed portless class's extent
            canonicalize the remaining signed boundary state
            retain one local choice and child witness tuple per state
    return the table
~~~

Tables and equality checks include the **whole rational weight**. Unlike
Section 11, coefficients cannot in general be compared separately before
search: reducing a fresh axis can compensate a scalar coefficient. The
algorithm uses no tensor-entry values; the only numerical calculations are
metadata legality, extents, and exact rational weights.

Let \(\mathscr W_{r_j}(p)\) be the root table for component \(j\), and
\(W_{L_j}(p)\) its source state. Define

\[
\Phi^{\mathrm{wire}}_\Sigma(p)
 =\operatorname{Defined}(s_\Sigma(p))\ \land
  \bigwedge_j [W_{L_j}(p)\in\mathscr W_{r_j}(p)].
\tag{17}
\]

Recover root witnesses, combine their independent assignments, and output
only the inferred axis fields. Let \(f^{\mathrm{axis}}_\Sigma(p)\) be that
assignment; preserve the supplied fields through \(D_\Sigma\) from (T).

**Theorem (Complete axis synthesis for metadata-weighted signed wiring).** A
syntax-directed terminating mathematical generator produces the procedures
(17) and \(f^{\mathrm{axis}}_\Sigma\). For every admitted specification and
legal source metadata in its stated domain,

\[
\boxed{\begin{aligned}
\Phi^{\mathrm{wire}}_\Sigma(p)
 &\iff \exists\alpha\;
   \operatorname{Defined}(s_\Sigma(p))\land
   \operatorname{Valid}_\Sigma(p,D_\Sigma(p,\alpha)),\\
\Phi^{\mathrm{wire}}_\Sigma(p)
 &\Rightarrow
   \operatorname{Valid}_\Sigma
     (p,D_\Sigma(p,f^{\mathrm{axis}}_\Sigma(p))).
\end{aligned}}
\tag{18}
\]

The existential variables are exactly the marked, independent axis slots.
The theorem asserts completeness for those slots **conditional on the supplied
fields**, not synthesis of a complete descriptor with unknown non-axis fields.

**Proof.** Induct on each original RHS tree. Leaves have their exact identity
state. Every inserted table entry combines legal witnesses and a legal local
choice, so it is realizable and its replay is exact by Lemma 5. Conversely,
any legal assignment determines child states present by induction. Its local
axis choice is enumerated, with every supplied-field restriction respected.
Equal child states have the same shapes, values, and parent legality by Lemma 5,
so retaining one witness per state loses no possible parent state. Independence
of axis holes allows the chosen witnesses to be combined. Thus each table
contains exactly the realizable states, with correct witnesses.

Lemma 5 makes root membership exactly component feasibility. Every admitted
operation is multilinear in its tensor operands: its weights and domains depend
on metadata, and its reductions are full finite sums. Hence independent capture
scaling gives the same distinct-input-set separation as Section 5. Positive domains and nonzero weights preserve the one-hot witnesses after
unit normalization. Component
witnesses combine without shared unknowns and with the common source shape,
proving (18).

Termination follows from finite trees, finite legal local axis domains, and
finite child tables. A rank bound is computed from child bounds and the bounded
number of fresh axes; a contraction may use at most the sum of child ranks,
while insertion adds its known number and keepdims has the child rank. Replay
and canonicalization traverse only ports, signed edges, axis lists, and table
entries. Removing a closed class uses multiplication by its stored extent;
mean uses a product of selected extents. Neither operation traverses the
interval of that extent. A uniform generator emits these postorder traversals,
metadata checks, exact weight arithmetic, table lookup, and witness recovery.
Its code is independent of the concrete ranks and extents. \(\square\)

Search can be combinatorial in axis choices and structural states. Integer and
rational arithmetic costs depend on bit lengths. The claim is absence of
coordinate-grid enumeration, not polynomial-time synthesis or runtime cost
independent of numerical representation size. Internal weight arithmetic must
be exact; it is not restricted to the `u64`/`i64` storage of emitted attributes.
The existing direct constructor and Section 11's procedure remain specializations;
this theorem does not make the direct constructor complete for fresh-axis nodes.

### 12.4 Compiler boundary and evidence

A declaration is admitted by recognizing or proving the signed-wiring contract
above, not merely by finding `axis` fields in its schema. No general equivalence
oracle for arbitrary operation definitions is assumed. An adapter must cover
**every legal axis assignment in the claimed completion problem**, after its
explicit domain and supplied-field restrictions. Dropping otherwise legal
assignments because their accesses are harder to recognize would invalidate
completeness. A deliberately restricted operation family must state that
restriction as part of the theorem's domain. Supplied `coordinate`
fields may occur only in a recognized realization of identity or full-axis
reflection, or in metadata legality/weight programs consistent with the contract;
arbitrary offsets, windows, and affine mixtures remain outside it.

The structural state-table algorithm and its exactness lemma are mathematical
extensions. The current TEPL compiler retains `AutoField`, preserves supplied
fields, and independently verifies candidate constructors; it does not yet
implement this complete state-table synthesis. Normal compilation must still
prove the written source conditions imply the admission domain, applicability,
and all definedness obligations as in Section 10. The theorem adds no hidden runtime guard and
makes no claim about strict floating-point reassociation.

Coverage must be audited **per case**, recording the actual axis holes, supplied
programs, extent domain, operation adapters, and tree assumptions. An operation
with only explicitly supplied extent/coordinate fields adds no axis-inference
case by itself. No updated XLA completeness count is asserted here.

### 12.5 Finite validation, separate from the proof

[`check_supplied_axis_synthesis.py`](../scripts/check_supplied_axis_synthesis.py) implements
the boundary-state constructor for broadcast, permutations, reversals,
contractions, aligned products, unit-axis removal, sums, means, and keepdims
variants. Its synthesis side uses only signed partitions, local axis choices,
shapes, and exact rational weights. The independent small-shape test oracle
uses concrete positional descriptors and tensor-entry polynomials; that oracle
is not used by the constructor or by the mathematical proof.

Across twelve shape families, **222,642** source-state/RHS-template comparisons
agreed with that oracle: **6,916** feasible and **215,726** infeasible. Every
retained witness was legal and reproduced the independently computed tensor
polynomial. The families include scalar captures, singleton inputs, equal and
unequal extents, insertion and target broadcasts, singleton expansion, keepdims,
means, reflected axes, and nonzero rational coefficients.

Targeted structural-only checks additionally use a supplied extent of
2^63 + 7, with no coordinate enumeration, and a closed-class coefficient larger
than `u64`. They check unit normalization, zero-extent rejection from this
fragment, mean normalization, coefficient compensation, equal products with
different dimension factorizations, and the factor of one shared reduced class
rather than one factor per original broadcast axis.

```sh
python3 research/attribute-synthesis/scripts/check_supplied_axis_synthesis.py
```

These finite checks validate the instantiated procedure on the tested cases.
They are not an arbitrary-rank proof, do not implement all possible contract
adapters or supplied-axis restrictions, and do not establish refinement of the
production synthesizer. Completeness for the stated mathematical contract is
the inductive argument above.

### 12.6 Linear contexts over separated additive components

The separated-RHS restriction can be relaxed for **unary linear contexts**.
For every legal axis choice \(a\), let \(T_a\) be an admitted unary operation
whose metadata, weight, and legality are independent of tensor entries. Its
wiring contract gives

\[
T_a(A+B)=T_a(A)+T_a(B).
\]

This includes permutation, reversal, supplied-extent broadcast, full-axis sum,
keepdims sum, and full-axis mean. The operands must have the same ordered
shape. Retain the read-once and distinct-capture-set conditions on the expanded
components; no cancellation between equal capture sets is admitted.
For this extension, multi-operand non-additive nodes still take single-component
children. Products of additive families require a further composition argument.

**Lemma (Joint component construction).** Equation (18) also holds when these
unary operations surround separated additions, using a joint state table.

**Proof.** At each original RHS node, retain a tuple of component states,
indexed by their distinct capture sets, together with their common output
shape and one compatible original-tree witness. Addition joins the component
families. A unary node enumerates one legal local axis choice and applies
that **same choice to every component**. Consequently its copies after formal
distribution are never independent holes. Distinct original axis slots remain
independent, and supplied fields and original-node legality are preserved.
The other nodes retain Section 12.3's component construction.

Induction shows that each table contains exactly the realizable joint states:
every original assignment is enumerated, and every retained witness is one
consistent assignment to the original tree. Equal tuples are interchangeable
in admitted parents because their component meanings and common shapes agree.
Independent capture scaling separates the distinct capture sets at the root;
Lemma 5 then makes component-state equality necessary and sufficient. Retain
one witness per **joint tuple**, rather than choosing component witnesses
independently. The tables are finite and inspect only axis structure and
metadata. Thus the same feasibility and witness guarantees follow. \(\square\)

In particular, `broadcast_add` is admitted when its supplied extents and
source conditions satisfy the theorem domain. Broadcasting \(X+Y\) uses one
injection for both components. The earlier exclusion was a compositional
restriction of the proof, not an inability to express broadcast semantics.

### 12.7 Checked normalization before axis synthesis

A normalization admits additional surface syntax only if it preserves
semantics, legality, supplied fields, and the completion problem. Concretely,
every original legal axis assignment must have a corresponding normalized
assignment and vice versa, with witness recovery. No new independent copies
of an original hole may be introduced. Under these conditions, applying the
complete procedure to the normalized graph proves (18) for the original
graph as well: transport feasibility and the recovered witness through that
correspondence.

**Unit-only reshape.** Inserting a unit axis is broadcast with supplied extent
one; removing a unit axis is a full sum over its one-element domain. Both
preserve every tensor entry and the order of all non-unit coordinates.
A checked adapter may therefore replace reshapes that only insert/remove
unit axes by these operations. Adapter-generated axis positions are determined
by the supplied shapes and remain fixed, rather than becoming extra holes.
The adapter must be exact for every legal completion in its declared domain;
general split/merge reshape does not follow.

**Scalar reciprocal normalization.** A further restricted normalization applies
to an independent rank-zero capture \(Y\) used only through its reciprocal.
Under exact-real arithmetic on the explicitly stated domain \(Y\ne0\),

\[
X/\operatorname{broadcast}_a(Y)
 =X\operatorname{broadcast}_a(1/Y).
\]

Replace \(1/Y\) on both sides by one virtual scalar input \(Z\), leaving its
broadcast axes and all supplied fields unchanged. The remaining graph must
satisfy the structural and additive assumptions above. This is not a
tensor-value-dependent metadata weight: \(Z\) is a value input.

The substitution ranges over all nonzero real \(Z\). For each fixed legal
axis assignment and fixed values of the other inputs, each output difference
of the normalized graphs is a polynomial in \(Z\). It vanishes for every
nonzero \(Z\) exactly when it is the zero polynomial, hence also vanishes at
zero. Universal equality on the reciprocal's range is therefore equivalent
to the normalized theorem's equality for unrestricted \(Z\). Axis legality
and supplied metadata are unchanged, so feasibility and axis witnesses
transfer in both directions, without enumerating tensor entries.

This admits `catalog_broadcast_divide_broadcast` in that arithmetic domain.
Its scalar broadcast has only the empty input-axis mapping; record this as
a forced axis assignment, not evidence of difficult axis search. The
normalization does not admit arbitrary division, correlated reciprocal inputs,
integer division, or IEEE exceptional-value behavior. The \(Y\ne0\) domain
must be declared and justified; the current XLA rule has no explicit nonzero
condition, so this result alone does not certify its unrestricted value domain.

These closure lemmas extend the mathematical algorithm. They are not covered
by the finite validation in Section 12.5 and do not assert implementation of
the normalizers or joint tables in the production compiler.
