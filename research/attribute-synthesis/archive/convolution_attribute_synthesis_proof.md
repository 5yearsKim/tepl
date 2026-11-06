# Complete synthesis of a valid convolution replacement

> Historical research material. Implementation references and reproduction
> instructions describe the earlier approach. See the
> [current synthesis design](../../../design/attribute-synthesis.md) and
> [research index](../README.md) for present scope.

**Research proof draft, October 22, 2026.** This gives an exact feasibility
predicate and constructor for replacing an admitted bilinear affine-access
expression with one valid convolution. It extends synthesis to spatial
coordinate arithmetic, rather than adding a pointwise wrapper. It is a
restricted mathematical theorem, not completeness for arbitrary convolution
graphs or a verification of the current compiler. A concrete source subfamily
is convolution with spatial slices before and/or after it.

## 1. Domain and admitted expressions

Fix exact-real arithmetic and arbitrary finite spatial rank \(r\ge1\).
Independent captures have shapes

\[
X:[N,I_1,\ldots,I_r,C],\qquad
W:[K_1,\ldots,K_r,C,F].
\]

Every listed dimension is positive. Source output shape is
\([N,O_1,\ldots,O_r,F]\), with every \(O_j>0\). Unlike the ordered-axis
theorem, this theorem permits singleton dimensions. Zero channel or kernel
domains and empty outputs are excluded because they can erase all access
information. Dtypes are fixed and compatible with the chosen exact semantics.

Source metadata determines integer vector \(a\) and matrices \(A,B\), and
the admitted source has exactly the meaning

\[
L[n,o,f]=\sum_{k\in\prod_i[0,K_i)}\sum_{c=0}^{C-1}
  X[n,g(o,k),c]W[k,c,f],\qquad g(o,k)=a+Ao+Bk.
\tag{1}
\]

All source accesses are in bounds. Batch, channel, and feature coordinates
are as displayed: the kernel read is the original independent capture at
\((k,c,f)\), without slicing, duplication, or a changed reduction domain.
The coefficients are computed from known source metadata, not RHS unknowns.

The fixed RHS is one valid convolution of the *same* \(X,W\), with independent
positive-integer stride and dilation vectors \(s,d\):

\[
R_{s,d}[n,o,f]=\sum_{k,c}X[n,s\odot o+d\odot k,c]W[k,c,f].
\tag{2}
\]

Its only attribute legality and spatial shape conditions are

\[
e_j=I_j-1-d_j(K_j-1)\ge0,\qquad
O'_j=\left\lfloor e_j/s_j\right\rfloor+1.
\tag{3}
\]

This is the fixed-layout, positive-stride/dilation, no-padding contract in
the [semantic dialect](../../../tests/fixtures/semantic/dialects/tensor.tepl). No groups, kernel
reversal, extra backend bounds, offsets, or shared stride/dilation restrictions
are synthesized. Admission requires a syntax-directed derivation of (1) from
the source operations; it must not assume an arbitrary semantic-equivalence
oracle. Once (1) is obtained, the theorem applies to any integer coefficients
whose source accesses are legal.

## 2. Equality forces the coordinate map

**Lemma 1 (Bilinear access separation).** For legal candidates of the same
output shape, equality for all independent \(X,W\) holds if and only if

\[
g(o,k)=s\odot o+d\odot k
\quad\text{for every output }o\text{ and kernel coordinate }k.
\tag{4}
\]

**Proof.** Equal maps give identical sums. Conversely, suppose the maps differ
at \((o,k)\). Choose any channel \(c\), batch \(n\), and feature \(f\),
which exist by positivity. Set only \(W[k,c,f]=1\), with all other kernel
entries zero. Set only the input entry at the source's selected coordinate
to one. The source value at \((n,o,f)\) is one and the RHS value is zero.
This contradicts equality for every input. Kernel coordinates distinguish
summands, so collisions between different input accesses do not affect the
argument. \(\square\)

**Lemma 2 (Exact finite-box coefficient constraints).** Equation (4) is
equivalent to the following conditions:

\[
\begin{aligned}
&a_j=0 &&\text{for every }j,\\
&A_{ji}=s_j\delta_{ji} &&\text{whenever }O_i>1,\\
&B_{ji}=d_j\delta_{ji} &&\text{whenever }K_i>1.
\end{aligned}
\tag{5}
\]

**Proof.** Evaluate (4) at zero to recover the constant term. For each
coordinate whose domain has size at least two, evaluate at its unit vector
and subtract the zero evaluation. This recovers its coefficient column.
Coordinates of singleton domains are always zero and impose no coefficient
constraint. Conversely, (5) makes the affine maps equal at every point of
the box. \(\square\)

Ignoring the singleton qualification would make the checker incomplete:
different coefficients can represent the same map on a singleton domain.

## 3. An existential-free checker and constructor

First check \(a=0\) and the off-diagonal clauses of (5):

\[
 i\ne j\Longrightarrow
 (O_i>1\Rightarrow A_{ji}=0)\land(K_i>1\Rightarrow B_{ji}=0).
\tag{6}
\]

Then process each spatial dimension independently:

\[
d_j^*=\begin{cases}B_{jj}&K_j>1,\\1&K_j=1,\end{cases}\qquad
e_j^*=I_j-1-d_j^*(K_j-1).
\tag{7}
\]

Reject if \(d_j^*\le0\) or \(e_j^*<0\). Otherwise set

\[
s_j^*=\begin{cases}A_{jj}&O_j>1,\\e_j^*+1&O_j=1.\end{cases}
\tag{8}
\]

Reject if \(s_j^*\le0\) or

\[
O_j\ne\left\lfloor e_j^*/s_j^*\right\rfloor+1.
\tag{9}
\]

Let \(\Phi_{\mathrm{conv}}(p)\) be the success Boolean of these checks.
The constructor returns \((s^*,d^*)\) only after success. Positivity is checked
before division; all operations have mathematical integer semantics.

**Theorem (Complete single-convolution replacement synthesis).** For every
admitted legal source configuration,

\[
\Phi_{\mathrm{conv}}(p)
\iff\exists s,d:\operatorname{Legal}(R_{s,d})\land
\operatorname{Shape}(R_{s,d})=\operatorname{Shape}(L)\land
\forall X,W:\ R_{s,d}=L.
\]

On success, the constructor returns such attributes. The generated checker
and constructor terminate uniformly over arbitrary finite spatial ranks.

**Proof, sufficiency.** Conditions (6)--(8) establish every active coefficient
equation in (5). The positivity and shape checks establish (3) and the required
ordered output shape. Thus the RHS is legal and Lemma 1 proves equality.

**Proof, necessity.** Let \((s,d)\) be any valid assignment. Lemmas 1--2 force
(6), force \(d_j=B_{jj}\) when \(K_j>1\), and force \(s_j=A_{jj}\) when
\(O_j>1\). When \(K_j=1\), dilation affects neither accesses nor shape, so
replacing it by one preserves validity. The residual \(e_j\) is consequently
the same for the candidate and (7). When \(O_j=1\), legality and shape require
only \(s_j>e_j\); replacing that stride by \(e_j+1\) preserves the sole
output access and shape. Hence (7)--(8) construct a valid representative
whenever any assignment exists, so no check rejects a feasible instance.
The algorithms use finite loops over coefficient matrices and integer
operations, proving termination and rank uniformity. \(\square\)

The result is existence-complete, not uniqueness: singleton domains can admit
infinitely many equivalent positive strides or dilations. No unbounded
attribute search is needed because the proof supplies canonical representatives.

## 4. A concrete source fragment and examples

Consider a spatial input slice with starts \(u\), strides \(t\), and legal
positive sizes; a valid convolution with captured strides \(s^0\) and
dilations \(d^0\); then a spatial output slice with starts \(v\), strides
\(q\), and legal positive sizes \(O\). Both slices preserve the full batch,
channel, and feature dimensions as applicable. Either slice may be absent.

Substitution of the declared access maps gives (1), with

\[
a=u+t\odot s^0\odot v,\qquad
A=\operatorname{diag}(t\odot s^0\odot q),\qquad
B=\operatorname{diag}(t\odot d^0).
\tag{10}
\]

Thus eliminating these slices by synthesizing one convolution is covered.
This admission and extraction uses operation contracts and finite metadata
arithmetic. The source's slice bounds and convolution legality establish
the legality of (1); arbitrary source slices are not assumed removable.

For one spatial dimension, let input length be 8, kernel length 3, and source
convolution stride/dilation both be 1. The full source output length is 6.

| Source output selection | Result of complete synthesis |
| --- | --- |
| Positions 0,2,4 | Stride 2, dilation 1; output length 3. |
| Positions 0,1 only | Infeasible: active coefficients force stride/dilation 1, which yield length 6, not 2. |
| Positions 1,3,5 | Infeasible: offset 1 conflicts with a zero-offset valid convolution. |
| Position 0 only | Feasible: dilation 1 and canonical stride 6 yield one output. The source stride coefficient is unobservable on this output domain. |

The last example is why directly copying or matching all coefficient values
is insufficient for completeness.

## 5. Generated programs and implementation boundary

~~~text
GENERATE_CONVOLUTION(source, rhs):
    check_source_templates_and_single_valid_convolution_rhs()
    emit extraction of I, K, O, a, A, B from source metadata
    emit CHECK_AND_BUILD(p):
        validate source metadata/domain
        check zero offset and active off-diagonal coefficients
        for spatial dimension j:
            d[j] = B[j,j] if K[j] > 1 else 1
            require d[j] > 0
            e = I[j] - 1 - d[j] * (K[j] - 1)
            require e >= 0
            s[j] = A[j,j] if O[j] > 1 else e + 1
            require s[j] > 0
            require O[j] == floor(e / s[j]) + 1
        return Solved(s,d)
~~~

Failed requirements produce NotApplicable for legal in-domain instances.
Source-domain failures and resource exhaustion remain separate statuses.
The Boolean projection supplies the generated where predicate; successful
construction supplies the attributes. No tensor values are evaluated.

The general compiler's
[index_inference.cc](../../../src/core/attr_synth/legacy/index_inference.cc)
already matches some coordinate coefficients and proposes constructors, but
that procedure explicitly rejects some ambiguous, inverse, or coupled cases.
This theorem does not establish that procedure's completeness. Implementation
would add source-template extraction, active-coordinate handling, the canonical
constructor, shape checks, and the corresponding metadata-program lowering.

Independent validation is available in
[validate_convolution_completeness.py](validate_convolution_completeness.py).
It compares the proposed checker with exact bilinear monomial signatures from
enumerated legal RHS attributes, including singleton and off-diagonal cases.
The run checked 7,580 legal 1D sources (872 feasible) and 3,721 legal 2D sources
(441 feasible), with no discrepancies. These finite checks support the proof
but do not establish arbitrary-rank correctness. They are a standalone research
check, not an additional default test-suite target.

Padding, zero domains, grouped convolution, altered kernel accesses, nonlinear
indexing, chains with several independently variable kernels, or synthesis of
attributes on multiple interacting RHS operations require new characterizations.
The additive product theorem can combine this solver with other exact component
solvers only after proving the same component-separation and attribute-independence
premises; neither theorem automatically covers their unrestricted combination.

