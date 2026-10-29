> Within the supported read-once fragment, tensor equivalence can be reduced to matching logical-axis connections and output ordering. Once these are known, RHS attributes can be constructed directly without searching through possible assignments.

## 2. Problem Definition

### 2.1 Tensor Expressions

입력 텐서 이름의 집합을 다음과 같이 정의한다.

\[
\mathcal X=\{X_1,\ldots,X_n\}
\]

각 입력 텐서 \(X_i\)는 symbolic shape \(s_i\)를 가지며, 실제 shape는 pattern matching 시 결정된다.

텐서 연산의 집합을 \(\mathcal O\)라 정의한다. 각 연산 \(o\in\mathcal O\)는 \(k\)개의 입력과 attribute \(a\)를 가지며, 텐서 표현식은 다음과 같이 정의한다.

\[
\boxed{
T ::= X_i \mid o_a(T_1,\ldots,T_k)
}
\]

### 2.2 Tensor Rewrite

Tensor rewrite는 다음 두 symbolic expression으로 구성된다.

\[
\boxed{L\Rightarrow R}
\]

여기서 LHS \(L\)의 attribute는 주어지고, RHS \(R\)의 attribute는 미정이다.

메타데이터와 미정 attribute를 다음과 같이 정의한다.

\[
\begin{aligned}
s &= (s_1,\ldots,s_n) && \text{Input shapes}\\
\alpha &= \text{LHS attributes}\\
p &= (s,\alpha) && \text{Metadata}\\
\beta &= \text{RHS attributes}
\end{aligned}
\]

\(p\)는 pattern matching 시 결정되는 구체적인 메타데이터이며, LHS의 모든 연산이 유효하다고 가정한다.

\(X=(X_1,\ldots,X_n)\)를 입력 텐서의 실제 값이라고 할 때, 두 표현식의 계산 결과를 다음과 같이 나타낸다.

\[
L_p(X),\qquad R_{p,\beta}(X)
\]

### 2.3 Valid Attribute Assignment

RHS attribute \(\beta\)가 유효하려면 다음 조건을 만족해야 한다.

\[
\boxed{
\begin{aligned}
\operatorname{Valid}(p,\beta)\iff{}&
\operatorname{Legal}_R(p,\beta)\\
&\land \operatorname{Shape}_L(p)
=\operatorname{Shape}_R(p,\beta)\\
&\land \forall X,\quad
L_p(X)=R_{p,\beta}(X)
\end{aligned}
}
\]

여기서 \(X\)는 \(p\)에 지정된 입력 shape를 만족하는 모든 텐서 값을 의미한다.

\(\operatorname{Legal}_R\)는 모든 중간 연산의 유효성을 포함하며, shape는 축 순서를 포함한 벡터이다. Attribute는 텐서 값이 아니라 metadata에만 의존한다. 따라서 존재성의 양화 순서는 \(\exists\beta\,\forall X\)이다.

### 2.4 Attribute Synthesis Problem

우리의 목표는 symbolic rewrite \(L\Rightarrow R\)로부터 두 함수를 생성하는 것이다.

**1. Feasibility**

유효한 RHS attribute가 존재하는 조건을 결정한다.

\[
\boxed{
\operatorname{check}(p)
\iff
\exists\beta:\operatorname{Valid}(p,\beta)
}
\]

**2. Attribute Construction**

조건이 성립하면 유효한 RHS attribute를 생성한다.

\[
\boxed{
\operatorname{check}(p)
\Rightarrow
\operatorname{Valid}(p,\operatorname{build}(p))
}
\]

두 조건은 모든 유효한 LHS 메타데이터 \(p\)에 대해 성립해야 한다.

따라서 본 연구의 목표는 특정 shape에서만 동작하는 해를 구하는 것이 아니라, symbolic rewrite로부터 모든 유효한 메타데이터에 적용할 수 있는 \(\operatorname{check}\)와 \(\operatorname{build}\)를 생성하는 것이다.

### 2.5 Scope and Assumptions

Sections 3–5의 수학적 알고리즘과 완전성 증명은 아래 조건을 공통으로 사용한다. 이후에는 이 조건들을 다시 나열하지 않고, 필요한 성질이 증명에서 어떻게 쓰이는지 설명한다. TEPL이 일반 검증으로 처리하는 더 넓은 범위는 Section 6.4에서 다룬다.

**연산의 의미**

사용하는 연산은 contraction, permutation, unary summation, addition이며, 각 연산의 계약은 다음과 같다.

| 연산 | Attribute, legality, 출력 규칙 |
| --- | --- |
| Permutation | 입력 축 위치의 전체 permutation \(\pi\)를 사용한다. 출력 위치 \(j\)는 입력 위치 \(\pi_j\)에 대응한다. |
| Unary summation | 범위 내의 중복 없는 축 위치 목록 \(d\)를 사용한다. 해당 축을 합산하여 제거하고, 남은 축은 입력의 상대적 순서를 유지한다. |
| Contraction | 네 ordered list \((b_\ell,b_r,k_\ell,k_r)\)를 사용한다. 각 목록은 범위 내이며 중복이 없고, 각 operand에서 batch와 contraction 목록은 서로 겹치지 않는다. 양쪽 batch 목록과 양쪽 contraction 목록은 각각 길이가 같고 대응 축의 크기도 같다. 대응하는 인덱스의 입력 값을 곱하고 contraction 축을 합산한다. 출력은 batch-list 순서, 왼쪽의 선택되지 않은 축 순서, 오른쪽의 선택되지 않은 축 순서로 구성된다. |
| Addition | 동일한 ordered shape를 가진 operand의 원소별 합이다. Attribute와 broadcasting은 없다. |

빈 attribute 목록을 허용한다. 위 조건이 연산의 전체 legality 계약이며, 추가 layout 제약이나 backend별 attribute 제약은 없다. Unary summation은 합산만을 의미하며, 다른 reducer나 keepdims는 포함하지 않는다.

**입력과 계산**

- 입력 rank는 임의의 유한 값이며, rank-zero 텐서도 허용한다. 존재하는 모든 입력 축의 크기는 2 이상이다.
- 각 입력 텐서의 값은 독립적으로 바뀔 수 있으며, 대칭성이나 sparsity 등 값에 대한 추가 가정은 없다.
- 계산은 exact real arithmetic을 사용한다. 유한 합의 순서 변경과 분배법칙을 사용할 수 있지만, strict floating-point 동등성을 보장하지는 않는다.
- Metadata \(p\)는 LHS의 모든 중간 연산을 합법적으로 만드는 shape와 attribute를 제공한다.

**표현식의 구조**

**Read-once component**는 비어 있지 않은 입력 집합의 각 텐서를 정확히 한 번 사용하는 contraction/permutation/unary-summation tree이다. 단일 component에서는 LHS와 RHS가 동일한 입력 집합을 사용한다.

Addition이 있으면 LHS를 선형성과 쌍선형성으로 전개하여 \(L=\sum_t L_t\)로 나타낸다. 실제 RHS는 처음부터 대응하는 항들의 명시적인 합 \(R=\sum_t R_t\)이어야 한다.

1. 각 \(L_t,R_t\)는 read-once component이다.
2. 대응하는 \(L_t,R_t\)는 동일한 입력 집합 \(U_t\)를 사용하며, 서로 다른 항은 서로 다른 입력 집합을 가진다. 단, 입력 집합 간 일부 입력을 공유할 수 있다.

**Attribute의 자유도**

모든 RHS attribute는 서로 독립적인 미지수이며, 위 연산 계약 외의 제약은 없다. 고정된 attribute나 여러 연산이 공유하는 unknown은 제외한다. LHS 전개로 복사된 captured attribute는 원래 값을 공유하며, RHS 전개로 하나의 unknown을 여러 독립적인 unknown으로 바꾸는 것은 허용하지 않는다.

규칙의 구조와 attribute 조건은 합성 전에 문법적으로 검사하고, 입력 shape와 LHS legality는 metadata로 확인한다. Singleton/zero extent, component 안의 반복 입력, 같은 입력 집합을 가진 여러 항, 상수, subtraction/cancellation 및 위 표에 없는 연산은 이 완전성 정리의 대상이 아니다. 범위 밖이라는 사실은 유효한 rewrite가 없다는 뜻이 아니라, 별도의 검증이 필요하다는 뜻이다.

## 3. Axis-Based Representation

텐서 표현식의 의미를 입력 축의 연결 관계와 출력 축의 순서로 나타내는 *Axis-Based Representation*을 정의한다.

### 3.1 Logical Axes

Logical axis는 [NumPy의 einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)에서 같은 인덱스 기호로 표시하는 축들을 하나로 묶은 것으로 이해할 수 있다. 예를 들어 `np.einsum('ik,kj->ij', A, B)`에서 두 입력의 `k`축은 같은 인덱스를 사용한다. 이 글에서는 이러한 축의 묶음을 입력 axis port들의 동치류로 표현한다.

하나의 read-once component의 입력 집합을 \(\mathcal X\)라 하자. 입력 텐서 \(X\)의 \(d\)번째 축을 port \((X,d)\)로 표시하면, 전체 port 집합은 다음과 같다.

\[
P=\{(X,d)\mid X\in\mathcal X,\ 0\le d<\operatorname{rank}(X)\}
\]

같은 인덱스를 사용하도록 직접 또는 간접적으로 연결된 port들을 \(p\sim q\)로 표시한다. 이 동치 관계로 묶인 집합 하나가 logical axis이며, 전체 logical axis 집합은 다음과 같다.

\[
\boxed{\mathcal A=P/{\sim}}
\]

축 크기가 같다는 이유만으로 같은 logical axis가 되지는 않는다. 또한 합산으로 출력에서 사라진 축도 입력 축 사이의 연결을 나타내므로 기록에 남긴다.

### 3.2 Axis Representation

앞서 정의한 logical axis를 이용해, attribute가 주어진 텐서 표현식 \(T\)의 축 연결과 출력 순서를 다음과 같이 나타낸다.

\[
\boxed{N(T)=(P,\sim,F)}
\]

여기서 \(P\)는 입력 axis port 집합, \(\sim\)은 같은 인덱스를 사용하는 port들의 연결 관계, \(F\)는 최종 출력에 남는 logical axis를 순서대로 나열한 것이다. 이를 \(T\)의 *Axis Representation*이라 부른다.

핵심은 \(N(T)\)만으로 어떤 입력 원소를 곱하고 어떤 인덱스를 합산할지 알 수 있다는 것이다. 연결된 입력 축에는 같은 인덱스를 넣어 원소들을 곱한다. \(F\)에 있는 축은 그 순서대로 출력에 남기고, 나머지 logical axis의 가능한 인덱스 조합에 대해 합산한다.

예를 들어, 행렬 곱 \(T=AB\)의 동치 관계 \(\sim_{AB}\)는 다음 세 동치류로 정해진다.

\[
i=\{(A,0)\},\qquad
k=\{(A,1),(B,0)\},\qquad
j=\{(B,1)\}.
\]

출력 축 순서가 \((i,j)\)이므로,

\[
\boxed{
N(T)=
\left(
\underbrace{\{(A,0),(A,1),(B,0),(B,1)\}}_{P},
\ \sim_{AB},
\ \underbrace{(i,j)}_{F}
\right)
}
\]

이다. 출력에 없는 \(k\)를 합산하면 다음과 같다.

\[
C_{ij}=\sum_k A_{ik}B_{kj}.
\]

### 3.3 Semantic Equivalence

두 표현식이 모든 입력에서 같은 결과를 내는 것과, 두 표현식의 축 연결 및 출력 순서가 같은 것은 동치이다.

**Lemma 1 (Axis Exactness).**

동일한 입력 집합을 사용하는 두 합법적인 component \(L,R\)에 대해 다음이 성립한다.

\[
\boxed{
\forall X,\quad L(X)=R(X)
\iff N(L)=N(R)
}
\]

\(N(L)=N(R)\)는 logical axis에 붙인 이름이 아니라, 어떤 입력 port들이 연결되고 각 출력 위치에 어떤 동치류가 놓이는지가 같다는 뜻이다. 출력 shape도 축 순서까지 포함하여 같아야 한다.

증명에는 read-once tree의 다음 성질을 사용한다. 각 동치류에는 적어도 하나의 입력 port가 있고, 같은 입력의 port가 둘 이상 들어가지는 않는다. 또한 하나의 logical axis는 각 노드의 출력에서 최대 한 번만 나타난다.

Contraction은 서로 겹치지 않는 입력 subtree의 축을 연결한다. 따라서 같은 자식에 따로 남은 두 축을 나중에 서로 합칠 수는 없으며, 제거된 축도 다시 나타나지 않는다. 이 성질은 Section 4의 축 분석에서도 사용한다.

**Proof.**

(\(\Leftarrow\)) Axis representation이 같으면, 각 출력 좌표에서 곱할 입력 원소와 합산할 인덱스 조합이 같다. 따라서 Section 3.2에서 설명한 합과 곱이 동일하며, 모든 입력에서 결과가 같다.

(\(\Rightarrow\)) 다음 대우를 보인다.

\[
\boxed{
N(L)\ne N(R)
\ \Rightarrow\
\exists X:\ L(X)\ne R(X)
}
\]

출력 shape가 다르면 이미 동등하지 않으므로, shape가 같은 경우만 생각한다. 각 출력 위치에도 port를 붙여 해당 logical axis와 연결하면, 출력 순서의 차이도 연결의 차이로 나타낼 수 있다.

두 축 표현이 다르면 한쪽에서만 연결된 두 port가 있다. 필요하면 \(L,R\)을 바꾸어, \(L\)에서는 연결되고 \(R\)에서는 분리된 port \(p,q\)를 고른다.

\[
p\sim_L q,\qquad p\not\sim_R q.
\]

\(R\)에서 \(p,q\)가 속한 동치류의 인덱스를 각각 0과 1로, 나머지는 0으로 정한다. 각 입력은 그 좌표에서만 1인 one-hot 텐서 \(X^*\)로 두고, 출력 좌표 \(y\)도 같은 방식으로 정한다. Rank-zero 입력의 값은 1로 둔다.

모든 동치류에 입력 port가 있으므로 \(R\)에서는 정확히 한 항만 1이다. \(L\)에서는 연결된 두 port가 서로 다른 인덱스 0과 1을 요구하므로 모든 항이 0이다.

\[
R(X^*)[y]=1\ne 0=L(X^*)[y].
\]

따라서 대우가 성립한다. \(\square\)

예를 들어 두 벡터의 축을 연결하는 내적과, 각 벡터를 따로 합산한 뒤 곱하는 계산은 다음 one-hot 입력으로 구별된다.

\[
\begin{aligned}
A^*&=(1,0),\qquad B^*=(0,1),\\
L(A^*,B^*)&=\sum_k A^*_kB^*_k
             =1\cdot0+0\cdot1=0,\\
R(A^*,B^*)&=\left(\sum_i A^*_i\right)
             \left(\sum_j B^*_j\right)
             =(1+0)(0+1)=1.
\end{aligned}
\]

## 4. Attribute Synthesis Algorithm

본 절에서는 LHS의 axis representation을 기반으로 RHS attribute를 생성하는 알고리즘을 정의한다. 아래에서 사용하는 logical axis와 최종 출력 순서 \(F\)는 LHS에서 가져온다.

알고리즘은 세 단계로 구성된다: (1) intermediate axes 결정, (2) ordering feasibility 검사, (3) attribute 생성.

### 4.1 Canonical Intermediate Axis Analysis

이 절의 목표는 RHS의 각 노드 \(v\)에서 실제로 남길 logical axis 집합 \(P_v\)를 계산하는 것이다. 이를 위해 먼저 제거하면 안 되는 축의 집합 \(M_v\)를 구한다. \(M_v\)는 보조 집합이며, 여기에 각 연산이 어떤 축을 제거할 수 있는지를 반영하여 \(P_v\)를 결정한다.

먼저 다음을 정의한다.

- \(U_v\): 노드 \(v\)의 subtree에 포함된 입력 텐서 집합
- \(\sigma(a)\): logical axis \(a\)를 사용하는 입력 텐서 집합
- \(F\): LHS의 최종 출력 축 순서
- \(M_v\): 노드 \(v\)에서 반드시 유지해야 하는 축 집합

Mandatory axes는 다음과 같다.

\[
\boxed{
M_v=\left\{
a\in\mathcal A\ \middle|
\sigma(a)\cap U_v\neq\varnothing
\land
(a\in\operatorname{set}(F)\lor\sigma(a)\nsubseteq U_v)
\right\}
}
\]

즉, 현재 subtree에서 사용되는 축(\(\sigma(a)\cap U_v\neq\varnothing\)) 중, 최종 출력에 필요(\(a\in\operatorname{set}(F)\))하거나 subtree 외부의 입력과도 연결되는(\(\sigma(a)\nsubseteq U_v\)) 축은 반드시 유지한다.

\(M_v\)의 축은 모두 남겨야 하므로 \(M_v\subseteq P_v\)이다. 다만 불필요한 축도 현재 연산이 제거할 수 없으면 \(P_v\)에 남는다. 예를 들어 dot은 양쪽이 공유하는 축 중 \(M_v\)에 없는 축을 합산하지만, 한쪽에만 있는 free 축은 그대로 남긴다.

이처럼 필요한 축은 유지하고, 나머지는 제거할 수 있는 가장 이른 지점에서 제거하여 canonical live set \(P_v\)를 정한다. 다음 연산별 규칙을 RHS에 bottom-up으로 적용한다.

입력 텐서 \(X\)의 축을 LHS의 logical axis 이름으로 표시한 순서 목록을 \(I_X\)라 하자. 예를 들어 행렬 곱의 두 입력은 \(I_A=(i,k)\), \(I_B=(k,j)\)이다. Unary 연산의 자식을 \(u\)라 하면, 각 노드의 규칙은 다음과 같다.

| 노드 | 축 집합 \(P_v\) | 의미 |
| --- | --- | --- |
| Input \(X\) | \(\operatorname{set}(I_X)\) | 입력의 모든 축에서 시작한다. |
| Permutation | \(P_u\) | 순서만 바꾸므로 축 집합은 그대로다. |
| Unary summation | \(P_u\cap M_v\) | 자식의 축 중 반드시 필요한 축만 남긴다. |
| Contraction | \(B_v\cup LF_v\cup RF_v\) | Batch와 free 축을 남기고 contraction 축 \(K_v\)를 제거한다. |

Contraction의 왼쪽과 오른쪽 자식을 각각 \(\ell,r\)이라 하면, 표에서 사용한 네 축 집합은 다음과 같다.

\[
\begin{aligned}
B_v &= P_\ell\cap P_r\cap M_v\\
K_v &= (P_\ell\cap P_r)\setminus M_v\\
LF_v &= P_\ell\setminus P_r\\
RF_v &= P_r\setminus P_\ell
\end{aligned}
\]

여기서 \(B_v,K_v,LF_v,RF_v\)는 각각 batch, reduction, left-free, right-free axes이다.

Contraction은 양쪽이 공유하는 completed non-output class를 제거한다. 한쪽에만 있는 class는 불필요하더라도 free axis로 남긴다. Unary summation은 남아 있는 completed non-output class를 제거할 수 있다. 여기서 completed는 해당 class의 모든 supporting input이 현재 subtree에 포함됨을 의미한다.

Bottom-up 귀납으로 \(M_v\subseteq P_v\)이며 \(P_v\)의 모든 class가 \(U_v\) 안에 support를 가짐을 확인할 수 있다. 전체 port 집합 \(P\)와 노드별 live set \(P_v\)는 구분한다.

마지막으로 RHS root에서 다음을 검사한다. 중간에는 불필요한 축이 남을 수 있지만, 최종 출력에는 필요한 축만 남아야 한다.

\[
\boxed{
P_{\mathrm{root}}=\operatorname{set}(F)
}
\]

### 4.2 Canonicalization Lemma

Canonical intermediate axes만 고려해도 유효한 RHS attribute assignment를 놓치지 않음을 보인다.

**Lemma 2 (Canonicalization Preserves Feasibility).**

유효한 RHS attribute assignment \(\beta\)가 존재한다고 하자.

앞에서는 입력의 축 순서를 \(I_X\), 최종 출력의 목표 순서를 \(F\)로 나타냈다. 이제 assignment \(\beta\)에 따른 각 노드 \(v\)의 출력 축 순서를 \(\operatorname{Ord}_v(\beta)\)로 나타낸다. Lemma 1에 의해 유효한 RHS는 LHS와 같은 partition을 가지므로, 중간 축도 LHS의 logical-axis identity로 표시할 수 있다.

그러면 다음을 만족하는 유효한 assignment \(\beta'\)가 존재한다.

\[
\boxed{
\begin{aligned}
&\operatorname{Valid}(p,\beta)\\
&\Rightarrow
\exists\beta':\operatorname{Valid}(p,\beta')
\land\\
&\qquad\forall v,\quad
\operatorname{Ord}_v(\beta')=\operatorname{Ord}_v(\beta)|_{P_v}
\end{aligned}
}
\]

여기서 \(S|_P\)는 sequence \(S\)에서 \(P\)에 포함된 축만 기존 순서를 유지하며 선택한 subsequence를 의미한다.

**예시.** \(A,B\)의 shape가 각각 \([2]\)이고, RHS가 \(\operatorname{sum}(\operatorname{dot}(A,B))\)라고 하자. Dot attribute는 \((b_\ell,b_r,k_\ell,k_r)\), sum attribute는 합산할 축 위치 목록이다.

| Assignment | Dot attribute | Dot 출력 shape | Sum attribute |
| --- | --- | --- | --- |
| \(\beta\) | \(([0],[0],[],[])\) | \([2]\) | \([0]\) |
| \(\beta'\) | \(([],[],[0],[0])\) | \([]\) (scalar) | \([]\) |

\(\beta\)는 공유 축을 batch로 남겼다가 sum에서 제거하고, \(\beta'\)는 dot에서 바로 제거한다. 빈 축 목록의 sum은 값을 그대로 반환하므로, 둘 다 scalar \(A_0B_0+A_1B_1\)을 계산한다.

**Proof.**

원래 해 \(\beta\)에서 \(P_v\)에 없는 축만 제거하고, 남은 축의 순서를 유지하는 해 \(\beta'\)를 구성한다. 필요한 축(\(M_v\))은 원래 해에서도 남아 있어야 한다. 자식에서 부모로 올라가며 다음 두 성질을 함께 보인다.

\[
P_v\subseteq\operatorname{set}(\operatorname{Ord}_v(\beta)),
\qquad
\operatorname{Ord}_v(\beta)|_{P_v}\text{를 합법적인 attribute로 실현할 수 있다.}
\]

Input은 그대로 둔다. Permutation은 자식에 남은 축을 원하는 순서로 재배열하면 된다. Unary summation은 필요한 축(\(P_v=P_u\cap M_v\))만 남긴다. 이 축들은 원래 출력에도 있고, summation은 남는 축의 순서를 바꾸지 않으므로 두 성질이 유지된다.

Contraction에서는 세 가지를 확인한다.

1. **공유 축(\(P_\ell\cap P_r\))은 여기서 연결한다.** 자식에 남긴 공유 축은 원래 해의 양쪽 자식에도 있다. 원래 해에서도 이 둘을 현재 노드에서 연결해야 한다. 그대로 두면 이후에는 같은 자식 안의 두 축이 되어, 나중에 연결할 수 없기 때문이다(Section 3.3).
2. **필요한 공유 축(\(B_v\))은 남기고, 나머지(\(K_v\))는 제거한다.** \(B_v\)는 이후에도 필요하므로 원래 해에서도 batch 축이다. \(K_v\)는 모든 입력 연결을 마쳤고 출력에도 필요하지 않으므로, 원래 batch로 남겼더라도 지금 합산할 수 있다.
3. **Free 축(\(LF_v,RF_v\))은 원래 역할을 유지한다.** \(a\in LF_v\)가 원래 오른쪽 자식의 출력에도 있었다고 가정하자. 양쪽에 이 축을 사용하는 입력이 있으므로 오른쪽에서는 반드시 남겨야 한다(\(a\in M_r\subseteq P_r\)). 이는 \(a\in LF_v=P_\ell\setminus P_r\)와 모순이다. 따라서 이 축은 원래도 left-free이며, right-free도 같은 논리로 성립한다.

따라서 남기는 축(\(P_v=B_v\cup LF_v\cup RF_v\))은 원래 출력에도 있고, batch/free 역할도 같다. \(B_v\)는 원래 batch 순서대로 두고, \(K_v\)는 양쪽에서 같은 순서로 짝지어 합산한다. Free 축은 자식의 순서를 따르므로, 출력은 정확히 \(\operatorname{Ord}_v(\beta)|_{P_v}\)이다.

선택한 축은 각각 한 번 존재하고 짝지은 축의 크기가 같으며, batch와 reduction 목록은 겹치지 않으므로 attribute는 합법적이다. 같은 logical axis끼리만 연결하고 모든 입력 연결이 끝난 뒤에만 축을 제거하므로, LHS의 연결 관계도 보존된다.

Root에서는 반드시 남길 축과 원래 출력 축이 모두 \(\operatorname{set}(F)\)이다. 앞의 포함 관계에 의해

\[
\operatorname{set}(F)\subseteq P_{\mathrm{root}}
\subseteq\operatorname{set}(F)
\]

이므로 \(P_{\mathrm{root}}=\operatorname{set}(F)\)이다. 최종 축 순서도 \(F\) 그대로이며 연결 관계가 같으므로, Lemma 1에 의해 \(\beta'\) 역시 유효하다. \(\square\)

이 정리는 모든 유효한 assignment의 중간 축 집합이 같다는 뜻이 아니다. 유효한 assignment가 있으면 같은 최종 결과를 내는 canonical assignment도 있다는 뜻이다.

### 4.3 Axis Ordering Feasibility

Intermediate axis 집합 \(P_v\)가 결정되어도, 각 연산의 출력 순서 제약을 만족하는지 확인해야 한다.

이를 위해 노드 \(v\)에서 반드시 만족해야 하는 ordered subsequence를 \(Req_v\)로 정의한다.

두 sequence \(A,B\)에 대해 \(A\preceq B\)는 \(A\)가 \(B\)의 subsequence임을, \(\Vert\)는 sequence concatenation을 의미한다.

\(S|_P\)는 \(S\)에서 집합 \(P\)에 속한 축만 원래 순서대로 남기는 filter이다.

Root에서 시작하여 top-down으로 다음과 같이 전달한다.

**Root**

\[
Req_{\mathrm{root}}=F
\]

**Permutation**

\[
Req_u=[]
\]

Permutation은 임의의 축 순서를 구성할 수 있으므로 자식에 ordering requirement를 전달하지 않는다.

**Unary Summation**

\[
Req_u=Req_v
\]

**Contraction**

\[
Req_\ell=Req_v|_{LF_v},
\qquad
Req_r=Req_v|_{RF_v}
\]

Batch 축의 순서 요구는 child에 전달하지 않는다. Batch attribute list는 child의 물리적 축 위치를 임의의 순서로 선택할 수 있으므로, 필요한 batch 순서는 현재 contraction에서 직접 구성할 수 있다. 반면 free 축의 순서는 child에서 상속되므로 해당 subsequence를 전달해야 한다.

Contraction의 출력은 batch, left-free, right-free 순서를 따라야 하므로 다음 조건을 검사한다.

\[
\boxed{
Req_v=
(Req_v|_{B_v})\Vert
(Req_v|_{LF_v})\Vert
(Req_v|_{RF_v})
}
\]

또한 입력 노드에서는 입력의 실제 axis order \(I_v\)에 대해 다음을 검사한다.

\[
\boxed{
Req_v\preceq I_v
}
\]

전체 feasibility condition은 다음과 같이 정의한다.

\[
\boxed{
\begin{aligned}
\operatorname{check}(p)={}&
[P_{\mathrm{root}}=\operatorname{set}(F)]\\
&\land\bigwedge_{\text{input }v}
[Req_v\preceq I_v]\\
&\land\bigwedge_{\text{contraction }v}
\left[
Req_v=
(Req_v|_{B_v})\Vert
(Req_v|_{LF_v})\Vert
(Req_v|_{RF_v})
\right]
\end{aligned}
}
\]

### 4.4 Attribute Construction

\(\operatorname{check}(p)\)가 참이면 RHS를 bottom-up으로 순회하여 실제 attributes를 생성한다. 각 연산의 출력 규칙에 맞춰 \(P_v\)의 축들을 \(Req_v\)를 만족하는 순서로 배치하고, logical axis 이름을 실제 축 위치로 바꾸어 attribute를 만든다.

생성한 assignment에서 노드 \(v\)의 출력 축 순서를 \(\operatorname{Ord}_v\)로 쓰며, 여기서는 assignment 인자를 생략한다. 생성된 순서는 다음 조건을 만족한다.

\[
\boxed{
\operatorname{set}(\operatorname{Ord}_v)=P_v,
\qquad
Req_v\preceq \operatorname{Ord}_v
}
\]

즉, 남기는 축은 정확히 \(P_v\)이고(\(\operatorname{set}(\operatorname{Ord}_v)=P_v\)), 그 순서는 요구된 subsequence를 포함해야 한다(\(Req_v\preceq\operatorname{Ord}_v\)).

**Input**

\[
\operatorname{Ord}_v=I_v
\]

**Permutation**

자식의 축을 재배열하여 \(Req_v\)를 먼저 배치하고, 나머지 축은 child의 기존 순서대로 뒤에 배치한다. Permutation attribute는 이 순서에 대응하는 dimension positions로 생성한다.

**Unary Summation**

\[
\operatorname{Ord}_v=\operatorname{Ord}_u|_{P_v}
\]

제거할 축 \(P_u\setminus P_v\)의 위치를 reduction attribute로 지정한다.

**Contraction**

Batch axes \(B_v\) 중 요구된 축을 \(Req_v|_{B_v}\)의 순서대로 먼저 배치하고, 나머지 batch axes를 고정된 순서로 추가하여 sequence \(b\)를 구성한다.

Reduction axes \(K_v\)도 고정된 순서 \(k\)로 나열한다.

\(\operatorname{pos}(S,A)\)는 목록 \(A\)의 축들을 그 순서대로 \(S\)에서 찾아 위치를 나열한다. 위치는 0부터 세며, 예를 들어 다음과 같다.

\[
\operatorname{pos}((i,k,j),(j,i))=[2,0].
\]

Contraction attribute는 \(\beta_v=(b_\ell,b_r,k_\ell,k_r)\)이다. 앞의 두 목록은 왼쪽·오른쪽 batch 위치, 뒤의 두 목록은 왼쪽·오른쪽 contraction 위치이며, 다음과 같이 생성한다.

\[
\boxed{
\begin{aligned}
b_\ell &= \operatorname{pos}(\operatorname{Ord}_\ell,b),\\
b_r &= \operatorname{pos}(\operatorname{Ord}_r,b),\\
k_\ell &= \operatorname{pos}(\operatorname{Ord}_\ell,k),\\
k_r &= \operatorname{pos}(\operatorname{Ord}_r,k).
\end{aligned}
}
\]

출력 축 순서는 다음과 같다.

\[
\boxed{
\operatorname{Ord}_v=
b\Vert
(\operatorname{Ord}_\ell|_{LF_v})
\Vert
(\operatorname{Ord}_r|_{RF_v})
}
\]

모든 RHS 노드에서 생성된 attributes를 결합하여 최종 assignment를 얻는다.

\[
\boxed{
\beta=\operatorname{build}(p)
}
\]

따라서 알고리즘은 가능한 RHS attribute 조합을 열거하지 않고, canonical axis analysis와 ordering constraints를 통해 직접 attribute를 생성한다.

위 절차는 rewrite의 고정된 트리 구조로부터 \(\operatorname{check}\)와 \(\operatorname{build}\) 프로그램으로 생성할 수 있으며, 생성된 프로그램은 지원 범위 내 임의의 유한 rank를 가진 텐서에 적용할 수 있다.


## 5. Correctness and Completeness

본 절에서는 Section 4의 attribute synthesis 알고리즘이 유효한 RHS attribute의 존재성을 정확하게 판별하고, 존재하는 경우 올바른 attribute를 생성함을 증명한다.

**Theorem 1 (Complete Constructive Attribute Synthesis).**

Section 2.5의 조건을 만족하는 symbolic tensor rewrite와 metadata \(p\)에 대해 다음이 성립한다. 단일 component의 \(\operatorname{check}\)와 \(\operatorname{build}\)는 Section 4의 알고리즘이며, addition을 포함하는 경우에는 Section 5.3에서 설명하듯 component별 검사 결과와 attribute assignment를 결합한다.

**Completeness**

\[
\boxed{
\operatorname{check}(p)
\iff
\exists\beta:\operatorname{Valid}(p,\beta)
}
\]

**Constructive Soundness**

\[
\boxed{
\operatorname{check}(p)
\Rightarrow
\operatorname{Valid}(p,\operatorname{build}(p))
}
\]

**증명 개요.** 먼저 단일 component에서 necessity(Lemma 3)와 sufficiency(Lemma 4)를 보인다. 이후 addition을 포함하는 규칙을 component별로 나누어 같은 결과를 적용한다.

### 5.1 Necessity

**Lemma 3 (Necessity).**

유효한 RHS attribute assignment가 존재한다면, Section 4의 feasibility condition을 반드시 만족한다.

\[
\boxed{
\exists\beta:\operatorname{Valid}(p,\beta)
\Rightarrow \operatorname{check}(p)
}
\]

**Proof.**

유효한 assignment \(\beta\)가 존재한다고 가정하자.

Lemma 2에 의해 각 노드의 intermediate axis 집합이 \(P_v\)인 canonical assignment \(\beta'\)가 존재한다.

이 assignment는 LHS와 동일한 최종 출력을 생성하므로,

\[
P_{\mathrm{root}}=\operatorname{set}(F)
\]

를 만족한다.

Root의 요구 순서 \(F\)에서 시작하여 top-down 귀납을 적용한다. 각 노드의 canonical output이 \(Req_v\)를 subsequence로 포함한다고 가정한다. Contraction의 출력은 batch, left-free, right-free의 세 block으로 이루어지므로 \(Req_v\)도 그 block 순서를 따라야 한다. 각 free subsequence는 해당 child에 존재해야 하며, batch 순서는 child의 순서와 무관하게 선택할 수 있다.

Unary summation은 남는 축의 순서를 보존하므로 child가 같은 요구를 만족해야 한다. Permutation은 child에 순서 요구를 주지 않는다. 따라서 Section 4.3의 전달 규칙은 canonical assignment에서 반드시 성립하는 요구만을 전달하며, 모든 contraction에서 다음 조건이 성립한다.

\[
Req_v=
(Req_v|_{B_v})\Vert
(Req_v|_{LF_v})\Vert
(Req_v|_{RF_v})
\]

입력 노드에서는

\[
Req_v\preceq I_v
\]

가 성립한다.

따라서

\[
\operatorname{check}(p)=\top.
\]

\(\square\)

### 5.2 Sufficiency

**Lemma 4 (Sufficiency).**

Feasibility condition이 성립하면, Section 4의 attribute constructor는 유효한 RHS assignment를 생성한다.

\[
\boxed{
\operatorname{check}(p)
\Rightarrow
\operatorname{Valid}(p,\operatorname{build}(p))
}
\]

**Proof.**

\(\operatorname{check}(p)=\top\)이라고 가정하자.

RHS 트리에 대한 structural induction을 통해 각 노드 \(v\)에서 다음이 성립함을 보인다.

\[
\boxed{
\operatorname{set}(\operatorname{Ord}_v)=P_v
\quad\land\quad
Req_v\preceq \operatorname{Ord}_v
}
\]

Input에서는 실제 입력 축 순서를 사용하므로 성립한다.

Permutation은 필요한 순서를 자유롭게 구성할 수 있으며, unary summation은 지정된 축만 제거하면서 나머지 순서를 유지한다.

Contraction에서는 \(B_v,K_v,LF_v,RF_v\)에 따라 batch 및 reduction attributes를 생성한다. Feasibility condition에 의해 요구되는 축 순서가 contraction의 출력 규칙을 만족하므로 귀납 조건이 유지된다. 각 축이 child에 한 번씩 존재하므로 모든 position lookup이 정의된다. 대응 축은 같은 LHS class에 속하여 크기가 같고, batch/reduction 집합은 서로 겹치지 않으므로 모든 attribute가 합법적이다.

축 집합과 순서 외에, 다음 partition invariant를 함께 유지한다.

> 노드 \(v\)까지 구성한 port partition은 LHS partition을 \(U_v\)의 입력 port들에 제한한 것과 같다. 출력에서 제거된 class도 이 partition에 포함한다.

Input에서는 한 입력의 서로 다른 축이 같은 class에 속하지 않으므로 성립한다. Permutation과 unary summation은 port를 병합하지 않아 partition을 보존한다. Contraction의 양쪽 subtree에 support가 있는 LHS class는 각 child에서 mandatory이므로 양쪽 canonical live set에 존재한다. 현재 contraction은 이 두 occurrence를 연결하고, 서로 다른 LHS class는 연결하지 않는다. 따라서 두 child의 partition을 합친 결과는 LHS partition을 현재 subtree에 제한한 것과 같다.

Class를 제거할 때에는 최종 출력에 필요하지 않고 모든 support가 이미 subtree 안에 있다. Partition invariant에 의해 그 supporting port들이 연결되어 있으므로 이후에 필요한 연결을 잃지 않는다. 따라서 root의 전체 port partition은 LHS와 같다.

Root에서는

\[
P_{\mathrm{root}}=\operatorname{set}(F),
\qquad
F\preceq \operatorname{Ord}_{\mathrm{root}}
\]

이므로,

\[
\operatorname{Ord}_{\mathrm{root}}=F.
\]

결국 RHS와 LHS의 axis representation이 일치한다.

\[
N(L)=N(R)
\]

Lemma 1에 의해 두 표현식은 모든 입력에 대해 동일한 결과를 생성한다. 또한 생성된 모든 attributes는 합법적이므로,

\[
\operatorname{Valid}(p,\operatorname{build}(p)).
\]

\(\square\)

### 5.3 Extension to Addition

Section 2.5에서 정한 additive component에 대해 전체 동등성을 항별 동등성으로 나눌 수 있음을 보인다.

\[
L=\sum_{t\in J}L_t,
\qquad
R=\sum_{t\in J}R_t
\]

LHS 전개는 captured attribute의 값을 유지하고, 같은 shape의 summand로 operand를 대체한다. 따라서 각 연산의 legality와 출력 shape를 보존한다.

각 입력 텐서 \(X_i\)에 독립적인 scalar \(z_i\)를 곱하면, multilinearity에 의해

\[
L_t((z_iX_i)_i)
=
\left(\prod_{X_i\in U_t}z_i\right)L_t(X)
\]

가 성립한다.

RHS에도 같은 scaling 식이 성립한다. 임의의 입력 값 \(X\)와 출력 좌표를 고정하고 \(z_i\)에 관한 다항식으로 비교하면, 서로 다른 \(U_t\)는 서로 다른 monomial을 생성한다. 따라서 합법적이고 같은 출력 shape를 가진 표현식에 대해 polynomial coefficient의 유일성으로 다음이 성립한다.

\[
\boxed{
\forall X,\ L(X)=R(X)
\iff
\forall t\in J,\ \forall X,\ L_t(X)=R_t(X)
}
\]

따라서 각 component에 Section 4의 알고리즘을 독립적으로 적용할 수 있다.

전체 feasibility condition은

\[
\boxed{
\operatorname{check}(p)
=
\bigwedge_{t\in J}\operatorname{check}_t(p)
}
\]

이며, 전체 attribute assignment는 각 component의 결과를 결합하여 구성한다.

\[
\boxed{
\operatorname{build}(p)
=
\bigcup_{t\in J}\operatorname{build}_t(p)
}
\]

여기서 각 component는 서로 겹치지 않는 RHS attribute identifier를 사용하므로 assignment의 합집합에 충돌이 없다. 성공한 각 component의 출력 shape는 해당 LHS component의 shape이며, LHS 전개가 shape를 보존하므로 모두 전체 LHS의 shape와 같다. 따라서 RHS addition도 합법적이다.

Axis partition은 component마다 별도로 구성한다. 같은 입력이 여러 항에 나타나더라도 addition은 출력 좌표를 정렬할 뿐, 서로 다른 항의 입력 port를 하나의 class로 병합하지 않는다.

이 과정에서 Section 2.5의 구조와 attribute 조건이 필요한 이유도 드러난다. 입력 집합이 같은 두 항은 scaling으로 분리되지 않는다. 또한 독립적인 정사각 행렬 \(A,B\)에 대해 \(A+\operatorname{transpose}(B)\)를 \(\operatorname{transpose}_\pi(A)+\operatorname{transpose}_\pi(B)\)로 만들려면, 첫 항은 identity를, 두 번째 항은 swap을 요구한다. 하나의 unknown을 공유하면 항별 해를 합칠 수 없는 예이다.

Lemma 3과 Lemma 4는 단일 component의 필요충분성과 constructor의 올바름을 보인다. 위의 항별 동등성과 assignment 결합을 적용하면 addition을 포함하는 전체 규칙에도 같은 보장이 성립한다. 따라서 Theorem 1의 증명이 완성된다. \(\square\)

## 6. Application to TEPL

앞 절의 정리는 명시한 수학적 합성 알고리즘에 대한 것이다. TEPL에 적용할 때에는 연산 선언을 이 모델로 옮기는 과정, 규칙의 조건 아래에서 constructor를 검증하는 과정, 그리고 현재 구현 범위를 구분해야 한다.

### 6.1 From Operation Declarations to the Model

TEPL의 연산 선언은 attribute schema, 축의 연결과 출력 순서를 나타내는 axes, 원소 값을 나타내는 semantics를 제공한다. 합성기는 연산 이름만으로 contraction이나 permutation이라고 판단하는 대신, 선언이 Section 2.5의 계약과 일치하는지 구조적으로 확인한다.

예를 들어 contraction에서는 선언의 네 field, axis binding, 출력 순서 및 값의 계산이 Section 2.5의 contraction 계약에 대응하는지 확인한다. 선언에 추가 제약이 있는지도 함께 검사한다.

선언 field와 표준 역할 사이의 adapter를 \(\rho_v\)라 하자. 허용된 선언에서는 field가 서로 다르고 빠짐없이 대응하므로, RHS 전체 adapter는 선언상의 assignment와 표준 모델의 assignment 사이의 전단사이다. LHS adapter는 captured field의 값과 공유 관계를 그대로 읽는다.

각 연산 계약에 대한 structural induction으로 이 변환이 legality, ordered shape 및 tensor value를 보존함을 보일 수 있다. 따라서 모델에 들어오는 선언은 attribute 해를 잃거나 추가하지 않는다. 지원 템플릿의 구조적 인식은 유한 검사이며, 임의의 연산 선언이 의미적으로 같은지를 결정하는 일반 oracle을 가정하지 않는다.

### 6.2 Verification and Condition Suggestions

사용자가 작성한 where 조건을 \(C(p)\), RHS attribute를 생성하는 프로그램을 \(f\)라 하자. TEPL은 합법적인 모든 source metadata에서 \(C(p)\)만으로 생성한 RHS의 유효성을 보장할 수 있는지 검증한다.

\[
\boxed{
\forall p:\quad
\operatorname{Legal}_L(p)\land C(p)
\Rightarrow \operatorname{Valid}(p,f(p)).
}
\]

여기서 \(\operatorname{Legal}_L\)은 LHS의 legality이다. 이 절의 metadata와 Valid에는 dtype 및 선택한 arithmetic policy도 반영하며, \(f(p)\)가 정의되지 않으면 유효하지 않은 것으로 본다.

완전성 정리의 가정과 TEPL의 dtype·arithmetic 조건을 만족하는 경우에는, source legality 아래에서 \(C(p)\Rightarrow\operatorname{check}(p)\)를 보이면 된다. 더 넓은 연산에서는 생성한 RHS의 유효성을 직접 검증한다(Section 6.4).

충분성을 증명하지 못하면 규칙을 거절한다. 이때 `--suggest-where`를 요청하면, 기존 조건에 더할 조건 \(G(p)\)를 찾아 새로운 합성과 독립 검증으로 다음을 확인한다.

\[
\boxed{
\forall p:\quad
\operatorname{Legal}_L(p)\land C(p)\land G(p)
\Rightarrow \operatorname{Valid}(p,f(p)).
}
\]

추천 조건은 실제로 만족 가능한지도 구체적인 metadata로 확인한다. 검색은 제한적이므로 항상 조건을 찾거나 가장 약한 조건을 보장하지는 않는다. 사용자가 추천 조건을 where에 추가하면 다시 검증하며, 컴파일러가 조건을 몰래 추가하거나 추천만으로 원래 규칙을 수락하지는 않는다. 검증된 프로그램은 match에서 얻은 metadata로 RHS attribute를 계산한다.

### 6.3 Example: Reassociation with a Suggested Condition

다음은 `examples/semantic/reassociate.tepl`에 작성한 where 조건 없는 reassociation 규칙이다. TensorLang의 import와 use 선언은 생략한다.

~~~tepl
rule reassociate {
    (source: dot (inner: dot A B) C)
    => (dot A (dot B C))
}
~~~

이 상태에서는 일반적인 dot의 reassociation을 증명하지 못해 규칙이 거절된다. 다음 명령으로 충분조건을 추천받을 수 있다.

~~~sh
tepl check examples/semantic/reassociate.tepl \
  --semantic-policy=algebraic --suggest-where --rule reassociate
~~~

추천된 규칙의 where 절은 다음과 같다.

~~~tepl
where {
    source.attrs.lhs_batching_dimensions == inner.axes.B.positions
}
~~~

`source`와 `inner`는 두 dot 연산에 붙인 이름이다. `inner.axes.B`의 `B`는 입력 텐서가 아니라 batch 축 그룹을 뜻한다. 추천 조건은 바깥 dot의 왼쪽 batch 축 목록이 안쪽 dot의 batch 출력 축 목록과 같도록 요구한다.

앞 절의 기호로는 원래 조건이 \(C(p)=\top\)이고, 추천된 조건이 \(G(p)\)이다. 이 where 절을 원래 규칙에 추가하고 다시 검사하면, TEPL은 RHS의 두 dot에 생략된 attribute를 생성하고 변환의 유효성을 검증한다. 추천만 받은 상태에서는 원래 규칙이 여전히 거절되며, 추천은 필요충분조건이 아닌 검증된 충분조건이다.

### 6.4 Broader Operators with Soundness

TEPL은 실용성을 위해 broadcast, slice, convolution뿐 아니라 reverse(축 내부의 원소 순서 뒤집기), negate(부호 반전), subtract(뺄셈), zero_like(같은 shape의 영 텐서 생성) 등 완전성 정리 밖의 연산도 지원한다.

이 경우 수락한 규칙의 정확성(soundness)은 보장하지만, 유효한 attribute가 존재할 때 반드시 찾아내는 완전성(completeness)은 보장하지 않는다. 합성이나 검증에 실패해도 유효한 RHS가 없다는 뜻은 아니다.

이 연산들은 좌표 변환이나 음수 계수, 항의 상쇄 등을 다루므로 앞 절의 축 분석만으로 처리하기 어렵다. TEPL은 RHS attribute 후보를 생성한 뒤, 독립적인 검증기로 작성된 where 조건 아래에서 RHS의 legality, 출력 shape 및 값의 동등성을 확인한다. 예를 들어 연속된 두 slice를 하나로 합칠 때는 시작 위치와 stride를 계산하고, 합성된 slice가 범위 안에서 원래와 같은 원소를 읽는지 검증한다.

현재 TEPL의 완전성 기반 합성은 contraction/permutation을 중심으로 구현되어 있으며, 본 정리의 전체 범위로 확장하는 작업은 남아 있다.
