> 본 연구는 axis-structural multilinear 계약을 만족하는 rewrite에서, 주어진 메타데이터를 보존하면서 RHS의 `axis` attribute만 합성하는 문제를 다룬다. 값의 계산은 각 독립적인 operand에 대해 선형이며, 좌표 관계는 전체 축의 연결·방향·출력 순서·축소로 표현한다. Addition은 별도의 합성 규칙으로 다룬다.

**문서 구성.** Section 2는 axis-only 합성 문제와 본문 정리의 범위를 정의한다. Sections 3–5는 \(N(T)=(P,\sim,F)\)를 이용한 중간 축의 직접 구성과 완전성을 증명한다. 구체적인 구성 규칙은 Appendix A에, 직접 구성의 확장은 Appendix B에, 확장 표현과 지원 경계는 Appendix C에 정리한다.

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

여기서 LHS \(L\)의 attribute는 주어진다. RHS \(R\)에서는 `auto`로 표시된 적격 `axis` field만 미정이며, 나머지 field는 명시한 식 또는 schema의 기본값으로 주어진다. 명시적으로 주어진 `axis` field도 합성 중 변경하지 않는다.

메타데이터와 미정 attribute를 다음과 같이 정의한다.

\[
\begin{aligned}
s &= (s_1,\ldots,s_n) && \text{Input shapes}\\
\alpha &= \text{LHS attributes}\\
p &= (s,\alpha) && \text{Metadata}\\
\beta &= \text{Completed RHS attributes}
\end{aligned}
\]

\(p\)는 pattern matching 시 결정되는 구체적인 메타데이터이며, LHS의 모든 연산이 유효하다고 가정한다.

\(X=(X_1,\ldots,X_n)\)를 입력 텐서의 실제 값이라고 할 때, 두 표현식의 계산 결과를 다음과 같이 나타낸다.

\[
L_p(X),\qquad R_{p,\beta}(X)
\]

#### Attribute의 역할과 합성 경계

Attribute의 역할은 정수 저장 타입과 구분한다.

| 선언 | 의미 | 저장 타입 | `auto` 허용 |
| --- | --- | --- | --- |
| `axis`, `axis[]` | 텐서의 축을 선택·정렬·대응시키는 축 위치 | `u64` | 허용 |
| `extent`, `extent[]` | 축의 크기 또는 명시적으로 주어진 목표 shape | `u64` | 허용하지 않음 |
| `coordinate`, `coordinate[]` | 좌표의 시작점, offset, stride, padding 양 등의 메타데이터 | `i64` | 허용하지 않음 |

예를 들어 축 위치 1은 `axis`, 그 축의 크기 8은 `extent`, 그 축 안에서의 시작 위치 2는 `coordinate`에 해당한다. `axis`는 숫자로 표현되는 attribute의 역할이며, 연산 선언의 `axes`는 논리적 축 구조를 나타낸다. 이전의 공개 타입 `index`는 사용하지 않는다.

`coordinate`는 메타데이터 parameter를 뜻하며, gather 등의 index tensor에 담긴 값을 뜻하지 않는다. 또한 `i64`로 저장한다는 사실만으로 음수 stride나 padding이 합법적인 것은 아니다. 각 연산의 계약이 허용 범위를 정한다.

**현재 구현의 제한.** `auto`는 non-optional `axis` 또는 `axis[]` field에만 허용한다. Optional field의 부재(`None`)까지 선택하는 기능은 구현되어 있지 않다. 이는 axis-only라는 의미적 경계와 별개이다. Optional axis까지 정리를 확장하려면 부재의 의미를 정의하고, 부재와 모든 합법적인 값의 선택을 빠짐없이 처리해야 한다. 빈 축 목록 `[]`은 `None`과 다르며 허용된다.

예를 들어 broadcast의 다음 선언은 축 mapping만 요청한다.

```tepl
derive {
    out.attrs = {
        dimensions: auto,
        new_extents: source.attrs.new_extents
    }
}
```

여기서 `new_extents`는 주어진 식으로 유지된다. 합성기가 축 후보의 출력 shape를 계산하는 것은 `extent` attribute를 합성하는 것과 다르다. Descriptor를 생략하더라도 미지수가 될 수 있는 것은 적격 `axis` field뿐이며, 나머지 필수 field는 명시하거나 기본값으로 제공해야 한다.

합성기는 `auto`로 표시된 axis field만 결정하며, 나머지 field는 주어진 식이나 기본값을 그대로 유지한다.

### 2.3 Valid Attribute Assignment

완성된 RHS attribute \(\beta\)가 유효하려면 다음 조건을 만족해야 한다.

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

\(\operatorname{Legal}_R\)는 모든 중간 연산의 유효성을 포함하며, shape는 축 순서를 포함한 벡터이다. 합성 문제에서는 여기에 주어진 field의 유효성과 보존 조건을 적용한다.

하나의 rewrite를 고정하자. 여기서 \(a\)는 `auto` axis 대입, \(F(p)\)는 주어진 field의 값, \(D(p,a)\)는 그 값을 보존하고 \(a\)를 채운 전체 RHS attribute \(\beta\)이다. 주어진 식은 metadata \(p\)에만 의존하며, \(\operatorname{Defined}\)는 정수 범위와 타입 변환을 포함해 그 식이 유효하게 평가됨을 뜻한다.

\[
\boxed{
\operatorname{Complete}(p,a)
\iff
\operatorname{Defined}(F(p))
\land
\operatorname{Valid}(p,D(p,a))
}
\]

Axis 대입값은 텐서 값이 아니라 metadata에만 의존한다. 따라서 존재성의 양화 순서는 \(\exists a\,\forall X\)이며, extent나 coordinate에 대한 존재 양화는 없다.

### 2.4 Attribute Synthesis Problem

우리의 목표는 symbolic rewrite \(L\Rightarrow R\)로부터 두 함수를 생성하는 것이다.

**1. Feasibility**

주어진 field를 보존하는 유효한 axis 대입값이 존재하는 조건을 결정한다.

\[
\boxed{
\operatorname{check}(p)
\iff
\exists a:\operatorname{Complete}(p,a)
}
\]

**2. Attribute Construction**

조건이 성립하면 `auto` slot을 채울 axis 대입값을 생성한다.

\[
\boxed{
\operatorname{check}(p)
\Rightarrow
\operatorname{Complete}(p,\operatorname{build}(p))
}
\]

두 조건은 정리가 명시한 domain 안의 모든 유효한 LHS 메타데이터 \(p\)에 대해 성립해야 한다. 생성된 전체 RHS descriptor는 \(D(p,\operatorname{build}(p))\)이다. 주어진 field의 식이 정의되지 않으면 \(\operatorname{check}(p)\)는 거짓이어야 한다.

따라서 본 연구의 목표는 특정 shape에서만 동작하는 해를 구하는 것이 아니라, symbolic rewrite로부터 해당 domain의 모든 유효한 메타데이터에 적용할 수 있는 \(\operatorname{check}\)와 \(\operatorname{build}\)를 생성하는 것이다. 이는 **주어진 field를 조건으로 하는 axis 합성의 완전성**이며, 미지의 extent나 coordinate까지 포함한 전체 descriptor 합성의 완전성이 아니다.

본문은 중간 축과 순서를 직접 구성하며, 부록은 더 넓은 표현과 별도 구조 상태 탐색을 구분한다. 어느 경우에도 tensor 좌표 격자를 열거하지 않는다. 지원 여부는 `axis` 타입만이 아니라 명시한 연산 계약과 합성 절차로 결정한다.

### 2.5 Scope and Assumptions

본문 정리는 **axis-structural multilinear** 계산 중 아래 선언 형태를 대상으로 한다. 전체 축을 연결·재배열·합산하며, 값은 독립적인 입력 원소들의 곱과 합으로 계산한다. Multilinear는 다른 입력을 고정하면 각 입력에 대해 선형이라는 뜻이다. Addition은 Section 5.3의 항별 결합으로 따로 다룬다.

추론 대상은 RHS의 독립적인 `axis` field뿐이며, extent와 coordinate는 추론하지 않는다(Section 2.2). Reverse, 새 축, 가중치 등을 포함하는 더 넓은 계약은 Appendix C.4로 구분한다.

아래 반례·경계 예시는 현재 직접 구성의 범위를 보여주며, 다른 알고리즘의 처리 가능성까지 배제하지 않는다.

| 조건 | 본문 정리의 제한 | 반례·경계 예시 |
| --- | --- | --- |
| **D1. 전체 축 구조** | 전체 축을 같은 방향으로 연결·재배열·합산한다. 새 축, 좌표의 분할·병합, 추가 가중치는 없다. | `slice([a,b,c,d], 시작=1, 길이=2) = [b,c]`처럼 축 내부의 일부 좌표만 선택한다. |
| **D2. 자유로운 RHS 축 선택** | 아래 선언의 RHS axis field는 모두 `auto`이다. 명시한 legality 외에 고정된 선택·선택 개수·layout 제약은 없으며, 빈 목록도 허용한다. | 공유 축을 batch로 남기도록 고정한 `dot` 뒤에서 `sum`으로 그 축을 제거한다. 나중에 합산하는 유효한 해가 존재한다. |
| **D3. 출력 순서 규칙** | 출력 축은 자유롭게 재배열하거나, 입력의 상대적 순서를 유지하거나, 정해진 축 그룹 순서로 이어 붙인다. | 두 그룹 \((i_1,i_2)\), \((j_1,j_2)\)를 \((i_1,j_1,i_2,j_2)\)로 교대로 배치하는 출력 선언. |
| **D4. 독립적인 axis 미지수** | 서로 다른 RHS field와 노드는 미지수를 공유하지 않는다. | \(A+\operatorname{transpose}(B)\)를 \(\operatorname{transpose}_\pi(A)+\operatorname{transpose}_\pi(B)\)로 바꾼다. 독립적인 정사각 입력에서 첫 항은 identity, 둘째 항은 swap을 요구하므로 공유 \(\pi\)로는 결합할 수 없다. |

본문에서 다루는 모든 operation은 어떤 입력 축을 연결하고, 어떤 축을 합산하며, 남은 축을 어떤 순서로 출력하는지를 선언한다.

예를 들어, contraction은 두 입력의 축을 대응시켜 같은 좌표의 원소를 곱한다. 대응된 축 중 출력에 남기는 축을 batch, 합산하여 제거하는 축을 contraction, 대응시키지 않고 그대로 남기는 축을 free라 한다. 출력은 batch → left-free → right-free 순서이며, free 축은 각 입력의 상대적 순서를 유지한다.

입력 텐서와 rewrite의 구조에는 다음을 가정한다.

- Rank는 유한하며, scalar도 허용한다.
- 입력 값은 서로 독립이고, 계산에는 반올림 오차가 없다고 가정한다. LHS의 모든 연산은 유효해야 한다.
- 각 항(component)은 하나 이상의 입력을 각각 한 번씩 사용하는 tree이다. 대응하는 LHS와 RHS 항은 같은 입력 집합을 사용한다.
- 덧셈이 있으면 LHS는 전개할 수 있어야 하고, RHS는 이미 항들의 합이어야 한다. 항마다 입력 집합은 달라야 하지만, 일부 입력을 공유해도 된다.

D2–D4에 의해 이 기본 범위의 RHS attribute \(\beta\)는 독립적인 axis 대입 \(a\)와 동일시한다. 따라서 이후 \(\operatorname{Valid}(p,\operatorname{build}(p))\)는 Section 2.3의 completion 조건의 특수 경우이다. LHS에서 복사되는 captured attribute는 원래 값을 유지한다.

새 연산으로 직접 구성을 확장하려면 Section 5.4의 성질을 증명해야 한다.

## 3. Axis-Based Representation

이 절은 Section 2.5의 선언 형태와 가정을 따른다. D1의 전체 축 구조를 \(N(T)\)로 표현하며, 더 넓은 표현은 Appendix C에서 다룬다.

텐서 표현식의 의미를 입력 축의 연결 관계와 출력 축의 순서로 나타내는 *Axis-Based Representation*을 정의한다.

### 3.1 Logical Axes

**Logical axis**는 같은 좌표를 사용하는 입력 축들을 묶은 것이다. 예를 들어 NumPy의 `np.einsum('ik,kj->ij', A, B)`에서는 A의 두 번째 축과 B의 첫 번째 축이 같은 기호 `k`를 사용하므로 하나의 logical axis에 속한다. 출력 `ij`에 없는 `k`는 합산된다.

입력 텐서 \(X\)의 축 위치 \(d\)를 **port** \((X,d)\)로 표시한다. 한 component의 입력 집합이 \(\mathcal X\)이면 모든 입력 port의 집합은 다음과 같다.

\[
P=\{(X,d)\mid X\in\mathcal X,\ 0\le d<\operatorname{rank}(X)\}
\]

두 port가 같은 logical axis에 속하면 \(p\sim q\)로 쓴다. 전체 logical axis 집합은 다음과 같다. \(P/{\sim}\)는 연결된 port들을 하나씩 묶은 집합이라는 뜻이다.

\[
\boxed{\mathcal A=P/{\sim}}
\]

크기가 같다는 이유만으로 축을 연결하지는 않는다. 합산되어 출력에서 사라진 축의 연결도 기록에 남긴다.

**Symbolic Axis Group.** 여러 logical axis의 순서 있는 목록을 symbolic axis group이라 부르고 \(\vec I,\vec J,\vec K\)로 표시한다. 예를 들어 \(\vec K=(k_1,k_2)\)는 두 개의 별도 축이며, 두 입력이 이 그룹을 공유하면 같은 위치의 축끼리 대응한다. 그룹 길이는 미리 고정하지 않는다. 이는 표기를 줄이기 위한 것으로, 정리와 증명은 개별 축을 기준으로 한다.

### 3.2 Axis Representation

앞서 정의한 logical axis를 이용해, attribute가 주어진 텐서 표현식 \(T\)의 축 연결과 출력 순서를 다음과 같이 나타낸다.

\[
\boxed{N(T)=(P,\sim,F)}
\]

여기서 \(P\)는 입력 axis port 집합, \(\sim\)은 같은 인덱스를 사용하는 port들의 연결 관계, \(F\)는 최종 출력에 남는 logical axis를 순서대로 나열한 것이다. 이를 \(T\)의 *Axis Representation*이라 부른다.

이 정의와 아래 예제는 기본 fragment에 대한 것이다. 방향, 새 출력 축, scalar weight 및 singleton 정규화가 필요한 경우에는 Appendix C의 확장 표현 \(N^+(T)\)를 사용하며, 본문의 \(N(T)\)와 구분한다.

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

같은 구조를 여러 축으로 확장한 contraction은 symbolic axis group으로 간단히 표시할 수 있다.

\[
\begin{aligned}
A\text{의 축 순서}:&\quad \vec I\Vert\vec K,\\
B\text{의 축 순서}:&\quad \vec K\Vert\vec J,\\
F=&\quad \vec I\Vert\vec J.
\end{aligned}
\]

\(\Vert\)는 그룹 안의 축들을 순서대로 이어 붙인다는 뜻이다. 두 입력의 \(\vec K\)에서 같은 위치의 축끼리 연결하고 합산한다. \(P\)는 여전히 모든 개별 입력 port를 기록하고, \(\sim\)은 각 대응 축의 연결을 기록한다. 따라서 그룹 표기를 사용해도 \(N(T)=(P,\sim,F)\)의 정의는 같다. 위 행렬 곱은 \(\vec I=(i)\), \(\vec K=(k)\), \(\vec J=(j)\)인 경우이다.

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

Contraction은 서로 겹치지 않는 입력 subtree의 축을 연결한다. 따라서 같은 자식에 따로 남은 두 축을 나중에 서로 합칠 수는 없으며, 제거된 축도 다시 나타나지 않는다. 이 성질은 Section 4의 중간 축 분석에서도 사용한다.

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

## 4. Direct Attribute Synthesis Algorithm

본 절에서는 LHS의 axis representation을 기반으로 RHS attribute를 생성하는 알고리즘을 정의한다. 아래에서 사용하는 logical axis와 최종 출력 순서 \(F\)는 LHS에서 가져온다.

알고리즘은 세 단계로 구성된다: (1) intermediate axes 결정, (2) ordering feasibility 검사, (3) attribute 생성. 각 중간 노드에서 하나의 canonical live set을 구성하며, 가능한 표현들의 집합을 탐색하지 않는다. 새로운 연산으로의 일반화 조건은 Section 5.4에 정리한다.

### 4.1 Determining the Output Axes at Each Node

**이 단계의 목표는 RHS의 각 노드에서 “어떤 축을 출력에 남길 것인가”를 결정하는 것이다.** 그 결과가 노드별 축 집합 \(P_v\)이다. 이후 Section 4.3에서 이 축들의 순서 요구와 실현 가능성을 검사하고, Section 4.4에서 실제 axis attribute를 구성한다.

\(P_v\)를 구하기 위해 먼저 앞으로 필요한 축 \(M_v\)를 찾는다. 그런 다음 필요한 축을 보존하면서 연산의 규칙에 따라 \(P_v\)를 구성한다. 즉, \(M_v\)는 반드시 지켜야 할 조건을 제공하고, \(P_v\)는 이 단계에서 구성할 결과이다.

먼저 다음을 정의한다.

- \(U_v\): 노드 \(v\)의 subtree에 포함된 입력 텐서 집합
- \(\sigma(a)\): logical axis \(a\)를 사용하는 입력 텐서 집합
- \(F\): LHS의 최종 출력 축 순서
- \(M_v\): 노드 \(v\)에서 반드시 유지해야 하는 축 집합
- \(P_v\): 노드 \(v\)의 출력에 남는 logical axes의 집합

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

\(P_v\)는 노드 \(v\)의 출력에 남는 logical axes의 집합이다. 앞으로 필요한 축을 보존해야 하므로 다음을 만족한다.

\[
\boxed{M_v\subseteq P_v}
\]

\(P_v\)는 이 조건과 연산의 합법적인 축 규칙에 따라 구성하며, 구체적인 구성 규칙은 Appendix A.1에 둔다. 포함 조건만으로 \(P_v\)가 하나로 결정되는 것은 아니다.

입력 \(X\)의 logical-axis 순서를 \(I_X\)라 하고, \(P_X=\operatorname{set}(I_X)\)에서 시작한다. \(F\)와 \(\sigma\)는 LHS에서 미리 구한다. RHS를 bottom-up으로 순회하면서 \(U_v\)와 \(M_v\)를 계산하고, 자식의 축 정보와 부록의 규칙으로 \(P_v\)를 구성한다.

그룹 표기는 같은 역할의 축들을 묶어 보여주지만, \(M_v,P_v\)는 개별 축의 집합이다. 그룹 일부만 필요하면 순서를 유지하며 나누고, 불필요한 부분을 제거할 수 있는지는 연산 규칙에 따른다.

**그룹으로 보는 예제.** RHS가 `dot(dot(A,B),C)`이고, LHS에서 얻은 축 연결과 최종 출력 순서가 다음과 같다고 하자. 서로 다른 그룹은 서로 다른 logical axes를 나타낸다.

\[
A:\vec I\Vert\vec K,\qquad
B:\vec K\Vert\vec J,\qquad
C:\vec J\Vert\vec L,\qquad
F=\vec I\Vert\vec L.
\]

중간 노드 \(v=\operatorname{dot}(A,B)\)에서는 \(U_v=\{A,B\}\)이다. 이 subtree 밖의 원래 입력은 C이며, 각 그룹의 역할은 다음과 같다.

| 그룹 | 앞으로 필요한 이유 | 이 dot에서의 처리 |
| --- | --- | --- |
| \(\vec I\) | 최종 출력에 필요 | 유지 |
| \(\vec J\) | 밖의 입력 C와 연결해야 함 | 유지 |
| \(\vec K\) | A와 B 안에서 역할이 끝남 | 대응 축들을 합산 |

이 경우에는 \(\vec K\)를 합산하여 \(M_v=P_v=\operatorname{set}(\vec I)\cup\operatorname{set}(\vec J)\)를 얻는다. 다른 연산에서는 불필요한 축을 제거하지 못해 \(P_v\)가 \(M_v\)보다 클 수 있다. \(P_v\)는 축 집합만 정하며, 출력 순서의 요구와 실현 가능성은 Section 4.3에서 다룬다.

D1의 전체 축 연결과 D2의 선택 자유도 아래, 부록의 규칙이 유효한 해를 놓치지 않음을 다음 절에서 증명한다. 전체 입력 port 집합 \(P\)와 노드별 출력 축 집합 \(P_v\)는 구분한다.

마지막으로 RHS root에서 다음을 검사한다. 중간에는 불필요한 축이 남을 수 있지만, 최종 출력에는 필요한 축만 남아야 한다.

\[
\boxed{
P_{\mathrm{root}}=\operatorname{set}(F)
}
\]

### 4.2 Canonicalization Lemma

D2의 선택 자유도를 사용하여, Appendix A.1의 canonical 중간 축으로 바꾸어도 유효한 해를 잃지 않음을 보인다.

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

중간 축 집합 \(P_v\)를 정한 뒤에는 각 노드가 필요한 출력 순서를 실현할 수 있는지 확인한다. 노드 \(v\)의 출력에 반드시 포함되어야 하는 ordered subsequence를 \(Req_v\)라 하며, root의 요구는 LHS 출력 순서이다.

\[
Req_{\mathrm{root}}=F.
\]

두 sequence \(A,B\)에 대해 \(A\preceq B\)는 \(A\)가 \(B\)의 subsequence임을 뜻한다. \(\Vert\)는 sequence concatenation이며, \(S|_P\)는 \(S\)에서 집합 \(P\)에 속한 축만 기존 순서대로 남기는 filter이다.

**그룹으로 보는 순서 요구.** \(\vec I=(i_1,i_2)\), \(\vec J=(j_1,j_2)\)일 때,

\[
Req_v=\vec I\Vert\vec J=(i_1,i_2,j_1,j_2)
\]

는 I의 축들이 먼저, J의 축들이 나중에 나타나야 한다는 뜻이며, 각 그룹 내부의 순서도 유지해야 한다. \(Req_v\)는 subsequence 요구이므로 그룹을 연속된 블록으로 배치할 필요는 없다. 예를 들어 \(x\)가 \(P_v\)에 남아 있는 다른 축이면 \((i_1,x,i_2,j_1,j_2)\)도 이 순서 요구를 만족하지만, \((i_2,i_1,j_1,j_2)\)는 만족하지 않는다. 전체 layout의 legality는 해당 연산의 계약도 만족해야 한다. 그룹 표기는 기존의 개별 축 순서 검사를 바꾸지 않는다.

D3의 출력 형식에 따라 각 노드에 **순서 전달 규칙**을 둔다. 이 규칙은 현재 요구 \(Req_v\), 이미 계산한 중간 축 집합 및 메타데이터를 읽고, 다음 두 결과를 직접 계산한다.

- \(C_v\): 해당 노드에서 만족해야 하는 순서 제약의 검사 결과.
- 각 자식 \(u\)에 대한 \(Req_u\): 자식에게 전달할 순서 요구.

입력 노드에는 자식이 없으며 실제 축 순서 \(I_v\)가 고정되어 있으므로,

\[
C_v=[Req_v\preceq I_v]
\]

로 둔다. 연산 노드의 규칙은 그 연산의 출력 계약에 따른다. 연산별 정의는 Appendix A에 모으고, 본문에서는 공통 인터페이스를 사용한다.

허용된 출력 형식에서는 부모의 요구를 현재 검사 \(C_v\)와 자식별 요구로 정확히 나눌 수 있다. 각 규칙은 Appendix A.2에 정의한다. 순서 전달의 필요성은 Section 5.1에서 직접 보이고, 구성의 세 귀납 성질은 Section 5.2에서 증명한다. \(C_v\)는 미지수나 일반적인 존재성 oracle이 아니라 이 규칙으로 계산하는 참·거짓 값이다.

Root에서 시작하여 요구를 top-down으로 전달한 뒤 다음을 검사한다. \(V_R\)는 입력 노드를 포함한 RHS의 모든 노드이다.

\[
\boxed{
\operatorname{check}(p)
=
[P_{\mathrm{root}}=\operatorname{set}(F)]
\land
\bigwedge_{v\in V_R}C_v
}
\]

이 식은 다음 두 질문을 뜻한다.

1. **최종 출력에 남는 축이 목표와 같은가?**
2. **모든 노드에서 요구하는 순서를 실현할 수 있는가?**

Root의 축 집합이 다르면 즉시 거짓을 반환한다. 그 외에는 요구를 전달하면서 검사하고, 실패한 경우 constructor를 실행하지 않는다. 각 규칙은 부적합한 요구를 포함해 유한한 검사 결과를 반환해야 한다. 따라서 `check(p)`는 명시한 메타데이터 domain에서 정의된다.

이 검사는 Section 2.5의 선언에서 도출한 규칙을 실행하며, 가능한 축 표현이나 attribute 대입을 열거하지 않는다.

### 4.4 Attribute Construction

**이 단계의 목표는 정해진 축과 순서를 실제 RHS axis attribute로 바꾸는 것이다.** \(\operatorname{check}(p)\)가 참이면 RHS를 bottom-up으로 순회하며 다음을 수행한다.

**1. 연산에서 실현 가능한 출력 순서를 구성한다.** 자식의 출력 순서와 해당 연산의 구성 규칙으로 노드 \(v\)의 출력 순서 \(\operatorname{Ord}_v\)를 정한다. 이 순서는 다음을 만족해야 한다.

\[
\boxed{
\operatorname{set}(\operatorname{Ord}_v)=P_v,
\qquad
Req_v\preceq\operatorname{Ord}_v
}
\]

D2–D3에 따른 Appendix A.3의 규칙으로 이 순서를 실현한다. 출력 순서를 임의로 고르는 것이 아니라, 자식의 순서와 연산의 허용 형식으로 구성한다.

**2. 선택한 축 그룹을 위치 목록으로 바꾼다.** \(\operatorname{pos}(S,\vec G)\)는 그룹 \(\vec G\)의 각 축이 순서 목록 \(S\)의 어디에 있는지, 그룹 순서대로 나열한 것이다. 위치는 0부터 센다.

예를 들어 자식의 출력 순서와 선택한 그룹이

\[
\operatorname{Ord}_u=(i_1,k_1,i_2,k_2),
\qquad \vec K=(k_1,k_2)
\]

이면,

\[
\boxed{\operatorname{pos}(\operatorname{Ord}_u,\vec K)=[1,3]}
\]

이다. 따라서 이 자식에서 \(\vec K\)를 선택하는 axis-list attribute는 `[1,3]`이 된다. 각 축이 목록에 한 번씩 존재하는지와 그 위치 목록이 연산의 계약을 만족하는지는 구성 규칙의 legality 조건으로 확인한다. Symbolic group을 실제 위치 목록으로 바꾸더라도 그룹 내부의 축 순서는 유지한다.

**3. 노드별 attribute를 결합한다.** D4에 의해 노드별 대입은 충돌하지 않으므로 이를 모아 최종 대입을 얻는다.

\[
\boxed{\beta=\operatorname{build}(p)}
\]

여기서 \(\beta\)는 Section 2.5의 기본 범위에서 서로 독립적인 RHS axis 대입이다. 이 절차는 가능한 attribute 조합을 열거하지 않고, 중간 축 분석과 순서 검사 결과로 attribute를 직접 구성한다.

Rewrite의 고정된 tree와 검증된 연산 규칙으로부터 \(\operatorname{check}\)와 \(\operatorname{build}\) 프로그램을 생성할 수 있다. 생성된 프로그램은 지원 범위 내 임의의 유한 rank에 적용하며, 작성 시 구체적인 축 수를 고정할 필요는 없다.

## 5. Correctness and Completeness

본 절에서는 Section 4의 attribute synthesis 알고리즘이 유효한 RHS attribute의 존재성을 정확하게 판별하고, 존재하는 경우 올바른 attribute를 생성함을 증명한다.

**Theorem 1 (Complete Constructive Attribute Synthesis).**

Section 2.5의 제한·선언 형태·rewrite 가정을 만족하는 규칙에 대해, Section 4에서 도출한 \(\operatorname{check}\)와 \(\operatorname{build}\)는 다음을 만족한다. 분리된 addition은 Section 5.3의 component별 결합을 사용한다.

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

**Lemma 3 (Necessity).** 유효한 RHS attribute assignment가 존재하면 Section 4.3의 검사를 통과한다.

\[
\boxed{
\exists\beta:\operatorname{Valid}(p,\beta)
\Rightarrow\operatorname{check}(p)
}
\]

**Proof.** 유효한 대입 \(\beta\)가 있다고 하자. D1–D2 아래 증명한 Lemma 2에 의해 같은 결과를 내면서 각 노드에 canonical 축 집합 \(P_v\)를 남기는 대입 \(\beta'\)가 존재한다. 따라서 root에서는

\[
P_{\mathrm{root}}=\operatorname{set}(F)
\]

이고, 실제 출력 순서가 \(F\)이므로 \(Req_{\mathrm{root}}=F\)를 만족한다.

이제 root에서 아래로 내려간다. 현재 노드의 합법적인 출력이 \(Req_v\)를 포함한다고 하자. D3에서 허용하는 출력 형식은 자유로운 재배열, 입력 순서를 유지한 부분목록, 정해진 순서로 이어 붙인 그룹들이다.

입력 순서를 유지하는 부분에서는 출력의 요구 순서가 자식 순서에도 포함되므로, 해당 요구를 자식에게 전달할 수 있다. 자유롭게 재배열하는 부분은 현재 노드에서 순서를 정하므로 자식에게 추가 순서를 요구하지 않는다. 그룹들을 이어 붙이는 경우에는 \(Req_v\)도 그 그룹 순서를 따를 수밖에 없어 현재 검사 \(C_v\)가 참이며, 각 입력에서 물려받는 부분의 요구도 해당 자식에서 만족한다. 재배열과 부분목록 형식의 local 검사는 참으로 정의되어 있다.

따라서 각 노드에서 \(C_v\)가 참이고, 자식에게 전달한 요구도 만족된다는 성질이 유지된다. 입력 노드에 도달하면 \(Req_v\preceq I_v\)이므로 입력 검사도 통과한다.

따라서 모든 노드의 \(C_v\)가 참이며, root의 축 집합 검사도 통과하므로 \(\operatorname{check}(p)=\mathrm{true}\)이다. \(\square\)

### 5.2 Sufficiency

**Lemma 4 (Sufficiency).** Section 4.3의 검사를 통과하면 Section 4.4가 유효한 RHS attribute를 구성한다.

\[
\boxed{
\operatorname{check}(p)
\Rightarrow\operatorname{Valid}(p,\operatorname{build}(p))
}
\]

**Proof.** \(\operatorname{check}(p)=\mathrm{true}\)라고 하자. RHS를 bottom-up으로 구성하면서 각 노드에서 다음 세 성질을 유지한다.

1. 구성한 attribute와 중간 연산이 합법적이다.
2. 출력 축은 \(P_v\)이고 요구 순서를 만족한다.
3. 입력 port의 연결은 LHS 연결을 해당 subtree의 입력에 제한한 것과 같다. 합산되어 출력에서 사라진 class도 연결 기록에 포함한다.

두 번째 성질은 다음 식으로 나타낸다.

\[
\operatorname{set}(\operatorname{Ord}_v)=P_v,
\qquad Req_v\preceq\operatorname{Ord}_v.
\]

입력 노드에서는 실제 입력 축 순서를 사용한다. 순서 검사를 통과했으므로 요구를 만족하고, 한 입력의 서로 다른 축은 서로 다른 logical axis이므로 연결 성질도 성립한다.

연산 노드에서는 자식들이 위 성질을 만족한다고 가정한다. Section 2.5의 제한과 Section 4.4의 구성 규칙으로 다음을 확인한다.

- **합법성.** 선택할 축은 해당 자식에 한 번씩 존재하므로 범위 내의 중복 없는 위치 목록을 얻는다. 연결하는 축은 같은 LHS class에 속하여 extent가 같고, batch와 합산 축은 서로 겹치지 않는다. 따라서 D2의 legality 조건을 만족하며, D4에 의해 자식의 대입과 충돌 없이 결합한다.
- **출력 축과 순서.** 구성은 \(P_v\)의 각 축을 정확히 한 번 남긴다. D3의 재배열 형식에서는 요구 축을 먼저 배치하고, 부분목록 형식에서는 자식이 만족한 요구를 유지한다. Block 형식에서는 자유롭게 정하는 batch 순서와 자식에게서 물려받는 free 순서를 결합한다. 현재 검사 \(C_v\)가 block 순서를 확인하므로 이 경우에도 \(Req_v\preceq\operatorname{Ord}_v\)이다.
- **연결 보존.** 양쪽 subtree에 걸친 LHS class는 각 자식의 mandatory 축이므로 양쪽에 남아 있으며, 구성은 같은 class의 축끼리만 연결한다. 재배열과 합산은 새로운 연결을 만들지 않는다. 또한 \(M_v\) 밖의 축만 합산하므로, 제거되는 축은 모든 supporting input이 현재 subtree 안에 있고 최종 출력에도 필요하지 않다. 따라서 이후 필요한 연결을 잃지 않으며, 합산한 class도 연결 기록에는 남는다.

따라서 세 성질은 모든 노드에서 유지된다.

Root에서는 \(P_{\mathrm{root}}=\operatorname{set}(F)\)이고 \(F\preceq\operatorname{Ord}_{\mathrm{root}}\)이다. 각 출력 축이 한 번씩 나타나므로 \(\operatorname{Ord}_{\mathrm{root}}=F\)이다. 전체 입력 연결도 LHS와 같으므로,

\[
N(L)=N(R)
\]

이다. Lemma 1에 의해 두 표현식은 모든 입력 값에서 같은 결과를 낸다. 구성한 attribute도 합법적이므로 \(\operatorname{Valid}(p,\operatorname{build}(p))\)가 성립한다. \(\square\)

### 5.3 Extension to Multiple Terms

Addition은 동일한 ordered shape의 텐서들을 원소별로 더하는 연산이며, attribute와 broadcasting은 없다.

Section 2.5에서 정한 additive component에 대해 전체 동등성을 항별 동등성으로 나눌 수 있음을 보인다. 이 절의 독립적인 component별 구성은 명시적으로 분리된 RHS 합을 대상으로 하며, Appendix C.4.3의 unary linear context에는 공동 상태 구성이 추가로 필요하다.

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

입력 집합이 같은 항들은 위 scaling으로 분리되지 않는다. 또한 component별 대입의 결합에는 D4가 필요하며, 공유 미지수의 반례는 Section 2.5에 제시했다.

Lemma 3과 Lemma 4는 단일 component의 필요충분성과 constructor의 올바름을 보인다. 위의 항별 동등성과 assignment 결합을 적용하면 addition을 포함하는 전체 규칙에도 같은 보장이 성립한다. 따라서 Theorem 1의 증명이 완성된다. \(\square\)
### 5.4 Generalizing the Direct Constructor

D1–D4와 Section 2.5의 선언 형태에서는 앞의 구체적인 규칙을 증명했다. 다른 선언을 추가하려면 다음 성질을 새 규칙에 대해 확인한다.

1. **해를 보존하는 중간 축 구성:** 원래 해를 canonical \(P_v\)로 바꾸어도 합법적인 대입과 전체 계산 결과를 보존한다(Lemma 2의 귀납 단계).
2. **정확한 순서 전달:** 가능한 부모 순서를 \(C_v\)와 자식별 요구로 빠짐없이 분해한다.
3. **합법적인 구성:** 검사를 통과한 자식 layout들에서 요구 순서와 전체 축 연결을 보존하는 attribute를 직접 구성한다.

각 규칙은 축·메타데이터에 대한 종료하는 절차여야 한다. 고정된 field가 있으면 이를 보존하고, 방향·가중치가 추가되면 그 제약도 함께 실현해야 한다.

**Proposition 2 (Composition of Direct Construction Rules).** 정확한 축 표현, read-once 구조, 독립적인 axis slot 아래에서 위 세 성질을 만족하는 규칙들을 조합하면 같은 필요성·충분성 증명을 적용할 수 있다.

**Proof.** 첫 성질로 Lemma 2의 bottom-up 귀납을 확장한다. 두 번째 성질로 유효한 해가 모든 순서 검사를 통과함을 보인다. 반대로 세 번째 성질을 bottom-up으로 적용하여 합법적인 대입을 구성한다. Root에서 전체 연결과 출력 순서가 목표와 같으므로 exactness에 의해 동등하다. \(\square\)

이는 새 연산의 세 성질 자체를 대신 증명하지 않는다. 현재 확인한 직접 확장은 Appendix B에, 더 넓은 표현과 지원 경계는 Appendix C에 정리한다.

## Appendix A. Concrete Construction Rules

Permutation은 입력 축의 전체 재배열이다. Unary summation은 범위 내의 중복 없는 축 위치 목록을 선택해 해당 축의 전체 범위를 합산하고 제거하며, 남은 축의 상대적 순서를 유지한다. 빈 목록을 허용하고 keepdims나 추가 legality 제약은 두지 않는다.

### A.1 Reduction Sets and Remaining Axes

Section 4.1의 \(P_v\)를 계산하기 위해, \(R_v\)를 현재 노드에서 합산하여 제거하도록 정한 축 집합이라 하자. 새 축을 만들지 않는 Section 2.5의 기본 연산에서는 다음 식을 사용한다.

\[
P_v=\left(\bigcup_{u\in\operatorname{children}(v)}P_u\right)\setminus R_v,
\qquad
R_v\subseteq\bigcup_{u\in\operatorname{children}(v)}P_u,
\qquad R_v\cap M_v=\varnothing.
\]

\(\operatorname{children}(v)\)는 해당 연산에 직접 입력을 주는 자식 노드들이다. 자식의 축을 같은 LHS logical-axis identity로 표시하므로 공유 축은 합집합에 한 번만 나타난다. 입력 노드는 별도의 초기값 \(P_X=\operatorname{set}(I_X)\)를 사용한다.

\(R_v\)는 위 조건만으로 임의 선택하지 않고, 자식의 축과 \(M_v\)를 사용하여 아래 규칙으로 정한다. 이 규칙은 해당 연산의 자유로운 axis attribute와 전체 legality 계약을 전제로 한다.

입력 텐서 \(X\)의 축을 LHS의 logical axis 이름으로 표시한 순서 목록을 \(I_X\)라 하자. 예를 들어 행렬 곱의 두 입력은 \(I_A=(i,k)\), \(I_B=(k,j)\)이다. Unary 연산의 자식을 \(u\)라 하면, 각 노드의 규칙은 다음과 같다.

| 노드 | 합산할 축 \(R_v\) | 남는 축 \(P_v\) |
| --- | --- | --- |
| Input \(X\) | 해당 없음: 초기값 사용 | \(\operatorname{set}(I_X)\) |
| Permutation | \(\varnothing\) | \(P_u\) |
| Unary summation | \(P_u\setminus M_v\) | \(P_u\cap M_v\) |
| Contraction | \(K_v=(P_\ell\cap P_r)\setminus M_v\) | \((P_\ell\cup P_r)\setminus K_v=B_v\cup LF_v\cup RF_v\) |

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

이 규칙의 projection/lifting 성질은 Section 4.2에서 증명한다. 일반적인 이름으로 표현하더라도 연산별 규칙과 그 증명 의무는 유지된다.

### A.2 Ordering Rules

Section 4.3의 \(C_v\)와 자식 요구는 기본 연산에 대해 다음과 같이 계산한다. 이 규칙은 Section 2.5의 자유로운 axis attribute와 Section 4.1의 canonical 축 집합을 전제로 한다.

| 노드 | 자식에게 전달할 요구 | Local 검사 \(C_v\) |
| --- | --- | --- |
| Input | 없음 | \(Req_v\preceq I_v\) |
| Permutation | \(Req_u=[]\) | \(\mathrm{true}\) |
| Unary summation | \(Req_u=Req_v\) | \(\mathrm{true}\) |
| Contraction | \(Req_\ell=Req_v|_{LF_v}\), \(Req_r=Req_v|_{RF_v}\) | 아래의 batch/free block 순서 검사 |

표는 \(Req_v\)가 \(P_v\)의 중복 없는 subsequence 요구인 경우에 적용한다. 잘못된 축이나 중복을 포함하는 요구에는 검사 규칙이 거짓을 반환한다. 정상적인 root 요구와 위 전달 규칙에서는 이 조건이 유지된다.

Permutation은 현재 노드에서 순서를 자유롭게 정하므로 자식에게 요구를 전달하지 않는다. Unary summation은 남은 축의 순서를 유지하므로 같은 요구를 자식에게 전달한다.

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


이 정의를 Section 4.3에 대입하면 전체 식은 입력의 subsequence 검사와 contraction의 block 순서 검사의 conjunction이 된다. Permutation과 summation의 local 검사는 참이지만, 자식에게 전달한 요구는 계속 검사하므로 순서 제약을 생략한 것이 아니다. 순서 전달의 필요성은 Section 5.1에서 직접 증명한다. 구성의 세 귀납 성질과 전체 tree에서의 충분성은 Section 5.2에서 증명한다.

### A.3 Attribute Construction Rules

Section 4.4의 공통 절차는 기본 연산에 대해 다음과 같이 구체화한다. 자식의 출력 순서는 이미 구성되어 있으며, Section 4.3의 검사를 통과한 경우에 적용한다.

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

위치 목록은 Section 4.4의 \(\operatorname{pos}\)로 계산한다.

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


이 규칙들이 합법적인 attribute와 요구를 만족하는 출력 순서를 구성하고 전체 축 연결을 보존한다는 것은 Section 5.2에서 bottom-up 귀납으로 증명한다.

## Appendix B. Extensions of the Direct Constructor

다음은 Section 4의 직접 구성을 유지하면서 추가할 수 있는 확장이다. 고정 배율의 근거는 아래에 제시하고, 방향 검사를 추가하는 reverse는 별도의 증명을 참조한다.

### B.1 Fixed Unary Scaling without Structural Search

기본 fragment의 tree 안에 attribute-free unary 연산 \(T\mapsto qT\)를 허용하자. 여기서 \(q\)는 선언에 고정된 0이 아닌 유리수이다. Negation은 \(q=-1\)인 경우이다. 이 연산은 shape와 축 순서를 바꾸지 않고 추가 legality 제약도 부과하지 않는다. Tensor 값에 의존하는 배율, 0 배율, axis 선택에 따라 달라지는 배율은 이 절의 대상이 아니다.

Component \(E\)의 모든 unary 배율을 곱한 값을 \(c_E\), 그 unary 노드를 제거한 tree를 \(\bar E\)라 하자. 남은 tree가 Section 2.5의 계약을 만족해야 한다. 유한 합의 선형성과 곱의 다중선형성으로

\[
E=c_E\bar E
\]

가 성립한다. Unary 노드를 제거해도 shape, legality 및 원래 axis slot의 대응은 유지된다. 따라서 동일한 입력 집합의 component에 대해 다음이 성립한다.

\[
\boxed{
\operatorname{check}_{\mathrm{scale}}(p)
=[c_L=c_R]\land\operatorname{check}_{\mathrm{direct}}(\bar L,\bar R;p)
}
\]

**증명.** 계수는 고정되어 있고 0이 아니다. 정규화된 두 tree의 축 표현이 다르면 Lemma 1의 one-hot 입력이 한쪽을 1, 다른 쪽을 0으로 만들어, 원래 표현도 서로 다르다. 축 표현이 같으면 그 연결을 만족하는 one-hot 입력에서 정규화된 값이 1이므로 원래 값의 동등성은 \(c_L=c_R\)를 요구한다. 계수와 축 표현이 모두 같으면 동등성은 자명하다. 그러므로 배율을 정확히 비교한 뒤 기존 직접 constructor를 실행하고, 얻은 axis 대입을 원래 tree의 같은 slot에 넣으면 된다. \(\square\)

이는 구조 상태 탐색 없이 가능한 직접 확장이다. 분리된 addition에는 Section 5.3의 서로 다른 입력 집합 조건 아래 component별로 적용한다. 다중 operand의 aligned product는 unary 배율과 다르므로, 이 논증으로 자동 포함하지 않는다.

### B.2 Reversal with an Additional Orientation Check

Reverse는 축 집합이나 출력 순서를 바꾸지 않으므로 기존 live-set과 ordering 분석을 유지할 수 있다. 다만 unsigned \(N(T)\)만 비교해서는 방향을 알 수 없다. 각 살아 있는 logical axis에 방향 bit를 붙이고 입력·출력의 방향과 연결된 operand의 방향이 일치하는지 별도로 검사해야 한다. Reverse 노드의 attribute는 자식과 출력 방향 bit가 다른 축을 선택하여 구성한다.

이는 전체 구조 상태를 탐색하는 방식과 구분되는 직접 확장이다. 자유롭게 선택할 수 있는 reverse 축 목록, 기존의 read-once·extent domain 등 제한 아래에서, canonicalization과 방향 제약의 동시 실현성은 [영문 Sections 3–4.2](../proofs/completeness.md#42-orientation-feasibility-without-descriptor-enumeration)에 증명되어 있다. 고정된 reverse 목록이나 추가 layout 제약까지 자동으로 포함하지 않으며, 한국어 본문의 unsigned constructor가 수정 없이 reverse를 처리한다는 뜻도 아니다.

## Appendix C. Extended Representation and Scope

### C.1 Domain of the Basic Representation

Sections 3–5의 기본 표현과 직접 구성 증명은 모든 입력 축의 크기가 **2 이상**인 특수 경우에 한정된다. 특히 Lemma 1의 분리 증명은 서로 다른 좌표 0과 1을 사용하므로 singleton에 그대로 적용할 수 없다. Appendix C.4의 확장 계약은 singleton도 허용한다. Appendix C의 확장 표현은 Appendix C.3.1에 따라 unit class를 제거한 뒤 비단위 class에 분리 논증을 적용하고, ordered shape를 별도로 보존한다. 따라서 기본 표현의 정의를 바꾸지 않고 singleton을 포함하는 확장 범위를 별도로 다룬다.

### C.2 Extended Axis Representation

본문의 \(N(T)=(P,\sim,F)\)는 기본 fragment의 의미를 정확히 표현한다. Appendix C.4의 더 넓은 계약에서는 그 표현을 다음과 같이 확장할 수 있다. 여기서 입력 집합과 합법적인 메타데이터는 고정한다.

\[
\boxed{N^+(T)=(s,\Pi,w)}
\]

- \(s\)는 singleton 위치를 포함한 전체 ordered output shape이다.
- \(\Pi\)는 입력 port와 출력 위치 port 사이의 정규화된 연결 구조이며, 상대적인 방향도 포함한다. 출력 port가 없는 입력 연결 class는 합산되는 좌표를 기록하고, 입력 port가 없는 출력 class는 독립적인 새 출력 축을 기록한다.
- \(w\)는 선언된 scaling과 닫힌 축의 합산 계수를 포함하는 정확한 유리수 가중치이다.

Appendix C.3의 규칙으로 unit class, 방향 표기 및 닫힌 축을 정규화한다. 입력에 연결된 class의 extent는 고정된 입력 shape에서, 출력에만 연결된 class의 extent는 \(s\)에서 알 수 있다. 입력과 출력 모두에 연결되지 않은 class는 가중치에 반영한다.

| 확장 대상 | 추가로 필요한 정보 |
| --- | --- |
| Reverse | 연결의 상대적인 방향 |
| Supplied-extent broadcast | 입력 연결이 없는 출력 축과 그 크기 |
| Rational scaling, mean, fresh-axis reduction | Scalar weight |
| Singleton 입력·출력 | Unit 연결의 정규화와 전체 output shape 보존 |

기본 fragment에서는 기존 \(P,\sim\)를 유지한 채 \(F\)의 각 위치에 출력 port를 붙이면 \(\Pi\)를 얻는다. 모든 방향은 같고 \(w=1\)이며, \(s\)는 입력 extent와 \(F\)에서 결정된다. 따라서 기존 표현은 확장 표현의 특수 경우로 포함된다. 이 대응을 설명하기 위해 본문에서 \(P\), \(\sim\), \(F\)의 정의나 행렬 곱 예제를 변경할 필요는 없다.

일반 계약을 만족하는 단일 read-once component에 대해서는 확장 표현의 exactness, 즉 모든 입력 값에서의 동등성과 \(N^+(L)=N^+(R)\)의 동치가 성립한다. 이 주장은 본문의 Lemma 1을 그대로 적용한 결과가 아니라, [영문 Section 12.2](../proofs/completeness.md#122-canonical-boundary-state-and-exactness)의 확장 exactness 논증에 따른다. Addition은 서로 다른 입력 집합의 component를 구분하고 원래 axis 선택의 공유를 보존하는 별도 구성이 필요하다.

확장 표현의 exactness만으로 Section 4의 직접 구성까지 완전해지는 것은 아니다. 직접 확장은 Appendix B와 Section 5.4를, 별도의 구조 상태 탐색은 Appendix C.4.4를 따른다.

### C.3 Canonicalization

다음은 Appendix C.4의 일반 계약에 필요한 정규화 규칙이다. 별도의 지원 연산을 추가하는 확장이 아니라, 같은 의미를 가진 축 구조를 같은 상태로 표현하기 위한 규칙이다. 자세한 exactness 증명은 [영문 Section 12.2](../proofs/completeness.md#122-canonical-boundary-state-and-exactness)에 제시되어 있다.

#### C.3.1 Singleton Normalization

크기 1인 축의 좌표는 항상 0이므로, 그 축의 연결과 방향은 값에 차이를 만들지 않는다. 따라서 canonical 연결 상태에서는 unit class를 제거하되, **전체 ordered output shape에는 크기 1인 축과 그 위치를 유지한다**. 이는 실제 텐서의 shape를 임의로 바꾸는 변환이 아니다. Unit 축에 대한 합산의 계수는 1이다. 이 정규화를 통해 일반 정리는 singleton 입력과 출력을 포함한다.

크기 1인 입력 축의 좌표는 항상 0이다. Supplied-target broadcast가 이 축을 확장하는 경우, 그 unit 좌표를 제거하고 주어진 목표 extent의 새 축을 도입하는 것으로 표현한다. 이는 살아 있는 비단위 좌표의 크기를 변경하는 것이 아니다.

#### C.3.2 Orientation Convention

각 비단위 연결 class에서 고정된 port 순서에 따라 대표 port를 하나 고르고, 그 방향을 정방향으로 둔다. 다른 port에는 대표와 방향이 같은지 반대인지를 기록한다. 대표를 바꾸거나 모든 방향 bit를 함께 뒤집어도 상대적인 좌표 관계는 같으므로, 이 기준으로 같은 연결이 항상 같은 표현을 갖게 한다. 반전은 크기 \(n\)인 축에서 \(t\mapsto n-1-t\)를 뜻한다.

#### C.3.3 Closed-Axis Weights

입력 port도 출력 port도 남지 않은 독립적인 축을 합산하면, 그 축의 extent만큼 동일한 값을 더한다. 크기가 \(n\)이면 이 class를 제거하고 전체 가중치에 \(n\)을 **class당 한 번** 곱한다. 예를 들어 scalar \(x\)를 길이 3으로 broadcast한 뒤 전체 합산하면 \(3x\)이다. 입력 port에 연결된 일반적인 reduction 축은 입력 값에 영향을 주므로 이 규칙으로 제거하지 않는다. 가중치는 정확한 유리수로 계산하며, 좌표를 열거하지 않는다.

### C.4 Broader Axis-Structural Multilinear Contract

이 절은 본문 Section 2.5보다 넓은 확장 계약을 정의한다. 이 계약에 속한다는 사실만으로 본문의 직접 구성 완전성이 성립하지는 않는다. 직접 확장은 Section 5.4의 증명 의무를, 별도의 구조 상태 탐색은 Appendix C.4.4의 논증을 따른다.

#### C.4.1 Axis-Structural Multilinear Contract

Addition을 제외한 각 연산은 다음 구조로 표현되어야 한다.

1. **전체 축의 연결과 방향.** 서로 다른 operand의 같은 extent를 가진 축을 연결할 수 있다. 각 연결은 좌표를 그대로 대응시키거나 축 전체를 반전한다. 크기 \(n\)인 축에서 허용되는 대응은 \(t\) 또는 \(n-1-t\)이다. 하나의 연결된 class에는 각 operand의 축이 최대 하나만 포함되며, 연결은 forest로 표현한다. 좌표를 반전의 고정점 등 특정 위치에 제한하는 추가 방정식은 허용하지 않는다.
2. **주어진 extent의 새 축.** 주어진 메타데이터나 선언 상수로 크기가 결정되는 독립적인 축을 도입할 수 있다. 크기가 같다는 이유만으로 두 축의 좌표를 동일시하지 않는다.
3. **출력과 전체 축소.** 연결된 축들 중 출력에 남길 축과 그 순서를 선택하고, 나머지 축은 전체 좌표 범위에 걸쳐 합산한다. 하나의 class는 출력에서 최대 한 번 나타난다. 유지되는 비단위 축의 크기를 변경하거나 그 좌표를 분할·병합하지 않는다.
4. **값의 계산.** 위 좌표 대응에 따라 각 operand의 값을 곱하고 전체 축소를 수행한 뒤, 0이 아닌 유리수 가중치 \(w\)를 적용한다. 가중치는 주어진 메타데이터, 자식의 ordered shape, 해당 연산의 axis 선택으로 결정되며, 텐서 값에 의존하거나 별도의 합성 미지수가 되어서는 안 된다.
5. **국소 legality와 유한 선택.** Legality 역시 주어진 메타데이터, 자식의 ordered shape, 해당 연산의 axis 선택으로 결정된다. 모든 합법적인 axis 선택은 rank와 메타데이터 목록 길이로 제한되는 유한 절차로 열거할 수 있어야 한다. 메타데이터 계산은 축·목록 순회와 정확한 산술로 종료하며, 텐서 좌표 구간을 순회하지 않는다.

고정된 메타데이터와 axis 선택 아래에서 각 연산은 독립적인 operand마다 선형이다. 대표적인 확장과 필요한 표현 정보는 Appendix C.2의 표에 정리했다.

#### C.4.2 Input and Arithmetic Domain

유한 rank, 독립 입력 값, exact real arithmetic 및 LHS legality는 Section 2.5를 따른다. 다만 이 확장에서는 입력 축과 새 축의 extent를 양수로 허용하여 singleton을 포함한다. Zero extent는 제외하고 Appendix C.3의 정규화를 적용한다. 주어진 field의 유효성과 보존은 Sections 2.2–2.3을 따른다.

#### C.4.3 Rewrite Structure and Addition

Read-once component와 서로 다른 component의 입력 집합 조건은 Section 2.5를 따른다. 독립 상수 component, component 내부의 반복 입력 및 같은 입력 집합을 가진 항 사이의 cancellation은 제외한다. Addition의 의미는 Section 5.3과 같지만, 이 확장에서는 RHS의 합 바깥에 계약 내 unary linear 연산도 허용한다. 비가산 다중 operand 노드의 각 자식은 단일 component로 제한한다.

Unary linear 연산을 합에 분배할 때에는 원래 하나였던 axis 선택을 모든 component에 동일하게 적용한다. 복사된 occurrence를 서로 독립적인 미지수로 바꾸어서는 안 된다. 구성 알고리즘은 component별 상태를 함께 기록하고, 원래 tree의 하나의 일관된 대입을 복원해야 한다. 서로 다른 원래 `auto` slot은 독립적이며, 여러 노드가 임의로 하나의 미지수를 공유하는 경우는 이 범위에 포함하지 않는다.

#### C.4.4 Alternative Structural-State Search

각 노드에서 가능한 정규화된 축 표현의 집합을 만들고, root에서 목표 표현과 일치하는 상태의 axis 대입을 복원하는 방법은 본문의 직접 구성과 별도이다. 모든 합법적인 local axis 선택과 주어진 field를 보존하고, 자식 상태가 부모 계산에 필요한 의미와 legality 정보를 모두 담으면 상태 집합에 대한 귀납으로 완전성을 보인다. 자세한 논증은 [영문 Sections 12.3–12.7](../proofs/completeness.md#123-complete-construction)을 참조한다. 좌표 격자나 원소별 다항식을 열거하지 않지만 상태 수는 조합적으로 증가할 수 있고, 정확한 산술 비용도 메타데이터의 bit 길이에 의존한다. 이 결과를 본문의 직접 구성 완전성으로 간주하지 않는다.

### C.5 Boundary and Checked Normalization

다음 표는 원래 직접 구성의 보장을 그대로 적용할 수 없는 경우를 구분한다. 추가 증명이 없다는 것은 직접 방식으로 지원하는 것이 불가능하다는 뜻이 아니다.

| 범위 | 현재 직접 구성에서 필요한 추가 작업 또는 실패 이유 |
| --- | --- |
| 고정되거나 제한된 RHS axis field | Canonical 대입이 주어진 field를 바꿀 수 있다. 각 제한 아래에서 projection/lifting을 다시 증명해야 한다. |
| Aligned elementwise multiplication 및 다중 operand product | 연결된 축을 출력에 남겨야 하므로, 일반 contraction처럼 해당 노드에서 즉시 합산할 수 없다. 별도의 live-set과 ordering adapter가 필요하다. |
| Supplied-extent broadcast | 입력 port가 없는 새 축을 만들 수 있다. 기존 입력-support 기반 \(M_v,P_v\)만으로는 이를 나타내지 못하며, 새 축의 생존과 위치를 위한 직접 규칙이 필요하다. |
| Mean, keepdims, 새 축 도입 후 reduction | 선택한 extent에 따른 계수, 유지되는 unit 출력 위치 또는 닫힌 축의 계수를 보존해야 한다. 기존 축 제거·순서 증명만으로 충분하지 않다. |
| Singleton 입력·출력, unit-only reshape | 원래 \(N(T)\)의 exactness domain 밖이다. 정규화와 shape 보존뿐 아니라 원래 대입을 복원하는 직접 adapter의 증명이 필요하다. |
| 합 바깥의 unary 연산, 여러 component가 공유하는 axis 선택 | 기존 Section 5.3은 component별 독립 구성을 사용한다. 한 선택이 모든 component에 동시에 맞는지 검사하는 추가 직접 절차가 필요하다. |
| 일반 slice·window, split/merge reshape, convolution, padding, cumulative sum, sort, tensor-index gather/scatter | 축 내부 선택·좌표 분할 및 결합·출력 위치에 따른 합산 범위·값에 따른 접근은 Appendix C.4의 전체 축 계약 밖이다. 비-axis attribute가 주어져도 이 경계는 바뀌지 않는다. 더 풍부한 의미나 아래의 검증된 정규화가 필요하다. |

고정된 batch 선택이 조기 합산을 막는 구체적인 반례는 Section 2.5의 D2 예시와 같다.

Aligned multiplication도 같은 구분이 필요하다. `sum(A * B)`는 올바른 내적이지만, `A * B` 자체에는 축을 합산하는 attribute가 없다. 여기서 “불필요하므로 지금 제거”를 적용할 수 없다. 해당 노드에서 축을 남기는 새로운 규칙을 만들고 Section 5.4의 성질을 증명할 수는 있으나, 원래 contraction 규칙의 재사용만으로 완전성이 성립하지는 않는다.

정리하면, 현재 직접 증명 범위는 Section 2.5와 Appendix B.1의 고정 unary 배율 확장이다. Reverse는 Appendix B.2의 별도 signed 직접 구성 증명으로 다룰 수 있다. 위 표의 나머지 항목은 현재 직접 constructor의 완전성 보장에 포함하지 않는다. 이 범위는 별도의 구조 상태 탐색 정리의 범위와 구분한다.

**검증된 정규화(checked normalization)**를 통해 계약 내 표현으로 바꿀 수 있는 제한된 경우는 포함한다. 정규화는 의미, legality, 주어진 field를 보존하고, 원래 문제와 정규화된 문제의 합법적인 axis 대입을 양방향으로 대응시키며, 원래 표현의 해를 복원할 수 있어야 한다. 새로운 독립적인 미지수를 만들어서는 안 된다.

예를 들어 unit 축만 삽입·제거하고 비단위 축의 순서를 보존하는 reshape는 unit broadcast와 singleton sum으로 정규화할 수 있다. 이때 adapter가 정한 축 위치는 고정하며, 원래 문제의 모든 합법적인 completion에 정규화가 유효한지 확인해야 한다. 이 조건을 통과하면 확장 계약의 표현으로 다룰 수 있다. 본문의 직접 구성에 포함하려면 Section 5.4의 성질도 증명해야 한다. 범위 밖이라는 사실은 유효한 rewrite가 없다는 뜻이 아니다.
