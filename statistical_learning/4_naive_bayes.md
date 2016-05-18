<!--
    作者：华校专
    email: huaxz1986@163.com
**  本文档可用于个人学习目的，不得用于商业目的  **
-->
# 朴素贝叶斯法

1.朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法：

- 对给定的训练集，首先基于特征条件独立假设学习输入、输出的联合概率分布。然后基于此模型，对给定的输入 \\(\mathbf x\\) ，利用贝叶斯定理求出后验概率最大的输出 \\(y\\)
- 朴素贝叶斯法不是贝叶斯估计。
> 贝叶斯估计即最大后验估计

2.先验概率：根据以往经验和分析得到的概率。如：
你在山洞门口，你觉得山洞中有熊出现的事件为 Y 。然后你听到山洞中传来一阵熊吼的事件为 X 。一开始你以为山洞中有熊的概率为 P(Y) 。听到熊吼之后认为有熊的概率为 P(Y/X) 。很明显 P(Y/X) > P(Y)。这里：

- P(Y) 为先验概率（根据以往的数据分析或者经验得到的概率）
- P(Y/X) 为后验概率（得到本次试验的信息从而重新修正的概率）

3.设 S 为试验 E 的样本空间。 \\(B_1,B_2,\cdots,B_n\\) 为 E 的一组事件。若 ：

- \\(B_i \bigcap B_j=\phi,i \ne j,i,j=1,2,\cdots,n\\)
- \\(B_1 \bigcup B_2 \bigcup \cdots \bigcup B_n=S\\)

则称 \\(B_1,B_2,\cdots,B_n\\) 为样本空间 S 的一个划分。

如果 \\(B_1,B_2,\cdots,B_n\\) 为样本空间 S 的一个划分，则对于每次试验，事件 \\(B_1,B_2,\cdots,B_n\\) 中有且仅有一个事件发生。

4.**全概率公式**：设试验 E 的样本空间为 S， A 为 E的事件， \\(B_1,B_2,\cdots,B_n\\) 为样本空间 S 的一个划分，且 \\(P(B_i) \ge 0(i=1,2,\cdots,n)\\) ，则有：
$$P(A)=P(A/B_1)P(B_1)+P(A/B_2)P(B_2)+\cdots+P(A/B_n)P(B_n)=\sum_{j=1}^{n}P(A/B_j)P(B_j)$$

5.**贝叶斯定理**：设试验 E 的样本空间为 S， A 为 E的事件， \\(B_1,B_2,\cdots,B_n\\) 为样本空间 S 的一个划分，且 \\(P(A) \gt 0,P(B_i) \ge 0(i=1,2,\cdots,n)\\) ，则有：
$$P(B_i/A)=\frac{P(A/B_i)P(B_i)}{\sum_{j=1}^{n}P(A/B_j)P(B_j)}$$

6.**贝叶斯分类器**：

设输入空间 \\(\mathscr X \subseteq \mathbb R^{n}\\) 为 n  维向量的集合。输出空间为类标记集合 \\(\mathscr Y =\\{ c_1,c_2,\cdots,c_k\\}\\) ， \\(X\\) 为定义在  \\(\mathscr X\\) 上的随机向量，\\(Y\\) 为定义在  \\(\mathscr Y\\) 上的随机变量。则 \\(P(X,Y)\\) 为 X 和 Y 的联合概率分布。训练数据集 \\(T=\\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\\}\\)   由 \\(P(X,Y)\\) 独立同分布产生

- 朴素贝叶斯法通过训练数据集学习联合概率分布 \\(P(X,Y)\\)。具体的学习下列概率分布：
	- 先验概率分布： \\(P(Y=c_k),k=1,2,\cdots,K\\)
	- 条件概率分布： \\(P(X=\mathbf x/Y=c_k)=P(X^{(1)}=\mathbf x^{(1)},X^{(2)}=\mathbf x^{(2)},\cdots, X^{(n)}=\mathbf x^{(n)}/Y=c_k),k=1,2,\cdots,K\\)
- 朴素贝叶斯法对条件概率做了条件独立性假设：
	$$P(X=\mathbf x/Y=c_k)=P(X^{(1)}=\mathbf x^{(1)},X^{(2)}=\mathbf x^{(2)},\cdots, X^{(n)}=\mathbf x^{(n)}/Y=c_k)\\\
=\prod_{j=1}^{n}P(X^{(j)}=\mathbf x^{(j)})\\\
,k=1,2,\cdots,K$$
	这意味着： <font color='red'>在分类确定的条件下，用于分类的特征是条件独立的。</font>该假设使得朴素贝叶斯法变得简单，但是可能牺牲一定的分类准确率。

- 根据贝叶斯定理：
	$$
	P(Y=c_k/X=\mathbf x)=\frac{P(X= \mathbf x/Y=c_k)P(Y=c_k)}{\sum_{j=1}^{K} P(X=\mathbf x/Y=c_j)P(Y=c_j)}
	$$

	由于分类特征的条件独立假设有：
	$$P(Y=c_k/X=\mathbf x)=\frac{P(Y=c_k)\prod\_{i=1}^{n}P(X^{(i)}= \mathbf x^{(i)}/Y=c_k)}{\sum_{j=1}^{K} P(X=\mathbf x/Y=c_j)P(Y=c_j)},k=1,2,\cdots,K$$

	于是朴素贝叶斯分类器表示为：
	$$
	y=f(\mathbf x)=\arg \max\_{c_k}\frac{P(Y=c_k)\prod\_{i=1}^{n}P(X^{(i)}= \mathbf x^{(i)}/Y=c_k)}{\sum_{j=1}^{K} P(X=\mathbf x/Y=c_j)P(Y=c_j)}
	$$

	意思是：给定\\(\mathbf x\\)，求得使\\( \frac{P(Y=c_k)\prod\_{i=1}^{n}P(X^{(i)}= \mathbf x^{(i)}/Y=c_k)}{\sum_{j=1}^{K} P(X=\mathbf x/Y=c_j)P(Y=c_j)}\\)最大的那个 \\(c_k\\)，该 \\(c_k\\)就是 \\(f(\mathbf x)\\) 的值。

	由于对有所的 \\(c_k,k=1,2,\cdots,K\\) ，上式的分母都相同（均为 \\(P(\mathbf x)\\)），因此上式重写为：$$y=f(\mathbf x)=\arg \max\_{c_k}{P(Y=c_k)\prod\_{i=1}^{n}P(X^{(i)}= \mathbf x^{(i)}/Y=c_k)}$$

7.贝叶斯分类器是后验概率最大化，等价于期望风险最小化。

令损失函数为：
$$
L(Y,f(X))= \begin{cases}
1, & Y \ne f(X) \\\
0, & Y=f(X)
\end{cases} \\\
R_{exp}(f)=E[L(Y,f(X)]=\sum\_{\mathbf x \in \mathscr X}\sum\_{y \in \mathscr Y}[L(y,f(\mathbf x))P(X=\mathbf x,Y=y)]
$$

根据特征的条件概率独立性假设，有：
$$
R_{exp}(f)=E[L(Y,f(X)]=\sum\_{\mathbf x \in \mathscr X}\sum\_{y \in \mathscr Y}[L(y,f(\mathbf x))P(X=\mathbf x,Y=y)]\\\
= \sum\_{\mathbf x \in \mathscr X}\sum\_{k=1}^{K}[L(c_k,f(\mathbf x))P(X=\mathbf x,Y=c_k)]\\\
=E_X[\sum\_{k=1}^{K}L(c_k,f(\mathbf x))P(c_k/X=\mathbf x)]
$$

为了使得期望风险最小化，只需要对 \\(X=\mathbf x\\) 逐个极小化。因此有：
$$
f(x)=\arg \min_{y \in \mathscr Y} \sum\_{k=1}^{K}L(c_k,f(x))P(c_k/X=\mathbf x)\\\
= \arg \min\_{y \in \mathscr Y} \sum\_{k=1}^{K} P(y \ne c_k/X=\mathbf x)\\\
= \arg \min\_{y \in \mathscr Y}(1-P(y=c_k/X=\mathbf x))\\\
= \arg \max\_{y \in \mathscr Y}P(y=c_k/X=\mathbf x)\\\
$$

即得到了后验概率最大化。

8.在朴素贝叶斯法中，学习意味着估计概率：

- \\(P(Y=c_k)\\)
- \\(P(X^{(j)}=\mathbf x^{(j)}/Y=c_k)\\)

可以用极大使然估计相应概率。

- 先验概率 \\(P(Y=c_k)\\) 的极大似然估计是 \\(P(Y=c_k)=\frac {1}{N} \sum_{i=1}^{N}I(y_i=c_k),k=1,2,\cdots,K\\)
- 设第 j 个特征 \\(\mathbf x^{(j)}\\) 可能的取值为 \\(a\_{j1},a\_{j2},\cdots,a\_{js_j}\\)。则条件概率
\\(P(X^{(j)}=a_{j\;l}/Y=c_k)\\) 的极大似然估计为：

$$
P(X^{(j)}=a\_{j\;l}/Y=c_k)=\frac{\sum\_{i=1}^{N}I(\mathbf x_i^{(j)}=a\_{j \; l},y_i=c_k)}{\sum_{i=1}^{N}I(y_i=c_k)}\\\
j=1,2,\cdots,n; \\\l=1,2,\cdots,s_j;\\\k=1,2,\cdots,K
$$

9.**朴素贝叶斯算法**：

- **输入**：训练集 \\(T=\\{(\mathbf x_1,y_1),(\mathbf x_2,y_2),\cdots,(\mathbf x_N,y_N)\\}\\) ,\\(\mathbf x_i=(\mathbf x_i^{(1)},\mathbf x_i^{(2)},\cdots,\mathbf x_i^{(n)})\\), \\(\mathbf x_i^{(j)}\\) 为第 i 个样本的第 j 个特征，其中 \\(\mathbf x_i^{(j)} \in \\{a\_{j1},a\_{j2},\cdots,a\_{js_j}\\}\\)， \\(a_{j \; l}\\)为第 \\(j\\) 个特征可能取到的第 \\(l\\) 个值。