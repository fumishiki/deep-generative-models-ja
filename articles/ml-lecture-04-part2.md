---
title: "第4回: 確率論・統計学: 30秒の驚き→数式修行→実装マスター 【後編】実装編"
emoji: "🎲"
type: "tech"
topics: ["機械学習", "確率論", "統計学", "数学", "Python"]
published: true
slug: "ml-lecture-04-part2"
difficulty: "intermediate"
time_estimate: "90 minutes"
languages: ["Python"]
keywords: ["確率分布実装", "MLE実装", "ベイズ推論", "SciPy", "統計的推定"]
---

# 第4回: 確率論・統計学【後編】

> 理論編は [【前編】第4回: 確率論・統計学](/articles/ml-lecture-04-part1) をご覧ください。

## Learning Objectives

この実装編を修了すると、以下ができるようになります:

- [ ] PyTorch `torch.distributions` で `log_prob`・`entropy`・`kl_divergence` を使いこなせる
- [ ] Gaussian MLE を `D.Normal(mu, sigma).log_prob(x).sum()` で実装・検証できる
- [ ] 自己回帰尤度 $\log p(\mathbf{x}) = \sum_t \log p(x_t \mid x_{<t})$ を PyTorch で数値安定に計算できる
- [ ] `torch.logsumexp` の数値安定性の根拠を式から説明できる
- [ ] Triton カーネルで各データ点並列の対数尤度計算を実装できる
- [ ] 多変量正規分布の条件付き分布（Schur 補行列）の導出と Cholesky 安定化の必要性を説明できる
- [ ] KL ダイバージェンス・Fisher 情報量・Cramér-Rao 下界の関係を導出から示せる

---

## 💻 Z5. 試練（75分）— 5トピック完全実装+検証

### 5.1 確率分布の完全実装 — PDF・CDF・サンプリング・MLE

確率分布を「使える」とはどういうことか。PDF を評価し、累積確率を計算し、サンプルを生成し、データからパラメータを推定する——この4つがセットだ。

**Gaussian: 最も重要な分布**

$X \sim \mathcal{N}(\mu, \sigma^2)$ のとき:

$$
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- shape: `data: (N,)` → `log_prob(data): (N,)` → `.sum()` でスカラー対数尤度
- `D.Normal(mu, sigma)` の第2引数は**標準偏差** $\sigma$（分散 $\sigma^2$ ではない）。混同すると尤度が全て間違う
- 数値安定化: `.log_prob()` は内部で $\log$ 空間計算を行い `exp(-...)` のアンダーフローを回避する

記号 ↔ 変数対応:
- $\mu, \sigma$ ↔ `mu_mle`, `sigma_mle`（スカラーテンソル）
- $\ell(\mu,\sigma) = \sum_i \log \mathcal{N}(x^{(i)}\mid\mu,\sigma)$ ↔ `D.Normal(mu_mle, sigma_mle).log_prob(data).sum()`
- $\mathcal{H}(\boldsymbol{\pi}) = -\sum_k \pi_k \log \pi_k$ ↔ `D.Categorical(probs=pi).entropy()`
- $D_{\mathrm{KL}}(p \| q)$ ↔ `D.kl_divergence(p, q)`（登録済みペアに対して閉形式）

検算: (i) MLE が対数尤度を最大化すること（摂動後の尤度が下がる）、(ii) 一様分布のエントロピー $= \log K$、(iii) $D_{\mathrm{KL}}(p\|p) = 0$、の3点でそれぞれ assert する。

$$
\ell(\mu, \sigma) = \sum_{i=1}^{N} \log \mathcal{N}(x^{(i)} \mid \mu, \sigma)
= -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x^{(i)}-\mu)^2
$$

```python
import torch
import torch.distributions as D

torch.manual_seed(42)

# --- Block 1 / 3: torch.distributions — Normal, Categorical, MultivariateNormal ---

# Gaussian MLE: mu_hat = x.mean(),  sigma_hat = x.std(unbiased=False)
mu_true, sigma_true = torch.tensor(2.0), torch.tensor(1.5)
data = D.Normal(mu_true, sigma_true).sample((500,))      # (500,)

mu_mle    = data.mean()                                  # mu
sigma_mle = data.std(unbiased=False)                     # sigma (biased MLE, ddof=0)

# log p(D) = sum_i log N(x_i | mu, sigma)
ll_mle       = D.Normal(mu_mle,       sigma_mle).log_prob(data).sum()
ll_perturbed = D.Normal(mu_mle + 0.1, sigma_mle).log_prob(data).sum()
assert ll_mle > ll_perturbed                             # MLE is the argmax

# Categorical: H[Uniform(K)] = log K
pi_uniform = torch.full((5,), 1.0 / 5)
H_cat = D.Categorical(probs=pi_uniform).entropy()        # scalar
assert abs(H_cat - torch.log(torch.tensor(5.0))) < 1e-5  # H = log K

# MultivariateNormal: already shown — torch handles Cholesky internally
mu_mv  = torch.zeros(2)
cov_mv = torch.tensor([[2.0, 0.8], [0.8, 1.0]])
dist_mv = D.MultivariateNormal(loc=mu_mv, covariance_matrix=cov_mv)
x0 = torch.tensor([1.0, -1.0])
print(f"log N(x0|mu,Sigma) = {dist_mv.log_prob(x0):.6f}")

# KL divergence (closed form for registered pairs)
p = D.Normal(0.0, 1.0)
q = D.Normal(1.0, 2.0)
kl_pq = D.kl_divergence(p, q)                           # KL[N(0,1) || N(1,2)]
assert kl_pq > 0 and D.kl_divergence(p, p) < 1e-6       # KL >= 0, KL(p||p)=0

print(f"mu_mle={mu_mle:.4f}, sigma_mle={sigma_mle:.4f}")
print(f"H[Uniform(5)]={H_cat:.4f}, log(5)={torch.log(torch.tensor(5.0)):.4f}")
print(f"KL[N(0,1)||N(1,2)]={kl_pq:.4f}")
```

**Bernoulli → Categorical: 離散分布の系譜**

$$
P(X=k \mid \mathbf{p}) = p_k, \quad k \in \{1,\ldots,K\},\quad \sum_k p_k = 1
$$

Bernoulli は $K=2$ の特殊ケース。Softmax が Categorical の出力層になる理由: $\mathbf{p} = \text{softmax}(\mathbf{z})$ とすれば $\sum_k p_k = 1$ が自動的に満たされる。

MLE: $N$ 個の観測 $x^{(1)},\ldots,x^{(N)}$ から:

$$
\hat{p}_k = \frac{\#\{i : x^{(i)} = k\}}{N}
$$

カウントを $N$ で割るだけ。交差エントロピー損失 $-\sum_k y_k \log p_k$ の最小化 = Categorical MLE だ。

**大数の法則 (LLN) と中心極限定理 (CLT) — 数値検証**

理論的に保証されているが、具体的にどう収束するか数値で確認する。

LLN: $\bar{X}_N \xrightarrow{P} \mu$（確率収束）

$$
P(|\bar{X}_N - \mu| > \epsilon) \leq \frac{\sigma^2}{N \epsilon^2}
$$

CLT: $\sqrt{N}(\bar{X}_N - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$（分布収束）

$$
Z_N = \frac{\bar{X}_N - \mu}{\sigma/\sqrt{N}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

記号 ↔ 変数対応:
- $\bar{X}_N = \frac{1}{N}\sum_{i=1}^N X_i$ ↔ `X.mean(axis=1)` shape `(n_trials,)`
- $Z_N$（標準化標本平均）↔ `Z_N: (n_trials,)` → `N(0,1)` に収束
- $\text{KS}$（Kolmogorov-Smirnov検定量）↔ CLT収束の定量的評価

**収束の速さ — Berry-Esseen 定理**:

CLT は $Z_N \xrightarrow{d} \mathcal{N}(0,1)$ を保証するが「いつ収束するか」は述べない。Berry-Esseen 定理が定量化する:

$$
\sup_x \left| P(Z_N \leq x) - \Phi(x) \right| \leq \frac{C \rho}{\sigma^3 \sqrt{N}}, \quad C \leq 0.4748
$$

ここで $\rho = \mathbb{E}[|X - \mu|^3]$（三次絶対中心モーメント）。Exponential$(1)$ では $\mu=1$, $\sigma^2=1$, $\rho=\mathbb{E}[|X-1|^3]=2$ なので:

$$
\text{誤差上界} \leq \frac{0.4748 \times 2}{\sqrt{N}} = \frac{0.9496}{\sqrt{N}}
$$

$N=5$: 誤差 $\leq 0.424$（Exponential の歪度 = 2 が大きいため収束が遅い）。  
$N=500$: 誤差 $\leq 0.042$（KS 検定で有意差が検出されにくい水準）。

**歪度と収束速度**: $\rho/\sigma^3$ は分布の「歪み」を捉える。正規分布自体の歪度は 0 だが、金融収益率や自然言語の単語頻度は Power-law（Zipf の法則）に従い、三次以上のモーメントが無限大になることがある。そのような分布では CLT の収束が保証されず、正規近似は危険だ。

LLN の収束速度は Chebyshev 不等式から直接導ける:

$$
P(|\bar{X}_N - \mu| > \epsilon) \leq \frac{\sigma^2}{N\epsilon^2}
$$

これは $O(1/N)$ の確率保証だが、標本平均の標準偏差 $\sigma/\sqrt{N}$ を見ると実質的な精度は $O(1/\sqrt{N})$。**データを 100 倍にしても精度は 10 倍にしかならない** — これが大規模データ収集の「限界収益逓減」の数学的根拠だ。

**解釈**: Exponential 分布は右裾が重い（歪度 = 2）が、$N=500$ で標本平均の分布はほぼ正規分布に収束する。LLN 誤差は $N$ が増えるにつれ $O(1/\sqrt{N})$ で減少する。

**解釈**: Exponential分布は右裾が重いが、N=500で標本平均の分布はほぼ正規分布に収束する。LLN誤差はNが増えるにつれ $O(1/\sqrt{N})$ で減少 — Chebyshev不等式の $O(1/N)$ より速い（期待値の収束速度）。

**Softmax と Categorical の完全実装**:

$p_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$（Softmax = Categorical の自然パラメータ $\boldsymbol{\eta}$ から期待値パラメータ $\boldsymbol{\pi}$ への変換）

記号 ↔ 変数対応:
- $\mathbf{z}$（logit）↔ `z: (K,)`
- $\boldsymbol{\pi} = \text{softmax}(\mathbf{z})$ ↔ `pi: (K,)`, `sum=1`
- $\mathcal{H}(\boldsymbol{\pi}) = -\sum_k \pi_k \log \pi_k$（エントロピー）↔ `H: float`

**エントロピー最大化の数学 — Lagrange 乗数法**:

「制約のもとでエントロピーを最大化すると一様分布が得られる」を示す。

問題: $\max_{\boldsymbol{\pi}} \mathcal{H}(\boldsymbol{\pi}) = -\sum_{k=1}^{K} \pi_k \log \pi_k$ subject to $\sum_k \pi_k = 1$, $\pi_k \geq 0$

Lagrangian を構成し停留条件を取る:

$$
\mathcal{L} = -\sum_k \pi_k \log \pi_k + \lambda\!\left(\sum_k \pi_k - 1\right)
$$

$$
\frac{\partial \mathcal{L}}{\partial \pi_k} = -\log \pi_k - 1 + \lambda = 0 \implies \pi_k = e^{\lambda - 1}
$$

全 $k$ で同じ値 → 正規化条件 $\sum_k \pi_k = 1$ より $\pi_k = 1/K$。このとき $\mathcal{H} = \log K$。よって:

$$
\mathcal{H}(\boldsymbol{\pi}) \leq \log K, \quad \text{等号は } \boldsymbol{\pi} = (1/K, \ldots, 1/K) \text{ のとき}
$$

**一点への集中で $\mathcal{H} \to 0$**: $\pi_1 \to 1$（one-hot）とすると $-1 \cdot \log 1 - \sum_{k \geq 2} 0 \cdot \log 0 = 0$（$0 \log 0 = 0$ と定義）。これがエントロピー最小。

**数値安定 softmax の核心**:

$$
\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}, \quad c = \max_k z_k
$$

$c$ を引いても比は変わらない（分子・分母に $e^{-c}$ が共通因子）。$c = \max_k z_k$ とすると $e^{z_k - c} \leq 1$ が保証され `exp` がオーバーフローしない。$\log \text{softmax}(\mathbf{z})_k = z_k - c - \log \sum_j e^{z_j - c}$ が `F.log_softmax` の計算式だ。

この $\log \sum_j e^{z_j}$ が `torch.logsumexp(z, dim=-1)` であり、数値安定に $\log Z$ を計算する基本ツールだ。次の identity は常に成立する:

$$
\log \sum_j e^{z_j} = c + \log \sum_j e^{z_j - c}, \quad c = \max_k z_k
$$

**大数の法則の直感**: Bernoulli$(p)$ の標準偏差は $\sqrt{p(1-p)}$。$p=0.3$ で $\sigma \approx 0.458$。標本平均の標準誤差は $0.458/\sqrt{N}$。$N=10^4$ で $\approx 0.0046$ — A/B テストで「数千サンプル必要」と言われる根拠だ。精度を2倍にするにはデータが4倍必要という $O(1/\sqrt{N})$ の壁は LLN の本質的な限界だ。

### 5.2 多変量正規分布 — 完全実装と直感

1次元Gaussianの自然な拡張は、「変数間の相関」を捉える。

**定義**:

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) =
\frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}}
\exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$

- shape: `x` は `(d,)`, `mu` は `(d,)`, `Sigma` は `(d,d)` 正定値対称行列
- Mahalanobis距離 $D_M^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})$ は「楕円体の距離」
- $\boldsymbol{\Sigma}^{-1}$ の直接計算は避ける: `torch.linalg.solve(Sigma, x - mu)` か `D.MultivariateNormal` を使う

**条件付き分布** (Schur complement 公式):

変数を $[\mathbf{x}_1, \mathbf{x}_2]$ に分割すると:

$$
p(\mathbf{x}_1 \mid \mathbf{x}_2) = \mathcal{N}(\boldsymbol{\mu}_{1|2},\, \boldsymbol{\Sigma}_{1|2})
$$

$$
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)
$$

$$
\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}
$$

$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$ は「Kalman gain」の形。$\mathbf{x}_2$ を観測することで、$\mathbf{x}_1$ の不確実性 $\boldsymbol{\Sigma}_{1|2}$ は元の $\boldsymbol{\Sigma}_{11}$ より必ず小さくなる（半正定値の意味で）。

**MLE**: 全微分してゼロ点を解くと:

$$
\hat{\boldsymbol{\mu}} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}^{(i)}, \quad
\hat{\boldsymbol{\Sigma}} = \frac{1}{N}\sum_{i=1}^N (\mathbf{x}^{(i)} - \hat{\boldsymbol{\mu}})(\mathbf{x}^{(i)} - \hat{\boldsymbol{\mu}})^\top
$$

サンプル平均とサンプル共分散行列がそのままMLE解だ（1次元と同じ構造）。


**Cholesky分解による安定実装**:

$\boldsymbol{\Sigma}$ が正定値 → $\boldsymbol{\Sigma} = LL^\top$ の Cholesky 分解が存在する。

$$
\log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) =
-\frac{d}{2}\log 2\pi - \frac{1}{2}\log|\boldsymbol{\Sigma}|
- \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})
$$

記号 ↔ 変数対応（PyTorch では Block 1 の `D.MultivariateNormal` が担う）:
- $\boldsymbol{\mu}$ ↔ `mu_mv: (d,)` テンソル
- $\boldsymbol{\Sigma}$ ↔ `cov_mv: (d,d)` 正定値対称テンソル
- $\log |\boldsymbol{\Sigma}| = 2\sum_i \log L_{ii}$（Cholesky 因子の対角積）↔ `dist_mv.log_prob(x)` に内包
- Mahalanobis 距離 $D_M^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ ↔ `v @ v`（`v = L^{-1}(x-mu)`）

shape: `x: (d,)`, `mu: (d,)`, `Sigma: (d,d)`, 出力 `log_prob: scalar`

**Cholesky 安定化の理由**: $\boldsymbol{\Sigma}^{-1}$ を直接計算すると数値誤差が $O(\kappa^2(\boldsymbol{\Sigma}))$ で増幅される（$\kappa$ = 条件数）。Cholesky 分解 $\boldsymbol{\Sigma} = LL^\top$ を経由すると:
- $\log|\boldsymbol{\Sigma}| = 2\sum_i \log L_{ii}$（対角成分の対数和）
- $\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = L^{-\top}L^{-1}(\mathbf{x}-\boldsymbol{\mu})$（前進代入 + 後退代入）

直接逆行列を求めるより数値誤差が $O(\kappa)$ で抑えられる。PyTorch の `D.MultivariateNormal` は内部で Cholesky 因子をキャッシュし、同じ分布で複数の `log_prob` 評価を行う場合に効率的だ。

**MLE と正定値制約**: $N$ 個の $d$ 次元サンプルから $\hat{\boldsymbol{\Sigma}} = \frac{1}{N}\sum_i (\mathbf{x}^{(i)}-\hat{\boldsymbol{\mu}})(\mathbf{x}^{(i)}-\hat{\boldsymbol{\mu}})^\top$ を計算する場合、$N < d$ では行列のランクが $N$ になり半正定値（$\hat{\boldsymbol{\Sigma}} \succeq 0$ だが $\hat{\boldsymbol{\Sigma}} \not\succ 0$）。Cholesky 分解が失敗する。

回避策: $\hat{\boldsymbol{\Sigma}}_\text{reg} = \hat{\boldsymbol{\Sigma}} + \epsilon \mathbf{I}$（$\epsilon \sim 10^{-6}$）で正則化。VAE のエンコーダ出力 $\boldsymbol{\Sigma}_\phi = \text{diag}(\boldsymbol{\sigma}_\phi^2)$ が対角行列に限定されるのも、フルランク共分散の推定困難を回避するためだ。

**Block 1 での確認**: 上の PyTorch ブロックで `D.MultivariateNormal(mu_mv, cov_mv).log_prob(x0)` が Cholesky 経由で安定に計算されることを示した。`covariance_matrix` に代えて `scale_tril=L`（Cholesky 因子直接渡し）も使える — 既に Cholesky 分解済みの場合は後者が効率的だ。

**条件付き分布**:

$$
\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)
$$

$$
\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}
$$

$\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$ は Kalman gain と同型。$\mathbf{x}_2$ を観測すると分散は必ず縮む: $\boldsymbol{\Sigma}_{1|2} \preceq \boldsymbol{\Sigma}_{11}$（半正定値順序）。

**Schur 補行列公式の導出**:

ブロック行列の逆行列を使う。$\boldsymbol{\Sigma} = \begin{pmatrix}\boldsymbol{\Sigma}_{11}&\boldsymbol{\Sigma}_{12}\\\boldsymbol{\Sigma}_{21}&\boldsymbol{\Sigma}_{22}\end{pmatrix}$ の逆行列のブロック $(1,1)$ 成分が $(\boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21})^{-1}$。これが Schur 補行列 $\boldsymbol{\Sigma}_{1|2}^{-1}$ だ。

結合正規分布の定義から条件付き分布を導く:

$$
\begin{aligned}
\log p(\mathbf{x}_1, \mathbf{x}_2) &= -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) + \text{const} \\
p(\mathbf{x}_1 \mid \mathbf{x}_2) &\propto p(\mathbf{x}_1, \mathbf{x}_2) \quad (\mathbf{x}_2 \text{ 固定})
\end{aligned}
$$

$\mathbf{x}_2 = \mathbf{a}$ を固定して $\mathbf{x}_1$ についての二次形式を整理すると:

$$
-\frac{1}{2}\bigl(\mathbf{x}_1 - \boldsymbol{\mu}_{1|2}\bigr)^\top \boldsymbol{\Sigma}_{1|2}^{-1}\bigl(\mathbf{x}_1 - \boldsymbol{\mu}_{1|2}\bigr) + \text{const}
$$

これは $\mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$ の対数密度だ。

**分散が縮む理由**: $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ において、引かれる項 $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$ は半正定値（$\mathbf{v}^\top \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}\mathbf{v} = \|\boldsymbol{\Sigma}_{22}^{-1/2}\boldsymbol{\Sigma}_{21}\mathbf{v}\|^2 \geq 0$）。よって $\boldsymbol{\Sigma}_{1|2} \preceq \boldsymbol{\Sigma}_{11}$（半正定値順序）— **観測するほど不確実性は必ず減少する**。

具体例: $\boldsymbol{\Sigma} = \begin{pmatrix}2 & 0.8 \\ 0.8 & 1\end{pmatrix}$, $\mathbf{x}_2 = -1$ を観測すると:

$$
\boldsymbol{\mu}_{1|2} = 1 + 0.8 \cdot 1^{-1} \cdot (-1 - (-2)) = 1 + 0.8 = 1.8
$$

$$
\boldsymbol{\Sigma}_{1|2} = 2 - 0.8^2/1 = 2 - 0.64 = 1.36 < 2 = \boldsymbol{\Sigma}_{11}
$$

正の相関 $\rho = 0.8/\sqrt{2 \cdot 1} \approx 0.566$ があるため、$x_2$ の観測が $x_1$ の予測を上方修正し、不確実性を $2 \to 1.36$（32\%削減）する。この公式は Kalman フィルタの更新式と同型であり、GPGPU 上の状態推定から VAE の事後分布計算まで広く使われる。

### 5.3 指数型分布族 — 統一的記述

Gaussian, Bernoulli, Poisson, Gamma... 一見バラバラに見える分布が「同じ文法」で書ける。

**標準形**:

$$
p(x \mid \boldsymbol{\eta}) = h(x) \exp\!\left(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta})\right)
$$

- $\boldsymbol{\eta}$: 自然パラメータ（natural parameter）
- $T(x)$: 十分統計量（sufficient statistic）— データの「要約」
- $A(\boldsymbol{\eta})$: 対数分配関数（log partition function）— 正規化定数

**Gaussian の場合** ($d=1$):

$$
\boldsymbol{\eta} = \begin{pmatrix}\mu/\sigma^2 \\ -1/(2\sigma^2)\end{pmatrix},\quad
T(x) = \begin{pmatrix}x \\ x^2\end{pmatrix},\quad
A(\boldsymbol{\eta}) = -\frac{\eta_1^2}{4\eta_2} + \frac{1}{2}\log\frac{\pi}{-\eta_2}
$$

**MLEの美しさ**: 指数型分布族のMLEは「理論的期待値 = 経験的期待値」という条件:

$$
\mathbb{E}_{p(x|\hat{\boldsymbol{\eta}})}[T(x)] = \frac{1}{N}\sum_{i=1}^N T(x^{(i)})
$$

Gaussianなら $T(x) = (x, x^2)$ なので、平均と二乗平均が一致する条件 = サンプル平均・分散がMLE。

**共役事前分布**: 事前分布を $p(\boldsymbol{\eta}) = h(\boldsymbol{\eta})\exp(\boldsymbol{\chi}^\top \boldsymbol{\eta} - \nu A(\boldsymbol{\eta}))$ と書くと、事後分布が同じ族に属する（共役性）。Gaussian-Gaussian 共役、Beta-Bernoulli 共役 はこの特殊ケース。


**指数型分布族の統一実装**:

抽象的に見えるが、Gaussian/Bernoulli/Poissonが同じクラスで書けることを確認する。

記号 ↔ 変数対応:
- $\boldsymbol{\eta}$（自然パラメータ）↔ `eta: (k,)` where $k$ = 十分統計量の次元
- $T(x)$（十分統計量）↔ `suff_stat(x)`: 充分な情報を持つ「データの圧縮表現」
- $A(\boldsymbol{\eta})$（対数分配関数）↔ `log_partition(eta)`: 正規化定数の対数
- MLE 条件 $\mathbb{E}[T(x)] = \overline{T}$ ↔ 経験的十分統計量と理論的期待値の一致

shape: Gaussian の場合 `eta: (2,)`, `T(x): (2,)` = $(x, x^2)$

**対数分配関数 $A(\boldsymbol{\eta})$ の3つの役割**:

1. **正規化**: $A(\boldsymbol{\eta})$ は $\int p(x \mid \boldsymbol{\eta}) dx = 1$ を保証する。

2. **期待値生成**: $\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \mathbb{E}_{p}[T(x)]$。Gaussian では $\partial_{\eta_1} A = \mu$（期待値）、$\partial_{\eta_2} A = \mu^2 + \sigma^2$（二次モーメント）。

3. **分散・共分散生成**: $\nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \text{Cov}[T(x)] \succeq 0$。$A$ が凸であることの直接の証明だ。

**$A$ の凸性と MLE の大域最適性**:

$A$ が凸 $\implies$ $-\sum_i \log p(x^{(i)} \mid \boldsymbol{\eta}) = \sum_i A(\boldsymbol{\eta}) - \boldsymbol{\eta}^\top T(x^{(i)}) + \text{const}$ も凸（$\boldsymbol{\eta}$ の線形項を引いた凸関数）。よって局所解 = 大域解。これが指数型分布族の「学習しやすさ」の本質だ。

**Gaussian の具体計算**:

$\boldsymbol{\eta} = (\eta_1, \eta_2) = (\mu/\sigma^2,\; -1/(2\sigma^2))$ から逆変換:

$$
\sigma^2 = -\frac{1}{2\eta_2}, \quad \mu = \eta_1 \cdot \sigma^2 = -\frac{\eta_1}{2\eta_2}
$$

対数分配関数:

$$
A(\boldsymbol{\eta}) = -\frac{\eta_1^2}{4\eta_2} + \frac{1}{2}\log\frac{\pi}{-\eta_2}
$$

これを $\eta_1$ で微分すると $-\eta_1/(2\eta_2) = \mu = \mathbb{E}[X]$、$\eta_2$ で微分すると $\eta_1^2/(4\eta_2^2) + 1/(2\eta_2) = \mu^2 + \sigma^2 = \mathbb{E}[X^2]$。

MLE 条件「$\mathbb{E}[T(x)] = \frac{1}{N}\sum T(x^{(i)})$」は Gaussian では $(\mathbb{E}[X], \mathbb{E}[X^2]) = (\bar{x}, \overline{x^2})$、つまりサンプル平均と二乗平均が一致する — これが MLE 解 $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \overline{x^2} - \bar{x}^2$ と等価だ。

**自然勾配法 (Natural Gradient) へのプレビュー**:

指数型分布族のパラメータ空間は Fisher 情報行列 $\mathbf{I}(\boldsymbol{\eta})$ が計量を与える「Riemannian 多様体」だ。通常の勾配降下と自然勾配降下の違い:

$$
\text{通常}: \boldsymbol{\eta}_{t+1} = \boldsymbol{\eta}_t - \alpha \nabla_{\boldsymbol{\eta}} \mathcal{L}, \quad
\text{自然勾配}: \boldsymbol{\eta}_{t+1} = \boldsymbol{\eta}_t - \alpha \mathbf{I}^{-1}(\boldsymbol{\eta}_t) \nabla_{\boldsymbol{\eta}} \mathcal{L}
$$

指数型分布族の特別な性質: 自然パラメータ空間での自然勾配 = 期待値パラメータ空間 $\boldsymbol{\mu} = \mathbb{E}[T(x)]$ での通常勾配。つまり $\mathbf{I}^{-1} \nabla_{\boldsymbol{\eta}} = \nabla_{\boldsymbol{\mu}}$。これが Adam などの適応的最適化の理論的基盤だ（第12回で詳説）。

**なぜ対数分配関数 $A(\boldsymbol{\eta})$ が重要か**: $A$ の一次微分が期待値、二次微分が共分散を与える。

$$
\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \mathbb{E}_{p(x|\boldsymbol{\eta})}[T(x)]
$$

$$
\nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \text{Cov}_{p}[T(x), T(x)] \succeq 0
$$

$A$ が凸 → 負の対数尤度も凸 → MLEは大域的最適解。これが指数型分布族の「学習しやすさ」の本質だ。

**自然勾配法 (Natural Gradient) へのプレビュー**:

指数型分布族のパラメータ空間は「Riemannian多様体」だ。Fisher情報行列 $\mathbf{I}(\boldsymbol{\eta})$ がその空間の計量を与える。

通常の勾配降下: $\boldsymbol{\eta}_{t+1} = \boldsymbol{\eta}_t - \alpha \nabla_{\boldsymbol{\eta}} \mathcal{L}$

自然勾配降下: $\boldsymbol{\eta}_{t+1} = \boldsymbol{\eta}_t - \alpha \mathbf{I}^{-1}(\boldsymbol{\eta}_t) \nabla_{\boldsymbol{\eta}} \mathcal{L}$

自然勾配は「パラメータ空間の距離」ではなく「分布空間のKL距離」でステップを制御する。同じ分布の変化量に対応するステップが、パラメータの値に依存しない — これがAdamなどの適応的最適化の理論的基盤だ（第12回で詳説）。

指数型分布族では自然勾配に閉形式がある: $\mathbf{I}^{-1}(\boldsymbol{\eta}) \nabla_{\boldsymbol{\eta}} \mathcal{L} = \nabla_{\boldsymbol{\mu}} \mathcal{L}$（期待値パラメータ空間の通常勾配と等価）。

### 5.4 実装演習: ガウス混合モデル（GMM）のMLE

第8回（EM算法）への橋渡しとして、2成分GMMのフィッティングを実装する。ここではEM算法の前段階として、単一ガウスのMLEを拡張する形で問題の困難さを体感する。

$$
\begin{aligned}
p(x \mid \theta) &= \pi\,\mathcal{N}(x \mid \mu_1, \sigma_1^2) + (1-\pi)\,\mathcal{N}(x \mid \mu_2, \sigma_2^2) \\[6pt]
\ell(\theta) &= \sum_{i=1}^{N} \log p(x_i \mid \theta) \\[6pt]
\mathcal{N}(x \mid \mu, \sigma^2) &= \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\end{aligned}
$$

**なぜ GMM の MLE は閉じた形で解けないのか**: 対数尤度に**和の対数** $\log[\pi \mathcal{N}_1 + (1-\pi)\mathcal{N}_2]$ が現れ、対数と和の順序を入れ替えられない。微分してもパラメータが互いに絡み合う:

$$
\frac{\partial \ell}{\partial \mu_1} = \sum_i \frac{\pi \mathcal{N}(x_i|\mu_1,\sigma_1^2)}{\pi \mathcal{N}(x_i|\mu_1,\sigma_1^2) + (1-\pi)\mathcal{N}(x_i|\mu_2,\sigma_2^2)} \cdot \frac{x_i - \mu_1}{\sigma_1^2} = 0
$$

右辺の分数 $r_i = P(\text{成分1} \mid x_i, \theta)$ は **責任度（responsibility）** と呼ばれる。$\mu_1$ の式が $r_i$ に依存し、$r_i$ が $\mu_2, \sigma_2, \pi$ に依存する → 全パラメータが連立する。

これを解く反復アルゴリズムが EM 算法（第8回）だ:

1. **E ステップ**: 現在の $\theta$ で $r_i$ を計算  
2. **M ステップ**: $r_i$ を固定して各パラメータを個別に更新  
   - $\hat{\mu}_1 = \sum_i r_i x_i / \sum_i r_i$（責任度で重み付けした標本平均）

単一 Gaussian との対数尤度の差（gap）が GMM の「モデル表現力の利得」を数値化する。$\text{gap} = \ell_\text{GMM} - \ell_\text{single} > 0$ は混合モデルが必要な証拠だ。大きいほど単峰分布の仮定が誤りだった度合いを示す。

**なぜGMMのMLEは閉じた形で解けないのか**: 対数尤度の中に**和の対数** $\log[\pi \mathcal{N}(x \mid \mu_1, \sigma_1^2) + (1-\pi)\mathcal{N}(x \mid \mu_2, \sigma_2^2)]$ が現れる。対数と和の順序を入れ替えられないため、微分しても各パラメータが互いに絡み合う。この困難が第8回のEM算法の動機だ。

### 5.5a 実装演習: ベイズ推論のグリッド近似

$$
\begin{aligned}
\theta &\sim \mathrm{Beta}(a,b), \quad x_i \sim \mathrm{Bernoulli}(\theta) \\[4pt]
p(\theta \mid \mathbf{x}) &\propto \theta^{a+h-1}(1-\theta)^{b+t-1} \\[4pt]
\theta \mid \mathbf{x} &\sim \mathrm{Beta}(a+h,\; b+t)
\end{aligned}
$$

ここで $h = \sum_i x_i$（表の回数）、$t = N - h$（裏の回数）。事前 Beta$(1,1)$（一様）から始め、データを観測するごとに指数 $(a,b)$ が $(a+h, b+t)$ に更新される。尤度関数 $L(\theta) = \theta^h (1-\theta)^t$ が Beta 分布と「同じ形」になっているのが共役性の核心だ。

**グリッド近似が実用的でない理由 — 次元の呪い**:

$d$ 次元のパラメータに各軸 $M$ 点のグリッドを張ると $M^d$ 点が必要だ:

| $d$ | $M=10$ | $M=100$ | メモリ（float64）|
|:---:|:-------:|:--------:|:----------------:|
| 2 | $10^2$ | $10^4$ | 80 KB |
| 5 | $10^5$ | $10^{10}$ | 80 GB |
| 10 | $10^{10}$ | $10^{20}$ | ≫ 宇宙の原子数 |

10次元で各軸100点は $10^{20}$ 点 — 物理的に不可能。これが**次元の呪い**だ。

**代替手法の三本柱**:

1. **MCMC（Markov Chain Monte Carlo）**: 事後分布から直接サンプリング。Metropolis-Hastings や Hamiltonian Monte Carlo (HMC) が代表例。次元が増えても（ある意味）スケールする — 第5回で詳説。

2. **変分推論（Variational Inference）**: 事後分布 $p(\theta|\mathbf{x})$ を簡単な族 $q_\phi(\theta)$ で近似し、$D_{\mathrm{KL}}(q_\phi \| p)$ を最小化（= ELBO 最大化）。VAE の核心。

3. **Laplace 近似**: 事後分布を最頻値（MAP）周りで二次近似する。MAP 推定 + Hessian でガウス近似を得る。大次元でも計算可能だが、マルチモーダルな事後分布に弱い。

**事後一致性（posterior consistency）**: 正則条件のもとで $N \to \infty$ のとき事後分布は真のパラメータ $\theta^*$ に集中する。事前分布の影響が薄れ、ベイズ推定は MLE に収束する — 「事前分布は正則化の一形態」と割り切れる理由だ。Beta-Bernoulli では $\hat{\theta}_\text{Bayes} = (a+h)/(a+b+N) \to h/N = \hat{\theta}_\text{MLE}$ as $N \to \infty$。

> **Note:** **実装の教訓**: データが増えるほど、事前分布の影響は薄れ、ベイズ推定はMLEに近づく。これは事後分布が「尤度に支配される」ため。逆に、データが少ないときは事前分布が結果を大きく左右する。

この現象を「事後一致性（posterior consistency）」と呼ぶ。$N \to \infty$ で事後分布は真のパラメータに集中する — 大数の法則のベイズ版だ。

### 5.5b 実装演習: 共役事前分布の解析的更新

グリッド近似が「数値的」ならば、共役事前分布は「解析的」だ。

**Gaussian-Gaussian 共役（既知分散、未知平均）**:

事前: $\theta \sim \mathcal{N}(\mu_0, \tau_0^2)$、尤度: $X_i \mid \theta \sim \mathcal{N}(\theta, \sigma^2)$

$$
\frac{1}{\tau_N^2} = \frac{1}{\tau_0^2} + \frac{N}{\sigma^2}, \quad
\mu_N = \tau_N^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{N \bar{x}}{\sigma^2}\right)
$$

精度（分散の逆数）が加法的に更新される。$N \to \infty$ で $\mu_N \to \bar{x}$（MLE）、$\tau_N^2 \to 0$。

記号 ↔ 変数対応:
- $\mu_0, \tau_0^2$ ↔ 事前分布の平均・分散（先験的知識）
- $\sigma^2$ ↔ 尤度の分散（既知と仮定）
- $\bar{x}, N$ ↔ 標本平均・サンプル数
- $\mu_N, \tau_N^2$ ↔ 事後分布の平均・分散（観測後の更新された信念）

**精度（precision）の加法性**:

精度 $\lambda = 1/\tau^2$（分散の逆数）で書くと更新式は美しくなる:

$$
\lambda_N = \underbrace{\lambda_0}_{\text{事前の精度}} + \underbrace{\frac{N}{\sigma^2}}_{\text{データの精度}}
$$

精度は**加法的に更新される**。1個のデータが $1/\sigma^2$ の精度を追加する — 精度の積み重ねが信念の強化だ。

$N \to \infty$: $\lambda_N \to \infty$（精度無限大 → 分散0 → 確信に収束）、$\mu_N \to \bar{x}$（MLE に収束）。  
事前分布 $\tau_0^2 \to \infty$（無情報）: $\mu_N \to \bar{x}$、$\tau_N^2 \to \sigma^2/N$（MLE の標準誤差の二乗）。

**強事前 vs 弱事前の比較**:

$\mu_0 = 0$（事前の信念：平均は0）、真値 $\theta^* = 3$:
- 強事前（$\tau_0^2 = 0.5$）: $N=1$ では事前に強く引っ張られ $\mu_N \approx 0.5$。$N=100$ で $\mu_N \approx 2.9$（ほぼ収束）
- 弱事前（$\tau_0^2 = 100$）: 小 $N$ でも MLE に近い値。事前の影響が最初から薄い

これが「L2 正則化 = Gaussian 事前分布」の直感だ。正則化係数 $\lambda$ は事前精度 $\lambda_0 = \lambda \sigma^2$ に対応する。正則化を強くする（$\lambda \uparrow$）= 事前分布を強くする（$\tau_0^2 \downarrow$）= データより事前知識を信じる。

**3推定量の比較**:

| 推定量 | 式 | 特徴 |
|:-------|:---|:-----|
| MLE | $\bar{x}$ | バイアスなし、小データ不安定 |
| MAP | $\mu_N$ | 事前+尤度、正則化と等価 |
| 事後平均 | $\mu_N$（Gaussian事後）| MAP=事後平均 |

### 5.5a KLダイバージェンス — 分布間の「距離」実装

KLダイバージェンスは確率論の全ての武器が集結する場所だ。VAEのELBO、diffusion modelの目的関数、情報理論の基礎 — 全てここに通じる。

$$
D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx = \mathbb{E}_{p}\left[\log \frac{p(X)}{q(X)}\right]
$$

**基本性質**:
- $D_{\mathrm{KL}}(p \| q) \geq 0$（Gibbs不等式、Jensen不等式から）
- $D_{\mathrm{KL}}(p \| q) = 0 \iff p = q$（ほぼ至る所で）
- 非対称: $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$（距離公理を満たさない）

**2つのGaussian間のKL（閉形式）**:

$$
D_{\mathrm{KL}}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) =
\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

記号 ↔ 変数対応:
- $\mu_1, \sigma_1^2$ ↔ `mu1, var1`（分布 $p$）
- $\mu_2, \sigma_2^2$ ↔ `mu2, var2`（分布 $q$）
- $D_{\mathrm{KL}}(p\|q)$ ↔ `kl_pq`（スカラー、非負）

shape: scalar inputs → scalar output。Block 1 では `D.kl_divergence(p, q)` が閉形式を自動計算する。

**非負性の証明 — Gibbs 不等式**:

$\log x \leq x - 1$（$x > 0$、等号は $x=1$ のみ）を使う:

$$
-D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p\!\left[\log \frac{q(X)}{p(X)}\right] \leq \mathbb{E}_p\!\left[\frac{q(X)}{p(X)} - 1\right] = \int q(x)\,dx - \int p(x)\,dx = 1 - 1 = 0
$$

よって $D_{\mathrm{KL}}(p \| q) \geq 0$、等号は $p = q$ のとき（$\log(q/p) = 0$ a.e.）。

**閉形式の導出（Gaussian-Gaussian）**:

$$
\begin{aligned}
D_{\mathrm{KL}}(\mathcal{N}_1 \| \mathcal{N}_2) &= \int p_1 \log \frac{p_1}{p_2}\,dx \\
&= \int p_1 \left[\log \frac{\sigma_2}{\sigma_1} + \frac{(x-\mu_1)^2}{2\sigma_1^2} - \frac{(x-\mu_2)^2}{2\sigma_2^2}\right] dx \\
&= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{2\sigma_2^2} + \frac{(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
\end{aligned}
$$

3行目で $\mathbb{E}_{p_1}[(X-\mu_1)^2] = \sigma_1^2$、$\mathbb{E}_{p_1}[(X-\mu_2)^2] = \sigma_1^2 + (\mu_1-\mu_2)^2$ を使った。

**非対称性の直感**: $D_{\mathrm{KL}}(p \| q)$ は「$p$ でサンプリングしつつ $q$ との違いを測る」。$q$ の裾が軽くて $p$ の裾が重い場合（$q(x) \ll p(x)$ で $p(x) > 0$）、$\log(p/q) \to +\infty$ となり KL が爆発する。逆方向 $D_{\mathrm{KL}}(q \| p)$ は $q$ でサンプリングするため、この爆発は起きない。

VAE の ELBO で $D_{\mathrm{KL}}(q_\phi \| p)$ を使う（$p$ を外側に置く）のは、$q_\phi$ の領域外での爆発を避けるためだ — 「mode-seeking」と呼ばれる性質。

**Block 1 での確認**: `D.kl_divergence(p, p)` が $10^{-6}$ 以下（数値精度の範囲で 0）になることを assert で検証した。

**VAEとの接続**: VAEのELBOには $D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ が登場する。$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$、$q_\phi = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ なら、次元独立なGaussian KLの閉形式が使える:

$$
D_{\mathrm{KL}}(q \| p) = \frac{1}{2} \sum_{j=1}^d (\sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2)
$$

第8回（VAE）でこの式が損失関数に直接現れる。

### 5.5c Fisher情報量 — Cramér-Rao下界の実装検証

Fisher情報量 $I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p(x;\theta)}{\partial \theta}\right)^2\right]$ は推定の難しさを定量化する。

等価な表現（対数尤度の曲率）:

$$
I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log p(x; \theta)}{\partial \theta^2}\right]
$$

**Cramér-Rao下界**: 任意の不偏推定量の分散は $1/(n I(\theta))$ より小さくできない:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n \cdot I(\theta)}
$$

記号 ↔ 変数対応:
- $\theta$ ↔ `theta: float`（推定対象パラメータ）
- スコア関数 $s(x;\theta) = \partial_\theta \log p(x;\theta)$ ↔ スカラー関数（各データ点で評価）
- $I(\theta) = \mathbb{E}[s^2]$ ↔ スコアの二乗期待値
- CR 下界 $1/(nI(\theta))$ ↔ 任意の不偏推定量の分散の下限

shape: `score(x): (N,)`, Fisher info: scalar, CR bound: scalar

**Cramér-Rao 下界の導出スケッチ**:

任意の不偏推定量 $\hat{\theta}(X)$（$\mathbb{E}[\hat{\theta}] = \theta$）について:

$$
\frac{\partial}{\partial \theta}\mathbb{E}[\hat{\theta}] = 1 \implies \int \hat{\theta}(x) \frac{\partial}{\partial \theta} p(x;\theta)\,dx = 1
$$

$\frac{\partial \log p}{\partial \theta} = \frac{1}{p}\frac{\partial p}{\partial \theta}$ を代入:

$$
\mathbb{E}[\hat{\theta}(X) \cdot s(X;\theta)] = 1
$$

スコアの期待値ゼロ（$\mathbb{E}[s] = 0$）から $\text{Cov}(\hat{\theta}, s) = \mathbb{E}[\hat{\theta} \cdot s] = 1$。Cauchy-Schwarz 不等式より:

$$
1 = \text{Cov}(\hat{\theta}, s)^2 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(s) = \text{Var}(\hat{\theta}) \cdot I(\theta)
$$

よって $\text{Var}(\hat{\theta}) \geq 1/I(\theta)$。$n$ 個の i.i.d. サンプルなら Fisher 情報が加法的（$I_n = n I(\theta)$）なので $\text{Var}(\hat{\theta}_n) \geq 1/(n I(\theta))$。

**Gaussian MLE の効率性**: $\hat{\mu} = \bar{X}$ の分散は $\sigma^2/n$。Fisher 情報 $I(\mu) = 1/\sigma^2$ なので CR 下界は $1/(n \cdot 1/\sigma^2) = \sigma^2/n$。比率 = 1.0 — 標本平均は **Fisher 効率的推定量**（CR 下界を達成する）。

**CR 下界が達成されない例**: 分散 $\sigma^2$ の推定では、標本分散 $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$ の分散は $2\sigma^4/n$、CR 下界は $2\sigma^4/n$（Fisher 情報 $I(\sigma^2) = n/(2\sigma^4)$）— 同じく効率的。しかし、分散の平方根 $\hat{\sigma}$ の推定量は一般に CR 下界を達成しない（デルタ法による変換で下界が変わるため）。

**Fisher 情報行列（多次元）**: $\mathbf{I}(\boldsymbol{\theta})_{ij} = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]$。これが Riemannian 計量になり、パラメータ空間の「KL 幾何学」を定義する。自然勾配 $\tilde{\nabla} = \mathbf{I}^{-1} \nabla$ はこの幾何学に沿った最急降下だ。

**検証**: 標本平均はCramér-Rao下界を**ぴったり達成**する（Fisher効率的推定量）。比率が全て≈1.0になる。

**スコアの期待値はゼロ**: $\mathbb{E}[s(X;\theta)] = 0$。$\int p(x;\theta) dx = 1$ を $\theta$ で微分すると導ける（正規化条件の微分）。Fisher情報量はスコアの分散だ。

$$
\mathbb{E}[s] = \int \frac{\partial \log p}{\partial \theta} p \, dx = \frac{\partial}{\partial \theta} \int p \, dx = \frac{\partial}{\partial \theta} 1 = 0
$$

**多次元Fisher情報行列 (FIM)**: $\mathbf{I}(\boldsymbol{\theta})_{ij} = \mathbb{E}[\partial_i \log p \cdot \partial_j \log p]$。自然勾配法 $\tilde{\nabla}_\theta \mathcal{L} = \mathbf{I}^{-1} \nabla_\theta \mathcal{L}$ はFIMでパラメータ空間の曲率を補正し、確率多様体上の最適解に最短経路で到達する。

### 5.6 モーメント母関数と特性関数

**モーメント母関数（MGF）**: $M_X(t) = \mathbb{E}[e^{tX}]$

MGFの $k$ 次微分は $k$ 次モーメントを与える: $M_X^{(k)}(0) = \mathbb{E}[X^k]$


MGFが存在しない分布もある（Cauchy分布など）。その場合は**特性関数** $\varphi_X(t) = \mathbb{E}[e^{itX}]$ を使う。特性関数は常に存在し、分布を一意に決定する。CLTの証明はしばしば特性関数を用いて行われる。

Gaussianの場合: $M_X(t) = \exp(\mu t + \frac{\sigma^2 t^2}{2})$。

**独立和の性質**: $X, Y$ が独立なら $M_{X+Y}(t) = M_X(t) M_Y(t)$。これがCLT証明の核心だ — サンプル和の特性関数が元の特性関数の積になり、$N \to \infty$ で正規分布の特性関数に収束する。

$$
M_X(t) = \mathbb{E}[e^{tX}] = \int e^{tx} p(x) \, dx
$$

記号 ↔ 変数対応:
- $t$ ↔ Laplace 変換の変数（実数）
- $M_X^{(k)}(0) = \mathbb{E}[X^k]$ ↔ $k$ 次微分で $k$ 次モーメントを取り出せる
- 特性関数 $\varphi_X(t) = M_X(it)$ ↔ $t$ を虚数軸にした MGF の解析接続

**CLT の証明スケッチ（特性関数経由）**:

$S_N = \frac{1}{\sqrt{N}}\sum_{i=1}^N (X_i - \mu)/\sigma$ の特性関数を計算する:

$$
\varphi_{S_N}(t) = \left[\varphi_{(X-\mu)/\sigma}\!\!\left(\frac{t}{\sqrt{N}}\right)\right]^N
$$

$(X-\mu)/\sigma$ の特性関数を $u = t/\sqrt{N}$ 周りでテイラー展開（$u \to 0$ として）:

$$
\varphi(u) = 1 + iu\mathbb{E}\left[\frac{X-\mu}{\sigma}\right] - \frac{u^2}{2}\mathbb{E}\left[\frac{(X-\mu)^2}{\sigma^2}\right] + O(u^3) = 1 - \frac{u^2}{2} + O(u^3)
$$

（$\mathbb{E}[(X-\mu)/\sigma] = 0$、$\mathbb{E}[(X-\mu)^2/\sigma^2] = 1$ を使用。）

$$
\varphi_{S_N}(t) = \left(1 - \frac{t^2}{2N} + O(N^{-3/2})\right)^N \xrightarrow{N \to \infty} e^{-t^2/2}
$$

$e^{-t^2/2}$ は $\mathcal{N}(0,1)$ の特性関数だ — CLT の証明完了。

**MGF が存在しない分布**: Cauchy 分布 $p(x) = \frac{1}{\pi(1+x^2)}$ の MGF は収束しない（重裾のため $\mathbb{E}[e^{tX}] = \infty$）。だが特性関数は常に存在する: $\varphi_X(t) = e^{-|t|}$。Cauchy 分布は CLT が適用されない代表例で、$N$ 個の平均は同じ Cauchy 分布に従い「収束しない」。

**Gaussian の MGF と積率**:

$M_X(t) = e^{\mu t + \sigma^2 t^2/2}$ を $t$ で展開するとモーメントが得られる:

$$
\begin{aligned}
\mathbb{E}[X] &= \mu \\
\mathbb{E}[X^2] &= \mu^2 + \sigma^2 \\
\mathbb{E}[X^3] &= \mu^3 + 3\mu\sigma^2 \\
\mathbb{E}[X^4] &= \mu^4 + 6\mu^2\sigma^2 + 3\sigma^4
\end{aligned}
$$

独立和の性質 $M_{X+Y}(t) = M_X(t) M_Y(t)$ から: $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$, $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ が独立なら $X+Y \sim \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$。これが Gaussian の再生性（reproductive property）だ。



### 5.7 自己回帰尤度の完全実装 — Topic 5

自己回帰モデルの「全て」はこの一式に収まる:

$$
\log p(\mathbf{x}) = \sum_{t=1}^{T} \log p(x_t \mid x_1, \ldots, x_{t-1})
$$

各ステップが Categorical 分布からのサンプリング + 対数確率の加算。

**記号 ↔ 変数対応（Block 2）**:
- $\mathbf{x} = (x_1,\ldots,x_T)$: トークン列 → `x: (T,)` int tensor
- $\mathbf{z}_t = f_\theta(x_{<t})$: モデル出力 logit → `logits: (T, V)` float tensor
- $\log p(x_t \mid x_{<t}) = \log \text{softmax}(\mathbf{z}_t)_{x_t}$ → `log_probs.gather(-1, x)`
- $\log p(\mathbf{x}) = \sum_t \log p(x_t \mid x_{<t})$ → `.sum()` over `(T,)` tensor
- Perplexity $= \exp(-\frac{1}{T}\log p(\mathbf{x}))$ → `torch.exp(-log_prob / T)`

shape: `logits: (T, V)`, `x: (T,)`, `log_p_tokens: (T,)`, `log_prob: scalar`

**数値安定化**: `F.log_softmax(logits, dim=-1)` が `logits - log Z` を計算。`log Z = torch.logsumexp(logits, dim=-1)` が max-shift trick で安定化される:

$$
\log \sum_j e^{z_j} = c + \log \sum_j e^{z_j - c}, \quad c = \max_k z_k
$$

**落とし穴**: `logits[:-1]` と `x[1:]` のペアにすること（次トークン予測）。`F.nll_loss` は `log_softmax` 適用済みテンソルを要求する点に注意（内部で再び `log_softmax` をかけない）。

検算: `log_prob <= 0`（確率は [0,1] → 対数は非正）、`perplexity >= 1.0`、`logsumexp` 経由の手動計算と assert で一致を確認する。

```python
import torch
import torch.nn.functional as F
import torch.distributions as D

torch.manual_seed(0)

# --- Block 2 / 3: MLE + autoregressive log-likelihood ---
# log p(x) = sum_{t=1}^{T} log p(x_t | x_{<t})

def autoregressive_log_prob(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    logits: (T, V) — raw scores at each position (teacher-forcing)
    x:      (T,)   — token ids in {0, ..., V-1}
    returns: log p(x_1, ..., x_T) as a scalar tensor
    """
    # F.log_softmax = z_k - logsumexp(z)  [numerically stable]
    log_probs = F.log_softmax(logits, dim=-1)          # (T, V)
    # gather: pick log p(x_t) for actual token at each position
    log_p_t = log_probs.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)  # (T,)
    return log_p_t.sum()                                # scalar: log p(x)

def nll_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Next-token prediction NLL: logits[:-1] predicts x[1:]."""
    return F.nll_loss(F.log_softmax(logits[:-1], dim=-1), x[1:], reduction='mean')

def perplexity(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    T = x.shape[0]
    return torch.exp(-autoregressive_log_prob(logits, x) / T)

# numerical verification
V, T = 50, 20
logits = torch.randn(T, V)
x = torch.randint(0, V, (T,))

lp  = autoregressive_log_prob(logits, x)
ppl = perplexity(logits, x)
nll = nll_loss(logits, x)

assert lp.item() <= 0,          "log probability <= 0"
assert ppl.item() >= 1.0,       "perplexity >= 1"

# cross-check via logsumexp
log_z   = torch.logsumexp(logits, dim=-1)              # (T,)  = log Z per position
ll_manual = (logits[range(T), x] - log_z).sum()
assert abs(lp.item() - ll_manual.item()) < 1e-5,       "logsumexp cross-check"

print(f"log_prob={lp:.4f}, perplexity={ppl:.2f}, NLL={nll:.4f}")
# log_prob <= 0,  perplexity >= 1,  NLL >= 0
```

**数値検算**: `softmax` の和 = 1 は `F.softmax(logits, dim=-1).sum(dim=-1)` が全て `≈ 1.0` で確認できる。Perplexity = 1 はモデルが確信を持って正解を予測する場合（実際は語彙サイズ $V = 50$ の一様分布で Perplexity ≈ $e^{\log 50} = 50$ が期待値）。

---

これで確率論の5トピック全ての実装が完了した。次は Triton カーネルで並列化する。

### 5.8 Triton カーネル — 並列対数尤度計算

確率モデルの評価において「全データ点の $\log p(x^{(i)} \mid \theta)$ を計算する」操作は**各データ点が独立**なので、GPU 上で embarrassingly parallel に実行できる。

各スレッドが1データ点の対数尤度を担当するカーネルを実装する。

$$
\ell_i(\mu, \sigma) = \log \mathcal{N}(x^{(i)} \mid \mu, \sigma) = -\frac{1}{2}\left(\frac{x^{(i)}-\mu}{\sigma}\right)^2 - \log\sigma - \frac{1}{2}\log(2\pi)
$$

**記号 ↔ 変数対応（Block 3）**:
- $x^{(i)}$ ↔ `x[pid * BLOCK + arange(BLOCK)]`（スレッドブロックが担当するデータ）
- $z_i = (x^{(i)}-\mu)/\sigma$ ↔ `z = (x - mu) / sigma`
- $\ell_i$ ↔ `log_p` → `out[...]` に書き込む
- $\frac{1}{2}\log(2\pi) \approx 0.9189385$ ↔ `0.9189385`（定数）

shape: `x_ptr: (N,)`, `out_ptr: (N,)`, 各スレッドブロックが `BLOCK` 点を処理

**数値安定化**: Gaussian log-prob では `exp` は不要 — 対数空間のまま計算できる。`z * z` のオーバーフローは実用的な $\sigma$ の範囲では問題にならない（FP32 の最大値 $\approx 3.4 \times 10^{38}$、$z^2 < 10^{76}$ まで安全）。

検算: Triton カーネルの出力 sum と `torch.distributions.Normal.log_prob(x).sum()` を比較し、誤差 $< 10^{-2}$ を assert する。

```python
import triton
import triton.language as tl
import torch
import torch.distributions as D

@triton.jit
def gaussian_log_lik_kernel(
    x_ptr,              # (N,) data points
    out_ptr,            # (N,) output log-likelihoods
    mu,                 # scalar mean
    sigma,              # scalar std (> 0)
    N,                  # number of points
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)   # (BLOCK,)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)   # (BLOCK,)

    # log N(x | mu, sigma) = -0.5*z^2 - log(sigma) - 0.5*log(2*pi)
    z     = (x - mu) / sigma                           # standardized
    log_p = -0.5 * z * z - tl.log(sigma) - 0.9189385  # 0.9189385 = 0.5*log(2*pi)

    tl.store(out_ptr + offs, log_p, mask=mask)


def gaussian_log_lik(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Parallel log N(x_i | mu, sigma) for all i."""
    N   = x.numel()
    out = torch.empty(N, device=x.device, dtype=x.dtype)
    BLOCK = 1024
    grid  = (triton.cdiv(N, BLOCK),)
    gaussian_log_lik_kernel[grid](x, out, mu, sigma, N, BLOCK=BLOCK)
    return out


# numerical check: Triton sum vs torch.distributions
torch.manual_seed(0)
x_test = torch.randn(10_000)
mu_val, sigma_val = 0.5, 1.2

ll_triton = gaussian_log_lik(x_test, mu_val, sigma_val).sum()
ll_torch  = D.Normal(mu_val, sigma_val).log_prob(x_test).sum()

assert abs(ll_triton.item() - ll_torch.item()) < 1.0   # float32 accumulation tolerance
print(f"triton sum={ll_triton:.4f},  torch sum={ll_torch:.4f}")
print(f"|diff| = {abs(ll_triton - ll_torch):.4f}  (float32 accumulation)")
```

### 5.9 理解度チェック — Z5 完了確認

<details>
<summary>Q1: SciPyで多変量正規分布の条件付き分布を計算する際の数値安定性の注意点は？</summary>

**A**: 共分散行列 $\Sigma$ が特異に近い場合、逆行列計算が不安定になる。対策：(1) `torch.linalg.solve` を使い直接逆行列を避ける、(2) Cholesky 分解で正定値性を確認（`torch.linalg.cholesky` が失敗 → 半正定値）、(3) 正則化項 $\Sigma + \epsilon I$ を追加（$\epsilon \sim 10^{-6}$）、(4) 条件数 $\kappa(\Sigma)$ を確認（$> 10^{10}$ なら危険）。`D.MultivariateNormal` は `scale_tril` 引数で Cholesky 因子を直接受け取れる。

</details>

<details>
<summary>Q2: ベイズ推論のグリッド近似が実用的でない理由と代替手法を説明せよ。</summary>

**A**: グリッド近似は次元の呪い（$d$ 次元で $N^d$ 点必要）。10次元で各軸100点なら $100^{10} = 10^{20}$ 点。代替手法：(1) MCMC（Metropolis-Hastings、HMC）で事後分布からサンプリング、(2) 変分推論（ELBO最大化）で近似分布 $q(\theta)$ を最適化、(3) Laplace近似で事後のモード周りを正規近似。

</details>

---

### 5.9 分布ファミリーの全体像と相互関係

第4回で登場した分布たちの関係を整理する。これを知っていると、新しい問題に直面したとき「どの分布を使うべきか」が見えやすくなる。

```mermaid
flowchart TD
  B["Bernoulli(p)\nP(X=1)=p"] --> C["Binomial(n,p)\nn回試行の成功数"]
  B --> CAT["Categorical(π)\nK値の離散分布"]
  CAT --> MULT["Multinomial(n,π)\nn回試行の出現数"]
  N1["Normal(μ,σ²)"] --> MVN["Multivariate Normal\n(μ,Σ)"]
  MVN --> GMM["Gaussian Mixture\nΣ_k π_k N(μ_k,Σ_k)"]
  BETA["Beta(a,b)\n∈[0,1]"] --> B
  GAMMA["Gamma(α,β)"] --> N1
  GAMMA --> POISSON["Poisson(λ)\n非負整数"]
  N1 --> CHI2["Chi-squared(k)\n=Gamma(k/2,2)"]
  EF["指数型分布族\np(x|η)=h(x)exp(η^T T(x)-A(η))"] --> B
  EF --> N1
  EF --> BETA
  EF --> GAMMA
  EF --> POISSON
  EF --> CAT
```

**覚えておくべき変換**:

| 変換 | 数式 | 用途 |
|:-----|:-----|:-----|
| $X \sim \mathcal{N}(0,1)$ → $X^2 \sim \chi^2(1)$ | 2乗変換 | 検定統計量 |
| $\sum_{k=1}^n Z_k^2 \sim \chi^2(n)$ | 加法性 | 分散推定 |
| $\text{Bernoulli}(p) = \text{Binomial}(1, p)$ | 特殊ケース | LLM出力 |
| $\text{Categorical}(\boldsymbol{\pi}) = \text{Multinomial}(1, \boldsymbol{\pi})$ | 特殊ケース | トークン予測 |
| $X \sim \text{Poisson}(\lambda)$ として $\lambda \to \infty$: $\mathcal{N}(\lambda, \lambda)$ | CLT | 正規近似 |

**第4回のトピック全カバレッジ確認**:

| トピック | 実装完了 | 重要度 |
|:---------|:---------|:-------|
| 確率分布（Gaussian/Categorical/Beta） | 5.1 ✅ | ⭐⭐⭐ |
| 多変量正規分布・条件付き分布 | 5.2 ✅ | ⭐⭐⭐ |
| 指数型分布族 | 5.3 ✅ | ⭐⭐⭐ |
| GMM・EM算法の前段 | 5.4 ✅ | ⭐⭐⭐ |
| ベイズ推論（グリッド）| 5.5a ✅ | ⭐⭐ |
| 共役事前分布（Gaussian-Gaussian）| 5.5b ✅ | ⭐⭐⭐ |
| KLダイバージェンス | 5.5a-KL ✅ | ⭐⭐⭐ |
| Fisher情報量・CR下界 | 5.5c ✅ | ⭐⭐⭐ |
| LLN・CLT | 5.1補足 ✅ | ⭐⭐ |
| 自己回帰尤度 | 5.7 ✅ | ⭐⭐⭐ |

> Progress: 85%

---

## 🔬 Z6. 新たな冒険へ（20分）— 確率論の研究系譜

### 6.1 VAE — 確率的生成モデルの統一

Kingma & Welling (2013)[^2] は確率論の全武器を一点に集約した。

観測 $\mathbf{x}$、潜在変数 $\mathbf{z}$、生成モデル $p_\theta(\mathbf{x} \mid \mathbf{z})$。問題: 事後分布 $p_\theta(\mathbf{z} \mid \mathbf{x})$ が intractable。

**解決**: 変分分布 $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p_\theta(\mathbf{z} \mid \mathbf{x})$ で近似し、ELBO（Evidence Lower BOund）を最大化:

$$
\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))
$$

左辺と右辺の差は $D_{\mathrm{KL}}(q \| p_\theta(\mathbf{z}|\mathbf{x})) \geq 0$ だから、等号はKLがゼロのとき。

**第4回との接続**:
- 第1項 $\mathbb{E}_{q}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ = Gaussian MLE の期待値版
- 第2項 $D_{\mathrm{KL}}(q \| p)$ = KL divergence（情報理論、第5回以降）
- 事前 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$ = 共役Gaussian の応用

### 6.2 Bayesian Deep Learning — 分布としてのネットワーク

ニューラルネットの重み $\mathbf{w}$ を点推定ではなく分布として扱う。

$$
p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mathbf{w}) \cdot p(\mathbf{w})
$$

これは第4回 §3 のベイズ更新の直接適用だ。問題: $\mathbf{w}$ が何百万次元でもグリッド近似は不可能 → 変分推論（VI）かMCMCが必要。

**Bayes by Backprop**: 重みを $q(\mathbf{w}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ でパラメータ化し、ELBOを勾配降下で最大化。「重みの不確実性」が予測の不確実性に変換される。

**なぜ今、再注目されるのか**: LLMのCalibration問題。「モデルが高確信度で誤答する」現象をBayesian手法で緩和できる可能性。

### 6.3 自己回帰の普遍性 — Malach (2023)

$$
\log p(\mathbf{x}) = \sum_{t=1}^{T} \log p(x_t \mid x_{<t})
$$

この連鎖規則は**任意の分布**に対して厳密に成立する（確率の乗法定理）。Malach (2023)[^5] は「十分な表現力を持つ自己回帰モデルはあらゆる確率分布を近似できる」ことを理論化した。

「GPT系LLMが画像・音声・タンパク質・コードを生成できる」の理論的根拠はここにある。連鎖規則のシンプルさが、適用範囲の広大さに直結する。

### 6.4 Diffusion Models — 確率過程と逆拡散

DDPM (Ho et al. 2020)[^6] は確率論の異なる側面を使う。

**Forward process** (拡散: データ → ノイズ):

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t I)
$$

各ステップで少量のノイズを加える。$T$ ステップ後: $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, I)$。

**Reverse process** (生成: ノイズ → データ): ニューラルネット $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ を学習。

**第4回との接続**: Forward processはGaussianの連続積。ELBO の最適化はVAEと同じ構造。第4回で学んだ「Gaussian同士の周辺化の閉形式」が $q(\mathbf{x}_t \mid \mathbf{x}_0)$ の分析的計算を可能にする。

### 6.5 研究系譜図

```mermaid
flowchart TD
  KO["Kolmogorov (1933)<br/>確率空間の公理化"]
  BE["Bayes & Price (1763)<br/>ベイズの定理"]
  CR["Cramér-Rao (1945/46)<br/>Fisher情報量・推定限界"]
  EF["指数型分布族<br/>十分統計量・共役事前分布"]
  CLT["中心極限定理<br/>ガウス分布の普遍性"]
  EM["EM算法 (Dempster 1977)<br/>潜在変数モデル学習"]
  VAE["VAE (Kingma 2013)<br/>深層生成モデル"]
  BBB["Bayes by Backprop (2015)<br/>Bayesian Deep Learning"]
  DDPM["DDPM (Ho 2020)<br/>拡散モデル"]
  AR["自己回帰普遍性 (Malach 2023)"]

  KO --> CLT
  BE --> EF
  CR --> EF
  EF --> EM
  CLT --> DDPM
  EM --> VAE
  VAE --> DDPM
  EF --> VAE
  BE --> BBB
  KO --> AR
```

> Progress: 95%

---

## 🎓 Z7. エピローグ（10分）— まとめと次回予告

### 7.0 数式↔実装対応表

| 数式 | 実装 | セクション |
|:-----|:-----|:-----------|
| $f(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$ | `D.Normal(mu, sigma).log_prob(x)` | 5.1 Block 1 |
| $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{N}\sum(x_i-\bar{x})^2$ | `x.mean()`, `x.std(unbiased=False)` | 5.1 Block 1 |
| $\mathcal{H}(\boldsymbol{\pi}) \leq \log K$ | `D.Categorical(probs).entropy()` | 5.1 Block 1 |
| $D_{\mathrm{KL}}(p\|q)$ (Gaussian 閉形式) | `D.kl_divergence(p, q)` | 5.1 Block 1 |
| $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})$ | `D.MultivariateNormal(mu, cov).log_prob(x)` | 5.1 Block 1 |
| $\boldsymbol{\mu}_{1\mid 2}, \boldsymbol{\Sigma}_{1\mid 2}$（Schur 補行列） | 導出参照 §5.2 | 5.2 |
| $p(x\mid\boldsymbol{\eta}) = h(x)\exp(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta}))$ | 自然勾配・期待値パラメータ解析 | 5.3 |
| $p(\mathbf{x}\mid\theta) = \pi\mathcal{N}_1 + (1-\pi)\mathcal{N}_2$ | EM 算法への橋渡し（§5.4） | 5.4 |
| $p(\theta\mid\mathbf{x}) \propto \theta^{a+h-1}(1-\theta)^{b+t-1}$ | Beta-Bernoulli 共役更新 | 5.5a |
| $\lambda_N = \lambda_0 + N/\sigma^2$（精度加法性） | 精度更新公式 | 5.5b |
| $D_{\mathrm{KL}}(p\|q) \geq 0$（Gibbs 不等式） | `D.kl_divergence` | 5.5a-KL |
| $\mathrm{Var}(\hat{\theta}) \geq 1/(nI(\theta))$（CR 下界） | CR 下界導出 §5.5c | 5.5c |
| $M_X(t) = e^{\mu t + \sigma^2 t^2/2}$ | CLT 証明スケッチ §5.6 | 5.6 |
| $\log p(\mathbf{x}) = \sum_t \log p(x_t\mid x_{<t})$ | `autoregressive_log_prob(logits, x)` | 5.7 Block 2 |
| Perplexity $= \exp(-\frac{1}{T}\log p)$ | `perplexity(logits, x)` | 5.7 Block 2 |
| $\ell_i = \log \mathcal{N}(x^{(i)}\mid\mu,\sigma)$ 並列 | `gaussian_log_lik_kernel[grid](...)` | 5.8 Block 3 |

### 7.1 本講義の核心 — 3つの持ち帰り

1. **確率は「わからなさ」の言語である。** 確率空間 $(\Omega, \mathcal{F}, P)$ という厳密な枠組みの上に、確率変数・期待値・条件付き確率が定義される。この言語なしに生成モデルは記述できない。

2. **ベイズの定理は「学習」の数式だ。** 事前分布（信念）+ 尤度（データ）→ 事後分布（更新された信念）。VAEのELBOも、LLMのファインチューニングも、この構造の変種だ。

3. **MLEは条件付きCategorical分布の最適化に帰着する。** LLMの学習は、各時刻 $t$ で $p(x_t \mid x_{<t})$ をCategorical分布としてMLE推定すること。本講義で学んだ全ての道具がここに集約される。

### 7.2 FAQ

<details><summary>Q: ベイズと頻度主義、結局どちらが正しいのか？</summary>

「正しさ」の基準が異なる。頻度主義は「推定量の長期的振る舞い」（繰り返し実験）で評価し、ベイズは「現在の知識の下での確信度」で評価する。MLの文脈では:

- **MLE**（頻度主義寄り）: 計算が簡単、漸近的に最適、大データ向き
- **ベイズ推論**: 不確実性の定量化が自然、小データ向き、事前知識を活用可能

実用上は「どちらか一方」ではなく、問題に応じて使い分ける。VAEは変分ベイズ、LLMの損失関数はMLEだ。
</details>

<details><summary>Q: なぜ正規分布がこんなに頻出するのか？</summary>

3つの理由がある:

1. **中心極限定理**: 多数の独立な微小効果の和は正規分布に近づく
2. **最大エントロピー**: 平均と分散を固定したとき、エントロピー最大の分布が正規分布
3. **計算の都合**: 正規分布の積・和・条件付きが全て閉じた形になる

3つ目が実用上最も重要だ。GANの潜在空間 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ やVAEの事前分布も、計算の容易さが選択の主因だ。
</details>

<details><summary>Q: 指数型分布族は実際にどこで使うのか？</summary>

至る所で。

- **VAE**: エンコーダの出力はガウス分布（指数型分布族）のパラメータ
- **EBM**: $p(\mathbf{x}) = \frac{1}{Z}\exp(-E(\mathbf{x}))$ は指数型分布族の一般化
- **GLM**: 一般化線形モデルの応答分布は指数型分布族
- **Softmax**: Categorical分布は指数型分布族。LLMの出力分布そのもの

第27回（EBM）と第9回（変分推論）で本格的に活用する。
</details>

<details><summary>Q: Cramér-Rao下界を知って何の役に立つのか？</summary>

「この推定問題でこれ以上の精度は原理的に不可能」という限界を知ることができる。

- モデル設計: 推定量の分散がCR下界に近ければ、これ以上のデータは不要
- 実験計画: Fisher情報量が大きい実験条件を選ぶことで、少ないデータで精密な推定が可能
- 理論解析: NNの表現力とFisher情報量の関係は活発な研究分野
</details>

<details><summary>Q: 「確率密度関数の値が1を超える」のは間違いでは？</summary>

いいえ、正しい。PDFは確率ではない。確率は密度の**積分**で得られる:

$$
P(a \leq X \leq b) = \int_a^b f(x) dx
$$

$f(x)$ 自体は非負であればいくらでも大きくてよい。例えば $\mathcal{N}(0, 0.01)$ のピークは $f(0) = \frac{1}{\sqrt{2\pi \cdot 0.01}} \approx 3.99$ で、1を大きく超える。積分すると必ず1になるが、密度値が1を超えること自体は何の問題もない。

</details>

<details><summary>Q: Multinomial分布とCategorical分布の違いは？</summary>

Categorical分布は「サイコロを1回振る」: $x \in \{1, \ldots, K\}$, $P(x=k) = \pi_k$。

Multinomial分布は「サイコロを $n$ 回振って、各面の出た回数を記録する」: $(c_1, \ldots, c_K) \sim \text{Multi}(n, \boldsymbol{\pi})$, $\sum_k c_k = n$。

LLMの文脈では:
- 1トークンの予測 = Categorical分布
- バッチ内の全トークン予測の統計 = Multinomial分布

Categorical = Multinomial($n=1$, $\boldsymbol{\pi}$) だ。
</details>

<details><summary>Q: 「尤度」と「確率」は何が違うのか？</summary>

**確率**: データ $x$ が可変で、パラメータ $\theta$ が固定 → $P(X=x \mid \theta)$

**尤度**: データ $x$ が固定で、パラメータ $\theta$ が可変 → $L(\theta; x) = P(X=x \mid \theta)$

数式は全く同じ。視点の違いだけだ。確率として見ると $\sum_x P(x \mid \theta) = 1$（データに関して正規化）。尤度として見ると $\int L(\theta; x) d\theta$ は一般に1にならない。

MLEは「このデータが最もよく生成されるようなパラメータ」を探す → 尤度関数の最大化。

</details>

<details><summary>Q: 条件付き期待値 E[X|Y] はなぜ確率変数なのか？</summary>

$\mathbb{E}[X \mid Y=y]$ は $y$ の関数として計算できる。例えば $(X,Y) \sim \mathcal{N}$ なら $\mathbb{E}[X \mid Y=y] = \mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y)$（線形）。

$Y$ が確率変数だから $\mathbb{E}[X \mid Y]$ も確率変数になる。重要な性質: **繰り返し期待値の法則**

$$
\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]
$$

これはELBOの導出でも使われる: $\log p(\mathbf{x}) = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z})/q(\mathbf{z})] + D_{\mathrm{KL}}(q \| p)$。

</details>

<details><summary>Q: この確率論の知識は第5回（測度論）でどう拡張されるのか？</summary>

本講義では「確率密度関数 $f(x)$ が存在する」と暗黙に仮定した。だが:

- 離散と連続が混じった分布は？
- $\mathbb{R}^d$ 上の全ての部分集合に確率を定義できるか？
- 「ほとんど確実に」とは何か？

第5回では測度論の言葉で $f(x) = \frac{dP}{d\lambda}$ （Radon-Nikodym導関数）として密度関数を厳密に定義する。確率過程（Markov連鎖、Brown運動）も導入し、拡散モデルのSDE定式化への橋渡しを行う。
</details>

### 7.3 確率論でよくある「罠」

<details><summary>罠6: 多次元Gaussianの「ほとんどの確率質量」は殻にある</summary>

1次元では Gaussian の確率質量は平均付近に集中する（$\pm 2\sigma$ に95%）。

$d$ 次元では全く違う。$\mathbf{x} \sim \mathcal{N}(\mathbf{0}, I_d)$ のノルム $\|\mathbf{x}\|$ は:

$$
\mathbb{E}[\|\mathbf{x}\|^2] = d, \quad \text{Var}(\|\mathbf{x}\|) = O(1)
$$

つまり $\|\mathbf{x}\| \approx \sqrt{d}$ に集中する（次元の呪い の現れ）。$d=1000$ では全サンプルが半径 $\approx 31.6$ の薄い球殻上にある。

VAEの潜在空間 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_{100})$ でサンプリングすると、$\|\mathbf{z}\| \approx 10$ の球殻からしかサンプルが来ない。これがVAEの「posterior collapse」問題の一因だ。

</details>



<details><summary>罠1: P(A|B) ≠ P(B|A) — 条件の逆転</summary>

「雨のとき傘を持つ確率90%」と「傘を持っているとき雨の確率」は全く違う。ベイズの定理なしにこの2つを混同するのが「検察官の誤謬」だ。DNA鑑定で「一致した = 犯人」と結論するのは $P(\text{一致} \mid \text{犯人})$ と $P(\text{犯人} \mid \text{一致})$ の混同。
</details>

<details><summary>罠2: 独立と無相関は違う</summary>

無相関: $\text{Cov}(X, Y) = 0$（線形関係がない）
独立: $P(X, Y) = P(X)P(Y)$（あらゆる関係がない）

独立 → 無相関だが、逆は成り立たない。$X \sim \mathcal{N}(0,1)$, $Y = X^2$ は無相関だが独立ではない。
</details>

<details><summary>罠3: 分散0でも分布は決まらない</summary>

Cramér-Rao下界 $\text{Var} \geq 1/(nI)$ は不偏推定量にしか適用されない。バイアスのある推定量はCR下界を下回ることがある（James-Steinの縮小推定量）。「バイアスを許容する代わりにMSEを下げる」のは、MLでは正則化として日常的に行われる。
</details>

<details><summary>罠4: MLEは常に最良ではない</summary>

小サンプルではMLEのバイアスが問題になる。分散推定量 $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N}\sum(x_i - \bar{x})^2$ は $\sigma^2$ を過小評価する。James-Steinの定理が示すのは、3次元以上ではMLEが「許容可能でない」（admissible でない）という衝撃的事実だ。
</details>

<details><summary>罠5: 事前分布が「主観的」は欠点か？</summary>

頻度主義者はベイズの「主観性」を批判する。だが:
- 「事前分布なし」は「一様事前分布」と等価 — これも主観的
- 弱情報事前分布は、物理的制約（パラメータの範囲等）を自然にエンコード
- データが十分あれば事前分布の影響は消える（事後一致性）

実用的には、事前分布は「正則化の一形態」と割り切ってよい。
</details>

### 7.4 次回予告 — 第5回: 測度論的確率論・確率過程入門

第4回で確率分布を「使える」ようになった。だが、以下の問いに答えられるだろうか:

- 「確率密度関数」とは厳密に何か？ なぜ点 $x$ での $f(x)$ は確率ではないのか？
- 離散と連続が混じった分布をどう扱うか？
- 「ほとんど確実に収束する」の「ほとんど」とは？
- Brown運動はなぜ微分不可能なのか？
- 拡散モデルのforward processを記述するSDEとは何か？

第5回では**測度論**の言葉で確率論を再構築する。Lebesgue積分、Radon-Nikodym導関数、確率過程、Markov連鎖、Brown運動 — 拡散モデルの数学的基盤がここに埋まっている。

そして `%timeit` が初登場する。Monte Carlo積分の計算コストを測り始めると、Pythonの「遅さ」が少しずつ見えてくる......。

> **Note:** **進捗: 100% 完了** 第4回: 確率論・統計学 — 全ゾーンクリア。お疲れさまでした。確率の言語を手に入れた今、第5回で測度論という「確率の文法」を厳密に定義する旅に出よう。

---


**第5回の核心概念プレビュー**:

```mermaid
flowchart LR
  P4["第4回\n確率分布\nMLE/ベイズ\nGaussian"] --> P5
  P5["第5回\n測度論\n確率過程\nSDE"] --> P8
  P8["第8回\nVAE\nELBO\n変分推論"] --> P15
  P15["第15回\nDiffusion\nSDE逆問題\nScore matching"]
```

特に「Radon-Nikodym導関数」は Score matching の数学的基礎だ。スコア関数 $\nabla_x \log p(x)$ はデータ分布の勾配を表し、拡散モデルのノイズ除去プロセスと直接対応する。

### 7.5 💀 パラダイム転換の問い

> **現実のデータは正規分布に従わない。それでも仮定する"本当の理由"は何か？**

CLTが「多数の独立微小効果の和→正規分布」を保証するから？ それは理由の一つだ。だが本質はもっと深い。

- ガウス分布は**最大エントロピー分布**だ。平均と分散だけを知っているとき、それ以上の仮定を置かない「最も情報量の少ない」分布がガウスだ
- ガウス分布の演算は**閉じている**。和・条件付き・周辺が全てガウスのまま。これは計算上の奇跡と言ってよい
- そして、正規分布が「間違っている」ことは**わかっている**上で使う。重要なのは「どの程度間違っているか」を定量化すること — KLダイバージェンス（第6回）がその道具だ

<details><summary>ベイズ脳仮説 — 脳は確率計算機か？</summary>

認知科学には「脳はベイズ推論を行っている」という仮説がある。感覚入力（尤度）と経験（事前分布）を組み合わせて世界の状態（事後分布）を推定する。

錯視現象は、強い事前分布が弱い尤度を上書きする例として解釈される。VAEのデコーダが「ぼやけた」画像を生成するのは、事前分布 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ が過度に滑らかな潜在空間を強制するため — ある意味、脳の錯視と同じ構造だ。

「正規分布を仮定する」のは、脳が「世界は滑らかだ」と仮定するのと同じかもしれない。
</details>

考えてみよう:

- **LLMの出力分布はCategorical。** 正規分布ではない。だがCategorical分布の自然パラメータ（logit）は連続値で、その空間では正規分布的な仮定が使われる
- **次元の呪い**: 100次元のガウス分布のサンプルは、ほぼ確実に原点から $\sqrt{100} = 10$ の距離にある。「高次元のガウスは球殻に集中する」— これが正規分布の直感が崩壊する瞬間だ
- **正規分布は"最も無知な"分布**: 最大エントロピー原理により、平均と分散しか知らないとき、余計な仮定を最も少なくする分布がガウス。「知らないことを正直に認める分布」とも言える


---

### 7.6 最新研究 (2020-2026)

#### 6.9.1 Fisher情報量の理論的進展

Fisher情報量は統計的推測の基礎であり、最近の研究はその応用範囲を拡大している。

**期待Fisher情報 vs 観測Fisher情報**

Fisher情報量には2つの表現がある:

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log p(X; \theta)}{\partial \theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2 \log p(X; \theta)}{\partial \theta^2}\right]
$$

前者は「期待」、後者は「観測」と呼ばれる。2013年のarXiv論文[^13]は、**期待Fisher情報を使った信頼区間が観測Fisher情報を使った場合より平均二乗誤差の意味で精度が高い**ことを証明した。2021年の続編[^14]では、この結果を区間推定の相対性能評価に拡張している。

**潜在変数モデルへの拡張**

2024年の研究[^15]は、潜在変数モデルに対するFisher情報量の明示的定義を可能にする新しい最尤推定フレームワークを提案した。従来、潜在変数 $\mathbf{z}$ を積分消去した周辺尤度 $p(\mathbf{x}; \theta) = \int p(\mathbf{x}, \mathbf{z}; \theta) d\mathbf{z}$ ではFisher情報量の計算が困難だった。この研究は、変分近似と組み合わせることで効率的な推定を実現している。

**テンソルモデルのFisher情報**

2025年の最新論文[^16]は、ポアソンCanonical Polyadic (CP) テンソル分解のFisher情報量を導出した。3次元以上のテンソルデータ（例: 時間×空間×周波数）の統計的性質を定量化することで、Cramér-Rao下界に基づく推定量の評価が可能になる。


#### 6.9.2 測度論的確率論の実用化

測度論は確率論の厳密な基礎を与えるが、「抽象的すぎて実用的でない」という誤解がある。最近の研究は、測度論的フレームワークの実用的応用を示している。

**Taylor測度と確率過程**

2025年のarXiv論文[^17]は、Taylor測度という概念を導入し、Brown運動、マルチンゲール、ランダムウォーク、時系列モデルを統一的に扱う枠組みを提案した。これはTaylor展開の一般化であり、確率過程の局所的性質を捉える。

**連続時間確率過程のMetric Temporal Logic**

2023年の研究[^18]は、連続時間確率過程がMetric Temporal Logic (MTL) の論理式を満たすかどうかの可測性を確立した。これは形式検証とモンテカルロ法を橋渡しする成果で、自動運転車の安全性検証などに応用されている。

**確率空間の構成**

arXiv論文[^19]は、決定論的過程から出発して抽象的確率空間を構成する手法を提案した。これは「確率的シミュレーションは決定論的アルゴリズムである」という哲学的洞察を形式化している。

#### 6.9.3 情報理論の最新展開

KLダイバージェンスとエントロピーは機械学習の中心概念だが、その理論はまだ発展途上だ。

**α-ダイバージェンスとベイズ最適化**

2024年の論文[^20]は、KLダイバージェンスを一般化したα-ダイバージェンスに基づく新しいベイズ最適化手法「Alpha Entropy Search (AES)」を提案した。α-ダイバージェンスは:

$$
D_\alpha(p \| q) = \frac{1}{\alpha(\alpha-1)} \left( \int p(x)^\alpha q(x)^{1-\alpha} dx - 1 \right)
$$

$\alpha \to 1$ でKLダイバージェンスに収束する。AESは獲得関数として、次の評価点での目的関数値と大域的最大値の「依存度」を最大化する。この依存度をα-ダイバージェンスで測ることで、KLベースの手法より探索と活用のバランスを柔軟に制御できる。

**Jensen-ShannonとKLの関係**

2025年の論文[^21]は、Jensen-Shannon (JS) ダイバージェンスとKLダイバージェンスの最適な下界を確立した:

$$
\text{JS}(p \| q) = \frac{1}{2} D_{\text{KL}}(p \| m) + \frac{1}{2} D_{\text{KL}}(q \| m), \quad m = \frac{p + q}{2}
$$

JSダイバージェンスはGANの目的関数として知られているが、KLとの定量的関係は長年不明だった。この成果により、GANの収束性理論が改善された。

**幾何学的情報理論 (GAIT)**

従来のKLダイバージェンスは確率分布を「点」として扱い、空間の幾何を無視する。2019年の論文[^22]は、確率分布の台（support）の幾何学的構造を考慮した新しいダイバージェンス「Geometric Information」を提案した。これは最適輸送理論とKLダイバージェンスを統合する試みだ。


#### 6.9.4 統計的推測の新理論

**Extended Likelihoodとランダム未知量**

2023年の論文[^23]は、従来の尤度理論を「固定された未知パラメータ」から「ランダムな未知量」へ拡張した。これは頻度主義とベイズ主義の中間的立場で、事前分布を仮定せずにランダム効果を扱える。

**Maximum Ideal Likelihood**

2024年の研究[^24]は、潜在変数モデルに対する新しい推定フレームワーク「Maximum Ideal Likelihood (MIL)」を提案した。従来のMLEは周辺化 $p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z}$ が困難だったが、MILは潜在変数を「理想的な観測」として扱うことで、計算可能な目的関数を導出する。漸近的にMLEと等価であり、信頼区間も構成できる。


#### 6.9.5 非正規化統計モデルとスコアマッチング

確率密度関数を正規化定数込みで計算するのは困難な場合が多い。Energy-Based Model (EBM) では $p(x) = \frac{1}{Z}\exp(-E(x))$ と表現するが、分配関数 $Z = \int \exp(-E(x))dx$ の計算が指数的に困難だ。

**スコアマッチング** [^9] は、正規化定数を計算せずに確率モデルを推定する手法だ。スコア関数 $s(x) = \nabla_x \log p(x)$ は正規化定数に依存しないことを利用する:

$$
s(x) = \nabla_x \log p(x) = \nabla_x \log \frac{1}{Z}\exp(-E(x)) = \nabla_x [-E(x) - \log Z] = -\nabla_x E(x)
$$

スコアマッチング目的関数:

$$
J(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}(x)}\left[\| \nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{data}}(x) \|^2\right]
$$

これは正規化定数なしで計算可能な形に変形できる（部分積分を用いた恒等式）。拡散モデル [^10] の理論的基盤の一つでもある。


#### 6.9.6 確率論とLLMの深い接続

LLMの訓練は、次トークン予測という確率的タスクに帰着する。この接続を明確にしよう。

**自己回帰モデルと連鎖規則**:

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

各時刻での条件付き分布 $p(x_t \mid x_{<t})$ はCategorical分布であり、Softmaxで定義される:

$$
p(x_t = k \mid x_{<t}) = \frac{\exp(z_k)}{\sum_{j=1}^{V} \exp(z_j)}, \quad z = f_\theta(x_{<t})
$$

**Cross-Entropy損失とMLE**:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) = -\frac{1}{T} \log p_\theta(\mathbf{x})
$$

これは負の対数尤度であり、最小化はMLEと等価だ。

**Perplexityと条件付きエントロピー**:

$$
\text{Perplexity} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(x_t \mid x_{<t})\right)
$$

これは条件付きエントロピー $H(X_t \mid X_{<t})$ の指数である。Perplexity=10は「各時刻で平均10個の候補から選択している」ことを意味する。

**確率的ランキングとTop-k/Nucleus Sampling**:

温度パラメータ $\tau$ を導入した確率分布:

$$
p_\tau(x_t = k) = \frac{\exp(z_k/\tau)}{\sum_j \exp(z_j/\tau)}
$$

- $\tau \to 0$: 決定論的（argmax）
- $\tau = 1$: 元の分布
- $\tau > 1$: より平坦（多様性増加）

Nucleus sampling（Top-p）は累積確率 $\sum_{k \in \text{top-p}} p(k) \geq p$ を満たす最小集合からサンプリング。これは「確率質量の上位p%」という動的閾値だ。


> **Note:** **LLMの確率論的解釈**: 次トークン予測モデルは、シーケンスの条件付き確率分布を学習している。サンプリング戦略（temperature, top-k, nucleus）は、この確率分布からの「制御されたランダム化」だ。決定論的生成（greedy）は最尤推定、確率的生成はベイズ推論の視点と対応する。

### 7.7 確率論から生成モデルへの橋

第4回で学んだ全てが、深層生成モデルの数学的土台だ。この橋を明示的に示しておく。

**VAE（第8回）への直接接続**:

| 第4回の概念 | VAEでの役割 |
|:------------|:------------|
| MLE | デコーダ $p_\theta(\mathbf{x}\mid\mathbf{z})$ の最大化 |
| KLダイバージェンス | ELBOの正則化項 $D_{\mathrm{KL}}(q_\phi\|p)$ |
| Gaussian MLE | エンコーダ $\mu_\phi, \sigma_\phi$ の出力 |
| 指数型分布族 | デコーダの出力分布設計 |
| 変分推論（事後一致性）| ELBO最大化による近似事後分布の学習 |

**Diffusion Models（第15回）への接続**:

| 第4回の概念 | Diffusionでの役割 |
|:------------|:------------------|
| Gaussian積の閉形式 | $q(\mathbf{x}_t\mid\mathbf{x}_0)$ の分析的計算 |
| 条件付きGaussian | 逆プロセス $p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ の形 |
| KL最小化 | ELBO = $\sum_t \mathbb{E}[D_{\mathrm{KL}}(q_t\|p_{t-1})]$ |

**LLM（第20回）への接続**:

| 第4回の概念 | LLMでの役割 |
|:------------|:------------|
| Categorical分布 | softmax出力層 |
| 連鎖規則 $\log p(\mathbf{x}) = \sum_t \log p(x_t\mid x_{<t})$ | 自己回帰目的関数 |
| MLE | 次トークン予測の最大化（交差エントロピー）|
| 指数型分布族 | logit空間の幾何学 |

確率論は「積み木」だ。ここで積んだ概念が、後半の全ての講義で呼び戻される。

> Progress: 100%

> **理解度チェック**
> 1. ベイズの定理 $p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$ において、事前分布・尤度・事後分布をそれぞれ同定せよ。
> 2. 正規分布 $\mathcal{N}(\mu, \sigma^2)$ の平均・分散を $\mu, \sigma$ で書き、$\sigma \to 0$ のとき分布はどう変化するか説明せよ。

---
> **📖 前編もあわせてご覧ください**
> [【前編】第4回: 確率論・統計学](/articles/ml-lecture-04-part1) では、確率論・ベイズの定理・指数型分布族の理論を学びました。

## 参考文献

### 主要論文

[^1]: Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2016). Variational Inference: A Review for Statisticians.
<https://arxiv.org/abs/1601.00670>

[^2]: Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
<https://arxiv.org/abs/1312.6114>

[^3]: Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
<https://arxiv.org/abs/1503.02531>

[^4]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
<https://arxiv.org/abs/2006.11239>

[^5]: Malach, E. (2023). Auto-Regressive Next-Token Predictors are Universal Learners.
<https://arxiv.org/abs/2309.06979>

[^6]: Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution.
<https://arxiv.org/abs/1907.05600>

[^7]: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations.
<https://arxiv.org/abs/2011.13456>

[^10]: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B. (2020). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021 (Oral)*.
[https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)

[^11]: Rezende, D.J., Mohamed, S. (2015). "Variational Inference with Normalizing Flows." *ICML 2015*.
[https://arxiv.org/abs/1505.05770](https://arxiv.org/abs/1505.05770)

[^12]: Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

[^13]: Relative Performance of Expected and Observed Fisher Information in Covariance Estimation for Maximum Likelihood Estimates. (2013). *arXiv preprint*.
[https://arxiv.org/abs/1305.1056](https://arxiv.org/abs/1305.1056)

[^14]: Relative Performance of Fisher Information in Interval Estimation. (2021). *arXiv preprint*.
[https://arxiv.org/abs/2107.04620](https://arxiv.org/abs/2107.04620)

[^15]: Maximum Ideal Likelihood Estimation: A Unified Inference Framework for Latent Variable Models. (2024). *arXiv preprint*.
[https://arxiv.org/abs/2410.01194](https://arxiv.org/abs/2410.01194)

[^16]: A Latent-Variable Formulation of the Poisson Canonical Polyadic Tensor Model: Maximum Likelihood Estimation and Fisher Information. (2025). *arXiv preprint*.
[https://arxiv.org/abs/2511.05352](https://arxiv.org/abs/2511.05352)

[^17]: The Taylor Measure and its Applications. (2025). *arXiv preprint*.
[https://arxiv.org/abs/2508.04760](https://arxiv.org/abs/2508.04760)

[^18]: On the Metric Temporal Logic for Continuous Stochastic Processes. (2023). *arXiv preprint*.
[https://arxiv.org/abs/2308.00984](https://arxiv.org/abs/2308.00984)

[^19]: A Probability Space at Inception of Stochastic Process. (2025). *arXiv preprint*.
[https://arxiv.org/abs/2510.20824](https://arxiv.org/abs/2510.20824)

[^20]: Alpha Entropy Search for New Information-based Bayesian Optimization. (2024). *arXiv preprint*.
[https://arxiv.org/abs/2411.16586](https://arxiv.org/abs/2411.16586)

[^21]: Dorent, R., et al. (2025). Connecting Jensen-Shannon and Kullback-Leibler Divergences: A New Bound for Representation Learning. *arXiv preprint*.
[https://arxiv.org/abs/2510.20644](https://arxiv.org/abs/2510.20644)

[^22]: GAIT: A Geometric Approach to Information Theory. (2019). *arXiv preprint*.
[https://arxiv.org/abs/1906.08325](https://arxiv.org/abs/1906.08325)

[^23]: Statistical Inference for Random Unknowns via Modifications of Extended Likelihood. (2023). *arXiv preprint*.
[https://arxiv.org/abs/2310.09955](https://arxiv.org/abs/2310.09955)

[^24]: Maximum Ideal Likelihood Estimation: A Unified Inference Framework for Latent Variable Models. (2024). *arXiv preprint*.
[https://arxiv.org/abs/2410.01194](https://arxiv.org/abs/2410.01194)

---

## 著者リンク

- Blog: https://fumishiki.dev
- X: https://x.com/fumishiki
- LinkedIn: https://www.linkedin.com/in/fumitakamurakami
- GitHub: https://github.com/fumishiki
- Hugging Face: https://huggingface.co/fumishiki

---


---

## ライセンス

本記事は [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ja)（クリエイティブ・コモンズ 表示 - 非営利 - 継承 4.0 国際）の下でライセンスされています。

### ⚠️ 利用制限について

**本コンテンツは個人の学習目的に限り利用可能です。**

**以下のケースは事前の明示的な許可なく利用することを固く禁じます:**

1. **企業・組織内での利用（営利・非営利問わず）**
   - 社内研修、教育カリキュラム、社内Wikiへの転載
   - 大学・研究機関での講義利用
   - 非営利団体での研修利用
   - **理由**: 組織内利用では帰属表示が削除されやすく、無断改変のリスクが高いため

2. **有料スクール・情報商材・セミナーでの利用**
   - 受講料を徴収する場での配布、スクリーンショットの掲示、派生教材の作成

3. **LLM/AIモデルの学習データとしての利用**
   - 商用モデルのPre-training、Fine-tuning、RAGの知識ソースとして本コンテンツをスクレイピング・利用すること

4. **勝手に内容を有料化する行為全般**
   - 有料note、有料記事、Kindle出版、有料動画コンテンツ、Patreon限定コンテンツ等

**個人利用に含まれるもの:**
- 個人の学習・研究
- 個人的なノート作成（個人利用に限る）
- 友人への元記事リンク共有

**組織での導入をご希望の場合**は、必ず著者に連絡を取り、以下を遵守してください:
- 全ての帰属表示リンクを維持
- 利用方法を著者に報告

**無断利用が発覚した場合**、使用料の請求およびSNS等での公表を行う場合があります。
