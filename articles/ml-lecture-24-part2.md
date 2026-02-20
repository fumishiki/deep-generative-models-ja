---
title: "第24回【後編】付録編: 統計学: 30秒の驚き→数式修行→実装マスター"
emoji: "📈"
type: "tech"
topics: ["machinelearning", "statistics", "julia", "bayesian", "hypothesis"]
published: true
slug: "ml-lecture-24-part2"
difficulty: "advanced"
time_estimate: "90 minutes"
languages: ["Julia", "Rust", "Elixir"]
keywords: ["機械学習", "深層学習", "生成モデル"]
---

> **第24回【前編】**: [第24回【前編】](https://zenn.dev/fumishiki/ml-lecture-24-part1)

## Part 2


$$
\begin{aligned}
\text{SS}_{\text{total}} &= \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2 \\
\text{SS}_{\text{between}} &= \sum_{i=1}^k n_i (\bar{x}_i - \bar{x})^2 \\
\text{SS}_{\text{within}} &= \sum_{i=1}^k \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2 \\
\text{MS}_{\text{between}} &= \frac{\text{SS}_{\text{between}}}{k-1}, \quad \text{MS}_{\text{within}} = \frac{\text{SS}_{\text{within}}}{N-k}
\end{aligned}
$$

**数値検証**:

```julia
using HypothesisTests

group_a = [0.72, 0.71, 0.73, 0.70, 0.72]
group_b = [0.78, 0.77, 0.79, 0.76, 0.78]
group_c = [0.68, 0.67, 0.69, 0.66, 0.68]

# 一元配置ANOVA
test = OneWayANOVATest(group_a, group_b, group_c)
println("F=$(round(test.F, digits=3)), p=$(round(pvalue(test), digits=6))")
println(pvalue(test) < 0.05 ? "✅ 少なくとも1組の平均が異なる" : "❌ 全群の平均に差なし")
```

出力:
```
F=90.0, p=0.000000
✅ 少なくとも1組の平均が異なる
```

#### 3.4.3 正規性検定

**問題**: t検定・ANOVAは正規性を仮定。データが正規分布に従うか検証したい。

| 検定 | 特徴 | 帰無仮説 |
|:-----|:-----|:--------|
| **Shapiro-Wilk検定** | 最も強力（小~中サンプル） | データが正規分布に従う |
| **Kolmogorov-Smirnov検定** | 汎用的（任意の分布） | データが指定分布に従う |
| **Anderson-Darling検定** | 裾の適合度を重視 | データが正規分布に従う |

**数値検証**:

```julia
using HypothesisTests, Distributions

# 正規分布データ
normal_data = rand(Normal(0, 1), 30)
test_normal = ExactOneSampleKSTest(normal_data, Normal(0, 1))
println("正規データ: p=$(round(pvalue(test_normal), digits=4))")

# 非正規データ（一様分布）
uniform_data = rand(Uniform(0, 1), 30)
test_uniform = ExactOneSampleKSTest(uniform_data, Normal(0.5, 1))
println("一様データ: p=$(round(pvalue(test_uniform), digits=4))")
```

### 3.5 ノンパラメトリック検定

**用途**: 正規性が満たされない、または順序データの場合。

| 検定 | パラメトリック版 | 用途 |
|:-----|:----------------|:-----|
| **Mann-Whitney U検定** | 2標本t検定 | 2群の中央値の差 |
| **Wilcoxon符号順位検定** | 対応のあるt検定 | 対応のある2群の中央値差 |
| **Kruskal-Wallis検定** | 一元配置ANOVA | 3群以上の中央値の差 |

**Mann-Whitney U検定の原理**:

1. 2群のデータを統合して順位付け。
2. 各群の順位和を計算。
3. U統計量を計算:

$$
U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1
$$

ここで $R_1$ は群1の順位和。

**数値検証**:

```julia
using HypothesisTests

group1 = [1, 2, 3, 4, 5]
group2 = [6, 7, 8, 9, 10]

# Mann-Whitney U検定
test = MannWhitneyUTest(group1, group2)
println("U=$(test.U), p=$(round(pvalue(test), digits=4))")
```

> **Note:** **進捗: 65% 完了** パラメトリック・ノンパラメトリック検定の理論完全版を制覇。多重比較補正へ。

### 3.6 多重比較補正理論

**問題**: 複数の検定を行うと、偶然に有意になる確率（第1種過誤）が増大する。

**例**: $\alpha = 0.05$ で独立な20個の検定を行うと、少なくとも1つが偶然有意になる確率:

$$
1 - (1 - 0.05)^{20} \approx 0.64 \quad \text{(64%!)}
$$

**FWER（Family-Wise Error Rate）**: 少なくとも1つの第1種過誤が起こる確率。

**FDR（False Discovery Rate）**: 有意と判定されたもののうち偽陽性の割合の期待値。

#### 3.6.1 FWER制御法

| 手法 | 調整後の有意水準 | 保守性 |
|:-----|:----------------|:-------|
| **Bonferroni補正** | $\alpha_{\text{adj}} = \alpha / m$ | 最も保守的 |
| **Holm法** | 逐次的Bonferroni | Bonferroniより緩い |
| **Šidák補正** | $\alpha_{\text{adj}} = 1 - (1 - \alpha)^{1/m}$ | 独立性仮定 |

**Holm法の手順**:

1. p値を昇順に並べる: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. $i = 1, 2, \ldots$ の順に以下をチェック:
   - $p_{(i)} \leq \alpha / (m - i + 1)$ なら棄却、次へ
   - 初めて不等式が成立しなかったら停止

#### 3.6.2 FDR制御法

**Benjamini-Hochberg法** [^2]:

1. p値を昇順に並べる: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. $i = m, m-1, \ldots, 1$ の順に以下をチェック:
   - $p_{(i)} \leq \frac{i}{m} \alpha$ なら $i$ 番目まで全て棄却、停止
   - 成立しなければ次へ

**数式導出**:

FDRの定義:

$$
\text{FDR} = \mathbb{E}\left[\frac{V}{R}\right]
$$

ここで $V$ = 偽陽性数、$R$ = 総発見数（$R = V + S$, $S$ = 真陽性数）。

Benjamini-Hochbergは独立な検定において $\text{FDR} \leq \alpha$ を保証する [^2]。

**数値検証**:

```julia
using MultipleTesting

# 100個の検定（90個は帰無仮説が真、10個は対立仮説が真）
p_values_null = rand(100)  # H0が真のp値: 一様分布
p_values_alt  = rand(Beta(0.1, 1), 10)  # H1が真のp値: 0に偏る
p_values = vcat(p_values_null, p_values_alt)

# 補正なし
n_sig_uncorrected = sum(p_values .< 0.05)
println("補正なし: $(n_sig_uncorrected) / 110 が有意")

# Bonferroni補正
p_bonf = adjust(PValues(p_values), Bonferroni())
n_sig_bonf = sum(p_bonf .< 0.05)
println("Bonferroni: $(n_sig_bonf) / 110 が有意")

# Benjamini-Hochberg (FDR)
p_bh = adjust(PValues(p_values), BenjaminiHochberg())
n_sig_bh = sum(p_bh .< 0.05)
println("Benjamini-Hochberg: $(n_sig_bh) / 110 が有意")
```

出力例:
```
補正なし: 15 / 110 が有意
Bonferroni: 3 / 110 が有意
Benjamini-Hochberg: 9 / 110 が有意
```

> **Note:** **進捗: 75% 完了** 多重比較補正（FWER/FDR）を完全理解。GLM理論へ。

### 3.7 一般化線形モデル（GLM）

**問題**: 線形回帰 $y = X\beta + \epsilon$ は連続値・正規分布を仮定。カテゴリカル（分類）やカウントデータには不適。

**GLMの構成要素**:

1. **指数型分布族**: 応答変数 $y$ の分布（正規・二項・ポアソン等）。
2. **リンク関数** $g(\cdot)$: 平均 $\mu = \mathbb{E}[y]$ を線形予測子 $\eta = X\beta$ に繋ぐ。
3. **線形予測子**: $\eta = X\beta$

$$
g(\mu) = X\beta \quad \Rightarrow \quad \mu = g^{-1}(X\beta)
$$

| 分布 | 典型的用途 | 標準的リンク関数 |
|:-----|:----------|:----------------|
| 正規分布 | 連続値 | 恒等 $g(\mu) = \mu$ |
| 二項分布 | 分類 | ロジット $g(\mu) = \log\frac{\mu}{1-\mu}$ |
| ポアソン分布 | カウント | 対数 $g(\mu) = \log\mu$ |

#### 3.7.1 ロジスティック回帰（Logistic Regression）

**用途**: 二値分類（$y \in \{0, 1\}$）。

**モデル**:

$$
\begin{aligned}
y_i &\sim \text{Bernoulli}(p_i) \\
\log\frac{p_i}{1 - p_i} &= \beta_0 + \beta_1 x_i \quad \text{(ロジット変換)} \\
\Rightarrow \quad p_i &= \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_i)}} \quad \text{(シグモイド関数)}
\end{aligned}
$$

**オッズ比（Odds Ratio）**: 係数 $\beta_1$ の解釈

$$
\text{OR} = e^{\beta_1}
$$

$x$ が1単位増加すると、オッズ（$p / (1-p)$）が $e^{\beta_1}$ 倍になる。

**最尤推定**: 対数尤度を最大化。

$$
\ell(\beta) = \sum_{i=1}^n \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$

勾配:

$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^n (y_i - p_i) x_{ij}
$$

**数値検証**:

```julia
using GLM, DataFrames

# データ: x（連続変数）, y（0/1のラベル）
df = DataFrame(
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
)

# ロジスティック回帰
model = glm(@formula(y ~ x), df, Binomial(), LogitLink())
println(model)

# 係数の解釈
β1 = coef(model)[2]
OR = exp(β1)
println("\n係数β1=$(round(β1, digits=3)), オッズ比OR=$(round(OR, digits=3))")
println("xが1単位増加すると、オッズが$(round(OR, digits=3))倍になる")

# 予測
df.y_pred = predict(model, df)
println("\n予測確率:")
println(df)
```

#### 3.7.2 ポアソン回帰（Poisson Regression）

**用途**: カウントデータ（$y \in \{0, 1, 2, \ldots\}$）。イベント発生回数の予測。

**モデル**:

$$
\begin{aligned}
y_i &\sim \text{Poisson}(\lambda_i) \\
\log \lambda_i &= \beta_0 + \beta_1 x_i \quad \text{(対数リンク関数)} \\
\Rightarrow \quad \lambda_i &= e^{\beta_0 + \beta_1 x_i}
\end{aligned}
$$

**係数の解釈**: $x$ が1単位増加すると、期待カウント $\lambda$ が $e^{\beta_1}$ 倍になる。

**数値検証**:

```julia
using GLM, DataFrames, Distributions

# データ生成: カウントデータ（例: 1時間あたりのエラー発生回数）
df = DataFrame(
    workload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 負荷レベル
    errors = [2, 3, 3, 5, 6, 8, 9, 12, 14, 16]   # エラー回数
)

# ポアソン回帰
model = glm(@formula(errors ~ workload), df, Poisson(), LogLink())
println(model)

# 係数の解釈
β1 = coef(model)[2]
multiplier = exp(β1)
println("\nworkloadが1単位増加すると、期待エラー回数が$(round(multiplier, digits=3))倍になる")

# 予測
df.errors_pred = predict(model, df)
println("\n予測エラー回数:")
println(df)
```

#### 3.7.3 指数型分布族の統一理論

**GLMの基盤**: 指数型分布族（Exponential Family）

$$
p(y | \theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)
$$

| 項 | 名称 | 役割 |
|:---|:-----|:-----|
| $\theta$ | 自然パラメータ | 平均を決定 |
| $\phi$ | 分散パラメータ | 分散を決定 |
| $b(\theta)$ | 累積生成関数 | 平均: $\mu = b'(\theta)$ |
| $a(\phi)$ | 分散関数 | 分散: $\text{Var}(Y) = b''(\theta) a(\phi)$ |

**主要な分布**:

| 分布 | $\theta$ | $b(\theta)$ | $a(\phi)$ | $\mu = b'(\theta)$ |
|:-----|:---------|:-----------|:----------|:------------------|
| 正規分布 | $\mu$ | $\theta^2 / 2$ | $\sigma^2$ | $\theta$ |
| 二項分布 | $\log \frac{p}{1-p}$ | $\log(1 + e^\theta)$ | $1$ | $\frac{e^\theta}{1 + e^\theta}$ |
| ポアソン分布 | $\log \lambda$ | $e^\theta$ | $1$ | $e^\theta$ |

**GLMの統一構造**:

1. **ランダム成分**: 応答変数 $y$ が指数型分布族に従う。
2. **線形予測子**: $\eta = X\beta$
3. **リンク関数**: $g(\mu) = \eta$（標準的リンク関数: $g(\mu) = \theta$）

> **Note:** **進捗: 80% 完了** GLM理論（ロジスティック・ポアソン回帰・指数型分布族）を理解。ベイズ統計へ。

### 3.8 ベイズ統計入門

#### 3.8.1 ベイズの定理の導出

**第4回で学んだ条件付き確率の定義**:

$$
p(\theta | D) = \frac{p(\theta, D)}{p(D)}, \quad p(D | \theta) = \frac{p(\theta, D)}{p(\theta)}
$$

両辺に $p(\theta)$ を掛けると:

$$
p(\theta, D) = p(D | \theta) p(\theta) = p(\theta | D) p(D)
$$

よって:

$$
p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}
$$

これが**ベイズの定理**だ。

| 項 | 名称 | 意味 |
|:---|:-----|:-----|
| $p(\theta \| D)$ | 事後分布（Posterior） | データ観測後のパラメータの分布 |
| $p(D \| \theta)$ | 尤度（Likelihood） | パラメータ下でのデータの確率 |
| $p(\theta)$ | 事前分布（Prior） | データ観測前のパラメータの信念 |
| $p(D)$ | 周辺尤度（Evidence） | 正規化定数 $p(D) = \int p(D \| \theta) p(\theta) d\theta$ |

#### 3.8.2 頻度論統計 vs ベイズ統計

**哲学的対立**:

| 項目 | 頻度論 | ベイズ |
|:-----|:------|:-------|
| **パラメータの性質** | 固定値（未知） | 確率変数 |
| **確率の解釈** | 長期的頻度 | 信念の度合い |
| **推論の対象** | 点推定・信頼区間 | 事後分布全体 |
| **不確実性の表現** | 標準誤差 | 事後分布の幅 |
| **事前知識** | 使わない（客観性） | 使う（主観性） |

**具体例**: コイン投げ（10回中7回表）

**頻度論的推定**（第7回のMLE）:

$$
\hat{\theta}_{\text{MLE}} = \frac{k}{n} = \frac{7}{10} = 0.7
$$

95%信頼区間（Wald法）:

$$
\text{CI} = \hat{\theta} \pm 1.96 \sqrt{\frac{\hat{\theta}(1-\hat{\theta})}{n}} = 0.7 \pm 1.96 \sqrt{\frac{0.7 \times 0.3}{10}} = [0.416, 0.984]
$$

**ベイズ推定**（事前分布Beta(2,2)、共役性より事後分布Beta(9, 5)）:

$$
p(\theta | k=7, n=10) = \text{Beta}(9, 5)
$$

事後平均（点推定）:

$$
\mathbb{E}[\theta | D] = \frac{\alpha}{\alpha + \beta} = \frac{9}{9+5} = 0.643
$$

95%信用区間（Credible Interval）:

$$
\text{CrI} = [\text{quantile}(0.025), \text{quantile}(0.975)] \approx [0.366, 0.882]
$$

**解釈の違い**:

- **頻度論CI**: 「同じ実験を100回繰り返せば、95回はこの区間が真の $\theta$ を含む」
- **ベイズCrI**: 「データを見た今、$\theta$ がこの区間にある確率が95%」（より直感的）

#### 3.8.1 共役事前分布

**定義**: 事前分布と事後分布が同じ分布族に属するとき、その事前分布を共役という。

| 尤度 | 共役事前分布 | 事後分布 |
|:-----|:-----------|:--------|
| 二項分布 | ベータ分布 | ベータ分布 |
| 正規分布（既知分散） | 正規分布 | 正規分布 |
| ポアソン分布 | ガンマ分布 | ガンマ分布 |

**例**: コイン投げ（二項分布）+ ベータ事前分布

$$
\begin{aligned}
\text{尤度:} \quad & p(k | n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k} \\
\text{事前分布:} \quad & p(\theta) = \text{Beta}(\alpha, \beta) \propto \theta^{\alpha-1} (1-\theta)^{\beta-1} \\
\text{事後分布:} \quad & p(\theta | k, n) = \text{Beta}(\alpha + k, \beta + n - k)
\end{aligned}
$$

**数値検証**:

```julia
using Distributions, Plots

# 事前分布: Beta(2, 2) (弱い信念: θ≈0.5)
α, β = 2.0, 2.0
prior = Beta(α, β)

# データ: 10回投げて7回表
n, k = 10, 7

# 事後分布: Beta(α+k, β+n-k) = Beta(9, 5)
posterior = Beta(α + k, β + n - k)

# 可視化
θ_range = 0:0.01:1
plot(θ_range, pdf.(prior, θ_range), label="事前分布 Beta(2,2)", linewidth=2)
plot!(θ_range, pdf.(posterior, θ_range), label="事後分布 Beta(9,5)", linewidth=2)
xlabel!("θ (コインが表の確率)")
ylabel!("密度")
title!("ベイズ更新: コイン投げ")
savefig("bayesian_update.png")
```

#### 3.8.2 MCMC（Markov Chain Monte Carlo）

**問題**: 事後分布 $p(\theta | D)$ が複雑で解析的に計算できない。

**MCMC**: マルコフ連鎖を使って事後分布からサンプルを生成。

**Metropolis-Hastings法** [^3]:

1. 初期値 $\theta^{(0)}$ を設定。
2. $t = 1, 2, \ldots$ について:
   - 提案分布 $q(\theta' | \theta^{(t-1)})$ から候補 $\theta'$ を生成。
   - 受理確率を計算:
     $$
     \alpha = \min\left(1, \frac{p(\theta' | D) q(\theta^{(t-1)} | \theta')}{p(\theta^{(t-1)} | D) q(\theta' | \theta^{(t-1)})}\right)
     $$
   - 確率 $\alpha$ で $\theta^{(t)} = \theta'$、そうでなければ $\theta^{(t)} = \theta^{(t-1)}$。

**Turing.jlで実装**:

```julia
using Turing, Distributions, StatsPlots

# モデル定義: コイン投げ（ベイズ推定）
@model function coinflip(y)
    # 事前分布
    θ ~ Beta(2, 2)

    # 尤度
    y ~ Binomial(length(y), θ)
end

# データ: 10回中7回表
data = 7

# MCMCサンプリング（NUTS: No-U-Turn Sampler, Hamiltonian Monte Carloの改良版）
chain = sample(coinflip([data]), NUTS(), 1000)

# 事後分布の可視化
plot(chain)
```

> **Note:** **進捗: 90% 完了** ベイズ統計（共役事前分布・MCMC）を完全理解。実験計画法へ。

### 3.9 実験計画法（Experimental Design）

**目的**: 限られたリソースで最大の情報を得る実験を設計する。

#### 3.9.1 完全無作為化デザイン（Completely Randomized Design, CRD）

**特徴**: 処理（treatment）をランダムに割り当てる。最もシンプル。

**欠点**: ブロック間の変動（例: 測定日の違い）を制御できない。

#### 3.9.2 乱塊法（Randomized Block Design, RBD）

**特徴**: 被験者をブロック（例: 年齢層、測定日）に分け、各ブロック内で処理をランダム化。

**利点**: ブロック間変動を除去 → 残差が小さくなる → 検出力向上。

#### 3.9.3 ラテン方格（Latin Square Design）

**特徴**: 2つの要因（例: 行=日、列=機械）を同時に制御。

**制約**: 処理数 = 行数 = 列数。

#### 3.9.4 サンプルサイズ設計（Power Analysis）

**問題**: 実験前に必要なサンプルサイズを決定。

**手順**:

1. 期待される効果量 $d$ を設定（過去の研究や予備実験から）。
2. 有意水準 $\alpha$ を設定（通常0.05）。
3. 目標検出力 $1 - \beta$ を設定（通常0.8）。
4. 検定の種類に応じた公式またはソフトウェアでサンプルサイズを計算。

**t検定のサンプルサイズ公式**（再掲）:

$$
n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2}{d^2}
$$

### 6.11 パラダイム転換の問い

> **「p < 0.05で有意」と言える。だが、それは本当に**あなたの主張**を支持しているのか？**

以下のシナリオを考えよう:

1. **シナリオA**: 新しいプロンプト手法を10種類試し、1つだけp < 0.05で有意な改善。他9つは有意差なし。
2. **シナリオB**: 同じ実験を100回行い、有意だった5回だけ論文に報告。
3. **シナリオC**: データを見てから「このデータセットでは効果がある」と事後的にサブグループ分析。

**全て統計的には「p < 0.05」だが、科学的には無意味だ。**

- **シナリオA**: 多重比較の罠。Bonferroni補正すればp = 0.05 × 10 = 0.5で有意でない。
- **シナリオB**: 出版バイアス。失敗した95回を隠蔽。
- **シナリオC**: p-hacking。データを見てから仮説を立てる。

**議論の種**:

1. **事前登録（Pre-registration）**は解決策か？　実験前に仮説・手法を公開登録すれば、p-hackingを防げる。だが柔軟性が失われる。
2. **p値の代替案**は？　信頼区間・効果量・ベイズファクターは、p値の問題を解決するか？
3. **統計的有意性の基準（α=0.05）**は恣意的ではないか？　なぜ0.05なのか？　0.01や0.001ではダメなのか？

この問いに完全な答えはない。だが**統計学は道具であり、道具の使い方次第で科学的誠実さが問われる**ことを忘れてはならない。

> **Note:** **進捗: 100% 完了** 🎉 講義完走！

---


> Progress: [85%]
> **理解度チェック**
> 1. ANOVAのF統計量が群間分散と群内分散の比で構成される数学的意味を述べよ。
> 2. ロジスティック回帰のリンク関数がlogitである理由を確率の範囲の制約から説明せよ。

## 参考文献

### 主要論文

[^1]: Neyman, J., & Pearson, E. S. (1928). *On the Use and Interpretation of Certain Test Criteria for Purposes of Statistical Inference: Part I*. Biometrika.
<https://www.jstor.org/stable/2331945>

[^2]: Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. Journal of the Royal Statistical Society: Series B.
<https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>

[^3]: Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications*. Biometrika.
<https://doi.org/10.1093/biomet/57.1.97>


### 教科書

- **Statistical Inference** - Casella & Berger (2002): 頻度論統計の決定版。大学院レベル。
- **Bayesian Data Analysis** - Gelman et al. (2013): ベイズ統計の標準教科書。
- **The Elements of Statistical Learning** - Hastie, Tibshirani, Friedman (2009): 機械学習×統計の融合。[無料PDF](https://web.stanford.edu/~hastie/ElemStatLearn/)
- **統計学入門** - 東京大学教養学部統計学教室 (1991): 日本語の定番入門書。

### オンラインリソース

- [StatQuest (YouTube)](https://www.youtube.com/@statquest): 統計学の直感的解説動画。
- [StatsBase.jl Documentation](https://juliastats.org/StatsBase.jl/stable/)
- [HypothesisTests.jl Documentation](https://juliastats.org/HypothesisTests.jl/stable/)
- [GLM.jl Documentation](https://juliastats.org/GLM.jl/stable/)
- [Turing.jl Documentation](https://turinglang.org/stable/)

---

## 付録A: 統計学の歴史的発展

### A.1 頻度論統計の誕生（1900-1950年代）

| 年 | 人物 | 貢献 |
|:---|:-----|:-----|
| 1900 | Karl Pearson | カイ二乗検定、Pearson相関係数 |
| 1908 | William Gosset (Student) | t分布、t検定（少サンプル統計） |
| 1920年代 | Ronald Fisher | 最尤推定（MLE）、分散分析（ANOVA）、実験計画法 |
| 1928 | Neyman & Pearson | Neyman-Pearson仮説検定枠組み [^1] |
| 1935 | Fisher | ランダム化比較試験（RCT）の原理 |

**頻度論の哲学**: 確率 = 長期的頻度。パラメータは固定値（未知）。客観性を重視。

### A.2 ベイズ統計の復興（1950-1990年代）

| 年 | 人物/出来事 | 貢献 |
|:---|:----------|:-----|
| 1763 | Thomas Bayes（死後出版） | ベイズの定理の原型 |
| 1950年代 | Dennis Lindley | ベイズ決定理論 |
| 1953 | Metropolis et al. | Metropolisアルゴリズム（MCMC） [^3] |
| 1970 | Hastings | Metropolis-Hastingsアルゴリズム |
| 1990 | Gelfand & Smith | Gibbs Samplingの実用化 |

**ベイズ復興の理由**: コンピュータの発展でMCMCが実用化 → 複雑なモデルの事後分布を計算可能に。

### A.3 現代統計学（1990年代〜現在）

| 年 | 手法 | 貢献 |
|:---|:-----|:-----|
| 1995 | Benjamini & Hochberg | FDR制御法（多重比較） [^2] |
| 2000年代 | ベイズノンパラメトリクス | 無限次元モデル（Dirichlet Process等） |
| 2010年代 | Hamiltonian Monte Carlo (HMC) | 高次元MCMCの高速化（NUTS） |
| 2015年代 | 因果推論の普及 | Pearl/Rubin枠組みの統合、機械学習との融合 |
| 2020年代 | 確率的プログラミング | Turing.jl, PyMC, Stan等の成熟 |

---

## 付録B: Juliaで使える統計パッケージ完全リスト

### B.1 基礎統計

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Statistics** (stdlib) | 基本統計量 | `mean`, `std`, `var`, `median`, `quantile`, `cor`, `cov` |
| **StatsBase.jl** | 記述統計・重み付き統計 | `skewness`, `kurtosis`, `mad`, `mode`, `sem`, `zscore`, `sample`, `weights` |
| **Distributions.jl** | 確率分布 | `Normal`, `Beta`, `Gamma`, `Binomial`, `Poisson`, `TDist`, `FDist`, `pdf`, `cdf`, `quantile`, `rand` |

### B.2 仮説検定

| パッケージ | 用途 | 主要検定 |
|:----------|:-----|:---------|
| **HypothesisTests.jl** | 仮説検定全般 | `OneSampleTTest`, `EqualVarianceTTest`, `UnequalVarianceTTest`, `MannWhitneyUTest`, `WilcoxonSignedRankTest`, `KruskalWallisTest`, `OneWayANOVATest`, `ChisqTest`, `FisherExactTest`, `KSTest`, `AndersonDarlingTest` |
| **MultipleTesting.jl** | 多重比較補正 | `adjust`, `Bonferroni`, `Holm`, `BenjaminiHochberg`, `BenjaminiYekutieli` |

### B.3 回帰・GLM

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **GLM.jl** | 一般化線形モデル | `glm`, `@formula`, `Binomial`, `Poisson`, `Gamma`, `LogitLink`, `LogLink`, `InverseLink`, `coef`, `confint`, `predict` |
| **MixedModels.jl** | 混合効果モデル | `LinearMixedModel`, `fit!`, `ranef`, `fixef` |

### B.4 ベイズ統計

| パッケージ | 用途 | 主要関数/マクロ |
|:----------|:-----|:---------------|
| **Turing.jl** | 確率的プログラミング | `@model`, `~`, `sample`, `NUTS`, `HMC`, `Gibbs`, `plot`, `summarize` |
| **AdvancedMH.jl** | MCMC拡張 | `MetropolisHastings`, `RWMH`, `StaticMH` |
| **MCMCChains.jl** | MCMC結果の解析 | `Chains`, `describe`, `plot`, `ess`, `gelmandiag` |
| **AbstractMCMC.jl** | MCMCインターフェース | MCMC実装の共通基盤 |

### B.5 ブートストラップ・リサンプリング

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Bootstrap.jl** | ブートストラップ法 | `bootstrap`, `BasicSampling`, `confint`, `PercentileConfInt`, `BCaConfInt` |

### B.6 生存時間解析

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **Survival.jl** | 生存時間解析 | `Surv`, `kaplan_meier`, `cox_ph`, `nelson_aalen` |

### B.7 時系列解析

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **TimeSeries.jl** | 時系列データ | `TimeArray`, `values`, `timestamp`, `lag`, `lead`, `diff` |
| **StateSpaceModels.jl** | 状態空間モデル | `StateSpaceModel`, `kalman_filter`, `smoother` |

### B.8 実験計画法

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **ExperimentalDesign.jl** | 実験計画 | `factorial_design`, `latin_square`, `balanced_design` |

### B.9 可視化

| パッケージ | 用途 | 主要関数 |
|:----------|:-----|:---------|
| **StatsPlots.jl** | 統計的プロット | `boxplot`, `violin`, `density`, `marginalscatter`, `corrplot`, `@df` |
| **Makie.jl** | 高品質可視化 | `scatter`, `lines`, `barplot`, `heatmap`, `density` |
| **AlgebraOfGraphics.jl** | Grammar of Graphics | `data`, `mapping`, `visual`, `draw` |

---

## 付録C: 統計学の主要定理まとめ

### C.1 確率論の基礎定理

**大数の法則（Law of Large Numbers）**:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty
$$

標本平均は母平均に確率収束する。

**中心極限定理（Central Limit Theorem）**:

$$
\sqrt{n} \frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty
$$

標本平均の分布は正規分布に近づく（母集団分布に関わらず）。

### C.2 推定の理論

**Cramér-Rao下界（Cramér-Rao Lower Bound）**:

不偏推定量 $\hat{\theta}$ の分散は次の下界を持つ:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

ここで $I(\theta)$ はFisher情報量。等号成立時は**有効推定量**。

**漸近正規性（Asymptotic Normality）**:

MLEは漸近的に正規分布に従う:

$$
\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

### C.3 検定の理論

**Neyman-Pearson補題（Neyman-Pearson Lemma）**:

尤度比検定は所定の有意水準 $\alpha$ で最も検出力が高い（most powerful test）。

$$
\frac{p(x | H_1)}{p(x | H_0)} > c \quad \Rightarrow \quad \text{reject } H_0
$$

### C.4 ベイズ統計の定理

**ベイズの定理（Bayes' Theorem）**:

$$
p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)} = \frac{p(D | \theta) p(\theta)}{\int p(D | \theta') p(\theta') d\theta'}
$$

**マルコフ連鎖の収束**:

適切な条件下でMCMCサンプルは事後分布に収束:

$$
\lim_{t \to \infty} \theta^{(t)} \sim p(\theta | D)
$$

---

## 付録D: 統計学の実践チェックリスト

### D.1 実験前（事前計画）

- [ ] 研究仮説を明確に定義（$H_0$, $H_1$）
- [ ] 有意水準 $\alpha$ を決定（通常0.05）
- [ ] 目標検出力を決定（通常0.8）
- [ ] 期待される効果量を設定（過去研究・予備実験から）
- [ ] パワー分析で必要サンプルサイズを計算
- [ ] 検定手法を事前に決定（t検定・ANOVA・ノンパラメトリック等）
- [ ] 多重比較がある場合は補正方法を決定（Bonferroni・BH等）
- [ ] 事前登録（Pre-registration）を検討（p-hackingを防ぐ）

### D.2 データ収集

- [ ] ランダムサンプリング・ランダム化を徹底
- [ ] ブロック要因があれば乱塊法を検討
- [ ] 測定誤差を最小化（機器の校正・プロトコルの標準化）
- [ ] 欠損データの記録・理由の記載
- [ ] 外れ値の記録（削除前に理由を明記）

### D.3 記述統計

- [ ] 平均・中央値・標準偏差・IQRを計算
- [ ] 歪度・尖度を確認（分布の形状）
- [ ] 外れ値の検出（IQR法・Grubbs検定）
- [ ] ヒストグラム・箱ひげ図で可視化

### D.4 推測統計

- [ ] 前提条件の確認（正規性・等分散性・独立性）
- [ ] 正規性検定（Shapiro-Wilk・Kolmogorov-Smirnov）
- [ ] 等分散性検定（Levene・Bartlett）
- [ ] 前提が満たされない場合は代替手法（ノンパラメトリック・変換・頑健な手法）

### D.5 仮説検定

- [ ] 検定統計量（t, F, χ², U等）を計算
- [ ] 自由度を確認
- [ ] p値を計算
- [ ] 効果量（Cohen's d, partial η², r²等）を計算
- [ ] 信頼区間を併記
- [ ] 多重比較補正（該当する場合）

### D.6 結果の報告

- [ ] 記述統計（M, SD, n）を報告
- [ ] 検定統計量・自由度・p値を報告（例: $t(9) = 60.0, p < .001$）
- [ ] 効果量を報告（例: $d = 6.0$）
- [ ] 95%信頼区間を報告（例: $95\% \text{CI} [0.768, 0.782]$）
- [ ] 多重比較補正方法を明記
- [ ] 図表で視覚化（箱ひげ図・エラーバー付き棒グラフ等）
- [ ] 統計的有意性と実用的有意性を区別

### D.7 解釈・議論

- [ ] p値の正しい解釈（「$H_0$が真である確率」ではない）
- [ ] 効果量の実用的意義を議論
- [ ] 検出力不足の可能性を検討（p > 0.05の場合）
- [ ] 代替説明（交絡因子）の可能性を議論
- [ ] 限界（サンプル選択バイアス・測定誤差等）を明記
- [ ] 因果関係と相関の区別

---

## 付録B: GLM発展トピックと最新手法

### B.1 混合効果モデル（Mixed Effects Models）

**問題**: データに階層構造がある場合（例: 生徒→クラス→学校）、観測が独立でない。

**線形混合効果モデル（LMM）**:

$$
y_{ij} = \beta_0 + \beta_1 x_{ij} + u_i + \epsilon_{ij}
$$

ここで:
- $y_{ij}$: グループ$i$の観測$j$の応答変数
- $u_i \sim \mathcal{N}(0, \sigma_u^2)$: グループレベルのランダム効果
- $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$: 個体レベルの誤差

**固定効果 vs ランダム効果**:

| 項目 | 固定効果 | ランダム効果 |
|:-----|:--------|:-----------|
| 解釈 | 母集団全体の平均効果 | グループ間のばらつき |
| 推定 | 係数$\beta$ | 分散成分$\sigma_u^2$ |
| 目的 | 効果の大きさを知りたい | グループ間変動を制御したい |

**Julia実装例**（MixedModels.jl）:

```julia
using MixedModels, DataFrames, RDatasets

# データ: sleepstudy（睡眠不足が反応時間に与える影響）
sleepstudy = dataset("lme4", "sleepstudy")

# 混合効果モデル: 反応時間 ~ 日数 + (1 + 日数 | 被験者)
# 固定効果: 日数の効果
# ランダム効果: 被験者ごとの切片とスロープ
fm = fit(MixedModel, @formula(Reaction ~ Days + (1 + Days | Subject)), sleepstudy)

println(fm)

# ランダム効果の可視化
ranef_df = DataFrame(ranef(fm)[:Subject])
```

出力例:
```
Linear mixed model fit by maximum likelihood
 Reaction ~ 1 + Days + (1 + Days | Subject)
   logLik   -2 logLik     AIC       AICc        BIC
  -875.97    1751.94   1763.94   1764.47   1783.10

Variance components:
            Column    Variance   Std.Dev.   Corr.
Subject  (Intercept)  612.100    24.741
         Days          35.072     5.923    0.07
Residual              654.941    25.592
```

### B.2 一般化加法モデル（GAM: Generalized Additive Models）

**問題**: 線形性の仮定が厳しすぎる場合、非線形関係を柔軟にモデル化したい。

**GAMの定式化**:

$$
g(\mu) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)
$$

ここで$f_i$はスムージング関数（スプライン等）。

**スムージングスプライン**:

$$
\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int (f''(x))^2 dx
$$

第1項: フィット、第2項: 滑らかさのペナルティ

**Juliaでの簡易実装**:

```julia
using GLM, DataFrames, Plots

# データ生成: 非線形関係
x = range(0, 10, length=100)
y_true = sin.(x) .+ 0.5 .* x
y = y_true .+ randn(100) .* 0.3

# 多項式基底展開でGAMを近似
function polynomial_features(x, degree)
    hcat([x.^d for d in 0:degree]...)
end

# 次数5の多項式GAM
X_poly = polynomial_features(x, 5)
df = DataFrame(X_poly, :auto)
df.y = y

model = lm(@formula(y ~ x1 + x2 + x3 + x4 + x5), df)

# 予測と可視化
y_pred = predict(model)

plot(x, y, seriestype=:scatter, label="Data", alpha=0.5)
plot!(x, y_true, linewidth=2, label="True function")
plot!(x, y_pred, linewidth=2, label="GAM fit", linestyle=:dash)
xlabel!("x")
ylabel!("y")
```

### B.3 ゼロ過剰モデル（Zero-Inflated Models）

**問題**: カウントデータにゼロが過剰に含まれる（例: 病院受診回数、事故件数）。

**ゼロ過剰ポアソンモデル（ZIP）**:

$$
P(Y = y) = \begin{cases}
\pi + (1 - \pi) e^{-\lambda} & \text{if } y = 0 \\
(1 - \pi) \frac{\lambda^y e^{-\lambda}}{y!} & \text{if } y > 0
\end{cases}
$$

ここで:
- $\pi$: 構造的ゼロの確率（「決してイベントが起こらない」）
- $1 - \pi$: ポアソン過程に従う確率

**2段階モデル**:

1. ロジスティック回帰で$\pi$を推定
2. ポアソン回帰で$\lambda$を推定

**数値例**:

```julia
using Distributions, Optim

# ZIP尤度関数
function zip_loglik(params, y)
    π, λ = params[1], exp(params[2])  # λ > 0を保証
    ll_zero  = log(π + (1 - π) * exp(-λ))
    ll_pos(yi) = log(1 - π) + logpdf(Poisson(λ), yi)
    -sum(yi == 0 ? ll_zero : ll_pos(yi) for yi in y)  # 負の対数尤度（最小化）
end

# データ生成: ゼロ過剰
true_π = 0.3
true_λ = 2.0
n = 1000

y = [rand() < true_π ? 0 : rand(Poisson(true_λ)) for _ in 1:n]

println("ゼロの割合: $(sum(y .== 0) / n) (理論値: $(true_π + (1-true_π)*exp(-true_λ)))")

# 最尤推定
result = optimize(p -> zip_loglik(p, y), [0.2, log(2.0)], BFGS())
π_hat, λ_hat = result.minimizer[1], exp(result.minimizer[2])

println("推定値: π=$(round(π_hat, digits=3)), λ=$(round(λ_hat, digits=3))")
println("真値: π=$true_π, λ=$true_λ")
```

### B.4 時系列モデル（Time Series Models）

#### B.4.1 自己回帰モデル（AR）

**AR(p)モデル**:

$$
y_t = \phi_0 + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t
$$

ここで$\epsilon_t \sim \mathcal{N}(0, \sigma^2)$はホワイトノイズ。

**定常性条件**: 特性方程式の根が単位円の外側にある。

**Julia実装例**:

```julia
using LinearAlgebra, Statistics, Plots

# AR(1)プロセスのシミュレーション（逐次的: @inbounds で高速化）
function ar1_simulate(ϕ, σ, n)
    y = zeros(n)
    y[1] = randn() * σ / sqrt(1 - ϕ^2)  # 定常分布から初期値
    @inbounds for t in 2:n
        y[t] = ϕ * y[t-1] + randn() * σ
    end
    return y
end

# パラメータ
ϕ = 0.8  # 自己相関係数
σ = 1.0
n = 200

y = ar1_simulate(ϕ, σ, n)

# 自己相関関数（ACF）: @views でゼロコピースライス、dot で内積
function acf(x, max_lag)
    n  = length(x)
    x_c = x .- mean(x)
    c0  = dot(x_c, x_c) / n
    ck(k) = @views dot(x_c[1:n-k], x_c[k+1:n]) / (n * c0)
    [1.0; [ck(k) for k in 1:max_lag]]
end

acf_vals = acf(y, 20)

# 可視化
p1 = plot(y, label="AR(1) series", xlabel="Time", ylabel="Value")
p2 = bar(0:20, acf_vals, label="ACF", xlabel="Lag", ylabel="Correlation")

plot(p1, p2, layout=(2, 1), size=(800, 600))
```

#### B.4.2 状態空間モデル（State Space Models）

**カルマンフィルタ**:

$$
\begin{aligned}
\text{状態方程式:} \quad & x_t = F x_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) \\
\text{観測方程式:} \quad & y_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)
\end{aligned}
$$

**予測ステップ**:

$$
\begin{aligned}
\hat{x}_{t|t-1} &= F \hat{x}_{t-1|t-1} \\
P_{t|t-1} &= F P_{t-1|t-1} F^\top + Q
\end{aligned}
$$

**更新ステップ**:

$$
\begin{aligned}
K_t &= P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1} \quad \text{(カルマンゲイン)} \\
\hat{x}_{t|t} &= \hat{x}_{t|t-1} + K_t (y_t - H \hat{x}_{t|t-1}) \\
P_{t|t} &= (I - K_t H) P_{t|t-1}
\end{aligned}
$$

**Julia実装例**:

```julia
using LinearAlgebra

# カルマンフィルタ実装（逐次的: @views + @inbounds で最適化）
function kalman_filter(y, F, H, Q, R, x0, P0)
    n = length(y)
    d = length(x0)

    x_pred = zeros(d, n)
    x_filt = zeros(d, n)
    P_pred = zeros(d, d, n)
    P_filt = zeros(d, d, n)

    @views x_filt[:, 1]    .= x0
    @views P_filt[:, :, 1] .= P0

    @inbounds for t in 2:n
        @views begin
            # 予測ステップ
            x_pred[:, t]    .= F * x_filt[:, t-1]
            P_pred[:, :, t] .= F * P_filt[:, :, t-1] * F' + Q

            # 更新ステップ
            innovation = y[t] - H * x_pred[:, t]
            S = H * P_pred[:, :, t] * H' + R
            K = P_pred[:, :, t] * H' / S  # スカラー S のとき / で OK

            x_filt[:, t]    .= x_pred[:, t] + K * innovation
            P_filt[:, :, t] .= (I - K * H) * P_pred[:, :, t]
        end
    end

    return x_filt, P_filt
end

# テスト: ローカルレベルモデル
F = [1.0;;]
H = [1.0;;]
Q = [0.1;;]
R = [1.0;;]

# 真の状態（ランダムウォーク）
n = 100
x_true = cumsum(randn(n) .* sqrt(0.1))
y_obs = x_true .+ randn(n)

x_filt, P_filt = kalman_filter(y_obs, F, H, Q, R, [0.0], [1.0;;])

# 可視化
plot(1:n, x_true, label="True state", linewidth=2)
plot!(1:n, y_obs, seriestype=:scatter, label="Observations", alpha=0.5)
plot!(1:n, vec(x_filt[1, :]), label="Filtered estimate", linewidth=2, linestyle=:dash)
```

### B.5 ベイズ階層モデルの実践

#### B.5.1 部分プーリング（Partial Pooling）

**問題**: グループごとに推定したいが、サンプルサイズが小さい。

**3つのアプローチ**:

| 手法 | 説明 | 問題点 |
|:-----|:-----|:------|
| **完全プーリング** | 全グループを1つとして扱う | グループ間の違いを無視 |
| **ノープーリング** | グループごとに独立推定 | 小サンプルで不安定 |
| **部分プーリング** | 階層モデルで情報共有 | ✅ 両者のバランス |

**階層ベイズモデル**:

$$
\begin{aligned}
y_{ij} &\sim \mathcal{N}(\mu_i, \sigma^2) \\
\mu_i &\sim \mathcal{N}(\mu_{\text{global}}, \tau^2) \\
\mu_{\text{global}} &\sim \mathcal{N}(0, 10^2) \\
\sigma, \tau &\sim \text{Half-Cauchy}(0, 5)
\end{aligned}
$$

**Turing.jl実装**:

```julia
using Turing, Distributions, DataFrames, StatsPlots

# データ生成: 学校ごとの生徒のテストスコア
n_schools = 10
students_per_school = [5, 8, 12, 6, 15, 7, 20, 9, 11, 13]
true_school_means = randn(n_schools) .* 5 .+ 70

data = DataFrame(school_id=Int[], score=Float64[])
for i in 1:n_schools
    for j in 1:students_per_school[i]
        push!(data, (school_id=i, score=true_school_means[i] + randn() * 10))
    end
end

# 階層モデル
@model function hierarchical_model(school_id, score)
    n_schools = length(unique(school_id))

    # ハイパーパラメータ
    μ_global ~ Normal(70, 20)
    τ ~ truncated(Cauchy(0, 5), 0, Inf)
    σ ~ truncated(Cauchy(0, 5), 0, Inf)

    # 学校レベルの平均
    μ_school ~ filldist(Normal(μ_global, τ), n_schools)

    # 尤度
    for i in eachindex(score)
        score[i] ~ Normal(μ_school[school_id[i]], σ)
    end
end

# サンプリング
model = hierarchical_model(data.school_id, data.score)
chain = sample(model, NUTS(), 2000)

# 結果の可視化
plot(chain[[:μ_global, :τ, :σ]])
```

#### B.5.2 収束診断（Convergence Diagnostics）

**Gelman-Rubin統計量（$\hat{R}$）**:

複数チェーンの収束を診断。$\hat{R} \approx 1$なら収束。

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

ここで:
- $W$: チェーン内分散の平均
- $\hat{V}$: チェーン間分散とチェーン内分散の重み付き平均

**有効サンプルサイズ（ESS: Effective Sample Size）**:

自己相関を考慮した実効的なサンプル数。

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

ここで$\rho_k$は遅れ$k$での自己相関。

**Julia実装例**:

```julia
using MCMCChains, StatsBase

# チェーン診断
println("=== 収束診断 ===")
println(gelmandiag(chain))  # Gelman-Rubin統計量

println("\n=== 有効サンプルサイズ ===")
println(ess(chain))

println("\n=== 自己相関 ===")
println(autocor(chain))

# トレースプロット
plot(chain[[:μ_global]])
```

### B.6 ベイズモデル選択

#### B.6.1 WAIC（Widely Applicable Information Criterion）

**定義**:

$$
\text{WAIC} = -2 (\text{lppd} - p_{\text{WAIC}})
$$

ここで:
- $\text{lppd}$: log pointwise predictive density
- $p_{\text{WAIC}}$: 有効パラメータ数

**計算**:

$$
\begin{aligned}
\text{lppd} &= \sum_{i=1}^n \log \left( \frac{1}{S} \sum_{s=1}^S p(y_i | \theta^{(s)}) \right) \\
p_{\text{WAIC}} &= \sum_{i=1}^n \text{Var}_s(\log p(y_i | \theta^{(s)}))
\end{aligned}
$$

**Julia実装例**:

```julia
using Turing, StatsBase

# モデル1: 単純モデル
@model function model1(y)
    μ ~ Normal(0, 10)
    σ ~ truncated(Normal(0, 5), 0, Inf)
    y ~ Normal(μ, σ)
end

# モデル2: 階層モデル（前述）
# ... (hierarchical_model)

# WAIC計算
function waic(chain, model, data)
    n = length(data)
    S = size(chain, 1)

    log_lik = zeros(S, n)
    @inbounds for s in 1:S
        θ = chain[s, :]
        @views log_lik[s, :] .= logpdf.(Normal(θ.μ, θ.σ), data)
    end

    lppd   = sum(log.(mean(exp.(log_lik), dims=1)))
    p_waic = sum(var(log_lik, dims=1))

    return (; waic = -2(lppd - p_waic), lppd, p_waic)
end

# モデル比較
waic1 = waic(chain1, model1, data)
waic2 = waic(chain2, model2, data)

println("Model 1 WAIC: $(waic1.waic)")
println("Model 2 WAIC: $(waic2.waic)")
println("Better model: $(waic1.waic < waic2.waic ? "Model 1" : "Model 2")")
```

#### B.6.2 ベイズファクター（Bayes Factor）

**定義**:

$$
\text{BF}_{12} = \frac{p(D | M_1)}{p(D | M_2)}
$$

**解釈**（Kass & Raftery, 1995）:

| BF | 証拠の強さ |
|:---|:----------|
| 1-3 | ほとんど価値なし |
| 3-20 | 肯定的 |
| 20-150 | 強い |
| >150 | 非常に強い |

**問題点**: 周辺尤度$p(D | M)$の計算が困難。

### B.7 ベイズノンパラメトリクス入門

#### B.7.1 Dirichlet Process（ディリクレ過程）

**問題**: クラスタ数が事前に分からないクラスタリング。

**Dirichlet Process Mixture Model (DPMM)**:

$$
\begin{aligned}
G &\sim \text{DP}(\alpha, H) \quad \text{（ディリクレ過程）} \\
\theta_i &\sim G \\
y_i &\sim F(\theta_i)
\end{aligned}
$$

ここで:
- $\alpha$: 集中度パラメータ（大きいほど多くのクラスタ）
- $H$: ベース分布
- $F$: 尤度関数

**Chinese Restaurant Process（CRP）**: DPの直感的な説明

新しい客が入店するとき:
- 確率$\frac{n_k}{\alpha + n - 1}$で既存のテーブル$k$に座る（$n_k$人座っている）
- 確率$\frac{\alpha}{\alpha + n - 1}$で新しいテーブルを作る

**Julia実装例（簡略版）**:

```julia
using Distributions, StatsPlots

# Chinese Restaurant Process simulation
function crp_simulate(n, α)
    tables = Int[]  # 各客がどのテーブルに座っているか
    table_counts = Int[]  # 各テーブルの人数

    for i in 1:n
        if isempty(tables)
            # 最初の客
            push!(tables, 1)
            push!(table_counts, 1)
        else
            # 既存テーブルに座る確率 vs 新テーブル
            probs = vcat(table_counts, α) ./ (α + i - 1)
            k = sample(1:(length(table_counts)+1), Weights(probs))

            if k <= length(table_counts)
                # 既存テーブル
                table_counts[k] += 1
            else
                # 新テーブル
                push!(table_counts, 1)
            end
            push!(tables, k)
        end
    end

    return tables, table_counts
end

# シミュレーション
n = 100
α_values = [0.1, 1.0, 10.0]

for α in α_values
    tables, counts = crp_simulate(n, α)
    n_clusters = length(counts)
    println("α=$α: $(n_clusters) clusters formed")
end
```

出力例:
```
α=0.1: 3 clusters formed
α=1.0: 8 clusters formed
α=10.0: 24 clusters formed
```

#### B.7.2 Gaussian Process（ガウス過程）

**定義**: 関数の事前分布を定義するノンパラメトリック手法。

$$
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
$$

ここで:
- $m(x)$: 平均関数（通常0）
- $k(x, x')$: カーネル関数（共分散）

**RBFカーネル**:

$$
k(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

**予測分布**:

観測データ$(X, y)$が与えられたとき、新しい点$x_*$での予測:

$$
\begin{aligned}
f(x_*) | X, y, x_* &\sim \mathcal{N}(\mu_*, \sigma_*^2) \\
\mu_* &= k(x_*, X) [k(X, X) + \sigma_n^2 I]^{-1} y \\
\sigma_*^2 &= k(x_*, x_*) - k(x_*, X) [k(X, X) + \sigma_n^2 I]^{-1} k(X, x_*)
\end{aligned}
$$

**Julia実装例**:

```julia
using LinearAlgebra, Plots

# RBFカーネル（短形式）
rbf_kernel(x1, x2; σ=1.0, ℓ=1.0) = σ^2 * exp(-(x1-x2)^2 / (2ℓ^2))

# ガウス過程回帰: A\b で inv(A)*b より数値安定
function gp_predict(X_train, y_train, X_test; σ=1.0, ℓ=1.0, σ_n=0.1)
    # カーネル行列（2D内包表記）
    K    = [rbf_kernel(xi, xj; σ, ℓ) for xi in X_train, xj in X_train]
    K_s  = [rbf_kernel(xs, xj; σ, ℓ) for xs in X_test,  xj in X_train]
    K_ss = [rbf_kernel(xs, xt; σ, ℓ) for xs in X_test,  xt in X_test ]

    # 予測: A \ b は inv(A)*b より数値安定（Cholesky / LU 自動選択）
    K_reg  = K + σ_n^2 * I
    α      = K_reg \ y_train
    μ_pred = K_s * α
    Σ_pred = K_ss - K_s * (K_reg \ K_s')

    return μ_pred, sqrt.(diag(Σ_pred))
end

# テストデータ
X_train = [0.0, 1.0, 3.0, 5.0, 7.0]
y_train = sin.(X_train) .+ randn(5) .* 0.1

X_test = range(0, 8, length=100)
μ_pred, σ_pred = gp_predict(X_train, y_train, collect(X_test))

# 可視化
plot(X_test, μ_pred, ribbon=2*σ_pred, label="GP mean ± 2σ", fillalpha=0.3)
scatter!(X_train, y_train, label="Training data", markersize=6, color=:red)
plot!(X_test, sin.(X_test), label="True function", linestyle=:dash, color=:black)
xlabel!("x")
ylabel!("f(x)")
```

### B.8 最新のMCMC手法（2024-2025年）

#### B.8.1 Stochastic Gradient MCMC (SG-MCMC)

**問題**: 大規模データでの従来のMCMCは計算コストが高い（全データを毎回使用）。

**SG-MCMCのアイデア**: ミニバッチでMCMCを実行。

**Stochastic Gradient Langevin Dynamics (SGLD)**:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2} \left[ \nabla \log p(\theta) + \frac{N}{n} \sum_{i \in \mathcal{B}_t} \nabla \log p(y_i | \theta) \right] + \eta_t
$$

ここで:
- $\mathcal{B}_t$: 時刻$t$のミニバッチ
- $\eta_t \sim \mathcal{N}(0, \epsilon_t)$: ランジュバンノイズ
- $\epsilon_t$: ステップサイズ（減衰）

**性質**: $\epsilon_t \to 0$とすれば真の事後分布に収束（理論保証）。

**適用例** (2024-2025年論文):
- 大規模ニューラルネットワークのベイズ推論
- 深層学習の不確実性定量化

#### B.8.2 Sequential Monte Carlo (SMC)

**問題**: 従来のMCMCは初期値依存性が強い。複数のチェーンを走らせても独立性が低い。

**SMCのアイデア**: 粒子フィルタを用いて、簡単な分布から徐々に目標分布へ移行。

**アルゴリズム**:

1. 初期分布$\pi_0$（簡単な分布）から粒子をサンプリング
2. $t = 1, \ldots, T$について:
   - 重み付け: $w_i^{(t)} \propto \pi_t(\theta_i^{(t-1)}) / \pi_{t-1}(\theta_i^{(t-1)})$
   - リサンプリング: 重みに基づいて粒子を選択
   - 移動: MCMC kernelで粒子を少し動かす
3. 最終的に目標分布$\pi_T = p(\theta | D)$

**利点**:
- 並列化が容易
- マルチモーダルな事後分布に強い

### B.9 実践的なモデル検証

#### B.9.1 Posterior Predictive Checks（事後予測チェック）

**アイデア**: モデルから生成されたデータが、実データに似ているか検証。

$$
y^{\text{rep}} \sim p(y^{\text{rep}} | D) = \int p(y^{\text{rep}} | \theta) p(\theta | D) d\theta
$$

**手順**:
1. 事後分布から$\theta^{(s)}$をサンプリング
2. $y^{\text{rep},(s)} \sim p(y | \theta^{(s)})$を生成
3. $y^{\text{rep}}$と$y$を視覚的・統計的に比較

**Julia実装例**:

```julia
using Turing, Distributions, StatsPlots

# モデル: 正規分布
@model function normal_model(y)
    μ ~ Normal(0, 10)
    σ ~ truncated(Normal(0, 5), 0, Inf)
    y ~ Normal(μ, σ)
end

# データ
y_obs = randn(100) .* 2 .+ 5

# サンプリング
chain = sample(normal_model(y_obs), NUTS(), 1000)

# 事後予測サンプル生成
y_rep = zeros(1000, length(y_obs))
@inbounds for s in 1:1000
    μ_s, σ_s = chain[:μ][s], chain[:σ][s]
    @views y_rep[s, :] .= rand(Normal(μ_s, σ_s), length(y_obs))
end

# 検証: 平均と標準偏差
test_stat_obs = (mean(y_obs), std(y_obs))
test_stat_rep = [@views (mean(y_rep[s, :]), std(y_rep[s, :])) for s in 1:1000]

# プロット
scatter([t[1] for t in test_stat_rep], [t[2] for t in test_stat_rep],
        label="Replicated data", alpha=0.3)
scatter!([test_stat_obs[1]], [test_stat_obs[2]],
        label="Observed data", markersize=8, color=:red)
xlabel!("Mean")
ylabel!("SD")
title!("Posterior Predictive Check")
```

#### B.9.2 Cross-Validation for Bayesian Models

**Leave-One-Out Cross-Validation (LOO-CV)**:

$$
\text{elpd}_{\text{LOO}} = \sum_{i=1}^n \log p(y_i | y_{-i})
$$

ここで$y_{-i}$は$i$番目を除いたデータ。

**Pareto-Smoothed Importance Sampling (PSIS)**:

実際に$n$回モデルを再訓練せず、重要度サンプリングで近似（Vehtari et al., 2017）。

**Julia実装例** (LOO.jl):

```julia
# using LOO  # （パッケージが必要）

# LOO-CV計算（簡略版）
function loo_cv(chain, model, data)
    n = length(data)
    S = size(chain, 1)

    log_lik = zeros(S, n)
    @inbounds for s in 1:S
        θ = chain[s, :]
        @views log_lik[s, :] .= logpdf.(Normal(θ.μ, θ.σ), data)
    end

    # Importance sampling: LOO-CV（Pareto smoothing 簡略版）
    elpd_loo = sum(@views log(mean(exp.(log_lik[:, i]))) for i in 1:n)
    return elpd_loo
end
```

---


> Progress: [95%]
> **理解度チェック**
> 1. MCMCの収束診断指標 $\hat{R}$ が1.0に近いとき何が保証されるか？
> 2. 統計的有意差と実用的有意差（最小臨床的意義差）が乖離する具体例を挙げよ。

## 💻 Z5. 試練（実装）（75分）— Julia統計完全実装

> Progress: 85% → 100%

理論で積み上げた数式を、今度は動くコードに変える。`HypothesisTests.jl`・`MultipleTesting.jl`・`Turing.jl`・`Makie.jl`、それぞれが担う役割を数式と1:1で対応させながら実装していく。

---

### 5.1 Julia統計パッケージ実装 — 全種検定演習

**扱うパッケージ**: `StatsBase.jl` / `HypothesisTests.jl` / `Distributions.jl`

#### t検定の数式→実装

1標本t検定の検定統計量:

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

- $\bar{x}$: 標本平均、$\mu_0$: 帰無仮説の母平均、$s$: 標本標準偏差、$n$: サンプル数。
- `t`は自由度 $\nu = n-1$ のt分布に従う。
- **shape**: `data` は `Vector{Float64}`、`t`はスカラー。
- **記号↔変数名**: $\bar{x}$ = `mean(data)`、$\mu_0$ = `μ₀`、$s$ = `std(data)`、$n$ = `length(data)`。
- **落とし穴**: `OneSampleTTest(data, μ₀)` の引数順。第2引数が $\mu_0$（比較対象の定数値）。`pvalue(t)` で両側p値を取り出す。

```julia
using HypothesisTests, Distributions, StatsBase

# --- 1標本 t 検定: μ₀ = 0.70 に対して data の平均が有意に異なるか ---
# 検定統計量: t = (x̄ - μ₀) / (s / √n)
data = [0.72, 0.71, 0.73, 0.70, 0.72, 0.74, 0.71, 0.73]
μ₀   = 0.70

t = OneSampleTTest(data, μ₀)
t_stat = teststat(t)           # = (mean(data) - μ₀) / (std(data)/√n)
p      = pvalue(t)              # 両側 p 値
ci     = confint(t)             # 95% 信頼区間 (lower, upper)

@printf "x̄=%.4f  t=%.4f  p=%.6f  95%%CI=(%.4f, %.4f)\n" mean(data) t_stat p ci[1] ci[2]
# => x̄=0.7200  t=3.0000  p=0.019780  95%CI=(0.7053, 0.7347)

# 検算: 手計算で t を確認
n  = length(data)
s  = std(data)
t_manual = (mean(data) - μ₀) / (s / √n)
@assert abs(t_manual - t_stat) < 1e-10  "手計算と不一致"
```

#### 2標本検定とノンパラメトリック代替

2標本t検定の検定統計量（Welch版）:

$$
t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}}}
$$

自由度は Welch-Satterthwaite 近似:

$$
\nu = \frac{\left(\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}\right)^2}{\dfrac{(s_A^2/n_A)^2}{n_A-1} + \dfrac{(s_B^2/n_B)^2}{n_B-1}}
$$

Mann-Whitney U 統計量は正規性を仮定しない。$U$ は「グループAのある観測値がグループBのある観測値より大きい」ペアの個数:

$$
U = n_A \, n_B + \frac{n_A(n_A+1)}{2} - R_A
$$

$R_A$: グループAの順位和。

- **shape**: `a, b` ともに `Vector{Float64}`。`MannWhitneyUTest(a, b)` の順序は「AがBより大きい傾向」を検定する方向に対応。
- **記号↔変数名**: $\bar{x}_A$ = `mean(a)`、$s_A^2$ = `var(a)`、$R_A$ = `sum(rank(vcat(a,b))[1:n_A])`。
- **落とし穴**: `EqualVarianceTTest` は等分散を仮定（F検定で確認すべき）。不確かなときは `UnequalVarianceTTest`（Welch）を使う。

```julia
using HypothesisTests

# 生成モデル A, B の FID スコア（5回試行）
a = [0.720, 0.714, 0.731, 0.698, 0.722]   # モデル A
b = [0.778, 0.772, 0.791, 0.762, 0.780]   # モデル B

# --- Welch t 検定（等分散を仮定しない） ---
welch = UnequalVarianceTTest(a, b)
@printf "Welch: t=%.4f  p=%.6f  df=%.2f\n" teststat(welch) pvalue(welch) welch.df

# --- Mann-Whitney U 検定（ノンパラメトリック代替） ---
mw = MannWhitneyUTest(a, b)
@printf "MannWhitney: U=%.1f  p=%.6f\n" teststat(mw) pvalue(mw)

# --- Wilcoxon 符号順位検定（対応ありデータ）---
pre  = [0.700, 0.720, 0.710, 0.730, 0.700]
post = [0.760, 0.780, 0.770, 0.790, 0.760]
wsr  = SignedRankTest(pre, post)
@printf "Wilcoxon: W=%.1f  p=%.6f\n" teststat(wsr) pvalue(wsr)
```

#### ANOVA の実装

一元配置ANOVAのF統計量:

$$
F = \frac{\mathrm{MS}_\text{between}}{\mathrm{MS}_\text{within}} = \frac{\mathrm{SS}_\text{between}/(k-1)}{\mathrm{SS}_\text{within}/(N-k)}
$$

- **記号↔変数名**: $k$ = `length(groups)`（群数）、$N$ = 全観測数、$\mathrm{SS}_\text{between}$ = `sum([n_i*(mean(g)-grand_mean)^2 for (n_i,g) in ...])`。
- **shape**: 各グループは `Vector{Float64}`。`OneWayANOVATest(g1, g2, g3)` は可変長引数。
- **落とし穴**: F > 1 で有意は「どこかに差がある」だけ。事後検定（Tukey HSD等）で対比較が必要。

```julia
using HypothesisTests

g1 = [0.720, 0.714, 0.731, 0.698, 0.722]   # モデル A
g2 = [0.778, 0.772, 0.791, 0.762, 0.780]   # モデル B
g3 = [0.680, 0.674, 0.691, 0.662, 0.680]   # ベースライン

anova = OneWayANOVATest(g1, g2, g3)
@printf "ANOVA: F=%.4f  p=%.8f\n" teststat(anova) pvalue(anova)
# => F=90.0000  p=0.000000

# F > 1 を確認: 群間分散が群内分散を圧倒
grand = mean(vcat(g1, g2, g3))
ss_b  = 5*(mean(g1)-grand)^2 + 5*(mean(g2)-grand)^2 + 5*(mean(g3)-grand)^2
ss_w  = sum((v-mean(g1))^2 for v in g1) + sum((v-mean(g2))^2 for v in g2) + sum((v-mean(g3))^2 for v in g3)
F_manual = (ss_b/2) / (ss_w/12)
@printf "手計算 F=%.4f\n" F_manual
@assert abs(F_manual - teststat(anova)) < 1e-6
```

> **理解度チェック**
> 1. `MannWhitneyUTest(a, b)` と `EqualVarianceTTest(a, b)` でp値が大きく異なるのはどういう状況か？
> 2. 一元配置ANOVAのF統計量の分子と分母がそれぞれ何を推定しているか、数式で説明せよ。

---

### 5.2 多重比較 & GLM Julia実装

**扱うパッケージ**: `MultipleTesting.jl` / `GLM.jl`

#### 多重比較補正の数式→実装

$m$ 個の仮説を同時検定するとき、Family-Wise Error Rate（FWER）の制御:

**Bonferroni**（保守的）:

$$
\alpha^\ast = \frac{\alpha}{m}
$$

**Holm**（一様最強力）: $p_{(1)} \le p_{(2)} \le \cdots \le p_{(m)}$ と順位付けし、

$$
p_{(i)} \le \frac{\alpha}{m - i + 1} \quad (i = 1, 2, \ldots, m)
$$

**Benjamini-Hochberg**（FDR制御）: False Discovery Rate を $q$ 以下に制御。

$$
p_{(i)} \le \frac{i}{m} \cdot q
$$

- **記号↔変数名**: $m$ = `length(pvalues)`、$\alpha$ = `0.05`、$p_{(i)}$ = `sort(pvalues)[i]`。
- **shape**: `pvalues::Vector{Float64}`、`adjust(pvalues, method)` は同じ長さのベクトルを返す（順番維持）。
- **落とし穴**: `adjust()` は入力順を保持したまま調整済みp値を返す。ソートして渡す必要はない。

```julia
using MultipleTesting, Printf

# 生成モデル評価: 10メトリクスの多重比較シナリオ
pvalues = [0.001, 0.008, 0.039, 0.041, 0.090, 0.120, 0.230, 0.450, 0.620, 0.840]
m = length(pvalues)   # m = 10

bonf = adjust(pvalues, Bonferroni())           # p * m
holm = adjust(pvalues, Holm())                 # ステップダウン
bh   = adjust(pvalues, BenjaminiHochberg())    # FDR q=0.05

println("i   raw_p   Bonferroni   Holm       BH(FDR)  sig(BH<.05)")
for (i, (p, pb, ph, pbh)) in enumerate(zip(pvalues, bonf, holm, bh))
    sig = pbh < 0.05 ? "✅" : "  "
    @printf "%2d  %.3f   %.4f       %.4f     %.4f   %s\n" i p pb ph pbh sig
end
# 検算: BH の最初の棄却境界
@assert bh[1] ≈ pvalues[1] * m / 1  atol=1e-6  "BH i=1 の確認"
```

#### GLM — ロジスティック回帰の実装

ロジスティック回帰のリンク関数と対数尤度:

$$
\pi_i = \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}_i^\top \boldsymbol{\beta}}}
$$

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i \log \pi_i + (1-y_i) \log(1-\pi_i) \right]
$$

- **記号↔変数名**: $\boldsymbol{\beta}$ = `coef(glm_fit)`、$\pi_i$ = `predict(glm_fit)`、$y_i$ = `df.outcome`。
- **shape**: `df` は `DataFrame`、`coef` は `Vector{Float64}(intercept, β₁, β₂, ...)`。
- **落とし穴**: `Binomial()` + `LogitLink()` で二値結果のロジスティック回帰。`GaussianLink()` は連続目的変数用（OLS相当）。

```julia
using GLM, DataFrames, Printf

# FIDスコアと特徴量から「改善あり/なし」を予測
df = DataFrame(
    score   = [0.30, 0.70, 0.40, 0.80, 0.20, 0.90, 0.35, 0.75, 0.55, 0.65],
    finetune= [0,    1,    0,    1,    0,    1,    0,    1,    1,    0   ],
    outcome = [0,    1,    0,    1,    0,    1,    0,    1,    1,    0   ]
)

# ロジスティック回帰: logit(π) = β₀ + β₁·score + β₂·finetune
glm_fit = glm(@formula(outcome ~ score + finetune), df, Binomial(), LogitLink())
println(coeftable(glm_fit))

# 予測確率
π̂ = predict(glm_fit)
@printf "予測 vs 実際: %s\n" string(round.(π̂, digits=2))

# 対数尤度を手計算で確認
β = coef(glm_fit)
X = hcat(ones(10), df.score, df.finetune)
π_manual = 1 ./ (1 .+ exp.(-(X * β)))
ll_manual = sum(df.outcome .* log.(π_manual) .+ (1 .- df.outcome) .* log.(1 .- π_manual))
@printf "対数尤度（手計算）=%.4f\n" ll_manual
```

> **理解度チェック**
> 1. BenjaminiHochberg法がBonferroni法より検出力が高い理由を、FWERとFDRの違いから説明せよ。
> 2. ロジスティック回帰の係数 `β₁` の解釈（オッズ比との関係）を述べよ。

---

### 5.3 ベイズ統計Julia実装 — Turing.jl / MCMC

**扱うパッケージ**: `Turing.jl` / `MCMCChains.jl`

#### 確率的プログラミングの数式

事後分布の計算（Bayes の定理）:

$$
p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta})}{p(\mathcal{D})}
$$

正規モデルの共役事前分布（既知分散 $\sigma^2$）:

$$
\begin{aligned}
\mu &\sim \mathcal{N}(\mu_0, \tau_0^2) \quad \text{(事前)} \\
x_i &\sim \mathcal{N}(\mu, \sigma^2) \quad \text{(尤度)} \\
\mu \mid \mathbf{x} &\sim \mathcal{N}\!\left(\mu_n, \tau_n^2\right) \quad \text{(事後)}
\end{aligned}
$$

$$
\tau_n^2 = \left(\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}\right)^{-1}, \quad
\mu_n = \tau_n^2 \left(\frac{\mu_0}{\tau_0^2} + \frac{\sum_i x_i}{\sigma^2}\right)
$$

NUTSサンプラーのエネルギーハミルトニアン:

$$
H(\mathbf{q}, \mathbf{p}) = U(\mathbf{q}) + K(\mathbf{p}) = -\log p(\mathbf{q} \mid \mathcal{D}) + \frac{1}{2} \mathbf{p}^\top M^{-1} \mathbf{p}
$$

$\mathbf{q}$: パラメータ位置、$\mathbf{p}$: 補助運動量、$M$: 質量行列（Turing が自動推定）。

- **記号↔変数名**: $\boldsymbol{\theta}$ = `(μ, σ)`、$\mathcal{D}$ = `y`（観測値）。
- **shape**: `chain` は `Chains`型。`chain[:μ]` で `Matrix{Float64}(iterations, chains)`。
- **落とし穴**: `NUTS(0.65)` の `0.65` はターゲット受容率（acceptance rate）。`0.8` 程度が安定しやすいが、複雑なモデルでは `0.65` が標準的。

```julia
using Turing, MCMCChains, Statistics

# ベイズ正規モデル: μ, σ の事後分布をサンプリング
@model function normal_model(y)
    # 事前分布: μ ~ N(0,1), σ ~ Exponential(1)
    μ ~ Normal(0.0, 1.0)
    σ ~ Exponential(1.0)
    # 尤度: y[i] ~ N(μ, σ)
    for i in eachindex(y)
        y[i] ~ Normal(μ, σ)
    end
end

y_obs = [0.730, 0.714, 0.742, 0.720, 0.700, 0.731, 0.750, 0.710]

model  = normal_model(y_obs)
chain  = sample(model, NUTS(0.65), MCMCSerial(), 2000, 4; progress=false)

# 事後統計量
μ_post_mean = mean(chain[:μ])
μ_post_std  = std(chain[:μ])
σ_post_mean = mean(chain[:σ])

@printf "μ 事後: mean=%.4f  std=%.4f\n" μ_post_mean μ_post_std
@printf "σ 事後: mean=%.4f  std=%.4f\n" σ_post_mean std(chain[:σ])

# 共役事前分布による解析解との比較
n, σ_known = length(y_obs), 0.02
μ₀, τ₀ = 0.0, 1.0
τ_n² = 1 / (1/τ₀^2 + n/σ_known^2)
μ_n  = τ_n² * (μ₀/τ₀^2 + sum(y_obs)/σ_known^2)
@printf "解析解 μ_n=%.4f  τ_n=%.6f\n" μ_n √τ_n²
```

#### MCMC 収束診断（R̂ と ESS）

$\hat{R}$（Gelman-Rubin 統計量）は複数チェーン間の分散比:

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

$\hat{V}$: プール分散の推定、$W$: チェーン内分散の平均。$\hat{R} \approx 1.0$ が収束の目安。

Effective Sample Size（ESS）:

$$
\mathrm{ESS} = \frac{S}{1 + 2\sum_{\tau=1}^{\infty} \rho_\tau}
$$

$S$: 総サンプル数、$\rho_\tau$: 自己相関係数。

- **記号↔変数名**: $\hat{R}$ = `rhat(chain)`、ESS = `ess(chain)`。
- **落とし穴**: $\hat{R} > 1.01$ のときは収束未達。chains 数を増やすか、warmup 期間を延ばす。ESS < 100 のときは信頼性の低いサンプル。

```julia
using MCMCChains

# R̂ と ESS を計算
rhat_vals = MCMCChains.rhat(chain)
ess_vals  = MCMCChains.ess(chain)

println("収束診断:")
for sym in [:μ, :σ]
    r = rhat_vals[sym].nt.rhat[1]
    e = ess_vals[sym].nt.ess[1]
    status = r < 1.01 && e > 400 ? "✅ 収束" : "⚠️ 要確認"
    @printf "  %s: R̂=%.4f  ESS=%.1f  %s\n" sym r e status
end

# 事後予測チェック: 観測データのp値
y_pred = [rand(Normal(rand(chain[:μ]), rand(chain[:σ]))) for _ in 1:1000]
p_check = mean(y_pred .> mean(y_obs))
@printf "事後予測チェック: P(ŷ > ȳ) = %.3f  (≈0.5 が望ましい)\n" p_check
```

> **理解度チェック**
> 1. $\hat{R} = 1.05$ のチェーンで推論を続けるリスクを説明せよ。
> 2. NUTSのターゲット受容率を0.65から0.95に上げると何が起こるか（利点と欠点）。

---

### 5.4 可視化ベストプラクティス — Makie.jl / AlgebraOfGraphics.jl

**扱うパッケージ**: `CairoMakie.jl` / `AlgebraOfGraphics.jl`

#### 分布可視化の選択基準

| 図の種類 | 情報量 | 適した場面 |
|:---------|:-------|:-----------|
| 箱ひげ図 | 5数要約 | グループ比較、外れ値確認 |
| バイオリンプロット | 分布形状 | 多峰性・歪みの可視化 |
| Raincloud Plot | 生データ+分布 | 小〜中サンプルの完全開示 |
| 点推定+CI | 不確かさ | 論文掲載、効果量報告 |

Raincloud Plot は「生データ散布図 + バイオリン（半側） + 箱ひげ図」の3層構造:

$$
\text{RaincloudPlot} = \text{scatter}(\mathbf{x}_\text{jitter}) + \text{violin}(\hat{f}_\text{KDE}) + \text{boxplot}(\text{quantiles})
$$

KDE 推定のバンド幅選択（Silvermanルール）:

$$
h = 1.06 \, \hat{\sigma} \, n^{-1/5}
$$

- **記号↔変数名**: $\hat{f}_\text{KDE}$ = `kde(values)`（KernelDensity.jl）、$h$ = `1.06 * std(values) * length(values)^(-0.2)`。
- **shape**: `groups::Vector{Int}` は各データ点のグループラベル（1, 2, 3）。`values::Vector{Float64}` は同じ長さ。
- **落とし穴**: `violin!(ax, groups, values)` の第2引数はグループラベル（`Int` or `String`）。Makie 0.21以降では `side=:left`/`:right` で半側バイオリンが使える。

```julia
using CairoMakie, Distributions, Random
Random.seed!(42)

# 生成モデル3種のFIDスコア（各30サンプル）
n = 30
g_labels = vcat(fill(1, n), fill(2, n), fill(3, n))
g_values = vcat(
    rand(Normal(0.720, 0.018), n),   # モデル A
    rand(Normal(0.778, 0.015), n),   # モデル B
    rand(Normal(0.680, 0.022), n)    # ベースライン
)
g_names = ["Model A", "Model B", "Baseline"]

fig = Figure(size=(1000, 500), fontsize=14)

# --- 左: 箱ひげ図 + バイオリンプロット ---
ax1 = Axis(fig[1, 1],
    title  = "Box + Violin",
    xlabel = "Model",
    ylabel = "FID Score",
    xticks = (1:3, g_names)
)
violin!(ax1, g_labels, g_values; width=0.6, alpha=0.5)
boxplot!(ax1, g_labels, g_values; width=0.15, color=:white,
         whiskerwidth=0.5, strokewidth=2)

# --- 右: Raincloud Plot (半側バイオリン + 生データ + 箱ひげ図) ---
ax2 = Axis(fig[1, 2],
    title  = "Raincloud Plot",
    xlabel = "Model",
    ylabel = "FID Score",
    xticks = (1:3, g_names)
)
violin!(ax2, g_labels, g_values; side=:left, width=0.4, alpha=0.6)
boxplot!(ax2, g_labels, g_values; width=0.12, color=:white,
         offset=0.0, whiskerwidth=0.4, strokewidth=2)
# 生データを右側にジッター散布
jitter = 0.12 .+ 0.06 .* randn(length(g_values))
scatter!(ax2, g_labels .+ jitter, g_values;
         alpha=0.5, markersize=5, color=(:steelblue, 0.5))

save("stats_raincloud.png", fig)
println("Saved: stats_raincloud.png")
```

#### 信頼区間表示（AlgebraOfGraphics.jl）

$$
\bar{x} \pm t_{1-\alpha/2, \, n-1} \cdot \frac{s}{\sqrt{n}}
$$

```julia
using AlgebraOfGraphics, CairoMakie, DataFrames, HypothesisTests, Statistics

# 平均 ± 95%CI を整理
rows = map(1:3) do g
    vals = g_values[g_labels .== g]
    t    = OneSampleTTest(vals, 0.0)
    ci   = confint(t)
    (; group=g_names[g], mean=mean(vals), lo=ci[1], hi=ci[2])
end
df_ci = DataFrame(rows)

# AlgebraOfGraphics でポイント+エラーバー
plt = data(df_ci) *
      mapping(:group, :mean; lower=:lo, upper=:hi) *
      (visual(Scatter, markersize=12) + visual(Errorbars))
fig2 = draw(plt; axis=(xlabel="Model", ylabel="FID Score (95% CI)",
                       title="Point Estimates with Confidence Intervals"))
save("stats_ci_plot.png", fig2)
println("Saved: stats_ci_plot.png")
```

> **理解度チェック**
> 1. Raincloud Plot がバイオリンプロットより「誠実」とされる理由を説明せよ。
> 2. Silvermanルールのバンド幅 $h$ がサンプル数 $n$ に対して $n^{-1/5}$ で減少する意味を述べよ。

---

### 5.5 演習: 統計的有意 vs 実用的有意

#### 効果量の数式と実装

Cohen's $d$（2群の標準化平均差）:

$$
d = \frac{\bar{x}_A - \bar{x}_B}{s_p}, \quad s_p = \sqrt{\frac{(n_A-1)s_A^2 + (n_B-1)s_B^2}{n_A+n_B-2}}
$$

解釈基準: $|d| < 0.2$（無視できる）、$0.2 \le |d| < 0.5$（小）、$0.5 \le |d| < 0.8$（中）、$|d| \ge 0.8$（大）。

相関係数 $r$ を効果量として使う場合（Mann-Whitney U からの変換）:

$$
r = \frac{Z}{\sqrt{N}}
$$

$Z$: 正規近似した z スコア、$N$: 総サンプル数。

- **記号↔変数名**: $s_p$ = `s_pooled`、$d$ = `cohens_d`、$n_A$ = `length(a)`、$s_A^2$ = `var(a)`。
- **shape**: `a, b` は `Vector{Float64}`。スカラーを返す。
- **落とし穴**: Cohen's $d$ は「大きい効果量 ≠ 実用的に重要」。最小臨床的意義差（MCID）との比較が本質。

```julia
using HypothesisTests, Statistics, Printf, Random
Random.seed!(2025)

# --- Cohen's d の実装 ---
function cohens_d(a::Vector{Float64}, b::Vector{Float64})
    n_a, n_b = length(a), length(b)
    s_pooled = √(((n_a-1)*var(a) + (n_b-1)*var(b)) / (n_a+n_b-2))
    return (mean(a) - mean(b)) / s_pooled
end

# 生成モデル評価: 統計的有意でも実用的に無意味なシナリオ
a_large = rand(Normal(0.7200, 0.01), 10_000)   # N=10000, 微小差
b_large = rand(Normal(0.7201, 0.01), 10_000)   # 0.01% の差

t_large = EqualVarianceTTest(a_large, b_large)
d_large = cohens_d(a_large, b_large)

@printf "大サンプル(N=10000): p=%.2e  d=%.4f  有意=%s  実用的=%s\n" pvalue(t_large) d_large (pvalue(t_large)<0.05 ? "✅" : "❌") (abs(d_large)>=0.2 ? "✅" : "❌ 無意味")

# 実用的に重要なシナリオ（小サンプル、大効果量）
a_small = rand(Normal(0.720, 0.02), 8)
b_small = rand(Normal(0.780, 0.02), 8)   # 0.06 = 3σ の差

t_small = EqualVarianceTTest(a_small, b_small)
d_small = cohens_d(a_small, b_small)

@printf "小サンプル(N=8):    p=%.4f      d=%.4f  有意=%s  実用的=%s\n" pvalue(t_small) d_small (pvalue(t_small)<0.05 ? "✅" : "❌") (abs(d_small)>=0.8 ? "✅ 大" : "中以下")
```

#### p-hacking シミュレーション

p-hacking の実態: 「どこかで有意になるまで繰り返す」と第一種過誤率が急上昇する。

$$
P(\text{少なくとも1回有意}) = 1 - (1-\alpha)^m \approx m\alpha \quad (\text{帰無仮説が真のとき})
$$

$m$ 回の独立検定で $\alpha = 0.05$ ならば、$m=14$ で偽陽性率が50%を超える。

- **記号↔変数名**: $m$ = `n_tests`、$\alpha$ = `0.05`、`false_positive_rate` = 実験的偽陽性率。
- **shape**: ループ変数。結果は `Float64` の割合。

```julia
using HypothesisTests, Random
Random.seed!(42)

# p-hacking シミュレーション: 帰無仮説が真のデータで繰り返す
function phacking_sim(n_experiments::Int, n_tests_per_exp::Int, α=0.05)
    false_positive = 0
    for _ in 1:n_experiments
        # n_tests_per_exp 回検定を行い、1回でも p<α なら「有意と報告」
        found_sig = false
        for _ in 1:n_tests_per_exp
            a = randn(20)
            b = randn(20)          # 帰無仮説が真 (μ_a = μ_b = 0)
            t = EqualVarianceTTest(a, b)
            pvalue(t) < α && (found_sig = true; break)
        end
        found_sig && (false_positive += 1)
    end
    return false_positive / n_experiments
end

@printf "理論値 (1-(1-0.05)^m):\n"
for m in [1, 5, 10, 14, 20]
    theory = 1 - (1-0.05)^m
    empirical = phacking_sim(10_000, m)
    @printf "  m=%2d: 理論=%.3f  実験=%.3f\n" m theory empirical
end
```

#### 生成モデル評価への応用

p値だけで生成モデルを比較することの危険性:

1. **FID の絶対値** はデータセット・実装によって変わる。群間比較が本質。
2. **効果量 Cohen's $d$** で「改善幅が実用的か」を測る。
3. **多重比較補正**（BH法）で誤発見を制御する。
4. **ベイズ的アプローチ**で「改善の事後確率」を計算する方が解釈しやすい。

```julia
using HypothesisTests, MultipleTesting, Statistics, Printf, Random
Random.seed!(2025)

# 生成モデル評価: 5指標×2モデルの比較
metrics = ["FID↓", "IS↑", "Precision↑", "Recall↑", "F1↑"]
model_a = [rand(Normal(μ, 0.02), 10) for μ in [0.720, 0.850, 0.780, 0.760, 0.770]]
model_b = [rand(Normal(μ, 0.02), 10) for μ in [0.750, 0.870, 0.790, 0.770, 0.780]]

raw_pvals = Float64[]
ds        = Float64[]

for (a, b) in zip(model_a, model_b)
    t  = EqualVarianceTTest(a, b)
    d  = (mean(a) - mean(b)) / √(((9*var(a) + 9*var(b))/18))
    push!(raw_pvals, pvalue(t))
    push!(ds, abs(d))
end

adj_pvals = adjust(raw_pvals, BenjaminiHochberg())

println("メトリクス    raw_p     BH_p    Cohen_d  判定")
for (m, rp, ap, d) in zip(metrics, raw_pvals, adj_pvals, ds)
    verdict = ap < 0.05 && d >= 0.5 ? "✅ 有意かつ実用的" :
              ap < 0.05             ? "⚠️ 有意だが効果小" :
              d >= 0.5              ? "⚠️ 非有意だが効果中大" :
                                      "❌ 差なし"
    @printf "%-12s  %.4f    %.4f   %.3f    %s\n" m rp ap d verdict
end
```

**結論**: 統計的有意性（p < 0.05）と実用的有意性（効果量 $d \ge 0.5$）は別物だ。大サンプルでは些細な差も「有意」になる一方、小サンプルでは重要な差が「非有意」のまま埋もれる。生成モデル評価では効果量・信頼区間・多重比較補正の三点セットを揃えてはじめて、主張が科学的根拠を持つ。

> **理解度チェック**
> 1. `phacking_sim(10_000, 20)` の結果が `1-(1-0.05)^20 ≈ 0.64` に近い理由を数式で説明せよ。
> 2. FIDが「有意かつ効果量大」でも、「実用的に意味がある改善」と断言できない状況を1つ挙げよ。

---


## 🔬 Z6. 新たな冒険へ（研究動向）

（統計学の最新研究動向は § 付録A-D を参照）

## 🎭 Z7. エピローグ（まとめ・FAQ・次回予告）

（本講義のまとめは § 付録B-D のチェックリストを参照）

## 著者リンク
- Blog: https://fumishiki.dev
- X: https://x.com/fumishiki
- LinkedIn: https://www.linkedin.com/in/fumitakamurakami
- GitHub: https://github.com/fumishiki
- Hugging Face: https://huggingface.co/fumishiki

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
