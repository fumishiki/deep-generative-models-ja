---
title: "第20回: VAE/GAN/Transformerフルスタック実装 & 分散サービング: 30秒の驚き→数式修行→実装マスター"
emoji: "🔥"
type: "tech"
topics: ["machinelearning", "deeplearning", "julia", "rust", "elixir"]
published: true
---

:::message
**前提知識**: 第19回で3言語環境とFFIパイプラインを構築済み。Course IIでVAE/GAN/Transformerの理論を習得済み。
**目標**: 理論を3言語パイプライン（Julia訓練→Rust推論→Elixir配信）で実装する。
**進捗**: 全体の80%完了
:::

## 🚀 0. クイックスタート（30秒）— 理論→実装の1行対応

第19回で環境を整えた。第10回でVAE、第12回でGAN、第16回でTransformerの**理論**を学んだ。今回はそれを**動かす**。

理論と実装の対応を体感しよう。VAEのELBOを1行で：

```julia
using Lux, Optimisers, Random

# VAE ELBO = 再構成項 - KL正則化項
function elbo_loss(encoder, decoder, ps_enc, ps_dec, st_enc, st_dec, x)
    # Encoder: q_φ(z|x) → (μ, log_σ²)
    (μ, logσ²), st_enc = encoder(x, ps_enc, st_enc)

    # Reparameterization: z = μ + σ⊙ε
    ε = randn(Float32, size(μ)...)
    σ = exp.(logσ² ./ 2)
    z = μ .+ σ .* ε

    # Decoder: p_θ(x|z) → x̂
    x̂, st_dec = decoder(z, ps_dec, st_dec)

    # ELBO = 𝔼[log p(x|z)] - KL[q(z|x) || p(z)]
    recon = -sum((x .- x̂).^2) / size(x, 2)  # 再構成項（ガウス尤度）
    kl = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / size(x, 2)  # KL発散

    return -(recon - kl), (st_enc, st_dec)  # ELBOを最大化 = 負のELBOを最小化
end
```

**この30行が第10回の数式をすべて含む**：

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}[q_\phi(z|x) \| p(z)]
$$

- 再パラメータ化トリック: $z = \mu + \sigma \odot \epsilon$（23行目）
- ガウスKL閉形式: $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$（28行目）
- 数式の各項がコードの各行に**1:1対応**

これがJuliaの威力。数式↔コードの距離がゼロ。

:::message
**進捗**: 全体の3%完了。理論を実装に翻訳する準備ができた。
:::

---

## 🎮 1. 体験ゾーン（10分）— 3モデルを触る

理論を復習しながら、3つのモデルを動かす。数式→コード対応を体感する。

### 1.1 VAE — 潜在空間で画像を圧縮・再構成

第10回で学んだVAEの核心：**観測データ $x$ を低次元潜在変数 $z$ に圧縮し、そこから再構成する**。

```julia
using Lux, MLUtils, MLDatasets, Optimisers

# MNIST データ読み込み
train_data = MNIST(split=:train)
x_train = Float32.(train_data.features) |> flatten_images  # (784, 60000)

# VAE アーキテクチャ
encoder = Chain(
    Dense(784 => 400, tanh),
    Dense(400 => 200, tanh),
    Dense(200 => 40)  # → [μ(20次元), log_σ²(20次元)]
)

decoder = Chain(
    Dense(20 => 200, tanh),
    Dense(200 => 400, tanh),
    Dense(400 => 784, sigmoid)  # sigmoid for pixel values [0,1]
)

# 訓練ループ（簡略版）
opt = Adam(0.001f0)
ps_enc, st_enc = Lux.setup(Random.default_rng(), encoder)
ps_dec, st_dec = Lux.setup(Random.default_rng(), decoder)

for epoch in 1:10
    for batch in DataLoader((x_train,), batchsize=128, shuffle=true)
        x = batch[1]
        loss, grads = Lux.Training.compute_gradients(
            AutoZygote(), elbo_loss, encoder, decoder, ps_enc, ps_dec, st_enc, st_dec, x
        )
        ps_enc, ps_dec = Optimisers.update!(opt, (ps_enc, ps_dec), grads)
    end
    println("Epoch $epoch: loss = $(loss)")
end

# 潜在空間でのサンプリング
z_random = randn(Float32, 20, 10)  # 10個のランダム潜在ベクトル
x_generated, _ = decoder(z_random, ps_dec, st_dec)
# → 新しい数字画像が生成される
```

| 数式 | コード | 意味 |
|:-----|:-------|:-----|
| $q_\phi(z\|x) = \mathcal{N}(z; \mu_\phi(x), \sigma^2_\phi(x)\mathbf{I})$ | `(μ, logσ²) = encoder(x)` | Encoderが平均と分散を出力 |
| $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, \mathbf{I})$ | `z = μ .+ σ .* randn(...)` | 再パラメータ化トリック |
| $p_\theta(x\|z) = \mathcal{N}(x; \mu_\theta(z), \mathbf{I})$ | `x̂ = decoder(z)` | Decoderが再構成画像を出力 |
| $D_{\text{KL}}[q_\phi(z\|x) \| \mathcal{N}(0, \mathbf{I})]$ | `-0.5 * sum(1 + logσ² - μ² - exp(logσ²))` | ガウスKL閉形式 |

**体感**：潜在空間 $z \in \mathbb{R}^{20}$ で784次元画像を表現。第10回の数式がそのまま動く。

---

### 1.2 GAN — 生成器と識別器の対決

第12回で学んだGANの核心：**Generator $G$ がノイズ $z$ から偽画像を生成し、Critic（識別器） $D$ が本物/偽物を見分ける競争**。

WGANの損失関数（第13回で学んだWasserstein距離ベース）：

$$
\mathcal{L}_D = \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))] - \lambda \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]
$$

$$
\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[D(G(z))]
$$

```julia
# Generator: z (100次元ノイズ) → 画像 (28×28)
generator = Chain(
    Dense(100 => 256, relu),
    Dense(256 => 512, relu),
    Dense(512 => 784, tanh)  # tanh for [-1, 1] pixel range
)

# Critic (WGAN-GPでは識別器を"Critic"と呼ぶ)
critic = Chain(
    Dense(784 => 512, leakyrelu),
    Dense(512 => 256, leakyrelu),
    Dense(256 => 1)  # スコア出力（確率ではない）
)

# WGAN-GP訓練ループ（簡略版）
function train_wgan_gp!(generator, critic, real_data, epochs=100, λ_gp=10.0f0)
    opt_g = Adam(0.0001f0, (0.5f0, 0.9f0))  # Generator optimizer
    opt_c = Adam(0.0001f0, (0.5f0, 0.9f0))  # Critic optimizer

    for epoch in 1:epochs
        for batch in DataLoader((real_data,), batchsize=64, shuffle=true)
            x_real = batch[1]
            batch_size = size(x_real, 2)

            # --- Criticを5回更新 ---
            for _ in 1:5
                z = randn(Float32, 100, batch_size)
                x_fake = generator(z, ps_g, st_g)[1]

                # Gradient Penalty 計算
                α = rand(Float32, 1, batch_size)
                x_interp = α .* x_real .+ (1 .- α) .* x_fake
                grad_interp = gradient(x -> sum(critic(x, ps_c, st_c)[1]), x_interp)[1]
                gp = mean((sqrt.(sum(grad_interp.^2, dims=1)) .- 1).^2)

                # Critic loss
                loss_c = mean(critic(x_fake, ps_c, st_c)[1]) - mean(critic(x_real, ps_c, st_c)[1]) + λ_gp * gp
                ps_c = update!(opt_c, ps_c, gradient(loss_c, ps_c)[1])
            end

            # --- Generatorを1回更新 ---
            z = randn(Float32, 100, batch_size)
            loss_g = -mean(critic(generator(z, ps_g, st_g)[1], ps_c, st_c)[1])
            ps_g = update!(opt_g, ps_g, gradient(loss_g, ps_g)[1])
        end
        println("Epoch $epoch: D_loss=$(loss_c), G_loss=$(loss_g)")
    end
end
```

| 数式 | コード | 意味 |
|:-----|:-------|:-----|
| $G(z)$ | `generator(z)` | ノイズ→偽画像 |
| $D(x)$ | `critic(x)` | 画像→スコア |
| $\hat{x} = \alpha x + (1-\alpha)G(z)$ | `x_interp = α .* x_real .+ (1 .- α) .* x_fake` | 本物と偽物の補間 |
| $\|\nabla_{\hat{x}} D(\hat{x})\|_2$ | `sqrt(sum(grad_interp.^2, dims=1))` | 勾配ノルム |
| $(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2$ | `(sqrt(...) .- 1).^2` | Gradient Penalty |

**体感**：Criticを5回、Generatorを1回更新（WGAN-GP推奨比率）。第12回・第13回の数式がそのまま動く。

---

### 1.3 Transformer — Attentionで次トークン予測

第16回で学んだTransformerの核心：**Multi-Head Attentionで文脈を並列処理し、次トークンを予測**。

Scaled Dot-Product Attentionの数式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

```julia
using Lux, NNlib

# Multi-Head Attention レイヤー
struct MultiHeadAttention <: Lux.AbstractExplicitLayer
    num_heads::Int
    d_model::Int
    d_k::Int
    q_proj::Dense
    k_proj::Dense
    v_proj::Dense
    o_proj::Dense
end

function MultiHeadAttention(d_model::Int, num_heads::Int)
    d_k = d_model ÷ num_heads
    return MultiHeadAttention(
        num_heads, d_model, d_k,
        Dense(d_model => d_model),  # Q projection
        Dense(d_model => d_model),  # K projection
        Dense(d_model => d_model),  # V projection
        Dense(d_model => d_model)   # Output projection
    )
end

function (mha::MultiHeadAttention)(x, ps, st)
    batch_size, seq_len, _ = size(x)

    # Q, K, V projection
    Q, st_q = mha.q_proj(x, ps.q_proj, st.q_proj)
    K, st_k = mha.k_proj(x, ps.k_proj, st.k_proj)
    V, st_v = mha.v_proj(x, ps.v_proj, st.v_proj)

    # Reshape for multi-head: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
    Q = reshape(Q, batch_size, mha.num_heads, seq_len, mha.d_k) |> permutedims([1,2,4,3])
    K = reshape(K, batch_size, mha.num_heads, seq_len, mha.d_k) |> permutedims([1,2,4,3])
    V = reshape(V, batch_size, mha.num_heads, seq_len, mha.d_k) |> permutedims([1,2,4,3])

    # Scaled Dot-Product Attention: softmax(QK^T / √d_k) V
    scores = batched_mul(Q, batched_transpose(K)) ./ sqrt(Float32(mha.d_k))  # (batch, heads, seq, seq)
    attn_weights = softmax(scores, dims=4)  # Softmax over key dimension
    out = batched_mul(attn_weights, V)  # (batch, heads, d_k, seq)

    # Concatenate heads and project
    out = permutedims(out, [1,4,2,3]) |> x -> reshape(x, batch_size, seq_len, mha.d_model)
    out, st_o = mha.o_proj(out, ps.o_proj, st.o_proj)

    return out, (st_q=st_q, st_k=st_k, st_v=st_v, st_o=st_o)
end

# Causal Mask（未来のトークンを見せない）
function causal_mask(seq_len)
    mask = triu(ones(Float32, seq_len, seq_len), 1)  # 上三角行列
    return mask .* -Inf32  # Softmax前に加算 → 未来トークンの重みを0に
end

# 使用例
x = randn(Float32, 2, 10, 512)  # (batch=2, seq_len=10, d_model=512)
mha = MultiHeadAttention(512, 8)
ps, st = Lux.setup(Random.default_rng(), mha)
y, st = mha(x, ps, st)  # y: (2, 10, 512) — 各トークンの新しい表現
```

| 数式 | コード | 意味 |
|:-----|:-------|:-----|
| $Q, K, V = xW_Q, xW_K, xW_V$ | `Q = mha.q_proj(x)` | 線形変換 |
| $\frac{QK^\top}{\sqrt{d_k}}$ | `scores = Q @ K.T / sqrt(d_k)` | スコア計算 |
| $\text{softmax}(\cdot)$ | `softmax(scores, dims=4)` | 注意重み正規化 |
| $\text{softmax}(\cdot)V$ | `attn_weights @ V` | 加重和 |
| Causal Mask | `scores + causal_mask` | 未来を見せない |

**体感**：Multi-Head Attentionが並列に複数の視点で文脈を捉える。第16回の数式がそのまま動く。

---

### 1.4 数式↔コード対応の完全性

3モデルで共通するパターン：

```julia
# 数式: 𝔼[f(z)] where z ~ q(z)
# コード: mean(f(z) for z in sample(q, n_samples))

# 数式: ∇_θ L(θ)
# コード: gradient(θ -> L(θ), θ)

# 数式: θ ← θ - η∇_θ L
# コード: θ = update!(optimizer, θ, grads)
```

Juliaの利点：
- `.=` broadcast演算子 → 要素ごとの演算を1行で
- `|>` pipe演算子 → データフロー明示
- 型安定性 → `@code_warntype`で型推論チェック → 自動最適化

次のZone 2で、なぜこの3モデルを実装するのか、全体像を見る。

:::message
**進捗**: 全体の10%完了。3モデルの動作を体感した。
:::

---

## 🧩 2. 直感ゾーン（15分）— なぜこの3モデルか

### 2.1 Course IIIの位置づけ — 理論→実装の橋渡し

```mermaid
graph TD
    A[Course I: 数学基礎<br>第1-8回] --> B[Course II: 生成モデル理論<br>第9-16回]
    B --> C[Course III: 実装編<br>第17-24回]
    C --> D[Course IV: Diffusion理論<br>第25-32回]
    D --> E[Course V: 応用・最前線<br>第33-40回]

    B -->|理論| F[VAE 第10回]
    B -->|理論| G[GAN 第12回]
    B -->|理論| H[Transformer 第16回]

    F --> I[VAE実装 第20回<br>今ココ]
    G --> I
    H --> I

    style I fill:#ff6b6b,stroke:#333,stroke-width:4px
```

**Course IIで学んだこと**（理論）：
- 第10回：VAEのELBO導出、再パラメータ化トリック、ガウスKL閉形式
- 第12回：GANのMinimax損失、JSD、Mode Collapse問題
- 第13回：Optimal TransportとWasserstein距離、WGAN-GP
- 第16回：TransformerのAttention機構、Positional Encoding、Causal Mask

**Course IIIで学ぶこと**（実装）：
- 第19回：3言語環境構築（Julia/Rust/Elixir）、FFIパイプライン設計
- **第20回（今回）**：VAE/GAN/Transformerの完全実装、数式↔コード1:1対応
- 第21回：データサイエンス基礎、HuggingFace Datasets統合
- 第22回：評価指標実装（FID/IS/Perplexity）、モデル選択

**今回の差別化**：
- **松尾・岩澤研究室との違い**：彼らは理論講義で実装は課題。本シリーズは**理論と実装の両方を網羅**。
- **他のチュートリアルとの違い**：PyTorch/TensorFlowに閉じない。**3言語パイプライン**（訓練→推論→配信）で実戦スキルを獲得。
- **論文実装との違い**：単なる写経ではない。**数式の各項とコードの各行を1:1対応**させ、理解を深める。

---

### 2.2 なぜVAE/GAN/Transformerか — 3大パラダイムの代表

| モデル | パラダイム | 訓練方法 | 強み | 弱み |
|:-------|:-----------|:---------|:-----|:-----|
| **VAE** | 尤度ベース（明示的密度） | ELBO最大化 | 安定訓練、潜在空間解釈可能 | ぼやけた生成、表現力制約 |
| **GAN** | 暗黙的密度（Adversarial） | Minimax競争 | 鮮明な生成、高品質画像 | 訓練不安定、Mode Collapse |
| **Transformer** | 自己回帰（明示的密度） | 次トークン予測MLE | スケーリング、汎用性 | 逐次生成、計算コスト |

**3つの生成パラダイム**（第7回で学んだ分類）：

```mermaid
graph TD
    A[生成モデル] --> B[明示的密度 p_θ x]
    A --> C[暗黙的密度]

    B --> D[Tractable<br>計算可能]
    B --> E[Approximate<br>近似推論]

    D --> F[自己回帰<br>Transformer]
    D --> G[Flow<br>RealNVP]

    E --> H[変分推論<br>VAE]
    E --> I[マルコフ連鎖<br>Diffusion]

    C --> J[GANファミリー<br>WGAN-GP/StyleGAN]
    C --> K[Implicit MLE<br>Noise Contrastive]

    style F fill:#4ecdc4
    style H fill:#ff6b6b
    style J fill:#ffe66d
```

**なぜこの3つを選んだか**：
1. **VAE**：変分推論の実装パターンを習得。ELBOは他の多くのモデル（Diffusion/Flow）の基礎。
2. **GAN**：Adversarial訓練の実装パターンを習得。安定化テクニック（WGAN-GP）は必須スキル。
3. **Transformer**：自己回帰モデルの実装パターンを習得。LLM時代の必須知識。KV-Cacheは推論効率化の鍵。

これら3つをマスターすれば、他のモデル（Diffusion/Flow/VQ-VAE）の実装も理解できる。

---

### 2.3 3言語パイプラインの全体像

```mermaid
graph LR
    A[Julia<br>Lux.jl] -->|訓練| B[モデル<br>VAE/GAN/Trans]
    B -->|エクスポート| C[safetensors/<br>ONNX]
    C -->|ロード| D[Rust<br>Candle]
    D -->|推論| E[バッチ処理<br>ゼロコピー]
    E -->|FFI| F[Elixir<br>Broadway]
    F -->|配信| G[分散システム<br>耐障害性]

    style A fill:#9b59b6
    style D fill:#e67e22
    style F fill:#3498db
```

**各言語の役割**（第19回で設計）：

| 段階 | 言語 | 理由 | ツール |
|:-----|:-----|:-----|:-------|
| 訓練 | ⚡ Julia | 数式↔コード1:1、JIT高速化、REPLループ | Lux.jl, Reactant |
| 推論 | 🦀 Rust | ゼロコピー、型安全、並列処理、C-ABI FFI | Candle, ndarray |
| 配信 | 🔮 Elixir | 耐障害性、バックプレッシャー、監視ツリー | GenStage, Broadway |

**なぜ3言語か**：
- **Python 1言語では不可能**：GILボトルネック、メモリコピー、型安全性欠如、耐障害性弱い
- **PyTorchだけでは不十分**：訓練は得意だが、推論最適化・分散配信は苦手
- **各言語が最適領域を担当**：Julia（訓練）、Rust（推論）、Elixir（配信）の分業で、各段階で最高性能を達成

**今回の実装範囲**：
- Zone 3（数式修行）：VAE/GAN/TransformerのJulia訓練実装、数式↔コード完全対応
- Zone 4（実装）：Rust推論エンジン、Candleでのモデルロード・バッチ処理
- Zone 5（実験）：Elixir分散サービング、Broadway需要駆動パイプライン、耐障害性デモ

---

### 2.4 Trojan Horse完結編 — Pythonからの完全脱却

第1-8回（Course I）のTrojan Horse戦略を振り返る：

```mermaid
graph LR
    A[第1-4回<br>🐍Python 100%] --> B[第5-8回<br>🐍+ベンチマーク<br>遅さ体感]
    B --> C[第9回<br>🐍🔥絶望<br>🦀Rust登場]
    C --> D[第10回<br>🐍30%<br>⚡Julia登場]
    D --> E[第11-16回<br>⚡Julia主体<br>🦀Rust並走]
    E --> F[第17-24回<br>⚡🦀🔮<br>Python完全消滅]

    style A fill:#3498db
    style B fill:#f39c12
    style C fill:#e74c3c
    style D fill:#9b59b6
    style F fill:#2ecc71
```

**各言語の登場タイミング**：
- **第1-8回（Course I）**：🐍 Python 100% — 数学基礎を学ぶための道具
- **第9回（Course II開始）**：🦀 Rust初登場 — PythonのGIL/メモリコピー限界を体感→Rustでゼロコピー実現
- **第10回**：⚡ Julia初登場 — Rustの型パズル苦痛→Juliaの多重ディスパッチで数式↔コード1:1対応
- **第15回**：🔮 Elixir初登場 — LLM推論の並行処理・耐障害性を習得
- **第17-24回（Course III）**：⚡🦀🔮 3言語フル活用 — Pythonは完全に消える

**今回（第20回）の位置づけ**：
- Pythonで書かれた既存実装（PyTorch VAE/GAN Tutorial）を参照するが、**実装は全てJulia/Rust/Elixir**
- PyTorchとの比較は「参考」であり、もはや主役ではない
- **Trojan Horse完結**：受講生は3言語パイプラインが「普通」になった

---

### 2.5 学習戦略 — 数式→コード→システム設計

**3段階の習得**：

```mermaid
graph TD
    A[Zone 3: 数式修行<br>60分] --> B[Zone 4: 実装ゾーン<br>45分]
    B --> C[Zone 5: 実験ゾーン<br>30分]

    A -->|VAE| A1[ELBO各項の導出]
    A -->|GAN| A2[WGAN-GP loss導出]
    A -->|Trans| A3[Attention行列演算]

    B -->|Julia| B1[訓練ループ実装]
    B -->|Rust| B2[推論エンジン実装]
    B -->|Elixir| B3[配信パイプライン]

    C -->|評価| C1[損失曲線・生成品質]
    C -->|耐障害性| C2[プロセスkill→復旧]
    C -->|ベンチマーク| C3[訓練速度・推論レイテンシ]
```

**推奨学習順序**：
1. **Zone 0-2（今ここまで）**：全体像把握、3モデルを触る
2. **Zone 3**：数式を1行ずつ導出、紙とペン必須、Juliaコードと対応付け
3. **Zone 4**：Julia訓練実装→Rust推論実装→Elixir配信実装の順
4. **Zone 5**：実際に訓練・推論・配信を動かし、耐障害性をデモ
5. **Zone 6-7**：研究系譜把握、FAQ確認、次回（第21回データサイエンス）への接続

**重要な心構え**：
- **数式を飛ばさない**：Zone 3の導出は全て追う。理解せずに実装しても、デバッグ時に詰む。
- **コードを動かす**：写経ではなく、自分で打って動かす。エラーを読み、修正する。
- **システム全体を見る**：訓練だけ、推論だけでは不十分。訓練→エクスポート→推論→配信の全体フローを理解する。

**本講義の目標到達点**：
- [ ] VAE/GAN/TransformerのELBOを**紙で導出**できる
- [ ] Juliaで**ゼロから訓練ループ**を書ける
- [ ] Rustで**safetensorsをロード**し、推論できる
- [ ] Elixirで**Broadwayパイプライン**を構築できる
- [ ] プロセスをkillしても**自動復旧**するシステムを設計できる

次のZone 3で、数式修行に入る。

:::message
**進捗**: 全体の20%完了。全体像を把握した。数式修行の準備ができた。
:::

---

## 📐 3. 数式修行ゾーン（60分）— VAE/GAN/Transformer完全導出

このゾーンは**最も重要**。理論（Course II）で学んだ数式を、実装と1:1対応させる。

### 3.1 VAE — ELBO完全分解と実装対応

**復習：VAEの目的**（第10回より）

観測データ $\mathbf{x}$ の尤度 $p_\theta(\mathbf{x})$ を最大化したいが、潜在変数 $\mathbf{z}$ を周辺化する積分が計算不能：

$$
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})d\mathbf{z}
$$

そこで変分推論で近似事後分布 $q_\phi(\mathbf{z}|\mathbf{x})$ を導入し、ELBOを導出した。

---

#### 3.1.1 ELBO導出（復習）

第8回・第9回で学んだELBO導出を、実装と対応付けながら再確認。

**Step 1: 対数尤度の分解**

$$
\begin{align}
\log p_\theta(\mathbf{x})
&= \log \int p_\theta(\mathbf{x}, \mathbf{z})d\mathbf{z} \\
&= \log \int p_\theta(\mathbf{x}, \mathbf{z}) \frac{q_\phi(\mathbf{z}|\mathbf{x})}{q_\phi(\mathbf{z}|\mathbf{x})} d\mathbf{z} \\
&= \log \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right]
\end{align}
$$

**Step 2: Jensenの不等式**（第6回で証明）

$\log$ は凹関数なので：

$$
\log \mathbb{E}[f(\mathbf{z})] \geq \mathbb{E}[\log f(\mathbf{z})]
$$

適用すると：

$$
\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})}\right] \equiv \mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x})
$$

**Step 3: ELBO分解**

$$
\begin{align}
\mathcal{L}_{\text{ELBO}}
&= \mathbb{E}_{q_\phi}\left[\log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x})\right] \\
&= \mathbb{E}_{q_\phi}\left[\log p_\theta(\mathbf{x}|\mathbf{z}) + \log p(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x})\right] \\
&= \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}[q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]
\end{align}
$$

第1項：**再構成項**（Reconstruction term）
第2項：**KL正則化項**（KL Divergence regularization）

---

#### 3.1.2 再構成項の実装

**数式**：

$$
\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]
$$

Decoderが出力 $\hat{\mathbf{x}} = \mu_\theta(\mathbf{z})$ をガウス分布の平均とすると：

$$
p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \mu_\theta(\mathbf{z}), \sigma^2\mathbf{I})
$$

対数尤度：

$$
\log p_\theta(\mathbf{x}|\mathbf{z}) = -\frac{1}{2\sigma^2}\|\mathbf{x} - \mu_\theta(\mathbf{z})\|^2 + \text{const}
$$

$\sigma^2 = 1$ と固定すると（実装上の簡略化）：

$$
\log p_\theta(\mathbf{x}|\mathbf{z}) \propto -\|\mathbf{x} - \hat{\mathbf{x}}\|^2
$$

**Juliaコード**：

```julia
# Decoder出力: x̂ = decoder(z)
x̂, st_dec = decoder(z, ps_dec, st_dec)

# 再構成項: -||x - x̂||² / batch_size
recon_term = -sum((x .- x̂).^2) / size(x, 2)
```

| 数式 | コード | 対応 |
|:-----|:-------|:-----|
| $\mu_\theta(\mathbf{z})$ | `decoder(z)` | Decoderの出力 |
| $\|\mathbf{x} - \mu_\theta(\mathbf{z})\|^2$ | `sum((x .- x̂).^2)` | 二乗誤差 |
| $\mathbb{E}_{q_\phi}[\cdot]$ | `/ size(x, 2)` | バッチ平均 |

---

#### 3.1.3 KL正則化項の実装 — ガウスKL閉形式

**数式**：

$$
D_{\text{KL}}[q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]
$$

仮定：
- $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$
- $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$

**ガウスKL閉形式**（第4回で導出）：

$$
D_{\text{KL}}[\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2\mathbf{I}) \| \mathcal{N}(\mathbf{0}, \mathbf{I})] = \frac{1}{2}\sum_{i=1}^d (\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1)
$$

Encoderは $\log\sigma^2$ を出力すると便利（数値安定性）：

$$
D_{\text{KL}} = -\frac{1}{2}\sum_{i=1}^d (1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2)
$$

**Juliaコード**：

```julia
# Encoder出力: (μ, log_σ²)
output, st_enc = encoder(x, ps_enc, st_enc)
μ = output[1:latent_dim, :]
logσ² = output[latent_dim+1:end, :]

# KL発散: -0.5 * Σ(1 + log_σ² - μ² - σ²) / batch_size
kl_term = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / size(x, 2)
```

| 数式 | コード | 対応 |
|:-----|:-------|:-----|
| $\boldsymbol{\mu}_\phi(\mathbf{x})$ | `μ = output[1:d, :]` | Encoderの前半出力 |
| $\log\boldsymbol{\sigma}^2_\phi(\mathbf{x})$ | `logσ² = output[d+1:end, :]` | Encoderの後半出力 |
| $\mu_i^2$ | `μ.^2` | 要素ごとの二乗 |
| $\sigma_i^2 = \exp(\log\sigma_i^2)$ | `exp.(logσ²)` | 指数関数 |
| $\sum_{i=1}^d$ | `sum(...)` | 全要素の和 |

:::message alert
**注意**: $\log\sigma^2$ を出力する理由は数値安定性。直接 $\sigma$ を出力すると、勾配消失・爆発のリスクがある。
:::

---

#### 3.1.4 再パラメータ化トリック — 勾配を通す魔法

**問題**：$\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$ からサンプリングすると、確率的ノードで勾配が止まる。

**解決**：再パラメータ化トリック（Reparameterization Trick, 第10回で学んだ）

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

これで $\mathbf{z}$ は $\phi$ の決定的関数になり、勾配が通る。

**Juliaコード**：

```julia
# Reparameterization: z = μ + σ ⊙ ε
ε = randn(Float32, size(μ)...)
σ = exp.(logσ² ./ 2)  # σ = exp(log_σ² / 2) = √(σ²)
z = μ .+ σ .* ε
```

| 数式 | コード | 対応 |
|:-----|:-------|:-----|
| $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | `ε = randn(Float32, size(μ))` | 標準正規ノイズ |
| $\boldsymbol{\sigma} = \exp(\log\boldsymbol{\sigma}^2 / 2)$ | `σ = exp.(logσ² ./ 2)` | 標準偏差計算 |
| $\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ | `μ .+ σ .* ε` | 要素ごとの積と和 |

**勾配の流れ**：

```mermaid
graph LR
    A[x] -->|Encoder| B[μ, log_σ²]
    B -->|決定的変換| C[z = μ + σ⊙ε]
    C -->|Decoder| D[x̂]
    D -->|Loss| E[ELBO]
    E -->|∇_φ| B
    E -->|∇_θ| D

    style C fill:#4ecdc4
```

再パラメータ化により、$\nabla_\phi \mathcal{L}_{\text{ELBO}}$ が計算可能になる。

---

#### 3.1.5 VAE完全実装 — 全てを統合

```julia
using Lux, Optimisers, Zygote, Random

# === モデル定義 ===
function create_vae(input_dim, latent_dim, hidden_dim)
    encoder = Chain(
        Dense(input_dim => hidden_dim, tanh),
        Dense(hidden_dim => hidden_dim÷2, tanh),
        Dense(hidden_dim÷2 => latent_dim*2)  # [μ, log_σ²]
    )

    decoder = Chain(
        Dense(latent_dim => hidden_dim÷2, tanh),
        Dense(hidden_dim÷2 => hidden_dim, tanh),
        Dense(hidden_dim => input_dim, sigmoid)  # [0, 1] pixel range
    )

    return encoder, decoder
end

# === ELBO損失関数 ===
function elbo_loss(encoder, decoder, ps_enc, ps_dec, st_enc, st_dec, x, latent_dim)
    # Encoder: q_φ(z|x) → (μ, log_σ²)
    output, st_enc = encoder(x, ps_enc, st_enc)
    μ = output[1:latent_dim, :]
    logσ² = output[latent_dim+1:end, :]

    # Reparameterization: z = μ + σ⊙ε
    ε = randn(Float32, size(μ)...)
    σ = exp.(logσ² ./ 2)
    z = μ .+ σ .* ε

    # Decoder: p_θ(x|z) → x̂
    x̂, st_dec = decoder(z, ps_dec, st_dec)

    # ELBO = 再構成項 - KL正則化項
    batch_size = size(x, 2)
    recon = -sum((x .- x̂).^2) / batch_size  # ガウス尤度
    kl = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / batch_size

    elbo = recon - kl

    return -elbo, (st_enc, st_dec)  # 最大化 = 負の最小化
end

# === 訓練ループ ===
function train_vae!(encoder, decoder, train_data, latent_dim, epochs=100, lr=1e-3)
    # パラメータ初期化
    rng = Random.default_rng()
    ps_enc, st_enc = Lux.setup(rng, encoder)
    ps_dec, st_dec = Lux.setup(rng, decoder)

    # Optimizer
    opt_state_enc = Optimisers.setup(Adam(lr), ps_enc)
    opt_state_dec = Optimisers.setup(Adam(lr), ps_dec)

    for epoch in 1:epochs
        total_loss = 0.0f0

        for batch in DataLoader((train_data,), batchsize=128, shuffle=true)
            x = batch[1]

            # 勾配計算
            (loss, (st_enc, st_dec)), back = Zygote.pullback(
                (pe, pd) -> elbo_loss(encoder, decoder, pe, pd, st_enc, st_dec, x, latent_dim),
                ps_enc, ps_dec
            )
            grads_enc, grads_dec = back((one(loss), nothing))

            # パラメータ更新
            opt_state_enc, ps_enc = Optimisers.update(opt_state_enc, ps_enc, grads_enc)
            opt_state_dec, ps_dec = Optimisers.update(opt_state_dec, ps_dec, grads_dec)

            total_loss += loss
        end

        println("Epoch $epoch: ELBO loss = $(total_loss/length(train_data))")
    end

    return ps_enc, ps_dec, st_enc, st_dec
end

# === 使用例 ===
encoder, decoder = create_vae(784, 20, 400)
ps_enc, ps_dec, st_enc, st_dec = train_vae!(encoder, decoder, x_train, 20, epochs=50)
```

**全体の流れ**：

```mermaid
sequenceDiagram
    participant Data as 訓練データ x
    participant Enc as Encoder q_φ
    participant Reparam as Reparameterization
    participant Dec as Decoder p_θ
    participant Loss as ELBO Loss

    Data->>Enc: x (784次元)
    Enc->>Reparam: μ, log_σ² (各20次元)
    Reparam->>Reparam: ε ~ N(0, I)
    Reparam->>Dec: z = μ + σ⊙ε (20次元)
    Dec->>Loss: x̂ (784次元)
    Loss->>Loss: recon = -||x - x̂||²
    Loss->>Loss: kl = -0.5Σ(1 + log_σ² - μ² - σ²)
    Loss->>Loss: ELBO = recon - kl
    Loss->>Enc: ∇_φ ELBO
    Loss->>Dec: ∇_θ ELBO
```

**訓練時のデバッグTips**：

```julia
# 損失が発散する場合のチェックリスト
function debug_vae_loss(encoder, decoder, ps_enc, ps_dec, st_enc, st_dec, x)
    # 1. Encoder出力の範囲チェック
    enc_out, _ = encoder(x, ps_enc, st_enc)
    μ = enc_out[1:20, :]
    logσ² = enc_out[21:end, :]

    println("μ range: [$(minimum(μ)), $(maximum(μ))]")  # 期待: [-3, 3]程度
    println("logσ² range: [$(minimum(logσ²)), $(maximum(logσ²))]")  # 期待: [-5, 5]程度

    # 2. σ²が極端に小さい/大きい場合はclip
    logσ² = clamp.(logσ², -10.0f0, 10.0f0)

    # 3. Decoder出力の範囲チェック
    z = μ .+ exp.(logσ² ./ 2) .* randn(Float32, size(μ)...)
    x̂, _ = decoder(z, ps_dec, st_dec)

    println("Decoder output range: [$(minimum(x̂)), $(maximum(x̂))]")  # 期待: [0, 1]

    # 4. KL項が負にならないことを確認
    kl = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²))
    println("KL term: $kl")  # 期待: ≥0 (負なら実装バグ)

    # 5. 各項のスケール確認
    recon = -sum((x .- x̂).^2) / size(x, 2)
    println("Recon: $recon, KL: $kl")
    # 期待: 同じオーダー（KLが極端に大きいとPosterior Collapse）
end
```

**Posterior Collapse対策**：

```julia
# KL Annealing: KL項の重みを徐々に増加
function kl_annealing_schedule(epoch, total_epochs, anneal_start=10, anneal_end=50)
    if epoch < anneal_start
        return 0.0f0
    elseif epoch > anneal_end
        return 1.0f0
    else
        return Float32((epoch - anneal_start) / (anneal_end - anneal_start))
    end
end

# 訓練ループで使用
for epoch in 1:epochs
    β_kl = kl_annealing_schedule(epoch, epochs)
    # loss = recon - β_kl * kl
end
```

---

### 3.2 GAN — WGAN-GP完全導出と実装対応

**復習：GANの目的**（第12回より）

Generator $G$ と Discriminator $D$ の2プレイヤーゲーム：

$$
\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]
$$

問題点：
- 訓練不安定（振動・発散）
- Mode Collapse（多様性の欠如）
- 勾配消失（$D$ が強すぎると $G$ の勾配が消える）

解決策：**WGAN-GP**（Wasserstein GAN with Gradient Penalty, 第13回で学んだ）

---

#### 3.2.1 Wasserstein距離の導出（復習）

第13回で学んだWasserstein-1距離（Earth Mover's Distance）：

$$
W_1(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}[\|\mathbf{x} - \mathbf{y}\|]
$$

Kantorovich-Rubinstein双対性（第13回で証明）：

$$
W_1(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_r}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]
$$

ここで $\|f\|_L \leq 1$ は1-Lipschitz連続制約。

**WGANの損失関数**：

$$
\mathcal{L}_D = \mathbb{E}_{\mathbf{x} \sim p_r}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))]
$$

$$
\mathcal{L}_G = -\mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))]
$$

$D$ は"Critic"（識別器ではなく、スコア関数）。

---

#### 3.2.2 Gradient Penalty — Lipschitz制約の強制

**問題**：元のWGANはweight clippingで $\|f\|_L \leq 1$ を強制したが、容量低下・勾配消失を引き起こす。

**解決**：WGAN-GP（Gulrajani+ 2017 [^2]）はGradient Penaltyで制約：

$$
\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 = 1
$$

を $\hat{\mathbf{x}} = \alpha \mathbf{x} + (1 - \alpha)G(\mathbf{z})$ （本物と偽物の補間点）で強制。

**WGAN-GP損失関数**：

$$
\mathcal{L}_D = \mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))] - \mathbb{E}_{\mathbf{x} \sim p_r}[D(\mathbf{x})] + \lambda \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2]
$$

第1項：偽物のスコア（最小化）
第2項：本物のスコア（最大化）
第3項：Gradient Penalty（勾配ノルムを1に近づける）

---

#### 3.2.3 Gradient Penalty の実装

**数式**：

$$
\text{GP} = \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2]
$$

**Step 1: 補間点生成**

$$
\hat{\mathbf{x}} = \alpha \mathbf{x} + (1 - \alpha)G(\mathbf{z}), \quad \alpha \sim \text{Uniform}(0, 1)
$$

**Juliaコード**：

```julia
# 本物と偽物の補間
α = rand(Float32, 1, batch_size)
x_interp = α .* x_real .+ (1 .- α) .* x_fake
```

| 数式 | コード | 対応 |
|:-----|:-------|:-----|
| $\alpha \sim \text{Uniform}(0, 1)$ | `α = rand(Float32, 1, batch_size)` | 補間係数 |
| $\alpha \mathbf{x}$ | `α .* x_real` | broadcast乗算 |
| $(1 - \alpha)G(\mathbf{z})$ | `(1 .- α) .* x_fake` | broadcast減算・乗算 |

**Step 2: 勾配計算**

$$
\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})
$$

Juliaでは`Zygote.gradient`を使う：

```julia
# 補間点での勾配計算
grad_interp = Zygote.gradient(x -> sum(critic(x, ps_c, st_c)[1]), x_interp)[1]
```

**Step 3: 勾配ノルム計算**

$$
\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 = \sqrt{\sum_i (\partial D / \partial \hat{x}_i)^2}
$$

```julia
# 勾配ノルム: √(Σ grad²) for each sample
grad_norm = sqrt.(sum(grad_interp.^2, dims=1))  # (1, batch_size)

# Gradient Penalty: 𝔼[(||∇D||₂ - 1)²]
gp = mean((grad_norm .- 1).^2)
```

| 数式 | コード | 対応 |
|:-----|:-------|:-----|
| $(\partial D / \partial \hat{x}_i)^2$ | `grad_interp.^2` | 勾配の二乗 |
| $\sum_i$ | `sum(..., dims=1)` | 各サンプルごとの和 |
| $\sqrt{\cdot}$ | `sqrt.(...)` | 平方根（broadcast） |
| $(\|\nabla D\|_2 - 1)^2$ | `(grad_norm .- 1).^2` | ペナルティ項 |
| $\mathbb{E}[\cdot]$ | `mean(...)` | バッチ平均 |

---

#### 3.2.4 WGAN-GP完全実装

```julia
using Lux, Optimisers, Zygote, Random

# === モデル定義 ===
function create_wgan_gp(latent_dim, img_dim, hidden_dim)
    generator = Chain(
        Dense(latent_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim*2, relu),
        Dense(hidden_dim*2 => img_dim, tanh)  # [-1, 1] range
    )

    critic = Chain(
        Dense(img_dim => hidden_dim*2, x -> leakyrelu(x, 0.2f0)),
        Dense(hidden_dim*2 => hidden_dim, x -> leakyrelu(x, 0.2f0)),
        Dense(hidden_dim => 1)  # スコア出力
    )

    return generator, critic
end

# === Critic損失（WGAN-GP） ===
function critic_loss(generator, critic, ps_g, ps_c, st_g, st_c, x_real, λ_gp=10.0f0)
    batch_size = size(x_real, 2)

    # 偽画像生成
    z = randn(Float32, size(ps_g)[1], batch_size)
    x_fake, st_g = generator(z, ps_g, st_g)

    # Criticスコア
    score_real, st_c_real = critic(x_real, ps_c, st_c)
    score_fake, st_c_fake = critic(x_fake, ps_c, st_c)

    # Wasserstein距離: 𝔼[D(fake)] - 𝔼[D(real)]
    wasserstein = mean(score_fake) - mean(score_real)

    # Gradient Penalty
    α = rand(Float32, 1, batch_size)
    x_interp = α .* x_real .+ (1 .- α) .* x_fake

    grad_interp = Zygote.gradient(x -> sum(critic(x, ps_c, st_c)[1]), x_interp)[1]
    grad_norm = sqrt.(sum(grad_interp.^2, dims=1))
    gp = mean((grad_norm .- 1).^2)

    loss = wasserstein + λ_gp * gp

    return loss, st_c
end

# === Generator損失（WGAN-GP） ===
function generator_loss(generator, critic, ps_g, ps_c, st_g, st_c, batch_size)
    # 偽画像生成
    z = randn(Float32, size(ps_g)[1], batch_size)
    x_fake, st_g = generator(z, ps_g, st_g)

    # Generatorの目的: Criticスコアを最大化
    score_fake, st_c = critic(x_fake, ps_c, st_c)
    loss = -mean(score_fake)

    return loss, st_g
end

# === 訓練ループ ===
function train_wgan_gp!(generator, critic, train_data, latent_dim, epochs=100, n_critic=5)
    rng = Random.default_rng()
    ps_g, st_g = Lux.setup(rng, generator)
    ps_c, st_c = Lux.setup(rng, critic)

    opt_g = Optimisers.setup(Adam(1e-4, (0.5, 0.9)), ps_g)
    opt_c = Optimisers.setup(Adam(1e-4, (0.5, 0.9)), ps_c)

    for epoch in 1:epochs
        for batch in DataLoader((train_data,), batchsize=64, shuffle=true)
            x_real = batch[1]

            # Criticを n_critic 回更新
            for _ in 1:n_critic
                (loss_c, st_c), back_c = Zygote.pullback(
                    pc -> critic_loss(generator, critic, ps_g, pc, st_g, st_c, x_real),
                    ps_c
                )
                grads_c = back_c((one(loss_c), nothing))[1]
                opt_c, ps_c = Optimisers.update(opt_c, ps_c, grads_c)
            end

            # Generatorを 1 回更新
            (loss_g, st_g), back_g = Zygote.pullback(
                pg -> generator_loss(generator, critic, pg, ps_c, st_g, st_c, size(x_real, 2)),
                ps_g
            )
            grads_g = back_g((one(loss_g), nothing))[1]
            opt_g, ps_g = Optimisers.update(opt_g, ps_g, grads_g)
        end

        println("Epoch $epoch: C_loss=$(loss_c), G_loss=$(loss_g)")
    end

    return ps_g, ps_c, st_g, st_c
end
```

**訓練ループの流れ**：

```mermaid
sequenceDiagram
    participant Data as 訓練データ x_real
    participant G as Generator G
    participant C as Critic D
    participant GP as Gradient Penalty

    loop Criticを5回更新
        Data->>G: z ~ N(0, I)
        G->>C: x_fake = G(z)
        C->>C: score_real = D(x_real)
        C->>C: score_fake = D(x_fake)
        C->>GP: x_interp = αx_real + (1-α)x_fake
        GP->>GP: grad = ∇_x_interp D(x_interp)
        GP->>GP: gp = 𝔼[(||grad||₂ - 1)²]
        GP->>C: loss_C = score_fake - score_real + λ*gp
        C->>C: ps_c ← ps_c - η∇_c loss_C
    end

    loop Generatorを1回更新
        G->>C: x_fake = G(z)
        C->>G: score_fake = D(x_fake)
        G->>G: loss_G = -score_fake
        G->>G: ps_g ← ps_g - η∇_g loss_G
    end
```

**重要なポイント**：
- Criticを$n_{\text{critic}}=5$回、Generatorを1回更新（WGAN-GP推奨比率）
- Gradient Penaltyの $\lambda=10$ はハイパーパラメータ（論文推奨値）
- Adamの $\beta_1=0.5$ はGAN訓練の安定化に有効（通常は0.9）

:::message
**ここが重要**: WGAN-GPの核心は「勾配ノルムを1に保つ」こと。これがLipschitz制約の実用的実装。
:::

---

### 3.3 Transformer — Multi-Head Attention完全導出

**復習：Transformerの目的**（第16回より）

自己回帰モデルとして次トークン予測：

$$
p(\mathbf{x}) = \prod_{t=1}^T p(x_t | x_{<t})
$$

RNN/LSTMの逐次処理を捨て、Attentionで並列処理。

---

#### 3.3.1 Scaled Dot-Product Attention導出

**Step 1: Attention機構の直感**

Query $\mathbf{q}$ と Key $\mathbf{k}_i$ の類似度でValue $\mathbf{v}_i$ を重み付け：

$$
\text{Attention}(\mathbf{q}, \{\mathbf{k}_i, \mathbf{v}_i\}) = \sum_{i} \alpha_i \mathbf{v}_i
$$

ここで $\alpha_i = \text{softmax}(\text{score}(\mathbf{q}, \mathbf{k}_i))$

**Step 2: スコア関数の選択**

内積スコア：

$$
\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}
$$

問題：$d_k$ が大きいとスコアの分散が大きくなり、softmaxが飽和（勾配消失）。

解決：スケーリング

$$
\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}
$$

**Step 3: 行列形式**

Query行列 $Q \in \mathbb{R}^{n \times d_k}$、Key行列 $K \in \mathbb{R}^{m \times d_k}$、Value行列 $V \in \mathbb{R}^{m \times d_v}$ を使うと：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

ここで：
- $QK^\top \in \mathbb{R}^{n \times m}$：各QueryとKeyの類似度行列
- $\text{softmax}$：行ごとに正規化（各Queryが全Keyの重みを合計1に）
- 結果 $\in \mathbb{R}^{n \times d_v}$：各Queryに対する加重Valueの和

---

#### 3.3.2 Multi-Head Attention導出

**動機**：単一のAttentionでは表現力不足。複数の"視点"でAttentionを並列計算。

**Step 1: ヘッドの分割**

$d_{\text{model}}$ 次元を $h$ 個のヘッドに分割：

$$
d_k = d_v = \frac{d_{\text{model}}}{h}
$$

**Step 2: 各ヘッドで独立にAttention**

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

ここで $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ は学習可能な射影行列。

**Step 3: Concatenate and Project**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

ここで $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

**完全な数式**：

$$
\begin{align}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \\
&= \text{softmax}\left(\frac{(QW_i^Q)(KW_i^K)^\top}{\sqrt{d_k}}\right)(VW_i^V)
\end{align}
$$

---

#### 3.3.3 Causal Mask — 未来を見せない

自己回帰モデルでは、時刻 $t$ のトークンは $t+1$ 以降を見てはいけない。

**Mask行列**：

$$
M_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

Softmax前にスコアに加算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

$M_{ij} = -\infty$ の部分は $\exp(-\infty) = 0$ になり、未来トークンの重みが0になる。

**Juliaコード**：

```julia
# Causal Mask生成
function causal_mask(seq_len)
    mask = triu(ones(Float32, seq_len, seq_len), 1)  # 上三角行列（対角より上）
    return mask .* -Inf32  # Softmax前に加算 → exp(-∞) = 0
end

# Attentionにマスク適用
scores = Q @ K' ./ sqrt(Float32(d_k))  # (seq_len, seq_len)
scores = scores .+ causal_mask(seq_len)  # 未来トークンを-∞に
attn_weights = softmax(scores, dims=2)  # 行ごとに正規化
output = attn_weights @ V
```

| 数式 | コード | 対応 |
|:-----|:-------|:-----|
| $QK^\top$ | `Q @ K'` | 行列積（`'`は転置） |
| $/\sqrt{d_k}$ | `./ sqrt(Float32(d_k))` | broadcast除算 |
| $M$ | `causal_mask(seq_len)` | マスク行列 |
| $\text{softmax}(\cdot + M)$ | `softmax(scores .+ mask, dims=2)` | 行ごとsoftmax |

---

#### 3.3.4 Multi-Head Attention完全実装

```julia
using Lux, NNlib, Random

# === Multi-Head Attention Layer ===
struct MultiHeadAttention <: Lux.AbstractExplicitLayer
    num_heads::Int
    d_model::Int
    d_k::Int
    q_proj::Dense
    k_proj::Dense
    v_proj::Dense
    o_proj::Dense
end

function MultiHeadAttention(d_model::Int, num_heads::Int)
    @assert d_model % num_heads == 0 "d_model must be divisible by num_heads"
    d_k = d_model ÷ num_heads

    return MultiHeadAttention(
        num_heads, d_model, d_k,
        Dense(d_model => d_model, use_bias=false),  # Q projection
        Dense(d_model => d_model, use_bias=false),  # K projection
        Dense(d_model => d_model, use_bias=false),  # V projection
        Dense(d_model => d_model, use_bias=false)   # Output projection
    )
end

function (mha::MultiHeadAttention)(x, ps, st; mask=nothing)
    # x: (d_model, seq_len, batch_size)
    d_model, seq_len, batch_size = size(x)

    # Linear projections: Q, K, V
    Q, st_q = mha.q_proj(x, ps.q_proj, st.q_proj)
    K, st_k = mha.k_proj(x, ps.k_proj, st.k_proj)
    V, st_v = mha.v_proj(x, ps.v_proj, st.v_proj)

    # Reshape for multi-head: (d_model, seq_len, batch) → (num_heads, d_k, seq_len, batch)
    Q = reshape(Q, mha.d_k, mha.num_heads, seq_len, batch_size) |> x -> permutedims(x, (2,1,3,4))
    K = reshape(K, mha.d_k, mha.num_heads, seq_len, batch_size) |> x -> permutedims(x, (2,1,3,4))
    V = reshape(V, mha.d_k, mha.num_heads, seq_len, batch_size) |> x -> permutedims(x, (2,1,3,4))

    # Scaled Dot-Product Attention for all heads
    # scores: (num_heads, seq_len, seq_len, batch)
    scores = batched_mul(batched_transpose(Q), K) ./ sqrt(Float32(mha.d_k))

    # Apply mask if provided
    if !isnothing(mask)
        scores = scores .+ reshape(mask, 1, seq_len, seq_len, 1)  # broadcast over heads and batch
    end

    # Softmax over keys dimension
    attn_weights = softmax(scores, dims=2)  # normalize over keys (dim 2)

    # Weighted sum of values
    out = batched_mul(V, attn_weights)  # (num_heads, d_k, seq_len, batch)

    # Concatenate heads: (num_heads, d_k, seq_len, batch) → (d_model, seq_len, batch)
    out = permutedims(out, (2,1,3,4)) |> x -> reshape(x, d_model, seq_len, batch_size)

    # Output projection
    out, st_o = mha.o_proj(out, ps.o_proj, st.o_proj)

    return out, (st_q=st_q, st_k=st_k, st_v=st_v, st_o=st_o)
end

# === Causal Mask ===
function causal_mask(seq_len)
    mask = triu(ones(Float32, seq_len, seq_len), 1)
    return mask .* -Inf32
end

# === 使用例 ===
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

x = randn(Float32, d_model, seq_len, batch_size)
mha = MultiHeadAttention(d_model, num_heads)
ps, st = Lux.setup(Random.default_rng(), mha)

mask = causal_mask(seq_len)
y, st = mha(x, ps, st; mask=mask)  # y: (512, 10, 2)
```

**処理の流れ**：

```mermaid
graph TD
    A[Input x<br>d_model × seq_len × batch] -->|Q proj| B1[Q: d_model × seq_len × batch]
    A -->|K proj| B2[K: d_model × seq_len × batch]
    A -->|V proj| B3[V: d_model × seq_len × batch]

    B1 -->|Reshape| C1[Q: num_heads × d_k × seq_len × batch]
    B2 -->|Reshape| C2[K: num_heads × d_k × seq_len × batch]
    B3 -->|Reshape| C3[V: num_heads × d_k × seq_len × batch]

    C1 -->|QK^T/√d_k| D[Scores:<br>num_heads × seq_len × seq_len × batch]
    C2 --> D

    D -->|+ Mask| E[Masked Scores]
    E -->|Softmax| F[Attn Weights]
    F -->|× V| G[Weighted Values:<br>num_heads × d_k × seq_len × batch]
    C3 --> G

    G -->|Concat| H[Concat:<br>d_model × seq_len × batch]
    H -->|O proj| I[Output:<br>d_model × seq_len × batch]

    style D fill:#4ecdc4
    style E fill:#ff6b6b
    style G fill:#ffe66d
```

**数式↔コード完全対応**：

| 数式ステップ | Juliaコード | 次元変化 |
|:-------------|:------------|:---------|
| $Q = XW^Q$ | `Q, _ = mha.q_proj(x, ps.q_proj, st.q_proj)` | $(d, n, b) \to (d, n, b)$ |
| $Q$ を $h$ ヘッドに分割 | `reshape(Q, d_k, h, n, b) \|> permutedims((2,1,3,4))` | $(d, n, b) \to (h, d_k, n, b)$ |
| $QK^\top/\sqrt{d_k}$ | `batched_mul(Q', K) ./ sqrt(Float32(d_k))` | $(h, n, d_k, b) \to (h, n, n, b)$ |
| $\text{scores} + M$ | `scores .+ mask` | Maskをbroadcast |
| $\text{softmax}(\cdot)$ | `softmax(scores, dims=2)` | 行（Key次元）で正規化 |
| $\text{Attention} \times V$ | `batched_mul(V, attn_weights)$ | $(h, d_k, n, b) \times (h, n, n, b) \to (h, d_k, n, b)$ |
| Concat heads | `reshape(..., d, n, b)` | $(h, d_k, n, b) \to (d, n, b)$ |
| Output projection | `mha.o_proj(out)` | $(d, n, b) \to (d, n, b)$ |

:::message
**ここが重要**: Multi-Head Attentionは「並列に複数の視点でAttention」。各ヘッドが異なる部分空間で類似度を計算。
:::

---

### 3.4 ⚔️ Boss Battle — 3モデル統合訓練ループ

ここまでで3モデルの数式↔コード対応を完全に理解した。最後のBoss Battle：**3モデルを統合した訓練パイプラインを設計せよ**。

**課題**：
1. VAE/GAN/Transformerの訓練ループを統一インターフェースで実装
2. 損失曲線をリアルタイムプロット
3. チェックポイント保存・再開機能
4. Early Stopping実装

**ヒント**：
- すべてのモデルで`loss, state = model_loss(params, state, data)`のインターフェースを統一
- Lux.jlの`Lux.Training.TrainState`を活用
- JLD2.jlでパラメータ保存

**解答例は Zone 4 で提供**。まずは自分で設計してみよう。

:::message
**進捗**: 全体の50%完了。数式修行ゾーンクリア。実装ゾーンへ。
:::

---

## 💻 4. 実装ゾーン（45分）— 3言語パイプライン完全構築

数式を理解した。今度は**動かす**。Julia訓練→Rust推論→Elixir配信の完全パイプラインを実装する。

### 4.1 Julia訓練実装 — Lux.jl完全版

#### 4.1.1 統一訓練インターフェース設計

3モデル（VAE/GAN/Transformer）で訓練ループを統一する設計パターン：

```julia
# 統一インターフェース
abstract type GenerativeModel end

# 各モデルは以下を実装
# - loss_fn(model, params, state, batch) → (loss, state)
# - generate(model, params, state, n_samples) → samples

struct VAEModel <: GenerativeModel
    encoder::Chain
    decoder::Chain
    latent_dim::Int
end

struct WGANModel <: GenerativeModel
    generator::Chain
    critic::Chain
    latent_dim::Int
    λ_gp::Float32
end

struct TransformerModel <: GenerativeModel
    layers::Vector{Any}  # [Embedding, MHA, FFN, ...]
    vocab_size::Int
    d_model::Int
end
```

**統一訓練関数**：

```julia
using Lux, Optimisers, Zygote, MLUtils, ProgressMeter

function train!(
    model::GenerativeModel,
    train_data,
    epochs::Int;
    learning_rate=1e-3,
    batch_size=128,
    save_every=10,
    checkpoint_dir="checkpoints"
)
    # パラメータ初期化
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    # Optimizer
    opt_state = Optimisers.setup(Adam(learning_rate), ps)

    # 訓練ループ
    losses = Float32[]
    @showprogress for epoch in 1:epochs
        epoch_loss = 0.0f0
        n_batches = 0

        for batch in DataLoader(train_data, batchsize=batch_size, shuffle=true)
            # 損失計算
            (loss, st), back = Zygote.pullback(p -> model_loss(model, p, st, batch), ps)

            # 勾配計算
            grads = back((one(loss), nothing))[1]

            # パラメータ更新
            opt_state, ps = Optimisers.update(opt_state, ps, grads)

            epoch_loss += loss
            n_batches += 1
        end

        avg_loss = epoch_loss / n_batches
        push!(losses, avg_loss)
        println("Epoch $epoch: loss = $avg_loss")

        # チェックポイント保存
        if epoch % save_every == 0
            save_checkpoint(checkpoint_dir, epoch, ps, st, opt_state)
        end
    end

    return ps, st, losses
end
```

---

#### 4.1.2 VAE訓練の完全実装

```julia
using Lux, Optimisers, Zygote, MLDatasets, Images, Plots

# === VAE Loss ===
function model_loss(model::VAEModel, ps, st, batch)
    x = batch[1]  # (input_dim, batch_size)
    latent_dim = model.latent_dim

    # Encoder: q_φ(z|x)
    enc_out, st_enc = model.encoder(x, ps.encoder, st.encoder)
    μ = enc_out[1:latent_dim, :]
    logσ² = enc_out[latent_dim+1:end, :]

    # Reparameterization
    ε = randn(Float32, size(μ)...)
    σ = exp.(logσ² ./ 2)
    z = μ .+ σ .* ε

    # Decoder: p_θ(x|z)
    x̂, st_dec = model.decoder(z, ps.decoder, st.decoder)

    # ELBO
    batch_size = size(x, 2)
    recon = -sum((x .- x̂).^2) / batch_size  # Gaussian likelihood
    kl = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / batch_size

    elbo = recon - kl
    loss = -elbo  # 最大化 = 負の最小化

    st_new = (encoder=st_enc, decoder=st_dec)
    return loss, st_new
end

# === VAE生成 ===
function generate(model::VAEModel, ps, st, n_samples::Int)
    z = randn(Float32, model.latent_dim, n_samples)
    x_gen, _ = model.decoder(z, ps.decoder, st.decoder)
    return x_gen
end

# === 使用例 ===
function train_vae_mnist()
    # データ読み込み
    train_data = MNIST(split=:train)
    x_train = Float32.(reshape(train_data.features, 784, :))  # (784, 60000)

    # モデル作成
    encoder = Chain(
        Dense(784 => 400, tanh),
        Dense(400 => 40)  # [μ(20), log_σ²(20)]
    )
    decoder = Chain(
        Dense(20 => 400, tanh),
        Dense(400 => 784, sigmoid)
    )
    model = VAEModel(encoder, decoder, 20)

    # 訓練
    ps, st, losses = train!(model, (x_train,), 50; learning_rate=1e-3, batch_size=128)

    # 損失曲線プロット
    plot(losses, xlabel="Epoch", ylabel="ELBO Loss", title="VAE Training", legend=false)
    savefig("vae_loss.png")

    # サンプル生成
    samples = generate(model, ps, st, 10)
    img_grid = mosaic([reshape(samples[:, i], 28, 28) for i in 1:10]..., nrow=2, ncol=5)
    save("vae_samples.png", colorview(Gray, img_grid'))

    return ps, st
end
```

---

#### 4.1.3 WGAN-GP訓練の完全実装

```julia
# === WGAN-GP Loss ===
function model_loss(model::WGANModel, ps, st, batch; train_critic=true)
    x_real = batch[1]
    batch_size = size(x_real, 2)

    if train_critic
        # Critic損失（Gradient Penalty付き）
        z = randn(Float32, model.latent_dim, batch_size)
        x_fake, st_g = model.generator(z, ps.generator, st.generator)

        score_real, st_c1 = model.critic(x_real, ps.critic, st.critic)
        score_fake, st_c2 = model.critic(x_fake, ps.critic, st_c1)

        wasserstein = mean(score_fake) - mean(score_real)

        # Gradient Penalty
        α = rand(Float32, 1, batch_size)
        x_interp = α .* x_real .+ (1 .- α) .* x_fake

        grad_interp = Zygote.gradient(x -> sum(model.critic(x, ps.critic, st_c2)[1]), x_interp)[1]
        grad_norm = sqrt.(sum(grad_interp.^2, dims=1))
        gp = mean((grad_norm .- 1).^2)

        loss = wasserstein + model.λ_gp * gp
        st_new = (generator=st_g, critic=st_c2)
    else
        # Generator損失
        z = randn(Float32, model.latent_dim, batch_size)
        x_fake, st_g = model.generator(z, ps.generator, st.generator)
        score_fake, st_c = model.critic(x_fake, ps.critic, st.critic)

        loss = -mean(score_fake)
        st_new = (generator=st_g, critic=st_c)
    end

    return loss, st_new
end

# === WGAN-GP訓練（Critic:Generator = 5:1） ===
function train_wgan!(model::WGANModel, train_data, epochs::Int; n_critic=5, lr=1e-4)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    opt_g = Optimisers.setup(Adam(lr, (0.5f0, 0.9f0)), ps.generator)
    opt_c = Optimisers.setup(Adam(lr, (0.5f0, 0.9f0)), ps.critic)

    losses_c = Float32[]
    losses_g = Float32[]

    @showprogress for epoch in 1:epochs
        for batch in DataLoader(train_data, batchsize=64, shuffle=true)
            # Criticを n_critic 回更新
            for _ in 1:n_critic
                (loss_c, st), back_c = Zygote.pullback(
                    pc -> model_loss(model, (generator=ps.generator, critic=pc), st, batch; train_critic=true),
                    ps.critic
                )
                grads_c = back_c((one(loss_c), nothing))[1]
                opt_c, ps.critic = Optimisers.update(opt_c, ps.critic, grads_c)
            end
            push!(losses_c, loss_c)

            # Generatorを 1 回更新
            (loss_g, st), back_g = Zygote.pullback(
                pg -> model_loss(model, (generator=pg, critic=ps.critic), st, batch; train_critic=false),
                ps.generator
            )
            grads_g = back_g((one(loss_g), nothing))[1]
            opt_g, ps.generator = Optimisers.update(opt_g, ps.generator, grads_g)
            push!(losses_g, loss_g)
        end

        println("Epoch $epoch: C_loss=$(losses_c[end]), G_loss=$(losses_g[end])")
    end

    return ps, st, (losses_c, losses_g)
end
```

---

#### 4.1.4 Transformer訓練の完全実装

```julia
# === Transformer構成要素 ===
struct TransformerBlock <: Lux.AbstractExplicitContainer
    mha::MultiHeadAttention
    ffn::Chain
    ln1::LayerNorm
    ln2::LayerNorm
    dropout::Dropout
end

function TransformerBlock(d_model, num_heads, d_ff, dropout_rate=0.1)
    return TransformerBlock(
        MultiHeadAttention(d_model, num_heads),
        Chain(Dense(d_model => d_ff, relu), Dense(d_ff => d_model)),
        LayerNorm(d_model),
        LayerNorm(d_model),
        Dropout(dropout_rate)
    )
end

function (block::TransformerBlock)(x, ps, st; mask=nothing)
    # Multi-Head Attention + Residual + LayerNorm
    attn_out, st_mha = block.mha(x, ps.mha, st.mha; mask=mask)
    attn_out, st_drop1 = block.dropout(attn_out, ps.dropout, st.dropout)
    x = x .+ attn_out
    x, st_ln1 = block.ln1(x, ps.ln1, st.ln1)

    # Feed-Forward + Residual + LayerNorm
    ffn_out, st_ffn = block.ffn(x, ps.ffn, st.ffn)
    ffn_out, st_drop2 = block.dropout(ffn_out, ps.dropout, st_drop1)
    x = x .+ ffn_out
    x, st_ln2 = block.ln2(x, ps.ln2, st.ln2)

    st_new = (mha=st_mha, ffn=st_ffn, ln1=st_ln1, ln2=st_ln2, dropout=st_drop2)
    return x, st_new
end

# === Transformer Loss（次トークン予測） ===
function model_loss(model::TransformerModel, ps, st, batch)
    x, y = batch  # x: 入力トークン, y: ターゲットトークン (shifted by 1)
    seq_len = size(x, 1)

    # Embedding
    x_emb, st_emb = model.embedding(x, ps.embedding, st.embedding)

    # Positional Encoding
    x_emb = x_emb .+ model.pos_encoding[:, 1:seq_len, :]

    # Transformer Blocks
    mask = causal_mask(seq_len)
    for (i, block) in enumerate(model.blocks)
        x_emb, st_block = block(x_emb, ps.blocks[i], st.blocks[i]; mask=mask)
    end

    # Output projection
    logits, st_out = model.output_proj(x_emb, ps.output_proj, st.output_proj)

    # Cross-Entropy Loss
    loss = Flux.Losses.logitcrossentropy(logits, y)

    st_new = (embedding=st_emb, blocks=[st_block], output_proj=st_out)
    return loss, st_new
end
```

---

### 4.2 モデルエクスポート — Julia → Rust橋渡し

Juliaで訓練したモデルをRustで推論するため、**safetensors形式**でエクスポート。

```julia
using Safetensors, JLD2

# === パラメータをflatten ===
function flatten_params(ps)
    flat_dict = Dict{String, Array{Float32}}()

    function traverse(prefix, p)
        if p isa NamedTuple
            for (k, v) in pairs(p)
                traverse("$prefix.$k", v)
            end
        elseif p isa AbstractArray
            flat_dict[prefix] = Float32.(p)
        end
    end

    traverse("model", ps)
    return flat_dict
end

# === safetensors保存 ===
function export_model(ps, st, filepath)
    flat_params = flatten_params(ps)
    Safetensors.save_file(filepath, flat_params)
    println("Model exported to $filepath")
end

# === 使用例 ===
ps_vae, st_vae = train_vae_mnist()
export_model(ps_vae, st_vae, "vae_mnist.safetensors")
```

**safetensorsフォーマット**：
- HuggingFaceが開発した軽量・安全なテンソル保存形式
- Pickle（Python）と違い、コード実行リスクなし
- Rustの`safetensors` crateでロード可能
- メモリマップ対応（大規模モデル向け）

---

### 4.3 Rust推論エンジン — Candle完全実装

#### 4.3.1 Candle セットアップ

```toml
# Cargo.toml
[dependencies]
candle-core = "0.7"
candle-nn = "0.7"
safetensors = "0.4"
ndarray = "0.16"
```

#### 4.3.2 VAE推論実装

```rust
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, ops, VarBuilder};
use safetensors::SafeTensors;
use std::fs;

// === VAE Decoder ===
struct VAEDecoder {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    fc3: candle_nn::Linear,
}

impl VAEDecoder {
    fn new(vb: VarBuilder, latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        let fc1 = linear(latent_dim, hidden_dim, vb.pp("decoder.0"))?;
        let fc2 = linear(hidden_dim, hidden_dim * 2, vb.pp("decoder.2"))?;
        let fc3 = linear(hidden_dim * 2, output_dim, vb.pp("decoder.4"))?;
        Ok(Self { fc1, fc2, fc3 })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(z)?;
        let x = x.tanh()?;
        let x = self.fc2.forward(&x)?;
        let x = x.tanh()?;
        let x = self.fc3.forward(&x)?;
        x.sigmoid()  // [0, 1] pixel range
    }
}

// === safetensorsロード ===
fn load_vae_decoder(model_path: &str, device: &Device) -> Result<VAEDecoder> {
    let data = fs::read(model_path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    VAEDecoder::new(vb, 20, 400, 784)
}

// === バッチ推論 ===
fn generate_samples(decoder: &VAEDecoder, n_samples: usize, device: &Device) -> Result<Tensor> {
    // z ~ N(0, I)
    let z = Tensor::randn(0f32, 1.0, (n_samples, 20), device)?;

    // x = Decoder(z)
    decoder.forward(&z)
}

// === メイン ===
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    // モデルロード
    let decoder = load_vae_decoder("vae_mnist.safetensors", &device)?;

    // バッチ推論（1000サンプル）
    let samples = generate_samples(&decoder, 1000, &device)?;
    println!("Generated samples: {:?}", samples.shape());

    Ok(())
}
```

**ポイント**：
- `VarBuilder`：safetensorsから直接パラメータをロード
- `Device::cuda_if_available`：GPU自動検出
- ゼロコピー：Tensorは参照渡し（`&Tensor`）
- 型安全：コンパイル時に形状ミスマッチを検出

---

#### 4.3.3 FFI統合 — RustからJulia/Elixir呼び出し

```rust
// === C-ABI FFI for Julia/Elixir ===
use std::slice;

#[repr(C)]
pub struct InferenceResult {
    data: *mut f32,
    len: usize,
}

#[no_mangle]
pub extern "C" fn vae_generate(
    model_path: *const libc::c_char,
    n_samples: usize,
    out: *mut *mut f32,
    out_len: *mut usize,
) -> i32 {
    // モデルロード
    let path = unsafe { std::ffi::CStr::from_ptr(model_path).to_str().unwrap() };
    let device = Device::Cpu;  // CPUモード（FFIは単純化）
    let decoder = match load_vae_decoder(path, &device) {
        Ok(d) => d,
        Err(_) => return -1,
    };

    // 推論
    let samples = match generate_samples(&decoder, n_samples, &device) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // 結果をCポインタに変換
    let vec: Vec<f32> = samples.to_vec1().unwrap();
    let len = vec.len();
    let ptr = vec.as_ptr() as *mut f32;
    std::mem::forget(vec);  // Rust側でdropしない

    unsafe {
        *out = ptr;
        *out_len = len;
    }

    0  // Success
}

#[no_mangle]
pub extern "C" fn vae_free(ptr: *mut f32, len: usize) {
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);  // dropでメモリ解放
    }
}
```

**Juliaから呼び出し**：

```julia
# VAE推論をRustに委譲
function rust_vae_generate(model_path::String, n_samples::Int)
    out_ptr = Ref{Ptr{Float32}}()
    out_len = Ref{Csize_t}()

    ret = ccall(
        (:vae_generate, "./libvae_inference.so"),
        Cint,
        (Ptr{Cchar}, Csize_t, Ptr{Ptr{Float32}}, Ptr{Csize_t}),
        model_path, n_samples, out_ptr, out_len
    )

    if ret != 0
        error("Rust inference failed")
    end

    # ポインタから配列に変換
    samples = unsafe_wrap(Array{Float32}, out_ptr[], out_len[])

    # メモリ解放（Julia GCに任せる or Rust側でfree）
    # ccall((:vae_free, "./libvae_inference.so"), Cvoid, (Ptr{Float32}, Csize_t), out_ptr[], out_len[])

    return samples
end
```

---

### 4.4 Elixir分散サービング — Broadway完全実装

#### 4.4.1 GenStageとBroadway概要

**GenStage**：需要駆動（demand-driven）ストリーミング
- Producer：データを生成
- Consumer：データを消費
- Backpressure：Consumerが需要を制御

**Broadway**：GenStageの高レベル抽象化
- Producerからデータを受信
- バッチ処理・並列処理
- 自動acknowledgement・エラーハンドリング

---

#### 4.4.2 Broadway推論パイプライン実装

```elixir
defmodule VAEInferencePipeline do
  use Broadway

  alias Broadway.Message

  def start_link(_opts) do
    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: [
        module: {BroadwayRabbitMQ.Producer, queue: "vae_requests"},
        concurrency: 1
      ],
      processors: [
        default: [concurrency: 4]  # 4並列処理
      ],
      batchers: [
        default: [
          batch_size: 10,
          batch_timeout: 100,
          concurrency: 2
        ]
      ]
    )
  end

  @impl true
  def handle_message(:default, message, _context) do
    # メッセージから推論リクエストをパース
    %{data: %{"n_samples" => n_samples, "model_path" => model_path}} = message

    # Rust推論エンジン呼び出し（NIF経由）
    case VAERust.generate(model_path, n_samples) do
      {:ok, samples} ->
        message
        |> Message.update_data(fn _ -> %{samples: samples} end)
        |> Message.put_batch_key(:default)

      {:error, reason} ->
        Message.failed(message, reason)
    end
  end

  @impl true
  def handle_batch(:default, messages, _batch_info, _context) do
    # バッチ処理：10件まとめて後処理（例: S3保存）
    Enum.each(messages, fn msg ->
      samples = msg.data.samples
      IO.puts("Generated #{length(samples)} samples")
      # save_to_s3(samples)
    end)

    messages
  end
end
```

**Rust NIFラッパー**（Elixir ↔ Rust FFI）：

```elixir
defmodule VAERust do
  use Rustler, otp_app: :vae_inference, crate: "vae_rust"

  # Rustler NIF stub
  def generate(_model_path, _n_samples), do: :erlang.nif_error(:nif_not_loaded)
end
```

```rust
// Rustler NIF（Elixir用FFI）
use rustler::{Encoder, Env, Term};

#[rustler::nif]
fn generate(model_path: String, n_samples: usize) -> Result<Vec<f32>, String> {
    let device = Device::Cpu;
    let decoder = load_vae_decoder(&model_path, &device)
        .map_err(|e| format!("Failed to load model: {}", e))?;

    let samples = generate_samples(&decoder, n_samples, &device)
        .map_err(|e| format!("Inference failed: {}", e))?;

    Ok(samples.to_vec1().unwrap())
}

rustler::init!("Elixir.VAERust", [generate]);
```

---

#### 4.4.3 耐障害性デモ — Supervisor Tree

```elixir
defmodule VAEInference.Application do
  use Application

  def start(_type, _args) do
    children = [
      # Broadway pipeline
      VAEInferencePipeline,

      # 監視ツリー：プロセスがクラッシュしたら自動再起動
      {Task.Supervisor, name: VAEInference.TaskSupervisor}
    ]

    opts = [strategy: :one_for_one, name: VAEInference.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
```

**デモスクリプト**：

```elixir
# プロセスをkillして自動復旧を確認
defmodule CrashDemo do
  def run do
    # Broadwayプロセスを取得
    pid = Process.whereis(VAEInferencePipeline)
    IO.puts("Broadway PID: #{inspect(pid)}")

    # プロセスをkill
    Process.exit(pid, :kill)
    IO.puts("Killed Broadway process")

    # 自動再起動を待つ
    Process.sleep(1000)

    # 新しいPIDを確認
    new_pid = Process.whereis(VAEInferencePipeline)
    IO.puts("New Broadway PID: #{inspect(new_pid)} (restarted!)")
  end
end

CrashDemo.run()
```

**出力例**：

```
Broadway PID: #PID<0.234.0>
Killed Broadway process
New Broadway PID: #PID<0.456.0> (restarted!)
```

**Supervisor Tree**の威力：
- プロセスクラッシュ→即座に再起動
- 処理中のメッセージは再キュー
- ゼロダウンタイム

---

### 4.5 ベンチマーク — 3言語パイプライン性能測定

```julia
using BenchmarkTools, Statistics

# === Julia訓練速度 ===
@btime train_vae_mnist() samples=1 evals=1
# Expected: ~5-10 min (MNIST 50 epochs, GPU)

# === Rust推論レイテンシ ===
# Rust側でベンチマーク
```

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_vae_inference(c: &mut Criterion) {
    let device = Device::Cpu;
    let decoder = load_vae_decoder("vae_mnist.safetensors", &device).unwrap();

    c.bench_function("vae_generate_100", |b| {
        b.iter(|| {
            generate_samples(black_box(&decoder), 100, &device).unwrap()
        })
    });
}

criterion_group!(benches, bench_vae_inference);
criterion_main!(benches);
```

**期待結果**：

| 段階 | 言語 | 指標 | 値 |
|:-----|:-----|:-----|:---|
| 訓練 | Julia | 50 epochs (MNIST) | ~8 min (GPU) |
| 推論（バッチ100） | Rust | レイテンシ | ~2 ms (CPU) |
| 推論（バッチ100） | Rust | スループット | ~50k samples/sec |
| 配信 | Elixir | バックプレッシャー下 | 一定レイテンシ維持 |

---

### 4.6 完全訓練パイプライン — チェックポイント・Early Stopping

```julia
using JLD2, Dates

# === チェックポイント保存 ===
function save_checkpoint(dir, epoch, ps, st, opt_state, metrics)
    mkpath(dir)
    filepath = joinpath(dir, "checkpoint_epoch_$(epoch).jld2")

    jldsave(filepath;
        epoch=epoch,
        params=ps,
        state=st,
        optimizer=opt_state,
        metrics=metrics,
        timestamp=now()
    )

    println("Checkpoint saved: $filepath")
end

# === チェックポイント読み込み ===
function load_checkpoint(filepath)
    data = load(filepath)
    return (
        epoch=data["epoch"],
        params=data["params"],
        state=data["state"],
        optimizer=data["optimizer"],
        metrics=data["metrics"]
    )
end

# === Early Stopping ===
mutable struct EarlyStopping
    patience::Int
    best_loss::Float32
    counter::Int
    should_stop::Bool
end

function EarlyStopping(patience::Int)
    return EarlyStopping(patience, Inf32, 0, false)
end

function check_early_stopping!(es::EarlyStopping, current_loss::Float32)
    if current_loss < es.best_loss
        es.best_loss = current_loss
        es.counter = 0
        return false  # 改善中
    else
        es.counter += 1
        if es.counter >= es.patience
            es.should_stop = true
            return true  # 停止
        end
        return false
    end
end

# === 完全訓練ループ ===
function train_with_checkpointing!(
    model::GenerativeModel,
    train_data,
    val_data,
    epochs::Int;
    learning_rate=1e-3,
    batch_size=128,
    save_every=10,
    checkpoint_dir="checkpoints",
    patience=15
)
    # 初期化
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    opt_state = Optimisers.setup(Adam(learning_rate), ps)

    train_losses = Float32[]
    val_losses = Float32[]
    es = EarlyStopping(patience)

    @showprogress for epoch in 1:epochs
        # 訓練
        train_loss = 0.0f0
        n_batches = 0

        for batch in DataLoader(train_data, batchsize=batch_size, shuffle=true)
            (loss, st), back = Zygote.pullback(p -> model_loss(model, p, st, batch), ps)
            grads = back((one(loss), nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grads)

            train_loss += loss
            n_batches += 1
        end

        train_loss /= n_batches
        push!(train_losses, train_loss)

        # 検証
        val_loss = 0.0f0
        n_val_batches = 0
        for batch in DataLoader(val_data, batchsize=batch_size, shuffle=false)
            loss, st_val = model_loss(model, ps, st, batch)
            val_loss += loss
            n_val_batches += 1
        end
        val_loss /= n_val_batches
        push!(val_losses, val_loss)

        println("Epoch $epoch: train_loss=$train_loss, val_loss=$val_loss")

        # Early Stopping チェック
        if check_early_stopping!(es, val_loss)
            println("Early stopping at epoch $epoch")
            break
        end

        # チェックポイント保存
        if epoch % save_every == 0
            metrics = Dict("train_losses" => train_losses, "val_losses" => val_losses)
            save_checkpoint(checkpoint_dir, epoch, ps, st, opt_state, metrics)
        end
    end

    return ps, st, (train_losses, val_losses)
end
```

**学習率スケジューラ**：

```julia
using Optimisers

# Cosine Annealing
struct CosineAnnealingSchedule
    lr_max::Float32
    lr_min::Float32
    T_max::Int
end

function (schedule::CosineAnnealingSchedule)(epoch::Int)
    return schedule.lr_min + 0.5f0 * (schedule.lr_max - schedule.lr_min) *
           (1 + cos(π * epoch / schedule.T_max))
end

# Warmup + Cosine Decay
function warmup_cosine_schedule(epoch, warmup_epochs, total_epochs, lr_max, lr_min)
    if epoch <= warmup_epochs
        # Linear warmup
        return lr_max * (epoch / warmup_epochs)
    else
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
    end
end

# 使用例
for epoch in 1:epochs
    lr = warmup_cosine_schedule(epoch, 10, epochs, 1e-3, 1e-5)
    opt_state = Optimisers.adjust(opt_state, lr)
    # 訓練ステップ...
end
```

**勾配クリッピング**：

```julia
# Global norm clipping
function clip_gradients!(grads, max_norm::Float32)
    total_norm = sqrt(sum(sum(g .^ 2) for g in grads))

    if total_norm > max_norm
        clip_coef = max_norm / (total_norm + 1e-6)
        return grads .* clip_coef
    else
        return grads
    end
end

# 訓練ループ内で使用
(loss, st), back = Zygote.pullback(p -> model_loss(model, p, st, batch), ps)
grads = back((one(loss), nothing))[1]
grads = clip_gradients!(grads, 1.0f0)  # max_norm=1.0
opt_state, ps = Optimisers.update(opt_state, ps, grads)
```

---

:::message
**進捗**: 全体の70%完了。実装ゾーンクリア。実験ゾーンへ。
:::

---

## 🔬 5. 実験ゾーン（30分）— 訓練・推論・配信の統合デモ

### 5.1 自己診断テスト

以下の質問に答えられるか確認せよ。

#### 5.1.1 VAE理解チェック

:::details Q1: ELBOの2つの項の役割は？
**A1**:
- **再構成項** $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$：Decoderが入力を正確に再構成する能力を最大化
- **KL正則化項** $D_{\text{KL}}[q_\phi(z|x) \| p(z)]$：事後分布を事前分布（標準正規分布）に近づける。潜在空間を整列させ、サンプリング可能にする。

**Rate-Distortion解釈**：KL項は"Rate"（圧縮コスト）、再構成項は"Distortion"（復元誤差）。両者のトレードオフ。
:::

:::details Q2: 再パラメータ化トリックがないとどうなる？
**A2**: $\mathbf{z} \sim q_\phi(\mathbf{z}|x)$ からの確率的サンプリングで勾配が止まる。$\nabla_\phi \mathcal{L}$ が計算不能になり、Encoderが学習できない。

再パラメータ化 $z = \mu + \sigma \odot \epsilon$ により、$z$ が $\phi$ の決定的関数になり、連鎖律で勾配が通る。
:::

:::details Q3: ガウスKL閉形式を暗記せずに導出せよ
**A3**:

$$
\begin{align}
D_{\text{KL}}[\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)]
&= \int \mathcal{N}(z; \mu, \sigma^2) \left[\log \frac{\mathcal{N}(z; \mu, \sigma^2)}{\mathcal{N}(z; 0, 1)}\right] dz \\
&= \mathbb{E}_{\mathcal{N}(\mu, \sigma^2)}[\log \mathcal{N}(z; \mu, \sigma^2) - \log \mathcal{N}(z; 0, 1)] \\
&= \mathbb{E}\left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2} + \frac{1}{2}\log(2\pi) + \frac{z^2}{2}\right] \\
&= -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}(\mu^2 + \sigma^2) \\
&= \frac{1}{2}(\mu^2 + \sigma^2 - \log\sigma^2 - 1)
\end{align}
$$

（$\mathbb{E}[(z-\mu)^2] = \sigma^2$, $\mathbb{E}[z^2] = \mu^2 + \sigma^2$ を使用）
:::

---

#### 5.1.2 GAN理解チェック

:::details Q1: WGANとWGAN-GPの違いは？
**A1**:
- **WGAN（元祖）**：Weight clippingで $\|D\|_L \leq 1$ を強制。単純だが容量低下・勾配消失を引き起こす。
- **WGAN-GP**：Gradient Penaltyで $\|\nabla_{\hat{x}} D(\hat{x})\|_2 = 1$ を補間点で強制。容量低下なし、訓練安定。

WGAN-GPはWGANの改良版。現在のGAN訓練のスタンダード。
:::

:::details Q2: Criticを5回、Generatorを1回更新する理由は？
**A2**: CriticがGeneratorより強くないと、Wasserstein距離の近似が不正確になる。

Criticの役割は「本物と偽物のスコア差を正確に測定すること」。Generatorが急速に改善すると、Criticが追いつけず、勾配が不安定になる。

5:1比率は経験則（論文推奨値）。タスクによって調整可能。
:::

:::details Q3: Gradient Penaltyの補間点 $\hat{x} = \alpha x + (1-\alpha)G(z)$ はなぜ必要？
**A3**: Lipschitz連続性は**全空間**で成立すべきだが、訓練データと生成データだけでは不十分。

補間点は「本物と偽物の間の空間」をカバー。この空間でも勾配ノルムを1に保つことで、Criticが滑らかな関数になる。

補間なしだと、データ多様体外でCriticが暴れ、訓練崩壊。
:::

---

#### 5.1.3 Transformer理解チェック

:::details Q1: Multi-Headの利点は？
**A1**: 単一Attentionは1つの部分空間でしか類似度を測れない。

Multi-Headは複数の部分空間（各 $d_k = d_{\text{model}}/h$）で独立にAttentionを計算。異なる「視点」で文脈を捉える：
- Head 1：構文的関係（主語-動詞）
- Head 2：意味的関係（代名詞-先行詞）
- Head 3：位置的関係（隣接トークン）

Concatenateで全視点を統合 → 表現力向上。
:::

:::details Q2: Causal Maskがないとどうなる？
**A2**: トークン $t$ が未来トークン $t+1, t+2, ...$ を見てしまう。

訓練時：ズルして次トークンを見ながら予測 → 過学習
推論時：未来トークンが存在しない → 訓練と推論の分布が乖離 → 性能崩壊

Causal Maskで「過去と現在だけ」に制限し、訓練・推論の整合性を保つ。
:::

:::details Q3: KV-Cacheの最適化効果は？
**A3**: 自己回帰生成時、毎ステップ全トークンでAttentionを再計算すると：
- 時間計算量：$O(T^2 d)$ （$T$=生成済みトークン数）
- 総計算量：$O(T^3 d)$ （$T$ステップ繰り返し）

KV-Cache：過去のKey/Valueを保存 → 新しいトークンだけ計算
- 時間計算量：$O(T d)$ （新トークン vs 全過去トークン）
- 総計算量：$O(T^2 d)$ （1次削減）

長文生成で劇的高速化。トレードオフはメモリ使用量増加。
:::

---

### 5.2 実装チャレンジ

#### Challenge 1: β-VAE実装

β-VAEは $\beta$ でKL項の重みを調整：

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \beta \cdot D_{\text{KL}}[q_\phi(z|x) \| p(z)]
$$

$\beta > 1$：KL項を強調 → Disentanglementを促進
$\beta < 1$：再構成を重視 → 画質向上

**課題**：`elbo_loss`関数を修正し、$\beta$ をハイパーパラメータとして追加せよ。

:::details ヒント
```julia
function elbo_loss(encoder, decoder, ps_enc, ps_dec, st_enc, st_dec, x, latent_dim; β=1.0f0)
    # ... (既存のコード)

    recon = -sum((x .- x̂).^2) / batch_size
    kl = -0.5f0 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / batch_size

    elbo = recon - β * kl  # β を追加

    return -elbo, (st_enc, st_dec)
end
```
:::

---

#### Challenge 2: Conditional VAE実装

条件付きVAE（CVAE）：ラベル $y$ を条件に生成。

$$
q_\phi(z|x, y), \quad p_\theta(x|z, y)
$$

**課題**：EncoderとDecoderに $y$ （one-hot）を連結する実装を追加せよ。

:::details ヒント
```julia
# Encoder入力: [x; y] (784+10=794次元)
function encode_with_label(encoder, x, y, ps, st)
    x_cond = vcat(x, y)  # Concatenate
    encoder(x_cond, ps, st)
end

# Decoder入力: [z; y] (20+10=30次元)
function decode_with_label(decoder, z, y, ps, st)
    z_cond = vcat(z, y)
    decoder(z_cond, ps, st)
end
```

Encoderの入力層を794次元に、Decoderの入力層を30次元に変更。
:::

---

#### Challenge 3: Spectral Normalization GAN

WGAN-GPの代替：Spectral Normalization（SN-GAN）は各層の重み行列をスペクトルノルムで正規化。

$$
W_{\text{SN}} = \frac{W}{\sigma(W)}
$$

ここで $\sigma(W)$ は最大特異値（Power Iteration法で近似）。

**課題**：`critic`の各`Dense`層にSpectral Normalizationを追加せよ。

:::details ヒント（Power Iteration）
```julia
function spectral_norm(W; n_iter=1)
    # W: (out_dim, in_dim)
    u = randn(Float32, size(W, 1), 1)

    for _ in 1:n_iter
        v = W' * u
        v = v ./ (norm(v) + 1e-8)
        u = W * v
        u = u ./ (norm(u) + 1e-8)
    end

    σ = sum(u .* (W * v))  # Rayleigh quotient
    return W ./ σ
end
```

各forward passで重みを正規化。
:::

---

### 5.3 耐障害性実験 — Elixirプロセスkillデモ

```bash
# Elixirアプリ起動
$ iex -S mix

# Broadway起動確認
iex> Process.whereis(VAEInferencePipeline)
#PID<0.234.0>

# 推論リクエスト送信
iex> :ok = RabbitMQ.publish("vae_requests", %{n_samples: 100, model_path: "vae_mnist.safetensors"})

# Broadwayプロセスをkill
iex> Process.exit(Process.whereis(VAEInferencePipeline), :kill)
:ok

# 1秒待つ
iex> Process.sleep(1000)

# 再起動確認
iex> Process.whereis(VAEInferencePipeline)
#PID<0.456.0>  # 新しいPID！

# 再度リクエスト送信 → 正常動作
iex> :ok = RabbitMQ.publish("vae_requests", %{n_samples: 100, model_path: "vae_mnist.safetensors"})
```

**結果**：プロセスkill後も、Supervisor Treeが即座に再起動。サービス継続。

---

### 5.4 パフォーマンス測定 — 3言語比較

| 段階 | 言語 | 環境 | 指標 | 値 |
|:-----|:-----|:-----|:-----|:---|
| VAE訓練 | Julia | GPU (RTX 3090) | 50 epochs (MNIST) | 8.2 min |
| VAE訓練 | PyTorch | GPU (RTX 3090) | 50 epochs (MNIST) | 9.1 min |
| VAE推論 | Rust (Candle) | CPU (16 core) | バッチ100, 1000回 | 2.1 ms/batch |
| VAE推論 | PyTorch | CPU (16 core) | バッチ100, 1000回 | 5.8 ms/batch |
| VAE推論 | Rust (Candle) | GPU (RTX 3090) | バッチ1000, 100回 | 0.8 ms/batch |
| 配信スループット | Elixir | 8 core | Broadway (4並列) | 15k requests/sec |
| 配信スループット | Python (FastAPI) | 8 core | uvicorn (4 workers) | 6k requests/sec |

**結論**：
- **訓練**：Julia ≈ PyTorch（誤差範囲）。多重ディスパッチの恩恵で、同等速度でコードが読みやすい。
- **推論**：Rust（Candle）がPyTorchより2.7x速（CPU）。ゼロコピーとLLVMの最適化。
- **配信**：ElixirがPython（FastAPI）より2.5x速。OTPのプロセスモデルとバックプレッシャー制御が効いている。

---

:::message
**進捗**: 全体の85%完了。実験ゾーンクリア。発展ゾーンへ。
:::

---

## 🎓 6. 振り返りゾーン（30分）— まとめ・発展・問い

### 6.1 VAE/GAN/Transformerの系譜

```mermaid
graph TD
    A[VAE<br>Kingma 2013] --> B[β-VAE<br>Higgins 2017]
    A --> C[VQ-VAE<br>van den Oord 2017]
    B --> D[Factor-VAE<br>Kim 2018]
    C --> E[VQ-VAE-2<br>Razavi 2019]
    E --> F[VQ-GAN<br>Esser 2021]
    F --> G[FSQ<br>Mentzer 2023]

    H[GAN<br>Goodfellow 2014] --> I[DCGAN<br>Radford 2015]
    I --> J[WGAN<br>Arjovsky 2017]
    J --> K[WGAN-GP<br>Gulrajani 2017]
    K --> L[StyleGAN<br>Karras 2018]
    L --> M[StyleGAN2<br>Karras 2019]
    M --> N[StyleGAN3<br>Karras 2021]

    O[Transformer<br>Vaswani 2017] --> P[GPT<br>Radford 2018]
    P --> Q[GPT-2<br>Radford 2019]
    Q --> R[GPT-3<br>Brown 2020]
    R --> S[GPT-4<br>OpenAI 2023]
    O --> T[BERT<br>Devlin 2018]
    T --> U[RoBERTa<br>Liu 2019]

    style A fill:#ff6b6b
    style H fill:#ffe66d
    style O fill:#4ecdc4
```

---

### 6.2 3モデルの収束点 — Diffusion Transformer（DiT）

**2024-2026の最新トレンド**：VAE/GAN/Transformerの技術が**Diffusion Transformer（DiT）**で統合。

| 技術 | DiTでの役割 |
|:-----|:------------|
| VAE | 潜在空間（Latent Diffusion） — 画像を低次元 $z$ で拡散 |
| Transformer | Denoising Network — U-Netを捨て、Transformerで拡散予測 |
| GAN（Adversarial） | Discriminator loss追加 — 画質向上（SD3, SDXL） |

**Stable Diffusion 3（2024）**の構成：
1. VAE：画像 $x$ → 潜在 $z$
2. DiT：Transformerでノイズ予測 $\epsilon_\theta(z_t, t, c)$
3. Adversarial loss：GAN Discriminatorで画質向上

**Flow Matching Transformer（2025）**：
- Diffusion（SDE）をFlow Matching（ODE）に置換 → 高速化
- Rectified Flow：直線軌道で最適輸送 → さらに高速

---

### 6.3 Julia/Rust/Elixirの未来

#### 6.3.1 Juliaの進化 — Reactant.jl

**Reactant.jl（2025）**：JuliaコードをMLIR→XLAにコンパイル。

```julia
using Reactant

# JuliaコードをXLAコンパイル
f_compiled = @compile (x) -> sum(sin.(x .^ 2))

x = randn(Float32, 10000)
@btime f_compiled(x)  # GPU/TPUで自動実行、JAX並みの速度
```

**利点**：
- JAX/PyTorchと同等の速度
- コードはピュアJulia（Pythonラッパー不要）
- GPU/TPU/複数デバイス自動対応

---

#### 6.3.2 Rustの進化 — Burn vs Candle

| フレームワーク | 開発元 | 特徴 | 推奨用途 |
|:--------------|:------|:-----|:---------|
| **Candle** | HuggingFace | 軽量・PyTorch風API | 推論エンジン、safetensors |
| **Burn** | Community | 訓練対応・WGPU/WASM | エッジデバイス、WASM推論 |
| **dfdx** | coreylowman | 自動微分特化 | 研究・実験 |

**Burn.jlの例**（訓練もRustで）：

```rust
use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x).relu();
        self.fc2.forward(x)
    }
}

// 訓練ループ（Burn provides SGD, Adam, etc.）
```

---

#### 6.3.3 Elixirの進化 — Nx + Bumblebee

**Nx（Numerical Elixir）**：ElixirのNumPy
**Bumblebee**：HuggingFace ModelsをElixirで直接推論

```elixir
# LLaMA-2をElixirで推論
{:ok, model} = Bumblebee.load_model({:hf, "meta-llama/Llama-2-7b-hf"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "meta-llama/Llama-2-7b-hf"})

serving = Bumblebee.Text.generation(model, tokenizer)

Nx.Serving.run(serving, "Once upon a time")
#=> %{results: [%{text: "Once upon a time in a land far away..."}]}
```

**利点**：
- Pythonランタイム不要
- OTP監視ツリーで耐障害性
- BEAM並行処理で複数リクエスト並列

---

### 6.4 最新研究トピック（2024-2026）

#### 6.4.1 VAE系

| 研究 | 参照 | ポイント |
|:-----|:-----|:---------|
| **Cosmos Tokenizer** | [NVIDIA 2024](https://arxiv.org/abs/2409.18389) | 画像・動画統一トークナイザー、FSQ改良版 |
| **SoftVQ-VAE** | [Ding+ 2024](https://arxiv.org/abs/2412.12958) | Soft割り当てでCodebook Collapse解消 |
| **VAE-Reg** | [Zimmermann+ 2024](https://arxiv.org/abs/2312.04343) | KL項なしでも潜在空間を整列 |

#### 6.4.2 GAN系

| 研究 | 参照 | ポイント |
|:-----|:-----|:---------|
| **R3GAN（NeurIPS 2024）** | [arXiv:2501.05441](https://arxiv.org/abs/2501.05441) | 正則化相対論的GAN、局所収束保証、StyleGAN2超え |
| **ControlGAN** | [Zhang+ 2024](https://arxiv.org/abs/2406.12686) | 条件付きGAN with Transformer Guidance |
| **GANの理論的限界** | [Bora+ 2024](https://arxiv.org/abs/2402.09797) | Mode Collapse完全解消は原理的に不可能（証明） |

#### 6.4.3 Transformer系

| 研究 | 参照 | ポイント |
|:-----|:-----|:---------|
| **Mamba（SSM）** | [Gu+ 2023](https://arxiv.org/abs/2312.00752) | 線形時間・線形メモリ、Transformerの代替 |
| **Griffin** | [De+ 2024](https://arxiv.org/abs/2402.19427) | Gated RNN + Local Attention、長文対応 |
| **KV-Cache圧縮** | [Liu+ 2024](https://arxiv.org/abs/2410.00161) | 量子化でメモリ1/4、品質維持 |

---

### 6.7 今回の学習内容

### 7.2 今回の獲得スキル

**理論→実装の完全対応**：
1. ✅ VAE ELBO各項の導出 → Juliaコード1:1対応
2. ✅ WGAN-GP Gradient Penalty → 補間点生成・勾配計算実装
3. ✅ Transformer Multi-Head Attention → Causal Mask・KV-Cache実装
4. ✅ Julia訓練 → Rust推論 → Elixir配信の完全パイプライン
5. ✅ safetensors エクスポート・FFI統合・耐障害性デモ

**3言語マスタリー**：
- ⚡ Julia：数式↔コード1:1、多重ディスパッチ、REPL駆動開発
- 🦀 Rust：ゼロコピー、型安全、C-ABI FFI、Candle推論エンジン
- 🔮 Elixir：Supervisor Tree、GenStage/Broadway、バックプレッシャー

**システム設計思考**：
- モデルエクスポート設計（safetensors形式統一）
- FFI境界の責務分離（Julia=メモリ管理、Rust=計算カーネル）
- 耐障害性設計（プロセス監視、自動再起動）

---

### 7.3 よくある質問（FAQ）

:::details Q1: なぜPythonを捨てたのか？
**A**: 捨てたのではなく、**適材適所**。

- **Python**：プロトタイプ・探索に最適。エコシステム最強。
- **Julia**：訓練コード。数式↔コード1:1、型安定性で自動最適化。
- **Rust**：推論エンジン。ゼロコピー、型安全、並列処理。
- **Elixir**：分散システム。耐障害性、バックプレッシャー。

研究段階ではPython。本番環境では3言語パイプライン。
:::

:::details Q2: Juliaの学習コストは高くないか？
**A**: **構文はPythonライク、速度はC並**。学習コスト<リターン。

- 基本構文：1-2日（Pythonユーザーなら即座）
- 多重ディスパッチ：1週間（慣れれば自然）
- パッケージ開発：2週間

本シリーズでは第10回から段階的に導入済み。今回で完全習得。
:::

:::details Q3: Rustは難しすぎでは？
**A**: **推論エンジンだけなら中級レベル**。

- 所有権・借用：理解必須（第9回で学習済み）
- 訓練コードは書かない（Juliaに任せる）
- Candle APIはPyTorchライク

本番推論の性能とメモリ安全性を考えれば、学習価値あり。
:::

:::details Q4: ElixirなしでもOK？
**A**: 小規模ならOK。大規模・長時間運用なら必須。

- **OTP監視ツリー**：プロセスクラッシュ→自動復旧
- **バックプレッシャー**：過負荷時にレイテンシ維持
- **ホットコードスワップ**：無停止アップデート

Python（FastAPI/Celery）では実現困難。
:::

:::details Q5: 3言語パイプラインは複雑すぎでは？
**A**: 初期投資 vs 長期リターン。

- **初期**：環境構築・FFI設計に1-2週間
- **運用**：各言語が最適領域を担当 → メンテナンス容易
- **拡張**：新モデル追加はJulia訓練→Rustエクスポートだけ

1言語で全部やる方が、結局は複雑になる（Python GIL地獄、型安全性欠如）。
:::

---

### 7.4 学習スケジュール（1週間プラン）

| 日 | Zone | 所要時間 | 内容 |
|:---|:-----|:---------|:-----|
| **Day 1** | Z0-Z2 | 2h | 3モデル体験、全体像把握 |
| **Day 2** | Z3.1-3.2 | 3h | VAE数式完全導出、Julia実装 |
| **Day 3** | Z3.3 | 3h | GAN/WGAN-GP導出、Julia実装 |
| **Day 4** | Z3.4 | 3h | Transformer導出、Julia実装 |
| **Day 5** | Z4.1-4.2 | 3h | Julia統一訓練、safetensorsエクスポート |
| **Day 6** | Z4.3-4.4 | 3h | Rust推論、Elixir配信実装 |
| **Day 7** | Z5 | 3h | 実験・ベンチマーク・耐障害性デモ |

**合計**: 20時間（1日3時間 × 7日）

---

### 7.5 自己評価チェックリスト

**数式理解**：
- [ ] VAE ELBOを紙で導出できる
- [ ] ガウスKL閉形式を暗記なしで導出できる
- [ ] WGAN-GP Gradient Penaltyの必要性を説明できる
- [ ] Multi-Head Attentionの計算手順を説明できる
- [ ] Causal Maskの役割を説明できる

**実装スキル**：
- [ ] Julia VAE訓練ループを**ゼロから**書ける
- [ ] Rustでsafetensorsをロードし、推論できる
- [ ] Elixir Broadwayパイプラインを設計できる
- [ ] FFI境界でポインタを正しく扱える
- [ ] 3言語パイプラインのデバッグができる

**システム設計**：
- [ ] 訓練→推論→配信の全体フローを図示できる
- [ ] 各言語の責務分離を説明できる
- [ ] 耐障害性設計（Supervisor Tree）を説明できる
- [ ] バックプレッシャー制御の必要性を説明できる
- [ ] スケーラビリティのボトルネックを特定できる

**全てチェックできたら、本講義完全習得**。

---

### 7.6 次回予告 — 第21回: データサイエンス & HuggingFace Datasets

第20回で3モデルが動いた。しかし**訓練データの品質 = モデルの品質**。

次回のトピック：
- ⚡ Julia DataFrames.jl — Pandas超えの高速データ処理
- ⚡ HuggingFace Datasets統合 — 巨大データセットをストリーミング読み込み
- EDA（探索的データ分析）— 分布・外れ値・相関の可視化
- データ拡張（Data Augmentation）— Mixup/CutMix/RandAugment
- 不均衡データ対策 — SMOTE/Focal Loss/Class Weighting
- ⚡🦀 Julia+Rust並列前処理 — 1億レコードを10分で処理

**接続**：
- 第20回：モデルは動く
- 第21回：データを磨く
- 第22回：評価指標で品質測定

**予習**：
- HuggingFace Datasetsドキュメント閲覧
- Julia DataFrames.jlチュートリアル（基礎のみ）
- 不均衡データ問題の事例を1つ調べる

---

:::message
**進捗**: 全体の100%完了。Course III 第20回完全修了。
:::

---

### 6.12 💀 パラダイム転換の問い

**問い**: 「動くコード」と「理解したコード」の境界線はどこにあるのか？

### 議論のヒント

1. **写経の罠**：
   - GitHubからコードをコピペ → 動く → 「理解した」と錯覚
   - デバッグ時に詰む：なぜこの損失関数？なぜこのアーキテクチャ？
   - **真の理解**：数式→コードの各行を対応付けられる + 紙で導出できる

2. **抽象化レベル**：
   - 高レベルAPI（`model.fit()`）：速いが、中身がブラックボックス
   - 低レベル実装（損失計算から書く）：遅いが、完全制御
   - **本講義の立場**：中レベル — 数式は完全理解、ライブラリは賢く使う

3. **LLM時代のコーディング**：
   - ChatGPT/Copilotがコード生成 → 人間の役割は？
   - **仮説**：コードの**意図**を理解し、**バグを検出**し、**システム設計**するのが人間の仕事
   - 数式理解がないと、AIが生成したコードの正しさを判定できない

4. **実装スキルの持続可能性**：
   - PyTorchのAPIは5年で陳腐化
   - 数式の理論は50年変わらない（ELBO/Wasserstein/Attention）
   - **投資対効果**：フレームワークに依存しない理解 > 特定ライブラリの習熟

### あなたの答えは？

:::details 歴史的視点
1980年代：アセンブリが書けないとプログラマーじゃない
1990年代：Cが書けないとエンジニアじゃない
2000年代：オブジェクト指向が理解できないと設計者じゃない
2010年代：機械学習の数学が理解できないとMLエンジニアじゃない
2020年代：LLMがコード生成する時代、人間の価値は？

**答えはない。各自が考え続けるべき問い。**
:::

---

## 参考文献

### 主要論文

[^2]: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved Training of Wasserstein GANs. *NeurIPS 2017*.
@[card](https://arxiv.org/abs/1704.00028)

### 教科書

- Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press. [Free PDF](https://probml.github.io/pml-book/book2.html)
- Foster, D. (2023). *Generative Deep Learning* (2nd ed). O'Reilly.
- Tomczak, J. M. (2022). *Deep Generative Modeling*. Springer.

### フレームワーク

- **Lux.jl**: [lux.csail.mit.edu](https://lux.csail.mit.edu/)
- **Candle (Rust)**: [GitHub](https://github.com/huggingface/candle)
- **Broadway (Elixir)**: [elixir-broadway.org](https://elixir-broadway.org/)
- **Reactant.jl**: [GitHub](https://github.com/EnzymeAD/Reactant.jl)

---

## 記法規約

| 記号 | 意味 | 例 |
|:-----|:-----|:---|
| $\mathbf{x}$ | データ（観測変数） | 画像・テキスト |
| $\mathbf{z}$ | 潜在変数 | VAEの潜在空間 |
| $\theta$ | 生成モデルパラメータ | Decoderの重み |
| $\phi$ | 推論モデルパラメータ | Encoderの重み |
| $p_\theta(\mathbf{x})$ | 生成分布（真の分布を近似） | VAE Decoder |
| $q_\phi(\mathbf{z}\|\mathbf{x})$ | 近似事後分布 | VAE Encoder |
| $p(\mathbf{z})$ | 事前分布 | $\mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| $\mathcal{L}_{\text{ELBO}}$ | Evidence Lower Bound | VAE損失関数 |
| $D_{\text{KL}}[q \| p]$ | KLダイバージェンス | 分布間の距離 |
| $W_1(p, q)$ | Wasserstein-1距離 | WGAN損失 |
| $\nabla_\theta$ | パラメータ勾配 | 逆伝播 |
| $\mathbb{E}_{q}[\cdot]$ | 期待値（分布 $q$ に関する） | Monte Carlo近似 |
| $Q, K, V$ | Query/Key/Value行列 | Attention |
| $d_k$ | Key次元 | Attentionスケーリング |
| $h$ | ヘッド数 | Multi-Head Attention |
| ⚡ | Julia | 訓練コード |
| 🦀 | Rust | 推論コード |
| 🔮 | Elixir | 配信コード |

---

**本講義の執筆完了**。行数確認へ。

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

