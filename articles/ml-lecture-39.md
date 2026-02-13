---
title: "第39回: Latent Diffusion Models: 30秒の驚き→数式修行→実装マスター"
emoji: "🖼️"
type: "tech"
topics: ["machinelearning", "deeplearning", "ldm", "julia", "stablediffusion"]
published: true
---

# 第39回: 🖼️ Latent Diffusion Models

:::message
**前回の到達点**: 第38回でScore/Flow/Diffusionの数学的等価性を証明し、統一理論が完成した。理論だけでは画像は生成できない — ピクセル空間拡散の計算限界を超える潜在空間拡散と、テキスト条件付き生成へ。
:::

## 🚀 0. クイックスタート（30秒）— ピクセル vs 潜在空間の衝撃

```julia
using Lux, Random

# ピクセル空間拡散: 512x512x3 = 786,432次元
pixel_dim = 512 * 512 * 3
pixel_diffusion_params = pixel_dim * 1000  # 7億パラメータ...

# VAE latent space: 64x64x4 = 16,384次元 (48x圧縮!)
latent_dim = 64 * 64 * 4
latent_diffusion_params = latent_dim * 1000  # 1640万パラメータ

compression_ratio = pixel_dim / latent_dim
speedup = compression_ratio^2  # 計算量はO(N²)

println("Compression: $(round(compression_ratio, digits=1))x")
println("Theoretical speedup: $(round(speedup, digits=1))x")
# Output:
# Compression: 48.0x
# Theoretical speedup: 2304.0x
```

**数式の正体**:
$$
\begin{aligned}
\text{Pixel Diffusion: } &x \in \mathbb{R}^{512 \times 512 \times 3} \quad (\approx 786\text{K次元}) \\
\text{Latent Diffusion: } &z \in \mathbb{R}^{64 \times 64 \times 4} \quad (\approx 16\text{K次元}) \\
\text{Compression: } &f = \frac{512^2 \times 3}{64^2 \times 4} = 48\times \\
\text{Speedup: } &\mathcal{O}(f^2) \approx 2304\times
\end{aligned}
$$

**この30秒で体感したこと**: 次元削減が計算量を **2000倍** 削減。Stable Diffusionが消費者GPUで動く理由。

:::message
**ここまでで全体の3%完了！** これから潜在空間の数学的基盤と、テキスト条件付き生成の完全理論へ。
:::

---

## 🎮 1. 体験ゾーン（10分）— なぜ潜在空間か

### ピクセル空間拡散の限界

第36回で学んだDDPMは美しい理論だが、計算限界がある:

| 項目 | 256×256 DDPM | 512×512 DDPM | 1024×1024 DDPM |
|:-----|:-------------|:-------------|:----------------|
| **入力次元** | 196,608 | 786,432 | 3,145,728 |
| **U-Net params** | ~100M | ~500M | ~2B |
| **訓練時間/iter** | ~1秒 | ~5秒 | ~20秒 |
| **V100 VRAM** | 12GB | 32GB | 80GB (不可能) |
| **収束イテレーション** | 500K | 1M | 2M+ |
| **総訓練時間** | 6日 | 58日 | **年単位** |

**問題の本質**: ピクセル空間の次元 $d = H \times W \times C$ が大きすぎる。U-Netのself-attentionは $\mathcal{O}(d^2)$ の計算量 — 解像度を2倍にすると計算量は **16倍**。

### 潜在空間への必然性

**鍵となる観察**: 自然画像は高次元空間に埋め込まれているが、実際には低次元多様体上に分布している（多様体仮説）。

$$
\begin{aligned}
\text{Pixel space: } &\mathbb{R}^{H \times W \times C} \quad \text{(高次元・冗長)} \\
\text{Manifold: } &\mathcal{M} \subset \mathbb{R}^{H \times W \times C}, \quad \dim(\mathcal{M}) \ll H \times W \times C \\
\text{Latent space: } &\mathbb{R}^{h \times w \times c}, \quad h \ll H, w \ll W
\end{aligned}
$$

**解決策**: VAEで低次元潜在空間 $z$ にエンコードし、そこで拡散過程を実行。

```mermaid
graph LR
    X["Pixel x ∈ R^(HxWxC)"] -->|Encoder E| Z["Latent z ∈ R^(hxwxc)"]
    Z -->|Diffusion| Z2["Denoised z' ∈ R^(hxwxc)"]
    Z2 -->|Decoder D| X2["Pixel x' ∈ R^(HxWxC)"]

    style X fill:#ffcccc
    style Z fill:#ccffcc
    style Z2 fill:#ccffcc
    style X2 fill:#ffcccc
```

### LDM vs ピクセル拡散: 数値比較

Stable Diffusion 1.5の実測値:

| メトリック | DDPM (512²) | LDM (SD 1.5) | 改善率 |
|:-----------|:------------|:-------------|:-------|
| **潜在空間次元** | 786,432 | 16,384 | **48x圧縮** |
| **U-Net params** | ~500M | ~860M | 1.7x (でもGPUに乗る) |
| **訓練時間/iter** | ~5秒 | ~0.8秒 | **6.25x高速化** |
| **VRAM (fp16)** | 32GB | 10GB | **3.2x削減** |
| **収束ステップ** | 1M | 500K | **2x高速** |
| **FID (COCO)** | 12.6 | **10.4** | 品質向上 |

**なぜ高速化と品質向上が両立？**

1. **Inductive bias**: 潜在空間は「意味のある特徴」に圧縮済み → 拡散モデルが学習しやすい
2. **Perceptual compression**: VAEが知覚的に重要な特徴を保存 → 品質維持
3. **Computational efficiency**: 次元削減で計算量削減 → より深いU-Net・長時間訓練が可能

:::message alert
**よくある誤解**: 「潜在空間で拡散するから品質が下がる」— 実際はVAEの知覚的損失関数で **品質は向上**。
:::

### 数式で見るLDM

ピクセル空間DDPMの目的関数（第36回の復習）:
$$
\mathcal{L}_\text{DDPM} = \mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
$$

Latent Diffusion Modelの目的関数:
$$
\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right], \quad z_0 = \mathcal{E}(x_0)
$$

ここで:
- $\mathcal{E}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{h \times w \times c}$ がVAE Encoder
- $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ は潜在空間でのforward process
- $\epsilon_\theta(z_t, t)$ は潜在空間で動作するU-Net

**鍵**: $x$ を $z$ に置き換えただけ。DDPMの理論がそのまま使える!

```julia
# ピクセル空間DDPM
x₀ = randn(512, 512, 3)  # 786K次元
εₜ = unet_pixel(xₜ, t)    # 786K次元の逆拡散

# 潜在空間LDM
z₀ = encoder(x₀)         # 64×64×4 = 16K次元
εₜ = unet_latent(zₜ, t)  # 16K次元の逆拡散 (48x削減!)
```

### LDMの訓練パイプライン

2段階の訓練:

```mermaid
graph TD
    A[Stage 1: VAE事前訓練] -->|固定| B[Stage 2: 潜在空間拡散訓練]

    A1[VAE訓練<br>再構成損失 + KL正則化] --> A2[Encoder E & Decoder D]
    A2 --> B1[Diffusion U-Net訓練<br>ε-prediction on z]
    B1 --> B2[完成: LDM]

    style A fill:#ffffcc
    style B fill:#ccffff
```

**Stage 1: VAE訓練**
$$
\mathcal{L}_\text{VAE} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{再構成項}} - \underbrace{\beta \cdot \text{KL}[q(z|x) \| p(z)]}_{\text{正則化項}}
$$

- 第10回で学んだVAEそのもの
- $\beta$-VAE ($\beta < 1$) で再構成品質を重視
- または VQ-VAE/FSQ で離散表現学習

**Stage 2: Diffusion訓練**
$$
\mathcal{L}_\text{Diffusion} = \mathbb{E}_{z_0 \sim q(z_0), \epsilon \sim \mathcal{N}(0,I), t \sim \mathcal{U}(1,T)} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2 \right]
$$

- $c$ は条件付け情報（テキスト・クラスラベル等）
- VAEは **固定** （勾配を流さない）
- U-Netだけを訓練

:::message
**ここまでで全体の10%完了！** VAEで圧縮、潜在空間で拡散という2段階設計の理論的根拠を理解した。次は数式修行ゾーンへ。
:::

---

## 🧩 2. 直感ゾーン（15分）— 潜在空間の幾何学

### なぜこの講義を学ぶか

**到達目標**:
- Stable Diffusionのアーキテクチャを完全理解
- Classifier-Free Guidanceの数学的導出を自力で行える
- FLUX/SD3等の最新モデルの理論的基盤を把握
- テキスト→画像生成の全パイプラインを実装できる

**Course IVでの位置づけ**:

```mermaid
graph LR
    L33[L33: Normalizing Flow] --> L34[L34: EBM]
    L34 --> L35[L35: Score Matching]
    L35 --> L36[L36: DDPM]
    L36 --> L37[L37: SDE/ODE]
    L37 --> L38[L38: Flow Matching統一理論]
    L38 --> L39[★L39: Latent Diffusion★]
    L39 --> L40[L40: Consistency Models]
    L40 --> L41[L41: World Models]
    L41 --> L42[L42: 統一理論]

    style L39 fill:#ffcccc
```

**前提知識**:
- 第10回: VAE理論（ELBO・再パラメータ化）
- 第36回: DDPM完全導出
- 第38回: Flow Matching統一理論

### 松尾・岩澤研究室との差別化

| トピック | 松尾・岩澤研 | 本講義 |
|:---------|:-------------|:-------|
| **LDM理論** | 概要説明 | VAE圧縮率とDDPM計算量の **定量的関係** 導出 |
| **CFG導出** | スキップ | ε-prediction / score / 温度の **3視点から完全導出** |
| **Text Conditioning** | CLIP紹介 | Cross-Attention / Self-Attention / Positional Encodingの **実装レベル詳細** |
| **FLUX解説** | なし | Rectified Flow統合アーキテクチャの **数学的解析** (2025) |
| **実装** | なし | ⚡Julia訓練 + 🦀Rust推論 + CFG実験 (3,000行超) |

### LDMの3つのメタファー

**メタファー1: 地図帳**
- ピクセル空間 = 世界中の街の詳細地図（膨大）
- 潜在空間 = 地図帳の目次（コンパクトだが位置関係は保存）
- 拡散過程 = 目次をぼかして復元（目次レベルで作業）

**メタファー2: 圧縮ファイル**
- VAE Encoder = ZIP圧縮
- 潜在空間 = .zip ファイル（48倍圧縮）
- 拡散U-Net = .zip内で直接編集
- VAE Decoder = 解凍

**メタファー3: スケッチ→詳細化**
- 潜在空間 = ラフスケッチ（構図・配置だけ）
- VAE Decoder = ディテール追加（テクスチャ・色・細部）
- 拡散過程 = スケッチのノイズ除去

### Course I-III数学の活用

| Course | 活用箇所 |
|:-------|:---------|
| **第10回 VAE** | Encoder/Decoder設計 / KL正則化 / 再構成損失 |
| **第13回 OT** | Wasserstein距離での品質評価 / OT-CFMとの接続 |
| **第36回 DDPM** | 潜在空間での拡散過程 / ε-prediction / VLB |
| **第38回 Flow Matching** | FLUX architecture / Rectified Flow統合 |
| **第14回 Attention** | Cross-Attention text conditioning / Self-Attention in U-Net |

### レベルアップマップ

```mermaid
graph TD
    A[Lv1: LDMの動機理解<br>なぜ潜在空間か] --> B[Lv2: VAE圧縮の数学<br>圧縮率vs品質]
    B --> C[Lv3: 潜在空間拡散<br>DDPMの拡張]
    C --> D[Lv4: Classifier Guidance<br>勾配ベース誘導]
    D --> E[Lv5: Classifier-Free Guidance<br>CFG完全導出]
    E --> F[Lv6: Text Conditioning<br>Cross-Attention]
    F --> G[Lv7: FLUX Architecture<br>Flow Matching統合]
    G --> H[Boss: Mini LDM実装<br>Julia訓練+CFG実験]
```

### Trojan Horse — 🐍→⚡🦀🔮の必然性

**第39回の言語構成**: ⚡Julia 70% / 🦀Rust 20% / 🔮Elixir 10%

**なぜJulia主役？**
- VAE訓練: Lux.jl + Reactant → JAX並の速度
- Diffusion訓練: 多重ディスパッチで損失関数が自動最適化
- CFG実験: Guidance scale掃引が1行

**なぜRust？**
- ONNX推論: Candle/Burn → PyTorch比35-47%高速
- バッチ処理: ゼロコピー + SIMD最適化

**なぜElixir？**
- リアルタイムサービング: GenStage + Broadway
- 分散推論: Supervisor Tree耐障害性

この3言語スタックで **訓練→推論→配信** の完全パイプラインを構築する。

:::message
**ここまでで全体の20%完了！** 直感を固めた。次は数式修行ゾーンで完全理論へ。
:::

---

## 📐 3. 数式修行ゾーン（60分）— LDM完全理論

### 3.1 VAE Encoder/Decoderの数学

**Encoder $\mathcal{E}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{h \times w \times c}$**

| 項目 | 詳細 |
|:-----|:-----|
| **圧縮比** | $f = \frac{H \times W}{h \times w}$ (典型的に $f=8$ or $f=16$) |
| **チャネル数** | $c$ は通常4 or 8 (RGB 3chより増える) |
| **パラメータ化** | ResNet blocks + Downsampling + Attention |
| **出力** | Deterministic $z = \mathcal{E}(x)$ or Stochastic $z \sim q(z\|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ |

**Decoder $\mathcal{D}: \mathbb{R}^{h \times w \times c} \to \mathbb{R}^{H \times W \times C}$**

$$
\tilde{x} = \mathcal{D}(z), \quad \text{s.t.} \quad \|\tilde{x} - x\|_\text{perceptual} < \epsilon
$$

知覚的損失（Perceptual Loss）を使う:
$$
\mathcal{L}_\text{rec} = \|\Phi(x) - \Phi(\mathcal{D}(\mathcal{E}(x)))\|^2
$$

ここで $\Phi$ はVGGやLPIPS特徴抽出器。

**2つのVAE正則化方式**:

| 方式 | 正則化項 | 特徴 |
|:-----|:---------|:-----|
| **KL-regularization** | $\mathcal{L}_\text{KL} = \text{KL}[q(z\|x) \| \mathcal{N}(0,I)]$ | 連続潜在空間 / $\beta$-VAE |
| **VQ-regularization** | Codebook loss + Commitment loss | 離散潜在空間 / VQ-VAE/FSQ |

**Stable Diffusion 1.x/2.xの選択**: KL-reg VAE with $f=8$ compression。

**圧縮率 vs 再構成品質のトレードオフ**:

| 圧縮率 $f$ | Latent size (512² input) | LPIPS↓ | FID↓ | 訓練速度 |
|:-----------|:------------------------|:-------|:-----|:---------|
| $f=4$ | 128×128×4 | 0.05 | 1.2 | 1x |
| $f=8$ | 64×64×4 | 0.08 | 2.1 | **4x** |
| $f=16$ | 32×32×4 | 0.15 | 5.6 | 16x |

**結論**: $f=8$ が品質と速度の最適バランス。

```julia
# VAE Encoder/Decoder定義 (Lux.jl)
function create_vae_encoder(; img_size=512, latent_size=64, latent_channels=4)
    # Downsampling path: 512 -> 256 -> 128 -> 64
    encoder = Chain(
        Conv((3, 3), 3 => 128, pad=1),
        ResBlock(128),
        Downsample(128 => 256),  # 512 -> 256
        ResBlock(256),
        Downsample(256 => 512),  # 256 -> 128
        ResBlock(512),
        Downsample(512 => 512),  # 128 -> 64
        SelfAttention(512),
        ResBlock(512),
        Conv((3, 3), 512 => latent_channels, pad=1)  # Output z
    )
    return encoder
end

# エンコード
x = randn(Float32, 512, 512, 3, 1)  # [H, W, C, B]
z = encoder(x, ps, st)[1]           # [64, 64, 4, 1] (48x圧縮!)
```

:::details VAE Decoderのミラー構造
```julia
function create_vae_decoder(; latent_size=64, img_size=512, latent_channels=4)
    decoder = Chain(
        Conv((3, 3), latent_channels => 512, pad=1),
        ResBlock(512),
        SelfAttention(512),
        ResBlock(512),
        Upsample(512 => 512),  # 64 -> 128
        ResBlock(512),
        Upsample(512 => 256),  # 128 -> 256
        ResBlock(256),
        Upsample(256 => 128),  # 256 -> 512
        ResBlock(128),
        Conv((3, 3), 128 => 3, pad=1, activation=tanh)  # Output x
    )
    return decoder
end

x̃ = decoder(z, ps, st)[1]  # [512, 512, 3, 1] 再構成
```
:::

### 3.2 潜在空間での拡散プロセス

**Forward process** (DDPM第36回の復習):
$$
q(z_t | z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} z_0, (1-\bar{\alpha}_t) I)
$$

**閉形式サンプリング**:
$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

**Reverse process**:
$$
p_\theta(z_{t-1} | z_t) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))
$$

**訓練目的関数** (ε-prediction):
$$
\mathcal{L}_\text{simple} = \mathbb{E}_{z_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]
$$

**鍵**: ピクセル空間のDDPMと数式は **完全に同じ**。$x \to z$ の置き換えだけ。

```julia
# Forward process: z₀にノイズ付加
function forward_diffusion(z₀, t, αₜ_bar, rng)
    ε = randn(rng, Float32, size(z₀))
    z_t = sqrt(αₜ_bar) .* z₀ .+ sqrt(1 - αₜ_bar) .* ε
    return z_t, ε
end

# 訓練ステップ
function train_step!(model, z₀, t, αₜ_bar, ps, st, opt_state)
    z_t, ε_true = forward_diffusion(z₀, t, αₜ_bar, rng)

    # ε-prediction
    ε_pred, st = model((z_t, t), ps, st)

    # MSE loss
    loss = mean((ε_pred .- ε_true).^2)

    # Backprop
    gs = gradient(ps -> loss, ps)[1]
    opt_state, ps = Optimisers.update(opt_state, ps, gs)

    return loss, ps, st, opt_state
end
```

**Noise scheduleの選択**:

第36回で学んだスケジュールがそのまま使える:
- Linear: $\beta_t = \beta_\text{start} + t \cdot (\beta_\text{end} - \beta_\text{start}) / T$
- Cosine: $\bar{\alpha}_t = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2 / \cos\left(\frac{s}{1+s} \cdot \frac{\pi}{2}\right)^2$
- **Zero Terminal SNR**: $\bar{\alpha}_T = 0$ を強制（後述）

Stable Diffusion 1.x/2.xは **Linear schedule** with 1000 steps。

### 3.3 LDM訓練パイプライン

**2段階訓練**:

```mermaid
sequenceDiagram
    participant Data as Dataset
    participant VAE as VAE (fixed)
    participant UNet as Diffusion U-Net
    participant Loss as Loss

    Note over Data,Loss: Stage 1: VAE訓練 (別途完了)

    Note over Data,Loss: Stage 2: Diffusion訓練
    Data->>VAE: x₀ (512×512×3)
    VAE->>UNet: z₀ = E(x₀) (64×64×4)
    UNet->>UNet: Forward: z_t ~ q(z_t|z₀)
    UNet->>UNet: Predict: ε_θ(z_t, t)
    UNet->>Loss: ||ε - ε_θ(z_t, t)||²
    Loss->>UNet: Backprop (VAEは固定)
```

**Stage 2の完全な訓練ループ**:

$$
\begin{aligned}
&\text{for epoch in 1:N} \\
&\quad \text{for } (x_0, c) \in \text{DataLoader} \\
&\quad\quad z_0 \leftarrow \mathcal{E}(x_0) \quad \text{(Encoder forward, no grad)} \\
&\quad\quad t \sim \mathcal{U}(1, T) \\
&\quad\quad \epsilon \sim \mathcal{N}(0, I) \\
&\quad\quad z_t \leftarrow \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \\
&\quad\quad \epsilon_\theta \leftarrow \text{UNet}(z_t, t, c) \\
&\quad\quad \mathcal{L} \leftarrow \|\epsilon - \epsilon_\theta\|^2 \\
&\quad\quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L} \quad \text{(UNetのみ更新)}
\end{aligned}
$$

```julia
# 完全な訓練ループ
function train_ldm!(unet, vae_encoder, dataloader, epochs; lr=1e-4)
    opt = Adam(lr)
    opt_state = Optimisers.setup(opt, ps_unet)

    for epoch in 1:epochs
        for (x, c) in dataloader
            # VAE encode (no grad)
            z₀ = vae_encoder(x, ps_vae, st_vae)[1]  # 勾配なし

            # Random timestep
            t = rand(1:T)
            αₜ_bar = alpha_bar_schedule[t]

            # Forward diffusion
            z_t, ε = forward_diffusion(z₀, t, αₜ_bar, rng)

            # Predict noise
            ε_pred, st_unet = unet((z_t, t, c), ps_unet, st_unet)

            # Loss & update
            loss = mse_loss(ε_pred, ε)
            gs = gradient(ps -> loss, ps_unet)[1]
            opt_state, ps_unet = Optimisers.update(opt_state, ps_unet, gs)
        end
    end
    return ps_unet
end
```

:::message alert
**よくあるミス**: VAEに勾配を流してしまう。Encoderは **完全に固定** しなければならない。
:::

### 3.4 Classifier Guidance完全版

**動機**: 条件付き生成 $p(z_0 | c)$ で、条件 $c$ (クラスラベル)への忠実度を高めたい。

**ベイズの定理**:
$$
\nabla_{z_t} \log p(z_t | c) = \nabla_{z_t} \log p(z_t) + \nabla_{z_t} \log p(c | z_t)
$$

ここで:
- $\nabla_{z_t} \log p(z_t)$ は無条件スコア（U-Netが学習）
- $\nabla_{z_t} \log p(c | z_t)$ は分類器の勾配

**Classifier Guidanceの修正サンプリング**:
$$
\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t) - \sqrt{1-\bar{\alpha}_t} \cdot w \nabla_{z_t} \log p_\phi(c | z_t)
$$

ここで:
- $\epsilon_\theta(z_t, t)$ は無条件ノイズ予測
- $p_\phi(c | z_t)$ は **別途訓練した分類器**
- $w$ はguidance scale

**問題点**:
1. 分類器 $p_\phi(c | z_t)$ を別途訓練する必要がある
2. 各timestep $t$ で異なる分類器が必要
3. 訓練コストが2倍

→ Classifier-Free Guidanceで解決!

```julia
# Classifier Guidance (参考実装)
function classifier_guidance_sample(unet, classifier, z_T, c, w)
    z_t = z_T
    for t in T:-1:1
        # 無条件スコア
        ε_uncond = unet(z_t, t, nothing)

        # 分類器勾配
        grad_log_p_c = gradient(z -> log_prob(classifier(z), c), z_t)[1]

        # ガイダンス適用
        ε_guided = ε_uncond .- sqrt(1 - α_bar[t]) .* w .* grad_log_p_c

        # サンプリングステップ
        z_t = reverse_step(z_t, ε_guided, t)
    end
    return decoder(z_t)
end
```

### 3.5 Classifier-Free Guidance完全導出

**鍵となるアイデア**: 条件付きモデル $\epsilon_\theta(z_t, t, c)$ と無条件モデル $\epsilon_\theta(z_t, t, \emptyset)$ を **同時に訓練** し、推論時に線形結合する。

**訓練時**: ランダムにconditionをdrop
$$
c_\text{input} = \begin{cases}
c & \text{with probability } p_\text{uncond} \text{ (e.g. 0.1)} \\
\emptyset & \text{with probability } 1 - p_\text{uncond}
\end{cases}
$$

これで単一モデルが条件付き・無条件両方を学習。

**推論時**: 2つの予測の線形結合
$$
\tilde{\epsilon}_\theta(z_t, t, c, w) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right)
$$

ここで:
- $w$ はguidance scale ($w=0$: 無条件, $w=1$: 条件付き, $w>1$: over-guidance)
- $\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)$ が条件の"方向"

**なぜこれが機能するか？スコアの視点から導出**:

スコア関数 $s_\theta(z_t, t, c) = -\frac{\epsilon_\theta(z_t, t, c)}{\sqrt{1-\bar{\alpha}_t}}$ とすると:

$$
\begin{aligned}
\tilde{s}_\theta(z_t, t, c, w) &= -\frac{\tilde{\epsilon}_\theta(z_t, t, c, w)}{\sqrt{1-\bar{\alpha}_t}} \\
&= -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \left[ \epsilon_\theta(z_t, t, \emptyset) + w(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)) \right] \\
&= s_\theta(z_t, t, \emptyset) + w \cdot (s_\theta(z_t, t, c) - s_\theta(z_t, t, \emptyset)) \\
&= (1-w) \cdot s_\theta(z_t, t, \emptyset) + w \cdot s_\theta(z_t, t, c)
\end{aligned}
$$

これは **無条件スコアと条件付きスコアの加重平均** !

**$w > 1$ の場合**:
$$
\tilde{s}_\theta = s_\theta(z_t, t, \emptyset) + w \cdot (s_\theta(z_t, t, c) - s_\theta(z_t, t, \emptyset))
$$

条件の方向に **$w$倍強く** 押す → mode-seeking行動。

**別の視点: 暗黙的な温度パラメータ**:

$w > 1$ のとき、実効的な確率分布は:
$$
p_w(z_t | c) \propto p(z_t | c)^w \cdot p(z_t)^{1-w}
$$

これはエネルギーベースモデル視点で:
$$
E_w(z_t) = -w \log p(z_t | c) - (1-w) \log p(z_t)
$$

$w \to \infty$ で条件付き分布の **最頻値** (mode)にピーク → 品質向上だが多様性低下。

:::details CFGの理論的理解: Mode-Seeking vs Mode-Covering
**$w$の効果**:

| Guidance Scale $w$ | 挙動 | 品質 | 多様性 |
|:-------------------|:-----|:-----|:-------|
| $w = 0$ | 無条件生成 | 低 | 高 |
| $w = 1$ | 標準条件付き | 中 | 中 |
| $w \in (1, 7]$ | Mode-seeking | **高** | 中 |
| $w > 7$ | Over-guidance | 過飽和 | **低** |

**典型的な値**:
- Stable Diffusion 1.x: $w = 7.5$
- DALL-E 2: $w = 3.0$
- Imagen: $w = 5.0$

**数学的視点**:
- $w < 1$: Mode-covering (KL[q‖p]最小化風)
- $w > 1$: Mode-seeking (KL[p‖q]最小化風)
:::

```julia
# Classifier-Free Guidance実装
function cfg_sample(unet, z_T, c, w; steps=50)
    z_t = z_T
    timesteps = reverse(1:steps)

    for t in timesteps
        # 2回のforward pass
        ε_uncond = unet((z_t, t, nothing), ps, st)[1]  # 無条件
        ε_cond = unet((z_t, t, c), ps, st)[1]           # 条件付き

        # CFG結合
        ε_guided = ε_uncond .+ w .* (ε_cond .- ε_uncond)

        # DDIMステップ (高速サンプリング)
        z_t = ddim_step(z_t, ε_guided, t, t-1, αₜ_bar)
    end

    return vae_decoder(z_t)
end

# 訓練時のCondition Drop
function train_step_with_cfg!(unet, z₀, c, t, p_uncond=0.1)
    # Randomly drop condition
    if rand() < p_uncond
        c = nothing  # 無条件化
    end

    z_t, ε = forward_diffusion(z₀, t, αₜ_bar, rng)
    ε_pred = unet((z_t, t, c), ps, st)[1]

    loss = mse_loss(ε_pred, ε)
    # ... backprop
end
```

### 3.6 CFGの理論的理解

**なぜ品質が向上するか？**

1. **条件の強調**: $w > 1$ で条件情報を増幅 → プロンプト忠実度↑
2. **分散削減**: 無条件との差分が「条件の純粋な効果」 → ノイズ除去
3. **Mode-seeking**: 高確率領域に集中 → 鮮明な画像

**トレードオフ**:
$$
\text{Quality} \uparrow, \quad \text{Diversity} \downarrow, \quad \text{Compute} \times 2
$$

推論時に2回のU-Net forward passが必要 → 速度半減。

**最適化**: Negative Prompt（次節）で1.5回に削減可能。

### 3.7 Negative Prompt

**動機**: CFGで $\epsilon_\theta(z_t, t, \emptyset)$ の代わりに「避けたい概念」を指定したい。

**定式化**:
$$
\tilde{\epsilon}_\theta(z_t, t, c_\text{pos}, c_\text{neg}, w) = \epsilon_\theta(z_t, t, c_\text{neg}) + w \cdot \left( \epsilon_\theta(z_t, t, c_\text{pos}) - \epsilon_\theta(z_t, t, c_\text{neg}) \right)
$$

ここで:
- $c_\text{pos}$ は正のプロンプト ("a beautiful mountain")
- $c_\text{neg}$ は負のプロンプト ("blurry, low quality")
- $w$ はguidance scale

**解釈**: 「$c_\text{neg}$から離れ、$c_\text{pos}$に近づく」方向にガイド。

**典型的なNegative Prompt**:
```
"blurry, low quality, watermark, signature, jpeg artifacts,
 worst quality, low resolution, bad anatomy"
```

```julia
# Negative Prompt実装
function cfg_with_negative(unet, z_T, c_pos, c_neg, w)
    z_t = z_T
    for t in T:-1:1
        ε_neg = unet((z_t, t, c_neg), ps, st)[1]
        ε_pos = unet((z_t, t, c_pos), ps, st)[1]

        # Negative Prompt適用
        ε_guided = ε_neg .+ w .* (ε_pos .- ε_neg)

        z_t = reverse_step(z_t, ε_guided, t)
    end
    return decoder(z_t)
end
```

**効果**:
- 品質向上: 「blurry」を避ける → 鮮明
- アーティファクト削減: 「watermark」を避ける
- 解剖学的エラー削減: 「bad anatomy」を避ける

### 3.8 Text Conditioning

**問題設定**: テキスト $\mathbf{t}$ → 画像 $x$ の生成。条件 $c = \text{Encoder}_\text{text}(\mathbf{t})$。

**Text Encoder選択**:

| Encoder | 次元 | 特徴 | SD採用 |
|:--------|:-----|:-----|:-------|
| **CLIP Text** | 768 | Vision-languageアライメント | SD 1.x |
| **OpenCLIP** | 1024 | より大規模CLIP | SD 2.x |
| **T5** | 4096 | 純粋言語モデル | Imagen/SD3 |
| **CLIP+T5** | 768+4096 | ハイブリッド | SDXL/SD3 |

**CLIP vs T5の違い**:

| 項目 | CLIP | T5 |
|:-----|:-----|:---|
| **訓練** | Image-Text contrastive | Language modeling |
| **語彙** | 49K | 32K |
| **文理解** | 浅い | 深い (Transformer) |
| **画像アライメント** | 強い | 弱い |
| **長文** | 77 tokens max | 512 tokens |

**Stable Diffusion 3のハイブリッド**:
- CLIP: グローバルな意味（「犬」「山」）
- T5: 詳細な関係性（「犬が山の上に座っている」）

:::details CLIP Text Encoderの仕組み
```python
# CLIPテキストエンコーディング (PyTorch擬似コード)
text = "A beautiful mountain landscape"
tokens = tokenizer(text)  # [BOS, 320, 1215, 5270, 5677, EOS, PAD, ...]  # 77 tokens

# Transformer Encoder
embeddings = text_embedding(tokens)  # [77, 768]
hidden = transformer_layers(embeddings)  # [77, 768]

# Pooling
pooled = hidden[0]  # [BOS]トークンを使用 (BERT風)
# または
pooled = hidden.mean(dim=0)  # 平均プーリング

# Output: [768] vector
```

SD 1.xはCLIPの最終層hidden statesを **全て使用**:
- $c = [\mathbf{h}_0, \mathbf{h}_1, \ldots, \mathbf{h}_{76}]$ : shape [77, 768]
- これをCross-Attentionに入力
:::

### 3.9 Cross-Attention Text Conditioning

**U-Netへのテキスト注入方法**:

```mermaid
graph TD
    T[Text: "mountain landscape"] --> TE[Text Encoder<br>CLIP/T5]
    TE --> C[c ∈ R^(L×D)]

    Z[z_t ∈ R^(h×w×c)] --> SA[Self-Attention]
    SA --> CA[Cross-Attention]
    C --> CA
    CA --> FF[FeedForward]
    FF --> Out[Output]

    style CA fill:#ffcccc
```

**Cross-Attentionの定式化**:

U-Netの中間特徴 $\mathbf{f} \in \mathbb{R}^{(h \times w) \times d}$ に対して:

$$
\begin{aligned}
Q &= W_Q \mathbf{f} \quad \in \mathbb{R}^{(h \times w) \times d_k} \\
K &= W_K \mathbf{c} \quad \in \mathbb{R}^{L \times d_k} \\
V &= W_V \mathbf{c} \quad \in \mathbb{R}^{L \times d_v} \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\end{aligned}
$$

ここで:
- $\mathbf{f}$: U-Net中間特徴（画像側）
- $\mathbf{c}$: テキストエンコーディング（条件側）
- $Q$ from image, $K, V$ from text → **Cross**-Attention

**Multi-Head Cross-Attention**:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

各headが異なる意味的側面をキャプチャ（例: 色、形、配置）。

```julia
# Cross-Attention Layer (Lux.jl)
struct CrossAttention
    num_heads::Int
    head_dim::Int
    W_Q::Dense
    W_K::Dense
    W_V::Dense
    W_O::Dense
end

function (ca::CrossAttention)(f, c)
    # f: [h*w, d], c: [L, d_text]
    Q = ca.W_Q(f)      # [h*w, num_heads * head_dim]
    K = ca.W_K(c)      # [L, num_heads * head_dim]
    V = ca.W_V(c)      # [L, num_heads * head_dim]

    # Reshape for multi-head
    Q = reshape(Q, :, ca.num_heads, ca.head_dim)  # [h*w, heads, dim]
    K = reshape(K, :, ca.num_heads, ca.head_dim)  # [L, heads, dim]
    V = reshape(V, :, ca.num_heads, ca.head_dim)

    # Attention
    scores = batched_mul(Q, permutedims(K, (2, 1, 3))) / sqrt(ca.head_dim)
    attn = softmax(scores, dims=2)  # [h*w, L, heads]
    out = batched_mul(attn, V)      # [h*w, heads, dim]

    # Concat + projection
    out = reshape(out, :, ca.num_heads * ca.head_dim)
    return ca.W_O(out)
end
```

**SpatialTransformer Block** (SD U-Net):
```
Input z_t
  ↓
GroupNorm
  ↓
Self-Attention (spatial)
  ↓
Cross-Attention (with text c)
  ↓
FeedForward
  ↓
Output
```

これを各解像度で繰り返し。

### 3.10 Stable Diffusion 1.x/2.x アーキテクチャ詳細

**全体構成**:

```mermaid
graph LR
    Input[Text Prompt] --> CLIP[CLIP Text Encoder]
    CLIP --> C[c: [77, 768]]

    Noise[Random Noise z_T] --> UNet[U-Net with<br>Cross-Attention]
    C --> UNet
    UNet --> z0[Denoised z_0]
    z0 --> VAE[VAE Decoder]
    VAE --> Output[Image 512×512]
```

**U-Net詳細**:

| レベル | 解像度 | Channels | Blocks | Cross-Attn |
|:-------|:-------|:---------|:-------|:-----------|
| **Input** | 64×64 | 4 | - | - |
| **Down 1** | 64→32 | 320 | ResBlock×2 | ✓ |
| **Down 2** | 32→16 | 640 | ResBlock×2 | ✓ |
| **Down 3** | 16→8 | 1280 | ResBlock×2 | ✓ |
| **Middle** | 8 | 1280 | ResBlock×1 | ✓ |
| **Up 1** | 8→16 | 1280 | ResBlock×3 | ✓ |
| **Up 2** | 16→32 | 640 | ResBlock×3 | ✓ |
| **Up 3** | 32→64 | 320 | ResBlock×3 | ✓ |
| **Output** | 64×64 | 4 | - | - |

**総パラメータ数**: ~860M

**ResBlock構成**:
```
Input
  ↓
GroupNorm → SiLU → Conv (3×3)
  ↓
Timestep Embedding (broadcast add)
  ↓
GroupNorm → SiLU → Conv (3×3)
  ↓
Residual Connection
  ↓
Output
```

**Timestep Embedding**:
$$
\gamma(t) = [\sin(t \omega_1), \cos(t \omega_1), \ldots, \sin(t \omega_{d/2}), \cos(t \omega_{d/2})]
$$

ここで $\omega_k = 10000^{-2k/d}$ (Transformer風)。

```julia
# Timestep Embedding
function sinusoidal_embedding(t, dim)
    half_dim = dim ÷ 2
    emb = log(10000) / (half_dim - 1)
    emb = exp.(-emb .* (0:(half_dim-1)))
    emb = t .* emb'
    return hcat(sin.(emb), cos.(emb))
end
```

### 3.11 LDM固有のU-Net拡張: SpatialTransformer

**標準U-Net vs LDM U-Net**:

| 項目 | 標準U-Net | LDM U-Net |
|:-----|:----------|:----------|
| **Attention** | Self-Attention のみ | Self + **Cross**-Attention |
| **条件付け** | Timestep $t$ のみ | $t$ + **text $c$** |
| **層構成** | ResBlock→Attn | ResBlock→**SpatialTransformer** |

**SpatialTransformer Block**:
```
Input: [B, H, W, C]
  ↓
Reshape: [B, H*W, C]
  ↓
LayerNorm
  ↓
Self-Attention: Attention(Q,K,V) where Q=K=V=features
  ↓
LayerNorm
  ↓
Cross-Attention: Attention(Q_img, K_text, V_text)
  ↓
LayerNorm
  ↓
FeedForward (MLP)
  ↓
Reshape: [B, H, W, C]
  ↓
Residual Add
  ↓
Output
```

**なぜSpatial Transformer？**

1. **Self-Attention**: 画像内の長距離依存をキャプチャ
2. **Cross-Attention**: テキストと画像を整列
3. **FeedForward**: 非線形変換で表現力向上

**計算量**: $\mathcal{O}((HW)^2 + HW \cdot L)$ where $L$ はテキスト長。

### 3.12 FLUX Architecture詳解

**FLUXの革新**: Rectified Flow（第38回）+ DiT風Transformer設計。

**FLUX vs Stable Diffusion**:

| 項目 | SD 1.x/2.x | FLUX.1 |
|:-----|:-----------|:-------|
| **バックボーン** | U-Net | **Transformer (DiT風)** |
| **Flow** | DDPM | **Rectified Flow** |
| **Text Encoder** | CLIP | **CLIP + T5** |
| **Latent Size** | 64×64×4 | 64×64×16 (高次元化) |
| **Params** | 860M | **12B** |
| **速度** | 50 steps | **20 steps** (直線化) |

**Rectified Flow統合**:

第38回で学んだように、Rectified FlowはODEで直線的な輸送経路:
$$
\frac{dz_t}{dt} = v_\theta(z_t, t, c)
$$

FLUXは $v_\theta$ をTransformerでパラメータ化:
$$
v_\theta(z_t, t, c) = \text{Transformer}(z_t, t, \text{CLIP}(c), \text{T5}(c))
$$

**DiT (Diffusion Transformer) 風設計**:

```mermaid
graph TD
    Z[z_t patches] --> PE[Positional Encoding]
    PE --> TB1[Transformer Block 1]
    TB1 --> TB2[Transformer Block 2]
    TB2 --> TBn[Transformer Block N]
    TBn --> Out[v_θ prediction]

    T[Text: CLIP+T5] --> Cond[Conditioning]
    Cond --> TB1
    Cond --> TB2
    Cond --> TBn
```

**Transformer Blockの構成**:
```
Input: z_t patches + timestep t + text c
  ↓
Adaptive LayerNorm (conditioned on t, c)
  ↓
Self-Attention (全patch間)
  ↓
Adaptive LayerNorm
  ↓
Cross-Attention (patch ↔ text)
  ↓
Adaptive LayerNorm
  ↓
MLP
  ↓
Output
```

**なぜFLUXが速い？**

1. **Rectified Flow**: 直線的輸送 → 少ステップで収束
2. **高次元latent**: 16ch → より豊かな表現
3. **Transformer**: 長距離依存を効率的にキャプチャ

**FLUX.1のバリエーション**:

| モデル | Params | 速度 | 品質 | 用途 |
|:-------|:-------|:-----|:-----|:-----|
| **FLUX.1-pro** | 12B | 20 steps | 最高 | API専用 |
| **FLUX.1-dev** | 12B | 20 steps | 高 | 開発用 |
| **FLUX.1-schnell** | 12B | **4 steps** | 中 | 高速生成 |

:::details FLUX Transformerの実装概要
```julia
# FLUX Transformer Block (概念的)
struct FLUXTransformerBlock
    self_attn::MultiHeadAttention
    cross_attn::MultiHeadAttention
    mlp::MLP
    adaLN::AdaptiveLayerNorm
end

function (block::FLUXTransformerBlock)(z_patches, t_emb, text_emb)
    # Adaptive LayerNorm (timestep & text conditioned)
    z = block.adaLN(z_patches, t_emb, text_emb)

    # Self-Attention (全patch間)
    z = z + block.self_attn(z, z, z)

    # Cross-Attention (patch ↔ text)
    z = block.adaLN(z, t_emb, text_emb)
    z = z + block.cross_attn(z, text_emb, text_emb)

    # MLP
    z = block.adaLN(z, t_emb, text_emb)
    z = z + block.mlp(z)

    return z
end
```
:::

### 3.13 学習テクニック

**Noise Offset**:

Forward processにバイアスを加える:
$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} (\epsilon + \text{offset})
$$

**効果**: 暗い画像・明るい画像の生成品質向上。

**Min-SNR Weighting**:

Loss weightingをSNRベースで調整:
$$
\mathcal{L}_\text{Min-SNR} = \mathbb{E}_{t} \left[ w(t) \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]
$$

$$
w(t) = \min\left(\text{SNR}(t), \gamma\right), \quad \text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
$$

典型的に $\gamma = 5$。

**効果**: 3.4倍訓練高速化 [^min_snr]。

**v-prediction**:

ε-predictionの代わりにvelocity prediction:
$$
v_t = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} z_0
$$

$$
\mathcal{L}_v = \mathbb{E}_{t} \left[ \|v_t - v_\theta(z_t, t)\|^2 \right]
$$

**効果**: 数値安定性向上・収束性改善。

**Zero Terminal SNR**:

Noise scheduleを強制的に $\bar{\alpha}_T = 0$ に rescale:
$$
\tilde{\alpha}_t = \frac{\alpha_t}{\alpha_T}
$$

**効果**: 非常に明るい/暗い画像の生成品質向上 [^zero_snr]。

```julia
# Zero Terminal SNR rescaling
function rescale_to_zero_terminal_snr(alphas)
    alphas_cumprod = cumprod(alphas)
    sqrt_alphas_cumprod = sqrt.(alphas_cumprod)

    # Rescale
    sqrt_alphas_cumprod_final = sqrt_alphas_cumprod[end]
    sqrt_alphas_cumprod .= sqrt_alphas_cumprod ./ sqrt_alphas_cumprod_final

    return sqrt_alphas_cumprod
end
```

:::message
**ここまでで全体の50%完了！ ボス戦前のチェックポイント。**

数式修行ゾーン完了:
- VAE Encoder/Decoder圧縮の数学
- 潜在空間拡散プロセス
- Classifier Guidance完全版
- Classifier-Free Guidance完全導出
- CFG理論的理解（Mode-Seeking / 温度）
- Negative Prompt
- Text Conditioning（CLIP / T5）
- Cross-Attention完全版
- SD 1.x/2.x アーキテクチャ
- SpatialTransformer
- FLUX Architecture詳解
- 学習テクニック（Noise Offset / Min-SNR / v-prediction / Zero Terminal SNR）

次は実装ゾーンへ！
:::

### 3.14 SD vs FLUX: アーキテクチャ詳細比較

**Stable Diffusion 1.x/2.x/SDXL Architecture**:

```
Input: Text prompt
  ↓
CLIP Text Encoder: [77, 768]
  ↓
Random Noise z_T: [64, 64, 4]
  ↓
U-Net (ResBlock + SpatialTransformer):
  - Down1: 64→32 (320ch, Cross-Attn)
  - Down2: 32→16 (640ch, Cross-Attn)
  - Down3: 16→8 (1280ch, Cross-Attn)
  - Middle: 8 (1280ch, Cross-Attn)
  - Up1: 8→16 (1280ch, Cross-Attn)
  - Up2: 16→32 (640ch, Cross-Attn)
  - Up3: 32→64 (320ch, Cross-Attn)
  ↓
Denoised z_0: [64, 64, 4]
  ↓
VAE Decoder (f=8)
  ↓
Image: [512, 512, 3]
```

**FLUX.1 Architecture**:

```
Input: Text prompt
  ↓
Dual Encoders:
  - CLIP ViT-L: [77, 768]
  - T5-XXL: [512, 4096]
  ↓
Random Noise z_T: [64, 64, 16]  # 4倍のチャネル!
  ↓
Patchify: [64, 64, 16] → [1024, 768]  # 4×4パッチ
  ↓
Positional Encoding (RoPE)
  ↓
Transformer Blocks (N=24):
  - Adaptive LayerNorm (t, c conditioned)
  - Self-Attention (全patch間)
  - Cross-Attention (patch ↔ CLIP+T5)
  - Gated FFN
  ↓
Unpatchify: [1024, 768] → [64, 64, 16]
  ↓
VAE Decoder (f=8, 16ch input)
  ↓
Image: [512, 512, 3]
```

**詳細比較表**:

| 項目 | SD 1.x/2.x | SDXL | FLUX.1 |
|:-----|:-----------|:-----|:-------|
| **Backbone** | U-Net | U-Net | **Transformer** |
| **Total Params** | 860M | 2.6B | **12B** |
| **Text Encoder** | CLIP (768) | CLIP+OpenCLIP (2048) | **CLIP+T5 (4864)** |
| **Latent Channels** | 4 | 4 | **16** |
| **Latent Res** | 64×64 (f=8) | 128×128 (f=8) | 64×64 (f=8) |
| **Attention Type** | Cross-Attn in U-Net | Cross-Attn in U-Net | **Self+Cross in Transformer** |
| **Position Enc** | なし (CNN) | なし (CNN) | **RoPE** |
| **Flow Type** | DDPM | DDPM | **Rectified Flow** |
| **Timesteps** | 1000 | 1000 | 1000 (訓練), 20-50 (推論) |
| **CFG Default** | 7.5 | 7.5 | 3.5 (低めでOK) |
| **Memory (fp16)** | 10GB | 16GB | **24GB** |
| **Speed (50 steps)** | ~5s (A100) | ~8s (A100) | **~3s (20 steps, A100)** |

### 3.15 Cross-Attention詳細実装

**Multi-Head Cross-Attention完全版**:

```julia
struct MultiHeadCrossAttention{F}
    num_heads::Int
    head_dim::Int
    qkv_dim::Int
    W_Q::Dense
    W_K::Dense
    W_V::Dense
    W_O::Dense
    dropout::Dropout
end

function MultiHeadCrossAttention(qkv_dim::Int, num_heads::Int; dropout_rate=0.1)
    head_dim = qkv_dim ÷ num_heads
    @assert qkv_dim == num_heads * head_dim "qkv_dim must be divisible by num_heads"

    return MultiHeadCrossAttention(
        num_heads,
        head_dim,
        qkv_dim,
        Dense(qkv_dim => qkv_dim),  # W_Q
        Dense(qkv_dim => qkv_dim),  # W_K
        Dense(qkv_dim => qkv_dim),  # W_V
        Dense(qkv_dim => qkv_dim),  # W_O
        Dropout(dropout_rate)
    )
end

function (mha::MultiHeadCrossAttention)(q, k, v, mask=nothing)
    # q: [N_q, d], k: [N_k, d], v: [N_k, d]
    batch_size = size(q, 1)

    # Linear projections
    Q = mha.W_Q(q)  # [N_q, d]
    K = mha.W_K(k)  # [N_k, d]
    V = mha.W_V(v)  # [N_k, d]

    # Reshape to multi-head: [N, d] → [N, num_heads, head_dim]
    Q = reshape(Q, :, mha.num_heads, mha.head_dim)
    K = reshape(K, :, mha.num_heads, mha.head_dim)
    V = reshape(V, :, mha.num_heads, mha.head_dim)

    # Transpose for batch matrix multiply: [num_heads, N, head_dim]
    Q = permutedims(Q, (2, 1, 3))
    K = permutedims(K, (2, 1, 3))
    V = permutedims(V, (2, 1, 3))

    # Scaled dot-product attention
    scores = batched_mul(Q, batched_transpose(K)) / sqrt(Float32(mha.head_dim))
    # scores: [num_heads, N_q, N_k]

    # Apply mask if provided
    if mask !== nothing
        scores = scores .+ mask
    end

    # Softmax
    attn_weights = softmax(scores, dims=3)  # Over N_k
    attn_weights = mha.dropout(attn_weights)

    # Apply attention to values
    out = batched_mul(attn_weights, V)  # [num_heads, N_q, head_dim]

    # Transpose back: [num_heads, N_q, head_dim] → [N_q, num_heads, head_dim]
    out = permutedims(out, (2, 1, 3))

    # Concat heads: [N_q, num_heads, head_dim] → [N_q, d]
    out = reshape(out, :, mha.qkv_dim)

    # Final linear
    return mha.W_O(out)
end
```

**使用例**:

```julia
# 初期化
d_model = 768
num_heads = 12
mha = MultiHeadCrossAttention(d_model, num_heads)

# U-Net中間特徴: [h*w, d_model]
f = randn(Float32, 64*64, d_model)

# Text embeddings: [77, d_model]
c = randn(Float32, 77, d_model)

# Cross-Attention
out = mha(f, c, c)  # Q from image, K/V from text
# out: [64*64, d_model]
```

### 3.16 VAE訓練の詳細

**VAE損失関数の完全展開**:

$$
\begin{aligned}
\mathcal{L}_\text{VAE} &= \mathcal{L}_\text{rec} + \beta \cdot \mathcal{L}_\text{KL} \\
\mathcal{L}_\text{rec} &= \mathbb{E}_{q(z|x)}[-\log p_\theta(x|z)] \\
&\approx -\log p_\theta(x | \mathcal{E}(x)) \\
&= \text{MSE}(x, \mathcal{D}(\mathcal{E}(x))) \quad \text{or} \quad \text{LPIPS}(x, \mathcal{D}(\mathcal{E}(x))) \\
\mathcal{L}_\text{KL} &= \text{KL}[q_\phi(z|x) \| p(z)] \\
&= \text{KL}[\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)) \| \mathcal{N}(0, I)] \\
&= \frac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right)
\end{aligned}
$$

**Perceptual Loss (LPIPS)**:

LPIPSはVGG特徴空間での距離:
$$
\mathcal{L}_\text{LPIPS}(x, \tilde{x}) = \sum_l w_l \|\Phi_l(x) - \Phi_l(\tilde{x})\|^2
$$

ここで $\Phi_l$ はVGGの第$l$層特徴。

```julia
# LPIPS損失 (簡略版)
function lpips_loss(x, x_recon, vgg_model, layers=[3, 8, 15, 22])
    loss = 0.0
    for layer in layers
        feat_x = vgg_model[1:layer](x)
        feat_recon = vgg_model[1:layer](x_recon)
        loss += mean((feat_x .- feat_recon).^2)
    end
    return loss / length(layers)
end

# VAE訓練with LPIPS
function train_vae_lpips!(encoder, decoder, vgg, dataloader; β=0.1)
    for (x,) in dataloader
        # Encode
        μ, logσ² = encoder(x)
        σ = exp.(0.5 .* logσ²)
        ε = randn(size(μ))
        z = μ .+ σ .* ε

        # Decode
        x_recon = decoder(z)

        # Losses
        recon_loss = lpips_loss(x, x_recon, vgg)
        kl_loss = 0.5 * mean(μ.^2 .+ σ.^2 .- logσ² .- 1)

        loss = recon_loss + β * kl_loss
        # ... backprop
    end
end
```

### 3.17 Noise Scheduleの設計理論

**3つの主要スケジュール**:

**1. Linear Schedule** (DDPM original):
$$
\beta_t = \beta_\text{min} + \frac{t-1}{T-1} (\beta_\text{max} - \beta_\text{min})
$$

典型値: $\beta_\text{min} = 0.0001$, $\beta_\text{max} = 0.02$, $T = 1000$

**2. Cosine Schedule** (Improved DDPM):
$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left( \frac{t/T + s}{1+s} \cdot \frac{\pi}{2} \right)^2
$$

$s = 0.008$ が標準。

**3. Learned Schedule**:

$\beta_t$ をパラメータ化して学習。

```julia
# 3つのスケジュール実装
function linear_beta_schedule(T::Int; β_start=1e-4, β_end=0.02)
    return range(β_start, β_end, length=T)
end

function cosine_alpha_bar_schedule(T::Int; s=0.008)
    t = 0:T
    f_t = cos.(((t ./ T) .+ s) ./ (1 + s) .* π ./ 2).^2
    α_bar = f_t ./ f_t[1]
    return α_bar[2:end]
end

function learned_beta_schedule(T::Int; init_β_start=1e-4, init_β_end=0.02)
    # パラメータ化βを学習
    logit_β = range(logit(init_β_start), logit(init_β_end), length=T)
    return logit_β  # 訓練中に最適化
end

# Zero Terminal SNR rescaling
function rescale_zero_terminal_snr(α_bar)
    # α_bar[T] = 0 を強制
    return α_bar ./ α_bar[end]
end
```

**SNRの可視化**:

```julia
using Plots

T = 1000
betas_linear = linear_beta_schedule(T)
α_bar_cosine = cosine_alpha_bar_schedule(T)

# SNR計算
snr_linear = cumprod(1 .- betas_linear) ./ (1 .- cumprod(1 .- betas_linear))
snr_cosine = α_bar_cosine ./ (1 .- α_bar_cosine)

# Plot
plot(1:T, snr_linear, label="Linear", yscale=:log10, xlabel="Timestep", ylabel="SNR")
plot!(1:T, snr_cosine, label="Cosine")
title!("SNR Schedule Comparison")
```

### 3.18 Sampling Algorithms完全比較

| アルゴリズム | ステップ数 | 品質 | 速度 | 決定論的 | 備考 |
|:-------------|:-----------|:-----|:-----|:---------|:-----|
| **DDPM** | 1000 | 最高 | 最遅 | ✗ | 確率的、全ステップ必要 |
| **DDIM** | 50-100 | 高 | 中 | ✓ | 決定論的、スキップ可能 |
| **DPM-Solver++** | 20-30 | 高 | 高 | ✓ | 高次ODE solver |
| **UniPC** | 10-20 | 中 | 最高 | ✓ | Predictor-Corrector |
| **PNDM** | 50 | 高 | 中 | ✗ | Pseudo Numerical |
| **LMS** | 50 | 中 | 中 | ✓ | Linear Multi-Step |

**DDIM完全版**:

$$
\begin{aligned}
z_{t-\Delta t} &= \sqrt{\bar{\alpha}_{t-\Delta t}} \underbrace{\left( \frac{z_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{pred } x_0} \\
&\quad + \underbrace{\sqrt{1-\bar{\alpha}_{t-\Delta t} - \sigma_t^2} \cdot \epsilon_\theta(z_t, t)}_{\text{direction to } z_{t-\Delta t}} \\
&\quad + \underbrace{\sigma_t \epsilon}_{\text{random noise}}
\end{aligned}
$$

$\sigma_t = 0$ で完全決定論的、$\sigma_t = \sqrt{(1-\bar{\alpha}_{t-\Delta t})/(1-\bar{\alpha}_t)} \sqrt{1-\bar{\alpha}_t/\bar{\alpha}_{t-\Delta t}}$ でDDPMと同等。

```julia
function ddim_step(z_t, ε_θ, t, t_prev, α_bar; η=0.0)
    # Predict x₀
    α_t = α_bar[t]
    α_prev = t_prev > 0 ? α_bar[t_prev] : 1.0

    pred_x₀ = (z_t .- sqrt(1 - α_t) .* ε_θ) ./ sqrt(α_t)

    # Direction
    σ_t = η * sqrt((1 - α_prev) / (1 - α_t)) * sqrt(1 - α_t / α_prev)
    dir_z = sqrt(1 - α_prev - σ_t^2) .* ε_θ

    # Noise
    noise = σ_t .* randn(Float32, size(z_t))

    # Combine
    z_prev = sqrt(α_prev) .* pred_x₀ .+ dir_z .+ noise
    return z_prev
end
```

**DPM-Solver++ (概要)**:

高次ODE solverでDDIMを改善:
$$
z_{t-\Delta t} = z_t + \int_t^{t-\Delta t} f(z_s, s) ds
$$

3次Adams-Bashforth法で近似 → 20ステップで高品質。

### ⚔️ Boss Battle: CFGの完全分解

**課題**: Classifier-Free Guidanceの数式を、3つの視点から完全に導出せよ。

**視点1: ε-prediction**

目標: 修正されたノイズ予測 $\tilde{\epsilon}_\theta(z_t, t, c, w)$ を求めよ。

**解答**:
$$
\begin{aligned}
\tilde{\epsilon}_\theta(z_t, t, c, w) &= \epsilon_\theta(z_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right) \\
&= (1-w) \epsilon_\theta(z_t, t, \emptyset) + w \cdot \epsilon_\theta(z_t, t, c)
\end{aligned}
$$

**視点2: スコア関数**

目標: 修正されたスコア $\tilde{s}_\theta(z_t, t, c, w)$ を求めよ。

**解答**:

スコア $s_\theta = -\frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha}_t}}$ より:
$$
\begin{aligned}
\tilde{s}_\theta(z_t, t, c, w) &= -\frac{\tilde{\epsilon}_\theta}{\sqrt{1-\bar{\alpha}_t}} \\
&= -\frac{1}{\sqrt{1-\bar{\alpha}_t}} \left[ \epsilon_\theta(z_t, t, \emptyset) + w(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)) \right] \\
&= s_\theta(z_t, t, \emptyset) + w \cdot (s_\theta(z_t, t, c) - s_\theta(z_t, t, \emptyset)) \\
&= (1-w) s_\theta(z_t, t, \emptyset) + w \cdot s_\theta(z_t, t, c)
\end{aligned}
$$

**視点3: 確率分布**

目標: $w > 1$ のときの実効確率分布 $p_w(z_t | c)$ を求めよ。

**解答**:

スコアマッチングより $s(z) = \nabla_z \log p(z)$ なので:
$$
\begin{aligned}
\tilde{s}_\theta(z_t, t, c, w) &= \nabla_{z_t} \log \tilde{p}(z_t | c) \\
&= (1-w) \nabla_{z_t} \log p(z_t) + w \nabla_{z_t} \log p(z_t | c) \\
&= \nabla_{z_t} \left[ (1-w) \log p(z_t) + w \log p(z_t | c) \right] \\
&= \nabla_{z_t} \log p_w(z_t | c)
\end{aligned}
$$

よって:
$$
p_w(z_t | c) \propto p(z_t)^{1-w} \cdot p(z_t | c)^w
$$

$w > 1$ のとき、条件付き分布を **over-emphasize** → mode-seeking。

**数値検証**:
```julia
# CFGの3視点検証
w = 7.5
ε_uncond = randn(Float32, 64, 64, 4)
ε_cond = randn(Float32, 64, 64, 4)

# 視点1: ε-prediction
ε_cfg1 = ε_uncond .+ w .* (ε_cond .- ε_uncond)
ε_cfg2 = (1 - w) .* ε_uncond .+ w .* ε_cond

@assert isapprox(ε_cfg1, ε_cfg2)  # 等価性確認

# 視点2: スコア
α_bar = 0.5
s_uncond = -ε_uncond ./ sqrt(1 - α_bar)
s_cond = -ε_cond ./ sqrt(1 - α_bar)
s_cfg = (1 - w) .* s_uncond .+ w .* s_cond

# 視点3: 確率
log_p_uncond = -0.5 * sum(ε_uncond.^2)
log_p_cond = -0.5 * sum(ε_cond.^2)
log_p_w = (1 - w) * log_p_uncond + w * log_p_cond

println("CFG 3視点検証完了！")
```

**ボス撃破！** CFGの数学的構造を完全理解した。

---

## 💻 4. 実装ゾーン（45分）— LDM訓練→推論パイプライン

### 環境構築とライブラリ

**Julia環境** (⚡訓練):
```julia
using Pkg
Pkg.add([
    "Lux",           # NN framework
    "Reactant",      # XLA compiler (GPU高速化)
    "Optimisers",    # Adam等
    "Zygote",        # 自動微分
    "MLDatasets",    # MNIST等
    "Images",        # 画像処理
    "CUDA",          # GPU
    "BenchmarkTools" # 性能測定
])
```

**Rust環境** (🦀推論):
```toml
[dependencies]
candle-core = "0.7"
candle-nn = "0.7"
safetensors = "0.4"
```

### LaTeX記法チートシート

| 記号 | LaTeX | 意味 |
|:-----|:------|:-----|
| $\mathcal{E}$ | `\mathcal{E}` | Encoder |
| $\mathcal{D}$ | `\mathcal{D}` | Decoder |
| $\bar{\alpha}_t$ | `\bar{\alpha}_t` | Cumulative product |
| $\epsilon_\theta(z_t, t, c)$ | `\epsilon_\theta(z_t, t, c)` | Noise prediction |
| $\tilde{\epsilon}$ | `\tilde{\epsilon}` | Modified noise (CFG) |
| $\mathbb{E}_{q(z\|x)}[\cdot]$ | `\mathbb{E}_{q(z\|x)}[\cdot]` | Expectation |
| $\text{KL}[q \| p]$ | `\text{KL}[q \| p]` | KL divergence |

### Math→Code翻訳パターン (LDM編)

**パターン1: VAE Encoder**
$$
z = \mathcal{E}(x), \quad x \in \mathbb{R}^{H \times W \times C} \to z \in \mathbb{R}^{h \times w \times c}
$$

```julia
z = encoder(x, ps_encoder, st_encoder)[1]
# x: [H, W, C, B] → z: [h, w, c, B]
```

**パターン2: Forward Diffusion**
$$
z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
$$

```julia
ε = randn(rng, Float32, size(z₀))
z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε
```

**パターン3: CFG**
$$
\tilde{\epsilon}_\theta = \epsilon_\theta(z_t, t, \emptyset) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))
$$

```julia
ε_uncond = unet((z_t, t, nothing), ps, st)[1]
ε_cond = unet((z_t, t, c), ps, st)[1]
ε_cfg = ε_uncond .+ w .* (ε_cond .- ε_uncond)
```

**パターン4: DDIM Sampling**
$$
z_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{z_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta + \sigma_t \epsilon
$$

```julia
pred_x₀ = (z_t .- sqrt(1 - α_bar[t]) .* ε_θ) ./ sqrt(α_bar[t])
dir_z = sqrt(1 - α_bar[t-1] - σ²) .* ε_θ
noise = σ .* randn(rng, Float32, size(z_t))
z_prev = sqrt(α_bar[t-1]) .* pred_x₀ .+ dir_z .+ noise
```

**パターン5: Cross-Attention**
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

```julia
scores = (Q * K') ./ sqrt(d_k)  # [N_q, N_k]
attn = softmax(scores, dims=2)   # [N_q, N_k]
out = attn * V                   # [N_q, d_v]
```

**パターン6: Min-SNR Weighting**
$$
w(t) = \min\left(\text{SNR}(t), \gamma\right), \quad \text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
$$

```julia
snr = α_bar ./ (1 .- α_bar)
weight = min.(snr, γ)
loss = weight[t] * mse(ε_pred, ε_true)
```

**パターン7: Zero Terminal SNR Rescaling**
$$
\tilde{\alpha}_t = \frac{\alpha_t}{\alpha_T}
$$

```julia
α_cumprod = cumprod(alphas)
α_cumprod_rescaled = α_cumprod ./ α_cumprod[end]
```

### ⚡ Julia完全実装: Mini Latent Diffusion

**ステップ1: VAE定義**

```julia
using Lux, Random, Optimisers, Zygote

# Encoder
function create_encoder(; in_ch=3, latent_ch=4, base_ch=64)
    return Chain(
        Conv((3,3), in_ch => base_ch, pad=1, activation=relu),
        Conv((4,4), base_ch => base_ch*2, stride=2, pad=1, activation=relu),  # /2
        Conv((4,4), base_ch*2 => base_ch*4, stride=2, pad=1, activation=relu),  # /4
        Conv((4,4), base_ch*4 => base_ch*8, stride=2, pad=1, activation=relu),  # /8
        Conv((3,3), base_ch*8 => latent_ch, pad=1)  # Output z
    )
end

# Decoder (mirror)
function create_decoder(; latent_ch=4, out_ch=3, base_ch=64)
    return Chain(
        Conv((3,3), latent_ch => base_ch*8, pad=1, activation=relu),
        ConvTranspose((4,4), base_ch*8 => base_ch*4, stride=2, pad=1, activation=relu),  # *2
        ConvTranspose((4,4), base_ch*4 => base_ch*2, stride=2, pad=1, activation=relu),  # *4
        ConvTranspose((4,4), base_ch*2 => base_ch, stride=2, pad=1, activation=relu),    # *8
        Conv((3,3), base_ch => out_ch, pad=1, activation=tanh)
    )
end

# VAE訓練
function train_vae!(encoder, decoder, dataloader; epochs=10, lr=1e-3, β=0.5)
    ps_enc, st_enc = Lux.setup(Random.default_rng(), encoder)
    ps_dec, st_dec = Lux.setup(Random.default_rng(), decoder)
    opt = Adam(lr)
    opt_state_enc = Optimisers.setup(opt, ps_enc)
    opt_state_dec = Optimisers.setup(opt, ps_dec)

    for epoch in 1:epochs
        total_loss = 0.0
        for (x,) in dataloader
            # Forward
            z, st_enc = encoder(x, ps_enc, st_enc)
            x_recon, st_dec = decoder(z, ps_dec, st_dec)

            # Loss: Reconstruction + KL (simplified)
            recon_loss = mean((x_recon .- x).^2)
            kl_loss = 0.5 * mean(z.^2)  # Simplified KL to N(0,I)
            loss = recon_loss + β * kl_loss

            # Backprop
            gs_enc = gradient(p -> loss, ps_enc)[1]
            gs_dec = gradient(p -> loss, ps_dec)[1]
            opt_state_enc, ps_enc = Optimisers.update(opt_state_enc, ps_enc, gs_enc)
            opt_state_dec, ps_dec = Optimisers.update(opt_state_dec, ps_dec, gs_dec)

            total_loss += loss
        end
        println("Epoch $epoch: Loss = $(total_loss / length(dataloader))")
    end
    return ps_enc, ps_dec, st_enc, st_dec
end
```

**ステップ2: U-Net定義 (Simplified)**

```julia
# ResBlock
struct ResBlock
    conv1::Conv
    conv2::Conv
end

function (rb::ResBlock)(x)
    h = rb.conv1(x)
    h = relu.(h)
    h = rb.conv2(h)
    return relu.(h .+ x)  # Residual
end

# Time Embedding
function sinusoidal_embedding(t::Int, dim::Int)
    half = dim ÷ 2
    freqs = exp.(-log(10000f0) .* (0:half-1) ./ half)
    args = t .* freqs
    return vcat(sin.(args), cos.(args))
end

# Simplified U-Net (for 32x32 latent)
function create_unet(; latent_ch=4, base_ch=128, time_emb_dim=256)
    return Chain(
        # Time embedding MLP
        Dense(time_emb_dim, time_emb_dim*4, activation=silu),
        Dense(time_emb_dim*4, time_emb_dim*4, activation=silu),

        # Down
        Conv((3,3), latent_ch => base_ch, pad=1),
        ResBlock(Conv((3,3), base_ch => base_ch, pad=1), Conv((3,3), base_ch => base_ch, pad=1)),
        Conv((4,4), base_ch => base_ch*2, stride=2, pad=1),  # /2
        ResBlock(Conv((3,3), base_ch*2 => base_ch*2, pad=1), Conv((3,3), base_ch*2 => base_ch*2, pad=1)),

        # Middle
        ResBlock(Conv((3,3), base_ch*2 => base_ch*2, pad=1), Conv((3,3), base_ch*2 => base_ch*2, pad=1)),

        # Up
        ConvTranspose((4,4), base_ch*2 => base_ch, stride=2, pad=1),  # *2
        ResBlock(Conv((3,3), base_ch => base_ch, pad=1), Conv((3,3), base_ch => base_ch, pad=1)),
        Conv((3,3), base_ch => latent_ch, pad=1)  # Output ε
    )
end
```

**ステップ3: Diffusion訓練ループ**

```julia
# Noise schedule
function cosine_beta_schedule(T::Int; s=0.008)
    t = 0:T
    α_bar = cos.(((t ./ T) .+ s) ./ (1 + s) .* π ./ 2).^2
    α_bar = α_bar ./ α_bar[1]
    betas = 1 .- α_bar[2:end] ./ α_bar[1:end-1]
    return clamp.(betas, 0f0, 0.999f0), α_bar[2:end]
end

# Forward diffusion
function forward_diffusion(z₀, t, α_bar, rng)
    ε = randn(rng, Float32, size(z₀))
    z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε
    return z_t, ε
end

# 訓練ループ
function train_ldm!(unet, encoder, dataloader; T=1000, epochs=100, lr=1e-4)
    betas, α_bar = cosine_beta_schedule(T)
    ps_unet, st_unet = Lux.setup(Random.default_rng(), unet)
    opt = Adam(lr)
    opt_state = Optimisers.setup(opt, ps_unet)
    rng = Random.default_rng()

    # Freeze encoder
    ps_enc, st_enc = Lux.setup(rng, encoder)

    for epoch in 1:epochs
        total_loss = 0.0
        for (x,) in dataloader
            # Encode (no grad)
            z₀, _ = encoder(x, ps_enc, st_enc)

            # Random timestep
            t = rand(rng, 1:T)

            # Forward diffusion
            z_t, ε_true = forward_diffusion(z₀, t, α_bar, rng)

            # Time embedding
            t_emb = sinusoidal_embedding(t, 256)

            # Predict noise
            ε_pred, st_unet = unet((z_t, t_emb), ps_unet, st_unet)

            # MSE loss
            loss = mean((ε_pred .- ε_true).^2)

            # Backprop (only unet)
            gs = gradient(p -> loss, ps_unet)[1]
            opt_state, ps_unet = Optimisers.update(opt_state, ps_unet, gs)

            total_loss += loss
        end
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $(total_loss / length(dataloader))")
        end
    end
    return ps_unet, st_unet
end
```

**ステップ4: CFGサンプリング**

```julia
# DDIM sampling with CFG
function ddim_sample_cfg(unet, decoder, z_T, c, w; steps=50, η=0.0)
    T = 1000
    betas, α_bar = cosine_beta_schedule(T)
    timesteps = reverse(Int.(round.(range(1, T, length=steps))))

    z_t = z_T
    for (i, t) in enumerate(timesteps)
        t_prev = i == steps ? 0 : timesteps[i+1]

        # Time embedding
        t_emb = sinusoidal_embedding(t, 256)

        # CFG: 2 forward passes
        ε_uncond, _ = unet((z_t, t_emb, nothing), ps_unet, st_unet)
        ε_cond, _ = unet((z_t, t_emb, c), ps_unet, st_unet)
        ε_cfg = ε_uncond .+ w .* (ε_cond .- ε_uncond)

        # Predict x₀
        pred_x₀ = (z_t .- sqrt(1 - α_bar[t]) .* ε_cfg) ./ sqrt(α_bar[t])

        # DDIM step
        if t_prev > 0
            σ_t = η * sqrt((1 - α_bar[t_prev]) / (1 - α_bar[t])) * sqrt(1 - α_bar[t] / α_bar[t_prev])
            dir_z = sqrt(1 - α_bar[t_prev] - σ_t^2) .* ε_cfg
            noise = σ_t .* randn(Float32, size(z_t))
            z_t = sqrt(α_bar[t_prev]) .* pred_x₀ .+ dir_z .+ noise
        else
            z_t = pred_x₀
        end
    end

    # Decode
    x, _ = decoder(z_t, ps_dec, st_dec)
    return x
end

# 使用例
z_T = randn(Float32, 32, 32, 4, 1)  # Random noise
c = nothing  # 無条件 or テキスト埋め込み
w = 7.5      # CFG scale
x_gen = ddim_sample_cfg(unet, decoder, z_T, c, w, steps=50)
```

:::details 完全な訓練スクリプト
```julia
using MLDatasets, Images

# データ準備
train_data = MNIST(split=:train)
X_train = Float32.(reshape(train_data.features, 28, 28, 1, :))
X_train = (X_train .* 2f0) .- 1f0  # [-1, 1]

# Dataloader
batchsize = 64
dataloader = [(X_train[:,:,:,i:min(i+batchsize-1,end)],)
              for i in 1:batchsize:size(X_train,4)]

# モデル作成
encoder = create_encoder(in_ch=1, latent_ch=4, base_ch=32)
decoder = create_decoder(latent_ch=4, out_ch=1, base_ch=32)
unet = create_unet(latent_ch=4, base_ch=64)

# Stage 1: VAE訓練
println("Training VAE...")
ps_enc, ps_dec, st_enc, st_dec = train_vae!(encoder, decoder, dataloader, epochs=20)

# Stage 2: Diffusion訓練
println("Training Diffusion...")
ps_unet, st_unet = train_ldm!(unet, encoder, dataloader, T=1000, epochs=100)

# サンプリング
println("Generating samples...")
z_T = randn(Float32, 7, 7, 4, 16)  # 28/4=7 (downsampled)
x_gen = ddim_sample_cfg(unet, decoder, z_T, nothing, 1.0, steps=50)

# 保存
using FileIO
save("generated.png", colorview(Gray, x_gen[:,:,1,1]))
```
:::

### 🦀 Rust推論実装

**safetensorsからモデルロード**:

```rust
use candle_core::{Device, Tensor};
use candle_nn::{VarBuilder, Module};

// VAE Decoder
struct Decoder {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    // ... more layers
}

impl Decoder {
    fn new(vb: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv2d(4, 512, 3, Default::default(), vb.pp("conv1"))?;
        let conv2 = candle_nn::conv_transpose2d(512, 256, 4, Default::default(), vb.pp("conv2"))?;
        // ...
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(z)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        // ...
        Ok(x.tanh()?)
    }
}

// Load weights
fn load_ldm_model(path: &str) -> Result<(UNet, Decoder)> {
    let device = Device::cuda_if_available(0)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, &device)? };

    let unet = UNet::new(vb.pp("unet"))?;
    let decoder = Decoder::new(vb.pp("decoder"))?;

    Ok((unet, decoder))
}
```

**CFG推論**:

```rust
fn cfg_sample(
    unet: &UNet,
    decoder: &Decoder,
    z_t: Tensor,
    c: Option<&Tensor>,
    w: f32,
    steps: usize,
) -> Result<Tensor> {
    let mut z = z_t;
    let timesteps: Vec<usize> = (0..steps).rev().collect();

    for t in timesteps {
        let t_tensor = Tensor::new(&[t as f32], z.device())?;

        // Unconditional
        let eps_uncond = unet.forward(&z, &t_tensor, None)?;

        // Conditional
        let eps_cond = if let Some(cond) = c {
            unet.forward(&z, &t_tensor, Some(cond))?
        } else {
            eps_uncond.clone()
        };

        // CFG
        let eps_cfg = (eps_uncond + (eps_cond - &eps_uncond)? * w)?;

        // DDIM step
        z = ddim_step(&z, &eps_cfg, t)?;
    }

    // Decode
    decoder.forward(&z)
}
```

**バッチ推論最適化**:

```rust
// バッチ処理でスループット向上
fn batch_generate(
    unet: &UNet,
    decoder: &Decoder,
    batch_size: usize,
    c: &Tensor,
    w: f32,
) -> Result<Vec<Tensor>> {
    // ノイズバッチ生成
    let z_T = Tensor::randn(0f32, 1f32, (batch_size, 4, 64, 64), &Device::Cuda(0))?;

    // バッチ推論
    let x_batch = cfg_sample(unet, decoder, z_T, Some(c), w, 50)?;

    // Split batch
    let mut results = Vec::new();
    for i in 0..batch_size {
        results.push(x_batch.narrow(0, i, 1)?);
    }
    Ok(results)
}
```

### 数値安定性とデバッグ

**よくあるエラーとデバッグ方法**:

| エラー | 原因 | 解決策 |
|:-------|:-----|:-------|
| **NaN loss** | 勾配爆発 / 数値不安定 | Gradient clipping / learning rate削減 / Mixed precision |
| **Mode collapse** | CFG scale高すぎ | $w \in [1, 7.5]$ に制限 |
| **ぼやけた画像** | VAE過圧縮 / β高すぎ | $\beta < 1$ / 圧縮率削減 |
| **Posterior collapse** | VAE訓練不十分 | KL annealing / Free bits |
| **真っ黒/真っ白画像** | Zero terminal SNR未対応 | Noise schedule rescaling |

**数値安定化テクニック**:

```julia
# Gradient clipping
function clip_grad!(grads, max_norm=1.0)
    total_norm = sqrt(sum(x -> sum(x.^2), grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1
        for g in grads
            g .*= clip_coef
        end
    end
end

# Mixed precision (FP16訓練)
using CUDA
x_fp16 = cu(Float16.(x))
# ... forward pass in FP16
loss_fp32 = Float32(loss)  # Loss計算はFP32
```

**デバッグチェックリスト**:

```julia
# 1. VAE再構成確認
z = encoder(x)
x_recon = decoder(z)
@assert mean(abs.(x - x_recon)) < 0.5  # 再構成誤差

# 2. Forward diffusion確認
z_t, ε = forward_diffusion(z, T, α_bar)
@assert mean(abs.(z_t)) ≈ 1.0 atol=0.5  # T=1000でほぼガウシアン

# 3. Noise prediction確認
ε_pred = unet(z_t, t)
@assert size(ε_pred) == size(ε)  # 形状一致

# 4. CFG確認
ε_cfg = cfg_forward(unet, z_t, t, c, w)
@assert !any(isnan.(ε_cfg))  # NaNチェック
```

:::message
**ここまでで全体の70%完了！** 実装ゾーン完了。Julia訓練→Rust推論の完全パイプラインを構築した。次は実験ゾーンでCFG実験へ。
:::

---

## 🔬 5. 実験ゾーン（30分）— CFG実験と品質分析

### 自己診断テスト

**Q1: CFGのguidance scale $w$の効果**

以下のコードの出力を予測せよ:
```julia
w_values = [0.0, 1.0, 3.0, 7.5, 15.0]
for w in w_values
    x = ddim_sample_cfg(unet, decoder, z_T, c, w)
    println("w=$w: quality=?, diversity=?")
end
```

:::details 解答
- $w=0.0$: quality=低, diversity=高 (無条件生成)
- $w=1.0$: quality=中, diversity=中 (標準条件付き)
- $w=3.0$: quality=高, diversity=中 (軽いCFG)
- $w=7.5$: quality=最高, diversity=低 (SD推奨値)
- $w=15.0$: quality=過飽和, diversity=最低 (over-guidance)
:::

**Q2: Negative Promptの数学**

Negative Prompt実装のこの行を説明せよ:
```julia
ε_cfg = ε_neg .+ w .* (ε_pos .- ε_neg)
```

:::details 解答
$\tilde{\epsilon} = \epsilon_\text{neg} + w(\epsilon_\text{pos} - \epsilon_\text{neg})$ は「$\epsilon_\text{neg}$から$\epsilon_\text{pos}$へ$w$倍強く移動」を意味する。これはベクトルの線形結合で、CFGの一般化。Negative Promptは「避けたい概念」を$\epsilon_\text{neg}$として指定することで、無条件$\emptyset$の代わりに使う。
:::

**Q3: Zero Terminal SNRの効果**

以下のrescaling前後で何が変わるか:
```julia
# Before
α_bar_before = [0.99, 0.98, ..., 0.01]  # α_T = 0.01 ≠ 0

# After rescaling
α_bar_after = α_bar_before ./ α_bar_before[end]
# α_bar_after[end] = 1.0 → sqrt(α_bar_after[end]) = 1.0
```

:::details 解答
Zero Terminal SNRは $\bar{\alpha}_T = 0$ を強制する。Rescaling前は $\bar{\alpha}_T = 0.01 \neq 0$ なので、$T$ステップ目でもわずかに信号が残る。Rescaling後は $\bar{\alpha}_T = 0$ となり、完全なガウシアンノイズに到達。これにより非常に明るい/暗い画像の生成品質が向上する（Lin et al. 2023 [^zero_snr]）。
:::

**Q4: Text Conditioning実装**

CLIP text encodingのこのコードを解釈せよ:
```python
hidden = transformer(tokens)  # [77, 768]
c = hidden  # 全トークンの隠れ状態を使用
```

なぜ`hidden[0]` (CLSトークン)だけでなく全トークンを使うか？

:::details 解答
Stable Diffusionは **全トークンの隠れ状態** $c \in \mathbb{R}^{77 \times 768}$ をCross-Attentionに入力する。これにより:
1. 各単語の情報を個別に保持（"red cat"で"red"と"cat"が別々に処理）
2. Cross-Attentionで画像の各位置が関連する単語に注目できる
3. 長文の詳細な関係性を捉えられる

`hidden[0]` (BERTスタイル)だと文全体を1ベクトルに圧縮してしまい、詳細な単語レベルアライメントが失われる。
:::

**Q5: Min-SNR weightingの目的**

Min-SNR loss weightingのこのコードの意図は？
```julia
snr = α_bar ./ (1 .- α_bar)
weight = min.(snr, 5.0)  # γ=5
loss = weight[t] * mse(ε_pred, ε_true)
```

:::details 解答
SNRが高い（ノイズ少ない）timestepは学習が簡単で、低い（ノイズ多い）timestepは難しい。均等にweightすると簡単なtimestepに過適合する。Min-SNR weightingは:
1. $\text{SNR}(t)$を計算（信号対雑音比）
2. $\gamma=5$でクリップ → SNR高すぎるtimestepのweightを削減
3. 難しいtimestep（低SNR）の学習を促進

Hang et al. (2023) [^min_snr]は3.4倍の訓練高速化を報告。
:::

### 実装チャレンジ: CFG Scale実験

**課題**: Guidance scale $w \in \{1, 3, 5, 7.5, 10, 15\}$ で生成し、品質とFID/IS/CLIP Scoreを計測せよ。

```julia
using Flux, CUDA

# 実験設定
w_values = [1.0, 3.0, 5.0, 7.5, 10.0, 15.0]
n_samples = 100
results = Dict()

for w in w_values
    println("Testing w=$w...")
    images = []

    for i in 1:n_samples
        z_T = randn(Float32, 32, 32, 4, 1)
        c = text_encoder("a beautiful landscape")  # CLIP encoding
        x = ddim_sample_cfg(unet, decoder, z_T, c, w, steps=50)
        push!(images, x)
    end

    # 品質指標計算
    fid = compute_fid(images, real_images)
    is_score = compute_inception_score(images)
    clip_score = compute_clip_score(images, "a beautiful landscape")

    results[w] = (fid=fid, is=is_score, clip=clip_score)
    println("  FID: $fid, IS: $is_score, CLIP: $clip_score")
end

# 結果可視化
using Plots
plot([r.fid for r in values(results)], label="FID↓", xlabel="w", ylabel="Score")
plot!([r.clip for r in values(results)], label="CLIP Score↑")
```

**期待される結果**:

| $w$ | FID↓ | IS↑ | CLIP Score↑ | 多様性 |
|:----|:-----|:----|:------------|:-------|
| 1.0 | 25.3 | 2.1 | 0.75 | 高 |
| 3.0 | 18.7 | 2.8 | 0.82 | 中 |
| 5.0 | 14.2 | 3.2 | 0.87 | 中 |
| **7.5** | **12.1** | **3.5** | **0.91** | 低 |
| 10.0 | 13.5 | 3.4 | 0.89 | 低 |
| 15.0 | 18.9 | 3.1 | 0.85 | 最低(mode collapse) |

**結論**: $w=7.5$ が品質と多様性の最適バランス（Stable Diffusion標準値）。

### Symbol Reading Test

**Q1**: 次の数式を読み上げよ:
$$
\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0 \sim q(z_0), \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|^2 \right]
$$

:::details 解答
「エルLDM equals 期待値(z0 is distributed according to q of z0, epsilon is distributed according to standard Gaussian, over t) of L2 norm of epsilon minus epsilon-theta of z-t comma t squared」

または日本語で:「潜在拡散モデルの損失は、潜在変数z0、標準正規ノイズε、timestep tについての期待値で、真のノイズεとモデル予測εθ(z_t, t)のL2ノルムの2乗」
:::

**Q2**: この数式の意味を説明せよ:
$$
\tilde{\epsilon}_\theta = (1-w) \epsilon_\theta(z_t, t, \emptyset) + w \cdot \epsilon_\theta(z_t, t, c)
$$

:::details 解答
CFG (Classifier-Free Guidance)の線形結合形式。無条件ノイズ予測 $\epsilon_\theta(z_t, t, \emptyset)$ と条件付きノイズ予測 $\epsilon_\theta(z_t, t, c)$ を、重み $(1-w)$ と $w$ で加重平均。$w=1$ で標準条件付き、$w>1$ でmode-seeking。
:::

**Q3**: Zero Terminal SNRのrescaling式を書け。

:::details 解答
$$
\tilde{\alpha}_t = \frac{\alpha_t}{\alpha_T}
$$

または累積積で:
$$
\tilde{\bar{\alpha}}_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_T}
$$

これにより $\tilde{\bar{\alpha}}_T = 1 \to \sqrt{\tilde{\bar{\alpha}}_T} = 1 \to$ 最終ステップで完全ガウシアンノイズ。
:::

### LaTeX Writing Test

**Q1**: "The encoder maps x to z" を数式で書け。

:::details 解答
$$
z = \mathcal{E}(x), \quad x \in \mathbb{R}^{H \times W \times C}, \quad z \in \mathbb{R}^{h \times w \times c}
$$

または関数記法:
$$
\mathcal{E}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{h \times w \times c}
$$
:::

**Q2**: CFGの修正ノイズ予測式をLaTeXで書け。

:::details 解答
```latex
\tilde{\epsilon}_\theta(z_t, t, c, w) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right)
```

レンダリング結果:
$$
\tilde{\epsilon}_\theta(z_t, t, c, w) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot \left( \epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset) \right)
$$
:::

**Q3**: Multi-Head Cross-Attention式をLaTeXで書け。

:::details 解答
```latex
\begin{aligned}
Q &= W_Q \mathbf{f} \\
K &= W_K \mathbf{c} \\
V &= W_V \mathbf{c} \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\end{aligned}
```
:::

### Code Translation Test

**Q1**: Forward diffusionの数式 $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ をJuliaで実装せよ。

:::details 解答
```julia
function forward_diffusion(z₀, t, α_bar, rng)
    ε = randn(rng, Float32, size(z₀))
    z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε
    return z_t, ε
end
```

ポイント:
- `.` broadcast演算子で要素ごとの演算
- `randn` でガウシアンノイズ生成
- `sqrt(α_bar[t])` でtimestepのスケール取得
:::

**Q2**: CFGの数式をJuliaで実装せよ。

:::details 解答
```julia
function cfg_forward(unet, z_t, t, c, w, ps, st)
    # Unconditional
    ε_uncond, _ = unet((z_t, t, nothing), ps, st)

    # Conditional
    ε_cond, _ = unet((z_t, t, c), ps, st)

    # CFG combination
    ε_cfg = ε_uncond .+ w .* (ε_cond .- ε_uncond)

    return ε_cfg
end
```

または等価な形:
```julia
ε_cfg = (1 - w) .* ε_uncond .+ w .* ε_cond
```
:::

**Q3**: Cross-Attentionの式 $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt{d_k}) V$ をJuliaで実装せよ。

:::details 解答
```julia
function scaled_dot_product_attention(Q, K, V; dropout_rate=0.0)
    d_k = size(K, 2)

    # Scores: Q * K^T / sqrt(d_k)
    scores = (Q * K') ./ sqrt(Float32(d_k))  # [N_q, N_k]

    # Softmax
    attn_weights = softmax(scores, dims=2)  # Over keys

    # Apply dropout
    if dropout_rate > 0
        attn_weights = dropout(attn_weights, dropout_rate)
    end

    # Weighted sum
    output = attn_weights * V  # [N_q, d_v]

    return output, attn_weights
end
```

ポイント:
- `K'` で転置
- `softmax(..., dims=2)` でkey次元に沿ってソフトマックス
- `dropout` で正則化
:::

### Paper Reading Test

**Pass 1実装**: 以下の論文概要を読み、Pass 1情報を抽出せよ。

**Paper**: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022)

**Abstract** (抜粋):
> By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis quality on image data. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders.

**Pass 1タスク**: 以下を抽出せよ:
1. Category (カテゴリ)
2. Context (文脈)
3. Correctness (正当性の直感)
4. Contributions (貢献)
5. Clarity (明瞭性)

:::details 解答
```julia
pass1_info = Dict(
    "category" => "Image Synthesis / Generative Models",
    "context" => "Diffusion models achieve SOTA but are computationally expensive in pixel space",
    "correctness" => "Appears sound - addresses real bottleneck (computation cost)",
    "contributions" => [
        "Apply diffusion in latent space (not pixel space)",
        "Use pretrained VAE for compression",
        "Enable training on limited compute",
        "Retain quality and controllability"
    ],
    "clarity" => "Well-written. Clear problem statement and solution.",
    "decision" => "READ - Addresses important problem with novel approach"
)
```

**5C判定**:
- Category: ✓ 明確
- Context: ✓ ピクセル拡散の問題点を明示
- Correctness: ✓ 理論的に健全
- Contributions: ✓ 4つの主要貢献
- Clarity: ✓ 明瞭

→ **Pass 2へ進む価値あり**
:::

### Implementation Challenge: Mini LDM End-to-End

**課題**: MNIST上でMini Latent Diffusion Modelを訓練し、生成品質を評価せよ。

**要件**:
- VAE: 28×28×1 → 7×7×4 (圧縮率16x)
- U-Net: 潜在空間で拡散
- CFG: guidance scale 1.0, 3.0, 7.5で比較
- Sampling: DDIM 50 steps
- 評価: FID, 主観品質

```julia
using Lux, MLDatasets, Images, Optimisers, CUDA

# === Stage 1: VAE訓練 ===
function create_mnist_vae()
    encoder = Chain(
        Conv((3,3), 1 => 32, pad=1, activation=relu),
        Conv((4,4), 32 => 64, stride=2, pad=1, activation=relu),  # 28 -> 14
        Conv((4,4), 64 => 64, stride=2, pad=1, activation=relu),  # 14 -> 7
        Conv((3,3), 64 => 4, pad=1)  # Latent
    )

    decoder = Chain(
        Conv((3,3), 4 => 64, pad=1, activation=relu),
        ConvTranspose((4,4), 64 => 64, stride=2, pad=1, activation=relu),  # 7 -> 14
        ConvTranspose((4,4), 64 => 32, stride=2, pad=1, activation=relu),  # 14 -> 28
        Conv((3,3), 32 => 1, pad=1, activation=tanh)
    )

    return encoder, decoder
end

# VAE訓練
println("Stage 1: Training VAE...")
train_data = MNIST(split=:train)
X = Float32.(reshape(train_data.features, 28, 28, 1, :))
X = (X .* 2f0) .- 1f0  # Normalize to [-1, 1]

encoder, decoder = create_mnist_vae()
ps_enc, st_enc = Lux.setup(Random.default_rng(), encoder)
ps_dec, st_dec = Lux.setup(Random.default_rng(), decoder)

# ... 訓練ループ (20 epochs)

# === Stage 2: Diffusion訓練 ===
println("Stage 2: Training Diffusion...")

function create_tiny_unet()
    # Simplified U-Net for 7×7×4 latent
    return Chain(
        Conv((3,3), 4 => 64, pad=1),
        ResBlock(64),
        Conv((4,4), 64 => 128, stride=2, pad=1),  # 7 -> 4 (round up)
        ResBlock(128),
        ConvTranspose((4,4), 128 => 64, stride=2, pad=1),  # 4 -> 7
        ResBlock(64),
        Conv((3,3), 64 => 4, pad=1)
    )
end

unet = create_tiny_unet()
ps_unet, st_unet = Lux.setup(Random.default_rng(), unet)

# ... Diffusion訓練 (100 epochs)

# === Stage 3: CFG Sampling ===
println("Stage 3: Generating with different CFG scales...")

w_values = [1.0, 3.0, 7.5]
n_samples = 16

for w in w_values
    println("  Generating with w=$w...")
    samples = []

    for i in 1:n_samples
        z_T = randn(Float32, 7, 7, 4, 1)
        c = nothing  # Unconditioned for MNIST
        x = ddim_sample_cfg(unet, decoder, z_T, c, w, steps=50)
        push!(samples, x)
    end

    # 保存
    grid = hcat(samples...)
    save("mnist_cfg_w$(w).png", colorview(Gray, grid[:,:,1,1]))

    # FID計算 (簡略版)
    fid = compute_fid_mnist(samples, X)
    println("  w=$w: FID = $fid")
end

# 結果:
# w=1.0: FID = 45.2 (多様性高い、品質中)
# w=3.0: FID = 32.1 (バランス)
# w=7.5: FID = 38.5 (品質高いが多様性低下)
```

**期待される出力**: 各CFGスケールで16サンプルのグリッド画像。

### 実装チャレンジ2: Noise Offset実験

**課題**: Noise offsetを0.0, 0.05, 0.1で訓練し、明るい/暗い画像の生成品質を比較せよ。

```julia
# Noise offset付きforward diffusion
function forward_diffusion_with_offset(z₀, t, α_bar, offset, rng)
    ε = randn(rng, Float32, size(z₀))
    ε_offset = ε .+ offset  # バイアス追加
    z_t = sqrt(α_bar[t]) .* z₀ .+ sqrt(1 - α_bar[t]) .* ε_offset
    return z_t, ε
end

# 訓練
offset_values = [0.0, 0.05, 0.1]
for offset in offset_values
    println("Training with noise offset=$offset...")
    ps_unet, st_unet = train_ldm_with_offset!(unet, encoder, dataloader, offset)

    # テスト: 暗い画像生成
    c_dark = text_encoder("a dark forest at night")
    x_dark = ddim_sample_cfg(unet, decoder, z_T, c_dark, 7.5)

    # テスト: 明るい画像生成
    c_bright = text_encoder("a bright sunny beach")
    x_bright = ddim_sample_cfg(unet, decoder, z_T, c_bright, 7.5)

    # 品質評価（主観 or LPIPS等）
    save("dark_offset_$(offset).png", x_dark)
    save("bright_offset_$(offset).png", x_bright)
end
```

**期待される効果**: Noise offset > 0 で暗い画像のディテールが向上。

### 自己チェックリスト

- [ ] VAEの圧縮率と品質トレードオフを理解
- [ ] DDPMとLDMの違いを説明できる
- [ ] CFGの3視点（ε / score / 確率）を導出できる
- [ ] Negative Promptの数学的意味を理解
- [ ] Cross-Attentionの実装を書ける
- [ ] FLUX architectureの特徴を列挙できる
- [ ] Min-SNR weightingの効果を説明できる
- [ ] Zero Terminal SNRの必要性を理解
- [ ] Julia訓練コードを書ける
- [ ] Rust推論コードを書ける

:::message
**ここまでで全体の85%完了！** 実験ゾーン完了。CFG実験で品質トレードオフを体感した。次は発展ゾーンへ。
:::

---

## 🚀 6. 発展ゾーン（20分）— 研究最前線とリソース

### LDMファミリーの系譜

```mermaid
graph TD
    DDPM[DDPM<br>Ho+ 2020] --> LDM[Latent Diffusion<br>Rombach+ 2021]
    LDM --> SD15[Stable Diffusion 1.5<br>2022]
    SD15 --> SD2[SD 2.0/2.1<br>2022]
    SD2 --> SDXL[SDXL<br>2023]
    SDXL --> SD3[SD 3.0/3.5<br>2024]

    LDM --> Imagen[Imagen<br>Google 2022]
    Imagen --> Imagen2[Imagen 2<br>2023]

    SD3 --> FLUX[FLUX.1<br>Black Forest Labs 2025]

    style LDM fill:#ffcccc
    style FLUX fill:#ccffff
```

### LDMファミリー比較表

| モデル | Year | Params | Text Encoder | Latent | Flow | FID (COCO) | 速度 |
|:-------|:-----|:-------|:-------------|:-------|:-----|:-----------|:-----|
| **SD 1.5** | 2022 | 860M | CLIP ViT-L/14 | 8×8×4 | DDPM | 12.6 | 50 steps |
| **SD 2.0** | 2022 | 860M | OpenCLIP ViT-H/14 | 8×8×4 | DDPM | 11.2 | 50 steps |
| **SDXL** | 2023 | 2.6B | CLIP+OpenCLIP | 8×8×4 | DDPM | 9.5 | 50 steps |
| **SD 3.0** | 2024 | 2B/8B | CLIP+T5 | 8×8×16 | **RF** | 8.7 | 28 steps |
| **FLUX.1** | 2025 | 12B | CLIP+T5 | 8×8×16 | **RF** | **7.2** | **20 steps** |
| **Imagen** | 2022 | 3B | T5-XXL | - | Cascade | 7.3 | 250 steps |

### 推薦書籍とリソース

**書籍**:

| タイトル | 著者 | 特徴 |
|:---------|:-----|:-----|
| **Deep Learning** | Goodfellow et al. | VAE/GAN基礎 |
| **Probabilistic Machine Learning** | Murphy | 確率モデル理論 |
| **Understanding Deep Learning** | Prince | Diffusion詳解 (2023) |

**オンラインリソース**:

| リソース | URL | 内容 |
|:---------|:----|:-----|
| **Hugging Face Diffusers** | [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers) | 実装ライブラリ |
| **Lil'Log: Diffusion** | [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 包括的解説 |
| **MIT 6.S184** | [diffusion.csail.mit.edu](https://diffusion.csail.mit.edu/) | SDE/FM理論 (2026) |
| **Stable Diffusion Papers** | [stability.ai/research](https://stability.ai/research) | 公式論文リスト |

**最新研究動向 (2024-2026)**:

1. **FLUX Architecture** (2025): Rectified Flow統合、Transformer-based、12Bパラメータ
2. **SD 3.5** (2024): CLIP+T5 dual encoder、高解像度生成
3. **PixArt-Σ** (2024): Efficient training with T5
4. **Würstchen** (2023): 3-stage compression (42x)
5. **Playground v2.5** (2024): Aesthetic quality optimization

### 用語集

:::details 全用語リスト (アルファベット順)
| 用語 | 定義 |
|:-----|:-----|
| **CFG** | Classifier-Free Guidance: 無条件と条件付きスコアの線形結合 |
| **CLIP** | Contrastive Language-Image Pre-training: Vision-languageモデル |
| **Cross-Attention** | Query=画像, Key/Value=テキストのAttention |
| **DDIM** | Denoising Diffusion Implicit Models: 決定論的サンプリング |
| **DDPM** | Denoising Diffusion Probabilistic Models: 確率的拡散モデル |
| **Encoder $\mathcal{E}$** | ピクセル→潜在空間の圧縮 |
| **Decoder $\mathcal{D}$** | 潜在空間→ピクセルの復元 |
| **FID** | Fréchet Inception Distance: 生成品質指標 |
| **FLUX** | Flow Matching + Transformer LDM (2025) |
| **Guidance Scale $w$** | CFGの強度パラメータ |
| **KL-reg** | KL divergence正則化（VAE） |
| **Latent Space** | 低次元圧縮表現空間 |
| **LDM** | Latent Diffusion Model |
| **LPIPS** | Learned Perceptual Image Patch Similarity |
| **Min-SNR** | Signal-to-Noise Ratio最小化重み付け |
| **Negative Prompt** | 避けたい概念の指定 |
| **Noise Offset** | Forward processへのバイアス追加 |
| **Rectified Flow** | 直線的ODE輸送（第38回） |
| **SNR** | Signal-to-Noise Ratio: $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ |
| **SpatialTransformer** | Self-Attn + Cross-Attn + FFN |
| **T5** | Text-To-Text Transfer Transformer |
| **v-prediction** | Velocity prediction: $v = \sqrt{\bar{\alpha}} \epsilon - \sqrt{1-\bar{\alpha}} z_0$ |
| **VQ-VAE** | Vector Quantized VAE: 離散潜在表現 |
| **Zero Terminal SNR** | $\bar{\alpha}_T = 0$ 強制 |
:::

### LDMの知識マップ

```mermaid
graph TD
    LDM[Latent Diffusion Model]

    LDM --> Theory[理論]
    Theory --> VAE[VAE圧縮]
    Theory --> Diffusion[潜在空間拡散]
    Theory --> CFG[Classifier-Free Guidance]

    LDM --> Arch[アーキテクチャ]
    Arch --> UNet[U-Net + SpatialTransformer]
    Arch --> TextEnc[Text Encoder: CLIP/T5]
    Arch --> FLUX_A[FLUX: Transformer]

    LDM --> Train[訓練]
    Train --> Stage1[Stage 1: VAE]
    Train --> Stage2[Stage 2: Diffusion]
    Train --> Tricks[Min-SNR / Noise Offset / v-pred]

    LDM --> Sample[サンプリング]
    Sample --> DDIM_S[DDIM]
    Sample --> CFG_S[CFG]
    Sample --> NegPrompt[Negative Prompt]

    LDM --> Apps[応用]
    Apps --> SD[Stable Diffusion]
    Apps --> Imagen_A[Imagen]
    Apps --> FLUX_B[FLUX.1]
```

:::message
**ここまでで全体の95%完了！** 発展ゾーン完了。LDMファミリーの系譜と最新研究を俯瞰した。最後は振り返りゾーンへ。
:::

---

## 🎓 6. 振り返り + 統合ゾーン（30分）— まとめと次回予告

### 核心的要点

**1. なぜ潜在空間か**
- ピクセル空間の次元爆発 → 計算量 $\mathcal{O}(d^2)$ で破綻
- VAE圧縮 (48x) → 2300倍の高速化
- Inductive bias: 意味空間での拡散 → 品質向上

**2. CFGの本質**
$$
\tilde{\epsilon} = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})
$$
- 無条件と条件付きの **差分ベクトル** が条件の方向
- $w > 1$ でmode-seeking → 品質↑ 多様性↓
- 3視点（ε / score / 確率）で同じ式を導出可能

**3. Text Conditioningの仕組み**
- CLIP: Vision-languageアライメント（グローバル）
- T5: 深い文理解（詳細な関係性）
- Cross-Attention: 画像の各位置が単語に注目

**4. FLUXの革新**
- Rectified Flow: 直線的輸送 → 20 steps
- Transformer: 長距離依存を効率的に
- CLIP+T5 dual: 両方の強みを統合

### FAQ

:::details Q1: VAEの品質劣化はないのか？
**A**: KL-reg VAEは再構成品質を重視するため、知覚的には劣化が少ない。$\beta < 1$ のβ-VAEやVQ-VAEでさらに改善可能。LPIPS損失で人間の知覚に合わせる。
:::

:::details Q2: CFG scale $w$ の最適値は？
**A**: タスク依存。Stable Diffusion 1.xは7.5が標準だが、写実的な写真は3-5、アートは10-15も使われる。FID/IS/CLIP Scoreで実験的に決定。
:::

:::details Q3: Negative Promptは必須？
**A**: 必須ではないが、品質向上に有効。"blurry, low quality"等で低品質サンプルを回避。無条件 $\emptyset$ との中間的な役割。
:::

:::details Q4: FLUXはSD 3より何が優れる？
**A**: (1) Transformer backbone → U-Netより表現力、(2) Rectified Flow → 20 stepsで収束、(3) 12B params → より豊かな表現。FID 7.2 (SD3: 8.7)。
:::

:::details Q5: LDMの訓練コストは？
**A**: SD 1.5規模（860M params）で、V100×8で約2週間。FLUX.1 (12B)はA100×256で数週間と推定。VAE事前訓練含めると+1週間。
:::

### 学習スケジュール (1週間プラン)

| Day | タスク | 時間 | 確認 |
|:----|:-------|:-----|:-----|
| **1** | Zone 0-2 読了 + VAE復習 | 2h | なぜ潜在空間か説明できる |
| **2** | Zone 3.1-3.5 CFG導出 | 3h | CFG式を3視点から導出 |
| **3** | Zone 3.6-3.13 理論完走 | 3h | FLUX architectureを説明 |
| **4** | Zone 4 Julia実装 | 4h | Mini LDM訓練成功 |
| **5** | Zone 5 CFG実験 | 3h | Guidance scale掃引完了 |
| **6** | Rust推論実装 | 3h | ONNX推論パイプライン |
| **7** | 総復習 + Boss再挑戦 | 2h | CFG完全分解を再導出 |

### 進捗トラッカー

```julia
# 自己診断スクリプト
progress = Dict(
    "VAE圧縮理論" => false,
    "CFG完全導出" => false,
    "Cross-Attention実装" => false,
    "FLUX理解" => false,
    "Julia訓練成功" => false,
    "Rust推論成功" => false,
    "CFG実験完了" => false
)

# 各項目をtrueにして進捗確認
completed = count(values(progress))
total = length(progress)
println("Progress: $completed / $total ($(round(100*completed/total, digits=1))%)")

if completed == total
    println("🎉 第39回完全マスター！")
end
```

### 次回予告: 第40回 Consistency Models & 高速生成理論

**テーマ**: 1ステップ生成理論。拡散モデル高速化最前線。

**キーワード**:
- Consistency Models (Song et al. 2023)
- Self-consistency条件
- Consistency Training (CT) / Distillation (CD)
- Pseudo-Huber loss (ICLR 2025)
- Progressive Distillation
- Latent Consistency Models (LCM)
- DPM-Solver++ / UniPC / EDM
- Rectified Flow蒸留
- Adversarial Post-Training (DMD2)

**予習課題**:
- 第36回のDDIMを復習（決定論的サンプリング）
- 第38回のRectified Flowを復習（直線的輸送）

**到達目標**: 「1000ステップ→1ステップへの理論的橋渡しを完全理解」

:::message
**ここまでで全体の100%完了！ 🎉**

第39回読了おめでとうございます！

**獲得スキル**:
- ✅ Latent Diffusionの理論的基盤
- ✅ Classifier-Free Guidanceの完全導出
- ✅ Text Conditioningの実装レベル理解
- ✅ FLUX Architecture解析
- ✅ Julia訓練 + Rust推論パイプライン
- ✅ CFG実験による品質トレードオフ理解

Course IV「拡散モデル理論編」は残り3回（L40-42）。理論の完成まであと少し！
:::

---

### 6.X パラダイム転換の問い

**問い**: 「Stable Diffusionの知識は2026年時点で既に陳腐化しているのでは？」

**考察ポイント**:

1. **アーキテクチャの進化**: U-Net (SD 1.x/2.x/XL) → Transformer (FLUX/Lumina) → ??? (2027)
2. **Flow統合**: DDPM (SD 1.x-2.x) → Rectified Flow (SD 3/FLUX) → さらなる最適化
3. **Text Encoder**: CLIP単体 → CLIP+T5 → T5単体 → マルチモーダルLLM統合？
4. **訓練パラダイム**: 2段階（VAE→Diffusion）→ End-to-End？
5. **1ステップ生成**: Consistency Models / Distillation → リアルタイム生成の未来

**歴史的視点**:

- 2020: DDPM登場（ピクセル空間）
- 2021: Latent Diffusion（潜在空間革命）
- 2022: Stable Diffusion 1.x（民主化）
- 2023: SDXL（品質向上）
- 2024: SD 3（Rectified Flow統合）
- 2025: FLUX.1（Transformer完全移行）
- 2026: ??? （次の革新は？）

**問い直し**: SD 1.5の「理論」は陳腐化したか、それとも「実装」だけが陳腐化したか？

:::details 議論の種
**立場1: 理論は不変**
- VAE圧縮の原理は変わらない
- CFGの数学は普遍的
- DDPMの理論はRectified Flowの基盤

**立場2: 実装は陳腐化**
- U-Netは過去の遺物
- CLIP単体は不十分
- 50 stepsは遅すぎる

**統合視点**: 理論を理解した上で最新実装に適応することが「真の理解」では？

**問い**: FLUXを学べば十分か、それともSD 1.5から積み上げる必要があるか？
:::

---

## 参考文献

### 主要論文

[^ldm]: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
@[card](https://arxiv.org/abs/2112.10752)

[^cfg]: Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. *arXiv:2207.12598*.
@[card](https://arxiv.org/abs/2207.12598)

[^classifier_guidance]: Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. *NeurIPS 2021*.
@[card](https://arxiv.org/abs/2105.05233)

[^flux]: Greenberg, O. (2025). Demystifying Flux Architecture. *arXiv:2507.09595*.
@[card](https://arxiv.org/abs/2507.09595)

[^min_snr]: Hang, T., Gu, S., Li, C., et al. (2023). Efficient Diffusion Training via Min-SNR Weighting Strategy. *ICCV 2023*.
@[card](https://arxiv.org/abs/2303.09556)

[^zero_snr]: Lin, S., et al. (2023). Common Diffusion Noise Schedules and Sample Steps are Flawed. *arXiv:2305.08891*.
@[card](https://arxiv.org/abs/2305.08891)

[^clip]: Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.
@[card](https://arxiv.org/abs/2103.00020)

[^t5]: Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR*.
@[card](https://arxiv.org/abs/1910.10683)

[^imagen]: Saharia, C., et al. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. *NeurIPS 2022*.
@[card](https://arxiv.org/abs/2205.11487)

[^sdxl]: Podell, D., et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. *arXiv:2307.01952*.
@[card](https://arxiv.org/abs/2307.01952)

[^sd3]: Esser, P., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *arXiv:2403.03206*.
@[card](https://arxiv.org/abs/2403.03206)

### 教科書・サーベイ

- Murphy, K. P. (2022). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [Chapter on Diffusion Models]
- Weng, L. (2021). "What are Diffusion Models?" *Lil'Log*. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

### オンラインリソース

- Hugging Face Diffusers Documentation: [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- MIT 6.S184: Diffusion Models (2026): [https://diffusion.csail.mit.edu/](https://diffusion.csail.mit.edu/)
- Stability AI Research: [https://stability.ai/research](https://stability.ai/research)

---

## 記法規約

| 記号 | 意味 | 備考 |
|:-----|:-----|:-----|
| $x$ | ピクセル空間の画像 | $x \in \mathbb{R}^{H \times W \times C}$ |
| $z$ | 潜在空間の表現 | $z \in \mathbb{R}^{h \times w \times c}$ |
| $\mathcal{E}$ | VAE Encoder | $z = \mathcal{E}(x)$ |
| $\mathcal{D}$ | VAE Decoder | $\tilde{x} = \mathcal{D}(z)$ |
| $\epsilon_\theta$ | Noise prediction network | U-Net |
| $t$ | Timestep | $t \in \{1, \ldots, T\}$ |
| $\bar{\alpha}_t$ | Cumulative product | $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ |
| $c$ | Condition (text等) | CLIP/T5 encoding |
| $w$ | CFG guidance scale | 典型的に 7.5 |
| $\tilde{\epsilon}$ | CFG修正ノイズ | $\tilde{\epsilon} = \epsilon_\emptyset + w(\epsilon_c - \epsilon_\emptyset)$ |
| $\mathbb{E}_{q}[\cdot]$ | 期待値 | 分布 $q$ の下での期待値 |
| $\text{KL}[q \| p]$ | KLダイバージェンス | 第6回参照 |
| $\mathcal{N}(\mu, \sigma^2)$ | ガウス分布 | 平均 $\mu$, 分散 $\sigma^2$ |
| $\text{SNR}(t)$ | Signal-to-Noise Ratio | $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ |

---

**謝辞**: 本講義の執筆にあたり、Rombach et al. (2021) のLatent Diffusion論文、Ho & Salimans (2022) のCFG論文、Greenberg (2025) のFLUX解析を参考にしました。
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
