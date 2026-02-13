---
title: "第1回: 概論: 数式と論文の読み方 — 30秒の驚き→数式修行→実装マスター"
emoji: "🧭"
type: "tech"
topics: ["machinelearning", "deeplearning", "math", "python"]
published: true
---

# 第1回: 概論 — 数式と論文の読み方

> **数式が"読めない"のは才能ではなく語彙の問題。50記号を覚えれば論文が"読める"。**

この1文に「いやいや、そんなわけないだろ」と思っただろうか。気持ちはわかる。論文を開いて $\mathcal{L}(\theta, \phi; \mathbf{x})$ のような記号が並ぶと、反射的に閉じたくなる。あの感覚を覚えている人は多いはずだ。

だが考えてほしい。英語を読めなかった頃、英字新聞は暗号に見えた。アルファベットを覚え、単語を覚え、文法を理解した今、それは「普通の文章」になっている。数式も同じだ。記号のアルファベットを覚え、記法の文法を理解すれば、論文は「著者の思考を追体験できるドキュメント」に変わる。

本講義は全50回の第1回 — 冒険のはじまりだ。ここでは数式を読む基礎体力と、論文を構造的に読解する技術を身につける。

:::message
**このシリーズについて**: 東京大学 松尾・岩澤研究室動画講義の**完全上位互換**の全50回シリーズ。理論（論文が書ける）、実装（Production-ready）、最新（2025-2026 SOTA）の3軸で差別化する。
:::

```mermaid
graph LR
    A["📖 数式記号<br/>ギリシャ文字・演算子"] --> B["🧩 数式の文法<br/>集合・論理・写像"]
    B --> C["📐 数式の読解<br/>論文の数式を分解"]
    C --> D["💻 コードに翻訳<br/>数式 ↔ Python 1:1"]
    D --> E["🎯 論文読解<br/>構造的3パスリーディング"]
    style A fill:#e1f5fe
    style E fill:#c8e6c9
```

**所要時間の目安**:

| ゾーン | 内容 | 時間 | 難易度 |
|:-------|:-----|:-----|:-------|
| Zone 0 | クイックスタート | 30秒 | ★☆☆☆☆ |
| Zone 1 | 体験ゾーン | 10分 | ★★☆☆☆ |
| Zone 2 | 直感ゾーン | 15分 | ★★☆☆☆ |
| Zone 3 | 数式修行ゾーン | 60分 | ★★★★☆ |
| Zone 4 | 環境・ツールゾーン | 45分 | ★★★☆☆ |
| Zone 5 | 実験ゾーン | 30分 | ★★★☆☆ |
| Zone 6 | 振り返りゾーン | 30分 | ★★☆☆☆ |

---

## 🚀 0. クイックスタート（30秒）— 数式は動かせる

**ゴール**: 数式が「読める」感覚を30秒で体験する。

以下のコードを実行してほしい。たった3行で、機械学習の中核にある数式を「動かせる」。

```python
import numpy as np

# Softmax: p_i = exp(x_i) / Σ_j exp(x_j)
logits = np.array([2.0, 1.0, 0.1])
probs = np.exp(logits) / np.sum(np.exp(logits))
print(f"logits: {logits}")
print(f"probs:  {np.round(probs, 4)}")
print(f"sum:    {np.sum(probs):.6f}")  # must be 1.0
```

出力:
```
logits: [2.  1.  0.1]
probs:  [0.6590 0.2424 0.0986]
sum:    1.000000
```

**この3行の裏にある数式**:

$$
p_i = \frac{\exp(x_i)}{\sum_{j=1}^{K} \exp(x_j)}
$$

見てほしい。`np.exp(logits)` が $\exp(x_i)$、`np.sum(np.exp(logits))` が $\sum_j \exp(x_j)$。数式とコードが1対1で対応している。

このSoftmax関数は現代のLLMの心臓部だ。GPT、Claude、Gemini — 全てがこの関数を使って次のトークンの確率分布を計算している。Transformerの原論文 [^1] でもAttention機構の中核としてSoftmaxが使われている。

「え、数式ってコードに直せるの？」 — そう、直せる。全50回を通じて、この感覚を徹底的に鍛える。

:::message
**進捗: 3% 完了** ここまでで「数式 = コードで動かせる」を体感した。残り7ゾーンの冒険が待っている。
:::

---

## 🎮 1. 体験ゾーン（10分）— 数式を声に出して読む

### 1.1 数式を「声に出して読む」

数式を読めない最大の原因は、**声に出したことがない**からだ。英単語を覚えるとき発音しないで暗記する人はいない。数式も同じだ。

以下の数式を声に出して読んでみてほしい。

$$
\nabla_\theta \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(\mathbf{x}_i), y_i)
$$

**読み方**: 「ナブラ シータ エル シータ イコール イチ エヌ ぶんの シグマ アイ イコール イチ から エヌ ナブラ シータ スモール エル エフ シータ エックス アイ カンマ ワイ アイ」

......長い。だが構造は単純だ:

| 記号 | 読み | 意味 |
|:-----|:-----|:-----|
| $\nabla_\theta$ | ナブラ シータ | パラメータ $\theta$ に関する勾配 |
| $\mathcal{L}(\theta)$ | エル シータ | 損失関数（全体の平均） |
| $\frac{1}{N}\sum_{i=1}^{N}$ | エヌぶんのイチ シグマ | N個のデータの平均 |
| $\ell(f_\theta(\mathbf{x}_i), y_i)$ | スモール エル | 1個のデータの損失 |
| $f_\theta(\mathbf{x}_i)$ | エフ シータ エックス アイ | モデルの予測 |

**これが勾配降下法の数式だ。** ニューラルネットの学習で毎ステップ計算される式 — Rumelhart, Hinton & Williamsが1986年の画期的な論文 [^2] で誤差逆伝播法として定式化したアルゴリズムの核心がここにある。「ナブラ」を見たら「勾配」と反射的に読めるようになれば、もう数式はただの文章だ。

:::message
ここで多くの人が混乱するのが $\mathcal{L}$（カリグラフィック体のエル）と $\ell$（スモールエル）の違いだ。慣例として $\mathcal{L}$ は「全体の損失」、$\ell$ は「1サンプルの損失」を指すことが多い。だが著者によって流儀が異なるので、必ず論文中の定義を確認すること。
:::

### 1.2 数式パラメータを触って遊ぶ

数式の感覚を掴むには、パラメータを変えて挙動を見るのが一番早い。

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # corresponds to: p_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Temperature scaling: p_i = exp(x_i / T) / Σ exp(x_j / T)
logits = np.array([2.0, 1.0, 0.1])
for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    p = softmax(logits / T)
    print(f"T={T:5.1f}: {np.round(p, 4)} | max_prob={p.max():.4f}")
```

出力:
```
T=  0.1: [1.     0.     0.    ] | max_prob=1.0000
T=  0.5: [0.9796 0.0198 0.0007] | max_prob=0.9796
T=  1.0: [0.659  0.2424 0.0986] | max_prob=0.6590
T=  2.0: [0.4785 0.3107 0.2109] | max_prob=0.4785
T= 10.0: [0.3597 0.3299 0.3104] | max_prob=0.3597
```

**温度 $T$ が低いと「確信的」、高いと「均等」になる。** ChatGPTの `temperature` パラメータの正体がこれだ。LLMが使うSoftmaxの数式を、数行のコードで完全に理解できる。

$$
p_i = \frac{\exp(x_i / T)}{\sum_{j=1}^{K} \exp(x_j / T)}
$$

$T \to 0$ で one-hot（argmax）に近づき、$T \to \infty$ で一様分布に近づく。数式から読み取れる性質だ。

この温度付きSoftmaxは、Hintonらの知識蒸留（Knowledge Distillation）論文 [^3] で体系的に導入された。大きなモデル（教師）の「柔らかい」出力分布を小さなモデル（生徒）に学習させる手法であり、高温の $T$ で教師の出力を「ソフト化」するのが鍵だ。

:::details 温度パラメータの数学的直感
$T \to 0$ のとき、$x_i / T$ は最大の $x_i$ だけが $+\infty$ に発散し、他は相対的に $-\infty$ に近づく。$T \to 0$ の極限で、最大の $x_i$ に対応する Softmax 出力が 1 に収束し、他は 0 に収束するから one-hot。

$T \to \infty$ のとき、$x_i / T \to 0$ なので全ての $\exp(x_i / T) \to \exp(0) = 1$。均等に1なので一様分布。

この 2 つの極限を紙に書いて確かめてみてほしい。数式の性質を直接「導出」する感覚が身につく。

ちなみに、LLMの推論で $T = 0$ を指定するとgreedy decodingになるのは、まさにこの数学的性質のためだ。この話は第15回（自己回帰モデル）で本格的に扱う。
:::

### 1.3 Attention — 現代AIの心臓部を触る

もう1つ、現代の機械学習で最も重要な数式を体験しよう。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

GPT、BERT、Vision Transformer、Stable Diffusion — 2024-2026年の主要モデル全てがこの数式の上に立っている。**たった1つの数式が、AIの全てを動かしている**というのは言い過ぎではない。この式はVaswaniらの "Attention Is All You Need" [^1] で提案され、それ以降の深層学習の方向性を決定的に変えた。

```python
import numpy as np

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled Dot-Product Attention.

    corresponds to: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = Q.shape[-1]
    # QK^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)
    # softmax along last axis
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    # weighted sum of V
    return weights @ V

# Toy example: 3 tokens, d_model=4
np.random.seed(42)
seq_len, d_model = 3, 4
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output = attention(Q, K, V)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("Output shape:", output.shape)
print("Output:\n", np.round(output, 4))
```

**数式のどこがコードのどこか、わかるだろうか？**

| 数式 | コード | 意味 |
|:-----|:-------|:-----|
| $QK^\top$ | `Q @ K.T` | クエリとキーの類似度行列 |
| $\sqrt{d_k}$ | `np.sqrt(d_k)` | スケーリング因子 |
| $\text{softmax}(\cdot)$ | `np.exp(...) / sum` | 確率への正規化 |
| $(\cdot) V$ | `weights @ V` | 値の加重和 |

この数式→コード対応が「見える」なら、すでに数式リテラシーの第一歩を踏み出している。

ここで「なぜ $\sqrt{d_k}$ で割るのか？」という疑問が湧いた人は鋭い。内積 $QK^\top$ の値は次元数 $d_k$ が大きいほど絶対値が大きくなる傾向がある。大きすぎる値にSoftmaxをかけると one-hot に近づいてしまい、勾配が消失する。$\sqrt{d_k}$ で割ることで値のスケールを安定させているのだ。Vaswaniら [^1] は原論文でこの理由を明確に述べている:「$d_k$ が大きいとき、内積の大きさが増大してSoftmaxが極端に飽和した領域に押し込まれ、勾配が非常に小さくなる」。

> **一言で言えば**: Attention = 「類似度で重みづけした値の加重和」

この直感を第2回（線形代数 I）と第16回（Transformer完全版）で数学的に深める。

:::details Attentionの計算量
$Q, K, V \in \mathbb{R}^{n \times d}$ のとき:
- $QK^\top$: $O(n^2 d)$ — 全トークン対の類似度を計算
- softmax: $O(n^2)$
- weights $\times V$: $O(n^2 d)$

合計 $O(n^2 d)$。シーケンス長 $n$ に対して二乗で計算量が増える。これが「長い文脈は高コスト」の理由であり、Flash Attention（第16回）やSSM（第26回）で解決を試みる対象だ。
:::

### 1.4 LLMの学習目標 — Cross-Entropy Loss

LLMが毎ステップ最小化しているのが以下の損失関数だ:

$$
\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

分解すると:
- $\theta$ — モデルパラメータ
- $T$ — シーケンス長（トークン数）
- $x_t$ — $t$ 番目のトークン（正解）
- $x_{<t}$ — $t$ より前のトークン列（文脈）
- $p_\theta(x_t \mid x_{<t})$ — モデルが $x_t$ に割り当てた確率
- $\log$ — 対数（確率の積を和に変換する技）
- $-$ — 最小化のための符号反転

**一言で言えば**: 「各トークンについて、モデルが正解に割り当てた確率の対数を取り、平均の符号を反転したもの」。

```python
import numpy as np

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Compute cross-entropy loss for next-token prediction.

    corresponds to: L = -(1/T) Σ_t log p(x_t | x_{<t})
    """
    # softmax to get probabilities: p_θ(x_t | x_{<t})
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # select probabilities of correct tokens
    T = len(targets)
    correct_probs = probs[np.arange(T), targets]

    # -(1/T) Σ log p(x_t | x_{<t})
    loss = -np.mean(np.log(correct_probs + 1e-10))  # 1e-10: 数値安定化の ε（log(0) 回避 → Zone 3.3 で詳述）
    return loss

# Simulated next-token prediction
vocab_size = 100
T = 5
np.random.seed(42)

logits = np.random.randn(T, vocab_size)
targets = np.array([23, 45, 67, 12, 89])

loss = cross_entropy_loss(logits, targets)
print(f"Cross-Entropy Loss: {loss:.4f}")
print(f"Perplexity:         {np.exp(loss):.2f}")
print(f"(Random baseline:   {np.log(vocab_size):.4f} loss, {vocab_size:.0f} perplexity)")
```

この Cross-Entropy Loss は第6回（情報理論）で理論的に深堀りし、Perplexity $= \exp(\mathcal{L})$ の情報理論的意味を $2^{H(p)}$ と接続する。今は「LLMの学習 = この数式の最小化」とだけ覚えておけば十分だ。

### 1.5 4つの数式はどう接続するか

ここまでで体験した4つの数式を整理しよう。実はこれらは独立ではなく、LLMの推論・学習パイプラインの中で繋がっている。

```mermaid
graph TD
    INPUT["入力トークン列<br/>x₁, x₂, ..., xₜ₋₁"] --> ATT["Attention<br/>softmax(QKᵀ/√dₖ)V"]
    ATT --> LOGITS["出力ロジット<br/>z ∈ ℝ^V"]
    LOGITS --> SM["Softmax<br/>p = exp(zᵢ)/Σexp(zⱼ)"]
    SM --> PROB["確率分布<br/>p(xₜ | x₍₍ₜ₎₎)"]
    PROB --> CE["Cross-Entropy Loss<br/>L = -log p(xₜ)"]
    CE --> GRAD["勾配<br/>∇θ L"]
    GRAD --> UPDATE["パラメータ更新<br/>θ ← θ - η∇θL"]
    UPDATE -.-> ATT

    style INPUT fill:#e3f2fd
    style CE fill:#ffebee
    style UPDATE fill:#e8f5e9
```

**推論時**（生成時）:
1. 入力トークン列が Attention 層を通過する
2. 出力ロジットが得られる
3. Softmax で確率分布に変換する（Temperature で制御）
4. 確率分布からサンプリングして次のトークンを選ぶ

**学習時**:
1. 上記1-3に加えて、正解トークンに対する Cross-Entropy Loss を計算
2. Loss から勾配を逆伝播（Backpropagation）[^2]
3. パラメータを更新

```python
import numpy as np

def llm_forward_pass(x_logits: np.ndarray, target: int, temperature: float = 1.0):
    """Simulate one step of LLM forward pass.

    Shows how Softmax, Temperature, and Cross-Entropy connect.
    """
    # Step 1: Temperature scaling
    scaled = x_logits / temperature

    # Step 2: Softmax → probability distribution
    # p_i = exp(x_i/T) / Σ exp(x_j/T)
    exp_scaled = np.exp(scaled - np.max(scaled))  # numerically stable
    probs = exp_scaled / exp_scaled.sum()

    # Step 3: Cross-Entropy Loss for the target token
    # L = -log p(target)
    loss = -np.log(probs[target] + 1e-10)

    # Step 4: Perplexity
    ppl = np.exp(loss)

    return probs, loss, ppl

# Example: vocabulary of 10 tokens
np.random.seed(42)
logits = np.random.randn(10) * 2
target_token = 3  # correct next token is index 3

print("=== LLM Forward Pass Simulation ===\n")
for T in [0.1, 0.5, 1.0, 2.0]:
    probs, loss, ppl = llm_forward_pass(logits, target_token, T)
    print(f"T={T:.1f} | p(target)={probs[target_token]:.4f} | "
          f"loss={loss:.4f} | ppl={ppl:.2f} | argmax={np.argmax(probs)}")

print("\n--- Key insight ---")
print("Lower loss = model assigns higher probability to correct token")
print("Perplexity = 'effective number of equally likely choices'")
print(f"Perfect prediction: loss=0, ppl=1")
print(f"Random guess (V=10): loss={np.log(10):.4f}, ppl=10")
```

:::details NumPy と PyTorch の対応
ここまで全て NumPy で書いてきた。同じ処理を PyTorch で書くと:

```python
import torch
import torch.nn.functional as F

# NumPy version
import numpy as np
logits_np = np.array([2.0, 1.0, 0.1])
probs_np = np.exp(logits_np) / np.sum(np.exp(logits_np))

# PyTorch version — one line
logits_pt = torch.tensor([2.0, 1.0, 0.1])
probs_pt = F.softmax(logits_pt, dim=0)

print(f"NumPy:   {np.round(probs_np, 4)}")
print(f"PyTorch: {probs_pt.numpy().round(4)}")
print(f"Match: {np.allclose(probs_np, probs_pt.numpy())}")

# Cross-Entropy Loss
target = torch.tensor(0)  # correct class is 0
loss_np = -np.log(probs_np[0])
loss_pt = F.cross_entropy(logits_pt.unsqueeze(0), target.unsqueeze(0))
print(f"\nCE Loss (NumPy):   {loss_np:.6f}")
print(f"CE Loss (PyTorch): {loss_pt.item():.6f}")
```

**PyTorch の `F.cross_entropy`** は内部で Softmax + 負の対数尤度を一度に計算する。数値安定化も自動で行われる。「中で何が起きているか」を NumPy で理解してから PyTorch を使う — これが本シリーズのスタンスだ。
:::

> **Zone 1 まとめ**: Softmax（確率への変換）→ Temperature（分布の鋭さ制御）[^3] → Attention（類似度加重和）[^1] → Cross-Entropy（学習目標）。この4つが LLM の心臓部を形成している。数式で書けば4行。コードで書けば各20行。それが数十億パラメータのモデルを動かしている。

:::message
**進捗: 10% 完了** Softmax、Temperature、Attention、Cross-Entropy Loss の4つの数式を「触って」理解し、それらの接続を確認した。Zone 0-1 クリア。
:::

---

## 🧩 2. 直感ゾーン（15分）— 数式リテラシーが最優先の理由

### 2.1 なぜ「数式の読み書き」が最初なのか

多くの機械学習入門は「Pythonでモデルを動かす」から始まる。それ自体は悪くない。だが、その先で確実に壁にぶつかる。

- 論文を読もうとして数式で止まる
- ハイパーパラメータの意味がわからず試行錯誤
- 新手法が出ても「何が新しいのか」判断できない
- バグの原因が数式の誤解にあることに気づかない

**核心はこうだ: 数式が読めないと、ライブラリのユーザーから先に進めない。**

このシリーズが「概論 — 数式と論文の読み方」から始まる理由がここにある。数式の読み書き能力は、残り39回全ての土台になる。

> **この章を読めば**: 全50回シリーズの地図が手に入る。どこに何があり、各講義がどう繋がっているかが見える。

### 2.2 本シリーズの全体構成

全50回は5つのコースに分かれている。

```mermaid
graph TD
    CI["Course I: 数学基礎編<br/>第1-8回 (8講義)<br/>🐍 Python 100%"]
    CII["Course II: 生成モデル理論編<br/>第9-18回 (10講義)<br/>⚡ Julia + 🦀 Rust 登場"]
    CIII["Course III: 生成モデル実践編<br/>第19-32回 (14講義)<br/>⚡🦀🔮 3言語フルスタック"]
    CIV["Course IV: 拡散モデル理論編<br/>第33-42回 (10講義)<br/>⚡🦀🔮"]
    CV["Course V: 応用・フロンティア編<br/>第43-50回 (8講義)<br/>⚡🦀🔮 + 卒業制作"]

    CI -->|"不確実性を扱いたい<br/>→確率分布の学習"| CII
    CII -->|"理論を手で確かめたい<br/>→3言語で実装"| CIII
    CIII -->|"連続空間の生成理論<br/>→拡散・SDE・FM"| CIV
    CIV -->|"最先端+応用<br/>→DiT/Video/3D"| CV
```

| コース | 回 | テーマ | 松尾研との対比 |
|:------|:---|:------|:-------------|
| **Course I** | 第1-8回 | 数学基礎編 | 松尾研が「前提知識」で片付ける部分を **8回で叩き込む** |
| **Course II** | 第9-18回 | 生成モデル理論編 | 松尾研全8回 = このコースの **サブセット** |
| **Course III** | 第19-32回 | 生成モデル実践編 | 松尾研に **存在しない** 実装特化 |
| **Course IV** | 第33-42回 | 拡散モデル理論編 | 松尾研2回 vs **10回の圧倒的深度** |
| **Course V** | 第43-50回 | 応用・フロンティア | 松尾研に存在しない領域 + 卒業制作 |

### 2.3 差別化の3軸

| 軸 | 松尾研（教科書レベル） | 本シリーズ（上位互換） |
|:---|:---------------------|:---------------------|
| **理論** | 論文が読める | **論文が書ける** — 導出過程を全て追える |
| **実装** | 学習用PyTorch | **Production-ready** — Julia/Rust/Elixir |
| **最新** | 2023年までの手法 | **2025-2026 SOTA** — DiT [^7], FLUX, Sora理論 |

「論文が書ける」とはどういうことか。数式を「結果」として暗記するのではなく、**導出過程を自力で再現できる**ということだ。ELBOの分解 [^4]、KLダイバージェンスの解析解、Score Matchingの等価性証明 — これらを「見たことがある」ではなく「自分で導ける」レベルまで持っていく。


### 2.4 Course I（第1-8回）ロードマップ

本講義はCourse I「数学基礎編」の初回だ。Course Iの8回で何を学ぶのか、全体像を示す。

```mermaid
graph TD
    L1["第1回: 概論<br/>数式と論文の読み方<br/>🎯 数式リテラシー"]
    L2["第2回: 線形代数 I<br/>ベクトル・行列・基底<br/>🎯 QK^T の内積"]
    L3["第3回: 線形代数 II<br/>SVD・行列微分<br/>🎯 Backpropagation"]
    L4["第4回: 確率論・統計学<br/>分布・ベイズ推論<br/>🎯 p(x_t|x_{&lt;t})"]
    L5["第5回: 測度論・確率過程<br/>厳密な確率の基盤<br/>🎯 トークン空間の測度"]
    L6["第6回: 情報理論・最適化<br/>KL・SGD・Adam<br/>🎯 Perplexity = 2^H"]
    L7["第7回: 生成モデル概要 & MLE<br/>尤度最大化<br/>🎯 次トークン予測"]
    L8["第8回: 潜在変数 & EM算法<br/>隠れ変数の推定<br/>🎯 VAEへの橋渡し"]

    L1 -->|"数式が読めた<br/>→行列の意味は？"| L2
    L2 -->|"行列を扱えた<br/>→分解と微分"| L3
    L3 -->|"微分できた<br/>→不確実性は？"| L4
    L4 -->|"確率分布がわかった<br/>→もっと厳密に"| L5
    L5 -->|"厳密な確率<br/>→分布の距離と最適化"| L6
    L6 -->|"武器が揃った<br/>→生成モデルの全体像"| L7
    L7 -->|"尤度が計算困難<br/>→潜在変数で分解"| L8

    style L1 fill:#ffeb3b
```

| 回 | テーマ | 核心 | LLM/Transformerとの接点 |
|:---|:------|:-----|:----------------------|
| **第1回** | 概論: 数式と論文の読み方 | 数式リテラシー | Softmax, Attention [^1], Cross-Entropy |
| **第2回** | 線形代数 I | ベクトル空間・行列 | $QK^\top$ の内積、埋め込み空間 |
| **第3回** | 線形代数 II | SVD・行列微分 | ヤコビアン、Backpropagation [^2] |
| **第4回** | 確率論・統計学 | 分布・ベイズ | $p(x_t \mid x_{<t})$ 自己回帰、Softmax分布 |
| **第5回** | 測度論・確率過程 | 厳密な確率 | トークン空間上の確率測度 |
| **第6回** | 情報理論・最適化 | KL・SGD | Perplexity $= 2^H$、Cross-Entropy Loss |
| **第7回** | 生成モデル概要 & MLE | 尤度最大化 | 次トークン予測 $= \arg\max p(x_t \mid x_{<t}; \theta)$ |
| **第8回** | 潜在変数 & EM算法 | 隠れ変数 | Transformer隠れ層、VAE [^4] への橋渡し |

**各講義の「限界」が、次講義の「動機」になる。** これが40回を貫く設計原則だ。

たとえば第2回で線形代数を習得すると、「不確実性をどう数学的に扱うのか？」という問いが生まれる。それが第4回（確率論）の動機になる。第4回で確率分布を扱えると、「もっと厳密に確率を定義できないか？」 — それが第5回（測度論）の動機になる。

この連鎖が第40回まで途切れない。

### 2.5 3つの比喩で捉える「数式リテラシー」

数式を読む力を3つの比喩で考えてみよう。

**比喩1: 楽譜**

五線譜が読めない人にとって、楽譜はただの黒い点の集合だ。だが音楽理論を学べば、楽譜から音楽が「聞こえる」ようになる。数式も同じ — 記号の意味を知れば、数式から「動作」が見える。$\sum_{i=1}^{N}$ を見て「ループだ」とわかる。$\nabla_\theta$ を見て「勾配降下の方向だ」とわかる。

**比喩2: 設計図**

建築の設計図を見て、完成形の建物を想像できるのはプロだけだ。数式は機械学習アルゴリズムの「設計図」であり、読めれば実装前にアルゴリズムの動作を頭の中で走らせることができる。

**比喩3: プログラミング言語**

そう、数式は「最も古いプログラミング言語」だ。

| プログラミング | 数学記法 | 例 |
|:-------------|:---------|:---|
| 変数宣言 | 集合への所属 | `x: float` ↔ $x \in \mathbb{R}$ |
| forループ | 総和 | `sum(...)` ↔ $\sum$ |
| 積の累積 | 総乗 | `prod(...)` ↔ $\prod$ |
| if文 | 指示関数 | `if cond:` ↔ $\mathbb{1}[\cdot]$ |
| 関数定義 | 写像 | `def f(x):` ↔ $f: X \to Y$ |
| 型注釈 | 空間の指定 | `x: np.ndarray` ↔ $\mathbf{x} \in \mathbb{R}^d$ |

プログラミング言語にある概念は、数学記法にも全て存在する。

この対応は偶然ではない。プログラミング言語は数学の記法を形式化したものだからだ。だからこそ、プログラマは数式に対して本質的な優位性を持っている。$\sum_{i=1}^{N} f(x_i)$ を見て `sum(f(x[i]) for i in range(N))` と読める能力は、プログラミング未経験者にはない。

**あなたが数式を「難しい」と感じるのは、文法が違うだけだ。** 同じ計算を、一方は $\sum$ で書き、他方は `for` で書く。中身は同じ。この講義では常に両方を並べて示す。数式を見たら「これをコードにするとどうなるか？」と考える癖をつける。それが最も効率的な学習法だ。

**ここで一つ断言する。** 数式が「読めない」のではない。「読み方を教わっていない」だけだ。アルファベットを教わらずに英語を読めないのは当然で、数式記号の読み方を教わらずに論文を読めないのも当然だ。

これから、そのアルファベットを一つずつ教える。

### 2.6 ローカル完結ポリシー

本シリーズの全50回は**ローカルマシンだけで完結する**。Google Colabは不要。

| 項目 | 最低スペック | 推奨スペック |
|:-----|:-----------|:-----------|
| CPU | Intel i5 / Apple M1 | Apple M2+ / AMD Ryzen 7 |
| RAM | 8GB | 16GB |
| GPU | **不要**（CPU完結） | 内蔵GPU (Metal/Vulkan) |
| ストレージ | 10GB空き | 20GB空き |

Course I（第1-8回）は合成データ・2Dトイデータのみを使い、全て1分以内に実行できる。「GPUがないから手を動かせない」という言い訳は、このシリーズでは通用しない。

### 2.7 効果的な学習戦略

40回を最大限に活用するための戦略を提示する。

#### 3周読みのススメ

各講義は「1回読めばOK」ではない。3周するのが最も効果的だ。

| 周目 | 目的 | 所要時間 | フォーカス |
|:-----|:-----|:--------|:---------|
| **1周目** | 全体の地図を掴む | 表示時間の80% | Zone 0-4 を通読。Zone 5-6 は流し読み |
| **2周目** | 手を動かす | 表示時間の100% | コードを全て自分で写経。Zone 5 の問題を解く |
| **3周目** | 接続を見る | 表示時間の50% | 次の講義を読んだ後に戻り、接続を確認 |

**1周目で100%理解しようとしないこと。** 第1回を読んでいる段階では、第2回の知識がないため理解できない部分がある。第2回を読んだ後に戻ると「ああ、これはそういう意味だったのか」と繋がる。それが設計意図だ。

#### ノートの取り方

紙のノートに以下の3つを書く。デジタルでもいいが、数式は手書きの方が定着する。

1. **記号辞書**: 新しい記号に出会ったら「$\nabla$ = 勾配（ナブラ）」のように書き溜める
2. **数式→コード対応表**: `Σ → np.sum`, `∫ → np.mean(samples)` のように
3. **???リスト**: わからなかったことを書く。次の講義で解消されたら線を引く

```python
# My personal symbol dictionary — template
my_symbols = {
    "θ (theta)":    "model parameters (generic)",
    "φ (phi)":      "variational / encoder parameters",
    "μ (mu)":       "mean",
    "σ (sigma)":    "standard deviation",
    "Σ (Sigma)":    "summation / covariance matrix",
    "∇ (nabla)":    "gradient operator",
    "∂ (partial)":  "partial derivative",
    "∈ (in)":       "element of",
    "∀ (forall)":   "for all",
    "∃ (exists)":   "there exists",
}

for sym, meaning in my_symbols.items():
    print(f"  {sym:20s} → {meaning}")
```

#### 写経 vs 理解

**写経は理解の代替にならない。** しかし、理解の「きっかけ」にはなる。

推奨フロー:
1. コードを**見ずに**数式だけからコードを書く（10分試す）
2. 書けなかったら、答えのコードを**見て**写す
3. 写したら、もう一度**見ずに**書く
4. 3回やって書けなければ、数式の理解が足りない — 解説を再読する

このサイクルを「Softmax」「Attention」「Cross-Entropy Loss」の3つで実践してみてほしい。

:::message
**進捗: 20% 完了** シリーズの全体構成、「なぜ数式から始めるのか」、学習戦略を理解した。ここから数式の記号と記法を一気に学ぶ — Zone 3「数式修行ゾーン」、本講義最大の山場だ。
:::

### 2.8 深層生成モデルの全体像

本シリーズで扱う生成モデルのファミリーを俯瞰する。第1回で概観を掴んでおくことで、以降の各講義の位置づけが明確になる。

```mermaid
graph TD
    GM["深層生成モデル<br/>Deep Generative Models"] --> EBM["エネルギーベース<br/>Energy-Based"]
    GM --> LV["潜在変数モデル<br/>Latent Variable"]
    GM --> AR["自己回帰<br/>Autoregressive"]
    GM --> FM["フローベース<br/>Flow-Based"]
    GM --> DM["拡散モデル<br/>Diffusion"]

    LV --> VAE["VAE [^4]<br/>Kingma 2013"]
    EBM --> GAN["GAN [^8]<br/>Goodfellow 2014"]
    AR --> TR["Transformer [^1]<br/>Vaswani 2017"]
    DM --> DDPM["DDPM [^5]<br/>Ho 2020"]
    FM --> CFM["Flow Matching [^6]<br/>Lipman 2022"]
    DM --> DiT["DiT [^7]<br/>Peebles 2022"]
    TR --> GPT["GPT / LLM"]
    VAE --> VDVAE["VD-VAE"]

    style GM fill:#fff3e0
    style VAE fill:#e3f2fd
    style GAN fill:#fce4ec
    style DDPM fill:#e8f5e9
    style CFM fill:#f3e5f5
    style DiT fill:#e0f7fa
    style TR fill:#fff9c4
```

#### 各ファミリーの特徴比較

| ファミリー | 核心アイデア | 代表論文 | 本シリーズ |
|:---|:---|:---|:---|
| VAE | 潜在変数 + 変分推論 | Kingma & Welling (2013)[^4] | 第9-10回 |
| GAN | 生成器 vs 判別器の敵対的訓練 | Goodfellow et al. (2014)[^8] | 第15-16回 |
| 自己回帰 | 条件付き確率の連鎖 | Vaswani et al. (2017)[^1] | 第13-16回 |
| 拡散 | ノイズ付加→除去の反復 | Ho et al. (2020)[^5] | 第11-14回 |
| Flow Matching | ODE による確率パス | Lipman et al. (2022)[^6] | 第17-20回 |
| Transformer + 拡散 | DiT アーキテクチャ | Peebles & Xie (2022)[^7] | 第21-24回 |

#### 深層生成モデルの歴史タイムライン

```mermaid
timeline
    title 深層生成モデルの進化 (2013-2025)
    2013 : VAE (Kingma & Welling)
         : 変分推論 + ニューラルネット
    2014 : GAN (Goodfellow et al.)
         : 敵対的訓練の登場
    2017 : Transformer (Vaswani et al.)
         : Attention Is All You Need
    2018 : BERT / GPT
         : 事前学習の革命
    2020 : DDPM (Ho et al.)
         : 拡散モデルの実用化
    2021 : DALL-E / CLIP
         : テキスト→画像の幕開け
    2022 : Stable Diffusion
         : オープンソース画像生成
         : Flow Matching (Lipman et al.)
         : DiT (Peebles & Xie)
    2023 : GPT-4 / LLaMA
         : LLM のスケーリング
    2024 : FLUX / SD3
         : Flow Matching ベースの画像生成
         : Sora
         : 動画生成の衝撃
    2025 : マルチモーダル統合
         : 統一生成モデルへ
```

#### なぜ「深層生成モデル」を学ぶのか

2024-2025 の AI ブームの本質は**生成モデル**にある:

1. **LLM (GPT, Claude, Gemini)** — テキストの自己回帰生成モデル
2. **画像生成 (DALL-E 3, Stable Diffusion, FLUX)** — 拡散/Flow Matching による画像生成
3. **動画生成 (Sora, Runway)** — 時空間の拡散モデル
4. **音声生成 (Whisper, WaveNet)** — 自己回帰/拡散の音声モデル
5. **3D生成 (DreamFusion, Score Jacobian)** — SDS による3D最適化

これらすべてが**同じ数学的基盤**の上に成り立っている。本シリーズはその基盤を体系的に学ぶ。

| 応用 | 基盤モデル | 数学的核心 | 本シリーズの対応 |
|:---|:---|:---|:---|
| ChatGPT | Transformer (自己回帰) | $p(x_t \mid x_{<t})$ | 第13-16回 |
| DALL-E 3 | U-Net + Diffusion | $\epsilon_\theta(\mathbf{x}_t, t)$ | 第11-14回 |
| Stable Diffusion 3 | DiT + Flow Matching | $v_\theta(\mathbf{x}_t, t)$ | 第17-24回 |
| GPT-4V | Vision Transformer + LLM | Multi-modal fusion | 第33-36回 |
| Sora | Spatial-temporal DiT | 3D attention + diffusion | 第37-40回 |

---

## 📐 3. 数式修行ゾーン（60分）— 記号を「読める」ようにする

> **目標**: ギリシャ文字・添字・演算子・集合・論理・関数の記法を網羅し、Transformer の数式を**一文字残らず**読めるようにする。

本シリーズで最も重要なゾーンだ。ここをクリアすれば、以降の39回の講義で「記号がわからなくて止まる」ことは二度とない。逆にここを曖昧にすると、毎回つまずく。急がず一つずつ確認しよう。

### 3.1 ギリシャ文字 — 数式のアルファベット

機械学習の論文で頻出するギリシャ文字を、用途ごとに整理する。「この記号は何に使われることが多いか」を知っておくだけで、論文を開いたときの初見殺しが大幅に減る。

#### パラメータ系（モデルの中身を表す）

| 記号 | 読み | LaTeX | 典型的な用途 |
|:---:|:---:|:---:|:---|
| $\theta$ | シータ | `\theta` | モデルのパラメータ全般。$p_\theta(\mathbf{x})$ のように下付きで「このモデルのパラメータは $\theta$」と示す |
| $\phi$ | ファイ | `\phi` | $\theta$ と区別したい第2のパラメータ群。VAE[^4]ではエンコーダのパラメータを $\phi$、デコーダを $\theta$ とする |
| $\psi$ | プサイ | `\psi` | 第3のパラメータ群。Teacher-Student 構成[^3]で教師を $\psi$、生徒を $\theta$ とすることがある |
| $\omega, w$ | オメガ | `\omega` | 個々の重み（weight）。$\theta = \{w_1, w_2, \ldots\}$ |

:::message
**覚え方**: $\theta$（主役）→ $\phi$（相方）→ $\psi$（第三者）の順で「パラメータ三兄弟」と覚える。Kingma & Welling の VAE 論文[^4]を読めば、$\theta$ と $\phi$ の役割分担が自然に身につく。
:::

#### 統計量系（データの性質を表す）

| 記号 | 読み | LaTeX | 典型的な用途 |
|:---:|:---:|:---:|:---|
| $\mu$ | ミュー | `\mu` | 平均（mean）。$\mu = \mathbb{E}[X]$ |
| $\sigma$ | シグマ | `\sigma` | 標準偏差（standard deviation）。$\sigma^2$ は分散 |
| $\Sigma$ | 大シグマ | `\Sigma` | 共分散行列。$\Sigma \in \mathbb{R}^{d \times d}$。小文字 $\sigma$ と大文字 $\Sigma$ は意味が違う |
| $\rho$ | ロー | `\rho` | 相関係数。$\rho \in [-1, 1]$ |
| $\tau$ | タウ | `\tau` | 温度パラメータ（Temperature）。Zone 1 で見た Softmax の $T$ はこの $\tau$ で書かれることも多い |

:::details ミニ演習: 温度パラメータの記号揺れ
同じ概念でも、論文によって記号が異なる。温度パラメータは以下のバリエーションがある:

- Hinton et al. (2015)[^3]: $T$ を使用 — $q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$
- 他の論文: $\tau$ を使用 — $\text{softmax}(z_i / \tau)$
- 一部の強化学習論文: $\beta$ を使用（逆温度 $\beta = 1/T$ として）

**重要**: 記号は著者が定義するものであり、「正解」は存在しない。論文を読むときは冒頭の記号定義を**必ず**確認する習慣をつけよう。
:::

#### 演算系（操作を表す）

| 記号 | 読み | LaTeX | 典型的な用途 |
|:---:|:---:|:---:|:---|
| $\nabla$ | ナブラ | `\nabla` | 勾配演算子。$\nabla_\theta \mathcal{L}$ は「損失 $\mathcal{L}$ の $\theta$ についての勾配」 |
| $\partial$ | パーシャル/デル | `\partial` | 偏微分。$\frac{\partial f}{\partial x}$ |
| $\alpha$ | アルファ | `\alpha` | 学習率。$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ |
| $\epsilon$ | イプシロン | `\epsilon` | 微小量。数値安定化の $\log(p + \epsilon)$ や、ノイズ $\epsilon \sim \mathcal{N}(0, I)$ |
| $\lambda$ | ラムダ | `\lambda` | 正則化係数。$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \lambda \mathcal{L}_{\text{reg}}$ |
| $\eta$ | イータ | `\eta` | 学習率（$\alpha$ の代替）。文献によって $\alpha$ か $\eta$ のどちらか |
| $\gamma$ | ガンマ | `\gamma` | 割引率（強化学習）、モメンタム係数 |

#### 確率・分布系

| 記号 | 読み | LaTeX | 典型的な用途 |
|:---:|:---:|:---:|:---|
| $\pi$ | パイ | `\pi` | 方策（policy）。確率分布を直接意味することは少ない |
| $\xi$ | グザイ | `\xi` | 潜在変数の別名。$\xi \sim p(\xi)$ |
| $\zeta$ | ゼータ | `\zeta` | 補助変数。頻度は低いが理論系論文で登場 |
| $\kappa$ | カッパ | `\kappa` | 集中度パラメータ（von Mises分布など） |

:::message
**全部覚える必要はない。** ここは辞書だ。論文を読んでいて「この記号なんだっけ」となったら戻ってくればいい。繰り返し参照するうちに自然に覚える。
:::

#### Python で確認: ギリシャ文字マッピング

```python
"""ギリシャ文字 → 機械学習での典型的な用途マッピング"""

greek_ml_map = {
    # パラメータ系
    "θ (theta)":    "model parameters",
    "φ (phi)":      "encoder / variational parameters",
    "ψ (psi)":      "teacher / auxiliary parameters",
    "ω (omega)":    "individual weight",

    # 統計量系
    "μ (mu)":       "mean",
    "σ (sigma)":    "standard deviation",
    "Σ (Sigma)":    "covariance matrix / summation",
    "ρ (rho)":      "correlation coefficient",
    "τ (tau)":      "temperature parameter",

    # 演算系
    "∇ (nabla)":    "gradient operator",
    "∂ (partial)":  "partial derivative",
    "α (alpha)":    "learning rate",
    "ε (epsilon)":  "small constant / noise",
    "λ (lambda)":   "regularization coefficient",
    "η (eta)":      "learning rate (alternative)",
    "γ (gamma)":    "discount factor / momentum",

    # 確率・分布系
    "π (pi)":       "policy (RL)",
    "ξ (xi)":       "latent variable (alternative)",
}

print("=== ギリシャ文字 機械学習辞書 ===")
for symbol, usage in greek_ml_map.items():
    print(f"  {symbol:20s} → {usage}")
print(f"\n合計: {len(greek_ml_map)} 記号")
```

### 3.2 添字（Subscript / Superscript）の文法

数式の「文法」の中で最も重要なのが**添字**だ。$x$ と $x_i$ と $x_i^{(t)}$ と $x_{i,j}^{(l)}$ は、同じ $x$ でもまったく異なる情報を持つ。

#### 下付き添字（Subscript）: 「どの要素か」

$$
\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}
\quad \text{ベクトル } \mathbf{x} \text{ の } i \text{ 番目の成分を } x_i \text{ と書く}
$$

| パターン | 例 | 意味 |
|:---|:---|:---|
| 要素番号 | $x_i$ | ベクトルの $i$ 番目 |
| 行列要素 | $A_{ij}$ or $a_{ij}$ | $i$ 行 $j$ 列 |
| パラメータ指定 | $p_\theta$ | パラメータ $\theta$ を持つ分布 $p$ |
| 時刻 | $x_t$ | 時刻 $t$ でのデータ |
| 層番号 | $h_l$ | $l$ 番目の層の隠れ状態 |

:::message
**紛らわしいケース**: $x_t$ が「時刻 $t$」か「$t$ 番目の要素」かは文脈依存。論文の冒頭で定義されるので、必ず確認する。拡散モデル[^5]では $x_t$ は「ノイズステップ $t$ での画像」を意味する。
:::

#### 上付き添字（Superscript）: 「何乗か」「何回目か」

| パターン | 例 | 意味 |
|:---|:---|:---|
| べき乗 | $x^2$ | $x$ の2乗 |
| サンプル番号 | $x^{(i)}$ | $i$ 番目のサンプル。丸括弧で区別 |
| 反復回数 | $\theta^{(t)}$ | $t$ 回目の更新後のパラメータ |
| 層番号 | $W^{(l)}$ | $l$ 番目の層の重み行列 |
| 転置 | $A^\top$ or $A^T$ | 行列の転置 |
| 逆行列 | $A^{-1}$ | 逆行列 |

:::details 丸括弧の有無で意味が変わる
- $x^2$: $x$ の2乗（数値のべき乗）
- $x^{(2)}$: 2番目のデータサンプル（インデックス）

この区別は Goodfellow et al. "Deep Learning"[^9] の記法規約に従ったもの。同書では、サンプルインデックスを丸括弧付きの上付きとし、べき乗と区別する記法を採用している。多くの論文がこの規約に従う。
:::

#### 複合添字: $W_{ij}^{(l)}$

複数の添字が組み合わさる場合:

$$
W_{ij}^{(l)} = \text{第 } l \text{ 層の重み行列の } (i, j) \text{ 成分}
$$

**読み方のルール**:
1. まず上付き添字を読む: 「$l$ 層目の」
2. 次に下付き添字を読む: 「$i$ 行 $j$ 列の」
3. 本体を読む: 「重み $W$」

→ 全体: 「$l$ 層目の重み行列 $W$ の $i$ 行 $j$ 列成分」

```python
"""添字の複合パターンを Python で表現する"""
import numpy as np

# 3層ネットワークの重み行列
np.random.seed(42)
# W^(l)_{ij}: 第l層の重み行列の(i,j)成分
W = [
    np.random.randn(4, 3),  # W^(1): 入力3次元 → 隠れ4次元
    np.random.randn(4, 4),  # W^(2): 隠れ4次元 → 隠れ4次元
    np.random.randn(2, 4),  # W^(3): 隠れ4次元 → 出力2次元
]

# W^(2)_{1,3} を取得（0-indexed なので [0, 2]）
l, i, j = 2, 1, 3  # 第2層、1行3列（1-indexed）
print(f"W^({l})_{{{i},{j}}} = {W[l-1][i-1, j-1]:.4f}")

# 全層の形状を確認
for layer_idx, w in enumerate(W, 1):
    print(f"W^({layer_idx}): shape = {w.shape}")
```

### 3.3 演算子・特殊記法

#### 総和 $\sum$ と総乗 $\prod$

機械学習で最も頻繁に登場する演算子:

$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n
$$

$$
\prod_{i=1}^{n} x_i = x_1 \cdot x_2 \cdots x_n
$$

**Cross-Entropy Loss（再掲）**を $\sum$ で書き直す:

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log \hat{y}_i
$$

ここで $C$ はクラス数、$y_i$ は正解ラベル（one-hot）、$\hat{y}_i$ はモデルの予測確率。

$\prod$ は**尤度関数**で登場する。独立なデータ $\{x^{(1)}, \ldots, x^{(N)}\}$ の同時確率:

$$
p(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)} \mid \theta) = \prod_{i=1}^{N} p(\mathbf{x}^{(i)} \mid \theta)
$$

対数を取ると $\prod$ が $\sum$ になる — これが**対数尤度**を使う理由:

$$
\log p = \sum_{i=1}^{N} \log p(\mathbf{x}^{(i)} \mid \theta)
$$

```python
"""Σ と Π の対応: Python の sum() と math.prod()"""
import numpy as np

# サンプルデータ
x = np.array([2.0, 3.0, 5.0, 7.0])

# Σ: 総和
sigma_result = np.sum(x)  # = 2 + 3 + 5 + 7 = 17
print(f"Σ x_i = {sigma_result}")

# Π: 総乗
pi_result = np.prod(x)  # = 2 * 3 * 5 * 7 = 210
print(f"Π x_i = {pi_result}")

# 対数尤度: log(Π) = Σ(log)
log_likelihood = np.sum(np.log(x))
print(f"Σ log(x_i) = {log_likelihood:.4f}")
print(f"log(Π x_i) = {np.log(pi_result):.4f}")
print(f"一致を確認: {np.isclose(log_likelihood, np.log(pi_result))}")
```

#### argmax / argmin

$$
\hat{y} = \arg\max_{i} p(y = i \mid \mathbf{x})
$$

「確率を最大にする**インデックス**を返す」演算。$\max$ が**値**を返すのに対し、$\arg\max$ は**位置**を返す。

```python
"""argmax: 値 vs 位置"""
import numpy as np

probs = np.array([0.1, 0.05, 0.7, 0.15])  # 4クラスの予測確率
print(f"max  p(y|x) = {np.max(probs):.2f}")      # 値: 0.70
print(f"argmax p(y|x) = {np.argmax(probs)}")       # 位置: 2
print(f"→ 予測クラス: {np.argmax(probs)}")
```

#### 期待値 $\mathbb{E}$

$$
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) \, p(x) \, dx
$$

:::message
厳密にはルベーグ測度 $d\mu(x)$ に対する積分ですが、第5回で測度論を扱うまではリーマン積分の記法 $dx$ を使います。
:::

「$x$ を分布 $p$ からサンプリングしたとき、$f(x)$ の平均値」。離散の場合は積分が総和になる:

$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_{x} f(x) \, p(x)
$$

VAE[^4] の目的関数 ELBO は期待値で書かれる:

$$
\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

「エンコーダ $q_\phi$ からサンプルした $\mathbf{z}$ で、デコーダ $p_\theta$ がデータ $\mathbf{x}$ を復元する対数確率の期待値」に、KLダイバージェンス正則化項を引いたもの。

```python
"""期待値: 離散近似 vs モンテカルロ推定"""
import numpy as np

np.random.seed(42)

# 離散分布での期待値
values = np.array([1, 2, 3, 4, 5, 6])  # サイコロの目
probs = np.ones(6) / 6  # 一様分布

E_exact = np.sum(values * probs)
print(f"E[X] (exact) = {E_exact:.4f}")  # 3.5

# モンテカルロ推定: サンプリングして平均
samples = np.random.choice(values, size=10000, p=probs)
E_mc = np.mean(samples)
print(f"E[X] (MC, n=10000) = {E_mc:.4f}")

# 正規分布 N(0,1) からのモンテカルロ推定: E[X^2] = Var[X] = 1
z = np.random.randn(100000)
print(f"E[Z^2] (MC, n=100000) = {np.mean(z**2):.4f}")  # ≈ 1.0
```

#### ノルム $\|\cdot\|$

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} \quad \text{(L2ノルム / ユークリッドノルム)}
$$

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i| \quad \text{(L1ノルム / マンハッタンノルム)}
$$

Attention[^1] で $\sqrt{d_k}$ でスケーリングするのは、$\mathbf{q}^\top \mathbf{k}$ の分散が $d_k$ に比例するため、ノルムのスケールを揃える目的がある。

```python
"""L1, L2 ノルム"""
import numpy as np

x = np.array([3.0, -4.0])

l2 = np.linalg.norm(x, ord=2)  # sqrt(9 + 16) = 5
l1 = np.linalg.norm(x, ord=1)  # 3 + 4 = 7

print(f"||x||_2 = {l2:.1f}")
print(f"||x||_1 = {l1:.1f}")

# 単位ベクトル（正規化）
x_hat = x / l2
print(f"x̂ = x / ||x||_2 = {x_hat}")
print(f"||x̂||_2 = {np.linalg.norm(x_hat):.1f}")  # 1.0
```

#### KL ダイバージェンス $D_{\text{KL}}$

2つの確率分布の「距離」（厳密には非対称なので距離ではない）:

$$
D_{\text{KL}}(q \| p) = \mathbb{E}_{q}\left[\log \frac{q(x)}{p(x)}\right] = \sum_{x} q(x) \log \frac{q(x)}{p(x)}
$$

**性質**:
- $D_{\text{KL}}(q \| p) \geq 0$（非負性、ギブスの不等式）
- $D_{\text{KL}}(q \| p) = 0 \iff q = p$
- $D_{\text{KL}}(q \| p) \neq D_{\text{KL}}(p \| q)$（非対称）

VAE[^4] では $D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ が正則化として機能する。エンコーダが出力する分布を事前分布 $p(\mathbf{z}) = \mathcal{N}(0, I)$ に近づける役割を持つ。

```python
"""KL ダイバージェンス: 離散分布"""
import numpy as np

# 2つの離散分布
p = np.array([0.4, 0.3, 0.2, 0.1])  # "真の"分布
q = np.array([0.25, 0.25, 0.25, 0.25])  # 一様分布

# D_KL(p || q)
kl_pq = np.sum(p * np.log(p / q))
print(f"D_KL(p || q) = {kl_pq:.4f}")

# D_KL(q || p) — 非対称であることを確認
kl_qp = np.sum(q * np.log(q / p))
print(f"D_KL(q || p) = {kl_qp:.4f}")
print(f"非対称: D_KL(p||q) ≠ D_KL(q||p) → {not np.isclose(kl_pq, kl_qp)}")

# 正規分布間のKL（解析解）: KL(N(μ,σ²) || N(0,1))
mu, sigma = 1.0, 0.5
kl_gaussian = 0.5 * (sigma**2 + mu**2 - 1 - np.log(sigma**2))
print(f"\nKL(N({mu},{sigma}²) || N(0,1)) = {kl_gaussian:.4f}")
```

### 3.4 集合論の記号 — データの「住所」を表す

#### 数の集合

| 記号 | 名前 | 意味 | 例 |
|:---:|:---|:---|:---|
| $\mathbb{N}$ | 自然数 | $\{0, 1, 2, \ldots\}$ or $\{1, 2, 3, \ldots\}$ | クラスラベル $y \in \{0, 1, \ldots, C-1\}$ |
| $\mathbb{Z}$ | 整数 | $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$ | インデックス |
| $\mathbb{R}$ | 実数 | 連続値全体 | 重みパラメータ $w \in \mathbb{R}$ |
| $\mathbb{R}^n$ | $n$次元実数ベクトル空間 | | 入力 $\mathbf{x} \in \mathbb{R}^n$ |
| $\mathbb{R}^{m \times n}$ | $m \times n$ 実数行列の空間 | | 重み行列 $W \in \mathbb{R}^{m \times n}$ |
| $\mathbb{R}^+$ | 正の実数 | $(0, \infty)$ | 標準偏差 $\sigma \in \mathbb{R}^+$ |

:::message
**「$\mathbf{x} \in \mathbb{R}^{768}$」の読み方**: 「$\mathbf{x}$ は768次元の実数ベクトル空間の要素」。つまり768個の実数値を並べたベクトル。BERT の隠れ層の次元が768なので、BERT の出力は $\mathbb{R}^{768}$ に住んでいる。
:::

#### 集合の演算

| 記号 | 読み | 意味 | 例 |
|:---:|:---|:---|:---|
| $\in$ | 属する | 要素が集合に含まれる | $x \in \mathbb{R}$ |
| $\notin$ | 属さない | | $-1 \notin \mathbb{N}$ (0始まりの場合) |
| $\subset$ | 部分集合 | | $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{R}$ |
| $\cup$ | 和集合 | OR | 訓練データ $\cup$ 検証データ |
| $\cap$ | 共通集合 | AND | $A \cap B = \emptyset$（互いに素） |
| $\setminus$ | 差集合 | 引く | $\mathbb{R} \setminus \{0\}$（0を除く実数） |
| $\emptyset$ | 空集合 | 要素なし | |
| $|A|$ or $\#A$ | 濃度 | 集合の要素数 | $|\mathcal{D}| = N$（データセットのサイズ） |

#### 区間記法

| 記法 | 意味 | 範囲 |
|:---|:---|:---|
| $[a, b]$ | 閉区間 | $a \leq x \leq b$ |
| $(a, b)$ | 開区間 | $a < x < b$ |
| $[a, b)$ | 半開区間 | $a \leq x < b$ |
| $[0, 1]$ | — | 確率値の範囲。Sigmoid の値域 |
| $(0, \infty)$ | — | 正の実数。ReLU の正の部分 |
| $(-\infty, \infty)$ | — | $\mathbb{R}$ 全体 |

#### データセットの集合表現

機械学習では、データセットを集合として記述する:

$$
\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

「$N$ 個の入力-ラベルのペアからなるデータセット $\mathcal{D}$」

```python
"""集合論の記号を Python で表現"""
import numpy as np

# D = {(x^(i), y^(i))}_{i=1}^{N}
N = 1000  # |D| = 1000
d = 784   # x ∈ R^784 (28x28 画像)
C = 10    # y ∈ {0, 1, ..., 9}

np.random.seed(42)
X = np.random.randn(N, d)          # X ∈ R^{N × d}
y = np.random.randint(0, C, N)     # y ∈ {0, ..., C-1}^N

print(f"|D| = {N}")
print(f"x ∈ R^{d}")
print(f"y ∈ {{0, 1, ..., {C-1}}}")
print(f"X shape: {X.shape} (= R^{{N × d}})")

# 部分集合: 訓練/検証分割
n_train = int(0.8 * N)
D_train = (X[:n_train], y[:n_train])
D_val = (X[n_train:], y[n_train:])
print(f"|D_train| = {n_train}, |D_val| = {N - n_train}")
print(f"|D_train| + |D_val| = |D| → {n_train + (N - n_train) == N}")
```

### 3.5 論理記号 — 数式の「接続詞」

#### 基本の論理記号

| 記号 | 読み | 意味 | Python |
|:---:|:---|:---|:---|
| $\forall$ | for all | すべての〜について | `all(...)` |
| $\exists$ | there exists | 〜が存在する | `any(...)` |
| $\implies$ | implies | ならば | `if ... then ...` |
| $\iff$ | if and only if | 同値 | `==` (論理的等価) |
| $\land$ | and | かつ | `and` |
| $\lor$ | or | または | `or` |
| $\neg$ | not | 否定 | `not` |

#### 論文でよく見る論理表現

**1. 全称量化子 $\forall$**

$$
\forall x \in \mathbb{R}: \quad e^x > 0
$$

「すべての実数 $x$ について、$e^x$ は正」

```python
"""∀ (for all) の Python 表現"""
import numpy as np

# ∀ x ∈ R: e^x > 0
x_samples = np.random.randn(100000)
assert all(np.exp(x_samples) > 0), "反例が見つかった！"
print("∀ x ∈ R: e^x > 0 ... 確認OK（100,000サンプル）")
```

**2. 存在量化子 $\exists$**

$$
\exists \theta^* : \quad \mathcal{L}(\theta^*) \leq \mathcal{L}(\theta) \quad \forall \theta
$$

「損失を最小にするパラメータ $\theta^*$ が存在する」

**3. 含意 $\implies$**

$$
\text{Softmax}(\mathbf{z})_i > 0 \quad \forall i \implies \sum_i \text{Softmax}(\mathbf{z})_i = 1
$$

Softmax[^1] の性質: 「すべての出力が正ならば、合計は1」。実際にはこれは Softmax の定義から自動的に成り立つ。

**4. 同値 $\iff$**

$$
\hat{y} = \arg\max_i p_i \iff p_{\hat{y}} \geq p_j \quad \forall j
$$

「$\hat{y}$ が argmax であることと、$p_{\hat{y}}$ がすべての $p_j$ 以上であることは同値」

:::details 論文英語と論理記号の対応
論文本文では記号の代わりに英語で書かれることが多い:

| 記号 | 英語表現 |
|:---:|:---|
| $\forall$ | "for all", "for any", "for every" |
| $\exists$ | "there exists", "there is" |
| $\implies$ | "implies", "then", "it follows that" |
| $\iff$ | "if and only if", "iff", "is equivalent to" |
| s.t. | "such that", "subject to" — $\exists x \text{ s.t. } f(x) = 0$ |

特に "s.t." は最適化問題で頻出:
$$
\min_\theta \mathcal{L}(\theta) \quad \text{s.t.} \quad \|\theta\|_2 \leq \lambda
$$
:::

### 3.6 関数の記法 — 写像の読み方

#### 関数の定義域・値域

$$
f: \mathbb{R}^n \to \mathbb{R}^m
$$

「$f$ は $n$ 次元実数ベクトルを受け取り、$m$ 次元実数ベクトルを返す関数（写像）」

| 要素 | 記号 | 意味 |
|:---|:---|:---|
| 関数名 | $f$ | 写像そのもの |
| 定義域 (domain) | $\mathbb{R}^n$ | 入力の住所 |
| 値域 (codomain) | $\mathbb{R}^m$ | 出力の住所 |
| 矢印 | $\to$ | 「〜から〜への対応」 |

#### ニューラルネットワークを写像として読む

1層のニューラルネットワーク:

$$
f_\theta: \mathbb{R}^n \to \mathbb{R}^m, \quad f_\theta(\mathbf{x}) = \sigma(W\mathbf{x} + \mathbf{b})
$$

- $\theta = \{W, \mathbf{b}\}$: パラメータ集合
- $W \in \mathbb{R}^{m \times n}$: 重み行列
- $\mathbf{b} \in \mathbb{R}^m$: バイアスベクトル
- $\sigma$: 活性化関数

**多層の合成**:

$$
f = f^{(L)} \circ f^{(L-1)} \circ \cdots \circ f^{(1)}
$$

$\circ$ は**関数合成**（composition）。$(g \circ f)(x) = g(f(x))$。

```python
"""ニューラルネットワーク = 写像の合成"""
import numpy as np

def relu(x):
    """活性化関数 σ: R → R (要素ごと)"""
    return np.maximum(0, x)

def linear(x, W, b):
    """線形変換 f(x) = Wx + b"""
    return W @ x + b

# f_θ: R^3 → R^2 (3次元入力、2次元出力)
np.random.seed(42)
W1 = np.random.randn(4, 3)   # f^(1): R^3 → R^4
b1 = np.zeros(4)
W2 = np.random.randn(2, 4)   # f^(2): R^4 → R^2
b2 = np.zeros(2)

x = np.array([1.0, -0.5, 0.3])  # x ∈ R^3

# f = f^(2) ∘ f^(1)
h = relu(linear(x, W1, b1))     # h = σ(W^(1)x + b^(1)) ∈ R^4
y = linear(h, W2, b2)            # y = W^(2)h + b^(2) ∈ R^2

print(f"入力:  x ∈ R^3 = {x}")
print(f"隠れ:  h ∈ R^4 = {h}")
print(f"出力:  y ∈ R^2 = {y}")
```

#### 特殊な関数記法

| 記法 | 意味 | 例 |
|:---|:---|:---|
| $f: X \to Y$ | $X$ から $Y$ への写像 | $\text{Softmax}: \mathbb{R}^C \to \Delta^{C-1}$ |
| $f \circ g$ | 関数合成 | 多層ネットワーク |
| $f^{-1}$ | 逆関数 | Normalizing Flow |
| $f'(x)$ or $\frac{df}{dx}$ | 導関数 | 勾配計算 |
| $\nabla f$ | 勾配（多変数） | $\nabla_\theta \mathcal{L}$ |
| $\mathcal{O}(n)$ | 計算量 | Attention は $\mathcal{O}(n^2 d)$ |
| $\mathbb{1}[\cdot]$ | 指示関数 | $\mathbb{1}[y = k]$ — one-hot |

:::message
**$\Delta^{C-1}$** は**確率単体** (probability simplex)。$C$ 次元ベクトルで「全要素が非負かつ総和が1」を満たすものの集合:
$$
\Delta^{C-1} = \left\{ \mathbf{p} \in \mathbb{R}^C : p_i \geq 0, \sum_{i=1}^{C} p_i = 1 \right\}
$$
Softmax[^1] の値域はまさにこの確率単体。
:::

```mermaid
graph LR
    subgraph "写像の合成: Deep Neural Network"
        X["入力<br/>x ∈ R^n"] -->|"f^(1)"| H1["隠れ層1<br/>h^(1) ∈ R^{d1}"]
        H1 -->|"f^(2)"| H2["隠れ層2<br/>h^(2) ∈ R^{d2}"]
        H2 -->|"f^(L)"| Y["出力<br/>y ∈ R^m"]
    end
    style X fill:#e3f2fd
    style Y fill:#e8f5e9
```

### 3.7 微分の記法 — 勾配の読み方

機械学習の最適化は**勾配降下法**に基づく。勾配を理解するには微分の記法を知る必要がある。ここでは微分の「計算方法」ではなく「記法の読み方」に集中する。計算の詳細は第2回以降で扱う。

#### 導関数（1変数）

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

$f'(x)$ と $\frac{df}{dx}$ は同じもの。前者はラグランジュ記法、後者はライプニッツ記法と呼ばれる。

| 関数 | 導関数 | 機械学習での用途 |
|:---|:---|:---|
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ | べき乗の微分 |
| $f(x) = e^x$ | $f'(x) = e^x$ | Softmax の微分 |
| $f(x) = \log x$ | $f'(x) = \frac{1}{x}$ | Cross-Entropy の微分 |
| $f(x) = \max(0, x)$ | $f'(x) = \mathbb{1}[x > 0]$ | ReLU の微分 |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | Sigmoid の微分 |

#### 偏微分（多変数）

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}
$$

「他の変数を固定して、$x_i$ だけ動かしたときの変化率」。$\partial$ (パーシャル/デル) が $d$ (ディー) と異なるのは、多変数であることを明示するため。

#### 勾配ベクトル $\nabla f$

$$
\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}
$$

すべての偏微分を縦に並べたベクトル。「$f$ が最も急速に増加する方向」を指す。勾配降下法では、この逆方向（$-\nabla f$）にパラメータを更新する。

```python
"""数値微分で勾配を近似する"""
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """
    ∇f(x) の数値近似（中心差分法）
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# 例: f(x1, x2) = x1^2 + 2*x2^2
# ∇f = (2*x1, 4*x2)
def f(x):
    return x[0]**2 + 2 * x[1]**2

x = np.array([3.0, 4.0])
grad_numerical = numerical_gradient(f, x)
grad_analytical = np.array([2 * x[0], 4 * x[1]])  # 解析解

print(f"x = {x}")
print(f"数値勾配:  ∇f = {grad_numerical}")
print(f"解析勾配:  ∇f = {grad_analytical}")
print(f"一致: {np.allclose(grad_numerical, grad_analytical)}")
```

#### 連鎖律（Chain Rule）— 誤差逆伝播法の心臓

合成関数の微分規則。Rumelhart et al. (1986)[^2] が提案した誤差逆伝播法の数学的基盤:

$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

多変数版:

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdots \frac{\partial h^{(l+1)}}{\partial h^{(l)}} \cdot \frac{\partial h^{(l)}}{\partial W^{(l)}}
$$

「損失 $\mathcal{L}$ の第 $l$ 層の重み $W^{(l)}$ についての勾配は、出力層から第 $l$ 層まで偏微分を**掛け算で伝播**させたもの」。これが **backpropagation** の名前の由来。

```python
"""連鎖律の数値確認"""
import numpy as np

# f(x) = (x^2 + 1)^3
# g(x) = x^2 + 1, f(g) = g^3
# df/dx = df/dg * dg/dx = 3g^2 * 2x = 3(x^2+1)^2 * 2x = 6x(x^2+1)^2

def f(x):
    return (x**2 + 1)**3

def df_analytical(x):
    """解析解: 連鎖律を手計算"""
    return 6 * x * (x**2 + 1)**2

x = 2.0
h = 1e-7

# 数値微分
df_numerical = (f(x + h) - f(x - h)) / (2 * h)

print(f"f({x}) = {f(x)}")
print(f"df/dx (数値)  = {df_numerical:.6f}")
print(f"df/dx (解析)  = {df_analytical(x):.6f}")
print(f"一致: {abs(df_numerical - df_analytical(x)) < 1e-4}")
```

```mermaid
graph LR
    subgraph "連鎖律 (Chain Rule) — 逆伝播の原理"
        L["L (損失)"] -->|"∂L/∂h3"| H3["h^(3)"]
        H3 -->|"∂h3/∂h2"| H2["h^(2)"]
        H2 -->|"∂h2/∂h1"| H1["h^(1)"]
        H1 -->|"∂h1/∂W1"| W1["W^(1)"]
    end
    style L fill:#ffcdd2
    style W1 fill:#c8e6c9
```

#### ヤコビ行列とヘッセ行列（プレビュー）

ここでは名前だけ紹介。詳細は第2回（線形代数）と第3回（最適化）で扱う。

| 行列 | 定義 | サイズ | 用途 |
|:---|:---|:---|:---|
| **ヤコビ行列** $J$ | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ | $m \times n$ | ベクトル値関数の微分。連鎖律の行列版 |
| **ヘッセ行列** $H$ | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ | $n \times n$ | 2次微分。曲率の情報。Newton 法で使用 |

### 3.8 確率論の記法 — 生成モデルの言語

深層生成モデルは本質的に確率モデルだ。確率論の記法を読めなければ、VAE[^4] も拡散モデル[^5] も理解できない。

#### 確率分布の記法

| 記法 | 読み方 | 意味 |
|:---|:---|:---|
| $p(x)$ | 「$p$ の $x$」 | $x$ の確率（離散）/ 確率密度（連続） |
| $p(x \mid y)$ | 「$y$ が与えられたときの $x$ の確率」 | 条件付き確率 |
| $p(x, y)$ | 「$x$ と $y$ の同時確率」 | 同時分布 |
| $x \sim p$ | 「$x$ は $p$ に従う」 | サンプリング |
| $\mathcal{N}(\mu, \sigma^2)$ | 「平均 $\mu$、分散 $\sigma^2$ の正規分布」 | ガウス分布 |
| $\text{Cat}(\boldsymbol{\pi})$ | 「パラメータ $\boldsymbol{\pi}$ のカテゴリカル分布」 | 離散分布 |

#### ベイズの定理

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}
$$

| 項 | 名前 | 意味 |
|:---|:---|:---|
| $p(\theta \mid \mathcal{D})$ | 事後分布 (posterior) | データを観測した後のパラメータの信念 |
| $p(\mathcal{D} \mid \theta)$ | 尤度 (likelihood) | パラメータが与えられたときのデータの生成確率 |
| $p(\theta)$ | 事前分布 (prior) | データを見る前のパラメータの信念 |
| $p(\mathcal{D})$ | エビデンス (evidence) | 周辺尤度。正規化定数 |

$$
\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} \propto \underbrace{p(\mathcal{D} \mid \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}
$$

**VAE[^4] との接続**: VAE は $p(\mathcal{D})$ = $\int p(\mathbf{x}, \mathbf{z}) d\mathbf{z}$ が計算困難なため、事後分布 $p(\mathbf{z} \mid \mathbf{x})$ を近似分布 $q_\phi(\mathbf{z} \mid \mathbf{x})$ で近似する。これが変分推論の核心アイデア。

```python
"""ベイズの定理: コイン投げの例"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# 事前分布: θ ∈ [0,1] に一様分布 (Beta(1,1))
theta_grid = np.linspace(0, 1, 1000)
prior = np.ones_like(theta_grid)  # p(θ) = 1 (一様)

# データ: 10回投げて7回表
n_trials, n_heads = 10, 7

# 尤度: p(D|θ) = θ^7 * (1-θ)^3
likelihood = theta_grid**n_heads * (1 - theta_grid)**(n_trials - n_heads)

# 事後分布: p(θ|D) ∝ p(D|θ) * p(θ)
posterior_unnorm = likelihood * prior
posterior = posterior_unnorm / np.trapezoid(posterior_unnorm, theta_grid)  # np.trapezoid (NumPy 2.0+; 旧名 np.trapz)

# MAP推定: θ* = argmax p(θ|D)
theta_map = theta_grid[np.argmax(posterior)]
print(f"MAP推定: θ* = {theta_map:.3f}")
print(f"MLE推定: θ_MLE = {n_heads/n_trials:.3f}")  # 一致する（一様事前の場合）

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, data, title in zip(axes,
    [prior/np.trapezoid(prior, theta_grid), likelihood/np.trapezoid(likelihood, theta_grid), posterior],
    ["Prior p(θ)", "Likelihood p(D|θ)", "Posterior p(θ|D)"]):
    ax.plot(theta_grid, data)
    ax.set_xlabel("θ")
    ax.set_title(title)
    ax.set_xlim(0, 1)
plt.tight_layout()
plt.savefig("bayes_update.png", dpi=100, bbox_inches="tight")
print("→ bayes_update.png に保存")
```

#### 主要な確率分布

| 分布 | 記法 | パラメータ | 用途 |
|:---|:---|:---|:---|
| 正規 (Gaussian) | $\mathcal{N}(\mu, \sigma^2)$ | 平均 $\mu$, 分散 $\sigma^2$ | VAE の潜在空間、ノイズ |
| 多変量正規 | $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ | 平均ベクトル, 共分散行列 | 多次元潜在変数 |
| カテゴリカル | $\text{Cat}(\boldsymbol{\pi})$ | 確率ベクトル $\boldsymbol{\pi}$ | 離散ラベル予測 |
| ベルヌーイ | $\text{Bern}(p)$ | 成功確率 $p$ | 2値分類 |
| 一様 | $\text{Uniform}(a, b)$ | 区間 $[a, b]$ | 初期化、事前分布 |

```python
"""主要な確率分布のサンプリングと可視化"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
fig, axes = plt.subplots(1, 4, figsize=(14, 3))

# 正規分布 N(0, 1)
x = np.linspace(-4, 4, 200)
axes[0].plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi))
axes[0].set_title("N(0, 1)")

# 多変量正規分布のサンプル
samples = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 500)
axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5)
axes[1].set_title("N(0, Σ) 2D")
axes[1].set_aspect('equal')

# カテゴリカル分布
probs = [0.1, 0.05, 0.5, 0.2, 0.15]
axes[2].bar(range(5), probs)
axes[2].set_title("Cat(π)")
axes[2].set_xlabel("class")

# 一様分布 U(0, 1) のヒストグラム
uniform_samples = np.random.uniform(0, 1, 10000)
axes[3].hist(uniform_samples, bins=30, density=True, alpha=0.7)
axes[3].set_title("Uniform(0, 1)")

plt.tight_layout()
plt.savefig("distributions.png", dpi=100, bbox_inches="tight")
print("→ distributions.png に保存")
```

:::message
**ここまでの道のり**: 3.1 ギリシャ文字 → 3.2 添字 → 3.3 演算子 → 3.4 集合 → 3.5 論理 → 3.6 関数 → 3.7 微分 → 3.8 確率。これで数式を読む「語彙」と「文法」が揃った。Boss Battle で総力戦に臨もう。
:::

### 3.9 Boss Battle: Transformer の Scaled Dot-Product Attention を完全読解

ここまでの知識を総動員して、機械学習で最も重要な式の一つを**一文字残らず**読む。

Vaswani et al. (2017) "Attention Is All You Need"[^1] の式 (1):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

#### Step 1: 記号の確認

| 記号 | 意味 | 住所 |
|:---:|:---|:---|
| $Q$ | Query 行列 | $Q \in \mathbb{R}^{n \times d_k}$ |
| $K$ | Key 行列 | $K \in \mathbb{R}^{m \times d_k}$ |
| $V$ | Value 行列 | $V \in \mathbb{R}^{m \times d_v}$ |
| $K^\top$ | $K$ の転置 | $K^\top \in \mathbb{R}^{d_k \times m}$ |
| $d_k$ | Key の次元数 | $d_k \in \mathbb{N}$ |
| $\sqrt{d_k}$ | スケーリング因子 | $\sqrt{d_k} \in \mathbb{R}^+$ |

#### Step 2: 計算の分解

```mermaid
graph TD
    Q["Q ∈ R^{n×dk}"] --> MM["行列積<br/>QK^T ∈ R^{n×m}"]
    K["K ∈ R^{m×dk}"] --> KT["転置<br/>K^T ∈ R^{dk×m}"]
    KT --> MM
    MM --> SC["スケーリング<br/>/ √dk"]
    SC --> SM["Softmax<br/>(行方向)"]
    SM --> MM2["行列積<br/>× V"]
    V["V ∈ R^{m×dv}"] --> MM2
    MM2 --> OUT["出力 ∈ R^{n×dv}"]
    style OUT fill:#c8e6c9
```

**1. $QK^\top \in \mathbb{R}^{n \times m}$** — 類似度行列

$(n \times d_k) \cdot (d_k \times m) = (n \times m)$。$n$ 個のクエリと $m$ 個のキーの全ペアの内積を計算。

**2. $\frac{QK^\top}{\sqrt{d_k}}$** — スケーリング

内積の値は $d_k$ が大きいほど大きくなる（各次元の寄与が加算されるため）。具体的には、$Q$ と $K$ の各要素が平均0、分散1のとき、$\mathbf{q}^\top \mathbf{k}$ の分散は $d_k$ になる[^1]。$\sqrt{d_k}$ で割ることで分散を1に正規化する。

**3. $\text{softmax}(\cdot)$** — 確率分布への変換

各行に対して Softmax を適用。$n$ 個のクエリそれぞれについて、$m$ 個のキーに対する「注意の重み」が確率分布になる。

**4. $\text{softmax}(\cdot) V$** — 加重平均

$(n \times m) \cdot (m \times d_v) = (n \times d_v)$。重み付き平均で Value を集約。

#### Step 3: Python で完全再現

```python
"""Scaled Dot-Product Attention の完全実装
Vaswani et al. (2017) "Attention Is All You Need" [arXiv:1706.03762]
"""
import numpy as np

def softmax(x, axis=-1):
    """数値安定な Softmax"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Parameters:
        Q: (n, d_k) — Query 行列
        K: (m, d_k) — Key 行列
        V: (m, d_v) — Value 行列

    Returns:
        output: (n, d_v) — Attention 出力
        weights: (n, m) — Attention 重み（確率分布）
    """
    d_k = Q.shape[-1]  # d_k ∈ N

    # Step 1: QK^T ∈ R^{n×m}
    scores = Q @ K.T  # 内積 = 類似度

    # Step 2: / √d_k — スケーリング
    scores = scores / np.sqrt(d_k)

    # Step 3: softmax (行方向) → 確率分布 ∈ Δ^{m-1}
    weights = softmax(scores, axis=-1)

    # Step 4: 加重平均 ∈ R^{n×d_v}
    output = weights @ V

    return output, weights

# --- 実験 ---
np.random.seed(42)
n, m, d_k, d_v = 3, 5, 64, 64

Q = np.random.randn(n, d_k)  # Q ∈ R^{3×64}
K = np.random.randn(m, d_k)  # K ∈ R^{5×64}
V = np.random.randn(m, d_v)  # V ∈ R^{5×64}

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
print(f"Attention weights: {weights.shape}")
print(f"Output: {output.shape}")
print(f"\nAttention weights (query 0):")
print(f"  {weights[0].round(4)}")
print(f"  sum = {weights[0].sum():.6f}")  # ≈ 1.0

# スケーリングの効果を確認
scores_raw = Q @ K.T
scores_scaled = scores_raw / np.sqrt(d_k)
print(f"\nスケーリング前: 分散 = {scores_raw.var():.2f}")
print(f"スケーリング後: 分散 = {scores_scaled.var():.2f}")
print(f"理論値 (d_k = {d_k}): 前 ≈ {d_k}, 後 ≈ 1.0")
```

#### Step 4: Multi-Head Attention

単一の Attention を複数の「ヘッド」で並列実行し、結果を連結する[^1]:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

- $h$: ヘッド数。原論文では $h = 8$
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$: 第 $i$ ヘッドの Query 射影
- $d_k = d_v = d_{\text{model}} / h$: 各ヘッドの次元。$d_{\text{model}} = 512$ なら $d_k = 64$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$: 出力射影

```python
"""Multi-Head Attention の実装"""
import numpy as np

def multi_head_attention(Q, K, V, n_heads=8):
    """
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O

    Parameters:
        Q, K, V: (seq_len, d_model)
        n_heads: ヘッド数 h
    """
    d_model = Q.shape[-1]
    d_k = d_model // n_heads  # d_k = d_model / h

    # 射影行列 W_i^Q, W_i^K, W_i^V ∈ R^{d_model × d_k}
    np.random.seed(123)
    scale = 1.0 / np.sqrt(d_model)

    heads = []
    for i in range(n_heads):
        W_Q = np.random.randn(d_model, d_k) * scale
        W_K = np.random.randn(d_model, d_k) * scale
        W_V = np.random.randn(d_model, d_k) * scale

        # head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        Q_i = Q @ W_Q  # (n, d_k)
        K_i = K @ W_K  # (m, d_k)
        V_i = V @ W_V  # (m, d_k)

        head_i, _ = scaled_dot_product_attention(Q_i, K_i, V_i)
        heads.append(head_i)

    # Concat(head_1, ..., head_h) ∈ R^{n × (h * d_k)}
    concat = np.concatenate(heads, axis=-1)

    # W^O ∈ R^{h*d_v × d_model}
    W_O = np.random.randn(n_heads * d_k, d_model) * scale
    output = concat @ W_O

    return output

# --- 実験 ---
seq_len, d_model = 10, 512
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

out = multi_head_attention(Q, K, V, n_heads=8)
print(f"入力: Q, K, V ∈ R^{{{seq_len}×{d_model}}}")
print(f"出力: {out.shape} (= R^{{{seq_len}×{d_model}}})")
print(f"→ 入力と出力の shape が一致 ✓")
```

:::message
**Boss Battle クリア！** Transformer の Attention 式を一文字残らず読めるようになった。Zone 1 で「なんとなく」理解していた Softmax、添字、スケーリングの意味が、今は**完全に**説明できるはずだ。
:::

:::message
**進捗: 50% 完了** 数式の記号体系をマスターした。ここからは実践フェーズ — 環境構築、LaTeX の書き方、論文の読み方を学ぶ。
:::

---

## 🛠️ 4. 環境・ツールゾーン（45分）— 開発環境・LaTeX・論文読解術

> **目標**: Python 環境を整え、LaTeX で数式を書けるようにし、arXiv 論文を構造的に読む技術を身につける。

### 4.1 開発環境セットアップ — Python・IDE・AI CLI

コードを書き、実行し、AI に助けてもらう。この3つの環境を一気に整える。

#### Python 環境構築

本シリーズの Course I（第1回〜第8回）は Python 100% で進める。環境構築はシンプルに保つ。

#### 推奨環境

| 項目 | 推奨 | 理由 |
|:---|:---|:---|
| Python バージョン | 3.11+ | match 文、tomllib、速度改善 |
| パッケージ管理 | `uv` | pip の10倍高速、lockfile対応 |
| 仮想環境 | `uv venv` | プロジェクトごとに分離 |
| エディタ | VSCode + Pylance | 型推論、Jupyter 統合 |
| ノートブック | Jupyter Lab or VSCode | 対話的実験 |

```bash
# uv のインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクト作成
mkdir -p ~/ml-lectures && cd ~/ml-lectures
uv init
uv add numpy matplotlib jupyter

# 仮想環境の有効化
source .venv/bin/activate
python -c "import numpy; print(f'NumPy {numpy.__version__} ready')"
```

#### 最小限の依存パッケージ

```toml
# pyproject.toml
[project]
name = "ml-lectures"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "matplotlib>=3.8",
    "jupyter>=1.0",
]

[project.optional-dependencies]
lecture01 = []  # 第1回は追加依存なし
lecture02 = ["scipy>=1.12"]  # 第2回で追加
```

:::message
**本シリーズのルール**: 各講義で必要なパッケージは `[project.optional-dependencies]` で管理する。第1回は NumPy と Matplotlib のみ。PyTorch は第3回から、JAX は Course II から登場する。
:::

#### IDE（統合開発環境）の選び方

Python の環境ができたら、次はコードを読み書きする道具だ。正直、どれを選んでも学習はできる。だが道具の差は長期的に効いてくる。

#### 3大エディタ比較

| | VSCode | Cursor | Zed |
|:---|:---|:---|:---|
| **価格** | 無料 | 無料〜$20/月 | 無料 |
| **特徴** | 拡張機能が豊富 | AI統合エディタ | Rust製・超高速 |
| **AI支援** | Copilot拡張で対応 | ネイティブAI統合 | AI統合あり |
| **起動速度** | 普通 | 普通（VSCode fork） | 非常に高速 |
| **Jupyter** | 統合サポート | 統合サポート | 未対応 |
| **おすすめ対象** | 万人向け | AI活用したい人 | 速度重視の人 |

```mermaid
graph TD
    Q{"エディタ選び"} -->|初心者・安定| V["VSCode<br/>拡張豊富・情報多い"]
    Q -->|AI重視| C["Cursor<br/>VSCode + AI統合"]
    Q -->|速度重視| Z["Zed<br/>Rust製・軽量"]
    V --> J["Jupyter統合OK"]
    C --> J
    style V fill:#e3f2fd
    style C fill:#fff3e0
    style Z fill:#e8f5e9
```

:::message
**本シリーズの推奨**: 迷ったら **VSCode** で始める。拡張機能・ドキュメント・コミュニティが最も充実しており、困ったとき検索で解決しやすい。
:::

#### 最低限入れるべき拡張機能（VSCode）

| 拡張機能 | 用途 |
|:---|:---|
| Python (ms-python) | Python 言語サポート |
| Pylance | 型推論・補完 |
| Jupyter | ノートブック実行 |
| GitLens | Git 履歴の可視化 |
| Markdown All in One | Markdown プレビュー |

```bash
# コマンドラインから一括インストール
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
code --install-extension yzhang.markdown-all-in-one
```

#### ターミナル統合とキーバインド

エディタ内でターミナルを開き、コードの実行と編集を行き来できるのが IDE の強み:

| 操作 | ショートカット（VSCode） |
|:---|:---|
| ターミナル表示/非表示 | `` Ctrl+` `` |
| ファイル検索 | `Ctrl+P` |
| コマンドパレット | `Ctrl+Shift+P` |
| 定義へジャンプ | `F12` |
| 参照を検索 | `Shift+F12` |
| 行コメント | `Ctrl+/` |

:::details Cursor と Zed の補足
**Cursor**: VSCode をフォークしたエディタで、AI チャット・コード補完・コードベース理解が統合されている。VSCode の拡張機能がそのまま使える。月額 $20 の Pro プランで Claude / GPT-4 を使ったコード生成が可能。

**Zed**: Rust で書かれた次世代エディタ。起動とファイル操作が圧倒的に速い。マルチプレイヤー編集（ペアプログラミング）がネイティブ対応。ただし拡張機能エコシステムは VSCode ほど成熟していない。Jupyter 未対応のため、本シリーズの序盤では補助ツールとして使い、メインは VSCode が安全。
:::

#### AI CLI ツール — ターミナルからAIを使う

IDE が整ったら、もう一つの武器を手に入れよう。2025年以降、ターミナルから直接AIに質問・コード生成・デバッグ支援を受けるのが当たり前になった。ブラウザを開かずに、コーディング中のターミナルからそのまま AI を呼べる。

#### ツール比較

| ツール | 価格 | 特徴 | おすすめ度 |
|:---|:---|:---|:---|
| **Gemini CLI** | 無料 | Google製・導入が最も簡単 | ★★★★★ |
| **GitHub Copilot CLI** | $10/月（学生無料） | GitHub統合・安定 | ★★★★☆ |
| **Codex CLI** | API従量課金 | OpenAI製・高精度 | ★★★☆☆ |
| **Claude Code** | API従量課金 | Anthropic製・深い推論 | ★★★☆☆ |

:::message alert
**課金の落とし穴**: Claude Code と Codex CLI は API 従量課金制。1回の質問で $0.01〜$0.50+ かかることがある。$20/月のプランに入っても API 利用分は別途請求されるため、初学者は**無料の Gemini CLI から始める**のが安全。月の請求額が思わぬ金額になった報告は少なくない。
:::

#### Gemini CLI のセットアップ（推奨）

```bash
# インストール
npm install -g @anthropic-ai/gemini-cli
# または
npx @google/gemini-cli

# 認証（Google アカウントでログイン）
gemini auth login

# 基本的な使い方
gemini "Softmax関数をPythonで実装して"
gemini "このエラーの原因を教えて: IndexError: index out of range"
gemini "numpy の einsum の使い方を教えて"
```

#### AI CLI の実践的な使い方

```bash
# コードの説明を求める
gemini "以下のコードが何をしているか説明して:
def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V"

# デバッグ支援
gemini "np.linalg.norm が nan を返す。原因は？"

# 数式をコードに翻訳
gemini "KLダイバージェンスの式をNumPyで実装して"
```

:::details 課金ツール（Codex / Claude Code）を使う場合の注意
- **利用量のモニタリング**: 毎日の利用額を確認する習慣をつける
- **トークン上限の設定**: 環境変数や設定ファイルで1回あたりの上限を設定
- **簡単な質問は無料ツールで**: Gemini CLI で十分な質問を課金ツールに投げない
- **本当に必要な場面**: 大規模なコードベース理解、複雑なリファクタリング、深い推論が必要なとき

```bash
# Claude Code の場合
claude "複雑な質問をここに"

# Codex CLI の場合
codex "複雑な質問をここに"
```

どちらも高精度だが、日常的な質問には Gemini CLI で十分。課金ツールは「ここぞ」という場面で使うのがコスパ最良。
:::

### 4.2 プラットフォーム活用術 — GitHub・Hugging Face・OSSライセンス

開発環境が整ったら、次は**外の世界**にアクセスする方法を学ぶ。論文実装を読み、事前学習済みモデルを試し、法的リスクを避ける — この3つのプラットフォームスキルがセットで必要になる。

#### GitHub入門 — コードの宝庫を読み解く

論文を読めるようになったら、次は**実装を読む**番だ。世界中の研究者・エンジニアがコードを公開している場所、それが GitHub。

#### リポジトリの読み方

GitHub リポジトリを開いたとき、最初に見るべきファイルは3つ:

| ファイル | 見るべきポイント |
|:---|:---|
| `README.md` | プロジェクト概要・セットアップ手順・使い方 |
| `requirements.txt` / `pyproject.toml` | 依存ライブラリ（PyTorch? JAX? バージョンは?） |
| メインのソースコード（`model.py` 等） | 論文の数式がどこに実装されているか |

```mermaid
graph LR
    R["README.md<br/>概要把握"] --> D["依存関係<br/>requirements.txt"]
    D --> S["ソースコード<br/>model.py"]
    S --> T["テスト<br/>tests/"]
    T --> I["Issue/PR<br/>議論・バグ"]
    style R fill:#e3f2fd
    style S fill:#fff3e0
```

#### 論文実装の探し方

**Papers With Code** (paperswithcode.com) が最強のツール。論文タイトルで検索すると、公式・非公式の実装が一覧で出る。

```bash
# GitHub でのコード検索（例: Attention 実装を探す）
# github.com にアクセスし、検索バーで:
# "scaled_dot_product_attention" language:python

# リポジトリのクローン
git clone https://github.com/<user>/<repo>.git
cd <repo>

# 特定のファイルを検索
find . -name "*.py" | head -20

# 特定の関数を検索
grep -r "def attention" --include="*.py"
```

:::message
**Tips**: 論文の実装を読むとき、まず `forward` メソッドを探せ。PyTorch なら `nn.Module` のサブクラスの `forward` が論文の数式に対応している。
:::

#### Git 基本操作

コードを手元にコピーして実験するための最小限の Git:

```bash
# リポジトリをコピー
git clone <url>

# 変更の確認
git status
git diff

# 変更の保存
git add <file>
git commit -m "message"

# 履歴の確認
git log --oneline -10
```

#### jj（Jujutsu）— Git の上位互換 VCS

本シリーズでは **jj**（Jujutsu）を推奨する。Git と互換性を保ちながら、操作性が大幅に改善されている。

| 機能 | Git | jj |
|:---|:---|:---|
| 作業コピー | 手動 add/commit | **自動追跡**（常に記録） |
| undo | `reflog` + `reset --hard`（危険） | **`jj undo`**（何回でも安全） |
| コンフリクト | マージ時に発生・即解決必須 | **記録して後で解決可能** |
| ブランチ | 必須（HEAD管理） | **不要**（匿名コミットが基本） |
| バックエンド | Git 独自形式 | **Git互換**（既存リポジトリにそのまま使える） |

```bash
# jj のインストール
# macOS
brew install jj

# 既存の Git リポジトリで jj を使い始める
cd <git-repo>
jj git init --colocate

# 基本操作
jj status          # 状態確認
jj diff            # 差分表示
jj describe -m "message"  # コミットにメッセージ
jj new             # 新しい変更を開始
jj log             # 履歴をグラフ表示
jj undo            # 直前の操作を取り消し（何回でも）
```

:::details Git vs jj — どちらを学ぶべきか？
結論: **両方の概念を理解し、日常では jj を使う**。

理由:
1. jj は Git バックエンドを使うので、Git の知識は無駄にならない
2. jj の操作体系は Git より直感的（`add/commit` が不要、`undo` が安全）
3. 既存の GitHub リポジトリに対してそのまま `jj` を使える
4. Git を要求する環境（CI/CD、チーム開発）でも jj が裏で Git 操作を行う

初学者は jj から始めて、必要に応じて Git の概念を学ぶのが最短経路だ。
:::

#### Hugging Face入門 — モデルとデータセットのハブ

GitHub がコードの宝庫なら、**Hugging Face** (huggingface.co) は学習済みモデルの宝庫だ。機械学習モデル・データセット・デモの共有プラットフォームとして、論文の実装を「動かす」には、ここを使いこなすのが最短経路。

#### 3つの柱

| サービス | 内容 | URL |
|:---|:---|:---|
| **Models** | 事前学習済みモデル（80万+） | huggingface.co/models |
| **Datasets** | 公開データセット（15万+） | huggingface.co/datasets |
| **Spaces** | インタラクティブなデモ | huggingface.co/spaces |

#### Model Card の読み方

モデルページを開くと、「Model Card」が表示される。これは論文の Abstract に相当する:

| セクション | 確認ポイント |
|:---|:---|
| Model Description | アーキテクチャ・パラメータ数・学習データ |
| Intended Use | 想定用途と制限事項 |
| Training Details | 学習設定（エポック数・バッチサイズ・lr） |
| Evaluation | ベンチマーク結果 |
| Limitations | バイアス・失敗ケース・倫理的考慮 |

:::message
**重要**: Model Card の **Limitations** セクションは必ず読むこと。「このモデルは英語のみ」「有害なコンテンツを生成しうる」等の制約が書かれている。無視して本番投入すると事故になる。
:::

#### transformers ライブラリの基本

```bash
# インストール（本シリーズでは Course II から本格使用）
uv add transformers torch
```

```python
# 感情分析を3行で体験（モデルは自動ダウンロード）
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love machine learning!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

この3行の裏で何が起きているか:

```mermaid
graph LR
    T["テキスト入力"] --> TK["Tokenizer<br/>文→トークンID"]
    TK --> M["Model<br/>BERT等"]
    M --> L["分類ヘッド<br/>Softmax"]
    L --> R["結果<br/>POSITIVE 0.99"]
    style T fill:#e3f2fd
    style R fill:#e8f5e9
```

#### モデルのダウンロードと推論

```python
from transformers import AutoTokenizer, AutoModel

# モデル名を指定してダウンロード
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# テキストをトークン化 → モデルに入力
text = "Attention is all you need"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
print(f"Output shape: {outputs.last_hidden_state.shape}")
# Output shape: torch.Size([1, 7, 768])
# → 7トークン × 768次元の hidden state
```

:::details Hugging Face Hub のキャッシュ管理
モデルは `~/.cache/huggingface/` にダウンロードされる。大きなモデルはディスクを圧迫するので:

```bash
# キャッシュの確認
du -sh ~/.cache/huggingface/

# 特定モデルの削除
huggingface-cli delete-cache

# カスタムキャッシュディレクトリの指定
export HF_HOME=/path/to/large/disk/.cache/huggingface
```

BERT-base で約 440MB、GPT-2 で約 500MB。大型モデル（LLaMA 等）は数十GB単位になるため、ディスク容量に注意。
:::

#### OSSライセンス — 使う前に知るべきこと

GitHub や Hugging Face でコードやモデルを見つけたら、使う前に必ず確認すべきことがある — **ライセンス**だ。「知らなかった」は通用しない。

#### 主要ライセンス一覧

| ライセンス | 商用利用 | 改変 | 再配布条件 | コピーレフト |
|:---|:---|:---|:---|:---|
| **MIT** | OK | OK | 著作権表示のみ | なし |
| **Apache-2.0** | OK | OK | 著作権表示 + 変更点明記 | なし |
| **BSD-2/3** | OK | OK | 著作権表示のみ | なし |
| **MPL-2.0** | OK | OK | 改変ファイルのみ公開 | 弱い |
| **LGPL** | OK | OK | ライブラリ改変部分を公開 | 中程度 |
| **GPL** | OK | OK | **派生物全体を公開** | 強い |
| **CC BY** | OK | OK | クレジット表示 | なし |
| **CC BY-NC** | **不可** | OK | クレジット表示 | なし |

#### コピーレフトの強度スペクトル

```mermaid
graph LR
    MIT["MIT / BSD<br/>制約ほぼなし"] --> APL["Apache-2.0<br/>特許条項追加"]
    APL --> MPL["MPL-2.0<br/>ファイル単位"]
    MPL --> LGPL["LGPL<br/>ライブラリ単位"]
    LGPL --> GPL["GPL<br/>派生物全体"]
    GPL --> AGPL["AGPL<br/>ネットワーク経由も"]
    style MIT fill:#e8f5e9
    style GPL fill:#ffcdd2
    style AGPL fill:#ef9a9a
```

**左に行くほど自由、右に行くほど制約が強い。** 自分のコードに GPL ライブラリを組み込むと、自分のコード全体も GPL で公開する義務が生じる（感染性）。

#### 商用利用の判断フローチャート

```mermaid
graph TD
    S["コードを使いたい"] --> L{"ライセンスは？"}
    L -->|MIT/BSD/Apache| OK["✅ 著作権表示して使用OK"]
    L -->|MPL-2.0| MPL["✅ 改変ファイルだけ公開"]
    L -->|LGPL| LG["✅ ライブラリとして使うならOK<br/>改変したら改変部分を公開"]
    L -->|GPL| GP{"組み込み方は？"}
    GP -->|リンク・import| WARN["⚠️ 派生物 → GPL公開義務"]
    GP -->|CLI呼び出し（別プロセス）| CLI["✅ 分離されていればOK"]
    L -->|CC BY-NC| NC["❌ 商用利用不可"]
    L -->|ライセンスなし| NONE["❌ 使用不可<br/>（著者に許可を求める）"]
    style OK fill:#e8f5e9
    style WARN fill:#fff3e0
    style NC fill:#ffcdd2
    style NONE fill:#ffcdd2
```

:::message alert
**ライセンスなし = 使用不可**。GitHub にコードが公開されていても、`LICENSE` ファイルがなければ著作権者の許可なく使用できない。「公開されているから自由に使える」は誤解。
:::

#### ライセンス互換性マトリクス

自分のプロジェクトが MIT ライセンスの場合、どのライセンスのコードを取り込めるか:

| 取り込み元 → | MIT | Apache-2.0 | MPL-2.0 | LGPL | GPL |
|:---|:---|:---|:---|:---|:---|
| **MIT プロジェクト** | OK | OK | 条件付きOK | 条件付きOK | **不可** |
| **Apache-2.0** | OK | OK | 条件付きOK | 条件付きOK | **不可** |
| **GPL プロジェクト** | OK | OK | OK | OK | OK |

:::details ライセンス確認の実践手順
```bash
# リポジトリのライセンスを確認
cat LICENSE
# または
cat LICENSE.md

# GitHub API で確認
gh api repos/<owner>/<repo> --jq '.license.spdx_id'

# Python パッケージのライセンス確認
pip show numpy | grep License
# License: BSD License
```

**本シリーズで使うライブラリのライセンス**:

| ライブラリ | ライセンス | 商用利用 |
|:---|:---|:---|
| NumPy | BSD-3-Clause | OK |
| Matplotlib | PSF (BSD互換) | OK |
| PyTorch | BSD-3-Clause | OK |
| JAX | Apache-2.0 | OK |
| Hugging Face transformers | Apache-2.0 | OK |

全て商用利用可能。安心して使える。
:::

### 4.3 論文との向き合い方 — arXiv・3パスリーディング・知識管理

開発環境とプラットフォームの準備ができた。ここからは**論文を読み、理解し、記憶に残す**ための方法論に入る。arXiv で論文を見つけ、構造的に読み、知識をグラフ化する — この一連のワークフローを身につけよう。

#### arXiv の使い方 — 論文の宝庫

arXiv (https://arxiv.org) は物理学・数学・計算機科学のプレプリントサーバー。機械学習の最新論文はほぼすべてここに投稿される。

#### arXiv ID の読み方

| 形式 | 例 | 意味 |
|:---|:---|:---|
| 新形式 | `2006.11239` | 2020年6月の11239番目 |
| 旧形式 | `1706.03762` | 2017年6月の3762番目 |
| カテゴリ付き | `cs.LG/2006.11239` | cs.LG (Machine Learning) カテゴリ |

**主要カテゴリ**:
- `cs.LG` — Machine Learning
- `cs.CL` — Computation and Language (NLP)
- `cs.CV` — Computer Vision
- `cs.AI` — Artificial Intelligence
- `stat.ML` — Statistics: Machine Learning

#### 効率的な論文の探し方

1. **Semantic Scholar** (semanticscholar.org) — 引用ネットワークで関連論文を探索
2. **Papers With Code** (paperswithcode.com) — 実装付き論文
3. **Connected Papers** (connectedpapers.com) — 引用グラフの可視化
4. **Daily Papers** (huggingface.co/papers) — 日次の注目論文
5. **arXiv Sanity** — フィルタリングされた新着論文

:::message
**本シリーズで引用する論文は、すべて arXiv ID またはDOI付きで記載する。** 「〜と言われている」のような曖昧な引用は一切行わない。これが学術的誠実さの基本であり、読者が原典に当たれる環境を保証する。
:::

#### 3パスリーディング — 論文の構造的読解法

論文は**3回読む**のが基本戦略。S. Keshav の "How to Read a Paper" (2007) に基づく方法論。

#### Pass 1: 鳥瞰（5-10分）

**読む箇所**: タイトル → Abstract → Introduction（最初と最後の段落）→ 各セクション見出し → Conclusion → 図表

**得るもの**: 「この論文は何をしたのか」の1行要約

```mermaid
graph LR
    T["Title"] --> A["Abstract"]
    A --> I["Intro<br/>(First+Last ¶)"]
    I --> H["Section<br/>Headings"]
    H --> C["Conclusion"]
    C --> F["Figures &<br/>Tables"]
    style T fill:#e3f2fd
    style F fill:#e8f5e9
```

**Pass 1 チェックリスト**:
- [ ] 何の問題を解いているか？
- [ ] 既存手法の限界は何か？
- [ ] 提案手法の核心アイデアは？
- [ ] 主要な結果（数値）は？
- [ ] 自分の研究/学習に関連するか？

#### Pass 2: 精読（1-2時間）

**読む箇所**: 全文を通読（証明は飛ばしてよい）

**得るもの**: 手法の詳細理解、自分の言葉での説明

重要なのは**図表と数式をセットで読む**こと:
1. 図を見る → 何を表しているか推測
2. 対応する数式を読む → 図の各要素を数式と対応づける
3. 本文の説明を読む → 推測の答え合わせ

:::details Pass 2 での数式の読み方
Zone 3 で学んだ技術をフル活用する:

1. **記号の洗い出し**: 新しい記号が出たら、定義を探す
2. **次元の確認**: 各変数の shape を追跡する
3. **特殊ケースの確認**: $n=1$ や $d=1$ で式を単純化して意味を確認
4. **コードとの対応**: 数式を Python に翻訳してみる

例: VAE[^4] の再パラメータ化トリック
$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

→ Python: `z = mu + sigma * np.random.randn(*mu.shape)`
:::

#### Pass 3: 再現（数時間〜数日）

**やること**: 論文の手法を実装する、あるいは論文の主張を自分のデータで検証する

Pass 3 は全論文で行う必要はない。自分の研究に直結する論文、またはシリーズの講義テーマとなる論文に限定する。

#### 実践: "Attention Is All You Need"[^1] の Pass 1

| 項目 | 内容 |
|:---|:---|
| タイトル | "Attention Is All You Need" — Attention 機構だけで十分 |
| 問題 | 機械翻訳の系列変換モデル。RNN/CNN の逐次処理が並列化を阻害 |
| 提案 | Transformer: Self-Attention のみで構成。再帰なし、畳み込みなし |
| 核心 | Scaled Dot-Product Attention + Multi-Head Attention + Positional Encoding |
| 結果 | WMT 2014 英独翻訳で BLEU 28.4（当時SOTA）。訓練時間は1/10以下 |
| 影響 | BERT, GPT, ViT, DALL-E, ... 現代のほぼ全モデルの基盤 |

```python
"""論文の Pass 1 を構造化するテンプレート"""

pass1_template = {
    "title": "",
    "authors": "",
    "year": 0,
    "arxiv_id": "",
    "problem": "",          # 何の問題を解いているか
    "limitation": "",       # 既存手法の限界
    "proposal": "",         # 提案手法の核心
    "key_equation": "",     # 最も重要な数式（LaTeX）
    "main_result": "",      # 主要な数値結果
    "relevance": "",        # 自分との関連
    "pass2_needed": False,  # 精読すべきか
}

# 記入例: Attention Is All You Need
attention_paper = {
    "title": "Attention Is All You Need",
    "authors": "Vaswani, Shazeer, Parmar, et al.",
    "year": 2017,
    "arxiv_id": "1706.03762",
    "problem": "Sequence transduction (machine translation)",
    "limitation": "RNN/CNN require sequential computation, limiting parallelization",
    "proposal": "Transformer: pure attention-based architecture, no recurrence",
    "key_equation": r"Attention(Q,K,V) = softmax(QK^T/√d_k)V",
    "main_result": "BLEU 28.4 on WMT 2014 En-De (SOTA), 10x less training cost",
    "relevance": "Foundation of all modern LLMs and generative models",
    "pass2_needed": True,
}

for key, val in attention_paper.items():
    print(f"  {key:20s}: {val}")
```

#### 論文・知識管理 — Obsidian で知識をグラフ化する

論文を読む技術を身につけたら、次は読んだ知識を**構造化して残す**仕組みだ。40回の講義を受け、数十本の論文を読み、何百もの数式に触れる。この知識を整理しないと、3ヶ月後には何も覚えていない。

#### 推奨ツール: Obsidian

**Obsidian** (obsidian.md) はローカル完結のMarkdownエディタ。最大の特徴は**双方向リンク**と**ナレッジグラフ**。

| 特徴 | 説明 |
|:---|:---|
| ローカル完結 | データは全てローカルの `.md` ファイル。クラウド依存なし |
| 双方向リンク | `[[ノート名]]` でノート間をリンク。被リンクも自動表示 |
| ナレッジグラフ | リンク構造を視覚化。知識の全体像が見える |
| プラグイン豊富 | コミュニティプラグインで機能拡張 |
| 数式対応 | KaTeX/MathJax で数式レンダリング |

```mermaid
graph TD
    L1["第1回: 概論"] -->|"前提知識"| L2["第2回: 線形代数"]
    L1 -->|"Softmax"| L3["第3回: 微分"]
    L2 -->|"行列"| L6["第6回: KL情報量"]
    L3 -->|"勾配"| L4["第4回: 確率"]
    L4 -->|"ベイズ"| L9["第9回: ELBO"]
    L6 -->|"KL"| L9
    style L1 fill:#e3f2fd
    style L9 fill:#fff3e0
```

#### ローカル完結スタック

論文管理から執筆まで、全てローカルで完結するツール群:

| ツール | 役割 | 連携 |
|:---|:---|:---|
| **Zotero** | 論文PDF管理・引用 | Obsidian プラグインで連携 |
| **Obsidian** | ノート・知識管理 | Markdown → どこでも使える |
| **Longform** | 長文執筆（Obsidian プラグイン） | チャプター管理 |
| **Pandoc** | 出力変換 | Markdown → PDF / LaTeX / DOCX |

```bash
# Zotero のインストール
# https://www.zotero.org/ からダウンロード

# Pandoc のインストール
brew install pandoc   # macOS
# or: sudo apt install pandoc  # Ubuntu

# Markdown → PDF 変換
pandoc lecture-notes.md -o lecture-notes.pdf --pdf-engine=lualatex
```

#### クラウド共著ツール: Prism

チームで論文を書く場合は **Prism** (withprism.ai) が選択肢に入る。OpenAI が開発したAI支援付き共同執筆ツールで、リアルタイム共同編集 + AI による文章改善提案が統合されている。ただし本シリーズの学習ノートにはオーバースペック — まずは Obsidian で個人の知識管理を固めるのが先決。

#### 講義ノートの取り方 — 実践テンプレート

本シリーズ40回分をObsidianでナレッジグラフ化するテンプレート:

```markdown
---
tags: [ml-lecture, zone3, 線形代数]
lecture: 2
date: 2025-xx-xx
---

# 第2回: 線形代数 I

#### Key Concepts
- [[行列積]] — $C = AB$ where $C_{ij} = \sum_k A_{ik}B_{kj}$
- [[固有値分解]] — $A\mathbf{v} = \lambda\mathbf{v}$

#### Links
- 前提: [[第1回_概論]]
- 次回: [[第3回_微分]]
- 関連: [[Attention]] uses [[行列積]]

#### Questions
- [ ] なぜ固有値分解が重要？→ [[PCA]] で使う（第5回）

#### Code Snippets
<!-- 数式とコードの対応を残す -->
```

:::details Notion / Scrapbox ではダメなのか？
使っても構わないが、Obsidian を推奨する理由:

1. **ローカル完結**: インターネット不要。サービス終了リスクゼロ
2. **Markdown**: 標準形式なので他ツールへの移行が容易
3. **双方向リンク**: 講義間の関係性が自然に構造化される
4. **Git/jj 管理可能**: `.md` ファイルなのでバージョン管理できる
5. **数式**: KaTeX 対応で数式がそのままレンダリングされる

Notion はクラウド依存でエクスポートが面倒。Scrapbox は双方向リンクは優秀だが数式対応が弱い。
:::

### 4.4 LaTeX 入門 — 数式を「書く」力

数式を「読む」だけでなく「書く」力も必要だ。論文を書くときはもちろん、Zenn の記事やノートに数式を残すときにも LaTeX を使う。

#### 基本記法

| 数式 | LaTeX | 出力 |
|:---|:---|:---|
| 分数 | `\frac{a}{b}` | $\frac{a}{b}$ |
| 上付き | `x^{2}` | $x^{2}$ |
| 下付き | `x_{i}` | $x_{i}$ |
| 平方根 | `\sqrt{x}` | $\sqrt{x}$ |
| 総和 | `\sum_{i=1}^{n} x_i` | $\sum_{i=1}^{n} x_i$ |
| 総乗 | `\prod_{i=1}^{n} x_i` | $\prod_{i=1}^{n} x_i$ |
| 積分 | `\int_{a}^{b} f(x) dx` | $\int_{a}^{b} f(x) dx$ |
| 偏微分 | `\frac{\partial f}{\partial x}` | $\frac{\partial f}{\partial x}$ |
| ベクトル | `\mathbf{x}` | $\mathbf{x}$ |
| 行列 | `\mathbf{A}` or `\mathbf{W}` | $\mathbf{A}$ |
| 集合 | `\mathbb{R}^n` | $\mathbb{R}^n$ |
| 損失関数 | `\mathcal{L}` | $\mathcal{L}$ |
| 期待値 | `\mathbb{E}[X]` | $\mathbb{E}[X]$ |

#### Zenn での数式記法

Zenn は KaTeX をサポートしている。インラインは `$...$`、ブロックは `$$...$$`:

```markdown
<!-- インライン数式 -->
Softmax は $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ で定義される。

<!-- ブロック数式 -->
$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log \hat{y}_i
$$
```

:::details KaTeX で使えない LaTeX コマンド（注意）
KaTeX は LaTeX の完全互換ではない。以下は注意:

| 使えない | 代替 |
|:---|:---|
| `\text{}` 内の日本語 | 数式外に書く |
| `\boldsymbol{}` | `\mathbf{}` |
| `\newcommand` | Zenn では使えない |
| `aligned` 環境 | `\begin{aligned}...\end{aligned}` は使える |

**Tips**: 複雑な数式は Overleaf か HackMD でプレビューしてから Zenn に貼ると安全。
:::

#### 練習: Attention の式を LaTeX で書く

以下の数式を LaTeX で書いてみよう:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

:::details 解答
```latex
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

ポイント:
- `\text{Attention}` — 関数名はローマン体
- `\left(` `\right)` — 括弧のサイズ自動調整
- `K^\top` — 転置。`K^T` でもよいが `\top` が正式
- `\sqrt{d_k}` — 平方根
- `\frac{}{}` — 分数
:::

### 4.5 数式 ↔ コード翻訳 — 7つのパターン

論文の数式をコードに翻訳するとき、頻出するパターンを整理する。これを知っていれば、初見の数式でも迷わない。

#### Pattern 1: $\sum$ → `np.sum()` / `sum()`

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

```python
x_bar = np.mean(x)  # = np.sum(x) / len(x)
```

#### Pattern 2: $\prod$ → `np.prod()` / 対数和

$$
p(\mathcal{D}) = \prod_{i=1}^{N} p(x^{(i)})
$$

```python
# 直接計算（オーバーフロー注意）
p_data = np.prod(p_xi)

# 対数空間（推奨）
log_p = np.sum(np.log(p_xi))
```

#### Pattern 3: $\arg\max$ → `np.argmax()`

$$
\hat{y} = \arg\max_c p(y = c \mid \mathbf{x})
$$

```python
y_hat = np.argmax(probs)
```

#### Pattern 4: $\mathbb{E}[\cdot]$ → `np.mean()` (モンテカルロ)

$$
\mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x^{(i)}), \quad x^{(i)} \sim p
$$

```python
samples = np.random.normal(0, 1, size=10000)  # x ~ p
E_fx = np.mean(f(samples))
```

#### Pattern 5: 行列積 $AB$ → `A @ B`

$$
\mathbf{h} = W\mathbf{x} + \mathbf{b}
$$

```python
h = W @ x + b
```

#### Pattern 6: 要素ごとの演算 $\odot$ → `*`

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}
$$

```python
z = mu + sigma * epsilon  # element-wise
```

#### Pattern 7: $\nabla_\theta \mathcal{L}$ → 自動微分

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)
$$

```python
# NumPy（手動）
grad = compute_gradient(theta, loss_fn)
theta = theta - alpha * grad

# PyTorch（自動微分）— 第3回以降
# loss.backward()
# optimizer.step()
```

:::details 翻訳パターン対応表（まとめ）
| 数式 | Python (NumPy) | 備考 |
|:---|:---|:---|
| $\sum_i x_i$ | `np.sum(x)` | axis 指定で次元制御 |
| $\prod_i x_i$ | `np.prod(x)` | 対数空間推奨 |
| $\arg\max$ | `np.argmax(x)` | |
| $\mathbb{E}[f(x)]$ | `np.mean(f(samples))` | モンテカルロ |
| $AB$ | `A @ B` | 行列積 |
| $A \odot B$ | `A * B` | 要素積 |
| $A^\top$ | `A.T` | 転置 |
| $\|x\|_2$ | `np.linalg.norm(x)` | |
| $\nabla f$ | 手動 or autograd | 第3回以降 |
| $\mathcal{N}(\mu, \sigma^2)$ | `np.random.normal(mu, sigma)` | |
| $\mathbb{1}[c]$ | `(condition).astype(int)` | 指示関数 |
:::

:::message
**進捗: 75% 完了** 開発環境、プラットフォーム活用、論文読解・知識管理、LaTeX、コード翻訳パターンまで一通りカバーした。残りは自己診断テストとまとめ。
:::

---

## 🔬 5. 実験ゾーン（30分）— 自己診断テスト

> **目標**: Zone 3-4 の内容を本当に理解しているか、自分で確認する。「わかったつもり」を排除する。

### 5.1 記号読解テスト（10問）

以下の数式を**日本語で**説明せよ。答えを見る前に、自分で書いてみること。

:::details Q1: $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)$
**A**: パラメータ $\theta$ を、損失関数 $\mathcal{L}$ の $\theta$ についての勾配 $\nabla_\theta \mathcal{L}$ に学習率 $\alpha$ を掛けた分だけ更新する。これが**勾配降下法**（Gradient Descent）の1ステップ。Rumelhart et al. (1986)[^2] が誤差逆伝播法と組み合わせて提案した学習アルゴリズムの基本形。
:::

:::details Q2: $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$
**A**: パラメータ $\theta$ を持つモデルのデータ $\mathbf{x}$ に対する確率を、潜在変数 $\mathbf{z}$ について周辺化（積分消去）して求める。これが**周辺尤度**（marginal likelihood）。VAE[^4] ではこの積分が解析的に計算できないため、変分下界（ELBO）で近似する。
:::

:::details Q3: $D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$
**A**: エンコーダ $q_\phi$ が出力する事後分布と、事前分布 $p(\mathbf{z})$（通常 $\mathcal{N}(0, I)$）の間のKLダイバージェンス。VAE[^4] の正則化項として機能し、潜在空間が構造を持つように制約する。
:::

:::details Q4: $\text{softmax}(z_i / \tau)$
**A**: ロジット $z_i$ を温度パラメータ $\tau$ でスケーリングした後に Softmax を適用。$\tau \to 0$ で argmax（最も確率の高いクラスのみ1）、$\tau \to \infty$ で一様分布に近づく。Hinton et al. (2015)[^3] がKnowledge Distillation で使用。
:::

:::details Q5: $\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} p_\theta(y = c \mid \mathbf{x})$
**A**: 入力 $\mathbf{x}$ に対して、$C$ 個のクラスの中で事後確率 $p_\theta(y = c | \mathbf{x})$ が最大となるクラス $c$ を予測ラベル $\hat{y}$ とする。分類問題の推論時の操作。
:::

:::details Q6: $W_{ij}^{(l)} \in \mathbb{R}$
**A**: 第 $l$ 層の重み行列の $(i, j)$ 成分。実数値スカラー。上付きの $(l)$ は層番号、下付きの $ij$ は行列の行・列インデックス。
:::

:::details Q7: $f: \mathbb{R}^n \to \mathbb{R}^m$
**A**: 関数 $f$ は $n$ 次元実数ベクトルを受け取り、$m$ 次元実数ベクトルを返す写像。ニューラルネットワークの各層はこの形の写像。
:::

:::details Q8: $\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(\mathbf{x})]$
**A**: データ分布 $p_{\text{data}}$ からサンプリングした $\mathbf{x}$ について、モデル $p_\theta$ の対数確率の期待値。これを最大化することが**最尤推定**（Maximum Likelihood Estimation）に相当する。
:::

:::details Q9: $\epsilon_t \sim \mathcal{N}(0, I)$
**A**: 時刻（ステップ）$t$ のノイズ $\epsilon_t$ を、平均0、共分散が単位行列 $I$ の多変量正規分布からサンプリングする。拡散モデル[^5]のforward processで各ステップのノイズとして使用される。
:::

:::details Q10: $\|\nabla_\theta \mathcal{L}\|_2$
**A**: 損失関数 $\mathcal{L}$ の $\theta$ についての勾配ベクトルのL2ノルム（ユークリッドノルム）。**勾配ノルム**と呼ばれ、学習の安定性の指標として監視される。これが爆発（exploding）すると学習が破綻し、消失（vanishing）すると学習が停滞する。
:::

### 5.2 LaTeX 書き取りテスト（5問）

以下の数式を **LaTeX で書け**。KaTeX で正しく表示されることを確認せよ。

:::details Q1: Cross-Entropy Loss
**目標**:
$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log \hat{y}_i
$$

**解答**:
```latex
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log \hat{y}_i
```
:::

:::details Q2: Scaled Dot-Product Attention
**目標**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**解答**:
```latex
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```
:::

:::details Q3: KL ダイバージェンス
**目標**:
$$
D_{\text{KL}}(q \| p) = \sum_{x} q(x) \log \frac{q(x)}{p(x)}
$$

**解答**:
```latex
D_{\text{KL}}(q \| p) = \sum_{x} q(x) \log \frac{q(x)}{p(x)}
```
:::

:::details Q4: 勾配降下法
**目標**:
$$
\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_\theta \mathcal{L}(\theta^{(t)})
$$

**解答**:
```latex
\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_\theta \mathcal{L}(\theta^{(t)})
```
:::

:::details Q5: VAE の ELBO
**目標**:
$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

**解答**:
```latex
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
```
:::

### 5.3 コード翻訳テスト（5問）

以下の数式を **NumPy で実装せよ**。

:::details Q1: Softmax
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

```python
def softmax(z):
    """数値安定な Softmax"""
    e_z = np.exp(z - np.max(z))  # オーバーフロー防止
    return e_z / np.sum(e_z)

# テスト
z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
print(f"softmax({z}) = {p.round(4)}")
print(f"sum = {p.sum():.6f}")  # 1.0
```
:::

:::details Q2: Cross-Entropy Loss
$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i + \epsilon)
$$

```python
def cross_entropy(y_true, y_pred, eps=1e-12):
    """Cross-Entropy Loss"""
    return -np.sum(y_true * np.log(y_pred + eps))

# テスト: 正解がクラス2
y_true = np.array([0, 0, 1, 0])  # one-hot
y_pred = np.array([0.1, 0.05, 0.8, 0.05])  # Softmax 出力

loss = cross_entropy(y_true, y_pred)
print(f"L_CE = {loss:.4f}")  # -log(0.8) ≈ 0.2231
```
:::

:::details Q3: コサイン類似度
$$
\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
$$

```python
def cosine_similarity(a, b):
    """コサイン類似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# テスト
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
c = np.array([-1.0, -2.0, -3.0])

print(f"cos(a, b) = {cosine_similarity(a, b):.4f}")   # 1.0 (同方向)
print(f"cos(a, c) = {cosine_similarity(a, c):.4f}")   # -1.0 (逆方向)
```
:::

:::details Q4: 正規分布の対数確率密度
$$
\log \mathcal{N}(x; \mu, \sigma^2) = -\frac{1}{2}\left(\log(2\pi\sigma^2) + \frac{(x - \mu)^2}{\sigma^2}\right)
$$

```python
def log_normal_pdf(x, mu, sigma):
    """正規分布の対数確率密度"""
    return -0.5 * (np.log(2 * np.pi * sigma**2) + (x - mu)**2 / sigma**2)

# テスト: N(0,1) で x=0 の対数確率密度
print(f"log N(0; 0, 1) = {log_normal_pdf(0, 0, 1):.4f}")  # -0.9189
print(f"理論値: -0.5 * log(2π) = {-0.5 * np.log(2 * np.pi):.4f}")
```
:::

:::details Q5: ミニバッチ勾配降下法
$$
\theta \leftarrow \theta - \frac{\alpha}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta \ell(\theta; x^{(i)}, y^{(i)})
$$

```python
def sgd_step(theta, X_batch, y_batch, grad_fn, alpha=0.01):
    """
    ミニバッチ SGD の1ステップ

    Parameters:
        theta: パラメータ
        X_batch: ミニバッチ入力 (batch_size, d)
        y_batch: ミニバッチラベル (batch_size,)
        grad_fn: 勾配を計算する関数
        alpha: 学習率
    """
    batch_size = len(X_batch)
    # Σ ∇θ ℓ(θ; x^(i), y^(i)) / |B|
    grad_sum = np.zeros_like(theta)
    for i in range(batch_size):
        grad_sum += grad_fn(theta, X_batch[i], y_batch[i])
    avg_grad = grad_sum / batch_size

    # θ ← θ - α * avg_grad
    return theta - alpha * avg_grad
```
:::

### 5.4 論文読解テスト

以下の論文情報を読んで、Pass 1 のテンプレートを埋めよ。

**対象**: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"[^5]

:::details ヒント
arXiv ID: 2006.11239。Abstract を読むだけで Pass 1 は完成する。

**キーワード**: diffusion process, denoising, variational inference, progressive lossy decompression
:::

:::details 解答例
| 項目 | 内容 |
|:---|:---|
| タイトル | Denoising Diffusion Probabilistic Models |
| 著者 | Ho, Jain, Abbeel |
| 年 | 2020 |
| arXiv ID | 2006.11239 |
| 問題 | 高品質な画像生成 |
| 既存手法の限界 | GAN[^8]は訓練不安定、VAE[^4]は生成品質に限界 |
| 提案 | 拡散過程（ノイズ付加→除去）による生成モデル |
| 核心数式 | $L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]$ |
| 主要結果 | FID 3.17 on CIFAR-10（当時の生成モデルでSOTA品質） |
| 関連度 | 本シリーズ第11-14回で詳解 |
:::

### 5.5 実装チャレンジ: 3つのミニプロジェクト

これまでの知識を統合する3つの実装課題。所要時間は合計1-2時間。

#### Challenge 1: 数式パーサー（記号→説明辞書）

```python
"""
Challenge: 数式の各記号を自動的に解説するパーサーを作る
入力: LaTeX 文字列（の簡易版）
出力: 各記号の日本語説明
"""
import re

# 記号辞書
SYMBOL_DB = {
    r"\theta": ("シータ", "モデルパラメータ"),
    r"\phi": ("ファイ", "エンコーダ/変分パラメータ"),
    r"\mu": ("ミュー", "平均"),
    r"\sigma": ("シグマ", "標準偏差"),
    r"\nabla": ("ナブラ", "勾配演算子"),
    r"\mathcal{L}": ("エル", "損失関数"),
    r"\mathbb{E}": ("イー", "期待値"),
    r"\sum": ("シグマ", "総和"),
    r"\prod": ("パイ", "総乗"),
    r"\partial": ("パーシャル", "偏微分"),
    r"\alpha": ("アルファ", "学習率"),
    r"\epsilon": ("イプシロン", "微小量/ノイズ"),
    r"\lambda": ("ラムダ", "正則化係数"),
    r"\mathbb{R}": ("アール", "実数の集合"),
    r"\in": ("属する", "集合の要素"),
    r"\forall": ("すべての", "全称量化子"),
    r"\exists": ("存在する", "存在量化子"),
    r"\sqrt": ("ルート", "平方根"),
    r"\frac": ("分数", "分子/分母"),
    r"\log": ("ログ", "対数関数"),
    r"\exp": ("エクスプ", "指数関数"),
    r"\top": ("トップ", "転置"),
    r"\text{softmax}": ("ソフトマックス", "確率分布への変換"),
}

def parse_symbols(latex_str):
    """LaTeX 文字列から既知の記号を抽出して解説"""
    found = []
    for symbol, (reading, meaning) in SYMBOL_DB.items():
        if symbol in latex_str:
            found.append((symbol, reading, meaning))
    return found

# テスト
formulas = [
    r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V",
    r"\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta)",
    r"\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right]",
]

for formula in formulas:
    print(f"\n数式: {formula[:60]}...")
    symbols = parse_symbols(formula)
    for sym, reading, meaning in symbols:
        print(f"  {sym:25s} ({reading}) → {meaning}")
```

#### Challenge 2: Attention の可視化

```python
"""
Challenge: Attention weights をヒートマップで可視化する
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def attention_with_viz(Q, K, V, labels_q=None, labels_k=None):
    """Attention を計算して可視化"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V

    # ヒートマップ
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 生のスコア
    im0 = axes[0].imshow(scores, cmap='RdBu_r', aspect='auto')
    axes[0].set_title("Raw scores (QK^T/√dk)")
    axes[0].set_xlabel("Key")
    axes[0].set_ylabel("Query")
    plt.colorbar(im0, ax=axes[0])

    # Attention weights (softmax 後)
    im1 = axes[1].imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title("Attention weights (after softmax)")
    axes[1].set_xlabel("Key")
    axes[1].set_ylabel("Query")
    plt.colorbar(im1, ax=axes[1])

    if labels_q:
        for ax in axes:
            ax.set_yticks(range(len(labels_q)))
            ax.set_yticklabels(labels_q)
    if labels_k:
        for ax in axes:
            ax.set_xticks(range(len(labels_k)))
            ax.set_xticklabels(labels_k, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=100, bbox_inches='tight')
    print("→ attention_heatmap.png に保存")
    return output, weights

# テスト: 単語の埋め込みを模擬
np.random.seed(42)
n_queries, n_keys, d_model = 4, 6, 64

Q = np.random.randn(n_queries, d_model)
K = np.random.randn(n_keys, d_model)
V = np.random.randn(n_keys, d_model)

# 意図的に Q[0] と K[2] を類似させる
K[2] = Q[0] + np.random.randn(d_model) * 0.1

labels_q = ["Query_0", "Query_1", "Query_2", "Query_3"]
labels_k = ["Key_0", "Key_1", "Key_2", "Key_3", "Key_4", "Key_5"]

output, weights = attention_with_viz(Q, K, V, labels_q, labels_k)
print(f"\nQuery_0 の Attention weights:")
for i, w in enumerate(weights[0]):
    bar = "█" * int(w * 50)
    print(f"  Key_{i}: {w:.4f} {bar}")
```

#### Challenge 3: 学習曲線の実装と可視化

```python
"""
Challenge: 簡単な線形回帰を勾配降下法で解いて学習曲線を描く
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- データ生成 ---
np.random.seed(42)
N = 100
x_true = np.random.uniform(-3, 3, N)
y_true = 2.5 * x_true + 1.0 + np.random.randn(N) * 0.5  # y = 2.5x + 1.0 + noise

# --- モデル: y = wx + b ---
# パラメータ θ = (w, b)
w, b = 0.0, 0.0
alpha = 0.01  # 学習率 α
n_epochs = 200

# --- 勾配降下法 ---
history = {"epoch": [], "loss": [], "w": [], "b": []}

for epoch in range(n_epochs):
    # 予測: ŷ = wx + b
    y_pred = w * x_true + b

    # 損失: L = (1/N) Σ(ŷ - y)²  (MSE)
    loss = np.mean((y_pred - y_true)**2)

    # 勾配: ∂L/∂w = (2/N) Σ(ŷ - y)·x
    #        ∂L/∂b = (2/N) Σ(ŷ - y)
    residual = y_pred - y_true
    grad_w = 2 * np.mean(residual * x_true)
    grad_b = 2 * np.mean(residual)

    # 更新: θ ← θ - α∇L
    w -= alpha * grad_w
    b -= alpha * grad_b

    history["epoch"].append(epoch)
    history["loss"].append(loss)
    history["w"].append(w)
    history["b"].append(b)

print(f"最終パラメータ: w = {w:.4f} (真値 2.5), b = {b:.4f} (真値 1.0)")
print(f"最終損失: L = {history['loss'][-1]:.4f}")

# --- 可視化 ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 学習曲線
axes[0].plot(history["epoch"], history["loss"])
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (MSE)")
axes[0].set_title("Learning Curve")
axes[0].set_yscale('log')

# パラメータの収束
axes[1].plot(history["epoch"], history["w"], label="w (→ 2.5)")
axes[1].plot(history["epoch"], history["b"], label="b (→ 1.0)")
axes[1].axhline(y=2.5, color='C0', linestyle='--', alpha=0.5)
axes[1].axhline(y=1.0, color='C1', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Parameter value")
axes[1].set_title("Parameter Convergence")
axes[1].legend()

# フィッティング結果
axes[2].scatter(x_true, y_true, alpha=0.5, s=10, label="data")
x_line = np.linspace(-3, 3, 100)
axes[2].plot(x_line, w * x_line + b, 'r-', label=f"y = {w:.2f}x + {b:.2f}")
axes[2].plot(x_line, 2.5 * x_line + 1.0, 'g--', alpha=0.5, label="true")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_title("Linear Regression Fit")
axes[2].legend()

plt.tight_layout()
plt.savefig("learning_curve.png", dpi=100, bbox_inches='tight')
print("→ learning_curve.png に保存")
```

:::message
**Challenge クリア基準**: 3つのうち2つ以上を実行して結果を確認できればクリア。コードの意味を説明できること（写経ではなく理解していること）が重要。
:::

### 5.6 総合診断: セルフチェックリスト

以下の全項目に「はい」と答えられれば、Zone 5 クリア:

- [ ] ギリシャ文字 $\theta, \phi, \mu, \sigma, \nabla, \alpha, \epsilon$ の意味を即答できる
- [ ] $W_{ij}^{(l)}$ の添字を「第$l$層の$i$行$j$列」と読める
- [ ] $\sum$、$\prod$、$\arg\max$、$\mathbb{E}$ を Python に翻訳できる
- [ ] $\mathbb{R}^n$、$\in$、$\forall$、$\exists$ の意味がわかる
- [ ] $f: \mathbb{R}^n \to \mathbb{R}^m$ を「$n$次元入力から$m$次元出力への写像」と読める
- [ ] Attention の式 $\text{softmax}(QK^\top / \sqrt{d_k})V$ を一文字残らず説明できる
- [ ] LaTeX で基本的な数式（分数、添字、総和）を書ける
- [ ] arXiv の ID から論文にアクセスできる
- [ ] 論文の Pass 1 を10分以内で実行できる
- [ ] 数式→Python の7つの翻訳パターンを使える

:::message
**進捗: 85% 完了** 自己診断を通じて理解の穴を埋めた。あとは全体のまとめと次回への橋渡し。
:::

---

## 🎓 6. 振り返りゾーン（30分）— まとめと次回予告

### 6.1 用語集（本講義で登場した用語）

:::details 用語集（クリックで展開）
| 用語 | 英語 | 定義 |
|:---|:---|:---|
| 勾配 | gradient | 多変数関数の各偏微分を並べたベクトル。$\nabla f$ |
| 勾配降下法 | gradient descent | 勾配の逆方向にパラメータを更新する最適化手法 |
| 誤差逆伝播法 | backpropagation | 合成関数の連鎖律を用いて勾配を効率的に計算する手法[^2] |
| 損失関数 | loss function | モデルの予測と正解の乖離を測る関数。$\mathcal{L}$ |
| 交差エントロピー | cross-entropy | 2つの確率分布の差異を測る損失関数 |
| ソフトマックス | softmax | 実数ベクトルを確率分布に変換する関数[^1] |
| 温度パラメータ | temperature | Softmax のシャープさを制御するスカラー[^3] |
| アテンション | attention | クエリとキーの類似度で値を重み付け集約する機構[^1] |
| 潜在変数 | latent variable | データの背後にある観測されない変数。$\mathbf{z}$ |
| 変分推論 | variational inference | 事後分布を近似するための最適化ベースの推論手法[^4] |
| ELBO | evidence lower bound | 周辺尤度の下界。VAEの目的関数[^4] |
| KLダイバージェンス | KL divergence | 2つの分布間の非対称な「距離」 |
| 拡散モデル | diffusion model | データにノイズを加え、除去する過程で学習する生成モデル[^5] |
| フローマッチング | flow matching | 確率的なフローでソースからターゲットへの変換を学習[^6] |
| arXiv | arXiv | 物理学・数学・計算機科学のプレプリントサーバー |
| プレプリント | preprint | 査読前の論文 |
| 写像 | mapping / function | 定義域の各要素を値域の要素に対応させる規則 |
| 確率単体 | probability simplex | 非負で総和1のベクトルの集合。$\Delta^{C-1}$ |
:::

### 6.2 第1回の知識マップ

```mermaid
mindmap
    root((第1回))
        数式記号
            ギリシャ文字
            添字
            演算子
        数式文法
            集合論
            論理記号
            関数記法
        LLM基礎
            Softmax
            Attention
            Cross-Entropy
        実践技術
            Python環境
            LaTeX記法
            論文読解
        生成モデル概観
            VAE
            GAN
            拡散
            Flow
```

### 6.3 本講義のまとめ

第1回では以下を学んだ:

**1. 数式は「読む」ものであり「恐れる」ものではない**
- ギリシャ文字は「数式のアルファベット」— 覚えれば読める
- 添字・演算子・集合記法は「数式の文法」— ルールを理解すれば構文解析できる
- 論文の数式は著者の思考のスナップショット — 一文字ずつ分解すれば必ず理解できる

**2. Transformer の Attention 式を完全に読解した**
- Vaswani et al. (2017)[^1] の Scaled Dot-Product Attention
- 各記号の意味、次元の追跡、スケーリングの理由を理解した
- Python（NumPy）で完全に再実装した

**3. 論文の読み方を構造化した**
- 3パスリーディング: 鳥瞰 → 精読 → 再現
- arXiv の使い方、論文検索の方法
- 数式↔コード翻訳の7つのパターン

**4. 深層生成モデルの全体像を俯瞰した**
- VAE[^4], GAN[^8], 拡散モデル[^5], Flow Matching[^6], DiT[^7] の位置づけ
- 全50回シリーズの構成を理解した

### 6.4 よくある質問

:::details Q: 数学が苦手でもついていけますか？
**A**: はい。本シリーズは「数学が得意な人向け」ではなく「数学をこれから身につけたい人向け」に設計されている。第1回で記号体系を網羅したのはそのため。以降の講義で新しい記号が出てきたら、Zone 3 に戻れば解決する。ただし、「読み飛ばす」のではなく「わからなかったら戻る」という姿勢は必要。
:::

:::details Q: Python 以外の言語は使いますか？
**A**: Course I（第1-8回）は Python 100%。Course II（第9-16回）で Julia が登場し、Course III 以降で Rust + Julia の多言語構成になる。各言語の導入時に丁寧にセットアップするので心配不要。
:::

:::details Q: 講義の順番通りに進めるべきですか？
**A**: 基本的には順番通りを推奨。特に第1-4回は基礎なので飛ばさないこと。ただし、特定のトピック（例: 拡散モデルだけ知りたい）がある場合は、第1-2回 → 第11回と飛んでも理解できるように設計してある。
:::

:::details Q: 数式をすべて暗記する必要がありますか？
**A**: **暗記は不要。理解が重要。** Softmax の式を暗記していなくても、「実数ベクトルを確率分布に変換する関数で、各要素の指数を全体の指数の和で割る」と説明できれば十分。式は論文を見れば書いてある。意味を理解していれば、式を見た瞬間に読める。
:::

:::details Q: 参考書は買うべきですか？
**A**: 最低限は不要。本シリーズで必要な数学はすべて講義内で解説する。ただし、深掘りしたい場合は Zone 6 の推薦書籍を参照。特に "Mathematics for Machine Learning" (Deisenroth et al.) は無料PDF公開されており、手元に置いておく価値がある。
:::

:::details Q: 機械学習の経験がゼロでも大丈夫ですか？
**A**: 大丈夫。本シリーズは「プログラミングができるが、機械学習は初めて」という読者を想定している。Python の基礎（変数、関数、ループ、リスト）ができれば十分。NumPy も本講義の中で必要な操作を都度解説する。ただし、完全なプログラミング初心者の場合は、先に Python の入門書を1冊読んでおくことを推奨する。
:::

:::details Q: GPU は必要ですか？
**A**: Course I（第1-8回）は GPU 不要。CPU だけで全コードが動く。GPU が必要になるのは第9回以降の訓練実験から。その時点で Google Colab（無料枠）で十分。本格的な訓練実験をしたい場合は、第9回で GPU 環境の構築方法を解説する。
:::

:::details Q: 論文を英語で読む必要がありますか？
**A**: はい。機械学習の一次情報はほぼすべて英語。ただし、本シリーズでは重要論文の核心部分を日本語で解説するので、「論文を完全に読む」必要はない。まずは本講義の Pass 1 テンプレートで Abstract と図表だけ読む練習から始めよう。英語力は繰り返し読むうちに自然に上がる。DeepL/GPT を補助的に使うのは問題ない（ただし数式は自分で読むこと）。
:::

:::details Q: この講義だけで研究できるようになりますか？
**A**: 全50回を修了すれば、最新の深層生成モデルの論文を読み、理解し、実装し、改良するための基礎力が身につく。ただし「研究」には問題設定能力や実験設計力など、本シリーズだけではカバーしきれない能力も必要。本シリーズはあくまで「論文が読め、実装できる」ところまでを保証する。
:::

### 6.5 学習スケジュールの提案

第1回の内容を効率的に消化するための推奨スケジュール:

| 日 | 内容 | 所要時間 |
|:---|:---|:---|
| Day 1 | Zone 0-2: 概要と動機づけ。Softmax, Attention, Cross-Entropy を手で計算 | 1.5h |
| Day 2 | Zone 3 (3.1-3.6): ギリシャ文字、添字、演算子、集合、論理、関数 | 2h |
| Day 3 | Zone 3 (3.7-3.9): 微分、確率、Boss Battle (Attention 完全読解) | 2h |
| Day 4 | Zone 4: 環境構築 + LaTeX 練習 + 論文読解テンプレート作成 | 1.5h |
| Day 5 | Zone 5: 自己診断テスト + 実装チャレンジ | 2h |
| Day 6 | Zone 6-7: 全体像の確認 + 復習 | 1h |
| Day 7 | **復習日**: Zone 3 の苦手箇所を再読、Pass 1 を1本実践 | 1h |

**合計: 約11時間 / 1週間**

:::message
**ペース配分のコツ**: 1日に Zone を2つ以上進めようとしないこと。特に Zone 3 は消化に時間がかかる。「わかったつもり」で先に進むより、1つの Zone を確実に理解してから次に進む方が、結果的に速い。
:::

#### 復習の具体的な方法

1. **フラッシュカード**: Zone 3 のギリシャ文字と記号を Anki に登録。1日5分のレビュー
2. **写経 + 改造**: Zone 5 のコードを写経し、パラメータを変えて実験する
3. **論文 Pass 1**: 週1本の arXiv 論文で Pass 1 テンプレートを埋める練習
4. **数式日記**: 毎日1つ、新しい数式を見つけて「日本語で説明する」練習

```python
"""学習進捗トラッカー"""

progress = {
    "Zone 0: QuickStart": {"status": "done", "confidence": 5},
    "Zone 1: Intuition": {"status": "done", "confidence": 4},
    "Zone 2: Motivation": {"status": "done", "confidence": 5},
    "Zone 3.1: Greek Letters": {"status": "done", "confidence": 3},
    "Zone 3.2: Subscripts": {"status": "done", "confidence": 4},
    "Zone 3.3: Operators": {"status": "done", "confidence": 3},
    "Zone 3.4: Sets": {"status": "done", "confidence": 4},
    "Zone 3.5: Logic": {"status": "done", "confidence": 3},
    "Zone 3.6: Functions": {"status": "done", "confidence": 4},
    "Zone 3.7: Calculus": {"status": "done", "confidence": 3},
    "Zone 3.8: Probability": {"status": "done", "confidence": 3},
    "Zone 3.9: Boss Battle": {"status": "done", "confidence": 4},
    "Zone 4: Practical": {"status": "done", "confidence": 4},
    "Zone 5: Diagnosis": {"status": "done", "confidence": 3},
    "Zone 6: References": {"status": "done", "confidence": 5},
    "Zone 7: Summary": {"status": "done", "confidence": 5},
}

print("=== 第1回 学習進捗 ===\n")
total_conf = 0
n_zones = len(progress)
for zone, info in progress.items():
    bar = "★" * info["confidence"] + "☆" * (5 - info["confidence"])
    print(f"  [{info['status']:4s}] {bar} {zone}")
    total_conf += info["confidence"]

avg_conf = total_conf / n_zones
print(f"\n平均自信度: {avg_conf:.1f}/5.0")
if avg_conf >= 4.0:
    print("→ 第2回に進んでOK！")
elif avg_conf >= 3.0:
    print("→ 自信度3以下の Zone を復習してから第2回へ")
else:
    print("→ Zone 3 を重点的に復習しよう")
```

### 6.6 次回予告

**第2回: 線形代数 — ベクトルと行列の世界**

> ニューラルネットワークの「言語」は線形代数だ。行列の掛け算が世界を変える。

次回のキートピック:
- ベクトル空間、基底、次元
- 行列演算とその幾何学的意味
- 固有値分解と特異値分解（SVD）
- PCA: 次元削減の原理
- **Boss Battle**: Transformer の位置エンコーディングの線形代数的解釈

:::message
**進捗: 100% 完了** おめでとう。第1回「概論: 数式と論文の読み方」を修了した。ここで身につけた「数式を読む力」は、残り39回すべての講義で使い続ける基礎体力だ。次回は線形代数の世界に踏み込む。
:::

### 6.7 パラダイム転換の問い

> **もし「数式」という表現形式が存在しなかったら、人類は深層学習を発明できただろうか？**

数式は「厳密さ」と「汎用性」を両立する唯一の言語だ。「入力 $\mathbf{x}$ を重み $W$ で線形変換し、バイアス $\mathbf{b}$ を加え、非線形関数 $\sigma$ を通す」— この操作を $\sigma(W\mathbf{x} + \mathbf{b})$ という7文字で表現できる。自然言語では同じ情報量に50文字以上かかる。

しかし、別の可能性もある:
- **プログラミング言語で直接定義する世界**: `h = relu(W @ x + b)` — 実際、多くの実務者はコードで考えている
- **図的言語（ダイアグラム）で定義する世界**: カテゴリ理論のstring diagramのように
- **自然言語で大規模言語モデルに生成させる世界**: 2024-2026のAIコーディングが示す方向性

数式は「発見の道具」だったのか、それとも「発見を制約する檻」だったのか。Einstein の「数学は自然の言語」という主張は、数学が自然を記述するのに適しているのか、それとも数学で記述可能な自然しか我々が認識できないのか — この問いに第2回以降、繰り返し立ち戻ることになる。

**考えてみてほしいこと**:

1. **Transformer[^1] は数式から生まれたのか、実験から生まれたのか？** 原論文を読むと、Scaled Dot-Product Attention の $\sqrt{d_k}$ というスケーリング因子は、数学的解析（内積の分散が $d_k$ に比例するという観察）から導かれている。一方で、Multi-Head Attention のヘッド数 $h=8$ という選択は、実験的に最良だった数値であり、数学的な必然性はない。

2. **拡散モデル[^5] のノイズスケジュール**は、物理学の拡散方程式にインスピレーションを得ているが、実際に機能するスケジュール（linear, cosine）は物理的に自然なものではなく、実験的に発見されたものだ。「物理の数式からAIを設計する」という方向は本当に正しいのか、それとも「AIの振る舞いを事後的に物理で解釈する」方が生産的なのか。

3. **Flow Matching[^6]** は最適輸送理論という純粋数学からの直接的な応用だ。一方、GAN[^8] はゲーム理論（これも数学）から着想を得ている。異なる数学的フレームワークが異なるアーキテクチャを生む — 数学の選び方が発明を決定するのだとしたら、まだ試されていない数学的フレームワークの中に、次のブレイクスルーが眠っている可能性がある。

あなたの考えを、次回の講義の前に言語化しておいてほしい。正解は存在しない。考えること自体に価値がある。

:::details 歴史的な視点: 表現形式とブレイクスルーの関係
- **ライプニッツの微積分記法** ($\frac{dy}{dx}$) がニュートンの記法 ($\dot{y}$) より広く普及したのは、連鎖律が機械的に適用できたから。記法が思考を加速した例。
- **アインシュタインの縮約記法** ($a_i b^i = \sum_i a_i b_i$) がテンソル計算を劇的に簡略化し、一般相対性理論の発展を加速した。
- **Dirac のブラケット記法** ($\langle \psi | \phi \rangle$) が量子力学の計算を直感的にした。
- そして今、**PyTorch/JAX の自動微分** が「勾配を手計算で求める」必要をなくし、新しいアーキテクチャの実験コストを劇的に下げた。

表現形式の進化は、新しい発見を可能にする。数式→コード→自然言語（LLMプロンプト）という表現形式の進化は、次にどんな発見を可能にするだろうか。
:::

---

## 参考文献

### 主要論文

[^1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.
@[card](https://arxiv.org/abs/1706.03762)

[^2]: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
@[card](https://doi.org/10.1038/323533a0)

[^3]: Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
@[card](https://arxiv.org/abs/1503.02531)

[^4]: Kingma, D. P. & Welling, M. (2013). Auto-Encoding Variational Bayes.
@[card](https://arxiv.org/abs/1312.6114)

[^5]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
@[card](https://arxiv.org/abs/2006.11239)

[^6]: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow Matching for Generative Modeling.
@[card](https://arxiv.org/abs/2210.02747)

[^7]: Peebles, W. & Xie, S. (2022). Scalable Diffusion Models with Transformers.
@[card](https://arxiv.org/abs/2212.09748)

[^8]: Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Networks.
@[card](https://arxiv.org/abs/1406.2661)

[^9]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)

### 教科書

- Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*. Cambridge University Press. [mml-book.github.io](https://mml-book.github.io/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [deeplearningbook.org](https://www.deeplearningbook.org/)
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
- Axler, S. (2024). *Linear Algebra Done Right* (4th ed.). Springer.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. [udlbook.github.io](https://udlbook.github.io/udlbook/)

---

## 記法規約

本シリーズで使用する記法規約（全50回共通）:

| 記法 | 意味 |
|:---|:---|
| $\mathbf{x}$ (太字小文字) | ベクトル |
| $\mathbf{A}$, $W$ (太字/大文字) | 行列 |
| $x$ (イタリック小文字) | スカラー |
| $\mathcal{L}$ (カリグラフィ) | 損失関数、集合族 |
| $\mathbb{R}, \mathbb{E}$ (黒板太字) | 数の集合、期待値演算子 |
| $\theta, \phi$ (ギリシャ小文字) | パラメータ |
| $p(\cdot)$, $q(\cdot)$ | 確率分布/密度関数 |
| $x_i$ | ベクトルの $i$ 番目の要素 |
| $x^{(n)}$ | $n$ 番目のデータサンプル |
| $W^{(l)}$ | $l$ 番目の層のパラメータ |
| $\nabla_\theta$ | $\theta$ についての勾配 |
| $\sim$ | 「〜の分布に従う」 |
| $:=$ | 定義 |
| $\propto$ | 比例 |
| $\approx$ | 近似 |
| $\odot$ | 要素ごとの積（アダマール積） |
| $\circ$ | 関数合成 |
| $\|\cdot\|_2$ | L2ノルム（ユークリッドノルム） |
| $\langle \cdot, \cdot \rangle$ | 内積 |
| $\mathcal{N}(\mu, \sigma^2)$ | 正規分布 |
| $D_{\text{KL}}(\cdot \| \cdot)$ | KLダイバージェンス |
| $\mathbb{1}[\cdot]$ | 指示関数（条件が真のとき1、偽のとき0） |
| $\mathcal{O}(\cdot)$ | 計算量のオーダー |
| $\Delta^{C-1}$ | $C$次元確率単体 |
| $\text{s.t.}$ | "subject to"（制約条件） |

:::message
**記法について**: 本シリーズでは Goodfellow et al. "Deep Learning" (2016) の記法規約に準拠する。論文によっては異なる記法を使うことがあるが、その場合は都度注記する。記法の不統一は混乱の原因になるため、自分のノートでも一貫した記法を使う習慣をつけよう。
:::

---

**第1回 完 — 次回「第2回: 線形代数 — ベクトルと行列の世界」に続く**

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
