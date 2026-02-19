#!/usr/bin/env python3
"""
Rewrite short/broken Course I articles after padding removal:
- ml-lecture-03-part2.md
- ml-lecture-07-part2.md
- ml-lecture-08-part1.md
- ml-lecture-08-part2.md

Goals:
- No copy/paste padding (no large repetitive drills/Q&A).
- Add careful, step-by-step explanations (JP prose).
- GitHub-first markdown (no :::, no @[card], no $$ blocks).
- time_estimate remains "90 minutes".
Constraints:
- Part1: only 1 python code block (Quickstart) in Z1.
- Part2: 1-3 python blocks total; each must be preceded immediately by a matching ```math block.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "articles"


def w(p: Path, s: str) -> None:
    p.write_text(s.rstrip() + "\n", encoding="utf-8")


def mermaid(title: str, body: list[str]) -> str:
    return "\n".join([f"*mermaid: {title}*", "", "```mermaid", *body, "```"])


def part08_1() -> str:
    # Part1: theory, only quickstart code in Z1.
    m = []
    m.append("---")
    m.append('title: "第8回: 潜在変数モデル & EM算法 (Part1: 理論編)"')
    m.append('emoji: "🧩"')
    m.append('type: "tech"')
    m.append('topics: ["機械学習", "数学", "統計学", "Python"]')
    m.append("published: false")
    m.append('slug: "ml-lecture-08-part1"')
    m.append('difficulty: "intermediate"')
    m.append('time_estimate: "90 minutes"')
    m.append('languages: ["Python"]')
    m.append('keywords: ["潜在変数", "EM", "Jensen不等式", "ELBO", "GMM", "責任度", "単調増加"]')
    m.append("---")
    m.append("")
    m.append("> **この講義について**")
    m.append("> 本講義は「数学基礎編（Course I）」第8回 Part1（理論編）。")
    m.append("> 第7回で出た「周辺化 `\\(p(x)=\\int p(x,z)dz\\)`」を、Jensen不等式から ELBO に落とし、EM の E/M-step に分解する。")
    m.append(">")
    m.append("> **後編はこちら**: [第8回 Part2（実装編）](/articles/ml-lecture-08-part2)")
    m.append("")
    m.append("## Learning Objectives")
    m.append("")
    m.append("- [ ] 潜在変数モデル `\\(p_\\theta(x)=\\int p_\\theta(x\\mid z)p(z)dz\\)` の難しさを「`log` と `∫` の順序」で説明できる")
    m.append("- [ ] Jensen不等式から ELBO を導出し、`\\(\\log p(x)=\\mathrm{ELBO}+\\mathrm{KL}\\)` を分解できる")
    m.append("- [ ] EM の単調増加の証明の骨格を、式変形の順序で説明できる")
    m.append("- [ ] GMM の責任度 `\\(\\gamma_{ik}\\)` と M-step 更新式を shape つきで説明できる")
    m.append("- [ ] EM が壊れる典型（singularity / label switching / 初期値依存）を「起き方」で説明できる")
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🚀 Z1. クイックスタート（30秒）")
    m.append("")
    m.append("EM を最短で言うなら、「`\\(\\log p_\\theta(x)\\)` を直接押し上げる代わりに、下界（ELBO）を反復で押し上げる手続き」。")
    m.append("")
    m.append("```math")
    m.append("\\log p_\\theta(x) = \\mathrm{ELBO}(q,\\theta) + D_{\\mathrm{KL}}\\bigl(q(z)\\|p_\\theta(z\\mid x)\\bigr)")
    m.append("```")
    m.append("")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("# 1D 2-component GMM EM (minimal quickstart)")
    m.append("np.random.seed(0)")
    m.append("N = 400")
    m.append("x = np.concatenate([")
    m.append("    np.random.normal(-2.0, 0.7, N // 2),")
    m.append("    np.random.normal(2.0, 0.7, N // 2),")
    m.append("])")
    m.append("")
    m.append("K = 2")
    m.append("pi = np.array([0.5, 0.5])")
    m.append("mu = np.array([-1.0, 1.0])")
    m.append("var = np.array([1.0, 1.0])")
    m.append("")
    m.append("")
    m.append("def normal_pdf(x, mu, var):")
    m.append("    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mu) ** 2 / var)")
    m.append("")
    m.append("")
    m.append("def loglik(x, pi, mu, var, eps=1e-12):")
    m.append("    p = sum(pi[k] * normal_pdf(x, mu[k], var[k]) for k in range(K))")
    m.append("    return float(np.sum(np.log(p + eps)))")
    m.append("")
    m.append("")
    m.append("for t in range(6):")
    m.append("    # E-step")
    m.append("    r = np.stack([pi[k] * normal_pdf(x, mu[k], var[k]) for k in range(K)], axis=1)")
    m.append("    r = r / r.sum(axis=1, keepdims=True)")
    m.append("")
    m.append("    # M-step")
    m.append("    Nk = r.sum(axis=0)")
    m.append("    pi = Nk / N")
    m.append("    mu = (r * x[:, None]).sum(axis=0) / Nk")
    m.append("    var = (r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk")
    m.append("")
    m.append("    print('t=', t, 'loglik=', loglik(x, pi, mu, var))")
    m.append("```")
    m.append("")
    m.append("> **Note:** ここでは雰囲気だけ。以降は「なぜ上がるのか」を Jensen から、順番を崩さずに出す。")
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 📖 Z2. チュートリアル（10分）— 潜在変数モデルの定式化")
    m.append("")
    m.append("潜在変数モデルは、観測 `\\(x\\)` を直接書くのではなく、潜在 `\\(z\\)` を介して生成過程を設計する。")
    m.append("")
    m.append("```math")
    m.append("p_\\theta(x) = \\int p_\\theta(x\\mid z)\\,p(z)\\,dz")
    m.append("```")
    m.append("")
    m.append("ここで最初の躓きは、**目的関数が `\\(\\log\\int\\)` の形になる**こと。")
    m.append("")
    m.append("```math")
    m.append("\\mathcal{L}(\\theta)=\\sum_{i=1}^N \\log p_\\theta(x_i)")
    m.append("=\\sum_{i=1}^N \\log \\int p_\\theta(x_i,z_i)\\,dz_i")
    m.append("```")
    m.append("")
    m.append("「積分が難しい」より前に、「`log` と `∫` の順序が悪い」。この順序の悪さが、勾配計算の形を壊す。")
    m.append("")
    m.append(mermaid("`log` と `∫` の順序が悪い", [
        "flowchart LR",
        "  A[want: maximize log p(x|θ)] --> B[p(x|θ)=∫ p(x,z|θ) dz]",
        "  B --> C[objective: log ∫ ... dz]",
        "  C --> D[hard to move log inside / hard gradients]",
        "  D --> E[introduce q(z) + Jensen]",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🌍 Z3. 世界観（20分）— Jensen → ELBO → EM の一本道")
    m.append("")
    m.append("### 3.1 Jensen不等式は「ログを期待値の内側に押し込む」道具")
    m.append("")
    m.append("`\\(\\log\\)` は凹関数。だから、期待値に対して次が成り立つ。")
    m.append("")
    m.append("```math")
    m.append("\\log \\mathbb{E}[Y] \\ge \\mathbb{E}[\\log Y]")
    m.append("```")
    m.append("")
    m.append("しかし今は `\\(\\mathbb{E}\\)` がない。そこで補助分布 `\\(q(z)\\)` を自分で差し込む。")
    m.append("")
    m.append("```math")
    m.append("p_\\theta(x)")
    m.append("=\\int p_\\theta(x,z)\\,dz")
    m.append("=\\int q(z)\\,\\frac{p_\\theta(x,z)}{q(z)}\\,dz")
    m.append("=\\mathbb{E}_{q(z)}\\Bigl[\\frac{p_\\theta(x,z)}{q(z)}\\Bigr]")
    m.append("```")
    m.append("")
    m.append("ここまで来れば Jensen が打てる。")
    m.append("")
    m.append("```math")
    m.append("\\log p_\\theta(x)")
    m.append("=\\log\\mathbb{E}_{q(z)}\\Bigl[\\frac{p_\\theta(x,z)}{q(z)}\\Bigr]")
    m.append("\\ge \\mathbb{E}_{q(z)}[\\log p_\\theta(x,z) - \\log q(z)]")
    m.append("```")
    m.append("")
    m.append("右辺を **ELBO** と呼ぶ。")
    m.append("")
    m.append("```math")
    m.append("\\mathrm{ELBO}(q,\\theta):=\\mathbb{E}_{q(z)}[\\log p_\\theta(x,z)-\\log q(z)]")
    m.append("```")
    m.append("")
    m.append("<details><summary>なぜ「わざわざ q(z) を入れる」のか（直感）</summary>")
    m.append("")
    m.append("潜在変数モデルの地獄は `\\(\\log\\int\\)`。")
    m.append("")
    m.append("- `\\(\\int\\)` の外に `\\(\\log\\)` があると、積分の中身を「確率の足し算」として扱えない")
    m.append("- 逆に `\\(\\mathbb{E}[\\log]\\)` の形にできると、`\\(\\log\\)` が積や指数の世界で素直に働く")
    m.append("")
    m.append("`q(z)` は、数学的には「期待値の入れ物」を作るための道具で、推論的には「潜在の仮の説明」を置く道具。")
    m.append("")
    m.append("</details>")
    m.append("")
    m.append("### 3.2 ELBO分解（ギャップは KL）")
    m.append("")
    m.append("ELBO は「下界」だが、それだけだと弱い。EM が強いのは、ギャップが KL として正体を持つから。")
    m.append("")
    m.append("```math")
    m.append("\\log p_\\theta(x) = \\mathrm{ELBO}(q,\\theta) + D_{\\mathrm{KL}}\\bigl(q(z)\\|p_\\theta(z\\mid x)\\bigr)")
    m.append("```")
    m.append("")
    m.append("ここは丁寧に言う。**ELBOは『正しい確率』ではない**。")
    m.append("")
    m.append("- ELBO は `q` と `θ` の関数で、`q` の取り方で値が変わる")
    m.append("- `\\(\\log p_\\theta(x)\\)` は `θ` だけの関数で、観測モデルが決める本体")
    m.append("")
    m.append("この差を埋めるのが `D_KL(q||p)`。KL が 0 なら、ELBO は本体に一致する。")
    m.append("")
    m.append("この式が一気に3つを言う。")
    m.append("")
    m.append("- `\\(D_{\\mathrm{KL}}\\ge 0\\)` だから `\\(\\mathrm{ELBO}\\le\\log p_\\theta(x)\\)`（下界）")
    m.append("- ギャップは「`q` と真の事後 `p(z|x,θ)` のズレ」")
    m.append("- ギャップを 0 にできれば、下界を上げることが本体を上げることになる")
    m.append("")
    m.append(mermaid("分解の見取り図", [
        "flowchart LR",
        "  L[log p_θ(x)] -->|=| E[ELBO(q,θ)]",
        "  L -->|+| K[KL(q(z)||p_θ(z|x))]",
        "  K -->|≥ 0| Gap[gap]",
    ]))
    m.append("")
    m.append("### 3.3 EM の2ステップ（交互最適化）")
    m.append("")
    m.append("EM は ELBO を、`q` と `θ` で交互に最大化する。")
    m.append("")
    m.append("```math")
    m.append("\\textbf{E-step:}\\quad q^{(t+1)} = \\arg\\max_q\\,\\mathrm{ELBO}(q,\\theta^{(t)})")
    m.append("```")
    m.append("")
    m.append("この最適解は `q^{(t+1)}(z)=p_{\\theta^{(t)}}(z\\mid x)`。理由は、ELBO 分解式で `q` を動かすと KL を最小化する問題になるから。")
    m.append("")
    m.append("```math")
    m.append("\\textbf{M-step:}\\quad \\theta^{(t+1)} = \\arg\\max_\\theta\\,\\mathrm{ELBO}(q^{(t+1)},\\theta)")
    m.append("```")
    m.append("")
    m.append(mermaid("EMの反復", [
        "flowchart TD",
        "  A[init θ^(0)] --> B[E-step: set q(z)=p(z|x,θ^(t))]",
        "  B --> C[M-step: maximize ELBO over θ]",
        "  C --> D{converged?}",
        "  D -- no --> B",
        "  D -- yes --> E[θ*]",
    ]))
    m.append("")
    m.append("### 3.4 単調増加の「順序」をもう一度固定する")
    m.append("")
    m.append("EM の単調増加は、計算テクニックではなく『不等式のつなぎ方』。この順序が崩れると、証明は崩れる。")
    m.append("")
    m.append("```math")
    m.append("\\log p_{\\theta^{(t)}}(x)")
    m.append("= \\mathrm{ELBO}(q^{(t+1)},\\theta^{(t)}) + D_{\\mathrm{KL}}(q^{(t+1)}\\|p_{\\theta^{(t)}}(z\\mid x))")
    m.append("= \\mathrm{ELBO}(q^{(t+1)},\\theta^{(t)})")
    m.append("```")
    m.append("")
    m.append("E-step で `q^{(t+1)}=p(z|x,θ^(t))` を選ぶと、KL が 0 になる。だからこの瞬間、ELBO が本体になる。")
    m.append("")
    m.append("次に M-step。")
    m.append("")
    m.append("```math")
    m.append("\\mathrm{ELBO}(q^{(t+1)},\\theta^{(t+1)}) \\ge \\mathrm{ELBO}(q^{(t+1)},\\theta^{(t)})")
    m.append("```")
    m.append("")
    m.append("ELBO が上がるなら、その上にある `\\(\\log p_\\theta(x)\\)` も下がりにくい。これが単調増加の骨格。")
    m.append("")
    m.append("---")
    m.append("")
    m.append("## ⚔️ Z4. Boss Battle（60分）— GMM-EM を多変量で導出する")
    m.append("")
    m.append("Part2 で実装するために、ここでやるべきことは一つ。")
    m.append("")
    m.append("**どの記号が、どの配列 shape に落ちるかを固定したまま、E-step と M-step を最後まで閉じる。**")
    m.append("")
    m.append("### 4.1 GMM のモデル")
    m.append("")
    m.append("観測は `x_i∈R^d`、成分は `k=1..K`。パラメータは `θ=(π,μ,Σ)`。")
    m.append("")
    m.append("```math")
    m.append("p_\\theta(x)=\\sum_{k=1}^K \\pi_k\\,\\mathcal{N}(x\\mid\\mu_k,\\Sigma_k),")
    m.append("\\quad \\sum_k \\pi_k=1,\\ \\pi_k\\ge 0")
    m.append("```")
    m.append("")
    m.append("潜在変数 `z_i` は離散で、`z_i=k` が「どのガウスから来たか」を表す。")
    m.append("")
    m.append(mermaid("GMM の生成", [
        "flowchart LR",
        "  S{choose k ~ Cat(π)} --> G[N(x|μ_k,Σ_k)]",
        "  G --> X[x]",
    ]))
    m.append("")
    m.append("### 4.2 完全データ対数尤度（ログを取る理由）")
    m.append("")
    m.append("one-hot 表現 `z_{ik}∈{0,1}`（`∑_k z_{ik}=1`）を使うと、積が和に変わる。")
    m.append("")
    m.append("```math")
    m.append("p_\\theta(x_i,z_i)=\\prod_{k=1}^K \\bigl(\\pi_k\\,\\mathcal{N}(x_i\\mid\\mu_k,\\Sigma_k)\\bigr)^{z_{ik}}")
    m.append("```")
    m.append("")
    m.append("```math")
    m.append("\\log p_\\theta(x_i,z_i)=\\sum_{k=1}^K z_{ik}\\bigl(\\log\\pi_k+\\log\\mathcal{N}(x_i\\mid\\mu_k,\\Sigma_k)\\bigr)")
    m.append("```")
    m.append("")
    m.append("### 4.3 E-step: 責任度（posterior over components）")
    m.append("")
    m.append("責任度は事後確率そのもの。式はベイズ則で1行だが、実装では数値安定性が勝負になる（Part2）。")
    m.append("")
    m.append("```math")
    m.append("\\gamma_{ik}:=p(z_i=k\\mid x_i,\\theta^{(t)})")
    m.append("=\\frac{\\pi_k^{(t)}\\,\\mathcal{N}(x_i\\mid\\mu_k^{(t)},\\Sigma_k^{(t)})}{\\sum_{j=1}^K \\pi_j^{(t)}\\,\\mathcal{N}(x_i\\mid\\mu_j^{(t)},\\Sigma_j^{(t)})}")
    m.append("```")
    m.append("")
    m.append("**最低限の検算**: `γ_{ik}≥0` と `∑_k γ_{ik}=1`。これが崩れた時点で以降は全部壊れる。")
    m.append("")
    m.append("<details><summary>責任度を「softmax」として見る</summary>")
    m.append("")
    m.append("E-step の分子は `\\(\\pi_k\\mathcal{N}(x_i|\\mu_k,\\Sigma_k)\\)`。これを `k` で正規化しているだけ。")
    m.append("")
    m.append("だから log 空間では、")
    m.append("")
    m.append("```math")
    m.append("\\gamma_{ik} = \\mathrm{softmax}_k\\bigl(\\log\\pi_k + \\log \\mathcal{N}(x_i\\mid\\mu_k,\\Sigma_k)\\bigr)")
    m.append("```")
    m.append("")
    m.append("実装で `log-sum-exp` が必要になる理由もここにある。")
    m.append("")
    m.append("</details>")
    m.append("")
    m.append("### 4.4 M-step: 更新式（重み付き統計量）")
    m.append("")
    m.append("`N_k` は「成分kに割り当てられた有効サンプル数」。この `N_k` を軸に更新が揃う。")
    m.append("")
    m.append("```math")
    m.append("N_k := \\sum_{i=1}^N \\gamma_{ik}")
    m.append("```")
    m.append("")
    m.append("混合比は正規化、平均は重み付き平均、共分散は重み付き共分散。")
    m.append("")
    m.append("```math")
    m.append("\\pi_k \\leftarrow \\frac{N_k}{N},\\qquad")
    m.append("\\mu_k \\leftarrow \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik}x_i,\\qquad")
    m.append("\\Sigma_k \\leftarrow \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik}(x_i-\\mu_k)(x_i-\\mu_k)^\\top")
    m.append("```")
    m.append("")
    m.append("shape を固定する。")
    m.append("")
    m.append("- `X ∈ R^{N×d}`")
    m.append("- `γ ∈ R^{N×K}`（行がデータ、列が成分）")
    m.append("- `μ ∈ R^{K×d}`")
    m.append("- `Σ ∈ R^{K×d×d}`")
    m.append("")
    m.append("### 4.5 単調増加（なぜ loglik が落ちないか）")
    m.append("")
    m.append("証明は「難しい定理」ではない。順序がすべて。")
    m.append("")
    m.append("1. `θ^(t)` を固定し、E-step で `q(z)=p(z|x,θ^(t))` を選ぶ。ここで KL ギャップが 0 になる。")
    m.append("2. `q` を固定し、M-step で ELBO を最大化する。ELBO が上がる。")
    m.append("3. ギャップ 0 のままなので `log p(x|θ)` も上がる。")
    m.append("")
    m.append("### 4.6 EM の拡張と「壊れ方」")
    m.append("")
    m.append("- **GEM**: M-step を厳密最大化できなくても、`Q` を増やせば単調性は保てる（最大化の代わりに改善）")
    m.append("- **ECM**: パラメータをブロックに分けて条件付き最大化（実装で現れやすい）")
    m.append("- **label switching**: 成分ラベルは置換しても同じ分布（「ラベル一致」を評価基準にしない）")
    m.append("- **singularity**: ある成分が一点に潰れると尤度が発散し得る（共分散が特異になる）")
    m.append("")
    m.append(mermaid("EMが壊れる典型", [
        "flowchart TD",
        "  A[bad init] --> B[one component dominates]",
        "  B --> C[another becomes empty (Nk→0)]",
        "  C --> D[Σ becomes singular]",
        "  D --> E[Cholesky fails / loglik spikes]",
        "  E --> F[fix: epsI / reinit / priors]",
    ]))
    m.append("")
    m.append("### 4.7 Missing data（欠損）としての潜在変数")
    m.append("")
    m.append("潜在変数は『構造の源』としてだけでなく『欠損』としても現れる。ここで大事なのは、欠損のメカニズム。")
    m.append("")
    m.append("- MCAR: 欠損が完全にランダム")
    m.append("- MAR: 欠損は観測変数に依存してもよい（未観測には依存しない）")
    m.append("- MNAR: 欠損が未観測に依存する（モデル化なしに扱うと破綻しやすい）")
    m.append("")
    m.append("EM は『欠損を潜在として埋める』手続きに見えるが、MNAR を無視して適用すると推定が系統的に歪む。")
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 参考文献")
    m.append("")
    m.append("[^1]: <https://arxiv.org/abs/0710.5696>")
    m.append("[^2]: <https://arxiv.org/abs/cs/0412015>")
    m.append("[^3]: <https://arxiv.org/abs/1105.1476>")
    m.append("[^4]: <https://arxiv.org/abs/1301.2294>")
    m.append("[^5]: <https://arxiv.org/abs/1601.00670>")
    m.append("")
    m.append("## 著者リンク")
    m.append("- Blog: https://fumishiki.dev")
    m.append("- X: https://x.com/fumishiki")
    m.append("- LinkedIn: https://www.linkedin.com/in/fumitakamurakami")
    m.append("- GitHub: https://github.com/fumishiki")
    m.append("- Hugging Face: https://huggingface.co/fumishiki")
    m.append("")
    m.append("## ライセンス")
    m.append("- CC BY-NC-SA 4.0")
    return "\n".join(m)


def part08_2() -> str:
    # Part2: 2 python blocks, each immediately preceded by a math block.
    m: list[str] = []
    m += [
        "---",
        'title: "第8回: 潜在変数モデル & EM算法 (Part2: 実装編)"',
        'emoji: "🧩"',
        'type: "tech"',
        'topics: ["機械学習", "数学", "統計学", "Python"]',
        "published: false",
        'slug: "ml-lecture-08-part2"',
        'difficulty: "intermediate"',
        'time_estimate: "90 minutes"',
        'languages: ["Python"]',
        'keywords: ["GMM", "EM", "log-sum-exp", "Cholesky", "BIC", "AIC", "初期化", "singularity"]',
        "---",
        "",
        "> **この講義について**",
        "> Part1 の導出を、壊れない形で動かす。式とコードの対応がずれた瞬間に破綻するので、shape と数値安定性を最初に固定する。",
        ">",
        "> **前編はこちら**: [第8回 Part1（理論編）](/articles/ml-lecture-08-part1)",
        "",
        "## Learning Objectives",
        "",
        "- [ ] `\\(\\gamma_{ik}\\)`（責任度）を log-sum-exp で安定に計算できる",
        "- [ ] 多変量GMMの M-step を shape つきで実装できる",
        "- [ ] 尤度単調増加を数値で確認できる（`assert` できる）",
        "- [ ] BIC/AIC によるモデル選択を「パラメータ数」から実装できる",
        "- [ ] singularity / empty component を検出して対処できる",
        "",
        "---",
        "",
        "## 🛠️ Z5. 実装ゾーン（60分）— GMM-EM を壊れない最小形で実装",
        "",
        "### 5.1 設計: どこが壊れるかを先に潰す",
        "",
        "GMM-EM は、式は短いのに実装が壊れやすい。理由は、**確率計算が指数関数と行列分解を含む**から。",
        "",
        "- E-step: `\\(\\pi_k\\mathcal{N}(x\\mid\\mu_k,\\Sigma_k)\\)` が underflow で 0 になり、`γ` が NaN になる",
        "- `\\(\\Sigma_k\\)` が数値誤差で非対称・非SPDになり、Cholesky が落ちる",
        "- `N_k` が小さくなり、平均/共分散が発散する（empty component）",
        "",
        "ここでは **log空間** と **Cholesky** を固定し、それ以外は極力素直に書く。",
        "",
        m[0] if False else "",  # keep list type stable
    ]
    m = [x for x in m if x != ""]
    m.append(mermaid("数値安定性の流れ", [
        "flowchart TD",
        "  A[raw π_k N(x|μ_k,Σ_k)] --> B[underflow/overflow]",
        "  B --> C[log space: log π_k + log N]",
        "  C --> D[normalize via log-sum-exp]",
        "  D --> E[exp -> γ]",
        "  E --> F[assert row-sum=1]",
    ]))
    m.append("")
    m.append(mermaid("EM のデータフロー（実装視点）", [
        "flowchart LR",
        "  X[X: N×d] --> E[e_step -> γ: N×K]",
        "  E --> M[m_step -> π,μ,Σ]",
        "  M --> L[loglik -> scalar]",
        "  L --> Stop{converged?}",
        "  Stop -- no --> E",
        "  Stop -- yes --> Out[model]",
    ]))
    m.append("")
    m.append("### 5.2 E-step: log-sum-exp で責任度を作る")
    m.append("")
    m.append("E-step の本体は「softmax」。ただし通常の softmax をそのまま当てると `exp` が壊れる。")
    m.append("")
    m.append("まず、log-sum-exp の式を固定する。")
    m.append("")
    m.append("```math")
    m.append("\\log\\sum_{k=1}^K e^{a_k} = m + \\log\\sum_{k=1}^K e^{a_k-m},\\quad m=\\max_k a_k")
    m.append("```")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("")
    m.append("def logsumexp(a, axis=-1):")
    m.append("    m = np.max(a, axis=axis, keepdims=True)")
    m.append("    s = np.sum(np.exp(a - m), axis=axis, keepdims=True)")
    m.append("    return (m + np.log(s)).squeeze(axis)")
    m.append("")
    m.append("")
    m.append("# sanity: overflow-safe")
    m.append("z = np.array([1000.0, 999.0, 998.0])")
    m.append("print('naive exp finite? ->', np.isfinite(np.sum(np.exp(z))))")
    m.append("print('logsumexp        ->', float(logsumexp(z)))")
    m.append("```")
    m.append("")
    m.append("ここから先の `a_k` は、GMM では `log π_k + log N(x|μ_k,Σ_k)` に対応する。")
    m.append("")
    m.append("### 5.3 E-step + M-step + loglik + BIC/AIC を1本にまとめる（load-bearing code）")
    m.append("")
    m.append("次の数式ブロックは、この後のコードと 1:1 で対応する。")
    m.append("")
    m.append("```math")
    m.append("\\log \\gamma_{ik}")
    m.append("= \\log \\pi_k + \\log \\mathcal{N}(x_i\\mid\\mu_k,\\Sigma_k)")
    m.append("- \\log\\sum_{j=1}^K \\exp\\Bigl(\\log \\pi_j + \\log \\mathcal{N}(x_i\\mid\\mu_j,\\Sigma_j)\\Bigr)")
    m.append("")
    m.append("N_k = \\sum_{i=1}^N \\gamma_{ik},\\quad")
    m.append("\\pi_k = \\frac{N_k}{N},\\quad")
    m.append("\\mu_k = \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik}x_i,\\quad")
    m.append("\\Sigma_k = \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik}(x_i-\\mu_k)(x_i-\\mu_k)^\\top")
    m.append("")
    m.append("\\mathrm{AIC}=2\\,k_{\\mathrm{params}}-2\\,\\log p(X\\mid\\hat\\theta),")
    m.append("\\qquad")
    m.append("\\mathrm{BIC}=\\log(N)\\,k_{\\mathrm{params}}-2\\,\\log p(X\\mid\\hat\\theta)")
    m.append("```")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("")
    m.append("def logsumexp(a, axis=-1):")
    m.append("    m = np.max(a, axis=axis, keepdims=True)")
    m.append("    s = np.sum(np.exp(a - m), axis=axis, keepdims=True)")
    m.append("    return (m + np.log(s)).squeeze(axis)")
    m.append("")
    m.append("")
    m.append("def log_mvnormal(X, mu_k, Sigma_k, eps=1e-6):")
    m.append("    # X: (N,d), mu_k: (d,), Sigma_k: (d,d)")
    m.append("    N, d = X.shape")
    m.append("    Sigma_k = 0.5 * (Sigma_k + Sigma_k.T) + eps * np.eye(d)")
    m.append("    L = np.linalg.cholesky(Sigma_k)")
    m.append("    Y = np.linalg.solve(L, (X - mu_k).T)  # (d,N)")
    m.append("    quad = np.sum(Y * Y, axis=0)          # (N,)")
    m.append("    logdet = 2.0 * np.sum(np.log(np.diag(L)))")
    m.append("    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)")
    m.append("")
    m.append("")
    m.append("def e_step(X, pi, mu, Sigma):")
    m.append("    N = X.shape[0]")
    m.append("    K = pi.shape[0]")
    m.append("    log_r = np.zeros((N, K))")
    m.append("    for k in range(K):")
    m.append("        log_r[:, k] = np.log(pi[k] + 1e-12) + log_mvnormal(X, mu[k], Sigma[k])")
    m.append("    log_norm = logsumexp(log_r, axis=1)")
    m.append("    gamma = np.exp(log_r - log_norm[:, None])")
    m.append("    row_sum = gamma.sum(axis=1)")
    m.append("    assert float(np.max(np.abs(row_sum - 1.0))) < 1e-6")
    m.append("    return gamma")
    m.append("")
    m.append("")
    m.append("def m_step(X, gamma, eps=1e-6, empty_thresh=1e-3):")
    m.append("    N, d = X.shape")
    m.append("    K = gamma.shape[1]")
    m.append("    Nk = gamma.sum(axis=0) + 1e-12")
    m.append("")
    m.append("    # empty component detection: re-init mean to a random data point")
    m.append("    rng = np.random.default_rng(0)")
    m.append("    for k in range(K):")
    m.append("        if Nk[k] / float(N) < empty_thresh:")
    m.append("            gamma[:, k] = 0.0")
    m.append("            gamma[rng.integers(0, N), k] = 1.0")
    m.append("    Nk = gamma.sum(axis=0) + 1e-12")
    m.append("")
    m.append("    pi = Nk / float(N)")
    m.append("    mu = (gamma.T @ X) / Nk[:, None]")
    m.append("")
    m.append("    Sigma = np.zeros((K, d, d))")
    m.append("    for k in range(K):")
    m.append("        Xc = X - mu[k][None, :]")
    m.append("        Sigma[k] = (gamma[:, k][:, None] * Xc).T @ Xc / Nk[k]")
    m.append("        Sigma[k] = 0.5 * (Sigma[k] + Sigma[k].T) + eps * np.eye(d)")
    m.append("    return pi, mu, Sigma")
    m.append("")
    m.append("")
    m.append("def loglik_gmm(X, pi, mu, Sigma):")
    m.append("    N = X.shape[0]")
    m.append("    K = pi.shape[0]")
    m.append("    log_r = np.zeros((N, K))")
    m.append("    for k in range(K):")
    m.append("        log_r[:, k] = np.log(pi[k] + 1e-12) + log_mvnormal(X, mu[k], Sigma[k])")
    m.append("    return float(np.sum(logsumexp(log_r, axis=1)))")
    m.append("")
    m.append("")
    m.append("def run_em(X, K, steps=30, seed=0):")
    m.append("    rng = np.random.default_rng(seed)")
    m.append("    N, d = X.shape")
    m.append("")
    m.append("    # init: choose K points as means, shared covariance")
    m.append("    idx = rng.choice(N, size=K, replace=False)")
    m.append("    mu = X[idx].copy()")
    m.append("    pi = np.ones(K) / K")
    m.append("    Sigma0 = np.cov(X.T) + 1e-3 * np.eye(d)")
    m.append("    Sigma = np.stack([Sigma0.copy() for _ in range(K)], axis=0)")
    m.append("")
    m.append("    ll_hist = []")
    m.append("    for t in range(steps):")
    m.append("        gamma = e_step(X, pi, mu, Sigma)")
    m.append("        pi, mu, Sigma = m_step(X, gamma)")
    m.append("        ll = loglik_gmm(X, pi, mu, Sigma)")
    m.append("        ll_hist.append(ll)")
    m.append("        if t >= 1:")
    m.append("            assert ll_hist[-1] >= ll_hist[-2] - 1e-6")
    m.append("    return pi, mu, Sigma, np.array(ll_hist)")
    m.append("")
    m.append("")
    m.append("def aic_bic(loglik, N, k_params):")
    m.append("    aic = 2.0 * k_params - 2.0 * loglik")
    m.append("    bic = np.log(float(N)) * k_params - 2.0 * loglik")
    m.append("    return aic, bic")
    m.append("")
    m.append("")
    m.append("# demo: synthetic 2D mixture")
    m.append("rng = np.random.default_rng(0)")
    m.append("X = np.vstack([")
    m.append("    rng.normal(loc=(-2.0, 0.0), scale=0.6, size=(200, 2)),")
    m.append("    rng.normal(loc=(+2.0, 0.0), scale=0.6, size=(200, 2)),")
    m.append("])")
    m.append("")
    m.append("for K in [1, 2, 3, 4]:")
    m.append("    pi, mu, Sigma, ll_hist = run_em(X, K, steps=20, seed=K)")
    m.append("    ll = float(ll_hist[-1])")
    m.append("    N, d = X.shape")
    m.append("    k_params = (K - 1) + K * d + K * (d * (d + 1) // 2)")
    m.append("    aic, bic = aic_bic(ll, N, k_params)")
    m.append("    print('K=', K, 'loglik=', ll, 'AIC=', aic, 'BIC=', bic)")
    m.append("```")
    m.append("")
    m.append("ここでの丁寧ポイントを、コードに紐づけて言い切る。")
    m.append("")
    m.append("- `log_mvnormal`: `Σ` を **対称化**してから `eps I` を足す。SPD を守ると Cholesky が生きる。")
    m.append("- `e_step`: `logsumexp` 正規化の後、`row_sum==1` を `assert` する。ここで壊れたら以降は全部嘘。")
    m.append("- `m_step`: `Nk` が小さい成分（empty component）を検出して再初期化する。実務ではこれがないと破綻が早い。")
    m.append("- `run_em`: 単調増加 `ll_t ≥ ll_{t-1}` を `assert` する（数値誤差分のスラックだけ許す）。")
    m.append("")
    m.append(mermaid("モデル選択（AIC/BIC）", [
        "flowchart LR",
        "  D[data X] --> F[fit EM for each K]",
        "  F --> L[loglik(K)]",
        "  L --> A[AIC(K)]",
        "  L --> B[BIC(K)]",
        "  A --> S[select K]",
        "  B --> S",
    ]))
    m.append("")
    m.append(mermaid("singularity と empty component の関係", [
        "flowchart TD",
        "  A[γ becomes nearly one-hot] --> B[Nk for some k -> 0]",
        "  B --> C[μ/Σ update becomes unstable]",
        "  C --> D[Σ loses SPD / collapses]",
        "  D --> E[loglik spikes, Cholesky fails]",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🔬 Z6. 子孫たち（20分）— EM が再登場する場所")
    m.append("")
    m.append("EM は「潜在を入れた結果、事後 `p(z|x)` を扱う必要が出た」状況で現れる。代表例だけ押さえる。")
    m.append("")
    m.append("- HMM: 潜在状態列 `z_{1:T}` の事後推論が本体（Baum-Welch）[^2]")
    m.append("- PPCA/FA: 線形ガウス潜在では閉じるが、EM で書くと更新の見通しが良い")
    m.append("- Variational EM: `q(z)` を制約付きで最適化し、ELBO を最大化する（VAE に接続）[^5]")
    m.append("- EP: KL とは違う射影で近似を作る系譜（Minka）[^4]")
    m.append("")
    m.append(mermaid("系譜図（最小）", [
        "flowchart TD",
        "  EM[EM] --> HMM[Baum-Welch]",
        "  EM --> FA[Factor Analysis]",
        "  FA --> PPCA[Probabilistic PCA]",
        "  EM --> VEM[Variational EM]",
        "  VEM --> VAE[VAE]",
        "  EM --> EP[Expectation Propagation]",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🎓 Z7. まとめ（10分）")
    m.append("")
    m.append("- EM の出発点は `\\(\\log\\int\\)` の形の悪さ")
    m.append("- Jensen によって ELBO を作り、ギャップが KL であることが「単調増加」を支える")
    m.append("- 実装は log 空間 + Cholesky + `assert` が本体（性能より先に正しさ）")
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 参考文献")
    m.append("")
    m.append("[^1]: <https://arxiv.org/abs/0710.5696>")
    m.append("[^2]: <https://arxiv.org/abs/cs/0412015>")
    m.append("[^3]: <https://arxiv.org/abs/1105.1476>")
    m.append("[^4]: <https://arxiv.org/abs/1301.2294>")
    m.append("[^5]: <https://arxiv.org/abs/1601.00670>")
    m.append("")
    m.append("## 著者リンク")
    m.append("- Blog: https://fumishiki.dev")
    m.append("- X: https://x.com/fumishiki")
    m.append("- LinkedIn: https://www.linkedin.com/in/fumitakamurakami")
    m.append("- GitHub: https://github.com/fumishiki")
    m.append("- Hugging Face: https://huggingface.co/fumishiki")
    m.append("")
    m.append("## ライセンス")
    m.append("- CC BY-NC-SA 4.0")
    return "\n".join(m)


def part03_2() -> str:
    # Part2: keep 2 python blocks with matching math directly above; add detailed prose.
    m: list[str] = []
    m += [
        "---",
        'title: "第3回: 線形代数 II: SVD・行列微分・テンソル — 万能ナイフSVDと逆伝播の数学 【後編】実装編"',
        'emoji: "🔬"',
        'type: "tech"',
        'topics: ["machinelearning", "deeplearning", "linearalgebra", "python"]',
        "published: true",
        'difficulty: "★★★★☆"',
        'time_estimate: "90 minutes"',
        'languages: ["Python"]',
        'keywords: ["SVD", "低ランク近似", "行列微分", "数値微分", "einsum", "shape", "Attention"]',
        "---",
        "",
        "# 第3回: 線形代数 II — SVD・行列微分・テンソル【後編】",
        "",
        "> **理論編へのリンク**: [第3回 Part1（理論編）](/articles/ml-lecture-03-part1)",
        "",
        "## Learning Objectives",
        "",
        "- [ ] truncated SVD（ランクk近似）を「shapeの契約」を落とさず実装できる",
        "- [ ] 最適性（捨てた特異値のエネルギーが誤差になる）を数値で検算できる",
        "- [ ] 行列微分の基本パターンを、`@` と `einsum` に1:1で落とせる",
        "- [ ] 数値微分で勾配を検算し、実装の嘘を炙り出せる",
        "",
        "---",
        "",
        "## 💻 Z5. 実装ゾーン（75分）— 「式が壊れていない」ことを証明する実装",
        "",
        "この実装編の主役は速度ではない。主役は **検算**。",
    ]
    m.append("")
    m.append("線形代数の実装が壊れる典型は、いつも同じ。")
    m.append("")
    m.append("- shape を暗黙にしてしまう（`(m,n)` と `(n,)` の区別が溶ける）")
    m.append("- 対称性/正定値性を落とす（数値誤差が勝つ）")
    m.append("- 「同じはず」を厳密一致で比較する（SVD の符号自由度で死ぬ）")
    m.append("")
    m.append(mermaid("壊れ方の分類", [
        "flowchart TD",
        "  B[bug] --> S[shape mismatch]",
        "  B --> N[numerical instability]",
        "  B --> I[indexing/transpose]",
        "  B --> C[conceptual mismatch]",
        "  S --> A[assert shapes]",
        "  N --> E[eps/symmetrize]",
    ]))
    m.append("")
    m.append("### 5.1 truncated SVD（低ランク近似）")
    m.append("")
    m.append("SVD を使う理由は一言で済む。「行列を、情報量の順に並べ替えられる」から。")
    m.append("")
    m.append("ここでのポイントは2つ。")
    m.append("")
    m.append("1. `A_k` の構成が shape どおりに書けていること")
    m.append("2. `||A-A_k||_F` が「捨てた特異値の二乗和」に一致すること")
    m.append("")
    m.append("```math")
    m.append("A = U\\Sigma V^\\top,\\quad \\Sigma=\\mathrm{diag}(\\sigma_1,\\dots,\\sigma_r),\\ r=\\min(m,n)")
    m.append("")
    m.append("A_k = U_{[:,1:k]}\\,\\Sigma_{1:k,1:k}\\,V^\\top_{[1:k,:]}")
    m.append("")
    m.append("\\|A-A_k\\|_F^2 = \\sum_{i=k+1}^{r} \\sigma_i^2")
    m.append("```")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("")
    m.append("def svd_rank_k(A: np.ndarray, k: int) -> np.ndarray:")
    m.append("    # A: (m,n)")
    m.append("    U, s, Vt = np.linalg.svd(A, full_matrices=False)")
    m.append("    # U: (m,r), s: (r,), Vt: (r,n)")
    m.append("    return U[:, :k] @ (s[:k, None] * Vt[:k, :])")
    m.append("")
    m.append("")
    m.append("def rel_fro_error(A: np.ndarray, B: np.ndarray) -> float:")
    m.append("    return float(np.linalg.norm(A - B, ord='fro') / np.linalg.norm(A, ord='fro'))")
    m.append("")
    m.append("")
    m.append("def tail_energy_bound(s: np.ndarray, k: int) -> float:")
    m.append("    num = float(np.sum(s[k:] ** 2))")
    m.append("    den = float(np.sum(s ** 2)) + 1e-12")
    m.append("    return float(np.sqrt(num / den))")
    m.append("")
    m.append("")
    m.append("rng = np.random.default_rng(0)")
    m.append("A = rng.normal(size=(128, 96))")
    m.append("U, s, Vt = np.linalg.svd(A, full_matrices=False)")
    m.append("")
    m.append("prev = 1.0")
    m.append("for k in [1, 5, 10, 20, 40, 80]:")
    m.append("    Ak = svd_rank_k(A, k)")
    m.append("    err = rel_fro_error(A, Ak)")
    m.append("    bound = tail_energy_bound(s, k)")
    m.append("    assert err <= prev + 1e-10")
    m.append("    assert abs(err - bound) < 1e-6")
    m.append("    prev = err")
    m.append("    print(f'k={k:3d}  rel_fro_err={err:.6f}')")
    m.append("```")
    m.append("")
    m.append("**丁寧ポイント**: `diag(s)` を作らない。`s[:k,None] * Vt[:k,:]` で `k×n` を直接作る。")
    m.append("")
    m.append("`U[:, :k]` が `m×k`、右側が `k×n` なので、積は `m×n` に戻る。ここで shape が合わないなら切り方が間違い。")
    m.append("")
    m.append("SVD の自由度（符号反転など）で「Uが一致しない」問題があるが、この実装では **再構成誤差** を検算しているので影響を受けない。")
    m.append("")
    m.append(mermaid("SVD→truncated SVD", [
        "flowchart LR",
        "  A[A] --> SVD[SVD]",
        "  SVD --> U[U]",
        "  SVD --> Sig[Σ]",
        "  SVD --> V[V^T]",
        "  U --> Cut[keep top-k]",
        "  Sig --> Cut",
        "  V --> Cut",
        "  Cut --> Ak[A_k]",
    ]))
    m.append("")
    m.append("### 5.2 行列微分（逆伝播の最小核）")
    m.append("")
    m.append("逆伝播の実装が壊れる瞬間は、「どれで微分しているか」が曖昧になった瞬間。")
    m.append("")
    m.append("ここでは二次形式を題材に、解析勾配と数値勾配を突き合わせる。")
    m.append("")
    m.append("shape:")
    m.append("")
    m.append("- `x ∈ R^d`")
    m.append("- `A ∈ R^{d×d}`")
    m.append("- `f(x) ∈ R`")
    m.append("")
    m.append("```math")
    m.append("f(x) = \\frac{1}{2}x^\\top A x,\\qquad")
    m.append("\\nabla_x f(x) = \\frac{1}{2}(A + A^\\top) x")
    m.append("")
    m.append("S = \\frac{1}{\\sqrt{d_k}}QK^\\top,\\quad P=\\mathrm{softmax}(S),\\quad Y=PV")
    m.append("```")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("")
    m.append("def f_quadratic(x: np.ndarray, A: np.ndarray) -> float:")
    m.append("    return float(0.5 * x.T @ A @ x)")
    m.append("")
    m.append("")
    m.append("def grad_x_analytic(x: np.ndarray, A: np.ndarray) -> np.ndarray:")
    m.append("    return 0.5 * (A + A.T) @ x")
    m.append("")
    m.append("")
    m.append("def grad_x_numeric(x: np.ndarray, A: np.ndarray, eps: float = 1e-6) -> np.ndarray:")
    m.append("    g = np.zeros_like(x)")
    m.append("    for i in range(x.shape[0]):")
    m.append("        xp = x.copy(); xm = x.copy()")
    m.append("        xp[i] += eps; xm[i] -= eps")
    m.append("        g[i] = (f_quadratic(xp, A) - f_quadratic(xm, A)) / (2.0 * eps)")
    m.append("    return g")
    m.append("")
    m.append("")
    m.append("rng = np.random.default_rng(1)")
    m.append("d = 8")
    m.append("x = rng.normal(size=(d,))")
    m.append("A = rng.normal(size=(d, d))")
    m.append("")
    m.append("g_a = grad_x_analytic(x, A)")
    m.append("g_n = grad_x_numeric(x, A)")
    m.append("rel = np.linalg.norm(g_a - g_n) / (np.linalg.norm(g_a) + 1e-12)")
    m.append("print('grad check (relative error)=', float(rel))")
    m.append("assert rel < 1e-6")
    m.append("")
    m.append("")
    m.append("# einsum: contract indices explicitly (shape contract)")
    m.append("N, d_k, d_v = 4, 6, 5")
    m.append("Q = rng.normal(size=(N, d_k))")
    m.append("K = rng.normal(size=(N, d_k))")
    m.append("V = rng.normal(size=(N, d_v))")
    m.append("")
    m.append("S = np.einsum('nd,md->nm', Q, K) / np.sqrt(float(d_k))")
    m.append("S = S - S.max(axis=1, keepdims=True)")
    m.append("P = np.exp(S); P = P / P.sum(axis=1, keepdims=True)")
    m.append("Y = np.einsum('nm,mv->nv', P, V)")
    m.append("")
    m.append("assert S.shape == (N, N) and P.shape == (N, N) and Y.shape == (N, d_v)")
    m.append("print('attention shapes:', S.shape, P.shape, Y.shape)")
    m.append("```")
    m.append("")
    m.append("**丁寧ポイント**: 数値微分は遅いが、検算としては最強。解析勾配と一致しないなら、実装は壊れている。")
    m.append("")
    m.append("`einsum` は高速化のための道具でもあるが、この段階ではそれより「添字で縮約を固定する」ことが価値。")
    m.append("")
    m.append(mermaid("二次形式の計算グラフ", [
        "flowchart LR",
        "  x[x] --> Ax[A x]",
        "  A[A] --> Ax",
        "  Ax --> xtAx[x^T(Ax)]",
        "  x --> xtAx",
        "  xtAx --> f[f=1/2 x^T A x]",
    ]))
    m.append("")
    m.append(mermaid("einsumでAttention", [
        "flowchart TD",
        "  Q[Q: N×d_k] --> S[S: N×N]",
        "  K[K: N×d_k] --> S",
        "  S --> P[P: N×N]",
        "  P --> Y[Y: N×d_v]",
        "  V[V: N×d_v] --> Y",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🔬 Z6. 研究ゾーン（20分）— 大規模化で何が変わるか")
    m.append("")
    m.append("SVD は大規模で重い。だから現実には「近似」が主役になる。")
    m.append("")
    m.append("- ランダム射影で部分空間を先に取る（randomized SVD）[^1]")
    m.append("- その上で小さい行列のSVDだけを解く（圧縮→決定論）")
    m.append("")
    m.append(mermaid("randomized SVD の流れ（概念）", [
        "flowchart LR",
        "  A[A] --> O[Ω]",
        "  O --> Y[Y=AΩ]",
        "  Y --> Q[orth(Y)]",
        "  Q --> B[B=Q^T A]",
        "  B --> S[SVD(B)]",
        "  S --> Ak[A_k]",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🎓 Z7. まとめ（10分）")
    m.append("")
    m.append("- truncated SVD の誤差は「捨てた特異値のエネルギー」")
    m.append("- 行列微分は「局所規則」へ還元できるが、実装は shape と検算が支える")
    m.append("- 数値微分は最後の審判。通るまで式と実装を疑う")
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 参考文献")
    m.append("")
    m.append("[^1]: <https://arxiv.org/abs/0909.4061>")
    m.append("[^2]: <https://arxiv.org/abs/1706.03762>")
    m.append("[^3]: <https://arxiv.org/abs/1502.05767>")
    m.append("[^4]: <https://arxiv.org/abs/1404.1100>")
    m.append("[^5]: <https://arxiv.org/abs/2002.01387>")
    m.append("")
    m.append("## 著者リンク")
    m.append("- Blog: https://fumishiki.dev")
    m.append("- X: https://x.com/fumishiki")
    m.append("- LinkedIn: https://www.linkedin.com/in/fumitakamurakami")
    m.append("- GitHub: https://github.com/fumishiki")
    m.append("- Hugging Face: https://huggingface.co/fumishiki")
    m.append("")
    m.append("## ライセンス")
    m.append("- CC BY-NC-SA 4.0")
    return "\n".join(m)


def part07_2() -> str:
    # Part2: 2 python blocks each preceded by math; detailed prose; no drills.
    m: list[str] = []
    m += [
        "---",
        'title: "第7回: 最尤推定と統計的推論 (Part2: 実装編)"',
        'emoji: "📊"',
        'type: "tech"',
        'topics: ["機械学習", "深層学習", "数学", "Python", "統計学"]',
        "published: false",
        'slug: "ml-lecture-07-part2"',
        'difficulty: "intermediate"',
        'time_estimate: "90 minutes"',
        'languages: ["Python"]',
        'keywords: ["最尤推定", "MLE", "Cross-Entropy", "KL", "forward KL", "reverse KL", "FID", "評価指標"]',
        "---",
        "",
        "> **この講義について**",
        "> Part1 の結論（MLE = cross-entropy 最小化 = `D_KL(p||q)` 最小化）を、数値で崩れない形に落とす。",
        ">",
        "> **前編はこちら**: [第7回 Part1（理論編）](/articles/ml-lecture-07-part1)",
        "",
        "## Learning Objectives",
        "",
        "- [ ] MLE の `argmax` を「損失最小化」として実装できる",
        "- [ ] `H(p,q)=H(p)+D_KL(p||q)` を数値で検算できる",
        "- [ ] forward KL / reverse KL の違いを、期待値の取り方として説明できる",
        "- [ ] FID の数式と shape を説明し、数値安定性を守って実装できる",
        "",
        "---",
        "",
        "## 🛠️ Z5. 実装ゾーン（60分）— MLE と KL を動かして確認する",
        "",
        "### 5.1 MLE = Cross-Entropy 最小化（離散の最小例）",
        "",
        "ここで壊れるのはいつも `softmax` と `log(0)`。先に防御する。",
    ]
    m.append("")
    m.append("記号↔変数名:")
    m.append("")
    m.append("- `\\(\\hat p\\)` ↔ `p_hat`")
    m.append("- `\\(q_\\theta\\)` ↔ `softmax(theta)`")
    m.append("- `\\(H(\\hat p,q_\\theta)\\)` ↔ `cross_entropy(p_hat,q)`")
    m.append("")
    m.append("検算（このコードの合否基準）:")
    m.append("")
    m.append("- `KL(p||q) ≥ 0`")
    m.append("- `H(p,q)=H(p)+KL(p||q)`")
    m.append("")
    m.append("```math")
    m.append("\\hat\\theta_{\\mathrm{MLE}}")
    m.append("=\\arg\\max_\\theta \\sum_{i=1}^N \\log q_\\theta(x^{(i)})")
    m.append("=\\arg\\min_\\theta \\Bigl(-\\sum_x \\hat p(x)\\log q_\\theta(x)\\Bigr)")
    m.append("")
    m.append("H(p,q)=-\\sum_x p(x)\\log q(x),\\quad")
    m.append("D_{\\mathrm{KL}}(p\\|q)=\\sum_x p(x)\\log\\frac{p(x)}{q(x)}=H(p,q)-H(p)\\ge 0")
    m.append("```")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("")
    m.append("def softmax(theta: np.ndarray) -> np.ndarray:")
    m.append("    z = theta - float(np.max(theta))")
    m.append("    e = np.exp(z)")
    m.append("    return e / float(np.sum(e))")
    m.append("")
    m.append("")
    m.append("def cross_entropy(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:")
    m.append("    return float(-np.sum(p * np.log(q + eps)))")
    m.append("")
    m.append("")
    m.append("def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:")
    m.append("    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))")
    m.append("")
    m.append("")
    m.append("counts = np.array([50, 30, 20])")
    m.append("p_hat = counts / float(np.sum(counts))")
    m.append("")
    m.append("theta = np.array([0.2, -0.1, 0.0])")
    m.append("q = softmax(theta)")
    m.append("")
    m.append("H_pq = cross_entropy(p_hat, q)")
    m.append("H_p = cross_entropy(p_hat, p_hat)")
    m.append("KL_pq = kl(p_hat, q)")
    m.append("")
    m.append("print('p_hat=', p_hat)")
    m.append("print('q    =', q)")
    m.append("print('H(p,q)=', H_pq)")
    m.append("print('H(p)  =', H_p)")
    m.append("print('KL    =', KL_pq)")
    m.append("")
    m.append("assert KL_pq >= -1e-12")
    m.append("assert abs(H_pq - (H_p + KL_pq)) < 1e-10")
    m.append("```")
    m.append("")
    m.append("この検算が通ると、Part1 の「三位一体」がコード上で固定される。")
    m.append("")
    m.append(mermaid("MLE と KL の関係", [
        "flowchart LR",
        "  A[max loglik] --> B[min -E_p log q]",
        "  B --> C[min cross-entropy H(p,q)]",
        "  C --> D[min KL(p||q) (up to constant H(p))]",
    ]))
    m.append("")
    m.append("### 5.2 forward / reverse KL（mode covering / seeking）")
    m.append("")
    m.append("言葉で覚えると混乱する。違いは期待値の取り方。")
    m.append("")
    m.append("```math")
    m.append("D_{\\mathrm{KL}}(p\\|q)=\\mathbb{E}_p[\\log p - \\log q],\\qquad")
    m.append("D_{\\mathrm{KL}}(q\\|p)=\\mathbb{E}_q[\\log q - \\log p]")
    m.append("```")
    m.append("")
    m.append("- `E_p[-log q]` は「`p` がいる場所で `q` が小さい」ことを強く罰する → 取りこぼしに弱い（mode covering）")
    m.append("- `E_q[-log p]` は「`q` が置いた場所で `p` が小さい」ことを強く罰する → 置き場を絞る（mode seeking）")
    m.append("")
    m.append(mermaid("mode covering / seeking（直感）", [
        "flowchart TD",
        "  F[forward KL: E_p[-log q]] --> C[punish missing mass where p is]",
        "  C --> MC[mode covering]",
        "  R[reverse KL: E_q[-log p]] --> S[punish placing q where p is small]",
        "  S --> MS[mode seeking]",
    ]))
    m.append("")
    m.append("### 5.3 FID を「式どおり」に実装する（数値安定性が本体）")
    m.append("")
    m.append("FID は、特徴空間で実分布と生成分布をガウス近似し、その距離を測る。実装の敵は行列平方根。")
    m.append("")
    m.append("shape:")
    m.append("")
    m.append("- `μ_r, μ_g ∈ R^d`")
    m.append("- `Σ_r, Σ_g ∈ R^{d×d}`")
    m.append("")
    m.append("落とし穴:")
    m.append("")
    m.append("- `Σ` が非対称になる → 対称化")
    m.append("- 小さい負の固有値が出る → 下からクリップ（`max(w,eps)`）")
    m.append("")
    m.append("```math")
    m.append("\\mathrm{FID}(r,g)")
    m.append("= \\|\\mu_r-\\mu_g\\|_2^2")
    m.append("+ \\mathrm{Tr}\\Bigl(\\Sigma_r + \\Sigma_g - 2(\\Sigma_r\\Sigma_g)^{1/2}\\Bigr)")
    m.append("```")
    m.append("```python")
    m.append("import numpy as np")
    m.append("")
    m.append("")
    m.append("def cov(X: np.ndarray) -> np.ndarray:")
    m.append("    Xc = X - X.mean(axis=0, keepdims=True)")
    m.append("    return (Xc.T @ Xc) / float(X.shape[0] - 1)")
    m.append("")
    m.append("")
    m.append("def sqrtm_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:")
    m.append("    A = 0.5 * (A + A.T)")
    m.append("    w, V = np.linalg.eigh(A)")
    m.append("    w = np.maximum(w, eps)")
    m.append("    return (V * np.sqrt(w)[None, :]) @ V.T")
    m.append("")
    m.append("")
    m.append("def fid_gaussian(mu_r, Sigma_r, mu_g, Sigma_g) -> float:")
    m.append("    d = mu_r.shape[0]")
    m.append("    Sigma_r = 0.5 * (Sigma_r + Sigma_r.T) + 1e-6 * np.eye(d)")
    m.append("    Sigma_g = 0.5 * (Sigma_g + Sigma_g.T) + 1e-6 * np.eye(d)")
    m.append("")
    m.append("    diff = mu_r - mu_g")
    m.append("")
    m.append("    Sr12 = sqrtm_psd(Sigma_r)")
    m.append("    middle = Sr12 @ Sigma_g @ Sr12")
    m.append("    middle_sqrt = sqrtm_psd(middle)")
    m.append("")
    m.append("    tr = float(np.trace(Sigma_r + Sigma_g - 2.0 * middle_sqrt))")
    m.append("    return float(diff @ diff + tr)")
    m.append("")
    m.append("")
    m.append("# synthetic features (stand-in for Inception features)")
    m.append("rng = np.random.default_rng(0)")
    m.append("N, d = 800, 16")
    m.append("Xr = rng.normal(loc=0.0, scale=1.0, size=(N, d))")
    m.append("Xg = rng.normal(loc=0.2, scale=1.1, size=(N, d))")
    m.append("")
    m.append("mu_r, mu_g = Xr.mean(axis=0), Xg.mean(axis=0)")
    m.append("Sigma_r, Sigma_g = cov(Xr), cov(Xg)")
    m.append("")
    m.append("fid = fid_gaussian(mu_r, Sigma_r, mu_g, Sigma_g)")
    m.append("fid0 = fid_gaussian(mu_r, Sigma_r, mu_r, Sigma_r)")
    m.append("print('FID=', fid)")
    m.append("print('FID (same)=', fid0)")
    m.append("assert fid >= -1e-6")
    m.append("assert abs(fid0) < 1e-6")
    m.append("```")
    m.append("")
    m.append(mermaid("FID の計算パイプライン", [
        "flowchart LR",
        "  R[real features] --> Mr[μ_r, Σ_r]",
        "  G[gen features] --> Mg[μ_g, Σ_g]",
        "  Mr --> FID[FID]",
        "  Mg --> FID",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🔬 Z6. 分類（20分）— 評価可能性で生成モデルを分ける")
    m.append("")
    m.append("生成モデルの比較は、見た目でやると議論が壊れる。`q_θ(x)` が評価できるかどうかで分けると整理される。")
    m.append("")
    m.append("- 明示的尤度: `q_θ(x)` が計算できる（NLL で評価できる）")
    m.append("- 暗黙モデル: サンプルは出せるが `q_θ(x)` が評価できない（FID などが必要）")
    m.append("")
    m.append(mermaid("評価可能性で分類", [
        "flowchart TD",
        "  A[generative model] --> E[explicit likelihood]",
        "  A --> I[implicit]",
        "  E --> NLL[NLL / bits-per-dim]",
        "  I --> FID[FID / sample metrics]",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 🎓 Z7. まとめ（10分）")
    m.append("")
    m.append("- MLE は `H(p,q)` を最小化し、定数差で `D_KL(p||q)` を最小化する")
    m.append("- forward/reverse KL は期待値の取り方が違うので挙動が違う")
    m.append("- FID は「行列平方根」が本体で、対称化と固有値クリップが安定性を決める")
    m.append("")
    m.append(mermaid("推論と評価の流れ", [
        "flowchart LR",
        "  Data[data] --> Fit[fit θ]",
        "  Fit --> Eval[evaluate]",
        "  Eval --> L1[NLL / KL]",
        "  Eval --> L2[FID]",
    ]))
    m.append("")
    m.append("---")
    m.append("")
    m.append("## 参考文献")
    m.append("")
    m.append("[^1]: <https://arxiv.org/abs/1706.08500>")
    m.append("[^2]: <https://arxiv.org/abs/1406.2661>")
    m.append("[^3]: <https://arxiv.org/abs/1701.07875>")
    m.append("[^4]: <https://arxiv.org/abs/1711.10337>")
    m.append("[^5]: <https://arxiv.org/abs/1601.00670>")
    m.append("")
    m.append("## 著者リンク")
    m.append("- Blog: https://fumishiki.dev")
    m.append("- X: https://x.com/fumishiki")
    m.append("- LinkedIn: https://www.linkedin.com/in/fumitakamurakami")
    m.append("- GitHub: https://github.com/fumishiki")
    m.append("- Hugging Face: https://huggingface.co/fumishiki")
    m.append("")
    m.append("## ライセンス")
    m.append("- CC BY-NC-SA 4.0")
    return "\n".join(m)


def main() -> None:
    w(ART / "ml-lecture-08-part1.md", part08_1())
    w(ART / "ml-lecture-08-part2.md", part08_2())
    w(ART / "ml-lecture-03-part2.md", part03_2())
    w(ART / "ml-lecture-07-part2.md", part07_2())


if __name__ == "__main__":
    main()
