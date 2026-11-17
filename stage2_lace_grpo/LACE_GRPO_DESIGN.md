# Stage 2 Post-Training Design: LACE-GRPO
**Latent-Anchored, CoT-Consistent, Epistemic-Pressure-Invariant Group Relative Policy Optimization**

> **Core Objective:** Post-train small open-weights LLMs (3B–8B parameters, trainable on a single NVIDIA A100 GPU) to refuse cleanly when they genuinely lack parametric knowledge or encounter false premises, while eliminating *True Answer Suppression* (over-refusal on facts the model demonstrably knows) and *Prompt-Pressure Fragility*.

---

## 1. Motivation: Grounding Stage 2 in Stage 1 Empirical Findings

Our Stage 1 behavioral evaluation across 6 frontier checkpoints on the 100-item Anti-Hallucination Paradox (AHP) suite (`data/benchmark_subset_100.json`: 75 true-premise items across SimpleQA, PopQA, Bamboogle, FalseQA-True; 25 false-premise items across FalseQA-False, MultiHoax) uncovered three concrete mechanisms behind the hallucination–refusal trade-off:

1. **True Answer Suppression vs. Calibrated Abstention (`T1`, `T9`):**
   Models frequently withhold facts they already possess in their weights. For example, `or_glm_5p2` had a 62.7% Latent Knowledge Ceiling but lost **17.3%** of true-premise items to *True Answer Suppression*; `or_minimax_m3` showed **16.0%** True Answer Suppression; and in DeepSeek's later RL checkpoint (`or_deepseek_v4p1_flash` $\rightarrow$ `or_deepseek_v4p1_flash_1017`), conservative refusals doubled (**16.0% $\rightarrow$ 32.0%**, McNemar $p = 4.88 \times 10^{-4}$) and True Answer Suppression worsened (**5.3% $\rightarrow$ 8.0%**).
2. **Alignment Pressure Fragility (`T3`, `P2`, `P3`):**
   Models do not learn an intrinsic, prompt-invariant knowledge boundary. Sweeping the system prompt from `TARGET_PERMISSIVE` to `TARGET_CONSERVATIVE` (`src/prompts.py`) induced massive behavioral swings: refusal rates on true facts jumped from **0.0%** to **45.3%–52.0%** (GLM-5.2, GLM-5.3, MiniMax M3), while permissive prompts caused **76.0%–95.8%** hallucination on false-premise questions. Conversely, reducing over-refusal naively caused `or_minimax_3p1_flash` to collapse into reckless guessing (**62.5%** false-premise hallucination at neutral).
3. **Chain-of-Thought (CoT) Self-Censorship (`T10`):**
   Auditing internal `<think>` / `<thought>` traces revealed cases where a model retrieved the exact gold entity during internal reasoning, then triggered learned caution heuristics (*"the instruction says highly certain... prefer silence"*) and overwrote its valid retrieval with `"I don't know"` or an incorrect guess.

---

## 2. Why Existing Calibration Methods Fall Short (Novelty Gap)

| Approach | Mechanism | Limitation Relative to Stage 1 Findings |
| :--- | :--- | :--- |
| **R-Tuning** *(Static SFT)* | Splits data by base confidence; supervised fine-tuning on `"I don't know"`. | Static off-policy boundary cannot adapt as retrieval sharpens during training and tends to over-generalize surface refusal templates. |
| **TruthRL** *(Ternary GRPO)* | Rewards `+1` (correct), `0` (refusal), `-1` (wrong) under a single neutral prompt. | **The Abstention Trap:** Under a `+1 / 0 / -1` payoff, whenever internal probability $p < 0.5$ (common on long-tail PopQA/SimpleQA), expected reward $2p - 1 < 0 < 0_{\text{refusal}}$. Furthermore, treating all refusals as `0` makes *True Answer Suppression* indistinguishable from *Calibrated Ignorance*. |
| **RLCR / Abstain-R1 / KARL** | Proper scoring rules (Brier score), clarification rewards, or two-stage boundary exploration under one static system prompt. | **Ignores Counterfactual Prompt Pressure & CoT Self-Censorship:** Models trained on a single neutral prompt remain vulnerable to conservative over-refusal and permissive jailbreak guessing (`T3`), and final-answer-only rewards never penalize self-censorship inside `<think>` blocks (`T10`). |

### What Is Novel in LACE-GRPO
LACE-GRPO is the first post-training RL framework to combine:
1. **Permissive Multi-Sample Self-Knowledge Probing ($\hat{\kappa}_\theta(x)$)** to separate *parametric ignorance* from *alignment-induced suppression* via a continuous, model-specific asymmetric refusal reward—eliminating TruthRL's $p < 0.5$ abstention collapse.
2. **CoT-to-Output Epistemic Consistency Penalties ($R_{\text{CoT}}$)** that inspect internal reasoning traces during rollout scoring and directly penalize suppressing a fact already retrieved in `<think>`.
3. **Counterfactual Prompt-Pressure Invariance inside GRPO**, generating paired rollouts across `(NEUTRAL, CONSERVATIVE, PERMISSIVE)` system prompts for the same item and penalizing cross-prompt flip-flopping (`suppression_proxy` and `induced_halluc_proxy` directly optimized end-to-end).

---

## 3. System Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │ Disjoint Training Pool (~4,000 QA items) │
                    │  • Single-Hop True (PopQA / SimpleQA)    │
                    │  • Multi-Hop True (HotpotQA / 2Wiki)     │
                    │  • False-Premise 1-Hop (FalseQA / KUQA)  │
                    │  • False-Premise M-Hop (Bridge-Corrupted)│
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
           ┌───────────────────────────────────────────────────────────┐
           │ Phase 1 (Pre-RL & Mid-RL Refresh): Permissive Probing     │
           │  • Sample M=8 rollouts @ T=0.7 with TARGET_PERMISSIVE     │
           │  • Compute empirical Latent Knowability k_hat(x) in [0,1] │
           └─────────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
           ┌───────────────────────────────────────────────────────────┐
           │ Phase 2: Counterfactual LACE-GRPO Training Loop (1x A100) │
           │  For each query x, sample G=6 rollouts across 3 prompts:  │
           │   [2x NEUTRAL, 2x CONSERVATIVE, 2x PERMISSIVE]            │
           │                                                           │
           │  Fast Hybrid Verifier (<5 ms/rollout):                    │
           │   1. strip_thinking() -> (answer_text, thinking_text)     │
           │   2. Stage-1 regex refusal check (src/grader.py)          │
           │   3. Exact/alias/numeric match + NLI debunk check         │
           │                                                           │
           │  Composite Reward R(x, p, y):                             │
           │   R = R_epistemic(x, y; k_hat) + R_CoT(x, y)              │
           │       + R_pressure_invariance(x, {y_p}) + R_format(y)     │
           └─────────────────────────────┬─────────────────────────────┘
                                         │
                                         ▼
           ┌───────────────────────────────────────────────────────────┐
           │ Phase 3: Held-Out Stage 1 Evaluation (run_experiment.py)  │
           │  • Locked benchmark_subset_100.json + 500-item held-out   │
           │  • OOD generalization check (Bamboogle & MultiHoax)       │
           │  • Full T1-T11 tables & P1-P10 plots w/ local LLM judge   │
           └───────────────────────────────────────────────────────────┘
```

---

## 4. Formal Specification of LACE-GRPO

### 4.1 Phase 1: Latent Knowability Probing ($\hat{\kappa}_\theta(x)$)
Small models (3B–8B) genuinely do not store every long-tail fact in their parameters. Before RL begins—and refreshed once halfway through training at step $S_{\text{refresh}} = \lfloor S_{\text{total}}/2 \rfloor$—we probe the current policy $\pi_\theta$ on every true-premise training question $x \in \mathcal{D}_{\text{true}}$ with $M=8$ high-temperature ($T=0.7$) rollouts using Stage 1's `TARGET_PERMISSIVE` prompt (which disables polite hedging and forces latent retrieval):

$$\hat{\kappa}(x) = \frac{1}{M} \sum_{j=1}^{M} \mathbb{I}\Big[\text{Correct}\big(\text{answer}(y_j), y^*\big)\Big] \in [0, 1]$$

This partitions $\mathcal{D}$ into a continuous spectrum rather than assuming all ground-truth facts are knowable by a 4B/7B model:
* **$\hat{\kappa}(x) = 0$:** Fact is genuinely absent from parametric weights (*Unknown-Answerable*).
* **$\hat{\kappa}(x) > 0$:** Fact is demonstrably encoded in weights (*Known-Answerable*, graded by retrieval reliability).
* **$x \in \mathcal{D}_{\text{false}}$:** Question contains a false or impossible premise; answering any entity is always a hallucination.

---

### 4.2 Phase 2: Composite Epistemic Reward Function

For a rollout $y = (z_{\text{think}}, a_{\text{final}})$ generated under system prompt $p \in \{\text{NEUTRAL}, \text{CONSERVATIVE}, \text{PERMISSIVE}\}$, total reward is:

$$R(x, p, y) = R_{\text{epistemic}}(x, y; \hat{\kappa}) + R_{\text{CoT}}(x, y) + R_{\text{press}}(x, p, y; \{y_{p'}\}) + R_{\text{fmt}}(y)$$

#### 1. Latent-Anchored Asymmetric Epistemic Reward ($R_{\text{epistemic}}$)
Let $\ell = \text{Label}(a_{\text{final}}) \in \{\text{CORRECT}, \text{REFUSAL}, \text{INCORRECT}\}$ for $x \in \mathcal{D}_{\text{true}}$ and $\{\text{CORRECTED}, \text{REFUSED}, \text{HALLUCINATED}\}$ for $x \in \mathcal{D}_{\text{false}}$:

* **True-Premise Items ($x \in \mathcal{D}_{\text{true}}$):**
  $$R_{\text{epistemic}}(x, y; \hat{\kappa}) = \begin{cases}
  +1.0 & \text{if } \ell = \text{CORRECT} \\
  \lambda_{\text{cal}}\big(1 - \hat{\kappa}(x)\big) - \lambda_{\text{sup}}\,\hat{\kappa}(x) & \text{if } \ell = \text{REFUSAL} \\
  -1.0 & \text{if } \ell = \text{INCORRECT}
  \end{cases}$$
  With default hyperparameters $\lambda_{\text{cal}} = 0.5$ and $\lambda_{\text{sup}} = 0.6$:
  * If $\hat{\kappa}(x) = 0.0$ (unknown fact): `REFUSAL` receives **$+0.5$** vs. `INCORRECT` (**$-1.0$**) $\rightarrow$ strongly incentivizes calibrated abstention.
  * If $\hat{\kappa}(x) = 1.0$ (reliably known fact): `REFUSAL` receives **$-0.6$** vs. `CORRECT` (**$+1.0$**) $\rightarrow$ penalizes over-refusal while keeping hallucination (**$-1.0$**) strictly worse than silence.
  * If $\hat{\kappa}(x) = 0.25$ (borderline/noisy recall): `REFUSAL` receives $+0.225$, maintaining caution unless retrieval succeeds.

* **False-Premise Items ($x \in \mathcal{D}_{\text{false}}$):**
  $$R_{\text{epistemic}}(x, y) = \begin{cases}
  +1.0 & \text{if } \ell = \text{CORRECTED} \quad \text{(explicitly identifies false premise)} \\
  +0.7 & \text{if } \ell = \text{REFUSED} \quad \text{(safe abstention)} \\
  -1.2 & \text{if } \ell = \text{HALLUCINATED} \quad \text{(accepts false premise)}
  \end{cases}$$
  Giving $+1.0$ to `CORRECTED` over $+0.7$ to `REFUSED` encourages active fallacy explanation over lazy template refusals, while ensuring safe refusal still receives high positive reward.

#### 2. CoT-to-Output Epistemic Consistency Penalty ($R_{\text{CoT}}$)
Using `strip_thinking(y)` from [`src/thinking.py`](../src/thinking.py) to isolate $z_{\text{think}}$:

$$R_{\text{CoT}}(x, y) = \begin{cases}
-\beta_{\text{censor}} & \text{if } x \in \mathcal{D}_{\text{true}} \land \text{GoldInText}(y^*, z_{\text{think}}) \land \ell \neq \text{CORRECT} \\
-\beta_{\text{censor}} & \text{if } x \in \mathcal{D}_{\text{false}} \land \text{FalsePremiseFlagInText}(z_{\text{think}}) \land \ell = \text{HALLUCINATED} \\
0 & \text{otherwise}
\end{cases}$$

*(Default $\beta_{\text{censor}} = 0.5$)*. Directly extinguishes the Stage 1 Table `T10` failure mode where models verify the correct answer in `<think>` and suppress it at output time.

#### 3. Counterfactual Prompt-Pressure Invariance ($R_{\text{press}}$)
For every training item $x$, we generate a paired counterfactual group of **$G = 6$ rollouts**:
* 2 rollouts conditioned on `TARGET_NEUTRAL` ($G_{\text{neut}}$)
* 2 rollouts conditioned on `TARGET_CONSERVATIVE` ($G_{\text{cons}}$)
* 2 rollouts conditioned on `TARGET_PERMISSIVE` ($G_{\text{perm}}$)

Let $\text{AnyCorrect}_{\text{neut/perm}}(x)$ be $1$ if any rollout in $G_{\text{neut}} \cup G_{\text{perm}}$ got `CORRECT` on $x$, and $\text{AnySafe}_{\text{neut/cons}}(x)$ be $1$ if any rollout in $G_{\text{neut}} \cup G_{\text{cons}}$ achieved `REFUSAL`/`REFUSED`/`CORRECTED` on an item where abstention/debunking is desired ($x \in \mathcal{D}_{\text{false}}$ or $\hat{\kappa}(x) = 0$). Then:

$$R_{\text{press}}(x, p, y) = \begin{cases}
-\gamma_{\text{sup}} & \text{if } p = \text{CONSERVATIVE} \land x \in \mathcal{D}_{\text{true}} \land \ell = \text{REFUSAL} \land \text{AnyCorrect}_{\text{neut/perm}}(x) = 1 \\
-\gamma_{\text{hal}} & \text{if } p = \text{PERMISSIVE} \land (x \in \mathcal{D}_{\text{false}} \lor \hat{\kappa}(x)=0) \land \ell \in \{\text{HALLUCINATED}, \text{INCORRECT}\} \land \text{AnySafe}_{\text{neut/cons}}(x) = 1 \\
0 & \text{otherwise}
\end{cases}$$

*(Default $\gamma_{\text{sup}} = 0.4,\ \gamma_{\text{hal}} = 0.4$)*.
* **Effect on Conservative Prompting:** Prevents the model from abandoning known facts when a user asks it to be careful (`suppression_proxy` $\rightarrow 0$).
* **Effect on Permissive Prompting:** Teaches the model to resist adversarial *"never refuse, just guess"* instructions on false premises or genuinely unknown facts (`induced_halluc_proxy` $\rightarrow 0$).

#### 4. Format & Conciseness Guard ($R_{\text{fmt}}$)
* $+0.1$ if `answer_text` is non-empty, properly closed (`</think>` present if thinking model), and concise ($\le 40$ words outside `<think>`), matching Stage 1 evaluation criteria; $-0.5$ if `EMPTY` or unclosed truncated CoT.

#### 5. Advantage Normalization & Anti-Collapse Guard
Because $R_{\text{press}}$ only penalizes `CONSERVATIVE` and `PERMISSIVE` (never `NEUTRAL`), pure group-level mean centering would artificially inflate `NEUTRAL` advantages whenever other prompts incur $-\gamma$ penalties, while per-prompt standard deviation with $n=2$ collapses to $0$ whenever both rollouts for a prompt agree. We therefore use a **prompt-centered baseline with pooled group standard deviation**:

$$\mu_p(x) = \frac{1}{|G_p|} \sum_{j \in G_p} R(x, p, y_j), \qquad A_i = \frac{R(x, p_i, y_i) - \mu_{p_i}(x)}{\sigma_{\text{group}}(x) + \epsilon}$$

**Zero-Variance Rescue (Anti-Collapse):** If all $6$ on-policy rollouts for a known item ($\hat{\kappa}(x) > 0$) return `REFUSAL`, $\sigma_{\text{group}} = 0$ and gradients vanish. Whenever this occurs, a custom `GRPOTrainer` generation/loss hook substitutes one cached `CORRECT` rollout from Phase 1's permissive probing into the group buffer (scoring its `old_per_token_logps` under $\pi_{\theta_{\text{old}}}$ with an importance-sampling clip / off-policy mask) so the policy always sees a positive learning signal toward retrieving the known fact.

---

## 5. Fast Hybrid In-Loop Verifier (Solving RL Judge Latency)

In Stage 1, grading with a local LLM judge (`google/gemma-4-e4b`) averaged ~11–13 seconds per call (`outputs/report.html`). Over ~24,000+ GRPO rollouts, calling an LLM judge in the training loop would take >75 hours.

LACE-GRPO decouples **training-time verification** (< 5 ms per rollout) from **final evaluation grading** (full Stage 1 judge):

1. **Step 1 — CoT Extraction:** Call `strip_thinking(raw_output)` from [`src/thinking.py`](../src/thinking.py), which returns `(answer_text, thinking_text)`.
2. **Step 2 — Deterministic Refusal Matching:** Reuse the compiled refusal patterns (`_RAW_PATTERNS`) from [`src/grader.py`](../src/grader.py) to immediately classify `REFUSAL` ($x \in \mathcal{D}_{\text{true}}$) or `REFUSED` ($x \in \mathcal{D}_{\text{false}}$).
3. **Step 3 — True-Premise Fact Verification:**
   * Unicode/case/punctuation normalization + article stripping.
   * Numeric tolerance & date range parser (handling SimpleQA-style acceptable ranges).
   * Token-F1 & expanded Wikidata/alias substring matching against `expected_answers`.
   * CoT leakage check (`GoldInText`) using the same normalized alias matcher over `thinking_text`.
4. **Step 4 — False-Premise Debunk Verification (`CORRECTED` vs. `HALLUCINATED`):**
   * Fast premise-rejection regexes (*"false premise"*, *"impossible"*, *"does not exist"*, *"never happened"*, *"fictional"*, *"no such"*).
   * Batched lightweight NLI cross-encoder (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, < 1 GB VRAM on GPU) checking if `answer_text` entails the dataset's `explanation` (why the premise is false) $\rightarrow$ `CORRECTED`, or asserts an entity $\rightarrow$ `HALLUCINATED`.
   * CoT premise-awareness check (`FalsePremiseFlagInText(z_think)`): applies the same premise-rejection regexes (and, when triggered, NLI entailment against `explanation`) to `thinking_text` to detect internal false-premise recognition before output hallucination.

At **Stage 3 (Evaluation)**, the final trained LoRA checkpoint is merged/served and evaluated strictly through `run_experiment.py` with the exact Stage 1 `google/gemma-4-e4b` judge for unbiased comparison.

---

## 6. Dataset Construction (Zero-Leakage & OOD Generalization)

**Inviolable Rule:** [`data/benchmark_subset_100.json`](../data/benchmark_subset_100.json) is strictly excluded from all training, probing, and hyperparameter tuning (verified via normalized question-string deduplication + fuzzy Jaccard similarity < 0.8 filter).

### Training & Validation Split (~4,000 Train / 400 Val)
Mirrors Stage 1's 5-subset distribution while holding out entire datasets (`Bamboogle`, `MultiHoax`) to prove Out-Of-Distribution (OOD) generalization:

| Quadrant | Training Source (Disjoint Splits) | Train Count | Held-Out OOD Eval Analog |
| :--- | :--- | :---: | :--- |
| **Single-Hop True (Long-Tail)** | `PopQA` (stratified across popularity quartiles, disjoint entities) + `TriviaQA` / `SimpleQA` (train slice) | 1,600 | `SimpleQA` (25) + `PopQA` (25) |
| **Multi-Hop True** | `HotpotQA` + `2WikiMultiHopQA` | 800 | `Bamboogle` (20) *(100% OOD dataset)* |
| **Single-Hop False Premise** | `FalseQA` train split (`label=1` false traps + `label=0` true controls) + `KUQA` | 800 | `FalseQA` (15: 10 False / 5 True) |
| **Multi-Hop False Premise** | Programmatic bridge-corrupted `HotpotQA` (swap verified bridge entity between hop 1 & hop 2 to break premise) | 800 | `MultiHoax` (15) *(100% OOD dataset)* |

*Note on `FalseQA label=0` inclusion:* Including single-hop weird-but-true questions (`label=0`) alongside false-premise questions (`label=1`) prevents the model from learning a superficial shortcut where any unusual phrasing triggers automatic refusal.

---

## 7. Single NVIDIA A100 Implementation & Compute Budget

### Target Small Models
1. **Primary Reasoning / Thinking Small Models** *(tests all 4 mechanisms including $R_{\text{CoT}}$)*:
   * `Qwen/Qwen3-4B` *(ideal for **A100-40GB** in `bf16` LoRA)*
   * `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or `Qwen/Qwen3-8B` *(ideal for **A100-80GB** in `bf16` LoRA, or **A100-40GB** in 4-bit QLoRA)*
2. **Non-Reasoning Instruct Control** *(demonstrates generality beyond CoT models, $R_{\text{CoT}}$ inactive)*:
   * `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.1-8B-Instruct`

### Recommended Single-A100 Hyperparameters & Memory Budget

| Parameter | A100-40GB Profile (`Qwen3-4B`) | A100-80GB Profile (`7B` / `8B`) |
| :--- | :--- | :--- |
| **Precision & Adapter** | `bf16` + LoRA (`r=32, alpha=64`, all linear) | `bf16` + LoRA (`r=32, alpha=64`, all linear) |
| **Trainer Engine** | TRL `GRPOTrainer` + colocated `vLLM` sleep mode (`use_liger_loss=True` / chunked CE) | TRL `GRPOTrainer` + colocated `vLLM` sleep mode |
| **vLLM Memory Cap** | `gpu_memory_utilization = 0.45` | `gpu_memory_utilization = 0.45` |
| **DeBERTa NLI Verifier** | ~0.9 GB VRAM (frozen, `fp16` inference) | ~0.9 GB VRAM (frozen, `fp16` inference) |
| **Peak VRAM Usage** | ~27–32 GB *(use `per_device_train_batch_size=1–2` if without Liger loss)* | ~52–64 GB |
| **Group Size ($G$)** | 6 (2 Neutral + 2 Conservative + 2 Permissive) | 6 (2 Neutral + 2 Conservative + 2 Permissive) |
| **Prompt Batch / Grad Accum** | 4 prompts/step $\times$ 4 grad accum (16 prompts = 96 rollouts/opt step) | 8 prompts/step $\times$ 2 grad accum (16 prompts = 96 rollouts/opt step) |
| **Max Completion Length** | 1,024 tokens (enough for CoT + concise answer) | 1,536 tokens |
| **KL Penalty ($\beta_{\text{KL}}$)** | `0.02` | `0.02` |
| **Learning Rate** | `5e-6` (cosine decay, 5% warmup) | `5e-6` (cosine decay, 5% warmup) |
| **Training Steps / Runtime** | ~500 steps (~2 epochs) $\approx$ **3.5–4.5 hours** | ~500 steps (~2 epochs) $\approx$ **4.5–6.0 hours** |

---

## 8. Experimental Protocol, Baselines & Ablations

All baselines and ablations are trained on the identical 4,000-item split with matched compute and evaluated via the untouched Stage 1 pipeline (`run_experiment.py`) on both `benchmark_subset_100.json` and a 500-item held-out multi-dataset test suite:

| Run ID | Method | What It Tests |
| :--- | :--- | :--- |
| **B0** | Base Model (Untrained) | Establishes base latent knowledge ceiling $\hat{\kappa}$, raw suppression, and false-premise hallucination rate. |
| **B1** | R-Tuning (Confidence-Split SFT) | Compares off-policy SFT abstention against online RL. |
| **B2** | TruthRL (Static `+1 / 0 / -1` GRPO) | Tests whether flat ternary rewards trigger the $p < 0.5$ over-refusal collapse on small models. |
| **B3** | RLCR / Brier-Calibrated GRPO | Compares pure confidence-calibration rewards without pressure invariance or CoT penalties. |
| **M1 (Ours)** | **LACE-GRPO (Full System)** | All 4 components active. |
| **A1** | `LACE-GRPO w/o LatentAnchor` | Replaces $\hat{\kappa}(x)$ asymmetric refusal reward with flat `0` $\rightarrow$ isolates over-refusal prevention. |
| **A2** | `LACE-GRPO w/o CoT Penalty` | Sets $\beta_{\text{censor}} = 0$ $\rightarrow$ isolates reduction in Stage 1 Table `T10` CoT self-censorship. |
| **A3** | `LACE-GRPO w/o Pressure Sweep` | Trains on `TARGET_NEUTRAL` only $\rightarrow$ isolates reduction in `T3` `suppression_proxy` & `induced_halluc_proxy`. |

### Quantitative Success Criteria (Evaluated with Stage 1 Metrics)
1. **True Answer Suppression (`true_suppression_rate`, `T9`):** Strictly lower than Base, R-Tuning, and TruthRL (target **< 5%**), paired with higher `robust_recall_rate`.
2. **False-Premise Calibration (`calibration_success_false`, `T1`/`T2`):** Substantial gain over Base on both FalseQA and OOD MultiHoax (target **> 85%**), with a higher share of `CORRECTED` (debunking) vs. blind `REFUSED`.
3. **Pressure Stability (`T3`, `P2`, `P3`):** Minimal gap between Permissive and Conservative sweeps (`suppression_proxy` and `induced_halluc_proxy` compressed toward 0, McNemar refusal shift non-significant on $\hat{\kappa}=1$ items).
4. **CoT Integrity (`T10`):** Near-elimination of cases where the gold entity appears inside `<think>` followed by a `REFUSAL` output.

---

## 9. Planned Repository Layout for Stage 2 Implementation

```
stage2_lace_grpo/
├── LACE_GRPO_DESIGN.md          # This architectural & research specification
├── configs/
│   ├── a100_40gb_qwen3_4b.yaml  # LoRA + GRPO + vLLM config for 40GB A100
│   └── a100_80gb_7b.yaml        # LoRA + GRPO + vLLM config for 80GB A100
├── data_builder/
│   ├── build_train_pool.py      # Curates 4k disjoint items + bridge-corrupted MultiHoax analog
│   └── dedup_guard.py           # Enforces zero overlap with data/benchmark_subset_100.json
├── probing/
│   └── probe_latent_kappa.py    # M=8 permissive probing -> writes per-item k_hat(x)
├── rewards/
│   ├── hybrid_verifier.py       # Reuses src/thinking.py & src/grader.py + DeBERTa NLI
│   └── lace_rewards.py          # Implements R_epistemic, R_CoT, R_press, R_fmt
├── train_lace_grpo.py           # TRL GRPOTrainer w/ counterfactual 3-prompt grouping & anti-collapse
└── eval_bridge.py               # Exports checkpoint to OpenAI-compatible server for run_experiment.py
```
