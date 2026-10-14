# Verification of case-study claims (primary-source check)

Method: arXiv PDFs downloaded fresh, text extracted with PyMuPDF; figure values read from rendered pages where no text/table exists.
Versions read: ScaleRL 2510.13786v1 (15 Oct 2025); DAPO 2503.14476v2 (20 May 2025); VAPO 2504.05118 (current); Spurious Rewards 2506.10947v2 (25 Feb 2026; v1 also checked); Reasoning or Memorization 2507.10532v3 (17 Dec 2025); 2512.16912v3 (26 Jan 2026).

Status labels: VERIFIED / WRONG (correct value) / NOT FOUND / FIGURE-ONLY.

---

## ScaleRL: "The Art of Scaling Reinforcement Learning Compute for LLMs" (2510.13786v1)

### 1. Experimental protocol

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Total compute >400k GPU-hours | VERIFIED | Abstract p.1; Intro p.2 | "spanned over 400,000 GPU-hours (on Nvidia GB200 GPUs)" |
| Hardware GB200 | VERIFIED | p.2; App. A.3 p.19 | "We used 80 Nvidia GB200 GPU for a single run" |
| 8B dense model: which one? | NOT FOUND | whole paper | Called only "an 8B dense model" (p.5, p.20). No name/family is given. SFT is in-house: a "curated data mix of reasoning traces", 32B tokens (A.3 p.19). |
| MoE = Llama-4 Scout 17B×16 | VERIFIED | p.3, p.10 | "larger MoE (Llama-4 17B×16)"; "Training the 17B×16 Llama-4 Scout MoE with ScaleRL" |
| Per single ablation ~3.5–4k GPU-h | VERIFIED | §3 p.5; A.3 p.19 | "we first ablate design choices on top of the baseline at 3.5k to 4k GPU-hours" |
| Per leave-one-out run 16k GPU-h | VERIFIED (one inconsistency) | p.3, §3 p.5, §4 p.8, A.3 p.19 | "leave-one-out ablation, consuming 16,000 GPU-hours per run". A.6 (p.21) says "∼1600 GPU hours per run", which looks like a typo. |
| Largest run 100k GPU-h | VERIFIED | Abstract; Fig. 1 | "a single RL run scaled up to 100,000 GPU-hours". The Scout run used 50k GPU-h. |
| GPUs per run | VERIFIED (with scope caveat) | A.3 p.19 | 80 GB200: "we set 64 of those as generators… The remaining 16 GPUs act as trainers". The text does not say whether the 30k–100k runs also used 80 GPUs. |
| Data: Polaris-53K (math) | VERIFIED | §2 p.4; A.3 p.19 | "Polaris-53K dataset… with a batch size of 768 (48 prompts with 16 generations each)". Deepcoder is added for the math+code runs. The 100k run used "batch size of 2048 prompts" (p.10). |
| Other settings | VERIFIED | A.3 p.19 | LR 5×10⁻⁷ constant, AdamW ε=10⁻¹⁵, 100 warmup steps; 12k thinking + 2k solution tokens; ±1 rewards |

### 2. ScaleRL recipe components: VERIFIED (exact list)
§4 p.8: "uses PipelineRL with 8 steps off-policyness, interruption-based length control for truncation, FP32 computation for logits, and optimizes the JScaleRL(θ) loss. This loss combines prompt-level loss aggregation, batch-level advantage normalization, truncated importance-sampling REINFORCE loss (CISPO), zero-variance filtering, and no-positive resampling."
That is 8 components, and they match the claimed list. "FP32 LM head" is the paper's own wording for FP32 at the logits/LM head (Fig. 2 MiniMax description: "FP32 precision fix at the LM head"). No-Positive-Resampling means a prompt with historical pass rate ≥0.9 is permanently removed (p.8).

### 3. Final fit A = 0.610, B = 1.92: VERIFIED as numbers, but they belong to a different run (WRONG if read as the 100k run)
- App. Table 1, p.25 ("Cmid, B, and A values for the large scale runs in Section 5"), row **ScaleRL: Cmid 2542, B 1.92, A 0.610**. Fig. 2 caption (p.3): "achieving an asymptotic reward of A = 0.61".
- That row is the standard batch-768 ScaleRL run. The 100k-GPU-h 8B run used batch 2048, which is row **ScaleRL-bs2048: Cmid 10909, B 1.70, A 0.645**. A.7 (p.21) agrees: "in the 100k GPU-hour run on the 8B dense model, fitting over (1.5k, 50k) gives B = 1.70, A = 0.645". A (0, 100k) fit gives B = 1.56, A = 0.655.
- Other rows: ScaleRL-Scout A = 0.710, B = 1.65; ScaleRL-32k A = 0.645, B = 1.89.

### 4. Predictability from early compute: VERIFIED
- Fig. 1 caption (p.1): "We fit a sigmoid curve… up to 50k (and 16k) GPU hours and extrapolate to 100k (and 45k) on the 8B (Scout MoE) models respectively." Steps: 7400 (8B) and 7100 (Scout).
- §5 p.10: fits are made "early in training for each setting (precisely, half the target budget)". The bs2048 run "yielded a fit that extrapolated from up to 50k GPU hours to the final 100k point."
- LOO runs (§4 p.9): "fit the sigmoidal curve up to 8000 GPU-hours and extrapolate to 16000 GPU-hours".
- Fits skip the first ~1.5k GPU-h (p.4, A.7). No quantitative prediction-error number is given for the 100k point. The agreement is shown only in the figure.

### 5. Instability onsets: VERIFIED
- A.6 p.21: "our baseline method… destabilizes beyond ∼3500 GPU-hours"; "∼5k GPU hours for ϵ = 0.26 in DAPO, ∼6k GPU hours with the FP32 precision fix…, and ∼7k GPU hours for CISPO."
- GRPO ~6k is a separate statement about the DeepSeek(GRPO) comparison recipe in Fig. 2. A.16 p.26: "We saw the training became unstable post 6k GPU Hours due to truncations". The cause given is truncations exceeding 10–15% at batch 768 (A.15 p.25).

### 6. Skipped, as requested (Fig. 8 caption says ±0.02, and App. p.21 says "A varied by at most ±0.015").

---

## DAPO (2503.14476v2)

### 7. Training setup
| Claim | Status | Location | Evidence |
|---|---|---|---|
| Qwen2.5-32B base | VERIFIED | §1 p.2 | "We experiment over Qwen2.5-32B [12] as the pretrained model for RL." |
| Prompt batch 512, 16 responses/prompt | VERIFIED | §4.1 p.8 | "the prompt batch size is 512 and we sample 16 responses for each prompt". Mini-batch 512 gives 16 gradient updates per rollout step. |
| Max gen 16,384 + 4,096 cache | VERIFIED | §4.1 p.8 | "expected maximum length as 16,384 tokens and allocate additional 4,096 tokens as the soft punish cache" (max 20,480) |
| Learning rate | VERIFIED | §4.1 p.8 | "constant learning rate of 1 × 10−6, incorporating a linear warm-up over 20 rollout steps". Other settings: ε_low = 0.2, ε_high = 0.28; eval avg@32, temp 1.0, top-p 0.7. |
| Number of training steps | FIGURE-ONLY | Fig. 1 p.1 | No total step count appears in the text. The Fig. 1 x-axis ("gradient update steps") runs 0–10,000. |
| 50 AIME24 using 50% of R1-Zero-Qwen-32B's steps (47) | VERIFIED | Abstract/§1 p.2; §4.2 p.8; Table 1 p.9 | "outperforming previous state-of-the-art results achieved by DeepSeek-R1-Zero-Qwen-32B [2] (47 points) using 50% training steps" |

Table 1 (p.9, "Main results of progressive techniques applied to DAPO"): Naive GRPO 30, + Overlong Filtering 36, + Clip-Higher 38, + Soft Overlong Punishment 41, + Token-level Loss 42, + Dynamic Sampling (DAPO) 50. R1-Zero-Qwen-32B is 47.

### 8. Repeated runs, seeds, one-at-a-time ablation, ablation order: NOT FOUND
- No repeated training runs, seeds, error bars or variance across runs are reported. "Repeat" appears only for evaluation: "we repeat the evaluation set for 32 times and report avg@32" (p.8).
- The ablation is cumulative only: "progressive techniques" (Table 1 title). There is no leave-one-out or one-at-a-time ablation.
- The paper gives no rationale for the order. The only order-related comment is that token-level loss "brings less performance improvement" but "enhances training stability" (p.9).

---

## VAPO (2504.05118)

### 9. Ablation table: VERIFIED with one correction
Table 1, p.8 ("Abalation results of VAPO", AIME24 avg@32):

| Row | Value |
|---|---|
| Vanilla PPO | 5 |
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| DAPO | 50 |
| w/o Value-Pretraining | 11 |
| w/o Decoupled-GAE | 33 |
| w/o Length-adaptive GAE | 45 |
| w/o Clip-Higher | 46 |
| w/o Token-level Loss | 53 |
| w/o Positive Example LM Loss | 54 |
| w/o Group-Sampling | 55 |
| VAPO | **60** (the table says 60; the text says 60.4) |

- The table entry for full VAPO is 60, not 60.4. The 60.4 appears in the text: "achieves a new SOTA score of 60.4 within just 5,000 steps" (§5.2 p.8).
- This is a one-at-a-time ablation: "removing each of these seven modifications from VAPO individually" (p.8).
- VAPO has a repeated-run statement: "consistently achieves peak scores of 60-61 across three repeated experiments" (p.8).
- Vanilla PPO configuration (re-quote, p.7): "The value network was initialized using a reward model, with the GAE λ set to 0.95 and γ set to 1.0. Sample-level loss was used, and the clip ϵ was set to 0.2." Also: actor LR 1e-6, critic LR 2e-6, 8192 prompts × 1 sample, mini-batch 512.
- VAPO settings (p.8): value warmup of 50 steps; λ_critic = 1.0; λ_policy = 1 − 1/(αl) with α = 0.05; ε_high = 0.28, ε_low = 0.2; positive-example LM loss weight 0.1; 512 prompts × 16 samples.

### 10. Skipped, as requested.

---

## Spurious Rewards (2506.10947; v2 read, v1 compared)

### 11. Which models gain from random rewards
MATH-500 after 300 steps (Fig. 1 p.2, bar labels; values are deltas against "Before RL"):

| Model | Before RL | Random reward Δ | Status |
|---|---|---|---|
| Qwen2.5-Math-7B | 49.4 | +21.4 | VERIFIED (text p.1/p.4: "random rewards… still produce a 21.4% performance boost"; ground truth +29.1) |
| Qwen2.5-7B (non-Math) | 41.6 | +16.4 | VERIFIED (Fig. 1 label) |
| Llama3.1-8B-Instruct | 36.8 | **−11.5** | VERIFIED (degrades) |
| OLMo2-7B | 9.0 | **−8.3** | VERIFIED (degrades) |
| Qwen2.5-Math-1.5B | ~0.36 | gains (MATH ~0.36 → ~0.7) | FIGURE-ONLY (Fig. 2b p.3). Text p.3: gains come "much slower (after 100 steps) and less so on AMC (only 4.9%)". The 36.2% baseline is from Table 2. |
| Qwen2.5-1.5B | ~0.03 | no gain; random stays near 0 | FIGURE-ONLY (Fig. 3a p.5) |
| Llama3.2-3B, Llama3.1-8B (base), Llama3.2-3B-Instruct, OLMo2-7B-SFT | — | flat or slightly down | FIGURE-ONLY (Fig. 3 p.5) |

- Section text (p.4): "Spurious rewards can benefit Qwen2.5 models, but rarely help non-Qwen models"; "each weak or spurious reward fails to help at least one other model and can be flat or even harmful."
- p.4: "smaller models benefit less from spurious (e.g., random) rewards".

### 12. Mechanism: clipping bias. VERIFIED
- §4 p.5: "When clipping… is removed and rewards are sampled independently of the responses, the expected training objective is zero, yielding no meaningful learning signal."
- Eq. (2) gives the bias: the expected gradient is +∇logπ when R_θ < 1−ε_c, 0 inside the clip band, and −∇logπ when R_θ > 1+ε_c. In words: "clipping increases the probability of tokens that are already likely under the policy" (p.5–6).
- Worked example (p.6): with π_old = 0.85 and ε = 0.2, upper clipping would require π > 1.02, which is unreachable, so the token "receives nonnegative gradient bias".
- High-prior behavior is code reasoning, whose frequency rises "from 65% to over 90%" (abstract).
- Ablation evidence: three no-clipping variants fail. Fig. 4 shows the average over multiple seeds.

### 13. Versions: VERIFIED (v2 is the latest; no ICML venue on the PDF)
- v2 (25 Feb 2026) is marked "Preprint. February 26, 2026." It carries no ICML 2026 or other venue line.
- **v1 → v2 change in mechanism framing:**
  - v1 abstract (12 Jun 2025) leaves the mechanism open: "the exact mechanism remains a topic for future work". In v1, clipping bias appears only in the body (Fig. 11 ablation; "we hypothesize that clipping bias similarly…").
  - v2 moves clipping bias into the abstract ("GRPO exhibits a clipping bias arising from the clip term, which can amplify high-prior behaviors") and gives it its own §4 with Eq. (2).
- The headline numbers (+21.4 random, +29.1 ground truth, Qwen2.5-Math-7B) are unchanged between versions.

---

## Reasoning or Memorization? (2507.10532v3)

### 14. RandomCalculation: construction VERIFIED; numeric example FIGURE-ONLY
- Construction (§ "Dataset Construction", p.6): expressions built from integers 0–100 plus their fractions, squares and cubes. They have 1–20 steps using +, −, ×, ÷, giving "20 sub-datasets, each containing 1,000 unique problems". Intro p.2: "guaranteeing that every instance post-dates the public release of Qwen."
- RL setup (p.6): 5-step and 10-step subsets with a 700/300 train/validation split. The reward is continuous, r = 1 − 0.5·min(|a−b|,1) − 0.5·min(|a−b|/(|b|+ε),1), not binary.
- Result text (p.6): "the performance improves steadily throughout training under correct rewards. However, training becomes unstable and inconsistent with random or incorrect rewards. Under inverted rewards, the performance collapses rapidly."
- Numeric example: **FIGURE-ONLY** (Fig. 7 p.6, metric is the continuous reward, approximate reading).
  - Qwen2.5-Math-7B, 5-step: correct ≈ 0.85 vs random trials ≈ 0.25–0.45. Greedy baseline ≈ 0.22, Max@16 ≈ 0.75.
  - 10-step: correct ≈ 0.70 vs random ≈ 0.10–0.15.
  - No table of these numbers exists.

---

## 2512.16912 (counter-paper)

### 15. Title, venue, claims
- **Title:** "Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward" (Chen, Li, Li, Yin, Chen, Lin; Columbia/CUHK-SZ/DAMO/NYU). VERIFIED.
- **Venue:** "Published as a conference paper at ICLR 2026" (page header; v3, 26 Jan 2026). VERIFIED.
- **Main claims (p.2):**
  - Clipping bias under random rewards "provides no meaningful learning signal and cannot directly improve performance".
  - Clipping "reduces entropy and drives the policy toward more deterministic, higher-confidence outputs".
  - "reduced entropy by itself does not guarantee performance gains".
  - The paper proposes a reward-misalignment model (§5).
  - It claims to overturn the view "that improvements under spurious rewards are limited to potentially contaminated Qwen-Math models". The models tested are Qwen-Math, Llama (R1-Distill) and QwQ-32B.
- **Non-Qwen gains:** VERIFIED. Fig. 4 Right (p.10, six independent runs): R1-Distill-Llama-8B 0.66 → 0.74±0.03; Qwen2.5-Math-7B 0.64 → 0.70±0.01; Qwen2.5-Math-1.5B 0.48 → 0.49±0.03 (MATH500 pass@1).
  - Caveat: the paper itself says the distilled Llama's "base and teacher models both exhibit contamination on MATH500" (p.10).
  - It also concedes that "base Llama models reliably degrade during random-reward training".
- **"Explanation via entropy minimization": WRONG as a summary; needs rewording.**
  - The paper says clipping acts as entropy minimization: "Under random rewards, clipping acts as an implicit entropy minimization mechanism" (p.9).
  - But it denies that entropy minimization explains the gains: "entropy minimization alone is insufficient for improvement" (abstract), and the gains "cannot be attributed to clipping bias or to causal effects of policy entropy" (p.2).
  - Its explanation is the reward-misalignment model: stronger models (more correct rollouts n_c > n_i) incur less misalignment damage, so they are more likely to improve. Entropy minimization is called "regularization rather than universally beneficial learning signals" (p.9).

---

## Compact list: WRONG / NOT FOUND / FIGURE-ONLY / needs rewording

1. **ScaleRL #1, 8B dense model identity: NOT FOUND.** The paper never names it ("8B dense model", in-house SFT).
2. **ScaleRL #1, GPUs per run:** 80 GB200 (64 generators, 16 trainers) is stated for the runs in general. It is not stated separately for the 100k run. A.6 also contains a "∼1600 GPU hours" LOO figure that conflicts with the 16k given elsewhere (paper typo).
3. **ScaleRL #3, needs rewording.** A = 0.610 and B = 1.92 (Cmid 2542) belong to the standard batch-768 ScaleRL run (App. Table 1). The 100k-GPU-h 8B run is bs2048: **A = 0.645, B = 1.70** (fit 1.5k–50k).
4. **ScaleRL #4:** the prediction is checked visually only. The paper gives no numeric prediction-error figure for the 100k point.
5. **DAPO #7, number of training steps: FIGURE-ONLY.** The Fig. 1 x-axis runs 0–10k gradient steps, with no stated total.
6. **DAPO #8: NOT FOUND.** There are no repeated runs or seeds, and no one-at-a-time ablation (Table 1 is cumulative, "progressive techniques"). The order is not justified.
7. **VAPO #9, minor correction:** full VAPO is **60** in Table 1. The 60.4 is from the text and Fig. 1.
8. **Spurious #11:** the per-model values for Qwen2.5-Math-1.5B, Qwen2.5-1.5B, the Llama3.2 models and OLMo2-7B-SFT are FIGURE-ONLY. Llama3.1-8B-Instruct (−11.5) and OLMo2-7B (−8.3) degrade under random reward.
9. **Spurious #13:** no ICML/venue version exists. v2 (Feb 2026) is a preprint, and it moved clipping bias from a body hypothesis (v1) into the abstract as the stated mechanism.
10. **RandomCalc #14, numeric example: FIGURE-ONLY.** Fig. 7 reports a continuous reward, not accuracy.
11. **2512.16912 #15, "explanation via entropy minimization": WRONG framing.** The paper says clipping induces entropy minimization but that this does not explain the gains. Its explanation is a reward-misalignment model.
