# Controlled ablation and meta-analysis studies of LLM RL algorithms, plus a 2017–Oct 2026 timeline

Read-depth flags used throughout:
- **[FULL]**: full HTML/PDF text fetched and read this session (sometimes in chunks; truncations noted).
- **[PARTIAL]**: only some sections were read; which ones is noted.
- **[ABS]**: only the arXiv abstract was read (via the arXiv API).
- **[PK]**: prior knowledge of the paper. The arXiv ID/title/date were checked against the arXiv API this session, but the body was NOT re-read. Treat as lower confidence and re-check before quoting.
- **[2nd]**: secondary source only (blog or aggregator).

Evidence grades (from the assignment): **A** = multiple independent controlled ablations; **B** = one controlled ablation or a clear mechanism; **C** = only the originating paper's claim; **D** = contradictory evidence.

---

## Q1. Per-study protocol and conclusions: what each controlled study held fixed, at what scale, with how many seeds, and what it concluded per design choice

### Takeaway
Only one LLM-RL study, ScaleRL, fits compute curves and separates ceiling (A) from efficiency (B). Its ablations ran on a single unnamed 8B dense model plus Llama-4 Scout, using one math dataset (Polaris-53K), and it quantified seed variance only for A (±0.02). Every other LLM "recipe" study is single-family (mostly Qwen), runs fewer than 1,000 steps at 1.5B–8B, reports figures without tables, and does not report seed counts. The two 2020 deep-RL studies (Andrychowicz; Engstrom) are far better powered statistically. Their main lesson carries over: implementation details matter more than the headline algorithm.

### Cited Findings

#### 1. ScaleRL: "The Art of Scaling Reinforcement Learning Compute for LLMs" (Khatri et al., Meta; arXiv 2510.13786, 15 Oct 2025; ICLR 2026) [FULL, read in two chunks]
- **Fit.** R_C = R₀ + (A−R₀)/(1+(C_mid/C)^B). A is the asymptotic pass rate, B is the efficiency exponent, and C_mid is the compute at the halfway point. Fits exclude the first ~1.5K GPU-hours. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- **Held fixed.** 8B dense model. The exact model is not named in the text; the extraction found only "8B dense model". Data was Polaris-53K math, with Deepcoder added for math+code runs. Validation used 1,000 held-out Polaris prompts, scored as mean pass rate over 16 samples every 100 steps. Context was 16,384 tokens (12,288 thinking, 2,048 solution, 2,048 prompt). Each batch was 48 prompts × 16 generations = 768. LR was a constant 5e-7 with AdamW (ε=1e-15, wd 0.01) and 100 warmup steps. Rewards were ±1. Each run used 80 GB200 GPUs (64 generators, 16 trainers). — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- **Budgets.** Design ablations used about 3.5–4K GPU-hours each, with fits on the first ~8K. Leave-one-out (LOO) runs used 16K. Scale runs used 30K–100K. The total study is reported as more than 400K GPU-hours. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786); total from search summary of the [ICLR 2026 version](https://proceedings.iclr.cc/paper_files/paper/2026/file/75ea88aca02640cca54c1231db8cacdd-Paper-Conference.pdf)
- **Seeds and variance.** Three independent ScaleRL runs varied A by "at most ±0.015", so the authors suggest an error margin of 0.02 on A. They call estimating B's error margin "difficult". Most single ablation arms appear to be single runs. — [arXiv 2510.13786 App. A.5](https://arxiv.org/abs/2510.13786)
- **Robustness to fit window (8B, 100K GPU-hour run).**

  | Window (GPU-hours) | B | A |
  |---|---|---|
  | 1.5k–50k | 1.70 | 0.645 |
  | 0–100k | 1.56 | 0.655 |
  | 5k–50k | 1.67 | 0.645 |

  Source: [arXiv 2510.13786 App. A.5](https://arxiv.org/abs/2510.13786)
- **Per-choice findings (ablated one at a time on top of a baseline):**
  - **Off-policy algorithm.** PPO-off-policy-k and PipelineRL-k reached similar A, and PipelineRL mainly improved B. k=8 was best, and k=4 and k=8 "performed equally well". However, App. A.11 says the off-policy choice affects "the asymptote, not just the efficiency exponent". This contradicts the abstract's framing (see Q4). — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **Loss type.** GSPO and CISPO both raised A over DAPO, and CISPO was chosen as marginally better late in training and more robust to the IS-clip parameter. In LOO, the CISPO→DAPO swap gave B 2.01 vs 1.77. DAPO's ε_max "directly alters the scaling behavior" and "governs the asymptotic error itself" (A.17.1). GSPO runs diverged mid-training: restarts worked for the 8B but not for Scout, and the original GSPO clip scale of 1e-4 "did not work well". — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **FP32 precision at the LM head.** A rose from 0.52 to 0.61. This mattered little in the 8B LOO but helped on Scout. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **Loss aggregation (sample / prompt / token average).** Prompt-level averaging gave the highest A. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **Advantage normalization (prompt / batch / none).** All performed similarly. Batch-level was chosen as "marginally better" and "theoretically sound". — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **Zero-variance filtering** (computing the loss only over the effective batch) gave higher A. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **No-Positive-Resampling** permanently drops prompts with pass rate ≥ 0.9, and it improved A. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - **Length control.** Interruptions inserted a stop phrase at a random length between 10k and 12k tokens. Swapping in a DAPO-style length penalty (L_max 14k, cache 2k, applied to correct traces only) "does not improve performance" at 16K GPU-hours. — [arXiv 2510.13786 App. A.10](https://arxiv.org/abs/2510.13786)
  - **Truncation as an instability driver.** At batch 768, truncation rates of 10–15% typically destabilized training. The baseline destabilized beyond ~3,500 GPU-hours once truncations exceeded 10%. ScaleRL kept truncations below 5% for more than 90% of training. — [arXiv 2510.13786 App. A.15](https://arxiv.org/abs/2510.13786)
  - **Entropy** decreased in every run. Batch 768 and batch 2048 had "nearly identical entropy trajectories per step" even though the 2048 run performed much better. The authors conclude entropy is not a reliable predictor of performance. — [arXiv 2510.13786 App. A.12](https://arxiv.org/abs/2510.13786)
- **Scale axes (Table 1: C_mid / B / A).**

  | Run | C_mid | B | A |
  |---|---|---|---|
  | ScaleRL (8B, 14k, bs768, 16 gens) | 2542 | 1.92 | 0.610 |
  | 32k generation length | 11272 | 1.89 | 0.645 |
  | bs512 | 2818 | 1.77 | 0.605 |
  | bs2048 | 10909 | 1.70 | 0.645 |
  | 8 gens | 2542 | 2.44 | 0.585 |
  | 24 gens | 3054 | 2.22 | 0.590 |
  | 32 gens | 2936 | 2.07 | 0.595 |
  | Scout 17B×16 MoE | 4242 | 1.65 | 0.710 |
  | math+code (math) | 2896 | 2.05 | 0.595 |
  | math+code (code) | 1675 | 1.09 | 0.615 |

  The paper says the Scout MoE beats the 8B using about 1/6 of the RL compute. — [arXiv 2510.13786 Table 1, App. A.13](https://arxiv.org/abs/2510.13786)
- **Final recipe.** PipelineRL with k=8 off-policy, interruption-based length control, FP32 logits, CISPO, prompt-level loss averaging, batch-level advantage normalization, zero-variance filtering, and No-Positive-Resampling. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- **LOO result.** Reverting any single component left A within about ±0.02, so the components mainly "affect efficiency parameters B and C_mid" (App. A.9). — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- **Extrapolation test.** An 8B run went to ~100K GPU-hours over ~7,400 steps, with the curve fit up to ~50K and extrapolated beyond. A Scout run went to ~50K GPU-hours. The extrapolation tracked the extended run, with fitted A = 0.645 for the 8B. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- **Comparison recipes reimplemented in their codebase** (App. A.16):
  - DeepSeek-style GRPO (ε 0.2/0.2, sample averaging) "became unstable post 6k GPU Hours due to truncations".
  - DAPO-style dynamic batch filling caused generator stalls in their codebase.
  - The excerpt gives no fitted A/B for these recipes; values appear only in figures.
  - Source: [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- **Stated limitations.** Generalization beyond the in-distribution validation set is not characterized. Multi-task results are preliminary. Generations-per-prompt effects at batch sizes above 2K are untested. Most ablations ran at 8B only. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)

#### 2. "Tricks or Traps? A Deep Dive into RL for LLM Reasoning" (Liu et al., Alibaba; arXiv 2508.08221) [FULL]
- **Held fixed.**
  - Framework: ROLL. Loss: PPO-clip with REINFORCE-style advantages, `init_kl_coef` 0, 1 PPO epoch.
  - Models: Qwen3-4B/8B in both Base and aligned versions.
  - Data: 5,000-prompt subsets at three difficulty levels (SimpleRL-Zoo-Easy; the easiest DeepMath-103k as "medium"; difficulty-weighted DeepMath as "hard"). Binary-answer items were removed and difficulty was labelled by GPT-4o.
  - Hyperparameters: LR 1e-6; 128 prompts × 8 responses; 500 steps; max response 8,192 (the appendix config says 8,000); temperature 0.99.
  - Evaluation: six math benchmarks.
  - Source: [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- **Seeds are not stated** (the config shows `seed: 42`). Curves are smoothed and most results are shown only in figures with no numeric tables. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- **Advantage normalization.**
  - Group-level normalization is robust across settings, and batch-level is more stable under large-scale rewards.
  - Removing std helps when rewards are concentrated (easy data). On hard data, mean-only and std-normalized versions are similar.
  - "Group mean + batch std" was best on 4B and 8B.
  - Source: [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- **Clip-higher.**
  - Base models: minor effect on entropy and sometimes worse accuracy. The clip rate was ~0.003.
  - Aligned models: it slows entropy collapse and gives consistent but small gains (the total aligned-model RL gain was ~2%).
  - Optimal ε_high: 0.32 for 4B-aligned; 0.28 for 8B, where the trend was non-monotonic.
  - Source: [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- **Loss aggregation.** Token-level is better for Base models. Sequence-level is better for aligned models "across most datasets and settings". Evidence comes from Qwen3-8B only. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- **Overlong filtering** helps at an 8k max length, but benefits diminish at 20k, where overlong samples are mostly degenerate repetition. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- **Lite PPO.**
  - Recipe: group-mean/batch-std advantage plus token-level aggregation, with no clip-higher, no dynamic sampling, no overlong filter, and no KL.
  - Claimed to beat GRPO and DAPO on Base Qwen3-4B/8B, which "collapse rapidly after reaching their peak". No numeric averages are given in the text.
  - Not clearly reported for aligned models.
  - Source: [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)

#### 3. "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO; Liu et al., Sea AI; arXiv 2503.20783) [FULL]
- **Two biases identified in GRPO.**
  - Per-response 1/|o_i| normalization favors short correct answers and long wrong ones.
  - Std normalization up-weights near-all-correct or near-all-wrong questions.
  - Source: [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- **Dr. GRPO** drops both terms, using a constant normalizer and advantage = R_i − mean(R). This is RLOO up to a factor of G/(G−1). — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- **Codebase audit.** Table 2 lists trl, OpenRLHF, verl, SimpleRL-Zero, and Open-Reasoner-Zero as length-normalizing the loss. — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- **Protocol.**
  - Model: Qwen2.5-1.5B with the R1 template. Sec. 3.3 uses Qwen2.5-Math-1.5B instead, so the paper is internally inconsistent.
  - Data and hyperparameters: MATH training set; max response 3,000 tokens; 8 samples per question; constant LR 1e-6; KL 0; clip 0.2.
  - Compute: 8×A100, about one day per run.
  - Seeds: three seeds for the variance study only.
  - Source: [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- **Result.** Removing either bias beats vanilla GRPO on reward and accuracy (Fig. 8). GRPO's response length keeps growing after reward plateaus, while Dr. GRPO's stays flat. No numeric token-efficiency metric or significance test appears in the text. — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- **Oat-Zero-7B** (Qwen2.5-Math-7B, MATH levels 3–5, Qwen-Math template) averaged 51.4 at a 3k-token evaluation budget, with AIME24 43.3. — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783). Under Hochlehnert et al.'s re-evaluation with 10 seeds, its AIME24 gain over base is not significant (28.0±3.1 vs 20.7). — [arXiv 2504.07086](https://arxiv.org/abs/2504.07086)
- **Base-model and template confounds.**
  - Qwen2.5-Math-7B scores 0.0 with the R1 template, 26.5 with the Qwen template, and 38.2 with no template. Templates set the starting accuracy, and RL largely recovers what a bad template destroys.
  - Llama-3.2-3B gains far less from RL (3.3→6.8) than math-pretrained variants (NuminaQA: 0.14→20.7).
  - Source: [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)

#### 4. "What Matters in On-Policy RL? A Large-Scale Empirical Study" (Andrychowicz et al., 2020; arXiv 2006.05990) [FULL via ar5iv, first half]
- **Scale.** More than 250,000 agents and more than 50 design choices on five MuJoCo tasks (Hopper, Walker2d, HalfCheetah, Ant, Humanoid). — [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- **Design.** Choices were grouped thematically and sampled at random within each group, while everything else was held at a PPOv2-like base configuration with 256 parallel environments. Each configuration was scored as the median over seeds, with seed score defined as the average over training (so it rewards speed). Results are reported as conditional 95th percentiles with binomial CIs. The seed count is ambiguous in the extraction (3 in one place, "33" elsewhere). — [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- **Recommendations.**
  - Use the PPO loss with clip ~0.25.
  - Initialize the last policy layer about 100× smaller, and use a softplus std with a low initial value.
  - Always normalize observations.
  - Use GAE λ=0.9, and **do not use PPO-style value clipping or Huber loss**.
  - Adam with lr 3e-4.
  - **Entropy and KL regularizers did not significantly help** except on HalfCheetah.
  - The initial action distribution was surprisingly influential.
  - Source: [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)

#### 5. "Implementation Matters in Deep Policy Gradients" (Engstrom et al., 2020; arXiv 2005.12729) [FULL via ar5iv]
- **Optimizations studied.** Nine code-level optimizations: value clipping, reward scaling, orthogonal init and layer scaling, Adam LR annealing, reward clipping, observation normalization, observation clipping, tanh activations, and global gradient clipping. — [arXiv 2005.12729](https://arxiv.org/abs/2005.12729)
- **Protocol.** A full 2⁴ factorial over the first four optimizations, with 5 seeds each on Humanoid and Walker2d plus an LR grid. At least 80 agents per estimate, with bootstrapped 95% CIs. — [arXiv 2005.12729](https://arxiv.org/abs/2005.12729)
- **Algorithm choice vs code-level optimizations (reward).**

  | Task | PPO | PPO-M (minimal) | TRPO | TRPO+ (with PPO's optimizations) |
  |---|---|---|---|---|
  | Walker2d | 3292 | 2735 | 2791 | 3050 |
  | Humanoid | 806 | 674 | 586 | 1030 |

  On all three tasks, the effect of the code-level optimizations (ACLI) is larger than the effect of the algorithm choice (AAI). — [arXiv 2005.12729](https://arxiv.org/abs/2005.12729)
- **Clipping and trust regions.**
  - "All three algorithms fail to maintain a ratio-based trust region." Max ratios violate 1+ε even with clipping.
  - PPO-NoClip "performs uniformly better than PPO-M", and clipping "is not necessary to achieve high performance".
  - Source: [arXiv 2005.12729](https://arxiv.org/abs/2005.12729)

#### 6. "The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization" (Huang et al.; arXiv 2403.17031) [FULL, first ~70%]
- **Setup.** Pythia 1B/2.8B/6.9B; **4 seeds per size**; a single LR of 3e-6 with no sweeps. — [arXiv 2403.17031](https://arxiv.org/abs/2403.17031)
- **PPO settings.** Per-token KL penalty β=0.05 folded into the reward; γ=1, λ=0.95; K=4 PPO epochs; clip 0.2; value clip 0.2; value coefficient 0.1; temperature 0.7; value model initialized from the RM; dropout disabled; advantage whitening. — [arXiv 2403.17031](https://arxiv.org/abs/2403.17031)
- **"EOS trick".** Responses without EOS get a −1 score. — [arXiv 2403.17031](https://arxiv.org/abs/2403.17031)
- **Reward whitening** lowers the preference rate and shortens outputs, but length-controlled win rates are similar. — [arXiv 2403.17031](https://arxiv.org/abs/2403.17031)
- **Over-optimization.** Some 1B runs reached KL ≈ 50–85 with high RM reward but won fewer than 20% of comparisons against references. A KL-to-reference anchor is load-bearing when the reward is a learned RM. — [arXiv 2403.17031](https://arxiv.org/abs/2403.17031)
- **PAD = EOS pitfall.** If PAD is set equal to EOS, the model may never learn to stop. — [arXiv 2403.17031](https://arxiv.org/abs/2403.17031)

#### 7. "A Sober Look at Progress in Language Model Reasoning" (Hochlehnert et al.; arXiv 2504.07086) [FULL, first ~75%]
- **Variance magnitudes.**
  - Across 20 seeds, Pass@1 std ranged from 5 to 15 points. One AIME24 question moves Pass@1 by 3.3 points.
  - Temperature swings were up to ~15 points; top_p swings up to ~8.
  - Hardware: the same model on different clusters differed by up to 8 points (OpenRS-1.5B, AIME24). Even with fixed Docker images and determinism, 2–5 points of noise remained.
  - Framework: lighteval vs evalchemy differed by 4.5 points on S1.1-7B.
  - Stable estimates need K ≥ 30 seeds.
  - Source: [arXiv 2504.07086](https://arxiv.org/abs/2504.07086)
- **Re-evaluation protocol.** 10 seeds for AIME/AMC and 3 for other benchmarks, with paired t-tests and Wilcoxon tests (p<0.01). Under this protocol:
  - Open-RS1/2/3 gains are not significant, although the originals claimed 10–15 points.
  - STILL-3 and L1 show no significant gain.
  - Oat-Zero, SimpleRL-Zoo, and LIMR show non-significant AIME24 gains but significant gains on MATH500/Olympiad.
  - Significant on several benchmarks: DeepScaleR (AIME25 30.3**, MATH500**), FastCuRL, Nemotron-RR (AIME24 48.3**), Skywork-OR1-7B (AIME24 60.7*), and DAPO-32B (43.0**).
  - Source: [arXiv 2504.07086](https://arxiv.org/abs/2504.07086)
- **Conclusions.** SFT on reasoning traces gave the most reliable gains. RL models show larger drops from AIME24 to AIME25, which the authors read as overfitting, and a mild Pass@k collapse. — [arXiv 2504.07086](https://arxiv.org/abs/2504.07086)
- **Limitations.** They used 10 seeds although their own analysis recommends 30. The 4,096-token cap for Qwen-Math models contradicts their own advice to use 32K or more. — [arXiv 2504.07086](https://arxiv.org/abs/2504.07086)

#### 8. "Spurious Rewards: Rethinking Training Signals in RLVR" (Shao et al.; arXiv 2506.10947, v2 Feb 2026) [FULL, first ~55%]
- **Setup.** GRPO with KL=0 and no entropy loss; DeepScaleR data; LR 5e-7; 64 prompts × 16 rollouts; 8×A100 for ~24 hours. The seed count is not visible in the read portion. — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- **MATH-500 gains on Qwen2.5-Math-7B.** Ground truth +29.1, incorrect labels +24.1, random reward (γ=0.5) +21.4. AIME25 gains under spurious rewards are small (−0.4 to +4.5). — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- **Other models.** OLMo2-7B is flat under spurious rewards, and Llama3.x models mostly fail to improve. Each weak or spurious reward "fails to help at least one other model". — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- **Mechanism.**
  - Qwen2.5-Math-7B writes Python in 65% of responses, and responses with code are more accurate (60.9% vs 28.0%).
  - Spurious RL raises code frequency to above 90% within 15 steps.
  - **Random-reward gains vanish when clipping is removed.** The clip term biases updates toward already-likely tokens.
  - Source: [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- **Replication on another family.** OLMo 3 ran random-reward RL-Zero as a contamination control, and it produced no gains on any benchmark. — [arXiv 2512.13961 §6, Fig. 27](https://arxiv.org/abs/2512.13961) [FULL via PDF text]

#### 9. 2025–2026 "recipe" papers with internal ablations
- **DAPO (arXiv 2503.14476) [FULL, ablation section]**
  - Setup: Qwen2.5-32B base; DAPO-Math-17K; 512 prompts × 16 responses; 16 gradient updates per rollout step; ε 0.2/0.28; max length 20,480 with a 4,096-token soft-punishment cache; KL removed.
  - Progressive AIME24 avg@32: naive GRPO 30 → +overlong filtering 36 → +clip-higher 38 → +soft overlong punishment 41 → +token-level loss 42 → +dynamic sampling 50. The DeepSeek-R1-Zero-Qwen-32B reference is 47.
  - Seed count is not stated. Each row appears to be a single run, and gains of 1–2 points are within the 5–15-point AIME seed std reported by [arXiv 2504.07086](https://arxiv.org/abs/2504.07086).
  - Source: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- **Skywork-OR1 (arXiv 2505.22312) [PARTIAL, first ~65%]**
  - Setup: mostly DeepSeek-R1-Distill-Qwen-7B on 32 H800s for 2,700 steps.
  - **Off-policy SGD steps cause premature entropy collapse.** On-policy (N_SGD=1) beat N_SGD=2 or 4 at equal batch size, and a larger batch did not prevent collapse.
  - KL β=1e-3 vs 0: KL "strongly pulls the actor back" and AIME24 fails to improve, so β=0 was used.
  - τ=0.6 collapses entropy within ~100 steps, while τ=1.0 learns faster.
  - Adaptive entropy control used target 0.2.
  - Multi-stage 8K→16K saved ~100 hours over 1,000 steps and reached the same ~60 AIME24.
  - Advantage-masking truncated responses at 8K was "a form of reward hacking" and was not used.
  - Group size (4/8/16) and batch (16/32/64) showed "no significant differences in entropy dynamics".
  - Seeds are not reported.
  - Source: [arXiv 2505.22312](https://arxiv.org/abs/2505.22312)
- **AceReason-Nemotron (arXiv 2505.16400) [FULL]**
  - Setup: R1-Distill-Qwen-7B/14B; GRPO with token-level loss; strictly on-policy (1 update per batch); KL=0; entropy coefficient 0.
  - **22 or 44 updates per batch → entropy collapse around step 100.** A single update per batch "consistently prevented collapse".
  - Hard-prompt data gave AIME24 65.9 vs 63.3 with full data (7B, 24K stage).
  - Starting directly at 24K length gave minimal gains.
  - Math-only RL raised LiveCodeBench v5 by +6.8 (7B) and +5.8 (14B).
  - Source: [arXiv 2505.16400](https://arxiv.org/abs/2505.16400)
- **Magistral (arXiv 2506.10910) [FULL]**
  - Modifications: no KL; loss normalized by the group's total tokens; ε_high 0.26–0.28 (0.3 for Small); zero-advantage groups dropped; soft length penalty. KL removal and length penalty are asserted without ablation.
  - **Advantage normalization (minibatch / group / none): "no significant difference"** in evaluation or length growth (controlled, no numbers).
  - Batch/minibatch (3B): performance degrades beyond 2 minibatches per batch, and runs are less stable at batch ≤ 1024.
  - Binary vs partial code reward: partial reward gave −2% on LiveCodeBench.
  - An entropy bonus was unstable across data mixes (qualitative), and ε_high was more reliable.
  - Magistral Small: SFT+RL 70.7 AIME24 vs RL-only 65.8 vs SFT-only 65.4.
  - Source: [arXiv 2506.10910](https://arxiv.org/abs/2506.10910)
- **MiniMax-M1 / CISPO (arXiv 2506.13585) [PARTIAL, RL sections]**
  - CISPO clips the stop-gradient IS weight rather than the token update, with only an upper bound and no KL.
  - The motivation is that low-probability "fork" tokens (However, Wait) get clipped after the first update when there are 16 off-policy update rounds.
  - On Qwen2.5-32B-base, CISPO "matches DAPO's performance with only 50% of the training steps". This is a single comparison shown in a figure.
  - An FP32 LM head raised the train/infer probability correlation from ~0.9x to ~0.99x.
  - Generations are truncated early when 3,000 consecutive tokens each have p > 0.99.
  - The RL run took 512 H800s for 3 weeks (~$0.53M).
  - Source: [arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
- **POLARIS (HKU NLP blog) [FULL; 2nd-tier, no paper]**
  - Difficulty data shaped as a mirrored J-curve (drop 8/8-solved items; drop >0.9 accuracy per stage); diversity-based initial temperature (1.4 for Qwen3-4B, 0.7 for R1-Distill-7B), raised in stages; no KL; no entropy loss.
  - **"Think shorter then longer" works for R1-Distill-7B but Qwen3-4B "degraded sharply"; train at the maximum length from the start.**
  - Mostly figure-only evidence.
  - Source: [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/)
- **JustRL (arXiv 2512.16649; ICLR 2026) [FULL]**
  - Setup: stock verl GRPO; LR 1e-6; batch 256 × 8; clip [0.8, 1.28]; no KL; no entropy regularization; 16K context; DAPO-Math-17k with no filtering or dynamic sampling; fixed hyperparameters on two 1.5B backbones; 32 A800s for ~15 days.
  - Results: 54.87 average (DeepSeek-1.5B backbone) vs ProRL-V2 53.08, using ~half the token budget; 64.32 (Nemotron-1.5B backbone) vs QuestA 63.81.
  - **Ablations: adding a DAPO overlong penalty plateaued at ~50% AIME24 vs ~55% and collapsed entropy to 0.5–0.6. Adding a robust verifier on top dropped to ~45%.**
  - No seed counts. Baselines are taken from prior reports. The authors admit they cannot isolate which component matters.
  - Source: [arXiv 2512.16649](https://arxiv.org/abs/2512.16649)
- **OLMo 3 / OlmoRL (arXiv 2512.13961) [FULL RL sections via PDF text]**
  - GRPO plus: zero-gradient filtering; "active sampling" (continuous refill to a fixed batch); token-level loss; no KL; clip 0.2/0.272; truncated importance sampling (cap ρ=2.0 for 32B Think and RL-Zero); **no std normalization** (citing Dr. GRPO).
  - Hyperparameters: group size 8; 32–128 prompts per batch; max asynchrony 1–8; 32K response length for Think.
  - **Footnote 34: "We experimented with additional changes (e.g., overlong filtering), but did not find these gave consistent improvements."**
  - **Active sampling ablation (RL-Zero Math):** keeps the batch full of non-zero-advantage samples with "greatly reduced loss variance". Figure only, no accuracy numbers.
  - **Infrastructure ablation:** tokens/s 881 (OLMo 2) → 2,949 with in-flight updates (MFU 0.30% → 1.01%).
  - RL-Zero 3.0→3.1 came from 16K (instead of 12K) length plus *not* masking truncated sequences. The two changes are confounded. Masking had caused batch-size variation that reduced stability.
  - The authors say RL experiments were too expensive to ablate the full space and that algorithm changes were tested on math only.
  - Source: [arXiv 2512.13961](https://arxiv.org/abs/2512.13961)
- **DeepSeek-V3.2 (arXiv 2512.02556) [PARTIAL, §3.1]**
  - GRPO plus: an unbiased KL estimator (K3 reweighted by π_θ/π_old), with KL "relatively weak or even omitted" for math; off-policy sequence masking (mask negative-advantage sequences whose mean log(π_old/π_θ) exceeds δ, value not given); Keep Routing for MoE; Keep Sampling Mask for top-p/top-k.
  - The advantage formula is mean-only, and per-sequence 1/|o_i| averaging is kept.
  - RL compute exceeds 10% of pretraining.
  - **No ablation numbers** are given for any modification.
  - Source: [arXiv 2512.02556](https://arxiv.org/abs/2512.02556)
- **GLM-4.5 (arXiv 2508.06471) [PARTIAL, RL sections]**
  - GRPO without KL; two-stage difficulty curriculum (stage 2 uses pass@8=0 but pass@512>0 problems); single-stage RL at 64K beat multi-stage length increase (which caused an "irreversible performance drop"); token-weighted mean loss "highly beneficial" for code.
  - **All curves come from a "smaller experimental model", figures only.**
  - Source: [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)
- **Qwen3 (arXiv 2505.09388) [PARTIAL]**
  - Reasoning RL used GRPO on 3,995 query–verifier pairs, with large batch, many rollouts, off-policy updates, and entropy kept stable or rising.
  - Qwen3-235B-A22B AIME24 went from 70.1 to 85.1 over 170 steps. No ablations.
  - Source: [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- **Kimi k1.5 (arXiv 2501.12599) [PARTIAL]**
  - KL-regularized objective against the previous iterate, i.e. online mirror descent, with a squared-loss surrogate, mean-reward baseline, and no value network.
  - Length reward λ = 0.5 − (len−min)/(max−min). Curriculum and prioritized sampling ∝ (1−success), plus partial rollouts.
  - Claims of superiority over ReST and of curriculum gains are figure-only.
  - Source: [arXiv 2501.12599](https://arxiv.org/abs/2501.12599)

#### 10. Other 2025–2026 controlled or meta studies found this session
- **Scaling Behaviors of LLM RL Post-Training (arXiv 2509.25300) [ABS].**
  - Full Qwen2.5 series, 0.5B–72B. Larger models are more compute- and data-efficient, and a power law in compute and data predicts test loss.
  - Learning-efficiency gains from scale show "latent saturation".
  - With limited data, repeating good data works: performance tracks optimization steps more than sample uniqueness.
  - Source: [arXiv 2509.25300](https://arxiv.org/abs/2509.25300)
- **How RL Reshapes LLM Reasoning (arXiv 2610.04158, Oct 2026) [PARTIAL].**
  - Compares ScaleRL's log-sigmoid law with a log-linear law on Qwen3-14B-Base RL runs. Both fit through step 200 with R² > 0.99, extrapolate reliably to ~2× compute on math, and are "often nearly indistinguishable" in the fitting range.
  - Math-only RL follows the log-linear law better, while Math+Code saturates and follows the sigmoid better.
  - Qwen and Gemma move in opposite directions on some out-of-domain tasks under identical RL data.
  - Source: [arXiv 2610.04158](https://arxiv.org/abs/2610.04158)
- **Prompt Dominance… GRPO at 1B on GSM8K (arXiv 2610.04928) [ABS].**
  - OLMo-2-1B. "Estimator variants sit within seed noise", with Dr. GRPO ahead on both seeds.
  - In the off-policy regime, "clipping is the whole story": no clipped ratio loses 4–6 points, while GRPO clipping or GSPO recovers it.
  - The prompt is first-order: a zero-shot prompt gives 0.08% base accuracy, so almost no group carries a gradient.
  - Source: [arXiv 2610.04928](https://arxiv.org/abs/2610.04928)
- **Scaling Laws for Collapse in Asynchronous GRPO (arXiv 2607.01083) [ABS].**
  - Llama-3.2-1B/3B plus Qwen3-8B. The largest stable LR scales as roughly S⁻¹ (constant S·η, where S is the sync interval), and collapse time scales as roughly η⁻¹.
  - Source: [arXiv 2607.01083](https://arxiv.org/abs/2607.01083)
- **M2PO (arXiv 2510.01161) [ABS].** Stale data "can be as informative as on-policy data" if extreme IS weights are suppressed by a second-moment constraint. On-policy performance was matched with staleness of 256 updates across 1.7B–32B models. — [arXiv 2510.01161](https://arxiv.org/abs/2510.01161)
- **SFT-then-RL Outperforms Mixed-Policy Methods (arXiv 2604.23747) [ABS].**
  - Many mixed-policy papers used a faulty SFT baseline because of a DeepSpeed CPU-offload optimizer bug and an OpenRLHF loss-aggregation bug.
  - Once fixed, SFT→RL beats published mixed-policy methods by +3.8 (Qwen2.5-Math-7B) and +22.2 (Llama-3.1-8B).
  - Source: [arXiv 2604.23747](https://arxiv.org/abs/2604.23747)
- **Comparative Analysis of PPO, GRPO, DAPO (arXiv 2512.07611) [ABS + 2nd].**
  - Qwen2.5-1.5B-Instruct on Countdown. Larger groups stabilize training, and the KL coefficient's effect is non-monotonic.
  - **DAPO's dynamic sampling did not help; the best results came with DS disabled.**
  - Source: [arXiv 2512.07611](https://arxiv.org/abs/2512.07611)
- **Group-standard-deviation identity (arXiv 2607.00152) [ABS].** For binary rewards, the group std is exactly the update magnitude. GRPO (divide by std), Dr. GRPO (don't divide), and DAPO (drop std=0 groups) are three settings of one dial. — [arXiv 2607.00152](https://arxiv.org/abs/2607.00152)
- **Your Group-Relative Advantage Is Biased (arXiv 2601.08521) [ABS].** The group-relative estimator underestimates advantages on hard prompts and overestimates them on easy ones. — [arXiv 2601.08521](https://arxiv.org/abs/2601.08521)
- **Defeating the Training-Inference Mismatch via FP16 (arXiv 2510.26788) [ABS].** BF16 rounding is the root cause of train/infer mismatch, and uniform FP16 gives "more stable optimization, faster convergence, and stronger performance". — [arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
- **No independent replication of ScaleRL's 100K-GPU-hour run or its fitted parameters was found.** The nearest is 2610.04158, which tests the functional form. — search: [WebSearch summary](https://www.emergentmind.com/papers/2510.13786); [arXiv 2610.04158](https://arxiv.org/abs/2610.04158)

### Inferences
- The LLM-RL ablation literature is statistically weaker than the 2020 deep-RL studies it echoes. Most LLM ablations are single-seed curves on AIME-sized benchmarks, where Hochlehnert et al. measured 5–15-point seed std. So any ablation reporting ≤3-point AIME differences without seeds (most DAPO rows, many Tricks-or-Traps panels) should be graded at most C.
- ScaleRL is the only study that measures what the assignment calls "ceiling vs efficiency". Even there, B has no error bar, and most per-choice A differences (0.585–0.610) are about the size of its own ±0.02 seed margin. Only FP32 logits (0.52→0.61), batch size (0.605→0.645), generation length (0.610→0.645), and model scale (0.61→0.71) clearly exceed it.

### Gaps
- ScaleRL's 8B model identity and the A/B values for individual design ablations (only figures) were not visible in the HTML text. The DeepSeek/Qwen/Magistral/MiniMax comparison A/B values are also missing.
- The Andrychowicz seed count is ambiguous in the extracted text.
- Skywork-OR1 clip-higher and entropy-coefficient sensitivity results were in the unread final third.
- Spurious Rewards' seed count was not visible.

---

## Q2. Where studies disagree: std normalization, token vs sequence aggregation, KL, clipping, overlong handling, dynamic sampling, length curriculum

### Takeaway
Every "trick" whose direction is contested has small effects under these protocols: std normalization, aggregation granularity, overlong filtering, dynamic sampling, and staged length. Where controlled comparisons exist, they mostly find "no significant difference" for the ceiling: ScaleRL, Magistral, and the 1B GRPO study for normalization. The disagreements are about efficiency and length dynamics, and they depend on data difficulty, base vs aligned model, and context budget.

### Cited Findings

**Std normalization of the advantage** — Grade: **A for "does not move the ceiling"**; **D for the direction of any efficiency effect**.

| Study | Setting | Finding |
|---|---|---|
| [Dr. GRPO, 2503.20783](https://arxiv.org/abs/2503.20783) | Qwen2.5(-Math)-1.5B | Removing std (difficulty bias) beats GRPO |
| [Tricks or Traps, 2508.08221](https://arxiv.org/abs/2508.08221) | Qwen3-4B/8B | Removing std helps only on easy data; similar on hard data; best is group-mean + batch-std |
| [ScaleRL, 2510.13786](https://arxiv.org/abs/2510.13786) | 8B | Prompt / batch / none "perform similarly" |
| [Magistral, 2506.10910](https://arxiv.org/abs/2506.10910) | Mistral | Minibatch / group / none: "no significant difference" |
| [2610.04928](https://arxiv.org/abs/2610.04928) | OLMo-2-1B | Estimator variants within seed noise [ABS] |
| [REINFORCE++, 2501.03262](https://arxiv.org/abs/2501.03262) | — | Global-batch normalization; argues prompt-level normalization is biased [ABS] |
| [OLMo 3, 2512.13961](https://arxiv.org/abs/2512.13961); [DeepSeek-V3.2, 2512.02556](https://arxiv.org/abs/2512.02556) | Production | Mean-only advantage, adopted without ablation |

**Loss aggregation (token vs sequence vs prompt)** — Grade: **D**.

| Study | Setting | Finding |
|---|---|---|
| [DAPO, 2503.14476](https://arxiv.org/abs/2503.14476) | Qwen2.5-32B | Token-level +1 AIME (41→42), apparently single run |
| [Dr. GRPO, 2503.20783](https://arxiv.org/abs/2503.20783) | Qwen 1.5B | Constant normalizer stops length growth of wrong answers |
| [Tricks or Traps, 2508.08221](https://arxiv.org/abs/2508.08221) | Qwen3-8B | Token-level better for Base, **sequence-level better for aligned** |
| [ScaleRL, 2510.13786](https://arxiv.org/abs/2510.13786) | 8B | **Prompt-level** best A, but within LOO noise |
| [GLM-4.5, 2508.06471](https://arxiv.org/abs/2508.06471) | Small proxy model, code | Token-weighted better, figure only |
| [MiniMax-M1, 2506.13585](https://arxiv.org/abs/2506.13585) | Production | Switched from token-level to sample-level with token normalization during length extension |
| [2604.23747](https://arxiv.org/abs/2604.23747) | Frameworks | A real OpenRLHF loss-aggregation bug changed SFT baselines [ABS] |

**KL to a reference policy** — Grade: **B** for "drop KL-to-SFT in verifiable math RL"; **B** for "keep KL with a learned RM".

| Study | Setting | Finding |
|---|---|---|
| [Skywork-OR1, 2505.22312](https://arxiv.org/abs/2505.22312) | R1-Distill-7B | β=1e-3 vs 0 controlled: KL stalls AIME24 |
| [DAPO](https://arxiv.org/abs/2503.14476), [Magistral](https://arxiv.org/abs/2506.10910), [GLM-4.5](https://arxiv.org/abs/2508.06471), [OLMo 3](https://arxiv.org/abs/2512.13961), [JustRL](https://arxiv.org/abs/2512.16649), [AceReason](https://arxiv.org/abs/2505.16400), [Polaris](https://hkunlp.github.io/blog/2025/Polaris/) | Various | All drop KL; mostly asserted, not ablated |
| [ProRL, 2505.24864](https://arxiv.org/abs/2505.24864) | Prolonged training | Keeps KL control plus reference resets [ABS] |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | Production | Keeps an unbiased-K3 KL, weak or omitted for math |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Production | KL to the *previous iterate* (mirror descent), not to SFT |
| [N+ details, 2403.17031](https://arxiv.org/abs/2403.17031) | Learned-RM RLHF | Without effective KL, RM over-optimization: KL 50–85 runs won <20% |
| [Andrychowicz 2020](https://arxiv.org/abs/2006.05990) | MuJoCo | KL/entropy regularizers did not help except on HalfCheetah |
| [2512.07611](https://arxiv.org/abs/2512.07611) | Countdown | KL coefficient effect non-monotonic [ABS] |

**Clipping / trust region**

- **Clip-higher** — Grade: **B, model-dependent**.

  | Study | Finding |
  |---|---|
  | [DAPO](https://arxiv.org/abs/2503.14476) | +2 AIME (36→38) |
  | [Tricks or Traps](https://arxiv.org/abs/2508.08221) | Helps aligned Qwen3; minor or harmful on Base |
  | [ScaleRL](https://arxiv.org/abs/2510.13786) | ε_max "governs the asymptotic error itself" |
  | [Magistral](https://arxiv.org/abs/2506.10910) | ε_high tuned at 0.26–0.28 as a more reliable entropy lever than an entropy bonus (qualitative) |

- **Clipping as a confounder.** Random-reward gains on Qwen-Math vanish without clipping, because clipping biases updates toward high-probability tokens. — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- **Clipping does not enforce a ratio trust region.** — [arXiv 2005.12729](https://arxiv.org/abs/2005.12729)
- **Clipping matters off-policy.** It is the whole effect in the off-policy regime at 1B. — [arXiv 2610.04928](https://arxiv.org/abs/2610.04928) [ABS]
- **Clipped-token-update vs clipped-IS-weight families** — Grade: **C/D**.

  | Study | Finding |
  |---|---|
  | [ScaleRL](https://arxiv.org/abs/2510.13786) | CISPO > DAPO; GSPO diverged in their hands |
  | [MiniMax](https://arxiv.org/abs/2506.13585) | CISPO matches DAPO in 50% of steps (originating paper) |
  | [GSPO, 2507.18071](https://arxiv.org/abs/2507.18071) | Claims MoE stability over GRPO, from its own paper [ABS] |
  | [dev.to blog](https://dev.to/g_factor/multi-reward-rl-part-2-benchmarking-grpo-dapo-and-cispo-on-unseen-tasks-3p4c) [2nd] | CISPO had the highest train reward but collapsed on held-out tasks (single author) |

**Overlong / truncation handling** — Grade: **D**.

| Study | Finding |
|---|---|
| [DAPO](https://arxiv.org/abs/2503.14476) | Overlong filter +6, soft punishment +3 at 16K |
| [Tricks or Traps](https://arxiv.org/abs/2508.08221) | Helps at 8K, not at 20K |
| [JustRL](https://arxiv.org/abs/2512.16649) | Overlong penalty *hurts* (≈55→50 AIME24) and collapses entropy |
| [OLMo 3](https://arxiv.org/abs/2512.13961) | Overlong filtering gave no consistent improvement; un-masking truncated sequences helped (confounded with a length change) |
| [Skywork-OR1](https://arxiv.org/abs/2505.22312) | Masking truncated responses at 8K led to reward hacking |
| [ScaleRL](https://arxiv.org/abs/2510.13786) | Length penalty did not beat interruptions; truncation rate above 10% predicts instability |

**Dynamic sampling / zero-variance filtering** — Grade: **D for accuracy**; **B for stability of the effective batch**.

| Study | Finding |
|---|---|
| [DAPO](https://arxiv.org/abs/2503.14476) | +8 AIME (42→50), the largest single row |
| [ScaleRL](https://arxiv.org/abs/2510.13786) | Zero-variance filtering raises A |
| [2512.07611](https://arxiv.org/abs/2512.07611) | DS does not help on Countdown |
| [JustRL](https://arxiv.org/abs/2512.16649) | SOTA at 1.5B without any filtering |
| [OLMo 3](https://arxiv.org/abs/2512.13961) | Active sampling reduces loss variance (figure only) |
| [2607.00152](https://arxiv.org/abs/2607.00152) | Formally, std=0 groups contribute exactly zero gradient, so filtering changes only the effective batch size and normalization [ABS] |

**Length curriculum (short→long vs max from the start)** — Grade: **D, base-model dependent**.

| Study | Finding |
|---|---|
| [Skywork-OR1](https://arxiv.org/abs/2505.22312) | Staged 8K→16K saves compute with the same accuracy |
| [AceReason](https://arxiv.org/abs/2505.16400) | Staging needed; starting at 24K gave minimal gains |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) | Single-stage at 64K is better; staging causes an irreversible drop |
| [Polaris](https://hkunlp.github.io/blog/2025/Polaris/) | Staging works for R1-Distill-7B but breaks Qwen3-4B |
| [ScaleRL](https://arxiv.org/abs/2510.13786) | Longer generation (32K) raises A but lowers efficiency (C_mid 2542→11272) |

**Data reuse / off-policy SGD steps** — Grade: **A for "multiple SGD passes per rollout batch accelerate entropy collapse"**.

| Study | Finding |
|---|---|
| [Skywork-OR1](https://arxiv.org/abs/2505.22312) | Controlled: N_SGD 2–4 collapse |
| [AceReason](https://arxiv.org/abs/2505.16400) | Controlled: 22/44 updates collapse around step 100 |
| [Magistral](https://arxiv.org/abs/2506.10910) | Degrades beyond 2 minibatches per batch |
| [M2PO](https://arxiv.org/abs/2510.01161) [ABS] | Counterpoint: very stale data is usable with second-moment IS control |
| [2607.01083](https://arxiv.org/abs/2607.01083) [ABS] | Stability boundary at constant S·η, i.e. staleness and LR trade off |

### Inferences
- Most "disagreements" come from different regimes, not contradictory measurements:
  - Std removal matters when many groups are near-all-correct (easy data, aligned models) and is neutral on hard data.
  - Overlong penalties help at 8–16K budgets, where truncation is common and informative, and hurt at larger budgets or when entropy is fragile.
  - Staged length helps distilled models that start very long (R1-Distill) and hurts models already trained to think long (Qwen3).
- The robust signals, each replicated across groups, are:
  - (a) fewer off-policy SGD steps per batch protects entropy;
  - (b) dropping KL-to-SFT is safe for verifiable math rewards;
  - (c) numerical train/infer consistency matters (FP32 head, FP16, TIS);
  - (d) truncation rate is an early-warning signal.
- Note that (c) is supported by ScaleRL's A jump, MiniMax's correlation fix, the FP16 abstract, and OLMo/DeepSeek practice, but is still mostly one controlled ablation (ScaleRL) plus mechanism: grade **B**.

### Gaps
- No study measured std normalization, aggregation granularity, and clip-higher on the same non-Qwen base with ≥5 seeds.
- No independent four-way controlled comparison of GRPO vs DAPO vs GSPO vs CISPO was found. Search found only partial comparisons from method-proposing papers and one blog.

---

## Q3. Which conclusions generalize across base models (Qwen vs Llama vs others) and scale?

### Takeaway
Very few. The base model is the dominant variable. Qwen2.5-Math responds even to random rewards because clipping amplifies its code-reasoning prior, while OLMo and Llama do not. Template and prompt choices swing starting accuracy from 0% to 38%. Base vs instruction-tuned flips the preferred loss aggregation and the value of clip-higher. Conclusions with multi-family support are narrow:
- larger and better-pretrained models give higher RL ceilings;
- spurious-reward gains are Qwen-Math-specific;
- off-policy reuse harms entropy across the DeepSeek-distilled Qwen and Mistral families.

### Cited Findings
- Spurious rewards help Qwen2.5-Math but are flat on OLMo2-7B and mostly fail on Llama3. Forcing code reasoning helps Qwen2.5-Math-7B by +15.0, but hurts Qwen2.5-7B by −19.4 and Llama3.1-8B-Instruct by −21.6. — [arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- OLMo 3 Base under random rewards gains on no benchmark, which independently confirms the Qwen-specificity. — [arXiv 2512.13961](https://arxiv.org/abs/2512.13961)
- Template effects: Qwen2.5-Math-7B scores 0.0 with the R1 template vs 38.2 with no template. RL gains on Llama-3.2-3B are small unless it is math-pretrained (NuminaQA variant 0.14→20.7). — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- In a 1B OLMo-2 GSM8K study, the zero-shot prompt gives 0.08% base accuracy and almost no learnable groups, while 3-shot gives 18.3%. — [arXiv 2610.04928](https://arxiv.org/abs/2610.04928) [ABS]
- Base vs aligned Qwen3 reverses the preferred aggregation, and clip-higher helps only aligned models. All experiments are Qwen3, and the authors acknowledge the generality limit. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
- Qwen3 vs Gemma-3 under identical math RL move in opposite directions on some out-of-domain tasks and on AIME25 coverage. — [arXiv 2610.04158](https://arxiv.org/abs/2610.04158)
- Length curricula help R1-Distill-7B but break Qwen3-4B. — [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/)
- Scale:
  - The Llama-4 Scout MoE reaches A = 0.71 vs 0.61 for the 8B. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - Across Qwen2.5 0.5B–72B, larger models are more compute- and data-efficient, with saturation in the efficiency term. — [arXiv 2509.25300](https://arxiv.org/abs/2509.25300) [ABS]
  - Post-RL performance is predictable from pretraining loss in a chess testbed and a 1B math check. — [arXiv 2607.16097](https://arxiv.org/abs/2607.16097) [2nd, search summary]
- With bugs fixed, SFT→RL beats mixed-policy methods by +22.2 on Llama-3.1-8B vs +3.8 on Qwen2.5-Math-7B. Method rankings can flip by base model when baselines are sound. — [arXiv 2604.23747](https://arxiv.org/abs/2604.23747) [ABS]
- "RL does not expand reasoning beyond the base" (pass@k at large k; six RLVR algorithms perform similarly) — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837) [ABS]. This is contested by ProRL, which shows prolonged RL with KL and reference resets beating base at all k on some tasks — [arXiv 2505.24864](https://arxiv.org/abs/2505.24864) [ABS]. 2610.04158 finds both patterns, task by task. — [arXiv 2610.04158](https://arxiv.org/abs/2610.04158)

### Inferences
- A blog verdict should treat any single-family (especially Qwen2.5-Math) algorithmic gain as grade C at best until it is replicated on Llama, OLMo, or Gemma. Random-reward controls (Shao et al.; OLMo 3) are now a cheap standard negative control.
- "Which algorithm" appears second-order to "which base model, which prompt template, which data difficulty". Yue et al. found six RLVR algorithms perform similarly against the base-model pass@k bound, consistent with Engstrom's 2020 finding that implementation details dominate the algorithm label.

### Gaps
- No controlled study ran the same ablation grid on Qwen, Llama, and OLMo at ≥7B with multiple seeds.
- ScaleRL's 8B identity is unknown, which limits cross-family interpretation of its findings.

---

## Q4. Which design choices change only efficiency vs the ceiling?

### Takeaway
On ScaleRL's 8B evidence, and within its ±0.02 margin on A, most algorithmic knobs are efficiency-only. This covers advantage normalization, loss aggregation, curriculum or filtering, off-policy algorithm, and CISPO vs DAPO in the LOO comparison. Choices that clearly moved A are:
- numerical precision of the logits;
- loss family and ε_max, partially;
- batch size;
- generation length;
- model scale.

The A vs B separation itself is fragile. 2610.04158 shows log-sigmoid and log-linear (no-ceiling) fits are nearly indistinguishable within the fitting window. ScaleRL's own text is internally inconsistent about the off-policy algorithm.

### Cited Findings

**ScaleRL abstract (via search summary).** Loss aggregation, normalization, curriculum, and off-policy algorithm "primarily modulate compute efficiency without materially shifting the asymptote". — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786); [search summary of ICLR version](https://proceedings.iclr.cc/paper_files/paper/2026/file/75ea88aca02640cca54c1231db8cacdd-Paper-Conference.pdf)

**ScaleRL by choice:**

| Effect on the curve | Choice | Evidence |
|---|---|---|
| Ceiling A moved | FP32 LM head | 0.52 → 0.61 |
| Ceiling A moved | Batch size | 512 → 2048: 0.605 → 0.645 |
| Ceiling A moved | Generation length | 14K → 32K: 0.610 → 0.645, with worse C_mid |
| Ceiling A moved | Model | 8B → Scout MoE: 0.61 → 0.71 |
| Ceiling A moved | DAPO's ε_max | "Governs the asymptotic error" |
| Ceiling A moved | GSPO / CISPO vs DAPO | "Raise A" (pre-LOO comparison) |
| Ceiling A moved | Zero-variance filtering; No-Positive-Resampling | "Higher asymptote", but within the LOO ±0.02 when reverted from the full recipe |
| Efficiency only (B, C_mid) | Advantage normalization | — |
| Efficiency only (B, C_mid) | Every LOO reversion | A within ±0.02 |
| Efficiency only (B, C_mid) | Generations per prompt (8–32 at fixed batch) | A 0.585–0.595 vs 0.610, B 2.07–2.44; roughly flat |
| Efficiency only (B, C_mid) | PipelineRL vs PPO-off-policy | Main text says B only; **App. A.11 says it affects A** |

All rows: [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)

**Other studies on ceiling vs efficiency:**
- Off-policy SGD reuse (N_SGD > 1) causes premature entropy collapse and lower final accuracy in Skywork and AceReason. Within their horizons, these runs plateaued lower, so this is plausibly a ceiling effect for a fixed training horizon. — [arXiv 2505.22312](https://arxiv.org/abs/2505.22312); [arXiv 2505.16400](https://arxiv.org/abs/2505.16400)
- An entropy–performance law R = −a·e^H + b implies a predictable ceiling at H=0. This frames entropy preservation as ceiling-relevant. — [arXiv 2505.22617](https://arxiv.org/abs/2505.22617) [ABS]. Counterpoint: ScaleRL found entropy trajectories non-predictive across batch sizes. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- Log-sigmoid and log-linear fits are both R² > 0.99 in-window and "often nearly indistinguishable". Which is right depends on the run (math-only looked log-linear, math+code saturated). — [arXiv 2610.04158](https://arxiv.org/abs/2610.04158)
- Data reuse: final performance is governed by the number of optimization steps rather than sample uniqueness. — [arXiv 2509.25300](https://arxiv.org/abs/2509.25300) [ABS]
- Prolonged RL with KL plus reference resets: the ProRL claim is that the boundary itself expands with duration. — [arXiv 2505.24864](https://arxiv.org/abs/2505.24864) [ABS]

### Inferences
- For the blog's "decisive vs not" verdicts, the defensible statement is narrow. Among loss-level algorithm changes, none has been shown by independent controlled evidence to raise the RL ceiling. What moves the ceiling in the only curve-fitting study is model scale, context length, batch size, and numerical precision. ScaleRL also attributes an A effect to loss family and ε_max, which is grade C because it is a single study with no error bar on these comparisons.
- Because A is weakly identified from runs shorter than saturation, "efficiency-only" verdicts are more robust than "raises the ceiling" verdicts. Any A-shift smaller than ~0.02–0.03 should be reported as unresolved.

### Gaps
- No second group has fit ScaleRL-style curves to the same design ablations.
- No error bar for B exists anywhere.
- The A values per individual ablation (beyond Table 1) were not extractable from the text.

---

## Q5. The 2026 consensus recipe in open frameworks and frontier reports, and which parts rest on controlled evidence vs folklore

### Takeaway
By late 2025 and 2026, open and frontier recipes have converged on a common baseline:
- group-sampled REINFORCE/GRPO with a mean-centered advantage;
- no KL-to-SFT for verifiable rewards;
- token- or prompt-level loss normalization;
- asymmetric clipping (0.2/0.27–0.28) or IS-weight clipping (CISPO);
- zero-variance group filtering with batch refill;
- asynchronous rollout with bounded staleness (1–8) and train/infer mismatch correction (TIS cap ~2, FP32 head or FP16, routing replay for MoE);
- ~1.0 sampling temperature;
- some truncation control.

Only a few elements have independent controlled support:
- dropping KL;
- limiting off-policy SGD reuse;
- precision alignment;
- zero-variance batch hygiene, for stability.

Std removal, token-level aggregation, overlong penalties, and staged length are adopted mostly on originating-paper or mechanism arguments.

### Cited Findings

**Concrete 2025–2026 recipes and their stated choices:**

| Recipe | Advantage | Aggregation | Clip / IS | KL | Off-policy handling | Data / sampling | Source |
|---|---|---|---|---|---|---|---|
| OlmoRL (Dec 2025) | Mean-only | Token-level | 0.2/0.272 | None | TIS cap 2.0; async ≤8 | Zero-gradient filter plus active sampling | [arXiv 2512.13961](https://arxiv.org/abs/2512.13961) |
| ScaleRL (Oct 2025) | Batch-std | Prompt-level | CISPO | — | PipelineRL k=8; FP32 head | Zero-variance filter, No-Positive-Resampling, interruptions | [arXiv 2510.13786](https://arxiv.org/abs/2510.13786) |
| DeepSeek-V3.2 (Dec 2025) | Mean-only | Per-sequence 1/\|o\| kept | GRPO clip | Unbiased-K3 KL, weak or none for math | Off-policy sequence masking; Keep Routing; Keep Sampling Mask | — | [arXiv 2512.02556](https://arxiv.org/abs/2512.02556) |
| DeepSeek-V4 (2026) | — | — | — | — | — | GRPO domain specialists, then multi-teacher on-policy distillation replacing the mixed RL stage | [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) [2nd: search summary of the report/model card] |
| JustRL (Dec 2025) | GRPO default | — | clip-higher 1.28 | None | — | No filtering, no length penalty; fixed hyperparameters | [arXiv 2512.16649](https://arxiv.org/abs/2512.16649) |
| Magistral (Jun 2025) | Minibatch norm | Group-token normalization | ε_high 0.26–0.3 | None | Async generators | Zero-advantage filter; soft length penalty | [arXiv 2506.10910](https://arxiv.org/abs/2506.10910) |
| MiniMax-M1 (Jun 2025) | — | Token-level | CISPO | None | FP32 head | Repetition early-stop | [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) |
| GLM-4.5 (Aug 2025) | GRPO | Token-weighted (code) | — | None | slime async for agentic RL | Difficulty curriculum; single-stage 64K | [arXiv 2508.06471](https://arxiv.org/abs/2508.06471) |

- **TIS origin.** Truncated importance sampling originates in a blog post by Yao et al. (2025) on inference/training engine mismatch, which OLMo 3 cites. The blog is cited here via OLMo 3 and was not read. — [arXiv 2512.13961](https://arxiv.org/abs/2512.13961)

**Evidence status per consensus element:**

| Element | Grade | Evidence | Caveat |
|---|---|---|---|
| Drop KL-to-SFT for verifiable math | B | Skywork controlled ablation; universal practice | Not valid for learned RMs (N+ details, [2403.17031](https://arxiv.org/abs/2403.17031)); DeepSeek-V3.2 still uses KL in non-math domains ([2512.02556](https://arxiv.org/abs/2512.02556)) |
| Few off-policy SGD steps per rollout batch | A | Skywork, AceReason, Magistral | Async staleness k≤8 is fine in ScaleRL ([2510.13786](https://arxiv.org/abs/2510.13786)); M2PO claims much more is fine with IS control ([2510.01161](https://arxiv.org/abs/2510.01161)) |
| Precision alignment (FP32 head / FP16 / TIS / routing replay) | B | ScaleRL ablation; MiniMax correlation measurement; FP16 paper [ABS]; DeepSeek "crucial" claim (no numbers) | |
| Zero-variance filtering / active sampling | B for stability, D for accuracy | DAPO +8; ScaleRL higher A; OLMo loss variance | 2512.07611 and JustRL show it is unnecessary |
| Mean-only advantage (no std) | Ceiling-neutral A; direction D | See Q2 | Practice follows Dr. GRPO's argument rather than decisive data |
| Token-level aggregation | D | Reversed for aligned models in Tricks or Traps; ScaleRL prefers prompt-level | |
| Clip-higher (~0.27–0.28) | B, model-dependent | | Spurious-reward and clip-bias results show clipping also injects a prior-amplifying bias ([2506.10947](https://arxiv.org/abs/2506.10947)) |
| CISPO / GSPO over token-clipped PPO | C | Originating papers plus ScaleRL alone | GSPO diverged in ScaleRL's hands |
| Overlong penalty / filtering | D | — | JustRL harms; OLMo no consistent gain; ScaleRL no gain vs interruption |
| Staged context length | D | — | Base-model dependent |
| Entropy bonus | D/C | — | Unstable (Magistral); Skywork uses adaptive target control; Polaris and JustRL use none |
| Temperature ~1.0, or calibrated to the base model | B | Skywork τ ablation (0.6 collapses) | Polaris uses model-specific temperatures |
| Evaluation with ≥10–30 seeds | B | Hochlehnert et al. | Most recipe papers still do not comply ([2504.07086](https://arxiv.org/abs/2504.07086)) |

### Inferences
- The 2026 "recipe" is best described as a stability and engineering package, not an algorithm. Its strongest evidence concerns not breaking training: entropy collapse from data reuse, truncation blow-ups, numerical mismatch. There is little evidence of raising the ceiling.
- That mirrors Engstrom et al.'s 2020 conclusion, where code-level choices mattered more than PPO vs TRPO.
- The JustRL result (plain GRPO plus clip-higher, no filtering or penalties, competitive at 1.5B) and Yue et al.'s "six algorithms perform similarly" both suggest the marginal value of most 2025 tricks is small once the basics are right.

### Gaps
- Exact verl, OpenRLHF, and slime default configs (2026 versions) were not inspected. The local `_external_repos/` could be checked by a code-focused researcher.
- The TIS blog, Seed1.5-Thinking/VAPO, GSPO, and Kimi K2 bodies were not read.
- Kimi K2.5, GLM-5, and Qwen3.5 2026 reports were not found by search. MiMo-V2.6 (Sep 2026) was seen only via a third-party blog: [orcarouter blog](https://www.orcarouter.ai/blog/mimo-v2-6-technical-report-rl-explained) [2nd].

---

## Q6. Dated timeline of LLM RL algorithms, 2017 → Oct 2026 (one line "what changed")

### Takeaway
The field moved through four phases:
1. PPO-based RLHF with a learned RM, KL to SFT, and a critic (2017–2022).
2. A turn toward offline and "RL-free" methods and back to REINFORCE-style critic-free methods (2023–early 2024).
3. Critic-free group-baseline RLVR for long-CoT reasoning, followed by rapid trick accretion such as DAPO (2024–mid 2025).
4. A shift of attention to off-policy and numerics correctness, sequence-level or IS-weight clipping, compute-scaling laws, and on-policy distillation (late 2025–2026).

### Cited Findings
All dates are the arXiv first-submission dates, verified via the arXiv API this session unless marked otherwise.

| Date | Method / report | What changed | Source |
|---|---|---|---|
| 2017-06-12 | Deep RL from Human Preferences (Christiano et al.) | Learned reward model from pairwise human preferences drives RL | [1706.03741](https://arxiv.org/abs/1706.03741) [PK] |
| 2017-07-20 | PPO (Schulman et al.) | Clipped-ratio surrogate as a cheap first-order trust region; becomes the RLHF default | [1707.06347](https://arxiv.org/abs/1707.06347) [PK] |
| 2019-09-18 | Fine-Tuning LMs from Human Preferences (Ziegler et al.) | PPO plus RM applied to LMs, with a KL-to-pretrained penalty in the reward | [1909.08593](https://arxiv.org/abs/1909.08593) [PK] |
| 2020-05-25 | Implementation Matters (Engstrom et al.) | Shows code-level details, not clipping, drive PPO's edge | [2005.12729](https://arxiv.org/abs/2005.12729) [FULL] |
| 2020-06-10 | What Matters in On-Policy RL (Andrychowicz et al.) | 250K-agent ablation; PPO loss, init, normalization recommendations | [2006.05990](https://arxiv.org/abs/2006.05990) [FULL] |
| 2020-09-02 | Learning to Summarize from HF (Stiennon et al.) | Scaled RLHF (PPO, per-token KL, RM) beats SFT on summarization | [2009.01325](https://arxiv.org/abs/2009.01325) [PK] |
| 2022-03-04 | InstructGPT (Ouyang et al.) | SFT → RM → PPO with KL plus pretraining mix (PPO-ptx) for instruction following | [2203.02155](https://arxiv.org/abs/2203.02155) [PK] |
| 2022-04-12 | Anthropic HH-RLHF (Bai et al.) | Online iterated RLHF; KL–reward relationship | [2204.05862](https://arxiv.org/abs/2204.05862) [PK] |
| 2022-12-15 | Constitutional AI | AI-feedback reward (RLAIF) replaces human labels for harmlessness | [2212.08073](https://arxiv.org/abs/2212.08073) [PK] |
| 2023-04-13 | RAFT | Reward-ranked fine-tuning: filter best-of-K and SFT, no policy gradient | [2304.06767](https://arxiv.org/abs/2304.06767) [PK] |
| 2023-05-29 | DPO | Closed-form KL-regularized optimum turns RLHF into an offline classification loss | [2305.18290](https://arxiv.org/abs/2305.18290) [PK] |
| 2023-05-31 | Let's Verify Step by Step | Process reward models (step-level supervision) beat outcome RMs for selection | [2305.20050](https://arxiv.org/abs/2305.20050) [PK] |
| 2023-08-17 | ReST | Grow/improve offline self-training loop with reward filtering | [2308.08998](https://arxiv.org/abs/2308.08998) [PK] |
| 2023-10-16 | ReMax | REINFORCE with the greedy-decode reward as baseline; drops the critic | [2310.10505](https://arxiv.org/abs/2310.10505) [PK] |
| 2023-12-11 | ReST-EM | EM-style self-training on binary-verified answers (math/code) | [2312.06585](https://arxiv.org/abs/2312.06585) [PK] |
| 2023-12-14 | Math-Shepherd | Automatic (MC-rollout) process labels used for PPO with a PRM | [2312.08935](https://arxiv.org/abs/2312.08935) [PK] |
| 2024-02-05 | GRPO / DeepSeekMath | Critic replaced by a group mean/std baseline over G samples; KL moved into the loss | [2402.03300](https://arxiv.org/abs/2402.03300) [PK] |
| 2024-02-22 | RLOO "Back to Basics" (Ahmadian et al.) | REINFORCE with a leave-one-out baseline beats PPO, DPO, and RAFT in RLHF; PPO machinery "unnecessary" | [2402.14740](https://arxiv.org/abs/2402.14740) [ABS] |
| 2024-03-24 | N+ Implementation Details of RLHF with PPO | Reproducible PPO-RLHF details (EOS trick, RM init, whitening) | [2403.17031](https://arxiv.org/abs/2403.17031) [FULL] |
| 2024-10-02 | VinePPO | Monte Carlo rollouts from intermediate states replace the value net for credit assignment | [2410.01679](https://arxiv.org/abs/2410.01679) [PK] |
| 2024-11-22 | Tülu 3 (RLVR) | Names RLVR: PPO with binary verifiable rewards on math, IF, and code | [2411.15124](https://arxiv.org/abs/2411.15124) [PK] |
| 2025-01-04 | REINFORCE++ | Critic-free with **global-batch** advantage normalization | [2501.03262](https://arxiv.org/abs/2501.03262) [ABS] |
| 2025-01-22 | DeepSeek-R1 / R1-Zero | GRPO plus rule-based outcome rewards from a base model produce long CoT; RL at frontier scale | [2501.12948](https://arxiv.org/abs/2501.12948) [PK] |
| 2025-01-22 | Kimi k1.5 | Mirror-descent objective (KL to previous iterate), length reward, partial rollouts, no value net | [2501.12599](https://arxiv.org/abs/2501.12599) [PARTIAL] |
| 2025-02-03 | PRIME | Implicit process rewards from an online-updated outcome RM | [2502.01456](https://arxiv.org/abs/2502.01456) [PK] |
| 2025-03-18 | DAPO | Clip-higher, dynamic sampling, token-level loss, overlong shaping, no KL | [2503.14476](https://arxiv.org/abs/2503.14476) [FULL, ablation] |
| 2025-03-26 | Dr. GRPO | Removes 1/\|o\| and std normalization (GRPO biases) | [2503.20783](https://arxiv.org/abs/2503.20783) [FULL] |
| 2025-03-31 | Open-Reasoner-Zero | Vanilla PPO (GAE λ=1, γ=1), no KL, scaled from base | [2503.24290](https://arxiv.org/abs/2503.24290) [PK] |
| 2025-04-07 | VAPO | Returns to a value-based PPO for long CoT (value pretraining, decoupled/length-adaptive GAE) | [2504.05118](https://arxiv.org/abs/2504.05118) [ABS] |
| 2025-04-10 | Seed1.5-Thinking | VAPO/DAPO-based production reasoning RL | [2504.13914](https://arxiv.org/abs/2504.13914) [PK] |
| 2025-04-18 | Does RL Really Incentivize Reasoning (Yue et al.) | Pass@k analysis: RLVR bounded by base; six algorithms similar | [2504.13837](https://arxiv.org/abs/2504.13837) [ABS] |
| 2025-04-09 | Sober Look (Hochlehnert et al.) | Evaluation-variance audit; many RL gains not significant | [2504.07086](https://arxiv.org/abs/2504.07086) [FULL] |
| 2025-05-14 | Qwen3 | GRPO reasoning RL on 3,995 pairs plus general RL with mixed rewards | [2505.09388](https://arxiv.org/abs/2505.09388) [PARTIAL] |
| 2025-05-22 | AceReason-Nemotron | Strict on-policy GRPO; math-then-code RL; staged length | [2505.16400](https://arxiv.org/abs/2505.16400) [FULL] |
| 2025-05-28 | Skywork-OR1 | MAGIC: adaptive entropy control, on-policy emphasis, no KL | [2505.22312](https://arxiv.org/abs/2505.22312) [PARTIAL] |
| 2025-05-28 | Entropy Mechanism (Clip-Cov / KL-Cov) | Entropy change ∝ cov(logp, A); restrict high-covariance tokens | [2505.22617](https://arxiv.org/abs/2505.22617) [ABS] |
| 2025-05-30 | ProRL | Prolonged RL with KL plus periodic reference resets; claims expanded pass@k | [2505.24864](https://arxiv.org/abs/2505.24864) [ABS] |
| 2025-05-30 | AReaL | Fully asynchronous RL system with a decoupled (behavior vs proximal) PPO objective for stale data | [2505.24298](https://arxiv.org/abs/2505.24298) [PK] |
| 2025-06-02 | Beyond the 80/20 Rule | Update only high-entropy "forking" tokens (~20%) | [2506.01939](https://arxiv.org/abs/2506.01939) [PK] |
| ~2025-06 | POLARIS (blog) | Difficulty-shaped data, staged temperature, YaRN at inference | [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/) [FULL]; date not verified |
| 2025-06-12 | Spurious Rewards | Random rewards help Qwen-Math via clipping bias; confounder warning | [2506.10947](https://arxiv.org/abs/2506.10947) [FULL] |
| 2025-06-12 | Magistral | Async GRPO variant, no KL, group-token normalization, ε_high ~0.28 | [2506.10910](https://arxiv.org/abs/2506.10910) [FULL] |
| 2025-06-16 | MiniMax-M1 (CISPO) | Clips the stop-gradient IS weight, not the update; FP32 LM head | [2506.13585](https://arxiv.org/abs/2506.13585) [PARTIAL] |
| 2025-07-24 | GSPO (Qwen) | Sequence-level (length-normalized) ratio and clipping; MoE stability | [2507.18071](https://arxiv.org/abs/2507.18071) [ABS] |
| 2025-07-28 | GMPO | Geometric mean of token-level ratio-weighted objectives to damp outliers | [2507.20673](https://arxiv.org/abs/2507.20673) [ABS] |
| 2025-07-28 | Kimi K2 | Production RL (Muon optimizer lineage) with budget control | [2507.20534](https://arxiv.org/abs/2507.20534) [PK, body not read] |
| 2025-08-08 | GLM-4.5 | GRPO without KL, difficulty curriculum, single-stage 64K, slime async | [2508.06471](https://arxiv.org/abs/2508.06471) [PARTIAL] |
| ~2025-08 | Truncated Importance Sampling (Yao et al. blog) | Corrects inference/training engine log-prob mismatch with a capped ratio | Cited via [2512.13961](https://arxiv.org/abs/2512.13961); blog not read |
| 2025-08-11 | Tricks or Traps (Lite PPO) | Group-mean/batch-std normalization plus token-level loss; minimal recipe | [2508.08221](https://arxiv.org/abs/2508.08221) [FULL] |
| 2025-09-23 | PipelineRL | In-flight weight updates mid-generation; ~2× faster at near on-policy | [2509.19128](https://arxiv.org/abs/2509.19128) [ABS] |
| 2025-09-29 | Scaling Behaviors of LLM RL Post-Training | Power-law compute/data scaling across Qwen2.5 0.5B–72B | [2509.25300](https://arxiv.org/abs/2509.25300) [ABS] |
| 2025-10-01 | M2PO | Second-moment IS constraint permits very stale data | [2510.01161](https://arxiv.org/abs/2510.01161) [ABS] |
| 2025-10-15 | ScaleRL | Sigmoid compute laws; A vs B separation; 100K GPU-hour predictive run | [2510.13786](https://arxiv.org/abs/2510.13786) [FULL] |
| 2025-10-30 | FP16 for RL | Switch BF16 → FP16 to remove train/infer mismatch | [2510.26788](https://arxiv.org/abs/2510.26788) [ABS] |
| 2025-11-25 | SAPO (Qwen) | Soft temperature-gated trust region replacing hard clipping; token-adaptive yet sequence-coherent | [2511.20347](https://arxiv.org/abs/2511.20347) [ABS] |
| 2025-12-02 | DeepSeek-V3.2 | Unbiased-K3 KL, off-policy sequence masking, Keep Routing, Keep Sampling Mask; RL above 10% of pretraining compute | [2512.02556](https://arxiv.org/abs/2512.02556) [PARTIAL] |
| 2025-12-08 | PPO vs GRPO vs DAPO comparison | Controlled small-model comparison; dynamic sampling not useful | [2512.07611](https://arxiv.org/abs/2512.07611) [ABS] |
| 2025-12-15 | OLMo 3 (OlmoRL) | Fully open recipe: mean-only advantage, token loss, TIS, active sampling, in-flight updates | [2512.13961](https://arxiv.org/abs/2512.13961) [FULL RL sections] |
| 2025-12-18 | JustRL | Fixed-hyperparameter plain GRPO matches complex recipes at 1.5B | [2512.16649](https://arxiv.org/abs/2512.16649) [FULL] |
| 2026-01-13 | Group-Relative Advantage Is Biased (HA-DW) | Difficulty-dependent bias in group baselines; adaptive reweighting | [2601.08521](https://arxiv.org/abs/2601.08521) [ABS] |
| 2026-02-05 | Rewards as Labels | Recasts RLVR as classification | [2602.05630](https://arxiv.org/abs/2602.05630) [title only] |
| 2026-02-06 | F-GRPO | GRPO variant protecting rare correct modes | [2602.06717](https://arxiv.org/abs/2602.06717) [title only] |
| 2026-04-26 | SFT-then-RL Outperforms Mixed-Policy | Framework bugs invalidated mixed-policy baselines | [2604.23747](https://arxiv.org/abs/2604.23747) [ABS] |
| 2026-04-26 | DeepSeek-V4 | GRPO specialists merged by multi-teacher on-policy distillation (replaces mixed RL stage) | [2606.19348](https://arxiv.org/abs/2606.19348) [2nd: search summary]; the arXiv API reports a first-submission date of 2026-04-26, which is unusual for a 2606 ID, so re-check it |
| 2026-06-30 | Group-Std Identity | GRPO, Dr. GRPO, and DAPO as three settings of one std "dial" | [2607.00152](https://arxiv.org/abs/2607.00152) [ABS] |
| 2026-07-01 | Scaling Laws for Collapse in Async GRPO | Stability boundary at constant staleness × LR | [2607.01083](https://arxiv.org/abs/2607.01083) [ABS] |
| 2026-07-17 | Understanding Reasoning from Pretraining to Post-Training | Post-RL performance predicted by pretraining loss (chess testbed) | [2607.16097](https://arxiv.org/abs/2607.16097) [2nd] |
| 2026-09 | MiMo-V2.6 (Xiaomi) | Single mixed RL run "towards self-improvement" | [orcarouter blog](https://www.orcarouter.ai/blog/mimo-v2-6-technical-report-rl-explained) [2nd only] |
| 2026-10-03 | How RL Reshapes LLM Reasoning | Log-sigmoid vs log-linear RL compute laws; Qwen vs Gemma divergence | [2610.04158](https://arxiv.org/abs/2610.04158) [PARTIAL] |
| 2026-10-04 | GRPO at 1B ablations | Estimators within seed noise; clipping decisive off-policy | [2610.04928](https://arxiv.org/abs/2610.04928) [ABS] |

### Inferences
- From 2023 to 2025, the advantage estimator was progressively simplified: critic → greedy baseline → leave-one-out → group mean/std → mean-only. At the same time, complexity moved into the trust-region and IS layer: token clip → asymmetric clip → IS-weight clip → sequence-level ratio → soft gates → TIS/masking. This is consistent with the controlled evidence: the estimator matters little, while off-policy and numerical correctness matter for stability.

### Gaps
- Several 2017–2024 entries are [PK]: IDs, titles, and dates were verified but the bodies were not re-read. Their one-liners should be spot-checked before publication.
- 2026 frontier reports (Kimi K2.5, GLM-5, Qwen3.5, MiniMax-M2.x) could not be located by search this session.

---

## Q7. Algorithm × design-axis table

### Takeaway
Most named algorithms differ on only one or two axes. Several axes are near-universal in 2025–2026: group baseline, outcome reward, no KL for math. The table separates what each method *specifies* from what was *controlled-tested*.

### Cited Findings
Legend: "—" = not specified or default inherited. "tok" = token-level (sum over all tokens / total tokens). "seq" = per-sequence mean, then mean over sequences. "PK" rows were not re-read this session.

| Algorithm (source) | Advantage / baseline | Normalization | Loss aggregation | Clip / trust region; ratio granularity | KL / entropy | Off-policy correction | Data / sampling | Reward type |
|---|---|---|---|---|---|---|---|---|
| PPO-RLHF (InstructGPT, N+ details) [1909.08593](https://arxiv.org/abs/1909.08593) PK; [2403.17031](https://arxiv.org/abs/2403.17031) | Learned critic, GAE (λ 0.95, γ 1) | Advantage whitening; optional reward whitening | Token (per-token returns) | Token ratio clip 0.2; value clip 0.2 | Per-token KL-to-SFT in reward (β 0.05) | PPO ratio over K=4 epochs | Prompt reuse | Learned RM (EOS score); −1 if no EOS |
| ReMax [2310.10505](https://arxiv.org/abs/2310.10505) PK | Greedy-sample reward baseline | — | Seq | None (REINFORCE) | KL in reward | None | 1 sample + greedy | Learned RM |
| RLOO [2402.14740](https://arxiv.org/abs/2402.14740) ABS | Leave-one-out mean over k | None | Seq (whole completion as one action) | None (on-policy REINFORCE) | KL in reward | None | k samples per prompt | Learned RM |
| GRPO [2402.03300](https://arxiv.org/abs/2402.03300) PK | Group mean | ÷ group std | Seq (1/\|o\| then mean) | Token clip 0.2 | K3 KL-to-ref in loss (β~0.04) | PPO ratio | G samples | Outcome or process RM |
| REINFORCE++ [2501.03262](https://arxiv.org/abs/2501.03262) ABS | Global-batch (optionally group-mean first) | Global-batch std | — | Token clip | KL (in reward) | PPO ratio | — | RM or verifiable |
| Kimi k1.5 [2501.12599](https://arxiv.org/abs/2501.12599) | Mean reward of k samples | None | — | Squared-loss regularizer (mirror descent) instead of clip | KL to previous iterate (τ); optimizer reset per iteration | Off-policy-tolerant surrogate; partial rollouts | Curriculum; prioritized ∝ (1−success) | Verifiable plus length reward |
| DAPO [2503.14476](https://arxiv.org/abs/2503.14476) | Group mean | ÷ group std | **Tok** | **Asymmetric token clip 0.2/0.28** | **None** | PPO ratio (16 updates per rollout) | **Dynamic sampling** (drop all-0/all-1 groups, refill) | Binary plus **soft overlong penalty**; overlong filtering |
| Dr. GRPO [2503.20783](https://arxiv.org/abs/2503.20783) | Group mean (= RLOO × G/(G−1)) | **None** | **Constant normalizer** (generation budget) | Token clip 0.2 | None (KL 0) | PPO ratio | — | Binary (Math-Verify) |
| Open-Reasoner-Zero [2503.24290](https://arxiv.org/abs/2503.24290) PK | Critic, GAE λ=1, γ=1 | — | — | Token clip | None | PPO | — | Binary |
| VAPO [2504.05118](https://arxiv.org/abs/2504.05118) ABS/PK | Pretrained critic; decoupled, length-adaptive GAE | — | Tok | Clip-higher | None | PPO | Dynamic sampling (PK) | Binary |
| Skywork-OR1 / MAGIC [2505.22312](https://arxiv.org/abs/2505.22312) | Group mean | ÷ std | Tok (removes 1/\|y\|) | Clip (higher tested) | **No KL; adaptive entropy bonus** (target 0.2) | Emphasizes on-policy (N_SGD=1) | Zero-advantage group rejection; multi-stage length; τ=1.0 | Binary |
| AceReason [2505.16400](https://arxiv.org/abs/2505.16400) | Group mean | ÷ std | Tok | Clip (irrelevant: strictly on-policy, ratio = 1) | None; entropy coefficient 0 | Strict on-policy | Hard-prompt filtering; staged 8K→32K; math then code | Binary |
| CISPO (MiniMax-M1) [2506.13585](https://arxiv.org/abs/2506.13585) | Group-relative | ÷ std | Tok (later sample-level with token normalization) | **Clips the stop-gradient IS weight** (upper only); no update dropped | None | IS-weighted REINFORCE; FP32 LM head | Dynamic sampling; repetition early-stop | Verifiable plus length penalty |
| Magistral [2506.10910](https://arxiv.org/abs/2506.10910) | Group mean | **Minibatch** normalization | Tok within group (÷ Σ\|o_i\|) | ε 0.2 / ε_high 0.26–0.3 | None; no entropy bonus | Async generators updated mid-generation | Zero-advantage filtering | Format gate 0.1 + correct 0.9 + language 0.1; soft length penalty |
| GSPO [2507.18071](https://arxiv.org/abs/2507.18071) ABS | Group-relative | ÷ std | Seq | **Sequence-level length-normalized ratio** and sequence clip | — | Sequence ratio (claimed to remove need for MoE routing replay) | — | Verifiable |
| GMPO [2507.20673](https://arxiv.org/abs/2507.20673) ABS | Group-relative | ÷ std | **Geometric mean** over tokens | Token clip (wider) | — | PPO ratio | — | Verifiable |
| Clip-Cov / KL-Cov [2505.22617](https://arxiv.org/abs/2505.22617) ABS | GRPO | — | — | Clip or KL on **high-covariance tokens** only | Entropy control via covariance | — | — | Verifiable |
| Lite PPO [2508.08221](https://arxiv.org/abs/2508.08221) | Group mean | **Batch std** | Tok | Standard clip 0.2 (no clip-higher) | None | PPO ratio (1 epoch) | No dynamic sampling, no overlong filter | Binary |
| M2PO [2510.01161](https://arxiv.org/abs/2510.01161) ABS | Group-relative | — | — | **Second-moment constraint on IS weights** (masks outliers) | — | Designed for staleness of 256 or more | — | Verifiable |
| ScaleRL [2510.13786](https://arxiv.org/abs/2510.13786) | Group (prompt) mean | **Batch std** | **Prompt-level** average | CISPO IS-weight clip | None | PipelineRL k=8; FP32 logits | Zero-variance filter; No-Positive-Resampling (pass ≥ 0.9); interruptions | ±1 verifiable |
| SAPO [2511.20347](https://arxiv.org/abs/2511.20347) ABS | Group-relative | — | — | **Soft temperature-gated** trust region; token-adaptive, sequence-coherent | — | Soft attenuation of off-policy tokens | — | Verifiable |
| DeepSeek-V3.2 GRPO [2512.02556](https://arxiv.org/abs/2512.02556) | Group mean | **None** (formula) | Seq (1/\|o\|) | Token clip | **Unbiased K3 KL** (IS-reweighted), domain-dependent | **Off-policy sequence masking** (negative advantage and divergence > δ); Keep Routing; Keep Sampling Mask | — | Verifiable, generative RM, agentic |
| OlmoRL [2512.13961](https://arxiv.org/abs/2512.13961) | Group mean | **None** | Tok | 0.2/0.272 token clip | None | **TIS** (cap 2.0); async ≤8 with in-flight updates | Zero-gradient filter plus **active sampling**; offline pass-rate filter (>62.5% removed) | SymPy math, test-case code, IF constraints, LLM judge for chat |
| JustRL [2512.16649](https://arxiv.org/abs/2512.16649) | Group mean | ÷ std (verl default) | — | Clip 0.2/0.28 | None; no entropy regularization (Table 1 vs Table 2 inconsistency) | — | No filtering; single stage 16K | Binary (DAPO verifier) |

### Inferences
- The axis with the most variation in 2025–2026 is the **trust-region / IS axis** (token clip, asymmetric clip, IS-weight clip, sequence ratio, geometric mean, soft gate, second-moment, TIS, sequence masking). It is also where controlled evidence of ceiling effects is thinnest (ScaleRL only) and where implementation divergence is most likely (GSPO diverged in ScaleRL; the CISPO held-out collapse in one blog).
- The advantage and normalization axes have converged toward a group mean with optional batch std and are empirically low-stakes.

### Gaps
- Exact default ε and normalization choices for GSPO, GMPO, SAPO, M2PO, VAPO, and REINFORCE++ come from abstracts or prior knowledge and should be verified against their method sections before the table is published.
