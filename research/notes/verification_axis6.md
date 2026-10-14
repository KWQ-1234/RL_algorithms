# Verification — Axis 6: off-policy data and train–inference mismatch

Checked 2026-10-14 against primary sources. arXiv PDFs were downloaded and text-extracted with PyMuPDF. The Thinking Machines post was fetched as raw HTML. The two Notion blogs were rendered in the built-in browser, with every block collected while scrolling.
Status key: **V** = verified, **W** = wrong (correct value given), **NF** = not found in source, **FIG** = figure-only (not stated in text), **N** = verified with a nuance that the blog wording should reflect.

## 1. R3 — Rollout Routing Replay (arXiv 2510.11370 v2)

| Claim | Status | Location / evidence |
|---|---|---|
| ~10% routers differ train vs infer (Qwen3-30B-A3B) | V | §3.2, Fig 3a: "approximately 10% of the routers select different experts during training compared to inference" |
| ~94% tokens differ in ≥1 layer | V | §3.2, Fig 3b: "94% of tokens select a different expert in at least one layer" |
| KL 1.535e-3 (MoE) vs 6.4e-4 (dense Qwen3-8B) | V | §3.1: "Qwen3-30B-A3B (MoE) is 1.535×10−3, while that of Qwen3-8B (Dense baseline) is 6.4 × 10−4" |
| Two identical Megatron passes → 8.4e-4 | V | §3.3, Fig 4: "KL divergence = 8.4 × 10(−4)". Note that this is larger than the dense train–infer KL (6.4e-4). With R3 the train–infer KL drops 1.5e-3 → 7.5e-4 (§4.3) |
| GRPO crash steps 60 / 105 / 120 | V | Table 1, Crash Step column. **120** = SFT model, mini_step=8, LR 1e-6. **60** = SFT, mini_step=1, LR 3e-6. **105** = Base, mini_step=1, LR 3e-6. LRs from §5.1: "mini step to 8 and the learning rate to 1 × 10−6 … increase the learning rate to 3 × 10−6" |
| GRPO+TIS crash at 105 | V/N | Table 1. The crash is at 105 only for **SFT, mini_step=1**. In Base mini_step=1, GRPO+TIS has no crash ("-"). TIS uses C=2 |
| R3 runs: no crash within 180 steps | V | Table 1: every +R3 row has Crash Step "-", with max_global_step=180 |
| SWE-bench Verified 31.8 vs 38.6 | V | Table 2: GRPO 31.80 (best step 70, crash step 90), GRPO+R3 38.60 (best step 160, no crash). Model is Qwen3-30B-A3B, mini_step=1, LR 2e-6 |
| Collapse when fraction of ratio>2 tokens exceeds 0.1 | N | §5.2 reports this as one observed run, not a general threshold: "in the SFT model trained using GRPO under the single mini-step setting, after 60 global steps, the value of 𝐹(𝜏= 2) exceeded 0.1". With R3, F(τ=2) stays below 1e-4 |
| Overhead < 3% | V | §4.1: "overall latency overhead below 3% during the rollout phase" |
| Best-checkpoint reporting | V | §5.1: "evaluate the model performance every 5 global steps and report the highest observed performance". Seeds are not mentioned, so each result is a single run |

## 2. Ring-1T / IcePop (arXiv 2510.18855)

| Claim | Status | Location / evidence |
|---|---|---|
| M(k)=k if k∈[α,β] else 0, k=π_train(θ_old)/π_infer(θ_old) | V | §2.3.2 Eq. (1)–(2). M multiplies the PPO-clipped term, and the KL coefficient γ is also present |
| α=0.5, β=5 | V | §2.3.4 recipe: "IcePop (α = 0.5, β = 5)". The same values are used in the §2.4.1 setup |
| Ring-mini-2.0 baseline collapses after ~180–200 steps | V/FIG | Appendix (Preliminary Analysis), Fig 10 caption: "The reward of baseline collapses after 180–200 steps" |
| IcePop and TIS stable | V | Same section: "both IcePop and TIS mitigate the instability of RL training within 600 gradient steps" |
| Narrow band [0.5, 2] destabilizes | V | Sensitivity analysis, Fig 13: "The narrow masking range [0.5, 2.0] immediately destabilizes training" |
| ~1–2‰ tokens masked; higher entropy | V | Fig 12: "IcePop maintains 1–2‰ of tokens clipped … clipped tokens show a higher proportion of high-entropy tokens" |
| +6% relative over TIS on AIME25 | V | §2.4.1: "expands the performance gap with TIS by relative 6%". The same sentence says it "improves the base score (63%) by over 14%". AIME25 is Avg@64. This is a single run (Fig 5) |

## 3. FP16 paper (arXiv 2510.26788)

| Claim | Status | Location / evidence |
|---|---|---|
| Sanity test on DeepSeek-R1-Distill-Qwen-1.5B | V | §4: a "perfectible" set of 1,460 MATH problems with initial accuracy between 20% and 80%. Context 8K, 8×A100, batch 64×8, 4 gradient steps per iteration |
| BF16 GRPO peak 73% (VeRL) / 84% (Oat) then collapse | V | §4.2: "reaching a peak accuracy of only 73% in VeRL and 84% in Oat before its performance degrades" |
| BF16 token-TIS collapse after 82% / 88% | V | §4.2: "collapsing after reaching 82% (VeRL) and 88% (Oat)" |
| BF16 GSPO NaN after 1,200 steps | V | Footnote 4: "In our VeRL experiment, the GSPO gradient norm became 'NaN' after 1200 steps". This is the VeRL run only |
| BF16 seq-MIS stable but 95% vs FP16 99% | V | §4.2: "maximum training accuracy of only 95% (vs. 99% in FP16)" |
| AIME24 34% vs 39% | V | Same sentence: "score of 34% (vs. 39% in FP16) on the AIME 2024 benchmark" |
| Sequence log-ratio mismatch ~24× smaller with FP16 | V | §3.5 (offline analysis, Fig 2): "FP16 maintains the mismatch at a much milder level (approximately 24× smaller)" |
| No seeds reported | V | The word "seed" does not appear anywhere in the paper |

## 4. VeXact (arXiv 2605.14220 v2)

The paper is titled "Diagnosing Training-Inference Mismatch in LLM RL via a Zero-Mismatch Reference" and comes from ByteDance and UVA.

| Claim | Status | Location / evidence |
|---|---|---|
| MoE REINFORCE, Qwen3-30B-A3B, vLLM train/val 0.574/0.293 → 0.255/0.067 after ~step 280; VeXact 0.753/0.534 | V | §3.2, Fig 2: "starts to degrade after step 280, with training and validation rewards decreasing from 0.574 and 0.293 to 0.255 and 0.067 … reaching 0.753 training reward and 0.534 validation reward". MoE is trained on DAPO and evaluated on AIME24 every 20 steps. REINFORCE uses batch-whitened advantages |
| Qwen3-1.7B GRPO numbers | V (values) | §4.1, Fig 3. Qwen3-1.7B, FSDP, trained on Sanity-Test-R1D-1.5B. VeXact holds training reward ~0.93. **vLLM recompute** drops ~0.87 → ~0.40 over the first 650 steps, partially recovers, falls again after ~1610 and collapses to near zero after ~1665. **vLLM bypass** degrades in one stage to ~0.4 with no collapse to zero. Also §4.x: on GSM8K, Qwen3-1.7B shows "17.3% of token logprob change directions are misaligned with their advantage sign" |

## 5. CIS (arXiv 2609.32444)

| Claim | Status | Location / evidence |
|---|---|---|
| 3 MoE models, 3 seeds, GSM8K training | V | §4.1. Models are Qwen1.5-MoE-A2.7B-Chat, DeepSeek-V2-Lite and Qwen3-30B-A3B. All are trained on GSM8K and evaluated with greedy pass@1 on five benchmarks. Table 1: "Mean ± standard deviation over three seeds" |
| Qwen3-30B-A3B 5-bench averages | V | Table 1: No corr 68.26±0.27, TIS 69.14±0.30, Seq-TIS 68.99±0.61, Seq-MIS 68.68±0.15, IcePop 68.99±0.51, GSPO 69.23±0.16, FP16 68.53±0.31, CIS 69.88±0.55. Two rows the claim omits: Base 67.84, Exact ratio 68.72, KPop† 69.06 |
| Qwen1.5-MoE: lifting small weights → 6.01 vs base 28.93 | V | §4.3 / Table 2 ablation, two-sided weight w=max{min{k,1+λφ},(1+p)/2}: "reduces the average accuracy from 34.78 to 6.01". It is also "below the base model (28.93)". The cap C=1.13 also collapses, to 8.88 |
| GSPO 28.90 below base | V | Table 1, Qwen1.5-MoE: GSPO 28.90±1.25 vs base 28.93 (a 0.03 gap) |
| Authors concede margins within seed variance | N | The concession is narrower than the claim. §4.2: "margins over IcePop on several transfer benchmarks are smaller than the across-seed variation". For the OOD pooled set, "its margin over IcePop is within the seed spread". The authors still claim the highest 5-bench average on all three models. They do not concede that the average margins are within noise. On Qwen3-30B-A3B, though, CIS 69.88±0.55 vs GSPO 69.23±0.16 is visibly close |

## 6. AReaL (arXiv 2505.24298)

| Claim | Status | Location / evidence |
|---|---|---|
| Staleness table (AIME24, w/o → with decoupled) | V | Table 2: η=0 42.0. 1: 41.8→42.1. 2: 40.0→41.8. 4: 23.3→42.2. 8: 35.7→41.0. 16: 35.8→38.7. ∞: 34.0→36.9 |
| Model size | V | Fig 5 caption: "with a 1.5B model on math reasoning tasks". Learning curves cover 250 steps. PPO uses 4 minibatches (Table 3) |
| Single run? | V | NeurIPS checklist Q7: "We present results within a single trial under the same fixed random seed". App. A: "fixed random seed of 1 across all experiments" |
| Decoupled objective | V | §5, Eq. (4)–(5): J = E_{a∼π_behav}[ Σ_t (π_prox/π_behav) · min(u_t Â, clip(u_t,1−ε,1+ε) Â) ], with u_t = π_θ/π_prox |
| Speedups 2.77× (1.5B) to 1.64× (32B) | V/N | 2.77× is stated (abstract, §7.2). **1.64× is not stated**. It is computed from Table 1 training hours vs Sync.AReaL (32B: 51.1/31.1 = 1.64). The other sizes are 7B 57.7/25.4 = 2.27 and 14B 48.8/21.9 = 2.23. Against VeRL hours, 32B is 46.4/31.1 = 1.49×. The 14B and 32B models are **coding** models (LiveCodeBench), not math |

## 7. PipelineRL (arXiv 2509.19128)

| Claim | Status | Location / evidence |
|---|---|---|
| Conventional RL stable up to G=32; G=64 diverged | V/N | §5: "G = 32 … the maximum G for which Conventional RL training was stable … Further increasing G to 64 results in divergence, see Figure 10". The paper also says G=32 eventually becomes "slower and unstable" |
| ~2× faster to the same reward | V | §5: "achieves the same reward values approximately ∼2x faster than this baseline" |
| Truncated IS c=5 | V | §5 setup: "importance weighted REINFORCE … clamp the importance weights to 5" |

## 8. M2PO (arXiv 2510.01161)

| Claim | Status | Location / evidence |
|---|---|---|
| Qwen3-Base-4B 50.7 (GRPO s=0) / 40.1 (GRPO s=256) / 51.3 (M2PO s=256) | V | Table 1 Avg column. GSPO s=256 = 43.2 |
| τ_M2 = 0.04 | V | §5: "we consistently set τM2 =0.04" |
| Best checkpoint? | V | §6.1: "evaluate … every 50 steps and report the performance of the checkpoint that obtains the best average performance on eight benchmarks" |
| Seeds? | V (none) | No seeds or repeated runs are mentioned |

## 9. Magistral (arXiv 2506.10910)

| Claim | Status | Location / evidence |
|---|---|---|
| "performance degrades when there are more than 2 minibatches" | V | Fig 6 caption: "degrades when there are more than 2 minibatches in a batch". The ablation used a 3B model, GRPO on math, and n_async=4096 |
| Final recipe: one update per batch | V | §6.3: "we opt to keep ratio nasync/nbatch ≤2 and nbatch = nminibatch during final training". The infra description also says "A single gradient step is performed" |

## 10. MiniMax-M1 (arXiv 2506.13585)

| Claim | Status | Location / evidence |
|---|---|---|
| FP32 LM head raised correlation ~0.9x → ~0.99x | V | §3.x: "improved the correlation between training and inference probabilities from approximately 0.9x to 0.99x". Fig 3 shows Pearson 0.987319 → 0.997135 |
| 16 off-policy updates per generation batch | V | CISPO motivation: "our setup, which involved 16 rounds of off-policy updates per generation batch" |

## 11. Scaling laws for collapse in async GRPO (arXiv 2607.01083 v3)

| Claim | Status | Location / evidence |
|---|---|---|
| Max stable LR ∝ 1/S | V | Abstract: "the largest stable learning rate scales approximately as S−1 … approximately constant product Sη" |
| Collapse time ∝ 1/η | V | Abstract: "estimated collapse times scale approximately as η−1". The cumulative LR budget t·η ≈ c2 depends on the model and setup |
| Bias O(Sη) | V/N | Abstract: "Under local regularity conditions, the analysis yields an O(Sη) upper bound on the staleness-induced update bias that resets at synchronization". It is an upper bound under assumptions, not a measured quantity |
| Models | V | Llama-3.2-1B/3B (Instruct) sweeps, plus Qwen3-8B at S=16 (timing law only, different budget). Vanilla GRPO with constant LR and periodic sync. One seed (42) per configuration. Three-seed repeats (42, 100, 2026) cover only 3B at S=32 (App. C) |

## 12. Qwen "Stabilizing RL with LLMs" / MiniRL (arXiv 2512.01374)

| Claim | Status | Location / evidence |
|---|---|---|
| FP8 inference / BF16 training stress test | V | §4.1: "We adopt the setting of FP8 inference and BF16 training, providing a stress test". The model is a cold-start SFT of Qwen3-30B-A3B-Base. TIS threshold is 5 |
| On-policy: removing IS correction → rapid collapse | V | §4.3, Fig 1: "Removing the training–inference IS correction causes rapid training collapse and a sharp drop in entropy". In the same setting, R3 gives no gain on-policy, and R3 without IS still fails. The authors say this contradicts Ma et al., whose runs used BF16 inference and ≤180 steps |
| R2 vs R3 by N | V | §4.4: "When off-policiness is small (gbs = 2 × mbs), R2 outperforms R3, while when off-policiness is large (gbs = 4 × mbs and gbs = 8 × mbs), R3 surpasses R2". N = 2, 4, 8 with mini-batch 1,024 |

## 13. Yao et al. blog (fengyao.notion.site/off-policy-rl; last updated Oct 13 2025)

| Claim | Status | Location / evidence |
|---|---|---|
| PPO ratio never sees π_sampler | V (paraphrase) | "Extension to Other Algorithms". Rollouts are sampled from π_sampler(θ_old), but the PPO ratio is π_learner(a,θ)/π_learner(a,θ_old), because the FSDP backend recomputes π_θ_old |
| TIS = min(π_learner/π_sampler, C) | V | "How to Fix It?": truncated importance ratio min(π_learner(a,θ)/π_sampler(a,θ), C), with "C is a hyper parameter". In PPO form it uses θ_old |
| Ratio 16 → 256× vanilla-IS vs 4× TIS-2 | V | "Vanilla-IS v.s. TIS": "reaches 16 for one token … amplified by 256 times via Vanilla-IS, 4 times via TIS-2, or 64 times via TIS-8" |
| FP32 LM-head patch did not remove mismatch | V | "Does higher-precision vLLM help?": "the mismatch problem still exists after applying both patches". The two patches are return-actual-sampling-probs and fp32 lm_head |
| R1-Distill-1.5B BF16: TIS no gain | V | Fig 3 / "Does TIS always help?": "introducing the additional Truncated Importance Sampling term cannot bring performance gain". The caption notes a standard bf16 rollout and that "the mismatch is not huge" |

## 14. Liu & Li blog "When Speed Kills Stability" (yingru.notion.site)

| Claim | Status | Location / evidence |
|---|---|---|
| vllm-kl by hardware | V | §3 hardware: "H20 … 5e-4 to 1e-3, for the L20 it is around 1e-3 to 1e-2, and for the A100, it is predominantly between 1e-2 and 1" |
| Disable cascade attention on A100 → ~1e-3 | V | Fig 21 text: "vllm-kl metric decreased from the range of 5e-2 to 1e-1 to around 1e-3" |
| Failed L20 run recovered on H20 | V | Fig 8: resumed "from the checkpoint at the 200th training step of the failed L20 experiment". The text says "The training immediately stabilized and recovered" |
| Qwen3-14B-Base multi-turn TIR collapse ~step 200 | V | Fig 14/16 captions: "The vanilla on-policy experiment without IS crashes after 200 steps". There are four failed runs in Fig 1, using on- and off-policy GRPO on L20 |
| Token-TIS delays but doesn't prevent | V/N | Fig 14: "the run with token-level TIS still collapses later". This holds for the TIR setting. For simpler reasoning RL, the blog says token-TIS "prevented gradient explosion, but training remained unstable" (Fig 15) |
| Seq-MIS best | N | Fig 17: MIS "surpassed the peak training rewards and test accuracy of both the vanilla and TIS experiments". The same caption says both MIS and TIS "suffered from training instability in the later stages". So Seq-MIS scored best but was not fully stable. Seq-TIS resumed from step 200 "remains stable" (Fig 14) |
| FP32 head ineffective | V | §4.1.1: "the mismatch problem still persisted after the patch and the model collapse was unavoidable" |
| Token-level bias O(T²·D_TV^max) | V | TL;DR "Part 1" and "Part 2" summaries: "Token-level IS (PPO/GRPO) has O(T²D_TV^max) bias from the surrogate's first-order approximation error". It is derived from the TRPO bound J ≥ L_μ − C·T²·D_TV^max. Seq-IS variance is O((1+χ̄²_max)^T) |

## 15. Thinking Machines "Defeating Nondeterminism in LLM Inference"

| Claim | Status | Location / evidence |
|---|---|---|
| 1,000 temp-0 completions, Qwen3-235B → 80 unique | V | Qwen3-235B-A22B-Instruct-2507, prompt "Tell me about Richard Feynman", 1000 tokens each: "we generate 80 unique completions, with the most common of these occuring 78 times". Divergence first appears at token 103 (992 vs 8) |
| Batch-invariant: all identical | V | "when we enable our batch-invariant kernels, all of our 1000 completions are identical" |
| RL without IS collapses (~step 318?) | V/N | "the run without importance weighting has a significant loss spike around Step 318". That step is a **loss spike** that coincides with a KL spike. The reward collapse is described only as "partway through training". Setup: Bigmath RLVR, Qwen 2.5-VL instruct 8B, max rollout 4096. With IS, KL stays ~0.001 |
| Bitwise identical → KL = 0 | V | "True On-Policy RL … KL-divergence stays flat at 0" |
| Timings 26 s / 55 s / 42 s | V | Table: vLLM default 26, Unoptimized Deterministic vLLM 55, + Improved Attention Kernel 42. Setup: Qwen-3-8B on one GPU, 1000 sequences of 90–110 tokens |

## 16. DeepSeek-V3.2 (arXiv 2512.02556)

| Claim | Status | Location / evidence |
|---|---|---|
| Off-Policy Sequence Masking: mask when Â<0 and mean log(π_old/π_θ) > δ | V | §3.1 Eq. (8)–(9). M=0 when Â<0 and (1/\|o\|)Σ log(π_old/π_θ) > δ, otherwise 1. π_old is "the sampling probability directly returned by the inference framework". The text says it "improves stability in certain training scenarios" |
| Keep Routing called crucial for MoE RL stability | V | "This Keep Routing operation was found crucial for RL training stability of MoE models". It has been used since DeepSeek-V3-0324 |
| Keep Sampling Mask | V | Top-p/top-k truncation masks from π_old are applied to π_θ, "ensuring both policies share identical action subspaces" |
| Unbiased k3 KL with ratio factor | V | Eq. (7): (π_θ/π_old)·[π_ref/π_θ − log(π_ref/π_θ) − 1]. The paper says "the gradient of this KL estimator becomes unbiased" |

## 17. Skywork-OR1 (arXiv 2505.22312) and AceReason (arXiv 2505.16400)

| Claim | Status | Location / evidence |
|---|---|---|
| Skywork: more SGD passes per rollout batch → faster entropy collapse | V | §4.4, Ablation 6 / Fig 16. N_SGD ∈ {1,2,4}, via more minibatches or reuse: "accelerates entropy collapse and exhibits worse test performance". Summary: "significantly accelerates entropy collapse and generally results in degraded test performance" |
| AceReason: 22 or 44 updates per batch collapse ~step 100 | **W** | §3.x ("On-policy training to stabilize entropy loss"): "applying multiple **(2 or 4)** gradient updates … led to rapid entropy collapse around 100 steps (see Figure 3c)". The correct count is **2 or 4**, not 22 or 44 |
| AceReason: one update avoids it | V | "using exactly one gradient update after model generation … consistently prevented collapse" |

---

## WRONG / NOT FOUND / needs rewording (compact)

1. **WRONG — AceReason (2505.16400):** the update counts are "2 or 4" gradient updates per batch, not 22 or 44.
2. **Not stated (derived) — AReaL 1.64× (32B):** this number is computed from Table 1 hours against Sync.AReaL (51.1/31.1). Against VeRL it is 1.49×. 14B and 32B are coding models.
3. **Reword — R3 "collapse when F(τ=2)>0.1":** this is one observed run (SFT, mini_step=1, after step 60), not a stated threshold.
4. **Reword — R3 GRPO+TIS crash at 105:** this applies only to SFT, mini_step=1. On Base, mini_step=1, GRPO+TIS did not crash.
5. **Reword — CIS concession:** the authors concede only per-benchmark margins over IcePop and the OOD pooled margin. They do not concede the 5-bench averages.
6. **Reword — Liu & Li "Seq-MIS best":** Seq-MIS had the best peak, but the caption says both MIS and TIS "suffered from training instability in the later stages".
7. **Reword — Thinking Machines step 318:** this is a loss spike (with a KL spike), not a stated reward-collapse step.
8. **Context — Ring-1T baseline collapse 180–200 steps:** this is stated only in the Fig 10 caption (appendix).
