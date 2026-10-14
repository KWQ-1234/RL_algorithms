# Verification: Axis 4 (clipping / trust regions / ratio granularity)

Checked 2026-10-14 against arXiv PDFs (text extracted with PyMuPDF; numbers read from table text, not summaries).
Status key: VERIFIED / PARTLY (core right, detail off) / WRONG / NOT FOUND / FIGURE-ONLY.

| # | Claim | Status | Location | Evidence (verbatim ≤25 words or cell values) |
|---|---|---|---|---|
| 1a | Tricks or Traps (2508.08221): base-model clip rate ≈0.003; raising ε_high has little or negative effect on base | VERIFIED | §4.2.1, p.10 | "the base models operate with a low policy clipping rate, approximately 0.003"; base: "minor effects on policy entropy and even damages the performance" |
| 1b | On aligned models clip-higher slows entropy collapse and helps | VERIFIED | §4.2.1, Fig. 7/8 | "raising the upper clipping bound notably slows the entropy collapse, leading to consistent performance improvements" |
| 1c | Best ε_high 0.32 (aligned Qwen3-4B), 0.28 (aligned Qwen3-8B), non-monotone | PARTLY | §4.2.3, Takeaway 6, Fig. 11 (grid 0.20/0.24/0.28/0.32) | 4B: "performance gradually improves as the upper bound of the clip increases. And at 0.32, it demonstrates the best". So 4B is **monotone** (the "scaling law"); only 8B is non-monotone: "does not show a progressive improvement… more prominent when… 0.28" |
| 1d | Single seed (42); ~500 steps | PARTLY | App. A.1 config | Config lists `seed: 42`, `max_steps: 500`. Paper never says "single seed" explicitly (no multi-seed runs or error bars reported). Fig. 11 x-axis runs to 750+ steps, so not all runs stop at 500 |
| 1e | Token-level agg. better on base, sequence-level better on aligned? | VERIFIED | §4.3.1, Takeaway 7, Fig. 12/17 (Qwen3-8B) | "token-level aggregation is best suited for base models, while response-level aggregation is preferable for instruction-tuned models" |
| 2a | Entropy Mechanism (2505.22617) Qwen2.5-7B avg: GRPO 38.6 / clip-higher 38.8 / Clip-Cov 40.4 / KL-Cov 40.6 | VERIFIED | Table 2, p.12 | Avg column: 38.6, 38.8, 40.4, 40.6 |
| 2b | Qwen2.5-32B avg 45.8 / 47.2 / 50.3 / 52.2 | VERIFIED | Table 2 | Avg column: 45.8, 47.2, 50.3, 52.2 |
| 2c | Clip-higher becomes unstable later | VERIFIED | §4.3, p.13 | "it gradually becomes unstable, with performance saturating and declining" |
| 2d | ΔH ≈ −η·Cov(log π, π·Â) for vanilla PG | VERIFIED | Theorem 1, §3.2 | H(π^{k+1}|s) − H(π^k|s) ≈ −η·Cov_{a∼π^k}(log π^k(a|s), π^k(a|s)·A(s,a)). (Theorem 2, natural PG: Cov(log π, A) without the π factor) |
| 2e | Fitted law R = −a·exp(H) + b | VERIFIED (note typo in paper) | Eq. 6, §2.4 | "R = −a exp(H) + b". The §2 Takeaway box writes "R = −a exp(H + b)", inconsistent with Eq. 6; cite Eq. 6 |
| 3a | ScaleRL (2510.13786): GSPO and CISPO improve asymptote A over DAPO | VERIFIED; numeric A FIGURE-ONLY | §3.2 "Loss type", Fig. 5a | "both GSPO and CISPO substantially outperform DAPO, improving the asymptotic pass-rate A by a large margin." No A values for Fig. 5a in text |
| 3b | LOO: CISPO B = 2.01 vs DAPO-loss B = 1.77, similar A | VERIFIED | §4 "Are the design choices worth it?", Fig. 7 table; App. A.10 | "more efficient than DAPO, as seen in LOO experiment (B = 2.01 vs B = 1.77)"; "reverting to DAPO yields similar asymptotic performance"; A.10: all LOO A "within ±0.02". LOO = 8B, 16k GPU-h |
| 3c | CISPO insensitive to ε_max ∈ {4,5,8} | VERIFIED | App. A.17 intro; A.17.2, Fig. 19b | "varying ϵmax for CISPO between {4, 5, 8} produced no significant differences in performance" (lower clip fixed at 0) |
| 3d | DAPO ε_max affects A non-monotonically; "controls the asymptotic" wording | VERIFIED (behavior) / NOT FOUND (exact wording) | App. A.17.1, Fig. 19a | "As ϵmax increases, the terminal reward A increases until an optimal range is reached, after which A decreases again"; "ϵmax governs the asymptotic error itself"; Fig. 19a caption: "fundamentally changes the asymptotic performance value A". The phrase "controls the asymptotic" does not appear |
| 3e | GSPO diverged mid-training multiple times; 8B recoverable from checkpoint, Scout not | VERIFIED | App. A.17.4 | "On multiple occasions, GSPO runs diverged mid-training… For 8B models, restarting from a stable checkpoint allowed recovery, but this strategy failed on… Scout" |
| 3f | Appendix sections | — | A.17 (Loss type stability), A.17.1 DAPO ratios, A.17.2 CISPO ratios, A.17.3 GSPO ablations (best scale 10⁻³; 4e-3 vs 5e-3), A.17.4 GSPO vs CISPO; A.10 LOO (Fig. 15) | Also: GSPO's original 10⁻⁴ scale "did not work well in our setting" |
| 4a | MiniMax-M1 (2506.13585): reflective tokens (However, Recheck, Wait, Aha) clipped after first on-policy update | VERIFIED | §3.1 "Issues of Token Clipping" | "these tokens were clipped out after the first on-policy update, preventing them from contributing to subsequent off-policy gradient updates." Observed on their hybrid-attention model, zero-RL |
| 4b | DAPO clip-higher "less effective" with 16 rounds of off-policy updates | VERIFIED | §3.1 | "less effective in our setup, which involved 16 rounds of off-policy updates per generation batch" |
| 4c | CISPO matches DAPO with 50% of steps (Qwen2.5-32B, AIME24) | VERIFIED in text; curve FIGURE-ONLY | §3.1 + Fig. 2 caption; §1 says "2x speedup" | "matches DAPO's performance with only 50% of the training steps" (Qwen2.5-32B-base, zero-RL, DAPO math data). No table; the numbers are only in the Fig. 2 curve |
| 5a | CE-GPPO (2509.20712) on R1-Distill-Qwen-1.5B, 5-bench avg: base 45.2, CISPO 47.6, GSPO 54.3, CE-GPPO 54.9 | VERIFIED | §5.4, Table 2 (AIME24/AIME25/HMMT25/MATH500/AMC23) | Avg: 45.2 / 47.6 / 54.3 / 54.9 (CE-GPPO w/ β1=0.5, β2=1). Text: CISPO "exhibits model collapse during training" |
| 5b | 7B AIME24: CE-GPPO 66.0 vs DAPO 59.7 | VERIFIED | Table 1 (DS-R1-Distill-Qwen-7B) | DAPO 59.7; CE-GPPO β1=0.75, β2=1: 66.0 (β1=0.5 gives 62.3). DAPO here = GRPO + clip-higher ε_high 0.28 |
| 6a | ARLArena (2602.21534) Qwen3-4B agentic avg: GSPO 52.28, GRPO 46.16, CISPO 34.03, SAPO 32.22 | VERIFIED | §4.1, Table 3 | Exact cells as claimed. Model is the "SFT version of Qwen3-4B"; avg over ALFWorld/WebShop/Sokoban/TIR-Math |
| 6b | CISPO collapse around step 130 | VERIFIED | §4.1, Fig. 2 (ALFWorld) | "such aggressiveness leads to training instability, with collapse occurring around step 130" (stated for CISPO and SAPO jointly) |
| 6c | Sequence masking: CISPO ALFWorld 54.42 → 78.88; SAPO 25.16 → 76.92 | VERIFIED (table) | Table 4 (ALFWorld success) | CISPO 54.42 → 78.88; SAPO 25.16 → 76.92. Paper's body text misprints CISPO baseline as "54.12" |
| 6d | Collapse attributed to negative-advantage, low-ratio sequences | VERIFIED | §4.1 "Rooted cause", Fig. 3, Finding 2 | "Training collapse is largely driven by the accumulation of negative-advantage sequences with low IS ratios." Note: the token/sequence-level diagnosis in Fig. 3 is run on **SAPO**, not CISPO |
| 7a | MiniRL (2512.01374) on-policy: token-level PG w/o length norm + train-infer IS correction most stable | VERIFIED | §4.3, Fig. 1 (gbs = mbs = 1,024) | "MiniRL, i.e., the basic policy gradient algorithm with IS correction, achieves the best performance and training stability." Length-norm: "suboptimal… although training remains stable" |
| 7b | Removing IS correction → rapid collapse with entropy drop | VERIFIED | §4.3 | "causes rapid training collapse and a sharp drop in entropy" |
| 7c | Routing replay gives no on-policy gain | VERIFIED (R3 only) | §4.3 | "Applying R3 in on-policy training does not yield performance gains"; R2 "inapplicable" on-policy (Table 1) |
| 7d | Off-policy (N = 2/4/8) needs both clipping and routing replay | VERIFIED | §4.4, Figs. 2–4 | "both Routing Replay and clipping become essential for stable training" |
| 7e | R2 better at N = 2, R3 better at N ≥ 4 | VERIFIED | §4.4 | "When off-policiness is small (gbs = 2 × mbs), R2 outperforms R3, while… (gbs = 4 × mbs and gbs = 8 × mbs), R3 surpasses R2" |
| 7f | Once stable, on- and off-policy reach similar peaks | VERIFIED | §4.5 / Fig. 5 discussion | "both on-policy and off-policy training—once stabilized—also consistently achieve similar peak performance" |
| 7g | Is GSPO run? | Only mentioned, not run | Intro; footnote 3 (§4.1) | The string "GSPO" never appears; sequence-level clipping cited as Zheng et al. 2025 in footnote 3: "we found that the current clipping strategy has worked decently". App. A compares objectives to GRPO and CISPO only. Setup: Qwen3-30B-A3B-Base cold-start, FP8 inference / BF16 training |
| 8a | GSPO (2507.18071): ~10% of activated experts change after one update (Qwen3-30B-A3B) | VERIFIED | §5.3 Background | "after each RL gradient update… roughly 10% of the experts activated under the new policy… different" (48-layer Qwen3-30B-A3B-Base) |
| 8b | GSPO clips two orders of magnitude more tokens yet trains more efficiently | VERIFIED | §5.2 "Curious Observation on Clipping Fractions", Fig. 2 | "a difference of two orders of magnitude in the fractions of clipped tokens"; Fig. 2 bars: GRPO 0.0013, GSPO 0.15 |
| 8c | 4 minibatches per rollout batch | VERIFIED | §5.1 | "each batch of rollout data is partitioned into four mini-batches". Clip ranges: GSPO 3e-4/4e-4; GRPO 0.2/0.27 |
| 9 | 2509.26114: clip-low ↑ entropy, clip-high ↓ entropy; clip-high dominates under standard params, so entropy falls even with random rewards | VERIFIED | Abstract; §2.3 "Empirical validation", Fig. 3–4 | "with symmetric clipping parameters (εlow = εhigh = 0.2), the effect of clip-high dominates that of clip-low, leading to a reduction in entropy." Setting: GRPO without std normalization (Dr. GRPO style), rollout batch 512 / optimizer batch 256, no KL or entropy loss, T = 1, lr 5e-7. Models: text names Qwen2.5-3B-Instruct and Llama3-8B-Instruct, but Fig. 3 caption says Qwen2.5-1.5B-Instruct (inconsistent in paper). Fig. 4: entropy falls across Qwen, Llama and OLMo |
| 10 | DeepSeek-R1 (2501.12948): GRPO clip ratio ε = 10 in first RL stage; reason | VERIFIED | §3.2.1 "Training Details of the First RL Stage" | "the GRPO clip ratio ε to 10". Reason: "A lower value can lead to the truncation of gradients for a significant number of tokens… while a higher value may cause instability". Context: 8,192 outputs per rollout split into 16 minibatches, 1 epoch |
| 11 | TOPR (2503.14286): PPO ε = 0.2 makes little progress off-policy (Llama-3-8B, GSM8K) because ratios leave [1−ε, 1+ε] | VERIFIED | §4.3, Fig. 1 | "For PPO, we use ϵ = 0.2."; "PPO makes little progress on the objective as most data points quickly fall outside of its [1 −ϵ, 1 + ϵ] range" (Llama 3 8B Instruct) |
| 12 | Spurious Rewards (2506.10947): disabling clipping removes random-reward gain on Qwen2.5-Math-7B; how many ways | VERIFIED: three ways | §4, Fig. 4 | "three separate no-clipping variants… disabling clipping in the implementation, increasing the mini-batch size to match the rollout size, or reducing the rollout size" (last two make π_θ = π_old). Derivation App. B.1 |
| 13a | Magistral (2506.10910): ε_high adjusted 0.26–0.28 to keep group entropy stable | VERIFIED | §2.1 | "We adjusted it between 0.26 and 0.28 during the training to keep the group entropy stable." (Magistral Medium; Magistral Small RL used ε_high = 0.3, §5.3) |
| 13b | Clip-higher instead of entropy bonus | VERIFIED | §7.4.2 "Entropy targeting", Fig. 12 | Entropy bonus "unstable as the effect… varies significantly depending on the dataset"; "we found it more effective to depend on εhigh" |
| 13c | Final recipe one update per batch | VERIFIED | §6.3 | "we opt to keep ratio nasync/nbatch ≤ 2 and nbatch = nminibatch during final training" |
| 14 | DCPO (2509.02333): 0.5 + ½√(1 − 4ε_low/q) ≤ r ≤ 0.5 + ½√(1 + 4ε_high/q); ε_low 0.16, ε_high 0.2 | VERIFIED (two details missing) | Eq. 4 (§2); App. A.6 Eq. 28; App. A.9 | Lower bound is ½√(**max**(1 − 4ε_low/q, 0)); both bounds capped at r_max = 10. "low/high clipping thresholds (0.16, 0.2) for DCPO", chosen to match GRPO ε = 0.2 at two boundary points. Derived from (r−1)·p ≤ ε with p = r·q (Eq. 3) |

## Notes

- **1c.** The blog should say the 4B curve is monotone over {0.20, 0.24, 0.28, 0.32}, with the best value at the top of the grid. The optimum may be higher than 0.32, which was not tested. Only 8B peaks inside the grid, at 0.28.
- **1d.** "Single seed" is an inference from the config. Safer wording: "reported config uses seed 42; no multi-seed variance reported".
- **3a.** The A values for DAPO, GSPO and CISPO appear only in Fig. 5a. The A values that do appear in text are the FP32 fix (0.52 → 0.61) and the final runs (Table 1: ScaleRL 8B 0.610, Scout 0.710).
- **4.** The clipping diagnosis was made on M1's hybrid-attention model. The 50% / 2× figure comes from a separate Qwen2.5-32B-base controlled study.
- **6.** Table 3's caption has its own inconsistency: it gives SAMPO avg 59.55, while the table cell says 60.21. This does not affect the claimed numbers.

## WRONG / NOT FOUND / qualified items

1. **1c (PARTLY WRONG):** "non-monotone" holds only for aligned Qwen3-8B (best 0.28). For aligned Qwen3-4B, performance rises monotonically up to 0.32, which is the edge of the grid.
2. **1d (qualified):** seed 42 and max_steps 500 are in the config. A single seed is never stated explicitly, and Fig. 11 runs past 750 steps.
3. **3a (FIGURE-ONLY):** no numeric A values are given for DAPO, GSPO or CISPO in the loss-type comparison.
4. **3d (NOT FOUND, wording):** the phrase "controls the asymptotic" does not appear. Use "ϵmax governs the asymptotic error itself" (A.17.1) or "fundamentally changes the asymptotic performance value A" (Fig. 19 caption).
5. **4c (FIGURE-ONLY for the data):** the 50% claim is stated in text, but the only supporting data is the Fig. 2 curve.
6. **6d (qualified):** the collapse diagnosis was done on SAPO, not CISPO.
7. **7c (qualified):** only R3 was tested on-policy, because R2 is inapplicable there.
8. **7g:** GSPO is not run in 2512.01374. It is only cited (footnote 3).
9. **9 (qualified):** the paper names its models inconsistently (Qwen2.5-3B vs 1.5B-Instruct).
10. **14 (qualified):** the formula is missing max(·, 0) under the lower-bound square root and the r_max = 10 cap.
11. **2e (paper typo):** the Takeaway box gives R = −a·exp(H + b), but Eq. 6 is R = −a·exp(H) + b. The claim matches Eq. 6.
