# Verification: data/sampling (axis 7) and reward signal (axis 8)

Date: 2026-10-14. Method: full-text PDFs from arxiv.org/pdf/<id> (latest version, noted below) extracted with PyMuPDF; Polaris from the original HKU NLP blog HTML. Tables with checkmarks or ambiguous column order were rendered to images and read directly. "p." means PDF page index, which can differ from the printed page number.
Status labels: VERIFIED / WRONG (correct value given) / NOT FOUND / FIGURE-ONLY (the claim is backed only by a plot, with no numbers in text or tables) / REWORD (the substance holds but the blog's phrasing or attribution is off).

---

## Data and sampling

### 1. ScaleRL, arXiv 2510.13786v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Group size G ∈ {8,16,24,32} at fixed total batch gives similar fits | VERIFIED (qualitative), see A values | §5 p.10; App. A.13 Table 1 p.25 | "leaves fitted scaling curves essentially unchanged (Appendix A.13)". Table 1 (Cmid / B / A): ScaleRL [16 gen, batch 768 = 48×16] 2542 / 1.92 / **0.610**; 8gen 2542 / 2.44 / **0.585**; 24gen 3054 / 2.22 / **0.590**; 32gen 2936 / 2.07 / **0.595**. |
| Note on the above | caveat | Table 1 | The A values span 0.585–0.610, a 0.025 spread, slightly more than the paper's own ±0.02 error margin (8gen vs 16gen). The Cmid of the 8gen run is identical to the baseline (2542), which may be a copy error in the paper. |
| Zero-variance filtering gives a higher asymptote (main text) | VERIFIED; magnitude FIGURE-ONLY | §3.2 p.7; Fig. 6a caption p.8 | "using the effective batch performs better asymptotically"; Fig. 6a caption: "find it achieves a higher asymptote". No A values are given in the text. |
| …but in the leave-one-out ablation (loo-batch) it falls within ±0.02 | VERIFIED | App. p.23 (text under Fig. 15, A.9–A.10); Fig. 7 p.9 | "We notice that all methods have similar values of A (within ±0.02 error margin range)." The per-run A/B table is embedded in the Fig. 7 image (FIGURE-ONLY). |
| No-Positive-Resampling (permanently drop prompts with pass rate ≥ 0.9) improves scalability and asymptote | VERIFIED; numbers FIGURE-ONLY | §3.2 p.8; Fig. 6b | "permanently removing any prompt with pass rate ≥0.9 from subsequent epochs… the curriculum improves scalability and the asymptotic reward A." No numeric A is given. |
| Truncation rate above ~10–15% predicts instability (batch 768) | VERIFIED (REWORD slightly) | App. A.15 p.25–26 | "At batch size 768, we observed that truncations in the range of 10–15% typically destabilized training"; "high truncation rates are a reliable warning signal of instability". The paper says "in the range of 10–15%", not "above". |
| Length penalty vs interruption | VERIFIED | App. A.10 "Controlling generation length" p.22; loo-lp in Fig. 15 | "replacing interruption with length penalty in our final ScaleRL recipe does not improve performance." Length-penalty settings: Lmax = 14k, Lcache = 2k. |
| Cmid 2542 (14k) vs 11272 (34k?) | Numbers VERIFIED; **"34k" WRONG → 32k** | Table 1 p.25; §5 p.10 | Table 1: ScaleRL 2542 / B 1.92 / A 0.610; **ScaleRL-32k** 11272 / 1.89 / 0.645. The main text says "Increasing the generation length from 14k to 32k". App. A.15 also says "34k generation length (batch 768)" once, an inconsistency inside the paper, so the blog should use 32k. |

### 2. JustRL, arXiv 2512.16649v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| No dynamic sampling / filtering | VERIFIED | §3.1 p.4 | "We train on DAPO-Math-17k… without offline difficulty filtering or online dynamic sampling strategies." |
| DAPO overlong penalty → AIME24 plateaus ~50 vs ~55 | VERIFIED | §4.4 p.8; Fig. 3 caption p.9 | "adding overlong penalty plateaus at 50% AIME 2024 (vs 55% baseline), and adding both modifications plateaus at 45%." The ablation ran 3,000+ steps. The full run reaches 58% (Fig. 1). |
| Entropy collapses to 0.5–0.6 | VERIFIED | p.9 | "driving entropy down to 0.5-0.6 compared to the 1.2-1.4 range in our base approach." Fig. 3 caption says both modifications collapse it. |
| Model is 1.5B | VERIFIED | title; §3.1 | DeepSeek-R1-Distill-Qwen-1.5B (the ablations) and OpenMath-Nemotron-1.5B. |
| Table 1 vs Table 2 inconsistency about KL | **No KL inconsistency; the inconsistency is about entropy** | Table 1 p.3 (rendered image); Table 2 p.5 | Table 1: JustRL rows have ✓ only under EC (Entropy Control) and ✗ under RKL (Reset KL), which agrees with Table 2 "Use KL Loss: No". However, Table 2 says "Use Entropy Regularization: No", while the Table 1 caption says JustRL models "use only entropy control". The text names clip-higher [0.8, 1.28] as "the one technique we do use", so the EC ✓ apparently refers to clip-higher, but the paper never states this. |

### 3. OLMo 3, arXiv 2512.13961v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Active sampling reduces loss variance | VERIFIED as a qualitative text claim; magnitude FIGURE-ONLY | §6 "Active sampling stabilizes training" p.65; Fig. 26 | "These consistent batch sizes have a stabilizing effect on training, and we see greatly reduced loss variance." Shown in an RL-Zero Math run. No numbers are given. |
| Footnote: overlong filtering gave no consistent improvement | VERIFIED | footnote 34, p.45 | "We experimented with additional changes (e.g., overlong filtering), but did not find these gave consistent improvements." |
| Offline filter removes prompts with pass rate > 62.5% | VERIFIED (scope caveat) | §4.4.2 Step 2, p.47 | "remove all samples that the model easily solves (that is, those with a pass rate greater than 62.5%)", using 8 rollouts at T = 1.0. **This applies only to 7B Think. The 32B relies on active sampling** and reuses the 7B-filtered data. |
| 3.0 → 3.1 gains from 16K length and not masking truncated samples | VERIFIED (scope: **RL-Zero**) | App. A.6.4 p.106 | "the major improvement comes from 1. setting completion length to 16k instead of 12k and 2. not masking truncated sequences". Reason given: batch-size variation from masking hurt stability. This is about Olmo RL-Zero 3.1 vs 3.0, not Think 3.1, which instead came from extending the RL run by 21 days. |

### 4. arXiv 2512.07611v1 (Lian, "Comparative analysis… PPO, GRPO, DAPO")

| Claim | Status | Location | Evidence |
|---|---|---|---|
| On Countdown, turning off dynamic sampling worked best | VERIFIED (with caveats) | Abstract p.1; §3.5 p.14; §4.2 Table 3 p.17–18 | "the best overall results are achieved with DAPO when DS is disabled." Model: Qwen2.5-1.5B-Instruct, G = 8, trained on Countdown and evaluated on GSM8K/MATH/BBH/MMLU-Pro. |
| Caveats | — | Table 3; Fig. 7 | Table 3 has **no DAPO-with-DS row**. Its rows are Base, PPO, GRPO and DAPO (No DS) only, e.g. GSM8K 48.4 / 50.3 / 50.8 / 53.3. The DS-on vs DS-off comparison is FIGURE-ONLY (Fig. 7a, Countdown accuracy; DS hinders after ~step 75). DS also added ">25% computation time per training step". This is a single-author, small-scale study, and its toy explanation says DS "may remove all correct responses", which does not match DAPO's DS (DAPO drops whole all-correct or all-wrong groups). |

### 5. BroRL, arXiv 2510.01180v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Survival after dynamic sampling: 41% at N=16 vs 62% at N=512 | Numbers VERIFIED; REWORD "group survival" | §4.2.4 Table 2 p.9 | "Dynamic Sampling Pass Rate": ProRL (16) 41%, BroRL (512) 62%; "62% of the generated **samples** are deemed useful for training". The paper calls it the fraction of samples, not groups. Throughput is 36.5 vs 72.4 samples/s. |
| Time-matched gain ~+1 on Math | **REWORD** | §4.2.3 Table 1 p.8 | Math: baseline 61.69; ProRL +535 steps / 133.8 h → 62.02; BroRL 98.1 h → 62.62; 122.8 h → 62.85; 173.8 h → 63.03. At roughly matched time (122.8 h vs 133.8 h) the gap is **+0.83**. At less time (98.1 h) it is +0.60. The +1.01 gap (63.03 vs 62.02) compares 173.8 h against 133.8 h, so it is not time-matched. |
| Base model | VERIFIED | §4.2.1 p.7 | Continued from the public **ProRLv2 checkpoint** (1.5B; already 3,000 RL steps at 8k context), with context expanded to 16,384, on 64 H100s. |

### 6. AceReason-Nemotron, arXiv 2505.16400v3

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Hard subset 65.9 vs full set 63.3 | VERIFIED; benchmark = **AIME24 avg@64**, 7B, Stage 3 (24K) | §4.3.2 Table 3 p.11 | Full (no filtering, #49K) 63.3 / AIME25 51.1; Easy (#10K) 64.4; Medium (#4.6K) 65.3; **Hard (#2.2K) 65.9** / AIME25 52.5. The starting checkpoint was 62.2. The Hard set excludes prompts the 7B model solves >6 of 16 times. |

### 7. Skywork-OR1, arXiv 2505.22312v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Masking truncated samples → rising clip ratio / length hacking; so they don't mask | VERIFIED; REWORD "length hacking" | §3.2.3 p.11–14; Fig. 7 | With Adv-Mask-Before, "the overall training accuracy continues to decline, and the clip ratio increases steadily. This appears to be a form of reward hacking." (In this paper, clip_ratio is the ratio of truncated responses.) The paper's main stated reason for not masking is different: "applying an advantage mask does not improve test-time scaling behavior" at 32K, while no-mask improves token efficiency. |
| Multi-stage length schedule effects | VERIFIED | §3.2.2 p.10–11 | 7B ablation, 8K → 16K at step 540 vs 16K from scratch: both reach "approximately 60" AIME24. The multi-stage run saves "approximately 100 training hours… over 1000 training steps". Stage I 8K cuts average response length from ~12.5K to 5.4K tokens with comparable AIME24 at 32K. |

### 8. Tricks or Traps, arXiv 2508.08221v3

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Overlong filtering helps clearly at 8k, much less at 20k | VERIFIED | §4.4.1 Takeaway 8, p.15; Fig. 13 | "when the filter threshold is restricted to 8k tokens, substantial benefits are evident… with a longer filter threshold, i.e., 20k tokens, the benefits… diminish significantly." Models: Qwen3-8B-Base / Qwen3-8B. Gains are shown only in curves (FIGURE-ONLY). |

### 9. GLM-4.5, arXiv 2508.06471v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Single-stage 64K beats multi-stage; multi-stage causes irreversible degradation | VERIFIED | §"Single-Stage RL at 64K Output Length", p.8; Fig. 6 | Quote: "This often leads to a significant and irreversible drop in performance, as the model's average output length decreases." The paper attributes this to SFT having conditioned the model on 64K outputs. |

### 10. Polaris blog (hkunlp.github.io/blog/2025/Polaris/)

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Length curriculum hurts Qwen3-4B | VERIFIED | §4 "Is 'Think Shorter, then Longer' necessary?" | "we observe drastic performance drop for Qwen3-4B even at a response length of 24K and response clip ratio of <15%. Such performance degeneration is irreversible". 40K from scratch beats both 24K and 24K→40K. |
| Temperature raised 1.4 → 1.5 during training | VERIFIED | §2 stage table | Polaris-4B: Stage 1/2/3 = 1.4 / 1.45 / 1.5. Polaris-7B: 0.7 / 1.0 / 1.1. |
| > ~1.55 collapses | VERIFIED (Qwen3-4B, AIME24 probe) | §2 "Temperature Zone" | "increasing temperature from 1.55 to 1.7 causes significant model performance collapse, which indicates the PCZone starts from temperature=1.55." The range 1.4–1.55 is the "Controlled Exploration Zone". |
| Difficulty filtering numbers | VERIFIED | §1 | DeepScaleR-40K: removing 8/8 gives **26K**; aggressive filtering (pass > 4/8 removed) gives **19K** and hurts. The final 7B set is **53K** (26K DeepScaleR + 27K AReaL). Qwen3-4B gets a further pass to **30K**. Rate estimates use 8 rollouts at T = 0.6 for the analysis. During training, samples with "Accuracy > 0.9" are removed at the end of each phase. |

### 11. LIMR, arXiv 2502.11886v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| 1,389 samples avg 58.1 vs full 8,523 avg 57.0 | VERIFIED | §3.2 Table 2 p.4 | Qwen-Math-7B (Qwen2.5-Math-7B), PPO. FULL 8,523: AIME24 32.5 / MATH500 76.6 / AMC23 61.9 / **Avg 57.0**. LIMR 1,389: 32.5 / 78.0 / 63.8 / **58.1**. RAND 1,389: 49.4. The average covers only 3 benchmarks. |

### 12. 1-shot RLVR, arXiv 2504.20571v3

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Llama-3.2-3B-Instruct 17.5 → 19.0 on MATH500 | **WRONG metric**: 17.5 → 19.0 is the **6-benchmark average**; MATH500 is **40.8 → 45.8** | §3.3 Table 4 p.8 | Llama row (MATH500 / AIME24 / AMC23 / Minerva / OlympiadBench / AIME25 / Avg): base 40.8 … 17.5; {π1} 45.8 … **19.0**. Full-set DSR-sub gets Avg 19.8, which beats π1; {π1, π13} gets 21.0. |
| Qwen2.5-Math-1.5B entropy-loss-only, 20 steps, MATH500 36.0 → 63.4 | VERIFIED | §4.2 Table 6 p.10 | "+Entropy Loss, 20 steps": M500 63.4, Avg 25.0 (base 36.0 / 17.6). Table 5 row 10 also shows 63.4. |
| Format-only reward 65.0 | VERIFIED (with an alternate value) | Table 6 p.10 | Format Reward M500 **65.0**, Avg 28.7. Table 3 (best checkpoint) and the Table 5 caption give **65.6** for the same format baseline. |

### 13. L1, arXiv 2503.04697v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Beats S1 budget forcing by 20–25 points at 512/1024 tokens | VERIFIED | §4 p.5 | "over 100-150% relative and 20-25% absolute performance gains at both 512 and 1024 token budgets." |
| AIME25 L1-Exact 22.5 vs DeepScaleR 27.3 | VERIFIED (specify DeepScaleR-**24K**) | App. Table 3 p.16 | L1-Exact 24.8 / **22.5**; DeepScaleR-24K 40.2 / **27.3**; DeepScaleR-4K 25.4 on AIME25. L1-Max gets 21.5. |
| Length error 20–40% out of distribution | VERIFIED | p.8 | "Although OOD datasets exhibit predictably higher errors (20-40%)". In-distribution math is "close to 3%". This is the mean relative error. |

### 14. DAPO, arXiv 2503.14476v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Dynamic sampling reaches the same performance in fewer steps without much wall-clock change | VERIFIED | §3.2 p.6; §4.2 p.9 | Quote (p.9): "the overall training time is not significantly affected… the model's convergence time is even reduced, due to fewer training steps required." p.6: "with dynamic sampling the experiment achieves the same performance faster". Table 1 ablation: +Dynamic Sampling 42 → 50 AIME24. |

---

## Reward signal

### 15. Pitfalls of rule- and model-based verifiers, arXiv 2505.22203v3

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Rule-based precision > 99% | VERIFIED | §3.2 p.3 | "all three verifiers exhibit near-perfect precision (> 99%)" (on DS-R1-Distill-Qwen-32B samples). |
| Average recall 86% | VERIFIED | §1 p.1 | "average recall rate of only 86%, which means 14% of correct responses are classified as incorrect." |
| HF Math-Verify recall 47% on WebInstruct | VERIFIED | §4.3 p.5 | "on WebInstruct-Verified, where the HF rule-based verifier attains only 47% recall, the performance gap widens to 3.6 points." |
| Hybrid 57.3 vs rule-only 55.0 (Qwen2.5-7B, DeepScaleR) | VERIFIED | §4.3 p.5; Table 2 p.6 | "reaching a peak of 57.3 – 2.3 points higher than using the rule-based verifier alone." The policy is Qwen2.5-7B Base. The hybrid = HF verifier + **DS-R1-Distill-Qwen-1.5B** (untrained) as the model-based verifier. |
| Trained 1.5B generative verifier hacked ~iteration 450 with outputs like a single "{" | VERIFIED | §5 p.6–7 | R1-Distill-Verifier-1.5B: "after approximately 450 training iterations, the training reward… diverges significantly from the oracle reward"; the policy outputs "a single simple character (such as '{')". Its best evaluation was only 55.6 vs 55.0. |
| Attack success 27.6–35% | **REWORD** | §6 Table 3 p.8 (rendered) | For R1-Distill-Verifier-1.5B: Adversarial Prefixes 35.0, Answer Explanation 27.6, Empty Symbols 29.5, Gibberish 10.6, Html Markdown 15.5, Prompt Injection 16.1. **27.6–35.0 covers only the top three patterns; the full range is 10.6–35.0.** The prefix rate rose from 21.7 (the untrained base) to 35. |
| xVerify near 0–1% | VERIFIED | Table 3 p.8 | xVerify-0.5B-I 0.0–0.4; xVerify-3B-Ia 0.0–1.1. |

### 16. TinyV, arXiv 2505.14625v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| 38.5% of responses judged wrong in Big-Math-RL-Verified are correct | VERIFIED | §1 p.2; Takeaway 1 p.4 | "among the 226K prompt-response pairs within seemingly incorrect prompt-response pairs (W), Prime Verifier mislabels 87K (38.5%) correct responses as incorrect." The abstract's "over 38% of model-generated responses" is looser wording. |
| ~20% of rollouts per step false negatives | VERIFIED | Takeaway 3 p.5 | "the false-negative ratio remains high at every training step, reaching 20% of rollouts on average." |
| +2.6–2.8 average | VERIFIED (computed from table) | Table 1 p.9 | Qwen2.5-7B Avg 54.46 vs Prime 51.82 (**+2.64**); Qwen2.5-Math-7B 59.97 vs 57.17 (**+2.80**). Most of the gain comes from the authors' own HardVerify-Math set (+10.0 / +6.4). On MATH/AMC/Olympiad the gains are mixed (AMC 43.37 vs 44.58 for 7B). |

### 17. Kimi k1.5, arXiv 2501.12599v4

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Excludes MC, true/false, proof questions; removes prompts guessable within 8 attempts without CoT | VERIFIED | §2.1 p.3 | "we exclude questions that are prone to such errors, such as multiple-choice, true/false, and proof-based questions." "prompt a model to guess potential answers without any CoT… N = 8". The N = 8 rule is stated for "general question-answering tasks". |

### 18. DeepSeek-R1, arXiv 2501.12948v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| No neural RM for reasoning (reason: reward hacking) | VERIFIED | §2.2 p.4 | "we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks… neural reward models are susceptible to reward hacking". |
| Preference rewards only in last 400 of 1,700 steps of stage 2 | VERIFIED | §3 p.8 | Quote: "The stage comprises a total of 1,700 training steps, during which general instruction data and preference-based rewards are incorporated exclusively in the final 400 steps." Text from v2; the Jan-2025 v1 may not contain these numbers. |

### 19. Entropy minimization, arXiv 2505.15134v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Qwen2.5-7B EM-RL 42.9 vs labeled GRPO 42.1 | VERIFIED (math Avg column; EM-RL-**token**) | Table 2 p.6 | Math Avg: GRPO N=4 **42.1**; EM-RL-TOKEN **42.9**. Labeled **RLOO gets 43.9**, which beats EM-RL. EM-RL-sequence gets 41.6. |
| Llama-3.1-8B MATH500 40.6 → 41.0 vs RLOO 52.0 | VERIFIED (Llama-3.1-8B-**Instruct**, "Math" column) | Table 5 left p.9 | 40.6 base; EM-RL-token 41.0; EM-RL-sequence 41.6; EM-FT 42.2; **W/ RLOO 52.0**. |

### 20. TTRL, arXiv 2504.16084v3

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Mistral-Nemo degrades 0.8 → 0 | VERIFIED but **REWORD** | Table 2 p.7 | Mistral-Nemo-Instruct **AIME 2024** 0.8 → 0 (Δ −0.8). On the same model, AMC rose 15.4 → 24.8 and MATH-500 rose 40.8 → 51.0. This is not a collapse, only a near-zero AIME score with one problem fewer. |

### 21. Intuitor, arXiv 2505.19590v5

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Offline self-certainty reward hacked by appending already-solved problems | VERIFIED | §5 p.8; Fig. 7 | "Around the 100th update step, the policy model learns to inflate its self-certainty reward by appending an auxiliary, already-solved problem to its answer". Response length spikes and validation accuracy collapses, while the online annotator prevents this. |

### 22. Rubrics as Rewards, arXiv 2507.17746v2

| Claim | Status | Location | Evidence |
|---|---|---|---|
| HealthBench Likert 23.9 → RaR 35.9 | VERIFIED (scope: HealthBench-1k ablation) | §6 Table 1 p.7 | Simple-Likert 23.9%; Reference-Likert 31.7%; RaR-Implicit-Synthetic 35.9%; RaR-Implicit-Human 34.8%; Expert-Answer-SFT 20.4%. This is the ablation trained on a 3.5k HealthBench subset, not the main Fig. 2 experiment, where the numbers are figure-only. |
| Without reference answers 32.7–34.2 | **REWORD (incomplete range)** | Table 3 p.8; Table 1 p.7 | Table 3 (rubrics generated without reference): GPT-4o 34.2, GPT-4o-mini 32.7, O3-mini 32.4, Qwen-72B 32.7, Qwen-32B 31.1, Qwen-7B 31.9. **The full range is 31.1–34.2.** Table 1's RaR-Implicit-Synthetic-NoRef is 32.0. v1 had the same values on a 0–1 scale. |

### 23. Yue et al., arXiv 2504.13837v5

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Omni-MATH-train pass@1 26.1 → 42.5, pass@256 66.3 → 64.3 | VERIFIED (scope: GRPO **step150 → step450**) | §4.4 p.10; App. Table 4 p.23 | Qwen2.5-7B base pass@1 9.9 / pass@256 **67.2**; step150 26.1 / 66.3; step300 33.6 / 65.3; step450 42.5 / 64.3. The endpoints are training checkpoints, not base → RL. |
| AIME24 k=1024: 13.3% solved only by base, 0% only by RL | VERIFIED | Table 2 p.8 | Base ✓/RL ✗ **13.3%**; ✗/✓ **0.0%**; ✓/✓ 63.3%; ✗/✗ 23.3%. The RL model is SimpleRLZoo; MATH500 at k=128 gives 3.6% / 1.0%. |
| R1-Distill-Qwen-7B above base at all k | VERIFIED (text); curves FIGURE-ONLY | §4.2 p.9; Fig. 7 | "the pass@k curve of the distilled model is consistently and significantly above that of the base model" (base = Qwen2.5-Math-7B). |

### 24. ProRL, arXiv 2505.24864v1

| Claim | Status | Location | Evidence |
|---|---|---|---|
| Reasoning Gym pass@1 4.24 → 59.06 "on tasks where base is weak" | Numbers VERIFIED; **REWORD scope** | Table 3 p.5; App. Table (Avg) | DeepSeek-R1-Distill-Qwen-1.5B 4.24 → Nemotron-Research-Reasoning-Qwen-1.5B **59.06**, which is the overall Reasoning Gym average, not a subset of weak tasks. Pass@1 is estimated from 16 samples. |
| Some math pass@128 decreases | VERIFIED | §"Diminished Reasoning Boundary" p.7 | "In some benchmarks (particularly in the math domain)… Although pass@1 improves, the pass@128 score… often declines." |

### 25. Invisible Leash, arXiv 2507.14843v4

| Claim | Status | Location | Evidence |
|---|---|---|---|
| ProRL-1.5B-v2: 48 expansions vs 175 shrinkages | VERIFIED | Table 1 p.4; §4.2 p.5 | "ProRL-1.5B-v2 loses 175 completions while gaining only 48 (ratio ≈3.6:1)", across 11 benchmarks. |
| AIME24 pass@8192 base 93.3% vs ProRL 83.3% | Numbers VERIFIED; **model is ProRL-1.5B-v1, not v2** | §4.2 p.6; App. Table 6 p.13 | Main text: "base pass@8192 = 93.3% vs. ProRL-1.5B's 83.3%". Table 6 is titled "Support dynamics… of ProRL-1.5B-**v1**". v1 is the paper's main model, and v2 is "additionally include[d]". |

---

## Compact list of problems

**WRONG**
- #1 ScaleRL: the long-context run is **32k**, not 34k (Table 1 "ScaleRL-32k", Cmid 11272; the paper itself says 34k once in A.15).
- #12 1-shot RLVR: Llama-3.2-3B-Instruct **17.5 → 19.0 is the 6-benchmark average**. On MATH500 it is 40.8 → 45.8.

**NOT FOUND**
- None. Every claim maps to a location. One item has no supporting comparison table: 2512.07611 has no DAPO-with-DS row, so its DS-on vs DS-off comparison is FIGURE-ONLY.

**Needs rewording / scoping**
- #1 ScaleRL group size: the A values are 0.585 / 0.610 / 0.590 / 0.595 for G = 8 / 16 / 24 / 32, a spread of 0.025, which is slightly beyond the ±0.02 margin. The truncation threshold is "in the range of 10–15%", not "above".
- #1 ScaleRL zero-variance and NPR: the asymptote gains are FIGURE-ONLY (no numeric A). The LOO tables sit inside a figure image.
- #2 JustRL: there is no KL inconsistency. The real inconsistency is Table 1 "EC ✓ / uses only entropy control" vs Table 2 "Use Entropy Regularization: No".
- #3 OLMo 3: the 62.5% offline filter applies to 7B Think only; the 16K/no-masking gains are for **RL-Zero** 3.0 → 3.1; the loss-variance reduction is qualitative or figure-only.
- #4 2512.07611: single-author, 1.5B study. The DS comparison is FIGURE-ONLY, and the paper's description of DS is inaccurate.
- #5 BroRL: 41% / 62% are fractions of **samples** ("Dynamic Sampling Pass Rate"), not groups. The time-matched Math gain is **+0.6 to +0.8** (+1.0 only when BroRL gets 40 extra hours).
- #7 Skywork-OR1: the paper says "a form of reward hacking", not "length hacking". Its main reason for not masking is no gain at 32K plus better token efficiency.
- #12 1-shot RLVR: the format baseline is 65.0 in Table 6 but 65.6 in Table 3.
- #13 L1: the comparison is with DeepScaleR-**24K** (27.3), not DeepScaleR-4K (25.4).
- #15 Pitfalls: the 27.6–35.0% range covers only the top 3 attack patterns; the full range is 10.6–35.0% for R1-Distill-Verifier-1.5B.
- #16 TinyV: the +2.6/+2.8 average is driven by the authors' own HardVerify-Math benchmark.
- #19 EM: labeled RLOO (43.9) beats EM-RL-token (42.9) on Qwen2.5-7B, so "EM-RL beats labeled RL" holds only against GRPO.
- #20 TTRL: Mistral-Nemo drops on AIME24 only (0.8 → 0), while AMC and MATH-500 improve by 9–10 points. It is not a collapse.
- #22 RaR: the no-reference range is **31.1–34.2** (Table 3) or 32.0 (Table 1), not 32.7–34.2. The 23.9 → 35.9 numbers come from the HealthBench-1k ablation.
- #23 Yue: 26.1 → 42.5 and 66.3 → 64.3 are GRPO step150 → step450; the base is 9.9 / 67.2.
- #24 ProRL: 4.24 → 59.06 is the overall Reasoning Gym average, not a subset of weak tasks.
- #25 Invisible Leash: the AIME24 93.3% vs 83.3% figure is for **ProRL-1.5B-v1**; the 48/175 counts are for v2.
