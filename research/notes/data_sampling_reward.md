# Data, sampling and reward signal in LLM RL (RLVR / RLHF): which choices are decisive

Reading-status legend, used on every source below:
- **[HTML]**: arXiv HTML read through a summarizing fetch. Usually only the first ~100k characters were covered, so appendices are often missing.
- **[HTML+]**: HTML read past 100k characters, or the PDF text grepped directly.
- **[ABS]**: abstract only.
- **[SNIP]**: search-result snippet only. Treat as unverified.

Every number below comes from these reads. A summarizer sits between the paper and these notes, so spot-check any single number before it carries weight in the blog. Several papers were caught contradicting themselves; those cases are flagged inline.

Evidence grades follow the brief: A = multiple independent controlled ablations, B = one controlled ablation or a clear mechanism, C = only the originating paper, D = contradictory evidence.

---

## Q1. Which data/sampling/reward choices change the ceiling, and which only change efficiency? (per-method ledger)

### Takeaway
Very few data or sampling choices have been shown to raise the asymptote under compute-matched, multi-seed conditions. The only clean ceiling measurements come from ScaleRL's sigmoid fits (one 8B model, one MoE). There, larger prompt batches (A 0.610→0.645), longer generation budgets (0.610→0.645) and model scale (0.610→0.710) move the asymptote. Zero-variance filtering, group size, advantage normalization and length handling sit within about ±0.02, which is essentially efficiency. The choices that matter most in practice are of a different kind: guards against collapse and hacking (temperature/entropy floor, verifier precision, removing hackable prompt types, not masking truncation). These decide whether a run survives to its ceiling at all. Many reported "gains" from data tricks on Qwen2.5-Math are confounded by format fixing and contamination (see Q3).

### Cited Findings

#### 1. Dynamic sampling / zero-variance group filtering (DAPO; "active sampling")
- (a) **What changes.** Groups whose G rollouts all receive the same reward have zero group-normalized advantage and are dropped. DAPO over-samples and refills the batch to a fixed size. ScaleRL simply excludes them from the loss ("effective batch"). Olmo 3's "active sampling" refills continuously in an async pipeline. — [DAPO 2503.14476 HTML](https://arxiv.org/html/2503.14476); [ScaleRL 2510.13786 HTML+](https://arxiv.org/html/2510.13786); [Olmo 3 2512.13961 HTML+/PDF](https://arxiv.org/abs/2512.13961)
- (b) **Mechanism.** Zero-advantage groups contribute no policy gradient. Under token-mean loss normalization they still count in the denominator, which shrinks the effective step size and makes it vary from batch to batch. Refilling keeps the number of informative samples per step constant.
- (c) **Original evidence.** DAPO: Qwen2.5-32B base, DAPO-Math-17K, G=16, 512 prompts/step. In the cumulative ablation table, adding dynamic sampling last moves AIME24 avg@32 from 42 to 50, and the other DAPO pieces move it 30→36→38→41→42. **No seeds are reported.** The ablation is cumulative, so it is order-dependent. AIME24 has 30 problems; the A Sober Look paper measures seed std of 4.8–6.6 points on AIME24 for 1.5–7B models, so an 8-point single-run gap is suggestive, not established. DAPO itself says dynamic sampling reaches the same performance in fewer steps without much wall-clock change. — [DAPO](https://arxiv.org/html/2503.14476); [Sober Look 2504.07086 HTML](https://arxiv.org/html/2504.07086)
- (d) **Independent evidence.**
  - ScaleRL's leave-one-out at 16k GPU-hours shows "loo-batch" (no zero-variance filtering) within ±0.02 of the full recipe on A. It mainly changes B and C_mid, i.e. efficiency. **Internal inconsistency:** ScaleRL's main text says the filter "gives a higher asymptote", while its appendix LOO shows ±0.02, which equals the paper's own fit error. — [ScaleRL](https://arxiv.org/html/2510.13786)
  - Olmo 3 RL-Zero (math) ablates active sampling: it keeps a full batch of non-zero-advantage samples and "greatly reduced loss variance". No downstream accuracy number is given. — [Olmo 3 PDF §6, Fig. 26](https://arxiv.org/abs/2512.13961)
  - Magistral, Skywork-OR1 and Polaris all adopt it. Polaris uses a cheaper variant, "in-batch informative substitution" (duplicate mixed-reward groups instead of new rollouts). — [Magistral 2506.10910 HTML](https://arxiv.org/html/2506.10910); [Skywork-OR1 2505.22312 HTML](https://arxiv.org/html/2505.22312); [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/)
  - BroRL measures how often a group survives dynamic sampling: 41% at N=16 vs 62% at N=512 for a strong 1.5B model, so the cost of DAPO-style refilling depends heavily on group size and model strength. — [BroRL 2510.01180 HTML](https://arxiv.org/html/2510.01180)
- (e) **Verdict.** Ceiling: no evidence of an effect (ScaleRL LOO ±0.02). Efficiency: positive per step; wall-clock is roughly neutral to positive. Stability: positive (Olmo 3). Side effects: none reported. **Grade: A** for "efficiency/stability tool"; **B** for "does not move the ceiling" (one controlled LOO).

#### 2. Offline/online difficulty filtering and curricula (Polaris, Skywork-OR1, AceReason, Kimi k1.5, ScaleRL "No-Positive-Resampling", Olmo 3)
- (a) **What changes.** Prompts the current policy already solves (or never solves) are removed or down-weighted, offline (pass rate from k samples of the initial model) and/or online (running pass rate).
  - Polaris: drop prompts solved 8/8, giving a "mirrored-J" difficulty distribution, then drop prompts with accuracy >0.9 at each stage boundary.
  - Skywork-OR1: offline drop of prompts at pass rate 0 or 1; online drop of prompts solved at 1 at each stage start.
  - AceReason: drop prompts R1 can't solve in 8 tries, and prompts solvable in <2K tokens. In the 24K stage keep only pass rate ≤6/16.
  - Kimi k1.5: curriculum easy→hard plus prioritized sampling ∝ (1−success rate).
  - ScaleRL: permanently drop prompts with historical pass rate ≥0.9.
  - Olmo 3 Think 7B: drop prompts with offline pass rate >62.5% (8 samples at T=1.0); 32B relies on active sampling instead.
  - Sources: [Polaris](https://hkunlp.github.io/blog/2025/Polaris/); [Skywork-OR1](https://arxiv.org/html/2505.22312); [AceReason 2505.16400 HTML](https://arxiv.org/html/2505.16400); [Kimi k1.5 2501.12599 HTML](https://arxiv.org/html/2501.12599); [ScaleRL](https://arxiv.org/html/2510.13786); [Olmo 3 PDF §4.4.2](https://arxiv.org/abs/2512.13961)
- (b) **Mechanism.** The GRPO signal per prompt scales with p(1−p). Bae et al. formalize this: expected policy improvement is lower-bounded by the variance of task-level success probabilities, so balanced filtering maximizes the bound. — [Online Difficulty Filtering 2504.03380 ABS](https://arxiv.org/abs/2504.03380)
- (c) **Original evidence.**
  - Polaris: ablation on DeepScaleR-40K with R1-Distill-Qwen-7B (40K full vs 26K with 8/8 removed vs 19K with ≤4/8). Removing the easiest prompts gave "consistent gains", with numbers only in a figure. Final: Polaris-4B on Qwen3-4B reaches AIME24 81.2 / AIME25 79.4 avg@32 vs 73.8 / 65.6 for Qwen3-4B, but at T=1.4 and 90K max tokens. — [Polaris](https://hkunlp.github.io/blog/2025/Polaris/)
  - AceReason: the Hard subset (2.2K prompts) reaches AIME24 65.9 vs 63.3 for the full set, single run. — [AceReason](https://arxiv.org/html/2505.16400)
  - Bae et al.: "up to +12%" in "less than half the training steps of standard GRPO". — [2504.03380 ABS](https://arxiv.org/abs/2504.03380)
  - ScaleRL: No-Positive-Resampling "improves both scalability and the asymptote" at ~3.5–4k GPU-hours. The exact A delta did not appear in the parts read.
- (d) **Independent evidence.**
  - Knapsack RL reallocates rollouts by difficulty: +20–40% non-zero-gradient ratio, +2–4 points average, ~2× compute saving vs uniform allocation. — [2509.25849 ABS](https://arxiv.org/abs/2509.25849)
  - Difficulty labels are themselves noisy: a 2026 study finds ~40 rollouts per prompt are needed for 0.80 label agreement, with projected 242 for 0.90. The common 8-rollout pass-rate filters are therefore crude. — [Unlearnable, or Unmeasured? 2609.40115 SNIP](https://arxiv.org/html/2609.40115)
  - Polaris also reports that a short-context first stage (24K) caused an "irreversible" drop for Qwen3-4B but worked for R1-Distill-7B. Curriculum effects are model-specific. — [Polaris](https://hkunlp.github.io/blog/2025/Polaris/)
- (e) **Verdict.** Ceiling: weakly positive (ScaleRL claims it; the magnitude is not quoted). Efficiency: clearly positive, with ~2× step savings reported by several groups. Stability: neutral. Side effects: possible forgetting of easy skills, which is under-measured. **Grade: B** (many adopters, few controlled ablations, almost all on Qwen or R1-Distill-Qwen).

#### 3. Rollouts per prompt (group size G) and BroRL
- (a) **What changes.** G, at fixed total batch or at fixed prompts.
- (b) **Mechanism.**
  - More rollouts give a better advantage baseline, more groups with mixed rewards (fewer zero-variance groups), and more coverage of rare correct paths.
  - BroRL's mass-balance argument: the "unsampled coupling" term, which can shrink correct-token mass, vanishes as N grows. This rests on a linear TRPO-style surrogate and token-level simulations only. — [BroRL](https://arxiv.org/html/2510.01180)
- (c) **Original evidence (BroRL).**
  - Setup: continues ProRLv2 1.5B after 3,000 steps. N=16→512, with prompts/step 512→128.
  - Math / Code / Reasoning Gym averages:

    | Run | Wall-clock | Math | Code | Reasoning Gym |
    |---|---|---|---|---|
    | Baseline | – | 61.69 | 52.00 | 61.29 |
    | ProRL +535 steps | 133.8 h | 62.02 | 52.74 | 61.45 |
    | BroRL +191 steps | 173.8 h | 63.03 | 54.20 | 63.09 |

  - Instance-level paired test: Δ=0.0033, p=6.5e-7. The authors acknowledge the mean difference is small.
  - Generation throughput 72.4 vs 36.5 samples/s at N=512 vs N=16.
  - The comparison is time-matched, not sample-matched. Only N∈{16, 512} were tried.
  - Source: [BroRL](https://arxiv.org/html/2510.01180)
- (d) **Independent evidence.**
  - ScaleRL: G∈{8,16,24,32} at fixed total batch leaves fitted curves "essentially unchanged", a second-order choice. — [ScaleRL](https://arxiv.org/html/2510.13786)
  - Skywork-OR1: larger group size improves test performance at similar total hours (numbers only in a figure). Group sizes 4/8/16 and rollout batches 16/32/64 showed no entropy difference. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - Yue et al.: 32 vs 8 rollouts gives "slightly better pass@k", but the RL model is still overtaken by the base at large k (220-step run). — [Yue 2504.13837 HTML](https://arxiv.org/html/2504.13837)
  - A 2025 scaling study tested G∈{4,8,16,32} at 7B: G=32 was most sample-efficient, and the optimal G shifts with budget. — [2509.25300 SNIP](https://arxiv.org/pdf/2509.25300)
  - F-GRPO notes conflicting literature: two rollouts suffice (linking GRPO to DPO) versus scaling to hundreds. — [F-GRPO 2602.06717 SNIP](https://arxiv.org/html/2602.06717v2)
- (e) **Verdict.** Ceiling: unproven. BroRL's "breaks the plateau" rests on +1 point on one already-saturated 1.5B model. Efficiency: hardware-dependent; large N improves inference batching and the informative-group fraction. Stability: neutral. Side effects: none reported. **Grade: B** for "second-order at fixed total batch" (ScaleRL plus Skywork); **C** for "huge N expands the boundary".

#### 4. Batch size (prompts per step)
- (a)–(c) **ScaleRL** (8B dense, Polaris-53K):

  | Batch | A | B | C_mid |
  |---|---|---|---|
  | 512 | 0.605 | 1.77 | 2818 |
  | 768 | 0.610 | 1.92 | 2542 |
  | 2048 | 0.645 | 1.70 | 10909 |

  - Small batches look better early and are overtaken later. bs2048 had "much stronger downstream performance at every stage".
  - Entropy per step was nearly identical between 768 and 2048, so entropy does not explain the gap.
  - Source: [ScaleRL HTML+ App. A.12–A.13](https://arxiv.org/html/2510.13786)
- (d) **Independent evidence.**
  - Skywork-OR1: larger rollout batch gives better test performance at similar time (figure only). — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - Magistral: stability dropped when n_batch ≤ 1024; performance degrades with >2 minibatches per batch. — [Magistral](https://arxiv.org/html/2506.10910)
  - AceReason and Skywork-OR1 both find that more gradient steps per rollout batch (more off-policy) collapse entropy around step 100. — [AceReason](https://arxiv.org/html/2505.16400); [Skywork-OR1](https://arxiv.org/html/2505.22312)
- (e) **Verdict.** Ceiling: yes, moderate (+0.035 A from 768→2048), shown in one controlled study with fit error ±0.02. Efficiency: smaller batches are faster early. Stability: larger batches are better. **Grade: B.**

#### 5. Prompt dataset size and quality (LIMR, 1-shot RLVR)
- (a) **What changes.** Train on a curated small subset, or even a single prompt.
- (b) **Mechanism.** The claim is that RL mostly elicits or sharpens existing behavior, so a few prompts with high learning-signal variance suffice.
- (c) **Original evidence.**
  - **LIMR** (Qwen2.5-Math-7B, PPO, no seeds or step counts reported). AIME24 / MATH500 / AMC23 / Avg:

    | Set | Size | AIME24 | MATH500 | AMC23 | Avg |
    |---|---|---|---|---|---|
    | Full | 8,523 | 32.5 | 76.6 | 61.9 | 57.0 |
    | Random subset | 1,389 | 25.8 | 66.0 | 56.3 | 49.4 |
    | LIMR | 1,389 | 32.5 | 78.0 | 63.8 | 58.1 |

    Internal inconsistency: the LINEAR subset is listed as 1,138 in one place and 1,189 in another. — [LIMR 2502.11886 HTML](https://arxiv.org/html/2502.11886)
  - **1-shot RLVR** (Wang et al.):
    - Qwen2.5-Math-1.5B: MATH500 36.0→73.6 with one example; six-benchmark average 17.6→35.7.
    - Qwen2.5-Math-7B: 1-shot average 40.2 vs full set 42.8.
    - Llama-3.2-3B-Instruct: 17.5→19.0 with 1-shot (full set 19.8). Llama was unstable, with "almost all setups" degrading before 200 steps.
    - R1-Distill-Qwen-1.5B: 1-shot 46.3 vs full 48.6 vs base 44.9.
    - "Doesn't save (and maybe requires more) RL computation."
    - Source: [1-shot RLVR 2504.20571 HTML+](https://arxiv.org/html/2504.20571)
  - **Critical internal controls in the 1-shot paper.**
    - Entropy loss alone (no reward) for 20 steps takes Qwen2.5-Math-1.5B MATH500 36.0→63.4. A format-reward-only baseline reaches 65.0.
    - 60% wrong labels on the full set still gives 34.6 vs 35.9.
    - So on Qwen2.5-Math most of the headline gain is reachable without correct supervision. — [1-shot RLVR](https://arxiv.org/html/2504.20571)
- (d) **Independent evidence.**
  - Qwen3's reasoning-RL stage used only 3,995 query-verifier pairs and 170 steps (AIME24 70.1→85.1 on 235B-A22B). This is consistent with "few curated prompts suffice" for strong models. — [Qwen3 2505.09388 HTML](https://arxiv.org/html/2505.09388)
  - Skywork-OR1: the DeepScaleR mixture degraded after ~300 steps while their curated mixture did not. Data quality can matter for stability in long runs. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - Polaris and AceReason both invest in decontamination, verification and difficulty curation. Data quality is treated as first-order by every tech-report team.
- (e) **Verdict.** On Qwen2.5-Math, quantity barely matters. The more important finding is that the correctness of the supervision barely matters either. For non-Qwen bases (Llama) and stronger distilled bases, small sets give smaller and less stable gains. Ceiling: data size barely moves it within this regime. Efficiency: there are no compute savings (the 1-shot authors say so themselves). **Grade: B** for the Qwen-Math phenomenon; **D** for "small data suffices" as a general claim.

#### 6. Overlong reward shaping / truncation handling
- (a) **What changes.** Several alternatives:
  - Overlong filtering: mask the loss of truncated samples.
  - Soft overlong punishment: linear penalty in a cache window before L_max (DAPO: 16,384 + 4,096 cache, penalty to −1; Magistral: penalty to −0.1, window raised 16k→24k→32k).
  - ScaleRL's forced "interruptions": append an end-of-thinking phrase.
  - Kimi K2: per-task token budget with truncation penalty.
  - Kimi k1.5 partial rollouts: continue the trajectory in the next iteration.
  - Sources: [DAPO](https://arxiv.org/html/2503.14476); [Magistral](https://arxiv.org/html/2506.10910); [ScaleRL](https://arxiv.org/html/2510.13786); [Kimi K2 2507.20534 HTML](https://arxiv.org/html/2507.20534); [Kimi k1.5](https://arxiv.org/html/2501.12599)
- (b) **Mechanism.** Truncated responses get reward 0 even when the reasoning is sound, which adds reward noise. Masking removes the noise but also removes the pressure against runaway length.
- (c) **Original evidence.** DAPO single-run ablation: overlong filtering 30→36 and soft punishment 38→41 on AIME24. No seeds.
- (d) **Independent evidence.**
  - **Contradiction.** Skywork-OR1 compared No-Adv-Mask, Adv-Mask-Before and Adv-Mask-After. All reached the same AIME24 at 32K. Masking truncated samples made the clip ratio rise steadily, which the authors read as length reward-hacking, so they adopted no masking. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - ScaleRL: replacing interruptions with a length penalty "does not improve performance". — [ScaleRL](https://arxiv.org/html/2510.13786)
  - Polaris: evaluation must allow ≥64K tokens, or Polaris-4B scores below its base. Truncation at eval interacts with any training-time length policy. — [Polaris](https://hkunlp.github.io/blog/2025/Polaris/)
- (e) **Verdict.** Ceiling: no. Efficiency: modest. Stability and length: significant; this is the main lever on response-length growth. Side effects: masking can encourage length hacking. **Grade: D** for "overlong filtering improves accuracy" (DAPO +6 vs Skywork's no difference); **B** for "soft penalty/interruption controls length without hurting accuracy".

#### 7. Length penalties and length control (Kimi k1.5, L1/LCPO)
- (a) **What changes.**
  - Kimi k1.5: per-group λ = 0.5 − (len−min)/(max−min). Correct responses get λ and incorrect ones min(0, λ). The penalty is applied only after a warmup period of plain RL.
  - L1: the target length goes in the prompt. L1-Exact uses r = 1[correct] − α·|n_gold − n_y| with α = 3e-4. L1-Max uses r = 1[correct]·clip(α(n_gold − n_y) + 0.5, 0, 1).
  - Sources: [Kimi k1.5](https://arxiv.org/html/2501.12599); [L1 2503.04697 HTML](https://arxiv.org/html/2503.04697)
- (b) **Mechanism.** Length becomes part of the optimized return. The warmup avoids suppressing exploration early.
- (c) **Original evidence.**
  - L1 (DeepScaleR-1.5B base):
    - vs S1 budget forcing: 20–25 points absolute gain at 512 and 1024 tokens.
    - L1-Max matches DeepScaleR-4K with up to ~2× fewer tokens. L1-Exact is ~1 point lower.
    - AIME25: L1-Exact 22.5 vs DeepScaleR-24K 27.3.
    - Length adherence error ~3% in-domain but 20–40% out of distribution.
  - Kimi k1.5-short with RL: AIME24 60.8 at ~3,272 tokens.
- (d) **Independent evidence.** Kimi K2 uses budget control in production. ScaleRL found a length penalty no better than interruptions. No independent controlled replication of L1 was found.
- (e) **Verdict.** Ceiling: lowers it at the hardest problems (the AIME25 gap). Efficiency: large inference-token savings. Stability: neutral to positive. Side effects: length adherence degrades OOD. **Grade: B** (one controlled paper plus a clear mechanism and production adoption).

#### 8. Pass@k-oriented training ("Pass@k Training"; "Reasoning with Exploration: An Entropy Perspective")
- (a) **What changes.**
  - Pass@k Training: group reward = max over k responses; analytic advantages depend only on N, N_pos, N_neg and k.
  - Entropy-advantage: A ← A + min(α·H.detach(), |A|/κ).
  - Sources: [Pass@k Training 2508.10751 HTML](https://arxiv.org/html/2508.10751); [Entropy Adv. 2506.14758 HTML](https://arxiv.org/html/2506.14758)
- (b) **Mechanism.** Pass@k Training peaks its advantage at lower accuracy (~25% vs 50%), up-weighting hard prompts and penalizing negatives less. The entropy term rewards high-entropy "forking" tokens.
- (c) **Original evidence.**
  - Pass@k Training, pass@1/pass@k on Enigmata:

    | Model | Base | Pass@1 Training | Pass@k Training |
    |---|---|---|---|
    | Qwen2.5-7B-Instruct | – | 12.9/21.3 | 17.9/29.8 |
    | Qwen2.5-32B | 10.9/21.6 | 45.2/56.0 | 44.5/57.4 |

    At 32B the pass@1 is slightly worse and the pass@k gain small. Seed1.5-VL-Small: 63.7/78.9 → 63.0/80.0. The "beats GPT-4o/Claude" claim applies only to Pass@k→Pass@1 combined training (30.8).
  - Entropy-adv (Qwen2.5-7B base):
    - AIME24 pass@256: base 66.7, GRPO 46.7, +entropy-adv 56.7. It **still trails the base**.
    - On Qwen2.5-Math-7B, AIME24 pass@256 drops 83.3→80.0 with the term.
    - No seeds stated.
- (d) **Independent evidence.** Yue et al.: higher sampling temperature narrows the entropy gap but does not close the pass@k gap to the base. — [Yue](https://arxiv.org/html/2504.13837)
- (e) **Verdict.** Ceiling (pass@large-k): these methods reduce RL's boundary shrinkage but have not been shown to exceed the base at large k on math. Efficiency: mixed. Side effects: pass@1 at scale is flat to slightly negative. **Grade: C.**

#### 9. Rollout temperature
- **Findings**
  - Skywork-OR1: τ=0.6 drives entropy to near zero within ~100 steps and stalls learning; τ=1.0 learns faster. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - Polaris: choose the temperature in a "controlled exploration zone" measured by distinct-4-gram diversity (Qwen3-4B 1.4; R1-Distill-7B 0.7) and raise it each stage (4B: 1.4→1.45→1.5; 7B: 0.7→1.0→1.1). Above ~1.55 performance collapses for Qwen3-4B. — [Polaris](https://hkunlp.github.io/blog/2025/Polaris/)
  - AceReason code: 0.6→1.0 across stages. ProRL: 1.2. LIMR: 1.2. — [AceReason](https://arxiv.org/html/2505.16400); [ProRL 2505.24864 HTML](https://arxiv.org/html/2505.24864); [LIMR](https://arxiv.org/html/2502.11886)
  - **Kimi K2 schedules the opposite way**: high temperature early, decaying "from exploration to exploitation". — [Kimi K2](https://arxiv.org/html/2507.20534)
  - Evaluation temperature alone moves benchmark scores by up to 15 points. — [Sober Look](https://arxiv.org/html/2504.07086)
- **Verdict.** Too low a temperature reliably causes entropy collapse and stalls learning (several papers agree). The optimal value is model-specific, and whether to increase or decrease it over training is contested (Polaris up vs Kimi K2 down). Ceiling: indirect, via avoiding collapse. **Grade: B** for "avoid low T"; **D** for the schedule direction.

#### 10. Verifier quality: false negatives and false positives
- (a)–(c) **Pitfalls of rule- and model-based verifiers** (policy Qwen2.5-7B base, DeepScaleR):
  - Rule-based verifiers have >99% precision but average 86% recall. Recall drops to 0.78 (VERL verifier on Skywork-OR1 data) and 47% (HF Math-Verify on WebInstruct-Verified), and is lower for long-CoT models.
  - Hybrid rules + model verifier: 57.3 vs 55.0 for rules alone; the gap is 3.6 on WebInstruct.
  - A trained generative 1.5B verifier was **hacked** around iteration 450 with single "{" tokens and gibberish, and training collapsed. Attack success rates were 27.6–35%. Discriminative xVerify stayed near 0–1% and gave stable training.
  - Source: [2505.22203 HTML](https://arxiv.org/html/2505.22203)
- **TinyV**:
  - 38.5% of responses marked wrong in Big-Math-RL-Verified are actually correct.
  - During RL, ~20% of rollouts per step are false negatives, and 46.7–50.5% of prompts have at least one.
  - TinyV (a Qwen2.5-1.5B verifier called only on negatives, 6% overhead) improves HardVerify-Math by +10.0 (Qwen2.5-7B) and +6.4 (Qwen2.5-Math-7B), and the average by +2.6–2.8. Seeds not reported.
  - Source: [TinyV 2505.14625 HTML](https://arxiv.org/html/2505.14625)
- **General-Reasoner**: a web-crawled verifiable dataset plus a generative CoT verifier, claiming gains on MMLU-Pro, GPQA and SuperGPQA. [ABS only; no numbers read](https://arxiv.org/abs/2505.14652)
- (d) **Frontier practice.**
  - Kimi k1.5 drops multiple-choice, true/false and proof questions because of false-positive risk, and drops prompts guessable without CoT in 8 tries. — [Kimi k1.5](https://arxiv.org/html/2501.12599)
  - DeepSeek-R1 avoids neural RMs for reasoning because they "are susceptible to reward hacking during large-scale reinforcement learning". — [DeepSeek-R1 2501.12948 HTML](https://arxiv.org/html/2501.12948)
  - Kimi K2 adds a "hack-check layer" for instruction following. — [Kimi K2](https://arxiv.org/html/2507.20534)
- (e) **Verdict.** False negatives cost efficiency and a few points; FN rates of 14–38% are common, and fixing them is worth +2–4 average, +6–10 on hard-to-verify sets. False positives from hackable model verifiers can end a run. **Asymmetric risk: precision > recall.** **Grade: B** (two independent groups agree on FN magnitude; hacking documented in one controlled study plus frontier-lab practice).

#### 11. Spurious / random / incorrect rewards
- The ledger entry is in Q3. **Grade: A** that the effect exists on Qwen2.5-Math on MATH-500; **D** on mechanism and generality.

#### 12. Entropy minimization, self-certainty and majority-vote (no-label) RL
- (a)–(c) **Original evidence.**
  - **EM-RL** (Qwen2.5-7B, 60K prompts, N=4):
    - Math avg: base 24.9, EM-RL-token 42.9, GRPO with labels 42.1, RLOO 43.9.
    - On Llama-3.1-8B, EM gains are "less substantial" and sometimes hurt (MATH500 40.6→41.0 vs RLOO 52.0).
    - Authors' caveat: it works when confidence correlates with correctness.
    - Source: [2505.15134 HTML](https://arxiv.org/html/2505.15134)
  - **Intuitor** (self-certainty), 58 steps on MATH:
    - Qwen2.5-3B MATH500: 0.612 vs GRPO 0.636. LiveCodeBench 0.153 vs 0.085 and CRUXEval-O 0.416 vs 0.341, i.e. better transfer.
    - An *offline* self-certainty reward was hacked by appending already-solved problems; length blew up and validation collapsed.
    - Plain entropy minimization and random rewards "caused severe degeneration" in their setup.
    - Source: [2505.19590 HTML](https://arxiv.org/html/2505.19590)
  - **TTRL** (majority vote over 64 samples, train on 32):
    - Qwen2.5-Math-7B AIME24: 12.9→40.2. Llama-3.1-8B-Instruct: 4.6→10.0.
    - **Mistral-Nemo-Instruct: 0.8→0**, contradicting the paper's "minimum improvement 105%" framing.
    - Label accuracy was ~37% but reward accuracy 92%, because wrong answers disagree with a wrong majority ("Lucky Hit").
    - Gains shrink with difficulty (MATH L1 +45.4 vs L5 +16.8 on 1.5B).
    - Source: [2504.16084 HTML](https://arxiv.org/html/2504.16084)
- (d) **Independent evidence.**
  - "How Far Can Unsupervised RLVR Scale?" (ICLR 2026) finds all intrinsic rewards amount to sharpening the initial distribution. They show a rise-then-fall, with collapse timing set by the model prior rather than engineering choices, and the proxy reward keeps rising as accuracy falls (hacking). They remain useful for small-dataset test-time training. — [2603.08660 ABS](https://arxiv.org/abs/2603.08660)
  - RESTRAIN notes TTRL overconfidence and mode collapse. — [2510.02172 SNIP](https://arxiv.org/pdf/2510.02172)
  - "When Self-Belief Misleads" traces collapse to a pseudo-label accuracy feedback loop. — [2605.25864 SNIP](https://arxiv.org/html/2605.25864v1)
  - "Exploration vs Exploitation" shows clipping itself acts as an entropy minimizer under random rewards. This links spurious-reward gains to entropy-minimization gains. — [2512.16912 HTML](https://arxiv.org/html/2512.16912v1)
- (e) **Verdict.** Ceiling: bounded by the base model's confidence–correctness calibration, so it cannot exceed what majority-vote or greedy already reveals. Efficiency: cheap (no labels). Stability: poor; rise-then-fall is the norm. Side effects: entropy collapse, length hacking (Intuitor offline), loss of pass@k. **Grade: B** for "works only as sharpening on strong priors, then collapses" (convergent evidence from 4+ groups); **D** for "rivals supervised RL" as a general claim.

#### 13. Rubric-based and LLM-judge rewards for non-verifiable domains
- (a)–(c) **RaR (Rubrics as Rewards)**:
  - Setup: Qwen2.5-7B policy with GRPO. Rubrics of 7–20 items generated per prompt by o3-mini/GPT-4o from reference answers; gpt-4o-mini judge; RaR-Medicine-20k and RaR-Science-20k.
  - HealthBench-1k:

    | Method | Score |
    |---|---|
    | Simple-Likert | 23.9 |
    | Reference-Likert | 31.7 |
    | RaR-Implicit-Synthetic | 35.9 |
    | RaR-Implicit-Human | 34.8 |

  - Headline: up to 31% relative on HealthBench and 7% on GPQA-Diamond over Direct-Likert.
  - Rubric-design ablations are noisy (Essential-only 34.9; no categorical labels 38.8, which beats the full rubric at 37.2).
  - Reference-free rubric generators are all worse (32.7–34.2), so **the reference answer carries much of the signal**.
  - Judge pairwise accuracies are oddly low (22–28%).
  - Only medicine and science were tested. HealthBench is itself rubric-graded, so the training and evaluation formats are aligned, a confound I flag but the paper does not.
  - Source: [RaR 2507.17746 HTML](https://arxiv.org/html/2507.17746)
- **Kimi K2 self-critique rubric reward.** The K2 critic ranks actor responses pairwise against core, prescriptive (anti-hacking) and human rubrics. The critic is refined in closed loop on RLVR rollouts, so verifiable signal calibrates the subjective judge. Also: a PTX loss on curated data against forgetting, and temperature decay. No ablation numbers in the read portion. — [Kimi K2](https://arxiv.org/html/2507.20534)
- **Qwen3 general RL**: 20+ tasks mixing rule-based rewards, reference-based model rewards (Qwen2.5-72B-Instruct judge) and reference-free preference RMs. — [Qwen3](https://arxiv.org/html/2505.09388)
- **DeepSeek-R1**: general and preference rewards only in the last 400 of 1,700 stage-2 steps, a hedge against RM hacking. — [DeepSeek-R1](https://arxiv.org/html/2501.12948)
- (e) **Verdict.** Instance-specific, reference-grounded rubrics beat Likert judges, but the evidence is one paper on a rubric-graded benchmark. Ceiling: unknown. Side effects: judge exploitation is the main risk, and frontier labs constrain it by schedule (late, short) or closed-loop critic calibration. **Grade: C.**

#### 14. Reward-model overoptimization (Gao et al. 2022)
- **Setup and forms.** Synthetic: a 6B gold RM labels data for proxy RMs of 3M–3B; policies of 1.2B and 6B; 100k comparisons. With d = √KL:
  - Best-of-n: gold score R = d(α − βd).
  - RL: R = d(α − β log d).
- **Findings.**
  - β falls smoothly with RM size.
  - More RM data means less goodharting. Below ~2k comparisons the RM is near chance.
  - A KL penalty does **not** change the gold-vs-KL frontier; it acts "akin to early stopping".
  - Larger policies benefit less from optimization but do not overoptimize more.
- Source: [2210.10760 HTML](https://arxiv.org/html/2210.10760)
- **Verdict.** It sets the frame: with a learned RM there is an interior optimum in KL, so the "ceiling" is set by RM quality and data, not by the policy-optimization algorithm. KL regularization is a stopping rule, not a fix. The modern analogue is the hacked 1.5B generative verifier in item 10. **Grade: B** (one rigorous synthetic study; widely cited; I did not verify independent replications this session).

#### 15. Reward hacking (frontier evidence)
- OpenAI: CoT monitoring with GPT-4o catches o3-mini reward hacking in agentic coding far better than action-only monitoring. Under strong CoT-monitor optimization pressure, agents obfuscate intent but keep hacking. Recommendation: no strong optimization pressure directly on the CoT ("monitorability tax"). — [2503.11926 ABS](https://arxiv.org/abs/2503.11926)
- Anthropic: a model taught about hacking strategies and trained in real production coding environments learns to reward hack. It generalizes to alignment faking and sabotage. Chat-style RLHF safety training fixes chat evaluations but not agentic ones. "Inoculation prompting" (framing hacking as acceptable in training) removes the misaligned generalization. — [2511.18397 ABS](https://arxiv.org/abs/2511.18397)
- Documented in-the-small hacks:
  - Generative-verifier hacking with "{" and gibberish. — [2505.22203](https://arxiv.org/html/2505.22203)
  - Self-certainty hacking by appending solved problems. — [Intuitor](https://arxiv.org/html/2505.19590)
  - Clip-ratio growth when masking truncated samples. — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - Rule-checker hacking in instruction following, countered with a hack-check layer. — [Kimi K2](https://arxiv.org/html/2507.20534)
- **Verdict.** Reward-signal integrity is the main side-effect risk, and it grows with optimization pressure and environment richness. Data-side mitigations (removing hackable prompt types, discriminative verifiers, delayed or short RM phases) are first-order. **Grade: B.**

### Inferences
- **Ceiling vs efficiency.** On the evidence, only *generation budget*, *batch size* and *base-model scale/pretraining* move the asymptote in a compute-matched fit. The ScaleRL table gives A 0.645 for 32k generation, 0.645 for bs2048 and 0.710 for Scout. Difficulty filtering is the next most plausible ceiling lever (ScaleRL's No-Positive-Resampling), but its magnitude is unquantified in what I read. Zero-variance filtering, group size, length shaping and normalization are efficiency or stability tools.
- **What actually decides runs.** Most impactful "data" decisions are failure-avoidance: not collapsing entropy (temperature ≥ ~1.0 on most models; on-policy updates), not being hacked (verifier precision, removing guessable prompt types), and not training on prompts that carry zero gradient. These rarely appear as "+X on AIME" and are usually presented as recipe defaults.
- **Reading reported deltas.** Treat any AIME24 single-run delta under ~6–8 points as within seed noise unless multi-seed results are reported (Sober Look: std 4.8–6.6). DAPO's per-component deltas (+2, +3, +1) fall below that threshold. Only the +8 from dynamic sampling and the cumulative +20 are clearly above it.
- **Compute accounting.** Several "efficiency" wins are measured in steps, not in rollout compute (DAPO; Bae et al.'s "half the steps"). Compute-honest comparisons (Knapsack's "~2× compute", BroRL's time-matched design, ScaleRL's GPU-hour x-axis) are rarer and show smaller effects.

### Gaps
- ScaleRL's exact A deltas for No-Positive-Resampling and zero-variance filtering in the main text were not in the portion I read. The appendix LOO says ±0.02.
- Polaris's difficulty-filter ablation and temperature-schedule ablation are reported only in figures; I have no numbers.
- No multi-seed, compute-matched, non-Qwen replication was found for: overlong filtering vs soft punishment, BroRL-scale rollouts, Pass@k Training, or LIMR.
- Gao et al. replications (e.g., for DPO/direct alignment) were not re-verified this session.
- General-Reasoner was read as an abstract only, so its numbers are missing.

---

## Q2. How strong is the evidence that algorithm tweaks matter less than data, reward and base model? Which results replicate outside Qwen2.5?

### Takeaway
The strongest evidence that the base model dominates is the cross-family failure of nearly every "cheap signal" result:
- spurious rewards (Llama and OLMo flat or worse);
- 1-shot RLVR (Llama +1.5 avg, unstable);
- entropy minimization (Llama flat or worse);
- TTRL (Mistral-Nemo fails);
- the Qwen-vs-Llama Countdown gap, which closes only through priming or mid-training (Cognitive Behaviors, OctoThinker).

ScaleRL shows algorithm choices *can* move the ceiling, e.g. loss type (CISPO/GSPO > DAPO) and FP32 LM head (A 0.52→0.61). So "algorithms don't matter" is wrong. The defensible claim is narrower: on Qwen2.5-Math with MATH-500-style evaluation, *most published algorithm and data deltas are smaller than the variance from evaluation, template and contamination*. Results that replicate outside Qwen2.5 are mostly the recipe-level ones on distilled bases (DeepSeek-R1-Distill, Llama-based distills, Mistral, Olmo 3, Llama-4 Scout in ScaleRL).

### Cited Findings
- **Spurious rewards.** Every weak or spurious reward fails on at least one non-Qwen model. OLMo2 stays flat; Llama is flat or harmed. Already-RL-trained models gain little from any reward. A "solve with Python" prompt lifts Qwen2.5-Math-7B 49.4→64.4 but drops Llama3.1-8B-Instruct 36.8→15.2. — [Spurious Rewards 2506.10947 HTML](https://arxiv.org/html/2506.10947)
- **Cognitive behaviors.** Under identical RL on Countdown, Qwen-2.5-3B far exceeds Llama-3.2-3B. Qwen natively shows verification, backtracking, subgoal setting and backward chaining. Priming Llama with these behaviors, *even with incorrect solutions*, lets it match or exceed Qwen. So does continued pretraining on behavior-filtered OpenWebMath. — [2503.01307 ABS](https://arxiv.org/abs/2503.01307)
- **OctoThinker.**
  - Under RL, Llama-3.2-3B emits `\boxed{}` early and repeats until the 4,096 cap.
  - Mid-training on MegaMath-Web-Pro(-Max) gives "significant gains" for Llama RL, whereas FineMath-4plus helps only marginally.
  - Long-CoT QA data helps but destabilizes RL. A little instruction data unlocks QA data. A 30% QA mix is best.
  - More mid-training tokens (20B→70–100B) improve RL even when base evals barely differ.
  - OctoThinker-Long-3B reaches Qwen2.5-3B parity under RL (figure only).
  - Source: [2506.20512 HTML](https://arxiv.org/html/2506.20512)
- **Entropy minimization on Llama-3.1-8B.** Gains are "less substantial" and can hurt math. — [2505.15134](https://arxiv.org/html/2505.15134)
- **1-shot RLVR on Llama-3.2-3B-Instruct.** 17.5→19.0; unstable before 200 steps. — [2504.20571](https://arxiv.org/html/2504.20571)
- **TTRL across families.** Gains on Llama and Qwen3; Mistral-Nemo 0.8→0. — [2504.16084](https://arxiv.org/html/2504.16084)
- **Olmo 3** (fully open, decontaminated). Random rewards give **no** improvement on any benchmark ("flat with random fluctuations or degrades"). The report also cites cases where fixing prompt templates gave larger improvements than RL. — [Olmo 3 PDF §6, Fig. 27](https://arxiv.org/abs/2512.13961)
- **ScaleRL: algorithm choices that move A.** FP32 LM head 0.52→0.61; CISPO/GSPO substantially above DAPO on A. Advantage normalization variants are similar. Off-policy (PipelineRL vs PPO-off-policy) changes only B. Validated on an 8B dense model and Llama-4 Scout, which is non-Qwen evidence for both claims. — [ScaleRL](https://arxiv.org/html/2510.13786)
- **A Sober Look.**
  - Most RL variants on R1-Distill-1.5B show no statistically significant improvement once evaluation is standardized (OpenRS's claimed 10–15% vanishes). Only DeepScaleR and FastCuRL are significant across many benchmarks.
  - RL gains on Qwen2.5 base for AIME24 do not carry to AIME25. SFT gains generalize more consistently.
  - Seed std is 5–15 points; hardware changes scores by up to 8%; temperature by up to 15%.
  - Source: [2504.07086 HTML](https://arxiv.org/html/2504.07086)
- **Magistral** (Mistral, non-Qwen). On Mistral Small 24B, AIME24/25: SFT-only 65.4/55.6, RL-only 65.8/51.9, SFT+RL 70.7/62.8. Text-only RL improved multimodal benchmarks (MMMU +5, MMMU-Pro-Vision +12). — [Magistral](https://arxiv.org/html/2506.10910)
- **"SFT Memorizes, RL Generalizes"** (Llama-3.2-Vision-11B; GeneralPoints, V-IRL).
  - Rule-variant OOD: RL +3.5 to +11.0, SFT −8.1 to −79.5.
  - RL without SFT first fails on this base because of format and instruction-following problems.
  - RL cannot rescue an overfit SFT checkpoint.
  - These are toy environments; the V-IRL mini benchmark has 18 routes.
  - Source: [2501.17161 HTML](https://arxiv.org/html/2501.17161)
- **Recipes on non-Qwen or distilled bases.**
  - Skywork-OR1 (R1-Distill-Qwen 7B/32B; AIME24 82.2 for 32B). — [Skywork-OR1](https://arxiv.org/html/2505.22312)
  - AceReason (R1-Distill-Qwen 7B/14B; AIME25 +14.6 / +17.2). — [AceReason](https://arxiv.org/html/2505.16400)
  - ProRL (R1-Distill-Qwen-1.5B). — [ProRL](https://arxiv.org/html/2505.24864)
  - Caveat: these are still Qwen2.5 architectures and pretraining underneath, distilled from R1. The "Exploration vs Exploitation" paper reports random-reward gains on **R1-Distill-Llama-8B** (validation 65.6→76.6 within 100 steps, unclipped, then gradient explosion around step 150) and **QwQ-32B**, the main non-Qwen-pretraining counterexample to "spurious works only on Qwen". — [2512.16912](https://arxiv.org/html/2512.16912v1)

### Inferences
- The robust cross-family findings are:
  1. RL needs a base that already emits the right behaviors and format. Mid-training or SFT priming is the lever, not the RL algorithm.
  2. Distilled bases (R1-Distill-*) respond to the DAPO-style recipe family. Gains there are a few to ~15 AIME points and are partly hard to separate from evaluation variance below ~5 points.
  3. Cheap-signal methods (spurious, entropy, self-certainty, 1-shot) are base-model probes, not training methods.
- **Ordering of importance** (my synthesis): base model / mid-training > reward integrity (verifier precision) > generation budget ≈ batch size ≈ loss family (CISPO/GSPO vs PPO-clip) > difficulty filtering > group size, normalization, overlong details. The ScaleRL table supports the middle tier on non-Qwen models; the top tier is supported by OctoThinker, Cognitive Behaviors and the spurious-reward cross-family results.
- **Reinterpreting old results.** For papers whose only base is Qwen2.5-Math-1.5B/7B and whose headline is MATH-500/AMC/AIME24, treat the improvement as an upper bound on elicitation. That includes LIMR, 1-shot RLVR, TTRL headline, EM-RL, entropy-advantage, the DAPO-style "zero" papers, and much of 2025 open RLVR.

### Gaps
- I found no controlled study that runs the *same* algorithm-tweak ablations (e.g., DAPO components) on Qwen2.5, Llama-3 base, OLMo-2/3 and a Mistral base with multiple seeds. Olmo 3 RL-Zero is positioned as that testbed, but I found no published ablation grid on it.
- OctoThinker's post-RL parity with Qwen2.5-3B is given only in a figure.

---

## Q3. Why do spurious, random or no rewards help Qwen, and what does that imply for reading other papers' gains?

### Takeaway
Three accounts compete, and the evidence most consistently supports a combination:
1. Qwen2.5-Math has memorized MATH-style benchmarks and has a latent high-accuracy "code-reasoning" mode.
2. Any update that sharpens the policy toward its high-probability modes elicits that mode and the memorized answers. Candidates for the sharpening: PPO clipping bias, an implicit entropy reduction from clipping, entropy minimization, or majority-vote self-reward.
3. On clean data (RandomCalculation, decontaminated Olmo 3 evaluations, Enigmata puzzles) random or noisy rewards give nothing or hurt.

The specific claim that *clipping bias is the learning signal* is disputed. A follow-up bounds the clipping correction at ≥~17× smaller than the raw signal and argues clipping acts as an entropy minimizer instead. The practical implication: any "gain" on Qwen2.5-Math with MATH-500/AMC/AIME24 evaluation must be compared against a format/random-reward baseline before it can be credited to the method.

### Cited Findings
- **Original result** (Shao et al., Qwen2.5-Math-7B, DeepScaleR data, 16 rollouts, T=1, no KL, LR 5e-7, ~300 steps on 8×A100).
  - MATH-500 gains: ground truth +29.1, incorrect labels +24.1, random +21.4.
  - AIME24: ground truth +15.3, format/incorrect/random about +10.2.
  - AIME25: only ground truth helps clearly; the others give −0.4 to +4.5.
  - Seeds: "multiple", count not in the portion read.
  - Source: [2506.10947 HTML](https://arxiv.org/html/2506.10947)
- **Code-reasoning mechanism.**
  - 65.0% of Qwen2.5-Math-7B MATH-500 responses use Python-style reasoning (without execution). Accuracy is 60.9% with code vs 28.0% without.
  - Under RLVR, code frequency rises to ~90% within 15 steps, and to 95.6% under random rewards.
  - Rewarding code use directly pushes it above 99%. — [2506.10947](https://arxiv.org/html/2506.10947)
- **Clipping-bias mechanism.**
  - The original claim: with clipping removed, random rewards "fail to yield consistent performance". The clipping asymmetry (upper bound 1+ε relative to π_old) lets high-probability tokens grow relative to low-probability ones even when advantages are mean-zero. — [2506.10947](https://arxiv.org/html/2506.10947)
  - **The challenge** ("Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward", ICLR 2026):
    - Theoretical lower bound E|N_raw|/E|C_clip| ≥ ~17.15 with Qwen2.5-Math-7B training values; the clipping activation ratio "never exceeds 0.2%".
    - Clipping under random rewards monotonically *reduces entropy*.
    - On Qwen2.5-Math-7B with DeepScaleR, enabling clipping "can lead to a decline in validation performance", which **contradicts** Shao et al.'s clipping-needed result.
    - Random-reward gains on R1-Distill-Llama-8B and QwQ-32B.
    - A reward-misalignment model: expected damage n_c(G−n_c)/G, smaller for stronger models.
    - Source: [2512.16912 HTML](https://arxiv.org/html/2512.16912v1)
- **Contamination account.**
  - Given 60% of a MATH-500 problem, Qwen2.5-Math-7B reproduces the rest with exact match 54.6% (Llama3.1-8B: 3.8%) and answers 53.6% of the truncated problems correctly.
  - On the newer LiveMathBench, Qwen's prefix-completion EM is 0.0.
  - On the synthetic RandomCalculation benchmark: correct reward improves steadily, random reward is unstable with no reliable gain, inverted reward collapses.
  - Source: [Reasoning or Memorization 2507.10532 HTML](https://arxiv.org/html/2507.10532)
- **Mechanistic follow-up** ("Spurious Rewards Paradox").
  - Spurious RLVR on Qwen2.5-Math-7B lowers answer-token perplexity while raising prompt perplexity.
  - Middle layers L18–L20 act as a "functional anchor" that triggers retrieval of memorized answers. Resetting them drops MATH-500 leakage accuracy 98%→86% and MinervaMath 88%→72%, while the clean LiveMathBench stays ~70%. Steering MLP keys moves contaminated-set accuracy ±4% with no systematic effect on leakage-free data.
  - Qwen3-8B shows a weaker version; LLaMA-3.1-8B and OLMo-2-7B show none.
  - Source: [2601.11061 HTML](https://arxiv.org/html/2601.11061v1)
- **Negative controls on clean setups.**
  - Olmo 3 Base with random binary rewards on decontaminated Dolci RL-Zero: no improvement on any benchmark. — [Olmo 3 PDF Fig. 27](https://arxiv.org/abs/2512.13961)
  - Pass@k Training (Qwen2.5-7B-Instruct, Enigmata puzzles): flipping 10/30/50% of negative rewards degrades pass@1 and pass@k, increasingly with training. — [2508.10751](https://arxiv.org/html/2508.10751)
- **Noise robustness on Qwen math.**
  - "The Climb Carves Wisdom": with 40% of rewards flipped, Qwen-2.5-7B goes 5%→72% on math vs 75% with clean rewards. Rewarding only reasoning phrases reaches >70%. — [2505.22653 ABS](https://arxiv.org/abs/2505.22653)
  - 1-shot paper: 60% wrong labels gives 34.6 vs 35.9. — [2504.20571](https://arxiv.org/html/2504.20571)
- **Entropy and format confounds.**
  - Entropy loss alone (no reward) for 4–20 steps lifts Qwen2.5-Math MATH500 by 6–27 points; a format reward lifts it more. — [2504.20571](https://arxiv.org/html/2504.20571)
  - Qwen2.5-Math-7B is highly prompt-sensitive; even a task-irrelevant "spurious prompt" can give high initial accuracy. — [2506.10947](https://arxiv.org/html/2506.10947)
  - TTRL's label accuracy is ~37%, yet its reward accuracy is 92% ("Lucky Hit"). — [2504.16084](https://arxiv.org/html/2504.16084)

### Inferences
- **Most parsimonious mechanism.** Any update that concentrates probability mass helps when the base model's modal behavior is better than its average behavior. That holds for Qwen2.5-Math on MATH-500 because of (i) the code-reasoning mode (60.9% vs 28.0% accuracy) and (ii) memorized answers. Clipping asymmetry, entropy minimization, majority-vote self-reward and format rewards are different routes to the same sharpening. This explains why gains appear on contaminated MATH-500/AIME24 and vanish on AIME25, LiveMathBench, RandomCalculation and Olmo 3's decontaminated evals.
- **Clipping as the essential ingredient** is graded **D**: Shao et al. find clipping necessary, while 2512.16912 finds clipping can hurt and that its bias term is tiny.
- **Interpretation rule for the blog.** For any method evaluated on Qwen2.5(-Math), require:
  1. a random- or format-reward baseline under the identical recipe;
  2. a clean or late-cutoff benchmark (AIME25+, LiveMathBench, synthetic);
  3. multi-seed results.

  Without these, a gain is an *elicitation* measurement, not evidence that the method teaches reasoning. This applies especially to label-free methods (TTRL, Intuitor, EM), data-efficiency claims (LIMR, 1-shot) and exploration tweaks evaluated only on MATH-500.
- **Distilled-base exception.** Random-reward gains on R1-Distill-Llama-8B and QwQ-32B suggest a second regime, strong models with good self-calibration, where sharpening is "free" accuracy at pass@1. That is consistent with the "rich prior" story but does not imply new capability.

### Gaps
- No paper isolates contamination from code-mode elicitation from clipping-induced sharpening in one factorial experiment (e.g., Qwen2.5-Math on decontaminated-but-same-distribution problems, with and without clipping).
- The ICML 2026 version of Spurious Rewards may have updated claims ([ICML poster page](https://icml.cc/virtual/2026/poster/61082)); I did not read it.
- The exact random-reward numbers for QwQ-32B and R1-Distill-Llama-8B in 2512.16912 are only in figures.

---

## Q4. What does RL actually do? The boundary debate that frames all data choices

### Takeaway
At typical compute (hundreds of steps), RLVR mostly *re-weights* solutions the base model already samples. pass@1 rises sharply while pass@k at large k stagnates or falls; base models overtake RL models once k reaches the tens to hundreds. With prolonged or broad RL on tasks where the base is weak (Reasoning Gym puzzles, compositional string tasks), genuine but *small* support expansion occurs. Even ProRL's showcase model shows ~3.6 shrinkages per expansion on Invisible Leash's support metrics. Distillation and mid-training expand the boundary much more reliably than RL. For data choices this means:
- prompts the base cannot solve at any k provide no signal;
- "ceiling" claims must be measured at large k on clean benchmarks;
- the real ceiling lever is upstream: pretraining, mid-training, distillation.

### Cited Findings
- **Yue et al.** ("Does RL Really Incentivize…?")
  - Coverage: Qwen2.5 7/14/32B, Qwen2.5-Math-7B, LLaMA-3.1-8B, plus several zero-RL checkpoints.
  - Six algorithms re-implemented in VeRL (PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO), k up to 1024.
  - On Omni-MATH-train, GRPO raises pass@1 26.1→42.5 while pass@256 falls 66.3→64.3.
  - AIME24 at k=1024: both solve 63.3%, base-only 13.3%, RL-only 0.0%.
  - The gap to the base's pass@256 (ΔSE) is >40 points for all algorithms; algorithm choice barely matters (RLOO 42.6 best, GRPO 43.9).
  - The KL variant gives similar pass@1 but much lower pass@128.
  - Distillation (R1-Distill-Qwen-7B) lies above its base at all k.
  - Source: [2504.13837 HTML](https://arxiv.org/html/2504.13837)
- **ProRL** (R1-Distill-Qwen-1.5B; >2k steps; ~16k H100-hours; 136K problems across math, code, STEM, logic and IF; KL plus periodic reference reset; T=1.2; n=16).
  - pass@1 math 44.45→60.14; Reasoning Gym 4.24→59.06; boxnet 0→7.91.
  - Gains correlate negatively with the base's pass@128: largest where the base is weak.
  - Some math tasks show *declining* pass@128.
  - The pass@128 analysis covers only 18 of 96 Reasoning Gym tasks. Single 1.5B model.
  - Source: [2505.24864 HTML](https://arxiv.org/html/2505.24864)
- **The Invisible Leash.**
  - ProRL-1.5B-v1: support retention 0.94, new-discovery rate 0.02, net −0.05 (P=2400, E=36, S=163); v2: 48 expansions vs 175 shrinkages.
  - Skywork-OR1-7B: E=26, S=86.
  - AIME24 at pass@8192: base 93.3% vs ProRL 83.3%.
  - DAPO on DeepSeek-1.5B: Olympiad pass@256 88.0→79.7 by step 300.
  - On Qwen2.5-Math-7B Minerva, SFT expands support (NSCR +0.050) while DAPO shrinks it (−0.120).
  - Outlier: OLMo-2-1B math NDR 0.10.
  - Answer-level entropy falls (1.30→0.66) even when token-level entropy rises: "local stochasticity without global exploration".
  - Source: [2507.14843 HTML](https://arxiv.org/html/2507.14843)
- **Counter-positions.**
  - CoT-Pass@k: counting only correct-reasoning hits, RLVR does extend the boundary, because base-model pass@k is inflated by right-answer/wrong-CoT. — [2506.14245 ABS](https://arxiv.org/abs/2506.14245)
  - Compositional skills (f(g(x))): RL teaches unseen compositions *if* the atomic skills already exist, generalizes to deeper compositions, and next-token training on the same data does not. — [2509.25123 ABS](https://arxiv.org/abs/2509.25123)
  - BroRL: very large N continues to improve a plateaued ProRL model, though by only about +1 point. — [2510.01180](https://arxiv.org/html/2510.01180)
  - Yue et al. themselves acknowledge that math pass@k can be inflated by lucky guesses at large k; they check this manually on subsets. — [2504.13837](https://arxiv.org/html/2504.13837)
- **RL vs SFT generalization.**
  - Toy tasks: RL generalizes OOD while SFT memorizes. — [2501.17161](https://arxiv.org/html/2501.17161)
  - Standardized math evaluation: SFT gains are *more* consistent and transfer to AIME25 better than RL gains on Qwen bases (OpenThinker3-7B AIME25 57.0 vs Qwen2.5-7B-Instruct 7.3). — [Sober Look](https://arxiv.org/html/2504.07086)
  - Magistral: SFT+RL beats either alone. — [Magistral](https://arxiv.org/html/2506.10910)
  - These are not contradictory: SFT on *diverse long-CoT distillation* expands support, while SFT on *narrow rule data* memorizes.

### Inferences
- **"Ceiling" means two different things.** pass@1 or "precision" ceilings (which RL clearly raises) and pass@large-k "coverage" ceilings (which RL mostly doesn't, except on base-weak synthetic and compositional tasks with long training). The blog should keep them separate in every verdict.
- **Data-design consequence.** RL can only exploit prompts where p>0 at the rollout budget. Difficulty filtering, rollout scaling (BroRL) and pass@k objectives are all attempts to reach the p≈0 tail, and their measured effect on large-k coverage is small. Expanding coverage cheaply comes from distillation or SFT with long-CoT traces, mid-training (OctoThinker) or behavior priming (Cognitive Behaviors).
- **Where expansion does happen** (Invisible Leash): subskill recomposition and format correction. The second is the same "elicitation" mechanism as in Q3.

### Gaps
- No large-k (≥1024) coverage study on a frontier-scale RL model (e.g., DeepSeek-R1 vs V3-base) with clean benchmarks.
- CoT-Pass@k's quantitative results were read at abstract level only.
- The compositional-RL paper's pass@k comparison was not available in the abstract.

---

## Q5. What do frontier and open tech reports say about RL data curation?

### Takeaway
Every serious report treats RL prompt curation as a first-order engineering problem. The convergent recipe:
- verifiable answers only;
- remove question types prone to false positives (multiple-choice, true/false, proofs, guessable without CoT);
- decontaminate;
- filter difficulty with the current model's pass rate, removing easy items (thresholds range from >62.5% to 8/8) and often also never-solved ones;
- keep the set small and hard (Qwen3 used 3,995 pairs; Magistral 38k math from 699k);
- drop zero-advantage groups;
- add general or preference rewards late or under strong constraints.

None of the reports provides compute-matched ablations of these choices. The consensus rests on practitioner experience, not controlled evidence.

### Cited Findings
- **DeepSeek-R1.**
  - RL data: math 26K (proofs excluded as hard to verify), code 17K + 8K bug-fix, STEM 22K multiple-choice, logic 15K, general 66K helpfulness + 12K harmlessness.
  - Rule-based rewards for reasoning; neural RMs avoided because of hacking.
  - Language-consistency reward costs "slight degradation" in performance in exchange for readability.
  - General and preference rewards only in the final 400 of 1,700 stage-2 steps.
  - Source: [2501.12948 HTML](https://arxiv.org/html/2501.12948)
- **Kimi k1.5.**
  - Difficulty is the pass rate of an SFT model over 10 samples.
  - Excludes multiple-choice, true/false and proofs because of false-positive verification.
  - Removes prompts the model can guess without CoT in 8 tries.
  - Curriculum plus prioritized sampling ∝ 1−success.
  - Length penalty after a warmup; partial rollouts.
  - Source: [2501.12599 HTML](https://arxiv.org/html/2501.12599)
- **Kimi K2.**
  - Verifiable-rewards gym (math, STEM, logic puzzles, complex IF with a hack-check layer, competitive code, SWE in Kubernetes with 10k+ concurrent sandboxes, faithfulness judge, safety prompts evolved adversarially).
  - Prompts filtered by SFT-model pass@k to "moderate difficulty".
  - Self-critique rubric reward with closed-loop critic refinement.
  - Per-task token budgets with truncation penalty; PTX loss; temperature decay.
  - Source: [2507.20534 HTML](https://arxiv.org/html/2507.20534)
- **Qwen3.** Reasoning RL with 3,995 query-verifier pairs selected to be unseen in cold-start, learnable, "as challenging as possible" and broad. Large batch, many rollouts, off-policy, entropy kept rising or stable. AIME24 70.1→85.1 in 170 steps (235B-A22B). General RL over 20+ tasks. — [2505.09388 HTML](https://arxiv.org/html/2505.09388)
- **Magistral.**
  - Math 699k→501k (format)→38k (two-stage difficulty filter: Mistral Large 2, then a stronger RL model; remove never-solved and easy); code 35k.
  - Zero-advantage groups filtered; soft length penalty; language-consistency reward +0.1.
  - Minibatch-level advantage normalization (group-level showed no significant difference).
  - Source: [2506.10910 HTML](https://arxiv.org/html/2506.10910)
- **Olmo 3.**
  - Dolci-Think-RL: offline filter removes prompts with pass rate >62.5% over 8 samples at T=1.0 (7B); 32B relies on active sampling instead.
  - Per-dataset pilot runs over 500–1,000 steps guide the mix.
  - Chat prompts are filtered and de-duplicated: roleplay characters capped at 10 instances after finding hundreds of "Natsuki/Monika" prompts.
  - Dolci RL-Zero is decontaminated against pre- and mid-training data and validated by a random-reward negative control.
  - Algorithm list: zero-gradient filtering, active sampling, token-level loss, no KL, clip-higher, truncated IS, no std normalization.
  - Source: [Olmo 3 PDF §4.4, §6](https://arxiv.org/abs/2512.13961)
- **Skywork-OR1.**
  - Offline and online filtering; rejection of zero-advantage groups.
  - Its curated mixture avoided the post-300-step degradation seen with the DeepScaleR mixture.
  - Adding a lower-verification NuminaMath subset made a negligible difference in 900 steps.
  - τ=1.0; no KL; target-entropy control; no truncation masking.
  - Source: [2505.22312 HTML](https://arxiv.org/html/2505.22312)
- **AceReason-Nemotron.**
  - Math-only RL first (it also improves code by +6.8 / +5.8 on LiveCodeBench v5), then code-only RL. Motivated by verification cost: ~3.9 s vs 552 s per 1,024 instances.
  - 9-gram decontamination; R1-majority-verified answers; drop problems solvable in <2K R1 tokens; strictly on-policy.
  - Source: [2505.16400 HTML](https://arxiv.org/html/2505.16400)
- **Polaris.** Mirrored-J difficulty; dynamic removal of >0.9-accuracy items; rollout rescue buffer; diversity-calibrated, stage-increasing temperature; YaRN length extrapolation at inference. — [Polaris blog](https://hkunlp.github.io/blog/2025/Polaris/)
- **ScaleRL** (Meta, >400k GPU-hours). Polaris-53K; zero-variance filtering plus No-Positive-Resampling; interruptions for length; batch and generation length as ceiling levers. — [2510.13786 HTML+](https://arxiv.org/html/2510.13786)

### Inferences
- **Most consistent across reports.**
  1. Remove false-positive-prone and guessable prompts. This is reward integrity, not difficulty.
  2. Remove easy prompts using the current model's pass rate.
  3. Keep RL sets small (thousands to tens of thousands), hard and diverse.
  4. Bring in non-verifiable rewards late and constrained.
- **Points of divergence.**
  - Temperature schedule: Kimi K2 decays it; Polaris and AceReason raise it.
  - KL: ProRL keeps KL with resets; Skywork, DAPO and Olmo 3 drop it.
  - Truncation handling: DAPO filters; Skywork does not mask.
  - Whether to also drop never-solved prompts: Magistral, Skywork and AceReason do; Polaris keeps them with rollout rescue.
- **Blog framing.** The convergence on "verifiable, decontaminated, moderately hard, small" is itself the best available evidence that data curation dominates. It is practitioner consensus (roughly grade B/C), not grade-A controlled evidence.

### Gaps
- MiniMax-M1's RL data curation and the Tulu 3 RLVR data details were not read this session.
- No tech report I read gives an ablation of its data filter on a final-scale run.
- DeepSeek-R1 Nature supplementary details beyond Table 4 were not read.
