# Verification: "the layer beneath" claims (RLVR boundary / base-model priors / spurious rewards)

Date: 2026-10-14. Sources: arXiv full-text PDFs (latest version served by arxiv.org/pdf/<id>), text extracted with PyMuPDF.
Legend: VERIFIED / WRONG (correct value) / NOT FOUND / FIGURE-ONLY (value exists only as a plot or as in-figure data labels).
"Fig-label" = number printed inside a figure (extractable, but the series-to-label mapping is inferred from order/legend).

---

## 1. Yue et al., "Does RL Really Incentivize Reasoning Capacity..." (2504.13837)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Distilled model (R1-Distill-Qwen-7B) exceeds its base at all k, unlike RL models | VERIFIED (qualitative); curve FIGURE-ONLY | Sec 4.2, Fig 7 (Minerva only). Base = Qwen2.5-Math-7B; RL comparator = Qwen2.5-Math-7B-Oat-Zero. Quote: "the pass@k curve of the distilled model is consistently and significantly above that of the base model." Shown for one benchmark (Minerva) only. |
| Base surpasses RL at large k "on most benchmarks" | VERIFIED, and paper's claim is stronger | Sec 1 bullets / Sec 3: "base models consistently surpass RLVR models across all benchmarks and LLM families as k increases." Example: Minerva, 32B, base ahead by ~9% at k=128. |
| Coverage analysis on AIME24 (base-only vs RL-only) | VERIFIED | Table 2 (AIME24, k=1024; Base vs SimpleRLZoo): both 63.3%, base-only 13.3%, RL-only 0.0%, neither 23.3%. MATH500 (k=128): 92.4 / 3.6 / 1.0 / 3.0. Table 5 (App C.7): Qwen2.5-7B-Base solves 23 AIME24 indices, SimpleRL-Qwen2.5-7B 19, RL set a strict subset (base-only = {4,11,17,19} = 4/30 = 13.3%). App C.7: the ~5 RL-only MATH500 problems are all solved by base at 1024 samples. |

## 2. ProRL (2505.24864)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Steps / base | VERIFIED | Sec 1: "more than 2k training steps"; base DeepSeek-R1-Distill-Qwen-1.5B -> Nemotron-Research-Reasoning-Qwen-1.5B. |
| Expansion strongest where base is poor | VERIFIED | Sec 4.1 "The Weaker the Start, the Stronger the Gain": "significant negative correlation between the base model's reasoning boundary and the extent of reasoning improvement" (pass@128, Fig 3; 18 of 96 Reasoning Gym tasks re-run at 256 samples). |
| pass@k decreases on some math tasks | VERIFIED | Sec 4.2 "Diminish" regime: "Although pass@1 improves, the pass@128 score ... often declines" (esp. math). Fig 4 math panel Fig-labels: pass@256 final 0.982 vs base 0.990 (pass@1 0.918 vs 0.844). Diminish panels: math, deepcoder_livecodebench, letter_counting; Plateau: gpqa_diamond, maze, palindrome; Sustained: codecontests, taco, dice. |
| Task-level: near-zero base pass@k becomes solvable | VERIFIED (boxnet); per-task values FIGURE-ONLY | Sec 4.3 / Fig 5, OOD task boxnet (unseen in training): "the base model exhibits no capability of solving the task." Fig-labels: base 0.000 at every k up to 256; final model 0.077 at k=1 -> 1.000 at k=256. Sec 1: "many tasks where the base model fails to produce any correct solutions regardless of the amount of sampling, while our RL-trained model achieves 100% pass rates (Figure 4)". Also graph_color generalizes to larger graphs (Fig 6, figure-only). |

## 3. "The Invisible Leash" (2507.14843)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Main model version | VERIFIED | Sec 4.1: ProRL-1.5B-v1 is the main RLVR method (base DeepSeek-R1-Distill-Qwen-1.5B); ProRL-1.5B-v2 also reported. |
| Support counts (Table 1, P/E/S/O) | VERIFIED | ProRL-1.5B-v1 Overall: P=2400, **E=36, S=163**, O=805 (SRR 0.94, NSCR -0.05); Math E=5 vs S=56; Non-Math E=31 vs S=107. ProRL-1.5B-v2 Overall: P=2388, E=48, S=175. Every row in Table 1 has S > E (e.g., Nemotron-1-7B 28 vs 56; Skywork-OR1-7B 26 vs 86; Phi4-Reason-Plus-14B 16 vs 23; OLMo-2-1B 83 vs 104). |
| Precise claim | VERIFIED, wording note | Abstract: "the shrinkage of empirical support generally outweighs the expansion of empirical support under larger sampling budgets." Sec 1: RLVR "primarily preserves rather than expands the base model's solution coverage"; framed as "support-constrained optimization". "Reweights" is a fair paraphrase but is not the paper's word; preservation (P) dominates, and S > E. |

## 4. Pass@k Training (2508.10751)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Improves pass@k without hurting pass@1 | VERIFIED (as the paper's claim) | Sec 1 contributions: "improving Pass@k performance while not harming Pass@1 scores." Caveat: Table 5 shows small pass@1 dips vs Pass@1 Training in some rows (LLaMA3.2-3B AIME24 12.7 vs 13.6; R1-Distill-1.5B 36.5 vs 36.7). |
| Key numbers on Qwen2.5-7B, AIME24 pass@256 vs base | NOT FOUND | No Qwen2.5-7B math/AIME24 result and no pass@256 anywhere. Main backbone is Qwen2.5-7B-**Instruct** on Maze/Enigmata; evaluation samples 32 (Maze) or 8 (other tasks) responses (App A.2). Table 1 (Qwen2.5-7B-Instruct, pass@1/pass@k): Enigmata 4.8/10.1 base -> 12.9/21.3 (P@1 T.) -> 17.9/29.8 (P@k T.); AIME 2025 4.2/15.8 -> 5.4/19.1 -> 7.1/22.4. AIME 2024 appears only in App E Table 5 for LLaMA and R1-Distill models, e.g. LLaMA3.1-8B-Instruct 3.4/17.9 base -> 4.4/32.1 (P@1) -> 7.1/40.0 (P@k); R1-Distill-Qwen-7B 43.2/80.9 -> 48.5/79.5 -> 48.2/80.9. |

## 5. Cheng, Huang, Zhu, Sojoudi, "How RL Reshapes LLM Reasoning: Transferability, Coverage, and Scaling Laws" (2610.04158)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Observes both regimes | VERIFIED, wording note | Abstract: "coverage at large sampling budgets increases on some tasks and decreases on others." Sec 2.3: most prevalent pattern is higher Pass@1 with lower Pass@1024 (esp. math); BoxNet expands (Qwen Pass@1 3.8%->9.9%, Pass@1024 34.0%->51.0%). The paper does not call these "sharpening" vs "expansion" regimes; it says coverage expands or shrinks, and argues both arise from reweighting an existing strategy library ("broader sampled coverage need not imply new strategies"). |
| Qwen3 and Gemma-3 move in opposite directions OOD under the same math RL | VERIFIED | Sec 2.2: "despite identical training data, Qwen3-Math-RL and Gemma-3-Math-RL shift in opposite directions on many OOD tasks." Models: Qwen3-14B-Base and Gemma-3-12B-IT; 48K UniReason math problems, GRPO, 200 steps. Coverage example (App F.2): AIME 2025 coverage 23/30 -> 25/30 for Qwen but 20/30 -> 18/30 for Gemma. Gemma BoxNet: Pass@1 9.1% -> 7.3% while coverage 16/100 -> 30/100. |
| log-sigmoid vs log-linear | skipped (already verified) | |

## 6. OctoThinker (2506.20512)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Mid-training makes Llama-3.2-3B match Qwen2.5-3B under RL | VERIFIED (text); numbers FIGURE-ONLY | Sec 5 (Fig 14): "OctoThinker-Long-3B ... reaches performance on par with Qwen2.5-3B"; hybrid and short branches "marginally inferior, especially on challenging benchmarks". Fig 1 caption: "matching Qwen2.5 performance." Fig 1 MATH500 Fig-labels (mapping inferred): Llama-3.2-3B-Base 7.4, Llama-3.2-3B-Zero 10.0, OctoThinker-Stable-3B-Base 22.4, OctoThinker-Long-3B-Base 25.8, OctoThinker-Long-3B-Zero 65.2, Qwen2.5-3B-Base 38.2, Qwen2.5-3B-Zero 66.4. |
| "mid-training on MegaMath-Web-Pro" causes the match | NEEDS REWORDING | MegaMath-Web-Pro is the finding that a high-quality math corpus helps (vs FineMath-4plus). The model that matches Qwen is the full Stable-then-Decay recipe: 200B tokens stable stage (Table 2: MegaMath-Web-Pro-Max 0.725, DCLM-Baseline 0.10, MegaMath text-code 0.10, QA 0.05, code 0.025) + 20B-token decay stage, Long-CoT branch. Only the Long branch is "on par". |

## 7. Gandhi et al., "Cognitive Behaviors that Enable Self-Improving Reasoners" (2503.01307)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Qwen-2.5-3B naturally shows verification/backtracking; Llama-3.2-3B does not | VERIFIED | Sec 1: "Qwen naturally exhibits these reasoning behaviors, particularly verification and backtracking, while Llama lacks them." |
| Countdown baseline numbers | VERIFIED | Sec 3: by end of RL, "Qwen achieves approximately 60% accuracy, substantially outperforming Llama's 30%." |
| Priming Llama matches/exceeds Qwen; incorrect solutions also work | VERIFIED | Abstract: priming "enables substantial improvements during RL, matching or exceeding Qwen's performance"; "models primed with incorrect solutions containing proper reasoning patterns achieve comparable performance to those trained on correct solutions" (Sec 4 says "identical", Fig 6). Empty-CoT controls stay at base Llama level (~30-35%, Fig 5). Precise structure: incorrect-primed ~= correct-primed, and correct-primed matches Qwen; per-condition curves are figure-only. |

## 8. Wu et al., "Reasoning or Memorization?" (2507.10532)

| Claim | Verdict | Location / evidence |
|---|---|---|
| On clean RandomCalculation, random/incorrect rewards give no reliable gain for Qwen2.5-Math-7B; correct rewards do | VERIFIED | Abstract: "only accurate reward signals yield steady improvements that surpass the base model's performance boundary ... whereas random or incorrect rewards do not." Sec 1: random rewards "yielding no reliable improvement, while inverse rewards rapidly erode" ability. Sec 4 (Fig 7, 5-step and 10-step subsets, 700 train / 300 val): under incorrect/random rewards Qwen "either maintains its base performance or exhibits only marginal and unstable improvements", which vanish from 5 to 10 steps; correct rewards surpass Max@16. Llama3.1-8B-Instruct fails to exceed Max@16 even with correct rewards. |

## 9. Chen et al., "Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward" (2512.16912, ICLR 2026)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Clipping term >= ~17x smaller than raw signal | VERIFIED (with scope) | Corollary 3.6: with Qwen-Math hyperparameters (eta=5e-7, eps=0.2, p+=0.001, G=16, L=4096, pi_min=1e-6), E[abs N_raw] / E[abs C+_tot] >= ~17.15. It is a lower bound on the ratio for the upper-clipping correction at those specific settings, not a general constant. |
| Clipping acts like entropy minimization | VERIFIED | Sec 4: "Under random rewards, clipping acts as an implicit entropy minimization mechanism." Fig 2: unclipped entropy rises, clipped entropy decreases monotonically. Paper also says clipping bias "provides no meaningful learning signal"; Fig 1: enabling clipping can lower validation accuracy while disabling it often improves it (contradicts Spurious Rewards' claim that gains need clipping). |
| Random-reward gains on R1-Distill-Llama-8B / QwQ-32B | VERIFIED (R1-Distill-Llama-8B); QwQ-32B partial | Sec 5 / Fig 4 Middle: R1-Distill-Llama-8B, clipped random-reward training, "improvements comparable to those in Figure 2"; Fig 4 Right Fig-labels (MATH500, base -> RLVR, mapping inferred): Qwen2.5-Math-1.5B 0.48 -> 0.49+-0.03, Qwen2.5-Math-7B 0.64 -> 0.70+-0.01, R1-Distill-Llama-8B 0.66 -> 0.74+-0.03. Sec 4.2: unclipped R1-Distill-Llama-8B goes 65.6% -> 76.6% in 100 steps, then gradient explosion near step 150. QwQ-32B: only Fig 3 Left (AIME training set), described as "steady early-epoch gains"; no numeric gain reported in text. Sec 1 frames it as "similar gains also arise in the Llama and QwQ families". |

## 10. Shao et al., "Spurious Rewards" (2506.10947, v2)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Random-reward gains on Qwen2.5-Math depend on clipping | VERIFIED (as the paper's claim) | Sec 4, Fig 4 (Qwen2.5-Math-7B): three no-clipping variants "fail to yield consistent performance"; "Random rewards yield consistent improvements only when clipping is present." Headline: MATH-500 +21.4 points with random rewards vs +29.1 with ground truth. Disputed by 2512.16912 (item 9). |
| Code-reasoning frequency 65% -> ~90% | VERIFIED | Abstract: "from 65% to over 90%"; Sec 5: 65.0% before training, "~90% within 15 steps and up to 95.6% for random rewards". |
| Accuracy with vs without code 60.9% vs 28.0% | VERIFIED | Sec 1: "answers with it have an accuracy of 60.9%, much higher than without (28.0% accuracy)" (Qwen2.5-Math-7B, MATH-500). |

## 11. Dr. GRPO, "Understanding R1-Zero-Like Training" (2503.20783)

| Claim | Verdict | Location / evidence |
|---|---|---|
| Qwen2.5-Math bases already show "aha"/self-reflection before RL | VERIFIED (text); counts FIGURE-ONLY | Sec 1: "Nearly all base models already exhibit the 'Aha moment', including DeepSeek-V3-Base." Sec 4: Qwen2.5 bases "are already strong math solvers and exhibit self-reflection patterns." Per-model counts (Qwen2.5-Math-1.5B/7B, Qwen2.5-7B, Llama-3.1-8B, DeepSeek-Math-7B, DeepSeek-V3-Base) are only in Fig 3 Right. The abstract's emphasis is DeepSeek-V3-Base; for Qwen2.5 it stresses strong reasoning without templates. Paper also finds self-reflection in R1-Zero is "not positively correlated with higher accuracy." |
| Template effects | skipped (already verified) | |

---

## Compact list: WRONG / NOT FOUND / needs rewording

- **NOT FOUND, #4 Pass@k Training:** there are no Qwen2.5-7B AIME24 pass@256 numbers. The backbone is Qwen2.5-7B-Instruct on Maze/Enigmata, with pass@k at k<=8 (32 samples for Maze). AIME24 results exist only for LLaMA and R1-Distill models (App E, Table 5). Also note the small pass@1 dips relative to Pass@1 Training.
- **Reword, #6 OctoThinker:** the match with Qwen2.5-3B comes from the full Stable-then-Decay recipe, not from MegaMath-Web-Pro alone. That recipe is 200B tokens, mostly MegaMath-Web-Pro-Max, followed by 20B decay tokens. Only the Long-CoT branch reaches "on par"; the Short and Hybrid branches are slightly behind. The numbers are figure-only.
- **Reword, #5 (2610.04158):** the paper does not use the labels "sharpening" and "expansion regime". It says finite-budget coverage expands or shrinks depending on task, base model and training mix, and attributes both to reweighting existing strategies. The opposite OOD shifts between Qwen3-14B-Base and Gemma-3-12B-IT are verified.
- **Reword, #1 Yue:** the paper says the base model wins at large k across *all* benchmarks, not "most". The distillation-above-base result is shown on one benchmark only (Minerva, Fig 7). The AIME24 split is: 13.3% of problems base-only, 0.0% RL-only (k=1024).
- **Scope caveat, #9:** "~17x" is a lower bound (17.15) for one specific Qwen-Math hyperparameter setting, on the upper-clip correction term. The QwQ-32B gain appears only as a figure. The paper disputes that clipping drives the gains, which conflicts with #10.
- **Scope caveat, #3 Invisible Leash:** "reweights" is a paraphrase. The paper's own terms are "preserves rather than expands" and "support-constrained optimization". For ProRL-1.5B-v1 the counts are E=36 vs S=163.
- **Figure-only:** ProRL boxnet and math pass@k values (in-figure labels); Dr. GRPO self-reflection counts; OctoThinker MATH500 bars; Cognitive Behaviors per-condition curves.
