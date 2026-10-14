# Verification: agentic / multi-turn RL claims

Date: 2026-10-14. Method: arXiv full-text PDFs extracted with PyMuPDF (page numbers = PDF page index); official blogs / HF model card fetched raw; verl docs read from the local checkout `_external_repos/verl` (HEAD 7dc39fe, 2026-05-22). Versions: Search-R1 v-latest (COLM 2025), MT-GRPO 2505.11821v3, SORL 2511.20718v3, MiniMax-M2 2605.26494v2.

Labels: VERIFIED / WRONG (correct value) / NOT FOUND / FIGURE-ONLY / NOT ACCESSIBLE.

---

## 1. Search-R1 (2503.09516): retrieved-token loss masking

**VERIFIED.** Table 4 (p.9; 7B only) and Table 6 (p.18; 7B + 3B), RL = PPO, base models.

| Model | Variant | NQ | TriviaQA | PopQA | HotpotQA | 2wiki | Musique | Bamboogle | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-base | w. mask | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | **0.431** |
| | w.o. mask | 0.388 | 0.567 | 0.391 | 0.325 | 0.321 | 0.108 | 0.304 | **0.343** |
| Qwen2.5-3B-base | w. mask | 0.406 | 0.587 | 0.435 | 0.284 | 0.273 | 0.049 | 0.088 | **0.303** |
| | w.o. mask | 0.346 | 0.484 | 0.365 | 0.241 | 0.244 | 0.053 | 0.104 | **0.262** |

- Datasets where no-mask is better: **3B only — Musique (0.053 vs 0.049) and Bamboogle (0.104 vs 0.088).** At 7B, mask wins on all seven.
- The captions say masking "achieves consistently better performance", which the 3B rows contradict on 2 of 7 datasets (the average still favours masking).

## 2. verl docs: re-tokenization breaks token IDs and PPO convergence

**VERIFIED, with a scope caveat.** File: `_external_repos/verl/docs/advance/agent_loop.rst`, section "Chat completion vs Token in token out" (lines 146–182).
- Line 154: "we found the token_ids from apply the final messages may not equal to the token_ids by concat prompt_ids and response_ids in each turn."
- Lines 181–182: "We have observed that apply_chat_template to the final chat history messages make PPO training not even converged in single-turn."
- Caveats: (a) the section opens with a `.. warning::` saying the conclusion "is still open to investigation and discussion". (b) The non-convergence observation is stated for **single-turn** training (DAPO); the multi-turn (retool) experience is cited only for the token-ID mismatch. (c) It names two causes: tool parsers altering content, and decode-encode non-invertibility.

## 3. GLM-5 (2602.15763): TITO gateway

**VERIFIED.** §4.1.2 "Optimizing Asynchronous Training Stability", p.16 (also summarised on p.12).
- Quote: "In practice, we implement a TITO Gateway that intercepts all generation requests from rollout tasks and records each trajectory's token IDs and metadata."
- Rationale (same paragraph): re-tokenization "can introduce subtle mismatches in token boundaries, whitespace/normalization handling, truncation, or special-token placement".
- Related: Kimi K2.5 (2602.02276, App. D, p.24) says "Our framework strictly follows a Token-in-Token-out paradigm" and also describes an "LLM Gateway" proxy for black-box environments.

## 4. SimpleTIR (2509.02479): void-turn filtering ablation

**VERIFIED.** Table 2 (p.9), Qwen2.5-7B (base, Zero-RL setting), AIME24 avg@32:

| SimpleTIR-7B | Naive Multi-Turn | Low Prob Filtering | High Ratio Filtering | Stop Generation w/o Filtering |
|---|---|---|---|---|
| 50.5 | 20.8 | 23.3 | 26.3 | 26.1 |

(Math500: 88.4 / 73.1 / 72.8 / 75.0 / 77.3.)
- **Peak vs final:** the Table 2 caption says "Considering the unstable training of ablated methods, we report the highest scores within 1000 gradient steps." So every Table 2 number, SimpleTIR's 50.5 included as written, is a **peak within 1000 steps**, not a final checkpoint.
- The checkpoint-selection rule for the main Table 1 is NOT FOUND (p.8 states only avg@32 with temperature 1).
- Abstract framing: "from a text-only baseline of 22.1 to 50.5". The 20.8 is the naive multi-turn TIR ablation, a different baseline from the 22.1.

## 5. GiGPO (2505.10978)

- **Gains — VERIFIED with a qualification.** Table 1 (p.8), 3 seeds (caption: "Results are averaged over 3 random seeds"), mean ± std reported. These are differences in **absolute points**; the paper writes "%".
  - Text (p.7): "GiGPOw/o std surpasses GRPO by 13.3% on ALFWorld and 10.6% on WebShop at 1.5B, and by 12.6% and 9.1%, respectively, at 7B." The cells confirm it:
    - 1.5B ALFWorld: 86.1±4.7 vs 72.8±3.6.
    - 1.5B WebShop success: 67.4±4.5 vs 56.8±3.8.
    - 7B ALFWorld: 90.2±2.3 vs 77.6±5.2.
    - 7B WebShop success: 75.2±3.8 vs 66.1±3.7.
  - So "+9–14 points" holds for the **w/o-std variant only (9.1–13.3)**. The w/-std variant ranges 6.7–13.9: ALFWorld 86.7 / 90.8, WebShop success 65.0 / 72.8.
  - The abstract claims are only ">12% on ALFWorld and >9% on WebShop".
- **Shared hyperparameters — VERIFIED.** p.7, Training details: "For ALFWorld and WebShop, all RL training methods (including ours and the baselines) use exactly the same hyperparameter configurations." The PPO baseline therefore also got no separate tuning. Group size N = 8.
- **Overhead "<0.002%" — quote VERIFIED but internally INCONSISTENT.** p.10: anchor grouping 0.01 s + step advantage 0.53 s vs 362.83 s per iteration (Qwen2.5-1.5B, ALFWorld); the paper calls this "< 0.002% of the total per-iteration training time".
  - The arithmetic is 0.54 / 362.83 = **0.15% (≈0.0015 as a fraction)**. "<0.002" is correct as a fraction, not as a percent.
  - Recommend citing "≈0.5 s of ≈363 s per iteration (~0.15%)".

## 6. MT-GRPO / turn-level credit (2505.11821v3)

- **"Only 2 turns" — VERIFIED for the GRPO experiments.** §4.3 (p.5): "evaluate three GRPO algorithms in a two-turn agent setting". Setup: Qwen2.5-7B, TriviaQA, Wikipedia search.
  - The v3 adds experiments the claim does not cover: MT-PPO with "maximum number of turns Nmax to 4" (p.7; NQ/HotpotQA, 7B), and a Sokoban VLM setting with "a maximum of three turns" (p.7).
  - The remark on p.5 notes that MT-GRPO needs G^(K−1) rollouts, which is exponential in turns.
- **Outcome-only GRPO EM 0 — VERIFIED.** Table 2 (p.5), GRPO-OR row:

| Tool Exec | Search Ans | XML Format | Exact Match |
|---|---|---|---|
| 0 | 0 | 0.04 | **0** |

  The comparison rows: GRPO-MR EM 0.3346; MT-GRPO EM 0.5010; Qwen2.5-7B-Base 0.0469; Qwen2.5-7B-Instruct 0.1559. These are validation reward scores, and no seeds or std are reported.

## 7. VAGEN (2510.16907)

- **"Most gain (0.55 → 0.76) from reasoning format" — WRONG / misattributed.**
  - The 0.55 in the paper is a **delta**, not a level. p.7: WorldModeling "achieves a substantial improvement over its untrained counterpart (+0.55)": 0.21 (Qwen2.5-VL-3B) → 0.76 (Table 2, p.6, Overall). That gain combines RL and the reasoning format.
  - The format-only contrast, all trained with VAGEN-Base, is: FreeThink 0.67 → WorldModeling 0.76 (+0.09); NoThink 0.28; StateEstimation 0.56; TransitionModeling 0.68.
  - A level of 0.55 does appear: "Turn-PPO w/ Mask" = 0.55 (GRPO w/ Mask 0.54, Vanilla-PPO 0.26). Those baselines **already use the WorldModeling strategy** (section header: "RL Baselines with World Model Reasoning Strategy"). So 0.55 → 0.76 is an **RL-algorithm** gap, not a format gap.
- **Bi-Level GAE + WorldModeling reward ≈ +0.06 — VERIFIED.** Table 5 (p.11): VAGEN-Base 0.76 → VAGEN-Full 0.82 Overall. The gain is mostly from Sokoban (0.61 → 0.79). For SVG, only Bi-Level GAE is applied.
- **LLM-judge reward hacking — VERIFIED.** p.11, "Reward Hacking and Over-optimization": agents "develop generic, broadly applicable responses that satisfy the LLM-as-a-Judge's criteria"; this is "particularly evident in agents trained with Bi-Level GAE". Details in App. E.3 (p.52; FrozenLake). Mitigations are in App. D.2.

## 8. LOOP (2502.01600): importance-ratio granularity and PPO vs GRPO

**VERIFIED.** Table 1 (p.6), AppWorld Test-Normal TGC (Test-C TGC in parentheses), Qwen2.5-32B-Instruct. Each number covers 5 eval runs of the checkpoint chosen by best dev performance.

| Variant | Test-N TGC | Test-C TGC |
|---|---|---|
| LOOP (token) | 71.3 ± 1.3 | 45.7 |
| LOOP (turn) | 64.1 ± 2.2 | 40.8 |
| LOOP (bandit = per-trajectory/sequence) | 53.3 ± 3.4 | 27.7 |
| LOOP RwNorm (token) | 61.9 ± 4.0 | — |
| PPO (learned critic, token) | 50.8 ± 3.7 | 26.4 |
| GRPO (token, strictly on-policy, with KL) | 58.0 ± 1.8 | 39.5 |
| GRPO no kl | 59.0 ± 1.4 | 42.7 |
| RLOO (traj, on-policy) | 57.2 ± 2.6 | 36.7 |

- p.6: "All versions of GRPO and LOOP share the same experimental setup and all hyperparameters."
- Wording caveat: the GRPO baseline is strictly on-policy (single gradient step per rollout batch), while LOOP takes multiple off-policy steps. GRPO-vs-PPO here is not a clean critic-vs-no-critic comparison.

## 9. SORL (2511.20718v3): turn-level importance ratios under off-policy updates

**VERIFIED (qualitative; the quantitative evidence is FIGURE-ONLY).**
- §4.1 Validation (p.8): "turn-level importance sampling reduces the norm of the PPO policy gradient by several orders of magnitude" (Fig. 3b; token- vs turn-level PPO, Qwen2.5-7B base, search task).
- The same paragraph qualifies it: "turn-level PPO still exhibits occasional sharp drops … fails to sustain its best performance." Turn-level IS **alone does not suffice**; clipping-triggered normalization is the second required component.
- p.12: turn-level vs sequence-level PPO comparison in App. A.6.1, where "turn-level importance sampling achieves stronger and more stable performance".

## 10. Practitioner's Guide (2510.01132): PPO vs GRPO on TextWorld

- **PPO beats GRPO — VERIFIED.** Table 10 (p.9), success rate (base model in parentheses):

| Task / model | PPO | RLOO | Reinforce++ | GRPO |
|---|---|---|---|---|
| w2-o3-q4 / 1.5B (0.15) | 0.88 | 0.51 | 0.18 | 0.18 |
| w4-o6-q8 / 1.5B (0.01) | 0.59 | 0.0 | 0.0 | 0.02 |
| w2-o3-q4 / 7B (0.65) | 0.98 | 0.97 | 0.72 | 0.79 |
| w4-o6-q8 / 7B (0.28) | 0.72 | 0.47 | 0.33 | 0.36 (text) |

- **Were both tuned? — NO evidence that GRPO was tuned.**
  - App. A (p.14): "we conducted systematic hyperparameter sweeps training Qwen-1.5B on TextWorld w2-o3-q4 tasks **using PPO**."
  - App. C (p.19): the GRPO defaults are listed as "batch size of 16 for Qwen3-8B" with group size 4, i.e. the SWE-Gym config. No GRPO sweep on TextWorld is described.
  - p.8 says only "consistent hyperparameters across experiments."
  - Minor internal inconsistency: App. A's "optimal" PPO config (actor 1e-6, critic 1e-5, KL 0.01) differs from App. C's PPO defaults (5e-7 / 5e-6 / 0.001).

## 11. 2505.15117 (empirical study of search-agent RL)

- **R1-distill collapses with GRPO, stable with PPO — VERIFIED.**
  - Table 3 (p.6): DeepSeek-R1-Distill-Qwen-7B Avg EM is PPO 0.344 vs GRPO 0.100. For comparison, Qwen2.5-7B-Base gets PPO 0.434 and GRPO 0.420.
  - p.7: "training with GRPO leads to training collapse." The PPO run is described as learning "slow and gradual" rather than simply "stable".
- **Random vs E5 (Exact) EM 0.241 vs 0.430 — VERIFIED.** Table 4 (p.8; Qwen2.5-7B-Base, PPO), avg EM: Random 0.241 / BM25 0.352 / E5 (HNSW) 0.379 / E5 (Exact) 0.430. The random engine's 0.241 comes with recall 0.000, so it reflects parametric knowledge only.
- **Intermediate retrieval reward does not help — VERIFIED.** Table 2 (p.6), λr = 0.1, avg EM:

| Model | Algo | w/o retrieval reward | w/ retrieval reward |
|---|---|---|---|
| 3B | PPO | 0.389 | 0.340 |
| 3B | GRPO | 0.382 | 0.385 |
| 7B | PPO | 0.434 | 0.419 |
| 7B | GRPO | 0.420 | 0.410 |

- **Format reward: 3B PPO 0.303 → 0.389 — VERIFIED** (Table 1, p.5, base model). **"Hurts at 14B" — needs rewording.** Table 1, 14B:
  - 14B-Base PPO: 0.479 → 0.459 (hurts).
  - 14B-Instruct PPO: 0.433 → 0.456 (helps).
  - 14B-Base GRPO: 0.462 → 0.491 (helps).
  - 14B-Instruct GRPO: 0.446 → 0.471 (helps).
  - Format reward also hurts in smaller-model rows: 3B-Instruct GRPO 0.336 → 0.319, 7B-Instruct PPO 0.385 → 0.370, 7B-Instruct GRPO 0.396 → 0.378.
  - Despite these rows, the paper's text (p.4) says "consistently improves final model performance, particularly for base LLMs."

## 12. DemyAgent (2510.11701)

- **Real vs stitched SFT, AIME24 avg@32 33.23 vs 4.38 — VERIFIED** (Qwen3-4B-Instruct-2507). Table 1 (p.5):
  - Qwen3-4B AIME24: avg@32 4.38% → 33.23% (+28.85); pass@32 35.15 → 75.66; maj@32 0.01 → 51.64.
  - Qwen2.5-7B-Instruct AIME24: avg@32 6.77 → 17.91.
  - Minor: the p.7 text says "initial accuracy of 29.79% and 33.23%" for AIME2024/AIME2025, which swaps Table 1 (AIME24 33.23, AIME25 29.79).
- **Token-level loss +~3.9 at 4B, comparable at 7B — VERIFIED (text numbers; the curves are FIGURE-ONLY, Fig. 4 p.8).**
  - p.8: "token-level loss exceeds sequence-level loss by 3.95% on AIME24 and 3.86% on AIME25 under the same training budget" (Qwen3-4B-RA-SFT).
  - For Qwen2.5-7B-RA-SFT the two "achieve comparable average@32 performance" (pp.7–8).
  - The paper attributes the 4B gain to stronger initial exploration, not to model size. It does not say whether "%" means points or a relative difference.

## 13. BrowseComp with test-time context management

- **DeepSeek-V3.2 (2512.02556) 51.4 → 67.6 — VERIFIED.**
  - Table 2 (p.13): BrowseComp cell "51.4/67.6*" (* = with context management).
  - p.14: "the score is 51.4 without context management".
  - §4.4 (p.17): "Discard-all … achieving a score of 67.6". Summary reaches 60.2.
  - Trigger: token usage above 80% of the 128K window.
- **Kimi K2.5 (2602.02276) 60.6 → 74.9 — VERIFIED.**
  - p.11: "K2.5 achieves 60.6% without context management techniques, 74.9% with Discard-all context management". The same numbers appear in the main table (p.12).
  - App. E (p.26): uses "the same discard-all strategy proposed by DeepSeek".
- **GLM-5 (2602.15763) 55.3 → 75.9 — numbers present but NEEDS REWORDING.**
  - Table 7 (p.23) reports BrowseComp **62.0** vs BrowseComp (w/ Context Manage) **75.9**. The same table lists DeepSeek-V3.2 51.4/67.6 and Kimi K2.5 60.6/74.9.
  - The 55.3 appears only in §4.2.4 (p.19), as the score with **no** context handling: keep-recent-5 lifts 55.3 → 62.0, and hierarchical (keep-recent + discard-all at T = 32k) reaches 75.9.
  - App. B (p.36): "Without context management, we retain details from the most recent 5 turns."
  - Like-for-like with the other two models is therefore **62.0 → 75.9**. 55.3 → 75.9 is GLM-5's full no-CM → HCM gain.
  - Minor internal inconsistency: Table 7's prose (p.24) says the w/-CM setting uses "discard-all … same as DeepSeek-V3.2", but §4.2.4 attributes 75.9 to the keep-recent + discard-all hybrid.

## 14. DeepSWE and SkyRL SA-SWE-32B

- **DeepSWE 23 → 42.2 — VERIFIED** (HF model card `agentica-org/DeepSWE-Preview` README; the Notion blog was not fetched).
  - Fig. 2 caption: "With just 200 steps of RL training, SWE-Bench-Verified score increases from 23→42% (+20%)."
  - Table: "DeepSWE-Preview (32B) | R2E-Gym | Agent | 42.2%". This is Pass@1 averaged over 16 runs, base Qwen3-32B (thinking mode), R2E-Gym scaffold, 64k context, 100 steps.
  - Note: the 23 is a rounded figure-caption value. The card's headline 59.0% is hybrid Best@16 test-time scaling (TTS).
- **SkyRL SA-SWE-32B 24.4 → 39.4 — VERIFIED.** SkyRL-Agent (2511.16108), abstract p.1: "trained from Qwen3-32B (24.4% Pass@1) purely with RL … reach 39.4% Pass@1 on SWE-Bench Verified".
  - Table 2 (p.10): Qwen3-32B 24.4; SA-SWE-32B 39.4 (4601 H100 h).
  - The same table lists DeepSWE at 36.4 under SkyRL's Simple ReAct vs 42.2 reported (9180 H100 h).

## 15. Async rollout / training speedups

- **Kimi ≥1.5× — VERIFIED, but the source is the Kimi-Researcher blog** (moonshotai.github.io/Kimi-Researcher), not a Kimi tech report.
  - Quote: "Turn-level Partial Rollout … this mechanism delivers substantial rollout acceleration (at least 1.5x)."
  - The figure is attributed to **turn-level partial rollout**. The separate "fully asynchronous rollout" paragraph gives no number.
  - NOT FOUND in Kimi k1.5 (2501.12599), K2 (2507.20534) or K2.5 (2602.02276); these describe partial rollout and async coroutines without a speedup figure.
- **SkyRL-Agent 1.55× — VERIFIED.** 2511.16108 abstract (p.1) and §3 (p.6): "an optimized asynchronous pipeline dispatcher that achieves a 1.55× speedup over naive asynchronous batching." The baseline is "naive Async Batch (Bounded)", not a synchronous pipeline.
- **verl 1.6–2.35× — VERIFIED.** File: `_external_repos/verl/docs/advance/fully_async.md`, lines 368–370, 128-card Qwen2.5-Math-7B:
  - "the benefit brought by streaming is approximately 1.6x, and after combining staleness and partial_rollout, the benefit reaches 2.35x."
  - Line 9 headline: "2.35x-2.67x performance improvement when training the Qwen2.5-7B model with 128 GPUs."
  - Context: this is **math RL (DAPO-style)**, not agentic multi-turn.
- **MiniMax Forge prefix-tree merging 40× — VERIFIED.**
  - Forge blog (minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm, §3.2): "this solution achieves a 40x training speedup".
  - MiniMax-M2 report (2605.26494v2, §6.2.5, p.22): "achieves up to 40× training speedup".
  - Note: this is a **trainer-side (forward/backward) speedup** from deduplicating shared prefixes, stated as "up to". It is not a rollout-throughput number, and no workload or baseline details are given.

---

## Summary: WRONG / NOT FOUND / needs rewording

1. **VAGEN (#7) — WRONG attribution.** "0.55 → 0.76 from reasoning format" mixes two things. +0.55 is the delta 0.21 → 0.76 from RL plus WorldModeling. The isolated format effect is FreeThink 0.67 → 0.76 (+0.09). 0.55 is the level of Turn-PPO w/ Mask, which already uses WorldModeling, so 0.55 → 0.76 is an RL-algorithm gap. (+0.06 from Bi-Level GAE + WM reward and the reward hacking are verified.)
2. **GLM-5 BrowseComp (#13) — reword.** Table 7 gives 62.0 → 75.9. 55.3 is the no-keep-recent number from §4.2.4. Use 62.0 → 75.9 when comparing with DeepSeek (51.4 → 67.6) and Kimi (60.6 → 74.9).
3. **GiGPO overhead (#5) — paper's own number is off.** "<0.002%" should be ≈0.15% (0.54 s / 362.83 s). "+9–14 points" holds only for GiGPO w/o std (9.1–13.3); the w/-std variant gives 6.7–13.9.
4. **2505.15117 format reward (#11) — reword "hurts at 14B".** It hurts only 14B-Base PPO (0.479 → 0.459) and helps the other three 14B rows. It also hurts several Instruct rows at 3B and 7B.
5. **Practitioner's Guide (#10) — "both tuned" not supported.** Only PPO was swept (App. A); the GRPO config shown is the Qwen3-8B / SWE-Gym default.
6. **SimpleTIR (#4) — reword.** All Table 2 numbers, the 50.5 included, are peaks within 1000 steps. The selection rule for main Table 1 is NOT FOUND.
7. **Kimi ≥1.5× (#15) — source and mechanism.** The figure is from the Kimi-Researcher blog and refers to turn-level partial rollout. NOT FOUND in the k1.5 / K2 / K2.5 reports.
8. **verl 1.6–2.35× (#15)** is a math-RL (Qwen2.5-Math-7B) benchmark, not agentic. **MiniMax 40×** is an "up to" trainer-side speedup, not rollout throughput. **SkyRL 1.55×** is measured against naive async batching, not a synchronous pipeline.
9. **verl re-tokenization (#2) — scope.** The non-convergence observation is stated for single-turn training, under a "still open to investigation" warning.
10. **MT-GRPO (#6) — scope.** The 2-turn limit applies to the GRPO experiments only; v3 adds MT-PPO with up to 4 turns and Sokoban with 3.
11. **Search-R1 (#1)** — no-mask beats mask on 3B Musique and Bamboogle, despite the "consistently better" caption.
12. **SORL (#9)** — the evidence is figure-based. The paper says turn-level IS alone still collapses; clipping-triggered normalization is also needed.
