# Consistency read-through — 2026-10-14

Scope: all `_sections/*.html` in filename order (00 → 99), checked against the corrections log (§9 of `plan/2026-10-14-rl-algorithms-blog-outline.md`) and spot-checked against `research/notes/verification_*.md`. No files were edited.

Checks that passed: figures are numbered 1–9 in page order, and every "Figure N" reference matches its caption. Every `href="#…"` resolves. There are no VERIFY or TODO markers and no doubled words. The header counts match the body: ~105 unique arXiv IDs, 9 figures, 5 case studies, 6 patterns, a 12-item checklist, 12 experiments and ~27.8k words (≈130 min). Body numbers agree with the corrections log on ScaleRL 32k / C_mid / batch A values, the DAPO rows (30/36/38/41/42/50), the VC-PPO 5.6 vs VAPO 5 split, the 8k vs 16k VC-PPO results, GenAC, the DPO 15.1 vs 20.6 split, BrowseComp 14–16 (GLM-5 62.0→75.9), VAGEN, Invisible Leash v1, Yue et al., Pass@k Training, AceReason, ARLArena/SAPO, Tricks-or-Traps ε_high, R1 v2, slime k1, verl token-mean, Ivison, Rafailov, DNO, GPO, the spurious-reward losses (in 14b/21/29) and 2512.16912.

---

## A. Factual contradictions (with the sources, the corrections log, or another section)

1. **05-axes.html:23** (map row ⑥) and **11-off-policy.html:67** (verdict table)
   - Text: "Remove train–inference mismatch at source — *beats every correction tested*"; table: "*≥ every IS variant where compared*".
   - Problem: the one multi-seed comparison the survey cites (CIS, 11:46 and 22:10) has FP16 at 68.53 on Qwen3-30B-A3B. That is *below* TIS 69.14, IcePop 68.99, GSPO 69.23 and CIS 69.88 (verification_axis6 l.61). The survey itself says all of these sit within ~1.6 pts. The map's wording is the strongest form of a claim its own evidence contradicts.
   - Fix: map → "fixes the cause; ≥ corrections in long/MoE runs, ties them on short dense runs". Table ceiling cell → "≥ IS variants in long or MoE runs (VeXact, FP16 paper); within noise of them in the CIS multi-seed test".

2. **07-normalization.html:51**, **23-patterns.html:19**, **31-open.html:17**
   - Text: "Every normalization rule is a choice of *concave* transform F" / "gradient ascent on some concave transform" / "for some concave F".
   - Problem: GRPO's F = 2·arcsin√p has F′ = 1/√(p(1−p)), so F″ = −(1−2p)/(2(p(1−p))^{3/2}), which is positive for p > ½. F is convex on (½, 1), and that convexity is exactly the easy-prompt up-weighting the survey criticizes. Σp is linear. The claim is false for the survey's own main example, and it appears three times.
   - Fix: "a choice of transform F with F′ = 1/s. ML weighting and pass@k give concave F; GRPO's arcsin√p is concave below p = ½ and convex above, which is the easy-prompt up-weighting."

3. **08-aggregation.html:47**
   - Text: "The constant normalizer of Dr. GRPO is *the only mode* that is exactly the policy gradient up to a learning-rate constant."
   - Problem: the table's own row 3, "Sequence mean of token sums, 1/G", is also exactly Σ_t R·∇log π averaged over responses, which is the policy gradient. In addition, 03:91 says Dr. GRPO's paper formula *is* 1/G, while 08:17 gives Dr. GRPO as 1/(G·L_max).
   - Fix: "The two length-free modes — token sums per sequence (1/G) and Dr. GRPO's constant 1/(G·L_max) — are exactly the policy gradient up to a learning-rate constant." Also align 08:17 with 03:91: "1/G in the paper's formula; 1/(G·L_max) in released code".

4. **07-normalization.html:36**
   - Text: "Magistral, Kimi and GLM-4.5's agentic stage *subtract a mean only*."
   - Problem: Magistral subtracts the group mean and then normalizes per minibatch, (Â−mean)/std (verification_axes123 12b). 21b:24 says "minibatch normalization, no group std" and 25:30 says "minibatch-normalized", so the sections contradict each other.
   - Fix: "Kimi and GLM-4.5's agentic stage subtract a mean only. ScaleRL, AReaL, REINFORCE++ and Magistral (per minibatch) use a batch-level standard deviation instead."

5. **21b-case-frontier.html:25 vs 46** (Qwen3 row vs the ablation count)
   - Text: the table gives Qwen3 "Ablates? — No". The prose says "Three ablate one component or only systems choices. Four — V3.2, V4, K2, M2 — ablate none."
   - Problem: the table has only two "Partial" rows (K2.5, OLMo 3) and five "No" rows. The 6/3/4 split works only if Qwen3 is partial, which is how verification_frontier classes it ("OPD vs RL paradigm").
   - Fix: change the Qwen3 cell to "Partial: OPD vs RL only", or change the prose to "Two ablate one component… Five — V3.2, V4, K2, M2, Qwen3 — ablate none."

6. **21b-case-frontier.html:40**
   - Text: "Explicit handling of train–inference mismatch *in every report from the second half of 2025 onward*."
   - Problem: the table contradicts it. Kimi K2 (Jul 2025: "Synchronous, partial rollouts"), GLM-4.5 (Aug 2025: FP8 rollout, which causes mismatch rather than handling it) and MiniMax-M2 ("Asynchronous pool") show no mismatch handling. The verification note's own wording was "now standard".
   - Fix: "Explicit handling of train–inference mismatch has become standard: routing replay and masks (V3.2), …"

7. **26-beyond.html:20, 22**
   - Text: "MiniLLM uses the *reverse KL* from student to teacher" / "a full-vocabulary *reverse-KL* loss".
   - Problem: the corrections log says to stop using "reverse/forward" anywhere and to write the direction. The rest of the survey (10, 28) complies.
   - Fix: "MiniLLM minimizes KL(π_student‖π_teacher), which is mode-seeking, …" / "a full-vocabulary KL(π_student‖π_teacher) loss".

8. **04-tension.html:48**
   - Text: "These gains generally *do not transfer* to Llama3 or OLMo2."
   - Problem: per the corrections log, random rewards make these models *worse*: Llama3.1-8B-Instruct −11.5, OLMo2-7B −8.3. 14b:17, 21:21 and 29:8 already say "hurt". This is the one place that still says "no transfer".
   - Fix: "Random rewards make Llama3.1-8B-Instruct (−11.5) and OLMo2-7B (−8.3) worse."

9. **09-trust-region.html:42**
   - Text: "[GSPO] *falls below the base model* on Qwen1.5-MoE in a three-seed comparison (axis ⑥)."
   - Problem: GSPO scores 28.90 ± 1.25 against a base of 28.93, a gap of 0.03 inside one seed std (verification_axis6 l.63). "Falls below" implies a real effect. The linked axis ⑥ text also never mentions this result (see #24).
   - Fix: "and ties the base model on Qwen1.5-MoE in a three-seed comparison (28.90 ± 1.25 vs 28.93; CIS, 2609.32444)".

10. **11-off-policy.html:33**
    - Text: "lifting their small weights upward collapsed held-out accuracy on Qwen1.5-MoE *from a 28.9 base to 6.0*."
    - Problem: the drop is from 34.78 (CIS) to 6.01, which is *below* the 28.93 base (verification_axis6 l.62). The sentence as written makes 28.9 the starting point.
    - Fix: "…cut the five-benchmark average on Qwen1.5-MoE from 34.8 to 6.0, below the 28.9 of the untrained base."

11. **27-scaling.html:16**
    - Text: "14k → 32k tokens and 512 → 2,048 prompts *each moved A by about 0.035*."
    - Problem: the batch change is 0.605 → 0.645, which is 0.040. Only 768 → 2,048 is 0.035.
    - Fix: "moved A by about 0.035 and 0.04 respectively".

12. **16-prm-credit.html:19**
    - Text: "after one epoch with a success reward plus a learned process reward, … 11.16 …, below its 24.90 starting point and the 30.58 from success reward alone."
    - Problem: the corrections log says 30.58 is SR-only trained for *5 epochs* and 11.16 is SR+PR trained for 1 epoch; they are not the same run. The sentence states the epoch count for one side only, so it reads as a matched comparison.
    - Fix: "…scored 11.16 on MATH after one epoch, below its 24.90 starting point; success reward alone reached 30.58 after five epochs."

13. **11-off-policy.html:59** vs **28-pitfalls.html:38** and **24-code.html:57**
    - Text: "Every framework surveyed ships the gap-1 corrections, but … *all of them leave them off by default*."
    - Problem: pitfall 11 says verl's dataclass defaults rollout IS to *sequence-level* while the YAML sets null, so the default depends on loading path.
    - Fix: "…all of them leave them off in their stock configurations (verl's dataclass and YAML disagree; see pitfalls)."

14. **11-off-policy.html:72** vs **18-case-scalerl.html:38**
    - Text: "FP32 LM head — Large in one forward ablation, *none in a full-recipe one*."
    - Problem: 18 says the full recipe shows "little difference" on the dense 8B but the fix "helps on Scout".
    - Fix: "Large in one forward ablation; little difference in the full recipe on dense 8B, helps on the Scout MoE".

15. **12-data-sampling.html:56**
    - Text: "Small curated datasets — Stability: *Unstable on Llama*."
    - Problem: the body (l.42) says only that one-shot RLVR "barely moves" Llama-3.2-3B (17.5 → 19.0). Nothing about instability is reported.
    - Fix: Stability cell "—"; ceiling cell "Large gains on Qwen-Math; small on Llama".

16. **11-off-policy.html:73** and **17-agentic.html:48**
    - Text: "Up to 2.8× wall-clock (1.5B)" (11) vs "1.5–2.4× reported" for async (17).
    - Problem: the corrections log gives AReaL's figure as 2.77× (1.5B). The 17 range leaves out AReaL, and its endpoints mix baselines: SkyRL's 1.55× is against naive async, and verl's 2.35× is on math.
    - Fix: 11 → "2.77× wall-clock (1.5B)". 17 → "1.5–2.8× reported (verl, AReaL on math; SkyRL 1.55× vs naive async), mostly outside agentic tasks".

17. **20-case-weak-baseline.html:17**
    - Text: "The vanilla PPO both papers start from uses GAE with λ = 0.95 …"
    - Problem: the corrections log notes that VC-PPO's paper is internally inconsistent about the default λ_actor. The sentence states it without a caveat.
    - Fix: add "(VC-PPO's text is not consistent about the actor's default λ; its critic uses 0.95)" or similar.

## B. Evidence-grade mismatches (same claim graded differently)

18. **16-prm-credit.html:28 vs 42**
    - Text: the prose grades "accuracy gains small and shrink with scale" as **D**. The table row "On-policy Monte Carlo or tree credit" carries "**B** small models · **C** at scale".
    - Fix: table grade → "**B** efficiency · **D** accuracy" to match l.26 and l.28.

19. **30-decision-guide.html:17**
    - Text: "With a critic, use λ = 1 *and do not initialize it from the reward model*. **A**"
    - Problem: 06:36 grades "critic not initialized from the RM; value pre-training" as **B**. Only λ = 1 is **A**.
    - Fix: split into "use λ = 1 **A**" and "do not initialize it from the reward model **B**".

20. **05-axes.html:16** (map ③) vs **08-aggregation.html:40**
    - Map: grade **B** only, ceiling "—" (not studied). Axis table: "**B** mechanism · **D** 'token-level is better'", ceiling "Unresolved" — and it *was* studied (ScaleRL LOO, Tricks or Traps).
    - Fix: map grade "B D"; ceiling symbol ◐ or ○, not —.

21. **05-axes.html:19** (map "CISPO or GSPO instead of PPO-clip") vs **09:54–55** and **25:29, 31**
    - Map gives **D** only. Axis ④ and the landscape give "**B** (CISPO: efficiency in large-batch RLVR; GSPO: MoE without replay) · **D** general".
    - Fix: map grade "B D", in keeping with the map's own two-grade convention.

22. **05-axes.html:22** (map: "Several optimizer steps per batch", Ceiling ↓) vs **11-off-policy.html:66**
    - Map marks the ceiling ↓. 11's table says fewer steps give "Similar peaks once stable", i.e. no ceiling effect. 10:53 says "Lower at fixed budget", which is not a ceiling claim either.
    - Fix: map ceiling → ○ or ◐ ("lower at fixed budget"), or change 11's cell.

23. **29-checklist.html:22** vs the grades used throughout
    - Text: "A paper that does not answer the first six cannot be placed … above **C**."
    - Problem: questions 1 and 5 (multiple families, a random-reward control) are failed by most papers graded **B** in the body: Tricks or Traps (14b:40, Qwen3 only, no seeds), ScaleRL batch/length (12:49–50), GiGPO (17:43), Magistral (25:30). Applied literally, the rule downgrades a large share of the survey's own B grades.
    - Fix: "…cannot earn **A**, and its single-study results are graded **B** at most for the setting tested."

24. **06-advantage.html:38** and **25-landscape.html:21**
    - Text: "RLOO vs Dr. GRPO — **A** algebra" / "Equals Dr. GRPO × G/(G−1) **A**".
    - Problem: 02:32 defines **B** as "one controlled study, *or a clear mathematical mechanism*". By that definition an algebraic identity is B, or it needs its own label.
    - Fix: either grade it "exact (algebra)" without a letter, or add a sentence to 02 that algebraic identities are stated without a grade.

25. **25-landscape.html:38** vs **09-trust-region.html:56**
    - Landscape: SAPO "**C**–**D**". Axis ④ table: "GMPO, SAPO — **C**".
    - Fix: make both C (or both C–D with the ARLArena collapse as the D).

26. **30-decision-guide.html:14**
    - Text: "one optimizer step per batch where affordable, *or bounded staleness with sampler-aware corrections* when not. **A**"
    - Problem: the A comes from "fewer steps per batch" (10:53, 11:66). The sampler-aware corrections are graded B (IcePop, decoupled PPO) or D (TIS alone).
    - Fix: split the bullet, or say "**A** for fewer steps; corrections **B**".

## C. Summary stated more strongly than the evidence section

27. **05-axes.html:40**
    - Text: "*The surrogate axis is where the D grades cluster.*"
    - Problem: the map has seven rows carrying D, in axes ① ② ④ ④ ⑥ ⑦ ⑧. Axis ④ holds two of them. They do not cluster there.
    - Fix: "The surrogate axis is where no general gain survives: its named losses are graded D for general claims, which is what the r ≡ 1 identity predicts…"

28. **25-landscape.html:45**
    - Text: "The advantage column *converges after early 2025 on a group mean without standard deviation*."
    - Problem: the table's own post-2025 rows include CISPO ÷ std, GSPO ÷ std, GMPO ÷ std and ScaleRL batch std. 03:129 says more accurately "a group mean, with or without dividing by the group's standard deviation".
    - Fix: "converges on a group mean, with the group-std division increasingly dropped or replaced by a batch-level one".

29. **05-axes.html:38**
    - Text: "The up-arrows in the ceiling column are batch size, generation length, removing train–inference mismatch, routing replay on MoE, and a verifier…"
    - Problem: the list leaves out λ = 1 (map row 1 also has ceiling ↑).
    - Fix: add "λ = 1 for PPO". The "none of them is a new loss" point still holds.

30. **10-kl-entropy.html:60**
    - Text: "the *two strongest results on this axis* are the same result. Reusing a batch… and running at a low temperature…"
    - Problem: low temperature is graded **B** (l.39). The two **A** results are batch reuse and the fixed-entropy-bonus negative result.
    - Fix: "explains why two of the findings on this axis are the same result."

31. **23-patterns.html:19**
    - Text: up-weighting easy prompts "has none [support] and *is a source of instability on easy data*."
    - Problem: 07:34 reports an *efficiency/direction* effect (removing std helps on easy data; LOOP loses points). Instability is not shown.
    - Fix: "…has none, and removing it helps on easy-heavy data (axis ②)."

32. **14-what-rl-changes.html:35**
    - Text: sharpening "is *the most economical explanation* for random, wrong or absent rewards helping some models."
    - Problem: 21:35 labels this reading "ours, not either paper's". 2512.16912's account is reward misalignment. The summary section states it more firmly than the case study.
    - Fix: "is our preferred reading of …, compatible with both published mechanisms (case study)".

33. **14-what-rl-changes.html:36**
    - Text: "Distillation and mid-training expand what the model can solve *far more reliably* than RL does."
    - Problem: the evidence in the survey is one distill-vs-base plot (Yue, Minerva only), Qwen3's pass@64 (93.3 vs 90.0) and OctoThinker (figures only). "Far more reliably" is not measured anywhere.
    - Fix: "…appear to expand coverage where RL does not (Yue et al.'s one distillation comparison; Qwen3's pass@64)".

34. **01-why.html:20** (claim 3 links "verdict map")
    - Text: "…routing replay…, masking tokens the policy did not write (*verdict map*)."
    - Problem: the map has no row for masking environment tokens (it is in #agentic and #decision-guide).
    - Fix: link "masking tokens" to `#agentic`, or add the row to the map.

## D. Internal links / figure references

35. **09-trust-region.html:42** → `#off-policy`
    - The GSPO-on-Qwen1.5-MoE claim links to axis ⑥, which mentions GSPO only on Qwen3-30B-A3B. Either add the Qwen1.5-MoE GSPO number to 11:46, or cite CIS (2609.32444) directly here (see also #9).

36. **24-code.html:62**
    - Text: "<a href='#pitfalls'>*The next sections*</a> collect the cases…"
    - Problem: the next sections are landscape, beyond and scaling; pitfalls comes four sections later.
    - Fix: "<a href='#pitfalls'>The pitfalls section</a> collects…"

37. **29-checklist.html:10**
    - Text: "A PPO baseline at 5 points and a DPO baseline that varies by 5.5 points … (<a href='#case-weak-baseline'>weak baseline</a>)".
    - Problem: the DPO half is in `#tension-baselines`, not the case study.
    - Fix: add "; <a href='#tension-baselines'>baselines</a>".

38. **02-how-to-read.html:10**
    - Text: "the <a href='#case-dapo-table'>case studies</a>"
    - Problem: the link text is plural but the link lands on the second case study.
    - Fix: href → `#case-scalerl` (the first one), or link text "DAPO case study".

39. **31-open.html:8**
    - Text: "Sigmoid and log-linear fits … often indistinguishable (<a href='#case-scalerl'>ScaleRL</a>)"
    - Problem: the finding is 2610.04158's, not ScaleRL's. The link text attributes it to ScaleRL.
    - Fix: "(<a href='#scalerl-cannot'>2610.04158, discussed in the ScaleRL case study</a>)".

40. **01-why.html:19**
    - Text: "…and whether the reward can be fooled (base models, axis ⑦)."
    - Problem: reward-fooling is axis ⑧, which has no link here.
    - Fix: add ", <a href='#reward'>axis ⑧</a>".

41. **24-code.html:9, 27** (source badges)
    - The text pins line numbers to commit `7dc39fe`, but the badges link to `blob/main`, so the line numbers will drift.
    - Fix: link to `blob/7dc39fe/verl/trainer/ppo/core_algos.py#L1329-L1361` (and `#L1168-L1195`).

## E. Wording, terminology and minor inconsistencies

42. **04-tension.html:30 vs 89; 29-checklist.html:9**
    - "about 25 points" (04:30, 29:9) vs "Up to ~26 pts" (04:89). The table value is 25.8.
    - Fix: use "about 26" everywhere, or "about 25" everywhere.

43. **04-tension.html:80, 12-data-sampling.html:14, 30b:17**
    - 04 treats differences "below about 0.02–0.03" as unresolved. 12 calls the group-size spread of 0.025 "just beyond the noise band" and grades it B. Exp 4 uses 0.03.
    - Fix: 12 → "a 0.025 spread, at the edge of the noise band, with no trend".

44. **18-case-scalerl.html:21**
    - The batch-size A values (0.605 → 0.645) are figure-only (Fig. 10a). 12:11 says so; 18 does not, despite the promise in 02:44.
    - Fix: add "(read from a figure)".

45. **18-case-scalerl.html:11**
    - Text: "shown in a figure *without an error figure*." The phrasing is awkward.
    - Fix: "shown in a figure, without an error estimate" (as in 27:27).

46. **30b-experiments.html:7 vs 23**
    - The shared protocol requires "at least five training seeds"; Experiment 10 says "≥ 3 seeds".
    - Fix: make Experiment 10 "≥ 5 seeds", or relax the protocol to "≥ 3".

47. **30b-experiments.html:29**
    - Text: "Experiments 1, 4 and 2 … the first…, the second…, the third…". The ordering is easy to misread.
    - Fix: "Experiment 1 tests the identity…, Experiment 4 the identifiability…, Experiment 2 the transfer…".

48. **30b-experiments.html:25** vs **10-kl-entropy.html:23**
    - Exp 12 predicts "k3 *diverges* at high KL". The body reports only that k3-as-loss "made little progress" in 70B→8B distillation.
    - Fix: "k3 lags or stalls at high KL".

49. **27-scaling.html:18** vs **18-case-scalerl.html:35**
    - 27 lists "the off-policy algorithm" among things that "mostly move B". 18 shows ScaleRL contradicting itself on exactly this (App. A.11: shifts the asymptote).
    - Fix: drop it from the list, or append "(ScaleRL is inconsistent on the off-policy algorithm)".

50. **25-landscape.html:35**
    - Text: ScaleRL "Components move B, not A". That holds in LOO only. The forward comparisons in the same paper move A (loss type, FP32 head, batch, length).
    - Fix: "In LOO, components move B, not A".

51. **23-patterns.html:7**
    - The heading "collapses to REINFORCE at r = 1" uses a different notation from the "r ≡ 1" in 03, 05, 09, 29, 30 and 30b.
    - Fix: "at r ≡ 1".

52. **24-code.html:3**
    - The heading says "ten config flags", but the table lists 9 rows covering about 15 flags.
    - Fix: "a handful of config flags", or "nine settings".

53. **28-pitfalls.html:3**
    - Heading "Pitfalls — 踩坑录 at the algorithm layer". Chinese in an otherwise English page. Harmless if it deliberately mirrors the infra survey; otherwise it is a leftover.
    - Fix: "Pitfalls at the algorithm layer".

54. **09-trust-region.html:44**
    - R1's ε = 10 is cited as plain 2501.12948. The corrections log puts ε = 10 among the v2-only details, and 10:29 cites v2 for the same paper.
    - Fix: cite `2501.12948v2`.

55. **17-agentic.html:30** (BrowseComp **A**)
    - The three numbers are vendor within-report test-time comparisons, not controlled studies in the 02:31 sense. The A is defensible, but a short caveat keeps the grade honest.
    - Fix: append "(three labs' own reports; no seeds)".

---

### Count by category
- A. Factual contradictions: 17 (#1–17)
- B. Grade mismatches: 9 (#18–26)
- C. Summary overstated vs evidence: 8 (#27–34)
- D. Link / figure: 7 (#35–41). All figure numbers and captions are consistent.
- E. Wording / terminology / minor: 14 (#42–55)
- Total: 55
