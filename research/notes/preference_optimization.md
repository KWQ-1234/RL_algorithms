# Preference-based post-training: the DPO family, online/iterative variants, and the RLHF-era PPO pipeline. Which changes are decisive?

## Q1. For each variant, what exactly changes in the loss, what problem does it target, and how strong is the evidence?

### Takeaway
Most offline DPO-family variants are re-parameterizations of the same pairwise contrastive objective. They change the link function (logistic, squared, hinge), the implicit reward (log-ratio, length-normalized log-prob, odds), or add an SFT/NLL anchor or a length term. Three kinds of change have multi-source support: (1) making data on-policy or semi-online, (2) length normalization or length control, and (3) an NLL/SFT anchor when the data is off-policy. For most other variants the headline gains come only from the originating paper. Independent tuned comparisons either shrink those gains to within noise or reverse them (SimPO, KTO, ORPO). PPO done well (advantage normalization, large batch, reference EMA, EOS handling) beats offline DPO by a modest margin. The margin is larger in reasoning and code, and PPO costs about 7–14x more compute.

### Cited Findings

**Reading-depth legend.** [F] = read the method/experiment sections of the arXiv HTML, or grepped them for the cited numbers. [P] = read part of the body. [A] = abstract only. Formulas marked "(SimPO Tab. 3)" were checked against the unified objective table in the SimPO paper ([2405.14734](https://arxiv.org/abs/2405.14734)). Formulas marked "(TRL)" were checked against the HuggingFace TRL DPOTrainer docs ([TRL docs](https://huggingface.co/docs/trl/main/en/dpo_trainer)).

Notation: ρ_θ(y) = log π_θ(y|x) − log π_ref(y|x), and h = ρ_θ(y_w) − ρ_θ(y_l).

#### A. RLHF-era baseline: InstructGPT RM + PPO + KL (+ptx)
- (a) Objective: maximize E[r_φ(x,y) − β log(π/π_SFT)] + γ·E_pretrain[log π] ("PPO-ptx"). The reward model (RM) is a Bradley–Terry (BT) model fit on human comparisons. In the appendix sweep, human Likert scores were poor at KL coefficient 0 and at 2. The optimum was "around 0.01 and 0.02". Of the PPO init variants, only the SFT model with a 10% pretraining mix stood out — [InstructGPT, 2203.02155](https://arxiv.org/abs/2203.02155) [P]
- (c) Outputs of the 1.3B PPO-ptx model were preferred over 175B GPT-3, with 95% CIs from human eval — [2203.02155](https://arxiv.org/abs/2203.02155) [P]
- "N+ implementation details" (Huang et al.): an open reproduction of the OpenAI TL;DR RLHF scaling behavior on Pythia 1B/2.8B/6.9B lists more than 20 details. Examples: disable dropout, otherwise PPO ratios ≠ 1 and the KL estimate is unreliable (Detail 7). Extract the RM score at EOS and apply the "EOS trick" for truncated or no-EOS responses (Detail 23). Optional reward whitening (Detail 24) and advantage whitening (Detail 25). The 2.8B and 6.9B RLHF models beat OpenAI's released 1.3B checkpoint — [2403.17031](https://arxiv.org/abs/2403.17031) [P]
- Xu et al. ablated PPO on CodeLlama-34B APPS pass@5 (Intro/Inter/Comp). Baseline PPO at batch 64 gave 18.0/2.4/1.1, below SFT at 38.6/10.1/3.9. Adding advantage normalization gave 38.1/11.4/4.6. Adding a large batch gave 42.3/14.6/7.5. Adding a reference-model EMA gave 44.4/18.0/9.1. HH-RLHF reward moved only from 0.706 to 0.718 across the same steps. The paper's text cites 33.7% for the small-batch intro baseline, which contradicts Table 3's 18.0 — [Is DPO Superior to PPO, 2404.10719](https://arxiv.org/abs/2404.10719) [F]
- (e) Verdict. Ceiling: highest among the methods here when tuned (see Q2). Efficiency: poor. In Tulu 3's controlled setup, PPO took about 28 h on 2 nodes versus about 4 h on 1 node for DPO, and scored slightly lower — [Tulu 3, 2411.15124](https://arxiv.org/abs/2411.15124) [F]. Stability: hinges on implementation details. Without advantage normalization and a large batch, PPO collapses below SFT on code (Xu). Side effects: length growth, mostly driven by the RM; see the failure modes below. **Grade A** for "a well-tuned PPO ≥ offline DPO". The size of the gap is task-dependent.

#### B. DPO (Rafailov et al. 2023)
- (a) L = −log σ(β·h), the closed-form reparameterization of KL-regularized reward maximization under BT — (TRL); [2305.18290](https://arxiv.org/abs/2305.18290) [A].
- (b) The gradient is weighted by σ(−βh), so pairs the model already orders correctly get small weight. TRL's own docs note that in practice the objective "is typically achieved by suppressing the likelihood of dispreferred completions rather than by increasing the likelihood of preferred ones" — [TRL docs](https://huggingface.co/docs/trl/main/en/dpo_trainer). Feng et al.'s gradient-field analysis finds that the loss "decreases the probability of producing human dispreferred data at a faster rate than it increases the probability of producing preferred data" — [2404.04626](https://arxiv.org/abs/2404.04626) [A]
- (c) The original paper reports DPO matching or beating PPO on IMDb sentiment, TL;DR and single-turn HH at small scale [A].
- (d) Independent evidence: see Q2 for Xu, Ivison, Tajwar, Tang, Lanchantin and Tulu 3.
- (e) Ceiling: below online methods when the data is offline and off-policy. Efficiency: excellent, with no rollouts and no RM. Stability: good, but β and LR are sensitive (HF sweep below). Side effects: likelihood displacement, length exploitation, intra-epoch overoptimization (section K). **Grade A** for "works cheaply"; **Grade A** for "offline DPO < online methods".

#### C. Loss-shape variants (link function, noise, unpaired data)
- **IPO (Azar et al.)**: (a) L = (h − 1/(2τ))² (SimPO Tab. 3). (b) A squared loss regresses the log-ratio margin to a finite target instead of pushing it to infinity. It targets DPO's overfitting when preferences are deterministic (BT logit → ∞) — [ΨPO/IPO, 2310.12036](https://arxiv.org/abs/2310.12036) [A]. (c) Original evidence: "illustrative examples" only, with no LLM-scale benchmark in the abstract. (d) Independent evidence:
  - GPO: tuned squared and logistic losses reach "similar" peak performance "at a similar level of KL" (T5X 700M on TL;DR, PaLM-2 judge). The squared loss needs β≈1 versus ≈10 for the others — [GPO, 2402.05749](https://arxiv.org/abs/2402.05749) [F]
  - HF blog: IPO first looked worse, but that came from a TRL bug (summed instead of averaged loss). After the fix, "IPO matched DPO" in the paired setting — [HF pref-tuning blog](https://huggingface.co/blog/pref-tuning)
  - Rafailov et al.: IPO "is less prone to overoptimization", and the 6.9B IPO model showed little to none — [2406.02900](https://arxiv.org/abs/2406.02900) [F]
  - Gorbatovski: on Llama-3.1-8B, AlpacaEval 2 LC is 28.18 for IPO versus 26.82 for DPO — [2502.01237](https://arxiv.org/abs/2502.01237) [F]
  - (e) Ceiling ≈ DPO; stability slightly better against overoptimization; no length advantage. **Grade A** that IPO is interchangeable with DPO; **B** for its overoptimization advantage (one controlled study).
- **SLiC-HF**: (a) max(0, δ − log π_θ(y_w|x) + log π_θ(y_l|x)) − λ log π_θ(y_w|x). The TRL `hinge` loss variant applies the hinge to the ref-normalized margin — (SimPO Tab. 3; TRL). (b) A ranking hinge plus an SFT anchor. It was the pre-DPO offline alternative to PPO, and it allows off-policy human data — [2305.10425](https://arxiv.org/abs/2305.10425) [A]. (c) TL;DR on T5. (d) GPO finds the hinge ≈ logistic ≈ squared after tuning [F]. In OAIF, SLiC also gains from becoming online. (e) **Grade A** that it is interchangeable with DPO.
- **GPO family** (Tang et al.): (a) L = E[f(β·h)] with convex f: logistic (DPO), squared (IPO), hinge (SLiC), plus exponential, truncated-quadratic and Savage losses. (b) Different f give different implicit regularization strengths. (c) The trade-offs are similar once β is tuned — [2402.05749](https://arxiv.org/abs/2402.05749) [F]. This is the primary "the variants tie" result; its limitation is the scale (T5X ≤ 11B, summarization only).
- **cDPO / robust DPO (rDPO)**: (a) Conservative DPO is label smoothing, (1−ε)·L(y_w,y_l) + ε·L(y_l,y_w). Robust DPO uses the de-biased version [(1−ε)L(y_w,y_l) − εL(y_l,y_w)]/(1−2ε). In TRL, `robust` with `label_smoothing` ∈ [0, 0.5) as the flip rate, typical 0.1 — (TRL). The cDPO formula is from Mitchell's note, which I did not read; the TRL doc confirms only the rDPO parameterization. (b) Targets preference label noise. rDPO proves sub-optimality O(1/(1−2ε)·√(d/n)) under log-linear policies — [2403.00409](https://arxiv.org/abs/2403.00409) [A]. (c) IMDb plus Anthropic HH only. (d) I found no independent LLM-scale ablation. (e) **Grade C.** The mechanism is clear, but adoption evidence is limited to library support.
- **KTO**: (a) L = E[λ_y − v(x,y)], with v = λ_D·σ(β(r_θ − z_0)) for desirable y and λ_U·σ(β(z_0 − r_θ)) for undesirable y, where r_θ = ρ_θ(y) and z_0 = KL(π_θ‖π_ref) is estimated on mismatched pairs and not backpropagated — [KTO, 2402.01306](https://arxiv.org/abs/2402.01306) [F]. (b) A pointwise "human-aware loss" that needs only thumbs-up or thumbs-down labels. The saturating sigmoid bounds per-example gradients. The authors recommend LR 5e-6 versus 5e-7 for DPO at 8B. The paper says that when only KTO skips SFT, response length "stays roughly the same", while DPO without SFT "causes the response length to increase dramatically". (c) The abstract says it "matches or exceeds" preference methods at 1B–30B. (d) Independent evidence contradicts itself:
  - Saeidi et al. find "KTO consistently outperforms the other alignment methods in all three scenarios" (13 benchmarks) — [2404.14723](https://arxiv.org/abs/2404.14723) [P]
  - HF's β sweep finds DPO > KTO > IPO on OpenHermes. On Zephyr, KTO beat DPO in all but one setting, but after the IPO fix DPO ≈ IPO > KTO on paired data — [HF blog](https://huggingface.co/blog/pref-tuning)
  - Eurus: for reasoning, KTO and NCA keep chosen rewards positive while DPO drives them negative, and "DPO hurts model performance on most benchmarks" — [2404.02078](https://arxiv.org/abs/2404.02078) [P]
  - WPO's table on Mistral-Base with the off-policy UltraFeedback (UF) dataset: official KTO checkpoint 14.9 LC versus DPO 20.6 — [2406.11827](https://arxiv.org/abs/2406.11827) [F]
  - Gorbatovski: pointwise DAAs (APO-Zero, NCA, Cal-DPO, ASFT; KTO itself not tested) lose to pairwise ones on Llama-3.1-8B and Mistral-7B (p = 0.014 on 8B) — [2502.01237](https://arxiv.org/abs/2502.01237) [F]
  - (e) Ceiling: ≤ DPO on paired chat data, and possibly better on reasoning when absolute likelihoods matter. Unique benefit: works on unpaired data. **Grade D** (contradictory).
- **NCA / APO (anchored)**: (a) NCA optimizes the absolute likelihood of each response rather than relative likelihoods, which prevents the chosen likelihood from falling. NCA is reported to beat DPO on math and code — [2402.05369](https://arxiv.org/abs/2402.05369) [A]. APO-zero raises winners and lowers losers; APO-down lowers both — (TRL); [2408.06266](https://arxiv.org/abs/2408.06266) [A]. (d) Gorbatovski: pointwise or anchored losses (APO-Zero 23.15, NCA 23.21 LC) trail pairwise ones (DPO 26.82) on UF chat [F]. **Grade D** for chat; **B** for "anchoring helps on reasoning" (Eurus plus NCA, both partly by the same groups).

#### D. Reference-free / SFT-merged variants
- **ORPO**: (a) L = −log p_θ(y_w|x) − λ·log σ(log odds_θ(y_w) − log odds_θ(y_l)), with p_θ(y|x) = exp((1/|y|)·log π_θ(y|x)) and odds = p/(1−p) (SimPO Tab. 3). (b) Merges SFT and preference learning into one stage with no reference model; the odds ratio supplies a "minor penalty for the disfavored style" — [2403.07691](https://arxiv.org/abs/2403.07691) [A]. (c) Phi-2, Llama-2-7B, Mistral-7B on UF; up to 12.20% AlpacaEval 2.0 and 7.32 MT-Bench, i.e. weak absolute numbers. (d) Independent evidence:
  - SimPO's tuned baselines rank ORPO below SimPO [F]
  - WPO table: official ORPO checkpoint 14.7 LC versus DPO 20.6 [F]
  - Gorbatovski: in single-stage form, ORPO needs an explicit SFT stage plus a β to compete. With both, ORPO reaches 28.25 LC versus DPO 26.82, but the authors report "+3.46" in one version and "+13.45" in another for the β-tuning gain — [2502.01237](https://arxiv.org/abs/2502.01237) [F]
  - (e) The "monolithic" claim, its main selling point, is not supported once it is compared against two-stage pipelines. **Grade B** (one controlled study plus consistent tables).
- **CPO**: (a) L = −log σ(β log π_θ(y_w) − β log π_θ(y_l)) − λ log π_θ(y_w), i.e. DPO with π_ref removed plus a behavior-cloning (NLL) term (SimPO Tab. 3). (b) Targets MT, where human references are "adequate but not perfect". ALMA-13B-R reached WMT-winner and GPT-4 level with 22K pairs — [2401.08417](https://arxiv.org/abs/2401.08417) [A]. (d) SimPO's tuned comparison finds CPO sometimes wins Arena-Hard, but with outputs "about 50% longer" [F]. Saeidi: "KTO and CPO don't require SFT" [P]. (e) **Grade C** outside MT; length side effects.
- **SimPO**: (a) L = −log σ((β/|y_w|)·log π_θ(y_w|x) − (β/|y_l|)·log π_θ(y_l|x) − γ). Reference-free, length-normalized, with target margin γ. Recommended β ≈ 2–2.5 and γ ≈ 0.5–1.5, though Llama-3-Instruct sometimes needs β = 10 — [SimPO, 2405.14734](https://arxiv.org/abs/2405.14734) [F]. (b) Aligns the training reward with the generation metric (average log-likelihood) and removes the reference model, giving about 20% less runtime and 10% less memory. (c) SimPO vs DPO, AlpacaEval 2 LC / Arena-Hard:
  - Mistral-Base: 21.5 vs 15.1 LC; 16.6 vs 10.4 AH
  - Mistral-Instruct: 32.1 vs 26.8; 21.0 vs 16.3
  - Llama-3-Base: 22.0 vs 18.2; 23.4 vs 15.9
  - Llama-3-Instruct: 44.7 vs 40.3; 33.8 vs 32.6
  - Baselines were "thoroughly" tuned (DPO β ∈ {0.01, 0.05, 0.1}). No seeds or variance are reported in the parts I read.
  - The "Instruct" setting is on-policy: 5 samples from the SFT model scored by PairRM. With ArmoRM instead of PairRM, performance "significantly improved".
  - The authors acknowledge GSM8K drops ("comparable to or worse than DPO"), that MT-Bench separates methods poorly, and that γ must be hand-tuned.
  - (d) Independent evidence:
    - Gorbatovski, tuned (Llama-3.1-8B UF): SimPO 27.65 LC / 21.5 AH versus DPO 26.82 / 19.0 versus IPO 28.18 / 19.1, i.e. near-ties — [2502.01237](https://arxiv.org/abs/2502.01237) [F]
    - Tulu 3 ablation (Table 18, avg over their eval suite; SFT checkpoint = 55.7): SimPO 51.8–52.9 (below SFT), DPO 55.2, PPO 54.5–55.5, length-normalized DPO 57.3. Tulu 3 picked length-normalized DPO — [2411.15124](https://arxiv.org/abs/2411.15124) [F]
    - The DPO baseline for the same Zephyr Mistral-7B-SFT on UF is 15.1 LC in SimPO's paper and 20.6 ± 0.7 (5 seeds) in WPO's paper — [2406.11827](https://arxiv.org/abs/2406.11827) [F]. That 5.5-point gap is larger than SimPO's claimed margin, and it comes from baseline setup alone.
  - (e) Ceiling: ≈ DPO on chat after tuning, and worse on Tulu 3's broad suite, including math. Efficiency: modestly better. Stability: no KL anchor, so it is more sensitive to LR and γ. Side effects: math regression. **Grade D** (original vs. Tulu 3/Gorbatovski).

#### E. Length-targeting variants
- **R-DPO** (Park et al.): (a) L = −log σ(β·h + α|y_w| − α|y_l|) (SimPO Tab. 3). (b) An explicit length penalty inside the margin. Park et al. link DPO's length exploitation to OOD bootstrapping. On Pythia-2.8B (HH, TL;DR), they report "up to 20% improvement in win rates when controlling for length" — [2403.19159](https://arxiv.org/abs/2403.19159) [P]. (d) Independent evidence: Rafailov et al. find length regularization "changes the KL–win-rate frontier but does not fix overoptimization", and "can underperform the standard objective at low KL". Length is "not the only exploited dimension" — [2406.02900](https://arxiv.org/abs/2406.02900) [F]. SimPO's tuned R-DPO baseline (α ∈ {0.05, 0.1, 0.5, 1.0}) trails SimPO [F]. (e) **Grade B.**
- **Length-normalized DPO (Tulu 3 / OLMo 3)**: (a) L = −log σ((β/|y_w|)·ρ_θ(y_w) − (β/|y_l|)·ρ_θ(y_l)), keeping π_ref, with β = 5 — [OLMo 3, 2512.13961](https://arxiv.org/abs/2512.13961) [F]; [Tulu 3](https://arxiv.org/abs/2411.15124) [F]. (b) Removes the sum-over-tokens incentive for longer chosen responses while keeping the KL anchor. (c) Tulu 3 Table 18: 57.3 vs 55.2 avg for DPO, with large β-sensitivity: β = 2 gives 46.8 and β = 5 at 1 epoch gives 57.3. (d) Apertus: "length normalization consistently improves downstream performance for both algorithms" (DPO and QRPO) — [2509.14233](https://arxiv.org/abs/2509.14233) [P]. (e) **Grade B+**: two labs plus a clear mechanism, but no seeds and a fragile β. LD-DPO (`ld_alpha` in TRL) down-weights tokens beyond the shared length (TRL); not evaluated here.

#### F. Anti-likelihood-displacement variants (anchoring the chosen response)
- **DPO-Positive (Smaug)**: (a) Adds a penalty λ·max(0, log π_ref(y_w) − log π_θ(y_w)) inside the sigmoid, penalizing any drop of the chosen likelihood below the reference. (b) Pal et al. prove standard DPO can lower the chosen likelihood while the ratio rises, especially for low edit-distance pairs — [2402.13228](https://arxiv.org/abs/2402.13228) [A]. Evidence: Smaug-72B was the first open model above 80% on the Open LLM Leaderboard, a benchmark of questionable relevance. **Grade C.**
- **DPO + NLL (RPO / IRPO / Llama 3 / Nemotron-4)**: (a) L = L_DPO + α·(−log π_θ(y_w|x)/|y_w|). Evidence for:
  - Pang et al. call the NLL term "crucial" for iterative reasoning DPO. Llama-2-70B-Chat GSM8K went from 55.6 to 81.6 — [2404.19733](https://arxiv.org/abs/2404.19733) [A]
  - Llama 3 adds NLL with coefficient 0.2 on the chosen response and masks formatting and termination tokens — [Llama 3, 2407.21783](https://arxiv.org/abs/2407.21783) [F]
  - Nemotron-4-340B observed "both chosen and rejected responses' likelihoods drop consistently". MT-Bench gains came with MMLU losses, so they added a weighted SFT loss (weight tuned in [1e-5, 1e-3]) "especially since our preference data is not generated from the reference policy" — [2406.11704](https://arxiv.org/abs/2406.11704) [P]
  
  Evidence against:
  - Lanchantin et al. found no benefit from adding NLL in online or semi-online DPO, "because chosen log probabilities do not substantially decrease" when on-policy — [2506.21495](https://arxiv.org/abs/2506.21495) [F]
  - Razin et al.: CHES filtering or diverse "gold" data mitigates unalignment and "goes beyond the improvement achieved when adding an SFT term" — [2410.08847](https://arxiv.org/abs/2410.08847) [P]
  - (e) Verdict: it helps, and the help is specific to off-policy data. **Grade B.**

#### G. Off-policy correction / data-distribution variants
- **RSO (statistical rejection sampling)**: (a) The loss is unchanged (DPO or hinge). The data changes instead: candidate pairs are drawn approximately from π* ∝ π_SFT·exp(r/β) by rejection sampling against a trained RM. (b) Targets the MLE mismatch, since DPO's estimator assumes pairs are sampled from the optimal policy — [2309.06657](https://arxiv.org/abs/2309.06657) [A]. (c) Beats DPO and SLiC on three tasks, judged by LLM and humans (T5-scale). (d) Xiong et al.'s iterative DPO beats RSO — [2312.11456](https://arxiv.org/abs/2312.11456) [A]. **Grade C.** Conceptually, RSO is the "pairs should come from near the policy" idea in its offline form, which Q2 shows is the dominant factor.
- **WPO**: (a) L = −w(x,y_w)·w(x,y_l)·log σ(β·h), where w(x,y) = exp((1/|y|)·Σ_t log[π_θ(y_t|·) / Σ_v π_θ(v|·)²]). This is "sampled alignment", a calibrated length-normalized sequence probability under the current policy, with no gradient through the weights. In TRL it is `use_weighting` — [WPO, 2406.11827](https://arxiv.org/abs/2406.11827) [F]. (b) Importance-style reweighting toward pairs the policy would actually generate, i.e. it simulates on-policy data. (c) 5 seeds, std reported, significance marked. Mistral-Base off-policy: DPO 20.6 (0.7) → WPO 24.4 (1.4) LC. Llama-3-8B-Instruct off-policy: 28.2 → 33.8. Hybrid (on-policy + UF): DPO 37.9 → WPO 42.0 (Mistral), DPO 44.2 → WPO 48.6 (Llama-3). (d) None found. (e) **Grade C**, though it is better reported than most (5 seeds). Note that WPO's own table shows switching to hybrid on-policy data (+17.3 LC for DPO on Mistral) dwarfs the loss change (+3.8).
- **TR-DPO**: (a) Same loss, but π_ref is updated by soft mixing π_ref ← α·π_θ + (1−α)·π_ref (α = 0.6 default) or a hard reset every τ = 512 steps — (TRL); [2404.09656](https://arxiv.org/abs/2404.09656) [A]. (b) Lets the policy move far from π_0 without overoptimizing. This is the offline analogue of Xu et al.'s reference EMA for PPO (which gave +2–4 pass@5 on APPS). **Grade C** for DPO; **B** that reference updating helps PPO (Xu).

#### H. Online / iterative DPO
- **OAIF (Guo et al.)**: (a) Same DAP loss (DPO, IPO or SLiC), but each step samples two responses from the current policy and labels them with an LLM annotator (PaLM-2). (c) Human raters prefer online over offline DAP versions about 66% of the time. Raters favor online DPO over SFT, RLHF and RLAIF 58.00% of the time in 4-way comparisons on TL;DR. Prompting the annotator to prefer shorter answers cut length from ~120 to ~40 while still improving over SFT — [2402.04792](https://arxiv.org/abs/2402.04792) [P].
- **Iterative DPO (Xiong et al.; RLHF Workflow)**: Theory: KL-regularized contextual bandit, with an online iterative DPO algorithm with exploration and finite-sample guarantees — [2312.11456](https://arxiv.org/abs/2312.11456) [A]. Practice: a BT or preference RM labels best-vs-worst on-policy samples, then DPO runs over the historical data. Settings: η = 0.1, LR 5e-7, batch 128, warm-start from the last iterate with π_0 as reference — [2405.07863](https://arxiv.org/abs/2405.07863) [P]. The model reaches 37.2 LC (gpt4_turbo config); the authors note GPT-based eval "highly depends on the configuration" [P].
- **Self-rewarding LMs**: the policy is its own LLM-as-judge; 3 iterations of DPO on Llama-2-70B. AlpacaEval 2.0 win rate goes 9.94 → 15.38 → 20.44 while average length goes 1092 → 1552 → 2552 chars (numbers from SPPO's Table 1, which reprints them) — [2401.10020](https://arxiv.org/abs/2401.10020) [P]; [2405.00675](https://arxiv.org/abs/2405.00675) [F]. No gains on math or logical reasoning [P]. Meta-Rewarding takes Llama-3-8B-Instruct from 22.9% to 39.4% LC and 20.6% to 29.1% Arena-Hard — [2407.19594](https://arxiv.org/abs/2407.19594) [A].
- **Semi-online DPO vs GRPO (Lanchantin et al., FAIR, 2025)**: Llama-3.1-8B-Instruct, with standard errors over seeds.
  - Math500: offline DPO 53.7, semi-online DPO 58.9 / 57.2, online DPO 58.7, GRPO 58.1
  - Non-verifiable (Athene-RM-8B reward, GPT-4-1106 judge), AlpacaEval LC: offline 53.2, semi-online 81.6, online 83.1, GRPO 75.2
  - Arena-Hard: offline 39.4, semi-online 61.1, online 62.8, GRPO 59.1
  - Caveats noted by the authors: entropy collapses in all on-policy regimes on math, length collapses without ref sync, and length grows under the RM. Checkpoints were selected by length-normalized RM reward — [2506.21495](https://arxiv.org/abs/2506.21495) [F]
- **Tulu 3 on-policy preference data**: completions from the Tulu 3 SFT model plus a model pool, judged by GPT-4o in an UltraFeedback-style pipeline. "Adding on-policy completions from the SFT model improves aggregate DPO performance over a fully off-policy mix" (Figure 11; numbers not extracted). Unused prompts beat reused SFT prompts slightly. The judge barely matters: avg 56.6–57.3 across GPT-4o, Llama-3.1-405B, GPT-4-Turbo, 4o-mini and 70B — [2411.15124](https://arxiv.org/abs/2411.15124) [F]
- (e) Verdict for the online/iterative class: ceiling high (≈ GRPO/PPO in Lanchantin; > offline everywhere). Efficiency: needs generation and a judge or RM, but semi-online syncing helps. Stability: entropy and length collapse risks. **Grade A.**

#### I. General-preference / game-theoretic methods
- **Nash-MD (NLHF)**: (a) Learn a pairwise preference model P(y ≻ y′|x) and find the regularized Nash equilibrium π* = argmax_π min_π′ [P(π ≻ π′) − τ·KL(π‖μ) + τ·KL(π′‖μ)] by mirror descent, sampling against a geometric mixture π_t^{1−ητ}·μ^{ητ}. (b) Targets BT's inability to represent intransitive preferences — [2312.00886](https://arxiv.org/abs/2312.00886) [A]. (c) A T5 summarization experiment. (d) Calandriello et al. prove online IPO (both samples from the current policy, labeled by a preference model) is equivalent to finding the Nash equilibrium via self-play, and propose IPO-MD — [2403.08635](https://arxiv.org/abs/2403.08635) [A]. **Grade B** for the theory; **C** for empirical superiority.
- **SPPO**: (a) Per iteration, L = E[(log(π_θ(y|x)/π_t(y|x)) − η·(P̂(y ≻ π_t|x) − 1/2))²], with P̂ estimated by PairRM against K samples. TRL approximates it with hard labels as `sppo_hard` (TRL). (c) Mistral-7B-Instruct-v0.2 with 60k UF prompts and PairRM-0.4B reaches 28.53 LC (Iter3); Llama-3-8B-Instruct reaches 38.77 LC — [2405.00675](https://arxiv.org/abs/2405.00675) [F]. From the paper's own Table 1, same pipeline: iterative DPO Iter1 23.81 LC versus SPPO Iter1 24.79, iterative IPO Iter1 23.78. The authors attribute IPO's later decline to exploiting PairRM's length bias (lengths 2660–2760 chars). (e) Most of the gain over the base model (17.11 LC) comes from on-policy iteration with PairRM, which iterative DPO also gets (23.81 at Iter1). **Grade C** for SPPO-specific gains.
- **DNO**: (a) A batched on-policy regression toward a Nash/self-improvement target. DNO-Prct is in practice "a properly implemented iterative DPO-like contrastive learning algorithm" with GPT-4-Turbo as the preference oracle, and it includes gold (teacher) responses in the candidate set — [2404.03715](https://arxiv.org/abs/2404.03715) [P]. (c) Orca-2.5-7B goes from 7% to 33% LC win rate, but responses run 2,500–3,000 chars and MT-Bench avg falls across iterations (DNO-More-Data: 7.42 → 6.97) [P]. (e) The gain is mostly distillation from a GPT-4-class teacher and judge, not the objective. **Grade C**; side effect: length.

#### J. BOND (Best-of-N distillation)
- (a) Minimize the Jeffreys divergence (mix of forward and backward KL) between π_θ and the Best-of-N distribution of π_ref under an RM. Quantiles are estimated by Monte Carlo, and an iterative moving anchor (EMA η = 0.02 in J-BOND) amplifies N. (b) Mode-covering plus mode-seeking distribution matching, with a known BoN reward–KL trade-off. (c) Gemma 2B/7B conversational prompts; J-BOND beats REINFORCE baselines on the reward/KL trade-off — [2407.14622](https://arxiv.org/abs/2407.14622) [P]. (d) Adoption without ablation: Gemma 3 post-training uses "improved versions of BOND, WARM, and WARP" — [Gemma 3, 2503.19786](https://arxiv.org/abs/2503.19786). (e) **Grade C** on evidence; the main pipeline adopter is Gemma.

#### K. DPO failure modes and theory
- **Likelihood displacement / decreasing chosen log-prob**:
  - Razin et al. (OLMo-1B, Gemma-2B, Llama-3-8B): preferring "No" over "Never" can raise "Yes". A single-token single-prompt setting already shows pervasive displacement. The CHES score (centered hidden-embedding similarity of y⁺ and y⁻) identifies the responsible samples. In refusal training on SORRY-Bench, Llama-3-8B-Instruct's refusal rate fell from 74.4% to 33.4%. Filtering the top length-normalized-CHES samples (keeping a 5% subset in their figure) or using diverse gold data fixes it — [2410.08847](https://arxiv.org/abs/2410.08847) [P]
  - Ren & Sutherland describe a "squeezing effect": the negative gradient on an already-unlikely y⁻ concentrates mass onto the argmax, so long off-policy DPO makes even desired outputs less likely. They say this explains on-policy DPO's benefit — [2407.10490](https://arxiv.org/abs/2407.10490) [A]
  - Tajwar et al.: when y_w is far from π_ref, the negative gradient increases the margin, but the "recovered probability mass will go into increasing likelihoods of other responses, not y_w" — [2404.14367](https://arxiv.org/abs/2404.14367) [P]
  - Rafailov et al.: declining chosen implicit reward "approximates a forward KL term" and is "not inherently harmful", but the dynamics are non-linear — [2406.02900](https://arxiv.org/abs/2406.02900) [F]
  - The NCA paper also attributes the decrease to DPO optimizing only relative likelihood — [2402.05369](https://arxiv.org/abs/2402.05369) [A]
- **Ranking accuracy**: most preference-tuned models have <60% ranking accuracy on common preference datasets. DPO "is empirically and theoretically ill-suited to fix even mild ranking errors in the reference model" — [Chen et al., 2405.19534](https://arxiv.org/abs/2405.19534) [A]
- **Overoptimization in DAAs (Rafailov et al. 2024)**: Pythia 1B/2.8B/6.9B on TL;DR with DPO, IPO and SLiC over 7 β values. Win rate vs √KL is hump-shaped and fits the Gao et al. form R(d) = d(α − β log d). Under lower KL constraints, most runs peak in the first ~25% of an epoch (the paper's caption and body text disagree on which regime). Implicit-reward accuracy and training loss barely correlate with win rate. With finite data the objective is not strictly convex and has a null space, so there are infinitely many optima, some of which put mass on OOD responses. In a toy tree MDP, DPO, IPO and SLiC move mass to OOD trajectories — [2406.02900](https://arxiv.org/abs/2406.02900) [F]
- **Theoretical OOD exploitation**: Xu et al.'s Theorem 4.1 says the set of PPO-reachable policies is a proper subset of the DPO-reachable set. In a synthetic test, DPO raised an OOD response's probability from 0.11 to 0.23 — [2404.10719](https://arxiv.org/abs/2404.10719) [F]
- **Length exploitation**: Singhal et al. find "even a purely length-based reward reproduces most downstream RLHF improvements", with RMs as the dominant source — [2310.03716](https://arxiv.org/abs/2310.03716) [A]. Park et al. find significant exploitation in DPO, linked to OOD bootstrapping — [2403.19159](https://arxiv.org/abs/2403.19159) [P]. Benchmark consequence: length-controlled AlpacaEval raises Spearman correlation with Chatbot Arena from 0.94 to 0.98 — [2404.04475](https://arxiv.org/abs/2404.04475) [A]. SimPO concedes that MT-Bench separates methods poorly [F].
- **Token-level theory ("From r to Q*")**: DPO in the token-level MDP is "a general inverse Q-learning algorithm" satisfying a Bellman equation, so it performs implicit credit assignment. Likelihood-based search over a DPO policy is equivalent to reward-guided search, and beam search over a DPO policy gives gains. The choice of reference policy explains declining implicit rewards — [2404.12358](https://arxiv.org/abs/2404.12358) [A]
- **Implicit reward generalization**: DPO's implicit RM fits training data like an explicit RM but generalizes worse OOD: mean −3%, max −7% accuracy across five shifts. This argues for an explicit RM in iterative DPO — [2409.03650](https://arxiv.org/abs/2409.03650) [A]

#### L. Reward models, only as they affect this family
- RM quality improvements barely transfer to PPO policies. Ivison et al.: the 13B Mix RM jumps RewardBench from 61.0 to 79.8, yet PPO avg goes 62.2 → 61.6. The 70B UF RM gives the best GSM (58.0) but only +0.6 avg — [2406.09279](https://arxiv.org/abs/2406.09279) [F]
- Generative RMs (GenRM; LLM judging with reasoning traces) match BT in-distribution and beat it by 10–45% OOD — [2410.12832](https://arxiv.org/abs/2410.12832) [A]. GenRM verifiers beat discriminative and DPO verifiers on Best-of-N for reasoning — [2408.15240](https://arxiv.org/abs/2408.15240) [A]. Industrial use: Nemotron 3 Super/Ultra run RLHF against large GenRMs (Qwen3-Nemotron-235B GenRM and a 550B GenRM) — [Nemotron 3 Super, 2604.12374](https://arxiv.org/abs/2604.12374); [NeMo RL docs](https://docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-super.html) (search-snippet level only).
- BT is sufficient but not necessary; any order-consistent objective works. 12,000+ setups were compared — [2411.04991](https://arxiv.org/abs/2411.04991) [A]
- Process vs outcome inside the DAA family: Step-DPO applies DPO at the step level with 10K step-wise pairs and gives about +3% on MATH for >70B models. Self-generated data beats human or GPT-4 data "due to the latter's out-of-distribution nature" — [2406.18629](https://arxiv.org/abs/2406.18629) [A]

### Inferences
- **Verdict table** (ceiling / efficiency / stability / side effects; grade):

| Method | Ceiling vs tuned DPO | Efficiency | Stability | Side effects | Grade |
|---|---|---|---|---|---|
| PPO (well-implemented) | ≥, +0.7 avg (Ivison); larger on code (Xu) | ~7–14× DPO compute (Tulu 3) | fragile without adv-norm / large batch / ref-EMA | RM-driven length | A |
| DPO | baseline | best | β/LR-sensitive; overopt within <1 epoch | displacement, length, OOD mass | A |
| IPO / SLiC / GPO losses | tie | same | IPO less overopt (1 study) | same as DPO | A (tie) |
| cDPO / rDPO | unknown at LLM scale | same | targets label noise | — | C |
| KTO | ≤ on paired chat; maybe ≥ on reasoning | same; unpaired data OK | saturating gradients | stable length w/o SFT (self-reported) | D |
| ORPO | < single-stage; tie only with SFT stage + β | saves SFT stage | — | — | B |
| CPO | unclear outside MT | no ref | — | +50% length (SimPO tab.) | C |
| SimPO | tie (Gorbatovski) to worse (Tulu 3) | ~20% faster | no KL anchor; γ hand-tuned | math drops | D |
| R-DPO / LN-DPO | LN-DPO best in Tulu 3; Apertus agrees | same | very β-sensitive | reduces length; doesn't cure overopt | B / B+ |
| DPOP / +NLL | helps off-policy; no gain on-policy | same | reduces displacement | can over-anchor to chosen | B |
| RSO | ≈ superseded by iterative DPO | needs RM sampling | — | — | C |
| WPO | +4–6 LC (5 seeds, own paper) | no extra gen | — | — | C |
| TR-DPO | unclear | same | allows larger KL | — | C |
| Online / iterative / semi-online DPO | ≈ GRPO; ≫ offline | needs gen + judge | entropy & length collapse | judge length bias | A |
| SPPO / DNO / Nash-MD | ≈ iterative DPO (+~1 LC at Iter1, SPPO table) | same as iterative | — | length growth; MT-Bench decline (DNO) | C |
| BOND | unclear vs PPO at scale | BoN sampling cost | good reward/KL frontier | — | C |
| Delta-learning pairs (data, not loss) | > SFT on chosen; better RL init | cheap (model-pair sampling) | — | — | B |

- Taxonomy: almost every "new loss" changes one of three knobs: (i) the link f (GPO), which is irrelevant after β tuning; (ii) the implicit reward's normalization (length, odds, no-ref), where length normalization matters and reference removal does not clearly help; (iii) an absolute-likelihood anchor (NLL, DPOP, NCA, APO), which matters only when data is off-policy or the task has single-correct answers (math). The decisive change in this whole family is not a loss. It is moving the data distribution onto the policy (RSO → WPO → iterative → semi-online → online).
- The repeated finding that baseline DPO varies by about 5 LC points across papers using the same SFT checkpoint and dataset (15.1 vs 20.6) implies that single-run AlpacaEval differences under ~5 LC between offline variants should be treated as noise, unless seeds and full baseline sweeps are reported.

### Gaps
- Only the abstract was read for: ORPO, CPO, RSO, SLiC-HF, IPO (formula from SimPO Tab. 3), rDPO, Nash-MD, online IPO, TR-DPO, NCA, APO/CLAIR, DPOP, IRPO, Feng et al., Ren & Sutherland, Chen et al., Lin et al., From r to Q*, GenRM papers, Step-DPO, Meta-rewarding, LC-AlpacaEval, Singhal et al., the original DPO paper. Their quantitative claims beyond the abstract are not verified here.
- The cDPO formula comes from Mitchell's 2023 note, which I did not read. Only the TRL label_smoothing/robust implementation was verified.
- The numeric effect size of Tulu 3's on-policy-data ablation (Figure 11) was not extracted, because it is shown only as a figure.
- No seed counts were found for SimPO, Gorbatovski main tables, Tulu 3 ablations or OLMo 3 (OLMo 3 explicitly says "evaluations are from one run only").
- No independent LLM-scale replication was found for rDPO/cDPO, WPO, TR-DPO, BOND or DNO.
- Arena-Hard's own style/length bias, and its later style-controlled version, were not verified in this session.

## Q2. Controlled comparisons: which factor dominates — on-policy-ness of data, data quality, the loss variant, or the reference model?

### Takeaway
Across every controlled study found, the ordering is roughly: on-policy-ness of the data, plus the contrast or quality of the pairs, first. Next is the online vs offline algorithm. Next is the reference/SFT starting point. The specific offline loss variant comes last. Ivison et al. quantify it for one 13B setup: data quality up to +8 pts, PPO vs DPO up to +2.5 (avg +0.7), RM scale mostly marginal, prompts smallest. Lanchantin et al. show that online or semi-online DPO closes the gap to GRPO entirely. So "PPO vs DPO" is mostly a proxy for "on-policy vs offline".

### Cited Findings
- **Xu et al. 2024, "Is DPO Superior to PPO"** (Llama-2-7B on HH-RLHF/SafeRLHF; CodeLlama-7/13/34B on APPS/CodeContests; no seeds reported):
  - PPO "is able to surpass other alignment methods in all cases". CodeContests 34B: PPO 22.4% test versus AlphaCode-41B 15.6%. DPO got 0.0% and DPO-Iter 3.2%, both below SFT at 15.2%.
  - HH OpenAssistant-RM score: PPO 0.718, DPO-Iter 0.678, DPO 0.611.
  - GPT-4 judged PPO vs DPO at 42/28/30 win/tie/loss. Humans judged it 45/26/29.
  - Reference-model distribution shift: on SafeRLHF, re-SFT'ing the reference on in-distribution safe data raised DPO's safety rate from 55.4% to 71.8%.
  - DPO used β = 0.1 and LR 1e-6 (2 epochs). There is no sign of a DPO β sweep, so the baseline-tuning asymmetry is a valid critique — [2404.10719](https://arxiv.org/abs/2404.10719) [F]
- **Ivison et al. 2024, "Unpacking DPO and PPO"** (Tulu 2 13B on Llama 2; same data for both algorithms, downsampled to 60,908 pairs):
  - PPO beats DPO by +0.7 avg, with reasoning +1.3, coding +2.9, safety +2.3 and truthfulness −2.5. PPO is +3.4 on AlpacaEval 2. PPO models use chain-of-thought more often. On UltraFeedback (fine-grained), avg is 61.0 for DPO versus 62.2 for PPO.
  - Data: synthetic per-aspect-annotated data (UltraFeedback, HelpSteer) beats human and web data "up to 15 times larger". Per-aspect annotation matters more than the quality of the completion-generating models.
  - RM: better RewardBench scores give minimal downstream change.
  - Prompts: adding unlabeled math/code prompts did not help overall.
  - Abstract summary: data up to 8%, algorithm up to 2.5% (math) / 1.2% (general), RM up to 5% in math only, prompts smallest — [2406.09279](https://arxiv.org/abs/2406.09279) [F]
- **Tajwar et al. 2024** (Pythia-1.4B/Mistral-7B; AlpacaFarm, UltraFeedback, didactic bandits; DPO LR swept over 1e-7 to 1e-5): methods with on-policy sampling or a negative gradient beat offline maximum-likelihood objectives (Pref-FT, Binary FeedMe). Both are "mode-seeking". The benefit is largest when high-reward responses lie in low-likelihood regions of π_ref, and the two effects are complementary — [2404.14367](https://arxiv.org/abs/2404.14367) [P]
- **Tang et al. 2024 (online vs offline gap)**: IPO-loss experiments show online is clearly better. Offline data coverage and data quality "cannot convincingly explain" the gap. Offline-trained policies are better at pairwise classification but worse at generation. The gap persists for contrastive and non-contrastive losses and is not closed by scaling the policy — [2405.08448](https://arxiv.org/abs/2405.08448) [A/P]
- **Lanchantin et al. 2025**: semi-online DPO ≈ online DPO ≈ GRPO, all ≫ offline DPO, on both verifiable math and non-verifiable chat (numbers in Q1-H). NLL adds nothing on-policy — [2506.21495](https://arxiv.org/abs/2506.21495) [F]
- **Tulu 3**: length-normalized DPO > DPO > PPO ≈ SFT > SimPO on their early-checkpoint suite (Table 18). Adding on-policy SFT-model completions improves DPO over fully off-policy data. The LLM judge choice barely matters (56.6–57.3) — [2411.15124](https://arxiv.org/abs/2411.15124) [F]
- **WPO table as a natural experiment**: switching DPO from off-policy UF to hybrid on-policy data gives +17.3 LC on Mistral-Base (20.6 → 37.9) and +16.0 on Llama-3-Instruct (28.2 → 44.2). The loss change (WPO) adds +3.8 and +5.6 — [2406.11827](https://arxiv.org/abs/2406.11827) [F]
- **SimPO's strongest results also use on-policy data**: the Instruct setting regenerates 5 SFT samples per prompt scored by PairRM. Swapping PairRM for ArmoRM "yields significantly improved performance" — [2405.14734](https://arxiv.org/abs/2405.14734) [F]
- **Pair contrast as a data-quality axis ("delta learning")**:
  - Geng et al.: preference pairs from a 3B (chosen) versus a 1.5B (rejected) model post-train an 8B model to match Tulu 3 on 11 benchmarks. SFT on that weak data hurts — [2507.06187](https://arxiv.org/abs/2507.06187) [A]
  - OLMo 3 Table 21 (7B Think, single run): continued SFT on Qwen3-32B chosen responses drops avg 70.3 → 64.5. DPO preferring Qwen3-32B over Qwen3-0.6B raises it to 72.9 (AIME24 71.0 → 75.7, LCB 67.0 → 72.6).
  - OLMo 3: refreshing the OLMo 2 pipeline with stronger generators "failed to yield any improvements … the chosen and rejected responses no longer had meaningful contrast". Deliberately degrading rejected responses fixed it — [2512.13961](https://arxiv.org/abs/2512.13961) [F]
  - CLAIR/APO: "preference data gives a better learning signal when the underlying responses are contrastive" — [2408.06266](https://arxiv.org/abs/2408.06266) [A]
- **Reference policy**: DPO is sensitive to KL strength. A stronger reference helps "only when it is similar to the model being fine-tuned" — [2407.13709](https://arxiv.org/abs/2407.13709) [A]. Xu et al.'s SafeRLHF re-SFT result above points the same way.
- **Llama 3's industrial A/B**: Meta "explored on-policy algorithms such as PPO" but found DPO "required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval". Its DPO data is "the most recent batches of preference data collected using the best performing models", i.e. near on-policy, over 6 rounds — [2407.21783](https://arxiv.org/abs/2407.21783) [F]
- **SFT data amount** matters less than expected: most DAAs reach about 95% of full-data performance with 5–10% of UltraChat SFT data — [2502.01237](https://arxiv.org/abs/2502.01237) [F]

### Inferences
- **Dominance ordering (my synthesis):**
  1. Data distribution: on-policy or semi-online, plus contrastive pairs. Effects are +15–30 LC (WPO, Lanchantin) or the difference between collapsing and working (Xu code).
  2. Data or annotation quality: up to +8 (Ivison).
  3. Online RL algorithm vs offline DPO given the same data: about +0.7–2.5 (Ivison), and largely recoverable by iterative DPO (Lanchantin, Xu's DPO-Iter).
  4. Reference/SFT starting point: large when there is distribution shift (Xu SafeRLHF +16 safety-rate points), otherwise small.
  5. Offline loss variant: ~0–2 LC after tuning (Q3).
- Xu et al.'s "PPO > DPO" and Llama 3's "DPO > PPO" are compatible. Xu uses offline DPO on fixed datasets against a heavily tuned PPO. Llama 3 uses near-on-policy, multi-round DPO against (presumably) a less tuned PPO at 405B scale. The data regime decides the result, not the algorithm.
- The RM-quality result (Ivison: RewardBench ≠ downstream) and the judge-insensitivity result (Tulu 3) together suggest that once a judge is "good enough", label noise is not the binding constraint. Coverage of the policy's own distribution and pair contrast are.

### Gaps
- No study found that sweeps loss × data on-policy-ness × reference model in one factorial design with seeds at ≥7B.
- Tang et al.'s model sizes and exact numbers were not extracted (abstract plus partial read).
- Ivison's PPO used a 13B base. Whether the +0.7 gap grows or shrinks at 70B+ is unknown from the sources read.
- The Llama 3 PPO-vs-DPO comparison is reported only qualitatively, with no numbers.

## Q3. Which DPO variants are interchangeable once β and learning rate are tuned?

### Takeaway
DPO, IPO, SLiC/hinge and the other GPO convex losses are interchangeable after β tuning (GPO; HF after its bug fix). SimPO, ORPO (given an SFT stage and a β) and IPO land within about 1.5 LC of DPO on Llama-3.1-8B in the only fully re-tuned modern sweep (Gorbatovski et al., ICML 2026). The real divides are pairwise vs pointwise objectives (pairwise wins on chat) and length-normalized vs raw sums (normalization helps). Reference-free SimPO was the worst choice in Tulu 3's suite. Most originating-paper margins are within baseline-tuning noise.

### Cited Findings
- **GPO (Tang et al., Google)**: "the peak performance is similar and is obtained at a similar level of KL divergence" across logistic (DPO), squared (IPO), hinge (SLiC), exponential, truncated-quadratic and Savage losses. Only the optimal β shifts: ≈1 for squared, ≈10 for the others — [2402.05749](https://arxiv.org/abs/2402.05749) [F]
- **Gorbatovski et al. "The Differences Between Direct Alignment Algorithms are a Blur"** (ICLR-W 2025 → ICML 2026): ≥6 β × 4 LRs per method, with β added to ORPO and ASFT, which originally lacked one.
  - Llama-3.1-8B UF, AlpacaEval 2 LC / AH: DPO 26.82/19.0, IPO 28.18/19.1, SimPO 27.65/21.5, ORPO 28.25/20.9, APO-Zero 23.15/17.3, NCA 23.21/15.1, Cal-DPO 23.19/15.2, ASFT 20.82/13.5.
  - The "ranking objective emerges as the primary determinant" (pairwise vs pointwise; p = 0.014 at 8B), while the scalar score type (ratio vs odds) is "secondary". Differences on 3B are mostly not significant — [2502.01237](https://arxiv.org/abs/2502.01237) [F]; [ICML 2026 listing](https://icml.cc/virtual/2026/poster/61150)
- **HuggingFace preference-tuning study** (Zephyr-7B-SFT on UF; OpenHermes-2.5-Mistral on orca_dpo_pairs; β from 0.01 to 0.9, MT-Bench, one seed): "beta matters a lot", and the best β is algorithm- and model-dependent (0.6 DPO, 0.3 KTO, 0.01 IPO on OpenHermes). After the IPO loss bug fix, IPO ≈ DPO. The authors' line: "DPO is the most robust and best performing" — [HF blog](https://huggingface.co/blog/pref-tuning)
- **Tulu 3** (a counter-example on a broad eval suite): only length-normalized DPO beat the SFT checkpoint overall; SimPO fell below it (51.8–52.9 vs 55.7) — [2411.15124](https://arxiv.org/abs/2411.15124) [F]
- **Rafailov et al.**: IPO differs from DPO/SLiC in its overoptimization profile (less of it), not in peak quality — [2406.02900](https://arxiv.org/abs/2406.02900) [F]
- **Baseline sensitivity across papers**: the same Zephyr-SFT + UF DPO baseline scores 15.1 LC in SimPO's paper versus 20.6 ± 0.7 (5 seeds) in WPO's paper — [2405.14734](https://arxiv.org/abs/2405.14734) [F]; [2406.11827](https://arxiv.org/abs/2406.11827) [F]
- **Saeidi et al.**: KTO best across three scenarios — [2404.14723](https://arxiv.org/abs/2404.14723) [P]. This contradicts the HF study and Gorbatovski's pairwise > pointwise finding.
- **Apertus**: QRPO ≈ DPO at 8B; QRPO > DPO at 70B; length normalization helps both — [2509.14233](https://arxiv.org/abs/2509.14233) [P]
- Web search for a 2025–26 study with a uniform tuning budget and seeds across DPO, SimPO, IPO, KTO and ORPO found none beyond Gorbatovski. Newer method papers (LSC-DPO, ξ-DPO, IRPO-EACL 2026) report self-run comparisons that inherit SimPO's protocol — e.g. [LSC-DPO, 2610.07592](https://arxiv.org/html/2610.07592v1) (search-snippet level only).

### Inferences
- **Interchangeable after tuning:** DPO ≈ IPO ≈ SLiC/hinge ≈ other GPO convex losses (Grade A). SimPO ≈ DPO on chat (Grade B, but D overall given Tulu 3). ORPO ≈ DPO only when converted to two-stage with β (Grade B).
- **Not interchangeable:**
  - Pointwise/anchored losses (KTO-like, NCA, APO-Zero, Cal-DPO) vs pairwise losses: pairwise is better on chat, while anchoring may help reasoning (Eurus, NCA).
  - Length-normalized vs raw: Tulu 3 and Apertus favor normalized.
  - Any offline loss vs any on-policy data regime.
- β (and LR) dominates loss choice. The optimal β changes by about an order of magnitude across losses (GPO, HF) and is very sensitive within LN-DPO (Tulu 3: β = 2 → 46.8, β = 5 → 57.3). Any method paper that grid-searches its own hyperparameters more finely than the baselines' will "win".

### Gaps
- Gorbatovski's per-method seed variance and LR-sensitivity tables were not read (only main results and abstract).
- No tuned, seeded, multi-variant comparison at ≥30B was found.
- The HF study used MT-Bench with one seed, which the SimPO authors themselves say discriminates poorly.

## Q4. Where does the field stand in 2025–2026: is DPO still used in frontier pipelines, and for what stage?

### Takeaway
DPO has moved from being the main RLHF replacement (2023–mid-2024: Zephyr, Tulu 2, Llama 3) to a supporting stage. Today it typically sits (i) as a cheap contrastive stage between SFT and RLVR in open recipes (Tulu 3, OLMo 2/3, where it gives a better RL initialization), or (ii) as a light final polish (Llama 4). It also appears (iii) in semi-online form as a cheaper GRPO substitute in research. Labs whose reports emphasize large-scale RL (DeepSeek-V3/R1, Qwen3, Kimi K2, GLM-4.5, Nemotron 3) do not describe DPO in their reports. They use GRPO-style online RL with rule rewards plus RM/GenRM rewards for the preference stage.

### Cited Findings
- **Llama 3 (2024)**: six rounds of RM → rejection-sampling SFT → DPO. DPO (LR 1e-5, β = 0.1, NLL 0.2, formatting-token masking) was chosen over PPO — [2407.21783](https://arxiv.org/abs/2407.21783) [F]
- **Llama 4 (2025)**: "lightweight supervised fine-tuning (SFT) > online reinforcement learning (RL) > lightweight direct preference optimization (DPO)". "SFT and DPO can over-constrain the model, restricting exploration during the online RL stage and leading to suboptimal accuracy, particularly in reasoning, coding, and math". The final DPO is used "to handle corner cases related to model response quality" — [Meta Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- **Tulu 3 (Nov 2024)**: SFT → length-normalized DPO (β = 5; LR 5e-7 at 8B, 2e-7 at 70B; 1 epoch; ~354k pairs mixing on- and off-policy) → RLVR with PPO.
  - 8B: SFT → DPO → final gives AlpacaEval 2 12.4 → 33.5 → 34.5, IFEval 72.8 → 81.1 → 82.4, MATH 31.5 → 42.0 → 43.7, but safety 93.1 → 87.2 → 85.5.
  - 405B: MATH 63.4 → 59.9 after DPO, recovered to 67.3 by RLVR. So DPO can regress math at scale.
  - PPO was used "mainly for RLVR" — [2411.15124](https://arxiv.org/abs/2411.15124) [F]
- **OLMo 3 (Dec 2025)**: Think and Instruct both use SFT → delta-learning length-normalized DPO → RLVR (GRPO variant).
  - Hyperparameters: β = 5; LR 8e-8 (7B Think) / 7e-8 (32B Think) / 1e-6 (7B Instruct); 150K/200K/260K pairs.
  - "DPO yields gains where SFT on the same data cannot". DPO "remains a better starting point" for RL: on the 7B subset (single run), SFT+DPO+RLVR = 74.1 vs SFT+RLVR = 71.9 vs SFT+DPO = 72.7.
  - Length-controlled Instruct DPO cut scores on AIME/MATH, but "ultimately yields … a more performant model post RL", and RL "progresses more reliably" from it — [2512.13961](https://arxiv.org/abs/2512.13961) [F]
  - A separate study finds Olmo-3-7B-Instruct's safety-related values rise in SFT but fall during DPO, mainly via fewer refusals — [MATS/ICLR 2026 short](https://iclr.cc/virtual/2026/10019757) (search-snippet level only)
- **Qwen2.5 (Dec 2024)**: "offline learning DPO and online learning GRPO". Offline RL targets "capabilities that are challenging for the reward model to evaluate, such as reasoning, factuality, and instruction-following". It uses ~150,000 SFT-model-resampled pairs, with checked positives and negatives, trained for 1 epoch — [2412.15115](https://arxiv.org/abs/2412.15115) [F]
- **Qwen3 (May 2025)**: the technical report text contains no mention of DPO. It uses GRPO for reasoning RL (3,995 query–verifier pairs, large batch, off-policy updates, entropy control) — [2505.09388](https://arxiv.org/abs/2505.09388) [grep of full HTML]
- **DeepSeek-V3 (Dec 2024)**: GRPO with rule- and model-based rewards across "coding, math, writing, role-playing, and question answering"; no DPO in the report — [2412.19437](https://arxiv.org/abs/2412.19437) [grep of full HTML]
- **Kimi K2 and GLM-4.5 (2025)**: no DPO in their technical reports; GLM-4.5 describes agentic RL infrastructure — [2507.20534](https://arxiv.org/abs/2507.20534); [2508.06471](https://arxiv.org/abs/2508.06471) [grep of full HTML]
- **NVIDIA**:
  - Nemotron-4-340B (2024) used DPO with an SFT loss, then iterative Reward-aware Preference Optimization (RPO) — [2406.11704](https://arxiv.org/abs/2406.11704) [P]
  - Llama-Nemotron (2025) used GRPO for reasoning, RLOO for instruction following, and "iterative online RPO" for RLHF (Arena-Hard 69.1 → 88.1 in two iterations) — [2505.00949](https://arxiv.org/abs/2505.00949) [F]
  - Nemotron 3 Nano's report HTML mentions no DPO. Super and Ultra (2026) run RLVR, SWE-RL and RLHF with GenRMs, plus multi-teacher on-policy distillation (Ultra) — [2512.20856](https://arxiv.org/abs/2512.20856); [Nemotron 3 Super](https://arxiv.org/abs/2604.12374); [Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007) (Super/Ultra at search-snippet level only)
- **Gemma 3 (2025)**: distillation plus RL "based on improved versions of BOND, WARM, and WARP"; no DPO in the report — [2503.19786](https://arxiv.org/abs/2503.19786) [grep]
- **Apertus (2025)**: length-normalized QRPO (absolute quantile rewards from Skywork-Reward-V2-Llama-3.1-8B), chosen over DPO after an ablation (QRPO ≈ DPO at 8B, > DPO at 70B) — [2509.14233](https://arxiv.org/abs/2509.14233) [P]
- **SmolLM3 (2025)**: per Ai2's write-up, it used a delta-learning-style pair construction (Qwen3-32B chosen vs Qwen3-0.6B rejected) that OLMo 3 converged on. Its use of APO is my recollection, not verified here — [Interconnects on Olmo 3](https://interconnects.ai/p/olmo-3-americas-truly-open-reasoning) (secondary)
- **Research evidence that online DPO still competes**: semi-online DPO ≈ GRPO on math and chat at 8B — [2506.21495](https://arxiv.org/abs/2506.21495) [F]

### Inferences
- The 2025–26 consensus role of DPO is "contrast injection" and "behavior shaping" (style, conciseness, refusals, corner cases) before or after RLVR, not "the RL stage". OLMo 3 and Llama 4 give opposite reasons for keeping it light or using it early: OLMo 3 finds DPO a better RL init, while Llama 4 finds heavy DPO over-constrains exploration. Both point to the same mechanism: DPO sharpens or narrows the policy (mode-seeking per Tajwar; entropy reduction). That helps a short RL budget and hurts a long one.
- Groups with large RL infrastructure (DeepSeek, Qwen, Moonshot, Zhipu, NVIDIA 2026) have absorbed preference learning into online RL with RM/GenRM rewards. This matches the Q2 finding that on-policy-ness, not the loss, is the decisive factor.
- For a 2026 practitioner, the defensible default from this evidence is: length-normalized DPO (β tuned around 5 for LN; LR ≤ 5e-7 at 8B, lower at 30B+) on contrastive, partly on-policy pairs. Add NLL only if the pairs are off-policy, then follow with RLVR or online RL. Expect math to regress temporarily (Tulu 3 405B; OLMo 3 length control).

### Gaps
- The OLMo 2 technical report was not read directly. Its use of the Tulu 3 length-normalized DPO pipeline is inferred from OLMo 3's statement about "the UltraFeedback-style LLM-judge preference pipeline employed in OLMo 2 and Tülu 3".
- The Llama 4 DPO details come only from Meta's blog. The arXiv "Llama 4 Herd" notes (2601.11659) could not be fetched as HTML.
- The Qwen3, DeepSeek and Kimi K2 conclusions rest on the absence of "DPO" in the report HTML. Internal unreported use cannot be ruled out. DeepSeek-R1 and DeepSeek-V3.x reports were not checked.
- The Nemotron 3 Super/Ultra post-training details are from search snippets and NVIDIA docs, not the PDFs.
- No closed-lab (OpenAI, Anthropic, Google Gemini) public detail on DPO usage in 2025–26 was found.
