# Verification: preference-optimization claims (15 items)

Method: arXiv PDFs downloaded fresh (2026-10-14) and extracted with PyMuPDF; Llama 4 blog read as rendered page text in a browser (direct curl was blocked). Page numbers are PDF page indices. Quotes are kept to 25 words or fewer.

---

## 1. GPO (Tang et al., 2402.05749)

- **Similar peak at similar KL across convex losses**: VERIFIED. p13 (Sec. on KL trade-off, Fig. 6): "For different loss functions, the overall trade-off curves look similar. Concretely, the peak performance is similar and is obtained at a similar level of KL divergence." Appendix A.4 (p19) repeats: "all algorithmic variants achieve the peak performance at the same value of KL divergence but with a different value of β."
- **Optimal β, squared ~1 vs. others ~10**: VERIFIED. Fig. 7 caption (p13): "squared loss and truncated squared loss peak at about β = 1 while others mostly peak at slightly larger values β∼10." The losses compared are Logistic, Square, Hinge, Exponential, Truncated square, Savage (Fig. 7 legend). Mapping DPO=logistic, IPO=squared, SLiC=hinge is on p2 and p4. Caveat (p14): the "exponential seems to be slightly better than others", and the authors say this "might not be statistically significant".
- **Setting, "T5X models up to 11B, summarization"**: NEEDS REWORDING. The 11B XXL model is the *golden preference model* (used to relabel the data and act as the judge). The trained policies are smaller. A.4 (p19): "All policies are trained with the Large T5X model (110 million parameters)". The task is a synthetic Gao-style setup on TL;DR summarization (Stiennon et al.). Sweep: 2·10^4 steps, LR 1e-5 and 3e-5 (p12).

## 2. Gorbatovski et al. (2502.01237), "The Differences Between Direct Alignment Algorithms are a Blur"

- **Llama 3.1 8B UF, AlpacaEval 2 LC**: VERIFIED. Table 2 (p6), Llama 3.1 8B UF LC% (std): DPO 26.82 (0.77), IPO 28.18 (0.83), SimPO 27.65 (0.77), ORPO 28.25 (0.71), APO Zero 23.15 (0.76), NCA 23.21 (0.80). The same table also has Cal-DPO 23.19, ASFT 20.82, SFT 10.27.
- **Pairwise vs. pointwise p = 0.014**: VERIFIED, with a caveat. p6: "at 8B all metrics reach p = 0.014, whereas the 3B results are mixed." The test is a one-sided exact permutation test over C(8,4)=70 group assignments. Table 15 (p30) notes p = 0.014 is "the minimum attainable value under this exact test". 8B Δ (pairwise − pointwise): AE LC +5.13, AE WR +7.36, AH +4.85, MT-Bench +0.25. At 3B: AE LC +0.49 (p = 0.057), MT-Bench +0.01 (p = 0.443).
- **Grid size**: VERIFIED (details). App. A.2.2 (p14): "Each method was trained with six or more distinct β values", with 4 LRs per β {3e-7, 5e-7, 7e-7, 1e-6}, all from an UltraChat SFT init. Table 7 (p15) lists the Llama β sets: DPO and IPO 6 values each, SimPO 7, ORPO 6, ASFT 6, APO-Zero 7, Cal-DPO 6. The best LR per β is selected. For SimPO on Llama 3.1 8B, γ/β = 0.5 is fixed and a single LR (6e-7) is used. This works out to roughly 24 or more runs per method on the Llama setups.
- **ORPO needed an SFT stage + β**: VERIFIED. Table 1 (p5), Llama 3.1 8B UF LC: Base→ORPO† (one-stage) 14.8, SFT→ORPO 24.1, SFT→DPO 23.4. p6: "tuning β provided additional gains of +3.46 for ORPO and +8.27 for ASFT on the AlpacaEval 2 LC WR" (Llama 3.1 8B UF).

## 3. Ivison et al., "Unpacking DPO and PPO" (2406.09279)

- **Model Tulu 2 13B**: VERIFIED. p4: "We base our exploration off TÜLU 2 13B".
- **Data quality worth up to ~8 points**: VERIFIED, needs scoping. Abstract: "High-quality preference data leads to improvements of up to 8% in instruction following and truthfulness." The gain is per category, not on the overall average.
- **PPO over DPO: avg +0.7, reasoning +1.3, coding +2.9**: VERIFIED. Table 2 (p5), "Avg. Δ b/w PPO & DPO" row: −0.1 (factuality), +1.3 (reasoning), +2.9 (coding), −2.5 (truthfulness), +2.3 (safety), +0.1 (IF), +0.7 (avg). p6: "PPO improves over DPO by an average of 1.3, 2.9, and 2.3 points for reasoning, coding, and safety". Note that truthfulness drops by 2.5.
- **Bigger/better RM: RewardBench 61.0 → 79.8, PPO avg 62.2 → 61.6**: VERIFIED, needs rewording. Table 3 (p6): 13B UltraF. RM has RewardBench 61.0 and PPO avg 62.2; 13B Mix RM has 79.8 and 61.6. Both of these are **13B**. The change is in the RM's *data* (UltraF → Mix), not its size. For the size change, 70B UltraF. RM gives 73.6 and 62.8, and 70B Mix RM gives 73.9 and 61.8. "Larger reward model" is therefore wrong for the 61.0→79.8 pair. It should say "RM trained on a larger/more diverse data mixture".

## 4. Xu et al., "Is DPO Superior to PPO" (2404.10719)

- **CodeLlama-34B APPS intro/inter/comp, PPO 18.0/2.4/1.1 → 44.4/18.0/9.1**: VERIFIED. Table 3 (p6), APPS pass@5: baseline PPO 18.0% / 2.4% / 1.1%. Adding Adv.Norm gives 38.1/11.4/4.6. Adding Large.Batch gives 42.3/14.6/7.5. Adding Ref.EMA gives 44.4/18.0/9.1. Caption: "Baseline PPO is trained with a batch size of 64 … APPS and CodeContest tasks, the base model utilized is CodeLlama-34B." Context: SFT is 38.6/10.1/3.9, so baseline PPO is far *below* SFT. The metric is pass@5. The three factors are listed on p6 as advantage normalization, large batch size, and reference-model EMA.
- **DPO 0% on CodeContests**: VERIFIED. Table 8 (p8), Code Llama 34B 10@1k: SFT 10.3% valid / 15.2% test; DPO 0.0% / 0.0%; DPO-Iter 3.5% / 3.2%; PPO 19.7% / 22.4%.

## 5. Tulu 3 (2411.15124)

- **LN-DPO 57.3 vs. DPO 55.2 vs. SFT 55.7; SimPO 51.8–52.9; LN-DPO β=2 → 46.8**: VERIFIED. Table 18 (p28; ablation on an early Tulu 3 SFT checkpoint, UltraFeedback only):

  | Algorithm | Settings | Avg |
  |---|---|---|
  | SFT base | | 55.7 |
  | SimPO | β=2, γ/β 0.5 | 51.8 |
  | SimPO | β=10, γ/β 0.3 | 52.9 |
  | DPO | β 0.1, 3 ep | 55.2 |
  | PPO | β 0.0325 | 54.5 |
  | PPO | β 0.05 | 55.5 |
  | DPO-norm | 5e-7, β=2, 3 ep | 46.8 |
  | DPO-norm | 5e-7, β=5, 1 ep | **57.3** |

  The β=2 run (46.8) used 3 epochs. The same LR with β=5 for 3 epochs gives 53.4, and with 1 epoch it gives 57.3. Table 18 also includes PPO.
- **β=5 used**: VERIFIED. Table 20 (p28), final DPO hyperparameters: β = 5 for 8B and 70B, 1 epoch, LR 5e-7 (8B) and 2e-7 (70B), length-normalized variant.
- **PPO vs. DPO compute cost**: VERIFIED. p29: "The PPO runtime is roughly 28 hours using two nodes, whereas the DPO runtime is about 4 hours using a single node." PPO got "comparable … (albeit slightly lower)" scores in that controlled setup.

## 6. Llama 3 (2407.21783)

- **Chose DPO over PPO, and why**: VERIFIED. Sec. 4.1.4 (p16): "We also explored on-policy algorithms such as PPO … but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval". The intro (p2) also says RL algorithms "tend to be less stable and harder to scale".
- **NLL term, coefficient 0.2**: VERIFIED. p16: "We add an additional negative log-likelihood (NLL) loss term with a scaling coefficient of 0.2 on the chosen sequences". Other settings: LR 1e-5, β = 0.1, and formatting tokens masked out.
- **Iterative rounds (6), on-policy-ish**: VERIFIED. Sec. 4.1.6 (p17): "we apply the above methods in six rounds." p16: training uses "the most recent batches of preference data collected using the best performing models from the previous alignment rounds", so the data "conforms better to the distribution of the policy". "On-policy-ish" is a fair paraphrase. The paper itself does not call it on-policy.

## 7. Rafailov et al., "Scaling Laws for RM Overoptimization in DAAs" (2406.02900)

- **Hump-shaped; Gao's form in √KL**: VERIFIED. p4: "performance exhibits a hump-shaped pattern". p5 uses Eq. 5, R(d) = d(α − β log d) with d = √KL, from Gao et al. It "accurately relates d and winrates for DAAs" and halves RMSE compared with a quadratic in KL. Caveat (p5): "a quadratic fit between d and winrates yields a similar error".
- **"Under low KL budgets many runs peak in the first ~25% of the epoch"**: WRONG direction / NEEDS REWORDING. The paper describes this for *wider* KL budgets, i.e. a lower KL constraint (small β). p4: "configurations with wider KL budgets achieve their best performance after training on only 25% of the data". Fig. 2 caption (p5): "Under a lower KL constraint, most experiments reach their best performance in the first 25% of the epoch". Fig. 2 covers 6.9B models only.
- **Pythia 1B/2.8B/6.9B**: VERIFIED. p4: "three model sizes - 1B, 2.8B, and 6.9B", Reddit TL;DR, with DPO/IPO/SLiC and seven β values each. The appendix adds Gemma2-2B and Anthropic HH.

## 8. OLMo 3 (2512.13961)

- **Delta-learning DPO 72.9 vs. SFT-on-chosen 64.5 vs. SFT start 70.3**: VERIFIED. Table 21 (p50), "Subset of Olmo 3 Think Benchmarks", Avg.: Qwen3 32B (chosen) 83.2; Qwen3 0.6B (rejected) 35.1; Dev. 7B SFT ckpt 70.3; Cont. SFT on chosen 64.5; Delta learning 72.9. This is the Think (thinking) model.
- **SFT+DPO+RLVR 74.1 vs. SFT+RLVR 71.9 (7B, single run)**: VERIFIED. Table 22 (p50): SFT 70.1, SFT + DPO 72.7, SFT + RLVR 71.9, SFT + DPO + RLVR 74.1. Caption: RLVR is "for 1000 steps … on our 7B model … evaluations are from one run only." The SFT row here reads 70.1, while Table 21's dev checkpoint reads 70.3, so the two tables do not share the same baseline number.
- **Length-controlled DPO lowers AIME/MATH initially but is better after RL**: VERIFIED. This is the Instruct section, p61: length control comes "with lower scores on length-sensitive evaluations—particularly math benchmarks such as AIME and MATH". It "ultimately yields to a more performant model post RL" and was conjectured to be due to the 8K RL context at 7B. The length control works by filtering chat/multi-turn pairs to a chosen−rejected difference of ≤100 tokens (p59). No numeric table was found for the post-RL comparison; the support is prose plus figures.

## 9. Meta Llama 4 blog (ai.meta.com/blog/llama-4-multimodal-intelligence/)

- **Pipeline lightweight SFT → online RL → lightweight DPO**: VERIFIED. "lightweight supervised fine-tuning (SFT) > online reinforcement learning (RL) > lightweight direct preference optimization (DPO)."
- **Over-constrain quote**: VERIFIED, with a wording fix. The exact text is: "A key learning was that SFT and DPO can over-constrain the model, restricting exploration during the online RL stage and leading to suboptimal accuracy". The blog says "SFT and DPO", not "heavy SFT/DPO". The remedy described is removing >50% of data tagged easy (95% of the SFT data for Behemoth), followed by lightweight SFT.

## 10. SPPO (2405.00675)

- **Iter-1 AlpacaEval LC 24.79 vs. iterative DPO 23.81, same table**: VERIFIED. Table 1 (p16, AlpacaEval 2.0) has both rows. Mistral-7B-SPPO Iter1: LC 24.79 (+7.69), WR 23.51, length 1855. Mistral-7B-DPO Iter1: LC 23.81, WR 20.44, length 1723. Context from the same table: DPO Iter2/3 are 24.23/22.30, IPO Iter1 is 23.78, and SPPO Iter3 is 28.53.

## 11. DNO (2404.03715)

- **"Response length grows and MT-Bench declines across iterations"**: WRONG as stated / NEEDS REWORDING.
  - *Length*: the paper reports a spike after iteration 1, followed by a decrease, not steady growth. p15: "lengths explode by at least a factor of 2 over the initializing SFT model, before inching down again in the next iteration." Table 2 (p13), student length in words: Iter0 162, Iter1 671, Iter2 361. In DNO-More-Data (Table 4, p15), length in chars is 1174 (SFT) → 2795, 2782, 2539, 3033, 2856, 2683, which is non-monotonic after the jump.
  - *MT-Bench*: there is no monotone decline. Table 4 MT-Bench avg: SFT ep1 6.88 → Iter1 6.53, Iter2 7.42, Iter3 7.21, Iter4 7.24, Iter5 6.75, Iter6 6.97. Over the same run, AlpacaEval LC rises 8.96 → 33.05. The fair claim is: "MT-Bench does not track the AlpacaEval gains (it peaks at iter 2 and drifts down)". Table 1 shows only final iterations (DNO Iter 3: LC 22.59, WR 24.97, MT 7.48). The authors note "the relatively low correlation between MT-Bench and AlpacaEval 2.0" (p15).

## 12. Lanchantin et al. (2506.21495), "Bridging Offline and Online RL for LLMs"

- **NLL term gives no gain on-policy**: VERIFIED, scoped. p8: "We experiment with adding the NLL term to our online and semi-online DPO configurations in the verifiable task. We did not observe any benefits". The explanation given is that chosen log-probs "do not substantially decrease during training in our case". Setup: Llama-3.1-8B-Instruct, verifiable math, NLL scale 1.0 (App. B.2), Fig. 10. The result covers both online *and semi-online* runs, on the verifiable task only.

## 13. Razin et al., "Unintentional Unalignment" (2410.08847)

- **DPO can decrease the likelihood of preferred responses and shift mass to undesired responses; CHES score**: VERIFIED. Abstract: likelihood displacement "can be catastrophic, shifting probability mass from preferred responses to responses with an opposite meaning". Example: preferring No over Never "can sharply increase the probability of Yes". Llama-3-8B-Instruct refusal rate went 74.4% → 33.4%. CHES = "centered hidden embedding similarity" score, used to filter the samples that drive displacement. Wording nuance: the paper stresses *opposite-meaning/harmful* responses. It calls displacement "benign" when the mass goes to similar-meaning responses (p2). Models: OLMo-1B, Gemma-2B, Llama-3-8B.

## 14. Nemotron-4 340B (2406.11704)

- **Both chosen and rejected likelihoods decrease in DPO**: VERIFIED. p12: "we observe both chosen and rejected responses' likelihoods drop consistently with their gap increasing, even if chosen responses are high-quality."
- **Used RPO / added SFT loss**: VERIFIED, but the attribution needs care. The *SFT loss* on chosen is the stated mitigation for the likelihood drop and overfitting (p12–13, weight tuned in [1e-5, 1e-3]). *RPO* is motivated separately, by DPO ignoring the reward gap and "unnecessarily 'unlearning' high-quality rejected responses" (p13). The pipeline is DPO followed by 3 iterations of RPO, and RPO keeps the chosen SFT loss (coefficient 1e-5).

## 15. DPO (Rafailov et al., 2305.18290)

- **Loss −log σ(β(ρ_w − ρ_l))**: VERIFIED. Eq. 7 (p4): L_DPO = −E log σ(β log πθ(yw|x)/πref(yw|x) − β log πθ(yl|x)/πref(yl|x)), with ρ = log πθ/πref.
- **Gradient weight σ(−βh)**: VERIFIED. p5: ∇L = −β E[σ(r̂(x,yl) − r̂(x,yw)) (∇log π(yw|x) − ∇log π(yl|x))], where r̂ = β log πθ/πref. So the weight is σ(−β h) with h = ρ_w − ρ_l, times an overall β prefactor. The paper notes that dropping the weight "can cause the language model to degenerate".

---

## Compact list: WRONG / NOT FOUND / needs rewording

1. **GPO (#1)**: NEEDS REWORDING. The 11B T5X-XXL is the golden preference model/judge. The trained policies are T5X-Large at 110M. Do not say "models up to 11B were aligned".
2. **Ivison (#3)**: NEEDS REWORDING. RewardBench 61.0→79.8 with PPO 62.2→61.6 compares two **13B** RMs (UltraF → Mix data), not a *larger* RM. The 13B→70B comparison is UltraF 61.0→73.6 RewardBench, PPO 62.2→62.8. "Up to 8%" is per-category (IF and truthfulness).
3. **Rafailov scaling laws (#7)**: WRONG direction. Early peaking (first ~25% of the epoch) is for *wider* KL budgets / *lower* KL constraint (small β), not low KL budgets. It is shown for 6.9B.
4. **DNO (#11)**: WRONG as stated. Length spikes ≥2× after iter 1, then inches down, so it is not steady growth. MT-Bench is non-monotone (6.53, 7.42, 7.21, 7.24, 6.75, 6.97 over 6 iterations of DNO-More-Data), with no consistent decline. Say instead that it decouples from the AlpacaEval gains.
5. **Llama 4 blog (#9)**: NEEDS REWORDING. The exact quote is "SFT and DPO can over-constrain the model, restricting exploration during the online RL stage". The word "heavy" is not in the blog.
6. **Tulu 3 (#5)**: minor. The LN-DPO β=2 → 46.8 run was 3 epochs. The 57.3 run is β=5 for 1 epoch, while β=5 for 3 epochs gives 53.4. Mention the epochs if β is blamed.
7. **Nemotron-4 (#14)**: minor. The SFT loss is the remedy for the likelihood drop. RPO addresses reward-gap-blind overfitting/unlearning of good rejected responses.
8. **Gorbatovski (#2)**: minor. p = 0.014 is the floor of an exact 70-assignment permutation test. It holds at 8B only, and 3B is mixed.
9. **Lanchantin (#12)**: minor scope. The result covers online and semi-online DPO on verifiable math, with Llama-3.1-8B-Instruct and NLL scale 1.0.
10. **OLMo 3 (#8)**: minor. These are Think-model ablations. The single-run note is correct. The length-control post-RL benefit is supported by prose only, with no numeric table found.

No item was NOT FOUND. All numeric cells in claims 2, 3, 4, 5, 8 and 10 match the PDFs exactly.
