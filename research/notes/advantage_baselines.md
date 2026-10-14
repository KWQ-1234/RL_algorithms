# Advantage estimation, baselines, critics and advantage normalization in LLM RL post-training

Scope note. The window is PPO (2017) through 2026-10-14. Reading depth is flagged per source. "[full]" means the method and experiment sections were read via arxiv.org/html or the PDF text. "[partial]" means part of the HTML was read and figures were not. "[abstract]" means only the abstract was read. "[snippet]" means only search-result snippets were seen. "[code]" means a local framework checkout was read. Derivations marked "(my derivation)" are mine, not the paper's.

Shared notation. A prompt x has G samples y_1..y_G with scalar rewards r_i. For binary rewards, p̂ is the group success rate and p is the true pass rate. S_i = ∇ log π(y_i|x). "Group" means per-prompt and "batch" means across all prompts in the step.

---

## Q1. What is the exact estimator for each method, and what does it change mathematically (bias, variance, implicit prompt reweighting)?

### Takeaway
Almost every critic-free estimator in use reduces to Â_i = (r_i − b(x)) · w(x). Here b is a per-prompt baseline: group mean, leave-one-out mean, greedy reward, quantile, or length-weighted mean. w is a per-prompt scale: 1, 1/std, 1/mean, or a global batch std. The baseline mainly controls variance, and is unbiased only when it does not depend on y_i itself. The scale w(x) changes the objective being optimized: it reweights prompts by pass rate. PPO/GAE variants differ from these in a second way: whether the advantage is bootstrapped from a learned V (λ<1) or is just a Monte-Carlo return minus a learned baseline (λ=1).

### Cited Findings

#### PPO + learned critic + GAE (Schulman 2017; GAE 2015)
- (a) Estimator. Â_t = Σ_l (γλ)^l δ_{t+l}, with δ_t = r_t + γV(s_{t+1}) − V(s_t). The DeepSeek-R1 appendix restates that PPO "typically" computes advantages with GAE from rewards and a learned value model [full, PDF text]. — [DeepSeek-R1, arXiv 2501.12948 App. A.3](https://arxiv.org/abs/2501.12948)
- With γ=λ=1 and only a terminal reward, the advantage reduces to R − V(s_t) and the value target is the terminal reward R. Open-Reasoner-Zero (ORZ) uses exactly this, plus batch-level advantage normalization and no KL. — [ORZ, arXiv 2503.24290](https://arxiv.org/abs/2503.24290) [full]
- Reward-signal decay. With λ<1, the end-of-sequence reward reaches token t scaled by λ^(T−t), which is "essentially zero" for early tokens in long CoT. — [VC-PPO, arXiv 2503.01491](https://arxiv.org/abs/2503.01491) [full]
- Classic-RL baseline for comparison. Andrychowicz et al. (250k agents, 5 continuous-control envs) recommend "GAE with λ=0.9". They found per-minibatch advantage normalization "seems not to affect the performance too much", and found separate (non-shared) wide value networks better on 4 of 5 envs. — [What Matters in On-Policy RL, arXiv 2006.05990](https://arxiv.org/abs/2006.05990) [partial, ar5iv]
- λ for LLMs, first piece of evidence: on RLHF with Llama-7B/HH, a lower λ (more bias) gave monotonically lower reward, and λ=1.0 was best. — [Back to Basics/RLOO, arXiv 2402.14740](https://arxiv.org/abs/2402.14740) [full]
- λ for LLMs, second piece: DeepSeek ran PPO vs GRPO on MATH with DeepSeek-Coder-V2-Lite (16B MoE, 2.4B active). "When λ is set to 0.95 (the default …), PPO performs considerably worse than GRPO… setting λ to 1.0, PPO's performance improves substantially, nearing that of GRPO." — [DeepSeek-R1 App. A.3, Fig. 4](https://arxiv.org/abs/2501.12948) [full, PDF text]
- λ for LLMs, third piece: in ORZ's ablation on ORZ-7B, λ=1.0 reward "rose quickly and kept growing". λ=0.95 progressed more slowly and its response length collapsed. No numbers are given. — [ORZ](https://arxiv.org/abs/2503.24290)
- Critic initialization bias. Initializing V from the reward model, which was trained only on EOS, gives low values to early tokens. That produces a positive advantage bias that grows with position, followed by length collapse. — [VC-PPO](https://arxiv.org/abs/2503.01491)

#### VC-PPO (ByteDance Seed, Mar 2025)
- (a) Two changes:
  - Value pretraining: freeze π_sft and train V with λ=1 (Monte-Carlo targets) for about 100 steps.
  - Decoupled GAE: λ_critic=1.0 and λ_actor=0.95 to 0.99.
- (b) Mechanism. Value pretraining removes the initialization bias. λ_critic=1 makes the value target unbiased. λ_actor<1 is a variance-reduction choice for the policy.
- (c) Original evidence. Qwen2.5-32B with a small cold-start SFT, verl, ±1 reward, KL=0. Train set is past AIME problems plus synthetic hard problems; test is the two most recent AIME years. 8k context for ablations. Seeds and steps are not reported. — [arXiv 2503.01491](https://arxiv.org/abs/2503.01491) [full]
  - Ablation (AIME pass@1): PPO 5.6, VC-PPO without decoupled GAE 29.4, VC-PPO 41.9.
  - λ_actor sweep (AIME pass@1): 0.9 → 34.3, 0.95 → 41.3, 0.99 → 41.9, 1.0 → 29.4.
  - Value pretraining steps (AIME pass@1): 50 → 20.6, 100 → 30.1, 150 → 25.1.
  - At 16k context: GRPO 38.9 vs VC-PPO 48.8 AIME pass@1. GPQA is slightly worse for VC-PPO (48.8 vs 49.4).
- Confounders:
  - The PPO baseline at 5.6 is a known-broken configuration (RM-initialized critic, λ=0.95).
  - The test set is "recent AIME", which is small and high-variance.
  - The value-pretraining step sweep is non-monotonic and noisy-looking.
- (d) Independent evidence. GenAC (Qwen3-8B-Base, DeepScaleR, 200 steps, shared PPO framework) found VC-PPO only marginally above the critic-free baselines. — [GenAC, arXiv 2604.10701](https://arxiv.org/abs/2604.10701) [full]
  - Single run: VC-PPO 50.39 avg@16, vs GRPO 50.22 and RLOO 49.93.
  - Over 3 seeds: VC-PPO 50.85 vs GRPO 49.96.
  - VC-PPO was more sample-efficient early, but "VC-PPO's advantage shrinks over time".

#### VAPO (ByteDance Seed, Apr 2025)
- (a) VC-PPO plus several additions. — [arXiv 2504.05118](https://arxiv.org/abs/2504.05118) [full]
  - Length-adaptive GAE: λ_policy = 1 − 1/(αl), with α=0.05.
  - Token-level loss.
  - Clip-higher: ε_low=0.2, ε_high=0.28.
  - Positive-example NLL loss with μ=0.1.
  - Group sampling: 512 prompts × 16 samples.
- (b) Mechanism. Length-adaptive λ makes Σλ^t = α·l, so each token's effective TD horizon scales with response length. My derivation: for l=8000, λ≈0.9975 and the effective horizon is about 400 tokens. The final reward reaches token 0 with weight about e^−20. So at early tokens the actor's advantage is almost entirely critic-bootstrapped. This makes VAPO a genuinely critic-dependent estimator, unlike ORZ's λ=1, where the critic acts only as a baseline.
- (c) Original evidence. Qwen2.5-32B, no SFT. AIME24 avg@32 from ablations, apparently single runs:

  | Configuration | AIME24 avg@32 |
  |---|---|
  | Vanilla PPO | 5 |
  | w/o value pretraining | 11 |
  | w/o decoupled GAE | 33 |
  | w/o length-adaptive GAE | 45 |
  | w/o clip-higher | 46 |
  | w/o token-level loss | 53 |
  | w/o positive-example LM loss | 54 |
  | w/o group sampling | 55 |
  | VAPO | 60.4 |
  | DAPO (same team) | 50 |
  | R1-Zero-Qwen-32B | 47 |

  - VAPO reached its score in about 5,000 steps, about 60% of DAPO's update steps.
  - The authors report peak 60–61 across "three repeated experiments", with no crashes.
- (d) Independent evidence. I found no external replication of VAPO's AIME numbers. Seed1.5-Thinking uses VAPO for its actor-critic line and DAPO for its critic-free line. It describes both as giving "robust and consistent training trajectories" and makes no head-to-head comparison. — [Seed1.5-Thinking, arXiv 2504.13914](https://arxiv.org/abs/2504.13914) [partial]

#### VinePPO (Kazemnejad et al., Oct 2024)
- (a) Estimator. V̂_MC(s_t) = (1/K) Σ_k R(η_k) from K=9 extra rollouts launched at each step boundary. Steps are split at newlines/punctuation. Â = r + V̂(s_{t+1}) − V̂(s_t), and one advantage is shared by all tokens of a step.
- (b) Mechanism. The value estimate is unbiased Monte-Carlo, so there is no learned-critic bias, but variance depends on K. This trades extra generation compute for per-step credit assignment.
- (c) Original evidence. DeepSeekMath-7B and RhoMath-1.1B (SFT'd), on MATH and GSM8K. — [arXiv 2410.01679](https://arxiv.org/abs/2410.01679) [partial]
  - The PPO critic "barely outperform[s] a random baseline when comparing alternative steps". This holds even though the critic's explained variance is 0.7–0.9.
  - VinePPO beats PPO in wall-clock: 3.0× faster for 1.1B and 1.51× faster for 7B.
  - Per-iteration cost is up to 5× PPO (1.1B) and 2× PPO (7B).
  - Training seeds are not stated.
- (d) Independent evidence. GenAC prices VinePPO at 4.3× PPO per iteration. — [GenAC](https://arxiv.org/abs/2604.10701)
  - A 2026 paper names a "Value Flattening" failure: critic predictions stay flat while Monte-Carlo values vary sharply across states. It proposes supervising V on only about 3 states per response (SP³O) and reports gains on Qwen3-Base. — [arXiv 2609.18708](https://arxiv.org/abs/2609.18708) [abstract]

#### REINFORCE (vanilla) and REINFORCE with moving-average baseline
- (a) Estimator. ∇J = E[(R − b) S], with b = the average reward over training steps.
- (c) Back to Basics (TL;DR win rates): REINFORCE with baseline 70.7, vanilla PG 70.4, PPO 67.6, RLOO k=4 77.9. — [arXiv 2402.14740](https://arxiv.org/abs/2402.14740)
- (d) Minimalist (Qwen2.5-Math-7B, average over Math500/Minerva/Olympiad): Reinforce 53.9, GRPO 56.3, PPO 52.5. — [arXiv 2504.11343](https://arxiv.org/abs/2504.11343) [full]
  - BASIS reports that REINFORCE "can collapse during training" (Qwen3-4B, DAPO-Math-17K). — [arXiv 2605.27293](https://arxiv.org/abs/2605.27293) [snippet]

#### ReMax (Li et al., Oct 2023)
- (a) Estimator. Â = r(x, y_sample) − r(x, y_greedy).
- (b) Mechanism. The baseline is independent of the sampled response, so the estimator is unbiased. The authors concede variance can increase in the worst case.
  - My inference: with binary rewards, Â ∈ {−1, 0, +1}, and every correct sample on a prompt the greedy decode already solves gets zero signal.
  - It needs an extra greedy generation per prompt.
- (c) Original evidence. Llama-2-7B on full-hh-rlhf and Mistral-7B-Instruct, with RLHF reward models. — [arXiv 2310.10505](https://arxiv.org/abs/2310.10505) [partial]
  - About 46% GPU-memory saving vs PPO (Section 5.2 says "about 50%").
  - 1.6× faster per epoch.
  - Mistral-ReMax reaches AlpacaEval 94.78% and MT-bench 7.739.
  - The authors say PPO "might achieve similar results" but they lacked tuning compute. No seeds are reported.
- (d) Independent evidence. In REINFORCE++ Table 1 (Llama-3-8B RLHF, run by another group), ReMax scores 45.1 vs GRPO 46.8. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262)
  - Reasoning-era usage is rare. verl still ships `remax`. — [code: verl/verl/trainer/ppo/core_algos.py:88-110]

#### RLOO (Ahmadian et al., Feb 2024; Kool et al. 2019)
- (a) Estimator. Â_i = r_i − (1/(G−1)) Σ_{j≠i} r_j.
  - The paper's own formula: (1/k) Σ_i [R(y_i) − (1/(k−1)) Σ_{j≠i} R(y_j)] S_i. — [arXiv 2402.14740](https://arxiv.org/abs/2402.14740) [full]
- (b) Mechanism, my derivation. Â_i^RLOO = (G/(G−1))(r_i − r̄), so RLOO is exactly Dr. GRPO's advantage times a constant. The leave-one-out baseline does not depend on y_i, so the estimator is unbiased. The plain group mean includes r_i, which introduces an O(1/G) bias that the scale factor corrects.
  - Dr. GRPO itself notes the equivalence "up to a scaling factor". — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- (c) Original evidence. Pythia-6.9B on TL;DR and HH, and Llama-7B on HH. Win rates on TL;DR / HH-Pythia / HH-Llama:
  - RLOO k=4: 77.9 / 43.7 / 64.1
  - PPO: 67.6 / 29.2 / 32.0
  - DPO: 66.6 / 39.0 / 61.9
  - Confounders: PPO was tuned only over LR {1e-6, 1e-5}, no seeds are reported, and the PPO result on HH-Llama (32.0) looks under-tuned.
- (d) Independent evidence.
  - GenAC: RLOO 49.93 vs GRPO 50.22. — [arXiv 2604.10701](https://arxiv.org/abs/2604.10701)
  - REINFORCE++ Table 1 (RLHF): RLOO 44.6 vs GRPO 46.8. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262)

#### GRPO (DeepSeekMath 2024; DeepSeek-R1 2025)
- (a) Estimator. A_i = (r_i − mean(r)) / std(r), applied to every token, with a clipped-ratio loss and a KL to a reference model. — [DeepSeek-R1 Eq. 13](https://arxiv.org/abs/2501.12948)
  - The original GRPO also uses 1/|o_i| per-sequence length normalization. — [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- (b) Mechanism.
  - The std term makes questions with low reward std, whether too easy or too hard, "given higher weights". — [Dr. GRPO](https://arxiv.org/abs/2503.20783)
  - Population-level prompt weight: w(p) = 1/√(p(1−p)), which upweights both tails. REINFORCE/RLOO has w(p)=1 and maximum likelihood has w(p)=1/p. — [MaxRL, arXiv 2602.02710](https://arxiv.org/abs/2602.02710) [partial]
  - My derivation: E[∇] ∝ Σ_x ∇p_x / √(p_x(1−p_x)) = ∇ Σ_x 2·arcsin(√p_x). So GRPO's binary-reward objective is Σ arcsin√p_x, not Σ p_x.
  - Worked example, my derivation, G=8 and one success: p̂=1/8, std≈0.331, A_+≈+2.65, A_−≈−0.38. Under Dr. GRPO the same group gives A_+=+0.875 and A_−=−0.125. At p̂=1/2 both schemes give |A|=1 vs 0.5.
- An effective-loss view: mean+variance calibration turns GRPO into a weighted contrastive loss against samples from the previous policy. The success-probability recurrence has a fixed point above the reference policy's success rate ("success amplification"). — [Mroueh, arXiv 2503.06639](https://arxiv.org/abs/2503.06639) [abstract]

#### Dr. GRPO (Liu et al., Mar 2025)
- (a) Changes. It removes the std term (Ã_i = r_i − mean(r)). It replaces the per-response 1/|o_i| with a constant such as the generation budget, which recovers an unbiased PPO objective.
- (b) Mechanism. Removing the length term removes a bias that "prefer[s] lengthier responses among incorrect ones". Removing std removes the difficulty bias.
- (c) Original evidence. Qwen2.5-1.5B with the R1 template, MATH questions. Max length 3000, G=8, LR 1e-6, 3 independent runs in App. C. Steps are not stated. — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783) [full; numbers only in figures]
  - Main text: Dr. GRPO "improves token efficiency while maintaining reasoning performance". GRPO keeps lengthening responses even after reward plateaus.
  - Appendix C claims "statistically significant improvements – both in token efficiency and final accuracy".
  - The final Oat-Zero-1.5B averages 42.1.
- Confounders. The two changes are ablated mostly jointly in the main text.
  - Same paper: Qwen2.5-Math-7B gains about 60% simply from removing the template (23.8 → 38.2 average). The authors hypothesize the model was pretrained on concatenated QA text. This is the canonical "Qwen2.5-Math-specific" confounder.
- (d) Independent evidence: see Q4.
  - Adopted in OLMo 3 (no std). — [arXiv 2512.13961](https://arxiv.org/abs/2512.13961) [full PDF text]
  - OpenRLHF `dr_grpo` is mean-centering only. — [code: OpenRLHF/openrlhf/trainer/ppo_utils/experience_maker.py:267]

#### REINFORCE++ and REINFORCE++-baseline (Hu et al., Jan 2025)
- (a) REINFORCE++ (k=1):
  - A_t = r − β Σ_{i≥t} KL_i, with the token-level KL folded into the return.
  - Global-batch normalization: A^norm = (A − mean_batch)/std_batch.
- (a) REINFORCE++-baseline (k>1): A′ = R − mean_group(R), then global batch normalization (A′ − mean_batch)/std_batch, plus a k2 KL loss. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262) [full]
- (b) The authors' argument. Per-prompt std normalization is biased because the centred reward and the local std are dependent. It explodes when std→0 with small G, and it overfits easy prompts.
- (b) My inference from code. OpenRLHF computes the batch mean and std over action tokens (`(all_advantages*mask).sum()/num_actions`). For reinforce_baseline, the sequence-level group-centred advantages therefore have a token-weighted mean ≠ 0 whenever correct and incorrect responses differ in length. Subtracting it adds a length-correlated shift to every advantage. — [code: OpenRLHF experience_maker.py:315-327]
  - For k=1 REINFORCE++, the "baseline" is just the batch-average reward. That is a prompt-independent baseline, so its variance is higher than any per-prompt baseline on mixed-difficulty batches.
- (c) Original evidence. No seeds, and LR, β and λ are not reported. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262)
  - Llama-3-8B RLHF on Chat-Arena-Hard: REINFORCE++ 46.7 vs GRPO 46.8 vs RLOO 44.6 vs ReMax 45.1.
  - Qwen2.5-Math RL-zero: AIME24 pass@8 21.04 vs 18.96, MATH500 72.0 vs 73.0.
  - ZeroTIR tool use average: RF++-baseline 24.10 vs GRPO 22.58 vs PPO 21.85.
  - A "30 AIME-24 training questions" overfitting test: GRPO 95.0 on train and 0.0 on AIME-25.
- (d) Independent evidence.
  - Logic-RL (K&K puzzles) reports REINFORCE++ > GRPO on most metrics. The R1-replication survey says Logic-RL found PPO (λ=γ=1) best. — [arXiv 2502.14768](https://arxiv.org/abs/2502.14768) [snippet]
  - Tool-Star finds the differences "relatively minor". — [arXiv 2505.16410](https://arxiv.org/abs/2505.16410) [snippet]
  - ScaleRL and Tricks-or-Traps: see Q4.

#### Batch-level vs group-level normalization (ScaleRL; Tricks or Traps; Magistral)
- ScaleRL's final estimator is Âᵢ = rᵢ − mean_group(r), then Â/std_batch(Â). — [ScaleRL, arXiv 2510.13786](https://arxiv.org/abs/2510.13786) [partial; Fig. 14b is image-only]
- Tricks or Traps' "Lite PPO" uses group mean + batch std. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221) [full]
- Magistral uses group mean and no group std, then a minibatch (mean+std) normalization. — [arXiv 2506.10910](https://arxiv.org/abs/2506.10910) [full]
- (b) Mechanism, my inference. Dividing by one batch-wide std is a per-step global rescaling. It does not change relative prompt weights, so the implicit objective stays Σ p_x like RLOO. Under Adam, a global per-step scalar is largely absorbed, except through interactions with gradient clipping, clipping ε, and Adam's EMA when the scale drifts. This is why its measured effect is small.

#### OPO (Hao et al., May 2025)
- (a) Baseline. The verl implementation uses a length-weighted mean, b(x) = Σ_i |y_i| r_i / Σ_i |y_i|, with plain r_i − b and no std. — [code: verl/verl/trainer/ppo/core_algos.py:640-690]
- (b) Mechanism. This is the variance-optimal constant baseline if ‖∇log π(y)‖² ∝ |y| (the optimal-baseline form b* = E[‖S‖² r]/E[‖S‖²]). The paper pairs it with strictly on-policy training (one gradient step per batch). — [arXiv 2505.23585](https://arxiv.org/abs/2505.23585) [abstract + code]
- (c) Claims "superior performance and training stability", lower policy shift and higher entropy on math. I read no numbers.
- (d) Independent evidence. In OTB's comparison, OPO scored 27.29 on AIME25 vs GRPO 25.31 and OTB 30.31. These are single peak scores. — [arXiv 2602.07078](https://arxiv.org/abs/2602.07078) [full]

#### Optimal Token Baseline, OTB (Li et al., Dec 2025 / Feb 2026, ICML 2026)
- (a) Estimator. A per-token baseline B̂_t = Σ_i G_t^(i) Ŵ_t^(i) / Σ_i Ŵ_t^(i). Here W_t is the cumulative "realized energy" Σ_{j≤t} ‖e_{y_j} − π_j‖² = Σ (1 − 2π(y_j) + ‖π_j‖²), a logit-gradient proxy that needs only a forward pass.
- (c) Original evidence. Qwen3-8B-Base on DAPO-Math-17k (G=16, 8k length), and Qwen2.5-7B tool-integrated reasoning (TIR). — [arXiv 2602.07078](https://arxiv.org/abs/2602.07078) [full]
  - AIME25 peak: OTB 30.31 vs GRPO 25.31.
  - TIR: OTB 28.13 vs GRPO 18.54.
  - "N=4 OTB ≈ N=32", with 66–68% token savings.
  - No seeds and no confidence intervals. The proxy rests on an isotropic-backbone assumption.
  - Shipped in verl as `optimal_token_baseline` and `tir_optimal_token_baseline`. — [code: verl core_algos.py:108-109, 870]

#### KRPO (Kalman-filter baseline, Wang et al., May 2025)
- (a) Estimator. A 1-D Kalman filter runs sequentially over the samples within a group: x̂_{i|i} = x̂_{i|i−1} + K_i(r_i − x̂_{i|i−1}), with Q=1e-5 and R=1e-2. Then Â_i = (r_i − x̂_{i|i})/(√P_{i|i} + ε). — [arXiv 2505.07527](https://arxiv.org/abs/2505.07527) [full]
- (b) Critique, my derivation.
  - x̂_{i|i} includes r_i itself, so the baseline depends on the action and the gradient is biased.
  - r_i − x̂_{i|i} = (1 − K_i)(r_i − x̂_{i|i−1}), which shrinks advantages.
  - √P_{i|i} depends only on Q, R and P_0, never on rewards, so the "normalization" is a fixed schedule over sample index.
  - The baseline is order-dependent within a group.
- (c) Original evidence. Llama-3.2-1B-Instruct on a single A6000.
  - OpenMath-hard: 33.98 → 51.86.
  - MATH500: 24.1 → 30.2.
  - Datasets are named "MATH500" and "AIME 1983–2024" as training sets, so train/test overlap is unclear.
  - One main seed; seeds 777 and 1234 were shown as curves only.
  - Grade at best C, with methodological red flags.

#### Quantile Advantage Estimation, QAE (Wu et al., Sep 2025, ICLR 2026)
- (a) Estimator. b_K(x) is the K-quantile of the group rewards (K=0.4). For binary rewards, b=0 if p̂ ≤ 1−K and b=1 otherwise. Then A_i = (r_i − b)/(std + ε).
- (b) Mechanism. On hard prompts only successes get signal (positive). On easy prompts only failures get signal (negative). About 80% of samples get zero advantage. The paper claims a two-sided entropy bound under first-order softmax updates.
- (c) Original evidence. Qwen3-8B/14B-Base and Qwen3-30B-A3B. — [arXiv 2509.22611](https://arxiv.org/abs/2509.22611) [full]
  - Qwen3-8B AIME24: 39.69 → 48.23 over DAPO clip-higher.
  - AIME25: +2.2.
  - On top of KL-Cov it is roughly flat.
  - No seeds reported.

#### Pass@k advantage shaping (2025–2026)
- Chen et al. derive closed-form advantages from N rollouts for the max-over-k reward. — [arXiv 2508.10751](https://arxiv.org/abs/2508.10751) [partial]
  - R̄ = 1 − C(N_neg,k)/C(N,k) and σ = √(R̄(1−R̄)).
  - A_pos = (1−R̄)/σ.
  - A_neg = [1 − R̄ − C(N_neg−1,k−1)/C(N−1,k−1)]/σ.
  - The total absolute advantage peaks at about 25% accuracy (k=8), vs about 50% for pass@1, so hard prompts are emphasized.
  - Qwen2.5-7B on Enigmata: pass@1 goes 12.9 (pass@1 training) → 17.9 (pass@k training) → 30.8 (pass@k then pass@1).
- PKPO gives unbiased low-variance pass@k gradient estimators for any k ≤ n, plus k-annealing. Tested on Gemma-2, no numbers in the abstract. — [arXiv 2505.15201](https://arxiv.org/abs/2505.15201) [abstract]
- Thrampoulidis et al. show that advantage-shaping hacks and direct pass@k gradients are equivalent: shaping implicitly maximizes a surrogate reward, and "hard-example upweighting" equals reward-level regularization. TMLR 2026. — [arXiv 2510.23049](https://arxiv.org/abs/2510.23049) [snippet]
- verl ships `grpo_passk`, where only the best response gets a non-zero advantage r_max − r_2nd. It cites arXiv 2503.19595. — [code: verl core_algos.py:472-497]

#### MaxRL (Tajwar et al., Feb 2026, ICML 2026 oral)
- (a) Estimator. ĝ(x) = (1/(N p̂)) Σ_j (r_j − p̂) S_j, i.e. advantage (r − p̂)/p̂ (my reading of Alg. 1). The gradient is 0 if p̂=0.
  - Prompt weight w(p) = (1 − (1−p)^T)/p, which tends to 1/p (maximum likelihood).
- (c) Original evidence. Qwen3-1.7B/4B-Base on POLARIS-53K, 1000 steps, 16 rollouts. — [arXiv 2602.02710](https://arxiv.org/abs/2602.02710) [partial]
  - Pass@k beats GRPO in 7 of 8 settings.
  - "2.3×–19.2×" test-time-compute speedups.
  - No seeds in the visible text.
- (d) Independent evidence. A CS224R report finds MaxRL fragile under reward-label noise. — [search summary](https://cs224r.stanford.edu/projects/pdfs/Petru%20Cristian%20Budianu%20Nicolas%20Bejar%20Arambula%20submission_416215724/CS224R___Final_Project_Report__2_.pdf) [snippet]

#### A*-PO (Gao et al., May 2025)
- (a) Two stages. Offline, estimate V* from reference-policy samples. Online, run least-squares regression with one generation per prompt.
- Claims up to 2× faster training and over 30% less peak memory vs PPO/GRPO/REBEL, with "competitive" math performance. — [arXiv 2505.20686](https://arxiv.org/abs/2505.20686) [abstract only; formulas not read]

#### Other 2026 variants seen only as snippets (not evaluated)
- CoRPO clips the baseline at a correctness threshold so that wrong answers never get positive advantage. — [arXiv 2511.04439](https://arxiv.org/abs/2511.04439) [snippet]
- HA-DW is a difficulty-anchored reweighting. — [arXiv 2601.08521](https://arxiv.org/abs/2601.08521) [snippet]
- Kernelized Advantage Estimation claims 60–70% lower value-estimate MSE than GRPO. — [arXiv 2604.28005](https://arxiv.org/abs/2604.28005) [snippet]
- GDPO normalizes each reward dimension separately within the group before summing (multi-reward). It is in verl as `gdpo`. — [arXiv 2601.05242](https://arxiv.org/abs/2601.05242); [code: verl core_algos.py:362]
- Asymmetric Prompt Weighting argues that optimal weights upweight low p when successes are rare. — [arXiv 2602.11128](https://arxiv.org/abs/2602.11128) [snippet]

### Inferences

Verdicts per method. Columns are ceiling, efficiency, stability and side effects; grade A–D as defined in the brief.

| Method | Ceiling | Efficiency | Stability | Side effects | Grade |
|---|---|---|---|---|---|
| GAE λ=1 (vs 0.95) for long CoT | Decisive vs a broken default; λ=0.95 underperforms | — | λ<1 → length collapse | Length | **A** (RLOO paper, DeepSeek-R1 App. A.3, ORZ, VC-PPO all agree) |
| Value pretraining / RM-free critic init | Decisive within PPO (VAPO 60→11 without it) but only vs PPO | — | Fixes early collapse | Length | **B** (one group, clear mechanism) |
| Decoupled GAE (λ_c=1, λ_a<1) | +12–27 AIME points within the same team's PPO | — | — | — | **B/C** (same team twice) |
| Length-adaptive GAE | +15 in a single VAPO ablation | — | — | — | **C** |
| VC-PPO/VAPO vs well-tuned critic-free | Not decisive: GenAC 50.4–50.9 vs GRPO 50.0–50.2; the original 10-point gaps compare against weaker configurations | Value-based is more sample-efficient early (GenAC) | Claimed better; no independent test | Higher cost | **D** (original +10 vs independent ≈0) |
| VinePPO | Not shown at reasoning-era scale | Fewer steps, 2–5× cost per step | — | — | **C** |
| ReMax | ≈ others in RLHF | Memory saving vs PPO only | — | Wasted signal with binary rewards | **C** |
| RLOO ≡ Dr. GRPO ×G/(G−1) | Interchangeable with GRPO | — | — | — | **A** for the equivalence (math); **B** for empirical parity |
| GRPO std term | Not decisive for final accuracy | Possibly faster at small scale | Can blow up on easy data | Lengthening (via 1/\|o\|, not std) | **D** (see Q4) |
| Dr. GRPO length fix | Small | Token efficiency | — | Shorter responses | **B** (mechanism clear; adopted by OLMo 3, DAPO-style token-level loss) |
| Batch-std normalization | ≈ group-std ("similar", "marginally better") | ≈ | Possibly more stable | — | **B** (ScaleRL, Tricks or Traps, Magistral all "small/no effect") |
| REINFORCE++ k=1 global baseline | ≈ GRPO in RLHF | Saves G× sampling | Claimed lower KL | — | **C** |
| OPO, OTB, KRPO, QAE, MaxRL, A*-PO | Each claims gains | — | — | — | **C**; KRPO **C/D** for methodology |
| Pass@k shaping | Raises pass@k at large k (several groups) | — | — | Pass@1 effect mixed | **B** for pass@k; **C** for pass@1 |

- Unifying view (my derivation). For binary rewards, the estimators differ mainly through the implied prompt weight w(p): RLOO/Dr. GRPO/batch-std use 1, GRPO uses 1/√(p(1−p)), MaxRL uses roughly 1/p, pass@k uses roughly k(1−p)^{k−1}, and QAE is a hard asymmetric threshold. These are different objectives (Σp, Σ2arcsin√p, Σlog p, Σ1−(1−p)^k). Their "differences" should show up as pass@1 vs pass@k trade-offs and hard-vs-easy prompt emphasis, not as large changes in asymptotic pass@1. That is what the controlled studies report.
- GRPO's upweighting near p→1 has no defender. Every 2025–26 variant that touches it either removes it (Dr. GRPO, OLMo 3, DeepSeek-V3.2, Magistral) or makes it asymmetric toward hard prompts (MaxRL, QAE, pass@k). The defensible half of GRPO's std weighting is the upweighting of low-p prompts.

### Gaps
- Exact numbers for ScaleRL's advantage-normalization ablation (Fig. 14b) are image-only. I could not read A and B values for each variant, or the per-ablation seed counts.
- Dr. GRPO's GRPO vs Dr. GRPO accuracy numbers are figure-only, and the separate contributions of std removal and length removal are not tabulated in the text I read.
- A*-PO, PKPO, OPO, value-flattening (2609.18708), Mroueh (2503.06639) and 2601.23135 were read at abstract level only.
- I found no independent replication of VAPO's 60.4, of VinePPO at reasoning-era scale, or of OTB/QAE/KRPO.
- I could not confirm whether the Minimalist paper's vanilla Reinforce used ±1 or {0,1} rewards. This matters for interpreting "dropping all-wrong prompts".

---

## Q2. What controlled comparisons exist that hold everything else fixed, and what did they find on final accuracy, sample efficiency and stability?

### Takeaway
Truly controlled, multi-seed comparisons of advantage estimators are rare. The best ones (ScaleRL, GenAC, the Minimalist paper, Magistral, the DeepSeek-R1 appendix) all find that GRPO, RLOO, Dr. GRPO, batch-normalized variants and a well-configured PPO end within about 1–2 points of each other on final accuracy. The larger gaps reported in method papers come from comparing against an untuned or broken baseline, most often PPO with λ=0.95 and an RM-initialized critic.

### Cited Findings
- **ScaleRL** (Meta, 2025). 8B dense model, Polaris-53K, 48 prompts × 16 generations, about 3.5–4k GPU-hours per ablation, more than 400k GPU-hours total. — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786) [partial]
  - Prompt-level, batch-level and no normalization "yield similar performance". Batch-level was chosen as "theoretically sound and marginally better".
  - Its framing is that normalization, loss aggregation, curriculum and off-policy choice "primarily modulate compute efficiency without materially shifting the asymptote". Exceptions: loss type, and FP32 at the LM head (A 0.52 → 0.61).
  - Leave-one-out runs: all methods have A within ±0.02. Across three runs, A varied by at most ±0.015.
  - ScaleRL did not test a critic.
- **GenAC** (2026). Qwen3-8B-Base, DeepScaleR, 200 steps, the same PPO-style loss for all methods, only the advantage estimator changed. — [arXiv 2604.10701](https://arxiv.org/abs/2604.10701) [full]
  - avg@16 over 6 benchmarks: GRPO 50.22, RLOO 49.93, VC-PPO 50.39, GenAC 51.90.
  - 3 seeds for GRPO (≈49.96, spread about 0.4) and VC-PPO (50.85). GenAC was not replicated.
  - Value-based methods are more sample-efficient early. VC-PPO's lead shrinks by the end.
  - Cost per iteration: GRPO 0.6× PPO, GenAC 2.1×, VinePPO 4.3×.
  - Hyperparameters "largely follow verl's example scripts". There was no per-method tuning.
- **Minimalist / RAFT++** (2025). Qwen2.5-Math-7B and Llama-3.2-3B-Instruct, Numina-Math, 1024 prompts per iteration, n=4. — [arXiv 2504.11343](https://arxiv.org/abs/2504.11343) [full]
  - Qwen average: Reinforce 53.9, PPO 52.5, RAFT++ 56.1, GRPO 56.3, Reinforce-Rej 56.4.
  - Llama average: 24.2, 26.9, 27.6, 28.4, 28.5 in the same order.
  - "GRPO's main advantage arises from discarding prompts with entirely incorrect responses". Std normalization "yields little additional gain over simply removing bad samples".
  - Training only on positives (RAFT++) collapses entropy faster.
- **DeepSeek-R1 App. A.3.** PPO λ=0.95 is clearly worse than GRPO, and PPO λ=1.0 is "nearing that of GRPO". — [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
  - DeepSeek-Coder-V2-Lite on MATH. A single figure; no seeds stated in the text I read.
- **Magistral.** Minibatch vs group vs no normalization: "we did not observe any significant effects on evaluation performance or the growth of the length". — [arXiv 2506.10910](https://arxiv.org/abs/2506.10910) [full]
- **Tricks or Traps.** Qwen3-4B/8B and their Base variants, ROLL framework, 5k prompts at easy/medium/hard levels, 500 steps, seed 42, single curves. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221) [full]
  - Removing std helps on easy data, where low std amplifies advantages and can cause "gradient explosions".
  - On hard data, mean-only and std-normalized versions are similar.
  - Group mean + batch std beats group mean + group std at both sizes.
  - Lite PPO (group mean + batch std + token-level loss) "outperforms" GRPO and DAPO on base models. No numbers are given in the text.
- **"Why GRPO Needs Normalization".** Per-prompt std normalization beats unnormalized and globally normalized variants at 1.5B, but only in a mid-training phase. At 7B the schemes are "statistically indistinguishable". GSM8K and MATH. — [arXiv 2601.23135](https://arxiv.org/abs/2601.23135) [abstract]
- **Back to Basics (RLHF).** RLOO > RAFT > REINFORCE ≈ vanilla PG > PPO on win rate. PPO was tuned only over LR {1e-6, 1e-5}. — [arXiv 2402.14740](https://arxiv.org/abs/2402.14740)
- **REINFORCE++ Table 1.** REINFORCE++ 46.7, GRPO 46.8, ReMax 45.1, RLOO 44.6 on Arena-Hard. Single runs; authors' own method. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262)
- **ORZ.** On ORZ-7B, GRPO "becomes unstable around step 240" (repeat and truncate rates reach 1.0) while PPO stays stable. This is one run per method, and the GRPO configuration is not described as tuned. — [arXiv 2503.24290](https://arxiv.org/abs/2503.24290)

### Inferences
- Results line up across ScaleRL, GenAC, Minimalist, Magistral and the DeepSeek-R1 appendix: on final accuracy, the advantage estimator is second-order once the critic is not broken and zero-variance groups are handled. The remaining effects are on efficiency (early sample-efficiency of value methods) and stability in specific regimes (std blow-up on easy data; GRPO length/repetition drift in ORZ).
- The four studies that report large estimator effects all have a soft baseline:
  - VAPO/VC-PPO compare against vanilla PPO scoring 5 to 5.6.
  - Back to Basics' PPO is under-tuned.
  - KRPO runs on a single GPU with a 1B model.
  - REINFORCE++ reports no seeds.
- Confounder checklist for the reasoning-era studies:
  - Qwen2.5-Math base: Minimalist and part of REINFORCE++.
  - Qwen3 only: Tricks or Traps, GenAC, QAE, OTB, MaxRL.
  - AIME-only headline numbers: VC-PPO, VAPO.
  - Short training: GenAC 200 steps, Tricks or Traps 500 steps, Dr. GRPO.
  - Single seed: most of them.

### Gaps
- I found no study that runs GRPO vs RLOO vs Dr. GRPO vs REINFORCE++-baseline vs tuned PPO on one base model with ≥3 seeds and long horizons (thousands of steps). ScaleRL comes closest but does not include a critic.
- I found no controlled critic-vs-critic-free comparison on a non-Qwen family at more than 8B parameters.

---

## Q3. Is the PPO-vs-GRPO-vs-RLOO-vs-REINFORCE++ difference decisive or interchangeable, and is the main real benefit of critic-free methods memory/compute? When does a learned value function help?

### Takeaway
For single-turn RLVR with outcome rewards they are largely interchangeable on final accuracy. The decisive benefits of going critic-free are memory, compute and having no λ or critic-initialization hyperparameters to tune. A learned critic has a credible, but not yet independently confirmed, edge in early sample efficiency. It is plausibly decisive only where per-step credit assignment matters: long multi-turn agentic horizons, and even there it is usually re-cast at turn level.

### Cited Findings
- **Costs.**
  - DeepSeek-R1: the value model "is usually of similar size as the policy model", and PPO "demands additional computational cost for hyperparameter optimization". GRPO is "a more practical alternative… with constrained resources". — [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
  - ReMax saves about 46% GPU memory vs PPO and runs 1.6× faster per epoch. — [arXiv 2310.10505](https://arxiv.org/abs/2310.10505)
  - GenAC per-iteration cost: GRPO 0.6× PPO; total-FLOPs estimate GRPO ≈ 4.2k vs PPO ≈ 7.5k vs VC-PPO ≈ 8.8k (units M). — [arXiv 2604.10701](https://arxiv.org/abs/2604.10701)
- **Hyperparameter fragility of PPO in long-CoT RL.**
  - PPO needs λ=1. — [DeepSeek-R1](https://arxiv.org/abs/2501.12948); [ORZ](https://arxiv.org/abs/2503.24290)
  - It also needs value pretraining (VAPO: 60 → 11 without it) and decoupled λ. — [VAPO](https://arxiv.org/abs/2504.05118); [VC-PPO](https://arxiv.org/abs/2503.01491)
- **Arguments against a critic in long CoT.**
  - Kimi k1.5 excludes the value network on purpose. A critic would penalize erroneous intermediate steps that the model later recovers from, and the authors argue those exploratory steps are "extremely valuable" for learning long CoT. — [arXiv 2501.12599](https://arxiv.org/abs/2501.12599) [partial]
  - Kimi K2 keeps the same mean-baseline objective. — [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) [partial]
- **Arguments for a critic.**
  - VAPO cites more precise credit assignment, lower variance and better sample reuse. — [arXiv 2504.05118](https://arxiv.org/abs/2504.05118)
  - ORZ's critic assigns lower value to repetitive text, giving more negative advantages to repetition tokens than GRPO would (a computed counterfactual, not a trained GRPO). — [arXiv 2503.24290](https://arxiv.org/abs/2503.24290)
- **Critic quality is the binding constraint.**
  - The PPO critic is near chance at ranking alternative steps. — [VinePPO](https://arxiv.org/abs/2410.01679)
  - Discriminative critics do not improve reliably with size, and their MSE varies sharply across seeds. A generative critic (GenAC) helps, at 2.1× cost. — [GenAC](https://arxiv.org/abs/2604.10701)
  - "Value Flattening". — [arXiv 2609.18708](https://arxiv.org/abs/2609.18708) [abstract]
  - In sparse-reward settings a learned critic can add more noise than signal. One paper proposes single-batch explained variance as a diagnostic. [snippet; paper ID not captured]
- **Multi-turn and agentic settings.**
  - Turn-PPO argues GRPO's sample-based advantages are unstable in multi-turn settings and that token-level PPO is the wrong granularity. It applies GAE with a turn-level critic. — [arXiv 2512.17008](https://arxiv.org/abs/2512.17008) [snippet]
  - MT-GRPO adds turn-level rewards instead. — [arXiv 2505.11821](https://arxiv.org/abs/2505.11821) [snippet]
  - GiGPO merges identical states to get state-level group advantages without a critic. [snippet, cited within search results]
  - In ZeroTIR multi-turn tool use, PPO (21.85) was below GRPO (22.58) and RF++-baseline (24.10). Authors' own runs. — [REINFORCE++](https://arxiv.org/abs/2501.03262)
  - OTB claims its per-token optimal baseline fixes collapse in TIR (28.13 vs GRPO 18.54) without any critic. — [arXiv 2602.07078](https://arxiv.org/abs/2602.07078)
  - GLM-4.5 agentic RL uses a plain group-mean baseline. — [arXiv 2508.06471](https://arxiv.org/abs/2508.06471)

### Inferences
- With γ=λ=1 (the setting that works), PPO's advantage is R − V(s_t): a Monte-Carlo return with a learned, state-dependent baseline. Its benefit over a group-mean baseline is pure variance reduction, which is consistent with "more sample-efficient early, similar ceiling" (GenAC).
- Real critic-driven credit assignment needs λ<1 for the actor (VC-PPO 0.95–0.99, VAPO length-adaptive). That is exactly where critic bias bites, so the method then depends on value pretraining.
- The critic's distinctive contribution in practice is a negative signal on degenerate continuations (repetition, ORZ) and stability on long horizons. Critic-free methods now get the same effect from token-level losses, overlong penalties and repetition handling.
- In agentic, multi-turn work, the frontier evidence leans toward critic-free group baselines with turn-level reward shaping or state grouping. Turn-level critics remain a research line. I would grade "a critic helps for multi-turn" as **C**.
- Is memory/compute the "main real benefit" of critic-free? Yes, roughly 40% less per-iteration compute and no critic hyperparameters, together with robustness to mis-tuning. On the evidence above, critic-free methods do not have a meaningful accuracy edge over a correctly configured PPO.

### Gaps
- I could not locate the paper behind the "critic injects noise; explained-variance diagnostic" snippet, so it is not cited by arXiv ID.
- Turn-PPO, MT-GRPO and GiGPO numbers were not read.
- Logic-RL's PPO > GRPO/REINFORCE++ claim on K&K puzzles was seen only via a survey snippet.

---

## Q4. Does std normalization matter? Evidence for and against.

### Takeaway
On final accuracy, std normalization rarely matters (ScaleRL "similar", Magistral "no significant effect", "Why GRPO Needs Normalization" says indistinguishable at 7B, Minimalist "little additional gain"). Its real effects are regime-dependent:
- It destabilizes training on easy or low-variance data (Tricks or Traps).
- It may speed small models mid-training (2601.23135).
- It implicitly changes the objective toward Σ arcsin√p, upweighting both hard and easy prompts.

The field is converging on removing per-group std (OLMo 3, DeepSeek-V3.2 formula, Magistral, Kimi) or replacing it with a batch-level std (ScaleRL, REINFORCE++, Lite PPO).

### Cited Findings
- **Against / neutral on final accuracy.**
  - ScaleRL: prompt-level, batch-level and none "yield similar performance". — [arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
  - Magistral: no significant effect on evaluation or length growth. — [arXiv 2506.10910](https://arxiv.org/abs/2506.10910)
  - Minimalist: little extra gain beyond filtering. — [arXiv 2504.11343](https://arxiv.org/abs/2504.11343)
  - 7B: indistinguishable. — [arXiv 2601.23135](https://arxiv.org/abs/2601.23135)
- **Against (bias / instability).**
  - Dr. GRPO: difficulty bias. Removing std (jointly with the length fix) gives "statistically significant" gains over 3 runs at 1.5B. — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
  - Tricks or Traps: on easy data, std normalization amplifies advantages and may cause gradient explosions. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
  - REINFORCE++: local std is biased and explodes with small G. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262)
- **For.**
  - 2601.23135: std normalization is an adaptive gradient, provably faster than REINFORCE when per-prompt variances are heterogeneous. Empirical gains at 1.5B. — [arXiv 2601.23135](https://arxiv.org/abs/2601.23135)
  - Tricks or Traps: group-level normalization shows "robust efficiency under each reward setting". This sits in tension with its own Section 4.1.2. — [arXiv 2508.08221](https://arxiv.org/abs/2508.08221)
  - A multi-reward setting reportedly becomes unstable when std is removed. — [GDPO, arXiv 2601.05242](https://arxiv.org/abs/2601.05242) [snippet]
- **Indirectly for upweighting hard prompts** (the low-p half of std's effect):
  - MaxRL w(p)≈1/p beats GRPO on pass@k in 7 of 8 settings. — [arXiv 2602.02710](https://arxiv.org/abs/2602.02710)
  - Pass@k training emphasizes about 25%-accuracy prompts. — [arXiv 2508.10751](https://arxiv.org/abs/2508.10751)
  - Asymmetric Prompt Weighting. — [arXiv 2602.11128](https://arxiv.org/abs/2602.11128)

### Inferences
- The "does std matter" debate conflates two mechanisms.
  - (i) Upweighting low-p (hard) prompts: plausibly helpful for exploration and pass@k, and the reason MaxRL and QAE work.
  - (ii) Upweighting high-p (easy) prompts: no defender, and the source of easy-data instability.
  - Symmetric 1/std does both. That explains the mixed results, since the sign of the net effect depends on the dataset's pass-rate distribution. Hence the Tricks-or-Traps easy-vs-hard split.
- Batch-std normalization is close to a no-op on the objective (a per-step global scalar under Adam). Its small reported benefit is most likely gradient-scale stabilization. That fits ScaleRL's "marginally better".
- Interaction: zero-variance filtering (DAPO, ScaleRL, OLMo 3, Magistral) removes the p̂ ∈ {0,1} cases where std is undefined. Std normalization's worst case is then p̂ = 1/G or (G−1)/G. With G=8 this gives about 2.65× weight on the lone success or lone failure (my calculation), which grows as √G.
- Grade **D** overall: contradictory claims, and a small effect where measured carefully.

### Gaps
- ScaleRL's per-variant A and B numbers are unread.
- No study I found separates "std on the low-p side only" from "std on both sides" in a controlled ablation, other than QAE's asymmetric baseline, which changes the baseline rather than the scale.

---

## Q5. What do frontier tech reports and major frameworks actually use?

### Takeaway
Every frontier reasoning report since 2025, except ByteDance Seed's VAPO line, uses a critic-free group-mean baseline. The trend through 2025–26 is away from per-group std: DeepSeek-R1 used mean+std, but DeepSeek-V3.2, OLMo 3, Kimi and Magistral use mean-only, sometimes with batch-level rescaling. Open frameworks default to GRPO, except verl and OpenRLHF, which still default to `gae` for legacy reasons.

### Cited Findings

Frontier reports:

| Report | Estimator | Source |
|---|---|---|
| DeepSeek-R1 / R1-Zero | GRPO, A_i = (r_i − mean)/std. KL to a reference that is periodically reset to the current policy. PPO vs GRPO in App. A.3. | [arXiv 2501.12948](https://arxiv.org/abs/2501.12948) [full PDF text] |
| DeepSeek-V3.2 | GRPO with formula "A_i,t = R_i − mean(R)". The prose says "normalizing", so prose and formula are inconsistent. Unbiased KL (k3 × importance ratio), off-policy sequence masking for negative-advantage samples, keep-routing, keep-sampling-mask. | [arXiv 2512.02556](https://arxiv.org/abs/2512.02556) [partial] |
| Kimi k1.5 / K2 | Mirror-descent squared loss (r − r̄ − τ log π/π_old)². Baseline r̄ = mean of K samples. No std, no value network. | [arXiv 2501.12599](https://arxiv.org/abs/2501.12599); [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) |
| Qwen3 | "employed GRPO". Reasoning RL is 3,995 query-verifier pairs over 170 steps (235B: AIME'24 70.1 → 85.1). Advantage details not given. | [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) [partial] |
| MiniMax-M1 | CISPO with group mean+std advantages, no critic, no KL. Matches DAPO with 50% of the steps (Qwen2.5-32B zero-RL). | [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) [full] |
| Seed1.5-Thinking | VAPO (actor-critic) and DAPO (critic-free). | [arXiv 2504.13914](https://arxiv.org/abs/2504.13914) |
| Magistral | Group mean, no group std, then minibatch normalization. No KL. Filters zero-advantage groups. Normalization ablation shows no effect. | [arXiv 2506.10910](https://arxiv.org/abs/2506.10910) |
| GLM-4.5 | Reasoning RL "builds upon GRPO… excluding the KL loss". Agentic RL formula uses r − r̄ with no std. Token-weighted mean loss is "highly beneficial" for code. | [arXiv 2508.06471](https://arxiv.org/abs/2508.06471) [partial] |
| OLMo 3 (OlmoRL) | A = r − mean_group(r), "No standard deviation normalization… This removes a difficulty bias". Token-level loss, no KL, clip-higher, truncated IS, zero-gradient filtering plus active sampling. | [arXiv 2512.13961](https://arxiv.org/abs/2512.13961) [full PDF text] |
| Tulu 3 | PPO for RLVR (value model). Initialization details only from a secondary summary. | [arXiv 2411.15124](https://arxiv.org/abs/2411.15124) [snippet] |
| ORZ | PPO with γ=λ=1, batch advantage normalization, no KL. | [arXiv 2503.24290](https://arxiv.org/abs/2503.24290) |
| ScaleRL recipe | Group mean + batch std, CISPO, zero-variance filtering. | [arXiv 2510.13786](https://arxiv.org/abs/2510.13786) |

Framework defaults (local checkouts in `_external_repos/`):

| Framework | Default and options | Location |
|---|---|---|
| verl | `adv_estimator: gae`, `gamma: 1.0`, `lam: 1.0`, `norm_adv_by_std_in_grpo: True`. Enum: gae, grpo, reinforce_plus_plus(_baseline), remax, rloo, opo, grpo_passk, gpg, rloo/grpo_vectorized, optimal_token_baseline, tir_optimal_token_baseline, gdpo. | `verl/verl/trainer/config/ppo_trainer.yaml:63-74`; `verl/verl/trainer/ppo/core_algos.py:88-110` |
| OpenRLHF | `--algo.advantage.estimator` default `gae`. Choices: gae, reinforce, rloo, reinforce_baseline, group_norm (GRPO), dr_grpo. Token-weighted global batch mean/std normalization for gae, reinforce, reinforce_baseline (`--algo.advantage.no_std_norm`). γ forced to 1 for group methods. | `OpenRLHF/openrlhf/cli/train_ppo_ray.py:478-490`; `OpenRLHF/openrlhf/trainer/ppo_utils/experience_maker.py:264-327` |
| slime / miles | `--advantage-estimator` default `grpo`. Choices: grpo, gspo, reinforce_plus_plus(_baseline), ppo (enables critic). | `slime/slime/utils/arguments.py:806-822`; `miles/miles/utils/arguments.py:2068` |
| SkyRL | `advantage_estimator: "grpo"`. | `SkyRL/skyrl/train/config/config.py:347` |
| AReaL | `NormConfig` defaults to batch mean + batch std. The GSM8K GRPO example does group reward normalization, then batch advantage normalization (double normalization). | `AReaL/areal/api/cli_args.py:40-75`; `AReaL/examples/math/gsm8k_grpo.yaml:75-81` |
| cosmos-rl | GRPO with an `unbiased` flag that drops std. | `cosmos-rl/cosmos_rl/dispatcher/algo/grpo.py:38-47` |
| RLinf | `adv_type` includes grpo, grpo_dynamic, reinpp_baseline. | `RLinf/rlinf/config.py:1239` |

### Inferences
- Adoption signal: critic-free group baselines dominate both reports and framework defaults.
  - The only frontier lab that kept a critic in production is ByteDance Seed (VAPO). verl, which ByteDance maintains, keeps `gae` as its default, which reflects its PPO heritage rather than current usage.
  - Newer frameworks (slime, miles, SkyRL) default to GRPO.
  - Per-group std is being dropped (OLMo 3, V3.2 formula, Magistral, Kimi, GLM agentic) or replaced by batch std (ScaleRL, AReaL, OpenRLHF's REINFORCE++ family).
- Frontier labs present the estimator choice as low-stakes. Magistral's "no significant effects" and DeepSeek's "PPO λ=1 nearing GRPO" are direct statements. Their stated reasons are cost and simplicity, not accuracy.

### Gaps
- Llama 4: I found no published RL algorithm detail, so it is not covered.
- I could not read Qwen3's exact advantage formula or whether Qwen3 uses std. Qwen's later GSPO line is outside this scope.
- Tulu 3's value-model initialization from a reward model is unverified (secondary summary only).
- The DeepSeek-V3.2 prose/formula inconsistency on std could not be resolved from the text I read.
