# Verification — Axis 5: KL and entropy regularization in LLM RL

Method: full-text PDFs from arxiv.org/pdf/<id> (latest version at time of check, 2026-10-14), text extracted with PyMuPDF 1.28.2; page numbers are PDF page indices (= printed page for most papers). Framework defaults read from local checkouts in `/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/` (commit dates 2026-05-15..22). Labels: VERIFIED / WRONG / NOT FOUND / FIGURE-ONLY / NEEDS REWORDING.

Notation warning used throughout: k1 = log(π/π_ref), k2 = ½(log π/π_ref)², k3 = π_ref/π − 1 − log(π_ref/π). Papers disagree on what "reverse KL" means — see item 2.

---

## 1. Tang & Munos, 2506.09477 (v1), "On a few pitfalls in KL divergence gradient estimation for RL"

The paper does not use the names k1/k2/k3. Its terms are "vanilla" (= k1), "variance reduced" (= k3), and "squared" (= k2).

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| k1 as a loss has zero expected gradient | VERIFIED | §3, Eq. (2), p.3 | E[∇KL̂_vanilla] = E[∇log π(y)] = 0; "amounts to adding a zero-mean noise to the aggregate gradient estimate." |
| k3 as a loss gives ∇KL(π_ref‖π), not ∇KL(π‖π_ref) | VERIFIED | §3, Eq. (3), p.3; App. A derivation, p.10 | E[∇KL̂_var-reduced] = ∇KL(π_ref, π) ≠ ∇KL(π, π_ref). Key insight: "The commonly adopted variance reduced gradient estimate incidentally minimizes reverse-KL divergence." (Their KL(a,b) = E_a log a/b, so KL(π_ref,π) = KL(π_ref‖π).) |
| k2 as a loss gives the correct gradient on-policy | VERIFIED (the paper does not say "on-policy" here; samples are y∼π) | §3 "Squared estimate", Eq. (4), p.3 | "the squared KL divergence estimate, despite being a biased estimate itself, produces an unbiased KL gradient estimate when differentiated". §6 adds that off-policy, the squared regularizer is "the unique regularization that keeps the optimal policy intact". |
| Per-token KL losses miss the effect on later tokens | VERIFIED | §4.1, p.5; App. D, p.10–11 | "the above gradient does not account for the impact that token y_t has on the future tokens"; Key insight: "Token-level losses only produce a partial derivative at best". |
| Low-KL (Llama-3.1-8B, MATH): estimator choice matters little | NEEDS REWORDING | §5.1, Fig. 7, p.6; App. E, p.11 | §5.1 says "8B Llama 3 model"; App. E says "Llama 3.1 instruction-tuned models … 8B and 70B" on the 7,500 MATH train prompts. Only k3-as-loss and the correct vanilla gradient (both β=0.001) "trace out a similar KL-performance curve". k1-as-loss is "the least KL efficient regardless of the value of β". Write it as: "k3-as-loss vs. correct gradient matters little in low KL; k1-as-loss still fails." |
| High-KL on-policy distillation 70B→8B: k3-as-loss barely makes progress | VERIFIED | §5.2, Fig. 8, p.7 | "the variance reduced (incorrect) estimate does not make much progress on the target metric." §5.3 adds that sequence-level estimators (vanilla, leave-one-out, cumulative) beat the best token-level loss. |
| Libraries named | VERIFIED | §1, p.1; §2.2, p.2; §4, p.5 | Open-source projects: **TRL** (von Werra et al.) and **Open Instruct** (Lambert et al., Tülu 3). Ziegler et al. (2019) is also cited for the KL-as-loss practice. Pseudocode or papers named: **GRPO** (Shao et al. 2024; Guo et al. 2025, i.e. DeepSeek-R1), **Liu et al. 2025** (Dr. GRPO / "Understanding R1-Zero-like training") and **Zhao et al. 2025** (d1). It does **not** name verl, OpenRLHF, slime or SkyRL. |

## 2. Liu et al., 2510.01555 (v2), "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization"

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| k1-in-reward and k2-as-loss gradients coincide on-policy | VERIFIED | Abstract p.1; §5.2, Eqs. (12)–(14), p.6 | "under on-policy conditions, the 'k2 as loss' formulation is, in fact, gradient-equivalent to 'k1 in reward'." |
| k3-as-loss is a first-order biased approximation | VERIFIED | §5.3, Eqs. (15)–(18), p.6–7; App. E Lemma E.1 | The coefficient is 1−δ (δ = π_ref/π_θ), described as "only a first-order surrogate of the principal coefficient −log δ". |
| β=0.5 on Qwen2.5-Math-1.5B: k1-as-loss curve ≈ no-KL curve | VERIFIED (the curves themselves are figure-only) | §6.1, p.8 (setup); Fig. 2 caption, p.8; text p.9 | "the actor model is Qwen2.5-Math-1.5B … set β = 0.5 for all KL regularized losses". Fig. 2: "The training dynamics are nearly indistinguishable". Setup: fully on-policy, temperature 1.0, entropy coefficient 0, 7,300 OpenR1-Math prompts. |

Caveats for the blog:
- **Terminology clash.** This paper calls KL(π_θ‖π_ref) "Reverse KL (RKL)" (Eq. 2, p.3). Tang & Munos use "reverse KL" for KL(π_ref‖π). Comedy (item 15) follows 2510.01555. Pick one convention and state it.
- The paper calls the k1-in-reward ≡ k2-as-loss equivalence "first proven in our work". Tang & Munos (June 2025, §3, Eq. 4) had already shown that differentiating the squared estimate gives the vanilla score-function gradient. Do not repeat the priority claim.
- It also asserts that ProRL's KL is "the 'k2 as loss'" (§2, p.2). The ProRL paper itself does not state the estimator (see item 4).

## 3. DeepSeek-R1, 2501.12948 — the numbers are only in **v2** (86 pp.)

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| KL estimator is k3 | VERIFIED | §2.1, Eq. (2), p.3; repeated in Supp. A.3, Eqs. (11)–(12), p.15 | D_KL(π_θ‖π_ref) = π_ref/π_θ − log(π_ref/π_θ) − 1. p.15 adds: "an unbiased estimator of the KL divergence (Schulman, 2020) is directly added in the loss". |
| β = 0.001 | VERIFIED | §2.1, p.3 (R1-Zero); §3.2.1, p.7 (R1 first RL stage); Supp. B.4.1, p.34 (R1-Zero-Qwen-32B) | "we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1" |
| Reference replaced by the latest policy every 400 steps | VERIFIED | §2.1, p.3; §3.2.1, p.7; B.4.1, p.34 | "Every 400 steps, we replace the reference model with the latest policy model." p.16 gives the rationale: "we periodically update the reference policy to the latest policy". |

Note: the original arXiv v1 (22 pp., Jan 2025) has **none** of these values (searched for 0.001 and 400). Cite v2 or the Nature 2025 version.

## 4. ProRL, 2505.24864 (v1)

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Keeps KL | VERIFIED | §2.3.1, Eq. (4), p.3–4 | L_KL−RL = L_GRPO − β D_KL(π_θ‖π_ref). "retaining a KL penalty is still beneficial for both stability and sustained entropy." |
| Periodically hard-resets the reference model **and optimizer** | VERIFIED, with nuance | §2.3.1, p.4; §3.3, p.4; App. E + Fig. 8, p.18–19 | "we hard-reset the reference policy πref to a more recent snapshot of the online policy πθ, and reinitialize the optimizer states." The resets have no fixed period. They happen "When validation performance stagnates or degrades" (p.4), "Occasionally" (p.18). App. E lists Runs 1–8. |
| β value / estimator | NOT FOUND | — | The text gives neither a β value nor which estimator is used (searched for β=, coefficient, k2, k3, low_var). |
| Ablation that removes KL | NOT FOUND | — | No KL-removal ablation (searched for "ablat", "without KL", "no KL"). The only argument is that DAPO-style removal "often applies to base models prior to any supervised fine-tuning" (p.4). |

## 5. Kimi k1.5, 2501.12599

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| The regularizer is relative to the previous iterate, not the SFT model | VERIFIED | §2.3.2, Eq. (2), p.5 | "At the i-th iteration, we use the current model πθi as a reference model". Objective: max E[r] − τ KL(π_θ‖π_θi). |
| Squared term τ·log(π_θ/π_old) | VERIFIED, NEEDS PRECISE WORDING | §2.3.2, surrogate L(θ) and Eq. (3), p.5 | The surrogate loss is E_{πθi}[(r − τ log Z − τ log(π_θ/π_θi))²]. The gradient (Eq. 3) contains −(τ/2)∇(log π_θ/π_θi)², which the paper calls "an l2-regularization". Eq. (2) itself is a KL term; the squared form appears only in the surrogate. No numeric τ is given. |
| Optimizer reset each iteration | VERIFIED | §2.3.2, p.5 | "Since each iteration considers a different optimization problem due to the changing reference policy, we also reset the optimizer at the start of each iteration." |

## 6. Skywork-OR1, 2505.22312 (v2)

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| KL with β=1e-3 in stage 2 stalls AIME24 | VERIFIED | §3.2.6, Ablation Exp. 5, p.16–17, Fig. 11 | Stage 1 used β=1e-3; stage 2 compares β=1e-3 vs 0. KL "rapidly decrease[s] toward zero … performance on AIME24 fails to improve significantly". The ablated loss is "token-level k3 loss". |
| Released models use β = 0 | VERIFIED | §3.2.6, p.17 | "we set β = 0 for all training stages of our released models." |
| Adaptive entropy control, target 0.2, active only when entropy is below target | VERIFIED | §3.2.5, Eq. (3.2), p.16 | α_k = c_k·I{e_k ≤ tgt-ent}; c_k moves by ±Δ; c_0 = 0; "we set tgt-ent = 0.2 and ∆= 0.005". |
| N_SGD ablation: more SGD steps per batch → faster entropy collapse | VERIFIED | §4.4, Ablation Exp. 6, Fig. 16, p.21–22 | Quadruples (N_SGD, D_R, D_T, N_reuse) are (1,64,64,1), (2,64,32,1), (2,64,64,2), (4,64,16,1), (4,64,64,4). "N_SGD ∈{2, 4} exhibit faster policy convergence, with entropy decaying to very small values within a few training steps." |
| Temperature 0.6 → entropy near 0 within ~100 steps | VERIFIED | §3.2.4, Ablation Exp. 4, p.15 | "when a low temperature is used (e.g., 0.6), the model either begins with extremely low entropy or its entropy quickly collapses to near zero within approximately 100 steps." |

Extra (§4.5, Ablation Exp. 8, p.24): with a constant entropy coefficient, "For αk = 5e-4, 1e-3, 5e-3, and 1e-2, the entropy eventually rises sharply, leading to model collapse." 1e-4 was also tested.

## 7. Cui et al., Entropy Mechanism, 2505.22617

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Reference-model KL keeps entropy stable but lowers performance | VERIFIED | §4.1, Fig. 10, p.11 | "Despite the reference KL achieves stable entropy values, it fails to improve policy and instead leads to a degradation in performance." The KL coefficients tested (Fig. 10 legend) are 0.1, 0.05, 0.01, 0.001 and 0. |
| Entropy coefficients 1e-4 and 1e-3 negligible, 0.01 explodes, 0.005 stabilizes but does not beat baseline | VERIFIED | §4.1 "Effect of Entropy Regularization", Fig. 9, p.10–11 (text on p.11) | "small coefficients have a minor influence on entropy (0.0001, 0.001), large coefficients lead to entropy explosion (0.01). Although setting the coefficient at 0.005 successfully stabilizes policy entropy, it does not outperform other baselines." |

The main setup uses a reference-KL coefficient of 0 by default (§2.2, p.5). The model used for Figs. 9–10 is not named in the text around them.

## 8. Magistral, 2506.10910

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Removed KL | VERIFIED | §2.1 "Eliminating KL divergence", p.3 | "the policy diverges substantially regardless, and maintaining a copy of the reference model for KL computation incurs a compute cost we find unjustified. We remove the KL penalty entirely." |
| "mainly hinders training" | NEEDS REWORDING (the word is "primarily") | §7.4.2 "Entropy targeting", p.16 | "we found that using a KL penalty primarily hinders training, consistent with previous findings [Yu et al., 2025]." They also tried an EMA-of-weights reference and found it "simpler to manually adjust εhigh". |
| Entropy bonus → explosion on math+code, decrease on math only | VERIFIED | §7.4.2, Fig. 12 caption, p.15–16 | "On a math and code dataset, entropy increases excessively with the entropy bonus (even with the same coefficient as in the math-only run)"; for math-only, "entropy drops with the entropy bonus". The experiments use a 3B model. |
| Uses ε_high instead | VERIFIED | §2.1, p.3; §7.4.2, p.16 | ε_high was "adjusted … between 0.26 and 0.28 during the training to keep the group entropy stable". The ε_high = 0.3 for the cold-started model is from a separate section (text near p.9). |

## 9. DAPO, 2503.14476

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Removes KL, with a stated reason | VERIFIED | §2.3 "Removing KL Divergence", p.3 | "during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary." |

## 10. InstructGPT, 2203.02155

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| β = 0.02 used | VERIFIED | App. C.4, p.42 | "These models are also used to compute the KL reward … with β = 0.02 (see Equation 2)." |
| β found best | VERIFIED (range) | App. E.7, Fig. 36, p.56 | "Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02." |
| Extra | — | App. E.6, Fig. 34, p.53 | Raising β to 2.0, "100 times of the default value", does not fix the regressions on public NLP datasets. That experiment uses the 1.3B model with the pretraining-loss coefficient set to 0. |

## 11. Huang et al., "The N+ Implementation Details of RLHF with PPO", 2403.17031

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| KL 50–85 runs had high RM score but <20% win rate | NEEDS REWORDING | §7.1 "PPO training results", observation 3, p.18 | "For 1B models, the KL divergence seems high (around 50 and 85 for two runs)." The runs got "higher RLHF Reward R(x, y) (Equation 3)", where Eq. 3 (p.3) is R = r_φ − β·KL, i.e. the KL-penalized reward, not the raw RM score. GPT judged them poor: "less than 20% of time GPT prefers them over reference summaries". |

Required fixes: (a) the two values are exactly "around 50 and 85" for two 1B runs, not a 50–85 range; (b) they had a higher *RLHF reward* (RM − βKL), not just a higher RM score; (c) the win rate is against the human-written **reference summaries** of TL;DR, as judged by GPT, not against the reference policy. Cause: the policy learned to concatenate strings without spaces (p.18).

## 12. Gao, Schulman & Hilton, 2210.10760

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| R_RL(d) = d(α_RL − β_RL log d), d = √KL | VERIFIED | §1, p.2 | d := √D_KL(π‖π_init); "for reinforcement learning, R_RL(d) = d (αRL −βRL log d)". BoN uses R_bon(d) = d(α_bon − β_bon d). |
| The KL penalty does not improve the gold-vs-KL frontier and acts like early stopping | VERIFIED | §3.6 "Effect of KL Penalty", p.7; Fig. 9 caption, p.8; also §1 bullet, p.2 | "The KL penalty only causes the gold RM score to converge earlier, but does not affect the KLRL-gold reward frontier, and so the effect of the penalty on the gold score is akin to early stopping". The paper itself hedges: "this result could be particularly sensitive to hyperparameters." The KL penalty is set to 0 in all other RL experiments. |

## 13. RL's Razor, 2509.04259

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Forgetting predicted by E_{x∼τ}[KL(π0‖π)] ("forward KL") | VERIFIED | §1, p.2; §6 + Table 1, p.9 | "forgetting is accurately predicted by E_x∼τ[KL(π0‖π)]". The paper calls KL(π0‖π) "forward" and KL(π‖π0) "reverse". |
| R² = 0.96 (ParityMNIST) | VERIFIED | §4 "KL as Predictor", p.6; Table 1, p.9 | "A quadratic fit achieves R2 = 0.96". Table 1 (MNIST): forward KL 0.96 ± 0.01. |
| R² = 0.71 (LLM) | VERIFIED | §4, p.6 (Fig. 11) | "with a quadratic fit achieving R2 = 0.71 (Figure 11)". |
| The theorem uses KL(π‖π0), i.e. the opposite direction | VERIFIED | Theorem 5.2, p.8; App. A Thm A.3, p.20–21 | π† = arg min over π ∈ P*∩Π of D_KL(π‖π0). Lemma 5.1 is also an I-projection, D_KL(q‖p). So the theorem's direction is the paper's own "reverse" KL, while the empirical predictor is "forward". |

Added nuance: in Table 1, reverse KL scores R² = **0.93 ± 0.01**, close to forward KL's 0.96. So "only forward KL predicts" is overstated. The paper's own wording is that reverse KL "showed a good signal".

## 14. ScaleRL, 2510.13786, Appendix A.12

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Entropy does not predict downstream performance | VERIFIED | App. A.12, p.24 | "while entropy decreases consistently during training, it is not necessarily a reliable predictor of downstream performance." |
| Batch 768 vs 2048: nearly identical per-step entropy, 2048 much better | VERIFIED (curves are figure-only, Fig. 16) | App. A.12, p.24 | "Despite the 2048-batch size run achieving much stronger downstream performance at every stage (Figure 10b), both runs followed nearly identical entropy trajectories per step". Entropy was measured "on the held-out validation set". |

## 15. Shah, Obando-Ceron, Jain et al., "A Comedy of Estimators", 2512.21852 (v4, 25 Aug 2026; COLM 2026) — full text read

Scope: k1 and k3 only. k2 is explicitly excluded (§1, p.2; §3, p.4). The target is sequence-level KL(π_θ‖π_ref), which the paper calls "reverse". The reference model is the initial base model.

Table 1 (p.4) summarizes the four configurations:

| Configuration | Unbiased gradient? | Observed training |
|---|---|---|
| k1 in reward | Yes | Stable |
| k1 in loss | No | Unstable |
| k3 in reward | No | Collapse |
| k3 in loss | No | Stable |

| Claim | Verdict | Location | Evidence |
|---|---|---|---|
| Biased-gradient configurations cause instability | VERIFIED, with nuance | Abstract; Obs. 1–2, Figs. 2–3, p.7–8 | k1-in-loss is unstable for Qwen2.5-7B at β ∈ {0.1, 1} and for Llama-3.1-8B-Instruct at every β except 0.1. k3-in-reward gives "complete or partial collapse for all β" (β ∈ {0.05, 0.1, 0.3, 1}). But k3-in-loss is also biased and "exhibits stable training" (Obs. 3, p.8). So the accurate statement is "two of the three biased configurations are unstable". |
| k3-in-loss interpretation | VERIFIED | Obs. 3, p.8; App. F.4 | Its gradient "is a sum of unbiased gradient estimates of the forward KL divergences computed at the token level". This matches Tang & Munos. |
| k1-in-reward beats k3-in-loss | VERIFIED | Obs. 3, p.8; Fig. 4 (Qwen) and Fig. 8 (Llama), App. D.1 | At β=0.05 for Qwen2.5-7B: "+19.06%" average relative improvement on out-of-domain tasks (MMLU college physics, chemistry, biology) vs "+6.21%" on in-domain tasks. For Llama-3.1-8B-Instruct: +15.94% in-domain and +3.65% out-of-domain. "Increasing β deteriorates performance." Exact accuracy cells are FIGURE-ONLY (bar labels). |
| KL helps stabilize async off-policy RL | VERIFIED | §4.2, Figs. 5–7, p.9–10 | Setup: Qwen3-4B-Instruct-2507 on Countdown with Dr. GRPO at async levels 5, 10 and 20; lr 1e-6; 16 rollouts per prompt; batch 512; max length 4096; up to 350 steps; **β = 0.005**. Also Qwen2.5-7B on MATH at async level 10. "K1-in-reward and K3-in-loss, both help stabilize the training in cases where it is otherwise unstable in the absence of any KL regularization." The paper adds that the no-KL gap "can be closed by using regular reference policy resets" (p.9). |
| Models and setup | VERIFIED | §4.1, p.6–7; App. B | On-policy runs: Qwen2.5-7B and Llama-3.1-8B-Instruct on Hendrycks MATH train, with RLOO, batch 256, max length 1024, lr 1e-6, K=5 samples per prompt, temperature 1.0, 3 seeds, mean@32. Eval: MATH500, MATH², and MMLU college physics, chemistry and biology. App. D.3 adds Qwen3-4B-Instruct-2507 on CodeIO with β=0.1. |
| Counter-evidence (App. D.3, p.~18) | Note | App. D.3, Fig. 11 | In the on-policy code-reasoning run, "No KL performs best in this policy setting". |
| Correct gradients via KL in both reward and loss | Note | App. D.2, Fig. 10 | With β=0.1 and 150 steps, k1-in-reward-and-loss and k3-in-reward-and-loss (both unbiased on-policy) perform similarly to k1-in-reward, and all beat k3-in-loss. |
| Libraries | Note | App. E, Table 2 | Recommended unbiased settings: VeRL `algorithm.use_kl_in_reward=True, algorithm.kl_penalty="kl"`; OpenRLHF "DO NOT set --use_kl_loss, --kl_estimator=k1"; SkyRL `kl_estimator_type="k1", use_kl_in_reward=True`. Prime-RL "seems to have removed the previously used KL in loss term entirely". |

## 16. Framework defaults (local checkouts, read-only)

Paths are relative to `/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/`.

### verl (7dc39fe, 2026-05-22)

| Setting | Value | Location |
|---|---|---|
| `use_kl_loss` | `false` | `verl/trainer/config/actor/actor.yaml:99`; dataclass `verl/workers/config/actor.py:170` |
| `kl_loss_type` | `low_var_kl` (k3, clamped to [−10, 10]) | `actor.yaml:112`; `actor.py:175`; implementation `verl/trainer/ppo/core_algos.py:2177-2183` |
| `kl_loss_coef` | `0.001` | `actor.yaml:109` |
| `entropy_coeff` | `0` | `actor.yaml:89`; `actor.py:165` |
| `algorithm.use_kl_in_reward` | `False` | `verl/trainer/config/ppo_trainer.yaml:77`; `verl/trainer/config/algorithm.py:655` |
| `algorithm.kl_penalty` | `kl` (k1) | `ppo_trainer.yaml:80`; `algorithm.py:656` |
| `kl_ctrl.kl_coef` | `0.001`, type fixed | `ppo_trainer.yaml:92` |

So by default neither KL path is on. If KL loss is switched on, it is k3-as-loss. verl also supports a "+" suffix (e.g. `k3+`) that keeps the forward value but takes its gradient from k2 via a straight-through trick (`core_algos.py:2138-2151`).

### slime (6961f59, 2026-05-21) — all in `slime/utils/arguments.py`

| Setting | Value | Location |
|---|---|---|
| `--use-kl-loss` | `default=False` | `:846` |
| `--kl-loss-type` | `default="k1"`, choices k1, k2, k3, low_var_kl | `:800-803` (default on `:803`) |
| `--kl-loss-coef` | `0.0` | `:851` |
| `--kl-coef` (in-reward) | `0.00` | `:777` |
| `--entropy-coef` | `0.0` | `:866` |
| `--ref-update-interval` | `None` | `:861-863` |
| `--use-unbiased-kl` | `False` | `:855-858`; multiplies by the IS ratio (`slime/backends/megatron_utils/loss.py:968-976`) |

The in-reward path uses the same `kl_loss_type` (`loss.py:612-622`). **Pitfall:** turning on `--use-kl-loss` without also changing `--kl-loss-type` gives k1-as-loss, whose expected gradient is zero (k1 = log_ratio, `slime/utils/ppo_utils.py:30-31`).

### SkyRL (ccc181e, 2026-05-21) — all in `skyrl/train/config/config.py`

| Setting | Value | Location |
|---|---|---|
| `use_kl_loss` | `True` | `:356` |
| `kl_estimator_type` | `"k3"` | `:352` |
| `kl_loss_coef` | `0.001` | `:358` |
| `use_kl_in_reward` | `False` (mutually exclusive with KL loss) | `:354` |
| `use_entropy_loss` | `False` | `:359` |
| `entropy_loss_coef` | `0.01` (only used if enabled) | `:360` |

The claim "SkyRL `use_kl_loss` defaults to True with k3" is VERIFIED: the default is k3-as-loss.

### OpenRLHF (c3188af, 2026-05-15) — all in `openrlhf/cli/train_ppo_ray.py`

The current flag names are `--algo.kl.*`. The Comedy paper uses the older `--use_kl_loss`.

| Setting | Value | Location |
|---|---|---|
| `--algo.kl.use_loss` | `default=False` | `:485` |
| `--algo.kl.estimator` | `default="k1"` | `:421-425` |
| `--algo.kl.init_coef` | `0.01` | `:419` |
| `--actor.entropy_coef` | `default=None` | `:431-435` |

When KL loss is off, KL goes into the reward through `compute_approx_kl` (`openrlhf/trainer/ppo_utils/experience_maker.py:205-210`). The CLI warns if the estimator is not k2/k3 with loss on, or not k1 with loss off (`:675-680`). The claim "KL in reward with k1" is VERIFIED.

### AReaL (4aef98d, 2026-05-21) — `areal/api/cli_args.py`

| Setting | Value | Location |
|---|---|---|
| `kl_ctl` | `0.1` | `:1458` |
| `kl_estimator` | `"k1"` | `:1459-1461` |

KL is applied **in the reward**: `rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)` (`areal/trainer/ppo/actor.py:202`). The reference model is created only if `kl_ctl > 0 and config.ref is not None` (`areal/trainer/rl_trainer.py:179`). The shipped examples set `kl_ctl: 0.0` (e.g. `examples/countdown/train_config.yaml:67`). No entropy-coefficient option was found in `cli_args.py`. The claim "KL in reward with k1" is VERIFIED, with the caveat that the dataclass default β=0.1 is off in every example config checked.

---

## Compact list: WRONG / NOT FOUND / needs rewording

1. **Tang & Munos, low-KL result (NEEDS REWORDING).** Only k3-as-loss vs the correct gradient were similar (β=0.001). k1-as-loss stayed the least KL-efficient. The model is "8B Llama 3" in §5.1 and "Llama 3.1 instruct" in App. E. The libraries named are TRL and Open Instruct only (not verl, OpenRLHF, etc.).
2. **"Reverse KL" naming (NEEDS REWORDING).** 2510.01555 and Comedy use "reverse KL" for KL(π‖π_ref); Tang & Munos use it for KL(π_ref‖π). RL's Razor calls KL(π0‖π) "forward". Fix one convention in the blog.
3. **2510.01555 priority claim (NEEDS REWORDING).** Do not repeat its "first proven" claim for the k1-in-reward ≡ k2-as-loss equivalence; Tang & Munos already had it. Its statement that ProRL uses k2-as-loss is not supported by the ProRL paper.
4. **DeepSeek-R1 values (NEEDS CITATION FIX).** β=0.001, k3 and the 400-step reference refresh are only in arXiv **v2** / Nature, not v1.
5. **ProRL (NOT FOUND ×2).** The paper gives no β, no estimator, and no KL-removal ablation. Resets are triggered by validation stagnation ("occasionally"), not on a fixed schedule.
6. **Kimi k1.5 (NEEDS PRECISE WORDING).** The objective is a KL to π_θi. The squared term appears in the surrogate, with coefficient τ/2 on (log π_θ/π_θi)² in the gradient. No τ value is given.
7. **Magistral (NEEDS REWORDING).** The word is "primarily hinders training", not "mainly".
8. **N+ details (NEEDS REWORDING).** The values are "around 50 and 85" for two 1B runs, not a range. Those runs had a higher RLHF reward (RM − βKL), not just a higher RM score. The <20% is GPT-judged preference vs the human **reference summaries**.
9. **InstructGPT (VERIFIED, add nuance).** β=0.02 was used; the best range was "around 0.01 and 0.02".
10. **RL's Razor (NEEDS NUANCE).** Reverse KL also reaches R²=0.93 on MNIST (Table 1), so forward KL is not uniquely predictive. The direction inconsistency between Thm 5.2 (KL(π‖π0)) and the empirical predictor (KL(π0‖π)) is confirmed.
11. **Comedy of Estimators (NEEDS NUANCE).** Not all biased configurations are unstable: k3-in-loss is biased but stable. k2 is excluded from the study. In App. D.3 (CodeIO), no-KL performed best. The async β is 0.005.
12. **Framework pitfall (NEW).** slime defaults `--kl-loss-type` to k1, so `--use-kl-loss` alone gives k1-as-loss (zero expected gradient). OpenRLHF prints a warning in the analogous case; slime does not.
13. **AReaL default (NEEDS NUANCE).** The dataclass default `kl_ctl=0.1` (k1 in reward) is overridden to 0.0 in the shipped example configs.
