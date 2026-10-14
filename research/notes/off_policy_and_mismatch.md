# Off-policyness in LLM RL: stale rollouts, training–inference mismatch, and their corrections

Scope: how off-policy the data is (multi-mini-batch PPO, async staleness, in-flight weight updates, numerical sampler/learner mismatch, MoE routing drift) and what each correction does. Research cut-off 2026-10-14.

Reading provenance (so the writer can calibrate):
- **Read in full (raw text):** Yao et al. TIS blog (Notion, via browser), Liu & Li "When Speed Kills Stability" blog (Notion, via browser), ScaleRL (arXiv HTML, grepped the full text), verl `rollout_corr.md` / `rollout_corr_math.md` docs and framework configs in the local checkouts.
- **Read via a summarizing fetch of the arXiv HTML (numbers come from the paper but passed through a summarizer):** FP16 (2510.26788), R3 (2510.11370), Qwen MiniRL (2512.01374), IcePop/Ring-1T (2510.18855), AReaL (2505.24298), PipelineRL (2509.19128), M2PO (2510.01161), TOPR (2503.14286), GSPO (2507.18071), DeepSeek-V3.2 (2512.02556), MiniMax-M1 (2506.13585), Magistral (2506.10910), Kimi K2 (2507.20534, first ~55% only), VeXact (2605.14220), CIS (2609.32444, first ~50% only), the Thinking Machines blog, and the LMSYS FP8 blog.
- **Abstract only:** TRM (2512.23075), "Beyond Precision" LR scheduling (2602.01826), async GRPO collapse scaling laws (2607.01083), GMC-GRPO (2610.01896), SAT (2607.18722), MIPU (2606.29526), DPPO (2602.04879), TBIK (2511.17826).
- **Search snippet only:** LlamaRL/AIPO (2505.24034), PRIME-RL docs, AsyPPO (2510.01656), LMSYS Miles-FSDP blog.
- **Local framework checkouts** (`/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/`) are at commits dated 2026-05-15 to 2026-05-22. Defaults may have changed since then.

---

## Q1. What bias does each source of off-policyness introduce, and what does each correction do mathematically?

### Takeaway
There are four distinct sources of off-policyness: the sampler/learner numeric gap at the same θ, staleness from multi-mini-batch updates, staleness from asynchrony or in-flight updates, and MoE routing flips. All of them break the same identity, E_{y~π_θ}[R ∇log π_θ]. The corrections sort along three axes: token vs sequence (or geometric-mean) aggregation, truncation vs masking, and which ratio is corrected (behavior→proximal vs proximal→current). In the token-level family, the variance is bounded but the gradient keeps a bias that grows with the horizon T. Plain sequence-level IS is unbiased, but at reasoning lengths its weights are numerically degenerate. So the methods that work in practice are all "trust-region filters" (masking, geometric-mean criteria) rather than true IS corrections.

### Cited Findings

**Common notation.** The data is sampled from μ (the inference engine at the behavior parameters). π_old is the learner engine at the behavior or proximal parameters, and π_θ is the current learner.

**(0) What the uncorrected estimator does**
- Hybrid frameworks actually compute E_{a~π_sampler(θ)}[R(a)·∇log π_learner(a,θ)], not the on-policy gradient. In PPO, the clipped ratio π_learner(θ)/π_learner(θ_old) never sees π_sampler, so the mismatch goes completely uncorrected — [Yao et al. blog](https://fengyao.notion.site/off-policy-rl)
- verl's maintainers call it "a critical implementation mistake" to treat the trainer-recomputed π_old as the behavior policy when π_rollout ≠ π_old. Their fix has a three-policy "decoupled" mode (π_rollout, π_old, π_θ) and a two-policy "bypass" mode (π_old := π_rollout) — [verl rollout_corr.md, local checkout](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/docs/algo/rollout_corr.md)
- MoE-specific source: in Qwen3-30B-A3B, about 10% of routers pick different experts in training vs inference, 94% of tokens differ in at least one layer, and there are about 6 router differences per token. The k3 train/infer KL is 1.535e-3 for the MoE vs 6.4e-4 for dense Qwen3-8B. Two identical Megatron forward passes alone give a KL of 8.4e-4 — [R3, arXiv 2510.11370](https://arxiv.org/abs/2510.11370)
- MoE staleness source: after one gradient update on Qwen3-30B-A3B, "roughly 10% of the experts activated under the new policy" differ from those under the old policy. This motivated Routing Replay (R2) for GRPO and the sequence-level GSPO — [GSPO, arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
- Sampling-truncation source: top-p/top-k at inference restricts μ's support while π_θ is trained over the full vocabulary. DeepSeek-V3.2's "Keep Sampling Mask" applies the inference truncation mask to π_θ during training so that both share the same action subspace — [DeepSeek-V3.2, arXiv 2512.02556](https://arxiv.org/abs/2512.02556)

**(1) Token-level truncated IS (TIS) — Yao, Liu et al., Aug 2025**
- (a) REINFORCE form: E_{a~π_sampler}[min(π_learner(a,θ)/π_sampler(a,θ), C)·R·∇log π_learner]. PPO form: multiply the standard clipped PPO term (ratio π_learner(θ)/π_learner(θ_old)) by min(π_learner(θ_old)/π_sampler(θ_old), C). The two ratios are clipped separately — [Yao et al.](https://fengyao.notion.site/off-policy-rl)
- (b) Truncation bounds the per-token variance amplification at C². The blog's example: a ratio of 16 amplifies gradient noise 256× under vanilla IS, 4× under TIS-2, and 64× under TIS-8. Ratios below 1 are left untruncated, and the authors argue that this is what blocks an entropy-collapse loop driven by negative-advantage tokens. They frame "PPO-IS" (putting μ in the PPO denominator) as worse, because clipping then fires even at θ = θ_old — [Yao et al.](https://fengyao.notion.site/off-policy-rl)
- Bias: per-token IS corrects the action distribution but not the state (prefix) distribution d_μ ≠ d_π. Liu & Li derive g_tok = E_{s~d_μ} E_{a~π}[A^{μ}(s,a)∇log π], which differs from the true gradient in both the state measure and the advantage — [Liu & Li blog](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda). Their follow-up blog states the bias as O(T²·D_TV^max) — [same blog, Part 1/2 summary](https://richardli.xyz/rl-collapse)
- TRM tightens the classical O(T²) surrogate-error bound to O(T^{3/2}) (Pinsker-marginal) and O(T) (mixed). All of these bounds depend on the **maximum** token-level divergence, which the authors call a sequence-level quantity that per-token clipping cannot control — [TRM, arXiv 2512.23075, abstract only](https://arxiv.org/abs/2512.23075)

**(2) Sequence-level IS, Seq-TIS, Seq-MIS, Geo-RS — Liu & Li, Sep 2025; verl "Rollout Correction"**
- (a) Seq-IS: weight ρ(y) = Π_t π(y_t|s_t)/μ(y_t|s_t), applied to the whole sequence. This is unbiased (off-policy REINFORCE). Seq-TIS uses min(ρ(y), C). Seq-MIS uses ρ(y)·𝟙{ρ(y) ≤ C}, so over-threshold sequences are rejected rather than clipped — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- (b) Variance of Seq-IS is O((1+χ̄²_max)^T), i.e. exponential in T. Truncation buys a controllable bias — [Liu & Li Part 2 summary](https://richardli.xyz/rl-collapse)
- Geo-RS: ρ_geo = ρ(y)^{1/T} = exp(mean_t log ρ_t), used only as an accept/reject criterion with bounds such as 0.999–1.001. It is length-invariant. In the doc's example, a per-token ratio of 1.1 gives ρ ≈ 2.6 at T=10 but ≈ 13,780 at T=100, so a plain sequence threshold systematically rejects long sequences. verl also provides K2/K3-divergence rejection, where K3_seq = mean_t(ρ_t − log ρ_t − 1) ≥ 0 — [verl rollout_corr_math.md](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/docs/algo/rollout_corr_math.md)
- Note: verl's combined "Geo-RS + Token-TIS" estimator is written in the math doc as 𝟙(geo filter)·Π_t min(ρ_t, C)·f(y). That is a product of truncated token ratios, not per-token TIS weights, which conflicts with its own description ("rollout_rs=seq_mean_k1 + rollout_is=token"). Treat the doc formula with caution — [verl rollout_corr_math.md §3.3.3](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/docs/algo/rollout_corr_math.md)

**(3) IcePop — Ling/Ring team (Ring-1T), Oct 2025**
- (a) GRPO token objective in which each token's term is multiplied by M(k) = k if k ∈ [α, β], else 0, with k = π_train(θ_old)/π_infer(θ_old). Defaults: α = 0.5, β = 5. KL coefficient 0 in the Ring-1T recipe — [Ring-1T, arXiv 2510.18855](https://arxiv.org/abs/2510.18855)
- (b) This is two-sided token **masking** rather than truncation: out-of-band tokens contribute zero gradient. About 1–2‰ of tokens are masked, and those tokens have higher-than-average entropy. The theory (Thm 1/2) argues that the mismatch δ_t = KL(π_infer‖π_train) compounds as δ_{t+1} ≥ (1 + ημ/2)δ_t, under assumptions A1–A4 that the paper does not verify empirically — [Ring-1T](https://arxiv.org/abs/2510.18855)

**(4) Calibrated IS (CIS), Sep 2026**
- (a) The mismatch is modeled as an additive log-odds shift ε_t, so the exact ratio is k_t = p_t + (1−p_t)e^{ε_t}. CIS uses f_λ = min{k_t, 1 + λ(1−p_t)}: the cap tightens for confident tokens, and there is no lower threshold. Defaults: λ = 2.3, with (1−p_t) floored at 5e-3 — [CIS, arXiv 2609.32444](https://arxiv.org/abs/2609.32444)
- (b) The paper argues that CIS bounds the variance term that exact IS leaves unbounded, at a controlled bias. Lifting small weights upward (two-sided bands) collapsed held-out accuracy to 6.01 vs a 28.93 base on Qwen1.5-MoE — [CIS](https://arxiv.org/abs/2609.32444)

**(5) Decoupled PPO — AReaL, May 2025 (staleness correction)**
- (a) J = E_{a~π_behav}[ Σ_t (π_prox/π_behav)·min(u_t·Â_t, clip(u_t, 1−ε, 1+ε)·Â_t) ], with u_t = π_θ/π_prox. π_prox is the parameters before the current update, recomputed by a forward pass — [AReaL, arXiv 2505.24298](https://arxiv.org/abs/2505.24298)
- (b) The trust region is re-centered on a recent policy, so clipping stays informative. The behavior ratio corrects staleness. The original code dropped tokens whose behavior weight exceeded a cap of 5 (token mask). The current default is `rejection_sampling=None`, which means no filtering, and the code warns users about this change. A `prox_logp_method="loglinear"` option approximates π_prox without the extra forward pass — [AReaL cli_args.py, local](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/AReaL/areal/api/cli_args.py); Yao et al. contrast AReaL's "drop" with TIS's "truncate" — [Yao et al.](https://fengyao.notion.site/off-policy-rl)
- Note: TIS-in-PPO, as Yao et al. write it, has the same form as decoupled PPO, with π_behav = sampler, π_prox = learner(θ_old), and a truncated rather than masked behavior weight.

**(6) Truncated-IS REINFORCE family for async: PipelineRL, AIPO (LlamaRL), CISPO**
- PipelineRL: importance-weighted REINFORCE with weights min(c, π/μ), c = 5, and a value baseline — [PipelineRL, arXiv 2509.19128](https://arxiv.org/abs/2509.19128)
- AIPO (LlamaRL): token-level min(π/μ, ρ)·A·∇log π. The generator must ship μ to the trainer — [LlamaRL, arXiv 2505.24034 (search snippet)](https://arxiv.org/abs/2505.24034). PRIME-RL uses a token-level variant of AIPO without entropy/KL terms — [PRIME-RL docs (snippet)](https://docs.primeintellect.ai/prime-rl/async)
- CISPO (MiniMax-M1): sg(clip(r_{i,t}, 1−ε_low, 1+ε_high))·Â·log π_θ, normalized by total tokens. The lower bound is effectively off and only ε_high is tuned. Unlike PPO, every token keeps a gradient — [MiniMax-M1, arXiv 2506.13585](https://arxiv.org/abs/2506.13585)

**(7) M2PO — second-moment masking for very stale data, Oct 2025**
- (a) The statistic is M̂₂ = mean (log r_i)², with r = π_θ/π_behav. Among "constrained" tokens (A > 0 and r > 1, or A < 0 and r < 1), the token with the largest (log r)² is removed repeatedly until the mean M̂₂ ≤ τ = 0.04. The loss is averaged over all tokens — [M2PO, arXiv 2510.01161](https://arxiv.org/abs/2510.01161)
- (b) The paper proves χ²(π_new‖π_behav) ≤ R²·M₂ given 1/R ≤ r ≤ R. M2PO replaces per-token ε-clipping with a batch-level variance budget. The paper's data show that ε-clipping preferentially removes high-entropy tokens as staleness grows — [M2PO](https://arxiv.org/abs/2510.01161)

**(8) TOPR — Tapered Off-Policy REINFORCE, Mar 2025**
- (a) Positive-reward trajectories get weight 1 (SFT-like, no ratio). Negative-reward trajectories get weight clip(π/μ, 0, 1) — [TOPR, arXiv 2503.14286](https://arxiv.org/abs/2503.14286)
- (b) With a⁻ = 0 the objective is bounded above, while naive REINFORCE with negative rewards can drive trajectory probabilities to 0. For positives, TOPR optimizes a lower bound on J — [TOPR](https://arxiv.org/abs/2503.14286)

**(9) DeepSeek-V3.2 Off-Policy Sequence Masking (OPSM)**
- (a) M_{i,t} = 0 iff Â_{i,t} < 0 **and** (1/|o_i|)Σ_t log(π_old/π_θ) > δ, where π_old is the probability returned by the inference engine. Otherwise M = 1, and M multiplies the clipped surrogate. The value of δ is not disclosed — [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- (b) This combines a geometric-mean (per-token-average log-ratio) trust-region test with TOPR-like asymmetry: only negative samples that are far off-policy are dropped, because "the model benefits the most by learning from its own mistakes" — [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556). Liu & Li say V3.2 adopted their geometric sequence masking — [Liu & Li blog news, Dec 2](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)

**(10) GSPO (Qwen), Jul 2025**
- (a) s_i(θ) = (π_θ(y_i|x)/π_θold(y_i|x))^{1/|y_i|}, clipped to [1−3e-4, 1+4e-4]. For comparison, GRPO used 0.2/0.27 — [GSPO](https://arxiv.org/abs/2507.18071)
- (b) A single length-normalized ratio per sequence removes per-token ratio noise, including noise from expert flips. GSPO clips two orders of magnitude more tokens than GRPO. The claim that it tolerates engine mismatch well enough to use inference-engine likelihoods directly is not tested in the paper — [GSPO](https://arxiv.org/abs/2507.18071)

**(11) Routing replay: R2 and R3**
- R3 records the inference engine's top-k routing mask and replays it in training, both for the old-policy recompute and for the update. Gate weights are still the softmax over the **training** logits, restricted to the replayed experts, to keep gradient flow. Reported rollout overhead is below 3% — [R3](https://arxiv.org/abs/2510.11370)
- R2 replays the training engine's routing at the behavior parameters. It leaves the target policy unchanged in the first mini-batch but changes it in later ones. R3 changes the target policy in every mini-batch — [Qwen MiniRL, arXiv 2512.01374](https://arxiv.org/abs/2512.01374)

**(12) Precision and kernel fixes (removing mismatch instead of correcting it)**
- FP16 for both training and inference: 10 vs 7 mantissa bits, "8 times more precision." Sequence-level log-ratio mismatch is about 24× smaller than with BF16. The costs are loss scaling and overflow risk — [FP16, arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
- FP32 LM head: MiniMax traced the mismatch to high-magnitude activations at the head. Moving the head to FP32 raised the train/inference probability correlation from about 0.9x to about 0.99x — [MiniMax-M1](https://arxiv.org/abs/2506.13585)
- Batch-invariant kernels: Thinking Machines argue the root cause of inference nondeterminism is a lack of batch invariance in three reductions (RMSNorm, matmul, attention), not "concurrency + atomics." On Qwen3-235B, 1,000 temperature-0 completions gave 80 unique outputs. With batch-invariant kernels all 1,000 were identical — [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- TBIK extends invariance to different tensor-parallel sizes and claims bitwise-identical vLLM vs FSDP outputs — [TBIK, arXiv 2511.17826, abstract only](https://arxiv.org/abs/2511.17826). VeXact reaches bitwise alignment with FSDP using batch-invariant RMSNorm/matmul/fused-MoE kernels plus attention with KV splitting disabled — [VeXact, arXiv 2605.14220](https://arxiv.org/abs/2605.14220)

**(13) Unbiased KL estimate.** DeepSeek-V3.2 multiplies the k3 estimator by π_θ/π_old, because tokens come from π_old. Uncorrected k3 gives unbounded weight where π_θ ≪ π_ref — [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556). Yao et al. show that under INT8 rollouts the k1 KL estimate frequently goes **negative**, a symptom of sampling from μ rather than π_old — [Yao et al.](https://fengyao.notion.site/off-policy-rl)

### Inferences
- **Sequence-level IS is not a usable estimator at reasoning lengths.** If the per-token log-ratio has a near-zero mean and std σ, then log ρ(y) has std ≈ σ√T. Using the small-divergence approximation KL ≈ ½E[(log r)²], R3's MoE KL of 1.5e-3 gives σ ≈ 0.055. At T = 16k that makes the std of log ρ about 7, so ρ(y) spans many orders of magnitude. Seq-TIS at C = 2 is then mostly a hard truncation, i.e. heavily biased. This is why everyone who succeeds uses geometric-mean or K3 criteria (GSPO, Geo-RS, DeepSeek OPSM): they are length-normalized trust-region tests, not IS corrections. Calling Seq-IS "the principled, unbiased fix" (Liu & Li) is true only in a regime nobody runs. (This is my calculation from the cited KL numbers, not a cited result.)
- **Masking vs truncation is the key design axis.** Truncation (TIS, CISPO, AIPO) keeps a bounded gradient from every token, which introduces bias but uses all the data. Masking (IcePop, MIS, AReaL's old drop, M2PO, OPSM, TRM) zeroes the gradient from out-of-region tokens or sequences, which also introduces bias, toward the well-matched region. The masked tokens are disproportionately low-probability or high-entropy (IcePop, M2PO, Liu & Li), so masking has an exploration cost. The CIS ablation that lifts small weights shows that the lower side matters: masking tokens with ratio < α (IcePop does) is benign, but lifting them is catastrophic.
- **There is a convergent asymmetric principle.** TOPR (truncate only negatives), DeepSeek OPSM (mask only far off-policy negatives), Yao's intuition (bias on ratio < 1 with negative advantage drives entropy collapse), and M2PO (constrain only the tokens that would be clipped) all treat negative-advantage, off-policy tokens as the dangerous part of the gradient. Of these, only TOPR has a theoretical boundedness argument.
- **R3 changes the objective rather than correcting toward it.** Replaying inference routing makes the trained policy "π_θ under frozen routing," which is a biased surrogate of the true π_θ. That explains why Qwen finds R3 gives no gain on-policy and that R2 beats R3 at low staleness.

### Gaps
- No paper reports a clean bias–variance measurement (e.g. gradient cosine to a high-sample ground-truth gradient) across TIS, MIS, Geo-RS, IcePop, and CIS at T > 8k. Everything is inferred from reward curves.
- The TRM experiments were not read (abstract only), so I cannot say whether its tighter bounds translate empirically.
- The δ value in DeepSeek OPSM and the IcePop masking rate at Ring-1T scale are undisclosed.

---

## Q2. What is the evidence that mismatch causes training collapse? Decisive or niche?

### Takeaway
Mismatch-driven collapse is real and reproducible, but its severity is highly conditional. It is decisive for (i) MoE models, (ii) long horizons and multi-turn agentic RL with out-of-distribution tool outputs, (iii) low-precision rollouts (FP8/INT8), and (iv) certain hardware or kernel paths. It is marginal for short-context dense math RL. Almost all collapse evidence is single-seed "crash step" curves, often scored as best-checkpoint. The one multi-seed controlled comparison found (CIS) shows small, mostly within-noise differences between corrections on short GSM8K runs. Part of the attributed "mismatch" was actually a kernel bug (vLLM cascade attention on A100).

### Cited Findings

**Originating evidence (mostly single runs)**
- **Yao et al. (Aug 2025).** On Qwen2.5-32B dense with DAPO (4×8 H100, only 250 steps, single run), TIS improved AIME. On Qwen2.5-0.5B GSM8K with INT8 rollouts, PPO degraded and TIS recovered BF16-level accuracy. On DeepSeek-R1-Distill-1.5B with BF16, TIS gave no gain. FP32 LM head plus "return actual sampling probs" vLLM patches did **not** remove the mismatch. They also found the max mismatch grows with sampler/learner parallelism differences and with response length (20K vs 4K), while the mean mismatch and KL barely change. Swapping sampler backend (vLLM, SGLang, SGLang-deterministic) had no consistent effect — [Yao et al.](https://fengyao.notion.site/off-policy-rl)
- **Liu & Li (Sep 2025).** Qwen3-14B-Base multi-turn TIR with GRPO on L20 GPUs: four runs (on- and off-policy) had gradient norms explode and collapse around step 200. vllm-kl spikes coincided with entropy, fsdp-ppl, and grad-norm spikes. Mismatch concentrates on tokens with low vLLM probability, especially in non-first-turn outputs after OOD tool responses. More tool turns gave earlier collapse — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- **Hardware.** vllm-kl was about 5e-4–1e-3 on H20, 1e-3–1e-2 on L20, and 1e-2–1 on A100. Resuming a failed L20 run on H20 recovered it. Disabling vLLM cascade attention on A100 dropped vllm-kl from 5e-2–1e-1 to about 1e-3 — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- **Liu & Li corrections.** Token-TIS and token-MIS delayed but did not prevent collapse in TIR. Seq-TIS (C=2) prevented collapse but plateaued at or below the vanilla peak. Seq-MIS beat both. In plain reasoning RL on H100, token-TIS prevented gradient explosion "but does not achieve better test performance." FP32 LM head, disabling chunked prefill, and enforce_eager were all ineffective. Lower top-p reduced vllm-kl spikes but slowed learning — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- **Mismatch is coupled to optimization.** In a "batch-filter" run that skipped batches with vllm-kl > 0.1, the model eventually produced only high-mismatch batches. The authors propose a feedback loop: BF16 sensitivity, then kernel-order amplification, then more mismatch — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- A follow-up by an overlapping group argues the mismatch is an optimization problem: gradient noise and mismatch rise together, and an LR decay triggered by response length keeps it safe — [Beyond Precision, arXiv 2602.01826, abstract only](https://arxiv.org/abs/2602.01826)
- **FP16 paper (Oct 2025).** Sanity test on DeepSeek-R1-Distill-1.5B (1,460 "perfectible" MATH problems, 4 gradient steps per iteration, IS clip 3). BF16 GRPO collapsed at peak 73% (VeRL) / 84% (Oat). BF16 + token-TIS collapsed after 82% / 88%. BF16 GSPO had NaN grad norm after 1,200 steps in VeRL. BF16 seq-MIS was stable but capped at 95% train accuracy vs 99% for FP16, with AIME24 at 34% vs 39%. LoRA BF16 collapsed around step 600 and OctoThinker-3B BF16 around step 150, while FP16 was stable in both. BF16-train/FP32-inference was stable but about 3× slower. No seeds are reported, and the base model differs between §5.3 and the appendix (Qwen3-14B vs Qwen2.5-14B) — [FP16](https://arxiv.org/abs/2510.26788)
- **R3 (Oct 2025).** Qwen3-30B-A3B in VeRL (Megatron + SGLang), 30K generation length, best checkpoint reported. GRPO crashed at step 60 (SFT init, 1 mini-step), 105 (Base, 1 mini-step), and 120 (SFT, 8 mini-steps). GRPO + TIS crashed at 105 (SFT, 1 mini-step). R3 runs did not crash within 180 steps. SWE-bench Verified: GRPO 31.8 vs GRPO+R3 38.6. TIS on top of R3 gave no clear gain. In a collapsed run, the fraction of tokens with ratio > 2 exceeded 0.1. R3 kept it below 1e-4 — [R3](https://arxiv.org/abs/2510.11370)
- **IcePop / Ring-mini-2.0 (16.8B total, 0.75B active).** The baseline reward collapsed after 180–200 steps. IcePop and TIS were both stable, with IcePop ahead by a relative 6% on AIME25. A narrow band [0.5, 2] "immediately destabilizes training" — [Ring-1T](https://arxiv.org/abs/2510.18855)
- **Thinking Machines.** Qwen2.5-VL-8B on BigMath: without IS correction there was a loss spike and collapse around step 318. With IS correction, KL stayed around 1e-3 and training was smooth. With bitwise-identical sampler/trainer, KL was exactly 0 and training was smooth — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- **MiniMax-M1.** The LM-head mismatch "prevented reward growth" until the FP32 fix (correlation about 0.9x → 0.99x) — [MiniMax-M1](https://arxiv.org/abs/2506.13585)

**Independent and controlled evidence**
- **Qwen MiniRL** (30B MoE cold-start, FP8 inference / BF16 training as a deliberate stress test, about 5–6 GPU-hours per step, "hundreds of thousands of GPU hours" total). On-policy, removing the train–inference IS correction causes "rapid collapse with a sharp entropy drop." R3 gives no on-policy gain — [Qwen MiniRL](https://arxiv.org/abs/2512.01374)
- **VeXact (zero-mismatch reference engine).** MoE REINFORCE on Qwen3-30B-A3B: with vLLM, train/val reward fell from 0.574/0.293 to 0.255/0.067 after about step 280, while VeXact reached 0.753/0.534. GRPO on Qwen3-1.7B: VeXact held about 0.93 train reward, while recompute-π_old dropped to about 0.40 by step 650 and to about 0 after step 1665. 17.3% of logprob change directions disagree with the advantage sign in bypass mode. TIS combined with r_corr-based sequence rejection "can track VeXact closely" — [VeXact, arXiv 2605.14220](https://arxiv.org/abs/2605.14220)
- **CIS — the closest to a controlled multi-seed comparison (3 MoE models, 3 seeds, GSM8K training).** Five-benchmark averages on Qwen3-30B-A3B: no correction 68.26, TIS 69.14, Seq-TIS 68.99, Seq-MIS 68.68, IcePop 68.99, GSPO 69.23, FP16 68.53, CIS 69.88. On Qwen1.5-MoE: no correction 30.99, IcePop 34.18, CIS 34.78, GSPO 28.90 (below base). No main-table baseline collapsed. The authors concede that CIS's margins over IcePop on some benchmarks are within seed variance — [CIS, arXiv 2609.32444](https://arxiv.org/abs/2609.32444)
- **LMSYS unified FP8.** For BF16-train / FP8-rollout MoE, TIS-clipfrac and the train/rollout log-prob gap "rise significantly" from 30B to 1T. Unified FP8 for both training and rollout lowers both. No reward or collapse outcomes are reported — [LMSYS FP8 RL blog](https://www.lmsys.org/blog/2025-11-25-fp8-rl/)
- **The MoE vs dense gap is consistent across sources.** Yao et al. note that dense 32B and MoE 30B show "different levels" of mismatch and that base models mismatch less than post-trained ones — [Yao et al.](https://fengyao.notion.site/off-policy-rl). R3 measures MoE KL at about 2.4× dense — [R3](https://arxiv.org/abs/2510.11370)

### Inferences
- **Decisive vs niche.** For MoE plus long context plus agentic multi-turn, mismatch handling is decisive: four independent groups report crashes within 60–320 steps without it (R3, Ring, Qwen, VeXact), and DeepSeek calls Keep Routing "crucial." For dense short-context math RL, it is a niche or second-order fix. Yao's 1.5B result, Liu & Li's H100 reasoning result, and the CIS table all show at most about 1-point effects. Writers should not generalize the dramatic 14B-TIR-on-L20 figures to all RL.
- **Bugs and hardware are confounded with mismatch.** The A100 cascade-attention result and the L20-vs-H20 resume experiment show that a large share of collapse is an implementation or hardware pathology. Monitoring vllm-kl or max-mismatch first, and fixing the engine, often beats any estimator change.
- **No correction removes the problem as reliably as removing the mismatch.** Wherever a zero- or near-zero-mismatch arm exists (FP16, true on-policy, VeXact), it matches or beats every IS variant. Among estimator fixes, the order of evidence is roughly: sequence- or geometric-level masking ≥ IcePop ≈ TIS for stability. None has consistently shown a higher ceiling than a mismatch-free run.
- Crash steps from different papers are not comparable: they depend on LR (R3's 8-mini-step run used LR 1e-6 vs 3e-6 for 1 mini-step), hardware, kernels, and data. The R3 result that more off-policy (8 mini-steps) crashed *later* than on-policy is plausibly an LR confound.

### Gaps
- No multi-seed, long-horizon (>1k steps, >16k tokens) controlled comparison of TIS / IcePop / Seq-MIS / FP16 / R3 on a large MoE exists in what I found. CIS is multi-seed but GSM8K-scale.
- I found no independent replication of FP16 at frontier MoE scale (235B+), and no reported FP16 overflow failure at scale. Overflow is a plausible risk that nobody has measured.
- VeXact's injected-mismatch magnitude study does not exist (the paper uses natural mismatch only).

---

## Q3. How much staleness can be tolerated, with and without correction?

### Takeaway
Uncorrected PPO/GRPO tolerates roughly 1–2 policy versions (or ≤2 mini-batch updates) before quality drops. With decoupled or truncated-IS objectives, single-group studies report parity with on-policy up to about 4–8 steps of lag, and M2PO claims parity at 256. Several 2026 results show the tolerable lag is not a constant: it scales inversely with learning rate (Sη ≈ const at the stability boundary) and depends on MoE routing. "Tolerable staleness" should therefore be stated as a staleness × LR budget, not a number.

### Cited Findings

**Synchronous multi-mini-batch (degree of off-policyness inside "on-policy" training)**
- verl's stock defaults are `train_batch_size: 1024`, `ppo_mini_batch_size: 256`, `ppo_epochs: 1`. That means 4 policy updates per rollout batch by default, so the last mini-batch is 3 updates stale — [verl legacy_data.yaml](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/verl/trainer/config/data/legacy_data.yaml), [verl actor.yaml](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/verl/trainer/config/actor/actor.yaml)
- Other reported update counts: MiniMax-M1 used **16** off-policy updates per generation batch — [MiniMax-M1](https://arxiv.org/abs/2506.13585). GSPO used 4 mini-batches — [GSPO](https://arxiv.org/abs/2507.18071). ScaleRL notes PPO-off-policy-k "has been used previously by Qwen3 and ProRL" — [ScaleRL](https://arxiv.org/abs/2510.13786)
- Magistral: "performance degrades when there are more than 2 minibatches in a batch." The final recipe used batch = minibatch (1 update), with n_async/n_batch ≤ 2 — [Magistral, arXiv 2506.10910](https://arxiv.org/abs/2506.10910)
- **Qwen MiniRL (controlled, 30B MoE).** Mini-batch 1,024 responses, global batch 1,024/2,048/4,096/8,192, i.e. N = 1/2/4/8 updates. On-policy (N=1) with plain PG + IS correction was most stable. Off-policy without clipping **or** without routing replay collapsed prematurely with lower peaks. R2 > R3 at N=2, R3 > R2 at N=4 and 8. Once stable, on- and off-policy runs reached similar peak performance — [Qwen MiniRL](https://arxiv.org/abs/2512.01374)
- R3 paper: at 8 mini-steps (LR 1e-6), GRPO crashed at step 120, while GSPO, GRPO+R3, and GSPO+R3 did not crash within 180 steps. With 4 mini-steps on ReasoningSFT, GSPO crashed at 90 — [R3](https://arxiv.org/abs/2510.11370)

**Asynchronous staleness, with and without correction**
- **AReaL** (1.5B, DeepScaleR math, 250 steps, single run; max staleness η vs AIME24 without → with decoupled objective): 0 → 42.0 (oracle); 1: 41.8 → 42.1; 2: 40.0 → 41.8; 4: 23.3 → 42.2; 8: 35.7 → 41.0; 16: 35.8 → 38.7; ∞: 34.0 → 36.9. Defaults: η = 4 (code), 8 (math). Async gave 2.77× (1.5B) down to 1.64× (32B) wall-clock speedup at equal accuracy — [AReaL](https://arxiv.org/abs/2505.24298)
- **PipelineRL** (Qwen2.5-7B, 1,000 steps, B = 1024). Conventional async with lag G = 32 "was the maximum G for which Conventional RL training was stable," and G = 64 diverged. PipelineRL reached the same reward about 2× faster. Its ESS was similar to G = 8, even though some early tokens had lag above 50k samples. A stale KV cache for mixed-policy sequences gave "only slightly higher divergence" than recomputing — [PipelineRL](https://arxiv.org/abs/2509.19128)
- **M2PO** (6 models, 1.7B–32B, 1,000 steps, verl, best checkpoint, seeds not reported). At staleness 256, GRPO dropped 2.6–10.6 points vs on-policy, while M2PO matched or beat on-policy. Examples: Qwen3-Base-4B 50.7 (GRPO s=0) / 40.1 (GRPO s=256) / 43.2 (GSPO s=256) / 51.3 (M2PO s=256); Qwen2.5-32B 51.6 / 47.0 / — / 52.6. GRPO's clipped-token fraction rose from 0.05% to 1.22% on the 32B. AReaL, TOPR, GPPO, and CISPO "break down early" at s = 256 on Llama-3.2-3B — [M2PO](https://arxiv.org/abs/2510.01161)
- **Ring-1T C3PO++** (token-budgeted partial rollouts, resumed under newer weights and purged after σ iterations): about 2.5× rollout speedup, about 1.5× end-to-end, reward "on par." No staleness numbers were disclosed — [Ring-1T](https://arxiv.org/abs/2510.18855)
- **SAT** (Qwen3-30B-A3B, SGLang + Megatron, fully decoupled async; abstract only): SAT-GSPO + R3 scored 35.83 at lag 1 vs 34.79 at lag 8 on AIME24 avg@8 — [SAT, arXiv 2607.18722](https://arxiv.org/abs/2607.18722)
- **Async GRPO collapse scaling laws** (Llama-3.2-1B/3B, Qwen3-8B; abstract only): the max stable LR ∝ S⁻¹ (sync interval S), so Sη ≈ constant. Collapse time ∝ η⁻¹ (a cumulative LR budget). The staleness-induced bias is O(Sη) — [arXiv 2607.01083](https://arxiv.org/abs/2607.01083)
- **GMC-GRPO** (theory plus Qwen3 experiments; abstract only): once the second moment is controlled, delay hurts mainly through clipping or rescaling bias. Group-level rescaling bias vanishes as G → ∞, but trajectory-wise clipping bias persists — [arXiv 2610.01896](https://arxiv.org/abs/2610.01896)
- The ASymPO, DORA, and AsyncWebRL 2026 papers came up in search, but I did not read them (search results only).
- **AsyPPO (2510.01656) is not a staleness method.** It is "Asymmetric PPO" with lightweight mini-critics, ICLR 2026 — [AsyPPO (search snippet)](https://arxiv.org/abs/2510.01656)

### Inferences
- **Without correction:** about 1–2 versions are safe (AReaL η ≤ 2 within about 2 points; Magistral ≤ 2 mini-batches). **With correction:** about 4–8 versions match on-policy in AReaL, PipelineRL, ScaleRL (k = 4 ≈ k = 8), and SAT. 256 (M2PO) is a single-group claim (grade C).
- AReaL's η = 4 uncorrected cell (23.3 AIME24, worse than η = 8 and η = ∞) is non-monotone. That is a sign of single-seed variance or a collapse event, and it inflates the apparent benefit of decoupling at η = 4.
- The Sη scaling law reconciles the conflicting tolerance numbers: papers that use LR 1e-6 (PipelineRL, M2PO) can tolerate far more lag than ones at 3e-6. Notes should quote staleness together with LR.
- For MoE, staleness tolerance is gated by routing drift. Qwen's result that off-policy MoE needs routing replay, and that R3 beats R2 only at N ≥ 4, makes "async MoE without routing replay" a red flag.

### Gaps
- No multi-seed staleness sweep at more than 7B dense or on a large MoE with matched LR.
- The SAT, GMC, and scaling-laws papers were read at abstract level only, so their numbers beyond the abstracts are unverified.
- Whether PipelineRL's "constant high lag of early tokens in long sequences" hurts is explicitly left open by its authors — [PipelineRL](https://arxiv.org/abs/2509.19128)

---

## Q4. Do ScaleRL's FP32 LM head and off-policy algorithm choices affect the asymptote or only efficiency?

### Takeaway
The ScaleRL paper contradicts itself on the off-policy algorithm. Its abstract and §3.1 say PipelineRL vs PPO-off-policy mainly changes efficiency, with "similar asymptotic performance," while Appendix A.11 says it shifts the asymptote and is "one of the most consequential design decisions." The FP32 head raises the asymptote in the forward ablation (A: 0.52 → 0.61). In the full-recipe leave-one-out on the dense 8B it makes "little difference," and on the Scout MoE it helps. The safe reading: both are robustness and stability fixes whose asymptotic effect shows up only when the rest of the recipe is weak (DAPO-style loss) or the model is MoE.

### Cited Findings
- Abstract: "Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote" — [ScaleRL, arXiv 2510.13786](https://arxiv.org/abs/2510.13786)
- §3.1 says "PipelineRL and PPO-off-policy achieve similar asymptotic performance, but PipelineRL substantially improves the compute efficiency." The Fig. 4a caption says PipelineRL is "much more efficient and slightly better in the large compute limit." Max off-policyness k = 4 and k = 8 "performed equally well," and k = 8 is the default — [ScaleRL](https://arxiv.org/abs/2510.13786)
- Appendix A.11 contradicts this: "this distinction affects the asymptotic performance … not just the efficiency exponent … Very few axes shift the asymptote in this way, making the choice of off-policy algorithm one of the most consequential design decisions in RL post-training" — [ScaleRL](https://arxiv.org/abs/2510.13786)
- **FP32 head.** "The precision fix dramatically improves the asymptotic performance" from 0.52 to 0.61 (numbers via summarizer; the HTML math was not extractable) — [ScaleRL](https://arxiv.org/abs/2510.13786). However, "while the FP32 precision fix makes little difference with dense 8B trained with ScaleRL (Figure 7), it provides large gains in GRPO/DAPO-style losses." On Scout 17B×16 MoE the LOO shows "FP32 precision improves overall scalability" — [ScaleRL](https://arxiv.org/abs/2510.13786)
- **LOO.** All LOO variants (including loo-fp32 and loo-8op = PPO-off-policy-8) reach asymptotes within ±0.02, which is the paper's own error margin from 3 independent runs. Differences are mainly in B (efficiency) — [ScaleRL](https://arxiv.org/abs/2510.13786)
- Final ScaleRL fit: A = 0.610, B = 1.92 (8B). Compute: more than 400k GB200 GPU-hours total; ablations at about 3.5–4k GPU-hours each, LOO runs at 16k each, and the largest run at 100k — [ScaleRL](https://arxiv.org/abs/2510.13786)
- **Contrary evidence on the FP32 head as a fix.** Yao et al.'s vLLM FP32-head patch did not remove the mismatch — [Yao et al.](https://fengyao.notion.site/off-policy-rl). Liu & Li resumed a collapsed run with an FP32 head and it still collapsed with a vllm-kl explosion — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda). The positive results are MiniMax-M1 (own claim, correlation 0.9x → 0.99x) — [MiniMax-M1](https://arxiv.org/abs/2506.13585)

### Inferences
- The asymptote claims for both knobs rest on single-run forward ablations at about 4k GPU-hours, against a ±0.02 noise band. The FP32 forward ablation (0.09 gap) is outside the noise. The PipelineRL "slightly better" asymptote is plausibly within it.
- **FP32 head:** it fixes the logit/softmax part of the mismatch only. It is necessary-but-insufficient when the mismatch originates in attention, matmul reduction order, or MoE routing (Liu & Li, Yao). It is cheap, so it is a sensible default, but it is not a stability guarantee.
- **PipelineRL vs PPO-off-policy-k:** the robust effect is efficiency (less idle time, about 2× in both the PipelineRL paper and ScaleRL). The asymptote effect is unresolved because the paper contradicts itself.

### Gaps
- ScaleRL's per-ablation (A, B) values for Fig. 4a/5b could not be extracted from the HTML (the math was rendered as images or MathML). Only A = 0.52 → 0.61 for FP32 came through, via a summarizer.
- No replication of ScaleRL's FP32-head asymptote effect by another group was found.

---

## Q5. What do frontier labs and open frameworks do by default now?

### Takeaway
Open frameworks all ship the corrections (TIS, MIS/Geo-RS, IcePop, routing replay R2/R3, staleness bounds) but leave them **off by default**, so users must opt in. Async modes are the exception: they force rollout log-probs with bounded staleness. Labs converge on "remove what you can, mask what remains": routing replay for MoE (DeepSeek, Qwen, Xiaomi), sequence-level or geometric masking of far off-policy samples (DeepSeek OPSM, Cognition reportedly Seq-MIS), token masking (Ring IcePop), FP32 head plus truncated-IS losses (MiniMax CISPO, ScaleRL), and few or no mini-batch updates (Magistral 1, Qwen on-policy preferred). MiniMax-M1 (16 updates) is the outlier.

### Cited Findings

**Frameworks (local checkouts, May 2026)**
- **verl.**
  - `algorithm.rollout_correction.rollout_is: null` (metrics only), `rollout_is_threshold: 2.0`, `rollout_rs: null`, `bypass_mode: false`.
  - Presets: `decoupled_token_is`, `decoupled_seq_is`, `decoupled_seq_is_rs` (Seq-MIS), `decoupled_geo_rs`, K3-RS, `bypass_pg_is`, and others. A "lower_upper" threshold string gives IcePop.
  - Router replay modes are `disabled` / `R2` / `R3` (default disabled).
  - Fully-async mode forces `use_rollout_log_probs: True` and `bypass_mode: True`, with `staleness_threshold: 0.1`, `trigger_parameter_sync_step: 4`, `partial_rollout: True`.
  - Sources: [rollout_correction.yaml](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/verl/trainer/config/algorithm/rollout_correction.yaml), [actor.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/verl/workers/config/actor.py), [fully_async_ppo_trainer.yaml](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml)
- **slime.** `--use-tis` off by default, `--tis-clip 2.0`, `--tis-clip-low 0`. Custom TIS/RS hook (example MIS at `examples/train_infer_mismatch_helper/mis.py`). `--use-routing-replay` (R2; the help text cites GSPO) and `--use-rollout-routing-replay` (R3) are both off. `--use-rollout-logprobs` (bypass) is mutually exclusive with TIS — [slime arguments.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/slime/slime/utils/arguments.py)
- **Miles** (slime fork). Same TIS and routing-replay flags, plus `icepop_function`, `--true-on-policy-mode` (batch-invariant SGLang ↔ FSDP/Megatron alignment), and `--max-weight-staleness` (default None = no staleness filter) — [miles arguments.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/miles/miles/utils/arguments.py), [corrections.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/miles/miles/backends/training_utils/loss_hub/corrections.py). The LMSYS blog describes Miles's FSDP true-on-policy mode as forcing SGLang-matching operators via `enable_batch_invariant_mode` — [LMSYS Miles FSDP blog (snippet)](https://lmsys.org/blog/2025-12-03-miles-fsdp)
- **AReaL.** Code defaults: `use_decoupled_loss=False`, `max_head_offpolicyness=0`, `rejection_sampling=None`. The shipped math example configs set `use_decoupled_loss: true` and `max_head_offpolicyness: 2–4` — [AReaL cli_args.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/AReaL/areal/api/cli_args.py), [examples/math/*.yaml](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/AReaL/examples/math/gsm8k_grpo.yaml)
- **SkyRL.** `off_policy_correction.tis_ratio_type: None`. Token TIS cap 2.0 (recommended 1.5–5), sequence TIS cap 5.0 (recommended 2–10). Geometric sequence mask at 0.99–1.01 ("MoE models may need a wider range"), product mask 0.5–2.0, and optional outlier-token masks. Fully async `max_staleness_steps: 4`. `moe_enable_routing_replay: False` — [SkyRL config.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/SkyRL/skyrl/train/config/config.py)
- **OpenRLHF.** `--algo.advantage.is_correction_enable` off, thresholds [0.5, 5.0], types `tis` / `icepop` / `seq-mask-tis` — [train_ppo_ray.py](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/OpenRLHF/openrlhf/cli/train_ppo_ray.py)
- **cosmos-rl.** `use_decoupled_loss=False`, `behav_imp_weight_cap=None`, and `off_policy_masking_delta=None` (DeepSeek-V3.2 OPSM, off by default) — [cosmos-rl config](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/cosmos-rl/cosmos_rl/policy/config/__init__.py)
- **RLinf.** No matches for these knobs in the checkout (grep), so it has no built-in mismatch/staleness corrections by these names — [local grep, RLinf](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/RLinf)
- The verl docs claim the presets were "validated with tens of thousands of GPU hours." No public data accompanies that claim in the doc — [verl rollout_corr.md](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/docs/algo/rollout_corr.md)

**Labs / tech reports**
- **DeepSeek-V3.2.** OPSM (mask negative-advantage sequences whose mean log(π_infer/π_θ) > δ), Keep Routing (used since V3-0324, "crucial for RL training stability of MoE models"), Keep Sampling Mask, an unbiased-KL k3 correction, and multiple mini-batches per rollout batch. No ablation numbers are given — [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
- **Qwen.** GSPO for Qwen3-era MoE RL, partly to avoid needing routing replay — [GSPO](https://arxiv.org/abs/2507.18071). The later MiniRL paper recommends on-policy PG with TIS (threshold 5), clip-style masking (ε_high 0.27, ε_low 0.2), and R2/R3 when off-policy. It makes no statement about Qwen3 production — [Qwen MiniRL](https://arxiv.org/abs/2512.01374)
- **Ling/Ring (inclusionAI).** IcePop [0.5, 5], C3PO++ budgeted partial rollouts, fixed MoE router bias during RL — [Ring-1T](https://arxiv.org/abs/2510.18855)
- **MiniMax-M1.** FP32 LM head, CISPO (only ε_high tuned), 16 off-policy updates per generation batch, AdamW β2 = 0.95 and eps = 1e-15, and repetition-based early truncation — [MiniMax-M1](https://arxiv.org/abs/2506.13585)
- **Magistral (Mistral).** PipelineRL-style mid-generation weight swaps without KV-cache refresh. No explicit IS correction beyond the GRPO clip (ε_high 0.26–0.3), no KL, and 1 update per batch — [Magistral](https://arxiv.org/abs/2506.10910)
- **Kimi K2.** K1.5-style objective, mean over i of (r − r̄ − τ·log(π_θ/π_old))², optimized with Muon. Colocated synchronous engines (full weight update in under 30 s) with partial rollouts for long-tail tasks. No IS or mismatch handling is mentioned in the portion read — [Kimi K2, arXiv 2507.20534](https://arxiv.org/abs/2507.20534)
- **Meta.** LlamaRL uses AIPO (token-level capped IS) for 1-to-n-step async — [LlamaRL (snippet)](https://arxiv.org/abs/2505.24034). ScaleRL uses PipelineRL-8 + CISPO + FP32 head — [ScaleRL](https://arxiv.org/abs/2510.13786)
- **Cognition SWE-grep** reportedly uses sequence-level MIS (per the Liu & Li news log; not verified at the primary source) — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- **Thinking Machines** demonstrated batch-invariant "true on-policy" RL (KL = 0). Inference costs: 26 s for default vLLM, 55 s for unoptimized deterministic, 42 s with an improved attention kernel — [Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

**KL-based off-policy diagnostics in use**
- vllm-kl (k3 estimator of KL(π_vllm‖π_fsdp) over model-generated tokens), fsdp-ppl, and grad-norm as early warnings — [Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)
- Max vs mean per-response |p_sampler − p_learner|: max is sensitive to parallelism and length, while mean and KL are not — [Yao et al.](https://fengyao.notion.site/off-policy-rl)
- The extreme-token fraction F(τ = 2), with collapse when F(2) > 0.1 — [R3](https://arxiv.org/abs/2510.11370)
- TIS-clipfrac and the train/rollout log-prob abs diff — [LMSYS FP8](https://www.lmsys.org/blog/2025-11-25-fp8-rl/)
- The batch second moment M̂₂ = mean (log r)² — [M2PO](https://arxiv.org/abs/2510.01161)
- verl `compute_offpolicy_metrics()` — [verl rollout_corr.md](/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/verl/docs/algo/rollout_corr.md)

### Inferences — consolidated verdict table (my judgment from the evidence above)

Grades: A = multiple independent controlled ablations; B = one controlled ablation or a clear mathematical mechanism; C = originating paper only; D = contradictory.

| Method | Ceiling | Efficiency | Stability | Side effects | Grade |
|---|---|---|---|---|---|
| Multiple mini-batch updates per rollout (N > 1) | No gain once stable (Qwen); degrades if > 2 (Magistral) | Faster per-sample convergence (Qwen) | Worse; MoE needs clipping + routing replay | Hidden off-policyness in "sync" defaults (verl = 4) | B |
| Decoupled PPO + staleness bound (AReaL) | Recovers on-policy up to η ≈ 8 (one 1.5B study) | Enables 1.6–2.8× wall-clock | Yes | Extra π_prox forward pass (loglinear approximation exists); M2PO says it breaks at s = 256 | B |
| In-flight updates (PipelineRL) | Unresolved (ScaleRL contradicts itself) | About 2× (PipelineRL; ScaleRL agrees) | OK at k ≤ 8 | Mixed-policy sequences with stale KV; early-token lag | B |
| M2PO | Claims on-policy parity at s = 256 | Allows extreme staleness | Yes (own) | Best-checkpoint reporting, no seeds | C |
| TOPR | Own gains on GSM8K/MATH | n/a | Bounded objective | Offline single-iteration setting only | C |
| Token-level TIS | +0 to +4 points | Neutral | Delays collapse (R3: 60 → 105); fails in TIR (Liu & Li) and in the FP16 sanity test; fine on Ring-mini | O(T²) state-distribution bias; reward logging biased (logged reward can drop while eval rises) | D |
| Seq-TIS / Seq-MIS / Geo-RS | Seq-MIS below FP16 (FP16 paper); no gain on short GSM8K (CIS) | Neutral | Best estimator fix in long-horizon TIR (Liu & Li); stable in FP16 paper | Masks long or low-probability samples | C for stability, D for ceiling |
| IcePop | Among the best in CIS (3 seeds); +6% relative over TIS (Ring) | Neutral | Stable; band [0.5, 2] destabilizes | Masks 1–2‰ high-entropy tokens | B |
| CIS | Best average on 3 MoE models (own, 3 seeds) | Neutral | — | New (Sep 2026) | C |
| FP16 train + infer | Highest in originating paper; ≈ no-correction in CIS | No speed loss vs BF16 (own); BF16 + FP32-infer is about 3× slower | Strong in originating paper | Loss scaling; overflow risk untested at frontier MoE; BF16-pretrained range | B for mismatch reduction, D for downstream gains |
| FP32 LM head | +0.09 A in forward ablation; none in dense LOO | Cheap | Did not stop collapse (Yao; Liu & Li) | Fixes only head-level mismatch | D |
| Batch-invariant / true on-policy (TML, VeXact, TBIK, Miles) | ≥ every IS variant where tested | 1.6–2.1× inference slowdown (TML numbers); VeXact recovers some throughput | Zero mismatch | Engineering cost; pins kernels | B |
| R3 / Keep Routing | +5.6 to +6.8 points over GRPO (R3 paper); Qwen: needed off-policy at N ≥ 4; no on-policy gain | < 3% overhead | Prevents MoE collapse | Biased target (frozen routing); R2 better at low staleness | B (A for MoE multi-mini-batch: R3 paper + Qwen controlled + DeepSeek production) |
| GSPO (sequence ratio) | > DAPO asymptote (ScaleRL); worst on Qwen1.5-MoE (CIS) | — | Stable at 8 mini-steps (R3 paper); crashed at 90 (R3 appendix); NaN at 1,200 (FP16 paper) | Clips about 100× more tokens | D |
| DeepSeek OPSM + Keep Sampling Mask | Undisclosed | — | Claimed | No ablation | C |
| Unbiased-KL k3 correction | — | — | — | Mathematically correct | B (mechanism) |

- **Bottom line for the blog.**
  - The decisive moves are (1) eliminating mismatch at the source (precision, kernels, routing replay) and (2) bounding staleness × LR.
  - Estimator patches (TIS, MIS, IcePop, CIS) are cheap insurance whose measured ceiling effect is small when the system is healthy and whose stability effect is large only when the system is already pathological.
  - Token-level TIS by itself is the weakest of the family and the most over-cited.

### Gaps
- Framework defaults were read from checkouts dated May 2026. Releases from June to October 2026 may have changed defaults (for example, enabling rollout correction by default).
- I could not verify the true-on-policy throughput cost inside RL frameworks (Miles/slime) from primary numbers. Only Thinking Machines' single-GPU inference timings are available.
- No primary source was found for Anthropic, OpenAI, or Google internal practice. Kimi K2's RL section beyond the first ~100k characters was not read.
