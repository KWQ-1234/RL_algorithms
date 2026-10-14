# Verification: what frontier tech reports actually do in RL post-training

Date: 2026-10-14. Method: arXiv full-text PDFs (latest version on the date of download) extracted with PyMuPDF; page numbers are PDF page indices. Official Hugging Face model cards fetched raw where noted. No secondary summaries used.

Versions read: DeepSeek-R1 2501.12948**v2** (4 Jan 2026); DeepSeek-V3.2 2512.02556v1; DeepSeek-V4 2606.19348v1; Kimi k1.5 2501.12599v4; Kimi K2 2507.20534v2; Kimi K2.5 2602.02276v2; MiniMax-M1 2506.13585v1; MiniMax-M2 series 2605.26494v2; Magistral 2506.10910v1; Qwen3 2505.09388v1; GLM-4.5 2508.06471v1; OLMo 3 2512.13961v2; Seed1.5-Thinking 2504.13914v3.

Conventions: "not stated" means the report is silent on that axis after a full-text search. Quotes are verbatim (≤25 words; math transcribed in plain text where marked "Eq., transcribed"). Ablation means a reported controlled comparison (figure or table). A stated claim with no comparison shown does not count.

---

## 0. DeepSeek-V4 provenance

| Question | Finding | Source |
|---|---|---|
| Correct ID | **2606.19348 is the official report.** The official HF model cards for `deepseek-ai/DeepSeek-V4-Pro` and `-Flash` link `https://arxiv.org/abs/2606.19348` as "Technical Report". | HF README.md, line 38 (both cards) |
| Submission date | arXiv abs page: "[v1] Sun, 26 Apr 2026 14:49:33 UTC (2,854 KB)". arXiv API `<published>2026-04-26T14:49:33Z`. One version only. | arxiv.org/abs/2606.19348; export.arxiv.org API |
| Release context | HF repos `DeepSeek-V4-Pro`, `-Flash`, `-Pro-Base`, `-Flash-Base` were created 2026-04-22. | HF API `createdAt` |
| ID/date inconsistency | **Real in arXiv's own metadata, but it does not mean the ID is wrong.** A 2606 prefix with a 26 Apr submission timestamp would fit a submission held before announcement in June. I could not confirm this (the OAI-PMH query returned a 503). | — |
| Final stage | **On-policy distillation (OPD) from RL-trained specialists, not RL.** p.28: "the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD; Gu et al., 2024; Lu and Lab, 2025)." HF card: "independent cultivation of domain-specific experts (through SFT and RL with GRPO), followed by unified model consolidation via on-policy distillation". | PDF p.28 §5.1; HF README line 51 |

---

## 1. DeepSeek-R1 (2501.12948v2)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | GRPO, no critic. A_i = (r_i − mean)/std over the group. | p.3 Eq. 3; p.15 Fig. 3 | "GRPO foregoes the value model, instead estimating the advantages from group scores." |
| 2 Aggregation | Written at **sequence level**: ratio π_θ(o_i\|q)/π_old(o_i\|q), averaged 1/G over outputs. Token-level normalization not stated. | p.3 Eq. 1 (= p.15 Eq. 11) | Eq., transcribed: "1/G Σ_i [min(π_θ(o_i\|q)/π_θold(o_i\|q) A_i, clip(…, 1−ε, 1+ε) A_i) − β D_KL]" |
| 3 Clipping | PPO-style symmetric clip. **ε = 10** in R1's first RL stage. Called crucial, but no data shown. R1-Zero ε: not stated. | p.7 §3.2.1; p.8 | "we set the learning rate to 3e-6, the KL coefficient to 0.001, the GRPO clip ratio ε to 10, and the sampling temperature to 1" |
| 4 KL / entropy | KL to reference, β = 0.001, k3 estimator added in the loss. **Reference reset to the current policy every 400 steps.** Entropy: not stated. | p.3; p.15 | "Every 400 steps, we replace the reference model with the latest policy model." |
| 5 Off-policy | 8,192 rollouts split into 16 minibatches, 1 inner epoch, so 16 updates per rollout batch. Async only for rule-reward computation (systems). IS / precision / routing fixes: not stated. | p.3; p.17 B.1 | "each rollout generates 8,192 outputs, which are randomly split into 16 mini-batches and trained for only a single inner epoch." |
| 6 Data / sampling | G = 16, 32 questions per step (batch 512). R1-Zero max length 32,768, then 65,536 after step 8.2k. Temperature 1.0, then 0.7 in RL stage 2. RL data: 26K math, 17K code, 22K STEM, 15K logic, 66K general. Proofs excluded. Difficulty filtering of RL prompts: not stated. | p.3; p.7; p.8; p.19 Table 4 | "we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward." |
| 7 Reward | Rule-based accuracy and format rewards. Language-consistency reward. Stage 2 adds a helpful RM (pairwise-trained) and a safety RM (pointwise), used only in the last 400 of 1,700 steps. Length penalty: not stated. | p.3 §2.2; p.7; p.8 | "general instruction data and preference-based rewards are incorporated exclusively in the final 400 steps." |
| 8 Ablation | **Yes, partial.** (a) PPO vs GRPO on DeepSeek-Coder-V2-Lite, MATH (Fig. 4, p.16): PPO with GAE λ = 0.95 is worse, and λ = 1.0 nearly matches GRPO. (b) Language-consistency reward on/off (Supp. B.6, Fig. 7, p.35–37). The clip-ratio claim has no data. RM reward hacking is shown observationally (Fig. 6). | p.16; p.35–37 | "with careful tuning (setting λ to 1.0), PPO's performance improves substantially, nearing that of GRPO." |

## 2. DeepSeek-V3.2 (2512.02556v1)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | GRPO, group mean only. **No std division**, unlike R1. | p.7 §3.1 | "calculated by subtracting the average reward of the group from the reward of output oi, i.e., Âi,t = Ri − mean(R)." |
| 2 Aggregation | Per-sequence token mean, then mean over the group (1/G Σ_i 1/\|o_i\| Σ_t). | p.7 Eq. 5 | Eq., transcribed: "1/G Σ_i 1/\|o_i\| Σ_t min(r_i,t Â_i,t, clip(r_i,t, 1−ε, 1+ε) Â_i,t) − β D_KL" |
| 3 Clipping | Symmetric PPO clip. ε value not stated. | p.7 | "ε and β are hyper-parameters controlling the clipping range and KL penalty strength, respectively." |
| 4 KL / entropy | k3 KL **corrected by the π_θ/π_old IS ratio**, so the gradient is unbiased. Strength varies by domain, weak or none for math. β values not stated. Entropy: not stated. | p.7 Eq. 7 | "For certain domains, such as mathematics, applying a relatively weak KL penalty or even omitting it entirely can yield improved performance." |
| 5 Off-policy | Several minibatch steps per rollout batch. **Off-Policy Sequence Masking**: drop negative-advantage sequences whose mean log(π_old/π_θ) exceeds δ, with π_old taken from the inference engine. **Keep Routing** (MoE routing replay, in use since V3-0324). **Keep Sampling Mask** (top-p/top-k truncation mask reused in training). δ not stated. Async: not stated. | p.7–8 Eq. 8–9 | "we mask negative sequences that introduce significant policy divergence, as measured by the KL divergence between the data-sampling policy πold and the current policy πθ." |
| 5 (cont.) | Keep Routing described as crucial. | p.8 | "This Keep Routing operation was found crucial for RL training stability of MoE models" |
| 6 Data / sampling | One mixed RL stage (reasoning + agent + alignment) after specialist distillation. Group size, max length, RL temperature: not stated. Synthetic agent tasks kept only if pass@100 > 0. Post-training compute > 10% of pre-training. | p.6; p.11; p.2 | "retain only instances with non-zero pass@100, resulting in 1,827 environments and their corresponding tasks (4,417 in total)." |
| 7 Reward | Rule-based outcome reward, length penalty, language consistency. Generative RM with per-prompt rubrics for general tasks. | p.6 | "rule-based outcome reward, length penalty, and language consistency reward. For general tasks, we employ a generative reward model where each prompt has its own rubrics" |
| 8 Ablation | **No for RL algorithm choices.** OPSM, Keep Routing and the KL change are backed by qualitative claims only. The §5 ablation is about synthetic agent-task data (Table 5, Fig. 5, p.15–16). | p.8; p.15 | "We empirically observe that this Off-Policy Sequence Masking operation improves stability in certain training scenarios" |

## 3. DeepSeek-V4 (2606.19348v1)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| Pipeline | Specialists (SFT, then GRPO RL) per domain, then multi-teacher OPD into one model. **The final stage is OPD, not RL.** | p.28 §5.1; p.5 | "the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD; Gu et al., 2024; Lu and Lab, 2025)." |
| 1 Advantage | Specialists use GRPO with prior hyperparameters; the form is not restated. The OPD stage has no advantage: it uses a full-vocabulary reverse-KL loss. | p.28 | "we implemented the Group Relative Policy Optimization (GRPO) algorithm, maintaining hyper-parameters closely aligned with our prior research" |
| 2 Aggregation | Not stated. | — | — |
| 3 Clipping | Not stated; inherited from prior work. | — | — |
| 4 KL / entropy | OPD loss is Σ_i w_i · D_KL(π_θ ‖ π_Ei) over >10 teachers, full vocabulary. The report rejects the token-level "log-ratio as advantage" estimator as high-variance. KL-to-reference in specialist RL: not stated. Entropy: not stated. | p.32–33 Eq. 29 | "we adopt full-vocabulary logit distillation in our OPD. Preserving the complete logit distribution in calculating reverse KL loss yields more stable gradient estimates" |
| 5 Off-policy | OPD samples from the student. FP4 QAT, with RL rollout on native FP4 weights to match deployment. Token-level WAL for preempted rollouts, because restarting from scratch biases toward short outputs. Routing replay / IS correction: not stated for V4. | p.33; p.34; p.34–35 | "it is mathematically incorrect to regenerate unfinished requests from scratch, as this introduces length bias." |
| 6 Data / sampling | Separate specialists per reasoning-effort mode, each with its own length penalty and context window. Group size / temperature: not stated. | p.29 | "For each mode, we apply distinct length penalties and context windows during RL training" |
| 7 Reward | No scalar RMs. Rubric-guided GRM, and the **actor itself is the GRM**, jointly RL-trained. Rule verifiers for easy-to-verify tasks. | p.29–30 | "we curate rubric-guided RL data and employ a Generative Reward Model (GRM) to evaluate policy trajectories. Crucially, we apply RL optimization directly to the GRM" |
| 8 Ablation | **None found** for RL or OPD choices. The full-vocabulary vs token-level OPD claim has no reported comparison. | — | — |

## 4. Kimi k1.5 (2501.12599v4)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | No value network. The baseline is the mean reward of k samples (an approximation of τ log Z). No std. | p.5 §2.3.2 | "this gradient resembles the policy gradient of (2) using the mean of sampled rewards as the baseline" |
| 2 Aggregation | Sequence-level: 1/k Σ_j over responses (Eq. 3). Token normalization: not stated. | p.5 Eq. 3 | "For each problem x, k responses are sampled using the reference policy πθi" |
| 3 Clipping | **No clipping.** Online mirror descent with a squared log-ratio (l2) penalty to the previous iterate. | p.5 | "the responses are sampled from πθi rather than on-policy, and an l2-regularization is applied." |
| 4 KL / entropy | KL to the **previous iterate π_θi** (τ), not to SFT. Optimizer reset each iteration. τ value: not stated. Entropy: not stated. | p.5 Eq. 2 | "we use the current model πθi as a reference model and optimize the following relative entropy regularized policy optimization problem" |
| 5 Off-policy | The surrogate tolerates off-policy data, with no IS ratio. **Partial rollouts**: unfinished trajectories go to a replay buffer and continue in the next iteration. Rollout workers are async. Precision / routing: not stated. | p.5; p.9 §2.6.2 | "If a trajectory exceeds the token limit during the rollout phase, the unfinished portion is saved to the replay buffer and continued in the next iteration." |
| 6 Data / sampling | Difficulty measured as SFT pass rate over 10 samples. MC, true/false and proof questions excluded. A prompt is "easy-to-hack" if guessed without CoT within N = 8 tries. Curriculum (easy to hard) plus prioritized sampling ∝ 1 − s_i. RL context 128k. Group size k and temperature: not stated. | p.3; p.6 | "We track the success rates si for each problem i and sample problems proportional to 1 −si" |
| 7 Reward | Rule / test-case rewards. Chain-of-Thought RM for math. Length reward λ = 0.5 − (len − min)/(max − min), warmed up. A separate long2short RL phase. | p.5–7; p.8 | "we promote shorter responses and penalize longer responses among correct ones, while explicitly penalizing long responses with incorrect answers." |
| 8 Ablation | **Yes.** Negative gradients (own method vs ReST, Fig. 10). Curriculum vs uniform sampling (Fig. 9). Model size vs context length (Fig. 8). | p.14–15 §3.5 | "the incorporation of negative gradients markedly enhances the model's efficiency in generating long CoT." |

## 5. Kimi K2 (2507.20534v2)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | k1.5 algorithm: r − r̄(x), mean over K samples. No critic. | p.13 §3.2.3 | "¯r(x) = 1/k ∑k i=1 r(x,yi) is the mean rewards of the sampled responses" |
| 2 Aggregation | Sequence-level 1/K Σ_i of the squared term. | p.13 Eq. | Eq., transcribed: "E_x[1/K Σ_i (r(x,y_i) − r̄(x) − τ log π_θ(y_i\|x)/π_old(y_i\|x))²]" |
| 3 Clipping | None (squared log-ratio regularizer). | p.13 | "τ > 0 is a regularization parameter that promotes stable learning." |
| 4 KL / entropy | τ-regularization to π_old. **Auxiliary PTX loss** on curated high-quality data. A temperature decay schedule stands in for any entropy term. KL to SFT and τ value: not stated. | p.13 | "we curate a dataset comprising hand-selected, high-quality samples and integrate it into the RL objective through an auxiliary PTX loss" |
| 5 Off-policy | Synchronous, colocated training. Partial rollout for long agentic tasks. IS / precision / routing fixes: not stated. Muon optimizer in RL. | p.13 §3.3.1; p.15 | "we adopt a hybrid colocated architecture for our synchronized RL training, where the training and inference engines live on the same workers." |
| 6 Data / sampling | Moderate-difficulty selection by SFT pass@k. Temperature decay. Per-task max-token budget. Group size K: not stated. | p.11; p.13 | "We assess the difficulty of each problem using the SFT model's pass@k accuracy and select only problems with moderate difficulty." |
| 7 Reward | Verifiable gym (rules, code, sandboxes, a faithfulness judge). **Self-Critique Rubric Reward**: K2 ranks its own outputs pairwise against rubrics, and the critic is refined on verifiable rollouts. Budget overruns are truncated and penalized. | p.12; p.13 | "The K2 critic then ranks all results by performing pairwise evaluations against a combination of rubrics" |
| 8 Ablation | **None for RL.** The only "Small-Scale Ablations" are for MuonClip pre-training (p.30). | p.30 | — |

## 6. Kimi K2.5 (2602.02276v2)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | r(x, y_j) − r̄(x), mean over K. No critic. | p.8 Eq. 1 | "¯r(x) = 1/K ∑K j=1 r(x,yj) is the mean reward of all generated responses." |
| 2 Aggregation | **Token-level**: 1/N, where N is the total number of generated tokens in the batch. | p.8 | "N = ∑K i=1 \|yi\| is the total number of generated tokens in a batch" |
| 3 Clipping | New **token-level log-ratio mask** [α, β]: gradients zeroed outside it, regardless of advantage sign. α, β: not stated. | p.8 | "our method relies strictly on the log-ratio to explicitly bound off-policy drift, regardless of the sign of the advantages." |
| 4 KL / entropy | Squared log-ratio τ term to π_old, kept from k1.5. KL to SFT and entropy: not stated. | p.8 Eq. 1 | "Here α,β,τ > 0 are hyperparameters" |
| 5 Off-policy | The mask targets train/inference mismatch. Inference log-probs are recorded for mismatch correction. Token-in-token-out. Partial rollout. Up to 100k concurrent agent tasks. Staleness bound: not stated. | p.8; p.24 App. D | "We also record log probabilities for all inference engine outputs to perform train-inference mismatch correction, ensuring stable RL training." |
| 6 Data / sampling | **Toggle**: every m iterations, alternate between a budget-limited phase (budget = ρ-th percentile of correct-answer lengths, applied only when accuracy > λ) and an unconstrained phase. Group size / temperature: not stated. | p.9 | "we propose Toggle, a training heuristic that alternates between inference-time scaling and budget-constrained optimization" |
| 7 Reward | Rule-based outcome reward, a budget-control reward, and GRMs with multiple rubric variants. Task-specific vision rewards (IoU, edit distance). PARL auxiliary rewards annealed to 0. | p.8–9; p.5–6 | "To mitigate reward hacking and overfitting to a single preference signal, we employ multiple alternative GRM rubrics tailored to different task contexts." |
| 8 Ablation | **Partial.** Toggle evaluated on K2 Thinking (Fig. 5: 25–30% fewer tokens). The token-level mask is called "essential" with no comparison shown. PARL vs single agent (Table 6) is a system-level comparison. | p.9–10 | "On average, Toggle decreases output tokens by 25∼30% with a negligible impact on performance." |

## 7. MiniMax-M1 (2506.13585v1)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | GRPO group-normalized (mean/std, Eq. 2). No critic. | p.5–6 | "Adopting the group relative advantage from GRPO and the token-level loss (Liu et al., 2025b; Yu et al., 2025), CISPO optimizes" |
| 2 Aggregation | Token-level 1/Σ\|o_i\|. **Changed for the 80K stage** to a sample-level loss with token-level normalization. | p.6 Eq. 4; p.12 | "Adopting combined sample-level loss and token-level normalization to alleviate negative-positive sample imbalance" |
| 3 Clipping | **CISPO**: clip the stop-gradient IS weight, not the update. The lower bound is effectively off and only ε_high is tuned (value not stated). ε_high is lowered in the 80K stage. | p.6–7 Eq. 4–5 | "we did not impose a lower bound on the IS weight by setting ϵIS low to a large value; instead, we only tuned ϵIS high." |
| 4 KL / entropy | No KL. No entropy bonus; CISPO is claimed to keep entropy in range. | p.7 | "There is no KL penalty term in CISPO similar to other recent works" |
| 5 Off-policy | **16 off-policy updates per generation batch.** **FP32 LM head** to fix train/inference probability mismatch (correlation 0.9x → 0.99x). AdamW β2 = 0.95, eps = 1e-15. Generation stops after 3,000 consecutive tokens with p > 0.99. Async / staleness / routing: not stated. | p.6; p.7–8 | "we increased the precision of the LM output head to FP32, thereby realigning the two theoretically identical probabilities" |
| 6 Data / sampling | Math kept if pass@10 is strictly between 0 and 0.9. DAPO dynamic sampling. Curriculum: rule-reward reasoning first, then general data mixed in. 40K run, then staged 40→48→56→64→72→80K. Easy samples removed using the 40K model. Group size / temperature: not stated. | p.9; p.7; p.11–12 | "retain only those samples with a pass rate strictly between 0 and 0.9" |
| 7 Reward | Rule verifiers. GenRM (with reference answers) for general tasks, recalibrated online when length-seeking is detected. DAPO overlong length penalty. | p.10–11; p.7 | "immediate GenRMs recalibration is triggered" |
| 8 Ablation | **Yes.** CISPO vs DAPO vs GRPO, zero-RL on Qwen2.5-32B-base, AIME24 (Fig. 2): CISPO matches DAPO in 50% of the steps. The FP32 head is supported by a probability-correlation diagnostic (Fig. 3), not an end-task ablation. | p.6–7 | "CISPO significantly outperforms both DAPO and GRPO with the same number of training steps." |

## 8. MiniMax-M2 series (2605.26494v2)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | Reward-to-go minus a trajectory-level baseline B_i. How B_i is computed (group or otherwise) and whether a critic exists: not stated. | p.18 Eq. 4 | "The advantage estimate is computed via reward-to-go with a trajectory-level baseline" |
| 2 Aggregation | Token-level 1/Σ\|o_i\| (CISPO). The training unit is each (s_t, a_t) completion. | p.17 Eq. 2; §6.1.3 | "Training operates on individual (st, at) pairs as atomic units" |
| 3 Clipping | CISPO with IS weight clipped to [0, 1 + ε_high]. ε value: not stated. | p.17–18 Eq. 3 | "The upper bound 1 + ϵIS high prevents excessively large policy updates, while the zero lower bound permits aggressive down-weighting" |
| 4 KL / entropy | Policy KL: not stated. The only KL is a top-K KL for co-training the MTP draft model. Entropy monitoring is used only for role-play RLHF reward hacking. | p.23; p.16 | "with entropy monitoring applied to mitigate reward hacking." |
| 5 Off-policy | Async Data Pool. **Windowed FIFO** scheduling (W = 0.3N) bounds out-of-order consumption. Fig. 5 is an illustrative example (lag 10). IS only via CISPO. Precision / routing fixes: not stated. | p.21–22 §6.2.4 | "In practice, W = 0.3N maintains near-FIFO distributional properties while substantially reducing cluster idle time." |
| 6 Data / sampling | Mixed-domain stages with shifting domain ratios, per-domain context expansion, and harder difficulty over time. Trajectories up to 192K tokens. Group size / temperature: not stated. | p.19 §6.1.6 | "Within each domain, the difficulty distribution of training tasks shifts progressively toward harder instances." |
| 7 Reward | r_t = α·r_process + β·r_speed + r_perf. Process rewards (language-mixing and tool-format penalties) plus a wall-clock completion-time reward. Rubric and agent-as-verifier judging for some tasks. | p.18 Eq. 5–7 | "We incorporate relative completion time as an explicit reward" |
| 8 Ablation | **None for the RL algorithm.** Reported ablations cover architecture (Tables 1–3) and reasoning-state persistence in agent use (§7.1). | p.4–6; p.24 | — |

## 9. Magistral (2506.10910v1)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | Group mean baseline (r_i − µ, no group std), then normalized across the **minibatch**. No critic. | p.3 | "We estimate the advantage of each token simply as ˆAi,t = ˆAi = ri −µ, where µ is the mean of rewards within a group." |
| 2 Aggregation | Sum of all token losses in the group, divided by the group's total length. | p.3 | "first adding token-wise loss for all tokens and all generations and then dividing by the total length of generations in the group" |
| 3 Clipping | Clip-Higher: ε_high 0.26–0.28 (Medium, adjusted during training) and 0.3 (Small). ε_low: not stated. | p.3; p.9 | "We adjusted it between 0.26 and 0.28 during the training to keep the group entropy stable." |
| 4 KL / entropy | **KL removed.** An entropy bonus was tried and rejected as unstable; ε_high is used instead. | p.3; p.15 §7.4.2 | "We remove the KL penalty entirely." |
| 5 Off-policy | Async generators. Weights updated **in flight**, without KV-cache recompute. A bounded queue limits off-policyness. Final runs use n_async/n_batch ≤ 2 and n_batch = n_minibatch (one update per batch). | p.5–6; p.11 §6.3 | "we opt to keep ratio nasync/nbatch ≤2 and nbatch = nminibatch during final training and further ablations." |
| 6 Data / sampling | Two-stage difficulty filtering with 16 samples per problem (Mistral Large 2, then an RL-trained 24B grader). All-equal-reward groups dropped. Batch 8k → 4k → 2k. l_max − l_cache 16k → 24k → 32k. Temperature 1.0 (Small). Group size: not stated. | p.7; p.8; p.9 | "we filter out all groups with zero advantage when forming training batches." |
| 7 Reward | Format 0.1 plus correctness 0.9 (binary; code uses 20 random tests). DAPO-style soft length penalty (up to −0.1). Language-consistency +0.1 via fastText. | p.3–4 | "If the classifier indicates that all three parts used the same language, we give an additional reward of 0.1." |
| 8 Ablation | **Yes, the most of any report.** Batch/minibatch size (Fig. 6). Advantage normalization (Fig. 7: no significant difference). Binary vs proportional code reward (Fig. 11). Entropy bonus vs ε_high (Fig. 12). Cross-domain (Table 5). RL vs distillation (Fig. 5). | p.10–15 | "we did not observe any significant effects on evaluation performance or the growth of the length as shown in Figure 7." |

## 10. Qwen3 (2505.09388v1)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | GRPO in Reasoning RL. Form not restated. The General RL algorithm: not stated. | p.10 §4.2 | "We ultimately collect a total of 3,995 query-verifier pairs, and employed GRPO (Shao et al., 2024) to update the model parameters." |
| 2 Aggregation | Not stated. | — | — |
| 3 Clipping | Not stated. | — | — |
| 4 KL / entropy | KL: not stated. Entropy is controlled, but the method is not stated. | p.10–11 | "controlling the model's entropy to increase steadily or remain stable, which is crucial for maintaining stable training." |
| 5 Off-policy | Off-policy training mentioned with no details (no update count, IS, or precision fix). | p.10 | "using a large batch size and a high number of rollouts per query, along with off-policy training to improve sample efficiency, is beneficial" |
| 6 Data / sampling | 3,995 query-verifier pairs: unseen in cold start, learnable, as hard as possible, broad. 170 steps (235B AIME24 70.1 → 85.1). Group size / length / temperature: not stated. | p.10–11 | "They are learnable for the cold-start model. (3) They are as challenging as possible." |
| 7 Reward | General RL over 20+ tasks: rules, a model judge with reference (Qwen2.5-72B-Instruct), and a scalar RM without reference. Small models use strong-to-weak **on-policy distillation** instead of RL. | p.12 §4.4–4.5 | "we provide a reference answer for each query and prompt Qwen2.5-72B-Instruct to score the model's response based on this reference." |
| 8 Ablation | **No RL algorithm ablations.** Table 21 (Qwen3-8B): OPD beats RL at about 1/10 of the GPU hours. Table 22: per-stage contributions. | p.20–21 | "distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours." |

## 11. GLM-4.5 (2508.06471v1)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | GRPO-based, critic-free. Agentic RL uses r − r̄ (mean). Std normalization: not stated. | p.7; p.10 §3.3.2 | "Our overall RL algorithm builds upon the GRPO [31] framework, excluding the KL loss term." |
| 2 Aggregation | **Token-weighted mean**, adopted after comparison with a sequence mean (code RL). Environment tokens are masked in agentic RL. | p.9; p.10 | "adopting a token-weighted mean loss is highly beneficial compared to a conventional sequence-mean loss." |
| 3 Clipping | Not stated. | — | — |
| 4 KL / entropy | **No KL.** Sampling temperature is raised when reward plateaus, capped at a ≤1% validation drop. Entropy bonus: not stated. | p.7; p.9 | "when the average reward of rollouts stabilizes, we identify this as a convergence phase and increase the sampling temperature" |
| 5 Off-policy | Synchronous colocated for reasoning / general RL. Async disaggregated for agentic RL. **FP8 rollout with BF16 training.** IS correction for FP8 / async and staleness bound: not stated. | p.13 (slime) | "our infrastructure supports BF16 for training while leveraging FP8 for inference to accelerate the data generation phase." |
| 6 Data / sampling | Two-stage difficulty curriculum (stage-2 pool has verified answers). **Single-stage RL at 64K** instead of progressive length. Expert-verified MCQs only for science. Group size: not stated. | p.7–8; p.9 | "this multi-stage approach is less effective than a single-stage RL process conducted directly at the maximum target length of 64K." |
| 7 Reward | Rule verifiers. Agent outcome reward, with zero reward on a tool-format error. General RL: RLHF RM plus RLAIF rubrics plus a critique model. Final model built through expert RL and self-distillation (SFT). | p.10; p.11; p.5 | "If the model fails to produce the correct tool format during agent trace generation, the process will be halted, and the trace will receive a zero reward." |
| 8 Ablation | **Yes, on a smaller proxy model.** Curriculum (Fig. 5). Single- vs multi-stage length (Fig. 6). Token-weighted vs sequence-mean loss (Fig. 7 left). Science data source (Fig. 7 right). | p.7–9 | "The comparison curves shown in this section are based on our smaller experimental model, not on GLM-4.5." |

## 12. OLMo 3 (2512.13961v2)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | Critic-free group mean, **no std normalization** (Dr GRPO). | p.45 Eq. 2 | "When calculating advantage, we do not normalize by the standard deviation of the group, following Liu et al. (2025b)." |
| 2 Aggregation | Token-level: 1/Σ\|y_i\| over the batch. | p.45 | "We use a token-level loss to normalize the loss by the total number of tokens across the batch" |
| 3 Clipping | Clip-higher: ε_low = 0.2, ε_high = 0.272 in all runs. | p.45; p.109 Table 49 | "We set the upper-bound clipping term in the loss to a slightly higher value than the lower bound" |
| 4 KL / entropy | **No KL.** Entropy: not stated. | p.45 | "We remove the KL loss as a common practice … as it allows less-restricted policy updates" |
| 5 Off-policy | Fully async actors and learner. **In-flight weight updates** without KV invalidation. **Truncated IS** (cap ρ = 2.0 for 32B Think and RL-Zero; "–" for 7B Think and 7B Instruct). Max asynchrony 1 / 8 / 8 / 8 (7B Think / 32B Think / 7B Instruct / RL-Zero). 1 minibatch (4 for 7B Instruct). | p.45; p.48–49; Table 49 | "To adjust for differences between log probabilities from the inference and training engines, we multiply the loss by the truncated importance sampling ratio" |
| 6 Data / sampling | Zero-gradient group filtering plus **active sampling** to refill the batch. Offline filter (7B) drops prompts with pass rate > 62.5% over 8 rollouts. Group size 8. Temperature 1.0. Response length 32,768 (Think), 8,192 (Instruct), 16,384 (RL-Zero). | p.45; p.47; Table 49 | "We then remove all samples that the model easily solves (that is, those with a pass rate greater than 62.5%)." |
| 7 Reward | Binary math verifier. Code tests (fraction-passed and all-pass both tried). IF constraint checkers. Qwen3-32B LM judge scoring in [0, 1] for chat. A length-control verifier was tried and not used. | p.45–46 | "We additionally experimented with puzzle problem … and length-control … verifiers, but did not find it useful for downstream performance." |
| 8 Ablation | **Partial.** Active sampling (Fig. 26, stability). Infrastructure throughput (Table 23). Overlong filtering and length-control verifiers reported as not helpful, with no data shown. | p.65; p.51; p.45 fn. 34 | "We experimented with additional changes (e.g., overlong filtering), but did not find these gave consistent improvements." |

## 13. Seed1.5-Thinking (2504.13914v3)

| Axis | Finding | Location | Quote |
|---|---|---|---|
| 1 Advantage | **Actor-critic (VAPO-style):** value pretraining on π_sft Monte-Carlo returns, decoupled GAE (λ_value = 1.0, λ_policy = 0.95), and length-adaptive λ_policy = 1 − 1/(αl). | p.7 §4.2 | "By employing different Generalized Advantage Estimation (GAE) parameters, such as λvalue = 1.0 and λpolicy = 0.95" |
| 2 Aggregation | Token-level loss. | p.7 | "Instead of defining the policy loss over entire responses, we define it over all tokens." |
| 3 Clipping | Clip-Higher with decoupled ε_low / ε_high. Values: not stated. | p.7 Eq. 1 | "By increasing the value of ϵhigh, we create more room for the increase of low-probability tokens." |
| 4 KL / entropy | KL: not stated. **Positive-example LM (NLL) loss** with coefficient µ. Entropy: not stated. | p.7 Eq. 2 | "We add a language model loss with a coefficient µ for positive examples" |
| 5 Off-policy | **Streaming Rollout System**: a fraction α of samples come from the latest model, the rest are partial generations continued from older snapshots. FP8 rollout. IS / precision correction: not stated. | p.8 §5.2 | "enabling dynamic adjustment of on/off-policy sample ratios through parametric α" |
| 6 Data / sampling | Problems with worst-of-N = 1 removed. MCQs rewritten as fill-in. Dynamic sampling drops accuracy 0/1 prompts. Online data distribution adaptation. Low-variance non-verifiable prompts removed. Group size / length / temperature: not stated. | p.3; p.4; p.7 | "Problems for which the model achieved a woN score (worst of N) of 1 are deemed too simple and removed." |
| 7 Reward | Seed-Verifier (rule-like LLM judge) and Seed-Thinking-Verifier (reasoning judge). **Pairwise generative RM** (P(YES)) for non-verifiable tasks. Hybrid verifier + RM. | p.5; p.6 | "we adopt the pairwise generative reward model mentioned in [9], which evaluates the superiority of two responses" |
| 8 Ablation | **Yes, partial.** DAPO vs VAPO (Table 4, AIME: Qwen-32B 50% vs 60%; Seed-150B-MoE 73% vs 79%). RFT initialization (Table 3: 58% vs 54%). Verifier accuracy (Table 1, not RL training). The pairwise-GenRM stability claim has no numbers. | p.10 | "this consistency suggests that Qwen-32B can effectively serve as a proxy model for investigating RL algorithms." |

---

## Cross-report summary

Abbreviations: GM = group mean baseline, std = group-std division, tok = token-level normalization over batch/group, seq = per-sequence, ns = not stated.

| Report | Critic? | Baseline / std | Aggregation | Clip / surrogate | KL to ref | Off-policy machinery | Reward types | Algorithm ablation? |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1 | No | GM + std | seq (Eq. 1) | PPO clip, ε = 10 (stage 1) | Yes, β = 0.001, ref reset / 400 steps | 16 minibatch updates per rollout | Rules + format + lang; helpful / safety RMs | Yes (PPO vs GRPO; LC reward) |
| DeepSeek-V3.2 | No | GM, no std | 1/G Σ 1/\|o\| | PPO clip, ε ns | IS-corrected k3, domain-dependent, ~0 for math | OPSM, Keep Routing, Keep Sampling Mask | Rules + length + lang; rubric GenRM | No |
| DeepSeek-V4 | No (specialists) | ns | ns | ns | Final stage = reverse-KL OPD | FP4 rollout, WAL; OPD on student samples | Rules; actor-as-GRM with rubrics | No |
| Kimi k1.5 | No | GM | seq | none (squared log-ratio) | to previous iterate (τ) | partial rollouts, replay buffer | Rules, CoT-RM, length reward | Yes (neg. grad, curriculum) |
| Kimi K2 | No | GM | seq | none (squared log-ratio) | to π_old (τ) + PTX loss | sync colocated; partial rollout | Rules + self-critique rubric | No |
| Kimi K2.5 | No | GM | tok (batch) | log-ratio token mask [α, β] | to π_old (τ) | inference logprobs for mismatch correction | Rules + budget + multi-rubric GRM | Partial (Toggle only) |
| MiniMax-M1 | No | GM + std | tok, then sample+tok (80K) | CISPO, ε_high only | None | 16 updates / batch; FP32 LM head | Rules + GenRM; overlong penalty | Yes (CISPO vs DAPO/GRPO) |
| MiniMax-M2 | ns | reward-to-go − B_i | tok | CISPO [0, 1+ε_high] | ns | async pool, Windowed FIFO | Process + speed + outcome; rubrics | No |
| Magistral | No | GM, minibatch-normalized | tok (group) | Clip-higher 0.26–0.3 | None | async, in-flight updates, n_async/n_batch ≤ 2 | Format + correct + soft length + lang | Yes (many) |
| Qwen3 | No (GRPO) | ns | ns | ns | ns | "off-policy training", no details | Rules; ref-judge; scalar RM | No (OPD vs RL only) |
| GLM-4.5 | No | GM (std ns) | token-weighted mean | ns | None | sync (reasoning) / async (agent); FP8 rollout | Rules; RM + RLAIF + critique | Yes (proxy model) |
| OLMo 3 | No | GM, no std | tok (batch) | 0.2 / 0.272 | None | async, in-flight, TIS cap 2.0, max async 8 | Rules; LM judge (chat) | Partial (active sampling) |
| Seed1.5-Thinking | **Yes** | value / GAE | tok | Clip-higher (values ns) | ns | streaming rollout (α); FP8 rollout | Verifiers; pairwise GenRM | Yes (DAPO vs VAPO) |

### Choices shared by nearly all

- **Critic-free, group-relative baseline.** 11 of 13 use one: R1, V3.2, V4 specialists, k1.5, K2, K2.5, M1, Magistral, Qwen3, GLM-4.5, OLMo 3. Seed1.5-Thinking is the only report with an explicit value model. M2 does not say how its baseline B_i is computed.
- **Group-std division is no longer the default.** Only R1 and M1 keep it explicitly. V3.2, Magistral (group step) and OLMo 3 drop it. k1.5, K2 and K2.5 never had it. GLM-4.5 and Qwen3 do not say.
- **No KL to a frozen SFT reference for reasoning RL.** It is removed outright in M1, Magistral, GLM-4.5 and OLMo 3. The Kimi line regularizes only toward the previous iterate. R1 keeps a small KL (0.001) but resets the reference every 400 steps. V3.2 keeps a corrected KL but weakens or omits it for math. Unstated: Seed, Qwen3, M2, V4.
- **Token-level aggregation**, explicit in K2.5, M1/M2, Magistral, GLM-4.5 (ablated), OLMo 3 and Seed. The exceptions are DeepSeek (per-sequence mean, or sequence-level in R1's Eq. 1) and k1.5/K2 (sequence-level squared loss).
- **Loosened or replaced PPO clipping.** Options: clip-higher (Magistral, OLMo 3, Seed), CISPO IS-weight clipping (M1, M2), a log-ratio mask (K2.5), no clip (k1.5, K2), or a very wide symmetric clip (R1 ε = 10).
- **Difficulty filtering or zero-variance group removal** appears in essentially every report that describes data: pass-rate windows, dynamic or active sampling, and curricula.
- **Explicit handling of train/inference mismatch** is now standard in reports from 2025H2 on. Examples: V3.2 (OPSM, Keep Routing, sampling mask), M1 (FP32 head), K2.5 (recorded logprobs plus mask), OLMo 3 (TIS), V4 (native-FP4 rollout plus WAL).
- **Length control in the reward or budget** appears in V3.2, V4, k1.5, K2, K2.5, M1 and Magistral. It is not stated for R1. OLMo 3 tried a length-control verifier and dropped it.
- **The trend toward distillation as the final consolidation step:** V4 (multi-teacher OPD replaces mixed RL), Qwen3 small models (on-policy distillation), GLM-4.5 (expert RL, then self-distillation SFT), and V3.2 (specialist distillation, then mixed RL).

### Which reports ablate their RL algorithm choices at all

- **Substantive ablations:**
  - Magistral: batch/minibatch, advantage normalization, reward granularity, entropy bonus vs ε_high.
  - MiniMax-M1: CISPO vs DAPO vs GRPO.
  - GLM-4.5: loss aggregation, length schedule, curriculum; on a proxy model.
  - Kimi k1.5: negative gradients vs ReST, curriculum.
  - Seed1.5-Thinking: VAPO vs DAPO, RFT init.
  - DeepSeek-R1: PPO vs GRPO on a smaller model; language-consistency reward.
- **Partial (one component or systems only):**
  - OLMo 3: active sampling; infra throughput.
  - Kimi K2.5: Toggle length heuristic.
  - Qwen3: OPD vs RL paradigm, not RL knobs.
- **None for RL algorithm choices:** DeepSeek-V3.2 (stability tricks backed by qualitative claims only), DeepSeek-V4, Kimi K2, MiniMax-M2.

---

## Uncertainties

1. **DeepSeek-V4 ID/date.** The ID 2606.19348 is confirmed by the official HF cards. The 26 Apr 2026 v1 timestamp alongside a 2606 prefix is unexplained. A moderation hold before announcement is the plausible reading, but the OAI-PMH record could not be retrieved (HTTP 503).
2. **DeepSeek-R1 aggregation.** v2 writes GRPO with sequence-level likelihood ratios (Eq. 1). Whether training normalizes per token is not stated. Do not assume the DeepSeekMath per-token 1/|o_i| form for R1.
3. **DeepSeek-V4 specialist RL.** It only says GRPO with "hyper-parameters closely aligned with our prior research". Whether OPSM, Keep Routing and the sampling mask carry over is inferred, not stated.
4. **MiniMax-M2 baseline B_i.** The text says "computed over trajectory i for variance reduction". This could be a group mean, a learned value, or something else, and the report does not specify. The Fig. 5 "Max Off-Policy Lag: 10" is a worked example, not the production setting.
5. **GLM-4.5 agentic objective (p.10)** is printed without a log-probability term (just mean of r − r̄). Read it as a typesetting omission, not as the actual loss.
6. **OLMo 3 TIS.** Table 49 shows "–" for the TIS cap on 7B Think and 7B Instruct. That could mean TIS was disabled for those runs or just unreported; the main text describes TIS as part of the formulation.
7. **Seed1.5-Thinking** never names "VAPO" as the final algorithm in §4.2. It lists VAPO's components (value pretraining, decoupled/length-adaptive GAE) plus DAPO's. "Actor-critic" is a direct reading of those listed components.
8. **Magistral ε_low**, and many ε / β / τ / δ / α values (V3.2, K2, K2.5, M1, M2, Seed), are not reported. The absence of a value is not evidence of a default.
9. **Qwen3** is too terse to fill most axes. "Off-policy training" and "controlling the model's entropy" are stated without mechanism.
10. **PDF extraction.** Symbols (ε, π, Σ) are transcribed from PyMuPDF output. Equation quotes marked "transcribed" are paraphrased into plain text, not character-exact. Page numbers are PDF page indices; printed page numbers may differ by one or more in some reports.
