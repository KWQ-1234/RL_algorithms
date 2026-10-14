# Verification: on-policy distillation (OPD) section + two pitfalls

Checked 2026-10-14 against primary sources: arXiv PDFs (text extracted with PyMuPDF) and the original blog HTML. Page numbers are PDF pages.

| # | Claim | Verdict |
|---|---|---|
| 1 | GKD trains the student on its own samples with teacher feedback; motivated by train–inference mismatch | VERIFIED (caveat: the headline Fig. 1 uses forward KL / JSD(0.1), not reverse KL) |
| 2 | MiniLLM uses reverse KL; mode-seeking rationale | VERIFIED |
| 3 | TM blog: per-token reverse KL as dense reward; 9–30x; Qwen3 numbers | VERIFIED. "~1/10 compute" is Qwen3's number, which the blog quotes. The blog's own figures are 9x / ~18x / ~30x |
| 4 | Qwen3: OPD beats RL at ~1/10 GPU hours | VERIFIED (1,800 vs 17,920 GPU h; Table 21) |
| 5 | DeepSeek-V4: mixed RL replaced by multi-teacher full-vocab reverse-KL OPD, >10 teachers | VERIFIED |
| 6 | reverse KL on student samples = PG reward log π_T − log π_S | VERIFIED in the TM blog (code) and MiniLLM (explicit). NOT stated in GKD |
| 7 | HF pref-tuning blog: TRL IPO summed instead of averaged; results changed after the fix | VERIFIED (in the Addendum; caveat: the body text is stale) |
| 8 | 2604.23747 abstract | VERIFIED |

## 1. GKD (arXiv 2306.13649v3, ICLR 2024)
- **Mechanism: VERIFIED.** Abstract, p.1: "GKD trains the student on its self-generated output sequences by leveraging feedback from the teacher on such sequences."
- **Why on-policy helps: VERIFIED.** §3.1, p.3–4: fixed-dataset KD "results in train-inference distribution mismatch". The paper frames it as imitation learning (DAgger, Ross et al. 2011). Errors cascade autoregressively. Related work (p.10) also calls this exposure bias.
- **Gradient detail.** p.4, Eq. 4: "we do not backpropagate through the student's sampling distribution pS(·|x)". The loss is supervised-style and per token.
- **Divergence: VERIFIED with a caveat.** GKD is divergence-agnostic and sweeps FKL, RKL, and JSD(0.1/0.5/0.9). JSD(β) interpolates FKL→RKL (p.3). The **headline Fig. 1 uses "JSD (0.1) on WMT and forward KL on other tasks"** (p.1 caption). Do not write that GKD's headline results use reverse KL. RKL/JSD(0.9) win in some settings: instruction tuning, where "on-policy GKD with reverse KL substantially outperforms" (p.9), and high temperature.
- **Headline numbers: VERIFIED.** p.2: relative gains over baseline KD of "2.1× on summarization, 1.7× on machine translation, and 1.9× on arithmetic reasoning". Task-agnostic (FLAN): +2% BBH, +1% MMLU absolute. Fig. 1 labels: +111% XSum, +70% WMT, +90% GSM8K. T5 students; teacher is T5-XL (~3B).

## 2. MiniLLM (arXiv 2306.08543)
- **Reverse KL: VERIFIED.** Abstract, p.1: replaces forward KLD "with reverse KLD … to prevent the student model from overestimating the low-probability regions of the teacher distribution."
- **Mode-seeking: VERIFIED.** §2.1, p.3: "Minimizing reverse KLD has been shown to cause the mode-seeking behavior … qθ assigns high probabilities to p's large modes and ignore the small ones."
- Optimized with policy gradient (§2.2), plus three stabilizers: single-step decomposition, teacher-mixed sampling (against reward hacking), and length normalization (p.2).

## 3. Thinking Machines, "On-Policy Distillation" (Kevin Lu et al., Oct 27, 2025), thinkingmachines.ai/blog/on-policy-distillation/
- **Per-token reverse KL as reward: VERIFIED.** "we choose the per-token reverse KL". KL(πθ‖π_teacher) = E_{x∼πθ}[log πθ(x_{t+1}|x_{1..t}) − log π_teacher(x_{t+1}|x_{1..t})]. Discount factor is **zero**: "the student only optimizes the immediate next token". The loss is a sampled-token estimate, not top-k or full logits: "we do not consider logit (top-k) distillation".
- **Qwen3 replication setup.** Student is Qwen3-8B-Base. The SFT init is trained on 400K OpenThoughts-3 prompts and reaches **60% AIME'24**. Target is 70%; extrapolated SFT needs ~2M prompts. The teacher is nominally Qwen3-32B for FLOP accounting, but "We actually use Qwen3-8B as a teacher". OPD reaches **70% AIME'24 in ~150 steps (~77K prompts, 4 samples/prompt)**.
- **Compute table: VERIFIED.**

  | Method | AIME'24 | Teacher FLOPs | Student FLOPs | CE vs SFT-2M |
  |---|---|---|---|---|
  | SFT-400K | 60% | 8.5e20 | 3.8e20 | – |
  | SFT-2M (extrap.) | ~70% | 3.4e21 | 1.5e21 | 1× |
  | RL | 68% | – | – | ≈1× |
  | OPD | 70% | 8.4e19 | 8.2e19 | **9–30×** |

  - **9x** when the SFT dataset is given or amortized. Teacher FLOPs are counted for OPD only.
  - **~18x** in GPU hours, because teacher log-probs parallelize.
  - **~30x** when the teacher's sampling cost for off-policy data is also counted.
- **"~1/10 compute": VERIFIED as a quote of Qwen3.** The blog reproduces Qwen3 Table 21 (55.0 → RL 67.6 @ 17,920 GPU-h vs OPD 74.4 @ 1,800 GPU-h). Quote: "reaching a higher score of 74.4 on AIME'24 at one-tenth the cost of RL". This is Qwen's number, not TM's own measurement.
- **Second experiment: RL teacher distilled back into the base model.** OPD "learns the RL-trained policy in approximately 7-10x fewer gradient steps, which corresponds to a compute efficiency of 50-100x". RL took 70 steps; OPD took <10.
- **Personalization (Qwen3-8B).** Internal-QA / IF-eval scores: Qwen3-8B 18% / 85%; + midtrain (70%) 36% / 79%; + midtrain + distill 41% / 83%.
- Note: the live page now carries an "Update (June 2026)" saying the Tinker cookbook switched to Qwen3.5-9B. The paper's numbers are unchanged.

## 4. Qwen3 Technical Report (arXiv 2505.09388)
- **Pipeline: VERIFIED.** §4.5 Strong-to-Weak Distillation (p.12) has two phases: off-policy, then on-policy. The student "produces responses in either /think or /no think mode" and is "fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence". **The direction of the KL is not stated.**
- **1/10 GPU hours: VERIFIED.** p.21 (Discussion): "distillation achieves significantly better performance than reinforcement learning while requiring approximately only 1/10 of the GPU hours". p.10 makes the separate claim of "only 1/10 of the GPU hours compared to the four-stage training method".
- **Table 21 (Qwen3-8B, same off-policy-distilled start, math+code queries only; pass@64 in parentheses):**

  | Method | AIME'24 | AIME'25 | MATH500 | LCB v5 | MMLU-Redux | GPQA-D | GPU h |
  |---|---|---|---|---|---|---|---|
  | Off-policy distill | 55.0 (90.0) | 42.8 (83.3) | 92.4 | 42.0 | 86.4 | 55.6 | – |
  | + RL | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 86.9 | 61.3 | 17,920 |
  | + OPD | 74.4 (93.3) | 65.5 (86.7) | 97.0 | 60.3 | 88.3 | 63.3 | 1,800 |

  On pass@64, the report says "reinforcement learning does not lead to any improvement in pass@64 scores", while OPD raises it.

## 5. DeepSeek-V4 (arXiv 2606.19348): VERIFIED (re-quoted)
- §5.1, p.28: "the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD…)".
- §5.1.2, p.33: "more than ten teacher models covering various domains are employed to distill a single student model".
- Also p.33: "we adopt full-vocabulary logit distillation in our OPD … calculating reverse KL loss yields more stable gradient estimates". The cited sources are Gu et al. 2024 (MiniLLM) and "Lu and Lab, 2025" (the TM blog).

## 6. Reverse KL on student samples as a policy-gradient reward log π_T − log π_S
- **TM blog: VERIFIED (implicitly, as code plus prose).** The prose says "We set the per-token advantage to the negative reverse KL". The pseudocode is `reverse_kl = sampled_logprobs - teacher_logprobs; advantages = -reverse_kl`, which gives advantage_t = log π_teacher − log π_student on the sampled token. Caveats: discount is 0, so this is per token rather than the return-to-go; it is a single-sample estimate; and they call it "a one-line change on top of RL implementations which use KL regularization".
- **MiniLLM: VERIFIED, explicit.** §2.2, Eq. 2, p.3: ∇L = −E Σ_t (R_t − 1) ∇log qθ(y_t|·). R_t = Σ_{t'≥t} r_{t'}, with **r_{t'} = log p(y_{t'}|·)/qθ(y_{t'}|·)**. Derived via the policy gradient theorem. This is the most precise primary source for the identity at sequence level.
- **GKD: NOT STATED.** GKD explicitly does *not* backpropagate through sampling (Eq. 4). It contrasts itself with MiniLLM's policy-gradient approach (p.10).
- **DeepSeek-V4 also describes it.** p.33: "prior works … reuse RL framework by replacing sg(log π_Ei/πθ) … as the per-token advantage estimate". V4 rejects this for its variance.

## 7. HF blog "Preference Tuning LLMs with Direct Preference Optimization Methods" (2024-01-18): VERIFIED
- From the Addendum: "the implementation of IPO in TRL was incorrect; in particular, the loss over the log-likelihoods of the completions needs to be averaged instead of summed". The fix is TRL PR #1265.
- On the changed results, the Addendum says: "The results are now consistent with the paper, with IPO on par with DPO and performing better than KTO in the paired preference setting."
- **Caveat:** some body paragraphs were not updated. The Results section still says IPO "appears to be worse than the base model in all but one setting" and "DPO > KTO > IPO". The Summary was updated to "DPO and IPO can achieve comparable results". Cite the Addendum and Summary, not the Results prose.

## 8. arXiv 2604.23747 (Limozin et al., "SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning"): VERIFIED (re-quoted)
- Abstract, p.1, names two bugs: "a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation" (affecting TRL, OpenRLHF, and Llama-Factory), and "a loss aggregation bug in OpenRLHF". The optimizer bug accounts for most of the gap.
- Once both are fixed, SFT-then-RL beats every mixed-policy method by **+3.8** (Qwen2.5-Math-7B) and **+22.2** (Llama-3.1-8B) points on math.
