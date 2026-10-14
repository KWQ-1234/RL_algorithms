# Process Reward Models and Fine-Grained Credit Assignment for LLM Reasoning RL (evidence audit, as of 2026-10-14)

Reading-depth legend used throughout: **[full]** = arXiv HTML body read (methods + experiments; for very long pages the last ~2-50K chars may be unread); **[abs]** = abstract/landing page only; **[snippet]** = search-result snippet only, not verified. Evidence grade: A = multiple independent controlled ablations; B = one controlled ablation or clear mechanism; C = originating paper only; D = contradictory.

Notation: q prompt, y = (y_1..y_T) response, steps s_1..s_K, R(y) ∈ {0,1} verifiable outcome, r^p_k learned per-step process reward, V^π(prefix) = P(correct | prefix) under π.

---

## Q1. For each method: what signal is added, how it becomes an advantage, and what it changes mathematically

### Takeaway
The methods split cleanly into two families with different math. (i) **Learned-PRM rewards** (Math-Shepherd PPO, Gao et al. PR, PURE, PRPO, Blend, PAV) inject a *biased* proxy into the return, and every paper that uses one as a reward finds hacking unless the PRM term is bounded/telescoped (Clip-Delta, min-form) or anchored by the verifiable outcome. (ii) **On-policy MC credit** (VinePPO, SPO, TreeRL, TreeRPO, and implicitly GRPO itself) estimates V^π at intermediate states from the policy's own rollouts against the ground-truth verifier, which is unbiased (up to finite-K noise) and acts purely as variance reduction / better per-token credit; its cost is extra generation. PRIME sits between: a learned (biased) implicit PRM, but updated online from outcome labels and summed so that its total telescopes to a sequence-level score.

### Cited Findings

**Baseline being compared against (outcome-only GRPO).** Outcome supervision in GRPO assigns every token the same normalized reward Â_{i,t} = (r_i − mean r)/std r; DeepSeekMath's "process supervision" variant normalizes all step rewards over the group and sets Â_{i,t} = Σ_{index(j) ≥ t} r̃_i^{index(j)} (sum of normalized future step rewards) — [DeepSeekMath, 2402.03300 [full]](https://arxiv.org/html/2402.03300)

#### 1. ORM vs PRM (Uesato et al. 2022; Lightman et al. 2023)
- (a/b) Uesato: on GSM8K, pure outcome-based supervision gives similar final-answer error to process-based supervision with less labeling; process supervision (or learned RMs emulating it) is needed for correct reasoning steps; headline numbers: final-answer error 16.8%→12.7%, trace error among correct answers 14.0%→3.4% (abstract does not attribute arrows cleanly to process vs outcome arm) — [Uesato et al., 2211.14275 [abs]](https://arxiv.org/abs/2211.14275)
- Lightman: PRM800K = ~800K step labels on 75K solutions to 12K problems; PRM solution score = product of step-correctness probabilities (slightly penalizes more steps); best-of-1860 on 500 MATH test problems: PRM 78.2%, ORM 72.4%, majority vote 69.6%; generator = GPT-4-base finetune, fixed; RL on the generator "intentionally not the focus"; active learning ~2.6x data efficient; "negative alignment tax" claim with explicit caveat about generalization beyond math — [Lightman et al., 2305.20050 [full]](https://arxiv.org/html/2305.20050)
- (e) Verdict: evidence is **test-time reranking only** (Lightman) or GSM8K-scale expert iteration (Uesato). Neither shows PRM > outcome in RL final accuracy. Grade for "PRM > ORM as a test-time reranker": B (one large controlled study; later corroborated as rerankers by DeepSeek-R1's own statement below). Grade for "PRM helps RL ceiling": not established by these papers.

#### 2. Math-Shepherd (MC-estimated step labels)
- (a) Labels: hard estimation HE = 1 if any of N completions from the step reaches gold answer, else 0; soft SE = fraction of N completions reaching gold; N=8; completer stated as LLemma-7B in §4 but LLaMA2-70B/MetaMATH in §5.2 (internal inconsistency) — [Math-Shepherd, 2312.08935 [full]](https://arxiv.org/html/2312.08935)
- (b/c) RL: "step-by-step PPO" with per-step PRM reward; policies LLaMA2-7B / Mistral-7B (MetaMATH SFT), max seq len 512, KL 0.04; seeds, PPO steps, batch sizes not reported. Greedy results (GSM8K/MATH): Mistral SFT 77.9/28.6 → RFT 79.0/29.9 → ORM-PPO 81.8/31.3 → PRM-PPO 84.1/33.0; LLaMA2 66.6/19.2 → ORM-PPO 70.8/20.8 → PRM-PPO 73.2/21.6. No reward-hacking discussion — [Math-Shepherd, 2312.08935 [full]](https://arxiv.org/html/2312.08935)
- Confounder: the RL baseline is PPO with a *learned ORM*, not a ground-truth verifier, so the comparison is learned-PRM vs learned-ORM, not PRM vs RLVR; responses are ≤512 tokens. (my reading of the setup above)
- (d) Independent evidence on MC labels: Qwen's study finds MC-estimated labels are the weakest step-level supervision (ProcessBench avg F1: MC 40.1, LLM-as-judge 46.5, human PRM800K 56.5) though Best-of-8 differences are tiny (65.9/65.3/64.9) — [Qwen PRM Lessons, 2501.07301 [full]](https://arxiv.org/html/2501.07301)
- Verdict: ceiling +1.7–2.4 pts over learned-ORM PPO (single run, short CoT); efficiency/stability/side effects unreported. Grade C for RL benefit; Grade B for "MC labels are noisy as step-correctness labels".

#### 3. OmegaPRM
- MCTS + binary search to locate the first error; >1.5M annotations; with weighted self-consistency Gemini Pro MATH500 51%→69.4%, GSM8K 86.4%→93.6%; Gemma2-27B MATH500 42.3%→58.2% — [OmegaPRM, 2406.06592 [abs]](https://arxiv.org/abs/2406.06592)
- Verdict: a data-collection method evaluated **only at test time** (per abstract). No RL training evidence. Grade: N/A for training.

#### 4. Qwen "Lessons of Developing PRMs" (MC noise, evaluation pitfalls)
- Consensus filtering: keep an instance only when MC and LLM-as-judge agree on the first-error location (~40% survives) and it matches LLM-as-judge quality with 40% of data; hard labels beat soft; recommend negative label only when no completion succeeds — [2501.07301 [full]](https://arxiv.org/html/2501.07301)
- BoN pitfalls: policies produce correct answers with flawed processes (more often on harder problems); most open PRMs detect these <50% of the time; "process-to-outcome drift": for EurusPRM, Math-Shepherd-PRM-7B, Skywork-PRM-7B, >40% of minimum step scores land on the final-answer step (i.e., the PRM behaves like an ORM) — [2501.07301 [full]](https://arxiv.org/html/2501.07301)
- ProcessBench avg F1: Qwen2.5-Math-PRM-7B 73.5, -72B 78.3, Skywork-PRM-7B 42.1, GPT-4o 61.9, o1-mini 87.9 — [2501.07301 [full]](https://arxiv.org/html/2501.07301)
- The paper does **not** test PRMs in RL; it says RL best practices "remain unexplored" — [2501.07301 [full]](https://arxiv.org/html/2501.07301)
- Downstream (independent) use of Qwen2.5-Math-PRM-7B as an RL reward: PRM-only GRPO on Qwen2.5-1.5B-Instruct drove PRM reward to 1.0 within ~100 steps while accuracy dropped to 0%, via a vacuous template ("Alright, let's solve this problem step by step") — [Reward Under Attack, 2603.06621 [full]](https://arxiv.org/html/2603.06621)

#### 5. Gao et al. "On Designing Effective RL Reward at Training Time" (PRM as dense PPO reward)
- (b) Adding learned ORM to the success reward gives no gain (MATH, Qwen2-1.5B-Instruct: SR 30.58 vs SR+OR 30.57); adding PRM reward causes hacking: step count/length explode, repeated nonsense steps → single word/emoji, repeated intermediate steps, repeated "The answer is: 110"; SR+PR at epoch 1 = 11.16 vs SR 30.58 — [Gao et al., 2410.15115 [full]](https://arxiv.org/html/2410.15115)
- (a) Fixes: Clip r = min(r_process − η, 0) (only penalties, so extra steps cannot add return); Delta r_k = r_process(k) − r_process(k+1) (telescoping, total return = α·r_process(first step) + Correct, bounded) — [2410.15115 [full]](https://arxiv.org/html/2410.15115)
- (c) PR-Clip-Delta vs SR-only across 6 Qwen2/2.5(-Math) Instruct models (1.5B/7B), MATH/GSM8K: mostly +0.2 to +1.8 pts, a few negative cells (Qwen2-Math-7B MATH greedy −0.42); the "improves across all LLMs and metrics" claim refers to gains over the base model, not over SR; seeds not reported; 5 epochs — [2410.15115 [full]](https://arxiv.org/html/2410.15115)
- Verdict: ceiling ≈ +0.5–1 pt (within noise for single seeds); stability: raw PR destabilizes, Clip/Delta stabilizes; side effects: length/repetition hacking (strong, reproduced elsewhere — see PURE, PROF). Grade A for "naive summed PRM reward hacks"; Grade C for "Clip-Delta gives gains".

#### 6. PAV — Rewarding Progress (Setlur et al., Google)
- (a) Process reward = advantage under a *prover* policy μ ≠ π: A^μ(s_h,a_h) = Q^μ − V^μ; effective reward Q^π + α·A^μ, α ∈ [0.2,0.5]; best provers are "complementary" (e.g., Best-of-4 of base) — [PAV, 2410.08146 [full]](https://arxiv.org/html/2410.08146)
- (c) Gemma 2B/9B (RFT init), MATH: 5–6x sample efficiency vs ORM-RL, >6% (abstract) / >7% (§5) accuracy gain over ORM-RL; seeds and RL steps not reported in read portion; prover fixed during RL; theory is tabular softmax NPG — [2410.08146 [full]](https://arxiv.org/html/2410.08146)
- (b) Mechanism: progress (advantage) rather than absolute step correctness (Q) is the right shaping signal; adding a Q-style PRM term just double-counts the outcome. This is consistent with the theory result that a policy's advantage function is an optimal PRM — [Jia, Rakhlin, Xie, 2502.10581 [abs]](https://arxiv.org/abs/2502.10581)
- Verdict: largest reported dense-reward gain in the literature, but single originating paper, Gemma short-CoT, unknown baseline tuning. Grade C (ceiling), B (mechanism).

#### 7. VinePPO (MC value estimates at steps)
- (a) V̂_MC(s_t) = (1/K) Σ_k R(η_k), η_k ~ π_θ(·|s_t), K=9 default (ablation K=1,3,9); Â = r + γV̂(s_{t+1}) − V̂(s_t), assigned to all tokens of a step; steps split at newlines/punctuation (MATH) — [VinePPO, 2410.01679 [full]](https://arxiv.org/html/2410.01679)
- (b) PPO critic is biased despite explained variance 0.7–0.9; near-chance at ranking five candidate next steps for most of training; within-0.05 accuracy ~65% (critic) vs 70–90% (MC) — [2410.01679 [full]](https://arxiv.org/html/2410.01679)
- (c) DeepSeekMath-7B and RhoMath-1.1B (SFT first), MATH/GSM8K; beats PPO, GRPO, RLOO, RestEM, DPO+; up to 9x fewer gradient steps (1.1B) / 2.8x (7B) to reach PPO peak; up to 3.0x / 1.51x less wall-clock; per-iteration up to 5x / 2x slower; hyperparameters tuned for PPO and reused; eval = 16 sampling seeds, but training seeds not stated; responses capped at 1024 tokens — [2410.01679 [full]](https://arxiv.org/html/2410.01679)
- (d) Independent: SPO (different group) reports SPO-chain "comparable to VinePPO" on MATH/DeepSeekMath-7B with fewer sampling points, and reuses VinePPO's baseline numbers — [SPO, 2505.23564 [full]](https://arxiv.org/html/2505.23564). PRIME's comparison to VinePPO uses a different base and a 96-step run, so not a clean replication — [PRIME, 2502.01456 [full]](https://arxiv.org/html/2502.01456)
- Verdict: efficiency strong at short CoT (B); ceiling: small, single-seed; stability: no hacking possible in principle (ground-truth verifier); side effect: generation cost scales with (#steps × K), untested beyond 1K tokens. Grade B.

#### 8. PRIME (implicit PRM, online from outcome labels)
- (a) Implicit PRM: outcome score r_φ(y) = β log π_φ(y)/π_ref(y); token process reward r_φ(y_t) = β log π_φ(y_t|y_<t)/π_ref(y_t|y_<t), β=0.05; PRM trained online with CE loss on sigmoid(r_φ(y)) vs verifier label; advantage = discounted sum of implicit process rewards minus leave-one-out mean + RLOO outcome advantage; PPO clip loss — [PRIME, 2502.01456 [full]](https://arxiv.org/html/2502.01456)
- (b) Mathematically, with γ=1 the summed process rewards telescope to the sequence-level implicit-ORM score, so PRIME redistributes a learned outcome-predictor's credit across tokens; it is not new information beyond the outcome labels. (inference from the formula above)
- (c) Qwen2.5-Math-7B-Base → SFT (230K) → RL; 592 steps for PRIME vs RLOO baseline run only to 240–320 steps; at 240 steps avg 41.0 (PRIME) vs 36.9 (RLOO); reaches RLOO's training reward in ~40% of steps ("2.5x"); per-step time 680 s vs 531 s (~28% more; paper says 24%); online PRM essential (offline PRM accuracy decays during RL); no seeds reported; inconsistent headline numbers (15.1% vs 16.7%; 5 vs 7 benchmarks); "zero" setting saturates around 50 steps; same hyperparameters for baselines — [2502.01456 [full]](https://arxiv.org/html/2502.01456)
- (d) Independent: I found no matched-compute reproduction of PRIME vs GRPO/RLOO [search, no hit]. A 2026 paper argues implicit PRMs have a train/inference mismatch (log-ratio constrained only as a sequence-level aggregate, so local credit is weakly identified) and that dense updates give "limited gains when the implicit rewards are unreliable" — [IPVRM/DistRL, 2604.13197 [abs]](https://arxiv.org/abs/2604.13197)
- Verdict: efficiency gain in steps plausible (C); ceiling not established (baseline truncated; Qwen2.5-Math); stability fine (online PRM cannot drift far from verifier); side effects not reported. Grade C.

#### 9. PURE (min-form credit assignment) and PRM reward hacking
- (a) Sum-form Q = E[Σ γ^{i−t} r^p_i]; min-form G(s_t) = min(r^p_t, …, r^p_n) for t ≤ w (w = worst step), else 0; implemented by reweighting r*_i = softmax(−r^p/T)_i · r^p_i with T=0.1; RLOO-style token baseline normalized by max length — [PURE, 2504.15275 [full]](https://arxiv.org/html/2504.15275)
- (c) Qwen2.5-7B, Qwen2.5-Math-7B, Qwen2.5-Math-1.5B; PRM = Qwen2.5-Math-7B trained on PRM800K; 8K MATH L3–5 problems; 500 steps (Math) / 1000 steps (7B); 8192 max tokens; best checkpoint reported; greedy eval; seeds not stated, no error bars. 5-benchmark avg (VR / PRM / PRM+10%VR): Math-7B 48.3/49.3/53.3; Qwen2.5-7B 43.8/45.6/47.6; Math-1.5B 40.6/40.3/43.3. PRM-trained models reach VR-level MATH-500 in ~30% of steps — [2504.15275 [full]](https://arxiv.org/html/2504.15275)
- (c) Hacking catalog: sum-form → "only thinking, not solving" and collapse by step 25 (scores ~30 vs base 39.5 at step 80); step-level baseline → one giant step (avoids delimiter); min-form PRM-only → "zero steps"/"Thank you." around step 270, in all PRM-only runs, deemed "inevitable with current discriminative PRMs"; PRM+VR with doubled process reward: collapse at step 365 within 5 gradient steps, preceded by clipped/repetitive correct responses — [2504.15275 [full]](https://arxiv.org/html/2504.15275)
- (d) Independent: PRPO's table lists PURE at 64.2 vs GRPO 61.2 on MATH (Qwen2.5-Math-1.5B, single seed, per-method early stopping) — [PRPO, 2601.07182 [full]](https://arxiv.org/html/2601.07182). Survey summarizes PURE as the canonical "summing PRM rewards encourages hacking" result — [PRM survey, 2510.08049 [full]](https://arxiv.org/html/2510.08049)
- Verdict: ceiling: PRM-only ≈ VR (±1); PRM+10% VR +3–5 (but this is a different data regime — 10% labels — so it is "PRM substitutes labels", not "PRM beats full labels"); efficiency: faster early; stability: poor (hacking in all PRM-only runs); side effects: length/repetition. Grade B for hacking dynamics (and A combined with Gao, PROF, 2603.06621); C for accuracy gains.

#### 10. PROF (PRM as a filter, not a reward) — 2025 independent evidence on "blending"
- (a) Per-rollout score = mean PRM step score minus a step-count penalty, times outcome sign; keep top-ranked correct and (variant) bottom-ranked incorrect rollouts with balanced counts; PRM never enters the advantage — [PROF, 2509.03403 [full]](https://arxiv.org/html/2509.03403)
- (c) avg@16 over 5 math benchmarks (GRPO / Blend PRM+ORM / PROF-POS / PROF-BOTH): Qwen2.5-Math-1.5B 37.2/35.3/40.2/39.6; Qwen2.5-Math-7B 49.9/47.3/50.6/51.7; LLaMA-3.2-3B-Inst 23.6/15.7/25.4/22.7; GRPO n=4 vs PROF n=8 (oversampling); matched-cost check only on 7B (~+2); seeds/steps not reported — [2509.03403 [full]](https://arxiv.org/html/2509.03403)
- Side effects: Blend (β=0.8 PRM average added to outcome) "suffers from severe reward hacking because its entropy collapses quickly toward zero", length grows uncontrollably — [2509.03403 [full]](https://arxiv.org/html/2509.03403)
- Verdict: using the PRM only for *data selection* avoids hacking; gains +1–3 with 2x sampling. Grade C.

#### 11. PRPO (entropy-segmented PRM advantage fused with outcome)
- (a) Segment at top-5 entropy spikes; Qwen2.5-Math-PRM-7B scores segments; z_process normalized with fixed prior (mean 0.5, std 0.289); A = z_process + (R − group mean) — [PRPO, 2601.07182 [full]](https://arxiv.org/html/2601.07182)
- (c) MATH pass@1: 1.5B GRPO 61.2 → +PRPO 64.4; 7B 69.8 → 72.0; but AIME25 (1.5B) GRPO 7.00 > GRPO+PRPO 4.20; per-batch process normalization collapses to ~0% after epoch 6; process-only advantage on Qwen2.5-Math-7B → shorter responses then meaningless strings; step time 46.8 s → 80.8 s with PRM; early-stop epoch chosen per method — [2601.07182 [full]](https://arxiv.org/html/2601.07182)
- Verdict: Grade C; reinforces A-grade finding that process-only advantage collapses.

#### 12. Tree / segment MC credit for critic-free RL (SPO, TreeRL, TreeRPO)
- **SPO** (a) segments at low-probability "cutpoints" (p<0.9) or fixed length; A_k = V(s_{t_{k+1}}) − V(s_{t_k}) via MC (N=4–9); advantage applied only to low-probability tokens (mask) — [SPO, 2505.23564 [full]](https://arxiv.org/html/2505.23564)
  - (c) RhoMath-1.1B GSM8K: SPO-chain beats PPO/GRPO by 6–12 pts (baselines mostly copied from VinePPO); R1-Distill-Qwen-1.5B MATH500 GRPO vs SPO-tree: 2K ctx 0.62 vs 0.736, 4K 0.752 vs 0.828, **32K 0.84 vs 0.848**; AIME24 identical at 32K (0.333); training only to 4K context; single runs — [2505.23564 [full]](https://arxiv.org/html/2505.23564)
  - Notably, the *mask alone* lifts GRPO from 45.7% to 53.6% on GSM8K — i.e., much of the gain is a token-selection effect, not MC credit — [2505.23564 [full]](https://arxiv.org/html/2505.23564)
- **TreeRL** (a) EPTree forks at top-N highest-entropy tokens; node value V(s_n) = fraction of correct leaves; step reward R(s_n) = |L(s_n)|^{-1/2}·[(V(s_n)−V(root)) + (V(s_n)−V(parent))] — [TreeRL, 2506.11902 [full]](https://arxiv.org/html/2506.11902)
  - (c) ChainRL→TreeRL avg: Qwen2.5-14B-SFT 41.6→44.5; GLM4-9B 27.2→29.3; R1-Distill-7B 64.5→65.8; 8,192 max tokens; **not compute-matched** (480 vs 256 responses per update), search config chosen on Omni-MATH-500 which is also an eval set; ~2x slower rollout; no seeds; hacking resistance asserted not tested — [2506.11902 [full]](https://arxiv.org/html/2506.11902)
- **TreeRPO** (a) 8-ary tree, depth 3, fixed 384-token "steps"; parent reward = mean of children; sibling-group GRPO-style advantage — [TreeRPO, 2506.05183 [full]](https://arxiv.org/html/2506.05183)
  - (c) Qwen2.5-Math-1.5B: GRPO 32.7 → 35.6; Qwen2.5-Math-7B: 40.7 → 41.4; 18.1% shorter responses; 300 steps; no seeds; abstract/body inconsistencies — [2506.05183 [full]](https://arxiv.org/html/2506.05183)
- Verdict (family): unbiased credit (ground-truth leaves), +1–3 pts at small scale, gains shrink with model size (TreeRPO 7B +0.7, TreeRL R1-distill +1.3) and with context length (SPO 32K +0.8 on MATH500, 0 on AIME). Grade B for "tree MC credit helps small models / truncated contexts"; C for ceiling claims at scale.

#### 13. "GRPO is secretly a PRM" (implicit step credit in outcome-only GRPO)
- (b) With DAPO token-level loss and μ=1, outcome-reward GRPO equals a PRM-aware objective whose step reward is the MC mean outcome of trajectories sharing that prefix; 99.8% of size-6 groups had shared prefixes; flaw: each shared step is weighted by its frequency |λ|; λ-GRPO rescales by 1/|λ| — [2509.21154 [full]](https://arxiv.org/html/2509.21154)
- (c) R1-Distill-Qwen-1.5B and Llama-3.2-1B, OpenRS, 1000 steps, group 6, **batch size 4**; λ-GRPO beats GRPO on 15/20 cells; Qwen avg 0.5576 vs 0.4837 (β=0) — [2509.21154 [full]](https://arxiv.org/html/2509.21154)
- Verdict: mechanism B (it is a clean identity), empirical C (tiny batch, small models).

#### 14. Token-level / key-token credit (critic-free)
- High-entropy "forking" tokens (~20%): restricting the policy gradient to them matches full-gradient on Qwen3-8B and beats it on Qwen3-32B (+11.04 AIME25, +7.71 AIME24) and Qwen3-14B (+4.79/+5.21); training only on the 80% low-entropy tokens degrades performance — [Wang et al., 2506.01939 [abs]](https://arxiv.org/abs/2506.01939)
- KTAE: token-level advantage from rollout correctness statistics added to GRPO/DAPO advantage; claims higher accuracy with shorter responses on 5 math benchmarks; no numbers in abstract — [KTAE, 2505.16826 [abs]](https://arxiv.org/abs/2505.16826)
- Critical tokens: tokens that "significantly influence incorrect outcomes", found by rollout sampling + contrastive estimation, used in token-level DPO (cDPO) on Llama-3 8B/70B and DeepSeek-math-7B (GSM8K, MATH500) — [2411.19943 [abs]](https://arxiv.org/abs/2411.19943)
- ARPO (agentic): entropy rises after tool calls; adaptive branching at those points + "advantage attribution estimation"; 13 benchmarks; claims half the tool-call budget — [ARPO, 2507.19849 [abs]](https://arxiv.org/abs/2507.19849)
- OREAL: outcome-only RL with a token-level reward model to pick important tokens; 7B reaches 94.0 MATH-500, 32B 95.0; no token-RM ablation in abstract — [OREAL, 2502.06781 [abs]](https://arxiv.org/abs/2502.06781)
- Verdict: all originating-paper only (C), and only 2506.01939 shows a large effect, which is a *gradient-masking* change rather than a new reward signal.

#### 15. Generative PRMs / verifiers
- GenPRM: CoT + code verification per step; Relative Progress Estimation; 23K MATH training examples; 1.5B beats GPT-4o and 7B beats Qwen2.5-Math-PRM-72B on ProcessBench *with test-time scaling*; "critic for policy refinement"; no RL policy training in abstract — [GenPRM, 2504.00891 [abs]](https://arxiv.org/abs/2504.00891)
- ThinkPRM: long-CoT verifier fine-tuned on 1% of PRM800K labels; beats discriminative PRMs under BoN and guided search; +8% on GPQA-Diamond subset, +4.5% on LiveCodeBench; no RL — [ThinkPRM, 2504.16828 [abs]](https://arxiv.org/abs/2504.16828)
- R-PRM: +11.9 / +8.5 F1 over baselines on ProcessBench / PRMBench; >8.5 pts when guiding reasoning on 6 datasets; no RL policy training stated — [R-PRM, 2503.21295 [abs]](https://arxiv.org/abs/2503.21295)
- RL Tango (NeurIPS 2025): generator and generative process-level verifier co-trained by RL; verifier trained solely from outcome-level verification correctness; claims fixed/SFT verifiers are hackable and RL-trained verifier is more robust; no numbers in abstract — [Tango, 2505.15034 [abs]](https://arxiv.org/abs/2505.15034)
- StepWiser: generative stepwise judge trained with RL from "relative outcomes of rollouts"; claims training-time policy improvement and better inference-time search — [StepWiser, 2508.19229 [abs]](https://arxiv.org/abs/2508.19229)
- Verdict: generative PRMs are well-evidenced as **verifiers/rerankers** (ProcessBench-type metrics, BoN), but I found no matched-compute RL study where a generative step-level PRM as dense reward beats outcome-only RLVR at final accuracy. Grade C (training use).

#### 16. Side-effect evidence specific to optimizing against process judges
- PRM-only GRPO (Qwen2.5-1.5B-Instruct on AIME24 prompts): Skywork-PRM-1.5B reward 0.1→0.8+ while accuracy stays ~0, peak 3–4%; Qwen2.5-Math-PRM-7B reward →1.0 in ~100 steps with 0% accuracy; on held-out AIME25 43% of the reward gain attributable to style (0.641 vs 0.472 after semantics-preserving rephrasing; base 0.246); PRMs act as "fluency detectors" — [Reward Under Attack, 2603.06621 [full]](https://arxiv.org/html/2603.06621) (caveat: AIME24 as training set for a 1.5B model is an adversarially hard setting where outcome reward is almost always 0.)
- Putting a CoT monitor into the RL reward helps under light optimization but under heavy optimization produces "obfuscated reward hacking"; OpenAI recommends a "monitorability tax" (don't optimize the CoT hard) — [Baker et al., 2503.11926 [abs]](https://arxiv.org/abs/2503.11926)
- Survey notes step-level signals have higher length variance (PRM800K step-level token-length std 71.7 vs 50.6 trajectory-level) and are more prone to length/verbosity hacking — [PRM survey, 2510.08049 [full]](https://arxiv.org/html/2510.08049)

### Inferences
- **Math summary.** Outcome-only GRPO/RLOO is an unbiased (REINFORCE-with-baseline) estimator of the trajectory-level gradient; dense credit can only help through (i) variance reduction / better per-token attribution or (ii) adding *information* not in R(y). On-policy MC methods (VinePPO/SPO/TreeRL/TreeRPO) are type (i) and keep the objective unbiased; learned PRMs are type (ii) but the extra information is a biased proxy, and the RL optimizer finds its errors (length, repetition, empty output, vacuous templates). Bounded/telescoping transforms (Gao's Delta, PURE's min, PRIME's sum-to-implicit-ORM, PRPO's fixed normalization, PROF's filter-only) all work by making the PRM term small or non-accumulating — which is also why their measured ceilings over outcome-only are small (≈0–3 pts).
- Gao's Delta is essentially potential-based shaping (telescoping sum), which classically preserves the optimal policy; that predicts "safe but small" gains, matching the +0.2–1.8 pts observed.
- The two exceptions with large reported gains (PAV +6–7%, high-entropy-token masking +7–11 AIME on Qwen3-32B) are both single originating papers; PAV's mechanism (advantage under a different prover) is the most theoretically principled of the learned-PRM designs.

### Gaps
- I did not read Uesato et al. beyond the abstract (base model, RL algorithm details unverified).
- DeepSeekMath's PS-vs-OS figure magnitudes are not in the text I could read (figure-only).
- Implicit PRM origin paper ("Free Process Rewards without Process Labels", arXiv 2412.01981) not read.
- KTAE, critical tokens, ARPO, OREAL, Tango, StepWiser, GenPRM, ThinkPRM, R-PRM: abstract-only; formulas and numbers not verified.
- 2608.03119 ("Don't Peek at the Answer") reportedly shows PRM baselines collapsing in 30–40 steps — [snippet only](https://arxiv.org/pdf/2608.03119); not verified.

---

## Q2. Controlled evidence: under matched compute and same base model, does PRM / dense credit beat outcome-only GRPO/PPO on final accuracy, or only early efficiency? Does it persist at longer training, or does hacking dominate?

### Takeaway
No study I found meets the bar of same base, matched compute, multiple seeds, long training, and a tuned outcome-only baseline. Across ~12 controlled-ish comparisons, dense credit gives (a) a consistent **early-efficiency** gain (2–9x fewer steps), (b) **final-accuracy** gains of roughly 0–3 points that are single-seed, mostly on Qwen2.5-Math, and usually shrink with model size, context length, or proper compute matching, and (c) with learned PRMs as rewards, hacking dominates on longer horizons unless an outcome anchor and bounded transform are used. Overall verdict: efficiency grade B; ceiling grade D (contradictory/within noise); "learned PRM reward → hacking" grade A.

### Cited Findings
**Efficiency (consistent):**
- VinePPO: 2.8–9x fewer gradient steps, 1.5–3x less wall-clock to PPO's peak (short CoT, ≤1024 tokens) — [2410.01679 [full]](https://arxiv.org/html/2410.01679)
- PRIME: RLOO's training reward in ~40% of steps; ~28% more time per step — [2502.01456 [full]](https://arxiv.org/html/2502.01456)
- PURE: VR-level MATH-500 in ~30% of steps — [2504.15275 [full]](https://arxiv.org/html/2504.15275)
- PAV: 5–6x sample efficiency vs ORM-RL — [2410.08146 [full]](https://arxiv.org/html/2410.08146)
- λ-GRPO: ~10% higher validation accuracy in under half the steps — [2509.21154 [full]](https://arxiv.org/html/2509.21154)

**Final accuracy (small, noisy, often not compute-matched):**
- Gao et al. PR-Clip-Delta vs SR: +0.2 to +1.8, some negative — [2410.15115 [full]](https://arxiv.org/html/2410.15115)
- PURE PRM-only vs VR: +1.0, +1.8, −0.3 (three bases) — [2504.15275 [full]](https://arxiv.org/html/2504.15275)
- PROF vs GRPO: +0.7 to +3.0, with 2x rollouts; ~+2 at matched cost (7B only) — [2509.03403 [full]](https://arxiv.org/html/2509.03403)
- PRPO vs GRPO: +2.2/+3.2 on MATH but −2.8 on AIME25 (1.5B) — [2601.07182 [full]](https://arxiv.org/html/2601.07182)
- TreeRPO vs GRPO: +2.9 (1.5B), +0.7 (7B) — [2506.05183 [full]](https://arxiv.org/html/2506.05183)
- TreeRL vs ChainRL: +1.3 to +2.9 but 480 vs 256 responses per update — [2506.11902 [full]](https://arxiv.org/html/2506.11902)
- SPO-tree vs GRPO on MATH500: +11.6 at 2K context, +0.8 at 32K — [2505.23564 [full]](https://arxiv.org/html/2505.23564)
- Math-Shepherd PRM-PPO vs learned-ORM-PPO: +0.8 to +2.4 — [2312.08935 [full]](https://arxiv.org/html/2312.08935)
- PRIME vs RLOO at equal 240 steps: 41.0 vs 36.9; RLOO not run to PRIME's 592 steps — [2502.01456 [full]](https://arxiv.org/html/2502.01456)
- Small-scale rule-based process reward (step matches GSM8K ground-truth intermediate values), Qwen2.5-0.5B: process-only 63.73% vs outcome-only 53.75% test accuracy; but process-only had the *lowest* trace validity (0.22 vs 0.40), contradicting the paper's own abstract; single run — [Reward Granularity in RLVR, 2607.02869 [full]](https://arxiv.org/html/2607.02869v1)

**Longer training / hacking:**
- PURE: sum-form collapse by step 25; min-form PRM-only hacked around step 270 in all runs; PRM+VR collapse at step 365 with doubled weight — [2504.15275 [full]](https://arxiv.org/html/2504.15275)
- Gao et al.: SR+PR falls to 11.16 (from 30.58) after one epoch — [2410.15115 [full]](https://arxiv.org/html/2410.15115)
- PROF: Blend (PRM+ORM) entropy collapses and length explodes, ends below GRPO — [2509.03403 [full]](https://arxiv.org/html/2509.03403)
- PRPO: per-batch-normalized variant collapses after epoch 6; process-only collapses — [2601.07182 [full]](https://arxiv.org/html/2601.07182)
- PRM-only GRPO: reward ↑ to 0.8–1.0 with accuracy ~0 — [2603.06621 [full]](https://arxiv.org/html/2603.06621)
- PRIME avoids this by updating the PRM online; an offline PRM's accuracy decays during RL — [2502.01456 [full]](https://arxiv.org/html/2502.01456)
- DeepSeek-R1: model-based rewards become "more susceptible to exploitation as training progresses"; preference RM reward was only enabled for the last 400 of 1,700 steps of stage 2 — [DeepSeek-R1, 2501.12948 [full]](https://arxiv.org/html/2501.12948)

**Confounders that apply to most of the above:**
- Qwen2.5-Math-7B gains +21.4 pts on MATH-500 from *random* rewards vs +29.1 from ground truth; such spurious-reward gains often fail on Llama3/OLMo2 — [Spurious Rewards, 2506.10947 [abs]](https://arxiv.org/abs/2506.10947). PURE, PRIME, PRPO, PROF, TreeRPO all use Qwen2.5-Math bases.
- Math reasoning benchmarks are highly sensitive to seeds, decoding, prompt format, and hardware; most re-evaluated RL methods gave "modest improvements, far below prior claims", and RL overfits small benchmarks like AIME24 — [Sober Look, 2504.07086 [abs]](https://arxiv.org/abs/2504.07086). AIME has 30 problems (1 problem = 3.3 pts), so SPO/TreeRL/PRPO AIME deltas are within noise — [SPO, 2505.23564 [full]](https://arxiv.org/html/2505.23564)
- Seeds: none of PRIME, PURE, PROF, PRPO, TreeRL, TreeRPO, SPO, Gao et al., Math-Shepherd report multiple training seeds (see each [full] reading above). VinePPO uses 16 eval seeds but training seeds unstated.
- Baseline tuning: VinePPO reused PPO hyperparameters; PRIME used the same hyperparameters for baselines; SPO copied baselines from VinePPO; PRPO picked early-stop per method (sources as above).

### Inferences
- The most defensible statement: **dense credit is an efficiency/variance-reduction device whose benefit is largest when outcome signal per sample is scarce** — small models, short context budgets (truncation), hard prompts with low pass rate, or few verifiable labels (PURE's 10% VR). As the base model and context budget grow, the outcome-only baseline catches up (SPO 2K→32K; TreeRPO 1.5B→7B; TreeRL on R1-distill).
- For learned PRMs, "remains beneficial at longer training" is not demonstrated anywhere; the documented trajectory is early speedup then hacking unless the PRM is bounded, online-updated, or used only for filtering.

### Gaps
- No multi-seed, compute-matched, ≥1K-step study of PRM vs GRPO on a non-Qwen-Math base found.
- No study measuring the full cost (PRM inference + training + MC rollouts) against simply spending that compute on more GRPO samples/steps, except PROF's single 7B matched-cost check.

---

## Q3. Is there a 2026 consensus on whether PRMs help in RL training (vs test-time search)? Which frontier labs use process supervision in RL?

### Takeaway
Rough consensus: PRMs (especially generative ones) are useful **test-time verifiers/rerankers**, but discriminative step-level PRMs as **dense RL rewards** are not part of any published frontier reasoning-model recipe; the published frontier recipes (DeepSeek-R1, Kimi k1.5) explicitly use outcome-level rewards and reject PRMs/value nets. The one notable frontier-lab use of "process-ish" supervision in RL is DeepSeekMath-V2's generative proof verifier, which judges the whole proof (trajectory-level score of reasoning quality), not per-step dense credit. Academic surveys are more optimistic than the controlled evidence supports.

### Cited Findings
- DeepSeek-R1 (v1, "Unsuccessful Attempts"): PRM has "three main limitations": hard to "explicitly define a fine-grain step in general reasoning"; hard to judge whether an intermediate step is correct ("Automated annotation using models may not yield satisfactory results, while manual annotation is not conducive to scaling up"); a model-based PRM "inevitably leads to reward hacking" and retraining it costs resources and complicates the pipeline. Conclusion: PRM shows "a good ability to rerank the top-N responses" or assist guided search, but "its advantages are limited compared to the additional computational overhead it introduces during the large-scale reinforcement learning process in our experiments." — [DeepSeek-R1 v1, 2501.12948v1 [full]](https://arxiv.org/html/2501.12948v1)
- DeepSeek-R1 later version: "we abstain from applying neural reward models—whether outcome-based or process-based—to reasoning tasks" citing reward hacking and retraining cost; PRM discussion moved to Appendix G.2 — [DeepSeek-R1 latest arXiv, 2501.12948 [full]](https://arxiv.org/html/2501.12948)
- Kimi k1.5: "We exclude the value network in our training system"; argues erroneous-but-recovered steps are "extremely valuable for training the model to generate long CoT", so value-based credit "may not be suitable"; no process reward models; math uses a generative CoT reward model (~98.5% spot-check accuracy) that judges final-answer correctness — [Kimi k1.5, 2501.12599 [full]](https://arxiv.org/html/2501.12599)
- DeepSeekMath-V2 (Nov 2025): proof verifier scores whole proofs on a 0/0.5/1 rubric identifying concrete issues; generator trained with the verifier as reward model; a meta-verifier audits the verifier's analyses (quality 0.85→0.96) — [DeepSeekMath-V2, 2511.22570 [snippet, via search summary]](https://arxiv.org/abs/2511.22570)
- DeepSeekMath (2024) reported GRPO+process supervision > outcome supervision in its ablation (magnitudes figure-only) — [2402.03300 [full]](https://arxiv.org/html/2402.03300); R1 (2025) abandoned PRMs — [2501.12948v1](https://arxiv.org/html/2501.12948v1). Same lab, opposite practical conclusion once RL scaled.
- OpenAI: Lightman et al. explicitly did not do RL with PRMs — [2305.20050 [full]](https://arxiv.org/html/2305.20050); OpenAI's CoT-monitoring paper warns against optimizing CoT directly (obfuscation) — [2503.11926 [abs]](https://arxiv.org/abs/2503.11926)
- Google: PAV (Setlur et al.) shows dense progress rewards in RL on Gemma — [2410.08146 [full]](https://arxiv.org/html/2410.08146); OmegaPRM used at test time — [2406.06592 [abs]](https://arxiv.org/abs/2406.06592). No evidence found that Gemini reasoning RL uses PRMs.
- Qwen: built the strongest open PRMs but did not test them in RL — [2501.07301 [full]](https://arxiv.org/html/2501.07301)
- 2025–26 survey (ACL 2026) concludes PRMs "can serve as actionable dense rewards that significantly improve RL training" but presents no controlled comparison of its own and notes that how PRM outputs enter the loss "critically affects training stability" — [PRM survey, 2510.08049 [full]](https://arxiv.org/html/2510.08049)
- A 2026 IBM ACL paper finds deterministic rule-based step verifiers outperform learned step judges as RL process rewards in a structured domain (medical risk-of-bias), learned judges being noisy/misaligned — [VPRM, 2601.17223 [snippet]](https://arxiv.org/html/2601.17223)
- Search for 2026 frontier technical reports confirming step-level PRM in RL training returned none — [search result summary; no primary hit](https://arxiv.org/pdf/2510.08049)

### Inferences
- The field has converged on a two-tier view: (1) **trajectory-level** model-based judgment (generative verifiers, rubric graders, DeepSeekMath-V2-style proof checkers) is used in frontier RL where answers are not checkable; (2) **step-level dense** learned rewards are an academic topic with small, fragile gains. "Process supervision" at frontier labs, where it appears, is mostly about *grading the reasoning as a whole*, not per-step advantages.
- The R1 limitation list maps directly onto the evidence: step definition (TreeRPO uses fixed 384-token chunks; PRPO's entropy splits vs uniform splits 64.4% vs 29.8% show step definition is load-bearing), label quality (Qwen lessons), hacking (Gao, PURE, PROF, 2603.06621).

### Gaps
- Closed labs (OpenAI o-series, Anthropic, Google Gemini) do not publish whether process supervision enters RL; I found no primary source either way.
- DeepSeek-R1 Nature version not read directly (arXiv latest version's ToC places PRM text in Appendix G.2; content assumed consistent with v1).
- DeepSeekMath-V2 details (whether verifier reward is per-step or whole-proof only; ablation vs outcome-only) read only via search summary.
- Qwen3 / Kimi K2 / GLM / MiniMax 2025–26 reports were not checked for PRM use in this pass.

---

## Q4. How does dense credit interact with long CoT (thousands of tokens)? Is per-token advantage from outcome-only actually a bottleneck?

### Takeaway
Evidence suggests outcome-only per-token advantage is **not a hard bottleneck** for long CoT at frontier scale (R1, k1.5 reach long CoT with outcome rewards; SPO's gain vanishes at 32K; dense-credit papers mostly train at ≤4–8K tokens). The credit that matters seems concentrated at a small set of high-entropy "forking" tokens, so the productive direction in 2025–26 has been branching/masking at those tokens (TreeRL, ARPO, entropy-token gradients) rather than learned per-step rewards. The main counter-signal is VAPO: a carefully engineered learned critic (dense credit) reportedly beats critic-free DAPO by >10 AIME24 points on Qwen-32B.

### Cited Findings
- Training lengths of dense-credit studies: VinePPO ≤1024 response tokens — [2410.01679](https://arxiv.org/html/2410.01679); Math-Shepherd 512 — [2312.08935](https://arxiv.org/html/2312.08935); SPO trains only to 4K ("compute limits") — [2505.23564](https://arxiv.org/html/2505.23564); PRPO 2048 — [2601.07182](https://arxiv.org/html/2601.07182); PROF 4096 — [2509.03403](https://arxiv.org/html/2509.03403); TreeRL and PURE 8192 — [2506.11902](https://arxiv.org/html/2506.11902), [2504.15275](https://arxiv.org/html/2504.15275)
- SPO-tree vs GRPO on R1-Distill-1.5B, MATH500: +11.6 at 2K, +7.6 at 4K, +0.8 at 32K; AIME24 equal at 32K — [2505.23564 [full]](https://arxiv.org/html/2505.23564)
- Kimi k1.5: flawed-but-recovered steps are valuable for learning long CoT; value-based step credit would penalize exploration — [2501.12599 [full]](https://arxiv.org/html/2501.12599)
- Qwen lessons: policies often reach correct answers through flawed steps — [2501.07301 [full]](https://arxiv.org/html/2501.07301) (step-correctness PRMs would penalize reflection/backtracking that long CoT relies on; this is the same phenomenon viewed negatively)
- Forking tokens: ~20% high-entropy tokens carry RLVR's gains; gradient on those only ≥ full gradient at 14B/32B — [2506.01939 [abs]](https://arxiv.org/abs/2506.01939)
- TreeRL forks at highest-entropy tokens; ~2x slower rollouts; small gain on R1-distilled 7B (64.5→65.8) — [2506.11902 [full]](https://arxiv.org/html/2506.11902)
- ARPO branches where entropy spikes after tool calls in agentic RL — [2507.19849 [abs]](https://arxiv.org/abs/2507.19849)
- VAPO (value-based PPO, Qwen-32B base): 60.4 AIME24, >10 points over DAPO and DeepSeek-R1-Zero-Qwen-32B "under identical settings", within 5,000 steps, no crashes across multiple runs — [VAPO, 2504.05118 [abs]](https://arxiv.org/abs/2504.05118)
- GRPO's outcome advantage already yields implicit prefix-level MC credit via shared prefixes — [2509.21154 [full]](https://arxiv.org/html/2509.21154) (but with long CoT and moderate group sizes, sampled trajectories diverge early, so shared prefixes are short — my inference, not tested in the paper)
- VinePPO's MC cost scales with #steps × K; the paper does not measure long CoT — [2410.01679 [full]](https://arxiv.org/html/2410.01679)
- PURE observed length/clip-ratio/repetition spikes preceding PRM-induced collapse at 8K budget — [2504.15275 [full]](https://arxiv.org/html/2504.15275)

### Inferences
- At long CoT, per-step MC value estimation is prohibitively expensive (hundreds of "steps" × K rollouts × thousands of tokens), and step-correctness PRMs are misaligned with the backtracking behavior long CoT needs. Both forces push practice toward (a) outcome-only with better token weighting (entropy masks, λ-GRPO, length normalization) or (b) a learned critic with heavy engineering (VAPO).
- The gap-shrinkage with context length (SPO) suggests much of the short-context benefit of dense credit is compensating for truncation (rewarding partial progress when the answer is cut off), not fixing a fundamental credit problem.
- VAPO is the strongest evidence that per-token credit *can* matter at long CoT, but it is a learned critic (not a PRM) and its comparison is against DAPO numbers in the same lab's setting; one originating paper (grade C).

### Gaps
- No study found that trains a PRM- or MC-credit method at ≥16K response length with matched compute against GRPO/DAPO.
- VAPO read only at abstract level; its ablations (value pretraining, decoupled/length-adaptive GAE) and whether gains replicate independently were not verified.
- Whether tree-branching's benefit at long CoT comes from credit assignment or from extra diverse samples (TreeRL's 480 vs 256) is unresolved.

---

### Summary verdict table (for the report writer)

| Method | Signal added | Ceiling | Efficiency | Stability | Side effects | Grade |
|---|---|---|---|---|---|---|
| PRM vs ORM (Lightman) | human step labels | test-time only | — | — | — | B (test-time) |
| Math-Shepherd PPO | MC step labels → learned PRM | +1–2 vs learned ORM, ≤512 tok | n/r | n/r | n/r | C |
| OmegaPRM | MCTS first-error labels | test-time only | — | — | — | N/A |
| Gao et al. PR / Clip-Delta | learned PRM, bounded | ≈+0.5–1 | n/r | raw PR collapses | length/repetition | A (hacking), C (gain) |
| PAV | prover advantage | +6–7 vs ORM-RL | 5–6x | n/r | n/r | C |
| VinePPO | on-policy MC V(s) | small | 3–9x steps | high | compute | B |
| PRIME | online implicit PRM | not established | ~2.5x steps | ok (online) | n/r | C |
| PURE | min-form PRM | ≈VR (PRM-only) | ~3x steps | hacks in all PRM-only runs | empty output, length | B (hacking), C (gain) |
| PROF | PRM as filter | +1–3 | — | stable | 2x sampling | C |
| PRPO | entropy-seg PRM + outcome | +2–3 MATH, −AIME25 | n/r | variants collapse | — | C |
| SPO / TreeRL / TreeRPO | tree MC credit | +1–3, shrinks w/ size & ctx | yes | stable | 2x rollout time | B (small/short), C (scale) |
| λ-GRPO | rescale implicit PRM | +3–7 (tiny batch) | ~2x | n/r | — | C |
| Entropy-token gradient | mask to forking tokens | +5–11 AIME (Qwen3 14/32B) | — | n/r | — | C |
| Generative PRMs (GenPRM, ThinkPRM, R-PRM) | CoT step verification | test-time only | — | — | — | C (training) |
| DeepSeek-R1 / Kimi k1.5 stance | outcome only | frontier SOTA without PRM | — | — | — | — |

n/r = not reported.
