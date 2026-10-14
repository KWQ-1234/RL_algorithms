# Verification: data-vs-loss, semi-online DPO, PRM reward hacking

Checked 2026-10-14 against arXiv PDFs (text extracted with PyMuPDF), not summaries.

---

## 1. WPO (arXiv 2406.11827): data setting matters more than DPO→WPO

**Verdict: VERIFIED for Mistral-base, with three caveats.** All numbers below come from **Table 1** (p. 6), one table with two model column groups. Metric: AlpacaEval 2 LC win rate, mean (std) over 5 random seeds.

| Data setting | Method | Mistral-Base (7B, zephyr SFT) | Llama-3-Instruct (8B) |
|---|---|---|---|
| Off-policy (UltraFeedback) | DPO | 20.6 (0.7) | 28.2 (0.5) |
| Off-policy | WPO | 24.4 (1.4) | 33.8 (1.3) |
| Hybrid | DPO | 37.9 (1.2) | 44.2 (1.2) |
| Hybrid | WPO | 42.0 (1.7) | 45.8 (1.3) |
| Hybrid + UltraFeedback | WPO | 43.1 (1.1) | 48.6 (1.3) |

- Data effect (DPO, off→hybrid): Mistral **+17.3** (20.6→37.9), Llama +16.0 (28.2→44.2).
- Loss effect (DPO→WPO, same data): Mistral off +3.8, hybrid +4.1. Llama off +5.6, hybrid **+1.6**.
- Caveat 1: "+3.8 to +5.6" covers only three of the four cells. The full range is **+1.6 to +5.6**, and the smallest gain is Llama in the hybrid setting.
- Caveat 2: Table 1 has **hybrid** only, not pure on-policy. Hybrid means 5 SFT samples plus 1 gpt-4-turbo output, labeled by gpt-4-turbo (p. 6). On-policy appears only in Fig. 3/4 bar charts, with no numbers. Per §4.3, on Mistral-base off-policy is *slightly better* than on-policy. The "+17" is therefore a hybrid gain (on-policy samples, a stronger GPT-4-turbo response and a GPT-4-turbo labeler together), not an on-policy gain.
- Caveat 3: going off→hybrid changes both the responses and the preference labeler (UltraFeedback labels vs gpt-4-turbo additive scoring). The two effects are confounded.

---

## 2. Lanchantin et al. (arXiv 2506.21495): offline vs semi-online vs online DPO vs GRPO

**Verdict: VERIFIED. All 8 numbers match.** Seed model: Llama-3.1-8B-Instruct.

**Table 1** (p. 5), Math500 acc (std err): Seed 47.4 (1.6); Offline DPO (s=∞) **53.7 (1.6)**; Semi-online DPO (s=100) **58.9 (1.2)**; Semi-online (s=10) 57.2 (1.1); Online DPO (s=1) **58.7 (1.2)**; GRPO **58.1 (1.3)**.

**Table 2** (p. 6), AlpacaEval LC win rate, GPT-4-1106 judge (std err): Seed 27.3 (1.3); Offline **53.2 (1.5)**; Semi-online (s=10) **81.6 (1.0)**; Semi-online (s=5) 78.7 (1.2); Online **83.1 (1.0)**; GRPO **75.2 (1.2)**.

- Note: "semi-online" refers to different sync intervals in the two tables: **s=100** for Math500, **s=10** for AlpacaEval.
- Other judge: with GPT-4o as AlpacaEval judge, the four values are 38.3 / 59.4 / 60.1 / 55.0. ArenaHard cells are reported with 95% CIs.
- **Uncertainty reported:**
  - Math: the paper reports "Standard error is computed over N = 50 random seeds". These are 50 sampled generations per problem (temp 0.6, top-p 0.9), i.e. **evaluation-sampling** seeds.
  - AlpacaEval: the paper gives a standard error. ArenaHard: 95% CIs.
  - **Training seeds: no multi-seed training is reported.** Each regime looks like a single training run with the best checkpoint picked on validation. So the error bars cover evaluation noise only, not run-to-run variance.

---

## 3. Gao et al. (arXiv 2410.15115): PRM reward hacking at RL training time

**Verdict: numbers VERIFIED, but the claim's framing is WRONG.** Source: **Table 1** (p. 9), Qwen2-1.5B-Instruct, MATH test, greedy (sampling):

| Row | Greedy | Sampling |
|---|---|---|
| Base model | 24.90 | 16.79 |
| Success Reward (SR) | **30.58** | 27.05 |
| SR + OR | 30.57 | 27.12 |
| SR + PR (E1) | **11.16** | 14.68 |
| SR + PR-Normed (E2) | 29.66 | 27.14 |
| SR + PR-Normed (E5) | 12.36 | 12.84 |
| SR + PR-Clip | 30.30 | 28.40 |
| SR + PR-Delta | 30.68 | 27.96 |
| SR + PR-Clip-Delta | **31.44** | 28.20 |

- **Framing correction:** the model did not go from 30.58 to 11.16 within one run. 30.58 is the **SR-only model after 5 epochs**. 11.16 is **SR+PR at epoch 1**, chosen as the early epoch because PR "degrades in later epochs". Both are 5-epoch PPO runs from a base of **24.90**. Correct wording: SR+PR scores 11.16 after 1 epoch, versus 24.90 for the base model and 30.58 for success-reward-only training. PR also uses a learned PRM, i.e. dense step rewards added to SR, with dense coefficient 1 and success coefficient 5.
- **Hacking behaviors** (§4.1, Fig. 1–3):
  - Generation length and step count rise sharply.
  - The model writes many short steps with "unnecessary or meaningless information".
  - In the end it "outputs only a single word or even emoji".
  - The PRM rewards these steps positively, so the return grows without bound as steps are added.
- **About "The answer is":** this string is *not* a reported learned behavior. It appears only in **Appendix C** synthetic probes (Fig. 3: (a) repeat nonsense steps such as "Step done." or "Step end.", (b) repeat an intermediate step, (c) repeat the last sentence "The answer is: 110"). PR gives higher returns than the ground truth for (a) and (b). Length penalty and length normalization still prefer the repeats in (a)/(b) or (c).
- **Fixes:**
  - **Clip** (Eq. 5): r = min(r_process − η, 0). Rewards are bounded above by 0, so only low-scoring steps are penalized.
  - **Delta** (Eq. 6): r_k = r_process(p_k) − r_process(p_{k+1}), and the last-step reward is dropped. The return telescopes to α·r_process(first step) + Correct, so it is bounded.
  - Effect: SR-only 30.58, PR-Clip 30.30, PR-Delta 30.68, PR-Clip-Delta 31.44. In other words they **stop the collapse**. The gain over SR-only is only **+0.86** greedy, from one run with no error bars.
  - In Table 3, Pass@16 is 61.70 both with and without Clip-Delta.
  - Paper typo: Table 3 prints "30.58↑4.68", but 30.58 − 24.90 = 5.68.
