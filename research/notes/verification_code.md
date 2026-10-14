# Code verification: verl / OpenRLHF / slime / SkyRL

Read-only checkouts under `/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/`. All paths below are relative to each repo root. Line numbers are for the commits listed here.

| Repo | `git log -1 --format='%h %cd'` |
|---|---|
| verl | `7dc39fe Fri May 22 23:32:51 2026 +0800` |
| OpenRLHF | `c3188af Fri May 15 13:25:02 2026 +0800` |
| slime | `6961f59 Thu May 21 16:45:52 2026 +0800` |
| SkyRL | `ccc181e Thu May 21 21:25:55 2026 -0700` |
| TRL | no local checkout (none under `_external_repos/`, and no `trl/trainer/dpo_trainer.py` anywhere on disk). **Part D skipped.** |

---

## A. verl

### A1. Policy loss and `agg_loss`

- `compute_policy_loss` (`verl/trainer/ppo/core_algos.py:1202-1275`) is marked `@deprecated("...compute_policy_loss_vanilla")` (line 1202). It contains the same math and takes `clip_ratio_c=3.0` as a keyword default (1211).
- Live path: `compute_policy_loss_vanilla`, registered as `"vanilla"` (`core_algos.py:1278-1369`). The ratio, clipping and dual-clip are at lines 1329-1354. The rollout-IS multiply and the `agg_loss` call are at 1357-1362.
- `agg_loss`: `core_algos.py:1138-1199`. Its branches are at 1168-1197.
- Clip params resolve at `core_algos.py:1313-1318`. `clip_ratio_low` and `clip_ratio_high` fall back to `clip_ratio` when they are `None`. `clip_ratio_c` comes from `config.get("clip_ratio_c", 3.0)`. `assert clip_ratio_c > 1.0` is at 1324.

Trimmed excerpt. Statements that wrap across several source lines are joined onto one line. `# ...` marks elided lines.

```python
# core_algos.py:1329-1354  (compute_policy_loss_vanilla)
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
# ...
pg_losses1 = -advantages * ratio
# ...
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)   # 1340-1342
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)                              # 1343-1345
# ...
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
# ...
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
# core_algos.py:1357-1361
if rollout_is_weights is not None:
    pg_losses = pg_losses * rollout_is_weights
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)

# agg_loss, core_algos.py:1168-1195
if loss_agg_mode == "token-mean":
    # ... batch_num_tokens = loss_mask.sum() only if None (raises if dp_size > 1)
    loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
    # ... global_batch_size = seq_mask.sum() only if None (raises if dp_size > 1)
    loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    if loss_agg_mode == "seq-mean-token-sum-norm":
        if loss_scale_factor is None:
            horizon = loss_mask.shape[-1]
            loss_scale_factor = horizon
        loss /= loss_scale_factor
elif loss_agg_mode == "seq-mean-token-mean":
    seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
    seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
    # ...
    loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
```

Notes:

- **Dual-clip applies only where `advantages < 0`** (1354). In that region the loss is `min(-A·c, max(-A·r, -A·clip(r)))`.
- **`seq-mean-token-sum-norm` divides by `loss_mask.shape[-1]` when `loss_scale_factor` is null** (1183-1186). The default is null (`actor.yaml:86`). The YAML comment calls this "response_length". On the engine path the mask comes from `data.select(...).to_padded_tensor()` (`verl/workers/utils/losses.py:91`). That makes `shape[-1]` the padded width of the micro-batch rather than a fixed constant. This is inferred from the code path and not tested at runtime.
- **The policy-loss registry** (`core_algos.py`) has `vanilla` (1278), `dppo_tv` (1372), `dppo_kl` (1453), `gspo` (1538), `sapo` (1614), `gpg` (1699), `clip_cov` (1735), `kl_cov` (1840), `geo_mean` (1920), `cispo` (2006) and `bypass_mode` (2351). `compute_policy_loss_reinforce` (2271) is not registered; `bypass_mode` calls it at 2457.
- **CISPO** (2006-2064): `clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)` (2038), then `pg_losses = -clipped_ratio.detach() * advantages * log_prob` (2039, 2043). It has no dual-clip, and `pg_clipfrac_lower` is a constant 0 (2057).

### A2. Advantage estimators

**Enum.** `AdvantageEstimator` (`core_algos.py:88-110`) has these values: `gae`, `grpo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `remax`, `rloo`, `opo`, `grpo_passk`, `gpg`, `rloo_vectorized`, `grpo_vectorized`, `optimal_token_baseline`, `tir_optimal_token_baseline`, `gdpo`.

**Registered estimators.** Each one has an `@register_adv_est` at these lines: gae 215, grpo 267, grpo_vectorized 334, gdpo 361, grpo_passk 471, reinforce_plus_plus_baseline 533, rloo 587, opo 639, reinforce_plus_plus 693, remax 732, gpg 768, rloo_vectorized 831, optimal_token_baseline 869, tir_optimal_token_baseline 988.

**Default estimator.** `algorithm.adv_estimator: gae` (`verl/trainer/config/ppo_trainer.yaml:71`). The dataclass also defaults to `"gae"` (`verl/trainer/config/algorithm.py:653`).

**GRPO** is `compute_grpo_outcome_advantage` (`core_algos.py:267-331`):

- **Std flag.** The kwarg is `norm_adv_by_std_in_grpo: bool = True` (273). The config default is `norm_adv_by_std_in_grpo: True` (`ppo_trainer.yaml:74`; `algorithm.py:654`). When it is False, the docstring says this matches Dr.GRPO (296).
- **Group statistics.** The group mean is `torch.mean` and the group std is `torch.std(scores_tensor)` (320-321). `torch.std`'s default is `correction=1`, so the std is **unbiased (n−1)**.
- **Singleton groups** use mean=0 and std=1 (315-317).
- **Normalization.** It computes `(score - mean) / (std + epsilon)` with `epsilon=1e-6` (272, 326). The scalar is then broadcast across the response mask (329).

**Vectorized GRPO** is `grpo_vectorized` (334-358). It uses `group_mean_std` (`verl/utils/groupwise.py:164-214`) with variance `var_num / (count - 1).clamp_min(1)` (210-211), so it is also unbiased. It is called with `eps=0.0` (352) and divides by `std_g + epsilon` (354).

### A3. Config defaults

Paths are under `verl/trainer/config/`. The YAML values and dataclass defaults match unless the table says otherwise.

| Key | Default | file:line |
|---|---|---|
| `algorithm.adv_estimator` | `gae` | `ppo_trainer.yaml:71`; `algorithm.py:653` |
| `algorithm.norm_adv_by_std_in_grpo` | `True` | `ppo_trainer.yaml:74`; `algorithm.py:654` |
| `algorithm.use_kl_in_reward` | `False` | `ppo_trainer.yaml:77`; `algorithm.py:655` |
| `algorithm.kl_penalty` | `kl` (=k1) | `ppo_trainer.yaml:80`; `algorithm.py:656` |
| `algorithm.kl_ctrl.type` / `kl_coef` | `fixed` / `0.001` | `ppo_trainer.yaml:89,92` |
| `algorithm.gamma` / `lam` | `1.0` / `1.0` | `ppo_trainer.yaml:65,68` |
| `actor.loss_agg_mode` | `token-mean` | `actor/actor.yaml:82`; `verl/workers/config/actor.py:163` |
| `actor.loss_scale_factor` | `null` | `actor/actor.yaml:86` |
| `actor.clip_ratio` | `0.2` | `actor/actor.yaml:36`; `actor.py:157` |
| `actor.clip_ratio_low` | `0.2` | `actor/actor.yaml:39`; `actor.py:158` |
| `actor.clip_ratio_high` | `0.2` | `actor/actor.yaml:42`; `actor.py:159` |
| `actor.clip_ratio_c` | `3.0` | `actor/actor.yaml:79`; `actor.py:162` |
| `actor.use_kl_loss` | `false` | `actor/actor.yaml:99`; `actor.py:170` |
| `actor.kl_loss_type` | `low_var_kl` (=k3) | `actor/actor.yaml:112`; `actor.py:175` |
| `actor.kl_loss_coef` | `0.001` | `actor/actor.yaml:109`; `actor.py:174` |
| `actor.entropy_coeff` | `0` | `actor/actor.yaml:89`; `actor.py:165` |
| `actor.ppo_epochs` | `1` | `actor/actor.yaml:115`; `actor.py:176` |
| `actor.ppo_mini_batch_size` | `256` (prompts; multiplied by `rollout.n`, see `ppo/ray_trainer.py:1260-1261`) | `actor/actor.yaml:18`; `actor.py:150` |
| `data.train_batch_size` | `1024` | `data/legacy_data.yaml:42` |
| `rollout.n` | `1` | `rollout/rollout.yaml:119` |
| `rollout.temperature` | `1.0` (val_kwargs: `0`, line 156) | `rollout/rollout.yaml:17` |
| `actor.policy_loss.loss_mode` | `"vanilla"` | `actor/actor.yaml:61`; `actor.py:93` |
| SAPO `tau_pos` / `tau_neg` | `1.0` / `1.05` | `actor/actor.yaml:46,49` |
| CISPO clip | no dedicated keys; it reuses `clip_ratio_low/high`, so the defaults are 0.2/0.2 (`core_algos.py:2024-2025`). The recipe uses `clip_ratio_low=10`, `clip_ratio_high=0.2`. | `examples/cispo_trainer/run_qwen3_8b_fsdp.sh:22-23`; `examples/cispo_trainer/README.md:16-17` |
| `rollout_correction.rollout_is` | YAML `null`; **dataclass default `"sequence"`** | `algorithm/rollout_correction.yaml:6`; `algorithm.py:175` |
| `rollout_correction.rollout_is_threshold` | `2.0` | `rollout_correction.yaml:11`; `algorithm.py:176` |
| `rollout_correction.rollout_rs` | `null` | `rollout_correction.yaml:14`; `algorithm.py:178` |
| `rollout_correction.rollout_rs_threshold` | `null` | `rollout_correction.yaml:17`; `algorithm.py:179` |
| `rollout_correction.bypass_mode` | `false` | `rollout_correction.yaml:20`; `algorithm.py:180` |
| `rollout_correction.loss_type` | `ppo_clip` | `rollout_correction.yaml:25`; `algorithm.py:181` |
| `rollout_correction.rollout_is_batch_normalize` | `false` | `rollout_correction.yaml:28`; `algorithm.py:177` |
| `AlgoConfig.rollout_correction` (dataclass) | `None` | `algorithm.py:663` |
| `actor.router_replay.mode` | `disabled` (options `disabled`, `R2`, `R3`) | `actor/actor.yaml:279`; `actor.py:66,72` |
| `rollout.enable_rollout_routing_replay` | `False` | `rollout/rollout.yaml:348` |

Notes on the table:

- **YAML path vs dataclass path.** With the stock YAML (`ppo_trainer.yaml:40` composes `rollout_correction.yaml`), rollout IS is off. A `RolloutCorrectionConfig()` built directly in Python has `rollout_is="sequence"` instead.
- **KL estimator names.** `kl_penalty` accepts `kl`/`k1`, `abs`, `mse`/`k2`, `low_var_kl`/`k3` and `full` (`core_algos.py:2166-2185`). A trailing `+` (e.g. `k3+`) makes the gradient follow k2 by straight-through (`core_algos.py:2138-2151`).

### A4. Where the token-mean denominator comes from

- **Global token count.** `batch_num_tokens` is the sum of `loss_mask` over **all DP ranks**. Each rank sums its local mask and then `all_reduce(SUM)` runs over the DP group, inside `forward_backward_batch`:
  - FSDP: `verl/workers/engine/fsdp/transformer_impl.py:612-618`
  - Megatron: `verl/workers/engine/megatron/transformer_impl.py:595-599`
  - veomni: 293-297; torchtitan: 318-322; automodel: 225-229 (each in its `transformer_impl.py`)
- **What the count covers.** `forward_backward_batch` runs once per PPO mini-batch. `verl/workers/engine_workers.py:264-301` iterates mini-batches of size `ppo_mini_batch_size * rollout.n / dp`, and each call `train_batch(mini_batch_td)` reaches `forward_backward_batch` via `verl/workers/engine/base.py:126`. So the denominator is the number of valid tokens in the **global mini-batch**. It is not per micro-batch and not the full rollout batch.
- **How it reaches `agg_loss`.** `verl/workers/utils/losses.py:65-68` copies `dp_size`, `batch_num_tokens` and `global_batch_size` into `config.global_batch_info`, which goes to `agg_loss`.
- **The seq-mean denominator** `global_batch_size` is `ppo_mini_batch_size * rollout.n` (`verl/trainer/ppo/ray_trainer.py:1260-1269`).
- **Micro-batch handling.** Each micro-batch loss is `local masked sum / global tokens * dp_size`. It is backpropagated with no further scaling (`fsdp/transformer_impl.py:632-640`), and DP gradient averaging cancels the `* dp_size`.

---

## B. OpenRLHF: group-mean advantages and the token-weighted batch normalization

`openrlhf/trainer/ppo_utils/experience_maker.py`:

```python
# 267-268  (group-mean baseline; rewards reshaped to [num_prompts, n_samples_per_prompt] at 252)
        elif args.algo.advantage.estimator in ["reinforce_baseline", "dr_grpo"]:
            rewards = rewards - rewards.mean(-1, keepdim=True)
# 315-328  (batch normalization, after per-token returns are built)
        if args.algo.advantage.estimator in ["gae", "reinforce", "reinforce_baseline"]:
            all_advantages = torch.cat([exp.advantages.flatten() for exp in experiences], dim=0).float()
            all_action_masks = torch.cat([exp.action_mask.flatten() for exp in experiences], dim=0)
            num_actions = all_action_masks.sum()

            mean = (all_advantages * all_action_masks).sum() / num_actions
            if not args.algo.advantage.no_std_norm:
                var = ((all_advantages - mean).pow(2) * all_action_masks).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd
```

**Facts from the code:**

- **Which estimators get batch-normalized.** The batch normalization runs only for `gae`, `reinforce` and `reinforce_baseline` (315). `rloo`, `group_norm` and `dr_grpo` skip it.
- **Mean and variance are token-weighted.** Both are masked over action tokens. They are computed over the **whole rollout batch**: `compute_advantages_and_returns` is called once on all experiences in `make_experience_batch` (80-96). The variance divides by `num_actions`, so it is the biased form.
- **`group_norm` uses unbiased std.** It divides by `rewards.std(-1)` (270), which is torch's default unbiased std, plus `1e-9`.
- **Per-token advantages.** `compute_reward` (`openrlhf/models/utils.py:82-110`) puts the centered scalar on the last action token and adds `-kl_coef * kl` per token. `get_cumulative_returns` with γ=1 (`experience_maker.py:398-410`; γ is forced to 1.0 for `reinforce_baseline` at 293-300) then gives `a_{i,t} = (r_i − r̄_g) − β Σ_{t'≥t} kl_{i,t'}`. With the KL term off (`--algo.kl.use_loss` makes `kl=0`, 205-213), `a_{i,t} = r_i − r̄_{g(i)}` is constant over the L_i valid tokens of response i.
- **Relevant defaults** (`openrlhf/cli/train_ppo_ray.py`): `--algo.advantage.estimator gae` (481), `--algo.kl.init_coef 0.01` (419), `--algo.kl.use_loss False` (485), `--algo.kl.estimator k1` (424), `--algo.advantage.no_std_norm False` (490). With these defaults the KL-in-reward term is active whenever a reference model exists.

**Can the batch-mean subtraction shift advantages by a length-dependent amount? Yes.** Take the KL term off. The batch mean at line 320 is

  m = Σ_i L_i (r_i − r̄_{g(i)}) / Σ_i L_i.

The group-mean step makes Σ_{i∈g} (r_i − r̄_g) = 0, but it does not make the **length-weighted** sum zero:

  Σ_{i∈g} L_i (r_i − r̄_g) = n · Cov_g(L, r)  (population covariance within the group).

So m = n Σ_g Cov_g(L, r) / Σ_i L_i. It is zero only when lengths are uncorrelated with reward within groups in aggregate, for example when all lengths are equal.

- **The shift is one scalar.** It applies to every token in the batch, but its value is set by how response length co-varies with reward. For example, if correct answers are shorter, then Cov < 0, m < 0, and every advantage moves up by |m|.
- **Group sums stop being zero.** After the shift, each group's advantages sum to −n·m (per sequence) rather than 0. A group with identical rewards (all-zero centered advantage) now gets −m on every token. That gives a nonzero gradient on prompts that carry no reward signal.
- **The scale is also length-weighted.** The std rescaling at 322-323 uses the same token weights.

---

## C. slime and SkyRL KL defaults

**slime**

- **`--kl-loss-type`** defaults to `"k1"`; the choices are `k1`, `k2`, `k3`, `low_var_kl` (`slime/utils/arguments.py:799-805`).
- **`--use-kl-loss`** is `store_true`, default `False` (`arguments.py:845-847`). **`--kl-loss-coef`** defaults to `0.0` (`arguments.py:848-853`).
- **No warning for `use_kl_loss` with k1.** The only check that combines `use_kl_loss` and KL settings is at `arguments.py:1602-1610`, and it only validates `ref_load`. A grep of `slime/` for `kl_loss_type` and for KL-related warnings found nothing else.
- **Estimator implementation** is in `slime/utils/ppo_utils.py:28-51`: k1 = `log_ratio`, k2 = `log_ratio**2/2`, k3/low_var_kl = `exp(-lr) - 1 + lr`. The clamp to ±10 applies only to `low_var_kl`. An optional `importance_ratio` multiplier is applied at 44-45.
- **For comparison, OpenRLHF does print a recommendation.** When `algo.kl.use_loss` is set and the estimator is not k2/k3, it calls `print(...)` (`openrlhf/cli/train_ppo_ray.py:675-680`). It does not use a logger warning.

**SkyRL** (`skyrl/train/config/config.py`, `AlgorithmConfig`)

- **`use_kl_loss`** defaults to `True` (356). **`kl_estimator_type`** defaults to `"k3"` (352); options are `k1`, `k2`, `k3`, `abs` (353).
- **Other defaults:** `kl_loss_coef: float = 0.001` (358), `use_kl_in_reward = False` (354), `advantage_estimator = "grpo"` (347).
- **Mutual exclusion** between `use_kl_in_reward` and `use_kl_loss` is asserted at `skyrl/train/utils/utils.py:248-250`.
- **Function default.** The KL function's own default is also `kl_estimator_type: str = "k3"` (`skyrl/backends/skyrl_train/utils/ppo_utils.py:92`).
- **No YAML.** SkyRL ships no YAML defaults for these keys; the dataclass is the source.

## D. TRL

Skipped: no local TRL checkout.
