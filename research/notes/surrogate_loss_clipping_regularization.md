# Surrogate loss 轴：trust-region/clipping、importance-ratio 粒度、loss aggregation、KL 与 entropy 正则

> 写作约定：术语保留英文。记号：token ratio `r_t = π_θ(y_t|·)/π_old(y_t|·)`，序列 `y_i` 长度 `|y_i|`，组大小 `G`，优势 `Â_i`（结果奖励时同一序列所有 token 共享）。"sg" = stop-gradient。证据等级：A = 多个独立受控消融；B = 一个受控消融或清晰的数学机制；C = 只有原论文；D = 证据相互矛盾。
> 阅读深度标注：[全文] = 读了 arXiv HTML 的方法+实验；[摘要] = 只读了 abstract；[代码] = 读了本地框架源码。
> 本地框架代码路径前缀：`/Users/qiwei/Desktop/Projects/RL_infra/_external_repos/`。

---

## Q1. 每个方法对目标/梯度估计器的精确改动是什么（哪些 token 梯度为零、步长如何被限制、引入什么偏差）？

### Takeaway
几乎所有 2025–2026 的"新 loss"都可以写成同一模板 `∇J = Σ_i Σ_t w_{i,t} · m_{i,t} · Â_i · ∇log π_θ(y_{i,t})`，差别只在三个旋钮：(1) 权重 `w`（token ratio / 序列几何平均 ratio / 截断且 stop-grad 的 ratio / 平滑门控），(2) 掩码 `m`（哪些 token 梯度被硬置零），(3) 归一化（`1/|y_i|`、`1/Σ|y_i|`、常数）。一个关键但常被忽略的事实：**在严格 on-policy（每批 rollout 只做一次梯度更新、且训练-推理概率一致）时 r_t ≡ 1，PPO-clip、DAPO、GSPO、GMPO、CISPO、SAPO 的梯度全部退化为同一个 REINFORCE 梯度，差异只剩归一化方式**；它们只在多 minibatch 复用（policy staleness）和训练-推理 mismatch 下才有区别。

### Cited Findings

**通用框架与机制**
- [代码] verl 的 vanilla PPO loss：`pg_losses = max(-A·r, -A·clip(r,1-ε_low,1+ε_high))`，对 `A<0` 再取 `min(·, -A·clip_ratio_c)`（dual-clip，默认 `clip_ratio_c=3.0`，断言 >1）；log-ratio 先 clamp 到 [-20,20] — `verl/verl/trainer/ppo/core_algos.py` 第 1278–1370 行（本地 checkout）。
- Qwen 团队把 token-level 目标解释为序列级奖励的一阶近似，只有在 training–inference discrepancy 和 policy staleness 都很小时才成立；IS correction、clipping、Routing Replay 都被解释为维持这一近似有效的手段 [全文] — [Zheng et al. 2512.01374](https://arxiv.org/abs/2512.01374)。
- 同一论文：长度归一化（`1/|y|`）"使一阶近似失效、得到有偏的 token-level 目标"；on-policy 网格里加长度归一化仍稳定但性能次优 — [2512.01374 HTML](https://arxiv.org/html/2512.01374)。

**PPO clip（Schulman 2017）**
- (a) `L = min(r_t Â, clip(r_t,1-ε,1+ε) Â)`。梯度：`Â>0 且 r_t>1+ε` 或 `Â<0 且 r_t<1-ε` 时为 0，其余为 `Â·r_t·∇log π`。注意它是单侧的：`Â<0 且 r_t ≫ 1` 时梯度 ∝ r_t 无界（这就是 dual-clip 要解决的）。 [代码] verl 同上。
- (b) Engstrom et al. 指出 PPO 相对 TRPO 的大部分收益来自 "code-level optimizations"，而不是 clipped objective 本身 [摘要] — [2005.12729](https://arxiv.org/abs/2005.12729)。（摘要未讨论 clipping 是否真的维持 trust region；该结论在全文中，我没读到全文。）
- (b) 偏差：clipping 本身对 entropy 有系统性偏置——clip-low 增加 entropy，clip-high 降低 entropy；标准参数下 clip-high 占主导，因此即使是**纯随机奖励**也会让 entropy 下降 [摘要] — [Park et al. 2509.26114](https://arxiv.org/abs/2509.26114)。
- (b) 偏差的可观测后果：Qwen2.5-Math-7B 在随机奖励下 MATH-500 提升 21.4 点（真实奖励 29.1 点）；作者归因于 GRPO clipping 偏向"已高概率"的 token（低概率 token 的有效上界更紧）；三种方式关掉 clipping 后随机奖励不再带来稳定提升；Llama3、OLMo2 上虚假奖励基本无效 [全文前半] — [Shao et al. 2506.10947](https://arxiv.org/html/2506.10947)。
- (c) 强 off-policy 场景下 PPO(ε=0.2) 在 Llama-3-8B/GSM8K 上"几乎没有进展"，因为大部分数据点落在 [1-ε,1+ε] 外 [全文] — [TOPR 2503.14286](https://arxiv.org/html/2503.14286)。
- (c) DeepSeek-R1 第一阶段 RL 把 GRPO clip ratio ε 设为 **10**（几乎等于不 clip），论文只说"太低会截断大量 token 的梯度、太高会不稳定"，没有给出消融 [全文] — [DeepSeek-R1 2501.12948](https://arxiv.org/html/2501.12948)。

**Dual-clip PPO（Ye et al. 2019）**
- (a) 对 `Â<0`，目标下界为 `c·Â`（c>1，常用 3）：`r_t > c` 时梯度为 0。[摘要] 原论文页面只提到名字（[1912.09729](https://arxiv.org/abs/1912.09729)，MOBA 游戏）；公式取自 [代码] verl `core_algos.py`。
- (b) 机制：限制 `Â<0` 且 `r_t` 很大（新策略大幅提高了一个坏 token 的概率）时的无界梯度——纯方差控制，偏向 behavior policy。
- (c)(d) 未找到任何 LLM 场景下 dual-clip 开/关的受控消融。框架默认：verl `clip_ratio_c: 3.0`（`verl/verl/trainer/config/actor/actor.yaml:79`），SkyRL `clip_ratio_c: 3.0`、`policy_loss_type` 可选 `"dual_clip"`（`SkyRL/skyrl/train/config/config.py`）。DCPO 设 `r_max=10`，"受 dual clipping 启发" [全文] — [DCPO 2509.02333](https://arxiv.org/html/2509.02333v2)。
- (e) Ceiling: 无证据；Efficiency: 无证据；Stability: 机制合理但未验证；Side effects: 未知。**等级 C（机制 B）**。

**DAPO clip-higher（非对称 ε）**
- (a) `clip(r, 1-ε_low, 1+ε_high)`，ε_low=0.2、ε_high=0.28；唯一改动是 `Â>0` 时上界放宽，使低概率 token 可以涨得更多 [全文] — [DAPO 2503.14476](https://arxiv.org/html/2503.14476)。
- (b) 机制：对 `Â>0` 的低概率 token，PPO 上界 `(1+ε)·p_old` 对 p_old 小的 token 是极小的绝对增量；放宽上界 = 去掉一部分 "clip-high 降 entropy" 偏置（与 [2509.26114](https://arxiv.org/abs/2509.26114) 的机制一致）。
- (c) 原始证据：Qwen2.5-32B base，AIME24 avg@32，逐项累加消融：+Overlong Filtering 30→36，+Clip-Higher 36→38，单次运行，未报告 seeds — [DAPO](https://arxiv.org/html/2503.14476)。
- (d) 独立证据（详见 Q3）：Tricks-or-Traps（Qwen3-4B/8B，base 与 aligned）— base 上影响很小甚至有害、base clip rate 约 0.003；aligned 上明显减缓 entropy collapse 并提升 [全文] [2508.08221](https://arxiv.org/html/2508.08221)；Entropy Mechanism（Qwen2.5-7B/32B）— clip-higher 38.6→38.8（7B）、45.8→47.2（32B），且"后期变得不稳定、性能下降" [全文] [2505.22617](https://arxiv.org/html/2505.22617)；ScaleRL（8B dense）— ε_max 改变 asymptote A，先升后降 [全文] [2510.13786](https://arxiv.org/html/2510.13786)。
- (e) 见 Q3 结论。**等级 B（有多项独立消融但结论随 base model 变化 → 对"普遍有效"是 D）**。

**DAPO token-level loss aggregation**
- (a) 从 `(1/G)Σ_i (1/|y_i|) Σ_t ℓ_{i,t}` 改为 `(1/Σ_i|y_i|) Σ_i Σ_t ℓ_{i,t}`（每个 prompt 组内按 token 总数归一）。DAPO 消融 +Token-level Loss 41→42 [全文] — [DAPO](https://arxiv.org/html/2503.14476)。
- 注意实现差异：DAPO 公式里归一化是**每个 prompt 组内**的 token 总数（期望在 q 外面），ScaleRL 把 DAPO/MiniMax 归为 "prompt average" [全文] [2510.13786](https://arxiv.org/html/2510.13786)；而 verl 的 `token-mean` 是**整个 global batch** 的 token 总数（`masked_sum(loss)/batch_num_tokens*dp_size`）[代码] `verl/verl/trainer/ppo/core_algos.py:1168-1173`。两者在 prompt 间长度差异大时给出不同权重。
- 其他内容见 Q1 "Loss aggregation 变体"。

**Overlong filtering（只谈 loss 侧）**
- (a) 被截断（未输出 EOS）的样本 loss mask 掉，不贡献梯度 — [DAPO](https://arxiv.org/html/2503.14476)。
- (b) 机制/偏差：条件化在"完成了的样本"上，相当于对截断样本既不奖励也不惩罚；去掉了"截断=错误"这一噪声负信号，但也去掉了对"无法终止的重复"的惩罚。
- (c) DAPO：30→36（AIME24，Qwen2.5-32B，最大长度 16k+4k cache）— [DAPO](https://arxiv.org/html/2503.14476)。
- (d) Tricks-or-Traps：8k 最大长度时"收益显著"，20k 时"收益大幅减小"；aligned 模型训练中"重复且无法终止"样本比例上升，加 overlong mask 后下降；最终 Lite PPO **不使用** overlong filtering [全文] — [2508.08221](https://arxiv.org/html/2508.08221)。AReaL 提供 `mask_no_eos_with_zero` 开关（默认 False）[代码] `AReaL/areal/api/cli_args.py`。
- (e) Ceiling: 未证明；Efficiency: 短上下文下有收益；Stability: 可能减少长度噪声；Side effects: 与长度预算强交互。**等级 B（一个独立消融显示效果依赖最大长度）**。

**GSPO（Qwen, 2507.18071）**
- (a) `s_i(θ) = (π_θ(y_i|x)/π_old(y_i|x))^{1/|y_i|} = exp((1/|y_i|)Σ_t log r_{i,t})`；目标 `(1/G)Σ_i min(s_i Â_i, clip(s_i,1-ε,1+ε) Â_i)`；梯度 `s_i·Â_i·(1/|y_i|)Σ_t ∇log π_θ(y_{i,t})`——**一个序列内所有 token 同权**，clip 一旦触发整条序列梯度为 0。clip 范围 3e-4/4e-4（GRPO 对照 0.2/0.27）[全文] — [GSPO](https://arxiv.org/html/2507.18071)。
- [代码] verl 实现：`log_seq_ratio = log_prob - sg(log_prob) + sg(mean_t(log r))`，强制 `seq-mean-token-mean` 聚合（`core_algos.py:1538-1612`）；OpenRLHF `--actor.policy_loss_type gspo`；SkyRL `policy_loss_type: "gspo"`。
- (b) 机制：用几何平均把 per-token ratio 噪声平均掉，降低方差；代价是 (i) 引入 `1/|y_i|` 长度归一化（Dr. GRPO 和 2512.01374 都认为这是有偏的），(ii) 整条序列一起 clip 浪费样本。作者称 GSPO clip 掉的 token 比 GRPO 多两个数量级却训练更高效 — [GSPO](https://arxiv.org/html/2507.18071)。
- (c) 原始证据：Qwen3-30B-A3B-Base cold-start（MoE）；每批 rollout 拆 4 个 minibatch；AIME24/LCB/CodeForces；文本中没有数值，只有曲线；未报告 seeds；称 MoE 每次更新约 10% 被激活专家变化，GRPO 需要 Routing Replay 才收敛 — [GSPO](https://arxiv.org/html/2507.18071)。
- (d) 独立证据见 Q4。**等级 D（对"普遍优于 token-level"）**。

**GMPO（geometric-mean, 2507.20673）**
- (a) `J = (1/G)Σ_i sgn(Â_i)·(Π_t |clip(r_{i,t})·Â_i|)^{1/|y_i|}`，clip 在 log 空间逐 token 做，范围 (e^-0.4, e^0.4)；梯度中每个 token 的权重是整条序列 ratio 的几何平均 — [GMPO](https://arxiv.org/html/2507.20673)。
- (b) 推断：这在代数上就是"带 token-level log-space clip 的 GSPO"——几何平均 ratio × `(1/|y_i|)Σ_t ∇log π`，区别只是 clip 粒度（token 而非序列）和 ε 尺度。
- (c) DeepSeek-R1-Distill-Qwen-7B 63.4 vs GRPO 59.3（+4.1）；Qwen2.5-Math-7B 52.7 vs 51.2（+1.5）；Qwen3-32B MATH500 96.7 vs 94.6；未报告 seeds 与训练步数；GRPO 基线"跟随 Dr. GRPO 设置"，未说明是否调参 [全文] — [GMPO](https://arxiv.org/html/2507.20673)。verl 有 `geo_mean` loss [代码]。
- (d) 未找到独立复现。**等级 C**。

**CISPO（MiniMax-M1, 2506.13585）**
- (a) `J = (1/Σ_i|y_i|) Σ_i Σ_t sg(clip(r_{i,t}, 1-ε_low, 1+ε_high))·Â_{i,t}·log π_θ(y_{i,t})`；不加下界（ε_low 设很大），只调 ε_high；无 KL；**没有任何 token 的梯度被置零**，权重被截断后 stop-grad [全文] — [MiniMax-M1](https://arxiv.org/html/2506.13585)。
- (b) 机制：这是截断 IS 的 off-policy REINFORCE（类似 V-trace/TIS），没有 trust region——步长只由 IS 权重上限和学习率限制；偏差：截断导致梯度偏向 behavior policy；动机是 "However/Recheck/Wait" 等低概率反思 token 在第一次 on-policy 更新后 r 变大、被 PPO clip 掉，之后的 16 轮 off-policy 更新不再贡献梯度 — [MiniMax-M1](https://arxiv.org/html/2506.13585)。
- 推断：CISPO "不丢 token" 同时也重新打开了 PPO 刻意关闭的区域 `Â<0 且 r<1-ε`（一个坏 token 已被压低还继续压）。ARLArena 报告的 CISPO/SAPO 崩溃正是来自"负优势、低 IS ratio 的序列"（见 (d)），与此一致。
- 实现陷阱 [代码]：verl 的 `cispo` 用 `clip_ratio_low/high`（默认 0.2/0.2），即默认把权重截断在 [0.8,1.2]，与原论文"无下界、ε_high 大"不同；SkyRL 默认 `cispo_eps_clip_low=0.0, cispo_eps_clip_high=5.0`（`SkyRL/skyrl/train/config/config.py`）。
- (c) Qwen2.5-32B-base zero-RL、DAPO 数据、AIME24：同步数下 CISPO > GRPO、DAPO，"以 50% 步数达到 DAPO 的性能"；文本无具体数值，未报告 seeds — [MiniMax-M1](https://arxiv.org/html/2506.13585)。
- (d) 独立：ScaleRL（8B dense）在 LOO 中 CISPO 效率 B=2.01 vs DAPO 1.77，A 相近；对 ε_max∈{4,5,8} 不敏感 — [2510.13786](https://arxiv.org/html/2510.13786)。反例：CE-GPPO（R1-Distill-Qwen-1.5B）五项平均 CISPO 47.6 vs GSPO 54.3 vs base 45.2 — [2509.20712](https://arxiv.org/html/2509.20712)；ARLArena（Qwen3-4B，agentic）CISPO 平均 34.03 vs GRPO 46.16，约第 130 步崩溃，加 sequence masking 后 ALFWorld 54.42→78.88 — [2602.21534](https://arxiv.org/html/2602.21534)。
- **等级 D（对普遍优越性）；B（对"大 batch 推理 RLVR 中提升效率且超参鲁棒"，仅 ScaleRL 一项受控研究）**。

**SAPO（soft gating, Qwen, 2511.20347）**
- (a) `L = -f(r_t)·Â`，`f(r) = (4/τ)·σ(τ(r-1))`，`Â>0` 用 τ_pos，否则 τ_neg；对 ∇log π 的有效权重为 `r·sech²(τ(r-1)/2)`（r=1 时为 1，远离 1 指数衰减）；受控实验 τ_pos=1.0、τ_neg=1.05，τ_neg=0.95 最不稳定 [全文] — [SAPO](https://arxiv.org/html/2511.20347)。[代码] verl `sapo`（强制 seq-mean-token-mean）、AReaL `use_sapo_loss`。
- (b) 机制：把 PPO 的硬指示函数（1/0）换成平滑核；在 r≈1 且序列内 log-ratio 方差小的假设下近似 GSPO 的序列门控，误差界 `(τ²/4)·Var_i`，作者称 Var_i 通常 < 0.02 — [SAPO](https://arxiv.org/html/2511.20347)。
- (c) Qwen3-30B-A3B cold-start（MoE）+ Qwen3-VL-30B-A3B；对照 GSPO、GRPO-R2；文本无具体数值、无 seeds、无训练步数；dense 模型（Qwen3-4B）只用于验证假设，没有 RL 训练结果 — [SAPO](https://arxiv.org/html/2511.20347)。
- (d) ARLArena（Qwen3-4B，agentic）：SAPO 平均 32.22，最差，ALFWorld 25.16；加 sequence masking 后 76.92 — [2602.21534](https://arxiv.org/html/2602.21534)。**等级 C（推理）/ D（跨任务）**。

**TOPR（truncated IS for negatives, 2503.14286）**
- (a) 正样本（R≥0）权重 1（无 IS），负样本权重 `sg(clip(π(y)/μ(y), 0, 1))`（**序列级** ratio），`ℓ = sg(α)·r·log π(y|x)` [全文] — [TOPR](https://arxiv.org/html/2503.14286)。
- (b) 机制：负样本的 IS 权重只向下截断，防止"把已经很低的坏序列继续压"时出现爆炸；正样本无 IS 则等价于 SFT-on-positives（偏差：不校正 off-policy）。称无需 KL 即稳定。
- (c) Llama-3-8B-Instruct，GSM8K/MATH，reward ±1，无 baseline，max 512 tokens，bootstrap CI，无 seed 数；朴素 REINFORCE 先好后崩，PPO(ε=0.2) 几乎不动；最佳"有效正样本比例"约 10–20% — [TOPR](https://arxiv.org/html/2503.14286)。
- (d) 未找到独立复现。短输出（512 tokens）设置，远离长 CoT。**等级 C**。

**ASPO（asymmetric IS, 2510.06062）**
- (a) `Â>0` 的 token 用"倒数"权重 `r̂ = π_old·π_θ / sg(π_θ²)`（数值 = 1/r，梯度仍经过 π_θ），即正样本梯度变为 `(π_old/π_θ)·Â·∇log π_θ`；`Â<0` 不变；外加 CISPO 式 soft dual-clip [全文 v1] — [ASPO](https://arxiv.org/html/2510.06062v1)。
- (b) 论点：PPO-clip 中已被提高概率的正 token 获得更大权重（自我强化 → entropy collapse、重复），倒数权重反过来照顾"落后"的 token。
- (c) DeepSeek-R1-Distill-Qwen-1.5B；数学六项平均 59.3 vs DAPO 53.5；LCB 31.0 vs 26.8；8×H800、每次约 7 天；无 seeds、无方差；论文的"相对 base +12.5%"实为绝对点数，代码 +17.0% 与表格（14.0 点）对不上 — [ASPO](https://arxiv.org/html/2510.06062v1)。**等级 C**。

**DCPO（dynamic clipping, 2509.02333）**
- (a) 由 `|(p/q − 1)·p| ≤ ε` 推出依赖旧概率 q 的 ratio 界：`0.5 + ½√max(1−4ε_low/q, 0) ≤ r ≤ 0.5 + ½√(1+4ε_high/q)`；ε_low=0.16、ε_high=0.2，r_max=10；聚合用 "Only-Token-Mean"（每条序列内 token 平均）；外加跨步累积的平滑优势标准化 [全文] — [DCPO](https://arxiv.org/html/2509.02333v2)。
- (b) 机制：低概率 token（q 小）上界变宽（比 clip-higher 更系统），高概率 token 上界收紧。
- (c) Qwen2.5-Math-7B AIME24 Avg@32 38.8 vs GRPO 32.1 / DAPO 31.6 / GSPO 34.9；Qwen2.5-14B AIME25 Avg@32 19.0 vs 10.5/15.3/9.9；最大长度 3072，400 步，无 seeds，基线用各自公开默认值、未扫参 — [DCPO](https://arxiv.org/html/2509.02333v2)。**等级 C（且主要证据在 Qwen2.5-Math 上）**。

**CE-GPPO（gradient-preserving clipping, 2509.20712）**
- (a) clip 区外的 token 不再零梯度，而是给常数权重：`Â<0, r<1-ε` 时权重 `β₁(1-ε)`；`Â>0, r>1+ε` 时 `β₂(1+ε)`；β₁=β₂=0 退化为 PPO [全文] — [CE-GPPO](https://arxiv.org/html/2509.20712)。
- (b) 四象限 entropy 分析：PA&LP（正优势低概率）被 clip → entropy collapse；NA&LP 被 clip → entropy explosion；β 调节二者 — 同上。
- (c) R1-Distill-Qwen-1.5B/7B，16k，ε=0.2，无 KL，≤1000 步；7B AIME24 66.0 vs DAPO 59.7；1.5B 平均 54.9 vs GSPO 54.3 vs CISPO 47.6；无 seeds。已被 ACL 2026 接收（据搜索结果）。**等级 C**。

**DPPO（divergence-based trust region, 2602.04879）**
- (a) [摘要+代码] 用直接的策略散度估计（TV 或 KL，配 Binary/Top-K 近似）替代 ratio clip；verl `dppo_tv` 实现为：`Â>0` 时要求 `p−p_old ≤ ε_high`，`Â<0` 时 `p−p_old ≥ −ε_low`，否则 mask；权重 `sg(min(r, 20))·Â·log π` — [2602.04879](https://arxiv.org/abs/2602.04879)，`verl/verl/trainer/ppo/core_algos.py:1372-1452`。
- (b) 论点：ratio 是单样本、噪声大的散度估计——对低概率 token 过度约束、对高概率 token 约束不足（与 DCPO、clip-higher 同一观察的另一种表述）。
- (c)(d) 只读了摘要，模型和数值未知。**等级 C（未核实）**。

**DGPO（2603.14389）、BAPO**
- DGPO：认为 soft-clipping 方法依赖的 ∇log π 在概率趋零时权重发散，主张用 ∇π 作为原语并做双侧解耦衰减；R1-Distill-Qwen 1.5B/7B/14B [摘要] — [2603.14389](https://arxiv.org/abs/2603.14389)。
- BAPO：自适应调整 clip 界以平衡正负样本贡献、防止负样本主导梯度（只见搜索摘要，未读原文，arXiv 号未核实）。**均为等级 C 或更低**。

**Clip-Cov / KL-Cov（Entropy Mechanism, 2505.22617）**
- (a) 计算每 token 的 `Cov = (log π − mean)(Â − mean)`；Clip-Cov：在协方差落于 [ω_low=1, ω_high=5] 的 token 中随机选比例 r=2e-4 的 token 并 detach 梯度；KL-Cov：对协方差 top-k（k=2e-3 for 7B，2e-4 for 32B）token 加 `β=1` 的 KL（对 π_old）惩罚 [全文] — [2505.22617](https://arxiv.org/html/2505.22617)。
- (b) 机制：softmax 策略下一步 entropy 变化 ≈ `−η·Cov(log π, π·Â)`（vanilla PG）或 `−η·Cov(log π, Â)`（natural PG）；高概率且高优势 token 驱动 entropy 下降 — 同上。
- (c) 平均分 Qwen2.5-7B：GRPO 38.6 / clip-higher 38.8 / Clip-Cov 40.4 / KL-Cov 40.6；32B：45.8 / 47.2 / 50.3 / 52.2；只改动 1e-4~1e-3 比例的 token；无 seeds — 同上。框架采用：verl `clip_cov`、`kl_cov`，SkyRL `clip_cov`/`kl_cov`（`KLCovConfig`）[代码]。
- (d) 未找到独立受控复现。**等级 C**。

**High-entropy minority tokens（"Beyond the 80/20 rule", 2506.01939）**
- (a) 只对 batch 内 entropy 前 20% 的 token（"forking tokens"）计算 policy gradient，其余 mask，按选中 token 数归一 [全文] — [2506.01939](https://arxiv.org/html/2506.01939)。
- (c) 在 DAPO 之上：Qwen3-32B AIME24 55.83→63.54、AIME25 45.63→56.67；14B +5.21/+4.79；**8B 六项平均仅 +0.53**（作者称"不受影响"）；只训练后 80% 低 entropy token 严重退化；比例 100% 显著下降（14B/32B）；无 seeds；无 KL/entropy loss — 同上。
- (d) 未找到独立复现。**等级 C**。

**NSR / PSR / W-REINFORCE（2506.01347）**
- (a) `L = L_PSR + L_NSR`；W-REINFORCE = `λ·L_PSR + L_NSR`，λ=0.1 [全文] — [2506.01347](https://arxiv.org/html/2506.01347)。
- (b) NSR 梯度：被采样的错误 token logit 下降（幅度 ∝ π(1−π)），未被采样 token 的 logit 按当前概率比例上升 → "按先验重新分配质量"的软重排序 — 同上。
- (c) Qwen2.5-Math-7B（MATH 训练，4k 上下文）：MATH Pass@1 PSR 74.1 / NSR 75.7 / GRPO 76.3 / W-REINFORCE 76.6；Pass@256 PSR 91.2（低于 base 96.9）/ NSR 96.9；AIME25 Pass@256 W-REINFORCE 56.7 vs GRPO 50.0（AIME 30 题，单题 = 3.3 点）；Llama-3.1-8B-Instruct 上所有 RL 都降低大 k 性能，NSR 降得最少；长时间 NSR 训练会退化；无 seeds、无步数 — 同上。
- (d) TOPR 也强调负样本处理（截断 IS）而非丢弃 [TOPR](https://arxiv.org/html/2503.14286)；Kimi k1.5 称相对 ReST（只拟合正样本）使用负梯度有更好的样本复杂度 [全文] — [k1.5 2501.12599](https://arxiv.org/html/2501.12599)；RL's Razor 反而发现"on-policy 数据比负梯度更重要" — [2509.04259](https://arxiv.org/html/2509.04259)。**等级 C（主效应集中在 Qwen2.5-Math，且 AIME 样本小）**。

**Loss aggregation 变体（精确权重）**
- token-mean（batch 级，verl 默认）：每 token 权重 `1/T_batch`；长序列总权重 ∝ 长度 [代码] `core_algos.py:1168`。
- prompt-average（DAPO 公式 / ScaleRL 选择）：组内 `1/Σ_i|y_i|`，再对 prompt 平均 — [ScaleRL](https://arxiv.org/html/2510.13786)、[DAPO](https://arxiv.org/html/2503.14476)；Magistral 同样"除以组内生成总长度" [全文] — [2506.10910](https://arxiv.org/html/2506.10910)。
- seq-mean-token-mean（原 GRPO、GSPO/SAPO 强制）：每 token 权重 `1/(G·|y_i|)`；Dr. GRPO 指出：正优势时短的正确答案更新更大，负优势时长的错误答案惩罚更小 → 错误答案越来越长 [全文] — [Dr. GRPO 2503.20783](https://arxiv.org/html/2503.20783)。
- seq-mean-token-sum：每 token 权重 `1/G`，梯度方差随长度增长 [代码] verl。
- Dr. GRPO 常数归一化（verl `seq-mean-token-sum-norm`，默认 `loss_scale_factor = response_length` 即 max tokens；SkyRL `seq_mean_token_sum_norm`，须显式设 `max_seq_len`）：`(1/G)Σ_iΣ_t ℓ / L_max`，对长度无偏（与 REINFORCE 相同，常数并入学习率）— [Dr. GRPO](https://arxiv.org/html/2503.20783)，[代码] `core_algos.py:1174-1186`。
- Dr. GRPO 还去掉 std 归一化（"问题难度偏差"）；它报告 OpenRLHF 按每条 response 归一化、trl/verl 按 batch 但仍依赖长度 — [Dr. GRPO](https://arxiv.org/html/2503.20783)。

**KL 估计器与位置（梯度层面）**
- 记 `δ = π_ref/π_θ`。k1 = `log(π/π_ref)`，k2 = `½(log π/π_ref)²`，k3 = `δ − 1 − log δ`。
- k1 当 loss：`E[∇log π] = 0`，零均值噪声，不起正则作用；k3 当 loss：期望梯度 = `∇KL(π_ref‖π)`（反向 KL 的梯度，不是 `KL(π‖π_ref)`），作者称之为 bug；k2 当 loss：期望梯度 = `∇KL(π‖π_ref)`（无偏）。原因：真实梯度 = pathwise 项 + score-function 项 `k̂·∇log π`，直接对估计量求导丢掉了后者 [全文] — [Tang & Munos 2506.09477](https://arxiv.org/html/2506.09477)。
- 序列级陷阱：逐 token 的 KL loss 只给出序列 KL 的偏导，忽略 y_t 对后续 token KL 的影响；正确的序列级估计是 `Σ_t (Σ_{s≥t} log ρ_s)·∇log π(y_t)`（cumulative）或 leave-one-out 版本 — 同上。点名 TRL、Open Instruct、GRPO 伪代码；**没有点名 verl/OpenRLHF** — 同上。
- k1-in-reward 梯度 = `E[log(π/π_ref)·∇log π]`；on-policy 下 k2-as-loss 与 k1-in-reward 梯度等价；k3-as-loss 梯度 = `E[(1 − π_ref/π)·∇log π]`，是 `−log δ ≈ 1 − δ` 的一阶有偏近似；off-policy 时 "as loss" 形式需要 IS，常被省略 → 有偏 [全文前 100k 字符] — [2510.01555](https://arxiv.org/html/2510.01555v1)。
- DeepSeek-V3.2 的修正：`D_KL = (π_θ/π_old)·[π_ref/π_θ − log(π_ref/π_θ) − 1]`，即给 k3 乘 IS ratio 以得到无偏梯度；理由：π_θ ≪ π_ref 的 token 上原 k3 梯度权重无界 [全文] — [DeepSeek-V3.2 2512.02556](https://arxiv.org/html/2512.02556)。
- Kimi k1.5/K2 的正则不是对 SFT 参考的 KL，而是对上一迭代 π_old 的平方项：`L = E[(1/K)Σ_i (r_i − r̄ − τ·log(π_θ(y_i)/π_old(y_i)))²]`，梯度 = 带均值 baseline 的 PG + `(τ/2)∇(log π_θ/π_old)²`（k2 形式的近端项），每次迭代重置优化器 [全文] — [k1.5](https://arxiv.org/html/2501.12599)、[K2 2507.20534](https://arxiv.org/html/2507.20534)。

### Inferences
- 推断（由公式直接得出）：GSPO、GMPO、SAPO 的序列级/平滑权重在 r≡1 时与 seq-mean-token-mean REINFORCE 完全相同；CISPO 在 r≡1 时与 token-mean REINFORCE 相同。因此**任何在 1 个 minibatch/rollout 且无训练-推理 mismatch 条件下报告的"新 loss 收益"都应怀疑是归一化差异或噪声**；真正的比较对象是 staleness（minibatch 数）与 mismatch（FP8 推理、MoE routing 漂移）的大小。
- 推断：2025–2026 的 clip 变体（clip-higher、DCPO、DPPO、CE-GPPO、ASPO、DGPO）都在修同一个问题——ratio clip 对低概率 token 的绝对步长过小、对高概率 token 过大——它们只是选择了不同的参数化。目前没有一项研究把它们放在同一个受控、多 seed 的框架下比较。
- 推断：CISPO/SAPO 一类"不丢 token"的方法放开了 `Â<0, r<1−ε` 区域，这正是 ARLArena、DeepSeek-V3.2 off-policy sequence masking 所针对的危险区域（负优势 + 高度 off-policy）。这说明"保留所有 token 梯度"与"对负样本做 off-policy 掩码"是一对需要搭配的设计。

### Gaps
- 未读到 Engstrom et al. 全文，无法确认"PPO 的 clip 实际上并不维持 trust region"的具体数值证据；Huang et al. "N+ Implementation Details of RLHF with PPO" 未读取。
- Dual-clip 在 LLM 中开/关的受控消融：未找到。
- BAPO、DGPO、DPPO 只读了摘要或搜索摘要；DPPO 的模型和数值未知。
- 没有找到在同一框架、多 seed 下比较 clip-higher / DCPO / CE-GPPO / ASPO / DPPO 的独立研究。

---

## Q2. 哪些选择改变渐近上限（ceiling），哪些只影响效率或稳定性？（核实 ScaleRL 关于 DAPO vs GSPO vs CISPO 的结论）

### Takeaway
ScaleRL 的"loss 类型影响 asymptote"只在**孤立前向比较**中成立（GSPO、CISPO 的 A 明显高于 DAPO）；在把其他修正（FP32 LM head 等）都加上后的 leave-one-out 中，**换回 DAPO loss 的 A 落在 ±0.02 噪声内，只有效率 B 下降（2.01→1.77）**。同一研究中对 A 影响最大、最清楚的是一个数值精度修复（FP32 LM head：A 0.52→0.61），以及 DAPO 的 ε_max 和 batch size。因此"loss 类型决定上限"这一标题式说法需要大幅打折。

### Cited Findings
- ScaleRL 拟合 `R(C) = R0 + (A − R0)/(1 + (C_mid/C)^B)`，A = 渐近 pass rate，B = 效率；三次独立 ScaleRL 运行给出 A 的误差 ±0.02（附录称 ±0.015）[全文] — [ScaleRL 2510.13786](https://arxiv.org/html/2510.13786)。
- 前向比较（图 5a）："GSPO 和 CISPO 都大幅超过 DAPO，以较大幅度提高渐近 pass-rate A"；CISPO 后期"略好于"GSPO；正文无各自的 A 数值 — 同上。
- LOO（16k GPU-hours/run）：换回 DAPO 的 A 相近，B：CISPO 2.01 vs DAPO 1.77；所有 LOO 变体 A 都在 ±0.02 内，主要差在 B 和 C_mid — 同上。
- CISPO 对 ε_max∈{4,5,8} 无显著差异；GSPO 原论文的 1e-4 尺度对 8B 不是最优，1e-3 尺度更好，上 clip 5e-3 略好于 4e-3；GSPO 多次中途发散，8B 可从稳定 checkpoint 重启恢复，Scout（17B×16 MoE）不能，作者未找到实现 bug — 同上（附录 A.17）。
- DAPO 的 ε_max 改变 A：先升后降，作者称它"控制渐近误差本身"，不只是速度 — 同上（A.17.1）。
- 聚合方式：prompt-average 在孤立比较中 A 最高（无数值）；LOO 中 sample-average 变体与其他变体 A 都在 ±0.02 内 — 同上（A.9、Fig.14a/15）。
- FP32 LM head：A 从 0.52 升到 0.61；不稳定点：GRPO 约 6k GPU-hours 后不稳定，DAPO(ε_max=0.26) 约 5k，FP32 修复约 6k，CISPO 约 7k — 同上。
- Entropy 在所有实验中单调下降，且不预测下游表现：batch 768 与 2048 的每步 entropy 几乎相同，但 2048 显著更好 — 同上（A.12）。
- 设置：8B dense（主消融）+ Llama-4 Scout；GB200，每 run 80 GPU；单个设计消融 3.5–4k GPU-hours；总计 >400k GPU-hours；**无 KL 项** — 同上。
- 独立旁证：Qwen 2512.01374 发现训练稳定后，不同 cold-start 初始化与 on/off-policy 最终达到相似峰值 [全文] — [2512.01374](https://arxiv.org/html/2512.01374)。
- Entropy-reward 经验律 `R = −a·exp(H) + b` 给出 H→0 时的"可预测上限" `−a + b`；系数与 RL 算法（GRPO/RLOO/PRIME）无关 [全文] — [2505.22617](https://arxiv.org/html/2505.22617)。这暗示：若 entropy 过早耗尽，上限由 entropy 预算决定，而 loss 选择只通过它对 entropy 消耗速度的影响间接作用。（与 ScaleRL "entropy 不预测下游"的结论存在张力。）

### Inferences
- ScaleRL 的 sigmoid 拟合中，"A 更低"可能部分是**不稳定被拟合成上限**：DAPO/GRPO 在 5–6k GPU-hours 后不稳定，若运行在平台期或下降期被截断，拟合会给出更低 A。所以前向比较里 DAPO 的 A 偏低，很可能是稳定性问题（在 LOO 中被 FP32 head 等修复后消失），而非 loss 的真实渐近差异。
- 按四个维度的初步分类（详见 Q7 的总表）：
  - **改变上限（有受控证据）**：数值精度/训练-推理一致性（FP32 head）、batch size、ε_max（非单调）；在长 CoT 中，最大长度预算与 overlong 处理会与上限交互。
  - **主要影响效率**：loss 类型（CISPO vs DAPO，在其他修复到位后）、聚合方式（在 LOO 中）。
  - **主要影响稳定性**：clipping/routing replay（off-policy 时）、IS correction（on-policy 时）、sequence-level 掩码负样本（agentic）、KL 估计器的偏差。
  - **主要是副作用**：长度归一化（长度漂移）、entropy bonus（entropy 爆炸/崩溃）。
- 只有 ScaleRL 一项研究对"上限 vs 效率"做了明确拟合，且只有 3 次重复，所以证据等级为 B。

### Gaps
- ScaleRL 正文没有给出 DAPO/GSPO/CISPO 各自的 A 数值，只有图；我未能读取图中的数值。
- 没有第二个独立的大规模研究复现 "loss 类型只影响 B" 这一结论。
- GSPO 在 ScaleRL LOO 中的表现（是否也落在 ±0.02 内）未在我读到的部分给出。

---

## Q3. clip-higher 的收益真的来自 entropy 保持吗？对不同 base model 稳健吗？

### Takeaway
机制上，提高 ε_high 确实会减少一个系统性的 entropy 下降偏置，经验上也能稳定地抬高 entropy；但"更高 entropy → 更好结果"并不稳健：在 base（非 aligned）模型上效果很小或为负（因为 clip 很少被触发，clip rate 约 0.3%），在 aligned/instruct 模型上收益更清楚；ScaleRL 发现 ε_max 对 A 是非单调的、而 entropy 本身不预测下游表现。结论：clip-higher 是**需要按模型调的二阶旋钮**，不是决定性的改动。

### Cited Findings
- DAPO 的动机：上 clip 限制低概率"探索" token 的增长；ε_high 从 0.2 提到 0.28 提高 entropy 与多样性；ε_low 保持 0.2，因为提高它会把 token 概率压向 0；AIME24 +2（36→38，单次运行，Qwen2.5-32B base）[全文] — [DAPO](https://arxiv.org/html/2503.14476)。
- Clip-high 降 entropy、clip-low 升 entropy；标准参数下 clip-high 占主导，即使随机奖励 entropy 也下降 [摘要] — [2509.26114](https://arxiv.org/abs/2509.26114)。
- Tricks-or-Traps（Qwen3-4B/8B，各有 base 与 aligned；500 步；seed 42，单 seed）：base 上改变上界对 entropy 影响很小、有时比 vanilla 更差，base clip rate 约 0.003；aligned 上提高上界"明显减缓 entropy collapse"并带来一致的下游收益；aligned 4B 在 0.32 最好，aligned 8B 在 0.28 最好、无单调趋势 [全文] — [2508.08221](https://arxiv.org/html/2508.08221)。
- Entropy Mechanism：clip-higher（ε=0.28）早期抬高 entropy，后期不稳定、性能下降；平均分提升只有 +0.2（7B）、+1.4（32B），远小于 Clip-Cov/KL-Cov [全文] — [2505.22617](https://arxiv.org/html/2505.22617)。
- ScaleRL：A 随 ε_max 先升后降（非单调）；entropy 不预测下游 — [2510.13786](https://arxiv.org/html/2510.13786)。
- MiniMax-M1：在每批 16 轮 off-policy 更新时，DAPO 的上 clip 放宽"效果较差"，促成了 CISPO — [MiniMax-M1](https://arxiv.org/html/2506.13585)。
- 生产配置：Magistral 训练中把 ε_high 在 0.26–0.28 之间调整"以保持组内 entropy 稳定"，Magistral Small 用 0.3；把 clip-higher 作为 entropy bonus 的替代 [全文] — [Magistral 2506.10910](https://arxiv.org/html/2506.10910)。ProRL 用 ε_high=0.4 [全文] — [ProRL 2505.24864](https://arxiv.org/html/2505.24864)。Qwen MiniRL 用 0.27 — [2512.01374](https://arxiv.org/html/2512.01374)。Skywork-OR1 称合适的 clip-higher 让 entropy 动态更慢更稳并提升测试性能（详细消融被截断）[全文前 100k] — [2505.22312](https://arxiv.org/html/2505.22312)。
- 80/20 论文认为 clip-higher 主要提高高 entropy token 的 entropy，但未单独验证 — [2506.01939](https://arxiv.org/html/2506.01939)。

### Inferences
- base 模型上 clip-higher 无效的机制性解释：若只有约 0.3% 的 token 触发上 clip（Tricks-or-Traps），ε_high 能改变的梯度质量就非常有限；clip-higher 的作用随 off-policy 程度（每批 minibatch 数）和策略漂移速度而放大。所以"对哪种模型有效"其实是"在哪种 staleness 下有效"的代理变量。
- "收益是否来自 entropy"：entropy 上升是可靠的一阶效应；但 ScaleRL（entropy 不预测下游）、Entropy Mechanism（entropy 升了但后期不稳定）表明，entropy 是必要条件的候选而非充分条件。在 aligned 模型上收益更大，与"aligned 模型初始 entropy 低、更容易 collapse"一致（推断）。
- 判定：Ceiling — 非单调地影响（ScaleRL）；Efficiency — 小；Stability — 混合（早期有利，Entropy Mechanism 称后期不稳定）；Side effects — entropy 升高，过大 ε_high 有不稳定风险。等级 **B（对 entropy 机制）/ D（对跨模型普遍收益）**。

### Gaps
- 没有多 seed 的 clip-higher 消融；所有报告的差距（+0.2 到 +2 点 AIME）都在 AIME 的单次运行噪声量级内。
- 没有找到在非 Qwen 家族（Llama、Mistral 等）上做的 clip-higher 受控消融。

---

## Q4. GSPO vs token-level：收益是 MoE/routing 不稳定特有的，还是普遍的？

### Takeaway
主要是 MoE/staleness 特有的，且即使在 MoE 上也可以被 Routing Replay 替代。原始 GSPO 只在 MoE（Qwen3-30B-A3B）上做了实验；Qwen 团队后续自己的受控研究（同一个 30B MoE）显示：on-policy 时最稳定的是**不带长度归一化的 token-level REINFORCE + 训练-推理 IS 修正**，off-policy 时 token-level + clipping + Routing Replay 即可稳定，论文实验里甚至没有跑 GSPO。dense 模型上的证据相互矛盾：ScaleRL（8B dense）中 GSPO 的 A 高于 DAPO 但多次发散；ARLArena（4B，agentic）中 GSPO 最好；CE-GPPO、DCPO（小 dense 模型）中 GSPO 有竞争力但不是最好。

### Cited Findings
- GSPO 的动机明确针对 MoE：每次梯度更新后约 10% 的激活专家改变，GRPO 需要 Routing Replay 才能收敛，而 GSPO"对单个 token 似然不敏感"；没有 dense 实验 [全文] — [GSPO](https://arxiv.org/html/2507.18071)。
- Qwen 2512.01374（Qwen3-30B-A3B cold-start，FP8 推理/BF16 训练，几十万 GPU-hours）：on-policy（gbs=mbs=1024）时 MiniRL（token-level，无长度归一化，含训练-推理 IS 修正）最好最稳；去掉 IS 修正则快速崩溃、entropy 骤降；on-policy 加 R3 没有收益。off-policy（N=2/4/8）时去掉 routing replay 或 clipping 都会过早崩溃；N=2 时 R2 优于 R3，N≥4 时 R3 优于 R2。GSPO 只在脚注中提到，未进入实验 [全文] — [2512.01374](https://arxiv.org/html/2512.01374)。
- DeepSeek-V3.2 用 "Keep Routing"（训练时强制使用推理时的专家路由），称其对 MoE RL 稳定性"至关重要"，自 V3-0324 起使用；其目标仍是 token-level GRPO 式 [全文] — [2512.02556](https://arxiv.org/html/2512.02556)。
- ScaleRL（8B dense）：GSPO、CISPO 的 A 都显著高于 DAPO；但 GSPO 多次中途发散，在 Scout（MoE）上无法恢复 — [2510.13786](https://arxiv.org/html/2510.13786)。
- SAPO 论文称 GSPO 和 GRPO-R2 在 30B MoE 上都出现"早期崩溃"（无数值）— [SAPO](https://arxiv.org/html/2511.20347)。
- ARLArena（Qwen3-4B SFT，ALFWorld/WebShop/Sokoban/TIR Math）：GSPO 平均 52.28 vs GRPO 46.16（+13.3%），ALFWorld 78.61 vs 62.36；作者把 GSPO 的优势归因于序列级 clipping 抑制了负优势低 ratio 序列；无 seeds [全文] — [2602.21534](https://arxiv.org/html/2602.21534)。
- CE-GPPO（R1-Distill-Qwen-1.5B dense）：GSPO 平均 54.3，仅次于 CE-GPPO 54.9，远高于 CISPO 47.6 — [2509.20712](https://arxiv.org/html/2509.20712)。
- DCPO（Qwen2.5-Math-7B dense）：GSPO AIME24 Avg@32 34.9，高于 GRPO 32.1、DAPO 31.6，低于 DCPO 38.8；Qwen2.5-14B AIME25 GSPO 9.9，最低 — [DCPO](https://arxiv.org/html/2509.02333v2)。
- 一篇 2026 论文（MinPRO）称理论上正确的修正是 prefix importance ratio，token-level 近似在 off-policy 漂移大时不稳定，在 dense 和 MoE 上都测试 [摘要] — [2601.22718](https://arxiv.org/abs/2601.22718)。（搜索摘要中称其 14B AIME24 pass@1 约为 GRPO 50.2 / GSPO 66.8 / CISPO 67.1，但我只读了摘要，**这些数字未核实**。）

### Inferences
- 根据 Q1 的公式，GSPO 与 token-level 只在 r≠1 时有区别。对 MoE，r≠1 的来源是 routing 漂移（训练-推理之间以及更新之间），这正是 Routing Replay 直接解决的；GSPO 是用"平均掉 token 噪声"间接解决。所以 GSPO 的优势在**高 staleness + MoE + 无 routing replay** 时最强，在 on-policy 或有 routing replay 时会消失。
- GSPO 把长度归一化重新引入（seq-mean-token-mean）；2512.01374 认为这是有偏的，Dr. GRPO 认为这会造成长度偏差。所以 GSPO 在 dense 模型上的表现取决于两个相反效应的净值：方差降低（有利）vs 长度归一化偏差（不利）。
- 在 agentic 多轮任务中（ARLArena），序列级 clipping 的价值可能主要来自"丢弃整条高度 off-policy 的负样本"，而非 MoE——这是另一个独立的机制（与 DeepSeek-V3.2 的 off-policy sequence masking 同类）。
- 判定（GSPO）：Ceiling — 有争议（ScaleRL 前向比较中较高；其他研究混合）；Efficiency — MoE 上更高（原论文），dense 上不确定；Stability — MoE 无 routing replay 时更好，但 ScaleRL 中会发散；Side effects — 长度归一化偏差，ε 尺度差三个数量级（需要重新调参）。**等级 D（普遍性）/ B（MoE 无 replay 时更稳）**。

### Gaps
- 没有找到在 dense 模型上、同一设置下比较 "GSPO vs token-level + IS 修正（无长度归一化）" 的受控研究。
- MinPRO 的具体数值和设置未核实。

---

## Q5. KL 到 reference 对推理 RLVR 是否重要？对 RLHF 呢？每种估计器的正确梯度是什么，常用框架的实现是否有偏？

### Takeaway
对 RLVR（可验证奖励的推理任务），2025 年的主流做法是去掉 KL 到 SFT reference：DAPO、Magistral、GLM-4.5、MiniMax-M1、ScaleRL、Skywork-OR1 都不用。少数受控证据（Skywork-OR1：β=1e-3 让 AIME 停滞；Entropy Mechanism：reference KL 保住了 entropy 但降低了性能）支持去掉。保留 KL 的系统（DeepSeek-R1、ProRL）都**定期把 reference 重置为当前策略**，因此它实际是近端/信任域项，而非锚定到 base。DeepSeek-V3.2 对数学用很弱或不用 KL。对 RLHF（学习到的 RM），KL 仍是防止 reward hacking 的标准手段，但 2025 年的论文没有给出新的受控证据。估计器方面：k3-as-loss（GRPO 默认）给出的是**反向 KL**的梯度（或一阶有偏近似），k1-as-loss 梯度期望为零，k2-as-loss 和 k1-in-reward 才是 `KL(π‖π_ref)` 的正确梯度；off-policy 时"as loss"形式需要 IS 修正。

### Cited Findings
**估计器梯度（数学，等级 B）**
- 见 Q1 的 KL 部分：[Tang & Munos 2506.09477](https://arxiv.org/html/2506.09477)、[2510.01555](https://arxiv.org/html/2510.01555v1)、[DeepSeek-V3.2](https://arxiv.org/html/2512.02556)。
- 经验：Llama-3.1-8B-Instruct + MATH（低 KL 区间）中估计器选择"实际差别不大"，错误的 k1 估计 KL 效率最低；70B→8B on-policy 蒸馏（高 KL 区间）中错误的 k3 估计"几乎没有进展"，序列级估计器优于最好的 token-level 估计器 — [2506.09477](https://arxiv.org/html/2506.09477)。
- Qwen2.5-Math-1.5B GRPO on-policy，β=0.5：k1-as-loss 与不加 KL 的曲线几乎一样（证实零均值梯度）；k2-as-loss 比 k3-as-loss 奖励方差更低、actor-ref 差距更小、entropy 略高（定性）— [2510.01555](https://arxiv.org/html/2510.01555v1)。
- "A Comedy of Estimators"（Qwen2.5-7B、Llama-3.1-8B-Instruct、Qwen3-4B-Instruct-2507）：梯度有偏的配置会导致训练不稳定，无偏配置在域内和域外都更好；KL 正则有助于稳定异步 off-policy RL；摘要无数值 [摘要] — [2512.21852](https://arxiv.org/abs/2512.21852)。
- 另一视角：2602.05494 认为 KL3 约束在数学上等价于一种把概率质量推向高置信动作的非对称 ratio clipping [搜索摘要，未读原文] — [2602.05494](https://arxiv.org/abs/2602.05494v2)。

**RLVR 中是否需要 KL**
- DAPO 去掉 KL：长 CoT 训练中模型分布本应大幅偏离初始模型 [全文] — [DAPO](https://arxiv.org/html/2503.14476)。
- Magistral："完全去掉 KL"，保留 reference 模型的算力"不值得"，第 7.4.2 节称 KL"主要是阻碍训练" [全文] — [Magistral](https://arxiv.org/html/2506.10910)。
- GLM-4.5："基于 GRPO 框架，去掉 KL loss 项" [全文] — [GLM-4.5 2508.06471](https://arxiv.org/html/2508.06471)。
- MiniMax-M1 CISPO 无 KL — [MiniMax-M1](https://arxiv.org/html/2506.13585)；ScaleRL 无 KL — [ScaleRL](https://arxiv.org/html/2510.13786)。
- Skywork-OR1：第二阶段加 β=1e-3 的 KL 把策略强力拉向 reference、KL→0，AIME24 不再显著提升；所有发布模型 β=0 [全文前 100k] — [2505.22312](https://arxiv.org/html/2505.22312)。
- Entropy Mechanism：reference KL 惩罚保持了 entropy 稳定但降低了性能 — [2505.22617](https://arxiv.org/html/2505.22617)。
- DeepSeek-R1：β=0.001（k3，加在 loss 中），**每 400 步把 reference 换成最新策略**，以平衡探索与稳定 [全文] — [2501.12948](https://arxiv.org/html/2501.12948)。
- DeepSeek-V3.2：不同领域需要不同 KL 强度，数学上较弱或不用 KL 可以提升性能；使用 IS 修正后的无偏 k3 — [2512.02556](https://arxiv.org/html/2512.02556)。
- ProRL（R1-Distill-Qwen-1.5B，>2k 步，约 16k GPU-hours）：保留 KL 并周期性硬重置 reference（并重置优化器），称 KL 对稳定性和持续 entropy"仍有益"，但**没有给出去掉 KL 的消融** [全文] — [ProRL](https://arxiv.org/html/2505.24864)。
- Kimi k1.5/K2 用相对上一迭代 π_old 的平方 log-ratio 正则（近端项），不是到 SFT 的 KL — [k1.5](https://arxiv.org/html/2501.12599)、[K2](https://arxiv.org/html/2507.20534)。
- DeepSeek-R1 指出，用基于模型的偏好奖励训练更多步"可能导致 reward hacking"（R1 第二阶段只在最后 400 步加入偏好奖励）— [2501.12948](https://arxiv.org/html/2501.12948)。

**RL's Razor / KL-to-base 与遗忘**
- 遗忘程度可由新任务输入上的前向 KL `E[KL(π₀‖π)]` 预测；ParityMNIST 二次拟合 R²：前向 KL 0.96、反向 KL 0.93、TV 0.80、权重 L1 变化 0.34；LLM 实验 R² 0.71；在相同新任务准确率下 RL 遗忘远少于 SFT（CKNNA 0.94 vs 0.56）；**主实验不使用 KL 正则**，结论是"on-policy 数据比负梯度更重要"；二元奖励下 PG 收敛到 KL 最小的最优策略（需要 e-flat 族假设，作者承认神经网络通常不满足）[全文] — [RL's Razor 2509.04259](https://arxiv.org/html/2509.04259)。
- 注意 RL's Razor 的定理用的是 `KL(π‖π₀)`，预测器用的是 `KL(π₀‖π)`，方向使用不一致 — 同上。

**框架实现（[代码]，本地 checkout）**
- verl：`use_kl_loss: false`（默认关），`kl_loss_type: low_var_kl`（k3，值 clamp 在 [-10,10]），`kl_loss_coef: 0.001`；`use_kl_in_reward: False`，`kl_penalty: kl`（k1）；提供 `"k3+"` 等后缀，用 straight-through 让前向值为 k3、反向梯度为 k2（注释明确说 k1、k3 的梯度期望不是 KL 的梯度）— `verl/verl/trainer/config/actor/actor.yaml:99-112`、`ppo_trainer.yaml:77-83`、`core_algos.py:2126-2190`。
- SkyRL：**默认 `use_kl_loss: True`**，`kl_estimator_type: "k3"`，`kl_loss_coef: 0.001`，`use_kl_in_reward: False`；`use_entropy_loss: False` — `SkyRL/skyrl/train/config/config.py`。
- slime：`--use-kl-loss` 默认 False，`--kl-loss-type` 默认 **k1**（若开启则 k1-as-loss 梯度期望为零），`--kl-loss-coef` 默认 0.0，有 `--use-unbiased-kl` 选项（默认 False）— `slime/slime/utils/arguments.py:795-860`。
- OpenRLHF：KL 默认放在 reward 中（`--algo.kl.init_coef` 默认 0.01，估计器默认 k1），`--algo.kl.use_loss` 默认 False；若 use_loss 且估计器不是 k2/k3 会打印提示 — `OpenRLHF/openrlhf/cli/train_ppo_ray.py:417-485,675-680`。
- AReaL：`kl_ctl` 默认 0.1、`kl_estimator` 默认 k1，以 reward 形式施加（`rewards = -kl_ctl * kl_estimator(old_logp, ref_logp)`），仅在配置了 ref 时生效 — `AReaL/areal/api/cli_args.py:1458-1462`，`AReaL/areal/trainer/ppo/actor.py:202`。

### Inferences
- 实现层面：k3-as-loss（GRPO/verl 的 `low_var_kl`、SkyRL 默认）给出的是反向 KL 梯度（Tang & Munos）或一阶近似（2510.01555）；在 β=0.001、策略距 reference 很近（δ≈1）时，`1 − δ ≈ −log δ`，偏差很小——这解释了为什么低 KL 区间下估计器选择"实际差别不大"。偏差只在高 KL 区间（蒸馏、长时间训练、不重置 reference）才显著。
- 框架默认配置几乎都避开了最坏情况：verl、slime 默认关闭 KL loss；OpenRLHF、AReaL 默认使用 k1-in-reward（正确的形式）。但 slime 开启 KL loss 时默认估计器是 k1，**这在数学上无正则作用**；SkyRL 默认开启 k3-as-loss。这些需要用户注意。
- 带 reference 周期重置的 KL（R1、ProRL）与 Kimi 的 π_old 近端项在功能上相似：都是信任域/近端正则，而不是"保持接近 base"。所以"RLVR 是否需要 KL"应拆成两个问题：(i) 锚定到 base 的 KL——对推理 RLVR 很可能有害或无用（多项报告，Skywork 是最接近受控的证据）；(ii) 近端 KL——可能有助于稳定性，尤其在 off-policy/异步时（Comedy of Estimators 的摘要声明），但证据有限。
- RL's Razor 表明 on-policy RL 自带隐式的"最小 KL"偏好，即使不加显式 KL 也遗忘较少；这为 RLVR 中去掉显式 KL 提供了理论上的理由，但其定理假设较强。
- 判定（RLVR 中的 KL-to-base）：Ceiling — 可能降低（Skywork、Entropy Mechanism）；Efficiency — 去掉可省下 reference 前向计算；Stability — 近端/重置版本可能有帮助（无受控消融）；Side effects — 保持 entropy、减少遗忘。**等级 B（去掉不会伤害 RLVR）；C（KL 对稳定性有帮助）**。估计器偏差：**等级 B（数学清楚）+ 两项小规模实验**。

### Gaps
- 没有找到 2025–2026 年针对 RLHF（学习到的 RM）中 KL 系数/估计器的新受控研究；RLHF 中 KL 防止 reward hacking 的证据主要来自更早的工作，本轮未读取。
- Comedy of Estimators 和 2510.01555 的大规模实验（附录 K/L）数值未读到。
- ProRL 未提供去掉 KL 的消融，所以"KL + 重置有助于长时间训练"仍是未验证的说法。

---

## Q6. Entropy bonus 与 entropy collapse 分析：什么是决定性的？

### Takeaway
固定系数的 entropy bonus 在 LLM RL 中很脆弱（对系数和数据都敏感，易爆炸），主流配置默认关闭；有效的替代方案是：自适应目标 entropy（Skywork-OR1）、clip-higher（Magistral）、协方差定向干预（Clip-Cov/KL-Cov）、只在高 entropy token 上训练（80/20）、减少 off-policy 更新次数。entropy collapse 的主要驱动因素是正优势高概率 token 的协方差项和 off-policy 多轮更新。但 entropy 本身不是可靠的性能预测指标（ScaleRL）。

### Cited Findings
- Entropy Mechanism：entropy 损失系数 0.0001/0.001 几乎无效，0.01 导致 entropy 爆炸，0.005 稳定了 entropy 但没超过基线 — [2505.22617](https://arxiv.org/html/2505.22617)。
- Magistral：entropy bonus 不稳定，效果随数据集变化显著——纯数学数据上 entropy 下降，数学+代码数据上爆炸；改用 clip-higher — [Magistral](https://arxiv.org/html/2506.10910)。
- Skywork-OR1：自适应 entropy 控制（目标 entropy 0.2，步长 Δ=0.005，只在 entropy ≤ 目标时施加 entropy loss）；off-policy 更新（N_SGD=2 或 4）使 entropy 在几步内跌到很小值、测试性能停止提升，on-policy（N_SGD=1）下降更缓且最终更高；温度 0.6 时 entropy 起点很低或 100 步内接近 0 — [2505.22312](https://arxiv.org/html/2505.22312)。
- Qwen3 技术报告：控制 entropy 使其"稳定增长或保持稳定"，称这对稳定训练至关重要；235B-A22B 在 170 步 RL 中 AIME'24 从 70.1 升至 85.1；未给出具体方法 [全文前 100k] — [Qwen3 2505.09388](https://arxiv.org/html/2505.09388)。
- GLM-4.5 用动态温度应对 entropy 下降（奖励稳定后提温，取使验证性能保持在最优 1% 以内的最高温度）— [GLM-4.5](https://arxiv.org/html/2508.06471)。
- 理论：entropy 变化 ≈ `−η·Cov(log π, π·Â)`；协方差大多为正，简单 prompt 上更大 — [2505.22617](https://arxiv.org/html/2505.22617)。
- CE-GPPO 四象限：PA&HP 驱动 collapse；NA&LP 降低 entropy；PA&LP、NA&HP 维持多样性 — [2509.20712](https://arxiv.org/html/2509.20712)。
- ScaleRL：entropy 不预测下游表现，batch size 是决定性因素 — [2510.13786](https://arxiv.org/html/2510.13786)。
- 框架默认：verl `entropy_coeff: 0`、slime `--entropy-coef 0.0`、SkyRL `use_entropy_loss: False` [代码]。

### Inferences
- 判定（固定 entropy bonus）：Ceiling — 无证据；Efficiency — 无；Stability — 负面（爆炸）；Side effects — 强烈依赖数据。**等级 A（"固定 entropy bonus 脆弱、不是决定性手段"有 3 个独立来源：Entropy Mechanism、Magistral、Skywork-OR1）**。
- 自适应 entropy 控制、Clip-Cov/KL-Cov、80/20 都只有原论文证据（**等级 C**）。
- off-policy 程度（每批 minibatch 数）是 entropy collapse 的最大可控驱动因素之一（Skywork-OR1 受控），这与 Q4 中"staleness 决定 loss 选择是否重要"的结论一致。

### Gaps
- Skywork-OR1 entropy 系数常数消融（1e-4~1e-2）的结果被截断，未读到。
- 没有找到 Clip-Cov/KL-Cov 或 80/20 的独立复现。

---

## Q7. 前沿技术报告用了什么？总体判定表

### Takeaway
前沿报告趋同于：无 KL-to-base（或周期重置的弱 KL）、token-level/组内 token 归一化、非对称或宽松的 clip（或 IS 截断）、按 0/1 准确率过滤 prompt、以及 MoE 的路由一致性措施。没有一家报告为其 loss 选择提供多 seed 的受控消融。

### Cited Findings
| 系统 | Loss / 粒度 | 聚合 | Clip / IS | KL | Entropy | 来源 |
|---|---|---|---|---|---|---|
| DeepSeek-R1 | GRPO token-level | 1/G·1/|o_i|（原 GRPO） | 第一阶段 ε=10 | k3-as-loss，β=0.001，每 400 步重置 ref | 未提 | [2501.12948](https://arxiv.org/html/2501.12948) |
| DeepSeek-V3.2 | GRPO token-level，Â=R−mean | 1/G·1/|o_i| | ε 未给出；负优势且 mean log(π_old/π_θ)>δ 的序列被 mask | IS 修正的无偏 k3；数学弱或无 KL | 未提；Keep Routing、Keep Sampling Mask | [2512.02556](https://arxiv.org/html/2512.02556) |
| Kimi k1.5 / K2 | 平方损失 `(r − r̄ − τ log π/π_old)²`（mirror descent） | 每题 K 个样本平均 | 无 clip | 无到 SFT 的 KL；对 π_old 的近端项 | K2 用温度衰减；PTX loss | [2501.12599](https://arxiv.org/html/2501.12599), [2507.20534](https://arxiv.org/html/2507.20534) |
| Qwen3 | GRPO（off-policy） | 未提 | 未提 | 未提 | 控制 entropy 稳定/增长 | [2505.09388](https://arxiv.org/html/2505.09388) |
| Qwen（后续，GSPO/MiniRL） | GSPO 序列级（2507.18071）→ MiniRL token-level + IS 修正（2512.01374） | GSPO：seq-mean-token-mean；MiniRL：无长度归一化 | GSPO 3e-4/4e-4；MiniRL ε_low=0.2、ε_high=0.27，TIS 阈值 5 | 未提 | — | [GSPO](https://arxiv.org/html/2507.18071), [2512.01374](https://arxiv.org/html/2512.01374) |
| MiniMax-M1 | CISPO | 组内 token 总数 | sg(clip IS)，无下界，只调 ε_high | 无 | — ；FP32 LM head | [2506.13585](https://arxiv.org/html/2506.13585) |
| Seed1.5-Thinking | VAPO（value-based）/ DAPO | token-level | clip-higher（数值未给出） | 未提 | 未提；正样本 LM loss | [2504.13914](https://arxiv.org/html/2504.13914) |
| Magistral | GRPO 修改版 | 组内总长度 | ε_high 0.26–0.28（Small 0.3） | 完全去掉 | 不用 bonus，用 clip-higher | [2506.10910](https://arxiv.org/html/2506.10910) |
| GLM-4.5 | GRPO 无 KL | 代码 RL 用 token-weighted mean | 未给出 | 去掉 | 动态温度 | [2508.06471](https://arxiv.org/html/2508.06471) |

- Seed1.5-Thinking：Qwen-32B dense 上 DAPO 50% vs VAPO 60%；Seed-150B-MoE 上 73% vs 79%（有限步数的消融）— [2504.13914](https://arxiv.org/html/2504.13914)。
- GLM-4.5：代码 RL 中 token-weighted mean"比常规的 sequence-mean loss 收敛更快、长度偏差更小"（来自较小的实验模型）— [2508.06471](https://arxiv.org/html/2508.06471)。

**总体判定表（本笔记的判断，依据见上文各节）**

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| PPO ratio clip（off-policy 时） | 无证据 | 允许复用样本 | off-policy 必需（MoE 研究） | entropy 偏置；随机奖励下放大先验（Qwen2.5-Math） | B |
| Dual-clip | ? | ? | 机制合理 | ? | C |
| Clip-higher | 非单调（ScaleRL） | 小 | 混合 | entropy ↑ | B / D（跨模型） |
| Token-level 聚合（vs seq-mean-token-mean） | 孤立时 prompt-avg 最好；LOO 中在噪声内 | base 模型上有帮助 | — | 去除长度偏差 | B |
| Dr. GRPO 常数归一化 | ? | 更好的 token 效率（Qwen2.5-Math） | 长度稳定 | 错误答案变短 | C（主要在 Qwen2.5/Qwen2.5-Math 上测试；MiniRL 的"无长度归一化最好"是间接旁证） |
| Overlong filtering | ? | 短上下文有帮助 | ? | 与最大长度强交互 | B |
| GSPO | 争议 | MoE 上更好 | MoE 无 replay 时更好；ScaleRL 中会发散 | 长度归一化；ε 需重调 | D |
| GMPO | ? | ? | 称更稳 | — | C |
| CISPO | LOO 中在噪声内 | B 更高（ScaleRL） | agentic 中崩溃 | 超参鲁棒 | D / B |
| SAPO | ? | ? | MoE 上称更稳；agentic 中崩溃 | — | C / D |
| TOPR、ASPO、DCPO、CE-GPPO、DPPO、DGPO | ? | 各自原论文称有提升 | ? | ? | C |
| Clip-Cov / KL-Cov、80/20、NSR | 原论文称有提升（以 Qwen 为主） | ? | ? | entropy ↑ / pass@k ↑ | C |
| 固定 entropy bonus | 无 | 无 | 负面 | 爆炸 | A（负面结论） |
| KL-to-base（RLVR） | 可能降低 | 去掉可省算力 | 近端版本可能有帮助 | 减少遗忘 | B / C |
| KL 估计器选择（k3 vs k2） | 低 KL 区间无差别 | — | 高 KL 时 k3-as-loss 有偏 | — | B |

### Inferences
- **最可能是决定性的（跨多项研究）**：(1) 减少 staleness 或正确处理 staleness（IS 修正、off-policy 时 clip/mask、MoE 的 routing 一致性）；(2) 数值层面的训练-推理一致性（FP32 LM head 是 ScaleRL 中单项 A 提升最大的改动；MiniRL 去掉 IS 修正会快速崩溃）；(3) 不要固定 entropy bonus；(4) 在推理 RLVR 中不需要锚定到 base 的 KL。
- **大多不是决定性的（或证据不足）**：大多数 2025–2026 年提出的 clip 变体、loss 粒度变体和 token 选择方法——它们的收益是在单 seed、短训练、通常使用 Qwen2.5-Math 或 R1-Distill-1.5B 的设置下报告的，并且彼此之间的排序在不同论文中会互相矛盾（例如 CISPO 在 ScaleRL 中最好，在 CE-GPPO 和 ARLArena 中最差）。
- **需要标注的混杂因素**：
  - Qwen2.5-Math：随机奖励也能提升 21.4 点 MATH-500，作者归因于 clipping 偏置 [2506.10947](https://arxiv.org/html/2506.10947)；Dr. GRPO 发现 Qwen2.5-Math 不加模板时平均准确率从 23.8 升到 38.2 [2503.20783](https://arxiv.org/html/2503.20783)。因此在 Qwen2.5-Math 上报告的 clip/loss 收益（DCPO、GMPO 的一部分、NSR、Dr. GRPO 本身）需要非 Qwen 模型复现。
  - AIME 小样本：AIME 只有 30 题，单题 = 3.3 点；多数论文报告 avg@16/32 但只跑一个训练 seed，因此大多数 +1~+3 点的差距在训练 seed 方差之内。"A Sober Look" 称很多 RL 方法的提升"远低于先前声称"，在 AIME'24 这类小基准上容易过拟合 [摘要] — [2504.07086](https://arxiv.org/abs/2504.07086)。ScaleRL 3 次重复给出 A 的误差 ±0.02 [2510.13786](https://arxiv.org/html/2510.13786)，这是目前唯一有量化 seed 方差的大规模研究。
  - 短训练：DCPO 400 步、3072 tokens；Tricks-or-Traps 500 步；TOPR 512 tokens。ScaleRL 显示早期效率与渐近上限可能排序不同。
  - 弱基线：DCPO 用各基线的公开默认值且未扫参；GSPO 的 ε 尺度在 ScaleRL 中需要重调一个数量级；verl 的 CISPO 默认截断在 [0.8,1.2]，与原论文不同——实现差异本身可能就是某些论文中"CISPO 很差"的原因（推断，未验证）。

### Gaps
- Qwen3、Seed1.5-Thinking 报告没有给出 clip 数值和 KL 设置；DeepSeek-V3.2 没有给出 ε、β、δ 的数值。
- 没有前沿实验室公开过多 seed 的 loss 选择消融。
- 未读取 Huang et al. "The N+ Implementation Details of RLHF with PPO"，因此本笔记没有涵盖 RLHF PPO 的实现细节（例如 EOS 处理、per-token KL reward）。
