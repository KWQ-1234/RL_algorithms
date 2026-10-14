# 多数 RL 算法改动只买效率，不抬上限

从 2017 年的 PPO 到 2026 年 10 月，受控条件下的大多数 LLM RL 算法改动只改变效率或稳定性，不改变渐近上限。这些改动包括 advantage 估计器、std 归一化、loss 聚合粒度、十余种 clip/IS 变体、新的 DPO 系 loss，以及作为奖励的学习型 PRM。有受控证据显示能移动上限的因素只有四个：base model 与 pretraining、生成长度预算、batch size、训练–推理数值一致性。唯一用 sigmoid 拟合把"上限 A"和"效率 B"分开测的研究是 ScaleRL（8B dense + Llama-4 Scout，总计 >400K GPU-hours）。它的 leave-one-out（LOO）结果是：撤回 recipe 中任一组件，A 都落在作者自报的 ±0.02 误差带内。明显超出误差带的只有 FP32 LM head（0.52→0.61）、batch 2048（0.610→0.645）、32k 生成长度（0.610→0.645）和换用 Scout MoE（0.710）（[ScaleRL](https://arxiv.org/abs/2510.13786)）。决定一次训练能否到达上限的是一组"正确性"措施，它们保证梯度估计量确实是它声称的那个量：

- **限制每批 rollout 的 off-policy SGD 次数**：多次复用导致 entropy collapse，有三组独立受控证据。
- **消除 MoE routing 与 BF16 带来的训练–推理 mismatch。**
- **多轮场景下屏蔽环境 token。**
- **保证验证器的精度。**

同时，混杂因素的量级普遍大于论文报告的算法增益：

- AIME 单题对应 3.3 点。
- 跨 seed 的 Pass@1 标准差为 5–15 点。
- Qwen2.5-Math-7B 在随机奖励下 MATH-500 也能涨 21.4 点。
- 同一 SFT checkpoint 加同一数据的 DPO baseline，在两篇论文里相差 5.5 个 LC 点。

对理论背景的读者，最省力的读法是把每个 surrogate loss 拆成三个旋钮：token 权重 w、梯度掩码 m、归一化。严格 on-policy 且推理引擎与训练引擎一致时 r≡1，此时 PPO-clip、DAPO、GSPO、CISPO、SAPO 的梯度都退化为同一个 REINFORCE，差别只剩归一化。所以这些方法之间的经验差距，本质上测的是 staleness 与 mismatch 有多大。整体证据等级偏低：多数结论只有 C（仅原论文），跨模型族、多 seed 的消融几乎不存在。这两点正是值得用小规模实验补上的空白。

> 本报告是"RL 算法与决策"blog 的证据底稿，与已有的 "RL Infrastructure for the Mathematically Inclined" 平行。它综合了 8 份研究笔记。凡是笔记里标注为"仅读摘要/片段"的数字，正文都会标出来，不作为确定结论使用。

## 判定口径：上限只有一项研究直接测过，误差带是 ±0.02

每条判断都在四个维度上给出结论：**ceiling（上限）**、**efficiency（效率）**、**stability（稳定性）**、**side effects（副作用）**。证据等级沿用任务定义：A = 多个独立的受控消融；B = 单个受控消融，或一个清晰的数学机制；C = 只有原论文的说法；D = 证据相互矛盾。同一改动在不同问题上可以有不同等级。例如，"std 归一化不移动上限"是 A，而它对效率的影响方向是 D。

阅读深度标记沿用笔记的标注：〔全文〕表示读了方法与实验部分；〔全文*〕表示通过摘要式抓取读了 HTML，数字经过摘要器转述，长文通常只覆盖前约 100k 字符；〔部分〕表示只读了部分章节；〔摘要〕表示只读了 abstract；〔片段〕表示只见过搜索摘要；〔先验〕表示 ID、标题、日期已核实，但正文本轮没有重读；〔二手〕表示博客或聚合来源；〔代码〕表示读了本地框架 checkout（`_external_repos/`，提交日期为 2026-05-15 到 05-22）。

"上限"有两层含义，必须分开写。**第一层是 pass@1 的渐近值**。ScaleRL 拟合的是 R_C = R₀ + (A−R₀)/(1+(C_mid/C)^B)，其中 A 是渐近 pass rate，B 是效率指数（[ScaleRL 〔全文〕](https://arxiv.org/abs/2510.13786)）。这个 A 的可辨识性很弱，原因有三：

- 三次独立 ScaleRL 运行之间，A 最多相差 ±0.015，作者因此建议用 ±0.02 作为误差带。B 的误差带作者认为"难以估计"，所以没有给。
- 2610.04158 发现，在 Qwen3-14B-Base 上，log-sigmoid 与 log-linear（没有上限）两种拟合在拟合窗口内 R² 都大于 0.99，"往往几乎无法区分"（[2610.04158 〔部分〕](https://arxiv.org/abs/2610.04158)）。
- 某些运行在 5–6k GPU-hours 之后会变得不稳定（GRPO 约 6k，DAPO(ε_max=0.26) 约 5k）。如果这种运行被截断后拿去拟合，会把"不稳定"拟合成"更低的 A"（[ScaleRL](https://arxiv.org/abs/2510.13786)）。

由此得出的规则是：**"只影响效率"的判定比"抬高上限"的判定更可靠；A 的差异小于 0.02–0.03 时，一律记为"未分辨"。** **第二层是 pass@large-k 的覆盖率**。RLVR 通常抬高 pass@1，但在大 k 下往往被 base model 反超（[Yue et al. 〔全文*〕](https://arxiv.org/abs/2504.13837)）。这一层的上限主要由 pretraining、mid-training 和 distillation 决定，见"奖励类型"一节。

## 混杂因素的量级普遍大于算法增益

2020 年的深度 RL 研究已经给出结论：**代码层面的实现细节比算法标签更重要**。Engstrom et al. 在 2⁴ 全因子设计中做了实验，每个格子 5 个 seed，每个估计量至少 80 个 agent，并用 bootstrap 95% CI。结果是，在全部三个任务上，code-level 优化（value clipping、reward scaling、正交初始化、LR annealing 等）的效应都大于 PPO 与 TRPO 之差。带上这些优化的 TRPO+ 在 Humanoid 上得分 1030，高于 PPO 的 806。他们还发现，三种算法"都没能维持基于 ratio 的 trust region"，并且 PPO-NoClip"一致优于 PPO-M"（[Engstrom et al. 〔全文〕](https://arxiv.org/abs/2005.12729)）。Andrychowicz et al. 用超过 25 万个 agent 得出类似结论：entropy 与 KL 正则除 HalfCheetah 外都没有显著帮助，PPO-style value clipping 也不该用（[2006.05990 〔全文前半〕](https://arxiv.org/abs/2006.05990)）。LLM RL 的消融文献在统计功效上明显弱于这两项研究：多数是单 seed、单一模型族（主要是 Qwen）、1.5B–8B 规模、少于 1000 步、只有图没有表。

**评测噪声**是第一个混杂因素。Hochlehnert et al. 的测量结果如下（[2504.07086 〔全文前~75%〕](https://arxiv.org/abs/2504.07086)）：

- 20 个 seed 下，Pass@1 的标准差为 5–15 点。AIME 单题等于 3.3 点。
- 改评测温度最多造成约 15 点的差异，改 top_p 约 8 点。
- 换集群最多差 8 点。即使固定 Docker 镜像并开启确定性，仍有 2–5 点噪声。
- 换评测框架（lighteval 与 evalchemy）相差 4.5 点。
- 稳定估计需要 K≥30 个 seed。

在 10 个 seed 加配对检验下，Open-RS 原本声称的 10–15 点增益不显著；Oat-Zero 的 AIME24 是 28.0±3.1，对比 base 的 20.7，同样不显著。他们自己的协议也有缺陷：只用了 10 个 seed（自己的分析建议 30 个），并且对 Qwen-Math 设了 4096 token 上限（自己的建议是 32K 以上）。按这些数字，**单次运行中小于 6–8 点的 AIME 差异应视为噪声**。DAPO 逐项消融里的 +2、+3、+1 都在这个范围内。

**Base model、模板与污染**是第二个混杂因素，而且常常是最大的那个：

- **模板**：Qwen2.5-Math-7B 用 R1 模板得 0.0，用 Qwen 模板得 26.5，不用模板得 38.2（[Dr. GRPO 〔全文〕](https://arxiv.org/abs/2503.20783)）。
- **随机奖励**：Qwen2.5-Math-7B 在 MATH-500 上，真实奖励带来 +29.1，错误标签 +24.1，随机奖励 +21.4。OLMo2 在这些设定下持平，Llama3 基本无效（[Spurious Rewards 〔全文前~55%〕](https://arxiv.org/abs/2506.10947)）。完全开源、做过去污染的 OLMo 3 把随机奖励当作阴性对照，在所有 benchmark 上都没有收益（[OLMo 3 〔全文 RL 节〕](https://arxiv.org/abs/2512.13961)）。
- **污染证据**：给出 MATH-500 题目的前 60%，Qwen2.5-Math-7B 补全其余部分的 exact match 为 54.6%，Llama3.1-8B 只有 3.8%；在更新的 LiveMathBench 上，Qwen 的 EM 为 0（[2507.10532 〔全文*〕](https://arxiv.org/abs/2507.10532)）。
- **不需要正确监督也能涨**：同在 Qwen2.5-Math-1.5B 上，只训 20 步 entropy loss、完全不用奖励，MATH500 就从 36.0 涨到 63.4；只给 format reward 能到 65.0；60% 标签错误时得 34.6，标签全对时 35.9（[1-shot RLVR 〔全文*〕](https://arxiv.org/abs/2504.20571)）。

随机奖励为什么有效，机制上存在争议（D）。Shao et al. 认为 clipping 偏置是必要条件，去掉 clipping 增益就消失。2512.16912 则给出 E|N_raw|/E|C_clip| ≥ ~17 的下界，认为 clip 的贡献远小于原始信号，打开 clipping 甚至会降低验证集表现；它还在 R1-Distill-Llama-8B 和 QwQ-32B 上观察到随机奖励增益（[2512.16912 〔全文*〕](https://arxiv.org/abs/2512.16912)）。两种解释的共同点是：**只要模型的众数行为好于平均行为，任何集中概率质量的更新都会显得有效。** clipping 不对称、entropy 最小化、多数投票、format 奖励，都是通往这种"锐化"的不同路径。

**Base 与 aligned 会让结论反转。** 在 Qwen3 上，loss 聚合的最优选择在 base 和 aligned 之间相反，clip-higher 也只对 aligned 模型有效（[Tricks or Traps 〔全文〕](https://arxiv.org/abs/2508.08221)）。长度课程对 R1-Distill-7B 有效，却让 Qwen3-4B "急剧退化"（[Polaris 博客 〔全文，二手级〕](https://hkunlp.github.io/blog/2025/Polaris/)）。在相同的数学 RL 下，Qwen3 和 Gemma-3 在部分域外任务上朝相反方向变化（[2610.04158 〔部分〕](https://arxiv.org/abs/2610.04158)）。

**弱 baseline 与框架 bug**是第三个混杂因素：

- VAPO 和 VC-PPO 用来对比的 vanilla PPO 只有 5 和 5.6 AIME。这是已知的坏配置：critic 从 RM 初始化，λ=0.95（[VC-PPO 〔全文〕](https://arxiv.org/abs/2503.01491)）。
- Back to Basics 里的 PPO 只在 LR ∈ {1e-6, 1e-5} 上调过参（[RLOO 〔全文〕](https://arxiv.org/abs/2402.14740)）。
- 同一个 Zephyr-SFT 加 UltraFeedback 的 DPO baseline，在 SimPO 论文里是 15.1 LC，在 WPO 论文里是 20.6±0.7（5 seeds）（[WPO 〔全文〕](https://arxiv.org/abs/2406.11827)）。这 5.5 点的差距大于 SimPO 声称的优势。
- 2604.23747 摘要称，许多 mixed-policy 论文的 SFT baseline 被 DeepSpeed CPU-offload 优化器 bug 和 OpenRLHF loss 聚合 bug 拖低了。修复之后，SFT→RL 反超 mixed-policy 方法：Qwen2.5-Math-7B 上 +3.8，Llama-3.1-8B 上 +22.2（[2604.23747 〔摘要〕](https://arxiv.org/abs/2604.23747)）。
- TRL 的 IPO 实现曾把 loss 写成求和而不是平均，修复前 IPO 看起来更差（[HF 博客](https://huggingface.co/blog/pref-tuning)）。

**算力记账与报告方式**是第四个混杂因素。许多"效率"收益按步数计，而不是按 rollout 算力计：

- PRIME 每步耗时多约 28%。
- TreeRL 每次更新用 480 条 response，对照组只有 256 条。
- PROF 用了 2 倍采样。

许多稳定性结论报告的是 best checkpoint 或窗口内峰值，包括 SimpleTIR、R3、M2PO，以及按方法分别选择 early stop 的 PRPO。

下表是由上述混杂因素得出的"读论文检查清单"。每一项都对应笔记中至少一个实际翻车案例。

| 检查项 | 机制或案例 | 不满足时的处理 |
|---|---|---|
| Base model 是否只有 Qwen2.5(-Math) | 随机奖励 +21.4（[2506.10947](https://arxiv.org/abs/2506.10947)）；模板 0→38.2（[2503.20783](https://arxiv.org/abs/2503.20783)） | 最多记 C；要求在 Llama、OLMo 或 Gemma 上复现 |
| 有没有随机奖励或 format 奖励的阴性对照 | OLMo 3 用它作污染对照（[2512.13961](https://arxiv.org/abs/2512.13961)） | 增益只算"elicitation 上界" |
| 评测集是否干净、是否晚于模型截止日期 | AIME24 的增益常常不迁移到 AIME25（[2504.07086](https://arxiv.org/abs/2504.07086)） | 看 AIME25+、LiveMathBench 或合成任务 |
| 训练 seed 数与评测 seed 数 | Pass@1 std 5–15；建议 K≥30 | 小于 6–8 点的单次 AIME 差异当噪声 |
| Baseline 是否与新方法同等调参 | VAPO 的 PPO=5；DPO 15.1 vs 20.6 | 拿独立实现的 baseline 数字对照 |
| 是否同算力（rollout 数、每步时间） | TreeRL 480 vs 256；PRIME +28%/step | 按 GPU-hours 重新换算 |
| 报告的是峰值还是终值 | SimpleTIR、R3、M2PO 报峰值 | 稳定性结论降一级 |
| 训练长度与上下文 | 多数消融 <1000 步、≤8K | 不外推到长 CoT 或长训练 |
| 每批 rollout 做几次梯度更新、引擎是否一致 | r≡1 时多数 loss 变体等价（见 clip 一节） | 先弄清楚比较的到底是 loss 还是 staleness |
| 框架实现与论文是否一致 | verl 的 CISPO 默认把权重截断在 [0.8,1.2]，原文无下界（〔代码〕） | 核对默认配置 |
| "上限"指 pass@1 还是 pass@k | RL 提高 pass@1，大 k 下常被 base 反超（[2504.13837](https://arxiv.org/abs/2504.13837)） | 两种上限分开报告 |
| 结论在 base 和 aligned 模型上是否一致 | 聚合方式与 clip-higher 的结论会翻转（[2508.08221](https://arxiv.org/abs/2508.08221)） | 标注适用的模型类型 |

## 十个设计轴（外加 DPO 系）上的逐项判定

本节按设计轴组织。每个轴先给数学机制，再给受控证据和矛盾，最后用一张判定表。表中"笔记推导"指笔记作者自己的推导，原论文中没有。

### Advantage 与 baseline：用不用 critic 不决定上限，λ 和初始化决定 PPO 能不能用

**机制。** 几乎所有在用的 critic-free 估计量都可以写成 Â_i = (r_i − b(x))·w(x)。其中 b 是逐 prompt 的 baseline，w 是逐 prompt 的缩放。只有当 b 不依赖 y_i 本身时，估计量才无偏。group mean 把 r_i 自己也算进去，会带来 O(1/G) 的偏差；乘上 G/(G−1) 正好把它消掉。所以 **RLOO 恰好等于 Dr. GRPO 的 advantage 乘一个常数**，这一点 Dr. GRPO 自己也承认"至多差一个缩放因子"（[Dr. GRPO](https://arxiv.org/abs/2503.20783)）。KRPO 的 Kalman baseline x̂_{i|i} 同样包含 r_i，所以有偏，而且依赖组内样本的顺序。它的"归一化" √P_{i|i} 只依赖 Q、R、P₀，与奖励无关（笔记推导，[KRPO 〔全文〕](https://arxiv.org/abs/2505.07527)）。ReMax 用 greedy 解码的奖励作 baseline，所以无偏；但在二值奖励下，greedy 已经能解出的 prompt 上，正确样本得到的信号为零（[ReMax 〔部分〕](https://arxiv.org/abs/2310.10505)）。

PPO 的 GAE 是另一种机制。λ<1 时，终点奖励传到 token t 时被乘上 λ^{T−t}，长 CoT 的早期 token 几乎收不到信号（[VC-PPO 〔全文〕](https://arxiv.org/abs/2503.01491)）。γ=λ=1 时，Â_t = R − V(s_t)，critic 只起 state-dependent baseline 的作用，带来的只是方差缩减。VAPO 的长度自适应 λ = 1−1/(αl)（α=0.05）是另一回事：取 l=8000，λ≈0.9975，有效视界约 400 token，终点奖励传到 token 0 时的权重约为 e^{−20}。所以 VAPO 早期 token 的 advantage 几乎完全由 critic bootstrap 得到，这是真正依赖 critic 的估计量（笔记推导，[VAPO 〔全文〕](https://arxiv.org/abs/2504.05118)）。

**证据。** "长 CoT 下 PPO 必须用 λ=1（或至少 critic 用 λ=1）"有四方证据，彼此一致：

- RLOO 论文在 HH 上发现 λ=1 最好（[2402.14740](https://arxiv.org/abs/2402.14740)）。
- DeepSeek-R1 附录 A.3：λ=0.95 时 PPO"明显差于 GRPO"，λ=1 时"接近 GRPO"（[DeepSeek-R1 〔全文〕](https://arxiv.org/abs/2501.12948)）。
- ORZ：λ=0.95 时响应长度崩溃（[ORZ 〔全文〕](https://arxiv.org/abs/2503.24290)）。
- VC-PPO：PPO 只有 5.6，去掉 decoupled GAE 是 29.4，完整 VC-PPO 是 41.9。

但 actor 端的 λ 在 VC-PPO 中不单调：λ_actor 取 0.95、0.99、1.0 时分别得 41.3、41.9、29.4。这只是同一团队的单次运行。

"critic 对比 critic-free"一项，原始论文的增益和独立复现结果冲突，所以记 **D**：

- VAPO 报告 AIME24 60.4，同团队的 DAPO 为 50。Seed1.5-Thinking 在 Qwen-32B 上 VAPO 对 DAPO 是 60 对 50，在 Seed-150B-MoE 上是 79 对 73（[Seed1.5-Thinking 〔部分〕](https://arxiv.org/abs/2504.13914)）。
- GenAC 在 Qwen3-8B-Base 上只换估计器、共享同一个 PPO 式 loss，跑 200 步。单次结果是 GRPO 50.22、RLOO 49.93、VC-PPO 50.39；3 个 seed 平均是 VC-PPO 50.85、GRPO 49.96。value 类方法前期样本效率更高，但"优势随时间缩小"。每次迭代的成本：GRPO 是 PPO 的 0.6 倍，VinePPO 是 4.3 倍。GenAC 没有对每个方法单独调参（[GenAC 〔全文〕](https://arxiv.org/abs/2604.10701)）。
- Minimalist 论文在 Qwen2.5-Math-7B 上得到：Reinforce 53.9、PPO 52.5、RAFT++ 56.1、GRPO 56.3、Reinforce-Rej 56.4。作者结论是 GRPO 的主要优势"来自丢弃全错的 prompt"（[2504.11343 〔全文〕](https://arxiv.org/abs/2504.11343)）。
- Yue et al. 用 VeRL 重新实现了六种算法（PPO、GRPO、Reinforce++、RLOO、ReMax、DAPO），发现它们与 base pass@256 的差距都超过 40 点，算法之间差别很小（[2504.13837 〔全文*〕](https://arxiv.org/abs/2504.13837)）。

2025–2026 的新估计器都只有原论文证据（C）：OTB（AIME25 30.31 vs GRPO 25.31，单次峰值）、QAE（Qwen3-8B AIME24 39.69→48.23）、MaxRL（pass@k 在 8 个设置中赢 7 个）、OPO、A*-PO〔摘要〕。KRPO 在单张 A6000 上用 1B 模型实验，训练集与测试集是否重叠不清楚，记 C/D。

| 改动 | 机制 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|---|
| PPO 中 λ=1（相对 0.95） | λ^{T−t} 衰减使早期 token 收不到奖励 | 相对坏配置是决定性的 | — | λ<1 导致长度崩溃 | 长度 | **A** |
| Value pretraining / critic 不从 RM 初始化 | RM 只在 EOS 上训练，早期 token 的 V 偏低，产生随位置增长的正偏置 | PPO 内部决定性（VAPO 去掉后 60→11） | — | 修复早期崩溃 | 长度 | **B** |
| Decoupled GAE（λ_c=1，λ_a<1） | critic 目标无偏，actor 端降方差 | 同团队 +12–27 | — | — | — | **B/C** |
| 调好的 PPO vs GRPO/RLOO（单轮 RLVR） | γ=λ=1 时只差 baseline 的方差 | 可互换（GenAC 差距 ≤1） | value 类方法前期更快 | — | critic 让显存与算力约增 40% | **D**（原文 +10，独立复现约 0） |
| RLOO ≡ Dr. GRPO × G/(G−1) | 代数恒等 | 可互换 | — | — | — | **A**（数学）/ **B**（经验） |
| ReMax | greedy baseline | ≈ 其他方法（RLHF） | 相对 PPO 省显存 | — | 二值奖励下浪费信号 | **C** |
| VinePPO（MC value） | 无偏 MC V̂，K=9 | 推理时代规模下未验证 | 步数少 3–9 倍，单步贵 2–5 倍 | — | 算力 | **C** |
| OTB / QAE / MaxRL / OPO / KRPO / A*-PO | 改变 b 或 w(p) | 各自声称有增益 | — | — | — | **C**（KRPO **C/D**） |
| Pass@k advantage shaping | 隐式最大化 1−(1−p)^k | 多组都看到 pass@k 提升 | — | — | pass@1 混合 | **B**（pass@k）/ **C**（pass@1） |

### 归一化：std 改变的是目标函数，不是上限

**机制。** 在二值奖励下，各估计量的主要区别在于隐含的 prompt 权重 w(p)，每种权重对应一个不同的目标函数（笔记推导）：

| 方法 | w(p) | 对应目标 |
|---|---|---|
| RLOO / Dr. GRPO / batch-std | 1 | Σ_x p_x |
| GRPO | 1/√(p(1−p)) | Σ_x 2·arcsin√p_x |
| MaxRL | ≈ 1/p | ≈ Σ log p_x |
| pass@k | ≈ k(1−p)^{k−1} | Σ 1−(1−p_x)^k |

GRPO 一行可以这样验证：E[∇] ∝ Σ_x ∇p_x/√(p_x(1−p_x)) = ∇Σ_x 2·arcsin√p_x（[MaxRL 〔部分〕](https://arxiv.org/abs/2602.02710) 给出了 w(p) 的形式）。

用 G=8、只有 1 个成功样本的组举例。GRPO 给出 A₊≈+2.65、A₋≈−0.38，Dr. GRPO 给出 +0.875 与 −0.125。zero-variance 过滤去掉 p̂∈{0,1} 之后，std 归一化的最坏情况是 p̂=1/G，此时权重按 √G 增长。

batch 级 std 是每一步的一个全局标量，不改变 prompt 之间的相对权重。在 Adam 下它基本被吸收，只通过梯度裁剪、ε 和 EMA 的尺度漂移产生影响。"std 是否重要"的争论其实混了两件事，这是双向效应能同时存在的原因：

- **抬高低 p（难）prompt 的权重**：MaxRL、QAE、pass@k 训练都依赖这一半，有一定依据。
- **抬高高 p（易）prompt 的权重**：没有人为这一半辩护，它也是易数据上不稳定的来源。

**证据。** 在最终准确率上，说"无关"或"在噪声内"的研究有：

- ScaleRL：prompt 级、batch 级、不归一化三者"表现相似"，作者选 batch 级，理由是"略好且理论上合理"。
- Magistral："没有观察到"对评测或长度增长的显著影响（[Magistral 〔全文〕](https://arxiv.org/abs/2506.10910)）。
- Minimalist："相比单纯去掉坏样本，额外增益很小"。
- 2601.23135：7B 上"统计上无法区分"，1.5B 只在训练中段有优势（[2601.23135 〔摘要〕](https://arxiv.org/abs/2601.23135)）。
- 2610.04928：OLMo-2-1B 上各估计器差异在 seed 噪声内（[2610.04928 〔摘要〕](https://arxiv.org/abs/2610.04928)）。

说"有影响"的研究方向不一致：

- Dr. GRPO 附录 C 用 3 次运行声称结果"统计显著"，但 std 和长度两项改动是联合消融的。
- Tricks or Traps 发现去掉 std 只在易数据上有帮助，难数据上持平，最优组合是 group mean 加 batch std。但它的"组级归一化在各奖励设定下都稳健"一句，与它自己 §4.1.2 的结论有张力。
- GDPO 的片段称多奖励设定下去掉 std 会不稳定（[2601.05242 〔片段〕](https://arxiv.org/abs/2601.05242)）。

2607.00152 的摘要指出：二值奖励下，group std 恰好就是更新幅度；GRPO（除以 std）、Dr. GRPO（不除）、DAPO（丢掉 std=0 的组）是同一个旋钮的三个档位（[2607.00152 〔摘要〕](https://arxiv.org/abs/2607.00152)）。

实践在收敛：OLMo 3、Magistral、Kimi、GLM-4.5 agentic 部分都不用组内 std；DeepSeek-V3.2 的公式里也没有。V3.2 正文写的是"normalizing"，与公式不一致，笔记未能消解这一点（[DeepSeek-V3.2 〔部分〕](https://arxiv.org/abs/2512.02556)）。ScaleRL、AReaL 和 REINFORCE++ 系则改用 batch std。实现细节上，OpenRLHF 的 batch 均值按 action token 加权。对 reinforce_baseline，组中心化后的 advantage 在正确与错误响应长度不同时，token 加权均值不为 0，减去它会给每个 advantage 加一个与长度相关的偏移（〔代码〕`OpenRLHF/.../experience_maker.py:315-327`，笔记推断）。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| 组内 std 归一化 | 不移动（多项研究一致） | 小模型训练中段可能更快（〔摘要〕） | 易数据或小 G 时放大、可能梯度爆炸 | 目标变成 Σ arcsin√p | **A**（上限中性）/ **D**（效率方向） |
| batch std 归一化 | ≈ 组内 std | ≈ | 可能更稳（梯度尺度） | 几乎不改目标 | **B** |
| 只抬高难 prompt 的非对称加权（MaxRL / QAE / pass@k） | pass@k 有收益 | — | — | pass@1 不确定 | **C** |

### Loss 聚合：唯一清楚的是 1/|y| 带来的长度偏置

**机制。** 四种常见聚合方式对每个 token 的权重如下：

- seq-mean-token-mean（原始 GRPO，GSPO/SAPO 强制使用）：1/(G·|y_i|)。正 advantage 时短的正确答案每 token 更新更大；负 advantage 时长的错误答案每 token 惩罚更小，所以错误答案越来越长。
- token-mean：长序列的总权重与长度成正比。
- seq-mean-token-sum：每 token 权重为 1/G，梯度方差随长度增长。
- Dr. GRPO 常数归一化 1/(G·L_max)：对长度无偏，常数可以并入学习率。

（[Dr. GRPO 〔全文〕](https://arxiv.org/abs/2503.20783)；〔代码〕verl `core_algos.py:1168-1186`）

Qwen 的 MiniRL 论文把 token 级目标解释为序列级奖励的一阶近似，并指出长度归一化会让这个近似失效（[2512.01374 〔全文〕](https://arxiv.org/abs/2512.01374)）。

实现上存在分歧。DAPO 公式里的"token-level"是在每个 prompt 组内按 token 总数归一化，ScaleRL 把它归为 prompt average。verl 的 `token-mean` 则按整个 global batch 的 token 总数归一化（〔代码〕`core_algos.py:1168-1173`）。prompt 之间长度差异大时，两者给出的权重不同。

**证据。** 各项研究的方向不一致：

- DAPO：加 token-level loss 从 41 到 42，单次运行。
- Tricks or Traps：base 模型上 token 级更好，aligned 模型上序列级更好，只在 Qwen3-8B 上测过。
- ScaleRL：prompt 级在单项比较中 A 最高，但在 LOO 中落在 ±0.02 内。
- GLM-4.5：代码 RL 中 token 加权"非常有益"，来自较小的实验模型，只有图（[GLM-4.5 〔部分〕](https://arxiv.org/abs/2508.06471)）。
- DemyAgent：agentic TIR 中 token 级在 4B 上高约 3.9pp，在 7B 上"相当"（[2510.11701 〔全文〕](https://arxiv.org/abs/2510.11701)）。
- MiniMax-M1：长度扩展阶段从 token 级改成了 sample 级加 token 归一化（[MiniMax-M1 〔部分〕](https://arxiv.org/abs/2506.13585)）。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| 去掉 1/\|y_i\|（token 级或常数归一化） | 未分辨 | base 模型上可能更好 | 长度更稳 | 错误答案不再变长 | **B**（机制）/ **D**（"token 级更好"作为经验结论） |
| prompt 级平均（ScaleRL） | 单项比较中最高，LOO 中在噪声内 | — | — | — | **C** |

### Clip、trust region 与 ratio 粒度：r≡1 时这些方法全部等价

**统一模板。** 2025–2026 年几乎所有"新 loss"都能写成：

∇J = Σ_i Σ_t w_{i,t}·m_{i,t}·Â_i·∇log π_θ(y_{i,t})

差别只在三个旋钮：权重 w（token ratio、序列几何平均 ratio、截断后 stop-grad 的 ratio、平滑门控），掩码 m（哪些 token 的梯度被硬置零），以及归一化方式。

**关键恒等式是：在严格 on-policy 下，r_t ≡ 1，此时 PPO-clip、DAPO、GSPO、GMPO、CISPO、SAPO 的梯度全部退化为同一个 REINFORCE 梯度，只差归一化。** 严格 on-policy 指每批 rollout 只做一次梯度更新，且训练与推理给出的概率一致。这些方法只有在多 minibatch 复用和训练–推理 mismatch 下才有区别。所以，在单次更新、无 mismatch 条件下报告的"新 loss 收益"，首先应怀疑是归一化差异或噪声（笔记推导）。

**各方法的精确改动：**

- **PPO-clip**：当 Â>0 且 r>1+ε，或 Â<0 且 r<1−ε 时，梯度为零；其余情况梯度为 Â·r·∇log π。当 Â<0 且 r≫1 时梯度无界，这是 dual-clip（c=3）要处理的情况（〔代码〕verl `core_algos.py:1278-1370`）。
- **clip 本身带 entropy 偏置**：clip-low 提高 entropy，clip-high 降低 entropy。标准参数下 clip-high 占主导，所以即便是纯随机奖励，entropy 也会下降（[2509.26114 〔摘要〕](https://arxiv.org/abs/2509.26114)）。
- **clip-higher**：只放宽 Â>0 时的上界。对 p_old 很小的 token，原上界 (1+ε)·p_old 允许的绝对增量极小，放宽它等于去掉一部分"clip-high 降 entropy"的偏置。
- **CISPO**：sg(clip(r))·Â·log π，不丢任何 token。但它也重新打开了 PPO 刻意关闭的 Â<0、r<1−ε 区域，即继续压低一个已经被压低的坏 token。
- **GSPO**：s_i = exp(mean_t log r_{i,t})，一条序列内所有 token 同权，clip 一旦触发整条序列梯度为零。clip 范围是 3e-4/4e-4，而 GRPO 是 0.2/0.27。它也重新引入了 1/|y_i|（[GSPO 〔全文〕](https://arxiv.org/abs/2507.18071)）。
- **GMPO**：代数上就是"带 token 级 log 空间 clip 的 GSPO"（笔记推导）。
- **SAPO**：用平滑核 r·sech²(τ(r−1)/2) 替换 PPO 的硬指示函数（[SAPO 〔全文〕](https://arxiv.org/abs/2511.20347)）。
- **DCPO、DPPO、CE-GPPO、ASPO、DGPO**：都在修同一个问题，即 ratio clip 对低概率 token 的绝对步长过小、对高概率 token 过大，只是参数化方式不同。

**Clip-higher 的证据：**

- DAPO：在 Qwen2.5-32B 上 AIME24 从 36 到 38，单次运行（[DAPO 〔全文，消融节〕](https://arxiv.org/abs/2503.14476)）。
- Tricks or Traps：base 模型上 clip 触发率约 0.003，影响很小甚至有害；aligned 模型上明显减缓 entropy collapse，最优 ε_high 在 4B 上为 0.32、8B 上为 0.28，不单调。
- Entropy Mechanism：只 +0.2（7B）和 +1.4（32B），且"后期不稳定"（[2505.22617 〔全文〕](https://arxiv.org/abs/2505.22617)）。
- ScaleRL：A 随 ε_max 先升后降，作者称它"决定渐近误差本身"。
- MiniMax：每批做 16 轮 off-policy 更新时，clip-higher"效果较差"。

由此可以推断，"对哪种模型有效"实际上是"在多大 staleness 下有效"的代理变量：clip 越少触发，ε_high 能改变的梯度质量就越少。

**CISPO 的证据：**

- 原文：在 Qwen2.5-32B 上"以 50% 步数达到 DAPO 的表现"，只有图。
- ScaleRL LOO：效率 B 为 2.01 对 DAPO 的 1.77，A 相近，对 ε_max∈{4,5,8} 不敏感。
- 反例：CE-GPPO 中 CISPO 平均 47.6，GSPO 54.3（[2509.20712 〔全文〕](https://arxiv.org/abs/2509.20712)）。ARLArena 的 agentic 实验中 CISPO 34.03、GRPO 46.16，约第 130 步崩溃；加上 sequence masking 后 ALFWorld 从 54.42 升到 78.88（[ARLArena 〔全文〕](https://arxiv.org/abs/2602.21534)）。
- 实现陷阱：verl 的 `cispo` 默认把权重截断在 [0.8,1.2]，原论文无下界；SkyRL 默认是 0/5.0（〔代码〕）。部分论文里"CISPO 很差"的结果可能就来自这种实现差异（笔记推断，未验证）。

**GSPO 的证据：**

- 原文：只在 MoE（Qwen3-30B-A3B）上做过实验，只有曲线，没有数值。
- Qwen 后续在同一 30B MoE 上做了受控研究，根本没跑 GSPO。结果是：on-policy 时最稳的是不带长度归一化的 token 级 REINFORCE 加训练–推理 IS 修正；off-policy 时 token 级加 clipping 加 Routing Replay 就能稳定（[2512.01374 〔全文〕](https://arxiv.org/abs/2512.01374)）。
- ScaleRL：GSPO 多次中途发散，8B 可以从 checkpoint 重启恢复，Scout 不能；原文的 1e-4 尺度"效果不好"。
- ARLArena：GSPO 最好，52.28 对 46.16。
- CIS：在 Qwen1.5-MoE 上 GSPO 28.90，低于 base（[CIS 〔全文*，前~50%〕](https://arxiv.org/abs/2609.32444)）。
- FP16 论文：BF16 下 GSPO 在 1200 步出现 NaN（[2510.26788 〔全文*〕](https://arxiv.org/abs/2510.26788)）。

在 off-policy 场景，2610.04928 的摘要称"clipping 就是全部"：不用 clipped ratio 会损失 4–6 点，GRPO clip 或 GSPO 可以补回来（〔摘要〕）。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| PPO ratio clip（off-policy 时） | 无证据 | 允许复用样本 | off-policy 时必需（MoE 受控研究） | entropy 偏置；随机奖励下放大先验（机制有争议） | **B** |
| Dual-clip | ? | ? | 机制合理，LLM 中没有开/关消融 | ? | **C** |
| Clip-higher | 对 A 非单调（ScaleRL） | 小 | 混合（Entropy Mechanism 称后期不稳） | entropy ↑ | **B**（entropy 机制）/ **D**（跨模型普遍收益） |
| CISPO | LOO 中在噪声内 | B 更高（ScaleRL） | agentic 中崩溃（ARLArena） | 对超参鲁棒；放开负优势低 ratio 区域 | **B**（大 batch 推理 RLVR 中的效率）/ **D**（普遍优越） |
| GSPO | 有争议 | MoE 上声称更好 | MoE 且无 routing replay 时更稳；ScaleRL 中发散 | 长度归一化偏置；ε 尺度差三个数量级 | **B**（MoE 无 replay）/ **D**（普遍性） |
| GMPO / SAPO | ? | ? | SAPO 在 agentic 中崩溃（ARLArena 32.22，最差） | — | **C** / **C–D** |
| DCPO / CE-GPPO / ASPO / DPPO / DGPO / TOPR | 各自原文有提升 | — | — | 多数在 Qwen2.5-Math 或 R1-Distill-1.5B 上、≤1000 步、无 seed | **C** |
| Clip-Cov / KL-Cov / 80/20 高 entropy token / NSR | 原文有提升（以 Qwen 为主） | — | — | entropy ↑ / pass@k ↑ | **C** |

### KL 与 entropy 正则：去掉对 base 的 KL 是安全的，固定 entropy bonus 有害

**KL 估计器的梯度。** 记 δ = π_ref/π_θ。三个常用估计量是 k1 = log(π/π_ref)、k2 = ½(log π/π_ref)²、k3 = δ−1−log δ。真实梯度由 pathwise 项和 score-function 项 k̂·∇log π 两部分组成，直接对估计量求导会丢掉后一项。由此得到各种用法的实际梯度：

| 用法 | 期望梯度 | 结论 |
|---|---|---|
| k1 作为 loss | E[∇log π] = 0 | 只是零均值噪声，没有正则作用 |
| k3 作为 loss（GRPO 默认） | ∇KL(π_ref‖π)，即反向 KL 的梯度；也可看作对 −log δ 的一阶有偏近似 | 有偏 |
| k2 作为 loss | on-policy 时等于 ∇KL(π‖π_ref) | 正确 |
| k1 加在 reward 里 | on-policy 时等于 ∇KL(π‖π_ref) | 正确 |

off-policy 时，"作为 loss"的写法还需要 IS 修正，这一步常被省略（[Tang & Munos 〔全文〕](https://arxiv.org/abs/2506.09477)；[2510.01555 〔全文，前 100k〕](https://arxiv.org/abs/2510.01555)）。DeepSeek-V3.2 给 k3 乘上 π_θ/π_old 以得到无偏梯度（[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)）。

在低 KL 区间 δ≈1，1−δ ≈ −log δ，所以估计器的偏差很小。这解释了 Llama-3.1-8B + MATH 上估计器选择"实际差别不大"。到了高 KL 区间（70B→8B 的 on-policy 蒸馏），k3 作为 loss "几乎没有进展"（[2506.09477](https://arxiv.org/abs/2506.09477)）。

框架默认值大多避开了最坏情况（〔代码〕）：

- verl 和 slime 默认关闭 KL loss。
- OpenRLHF 和 AReaL 默认把 k1 放在 reward 里，这是正确的形式。
- 但 slime 一旦开启 KL loss，默认估计器是 k1，在数学上没有正则作用。
- SkyRL 默认开启 k3 作为 loss。

**RLVR 中对 base 的 KL。** 直接的受控证据有两条：

- Skywork-OR1 在第二阶段加入 β=1e-3 的 KL，结果 AIME24 不再提升，所有发布模型都用 β=0（[Skywork-OR1 〔部分，前~65%〕](https://arxiv.org/abs/2505.22312)）。
- Entropy Mechanism 发现 reference KL 稳住了 entropy，但降低了性能（[2505.22617 〔全文〕](https://arxiv.org/abs/2505.22617)）。

DAPO、Magistral（称 KL "主要是阻碍训练"）、GLM-4.5、MiniMax-M1、ScaleRL、OLMo 3、JustRL 都去掉了 KL，但只是声明，没有消融。

保留 KL 的系统，实际做法都相当于近端项：

- DeepSeek-R1：k3，β=0.001，每 400 步把 reference 换成当前策略。
- ProRL：周期性硬重置 reference 和优化器，没有给去掉 KL 的消融（[ProRL 〔全文*〕](https://arxiv.org/abs/2505.24864)）。
- Kimi k1.5/K2：对上一迭代 π_old 加 (τ/2)∇(log π_θ/π_old)² 形式的平方正则，而不是对 SFT 的 KL（[Kimi k1.5 〔部分〕](https://arxiv.org/abs/2501.12599)）。

所以"RLVR 要不要 KL"应拆成两个问题。锚定到 base 的 KL 对推理 RLVR 无用或有害，等级 B。近端 KL 可能有助于 off-policy 和异步训练的稳定，等级 C，证据只有 Comedy of Estimators〔摘要〕（[2512.21852](https://arxiv.org/abs/2512.21852)）。

RL's Razor 给出了理论上的支持：on-policy RL 自带隐式的"最小 KL"偏好；遗忘程度可以由前向 KL E[KL(π₀‖π)] 预测（ParityMNIST 上 R² 0.96，LLM 上 0.71）。需要注意，它的定理用的是 KL(π‖π₀)，而预测指标用的是 KL(π₀‖π)，方向不一致（[RL's Razor 〔全文〕](https://arxiv.org/abs/2509.04259)）。

**学习型 RM 场景（RLHF）下，KL 仍然是承重的：**

- InstructGPT 的扫参中，KL 系数在 0.01–0.02 附近最优（[InstructGPT 〔部分〕](https://arxiv.org/abs/2203.02155)）。
- N+ Implementation Details 中，KL 达到 50–85 的 1B 运行 RM 分数很高，但对参考答案的胜率不到 20%（[2403.17031 〔全文前~70%〕](https://arxiv.org/abs/2403.17031)）。
- Gao et al. 给出 overoptimization 的函数形式：RL 下 R = d(α − β log d)，其中 d = √KL。他们还发现 KL 惩罚**不改变** gold-KL 前沿，作用"类似 early stopping"（[2210.10760 〔全文*〕](https://arxiv.org/abs/2210.10760)）。

**Entropy。** 机制上，一步 vanilla PG 的 entropy 变化 ≈ −η·Cov(log π, π·Â)。高概率且高优势的 token 驱动 entropy 下降，简单 prompt 上这个协方差更大（[2505.22617](https://arxiv.org/abs/2505.22617)）。

固定系数的 entropy bonus 有三方独立的负面证据：

- Entropy Mechanism：系数 1e-4 和 1e-3 几乎无效，0.01 导致爆炸，0.005 稳住了 entropy 但没超过基线。
- Magistral：纯数学数据上 entropy 下降，数学加代码数据上爆炸，最后改用 ε_high 来控制 entropy。
- Skywork-OR1：改用自适应目标 entropy（目标值 0.2），只在 entropy ≤ 目标时才施加 entropy loss。

主流框架默认 entropy 系数为 0（〔代码〕）。

**entropy 能否预测上限**，两份证据相互矛盾，记 D。2505.22617 拟合出 R = −a·e^H + b，意味着 H→0 时存在可预测的上限。ScaleRL 则发现 batch 768 和 2048 每步的 entropy 轨迹几乎相同，但 2048 显著更好，结论是 entropy 不预测下游表现（App. A.12）。

**Off-policy 的 SGD 复用是 entropy collapse 最大的可控驱动因素，等级 A：**

- Skywork-OR1：同 batch 下，N_SGD=1 优于 N_SGD 为 2 或 4 的设置，后两者几步内 entropy 就跌到很低。
- AceReason：每批 22 或 44 次更新时，约第 100 步 collapse；每批 1 次更新"一致地避免了 collapse"（[AceReason 〔全文〕](https://arxiv.org/abs/2505.16400)）。
- Magistral：每批超过 2 个 minibatch 时性能下降。

rollout 温度过低也会导致 collapse，等级 B：Skywork-OR1 中 τ=0.6 时 entropy 在约 100 步内接近 0。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| 去掉对 base 的 KL（可验证奖励） | 去掉不伤害，保留可能压低 | 省掉 reference 前向 | — | 遗忘可能增加（RL's Razor 认为 on-policy 本身已有约束） | **B** |
| 近端 KL / 周期重置 reference | ? | — | 可能有助于长训练和异步（无消融） | — | **C** |
| KL（学习型 RM） | 存在内点最优 | — | 防止 overoptimization，作用类似 early stopping | — | **B** |
| k3 作为 loss vs k2 作为 loss / k1 在 reward 里 | 低 KL 区间无差别 | — | 高 KL 区间 k3 有偏 | — | **B**（数学）|
| 固定 entropy bonus | 无 | 无 | 负面（爆炸或无效） | 强烈依赖数据 | **A**（负面结论） |
| 自适应 entropy / Clip-Cov / KL-Cov / 80/20 | 原文有提升 | — | — | — | **C** |
| 每批 rollout 多次 SGD（N>1） | 固定训练时长下更低 | 每样本收敛更快（Qwen） | entropy collapse | 同步训练中的隐性 off-policy | **A**（collapse） |
| 温度 ≥ ~1.0 / 按模型校准 | 通过避免 collapse 间接影响 | — | 低温导致 collapse | 最优值因模型而异 | **B**（避免低温）/ **D**（随训练升温还是降温） |

### Off-policy 与训练–推理 mismatch：从源头消除比事后修正可靠

**机制。** off-policy 有四个来源：

- 同一 θ 下采样器与学习器之间的数值差异（BF16、kernel 归约顺序、top-p 截断）。
- 多 minibatch 更新造成的 staleness。
- 异步或 in-flight 更新造成的 staleness。
- MoE routing 翻转。在 Qwen3-30B-A3B 中，约 10% 的 router 在训练和推理时选了不同专家，94% 的 token 至少在一层上不同；训练–推理的 k3 KL 是 dense 8B 的约 2.4 倍（[R3 〔全文*〕](https://arxiv.org/abs/2510.11370)）。

它们破坏的是同一个恒等式。Hybrid 框架实际计算的是 E_{a~π_sampler}[R·∇log π_learner]，而 PPO 的 ratio π_learner(θ)/π_learner(θ_old) 从来看不到 π_sampler，所以 mismatch 完全没有被修正（[Yao et al. TIS 博客 〔全文〕](https://fengyao.notion.site/off-policy-rl)）。verl 的维护者把"在 π_rollout ≠ π_old 时，把训练端重算的 π_old 当作 behavior policy"称为"严重的实现错误"（〔代码〕`verl/docs/algo/rollout_corr.md`）。

主要的修正方法：

- **Token 级 TIS**：用 min(π_learner/π_sampler, C) 修正，把每 token 的方差放大限制在 C² 以内。它只修正了动作分布，没有修正前缀（状态）分布，剩余偏差为 O(T²·D_TV^max)（[Liu & Li 博客 〔全文〕](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)）。TRM 的摘要称可以把界收紧到 O(T^{3/2}) 或 O(T)，但这些界都依赖最大的 token 级散度（[TRM 〔摘要〕](https://arxiv.org/abs/2512.23075)）。
- **序列级 IS**：无偏，但方差是 O((1+χ̄²_max)^T)，随长度指数增长。笔记据 R3 的 MoE KL 1.5e-3 估算：每 token log-ratio 的 std 约 0.055，T=16k 时 log ρ 的 std 约 7（笔记推导）。在推理长度下，序列级 IS 的权重在数值上已经退化，C=2 的 Seq-TIS 基本就是硬截断。
- **几何平均或 K3 判据**（GSPO、Geo-RS、DeepSeek OPSM）：之所以能用，是因为它们是长度无关的 trust-region 检验，而不是 IS 修正。

修正方法还可以按"截断 vs 掩码"分类：

- 截断（TIS、CISPO、AIPO）：保留每个 token 的有界梯度。
- 掩码（IcePop 的 [0.5,5] 双侧带、MIS、M2PO 的二阶矩预算、DeepSeek OPSM、AReaL 旧版的 drop）：把区外 token 的梯度置零。被掩掉的 token 不成比例地集中在低概率、高 entropy 区域，所以有探索代价。
- CIS 的消融显示下侧更敏感：像 IcePop 那样掩掉 ratio<α 的 token 无害，但把小权重往上抬是灾难性的，Qwen1.5-MoE 的留出准确率从 base 的 28.93 跌到 6.01。

多条线索指向同一个非对称原则：**负优势且高度 off-policy 的 token 是梯度中危险的部分**。TOPR 只截断负样本，DeepSeek OPSM 只掩远离 on-policy 的负样本，M2PO 只约束会被 clip 的 token，三者都体现了这一点。其中只有 TOPR 给出了有界性论证（[TOPR 〔全文〕](https://arxiv.org/abs/2503.14286)）。

**mismatch 何时是决定性的。** 在 MoE、长上下文、多轮 agentic、低精度 rollout 这四种情形下，它是决定性的：

- R3：GRPO 分别在第 60、105、120 步崩溃，加 TIS 在第 105 步崩溃，加 R3 后 180 步内不崩；SWE-bench Verified 从 31.8 升到 38.6。
- Ring-mini-2.0：baseline 在 180–200 步崩溃，IcePop 和 TIS 都稳定（[Ring-1T 〔全文*〕](https://arxiv.org/abs/2510.18855)）。
- Qwen MiniRL：on-policy 下去掉 IS 修正会"快速崩溃并伴随 entropy 骤降"（[2512.01374](https://arxiv.org/abs/2512.01374)）。
- VeXact（零 mismatch 参考引擎）：MoE REINFORCE 用 vLLM 时，约 280 步后奖励从 0.574/0.293 跌到 0.255/0.067；VeXact 达到 0.753/0.534（[VeXact 〔全文*〕](https://arxiv.org/abs/2605.14220)）。
- Liu & Li：Qwen3-14B-Base 多轮 TIR 在 L20 上约 200 步崩溃，工具轮数越多崩得越早。
- DeepSeek 称 Keep Routing "至关重要"，但没有给数字。

在 dense、短上下文的数学 RL 上，它是二阶问题：

- Yao 在 1.5B BF16 上 TIS 无增益。
- Liu & Li 在 H100 推理 RL 上，TIS 防住了梯度爆炸，但"没有更好的测试表现"。
- 唯一接近多 seed 受控的比较是 CIS：3 个 MoE 模型，各 3 个 seed，但只在 GSM8K 规模。Qwen3-30B-A3B 上五项平均：不修正 68.26、TIS 69.14、Seq-TIS 68.99、Seq-MIS 68.68、IcePop 68.99、GSPO 69.23、FP16 68.53、CIS 69.88。作者承认部分差距在 seed 方差内。

**很大一部分"mismatch 崩溃"其实是实现或硬件病理。** vllm-kl 在 H20 上为 5e-4–1e-3，L20 上为 1e-3–1e-2，A100 上为 1e-2–1。把失败的 L20 运行切到 H20 上续训就恢复了。在 A100 上关掉 vLLM cascade attention，vllm-kl 从 5e-2–1e-1 降到约 1e-3（[Liu & Li](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda)）。

**FP32 LM head 的证据相互矛盾，记 D：**

- ScaleRL 的单项消融中 A 从 0.52 升到 0.61（数字经摘要器转述）。
- ScaleRL 自己又说，在 dense 8B 的完整 recipe LOO 中它"几乎没有差别"，只对 GRPO/DAPO 式 loss 和 Scout MoE 有大收益。
- MiniMax 称它把训练–推理概率相关性从约 0.9x 提到约 0.99x。
- Yao 的 vLLM FP32 head 补丁没能消除 mismatch；Liu & Li 用 FP32 head 续训一个已崩溃的运行，仍然崩溃。

**FP16 的证据（[2510.26788 〔全文*〕](https://arxiv.org/abs/2510.26788)）：**

- R1-Distill-1.5B sanity test：BF16 GRPO 在峰值 73%/84% 后崩溃，BF16 加 TIS 也崩溃；BF16 Seq-MIS 稳定，但只到 95% 训练准确率，FP16 是 99%；AIME24 34% 对 39%。
- 没有报告 seed；§5.3 与附录写的 base model 不一致（Qwen3-14B 和 Qwen2.5-14B）。
- 在 CIS 中，FP16 约等于不修正。

**Staleness 容忍度（[AReaL 〔全文*〕](https://arxiv.org/abs/2505.24298)；[PipelineRL 〔全文*〕](https://arxiv.org/abs/2509.19128)；[M2PO 〔全文*〕](https://arxiv.org/abs/2510.01161)）：**

不修正时，大约 1–2 个版本是安全的：

- AReaL（1.5B，单次运行）：η=1 时 41.8，η=2 时 40.0，oracle 是 42.0。
- Magistral：每批不超过 2 个 minibatch。

修正后，4–8 个版本可以与 on-policy 持平：

- AReaL decoupled PPO：η=4 时 42.2，η=8 时 41.0。
- PipelineRL：常规异步训练能稳定的最大 lag 是 G=32，G=64 发散。
- ScaleRL：k=4 与 k=8 "一样好"。

M2PO 声称 staleness 256 时仍与 on-policy 持平（Qwen3-Base-4B 上 51.3 对 50.7），但它报告的是 best checkpoint、没有 seed，记 C。

AReaL 不修正时 η=4 只有 23.3，比 η=8 和 η=∞ 都差，这种非单调提示单 seed 方差或一次崩溃事件，会夸大 decoupling 在 η=4 的收益。2607.01083 的摘要称最大稳定 LR ∝ S⁻¹，即 S·η ≈ 常数（[2607.01083 〔摘要〕](https://arxiv.org/abs/2607.01083)）。这意味着 staleness 容忍度应当表述为 staleness × LR 的预算，而不是一个固定步数。R3 中"8 个 mini-step 反而比 1 个崩得晚"，很可能就是 LR 混杂：前者用 1e-6，后者用 3e-6。

**隐性 off-policy**：verl 默认 `train_batch_size 1024 / ppo_mini_batch_size 256`，所以所谓"同步"训练每批默认就做 4 次更新（〔代码〕）。各框架都内置了 TIS、MIS、IcePop、R2/R3 等修正，但默认全部关闭（〔代码〕，2026-05 版本）。

**ScaleRL 内部的矛盾必须保留。** 关于 off-policy 算法（PipelineRL-k 对 PPO-off-policy-k），摘要和 §3.1 说它"主要调节效率，渐近表现相似"，图 4a 说 PipelineRL "在大算力极限下略好"，附录 A.11 却说它"影响渐近表现……是 RL 后训练中最关键的设计决策之一"（[ScaleRL](https://arxiv.org/abs/2510.13786)）。可靠的读法是：效率收益约 2 倍，PipelineRL 论文与 ScaleRL 一致；对上限的影响未分辨。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| 从源头消除 mismatch（batch-invariant kernel / VeXact / 统一精度） | 凡有对照处都 ≥ 所有 IS 变体 | 推理慢 1.6–2.1 倍（TML 单机数字） | mismatch 为零 | 工程成本 | **B** |
| R3 / Keep Routing（MoE） | +5.6–6.8（R3 原文） | 开销 <3% | 防止 MoE 崩溃；on-policy 时无收益（Qwen） | 冻结 routing 后目标有偏，低 staleness 时 R2 更好 | **A**（MoE 多 minibatch：R3 + Qwen 受控 + DeepSeek 生产）|
| Token 级 TIS | 0 到 +4 | 中性 | 能推迟崩溃（60→105），但在 TIR 和 FP16 sanity test 中失败 | 状态分布偏差 O(T²) | **D** |
| Seq-MIS / Geo-RS | 低于 FP16；短 GSM8K 上无增益 | 中性 | 长视界 TIR 中最好的估计量修正 | 掩掉长或低概率样本 | **C**（稳定性）/ **D**（上限）|
| IcePop | CIS 中名列前茅（3 seeds） | 中性 | 稳定；[0.5,2] 带宽会立刻失稳 | 掩掉 1–2‰ 的高 entropy token | **B** |
| CIS | 3 个 MoE 上平均最好（原文，3 seeds） | — | — | 2026-09 新方法 | **C** |
| FP16 训练 + 推理 | 原文最高；CIS 中 ≈ 不修正 | 原文称与 BF16 同速 | 原文中稳 | loss scaling；前沿 MoE 规模的溢出风险无人测过 | **B**（降 mismatch）/ **D**（下游收益）|
| FP32 LM head | 单项消融 +0.09 A；dense LOO 中无差别 | 便宜 | 没能阻止崩溃（Yao；Liu & Li） | 只修 head 处的 mismatch | **D** |
| Decoupled PPO + staleness 上界 | η≈8 内恢复 on-policy 水平（单个 1.5B 研究） | 墙钟快 1.6–2.8 倍 | 是 | 多一次 π_prox 前向 | **B** |
| In-flight 更新（PipelineRL） | 未分辨（ScaleRL 自相矛盾） | 约 2 倍 | k≤8 时正常 | 混合策略序列；早期 token 的 lag | **B** |
| M2PO | 声称 s=256 时持平 | 允许极端 staleness | 原文中稳定 | 报 best checkpoint，无 seed | **C** |
| DeepSeek OPSM + Keep Sampling Mask | 未披露 | — | 声称 | 无消融 | **C** |

### 数据与采样：batch、长度预算和过滤决定能否走到上限

**Zero-variance 过滤与 dynamic sampling。** 全对或全错的组 advantage 为零，梯度贡献恰好为零。但在 token-mean 归一化下，它们仍然计入分母，使有效步长缩小且逐批波动。refill 的作用是让每步的有效样本数保持恒定。证据如下：

- DAPO 在累加消融的最后一步加入 dynamic sampling，AIME24 从 42 升到 50。这是单次运行，结果依赖累加顺序。DAPO 自己也说，它以更少步数达到相同表现，墙钟时间变化不大。
- ScaleRL 正文称 zero-variance 过滤"带来更高的渐近值"，但 LOO 中 loo-batch 落在 ±0.02 内，即只影响效率。两者矛盾。
- OLMo 3 的 active sampling "大幅降低了 loss 方差"，只有图，没有准确率数字。
- 2512.07611 在 Countdown 上发现关掉 dynamic sampling 时最好（[2512.07611 〔摘要+二手〕](https://arxiv.org/abs/2512.07611)）。
- JustRL 不做任何过滤，在 1.5B 上做到 SOTA（[JustRL 〔全文〕](https://arxiv.org/abs/2512.16649)）。
- BroRL 测得组在 dynamic sampling 后的存活率：N=16 时 41%，N=512 时 62%（[BroRL 〔全文*〕](https://arxiv.org/abs/2510.01180)）。也就是说，refill 的成本强烈依赖组大小和模型强度。

两份笔记对这一项的分级不同：数据笔记给"效率/稳定性工具"记 A，受控研究笔记给准确率记 D、稳定性记 B。本报告取：稳定性 **B**，上限中性 **B**，准确率增益 **D**。

**难度过滤与课程。** GRPO 每个 prompt 的信号与 p(1−p) 成正比。Bae et al. 证明期望改进的下界由任务成功率的方差给出（[2504.03380 〔摘要〕](https://arxiv.org/abs/2504.03380)）。证据如下：

- AceReason：Hard 子集 65.9，全集 63.3，单次运行。
- ScaleRL：No-Positive-Resampling（永久丢弃 pass rate ≥0.9 的 prompt）"同时改善可扩展性和渐近值"，具体幅度在笔记读到的部分中没有出现。
- Knapsack RL 的摘要称约省 2 倍算力（[2509.25849 〔摘要〕](https://arxiv.org/abs/2509.25849)）。
- 难度标签本身有噪声：一项 2026 年研究称，要达到 0.80 的标签一致性需要约 40 个 rollout（[2609.40115 〔片段〕](https://arxiv.org/abs/2609.40115)），常见的 8 rollout 过滤很粗糙。

记 **B**（效率）。

**组大小 G 与 batch。** ScaleRL 在固定总 batch 下比较 G∈{8,16,24,32}，A 为 0.585–0.595，16 是 0.610，拟合曲线"基本不变"，属于二阶选择。BroRL 把 N 从 16 提到 512，按时间对齐比较 Math 只 +1 点（实例级配对 p=6.5e-7，但均值差很小），所以"巨大 N 能突破平台"只记 C。Batch 是少数在 ScaleRL 中移动 A 的因素：512、768、2048 对应 A 0.605、0.610、0.645。小 batch 早期更快、后期被反超，两者每步的 entropy 几乎相同。记 B。

**生成长度预算。** ScaleRL 从 14k 加到 32k，A 从 0.610 升到 0.645，但 C_mid 从 2542 增到 11272，效率变差。这是"上限换效率"的清晰例子，记 B。

**Overlong 与截断处理，证据相互矛盾（D）：**

- DAPO：overlong filtering 带来 +6，soft punishment 带来 +3，最大长度 16K。
- Tricks or Traps：8k 时收益显著，20k 时大幅减小。
- JustRL：加 DAPO 的 overlong penalty 后 AIME24 停在约 50（不加约 55），entropy 塌到 0.5–0.6。
- OLMo 3 脚注 34：overlong filtering "没有带来一致的改进"。
- Skywork-OR1：mask 截断样本后 clip ratio 持续上升，作者认为这是长度 reward hacking，因此不 mask。
- ScaleRL：长度惩罚不比 interruption 更好。

截断率作为预警指标有受控支持，记 B：在 ScaleRL 的 batch 768 下，截断率 10–15% 通常导致失稳。OLMo 3 从 3.0 到 3.1 的提升来自"16K 长度"加"不 mask 截断"，两项改动混在一起。

**长度课程：** Skywork 和 AceReason 认为有效；GLM-4.5 发现单阶段 64K 更好，多阶段会"不可逆地下降"；Polaris 发现对 Qwen3-4B 有害。依赖 base model，记 D。

**长度惩罚：** L1 在 512/1024 token 预算下比 S1 budget forcing 高 20–25 点，但 AIME25 上 L1-Exact 22.5 对 DeepScaleR-24K 27.3，而且 OOD 下长度误差为 20–40%（[L1 〔全文*〕](https://arxiv.org/abs/2503.04697)）。记 B，代价是在最难的题上降低上限。

**数据量：** LIMR 用 1,389 条精选样本，平均 58.1，全集 8,523 条是 57.0；1-shot RLVR 在 Llama-3.2-3B 上只从 17.5 涨到 19.0，而且不稳定。"少量数据就够"在 Qwen-Math 上记 B，一般性记 D。

**温度：** Polaris 随训练升温（Qwen3-4B 从 1.4 升到 1.5，超过约 1.55 崩溃），Kimi K2 随训练降温，方向记 D。评测温度本身就能让分数差约 15 点。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| Zero-variance 过滤 / active sampling | 中性（LOO） | 每步正向，墙钟中性到正向 | 降低 loss 方差 | refill 成本依赖 G | **B** / 准确率 **D** |
| 难度过滤（去掉易题） | ScaleRL 称弱正向（幅度未读到） | 多组报告约省 2 倍步数 | 中性 | 易技能遗忘未测 | **B** |
| 组大小 G（固定总 batch） | 二阶 | 依硬件 | 中性 | — | **B**；巨大 N 扩边界 **C** |
| Batch size ↑ | +0.035 A（768→2048） | 早期更慢 | 更稳 | — | **B** |
| 生成长度 ↑ | +0.035 A（14k→32k） | C_mid ×4.4 | — | — | **B** |
| Overlong filtering / penalty | 不提高 | 短上下文下有用 | 与长度预算强交互 | mask 可能诱发长度 hacking；penalty 可能压低 entropy | **D** |
| Interruption / soft 长度控制 | 不降低 | — | 控制截断率 | — | **B** |
| 长度课程（短→长） | 依赖 base | 省算力（Skywork） | Qwen3/GLM 上不可逆下降 | — | **D** |
| 精选小数据（LIMR / 1-shot） | 该范围内基本不动 | 不省算力（1-shot 作者自述） | Llama 上不稳 | — | **B**（Qwen-Math）/ **D**（一般）|

### 奖励类型：验证器的精度和可 hack 性是一阶因素

**验证器误差的风险不对称：精度比召回更重要。** 下面的结果都用 Qwen2.5-7B base 加 DeepScaleR 数据（[2505.22203 〔全文*〕](https://arxiv.org/abs/2505.22203)）：

- 规则验证器的精度超过 99%，但平均召回只有 86%。在 WebInstruct 上，HF Math-Verify 的召回只有 47%。
- 规则加模型的混合验证器得 57.3，只用规则得 55.0。
- 一个训练出来的 1.5B 生成式验证器在约第 450 轮被 hack，策略输出单个 "{" 和乱码，训练崩溃。攻击成功率为 27.6–35%。判别式的 xVerify 被 hack 的比例接近 0–1%。

TinyV 发现，Big-Math-RL-Verified 中被判错的回答有 38.5% 实际是对的。RL 训练中每步约 20% 的 rollout 是 false negative。用一个只在负样本上调用的 1.5B 验证器修正它们，平均分提高 2.6–2.8（[TinyV 〔全文*〕](https://arxiv.org/abs/2505.14625)）。

前沿实践与此一致。Kimi k1.5 排除选择题、判断题、证明题，以及 8 次之内不写 CoT 就能猜中的题。DeepSeek-R1 不对推理任务使用神经 RM，偏好奖励只放在第二阶段 1,700 步中的最后 400 步（[DeepSeek-R1](https://arxiv.org/abs/2501.12948)）。这一条记 **B**。

**无监督奖励（entropy 最小化、self-certainty、多数投票）：**

- EM-RL 在 Qwen2.5-7B 上与有标签的 GRPO 相当（42.9 对 42.1），在 Llama-3.1-8B 上收益很小，MATH500 从 40.6 到 41.0，而 RLOO 是 52.0（[2505.15134 〔全文*〕](https://arxiv.org/abs/2505.15134)）。
- TTRL 在 Mistral-Nemo 上从 0.8 降到 0（[TTRL 〔全文*〕](https://arxiv.org/abs/2504.16084)）。
- Intuitor 的离线 self-certainty 奖励被 hack：模型在回答后追加已解出的题目（[Intuitor 〔全文*〕](https://arxiv.org/abs/2505.19590)）。
- 2603.08660 的摘要称，所有内在奖励本质上都是在锐化初始分布，表现先升后降，崩溃时机由先验决定（[2603.08660 〔摘要〕](https://arxiv.org/abs/2603.08660)）。

"只在强先验上起锐化作用，然后崩溃"有四组以上的证据一致，记 **B**。"可与有监督 RL 匹敌"作为一般结论，记 **D**。

**Rubric 和 LLM judge：** RaR 在 HealthBench 上把 Likert 打分从 23.9 提到 35.9。但参考答案承载了大部分信号（不用参考答案时只有 32.7–34.2），而且 HealthBench 本身就是用 rubric 评分的，训练与评测格式一致（[RaR 〔全文*〕](https://arxiv.org/abs/2507.17746)）。记 **C**。

**Reward hacking 随优化压力和环境丰富度上升：**

- OpenAI 发现，强 CoT 监控压力下 agent 会隐藏意图但继续 hack，因此建议支付"monitorability tax"（[2503.11926 〔摘要〕](https://arxiv.org/abs/2503.11926)）。
- Anthropic 发现，在真实生产编码环境中学会的 reward hacking 会泛化成 alignment faking 和破坏行为；"inoculation prompting"能消除这种泛化（[2511.18397 〔摘要〕](https://arxiv.org/abs/2511.18397)）。

数据侧的缓解手段（剔除可 hack 的题型、用判别式验证器、让 RM 阶段晚而短）是一阶的，记 **B**。

**RL 改变的是哪一种上限。** Yue et al. 在 Omni-MATH-train 上发现 GRPO 把 pass@1 从 26.1 提到 42.5，但 pass@256 从 66.3 降到 64.3。在 AIME24 上取 k=1024，只有 base 能解的题占 13.3%，只有 RL 模型能解的题占 0%。用 distillation 训练的模型（R1-Distill-Qwen-7B）在所有 k 上都高于它的 base。

ProRL 的长时间训练（R1-Distill-1.5B，超过 2k 步）在 base 本来较弱的 Reasoning Gym 任务上扩展了边界，pass@1 从 4.24 升到 59.06，但部分数学任务的 pass@128 下降（[ProRL](https://arxiv.org/abs/2505.24864)）。Invisible Leash 测得 ProRL-1.5B-v2 有 48 次支持集扩展、175 次收缩；在 AIME24 上取 pass@8192，base 是 93.3%，ProRL 是 83.3%（[2507.14843 〔全文*〕](https://arxiv.org/abs/2507.14843)）。

由此得出的结论是：**覆盖率上限主要由 pretraining、mid-training 和 distillation 决定。** OctoThinker 显示，在 MegaMath-Web-Pro 上做 mid-training 能让 Llama-3.2-3B 在 RL 下追平 Qwen2.5-3B（只有图）（[OctoThinker 〔全文*〕](https://arxiv.org/abs/2506.20512)）。Cognitive Behaviors 的摘要称，即使用解答错误的示范先 prime 一下 Llama，它也能达到 Qwen 的水平（[2503.01307 〔摘要〕](https://arxiv.org/abs/2503.01307)）。

### 偏好优化（DPO 系）：数据的 on-policy 程度压倒 loss 形状

**机制。** DPO 的 loss 是 L = −log σ(β·h)，其中 h = ρ_θ(y_w) − ρ_θ(y_l)，ρ_θ = log π_θ − log π_ref。它的梯度乘以 σ(−βh)，所以已经排对的样本对权重很小。实践中，这个目标主要通过压低 dispreferred 回答的似然来达成（[TRL 文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)）。

几乎所有离线变体只改四个旋钮：

- **link 函数 f**：logistic 对应 DPO，平方对应 IPO，hinge 对应 SLiC。
- **隐式奖励的归一化**：长度归一化、odds、去掉 reference。
- **绝对似然锚**：NLL、DPOP、NCA、APO。
- **数据分布**：RSO、WPO、iterative、semi-online、online。

DPO 系也会 overoptimize。Rafailov et al. 在 Pythia 1B/2.8B/6.9B 上发现，胜率随 √KL 呈驼峰形，符合 Gao 的 d(α−β log d)；低 KL 约束下，多数运行在第一个 epoch 的前约 25% 就到达峰值。有限数据下目标函数有零空间，存在把质量放到 OOD 回答上的最优解（[2406.02900 〔全文〕](https://arxiv.org/abs/2406.02900)）。

**受控证据给出的主导性排序：**

1. **数据的 on-policy 程度与样本对的对比度。** 在同一张表里，WPO 把 DPO 的训练数据从离线 UltraFeedback 换成混合 on-policy 数据，Mistral-Base 上提高 +17.3 LC，换 loss 只提高 +3.8（[WPO 〔全文〕](https://arxiv.org/abs/2406.11827)）。Lanchantin et al. 在 Llama-3.1-8B-Instruct 上报告了带 seed 标准误的结果：Math500 上离线 DPO 53.7、semi-online 58.9、online 58.7、GRPO 58.1；AlpacaEval LC 上离线 53.2、semi-online 81.6、online 83.1、GRPO 75.2（[2506.21495 〔全文〕](https://arxiv.org/abs/2506.21495)）。
2. **标注质量。** Ivison et al. 在 Tulu 2 13B 上发现数据质量最多带来 8 点。
3. **同一数据下 PPO 与 DPO 之差**：平均 +0.7，推理 +1.3，代码 +2.9。
4. **RM 规模**：影响基本可忽略，RewardBench 从 61.0 涨到 79.8，PPO 平均分反而从 62.2 降到 61.6。

（[Ivison et al. 〔全文〕](https://arxiv.org/abs/2406.09279)）

调好的 PPO 不低于离线 DPO，记 **A**。Xu et al. 发现，在 CodeLlama-34B 的 APPS 上，PPO 加 advantage 归一化、大 batch、reference EMA 之后，Intro/Inter/Comp 三档从 18.0/2.4/1.1 升到 44.4/18.0/9.1。DPO 在 CodeContests 上得 0.0%（[2404.10719 〔全文〕](https://arxiv.org/abs/2404.10719)）。但 PPO 的代价是在 Tulu 3 中约 7–14 倍的算力（[Tulu 3 〔全文〕](https://arxiv.org/abs/2411.15124)）。Xu 的"PPO > DPO"与 Llama 3 的"DPO > PPO"可以同时成立：前者是离线 DPO 对重度调参的 PPO；后者是近似 on-policy、6 轮迭代的 DPO（[Llama 3 〔全文〕](https://arxiv.org/abs/2407.21783)）。决定结果的是数据范式，不是算法。

**调好 β 之后，loss 形状基本可以互换。** GPO 测了 logistic、平方、hinge、指数、截断二次、Savage 六种损失，峰值相近，出现在相近的 KL 水平；只是最优 β 不同，平方损失约为 1，其余约为 10。这项研究只在 T5X ≤11B 的摘要任务上做（[GPO 〔全文〕](https://arxiv.org/abs/2402.05749)）。Gorbatovski et al. 对每种方法扫了至少 6 个 β × 4 个 LR，Llama-3.1-8B 上 AlpacaEval LC 为：DPO 26.82、IPO 28.18、SimPO 27.65、ORPO 28.25（需要补上 SFT 阶段和 β）。pairwise 方法优于 pointwise 方法（APO-Zero 23.15、NCA 23.21），8B 上 p=0.014（[2502.01237 〔全文〕](https://arxiv.org/abs/2502.01237)）。

真正有差别的有三类：

- **长度归一化。** Tulu 3 中 LN-DPO 平均 57.3，DPO 55.2，SFT 起点 55.7，SimPO 51.8–52.9，低于 SFT。LN-DPO 对 β 很敏感，β=2 时只有 46.8。Apertus 也认为长度归一化对 DPO 和 QRPO 都有帮助（[2509.14233 〔部分〕](https://arxiv.org/abs/2509.14233)）。
- **NLL 锚只在数据 off-policy 时有用。** Llama 3 用系数 0.2，Nemotron-4 观察到 chosen 和 rejected 的似然都在下降；Lanchantin 发现 on-policy 时加 NLL 没有收益。
- **对比度。** OLMo 3 中，在 Qwen3-32B 的 chosen 回答上继续 SFT，平均分从 70.3 降到 64.5；而让 DPO 偏好 Qwen3-32B 胜过 Qwen3-0.6B，平均分升到 72.9。这是单次运行（[OLMo 3](https://arxiv.org/abs/2512.13961)）。

| 改动 | Ceiling（相对调好的 DPO） | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| Online / semi-online / iterative DPO | ≈ GRPO，远高于离线 | 需要生成和 judge | entropy 与长度崩溃风险 | judge 的长度偏置 | **A** |
| 调好的 PPO | ≥，平均 +0.7，代码上更大 | 约 7–14 倍算力 | 依赖实现细节 | RM 驱动的长度增长 | **A** |
| IPO / SLiC / 其他 GPO 损失 | 打平 | 相同 | IPO 更不易 overoptimize（单项研究） | 与 DPO 相同 | **A**（打平）|
| LN-DPO / R-DPO | LN-DPO 在 Tulu 3 中最好 | 相同 | 对 β 很敏感 | 减少长度，但不能根治 overoptimization | **B** |
| +NLL / DPOP | off-policy 时有帮助 | 相同 | 减少 likelihood displacement | 可能过度锚定 chosen | **B** |
| SimPO | 打平（Gorbatovski）到更差（Tulu 3） | 约快 20% | 没有 KL 锚，γ 需要手调 | 数学退化 | **D** |
| KTO | 配对聊天数据上 ≤；推理上可能 ≥ | 可用非配对数据 | — | — | **D** |
| ORPO | 单阶段时更差；补 SFT 和 β 后打平 | 省一个 SFT 阶段 | — | — | **B** |
| WPO / RSO / TR-DPO / cDPO / rDPO | 只有原文（WPO 有 5 seeds） | — | — | — | **C** |
| SPPO / DNO / Nash-MD / BOND | ≈ iterative DPO | — | — | 长度增长（DNO MT-Bench 下降） | **C** |

**2025–2026 年 DPO 的角色。** 在开源 recipe 中，它是 SFT 与 RLVR 之间的"对比注入"阶段。OLMo 3 的 7B 子集单次运行中，SFT+DPO+RLVR 得 74.1，SFT+RLVR 得 71.9。Llama 4 只把它作为最后的轻量打磨，理由是重度 SFT 和 DPO "会过度约束模型、限制在线 RL 的探索"（[Meta Llama 4 博客](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)）。DeepSeek-V3、Qwen3、Kimi K2、GLM-4.5 的报告正文中都没有出现 DPO；这是"报告里没有"，不排除内部未公开的使用。这两种用法看似相反，背后是同一个机制：DPO 让策略变尖、变窄，对短的 RL 预算有利，对长的 RL 预算不利。这一条是笔记的推断。

### PRM 与细粒度 credit assignment：能提效率，学到的 PRM 当奖励必被 hack

**机制。** 方法分成两族，数学性质不同：

- **On-policy MC credit**（VinePPO、SPO、TreeRL、TreeRPO）：用策略自己的 rollout 对着真实验证器估计中间状态的 V^π。在 K 有限带来的噪声之内，这是无偏的，作用只是降方差，代价是额外的生成。
- **学习型 PRM 作为奖励**：往 return 里注入一个有偏的代理信号，RL 优化器会找到它的错误。

几种折中方案各有数学解释：

- Gao et al. 的 Delta 变换 r_k = r(k) − r(k+1) 会伸缩求和，本质上是 potential-based shaping。经典结论是它保持最优策略不变，所以可以预期"安全但收益小"。
- PRIME 在 γ=1 时，逐 token 的隐式奖励求和后等于序列级的隐式 ORM 分数，所以不会带来超出 outcome 标签的新信息。
- PAV 用另一个 prover 策略下的 advantage A^μ 作为过程奖励，与"策略自身的 advantage 就是最优 PRM"的理论结果一致（[PAV 〔全文〕](https://arxiv.org/abs/2410.08146)；[2502.10581 〔摘要〕](https://arxiv.org/abs/2502.10581)）。
- "GRPO is secretly a PRM"：用 token 级 loss 且 μ=1 时，outcome GRPO 等价于一个隐式的 MC 过程奖励。缺陷是每个共享前缀按它出现的频数 |λ| 加权（[2509.21154 〔全文〕](https://arxiv.org/abs/2509.21154)）。

**学习型 PRM 当奖励必被 hack，记 A：**

- Gao et al.：SR+PR 训练 1 个 epoch 后从 30.58 掉到 11.16，出现重复步骤、emoji 和反复输出 "The answer is"（[2410.15115 〔全文〕](https://arxiv.org/abs/2410.15115)）。
- PURE：sum-form 在第 25 步崩溃；min-form 的纯 PRM 运行全部在约第 270 步出现"零步"或 "Thank you."（[PURE 〔全文〕](https://arxiv.org/abs/2504.15275)）。
- PROF：PRM 与 ORM 混合时 entropy 迅速塌到 0（[PROF 〔全文〕](https://arxiv.org/abs/2509.03403)）。
- 纯 PRM 的 GRPO：PRM 奖励在约 100 步内到 1.0，准确率为 0%（[2603.06621 〔全文〕](https://arxiv.org/abs/2603.06621)）。

**效率收益一致，记 B：**

- VinePPO：达到 PPO 峰值所需梯度步少 2.8–9 倍，墙钟少 1.5–3 倍，响应长度 ≤1024。
- PRIME：约 40% 的步数达到 RLOO 的训练奖励，但每步多约 28% 时间。
- PURE：约 30% 的步数达到 VR 水平。

**最终准确率的增益只有约 0–3 点，且随模型规模和上下文长度缩小，记 D：**

- SPO-tree 对 GRPO：2K 上下文 +11.6，4K +7.6，32K +0.8，AIME24 在 32K 时持平（[SPO 〔全文〕](https://arxiv.org/abs/2505.23564)）。
- TreeRPO：1.5B +2.9，7B +0.7。
- PRPO：MATH +2.2/+3.2，但 AIME25 −2.8。
- 这些研究都没有报告多个训练 seed，多数用 Qwen2.5-Math 作 base。

SPO 有一个值得单独指出的发现：只加 token 掩码就把 GRPO 在 GSM8K 上从 45.7% 提到 53.6%。也就是说，很大一部分收益来自 token 选择，而不是 MC credit。

在长 CoT 场景，outcome-only 的逐 token advantage 不是硬瓶颈。R1 和 k1.5 都只用 outcome 奖励训练出了长 CoT，而 dense credit 的研究多在 ≤4–8K token 下训练。唯一的反例信号是 VAPO：学习型 critic 比 DAPO 高 10 点以上，但只是原文，记 C。

前沿实践：DeepSeek-R1 v1 在"Unsuccessful Attempts"中列出 PRM 的三点局限：步骤难定义、中间步骤正误难判、模型式 PRM 必然被 hack；结论是相对额外开销"优势有限"（[R1 v1 〔全文〕](https://arxiv.org/html/2501.12948v1)）。Kimi k1.5 刻意不用 value network，理由是那些犯错后又被纠正的步骤对学习长 CoT"极有价值"。DeepSeekMath-V2 用的是整份证明级别的验证器，不是逐步的 dense credit，这一点只见于搜索摘要（[2511.22570 〔片段〕](https://arxiv.org/abs/2511.22570)）。同一个实验室，2024 年的 DeepSeekMath 报告过程监督优于结果监督（幅度只有图），2025 年的 R1 却放弃了 PRM。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| 学习型 PRM 求和作奖励 | 负（会崩溃） | 早期快 | 崩溃 | 长度、重复、空输出 | **A**（hacking）|
| 有界或伸缩变换（Clip-Delta / min-form / PRIME） | +0.2–1.8，在噪声内 | 2–3 倍步数 | 纯 PRM 仍会被 hack | 长度 | **C**（增益）|
| PRM 只用作数据过滤（PROF） | +1–3（2 倍采样） | — | 稳定 | 采样成本 | **C** |
| On-policy MC / 树形 credit | +1–3，随规模和上下文缩小 | 3–9 倍步数 | 稳定（真实验证器） | rollout 慢约 2 倍 | **B**（小模型/短上下文）/ **C**（大规模）|
| PAV（prover advantage） | 相对 ORM-RL +6–7% | 5–6 倍 | — | — | **C**（结果）/ **B**（机制）|
| 高 entropy token 掩码（80/20） | Qwen3-32B AIME24 +7.71、AIME25 +11.04；8B 只 +0.53 | — | — | — | **C** |
| PRM 作为测试时 reranker | Lightman：BoN 下 78.2 对 72.4 | — | — | — | **B**（测试时）|

### 多轮 agentic：正确性约束决定成败，turn-level credit 仍是 C

**环境 token 屏蔽与 TITO 是正确性要求。** 如果不 mask，策略梯度会去提高检索器或解释器输出文本的似然。这些文本不是策略采样出来的，所以这相当于对环境输出做 behavior cloning，importance ratio 也失去了 on-policy 意义。Search-R1 在 Qwen2.5-7B-base 上 PPO 训练，7 个 QA 测试集的平均 EM 为：带 mask 0.431，不带 0.343。在 3B 上是 0.303 对 0.262，而且有两个数据集上不 mask 反而略好（[Search-R1 〔全文〕](https://arxiv.org/abs/2503.09516)）。其余所有系统都直接采用 mask，没有做消融，记 **B**。

verl 报告，用 `apply_chat_template` 重新拼接 token 无法复现逐轮的 token ID，会导致"PPO 训练连单轮都不收敛"（〔代码〕verl `agent_loop.rst`）。GLM-5 为此加了一个 token-in-token-out 网关（[GLM-5 〔全文〕](https://arxiv.org/abs/2602.15763)）。

**轨迹过滤。** SimpleTIR 规定：只要某一轮既没有代码块也没有最终答案（void turn），就丢掉整条轨迹。Qwen2.5-7B 的窗口内峰值从 naive 的 AIME24 20.8 升到 50.5。通用的低概率 token 过滤只到 23.3，高 ratio 过滤只到 26.3，都替代不了它。这是单次运行、报告峰值的结果，记 **B−/C**（[SimpleTIR 〔全文〕](https://arxiv.org/abs/2509.02479)）。

截断或超时轨迹的屏蔽（DeepSWE 的 compact filtering、SkyRL-Agent、Tongyi、Kimi-Researcher、GLM-5）被普遍采用，但没有任何一项消融，记 **C**。它需要配合显式的预算项一起用，否则没有东西惩罚"耗尽预算"。

**Turn-level credit：**

- **GiGPO**：用 3 个 seed，在 ALFWorld 和 WebShop 上比 GRPO 高 9–14pp，超过了 ±3–4 的标准差。但各方法用的是同一组超参，baseline 没有单独调。它的收益依赖于环境状态精确重复，在 SWE 或网页浏览中观测几乎不会重复，这时它会退化为 GRPO（笔记推断）。HCAPO 等后续论文引用的 GiGPO 数字都是从原文复制的，没有重跑。记 **B−**（[GiGPO 〔全文〕](https://arxiv.org/abs/2505.10978)）。
- **MT-GRPO**：只测了 2 轮，rollout 数是 G^{K−1}。它的 outcome-only GRPO 基线 EM 为 0，与 Search-R1 在同族模型上的 0.35–0.43 矛盾。记 **C**。
- **VAGEN**：大部分增益（0.55→0.76）来自推理格式，bi-level GAE 加世界模型奖励只多 +0.06，而且出现了讨好 judge 的 hacking。记 **C**。
- **LOOP**：在 AppWorld 上，不用 std 归一化的 leave-one-out baseline 有 3 次运行的结果，token 级 ratio 平均 66.4±4.8；std 除法大约损失 9pp。记 **B**（[LOOP 〔全文〕](https://arxiv.org/abs/2502.01600)）。
- **Ratio 粒度**：LOOP 认为 token 级最好，SORL 认为 turn 级 ratio 在 off-policy 时更稳（[SORL 〔全文〕](https://arxiv.org/abs/2511.20718)）。记 **D**，两者差别很可能来自 off-policy 程度不同。

**PPO-with-critic 对 GRPO 在多轮上记 D：**

- critic 胜：Search-R1 7B-base；Practitioner's Guide 的 TextWorld，但那里只有 PPO 调过参（[2510.01132 〔全文〕](https://arxiv.org/abs/2510.01132)）；R1-distill 用 GRPO 崩溃，用 PPO 稳定（[2505.15117 〔全文〕](https://arxiv.org/abs/2505.15117)）。
- critic 负：LOOP 中 PPO 50.8，GRPO 58.0；GiGPO 1.5B 中 PPO 54.4，GRPO 72.8。

**非算法改动的同设定差距远大于算法差距：**

- 检索器质量：随机检索器对精确 E5 检索器，EM 0.241 对 0.430（[2505.15117](https://arxiv.org/abs/2505.15117)）。
- 冷启动数据：真实工具轨迹对拼接合成轨迹，AIME24 avg@32 33.23 对 4.38（[DemyAgent](https://arxiv.org/abs/2510.11701)）。
- 测试时上下文管理：三个实验室独立报告 BrowseComp 提高 7–16 点。DeepSeek-V3.2 从 51.4 到 67.6，Kimi K2.5 从 60.6 到 74.9，GLM-5 从 55.3 到 75.9。记 **A−**。
- 保留跨工具调用的 thinking：MiniMax 称 BrowseComp 从 31.4 到 44.0，这个数字来自搜索摘要转述的厂商博客，记〔片段〕。

**奖励设计：**

- format 奖励帮助小的 base 模型（3B PPO 从 0.303 到 0.389），对 7B PPO 持平，对 14B 有害；SFT 冷启动之后就不需要了。记 B。
- 中间检索奖励持平或有害。记 B（[2505.15117](https://arxiv.org/abs/2505.15117)）。
- SWE 任务中二值奖励与按测试比例给分哪个好，结论随先验强弱翻转。记 D。

**异步 rollout：** 吞吐提升有多组独立测量，Kimi 至少 1.5 倍，SkyRL-Agent 1.55 倍，verl 1.6–2.35 倍。记 **B**（效率）。对上限的影响未知，没有任何 agentic 研究在固定总样本数下比较同步与异步的最终准确率。

| 改动 | Ceiling | Efficiency | Stability | Side effects | 等级 |
|---|---|---|---|---|---|
| 环境 token 屏蔽 + TITO | 大（7B +0.088 EM） | — | 机制上正向 | 无 | **B** |
| Void-turn 轨迹过滤 | +30 AIME（峰值） | — | 梯度范数有界 | 丢弃格式错误的失败样本 | **B−/C** |
| 截断/超时轨迹屏蔽 | 无消融 | — | 声称有效 | 需配合预算惩罚 | **C** |
| GiGPO（anchor state） | +9–14pp（≤7B，3 seeds） | 同 rollout 预算 | — | 需要状态重复 | **B−** |
| MT-GRPO / VAGEN / ArCHer / RAGEN 方差过滤 | — | MT-GRPO 指数级成本 | — | judge hacking（VAGEN） | **C** |
| LOO baseline 不用 std 归一化 | std 约损 9pp（LOOP） | — | — | — | **B** |
| Token / turn / sequence ratio | — | — | 取决于 off-policy 程度 | — | **D** |
| 多轮中 PPO vs GRPO | 符号不定 | — | — | — | **D** |
| 测试时上下文管理 | BrowseComp +7–16 | — | — | — | **A−** |
| 训练时上下文管理 | — | — | — | — | **C** |
| 异步 rollout + staleness 上界 | 未知 | 1.5–2.4 倍 | 需要 off-policy 控制 | — | **B**（效率）|

## 算法 × 设计轴对照表：多数命名算法只改一到两个轴

下表区分两件事：每个方法自己**规定**了什么，以及它的核心声称有多强的**受控证据**。"—" 表示未规定或沿用默认。tok 表示 token 级聚合（除以 token 总数），seq 表示先在序列内平均再在序列间平均。〔先验〕或〔摘要〕行中的超参数来自摘要或先验知识，发表前需要对照原文的方法节核对。

| 算法（来源，阅读深度） | Advantage / baseline | 归一化 | 聚合 | Clip / ratio 粒度 | KL / entropy | Off-policy 修正 | 数据 / 采样 | 奖励 | 核心声称的等级 |
|---|---|---|---|---|---|---|---|---|---|
| PPO-RLHF（[InstructGPT](https://arxiv.org/abs/2203.02155)〔部分〕；[N+](https://arxiv.org/abs/2403.17031)〔全文~70%〕） | critic + GAE（λ 0.95，γ 1） | advantage whitening | tok | token clip 0.2；value clip 0.2 | 每 token 对 SFT 的 KL 放进 reward（β 0.05） | PPO ratio，K=4 个 epoch | prompt 复用 | 学习型 RM；没有 EOS 记 −1 | 调好时 ≥ 离线 DPO：**A** |
| ReMax（[2310.10505](https://arxiv.org/abs/2310.10505)〔部分〕） | greedy 奖励作 baseline | — | seq | 无 | KL 在 reward 里 | 无 | 1 个样本 + 1 个 greedy | RM | **C** |
| RLOO（[2402.14740](https://arxiv.org/abs/2402.14740)〔全文〕） | leave-one-out | 无 | 整条序列 | 无 | KL 在 reward 里 | 无 | k 个样本 | RM | 与 GRPO 可互换：**B** |
| GRPO（[2402.03300](https://arxiv.org/abs/2402.03300)〔全文〕） | group mean | ÷ group std | seq（1/\|o\|） | token clip 0.2 | k3 作为 loss，β 约 0.04 | PPO ratio | G 个样本 | outcome 或 process RM | std 与 1/\|o\| 有偏：**B** |
| DPO（[2305.18290](https://arxiv.org/abs/2305.18290)〔摘要〕） | 隐式 log-ratio | — | 序列求和 | — | 隐式 KL（β） | 无（离线） | 偏好对 | BT 偏好 | 便宜可用 **A**；低于 online **A** |
| IPO / SLiC / GPO（[2402.05749](https://arxiv.org/abs/2402.05749)〔全文〕） | 同 DPO | — | 序列求和 | — | link 函数不同 | 无 | 偏好对 | BT | 与 DPO 打平：**A** |
| KTO（[2402.01306](https://arxiv.org/abs/2402.01306)〔全文〕） | 相对 z₀ 的逐点值 | — | — | — | z₀=KL 估计 | 无 | 非配对数据 | 二值好/坏 | **D** |
| SimPO（[2405.14734](https://arxiv.org/abs/2405.14734)〔全文〕） | 长度归一化的 log-prob，无 ref | 1/\|y\| | — | — | 无 ref；margin γ | 无 | Instruct 设定为 on-policy | PairRM / ArmoRM | **D** |
| LN-DPO（[Tulu 3](https://arxiv.org/abs/2411.15124)〔全文〕） | 长度归一化的 ρ | 1/\|y\| | — | — | β=5 | 部分 on-policy | delta-learning 对比对 | GPT-4o judge | **B** |
| Semi-online DPO（[2506.21495](https://arxiv.org/abs/2506.21495)〔全文〕） | DPO | — | — | — | ref 同步 | 周期性同步 | on-policy 对 | RM / 验证器 | ≈ GRPO：**A** |
| Math-Shepherd PPO（[2312.08935](https://arxiv.org/abs/2312.08935)〔全文〕） | critic + 逐步 PRM | — | tok | token clip | KL 0.04 | PPO | — | MC 标注训出的 PRM | **C** |
| VinePPO（[2410.01679](https://arxiv.org/abs/2410.01679)〔全文〕） | 每步 MC V̂（K=9） | — | 步级共享 | token clip | — | PPO | 额外 rollout | outcome | 效率 **B** |
| REINFORCE++（[2501.03262](https://arxiv.org/abs/2501.03262)〔全文/摘要，两份笔记标注不一〕） | 全 batch（可先减组均值） | batch std | — | token clip | KL 在 reward 里 | PPO ratio | — | RM 或可验证 | **C** |
| Kimi k1.5（[2501.12599](https://arxiv.org/abs/2501.12599)〔部分〕） | K 个样本的均值 | 无 | — | 平方损失近端项，无 clip | 对上一迭代 π_old 的正则 | 部分 rollout | 课程；按 1−成功率采样 | 可验证 + 长度 | **C** |
| PRIME（[2502.01456](https://arxiv.org/abs/2502.01456)〔全文〕） | 隐式 PRM + RLOO | — | — | PPO clip | — | PPO | — | outcome（在线训练 PRM） | **C** |
| DAPO（[2503.14476](https://arxiv.org/abs/2503.14476)〔全文，消融节〕） | group mean | ÷ std | **tok** | **0.2/0.28** | **无** | PPO，16 次更新/批 | **dynamic sampling** | 二值 + **soft overlong** | 逐项都是单次运行：**C**；组件见各轴 |
| Dr. GRPO（[2503.20783](https://arxiv.org/abs/2503.20783)〔全文〕） | group mean（= RLOO×(G−1)/G） | **无** | **常数** | 0.2 | 无 | PPO | — | 二值 | 去长度偏置 **B** |
| ORZ（[2503.24290](https://arxiv.org/abs/2503.24290)〔全文〕） | critic，λ=γ=1 | batch | — | token clip | 无 | PPO | — | 二值 | **B**（λ=1）|
| VC-PPO / VAPO（[2503.01491](https://arxiv.org/abs/2503.01491)；[2504.05118](https://arxiv.org/abs/2504.05118)〔全文〕） | 预训练 critic；decoupled / 长度自适应 GAE | — | tok | clip-higher | 无 | PPO | group sampling | 二值 + 正样本 NLL | 相对 critic-free 的优势：**D** |
| Skywork-OR1（[2505.22312](https://arxiv.org/abs/2505.22312)〔部分〕） | group mean | ÷ std | tok | clip | **无 KL；自适应 entropy** | N_SGD=1 | 拒绝零优势组；多阶段长度；τ=1 | 二值 | on-policy 防 collapse **A** |
| AceReason（[2505.16400](https://arxiv.org/abs/2505.16400)〔全文〕） | group mean | ÷ std | tok | （严格 on-policy，r=1） | 无 | 严格 on-policy | 难题过滤；先数学后代码 | 二值 | **A**（同上）|
| CISPO / MiniMax-M1（[2506.13585](https://arxiv.org/abs/2506.13585)〔部分〕） | group | ÷ std | tok | **sg(clip IS 权重)**，只有上界 | 无 | FP32 LM head | 重复早停 | 可验证 + 长度 | **B / D** |
| Magistral（[2506.10910](https://arxiv.org/abs/2506.10910)〔全文〕） | group mean | **minibatch** | 组内 tok | ε_high 0.26–0.3 | 无；无 entropy bonus | 异步，1 次更新/批 | 过滤零优势组 | format 门控 + 正确 + 语言 | 归一化无影响 **B** |
| GSPO（[2507.18071](https://arxiv.org/abs/2507.18071)〔全文〕） | group | ÷ std | seq | **序列几何平均 ratio**，3e-4/4e-4 | — | 声称不需要 R2 | — | 可验证 | **B / D** |
| GMPO（[2507.20673](https://arxiv.org/abs/2507.20673)〔全文〕） | group | ÷ std | 几何平均 | token 级 log 空间 clip | — | PPO | — | 可验证 | **C** |
| Lite PPO（[2508.08221](https://arxiv.org/abs/2508.08221)〔全文〕） | group mean | **batch std** | tok | 0.2 | 无 | 1 个 epoch | 无 DS、无 overlong | 二值 | **C**（只有图）|
| TIS（[Yao et al. 博客](https://fengyao.notion.site/off-policy-rl)〔全文〕） | 不变 | — | — | 额外乘 min(π_learner/π_sampler, C) | — | **token 级截断 IS** | — | — | **D** |
| AReaL decoupled PPO（[2505.24298](https://arxiv.org/abs/2505.24298)〔全文*〕） | — | batch | — | clip 以 π_prox 为中心 | — | behavior→prox 的权重 | staleness η 上界 | — | **B** |
| PipelineRL（[2509.19128](https://arxiv.org/abs/2509.19128)〔全文*〕） | value baseline | — | — | min(c, π/μ)，c=5 | — | in-flight 更新 | — | — | 效率 **B** |
| M2PO（[2510.01161](https://arxiv.org/abs/2510.01161)〔全文*〕） | group | — | 全 token | 二阶矩 M₂≤0.04 掩码 | — | 支持 s≥256 | — | 可验证 | **C** |
| IcePop（[2510.18855](https://arxiv.org/abs/2510.18855)〔全文*〕） | GRPO | — | tok | 双侧掩码 [0.5,5] | KL 0 | 训练/推理 ratio 掩码 | — | — | **B** |
| R3（[2510.11370](https://arxiv.org/abs/2510.11370)〔全文*〕） | — | — | — | — | — | 回放推理时的 routing | — | — | MoE 中 **A** |
| ScaleRL（[2510.13786](https://arxiv.org/abs/2510.13786)〔全文〕） | group mean | **batch std** | **prompt 级** | CISPO | 无 | PipelineRL k=8；FP32 logits | zero-variance 过滤；No-Positive-Resampling；interruption | ±1 | 各组件 LOO 下 A 在 ±0.02 内 |
| SAPO（[2511.20347](https://arxiv.org/abs/2511.20347)〔全文〕） | group | — | seq | **平滑温度门控** | — | 软衰减 | — | 可验证 | **C / D** |
| Qwen MiniRL（[2512.01374](https://arxiv.org/abs/2512.01374)〔全文〕） | group | — | tok，**无长度归一化** | 0.2/0.27 | — | TIS 阈值 5；off-policy 时用 R2/R3 | — | 可验证 | 受控 **B** |
| DeepSeek-V3.2（[2512.02556](https://arxiv.org/abs/2512.02556)〔部分〕） | group mean | 公式中无（正文写 normalizing） | seq（1/\|o\|） | token clip | **无偏 k3**，随领域调整 | **OPSM**；Keep Routing；Keep Sampling Mask | 1,800+ 合成环境 | 可验证 + GenRM + agentic | 无消融：**C** |
| OlmoRL（[2512.13961](https://arxiv.org/abs/2512.13961)〔全文 RL 节〕） | group mean | **无** | tok | 0.2/0.272 | 无 | **TIS 上限 2.0**；异步 ≤8 | zero-gradient 过滤 + active sampling；离线去掉 pass>62.5% | 多领域 | 随机奖励对照为阴性 |
| JustRL（[2512.16649](https://arxiv.org/abs/2512.16649)〔全文〕） | group mean | ÷ std（verl 默认） | — | 0.2/0.28 | 无（Table 1 与 Table 2 不一致） | — | 无过滤；单阶段 16K | 二值 | 简单 recipe 有竞争力：**C** |
| Search-R1（[2503.09516](https://arxiv.org/abs/2503.09516)〔全文〕） | critic / group | — | 只算 LLM token | clip | KL 同样 mask | — | — | outcome EM | mask **B** |
| GiGPO（[2505.10978](https://arxiv.org/abs/2505.10978)〔全文〕） | episode + **anchor state** 组 | std 或 1 | — | clip | KL 0.01 | — | — | 环境奖励 | **B−** |
| LOOP（[2502.01600](https://arxiv.org/abs/2502.01600)〔全文〕） | LOO，K=6 | **无** | 对 LLM token 取平均 | per-token ratio | 无 KL 更好 | — | — | 单元测试通过比例 | **B** |
| SimpleTIR（[2509.02479](https://arxiv.org/abs/2509.02479)〔全文〕） | GRPO | ÷ std | — | 0.2/0.28 | 无 | — | **void-turn 轨迹过滤** | 二值 | **B−/C** |
| MiniMax Forge / M2.5（[博客](https://www.minimax.io/blog/forge-scalable-agent-rl-en-1779896141)〔全文〕） | **reward-to-go** − B | — | — | CISPO clip(r,0,1+ε) | — | 窗口 FIFO 调度 | 前缀树合并 | 速度 + 表现 + 过程 | 无消融：**C** |

整张表的规律是：变化最多的轴是 **trust-region / IS 轴**（token clip、非对称 clip、IS 权重 clip、序列 ratio、几何平均、软门控、二阶矩约束、TIS、序列掩码），而这个轴上关于上限的受控证据最薄，只有 ScaleRL 一项。advantage 和归一化轴已经收敛到 group mean 加可选的 batch std，经验上影响很小。

## 时间线：估计器越来越简单，复杂度移到 trust-region 和数值层

2023 到 2025 年，advantage 估计器一路简化：critic → greedy baseline → leave-one-out → group mean/std → mean-only。同一时期，复杂度移进了 trust-region 与 IS 层：token clip → 非对称 clip → IS 权重 clip → 序列 ratio → 软门控 → TIS 和掩码。这与受控证据的结论一致：估计器本身影响不大，off-policy 修正和数值正确性决定训练是否稳定。

下表中，标"✓"的日期经 arXiv API 核实为首次提交日期（同一行有多个条目时，✓ 只适用于受控研究笔记时间线里出现过的条目）；其余日期按 arXiv ID 推出年月。〔先验〕条目只核实了 ID、标题和日期，一句话描述在发表前需要抽查。

| 日期 | 方法 / 报告 | 改了什么 | 阅读深度 |
|---|---|---|---|
| 2017-06 ✓ | Deep RL from Human Preferences（[1706.03741](https://arxiv.org/abs/1706.03741)） | 用成对偏好学 RM 来驱动 RL | 〔先验〕 |
| 2017-07 ✓ | PPO（[1707.06347](https://arxiv.org/abs/1707.06347)） | clipped ratio 作为廉价的一阶 trust region | 〔先验〕 |
| 2019-09 ✓ | Ziegler et al.（[1909.08593](https://arxiv.org/abs/1909.08593)） | PPO + RM 用于 LM，KL 放进 reward | 〔先验〕 |
| 2020-05/06 ✓ | Engstrom；Andrychowicz（[2005.12729](https://arxiv.org/abs/2005.12729)；[2006.05990](https://arxiv.org/abs/2006.05990)） | 实现细节比算法标签更重要 | 〔全文〕/〔全文前半〕 |
| 2022-03 ✓ | InstructGPT（[2203.02155](https://arxiv.org/abs/2203.02155)） | SFT → RM → PPO-ptx | 〔先验〕/〔部分〕 |
| 2022-10 | RM overoptimization（[2210.10760](https://arxiv.org/abs/2210.10760)） | gold–KL 的函数形式；KL 作用类似 early stopping | 〔全文*〕 |
| 2023-05 ✓ | DPO（[2305.18290](https://arxiv.org/abs/2305.18290)）；Let's Verify（[2305.20050](https://arxiv.org/abs/2305.20050)） | 闭式离线目标；PRM 优于 ORM（测试时） | 〔摘要〕/〔全文〕 |
| 2023-10 ✓ | ReMax（[2310.10505](https://arxiv.org/abs/2310.10505)）；IPO（[2310.12036](https://arxiv.org/abs/2310.12036)） | 去掉 critic；平方 link 函数 | 〔部分〕/〔摘要〕 |
| 2023-12 ✓ | Math-Shepherd（[2312.08935](https://arxiv.org/abs/2312.08935)） | MC 自动步级标注 + PPO | 〔全文〕 |
| 2024-02 ✓ | GRPO（[2402.03300](https://arxiv.org/abs/2402.03300)）；RLOO（[2402.14740](https://arxiv.org/abs/2402.14740)）；KTO；GPO | group baseline 取代 critic；DAP 损失族调参后打平 | 〔全文〕 |
| 2024-03/04 | N+ Details（[2403.17031](https://arxiv.org/abs/2403.17031)）；Is DPO Superior to PPO（[2404.10719](https://arxiv.org/abs/2404.10719)） | PPO-RLHF 的可复现细节；调好的 PPO > DPO | 〔全文〕 |
| 2024-05/06 | SimPO；Ivison et al.（[2406.09279](https://arxiv.org/abs/2406.09279)）；WPO | 无 ref 长度归一化；数据 > 算法 > RM | 〔全文〕 |
| 2024-10 ✓ | VinePPO（[2410.01679](https://arxiv.org/abs/2410.01679)）；PAV（[2410.08146](https://arxiv.org/abs/2410.08146)） | 用 MC value 取代 value 网络；prover advantage 作为过程奖励 | 〔全文〕 |
| 2024-11 ✓ | Tülu 3（[2411.15124](https://arxiv.org/abs/2411.15124)） | 提出 RLVR 这一名称；LN-DPO | 〔全文〕 |
| 2025-01 ✓ | REINFORCE++；DeepSeek-R1（[2501.12948](https://arxiv.org/abs/2501.12948)）；Kimi k1.5 | batch 归一化；GRPO + 规则奖励产生长 CoT；mirror descent | 〔全文〕/〔部分〕 |
| 2025-02 ✓ | PRIME（[2502.01456](https://arxiv.org/abs/2502.01456)）；LOOP（[2502.01600](https://arxiv.org/abs/2502.01600)） | 在线隐式 PRM；agent 场景的 LOO PPO | 〔全文〕 |
| 2025-03 ✓ | DAPO；Dr. GRPO；ORZ；VC-PPO；Search-R1；TOPR | clip-higher / dynamic sampling / token loss；去除两种偏置；λ=1；环境 token 屏蔽 | 〔全文〕 |
| 2025-04 ✓ | VAPO；Yue et al.（[2504.13837](https://arxiv.org/abs/2504.13837)）；Sober Look（[2504.07086](https://arxiv.org/abs/2504.07086)）；RAGEN；PURE | value-based 回归；pass@k 边界争论；评测方差审计；Echo Trap；min-form PRM 与 PRM hacking 目录 | 〔全文〕/〔全文*〕 |
| 2025-05 ✓ | Qwen3；AceReason；Skywork-OR1；Entropy Mechanism；ProRL；AReaL；GiGPO | 严格 on-policy；entropy = 协方差；长训练 + reference 重置；解耦异步 | 〔部分〕/〔全文〕/〔全文*〕 |
| 2025-06 ✓ | Spurious Rewards；Magistral；MiniMax-M1（CISPO）；80/20 forking tokens | 随机奖励这个混杂因素；IS 权重 clip；FP32 head | 〔全文〕/〔部分〕 |
| 2025-07 ✓ | GSPO；GMPO；Kimi K2 | 序列级 ratio；几何平均 | 〔全文〕/〔部分〕 |
| ~2025-08 | TIS（[Yao et al. 博客](https://fengyao.notion.site/off-policy-rl)）；GLM-4.5；Tricks or Traps | 训练–推理 mismatch 修正；单阶段 64K；Lite PPO | 〔全文〕/〔部分〕 |
| 2025-09 | Liu & Li 博客；PipelineRL；SimpleTIR；PROF | Seq-MIS / Geo-RS；in-flight 更新；void-turn 过滤；PRM 只做过滤 | 〔全文〕/〔全文*〕 |
| 2025-10 ✓ | ScaleRL（[2510.13786](https://arxiv.org/abs/2510.13786)）；M2PO；FP16；R3；IcePop；BroRL | 上限与效率分开拟合；二阶矩掩码；统一精度；routing 回放；超大 N rollout | 〔全文〕/〔全文*〕 |
| 2025-11 ✓ | SAPO（[2511.20347](https://arxiv.org/abs/2511.20347)） | 软门控 trust region | 〔全文〕 |
| 2025-12 ✓ | DeepSeek-V3.2；Qwen MiniRL（[2512.01374](https://arxiv.org/abs/2512.01374)）；OLMo 3；JustRL | 无偏 KL + OPSM；token 级 + IS 的受控研究；全开源 recipe；简单 recipe | 〔部分〕/〔全文〕 |
| 2026-01/02 | HA-DW；MaxRL；OTB；GLM-5；Kimi K2.5；ARLArena | 难度偏置；最大似然加权；TITO 网关；agentic 稳定性比较 | 〔摘要〕/〔部分〕/〔全文〕 |
| 2026-04 | GenAC（[2604.10701](https://arxiv.org/abs/2604.10701)）；SFT-then-RL（[2604.23747](https://arxiv.org/abs/2604.23747)） | 估计器的受控比较；框架 bug 让 baseline 失效 | 〔全文〕/〔摘要〕 |
| 2026-04（ID 为 2606） | DeepSeek-V4（[2606.19348](https://arxiv.org/abs/2606.19348)） | GRPO 领域专家模型 + 多教师 on-policy 蒸馏 | 〔二手〕；ID 与日期不一致，需复查 |
| 2026-05 | VeXact（[2605.14220](https://arxiv.org/abs/2605.14220)）；MiniMax-M2 系列 | 零 mismatch 参考引擎 | 〔全文*〕 |
| 2026-06/07 ✓ | Group-Std Identity（[2607.00152](https://arxiv.org/abs/2607.00152)）；异步 collapse 定标律（[2607.01083](https://arxiv.org/abs/2607.01083)） | 三种归一化是同一个旋钮；S·η≈常数 | 〔摘要〕 |
| 2026-09 | CIS（[2609.32444](https://arxiv.org/abs/2609.32444)）；MiMo-V2.6 | 校准过的 IS；单次混合 RL 运行 | 〔全文*，前~50%〕/〔二手〕 |
| 2026-10 ✓ | How RL Reshapes Reasoning（[2610.04158](https://arxiv.org/abs/2610.04158)）；GRPO at 1B（[2610.04928](https://arxiv.org/abs/2610.04928)） | log-sigmoid 与 log-linear 无法区分；估计器差异在 seed 噪声内 | 〔部分〕/〔摘要〕 |

## 2026 共识 recipe：哪些有受控证据，哪些是 folklore

到 2025 年末至 2026 年，开源与前沿 recipe 收敛到大致相同的形态：

- group 采样的 REINFORCE/GRPO，advantage 只做均值中心化；
- 可验证奖励下不加对 SFT 的 KL；
- token 级或 prompt 级的 loss 归一化；
- 非对称 clip（0.2/0.27–0.28）或 IS 权重 clip（CISPO）；
- zero-variance 过滤加 batch 补齐；
- 异步 rollout，staleness 上界 1–8，加训练–推理 mismatch 修正（TIS 上限约 2、FP32 head 或 FP16、MoE 的 routing replay）；
- 采样温度约 1.0；
- 某种截断控制。

这套 recipe 更准确的描述是"一个**稳定性与工程包**"，而不是一个算法。它证据最强的部分都是关于"不把训练搞坏"的，关于"抬高上限"的证据很少。

| 共识元素 | 证据 | 等级 | 定性 |
|---|---|---|---|
| 每批 rollout 少做 off-policy SGD | Skywork、AceReason、Magistral 三组独立受控；Qwen MiniRL | **A** | 受控证据支持 |
| MoE 的 routing 一致性（R2/R3/Keep Routing） | R3 原文 + Qwen 受控研究 + DeepSeek 生产经验 | **A**（MoE，多 minibatch）| 受控证据支持 |
| 推理 RLVR 中去掉对 SFT 的 KL | Skywork 受控 + Entropy Mechanism；普遍采用 | **B** | 证据支持；不适用于学习型 RM |
| 训练–推理数值一致（FP32 head / FP16 / TIS） | ScaleRL 单项消融 + MiniMax 相关性 + FP16 原文；与 FP32 head 无效的报告矛盾 | 消除 mismatch **B**；单项补丁 **D** | 机制支持；具体补丁有争议 |
| 不加固定 entropy bonus | 三方独立负面证据 | **A** | 证据支持 |
| 温度约 1.0 / 按模型校准 | Skywork τ 消融 | **B** | 证据支持；随训练怎么调是 folklore |
| 环境 token 屏蔽 + TITO（agentic） | Search-R1 消融 + 机制 + verl 报告 | **B** | 正确性要求 |
| Zero-variance 过滤 / active sampling | DAPO +8（单次）；ScaleRL 正文与 LOO 矛盾；OLMo 方差图；反例 2512.07611 与 JustRL | 稳定性 **B**；准确率 **D** | 半有据 |
| 只做均值中心化（去掉 std） | 多项研究：上限中性 | 上限中性 **A**；方向 **D** | 依据是 Dr. GRPO 的论证而非决定性数据，**接近 folklore** |
| Token 级聚合 | 方向随 base/aligned 翻转；ScaleRL 偏好 prompt 级 | **D** | **folklore**（去长度偏置的机制本身是 B）|
| Clip-higher 约 0.27–0.28 | 多项消融，但随模型类型变化 | **B / D** | 需要按模型调的二阶旋钮 |
| CISPO / GSPO 优于 token-clipped PPO | 原论文 + ScaleRL；其他研究排序相反 | **C / D** | **folklore** |
| Overlong 惩罚或过滤 | DAPO 正向；JustRL 有害；OLMo 无一致收益；Skywork 称 mask 会 hacking | **D** | **folklore** |
| 分阶段增加上下文长度 | 依赖 base model | **D** | **folklore** |
| 难度过滤 | 多个采用方，少量受控 | **B**（效率）| 实践共识 |
| 异步 + staleness 上界 | 吞吐有多组独立测量 | **B**（效率）| 对上限的影响未知 |
| 评测用 ≥10–30 个 seed | Hochlehnert | **B** | 多数论文仍不遵守 |

## 笔记之间和笔记内部的矛盾

下表的矛盾在正文中都保留了，没有被抹平。凡是写"未分辨"的地方，都不应在 blog 中写成定论。

| 来源 | 说法一 | 说法二 | 本报告的处理 |
|---|---|---|---|
| ScaleRL：off-policy 算法 | 摘要、§3.1：主要影响效率，渐近值相似；图 4a："略好" | 附录 A.11：影响渐近值，"最关键的设计决策之一" | 效率约 2 倍有据；上限**未分辨** |
| ScaleRL：zero-variance 过滤 / No-Positive-Resampling | 正文："更高的渐近值" | LOO：撤回后 A 在 ±0.02 内 | 上限**未分辨**，记为效率 |
| ScaleRL：loss 类型 | 单项比较：GSPO/CISPO 的 A 明显高于 DAPO | LOO：撤回 CISPO 后 A 在 ±0.02 内，只有 B 从 2.01 降到 1.77 | 单项比较中 DAPO A 偏低很可能来自不稳定被拟合成上限（笔记推断）|
| ScaleRL：FP32 head | 单项比较 A 0.52→0.61 | dense 8B LOO "几乎没差别"；Yao、Liu & Li 称它没能阻止崩溃 | **D** |
| ScaleRL：A 的误差 | 三次运行最多 ±0.015 | 建议 ±0.02；B 没有误差带 | 用 ±0.02 |
| Spurious Rewards 与 2512.16912 | clipping 是随机奖励增益的必要条件 | clip 项至少小 17 倍；打开 clipping 可能有害；非 Qwen 模型上也有增益 | 机制 **D**；现象在 Qwen-Math 上是 A |
| Tricks or Traps 内部 | 组级归一化"在各奖励设定下都稳健" | §4.1.2：易数据上 std 放大并可能导致梯度爆炸 | 方向 **D** |
| DeepSeek-V3.2 | 正文写 "normalizing" | 公式 A=R−mean(R)，没有 std | 无法消解 |
| Dr. GRPO | 正文模型写 Qwen2.5-1.5B | §3.3 写 Qwen2.5-Math-1.5B | 记为内部不一致 |
| REINFORCE++ / RLOO / GSPO / VAPO / 80/20 / M2PO / FP16 的阅读深度 | 一份笔记标〔全文〕 | 另一份标〔摘要〕 | 数字取自读了全文的那份笔记 |
| 2505.22617 与 ScaleRL | R = −a·e^H + b，entropy 决定上限 | entropy 不预测下游表现 | **D** |
| Yue et al. 与 ProRL | RLVR 不扩展大 k 边界 | 长训练能扩展（只在 base 较弱的任务上） | 分任务讨论；2610.04158 两种模式都观察到了 |
| VAPO/VC-PPO 与 GenAC | 比 critic-free 高 10 点以上 | 约 0（≤1 点） | **D** |
| CISPO 的排名 | ScaleRL 最好 | CE-GPPO、ARLArena 中最差 | **D**；实现默认值不同（verl [0.8,1.2]）|
| GSPO 的稳定性 | R3 论文：8 个 mini-step 时稳定 | ScaleRL 中发散；FP16 论文中出现 NaN；R3 附录中 4 个 mini-step 时第 90 步崩溃 | **D** |
| Overlong 处理 | DAPO +6/+3 | JustRL 有害；Skywork 中 mask 导致 hacking；OLMo 无一致收益 | **D** |
| 长度课程 | Skywork、AceReason 有效 | GLM-4.5、Polaris（Qwen3-4B）不可逆下降 | **D**，依赖 base |
| 温度调度 | Polaris 随训练升温 | Kimi K2 随训练降温 | **D** |
| PPO vs DPO | Xu：PPO 胜出 | Llama 3：DPO 更好 | 数据范式不同，可以调和 |
| DPO baseline | SimPO 论文中 15.1 LC | WPO 论文中 20.6±0.7 | 小于 5 LC 的单次差异当噪声 |
| PPO vs GRPO（多轮） | Search-R1 7B、TextWorld 中 PPO 胜 | LOOP、GiGPO 1.5B 中 GRPO 胜 | **D** |
| MT-GRPO 基线 | outcome-only GRPO EM=0 | Search-R1 中同类设置为 0.35–0.43 | 基线失败很可能是设置问题 |
| DeepSeekMath 与 DeepSeek-R1 | 过程监督优于结果监督 | 放弃 PRM | 规模扩大后结论翻转 |
| RL's Razor | 定理用 KL(π‖π₀) | 预测指标用 KL(π₀‖π) | 方向不一致 |
| AReaL staleness 表 | 不修正时 η=4 只有 23.3 | 不修正时 η=8 为 35.7 | 非单调，提示单 seed 噪声 |
| R3 的崩溃时间 | 8 个 mini-step 比 1 个崩得晚 | 两者 LR 不同（1e-6 对 3e-6） | 很可能是 LR 混杂 |
| Hochlehnert | 建议 K≥30 个 seed、≥32K token | 自己用 10 个 seed，Qwen-Math 设 4096 上限 | 结论方向可信，绝对值保守 |
| GiGPO 开销 | 称 <0.002% | 用它自己的数字算约 0.15% | 两者都很小 |
| DeepSeek-V4 | arXiv API 给出首次提交 2026-04-26 | ID 前缀是 2606 | 需复查；RL 与 OPD 的分工也有分歧 |

## 值得自己跑的小规模实验

下面的空白都满足两个条件：在 1.5B–8B、单节点或少数节点上可以做；结果能直接改变上文某条判定的等级。每个实验都应遵守检查清单：同时用 Qwen 与至少一个非 Qwen 模型族（OLMo 或 Llama）、≥5 个训练 seed、干净的评测集、按 GPU-hours 记账、加随机奖励阴性对照。

| # | 要回答的问题（对应判定） | 设计 | 判别标准 |
|---|---|---|---|
| 1 | r≡1 恒等式在实践中是否成立（clip 轴） | 用 FP16 或 batch-invariant kernel 让 mismatch 为零，每批只做 1 次更新，比较 GRPO / DAPO / GSPO / CISPO / SAPO，并对齐归一化；再把每批更新次数 N 设为 1/2/4/8 | N=1 时各方法差异应落在 seed 噪声内；差异随 N 增大而出现，就说明这些 loss 测的是 staleness |
| 2 | 跨模型族的组件网格（归一化、聚合、clip-higher 都是 D） | Qwen2.5-Math-1.5B、Qwen3-1.7B(Base/Instruct)、OLMo-2-1B、Llama-3.2-1B 上做 2×2×2 因子设计，各 5 个 seed，≥1000 步 | 把方向翻转归因到具体的 base 或 aligned 属性 |
| 3 | 只在低 p 侧做 std 加权与双侧对称 1/std 的区别（归一化轴） | 三个分支：w=1/√(p(1−p))；只在 p<0.5 时加权；w=1；在不同的 pass-rate 分布上（易、混合、难） | 验证"有益的一半与有害的一半"这一分解 |
| 4 | A 能否被辨识（上限的口径） | 复现 ScaleRL 式拟合：用前 50% 的算力拟合 sigmoid 和 log-linear，用后 50% 检验外推，并用 bootstrap 给 A、B 置信区间 | 若 A 的 CI 宽度大于 0.03，则"抬上限"类结论整体降级 |
| 5 | 截断处理与长度预算的交互（D） | {overlong filter, soft penalty, interruption, 不处理} × 最大长度 {4K, 8K, 16K}，并记录截断率 | 检验"短预算下有益、长预算下有害"，以及"截断率 >10% 预示失稳" |
| 6 | 估计器的偏差与方差（off-policy 轴） | 人为注入已知幅度的 mismatch（例如 FP8 rollout），在 T∈{1K,4K,16K} 下用大样本得到"真梯度"，测 TIS / Seq-MIS / Geo-RS / IcePop / CIS 的梯度余弦相似度 | 文献中完全缺失这一测量；同时可以检验 Seq-IS 的 log ρ std≈σ√T |
| 7 | S·η 定标律（staleness 容忍度） | staleness S ∈{1,2,4,8,16} × LR ∈{3e-7,1e-6,3e-6}，记录首次崩溃步数 | 检验摘要级的结论：最大稳定 LR ∝ S⁻¹ |
| 8 | Clip-higher 的效应是否随 staleness 放大 | ε_high ∈{0.2,0.28,0.35} × N∈{1,4}，在 base 与 aligned 模型上，记录 clip 触发率 | 检验"对哪种模型有效"其实是"在多大 staleness 下有效" |
| 9 | 同算力下 dense credit 与更多 GRPO 样本的对比（PRM 轴 D） | 把 tree/MC credit 的额外 rollout 预算改给 GRPO，用于增大 G 或训练更多步，分别在 4K 与 16K 上下文下 | 检验"dense credit 只是补偿截断与样本稀缺" |
| 10 | DPO 的因子设计（偏好轴） | loss（DPO / IPO / SimPO / LN-DPO）× 数据（离线 / semi-online）× β 网格，1–8B，≥3 个 seed | 量化"数据 on-policy 程度压倒 loss"的效应大小比 |
| 11 | Anchor-state 在开放环境中的重复率（GiGPO） | 在 SWE 或网页环境的 rollout 中统计精确重复与模糊重复状态的比例 | 重复率接近 0，则 GiGPO 在该类环境中退化为 GRPO |
| 12 | 框架默认 KL 的数学有效性 | slime 开 KL 用默认 k1-as-loss，对比 k2-as-loss 和不加 KL，在高 KL 区间（如 on-policy 蒸馏） | 在训练中复现"k1-as-loss 等于没有 KL"和"高 KL 下 k3 有偏" |

## 结论

这些证据改变了"RL 算法"这个问题本身的提法。2023 到 2026 年，被冠以新名字的改动大多不在估计量层面，而在两个"实现层"上：一是 trust region / IS，二是数值一致性。而真正有受控证据的结论，几乎都可以表述为"让梯度估计量等于它声称的那个量"：

- 每批 rollout 只做一次或很少几次更新；
- 消除 routing 与精度带来的 mismatch；
- 屏蔽不是策略采样出来的 token；
- 不让可 hack 的验证器或 PRM 进入 return。

这与 Engstrom et al. 在 2020 年的发现在结构上是同一件事：决定成败的是实现正确性，不是算法标签。对理论读者有一个可操作的推论：既然 r≡1 时 loss 变体之间只剩归一化差异，那么文献中 GRPO、DAPO、GSPO、CISPO 之间的经验排序，应当被读成对各自实验中 staleness 与 mismatch 大小的间接测量。这些排序在不同论文中互相矛盾，恰好是这一读法的旁证。

上限在算法之外：base model 与 mid-training 决定 pass@k 意义上的覆盖率，生成长度和 batch 决定 pass@1 意义上的渐近值，数据的 on-policy 程度决定偏好学习能走多远。由此得到两个尚未被充分利用的研究方向。第一，上限本身的可辨识性：A 的置信区间，以及 sigmoid 与 log-linear 能否区分。在把任何"抬高上限"的论断写进 blog 之前，这一点应当先被测清楚。第二，几乎所有 D 级判定都可以用同一套小规模跨族多 seed 网格来消解。这套网格的成本远低于文献中单个"新 loss"论文的算力，而它产出的证据等级会高出一到两级。
