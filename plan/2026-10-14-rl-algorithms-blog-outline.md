# RL Algorithms blog — 章节大纲

- 日期：2026-10-14
- 证据底稿：[reports/LLM RL 算法改动证据评估.md](../reports/LLM%20RL%20算法改动证据评估.md)，以及 `research_notes/LLM RL 算法改动证据评估/` 下的 8 份笔记
- 平行对象：本仓库的 "RL Infrastructure for the Mathematically Inclined"（`_sections/00–32, 99`）

## 0. 定位与对齐原则

**一句话论点。** infra blog 讲的是"工程选择悄悄改变了被优化的统计对象"。这篇讲它的对偶：**算法的名字也是表面，真正被优化的是十个设计轴上的一组选择，再加上实验条件。** 受控证据显示，多数命名改动只买到效率或稳定性，不抬上限。上限由 base model、生成长度、batch 和数值一致性决定。真正决定成败的是一组"正确性"措施：它们保证梯度估计量就是它声称的那个量。

**结构对齐方式。** 沿用 infra blog 的骨架和文件编号，读者可以按同一位置对照两篇：

| infra blog | 本篇的对应物 |
|---|---|
| 01 Preface（surface vs underneath） | 01 Preface（name vs estimator） |
| 02 How to read（三个 user story） | 02 How to read（读论文 / 选 recipe / 找空白） |
| 03 Skeleton（五阶段 loop） | 03 Skeleton（一个梯度估计量，三个旋钮） |
| 04 Tension（为什么 loop 不够） | 04 Tension（为什么估计量不够：测量问题） |
| 05–11 六个 engineering primitives ①–⑥ | 05–13 八个设计轴 ①–⑧ |
| 12–14 The layer beneath（CUDA / Megatron / quantization） | 14–14b The layer beneath the loss（RL 实际改变了什么；base model） |
| 14b–21 case studies | 15–21b：三个算法族 + 五个证据案例 |
| 22 Recent / 23 Patterns / 24 Code / 25 Landscape / 26 Beyond / 27 Scaling | 同名同位 |
| 28 Pitfalls / 29 Checklist | 同名同位（内容换成算法与评测层面） |
| 30 Reading plan（how to do this yourself） | 30 Decision guide + 30b 十二个小实验（how to do this yourself） |
| 31 Open / 32 Reading | 同名同位 |

**每个轴小节都用同一个模板**，对应 infra 的 "invariant / implementation / what's mathematically interesting"：

1. 这个旋钮改的是什么，用公式写出来。
2. 机制：偏差、方差，或者隐含的目标函数。
3. 受控证据，以及证据之间的矛盾。
4. 判定表：ceiling / efficiency / stability / side effects，加 A–D 等级。
5. What's mathematically interesting。

**写作规则（来自既有反馈）：**

- 先写目标，再写实现。
- 每个行为都钉到具体的公式、行号或表格。
- 不写似是而非的机制解释，代码片段只作教学用途。
- 句子短而密。
- 只读过摘要或片段的数字，正文要么不用，要么明说"摘要称"。

**默认假设，开写前请确认：**

1. 正文用英文，和 infra blog 一致。大纲用中文。
2. 新开平行仓库 `~/Desktop/Projects/RL_algorithms/`，复制 `_sections/`、`templates/`、`scripts/build_index.py` 的脚手架。本仓库只保留 plan、report 和 notes。
3. 标题暂定 **"RL Algorithms for the Mathematically Inclined"**，副标题 *"Which changes to LLM post-training actually matter — and how to tell"*。

---

## 1. 开篇四节（frame）

### 00-article-header.html

- 标题和副标题（见上）。
- dek：先给结论，再列出内容清单：ten design axes、three algorithm families、five evidence case studies、six patterns、12-question checklist、decision guide、12 small experiments。
- 统计行：约 90 min read · ~150 papers & reports · N 张图 · 4-dimension verdicts with A–D evidence grades。

### 01-why.html — Preface: what this survey is for

- 目的：先立论点和边界。
- **Name vs estimator。** "We use GRPO / DAPO / GSPO" 是表面；底下是十个轴上的选择，加上 base model、评测协议和 staleness。
- **边界：** 一个改动只要会改变 reward 曲线的解读、两个算法之间的比较、或者结论的可复现性，就展开讲；纯调参技巧一句带过。
- 结论先行（三句话）：
  1. 多数命名改动只买效率或稳定性。
  2. 上限在 loss 之外。
  3. 决定成败的是估计量的正确性。
- 开篇引语：Engstrom et al. 2020 的结论，即 code-level 优化的效应大于 PPO 与 TRPO 之差（[2005.12729]）。这与 infra blog 的 Chenyang 引语互为镜像。
- 收尾：读完以后，读者拿到一篇新论文时能问出的五个问题。和 infra preface 结尾的问题清单同构。

### 02-how-to-read.html — How to read this survey

- 三个 user story：
  1. **Interpretation**：读一篇新 RL 算法论文，判断它的增益能不能信。
  2. **Decision**：给一次训练选 recipe，知道哪些旋钮必须做对、哪些可以随便选。
  3. **Research**：找出证据最薄、最值得自己做实验的地方。
- 判定口径：
  - 四个维度：ceiling / efficiency / stability / side effects。
  - 证据等级：A = 多个独立受控消融，B = 单个受控消融或清晰机制，C = 只有原论文，D = 证据矛盾。
- **"上限"的两种含义必须分开：** pass@1 的渐近值（ScaleRL 的 A），和 pass@large-k 的覆盖率（Yue et al.）。
- 短路径：03 → 04 → 05 判定总图 → 23 → 30。长路径：全文。

### 03-skeleton.html — The skeleton: one gradient estimator, three knobs

- 目的：给全文一张"地图"，对应 infra 的五阶段 loop。
- 一次 RL 更新拆成七步，每一步对应一个设计轴：
  1. sample：由 π_sampler 生成数据。
  2. score：打分得到 r。
  3. baseline：计算 Â（轴 ①、②）。
  4. weight & mask：确定 w 和 m（轴 ④、⑥）。
  5. aggregate：选归一化方式（轴 ③）。
  6. regularize：KL / entropy（轴 ⑤）。
  7. step。
  
  数据与奖励（轴 ⑦、⑧）位于 loop 的入口。
- 统一模板 ∇J = Σ_i Σ_t w_{i,t}·m_{i,t}·Â_i·∇log π_θ(y_{i,t})。
- **核心恒等式：** 严格 on-policy 且训练与推理一致时 r≡1，此时 PPO-clip、DAPO、GSPO、GMPO、CISPO、SAPO 都退化为同一个 REINFORCE，只差归一化。所以 loss 之间的经验排序测的是 staleness 和 mismatch。
- 预览小表：把 GRPO / Dr. GRPO / DAPO / GSPO / CISPO / RLOO 映射到这三个旋钮（完整版放在 25）。
- **Fig 1**：估计量流水线图，每个箭头标出对应的轴号。
- 必须核实：GSPO / SAPO 在 r=1 时的梯度形式。要对照原文方法节推导一遍，因为报告中的推导出自笔记。

### 04-tension.html — Why the estimator is not enough: the measurement problem

- 目的：对应 infra 的 "Why the loop is not enough"。估计量讲清楚以后，还缺"怎么测出它好"。
- 四类混杂因素，各配一个量级：
  1. **评测噪声：** Pass@1 跨 seed 标准差 5–15；AIME 一题等于 3.3 点；只改评测温度最多差约 15 点（Hochlehnert, 2504.07086）。
  2. **Base、模板和污染：** Qwen2.5-Math 换模板，分数在 0 到 38.2 之间；随机奖励 +21.4；MATH-500 前缀补全 EM 54.6% 对 3.8%。
  3. **弱 baseline 和框架 bug：** VAPO 的 PPO 基线只有 5；同一个 DPO baseline 在两篇论文中是 15.1 和 20.6；2604.23747 报告的 bug。
  4. **算力记账：** 报告按步数而不是 GPU-hours；报峰值而不是终值。
- **上限的可辨识性：** ScaleRL 的 A 误差带为 ±0.02，B 没有误差带；2610.04158 发现 sigmoid 和 log-linear 在拟合窗口内无法区分。由此得出规则：A 的差异小于 0.02–0.03 时记为"未分辨"。
- **Fig 2**：横轴是各篇论文声称的增益，纵轴叠加噪声下界。直观展示 DAPO 逐项消融的 +1/+2/+3 落在哪里。
- 收尾句，引出 05："下面每个轴的判定都先过这四道筛子。"

---

## 2. 设计轴（对应 infra 的六个 primitives）

### 05-axes.html — Eight design axes worth judging

- 一段导语，加上 **Fig 3 判定总图**：8 个轴的若干改动排成行，列是四个维度，格子按 A–D 着色。读者翻到这里就能看到全貌。
- 说明 ⑨ credit assignment 和 ⑩ 多轮特有选择为什么放到 16、17 的"算法族"里：它们自带一套完整的方法族，按 infra 的 case study 形式写更清楚。

### 06-advantage.html — ① Advantage and baseline

- 公式：Â_i = (r_i − b(x))·w(x)。
- **恒等式：** RLOO = Dr. GRPO × G/(G−1)。group mean 因为包含 r_i 本身，带 O(1/G) 偏差。KRPO 的 baseline 同样包含 r_i。
- **GAE：** λ<1 时，终点奖励传到 token t 要乘 λ^{T−t}；λ=1 时 critic 只是 state-dependent baseline。VAPO 的长度自适应 λ 有效视界约 400 token，因此它依赖 critic bootstrap。
- 判定：
  - λ=1：**A**。
  - critic 不从 RM 初始化：**B**。
  - "调好的 PPO 对 GRPO / RLOO"：**D**。原文报 +10；GenAC 中差距 ≤1（3 seeds 下为 50.85 对 49.96）。
  - critic-free 的真正收益是成本（约 0.6×），不是准确率。
- 前沿实践：从 critic 转向 group mean，并逐步去掉 std；Seed 是唯一还在用 critic 的前沿团队。
- What's interesting：critic 何时从"降方差"变成"引入偏差"——取决于 λ 和有效视界。
- 必须核实：GenAC 是否为每个方法单独调参；DeepSeek-R1 附录 A.3 的原句。

### 07-normalization.html — ② Normalization: std changes the objective, not the ceiling

- **隐式目标表：** w(p)=1 对应 Σp；GRPO 的 1/√(p(1−p)) 对应 Σ 2·arcsin√p；MaxRL 的 ≈1/p 对应 Σ log p；pass@k 对应 Σ 1−(1−p)^k。
- 用 G=8、只有一个成功样本的组举例：GRPO 给出 +2.65 / −0.38，Dr. GRPO 给出 +0.875 / −0.125。
- **batch std 是一个全局标量**，在 Adam 下基本被吸收。
- **把 std 争论拆成两半：** 抬高难 prompt 的权重有依据；抬高易 prompt 的权重没人辩护，而且是易数据上不稳定的来源。
- 判定：上限中性 **A**；效率方向 **D**；非对称加权 **C**。
- **Fig 4**：四条 w(p) 曲线画在同一张图上。
- What's interesting：每种归一化都等价于选了一个关于 p 的凹变换。
- 必须核实：2607.00152（Group-Std Identity）和 2601.23135 目前都只读了摘要，正文使用前必须读全文。

### 08-aggregation.html — ③ Loss aggregation: the 1/|y| length bias

- 四种聚合方式的逐 token 权重。seq-mean-token-mean 会让错误答案越来越长（Dr. GRPO 的机制）。
- **实现分歧：** DAPO 的 token-level 是在 prompt 组内归一化；verl 的 `token-mean` 是按整个 global batch 归一化。
- Qwen MiniRL 的解释：token 级目标是序列级奖励的一阶近似，长度归一化会破坏这个近似。
- 判定：去长度偏置的机制 **B**；"token 级更好"作为经验结论 **D**（结论在 base 与 aligned 模型之间翻转；ScaleRL 偏好 prompt 级）。
- 本节最短；用一张权重对照表就够。

### 09-trust-region.html — ④ Clipping, trust regions, and ratio granularity

- 本节是全文的中心之一，也是命名算法最多的轴。
- **Fig 5（ratio × advantage 四象限图）：** 标出 PPO-clip、dual-clip、clip-higher、CISPO、GSPO、IcePop 各自在哪个区域把梯度置零或截断。重点标注 CISPO 重新打开的 Â<0、r<1−ε 区域。
- clip 本身带 entropy 偏置：clip-low 提高 entropy，clip-high 降低 entropy。由此解释 clip-higher。
- 证据：
  - clip-higher：作为 entropy 机制 **B**，作为普遍收益 **D**。
  - CISPO 的排名在 ScaleRL 与 CE-GPPO、ARLArena 之间相反（**D**）；还有一个实现陷阱：verl 默认把权重截断在 [0.8,1.2]。
  - GSPO：只在 MoE 上验证过；Qwen 自己的受控研究没有跑它。
  - 新 clip 变体（DCPO、CE-GPPO、ASPO、DPPO 等）都记 **C**。
- 推论："对哪种模型有效"实际上是"在多大 staleness 下有效"的代理变量。
- What's interesting：GMPO 在代数上等于"带 token 级 log 空间 clip 的 GSPO"；SAPO 的 sech² 核是 clip 指示函数的平滑版本。
- 必须核实：GSPO、GMPO、SAPO 的默认 ε 和归一化；2610.04928 "clipping is all you need" 目前只读了摘要。

### 10-kl-entropy.html — ⑤ KL and entropy regularization

- **KL 估计器梯度表：**
  - k1 作为 loss 的期望梯度为 0，没有正则作用。
  - k3 作为 loss（GRPO 默认）有偏。
  - k2 作为 loss，或 k1 放进 reward，梯度正确。
  - 低 KL 区间这些差别可以忽略；高 KL 区间（如 on-policy 蒸馏）k3 会卡住。
- **把"要不要 KL"拆开：**
  - 推理 RLVR 中锚定到 base 的 KL：**B**，去掉安全（Skywork 中 β=1e-3 让 AIME 停止提升）。
  - 近端 KL 或周期重置 reference：**C**。
  - 学习型 RM 场景下 KL 仍然承重：**B**（InstructGPT；N+ details；Gao 指出它的作用类似 early stopping）。
- **Entropy：**
  - 一步更新的 entropy 变化 ≈ −η·Cov(log π, π·Â)。
  - 固定系数的 entropy bonus：**A**（负面结论）。
  - 每批 rollout 多次 SGD 导致 entropy collapse：**A**（Skywork、AceReason、Magistral 三组独立研究）。全文最硬的结论之一。
  - entropy 能否预测上限：**D**。
- 交叉引用 28 Pitfalls：slime 开启 KL 后默认 k1-as-loss；SkyRL 默认开启 k3 loss。
- What's interesting：RL's Razor 的定理与它的预测指标使用的 KL 方向不一致。

### 11-off-policy.html — ⑥ Off-policy and train–inference mismatch

- 与 infra blog 的 #async、#quantization、#pitfalls（第 1、5、6 条）互链。本节只讲**估计量层面**，不重复系统机制。
- **四个 off-policy 来源：**
  1. 数值差异。
  2. 多 minibatch 更新。
  3. 异步训练。
  4. MoE routing 翻转（Qwen3-30B-A3B 中 94% 的 token 至少在一层上不同）。
- **关键缺陷：** PPO 的 ratio 从来看不到 π_sampler，所以 mismatch 完全没被修正。
- **修正谱系：**
  - token TIS：剩余偏差 O(T²·D_TV)。
  - 序列 IS：方差随长度指数增长；在 16k 长度下 log ρ 的 std 约为 7。
  - 几何平均或 K3 判据：本质是长度无关的 trust-region 检验，不是 IS 修正。
  - 截断 vs 掩码。
  - CIS 的结论：下侧更敏感，把小权重往上抬是灾难性的。
- **非对称原则：** 负优势且高度 off-policy 的 token 是梯度中危险的部分（TOPR、OPSM、M2PO）。
- **何时决定性、何时二阶：** 在 MoE、长上下文、多轮、低精度 rollout 下是决定性的；在 dense 短上下文数学 RL 中是二阶的。很大一部分崩溃其实是实现或硬件病理（vLLM cascade attention；L20 与 H20 的差异）。
- 判定：
  - 从源头消除 mismatch：**B**。
  - R3 / Keep Routing：**A**（MoE，多 minibatch）。
  - token TIS：**D**。
  - FP32 head：**D**。
  - decoupled PPO + staleness 上界：**B**。
- **Staleness 预算应当写成 S·η，而不是固定步数**（2607.01083 摘要称最大稳定 LR ∝ S⁻¹，必须核实）。
- **隐性 off-policy：** verl 默认 1024/256，即每批 4 次更新。
- **Fig 6**：off-policy 来源 × 修正方法的矩阵。
- 必须核实：2607.01083 全文；FP16 论文的 base model 不一致；ScaleRL 附录 A.11 的原句。

### 12-data-sampling.html — ⑦ Data and sampling

- **能移动 A 的因素：** batch（512→2048 时 A 从 0.605 到 0.645），生成长度（14k→32k 时从 0.610 到 0.645，但 C_mid 增至 4.4 倍）。这是"上限换效率"的清楚例子。
- zero-variance 过滤：稳定性 **B**，准确率 **D**（DAPO 报 +8；JustRL 不做过滤也达到 SOTA；ScaleRL 正文与 LOO 矛盾）。
- 难度过滤：**B**（效率）。但难度标签很噪，8 个 rollout 判出的难度很粗。
- 组大小 G 是二阶选择；"巨大 N 能突破平台"记 **C**。
- overlong / 截断处理：**D**。截断率超过 10% 作为失稳预警：**B**。
- 长度课程：**D**，依赖 base。温度：避免低温 **B**，随训练升温还是降温 **D**。
- 精选小数据：在 Qwen-Math 上 **B**，作为一般结论 **D**。

### 13-reward.html — ⑧ Reward signal and verifiers

- 验证器的精度比召回更重要。召回不足（TinyV 发现 38.5% 的假阴性）可以修，修复后 +2–3；精度一旦被 hack 就崩溃（单个 "{" 攻击）。判别式验证器更稳。**B**。
- 前沿数据实践：剔除可猜中或可 hack 的题型；RM 阶段放在晚期且短。
- 无监督奖励（entropy 最小化、TTRL、Intuitor）：只能锐化强先验，然后崩溃。**B**。
- Rubric 奖励：**C**。评测与训练格式一致的问题要点明。
- Reward hacking 随优化压力上升（CoT monitorability；inoculation prompting）。这两篇只读了摘要，只能作为背景一句话带过。
- 本节以"哪种上限"为过渡，引出 14。

---

## 3. 损失函数之下（对应 infra 的 "layer beneath"）

### 14-what-rl-changes.html — The layer beneath the loss: what RL actually changes

- **锐化与扩展：**
  - Yue et al.：pass@1 上升，pass@256 下降；k=1024 时，只有 base 能解的题占 13.3%，只有 RL 模型能解的占 0%。
  - ProRL：在 base 较弱的任务上扩展了边界。
  - Invisible Leash：48 次支持集扩展，175 次收缩。
  - distillation 在所有 k 上都高于 base。
- **统一读法：** 只要模型的众数行为好于平均行为，任何集中概率质量的更新都会显得有效。clipping 不对称、entropy 最小化、多数投票、format 奖励都属于这一类。
- **Fig 8**：pass@k 曲线交叉的示意图。
- 必须核实：Yue et al. 和 Invisible Leash 的数字目前只读过摘要器转述版（〔全文*〕），需要回原表核对。

### 14b-base-model.html — Base models, mid-training, and the ceiling outside RL

- 模板、污染、Qwen 与 Llama/OLMo 的分化（随机奖励、1-shot、TTRL 在非 Qwen 模型上都失败）。
- OctoThinker 的 mid-training，以及 Cognitive Behaviors 的 priming（摘要），把这个差距补上。
- **对 blog 的意义：** 覆盖率上限由 pretraining、mid-training 和 distillation 决定；RL 主要决定 pass@1 的渐近值。

---

## 4. 案例研究（对应 infra 的 14b–21）

三个**算法族**（各自带判定表）：

### 15-preference.html — Family case study: DPO and the preference-optimization family

- 四个旋钮：link 函数、隐式奖励归一化、似然锚、数据分布。
- **主导性排序：**
  1. 数据的 on-policy 程度：WPO 中换数据 +17.3、换 loss +3.8；semi-online DPO ≈ online ≈ GRPO（**A**）。
  2. 标注质量。
  3. PPO 与 DPO 之差（平均 +0.7）。
  4. RM 规模（基本可忽略）。
- 调好 β 后 loss 基本打平：**A**（GPO；Gorbatovski）。真正有差别的是长度归一化、off-policy 下的 NLL 锚、对比度。
- SimPO 和 KTO 记 **D**。Xu（PPO > DPO）与 Llama 3（DPO > PPO）可以同时成立，区别在数据范式。
- 2025–26 的角色：OLMo 3 把 DPO 用作 SFT 与 RLVR 之间的"对比注入"；Llama 4 只做轻量打磨。
- 必须核实：约 20 篇只读了摘要（IPO、ORPO、CPO、RSO 等），判定表中凡依赖它们的行都要补读；SmolLM3/APO 是先验记忆，未核实。

### 16-prm-credit.html — Family case study: process rewards and dense credit assignment

- 两族方法数学性质不同：on-policy MC credit 无偏，只降方差；学习型 PRM 注入有偏的代理信号。
- 学习型 PRM 求和作奖励必被 hack：**A**（四个独立崩溃案例）。
- 效率提升 2–9 倍：**B**。终值增益 0–3 点，随模型规模和上下文缩小：**D**（SPO 在 2K 上下文 +11.6，在 32K 上 +0.8）。
- 机制：Delta 变换就是 potential-based shaping；PRIME 求和后回到 outcome 分数；"GRPO is secretly a PRM"。
- 前沿实践：DeepSeek-R1 放弃 PRM 的三条理由；Kimi 不用 value network。
- 读者要带走的一句话：dense credit 在长 CoT 下不是瓶颈。

### 17-agentic.html — Family case study: agentic multi-turn RL

- 与 infra 的 #multi-turn 互链。那边讲 turn loop 的实现，这边讲算法判定。
- 正确性约束决定成败：环境 token 屏蔽与 TITO（**B**）；void-turn 轨迹过滤（**B−/C**）。
- Turn-level credit：GiGPO 记 **B−**，但它依赖环境状态精确重复；MT-GRPO 和 VAGEN 记 **C**。
- ratio 粒度、多轮中 PPO 对 GRPO：都记 **D**。
- **同设定下的非算法差距远大于算法差距：** 检索器质量（0.241 对 0.430）、冷启动数据（4.4 对 33.2）、测试时上下文管理（+7–16，**A−**）。
- 前沿实践：group baseline + ratio 掩码 + staleness 上界 + 截断过滤 + 预算惩罚。

五个**证据案例**（每篇 1–2 千词，各教一种读论文的技能）：

### 18-case-scalerl.html — Case study: ScaleRL — what a ceiling-vs-efficiency study can and cannot tell

- 实验协议、sigmoid 拟合、LOO 的设计。
- 它的四处内部矛盾：off-policy 算法、zero-variance 过滤、loss 类型、FP32 head。
- 推断：单项比较中 DAPO 的 A 偏低，可能是不稳定被拟合成了更低的上限。
- 教的技能：分清"抬上限"和"提效率"，以及 A 的可辨识性问题。

### 19-case-dapo-table.html — Case study: re-reading DAPO's ablation table against the noise floor

- 用 04 的噪声下界逐行重读 DAPO 的累加消融表；再以 JustRL（简单 recipe）和 OLMo 3（overlong 过滤无一致收益）作对照。
- 教的技能：累加式消融的顺序依赖问题，以及单 seed 下的结论下界。

### 20-case-weak-baseline.html — Case study: anatomy of a vanishing 10-point gain (VC-PPO / VAPO vs GenAC)

- 基线 PPO 只有 5 分的原因是 λ=0.95 加上从 RM 初始化 critic。GenAC 在只换估计器的条件下，差距约为 0。
- 教的技能：先检查 baseline 是否落在已知的坏配置上。

### 21-case-spurious.html — Case study: spurious rewards on Qwen2.5-Math — anatomy of a confounder

- 现象：A 级（Qwen-Math + MATH-500）。机制：D 级（Shao 认为是 clipping 偏置，2512.16912 认为 clip 项至少小 17 倍）。
- 污染证据，以及 OLMo 3 的阴性对照。
- 教的技能：要求论文提供随机奖励对照、干净的评测集、多个 seed。

### 21b-case-frontier.html — Case study: what frontier reports actually do

- 对应 infra 的 DeepSeek V4 案例。逐家列出：DeepSeek R1 → V3.2 → V4、Kimi k1.5 / K2 / K2.5、MiniMax-M1 / M2、Magistral、OLMo 3、Qwen3、GLM-4.5 / 5。
- 每家在八个轴上的选择，以及**哪些选择有消融、哪些只是声明**。
- 必须核实：DeepSeek-V4 的 arXiv ID 前缀 2606 与首次提交日期 2026-04 不一致；V4 最终阶段是 RL 还是 OPD；Kimi K2.5、GLM-5、Qwen3.5 在长报告后半部分是否有消融（笔记只读了前约 100k 字符）。

---

## 5. 收束部分（与 infra 22–32 同位）

### 22-recent.html — Recent results (2026)

- GenAC：估计器的受控比较。
- CIS：多 seed 的 mismatch 修正比较。
- VeXact：零 mismatch 的参考引擎。
- 2610.04158：上限 A 的可辨识性。
- 2607.01083：S·η 定标律。
- 2607.00152：Group-Std Identity。
- MaxRL：最大似然加权。
- 每条两三句。所有标〔摘要〕的条目，正文使用前都要补读。

### 23-patterns.html — Recurring patterns — the algebraic view

六个模式，各配一个公式和一句话：

1. **Collapse to REINFORCE**：r≡1 时所有 surrogate 等价，差异只来自 off-policy。
2. **Prompt weighting is an implicit objective**：w(p) 对应 Σf(p)。
3. **Sharpening**：众数好于均值时，任何集中概率质量的更新都显得有效。
4. **The dangerous quadrant**：负优势且远离 on-policy 的 token。
5. **Bounded, non-accumulating proxies**：可 hack 的代理信号只能有界、伸缩或只作过滤，不能进入 return。
6. **Data on-policyness dominates loss shape**。

综合：正确性 > 估计量 > 名字。这与 Engstrom 2020 是同一个结论。

### 24-code.html — Reading real code: one loss function, ten config flags

- 摘录 verl `core_algos.py` 中的 `compute_policy_loss` / `agg_loss` 及其 config。每个 flag 对应一个轴：`adv_estimator`、`norm_adv_by_std_in_grpo`、`loss_agg_mode`、`clip_ratio_low` / `high` / `c`、`use_kl_loss` / `kl_loss_type`、`entropy_coeff`、rollout correction、`ppo_mini_batch_size`。
- 再给一张框架默认值对照表：verl、slime、SkyRL、OpenRLHF、AReaL、Miles。数据来自 2026-05 的本地 checkout，开写前要 `git pull` 后重查。

### 25-landscape.html — The algorithm landscape

- 算法 × 设计轴对照表：从报告中的 45 行精简到约 25 行。
- **Fig 9 时间线**：两条带。一条是估计器的简化（critic → group mean → mean-only），一条是复杂度向 trust-region / IS / 数值层的转移。
- 时间线中标〔先验〕的一句话描述要抽查。

### 26-beyond.html — Beyond verifiable single-turn reasoning

- RLHF 与学习型 RM（KL 仍然承重）；rubric 与 LLM judge 奖励；on-policy distillation（DeepSeek-V4 的多教师 OPD）。
- **研究缺口：** OPD 不在本轮 8 个调研方向里，写这一节之前需要补一轮调研。多模态 RL 只写一句话或不写。

### 27-scaling.html — The scaling chain

- 哪些因素移动 A（batch、生成长度、模型、数值一致性），哪些只移动 B。
- ScaleRL sigmoid 的读法；staleness 写成 S·η 预算；截断率作为扩展时的预警信号。
- **Fig 7**：A / B / C_mid 的示意图。

### 28-pitfalls.html — Pitfalls — 踩坑录 at the algorithm layer

和 infra 的 28 互补：那边是系统 bug，这边是"数学上悄悄错了"。

1. slime 开启 KL 后默认 k1-as-loss，等于没有 KL。
2. SkyRL 默认开启 k3 loss。
3. verl 的 CISPO 默认把权重截断在 [0.8,1.2]，原论文无下界。
4. verl 的 `token-mean` 按 batch 归一化，不同于 DAPO 的 prompt 级。
5. verl 默认 1024/256，"同步"训练实际每批 4 次更新。
6. OpenRLHF 的 batch 均值按 token 加权，会引入与长度相关的偏移（笔记推断，需复核）。
7. TRL 的 IPO 曾把 loss 写成求和而不是平均。
8. DeepSpeed CPU-offload 和 OpenRLHF 的聚合 bug 拖低了 SFT baseline（2604.23747，只读了摘要）。
9. 多轮场景下重新套 chat template，导致 token 不一致。
10. vLLM cascade attention 造成 mismatch。
11. 评测温度和 top_p 设置不同。

每条格式：现象 → 数学后果 → 如何检测。

### 29-checklist.html — A researcher's checklist — what to verify before trusting an RL algorithm result

- 报告中的 12 项检查清单。和 infra 一样按"影响频率"排序，前三项最重要：base model、seed 与噪声、baseline 调参。
- 结尾句式对齐 infra："A paper that doesn't answer the first three…"

### 30-decision-guide.html — A decision guide — what to set, in what order

- 这一节是标题里 "decision" 的落点。
- **四档：**
  1. **必须做对（A/B 正确性）：** 每批少做 off-policy SGD；MoE 用 routing 一致性；从源头消除 mismatch；多轮屏蔽环境 token 并保证 TITO；保证验证器精度；PPO 用 λ=1；推理 RLVR 去掉对 base 的 KL；不加固定 entropy bonus。
  2. **便宜、中性的默认值：** group mean（是否加 batch std 都行）；去掉 1/|y|；zero-variance 过滤；温度约 1.0；截断率监控。
  3. **按模型调的旋钮（D）：** clip-higher 的 ε_high；聚合粒度；overlong 处理；长度课程；温度调度。
  4. **可以忽略（C / folklore）：** 绝大多数新 clip 变体；"CISPO / GSPO 普遍优于 PPO"；turn-level credit（开放环境中）。
- **Fig 10 决策树：** dense / MoE × 同步 / 异步 × 可验证奖励 / 学习型 RM × 单轮 / 多轮 × base / aligned 起点。
- 偏好学习分支：先把数据做成 on-policy，再选 loss（LN-DPO 或 DPO，加 β 扫参）。

### 30b-experiments.html — Twelve small experiments — how to do this yourself

- 对应 infra "how to do this yourself" 的位置，直接用报告中的 12 个实验。
- 每个实验写四项：要改变哪条判定的等级、设计、判别标准、预估算力。
- **统一协议：** Qwen 加至少一个非 Qwen 模型族；≥5 个训练 seed；干净的评测集；按 GPU-hours 记账；加随机奖励阴性对照。
- 标出优先级：#1（检验 r≡1 恒等式）、#4（A 的可辨识性）、#2（跨模型族网格）最能改变全文结论。
- 若之后真的跑了，结果回填到对应的轴小节，这一节改写为"what we found"。

### 31-open.html — Open questions at the algorithm–theory boundary

1. 上限 A 的可辨识性与置信区间。
2. token 级 IS 偏差的紧界：现有结果是 O(T²·D_TV)，TRM 声称可以做到 O(T^{3/2})。
3. S·η 定标律的理论解释。
4. 应该选哪个 f(p)：arcsin√p、log p、还是 pass@k。
5. 锐化与扩展的形式化：RL 何时能扩展支持集。
6. 代理信号的安全性：什么时候过程奖励等价于 potential-based shaping。
7. Entropy 的协方差动力学，以及 collapse 是否可预测。
8. KL 方向问题：RL's Razor 的定理与预测指标不一致。

### 32-reading.html — A reading list for the theoretically inclined

约 20 篇，分四组：

- **受控研究：** ScaleRL、Tricks or Traps、Dr. GRPO、Engstrom、Andrychowicz、Sober Look。
- **机制与理论：** Entropy Mechanism、Tang & Munos KL、TOPR、RL's Razor、Gao overoptimization。
- **off-policy：** Yao TIS、Liu & Li、R3、Qwen MiniRL、AReaL。
- **算法族：** Ivison、Lanchantin semi-online、GPO、PURE、SPO、GiGPO、Search-R1。

### 99-article-footer.html

- 与 infra blog 互链；注明证据截止日期（2026-10）和版本号。

---

## 6. 图表清单

| # | 位置 | 内容 |
|---|---|---|
| Fig 1 | 03 | 估计量流水线图，标出各轴号 |
| Fig 2 | 04 | 各论文声称的增益 vs 噪声下界 |
| Fig 3 | 05 | 判定总图（改动 × 四维，按 A–D 着色） |
| Fig 4 | 07 | 四条 w(p) 曲线 |
| Fig 5 | 09 | ratio × advantage 四象限，标出各 clip 变体的作用区域 |
| Fig 6 | 11 | off-policy 来源 × 修正方法的矩阵 |
| Fig 7 | 27 | ScaleRL sigmoid 中 A、B、C_mid 的含义 |
| Fig 8 | 14 | pass@k 曲线交叉示意 |
| Fig 9 | 25 | 两条带的时间线 |
| Fig 10 | 30 | 决策树 |

## 7. 开写前的核实闸门

每一节动笔之前，先把该节论证所依赖、但标为〔摘要〕〔片段〕〔先验〕〔全文*〕的数字回原文核对。各节"必须核实"条目已列在上面。全局性的几项：

- **ScaleRL：** 正文与附录 A.11 的原句；Fig. 14b 中各归一化变体的数值（只在图里）。
- **跨节通用的几篇：** Yue et al.、FP16、R3、AReaL 的数字目前都经过摘要器转述，要回原表。
- **框架默认值：** 各框架 `git pull` 后重新 grep（现有结果来自 2026-05 的 checkout）。
- **DeepSeek-V4：** ID 前缀与提交日期不一致。
- **补一轮调研：** on-policy distillation（26 节需要）。

## 8. 建议的写作顺序

1. 03 Skeleton + 04 Tension：这两节定下全文框架，也决定 Fig 1、Fig 2。
2. 09 Trust region + 11 Off-policy + 23 Patterns：r≡1 主线所在。
3. 其余轴（06–08、10、12–13）。
4. 14 / 14b 和三个算法族（15–17）。
5. 五个证据案例（18–21b）。
6. 收束部分（22, 24–32），最后写 01 / 02 / 00。开篇最后写，确保结论与正文一致。

## 9. 已核实的更正（2026-10-14，回原文核对 03/04 所用数字）

详见 `research/notes/verification_03_04.md`。会影响后续章节的几条：

- **ScaleRL 生成长度（二次核实后订正）：** Table 1 是 "ScaleRL-32k"，A 0.610→0.645，C_mid 2542→11272；"34k" 只在附录 A.15 和图 9 图例中出现一次。以 Table 1 为准。
- **ScaleRL batch：** 512/768/2048 对应 A = 0.605/0.610/0.645，同样只在图中（Fig. 10a）。
- **FP32 LM head 0.52→0.61：** 是在 DAPO-loss baseline 上测的（§3.2, Fig. 5b），不是完整 recipe。
- **VC-PPO / VAPO 的 PPO baseline：** VC-PPO 是 5.6（AIME pass@1），VAPO 是 5（AIME24 avg@32），两篇论文、两种指标，不能合写成"约 5"。
- **SimPO 15.1 与 WPO 20.6：** 不是同一个 SFT checkpoint（SFT 起点 LC 8.4 vs 9.5），只能说"同一配方"。原报告"5.5 点大于 SimPO 声称的优势"未经核实，已删。
- **CISPO 的 ε_high：** MiniMax-M1 没有给出具体数值，只说下界实际关闭、上界调过。
- **Sober Look "5–15 点 seed std"：** 是跨全部模型和 benchmark 的范围；AIME24 单独是"最高 15%"。另外，它是评测 seed 的方差，不是训练 seed 的方差，正文已分开写成三层噪声。
- **2610.04158：** "R² > 0.99" 是平均值（Fig. 5 图注），逐 benchmark 为 0.845–0.987。
- **新增事实（用于 03）：** 提出这些 surrogate 的论文自己的实验都不是 r ≡ 1：DAPO 每个 rollout step 16 次更新，MiniMax-M1 16 次，GMPO 8 次，GSPO 和 SAPO 每批 4 个 minibatch。
- **Gao et al. 2410.15115（写 16 PRM 节时用）：** 30.58 是 SR-only 训 5 epoch，11.16 是 SR+PR 训 1 epoch，base 是 24.90；不是同一次运行里"从 30.58 掉到 11.16"。"The answer is" 只出现在附录的人工构造探针里，不是观察到的 hacking 行为。Clip/Delta 修复后最好 31.44，只比 SR-only 高 0.86，单次运行。
- **WPO 的 "hybrid" 数据不是 on-policy：** 5 条 SFT 样本 + 1 条 gpt-4-turbo 输出，并由 gpt-4-turbo 重新标注；数据效应 +16–17，loss 效应 +1.6–5.6。
- **Tricks or Traps 的 ε_high：** 只有 aligned 8B 非单调（0.28 最好）；aligned 4B 在 0.32（网格上限）之前单调上升。
- **ARLArena：** 崩溃原因诊断做在 SAPO 上，不是 CISPO；sequence masking 的修复对两者都有效。
- **AceReason：** 是"每批 2 或 4 次梯度更新"导致约第 100 步 entropy collapse，不是 22 或 44 次；每批 1 次避免 collapse。
- **AReaL 速度：** 正文只给 2.77×（1.5B）；1.64× 是由 32B 编码模型的表格训练时长算出来的，不是论文原话。
- **R3：** "ratio>2 的 token 比例超过 0.1 即崩溃"只是单次运行的描述，不是论文给出的阈值。
- **KL 方向命名：** Tang & Munos 把 KL(π_ref‖π) 叫 reverse KL，2510.01555 和 Comedy 把 KL(π‖π_ref) 叫 reverse KL。全文统一不用 "reverse/forward"，直接写方向。
- **DeepSeek-R1 的 KL 细节（β=0.001、k3、每 400 步重置 reference）只在 arXiv v2 / Nature 版里有，引用 v2。**
- **新坑（28 节）：** slime 打开 `--use-kl-loss` 而不改 `--kl-loss-type` 就是 k1-as-loss（期望梯度为零）；OpenRLHF 会打警告，slime 不会。
- **verl `token-mean`：** 除以的是每个 PPO mini-batch（一次优化步）的有效 token 数，跨 DP 求和；既不是整个 rollout batch，也不是 prompt 组。此前大纲和报告写的"global batch"不准确。
- **VC-PPO 5.6/29.4/41.9 来自 8k 上下文消融；16k 主结果是 VC-PPO 48.8 vs GRPO 38.9。论文对默认 λ_actor 的说法前后不一致。**
- **LOOP：** 9 点是 best checkpoint（71.3→61.9）；按 3 次运行均值算是 6.8 点。
- **1-shot RLVR：** Llama-3.2-3B 的 17.5→19.0 是 6 个 benchmark 平均；MATH500 上是 40.8→45.8。
- **RaR：** 不给参考答案时仍有 31.1–34.2（Likert 基线 23.9），所以参考答案只解释部分增益，不是大部分。
