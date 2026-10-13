# 事实验证门控：Agent 长期记忆写入的 learned calibrated gating
## 最终方案定稿（基于三角度设计与对抗攻击综合）

调研日期：2026-09-16。本文档由三个独立方法设计（theory-first / systems-first / evaluation-first）及各自受到的 novelty/feasibility/reviewer-2 三镜头对抗攻击综合而成。采纳决策依据攻击裁决：fatal/serious 攻击且无 fix 的部件淘汰；存活部件按互补性组装；每个采纳注明来源设计。

---

## 执行摘要（≤600字）

**最终方案核心**：一个可训练、经校准的事实验证门，在 agent 把候选事实写入长期记忆前，通过误差独立的验证信号（环境工具交互：检索 oracle、结构化 DB、一致性审计）产生可靠性估计，训练目标为反事实边际效用（写 vs 不写某条事实对下游 QA 的因果贡献），在查询分布移位下保持校准，解决现有方法"写入时打分无法预测未来效用"（PA-Bench）与"LLM 自评系统性膨胀"（LUCID）的结构性缺陷。

**骨架来源**：以 systems-first 为工程主线（验证工具集+门训练管线+成本核算），嫁接 theory-first 的理论部件（误差独立性形式化+污染动力学界），淘汰 evaluation-first（受到 fatal 攻击：样本量不足支撑校准、合成数据循环论证、全量基线低估）。

**关键采纳**：
- 验证信号生成（systems）：5 工具（检索/DB/计算/来源回查/记忆交叉核对）+ 本地 4B 门模型学习证据聚合，架构隔离 LLM 自评与记忆库（误差独立的构造性保证）。
- 反事实标签（systems）：离线跑无门控 agent，对每条候选构造 M 含 f vs M\\f 重跑查询，边际效用 = 准确率差；用证据归因（Fine-Mem）缩方差。
- 校准（theory+systems）：温度标定 + conformal risk control 给出 FDR≤α 保证（分布移位下报告违约幅度）。
- 污染定理（theory，降级）：EI 门 vs 相关门的污染界，作为 related work 的理论对比点，但主实验不以此为核心卖点（攻击指出线性模型假设脆弱、LUCID 重新包装）。

**最大淘汰**：evaluation-first 设计的 1500 事实规模（攻击：4B 模型需 10K-100K 才收敛、ECE 需≥300 样本、反事实评估 5M API 调用爆预算）；theory-first 的三个命题作为主卖点（攻击：Prop 1 是教科书 CRC、Prop 2 线性假设自认脆弱、Prop 3 被 CRC 覆盖降为细节）。

**剩余最大风险**：Grounding Agent Memory（2609.11060，6 天前）做了 training-free 版环境探测验证，正面撞车；我们的 learned+calibrated+反事实效用 三点组合差异化可辩护但审稿人可能不买账；唯一对冲 = 理论部件（污染界与 ρ-相关门 FDR 下界）+ 尽早挂 arXiv 占 priority（即使机制被撞，理论独立成立）；实验风险 = 误差独立性若实测 ρ̂ 不够小，推论钝化。

---

## 1. 核心主张（一句话）

候选事实写入长期记忆前，一个可训练、经校准的验证门基于环境交互的误差独立信号产生可靠性估计，以反事实边际效用为训练目标，在同等存储预算下降低记忆错误率并在查询分布移位时保持校准，解决"写入时打分无法预测未来效用"与"LLM 自评系统性膨胀"的结构性缺陷。

---

## 2. 方法（形式化 + 算法 + 训练信号）

### 2.1 形式化设定（来源：theory-first §一、二，简化）

候选事实 c = (subject, relation, value, time-scope, provenance)，潜在真值 y∈{0,1}。两类信号：
- **写入者自评** s(c)∈(0,1)：冻结 LLM（deepseek-v4-pro）的 P(y=1) 估计（verbalized confidence + k 次自一致采样），误差 ε_s。LUCID 已证：E[ε_s]>0（Echo Gap）且 ε_s 与记忆库内容相关（复利）。
- **验证信号** v(c)∈(0,1)：由验证动作产生、经本地小模型 G_θ 聚合的 P(y=1) 估计，误差 ε_v。
- **误差独立性（EI）**：给定 y，ε_v ⊥ ε_s，且 ε_v 与记忆库状态 M_t 独立（v 不读记忆库、不读写入者 rationale——架构强制）。EI 不是白嘴假设：**实测 ρ̂ = Corr(ε_v, ε_s) 并入主表**，理论结论以 ρ 为参数给出。

**写入决策**：g(c) = 1{ q(c) ≥ τ_α }，q(c) = σ( logit v(c) + λ·logit s(c) − logit π_0 )，π_0 为候选池先验真率，λ∈[0,1] 是对 s 的收缩系数（由 ρ̂ 与实测 Echo Gap 估计；λ=0 退化为纯验证门）。τ_α 由 conformal risk control 在 held-out 校准集上设定，目标 FDR≤α。

### 2.2 验证信号如何产生（来源：systems-first §2，误差独立的关键）

环境提供 5 个验证工具（证据全部是程序化/环境交互产物，不含 LLM judge）：
1. **检索器**：对环境文档库做 BM25（+可选本地嵌入）检索，证据特征 = 命中片段与候选的匹配分、支持/反驳片段计数。
2. **结构化数据库**：世界状态 SQL 查询，证据 = 精确匹配/矛盾/查无（覆盖率 ρ_db 受控，非全知）。
3. **计算器/Python**：数值型与派生事实重算，证据 = 数值相等性。
4. **来源回查**：调出候选事实的原始话轮，程序化比对措辞置信度漂移（hedge 词消失、量词强化）——直接针对 Manufactured Confidence（2606.29279）与来源洗白（2607.29167）的失败模式。
5. **记忆交叉核对**：与现有高置信条目的矛盾检测（内部工具，近零成本）。

**门 G_θ**：本地 Qwen2.5-4B LoRA，输入 (事实嵌入, 上下文, 证据特征包) → v(c)；**架构上不接触 s、写入者 rationale、记忆库内容**（EI 的构造性保证）；输出经温度标定。

**融合与三路决策**（来源：theory-first §一）：q ≥ τ_α 写入；q ≤ τ_rej 拒绝；中间进 quarantine（限额缓冲，下次遇到同实体时复验）——显式回应 GovMem 自认的保守/审查负担问题，quarantine 预算入成本报表。

### 2.3 反事实边际效用训练信号（来源：systems-first §3，误差独立的第二重保证）

**Ground-Truth-First 脚本先行协议**（仿 Veracium 2607.21962 + Nous 2606.22030 矛盾/可靠性协议，全程本地可跑）：
1. 先生成事实脚本：实体/属性/真值/有效期/来源通道（带可靠性参数）/注入矛盾与过时更新。
2. deepseek-flash 把脚本渲染成多会话交互流与一个可查询的合成世界（SQLite 事实库 + 带可配噪声与覆盖率的工具 API），验证动作因此有真实语义。
3. 写入时抽出的每条候选事实与脚本比对得到精确真值标签 y。

**反事实标签生产**：离线跑无门控 agent 写全部事实到记忆 M 并执行查询 Q 得准确率 A_write，再对每个事实 f 构造反事实 M\\{f} 重跑相同查询得 A_reject，标签 y(f) = A_write - A_reject 为该事实的边际效用；用证据归因（Fine-Mem）只重跑引用 f 的查询降方差；收集 {(事实, 上下文, V, 边际效用)} 数据集训练门控模型预测边际效用，训练后温度标定输出概率。

**误差独立性的三重保证**：
- 标签由脚本先于渲染而定，不经过任何 LLM 的真值判断，故与写入者（deepseek）的自评偏差在构造上独立。
- 门 G_θ 的监督全部来自这些脚本标签，且训练/推理时架构上隔离 s、rationale 与记忆库。
- 在 held-out 上实测报告 ρ̂ = Corr(ε_v, ε_s)，把 EIA 从假设变成被审计的性质；确定性检查器子集（时间一致性/算术/矛盾）单独成列证明 ρ=0 by construction（theory-first 风险§缓解）。

训练规模约 **10K-20K 候选**（feasibility 攻击：1500 不足以让 4B 收敛，需 10K+），deepseek-flash 批量渲染成本可控，G_θ 在本地 GPU LoRA 微调。

### 2.4 校准（来源：theory-first Prop 1 + systems-first）

温度标定 + conformal risk control（CRC）给出 **FDR≤α 保证**（分布移位下报告违约幅度，不过度承诺）：
- Prop 1(ii) 的分布无关版本：以 1−q 为损失、用 CRC 选 τ_α，则对 i.i.d. 校准/测试候选流，门控后记忆库满足 E[被写入条目中假条目比例] ≤ α（标准 conformal risk control，单调损失+交换性，工程零风险）。
- 分布移位协议：训练查询分布 A / 评测不相交分布 B，报告 ECE 在移位下的校准曲线（bins≥30 样本/bin，bootstrapped 置信区间），承认 CRC 保证限同分布、移位下为 best-effort。

---

## 3. 理论部件（来源：theory-first，降级为 related work 对比点）

### Prop 2'（污染动力学界，简化版）

设 p_t = 第 t 轮记忆库污染率，每轮写入流量占比 μ、更新/驱逐净化率 λ_c。自评门控系统的假写入率随污染耦合上升：b(p) = b_0 + κp（κ>0 即 LUCID 复利系数）。
- **自评/相关门**：p_{t+1} = (1−λ_c)p_t + μ(b_0+κp_t)，不动点 p* = μb_0/(λ_c−μκ)，且 μκ ≥ λ_c 时 p_t→1（污染发散）。
- **EI 门**：b(p) ≡ α 与 p 无关（EI 对记忆状态独立 ⇒ 假写入率不随污染上升），p_t 几何收敛到 p* = μα/λ_c。
- **相关判官下界（杀手推论）**：若门的信号与写入者误差相关 Corr(ε_v,ε_s)=ρ 且 s 继承记忆偏差，则有效复利系数 κ_eff ≥ ρκ，p* ≥ μb_α/(λ_c−μρκ)。含义：LLM-judge 门（高 ρ）**原理上**不能把污染压到相关地板之下；这是对 Mem0/A-MAC/MemGuard 类打分门的定理级批判。

**降级理由**（攻击裁决）：
- novelty 攻击：Prop 2(c) 的 ρ-相关门下界是 LUCID 相关性结果的代数重新包装，非新定理；"LLM-judge 不能压到相关地板"直接继承 LUCID，本科生级推导（theory-first 自认）。
- feasibility/reviewer2 攻击：线性模型假设自认脆弱（"实测可能偏离线性区"），且无法给出 κ、λ_c 的 closed-form 估计，实验只能事后拟合验证假设（若不成立则"可证明抑制"保证坍塌）。
- **采纳方式**：不作为主卖点，降级为 related work 的理论对比——在讨论"为什么 LLM-judge 门不够"时引用推论作为理论支撑，主实验以实证对照（LLM-judge 门 vs 本方法在长时程污染曲线上的差异）为证据，理论界仅作辅助论证。

---

## 4. 实验设计

### 4.1 基准矩阵

| 用途 | 基准 | 说明 |
|---|---|---|
| 主战场 | 自建脚本先行基准（工作名待撞名检查） | 含矛盾证据、可靠性分层来源、时间性更新、与训练查询分布不相交的下游 QA；**规模扩至 ≥10K 候选以支撑 ECE**（feasibility 攻击） |
| 外部榜（可选） | LongMemEval-S (knowledge-update/abstention 子类) | **依赖 GitHub 数据，列为"需用户侧下载"可选项**，主结论不依赖它 |
| 操作级 GT（可选） | HaluMem (抽取/更新幻觉) + MemOps gold traces | 同上；若不可得，自建基准按 Nous/PA-Bench 协议参数复刻其压力条件 |
| 泄漏控制（可选） | PA-Bench query-blind 协议 | 同上 |
| legacy 对照（绝不用于写入消融） | LoCoMo（写明子集口径） | Nous 消融已证无区分度 |

### 4.2 基线（全部自行复测，对等输入）

1. **全量存储+检索**：cloud-embedding（MemDelta 教训）+ rerank + 大上下文模型（128k 窗口），成本核算含嵌入存储；**加入预算扫描**：budget ∈ {10%, 25%, 50%, 75%, 100%} × {FIFO, LRU, RANDOM 驱逐}，画准确率-存储曲线（feasibility 攻击）。
2. **LLM-judge 门对照**：deepseek-v4-pro + **相同工具证据特征**（只是把 learned gate 换成 prompt），保证输入对等（reviewer2 攻击）。
3. **A-MAC 多因子**：复现其五因子准入（future utility、factual confidence、semantic novelty、temporal recency、content type prior），无校准版。
4. **被动 "unverified" 标签**：Manufactured Confidence 的失败干预，动机基线。
5. **未校准学习门**：同方法但去 CRC，裸 logits 阈值。

### 4.3 杀手实验（直接检验核心主张）

**长时程污染压力测试**：长时程多会话 agent 反复巩固与再巩固记忆，注入来源洗白（2607.29167）与 Manufactured Confidence 式置信改写条件，画四条污染曲线 p_t（无门/被动标签/LLM-judge 门/本方法），**叠加线性模型理论预测**（若实测拟合不佳则承认理论限 toy model，但实证对照仍成立）；同图给实测 κ、ρ̂ 与理论界的对照（检验 Prop 2 假设）。

**PA-Bench query-blind 复刻**：去掉"未来问题泄漏"后安全关键事实存活率（全记忆 97% vs 现有免训练 0-1%），本方法目标 >90%（evaluation-first 攻击）。

### 4.4 消融

- 去校准：裸 logits 阈值 vs 温度标定 vs CRC（α 扫描 → 实测 FDR 覆盖曲线，检验 Prop 1）。
- 验证信号换成 LLM-judge：检验推论 c 的 ρ-FDR 下界。
- 纯 v / 纯 s / 融合：λ 扫描。
- 验证预算扫描：成本-风险 Pareto。
- **分布移位**：训练查询分布 A / 评测不相交 B 下的 ECE 与 FDR（bins≥30 样本，bootstrapped CI）。
- 换 reader：deepseek-v4-pro→flash，证明门-reader 解耦。
- **learned 增益量化**（feasibility 攻击）：报 learned（4B/1.7B/LR 特征）三阶梯在主指标上的置信区间，若 LR baseline 在 CI 内与 4B 重叠则承认 learned 增益 limited to calibration。

### 4.5 协议卫生

- 同报 token-F1 + LLM-judge + 分歧率（Nous 27.5 分教训）。
- 报告评分对象（TIAP）。
- MERIT/Total Recall 口径的逐操作 token/美元与 break-even（验证成本 vs 污染损失）；**成本表分 one-time（训练+缓存构建）+ per-candidate（门推理+工具调用）+ per-query（检索），给 break-even 曲线**（feasibility 攻击）。
- 固定嵌入与 judge 模型、单变量对照（MemDelta 教训）。
- **误差相关性实证**（reviewer2 攻击）：报告 Pearson r 的 [n, 95%CI, p 值]，对照组须同；bootstrapping 残差相关性分布；ablate 抽取上下文 x（只给候选 c+证据 E），若性能不降，证明门未依赖 flash 措辞风格。

---

## 5. Novelty 防线（逐近邻一段）

**四点交集**（learned + calibrated + 误差独立信号 + declarative facts，每近邻至少缺两点）：

1. **Grounding Agent Memory（2609.11060，6 天前）**：training-free curator、限企业程序知识、无校准无理论。**我们**：可训练门 + FDR 保证 + 污染定理；其探针机制被收编为我们动作集的一个通道，且我们证明"为什么必须是误差独立信号"——它只做了 how，我们补 why 与 guarantee。**风险承认**：正面碰撞最危险，必须在相关工作正面处理并强调四点组合差异，标题避开"environment probing"改强调"calibrated"+"counterfactual utility"。

2. **MemGuard（2608.21867）**：verifier 是打分器（高 ρ，恰落入我们推论 c 的下界）、经验/轨迹记忆、无训练无校准；其元数据生命周期被我们以校准置信版收编为附件。

3. **GovMem（2607.02579，已接收 mlise 2026）**：规则式 writepath governance、training-free、自认保守与审查负担高；我们用可调 α 的 CRC 与限额 quarantine 把这个负担变成显式预算参数。

4. **TRUSTMEM（2606.25161）**：验证的是记忆转移对源对话的忠实性——忠实转移完全可以固化一条对世界为假的事实；我们门控的是对世界的真，正交且必须同引。

5. **LUCID（2608.00017）**：证明 Echo Gap 与 EIA 必要性，是记忆估值的去膨胀算法，无验证动作、无学习门、无写入 FDR/污染界；我们是其构造性对偶：以 EIA 为公理交付机制 + Prop 2（虽降级但仍独立）。

6. **A-MAC（2603.04549，ICLR'26 MemAgents WS）**："factual confidence" = LLM 单次打分（违反 EI）、无学习无校准。

7. **Voyager/AWM**：执行成功门控只对代码/技能成立；我们交付 declarative 域的等价物，这正是 00 报告指出的结构性空洞。

8. **SP-KV/TTCD/Tang 2608.11654**：层级（KV/fast-weights）或纯形式化，不触及事实验证；Tang 的 coverage-vs-precision 提法被我们实例化到 precision 侧，引用并区分。

9. **RL 写入策略系（Memory-R1/Mem-α/MemAgent/Fine-Mem/MemReader）**：已有 ACL/ICLR venue anchor，我们的差异 = 误差独立验证信号（他们 QA-reward 经 reader 间接传导）+ 校准（他们无）+ 反事实边际效用作直接训练目标（Fine-Mem 证据回溯是事后归因，非写入时前瞻）。

理论贡献本身的 novelty：记忆污染的动力系统界与 ρ-相关门 FDR 下界在整个撞车名单中无先例（虽被降级，仍独立成立）。

---

## 6. 风险与缓解

### 6.1 撞车窗口（最大生存威胁）

- **Grounding Agent Memory（2609.11060，6 天前）**：正面碰撞，窗口正在关闭。唯一对冲 = learned+calibrated+反事实效用 三点组合 + 污染定理 + 尽早挂 arXiv 占 priority（即使机制被撞，理论独立成立）。
- NeurIPS 2026 放榜在即（Memory-R2/CMI-Mem/ICML 等大概率升 venue），**必须抢在放榜前挂 arXiv** 否则 novelty 论证需重写。

### 6.2 实验风险

- **误差独立性若实测 ρ̂ 不够小**：推论 c 的实证对照会钝化。缓解 = 确定性检查器子集（ρ=0 by construction）单独成列；承认 4B 头的 ρ̂ 是经验量需实测，把 EI 从公理降为"architectural best-effort+audited"。
- **线性模型假设偏离**：Prop 2 的动力学模型是风格化，κ、λ_c 的实测可能偏离线性区。缓解 = 把模型假设作为压力测试中被检验的对象而非教条，实证对照（污染曲线）不依赖理论假设成立。
- **合成基准真实性质疑**：外部基准全部依赖用户侧下载，若下载不成主表只剩自建基准，外部效度弱。缓解 = 自建基准按 Nous/PA-Bench 协议参数复刻其压力条件，并在讨论承认局限。

### 6.3 工程风险

- **4B 基座权重本机不可达**：须用户侧一次性下载，阻塞 M2。
- **DeepSeek 推理型模型成本**：verbalized confidence 质量与验证动作 token 成本可能超预算。缓解 = 提前用小规模 pilot 测成本。
- **反事实标签噪声大**：证据归因缩方差 + 报灵敏度分析。

---

## 7. 三个月里程碑排期（周粒度）

| 周 | 里程碑 | 关键路径 | 需用户协助 |
|---|---|---|---|
| W1-2 | 脚本先行数据管线 + 合成世界 | deepseek-flash 批量渲染 10K-20K 候选 | **W1**：4B 基座权重下载（Qwen2.5-4B，约 8GB）|
| W2-3 | 全部基线跑通 + 判分管线 | 全量存储+检索（cloud-embed）、LLM-judge、A-MAC、未校准门 | 无 |
| W4-5 | 反事实标签收集 | 离线 agent 重跑 M 含 f vs M\\f，证据归因 | 无 |
| W5-6 | 门训练/校准 | 本地 GPU LoRA 微调 G_θ，温度标定+CRC | 无 |
| W7-8 | 主表 + Prop 1、消融 | FDR 覆盖曲线、分布移位 ECE、learned 增益量化 | 无 |
| W9-10 | 污染压力测试 + Prop 2 验证 | 长时程四条曲线、线性假设拟合优度 | 无 |
| W10-11 | 外部基准（可选）+ 成本核算 | **W10**：若用户可下载 LongMemEval-S/HaluMem/MemOps，此周跑；否则跳过 | **W10**（可选）：外部数据集下载 |
| W11-12 | 成稿 + arXiv | 初稿→内部 review→定稿；**W12 中旬前必须挂 arXiv**（抢在 NeurIPS'26 放榜前） | 无 |

**关键路径总时长**：12 周，slack 0 周（紧张）。**单点故障**：W1 权重下载（若延误 1 周，W4 标签收集可压缩 1 天补回）；W10 外部数据集可选（主结论不依赖）。

---

## 8. 与原调研报告 00 的差异

原报告 §5 主方案 = 条目级校准效用写入门（效用估计量+反事实边际效用），备选 = 事实验证门控。**最终采纳备选方向**，原因：
- 撞车窗口更紧迫（Grounding Agent Memory 6 天前，验证门控比效用估计量更直接面对竞争）。
- 验证门控的理论抓手（LUCID EIA + 污染定理）与失败模式动机（Manufactured Confidence/AuthMem-Bench）更强。
- 效用估计量方向的最强近邻（Fine-Mem ACL'26、Memory-R2、CMI-Mem）已做证据回溯/局部 rerollout/内在奖励，novelty 需要更细的划界。

但两方向在**训练信号**上收敛：都用反事实边际效用（原方案核心）；验证门控只是把"如何产生验证信号"这一步拉到前台作为主卖点，效用仍是最终训练目标。

---

## 附录：攻击裁决摘要（存档）

| 设计 | Angle | Novelty verdict | Feasibility verdict | Reviewer2 verdict | 存活判定 |
|---|---|---|---|---|---|
| theory-first | 理论优先 | fatal（10 攻击，含 LUCID 重新包装、MemAgent Oral 已占、线性模型脆弱、EI 哲学攻击、A-MAC 词面撞车） | - | - | 理论部件降级采纳，Prop 2 作 related work 对比点 |
| systems-first | 系统优先 | serious（8 攻击，含 Grounding 6 天前、MemGuard 元数据翻版、标题地雷） | - | serious（8 攻击，含合成循环、特征基线够用、EI 实证验证不清、conformal 移位违约、全量基线低估） | 工程主线，嫁接 theory 理论 |
| evaluation-first | 评测优先 | fatal（5 攻击，含 PA-Bench 泄漏、Grounding 6 天前、校准是工程非贡献） | fatal（5 攻击，含反事实标签 5M 调用、1500 样本不足 4B 收敛、oracle 循环或 9 周标注、基线重实现 7 周、基础设施 11-15 周） | fatal（样本量不足 ECE、合成循环、全量基线低估） | 淘汰（三镜头均 fatal/serious） |
