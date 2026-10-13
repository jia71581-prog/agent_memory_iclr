# 撞车扫描笔记（2026-09-17 执行）

任务：为 QC-Lifecycle（training-free 查询条件遗忘 + 生命周期跟踪；核心 = 查询时态分类器按 current/historical 条件化渲染事实演化链视图）扫描 2026-09 以来新竞品。

## 检索方法与口径
- 通道：arXiv export API（curl --noproxy，submittedDate 降序，每查询 sleep 3s）；OpenReview api2 notes/search 核实投稿记录。tavily/WebSearch 按指令禁用。
- 关键词 13 组：temporal memory、query-conditioned memory、memory lifecycle、supersession、knowledge update + memory、stale memory、forgetting + agent memory、conversational memory + temporal、temporal intent、memory view、fact evolution、training-free + forgetting、time-aware + memory。
- 所有 arXiv ID 均来自 API 返回原文，未编造。日期为 API 的 published/updated 字段。

## 已知竞品状态复查（2026-09-17 核实）
- **RD-Forget（arXiv 2609.10263）**：id_list 确认最新版仍为 **v1**（提交/更新均 2026-09-09），无 v2。OpenReview api2 按 "RD-Forget" 与全标题 "What Should an Agent Forget" 检索均 0 条，**未见公开投稿记录**。
- **MemoryLACE（arXiv 2609.03201）**：id_list 确认最新版仍为 **v1**（2026-09-02），无 v2。OpenReview 按 "MemoryLACE" 与 "Lifecycle-Aware Consolidation" 检索均 0 条，**未见公开投稿记录**。
- OpenReview 检索 "query-conditioned memory" 仅命中 1 条：ICML 2026 Submission1757 的官方评审，对象为 SCHEMAS（多智能体异构图框架），与本方向无关。

---

## 第一档：直接撞车

### 1. RD-Forget — What Should an Agent Forget? Separating What Is Stored from What Is Used
- **arXiv 2609.10263v1 | cs.AI | 提交 2026-09-09**
- 摘要要点：training-free 框架，分离"存什么"与"用什么"。保留完整 source archive；**query-conditioned memory view** 控制证据对当前回答的影响。冻结 LLM curator 抽取证据、按 semantic slot 分组、保留多跳关系。**same-slot replacement links 在 current-state 语境下抑制被 supersede 的旧值；intent-aware retrieval 使历史证据在历史查询下重新可用**。rate-distortion 公式在记忆预算内构建 answer-time view。
- 撞车判定：**直接撞车**。training-free、查询条件化视图、按查询时态意图切换 current/historical 可见性、supersession 链抑制——与 QC-Lifecycle 核心机制逐条重合。差异空间需在方法细节里找：其视图构建走 rate-distortion 预算化路线（QC-Lifecycle 若主打显式"事实演化链渲染 + 查询时态分类器"仍有区分点，但必须在论文中正面对比或抢时间）。此前已知，本轮确认无 v2、无 OpenReview 记录，即尚未见其投稿动作。

（本轮 2026-09-01 之后新提交中未发现第二篇达到"直接撞车"档的工作。）

---

## 第二档：近邻，必须引用/正面区分

### 2. MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval
- **arXiv 2609.03201v1 | cs.CL, cs.LG | 提交 2026-09-02**
- 显式建模文本证据生命周期（sparse merge / supersession / contradiction 关系），检索时重构 relation-aware evidence units，**同时暴露 current、historical、supporting、conflicting 证据**给下游推理。评测含 BEAM 等。
- 距离：生命周期跟踪端撞车（supersession 关系链≈事实演化链），但**不做查询时态条件化**——它把 current/historical 一并暴露，交给 reader 分辨；QC-Lifecycle 的 per-query 时态分类器视图是关键差异点。必引 + 正面对比。

### 3. Fortunate Recall: Ontology-Driven Memory Lifecycle Management (FR / FR-Bank / LifecycleBench)
- **arXiv 2609.10413v1 | cs.AI | 提交 2026-09-09**
- 10+1 类行为本体给个人事实分类，按类别施加生命周期策略：差异化时间衰减、**slot-key supersession、event-time validity、category-aware retrieval routing**，全部为 LLM 抽取元数据上的确定性函数。自建 **LifecycleBench**（516 题时间消歧基准），76.9% 超 Mem0/A-MEM/Memory-R1/MemoryOS（61%–70.5%）；LongMemEval-S 75.2%。
- 距离：近邻偏"存储侧策略"。生命周期管理 + supersession + 时间消歧全面重叠；但条件化对象是**事实类别**而非**查询时态意图**，无 per-query current/historical 视图切换。警惕点：LifecycleBench 可能被审稿人要求作为评测基准；必引。

### 4. MemPrism: Task-Conditioned Relational Memory Views for Long-Horizon Agents
- **arXiv 2608.06745v1 | cs.AI | 提交 2026-08-07**
- 分离持久经验存储与 decision-time working memory；按当前任务上下文**动态构建关系视图**（view policy 选择关系结构/证据范围/粒度 + 确定性 composer 渲染），frozen task policy 消费视图。评测在长程 embodied/web agent。
- 距离：机制近邻——"存储与条件化视图分离 + 确定性渲染"与 QC-Lifecycle 同构，但条件化维度是任务上下文而非查询时态，且不做 supersession/生命周期。必引，说明条件化轴不同。

### 5. StateMemBench: Can Agent Memory Systems Track Evolving State?
- **arXiv 2608.19652v1 | cs.AI, cs.CL | 提交 2026-08-20**
- 定义 state tracking：回答须反映 current state 而非 superseded state。234 个多会话场景，closed-pool 评分区分"答成当前态/答成被替代态/其他错误"。现有记忆系统与长上下文基线均表现差。
- 距离：评测侧近邻。其"current vs superseded"评分口径与 QC-Lifecycle 主张直接相关；可作外部基准或对照叙事，必引。

### 6. TEPA: Revoking Stale Memories for Conflict-Robust Language Agents
- **arXiv 2608.07429v2 | cs.AI | 提交 2026-08-07，更新 2026-08-10**
- 把 validity 变成记忆的显式状态：keyed precedents，同 key 新证据矛盾时 revoke 旧 precedent；检索只取 active 证据，**revoked 历史保留可审计**。
- 距离：生命周期端近邻（revocation≈supersession 的写时实现）。差异：写时改状态 vs QC-Lifecycle 的查询时条件化（TEPA 的历史保留仅"for audit"，非按历史查询意图重新供给回答）。必引。

### 7. STAIR: Semantic-Temporal Automaton for Interpretable Reasoning in Temporal QA
- **arXiv 2608.16224v1 | cs.CL, cs.AI | 提交 2026-08-17**
- answer-free LLM adapter 把复杂问题映射为 **normalized temporal intents**，确定性时间自动机在规范化证据上执行对应策略。
- 距离：组件级近邻——"查询→时态意图分类→确定性执行"与 QC-Lifecycle 的查询时态分类器同构，但场景是温度较低的 temporal QA（非 agent memory 生命周期）。建议引用以承认"查询时态意图分类"已有先例，把新颖性压在"意图驱动的记忆生命周期视图渲染"上。

### 8. Beyond Dialogue Time: Temporal Semantic Memory (TSM)
- **arXiv 2601.07468v2 | cs.AI | 提交 2026-01-12，v2 更新 2026-09-02**
- 语义时间线（按事实实际发生时间而非对话时间组织）+ durative memory（持续状态与演化模式）。
- 距离：近邻。时间维建模与"事实演化"叙事重叠，但无 supersession 条件化视图。9 月刚出 v2，注意其新版是否加了查询时态条件化（本轮仅见摘要，未见该表述）。

### 9. StateAuditor / STALE benchmark: When Memory Updates but Behavior Does Not
- **arXiv 2608.01619v1 | cs.AI | 提交 2026-08-03**
- STALE 基准的 IPA gap（记忆已更新但行为未跟上）；从 stored state 反向审计草稿，验证 provenance 与时间先后，触发修复。
- 距离：近邻（stale 依赖的读时修复）。与查询条件化视图互补而非相同；建议引用。

---

## 第三档：相关但不同轴（引用可选/按篇幅取舍）

- **2609.16730v1 LSREP + ICE v2**（2026-09-15）：会话记忆的纵向 lifecycle 评测协议（ordered replay + lifecycle schedules + evolving reference answers）。评测方法学相关。
- **2609.08279v1 What Eviction Destroys**（2026-09-08）：restore-counterfactual 审计遗忘的不可逆损失。遗忘评测侧。
- **2608.28978v1 Selective Forgetting graph memory**（2026-08-29）：图记忆+剪枝，负结果（图不敌 flat vector）。
- **2609.12354v1 CueMem**（2026-09-11）：cue→source-turn 重构证据上下文；无时态条件化。
- **2609.07093v2 MemLoc**（v2 更新 2026-09-14）：Retrieve-Localize-Generate 会话记忆 QA；检索工程侧。
- **2608.20685v1 MemStrata / Temporal Validity on Software Histories**（2026-08-21）：确定性 (s,r,o) supersession memory 在真实 GitHub 修复上消除 stale-fact；领域限定 code assistant。supersession 机制相关，可引。
- **2608.25553v3 When Stale Constraints Go Unchecked**（2026-08-26）：预算内 provenance 复核失败的行为学研究。
- **2609.03340v1 PlanFence**（2026-09-03）：分布式 agent 的 stale-plan 依赖域验证。
- **2609.00243v1 Invalidation Contracts**（2026-08-31）：跨 episode 缓存失效协议层。
- **2609.01852v1 The Memory Trust Gap**（2026-09-01）：stale 记忆压过权威工具的能力分层失败研究。
- **2609.11060v1 Grounding Agent Memory / Env-Probing Curation**（2026-09-10）：curator 用只读环境探针刷新 stale 记忆。
- **2609.05339v1 Memory Portability across Model Upgrade**（2026-09-04）。
- **2608.17756v2 D2ACCI**（2026-08-18）：evidence-preserving 记忆诊断协议。
- **2609.07471v1 MEMO**（2026-09-07）：多模态证据组织/预算化 readout；命中 query-conditioned 关键词但主轴是模态选择。
- **2608.15780v1 SDF（Google Discover）**（2026-08-16）：推荐系统里的 supersession+decay 双滤波；跨领域概念对齐，可作 motivation 引用。
- **2604.00131v3 Oblivion**（v3 更新 2026-09-01）：decay 驱动可及性遗忘，读/写路径解耦。
- **2606.20954v2 LRE（Learning What Not to Forget）**（v2 更新 2026-09-03）：KB 级学习型 eviction 打分器。
- **2609.10413 之外的 lifecycle 关键词命中**（2501.09166v2 Attention...Until You Need Retention，v2 更新 2026-09-14）：记忆保持架构宣言式论文，弱相关。

## 结论
1. **直接撞车仍只有 RD-Forget（2609.10263）一篇**，且截至 2026-09-17 停留在 v1、无 OpenReview 投稿痕迹；但其摘要与 QC-Lifecycle 核心机制逐条重合，时间窗口是第一风险。
2. 9 月新增最需警惕的是 **Fortunate Recall（2609.10413）**：同日提交、lifecycle+supersession+时间消歧全覆盖、自带 LifecycleBench 基准；虽无 per-query 时态条件化，但会压缩"生命周期管理"侧的新颖性表述空间。
3. **MemPrism（2608.06745）+ STAIR（2608.16224）** 合起来覆盖了"条件化视图渲染"与"查询时态意图分类"两个组件的先例；QC-Lifecycle 的 novelty 陈述应收窄为"**查询时态分类驱动的记忆生命周期视图**"这一组合，并在相关工作中逐一切割。
