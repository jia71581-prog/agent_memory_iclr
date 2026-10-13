# 竞品全文精读对照笔记（QC-Lifecycle 差异化裁决）

日期：2026-09-17。来源：arXiv PDF 全文（非摘要）。
- A. RD-Forget = arXiv:2609.10263v1 (2026-09-09), "What Should an Agent Forget? Separating What Is Stored from What Is Used", Yuhang Li & Yuchen Li (北航/华东师大), ACM 双栏 8 页。
- B. MemoryLACE = arXiv:2609.03201v1 (2026-09-02), "MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval", Yacoubi et al. (TUM/inovex/Cerebras), IEEE 双栏 8 页。

本笔记对照基准 = `/root/agent_memory_iclr/12_qc_lifecycle_plan.md` §2 四点差异化主张。

---

## A. RD-Forget 全文精读

### A.1 方法细节

**总体管线（Alg.1, p.4）**：Curate → MaterializeActive → 建 supersession 链 →（空则词法兜底）→ Eligible(Eq.5) → RankAndCollect → LLM Select → 贪心 Pack → 答。存储 H 不删，视图 Z(q) 每查询重建。

**(1) intent-aware retrieval 的真实实现（关键发现）**：
- 核心是一个二值开关 h(q)∈{0,1}（"historical or revision-chain request"），进 Eq.5 的资格池：E(q) = A ∪ (D if r=1) ∪ (S if r=1 且 h(q)=1)。（§3.4, p.3）
- **h(q) 的实现被藏在 §4.2 (p.4) 一句话里："ℎ(𝑞) combines lexical cues with task-type labels for contradiction resolution, temporal reasoning, and preference history. These auxiliary inputs are implicit in Algorithm 1."**
- 三个要害：(i) 它不是纯 LLM 黑盒（计划书 §2.1 的"LLM 隐式"描述不准确），是词法线索+**基准 gold task-type 标签**的启发式——直接吃了评测集元数据，真实部署无此标签；(ii) 只有 historical/非historical 二分，无 current/historical/neutral 三分，无规则表；(iii) **全文没有任何对 h(q) 本身准确率的评测，没有误分类传导分析**。−QC 消融（§4.4）删的是"curator 看不见问题"（整个查询条件化），不是单独消融时态分类器；−RESCUE 删资格开关但也不是分类器质量实验。
- 另有"intent-aware extension"（§3.2, p.3）：对 contradiction-resolution/evolution/recommendation 三类问题给 curator 不同的抽取指令（保留冲突主张/保留转折点/组合偏好+约束）——这是 prompt 分支，同样依赖任务类型标签，无分类器评测。

**(2) semantic slot 构建管线（§3.2, p.3）**：冻结 LLM curator 对 (H,q) 抽取条目：原子陈述 + 槽 `subject|relation|scope` + 源证据串 + 可选 validity time。curator 被提示抽"回答所需的最小完备证据集"，multi-hop 时保留整条关系链（−CLOSURE 消融即删链式扩展指令）。scope 字段做局部化（"拒绝 Course A ≠ 拒绝学 Python"，Fig.2, p.4）。curation 为空时词法兜底：按与问题重叠度取 ≤20 行源文本（≈我们的 BM25 兜底已被做过，注意）。

**(3) same-slot replacement link 判定规则（§3.3, p.3, Eq.4）**：materializer 只做**字符串归一化（去空白+小写）后的槽名精确匹配**：slot(m_o)=slot(m_n) 且 status(m_n)=ACTIVE 且两者有内容 → m_o 标 SUPERSEDED、superseded_by 指向 m_n。"槽内最后一个 active 条目为替换目标"。**没有实体对齐/别名归并/跨会话链排序逻辑——同一事实两次抽取若槽字符串不同即漏链**；时序判定完全外包给数据格式约定（§3.2："MAB-FC 用序号、AMB-Text 用会话边界、LME-KU 用 query date+最新显式更新"），无 (date,会话序) 复合键这类通用规则。未挂链的旧行不进 materialized memory（只留在 H）。

**(4) rate-distortion 公式（§3.1, p.2, Eq.1-3）**：Z*(q) ∈ argmin_{Z⊆C(q)} D_q(Z) s.t. Σc(m)≤B；D_q(Z)=E[d(Y,g_θ(q,Z))|q,H]；c(m)=Tok_o200k(m.text)+8，**B=2048 固定**。实际实现＝LLM curation+状态资格+词法排序(active 加 0.4 分)+LLM selector+贪心装箱（§3.5, p.3），R-D 只是叙事框架，无任何优化性质或预算扫描。**全文只跑一个 B 值。**

**(5) contradiction 与多值属性**：**无 contradiction 关系类型**——唯一关系是 supersession。同期矛盾只在 intent extension 的 prompt 里处理（"contradiction-resolution questions retain the relevant claims"），不建关系、不标注、不裁决。多值共存靠 curator 自由裁量："Within a slot, the curator distinguishes a replacement from a related observation that can coexist with the current value"（§3.2, p.3）——LLM 隐式判断，无多值属性词表、无规则。

### A.2 实验（§4-6, p.4-6）

- **任务**：主套件 AMB-Text（LoCoMo 第一场对话 86 问，经 AgentMemoryBench 分发）、**LME-KU（LongMemEval oracle 版 78 道 knowledge-update 问，供 oracle sessions + query date）**、MAB-FC（MemoryAgentBench Conflict_Resolution 100 问）；intent 套件 BEAM 100K 档 150 问 + PersonaMem-v1 32K 档 579 问。
- **要害：LME-KU 是 oracle-session 设定（无检索大海捞针），且只用 KU 子类 78 问。不占我们的"LongMemEval-S cleaned 全量 500 问 + 真检索"设定。**
- **基线只有 ACE 和 ReasoningBank**（经验/上下文工程系统，非记忆生命周期系统）——没有 Mem0/A-MEM/MemGPT/时间衰减/FIFO 任何记忆基线。
- 骨干：Qwen3.5-flash、GPT-5.6-Luna、MiniMax-M2.5、Kimi-K2.5。**judge 与被评系统同族同配置**，作者自认"within-protocol evaluation rather than fully independent external judgments"（§4.2, p.4）。
- 主数字（Table 1, p.6）：12/12 组合全胜。Luna：AMB-Text 89.53 / LME-KU 93.59 / MAB-FC 77.00。对更强基线增益：AMB-Text +1.16~+19.77、LME-KU +6.41~+25.64、MAB-FC +11~+26。BEAM +8.47~+15.67、PersonaMem +10.88~+20.72（Table 3, p.6）。
- 消融（Luna only, Table 2, p.6）：Full 91.86/93.59/74.00；−FORGET 最惨（68.60/60.26/51.00，即不抑制旧值伤最大）；−QC 次惨（77.91/82.05/56.00）；−RESCUE/−SLOT/−CLOSure 各掉 2.5~10 分。**消融全部只报端到端准确率，无错误分解。**

### A.3 自认局限

**无 Limitations/Future work 节**（§7 Discussion + §8 Conclusion 均为正面总结）。文中散见的自我限定仅两处：(i) §4.2 judge 同族非独立；(ii) §7 "Slot granularity governs this separation in practice"——承认槽粒度是实际瓶颈（curator 分错互补/竞争即失败），但未量化。

---

## B. MemoryLACE 全文精读

### B.1 方法细节

**记忆表示（§III-A, p.2）**：m_i=(x_i 原子自然语言重述, τ_i 证据锚定时间元数据[事件时间/有效期], z_i 实体/关键词/主题/地点, p_i 支撑轮次 provenance, a_i 活跃位, R_i 生命周期关系集)。全库=稀疏关系图 G=(M,E)：有向 supersession 边 + 对称 merge/contradiction 边。supersession 双向存（新→旧 supersedes，旧→新 superseded_by）。

**(1) 关系抽取管线（§III-B, p.2-3）——LLM 判定而非规则**：
- 构建：沿用 SimpleMem 滑窗（20 轮窗、2 轮重叠），LLM 把窗转为原子记忆；同轮重复抽取合并 provenance，异轮重复留待 merge 链。
- 候选：按时间顺序整合（后证据不能先于其修改的状态）；语义相似(12)+同实体匹配(8) 检索候选，扩一跳生命周期邻居，总候选预算 15，cross-encoder (ms-marco-MiniLM-L6-v2) 重排取 5。
- 判定：**LLM 整合模块** g(m_new, C) → (r, c*)，每步只选一个关系+一个目标，R={merge, update, contradict, no action}；判定时评估"subject, attribute, claim, and temporal scope"以防主题相似误改状态。**"the backbone is used for memory construction, consolidation, batch evolution, query analysis, and answer generation"（§IV-B, p.4）→ 关系判定=LLM 调用，非规则、非可审计决策表。**
- 状态更新：update → 旧 a←0 新 a←1 建双向链；**contradict → 双方都保持 active + 对称链（"available evidence does not resolve their conflict"，存而不裁决）**；merge 默认 linked-merge（保原文互链），备选 compact-merge（合成一条，丢原件）。
- Local batch evolution：新增一组记忆后，LLM 联审该组+其候选，补建单步决策漏掉的关系（如一条新记忆牵涉多条旧记忆），只动链接和活跃位，不改文本。

**(2) evidence unit 重建（§III-C, p.3-4）**：
- 检索只搜 active 集：4 通道（语义 Qwen3-Embedding-0.6B 40 条 / 关键词 30 / 结构化模板 30 / 主题 20）RRF 融合 (k=60) 得锚点 A_q。
- **lifecycle expansion（无条件）**：E_q = A_q ∪ L(A_q)，L=沿 supersession 谱系全链回溯（上限 80 节点）+ 一跳 contradiction（上限 4 条）。
- 证据单元 = 检索池内由 supersession/contradiction/merge 连成的连通分量，supersession 边内部按旧→新排；cross-encoder 对**整个单元**打分重排；按预算装箱（终上下文 64 条记忆），链保旧→新顺序，超预算单元保高优先级部分。
- 生成时按 lifecycle 顺序呈现，**给 LLM 标出 current / superseded / conflicting** 三种身份。

**(3) 是否查询条件化——没有**：expansion 对任何查询都执行，current/historical/conflicting 证据一律全部进上下文，"The model can therefore prioritize active evidence for current-state questions, recover earlier states for historical or sequential questions"（§III-C4, p.4）——**按查询意图取舍完全交给答题 LLM 隐式完成，系统侧无时态分类器、无按意图的抑制/放行**。§IV-B 提了一次"query analysis"由 backbone 承担，但方法节从未定义该步骤（疑指结构化模板通道的查询解析），全文无查询意图分类机制或其评测。

### B.2 实验（§IV-VI, p.4-7）

- **基准：BEAM 100K 子集（20 对话 400 问、10 能力）+ StructMemEval 主集 51 场景。不含 LongMemEval**（仅在相关工作引用 [21]）。
- 基线：BEAM 上 SimpleMem / LIGHT / Hindsight；StructMemEval 上 Mem-Agent。骨干 Qwen3.5-4B/9B (FP8)、GPT-5.5；judge 固定 GPT-4o-mini（比 RD-Forget 干净）。温度 0。
- 主数字（Table I, p.5）：BEAM overall 4B 45.4（vs Hindsight 40.1）、9B 51.9（vs Hindsight 50.3，10 类中 6 胜；temporal −8.7、info −5.2、summ −4.9 落后）。9B contradiction 40.3 vs Hindsight 40.9（基本平）。运行时 −66.6%（7h31m vs 22h30m，Table II）。StructMemEval 52.08 vs 35.00（state/tree 双 100%，count 0%，recsys 8.33%）。
- **消融（Table IV, p.6，9B）**：去双 reranker −1.23；compact merge −1.43（contradiction +3.8 但 multi-hop −12.3）；去 batch evolution −1.68；**去 temporal awareness −4.97（contradiction 40.3→26.6、temporal 26.3→16.3）；去 lifecycle expansion −5.16（update 51.9→35.0、multi-hop −13、temporal −12.5，contradiction 不变 40.3）**；全删 −8.55。
- **关键定性发现（§VI, p.6）**："without it [temporal awareness], a valid update is more easily misread as a contradiction"——**他们已用消融叙事点名了"更新被误读为矛盾"这一失败模式**（即我们 conflict-confusion 的机制级前身），但只报能力类别分数变化，没有逐答案错误归类。

### B.3 自认局限

无独立 Limitations 节。自认要点：(i) 结论/§V-B：count-based 0%、recsys 场景级 8.33%——"global numerical accumulation or ranking across many distributed preference signals"不适配，future work=外挂聚合/计算器/代码执行算子；(ii) §V-A：设计取舍导致 exhaustive summarization 与显式时间区间计算弱于 Hindsight；(iii) knowledge update 提升甚微的自我解释："a single active-memory lookup already suffices"。

---

## C. 四点差异化主张逐条裁决

### (a) 显式可审计查询分类 + 误分类传导分析 —— 【仍成立】（且论据比原计划更锋利）
- RD-Forget：h(q) 存在但=词法线索+**gold task-type 标签**（§4.2, p.4），二值而非三分，无规则表，**h(q) 自身准确率与误分类代价全文零评测**；且吃基准标签在真实部署不可得——这是可攻击的协议缺陷。计划书 §2.1 "LLM 隐式黑盒"表述需修正为"启发式+标签泄漏、未评测"。
- MemoryLACE：完全没有查询意图分类，全量暴露交给答题 LLM（§III-C4）。
- 我们的三分类（historical/current/neutral）+确定性正则规则表+oracle/关键词/LLM 三档消融量化传导（计划 §3 消融(c)）在两文均无对应物。**注意 RD-Forget 已占"historical intent 重新放行旧值"这个机制本身，卖点必须落在"分类器显式化+可审计+误分类传导量化"，不能落在"有没有意图开关"。**

### (b) 替代 vs 矛盾二分 + 多值属性处理 —— 【需收缩】
- **关系层面的 supersession/contradiction 二分已被 MemoryLACE 占**：R={merge, update, contradict, no action} 是显式四类（§III-B2, p.3），矛盾对称链+双 active 存而不裁决（§III-B3），且 Table IV temporal-awareness 消融已论证"时间锚定区分有序更新 vs 真矛盾"（§VI, p.6）。RD-Forget 确实只有 supersession（矛盾仅 prompt 处理），但不能再声称"二分是我们首创"。
- **仍属我们的**：(i) 矛盾的**查询条件化呈现策略**（current→链头最新值+CONFLICT 标注；historical→全链演化）——MemoryLACE 一律全暴露、RD-Forget 无矛盾关系，两家都没有"按意图选择呈现策略"；(ii) **规则化/可审计**的关系判定（两家关系判定都是 LLM 调用：RD-Forget curator、MemoryLACE consolidation g）；(iii) **多值属性词表**——RD-Forget 靠 curator 自由裁量"可共存的相关观察"（§3.2），MemoryLACE 无多值机制（merge 只管重复证据），显式多值属性词表+结构启发两家均无。
- **写作动作**：把主张从"我们区分替代 vs 矛盾"改写为"我们把（MemoryLACE 已引入的）替代/矛盾区分做成规则可审计判定，并首次将其与查询时态条件化耦合成呈现策略，加多值属性词表"。

### (c) 错误类型学三分（stale-fact / history-loss / conflict-confusion） —— 【仍成立】（需加一句让步引用）
- RD-Forget：只报端到端准确率+消融分数（Table 1-3），零错误分析，连 Limitations 节都没有。
- MemoryLACE：报 BEAM 能力类别分解+消融 Δ，并**定性**点名"update 被误读为 contradiction"失败模式（§VI）——机制级叙事已有，但没有逐答案错误归类、没有逐基线错误构成、没有 restore-counterfactual 审计。
- 我们的定位：把该失败模式**量化为逐系统、逐预算的错误构成**。写作时应引 MemoryLACE §VI 作为动机证据（"竞品消融已暗示此混淆存在但未测量"），反而增强而非削弱主张。

### (d) 预算扫描下错误构成 —— 【仍成立】（最干净的一条）
- RD-Forget：有 R-D 预算叙事但 **B=2048 单点固定**（Eq.3, p.2-3），无扫描、无预算-错误关系；intent 套件只是换了更大的 cap（8192/2048），仍非扫描。
- MemoryLACE：上下文固定 64 条记忆（§IV-B），无预算维度实验。
- 1k/2k/4k/8k 预算 × 三类错误构成的二维分析两家完全空白。且可反打 RD-Forget："自称 rate-distortion 却从未画过 rate-distortion 曲线"。

---

## D. 超出四点主张的重要发现（写论文/做实验用）

1. **LongMemEval 阵地未失守但需精确措辞**：RD-Forget 用了 LongMemEval 的 **oracle 版 KU 子类 78 问**（喂 oracle sessions）。我们是 LongMemEval-S cleaned 全量 500 问+真检索（大海捞针）。相关工作里必须写明这一区别，不可笼统说"竞品未用 LongMemEval"。
2. **基线弱点可打**：RD-Forget 基线只有 ACE/ReasoningBank（非记忆系统）；MemoryLACE 未与 Mem0/A-MEM 直接比。我们计划的全量记忆/时间衰减/FIFO/RD-Forget 式四基线阵容按"记忆策略同台"逻辑站得住。
3. **评测卫生可打**：RD-Forget judge 与被评系统同族同配置（自认 within-protocol）；我们固定 deepseek-v4-pro judge+双指标+分歧率更干净。MemoryLACE 用固定 GPT-4o-mini judge，这点它是干净的。
4. **两家共同的结构成分与我们撞车的部分**（复现 RD-Forget 式基线时要如实实现）：词法兜底（RD-Forget 有 ≤20 行 lexical fallback，与我们 BM25 兜底同精神）；active 条目加权检索；superseded 状态默认排除。
5. **MemoryLACE 给我们的免费弹药**：其消融显示矛盾解析对 lifecycle expansion 不敏感（40.3 不变）但对 temporal awareness 极敏感（→26.6）——支持我们"矛盾处理瓶颈在时序判定而非检索连通性"的设计直觉，可引用。
6. **风险更新**：RD-Forget 的 scope 字段+intent extension 覆盖了部分"局部更新不推翻全局偏好"场景（Fig.2）；我们多值属性词表的案例设计应避开"偏好 scope"这一它已演示的类型，主打 hobbies/medications 等真多值属性。
7. 两篇均为 v1、仅 arXiv、无 venue 标注（RD-Forget 用 ACM 模板、MemoryLACE 用 IEEE 模板，可能分别在投）。concurrent-work 处理成立，但 4 周窗口判断不变。

## E. 计划书需修订的两处事实错误

1. §1 对 RD-Forget "intent-aware retrieval 在摘要中是黑盒（LLM 隐式）"→ 实为"词法线索+gold task-type 标签的启发式，未评测"（§4.2 原文）。攻击点从"黑盒"改为"标签泄漏+未评测+二分类"。
2. §2.2 "MemoryLACE 暴露 conflicting 证据但不裁决；RD-Forget 只处理 supersession 不处理 contradiction"——前半句对，后半句需补：RD-Forget 在 intent extension 的 prompt 层面处理 contradiction-resolution 问题（保留冲突主张），只是无关系类型。且"替代 vs 矛盾二分"首创权属 MemoryLACE，主张 (b) 按 §C(b) 收缩。
