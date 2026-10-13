# QC-Lifecycle 项目计划（Query-Conditioned Forgetting with Lifecycle Tracking）

日期：2026-09-16。用户已拍板推荐 1。本文档 = 撞车核实 + 差异化设计 + 实验计划 + 排期。

## 0. 硬约束（用户指定）
- **完全 training-free**：无 RL、无神经网络训练（4×4090 资源限制）。规则/启发式/冻结 LLM 调用均可。
- **单数据集起步**：LongMemEval-S（用户："数据集太大先只做一个"）。
- LLM 后端 = DeepSeek API（deepseek-v4-pro 主评测 / deepseek-flash 高吞吐操作），--noproxy 直连。

## 1. 撞车核实（2026-09-16，arXiv API 全文摘要核对）

两篇锚点竞品**均真实存在**，且比调研代理的一句话概括更接近本方案：

**RD-Forget (2609.10263, 2026-09-09,仅 arXiv)** 摘要核实要点：
- 已做：存储/使用分离（retained source archive + query-conditioned view）;冻结 LLM curator 抽取证据并分组进 semantic slots;**same-slot replacement links 在 current-state 语境抑制被取代值**;**intent-aware retrieval 让早期证据重新可用**（即历史查询可访问旧值）;rate-distortion 预算化视图构造。
- ⚠️ 调研代理称其"slots 无 lifecycle"**不准确**——它的 replacement links 就是 supersession 跟踪的一种。真实差距见 §2。

**MemoryLACE (2609.03201, 2026-09-02,仅 arXiv)** 摘要核实要点：
- 已做：sparse merge / supersession / contradiction 三类生命周期关系 + 原子记忆 + provenance;检索时重建 relation-aware evidence units（暴露 current/historical/supporting/conflicting 证据）;BEAM/StructMemEval 最优 + 66.6% 运行时缩减。
- 未做：**查询条件化**——它把全部关系暴露给下游推理，不按查询意图（current-state vs historical）主动抑制/放行。

**结论**：QC-Lifecycle = "RD-Forget 的查询条件化 + MemoryLACE 的完整生命周期关系"的组合，调研代理自己也标了风险："may face incremental critique if not sufficiently distinguished"。两篇都是 9 月上旬 arXiv 预印本（无 venue），可按 concurrent work 处理，但差异化必须超出"简单组合"。

## 2. 差异化设计（必须超出 A+B 组合）

**2026-09-16 全文精读裁决更新（依据 13_competitor_deepread.md，取代下述初版判断中与之冲突处）：**
- (a) 查询分类【仍成立，论据更锋利】：RD-Forget 的 h(q) 实为"词法线索+基准 gold task-type 标签"的二值启发式，**吃了评测集元数据**且全文零评测其准确率。我们的表述从"LLM 黑盒"改为"标签泄漏+未评测+二分类 vs 我们无泄漏三分类+误分类传导分析"。
- (b) 替代vs矛盾二分【需收缩】：MemoryLACE 已有 {merge, update, contradict, no action} 四类 LLM 判定关系，二分首创权让给它。仍属我们的：矛盾的**查询条件化呈现策略**、**规则可审计判定**（两家均 LLM 调用）、**多值属性词表**（两家均无）。
- (c) 错误类型学三分【仍成立】：两家均只报端到端分数；MemoryLACE 消融定性点名"update 被误读为 contradiction"但未量化——引作动机证据。
- (d) 预算扫描错误构成【仍成立，最干净】：RD-Forget 自称 rate-distortion 却 B=2048 单点固定从未画 R-D 曲线；MemoryLACE 固定 64 条。
- 协议要点：RD-Forget 的 KU 实验用的是 **oracle 版 78 问无检索**设定；我们的 S cleaned 全量+真检索设定未被占。其基线仅 ACE/ReasoningBank、judge 与被评系统同族，两处协议弱点可在实验节对打。RD-Forget 有 lexical fallback，复现基线须如实实现。两篇均无 venue 的 v1 预印本，concurrent-work 处理成立。

真实空隙（初版判断，(b) 已按上方收缩）：
1. **查询意图分类是显式且可审计的决策**：RD-Forget 的 intent-aware retrieval 机制在摘要中是黑盒（LLM 隐式）；我们把"查询时态判定 →关系过滤规则"做成确定性、可解释的决策表，并**单独评测查询分类器的准确率与下游误差的传导**（误分类代价分析——两篇都没有）。
2. **矛盾的查询条件化处理**：MemoryLACE 暴露 conflicting 证据但不裁决；RD-Forget 只处理 supersession（替代）不处理 contradiction（同时刻矛盾）。我们区分**替代（时间性更新）vs 矛盾（可靠性冲突）**两类关系，查询条件化地选择呈现策略：current-state 查询给最新+标注冲突，historical 查询给完整演化链。
3. **错误类型学与 restore-counterfactual 审计**：借 What-Eviction-Destroys（2609.08279）协议，把错误分为"过时事实误导（stale-fact error）"vs"历史缺失（history-loss error）"vs"矛盾混淆（conflict-confusion error）"三类，逐基线报告错误构成——这是**分析性贡献**，两篇竞品都只报端到端准确率。
4. **预算扫描下的鲁棒性**：1k/2k/4k/8k token 预算下三类错误的构成变化（RD-Forget 有预算化但未做错误分解）。

**论文定位**（防 incremental 批评）：不是"又一个记忆系统"，而是"**时态查询条件化的系统性研究**：什么时候该抑制、什么时候该暴露记忆的历史层"——以错误类型学与误分类代价分析为分析主轴，方法本身是轻量可复现的参考实现。

## 3. 实验设计

### 数据（已就绪路径）
- LongMemEval-S **cleaned 版**（原版已 deprecated，噪声会话影响答案正确性）：`/root/agent_memory_iclr/data/longmemeval_s_cleaned.json`（278MB，下载中）+ oracle 版（15MB，调试用）。来源 hf-mirror（实测 200）。
- 500 问、五能力标签；重点子类 = knowledge-update（时态更新）与 temporal-reasoning；abstention 子类做辅助。

### 系统对比（5 系统，全部自行复测）
1. 全量记忆+检索（无遗忘基线，强基线）
2. MemoryBank 式时间衰减
3. MemoryOS 式 FIFO 驱逐
4. RD-Forget 式（按摘要精神复现：slot 替代链+查询条件视图,无矛盾关系/无错误分解）
5. **QC-Lifecycle（本方法）**：规则化 supersession/contradiction 抽取（deepseek-flash 做实体-属性对齐,规则做时序链接）+ 确定性查询时态分类器 + 条件化视图构造

### 消融
(a) 去查询条件化（全部暴露）＝MemoryLACE 精神；(b) 去生命周期（只按相似度检索）；(c) 查询分类器换 oracle 标签（上界）vs 关键词启发式 vs LLM 分类，量化误分类传导。

### 指标与协议卫生
- 主指标：分子类准确率（LLM-judge, deepseek-v4-pro 固定 prompt 多次取均）+ token-F1 双报 + 分歧率（Nous 27.5 分教训）。
- 错误类型学三分（本文核心分析）+ 预算扫描 1k/2k/4k/8k。
- 逐操作 token 成本（MERIT 口径）；固定嵌入模型（MemDelta 教训）。

**2026-09-16 协议对齐更新（依据 15_protocol_alignment.md）：**
- **主表必须跑全量 500 题**（cleaned 版）：文献无人用 60 题样本报主表；60 题降级为开发集/消融，报 bootstrap CI（MemDelta n=100 先例要求 CI+McNemar）。
- **判分双轨**：社区事实标准是 GPT-4o-mini+官方分题型 prompt（2025-10 后四篇一致），但本项目仅有 DeepSeek 通道 → 论文主表用 deepseek-v4-pro judge 并显式声明口径不可与文献直接对比；同表报 token-F1；讨论节引用第三方复现波动 >20pp 证明跨论文数字本就不可比。
- **cleaned 版声明**：本地文件 sha256 与 2609.08279（COLM 2026 WS）pin 的一致（d6f21ea9...c3a442），是唯一已知使用 cleaned 的先例，论文必须声明版本并注明与原版数字不可直接对比（cleaned 删噪声会话 115k→102k tokens，题数与题型分布不变）。
- **可比数字锚点**：LongMemEval-S 自报数字仅 5 篇可用——原文 GPT-4o 全上下文 0.606、Zep 71.2%、LightMem 68.64、LycheeMemory V2 92.20（KU 97.44/TR 87.22，当前最高）、CueMem 75.20（自评自判不可比）。Mem0/MemoryOS/SeCom 从未在 LME 自评。
- **子类难点弹药**：KU 主瓶颈=旧值干扰（RD-Forget 去遗忘掉 33pp）+检索漏 pre-update 证据；TR 主瓶颈=时间无关检索，全上下文仅 34-48%，区分度最大。天花板参考：KU ~97、TR ~87。

### 成功线
current-state 子类相对最强基线 +3%，historical/temporal 子类 +2%，中性子类退化 <1%。若达标 → 扩展 LoCoMo 对照（仅端到端参考,不做写入消融）并起草论文。

## 4. 排期（周粒度）
| 周 | 内容 |
|---|---|
| W1 | 数据加载器+评测管线（judge/F1/成本）;全量基线+FIFO/时间衰减基线跑通 |
| W2 | 生命周期抽取管线（flash 实体对齐+规则链接）;RD-Forget 式基线;QC-Lifecycle 主方法 |
| W3 | 消融+预算扫描+错误类型学分析 |
| W4 | 若正向:LoCoMo 参考 + 论文初稿;arXiv 尽早挂（RD-Forget/MemoryLACE 均 9 月初,时间窗敏感） |

## 5. 风险

**2026-09-17 撞车扫描更新（依据 14_fresh_scan.md）：**
- 直接撞车仍只有 RD-Forget（确认仍 v1、无 OpenReview 记录;MemoryLACE 同）。
- 新警惕三篇：**Fortunate Recall（2609.10413，09-09）**本体驱动生命周期策略+自带 LifecycleBench 516 题时间消歧基准，压缩"生命周期管理"侧新颖性，审稿人可能要求跑其基准；**MemPrism（2608.06745）**任务条件化记忆视图（条件轴是任务非查询时态）；**STAIR（2608.16224）**LLM 把问题映射为时态意图再确定性执行——"查询时态意图分类"组件有先例，须引用。
- **novelty 陈述收窄为组合**："查询时态分类驱动的记忆生命周期视图" + 错误类型学 + 预算-错误构成分析;相关工作对上述各篇逐一切割。

- **最大风险**：RD-Forget 作者（或他人）快速迭代补上矛盾处理/错误分解 → 对冲 = 4 周内挂 arXiv,分析性贡献（错误类型学）独立于方法存活。
- LongMemEval-S 的 knowledge-update 子类规模可能不足以支撑三类错误的统计功效 → 预案 = 借 Nous 协议在 LME 会话流上受控注入矛盾对（合成增强,透明标注）。
- LLM-judge 不稳定 → 双指标+分歧率+固定 judge 版本。
- **抽取召回是 pipeline 型系统的固有风险面（60 题 v2 实证）**：转述/顺带提及型更新事实漏抽 → 视图含 stale fact 反而误导（Rachel/mortgage/family-trip 三错例同模式）。v3 修复 = 抽取 prompt 强化更新型/转述事实 + 回答 prompt 原文优先。此现象本身进论文错误类型学（extraction-miss → stale-fact error）。

## 6. v4 管线重构(2026-09-17,全量批跑前 preflight 审查落地)

24-agent 对抗审查(4 视角 × 独立核实)确认 17 项问题,已全部修复。关键变更:

**架构**
- 缓存改会话级内容哈希(`lifecycle_cache/sessions/`,key=md5(会话文本)):全库 23,867 会话实例仅 18,464 唯一,跨题去重省 ~23% 抽取开销;关系标注移到装配期(纯代码),规则可迭代不需重抽;抽取失败不写缓存 → 断点重试天然廉价。EXTRACT_PROMPT 移除日期行(日期由代码盖章),缓存与 prompt 严格等价。
- 关系规则 v4(时间戳含时分,可全序):同会话不同值 = **cooccurring**(共现噪声,非冲突);跨会话同时间戳 = **contradiction**(不可排序);更晚时间戳 = **supersession**。旧规则的 contradiction 结构上只能同会话触发(v3 缓存 138/138 全是同会话转述噪声),已推翻。多值结构启发改为"单会话内 ≥3 不同值"。
- current 视图 = 最新会话全部共现值;[CONFLICT] 仅在真 contradiction 时标注。视图渲染确定性排序 + 行边界截断。

**协议卫生(臂间对称性)**
- 统一 `answer_chat`:全部系统同模型、同 max_tokens 6144、同空答加倍重试(旧版基线 2048 无重试,n60 已现 2/60 空答判负,系统性偏袒本方法 ~3%)。
- 回答 prompt 改中性双源表述(旧"TRUST THE RAW TURNS"稀释 qc vs qc_no_qc 消融)。
- llm_judge:空输出/无判定词加倍配额重试,仍不可解析则抛异常记错误行,绝不静默判负。
- token_f1 SQuAD 式规范化(旧纯空白切分被 markdown 标点打压)。
- 抽取 max_tokens 4096→8192 + 零解析重试 16384(v2 台账 25.5% 调用撞上限返回空抽取)。

**runner**
- run_qc argv 解析移入 `__main__`(旧版 `import run_qc` 在 run_full500 下直接崩,三个 QC 臂会全灭)。
- run_full500 resume 只认含 judge 的行(错误行重试);存全量 pred(4000 字)+ qlabel/view_mode/mem_units,保住 post-hoc 重判分与错误类型学;汇总对未决错误行大声告警。
- 新增 **qc_no_lifecycle** 臂(BM25-only, k=20 与主方法一致)= 归因对照,隔离"视图贡献"vs"检索配置差异"(主方法 k=20+视图 vs 基线 k=30 存在混淆)。
- launch_full500.sh:set -e;预热与四个无缓存臂并行;QC 臂 gated on 预热成功标记。
- 查询分类器补 `\bprevious\b`(排除 previous chat/conversation 等会话元指称),historical 触发 36→44/500。

**遗留(不阻塞全量,W2-W3 排期)**
- RD-Forget 式基线(计划系统 4)未实现;各臂独立可后补,缓存可复用。
- judge 单票 n=1、无官方分题型 prompt:全量 pred 已存,post-hoc 多票/换 prompt 重判是廉价重算;≥40 题第二 judge 校准待做。
- 台账跨断点续跑只记当轮(已改时间戳文件名,报告成本时跨文件求和)。

## 7. 成本决策(2026-09-17,用户指令)
- 实测标定:thinking-on 全口径全量 500 ≈ 850 CNY(flash 抽取 reasoning 燃烧占 ~60%)。
- 抽取改 thinking-off(`thinking:{"type":"disabled"}`,实测 reasoning_tokens=0):小样本对照事实召回持平;nt 抽取重建 60 题缓存后 KU 门保持 0.90(9/10,错例=mortgage 旧值,v2 已知模式)。缓存目录 sessions_nt/,旧 thinking-on 缓存留 sessions/ 可回退。
- **用户指定回答模型 = deepseek-flash**(common.ANSWER_MODEL);judge 保留 v4-pro(每题分厘级成本,且判分模型≠答题模型可挡"自评自判"批评,较文献口径反而更干净)。论文声明:答题 flash(thinking-on)、判分 v4-pro、抽取 flash(thinking-off),全臂统一无不对称。
- flash 答题口径 KU 双门(qc vs full 同 10 题)结果见 results/ku_flash_gate.json。
- 预计全量成本(nt 抽取 + flash 答题 + v4-pro 判分)≈ 180-200 CNY;门后按余额差精确复标。
