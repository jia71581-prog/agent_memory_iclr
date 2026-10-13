# LLM Agent 长期记忆系统调研笔记(ICLR 相关工作基础)

- 日期:2026-09-16
- 检索通道:arXiv API(export.arxiv.org)、OpenAlex、OpenReview API2、Crossref;全部经 `curl --noproxy '*'` 直连,HTTP 200 实测。
- 纪律:所有 arXiv ID 均来自 arXiv API 返回,venue 来自 OpenReview/OpenAlex/Crossref 返回;摘要中读不出的机制细节标注 `[正文,待复核]`(基于对论文正文的已知理解,写作引用前需对照全文),完全查不到的标"未核实"。
- 引用计数为 OpenAlex 2026-09 快照,OpenAlex 对 arXiv 论文普遍低估(同一论文常拆成多条记录),仅作相对参考。

---

## 0. 快速索引表

| 系统 | arXiv ID | 首发年 | Venue(已核实) | 记忆表示 | 写入决策 |
|---|---|---|---|---|---|
| MemGPT | 2310.08560 | 2023 | arXiv preprint(CoRR 2023) | 分层:上下文内 + 外部存储 | LLM 自主 function call |
| Generative Agents | 2304.03442 | 2023 | UIST 2023 | 自然语言事件流 + 反思 | 全量存储 + LLM 反思生成 |
| Reflexion | 2303.11366 | 2023 | NeurIPS 2023 poster | 情景反思文本缓冲 | 失败触发 + LLM 生成反思 |
| MemoryBank | 2305.10250 | 2023 | AAAI 2024 | 分层摘要 + 用户画像 | 启发式:Ebbinghaus 遗忘曲线 |
| Voyager | 2305.16291 | 2023 | TMLR | 可执行代码技能库 | 环境验证:成功才入库 |
| ExpeL | 2308.10144 | 2023 | AAAI 2024 | 自然语言 insights + 轨迹 | LLM 跨轨迹归纳 |
| MemoryLLM | 2402.04624 | 2024 | ICML 2024 poster | 潜在空间固定记忆池 | 训练得到的压缩更新(自动) |
| HippoRAG | 2405.14831 | 2024 | NeurIPS 2024 poster | 开放知识图谱 + 语料 | 全量离线索引(OpenIE) |
| AWM | 2409.07429 | 2024 | arXiv(正式 venue 未核实) | 工作流(共用子程序) | LLM 归纳 + 成功过滤 |
| Zep/Graphiti | 2501.13956 | 2025 | arXiv preprint | 时序知识图谱 | 全量摄取 + 时间性边失效 |
| M+ | 2502.00592 | 2025 | ICML 2025 poster | 潜在记忆池 + 长期库 | 自动驱逐 + 共训检索器 |
| A-Mem | 2502.12110 | 2025 | NeurIPS 2025 poster | Zettelkasten 互联笔记网 | LLM 生成笔记 + LLM 决定链接/演化 |
| HippoRAG 2 | 2502.14802 | 2025 | ICML 2025 poster | KG + 深度段落整合 | 全量离线索引 |
| Mem0 | 2504.19413 | 2025 | IOS FAIA 2025(DOI 10.3233/faia251160) | 事实条目 / 图变体 | LLM 抽取 + 记忆操作 |
| MemoryOS | 2506.06326 | 2025 | EMNLP 2025 main | 短/中/长期三级页式存储 | 启发式:FIFO + 分段分页 |
| G-Memory | 2506.07398 | 2025 | NeurIPS 2025 spotlight | 三层图(insight/query/交互) | LLM 归纳 + 层级同化 |
| MemAgent | 2507.02259 | 2025 | ICLR 2026 Oral | 固定长度文本记忆(覆写) | 可训练策略(RL/DAPO) |
| MemOS | 2507.03724 | 2025 | arXiv preprint | MemCube:明文/激活/参数统一 | 系统调度层 [摘要未详,待读全文] |
| MIRIX | 2507.07957 | 2025 | arXiv preprint | 六类记忆(核心/情景/语义/程序/资源/金库) | 多 agent LLM 路由更新 |
| Memp | 2508.06433 | 2025 | arXiv preprint | 程序性记忆(步骤指令+脚本) | LLM 蒸馏 + 动态更新/废弃 |
| Memory-R1 | 2508.19828 | 2025 | arXiv preprint | 外部记忆库条目 | 可训练策略(RL:ADD/UPDATE/DELETE/NOOP) |
| Mem-α | 2509.25911 | 2025 | arXiv preprint | 核心/情景/语义多组件 | 可训练策略(RL,QA 准确率为奖励) |
| LightMem | 2510.18866 | 2025 | arXiv preprint | 三阶段(感觉/短期/长期) | 轻量压缩过滤 + 睡眠期离线整固 |

2026 年新系统(RL 写入策略、写路由、原生记忆)见 §4。

---

## 1. 核心系统详解

### 1.1 对话与个人助理型外部记忆

#### MemGPT — 2310.08560(2023,arXiv preprint;CoRR 2023,未见会议接收记录)
- 链接:https://arxiv.org/abs/2310.08560 (Packer et al.)
- WHY:LLM 上下文窗口有限,长对话/长文档任务中信息丢失。
- HOW:借鉴操作系统分层存储(虚拟内存),把 LLM 上下文当"主存",外部存储当"磁盘";通过"virtual context management"在两层之间搬运数据,用中断管理控制流。多级:main context(系统指令+工作上下文+FIFO 队列)与 external context(recall storage / archival storage)[正文,待复核]。
- WHAT:文档分析与多会话聊天上均超过固定上下文基线;衍生出开源系统 Letta。局限:依赖底层 LLM 的函数调用可靠性;摘要未报告失败模式。
- 写入决策:**LLM prompt 判断(自主 function call)**——模型自己发起 memory write/archive 调用,决定把什么写进 recall/archival 存储 [正文,待复核;摘要仅说 "intelligently manages different memory tiers"]。无可训练写入策略,无显式遗忘模型。

#### Generative Agents — 2304.03442(2023,UIST 2023)
- 链接:https://arxiv.org/abs/2304.03442 (Park et al.)
- WHY:可信的人类行为模拟需要 agent 记住并反思长期经历。
- HOW:memory stream **以自然语言完整记录全部经历**(摘要原文 "store a complete record of the agent's experiences");检索按 recency + importance + relevance 加权,importance 由 LLM 打分 [正文,待复核];周期性把记忆综合为更高层 reflection,并用于规划。
- WHAT:25 个 agent 的沙盒小镇产生可信个体与涌现社会行为(情人节派对传播);消融证明 observation/planning/reflection 各自关键。局限:成本高,记忆全量增长。
- 写入决策:**全量存储**(写入无选择),选择性全部发生在读取侧(三因子打分);reflection 是唯一的"生成式写入",由重要性累积阈值触发 [正文,待复核]。
- 引用:OpenAlex 两条记录合计约 2000+,是本领域被引最高的记忆架构论文。

#### MemoryBank — 2305.10250(2023,AAAI 2024)
- 链接:https://arxiv.org/abs/2305.10250 (Zhong et al.)
- WHY:长期陪伴/心理咨询场景需要跨会话记忆与用户画像。
- HOW:存储对话记录、逐日事件摘要与不断更新的用户画像(分层摘要)[正文,待复核];核心机制:受 **Ebbinghaus 遗忘曲线**启发的记忆更新——按时间流逝与记忆重要性决定遗忘或强化(摘要明确)。
- WHAT:SiliconFriend 聊天机器人在真实与模拟长对话中显示长期陪伴能力。局限:评测偏定性;遗忘参数为手工设定。
- 写入决策:**启发式打分 + 显式遗忘机制**——第一批把"遗忘"作为一等公民的系统;写入本身近全量(对话摘要),保留与否由时间衰减规则决定。

#### Mem0 — 2504.19413(2025;Crossref 显示同名论文发表于 IOS "Frontiers in Artificial Intelligence and Applications" 2025,DOI 10.3233/faia251160)
- 链接:https://arxiv.org/abs/2504.19413 (Chhikara et al.)
- WHY:生产环境多会话对话一致性;全上下文方案延迟/成本不可行。
- HOW:两阶段 pipeline:从对话增量**抽取 salient 信息**,与既有记忆整合(摘要:"dynamically extracting, consolidating, and retrieving");整合时执行 ADD/UPDATE/DELETE/NOOP 记忆操作 [正文,待复核];变体 Mem0-g 用图结构记忆表示关系。
- WHAT:LOCOMO 上超过六类基线;比全上下文降低 91% p95 延迟、省 90%+ token。局限:LOCOMO 单一基准;后续多篇论文(如 MemoryOS、MIRIX)报告在不同设置下反超它。
- 写入决策:**LLM prompt 判断(抽取式)+ LLM 决定记忆操作**——典型的"LLM 决定存什么、怎么改"路线,无训练信号;写入质量完全取决于抽取 prompt。

#### MemoryOS — 2506.06326(2025,EMNLP 2025 main,DOI 10.18653/v1/2025.emnlp-main.1318)
- 链接:https://arxiv.org/abs/2506.06326 (Kang et al.)
- WHY:固定上下文 + 记忆管理不足 → 长期记忆与个性化短板。
- HOW:OS 式三级存储:短期(对话页)、中期(topic 段)、长期个人记忆;短→中期用 **dialogue-chain FIFO**,中→长期用**分段分页(segmented paging)**组织;四模块:存储/更新/检索/生成(摘要明确);中期段的热度(visit 频率)决定晋升 [正文,待复核]。
- WHAT:LoCoMo 上对 GPT-4o-mini 基线 F1 +49.11%。局限:规则参数(页大小、热度阈值)手工设定。
- 写入决策:**启发式规则(FIFO + 分页 + 热度晋升)**——与 MemoryBank 同属规则派,但规则从"遗忘曲线"换成"OS 页面置换"。

#### MIRIX — 2507.07957(2025,arXiv preprint)
- 链接:https://arxiv.org/abs/2507.07957 (Wang & Chen)
- WHY:扁平、窄域记忆组件无法支撑多模态、用户级长期记忆。
- HOW:六类记忆(Core / Episodic / Semantic / Procedural / Resource / Knowledge Vault)+ **多 agent 框架动态控制与协调更新和检索**(摘要明确);支持屏幕截图等多模态输入。
- WHAT:ScreenshotVQA 比 RAG 基线 +35% 且存储省 99.9%;LOCOMO 85.4% SOTA(该榜单随后被多篇 2026 论文质疑饱和,见 §5)。
- 写入决策:**LLM 多 agent 路由判断**——由 meta/路由 agent 决定新信息该写入哪类记忆、如何更新;无训练信号。

#### LightMem — 2510.18866(2025,arXiv preprint)
- 链接:https://arxiv.org/abs/2510.18866 (Fang et al.)
- WHY:现有记忆系统在线开销过大(token/API 调用)。
- HOW:仿 Atkinson-Shiffrin 三级:感觉记忆(轻量压缩**先过滤无关信息**并按 topic 分组)→ topic 感知短期记忆(组织+摘要)→ 长期记忆 **sleep-time update**(整固与在线推理解耦,离线执行)。
- WHAT:LongMemEval/LoCoMo 上 QA 最高 +7.7%/+29.3%,token 省至 1/38,在线成本最高降百倍级。局限:另有复现论文(2607.29104)称 naive RAG 在同类管理任务上可比,见 §5。
- 写入决策:**启发式轻量过滤(写入前压缩去冗)+ 离线整固**——把写入决策的重活移到"睡眠期",是"写入时机"维度的代表。

### 1.2 检索/知识图谱型(非参数持续学习)

#### HippoRAG — 2405.14831(2024,NeurIPS 2024 poster)
- 链接:https://arxiv.org/abs/2405.14831 (Gutiérrez et al.)
- WHY:RAG 无法像人脑一样把新经验深度整合进已有知识(联想/多跳弱)。
- HOW:仿海马体索引理论:LLM 做 OpenIE 把**全部语料**离线抽成开放知识图谱(海马索引),查询时以 query 实体为种子跑 **Personalized PageRank** 做单步图检索(神经皮层=LLM,海马=KG)。
- WHAT:多跳 QA 超 SOTA 至多 20%,单步检索达到甚至超过迭代检索(IRCoT)且便宜 10-30×。局限:事实型简单检索反而弱于普通 RAG(HippoRAG 2 论文自述)。
- 写入决策:**全量离线索引**——不做选择性写入,所有文档都进 KG;"记忆管理"只有索引构建,无更新/遗忘。

#### HippoRAG 2 — 2502.14802 "From RAG to Memory"(2025,ICML 2025 poster)
- 链接:https://arxiv.org/abs/2502.14802 (Gutiérrez et al.)
- WHY:结构增强 RAG(含 HippoRAG 1)在基础事实记忆上掉分;要同时赢事实/意义构建/联想三类任务。
- HOW:在 PPR 基础上加深段落整合(passage 节点入图)与更有效的在线 LLM 过滤(recognition memory)[正文,待复核;摘要说 "deeper passage integration and more effective online use of an LLM"]。
- WHAT:联想任务超最强 embedding 检索 7%,事实与 sense-making 同步占优;定位为"非参数持续学习"。
- 写入决策:**全量离线索引**(同 HippoRAG);LLM 参与的是检索侧过滤而非写入侧。

#### Zep / Graphiti — 2501.13956(2025,arXiv preprint)
- 链接:https://arxiv.org/abs/2501.13956 (Rasmussen et al.)
- WHY:企业 agent 需要动态融合对话与业务数据,静态文档 RAG 不够。
- HOW:核心引擎 **Graphiti**:时序感知知识图谱,动态摄取非结构化对话与结构化业务数据;实体/关系抽取入图,**维护历史关系**——新事实与旧事实冲突时不删除而是给旧边打时间失效标记(bi-temporal edge invalidation)[正文,待复核;摘要说 "maintaining historical relationships"]。
- WHAT:DMR 基准 94.8% 超 MemGPT 93.4%;LongMemEval 提升至多 18.5% 且延迟降 90%。局限:公司论文,基线设置由厂商选择。
- 写入决策:**全量摄取 + LLM 抽取 + 时间性失效**——写入不筛选,但通过时序边失效实现"软遗忘/版本化",是 KG 派处理记忆更新冲突的代表。

### 1.3 参数/潜在空间型(模型内记忆)

#### MemoryLLM — 2402.04624(2024,ICML 2024 poster)
- 链接:https://arxiv.org/abs/2402.04624 (Wang et al.)
- WHY:部署后 LLM 静态,注入新知识难。
- HOW:transformer + 固定大小潜在记忆池(各层 hidden states,共 1B 参数);新文本经前向计算压缩进记忆池,自更新;旧记忆随更新逐步稀释(设计上的指数衰减遗忘)[正文,待复核]。
- WHAT:模型编辑与长上下文基准有效;近百万次更新后无性能退化。局限:>20k token 后旧知识保留急剧下降(M+ 论文自述)。
- 写入决策:**训练得到的自动压缩(无显式选择)**——写入策略隐含在网络权重里,端到端训练,不可解释、不可配置;遗忘=固定速率稀释。

#### M+ — 2502.00592(2025,ICML 2025 poster)
- 链接:https://arxiv.org/abs/2502.00592 (Wang, Krotov, et al.)
- WHY:MemoryLLM 超过 20k token 即遗忘。
- HOW:在 MemoryLLM 上加长期记忆层:被驱逐的记忆 token 转入长期库,由**共训检索器**(co-trained retriever)在生成时动态取回。
- WHAT:知识保留从 <20k 扩展到 160k+ token,GPU 显存开销相近。
- 写入决策:**自动驱逐进长期库(规则)+ 检索器共训(读侧可训练)**——写入侧仍无选择性判断,可训练的是"读"。

#### MemAgent — 2507.02259(2025,ICLR 2026 Oral)
- 链接:https://arxiv.org/abs/2507.02259 (Yu et al.)
- WHY:线性复杂度处理无限长文档且外推不掉分。
- HOW:agent 工作流分段读文本,维护固定长度文本型记忆,每段**用覆写策略更新记忆**;用多对话 RL(扩展 DAPO)端到端训练该覆写行为(摘要明确)。
- WHAT:8K 上下文训练(32K 文本)外推到 3.5M token QA 损失 <5%,512K RULER 95%+。
- 写入决策:**可训练策略(RL)**——"保留什么、覆写什么"直接由任务奖励优化,是"写入决策可训练化"的旗舰证据(且拿到 ICLR 2026 Oral,对我们的投稿是关键 anchor)。

#### Metis — 2607.26760(2026,arXiv preprint)
- 链接:https://arxiv.org/abs/2607.26760 (Zhang et al.)
- WHY:agent 各能力已内化进基础模型,唯记忆仍靠外挂模块。
- HOW:"memory foundation model" 原型:骨干内持久、动态演化的原生记忆状态 + **原生记忆过程**(通过模型计算自主存取);mid-training 学得;在线维护 gradient-free,更新只需一次前向;推理时权重冻结,记忆状态经标准前向计算自主变换。
- WHAT:展现原生记忆能力并给出强弱项分析;开源 checkpoint。
- 写入决策:**模型内生(mid-training 学到的前向计算)**——写入决策完全内化,连显式策略都没有;代表 2026 年"external memory → native memory"的转向。

### 1.4 经验/技能型(agent 自我改进记忆)

#### Reflexion — 2303.11366(2023,NeurIPS 2023 poster)
- 链接:https://arxiv.org/abs/2303.11366 (Shinn et al.)
- WHY:语言 agent 无法像 RL 那样从试错中快速学习(样本/微调代价大)。
- HOW:不更新权重,用语言反馈强化:agent 对任务反馈信号做**言语自我反思**,反思文本存入 episodic memory buffer,在后续尝试中作为上下文;支持标量/自由文本、内/外部反馈。
- WHAT:HumanEval pass@1 91%(超当时 GPT-4 的 80%);决策/编码/推理三域有效。局限:反思质量依赖模型自评能力;buffer 容量有限(通常保留最近 1-3 条)[正文,待复核]。
- 写入决策:**失败/反馈触发 + LLM 生成**——写入时机是启发式(试验结束、有失败信号),写入内容是 LLM 生成的反思;滑动窗口即遗忘。

#### Voyager — 2305.16291(2023,TMLR)
- 链接:https://arxiv.org/abs/2305.16291 (Wang et al.)
- WHY:Minecraft 中终身学习:持续探索、技能积累、无人干预。
- HOW:三组件:自动课程;**不断增长的技能库**(以可执行代码存储行为,embedding 索引描述);迭代 prompting(环境反馈+执行错误+**自我验证**)。技能=时间扩展、可解释、可组合。
- WHAT:独特物品 3.3×,科技树解锁最快 15.3×;技能库可迁移到新世界。局限:限 Minecraft/代码技能;GPT-4 依赖。
- 写入决策:**环境反馈验证——只有通过执行与自我验证的技能才入库**。这是"写入门控=外部可验证成功信号"的代表,与纯 LLM 判断路线形成对照。

#### ExpeL — 2308.10144(2023,AAAI 2024)
- 链接:https://arxiv.org/abs/2308.10144 (Zhao et al.)
- WHY:API 黑盒模型无法微调;要从经验学习而不改参数。
- HOW:训练任务上自主收集成功/失败轨迹,用自然语言**抽取跨任务 insights**;推理时同时召回 insights 与相似过往经验。insight 库由 LLM 以 ADD/UPVOTE/DOWNVOTE/EDIT 操作维护 [正文,待复核]。
- WHAT:随经验积累性能持续上升;有迁移学习迹象。
- 写入决策:**LLM 跨轨迹归纳 + 投票式增删改**——写入内容是抽象规则而非原始轨迹;LLM 同时管理保留与淘汰。

#### AWM (Agent Workflow Memory) — 2409.07429(2024,arXiv;正式 venue 在 OpenReview/Crossref 均未检索到,未核实)
- 链接:https://arxiv.org/abs/2409.07429 (Wang, Mao, Fried, Neubig)
- WHY:web agent 长程任务中动作轨迹复杂,人类靠复用任务工作流。
- HOW:从过往经验中**归纳常被复用的例程(workflow)**,并选择性提供给 agent 引导后续生成;离线(从训练例)与在线(从测试流,基于模型自评成功的轨迹归纳)两种模式 [在线成功过滤为正文细节,待复核]。
- WHAT:Mind2Web/WebArena 相对成功率 +24.6%/+51.1%,且减少步数;跨任务/网站/域泛化。
- 写入决策:**LLM 归纳(工作流抽象)+(在线模式)自评成功过滤**——写入单位不是事实而是"过程性技能",与 Voyager 的差别是无环境级验证、靠模型自评。

#### Memp — 2508.06433(2025,arXiv preprint)
- 链接:https://arxiv.org/abs/2508.06433 (Fang et al.)
- WHY:程序性记忆要么手工工程要么固化在参数里,脆弱。
- HOW:把过往轨迹蒸馏为细粒度步骤指令 + 高层脚本抽象,系统研究 Build/Retrieval/**Update** 策略;动态机制持续更新、纠正、**废弃(deprecate)**内容。
- WHAT:TravelPlanner/ALFWorld 成功率与效率随记忆细化稳步上升;强模型构建的程序记忆迁移给弱模型仍有大幅收益。
- 写入决策:**LLM 蒸馏 + 显式生命周期管理(含淘汰)**——少数把"记忆折旧"作为实验变量的工作。

#### G-Memory — 2506.07398(2025,NeurIPS 2025 spotlight)
- 链接:https://arxiv.org/abs/2506.07398 (Zhang et al.)
- WHY:多 agent 系统(MAS)记忆过于简陋,忽略协作轨迹,缺跨试验与 agent 级定制。
- HOW:借组织记忆理论,三层图(insight / query / interaction graph)管理 MAS 交互;查询时双向遍历取"高层可泛化 insight"+"细粒度压缩轨迹";执行后**整个层级同化新协作轨迹**而演化。
- WHAT:嵌入式动作 +20.89%、知识 QA +10.12%,即插即用于三种 MAS 框架。
- 写入决策:**LLM 归纳 + 层级图同化**——写入决策发生在"如何把新轨迹合并进三层图",由 LLM 驱动,无训练信号。

### 1.5 RL 训练写入策略(2025 下半年起的新范式)

#### Memory-R1 — 2508.19828(2025,arXiv preprint;OpenAlex 记录 2026,venue 空)
- 链接:https://arxiv.org/abs/2508.19828 (Yan et al.)
- WHY(直接点名本维度):现有管道"static and heuristic-driven, **lacking a learned mechanism for deciding what to store, update, or retrieve**"。
- HOW:两个 RL 微调 agent:Memory Manager 学结构化操作 **ADD/UPDATE/DELETE/NOOP**;Answer Agent 预选并推理相关条目;PPO/GRPO,结果驱动奖励。
- WHAT:仅 152 个训练 QA 对即超强基线,跨 LoCoMo/MSC/LongMemEval 与 3B-14B 模型泛化。
- 写入决策:**可训练策略(outcome-driven RL)**——写入操作被显式建模为动作空间。

#### Mem-α — 2509.25911(2025,arXiv preprint)
- 链接:https://arxiv.org/abs/2509.25911 (Wang et al.)
- WHY(同样点名):预定义指令+工具下,"language models may lack the ability to determine **which information to store**, how to structure it, and when to update it" → 次优构建与信息丢失。
- HOW:RL 框架训练 agent 管理复杂记忆系统(core/episodic/semantic 三组件+多工具);奖励=对全交互历史的下游 QA 准确率;配套多轮交互训练集。
- WHAT:显著超记忆增强基线;30k token 训练泛化到 400k+(13×)。
- 写入决策:**可训练策略(RL,直接以记忆构建质量为优化目标)**。

#### 同方向补充(均 arXiv preprint,写作时可合并引用)
- **MemBuilder** 2601.05488(2026):RL + 归因式稠密奖励(会话级合成问题提供中间奖励;按组件下游贡献加权梯度),4B 模型超闭源基线。https://arxiv.org/abs/2601.05488
- **DeltaMem** 2604.01560(2026):单 agent 端到端记忆管理;合成对话+操作级更新标签;Memory-based Levenshtein Distance 奖励;LoCoMo/HaluMem/PersonaMem 超产品级基线。https://arxiv.org/abs/2604.01560
- **AdaMem** 2606.21144(2026):把本维度形式化为 **preference-conditioned write control**:自然语言 Memory Policy 按交互情境个性化"什么值得写",从周期反馈更新;发现当前模型"难以把策略转化为可靠的选择性写入行为"(policy execution 是核心瓶颈)。https://arxiv.org/abs/2606.21144
- **Dual-Layer Agentic Memory** 2608.22215(2026):把管理移到**写阶段**:cost-aware epistemic routing 把输入三分类(non-write / write-new / write-update),小-大模型级联;周期性把高价值外部记忆经 SFT **整固进参数**;剪掉 68% 冗余记忆仍保 98%+ QA EM。https://arxiv.org/abs/2608.22215

### 1.6 系统/OS 型统一框架

#### MemOS — 2507.03724(2025,arXiv preprint)
- 链接:https://arxiv.org/abs/2507.03724 (Li et al.)
- WHY:LLM 缺乏明确定义的记忆管理系统;RAG 是无状态权宜,无生命周期控制。
- HOW:把记忆当可管理系统资源:统一**明文、激活、参数**三类记忆的表示/调度/演化;基本单元 **MemCube** 封装内容+溯源/版本元数据,可组合、迁移、融合,支持三类记忆间转换。
- WHAT:定位为记忆中心系统框架(可控性/可塑性/可演化性);摘要不含定量主结果。
- 写入决策:**系统调度层策略**;具体写入判定机制摘要未说明,待读全文。

---

## 2. 初步 Taxonomy:记忆表示 × 写入决策方式

| 记忆表示 \ 写入决策 | 全量存储(读侧过滤) | 启发式规则/打分 | LLM prompt 判断 | 环境/结果验证门控 | 可训练策略(RL/SFT) | 内生(训练进权重) |
|---|---|---|---|---|---|---|
| 扁平自然语言条目/事件流 | Generative Agents | MemoryBank(遗忘曲线) | Mem0、A-Mem(笔记+链接演化) | — | Memory-R1、DeltaMem、AdaMem* | — |
| 分层/页式结构化存储 | — | MemoryOS(FIFO+分页)、LightMem(过滤+睡眠整固) | MemGPT(自主分页)、MIRIX(多 agent 路由六类记忆) | — | Mem-α、MemBuilder | — |
| 知识图谱/时序图 | HippoRAG、HippoRAG 2(全量索引) | Zep/Graphiti(全量+时间性边失效) | G-Memory(层级图同化) | — | — | — |
| 潜在向量/隐状态 | — | M+(自动驱逐规则) | — | — | MemAgent(RL 覆写文本记忆)† | MemoryLLM、Metis |
| 程序性:代码/工作流/insight | — | — | ExpeL(归纳+投票)、Memp(蒸馏+废弃)、AWM(归纳) | Voyager(执行+自验证成功才入库)、AWM 在线(自评成功过滤) | — | — |
| 混合/OS 抽象 | — | — | MemOS(调度层,细节待读全文) | — | — | (MemOS 参数记忆入口) |

\* AdaMem 的策略本身是自然语言、由反馈更新,介于"LLM 判断"与"可训练"之间。
† MemAgent 记忆表示是文本,但覆写策略经 RL 训练,故归入"可训练"列。

**轴上观察**:
1. 2023 年系统占满"全量/启发式/LLM 判断"三列;"可训练策略"列在 2025-08 之前几乎为空,2025-08 至 2026-06 密集出现 6+ 个系统(Memory-R1、Mem-α、MemAgent、MemBuilder、DeltaMem、AdaMem、CMI-Mem 2607.20553、ContextPilot 2608.28476)。
2. "环境验证门控"列只有代码/技能类记忆占据——因为只有可执行记忆才有廉价的客观成功信号;事实型记忆缺乏等价验证器,这是明显的结构性空白。
3. KG 派全部走全量写入,把智能放在检索(PPR)或失效(时序边)上,没有 KG 系统做选择性写入。

---

## 3. 写入决策机制:总体格局

**五代演进**(可直接支撑相关工作叙事):
1. **全量存储,读侧智能**(2023:Generative Agents、HippoRAG 系):写什么不做决策,靠检索打分/图算法过滤。成本随交互线性增长;Mem0/MIRIX 等以延迟与 token 成本为攻击点。
2. **启发式规则**(2023-2025:MemoryBank 遗忘曲线、MemoryOS FIFO/分页、LightMem 感觉过滤):写入/保留由人工规则决定,便宜可控,但规则与任务无关、不自适应。
3. **LLM prompt 判断**(2023-2025 主流:MemGPT、Mem0、A-Mem、ExpeL、AWM、MIRIX、G-Memory、Memp):由(冻结)LLM 决定抽什么、存哪、改不改。灵活但无优化目标——Memory-R1 与 Mem-α 的摘要都明确把"缺 learned mechanism 决定存什么"当作攻击的核心缺陷;AdaMem 进一步实证:即使给了明确的写入策略,模型也难以可靠执行选择性写入。
4. **可训练写入策略**(2025-08 起爆发:MemAgent RL 覆写→ICLR 2026 Oral;Memory-R1 操作级 RL;Mem-α 以下游 QA 为奖励的构建 RL;MemBuilder 稠密归因奖励;DeltaMem 编辑距离奖励;Dual-Layer 的写路由级联+参数整固)。共同点:把 ADD/UPDATE/DELETE/NOOP 或覆写/路由作为动作空间,用下游任务结果做奖励。开放问题:奖励稀疏(MemBuilder 的动机)、训练数据合成、跨记忆架构泛化。
5. **内生记忆**(2024-2026:MemoryLLM→M+→Metis"memory foundation model"):写入决策消失于前向计算,由 mid-training 学得;与外挂系统的对照是 2026 年的活跃辩论(Metis 摘要明言外挂模块 vs native capability)。

**遗忘/淘汰机制的分布**:显式遗忘仍是少数派——MemoryBank(时间衰减)、MemoryLLM(固定稀释)、Zep(时间性失效,不删除)、Memp(deprecate)、Reflexion(滑动窗口)、Dual-Layer(写前剪枝 68%);多数 LLM 判断派(Mem0、A-Mem、MIRIX)有 DELETE 操作但触发完全依赖 prompt。2026-09 出现专门审计遗忘代价的工作(What Eviction Destroys,2609.08279)。

**评测基准现状**:LoCoMo 与 LongMemEval 是事实/对话线的两个事实标准(Mem0、MemoryOS、MIRIX、LightMem、Memory-R1 都用);WebArena/Mind2Web(AWM)、ALFWorld/TravelPlanner(Memp)是经验线标准。注意:Anatomy of Agentic Memory(2602.19320)指出基准饱和、judge 敏感、backbone 依赖等系统性问题;Harness the Memory(2608.15008)26 指标对照发现**没有单一记忆基底全面占优**;2608.12888 发现 agent 直接搜索原始聊天记录可匹敌结构化记忆;2607.29104 复现称 naive RAG 不输 LightMem——相关工作中应引用这组怀疑论以示公允。

---

## 4. 2025-2026 其他新系统扫描(简条目,均 arXiv API 返回,未逐一读全文)

| 系统 | ID | 日期 | 一句话 | 写入决策线索(仅摘要) |
|---|---|---|---|---|
| MAGMA | 2601.03236 | 2026-01 | 多图 agentic 记忆架构(语义/情景/程序等多图) | 摘要未详,待读全文 |
| AtomMem | 2606.19847 | 2026-06 | Fact Executor 选择性抽取高价值原子事实,层级事件+时间画像 | LLM 选择性抽取(value-dense) |
| Filesystem-Based Memory | 2607.26637 | 2026-07 | 以文件系统为记忆基底,谈组织/演化/可持续性 | 待读全文 |
| CMI-Mem | 2607.20553 | 2026-07 | CMI 增强 RL 做可泛化记忆管理 | 可训练策略 |
| ContextPilot | 2608.28476 | 2026-08 | 细粒度 RL 教 agent 主动上下文管理 | 可训练策略 |
| RippleMem | 2608.13334 | 2026-08 | 联想式回忆(从孤立检索到关联召回) | 待读全文 |
| MemFuse | 2608.18704 | 2026-08 | 碎片观察的多源记忆融合 | 待读全文 |
| Zero-Mem | 2607.29377 | 2026-07 | 零 token 记忆操作(降在线开销) | 待读全文 |
| AutoMem | 2608.14621 | 2026-07 | 文本梯度递归自改进,自动搜索记忆架构 | 架构级搜索(含写策略) |
| Metis | 2607.26760 | 2026-07 | 首个 memory foundation model(见 §1.3) | 内生 |
| What Eviction Destroys | 2609.08279 | 2026-09 | 恢复-反事实审计:遗忘/驱逐的真实代价 | (审计类) |
| Retrieval-Driven Memory Reconsolidation | 2609.16053 | 2026-09 | 检索触发的记忆再整固 | 待读全文 |

多智能体记忆:G-Memory(§1.4)之外还有 LEGOMem(AAMAS 2026,模块化程序记忆,Crossref DOI 10.65109/vlua1303)。

---

## 5. 相关综述与评测论文(定位我们的 taxonomy 贡献时引用)

| 论文 | ID | venue | 要点 |
|---|---|---|---|
| A Survey on the Memory Mechanism of LLM-based Agents | 2404.13501 | ACM TOIS 2025(OpenAlex,引 110+) | 最早的系统综述;按设计/评测/应用组织 |
| A Survey of Agent Memory in the Second Half | 2602.06052 | arXiv 2026 | 三维:substrate × cognitive mechanism × subject;明确指出"memory management itself is becoming a trainable capability"——与我们的写入决策维度直接呼应 |
| From Storage to Experience | 2605.06716 | arXiv 2026 | 演化框架:Storage → Reflection → Experience 三阶段 |
| Anatomy of Agentic Memory | 2602.19320 | arXiv 2026 | 四类记忆结构 taxonomy + 评测系统性缺陷(基准饱和/judge 敏感/backbone 依赖/系统开销) |
| Harness the Memory | 2608.15008 | arXiv 2026 | 26 指标控制变量评测:无单一 substrate 全面占优,提出 substrate routing |
| Always-On Agents | 2606.30306 | arXiv 2026 | 持久记忆/状态/治理综述 |
| 记忆安全综述 | 2604.16548 | arXiv 2026 | 记忆生命周期的攻防与治理(记忆投毒等) |

**对 ICLR 相关工作写作的三个提示**:
1. 现有综述的 taxonomy 轴是"表示/认知类型/主体",**把"写入决策机制"作为第一性组织轴的系统梳理仍缺位**;Memory-R1、Mem-α、AdaMem 三篇摘要不约而同把"decide what to store"点为未解问题,可作为动机引言的直接证据链。
2. 引用锚点按 venue 强度排序:MemAgent(ICLR 2026 Oral)、G-Memory/A-Mem(NeurIPS 2025)、HippoRAG 2 与 M+(ICML 2025)、MemoryOS(EMNLP 2025)、HippoRAG(NeurIPS 2024)、MemoryLLM(ICML 2024)、Reflexion(NeurIPS 2023)、Generative Agents(UIST 2023)、MemoryBank/ExpeL(AAAI 2024)、Voyager(TMLR)。MemGPT、Mem0、Zep、MemOS、MIRIX 均以 preprint/产业系统身份引用(Mem0 有 IOS FAIA 2025 DOI)。
3. 怀疑论对照组(2608.12888 原始日志搜索、2607.29104 naive RAG 复现、2602.19320 基准饱和)+ 遗忘审计(2609.08279)可用来论证:写入侧的选择性不是效率锦上添花,而是评测可信性问题。

---

## 附:核实状态清单
- 已核实 venue(OpenReview/OpenAlex/Crossref 返回):Reflexion、Generative Agents、MemoryBank、ExpeL、Voyager、MemoryLLM、HippoRAG、HippoRAG 2、M+、A-Mem、G-Memory、MemAgent、MemoryOS、Mem0(FAIA DOI)、TOIS 综述。
- 检索无正式 venue 记录(按 preprint 引用):MemGPT、Zep、MemOS、MIRIX、Memp、AWM、LightMem、Memory-R1、Mem-α、MemBuilder、DeltaMem、AdaMem、AtomMem、Dual-Layer、Metis、MAGMA。
- 标注 [正文,待复核] 的机制细节在引用前需对照全文 PDF(arxiv.org/abs/<id> 可直连)。
