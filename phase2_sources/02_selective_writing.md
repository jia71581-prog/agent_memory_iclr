# 选择性记忆写入:LLM Agent 如何判断"什么值得记" (memory write policy / importance estimation / selective retention / curation / forgetting)

调研日期:2026-09-16。检索通道:arXiv API(export.arxiv.org,布尔检索)+ OpenAlex(引用数/venue)+ OpenReview(接收状态)。所有 arXiv ID 均来自 API 返回;引用数来自 OpenAlex(同一论文常分裂为多条记录,数字为下界);标注"正文机制"的字段来自论文正文的公知描述而非摘要,标注"摘要未说明"的字段无法从 API 摘要核实。

---

## 1. 启发式重要性打分(不可训练的手写打分函数)

### 1.1 Generative Agents (2023, arXiv:2304.03442, UIST 2023)
- URL: https://arxiv.org/abs/2304.03442 。OpenAlex 引用 ~1823+(主记录)。
- "值得记住"的形式化:**写入端无筛选**——所有观察全部写入 memory stream;选择性发生在检索端,score = recency + importance + relevance 线性加权;importance 由 LLM 对每条记忆打 1-10 分(正文机制,摘要未详述)。reflection 由近期记忆 importance 累计和超阈值触发,生成高层反思再写回。
- 可训练:否。训练信号:无。
- 证据强度:消融显示 observation/planning/reflection 各自对"可信度"人工评估有贡献(摘要);**评测目标是行为可信度,不是任务效用**。
- 局限:importance 分数无校准、与下游任务效用无耦合;写入不做取舍,容量问题靠检索兜底。

### 1.2 MemoryBank (2023, arXiv:2305.10250, AAAI 2024)
- URL: https://arxiv.org/abs/2305.10250 。OpenAlex 引用 ~189(AAAI 记录)。
- 形式化:艾宾浩斯遗忘曲线——按"时间流逝 + 记忆相对重要性"决定遗忘与强化(摘要原文);正文机制:保留概率 R=e^(-t/S),每次被检索命中则记忆强度 S 增大。写入端为逐日摘要+用户画像合成。
- 可训练:否。训练信号:无。
- 证据强度:SiliconFriend 陪伴机器人,真实对话定性 + ChatGPT 模拟用户定量;**遗忘机制本身是否带来增益的消融:摘要未说明**。
- 局限:曲线参数任意;"重要性"如何量化摘要未说明;评测围绕共情/回忆而非记忆管理本身。

### 1.3 SCM: Self-Controlled Memory (2023, arXiv:2304.13343)
- URL: https://arxiv.org/abs/2304.13343
- 形式化:memory controller 决定**何时以及如何**调用 memory stream 中的记忆;写入侧近乎全存。不可训练,plug-and-play。局限:控制器判断纯 prompt,写入选择性弱。

### 1.4 HiMem (2026, arXiv:2601.06377)
- URL: https://arxiv.org/abs/2601.06377
- 形式化:Topic-Aware Event–Surprise 双通道分段构建 Episode Memory,多阶段抽取管线构建 Note Memory,层级链接。不可训练。证据:长程对话基准(摘要未给数字)。与线索 4 交叉。

### 1.5 MemSIF (2026, arXiv:2608.01742) —— 对空白分析关键
- URL: https://arxiv.org/abs/2608.01742
- 贡献:显式命名两类失配:**TSM(时间-结构失配)**与 **DUM(Delayed Utility Manifestation:写入时显著性不能可靠预测未来查询效用)**,并以结构化交互→双轨事实记忆缓解。不可训练(摘要未说明有学习组件)。
- 意义:这是对"写入时打分 ≠ 未来价值"最直接的问题陈述之一。

---

## 2. LLM prompt 式判断(冻结 LLM 当写入判官)

### 2.1 Mem0 (2025, arXiv:2504.19413)
- URL: https://arxiv.org/abs/2504.19413 。OpenAlex 引用 ~42+。
- 形式化:两阶段——LLM 从对话增量中**抽取显著事实**(extraction),再对每条候选事实与现有相似记忆比较,由 LLM 决策 ADD / UPDATE / DELETE / NOOP(正文机制;摘要表述为 dynamically extracting, consolidating);变体加图记忆。
- 可训练:否(判断由冻结 LLM prompt 完成)。训练信号:无。
- 证据强度:LOCOMO 上对 6 类基线系统性对比,LLM-as-a-Judge 相对 OpenAI memory +26%,p95 延迟 -91%,token 成本 -90%+。工程证据强,科学消融(哪步判断贡献多少)摘要未说明。
- 局限:什么算"salient"完全托付给 LLM 的先验;无任何来自下游效用的反馈回路;评测限对话 QA。

### 2.2 A-Mem (2025, arXiv:2502.12110, NeurIPS 2025 poster)
- URL: https://arxiv.org/abs/2502.12110
- 形式化:Zettelkasten 式笔记构建——每条新记忆生成含上下文描述/关键词/标签的结构化 note,动态与旧记忆建立链接,并触发旧记忆的更新演化(memory evolution)。写入选择性弱(每条都建笔记),创新在**记什么形式/怎么组织**而非"是否记"。
- 可训练:否。证据:六个基础模型上的实证(摘要未给数字)。局限:D-MEM(见 4.3)指出其 append-and-evolve 导致 O(N^2) 写延迟与高 token 成本。

### 2.3 MemGPT (2023, arXiv:2310.08560)
- URL: https://arxiv.org/abs/2310.08560
- 形式化:OS 式虚拟上下文管理,LLM 通过 function call 自主在主上下文与外部存储间**搬运**信息(何时写出/召回自主决定),中断机制管理控制流。不可训练。证据:长对话与文档分析两域。局限:写什么的判断仍是 prompt 内隐;无重要性度量。

### 2.4 Reflective Memory Management, RMM (2025, arXiv:2503.08026, Google)
- URL: https://arxiv.org/abs/2503.08026
- 形式化:**Prospective Reflection**(前瞻:以未来检索为目的决定以什么粒度/形式把对话蒸馏进记忆)+ **Retrospective Reflection**(回顾:根据生成中被引用情况,以在线 RL 微调 reranker)。注意:其 RL 作用在检索排序侧,不在写入侧。
- 可训练:部分(检索器可学,写入仍 prompt)。训练信号:引用/归因信号(citation-based)。
- 意义:首批把"写入为未来检索服务"显式化的工作,但学习信号没有打到写入决策上。

### 2.5 其他同类
- **TiM: Think-in-Memory** (2023, arXiv:2311.08719, https://arxiv.org/abs/2311.08719):存"想过的思考"而非原始文本,回避重复推理;写入内容=推理产物。不可训练。
- **MemInsight** (2025, arXiv:2503.21760, https://arxiv.org/abs/2503.21760):LLM 自主给历史交互做属性增强(标注),改善表示与检索;写入判断仍是 prompt。
- **Zep** (2025, arXiv:2501.13956, https://arxiv.org/abs/2501.13956):时序知识图 Graphiti,新事实到来时旧边失效/更新(temporal invalidation),工业系统;DMR 上超 MemGPT。
- **SeCom** (2025, arXiv:2502.05589, https://arxiv.org/abs/2502.05589):实证"记忆单元粒度很重要"(turn/session/摘要各有失配),提出 segment 级构建+压缩去噪。对"写入粒度"这一子问题的少见系统实验。
- **GAM** (2026, arXiv:2604.12285, https://arxiv.org/abs/2604.12285):把"编码"与"巩固"解耦,事件进展图仅在语义转移时并入主题联想网络——写入时机由语义转移触发。

---

## 3. 可学习 / RL 写入策略(2025-2026 的主战场)

### 3.1 Memory-R1 (2025-08, arXiv:2508.19828, ACL 2026)
- URL: https://arxiv.org/abs/2508.19828
- 形式化:Memory Manager 学习结构化操作 ADD/UPDATE/DELETE/NOOP,Answer Agent 预筛+推理;两者都用 outcome-driven RL(PPO/GRPO)微调。
- 训练信号:下游 QA 答对与否(outcome reward);**仅 152 个训练 QA 对**。
- 证据:LoCoMo/MSC/LongMemEval 三基准、3B-14B 多尺度超过强基线。摘要明确批评现有管线"static and heuristic-driven, lacking a learned mechanism for deciding what to store"。
- 局限:reward 经由 Answer Agent 间接传导,写入操作的 credit assignment 粗;操作空间固定为四种。

### 3.2 MEM1 (2025-06, arXiv:2506.15841, ICLR 2026 Poster)
- URL: https://arxiv.org/abs/2506.15841
- 形式化:端到端 RL 学习**恒定大小的内部状态更新**——每轮把旧状态与新观察整合,策略性丢弃无关/冗余信息;记忆=推理状态本身,无显式打分函数。
- 训练信号:任务成绩(多轮 QA/webshop),环境由既有数据集组合合成。
- 证据:16-objective multi-hop QA 上 MEM1-7B 性能 3.5x、内存 3.7x 优于 Qwen2.5-14B;超训练 horizon 泛化。
- 局限:"值得记"完全内隐在策略里,不可解释、不可移植到外部记忆库场景。

### 3.3 MemAgent (2025-07, arXiv:2507.02259)
- URL: https://arxiv.org/abs/2507.02259
- 形式化:分段读长文,固定长度记忆用 **overwrite 策略**更新(写=覆盖旧内容的决策);multi-conv DAPO RL 端到端训练。
- 证据:8K 训练外推 3.5M token QA 损失 <5%,512K RULER 95%+。
- 局限:面向长文档 QA 的工作记忆,不是跨会话长期记忆;"值得记"=对当前任务答案有用。

### 3.4 Mem-α (2025-09, arXiv:2509.25911;OpenReview 显示 Submitted to ICLR 2026)
- URL: https://arxiv.org/abs/2509.25911
- 形式化:RL 训练 agent 操作复杂三层记忆系统(core/episodic/semantic + 多工具);动机明确:"language models may lack the ability to determine which information to store"。
- 训练信号:对完整交互历史的下游 QA 精度;专门构建的多轮交互+评测问题训练集。
- 证据:训练最长 30k token,泛化到 400k+(13x);超过既有记忆增强基线。
- 局限:reward 仍是 QA 采样查询分布的期望;数据集合成质量决定学到的"价值观"。

### 3.5 Memory-as-Action / MemAct (2025-10, arXiv:2510.12635)
- URL: https://arxiv.org/abs/2510.12635
- 形式化:把工作记忆管理(上下文的删除/插入 in-place 编辑)作为**策略动作**,与任务动作联合端到端 RL;提出 DCPO 解决动态上下文更新破坏训练的问题。
- 证据:MemAct-RL-14B 匹配 16x 大模型精度,平均上下文 -51%。
- 局限:上下文编辑=工作记忆层,不解决持久跨任务记忆的写入。

### 3.6 AgeMem (2026-01, arXiv:2601.01885)
- URL: https://arxiv.org/abs/2601.01885
- 形式化:LTM+STM 统一进 agent 策略,记忆操作(store/retrieve/update/summarize/discard)作为工具动作;三阶段渐进 RL + **step-wise GRPO 专门应对记忆操作导致的稀疏且不连续奖励**。
- 证据:5 个长 horizon 基准、多骨干一致提升。
- 意义:官方承认标准 GRPO 在记忆操作上失效,需要专门算法。

### 3.7 Fine-Mem (2026-01, arXiv:2601.08435) —— 直击奖励稀疏
- URL: https://arxiv.org/abs/2601.08435
- 形式化:批评"以最终任务成绩为主 reward 导致严重奖励稀疏与无效 credit assignment";提出 Chunk-level Step Reward(用 chunk 专属辅助 QA 给每步即时奖励)+ Evidence-Anchored Reward Attribution(把全局奖励按推理中实际被引用的记忆条目回溯分配)。
- 证据:Mem-α 基准与 MemoryAgentBench 上超强基线,跨骨干泛化。
- 意义:目前对"单条写入操作的价值如何度量"给出最细粒度的工程答案:**一条记忆的价值 = 它作为证据被下游推理引用并答对的贡献**。

### 3.8 Memory-R2 (2026-05, arXiv:2605.21768) —— 指出 RL 训练的结构性缺陷
- URL: https://arxiv.org/abs/2605.21768
- 形式化:指出记忆使 agent 的过去动作成为未来环境的一部分——不同 rollout 写入不同记忆后不再共享同一中间记忆状态,**GRPO 等组相对方法的"同一环境采样"假设被破坏**,轨迹级奖励对长 horizon 记忆操作给出有噪/有偏的 credit;提出公平 credit assignment 训练框架。
- 意义:从优化理论层面论证了为什么 naive RL 学写入策略不稳。

### 3.9 CMI-Mem (2026-07, arXiv:2607.20553) —— 内在信息量奖励
- URL: https://arxiv.org/abs/2607.20553
- 形式化:混合奖励 = extrinsic QA 项(端任务正确性)+ intrinsic **条件互信息 CMI** 项(新输入相对当前记忆状态贡献的信息量,不依赖采样查询);明确批评"QA reward 通过采样查询分布和固定 reader 来给记忆定价"。
- 证据:跨记忆使用场景迁移提升,训练/推理更高效;代码与 4B checkpoint 开源。
- 意义:第一批把"记忆价值"与具体查询分布解耦的可计算代理指标。

### 3.10 其他
- **ICML: Interactive Memory Learning** (2026-09, arXiv:2609.17088, https://arxiv.org/abs/2609.17088):批评静态启发式"passively archived without adaptive memory valuation";Planner 选择性编码高价值信息 + Trigger,专家数据合成 + 在线 RL。
- **MemFactory** (2026-03, arXiv:2603.29493, https://arxiv.org/abs/2603.29493):承认 RL 记忆管道"highly fragmented and task-specific",给统一训练/推理框架(基础设施,非新策略)。
- **BudgetMem** (2025-11, arXiv:2511.04919, https://arxiv.org/abs/2511.04919):特征式显著性打分(实体密度、TF-IDF、话语标记、位置偏置)+ 学习门控,预算约束下决定存什么。证据强度弱(独立作者,基准细节摘要未说明)。
- **TRUSTMEM** (2026-06, arXiv:2606.25161, https://arxiv.org/abs/2606.25161):Memory Transition Verifier 按 coverage/preservation/faithfulness 评估记忆更新转移,同一记忆状态下构造候选更新的偏好对做偏好引导 RL——把"好的写入"定义为可验证的转移属性而非下游 QA。
- **MIA** (2026-04, arXiv:2604.04503, https://arxiv.org/abs/2604.04503):Manager-Planner-Executor,交替 RL,记忆为压缩检索轨迹。
- **StackPlanner** (2026-01, arXiv:2601.05890, https://arxiv.org/abs/2601.05890):RL 学习检索/利用协调经验(多智能体侧)。

---

## 4. Surprise / 新颖性驱动写入

### 4.1 EM-LLM (2024, arXiv:2407.09450, ICLR 2025 Poster)
- URL: https://arxiv.org/abs/2407.09450
- 形式化:**Bayesian surprise**(token 级预测意外度)+ 图论边界精化,在线把 token 流切成事件单元;检索两阶段(相似度+时间邻接)。无微调。
- 注意:surprise 决定**事件边界(记忆单元怎么切)**,不是"是否保留"——所有事件都存;它解决表示粒度而非取舍。
- 证据:LongBench 等(摘要未给数字);无需微调即可近乎无限上下文。

### 4.2 Titans (2024-12, arXiv:2501.00663, NeurIPS 2025 poster)
- URL: https://arxiv.org/abs/2501.00663
- 形式化:神经长期记忆模块在**测试时**按记忆目标的梯度在线更新——正文机制:momentary surprise = 关联损失的梯度,加动量(past surprise)与遗忘门(weight decay);"越意外越该记"。
- 可训练:是(记忆模块参数即记忆,测试时持续学习)。训练信号:自监督关联/重构损失,非任务效用。
- 证据:语言建模/常识/基因组/时序超 Transformer 与现代线性 RNN,2M+ 上下文 needle 任务。
- 局限:参数化记忆,不产生可审计的显式记忆条目;surprise 是可压缩性的代理,不等于对未来任务有用。

### 4.3 D-MEM (2026-03, arXiv:2603.14597)
- URL: https://arxiv.org/abs/2603.14597
- 形式化:**奖励预测误差 RPE 路由**——轻量 Critic Router 对刺激评 Surprise + Utility 两维;低 RPE 输入旁路/进 O(1) 快缓冲,高 RPE(事实矛盾、偏好转变)触发"多巴胺"信号启动 O(N) 记忆演化管线。
- 可训练:Critic Router 是否训练摘要未说明。证据:自建 LoCoMo-Noise(注入噪声),token -80%+,多跳与对抗鲁棒性超基线。
- 意义:少见的把"值不值得触发昂贵写入"显式建为两维打分并按成本分级的设计。

### 4.4 Nous (2026-06, arXiv:2606.22030) —— 负结果,对空白分析关键
- URL: https://arxiv.org/abs/2606.22030
- 形式化:实体-属性对为分类概率分布,闭式贝叶斯更新;信息论 surprise 驱动信念修正,熵驱动遗忘。
- 核心发现:**LoCoMo 消融显示贝叶斯信念更新相对朴素 last-write-wins 几乎无增益,因为现有对话记忆基准很少包含矛盾或可靠性不同的证据**;在受控矛盾基准上则显著更好。
- 意义:直接证明现有基准区分不出写入/更新策略的优劣——评测层面的空白证据。

### 4.5 相关
- **HiMem**(见 1.4):event-surprise 分段。
- **RecMem** (2026-05, arXiv:2605.16045, https://arxiv.org/abs/2605.16045):以**语义重现(recurrence)**为"值得抽取"的信号——先低成本存潜意识层,观察到语义相似交互反复出现才调 LLM 做情景/语义记忆抽取;动机是 token 成本(eager consolidation 太贵)。可看作 novelty 的反面:重复度高才值得固化。

---

## 5. 反思 / 经验蒸馏式写入(记"教训/技能"而非"事实")

### 5.1 Reflexion (2023, arXiv:2303.11366, NeurIPS 2023)
- URL: https://arxiv.org/abs/2303.11366 。OpenAlex 引用 ~497+(主记录)。
- 形式化:**写入触发器 = 任务反馈信号(通常是失败)**;把标量或自由文本反馈转成言语反思,存入 episodic buffer 供后续 trial。无权重更新。
- 证据:决策/推理/编程(HumanEval pass@1 91%,正文数字)显著提升。
- 局限:记忆生命周期限于同一任务的重试;跨任务保留什么无机制;无容量管理。

### 5.2 ExpeL (2023, arXiv:2308.10144, AAAI 2024)
- URL: https://arxiv.org/abs/2308.10144 。OpenAlex 引用 ~119+。
- 形式化:跨任务自主收集经验池,从成功/失败轨迹对比中用自然语言**提炼 insights**;正文机制:insight 池由 LLM 以 ADD/UPVOTE/DOWNVOTE/EDIT 操作维护(投票即软性重要性计数)。无参数更新。
- 证据:多任务上超 Reflexion/ReAct;有迁移实验。
- 局限:insight 质量无 ground truth;UPVOTE 计数是唯一的价值累积机制,未与下游效用闭环。

### 5.3 Agent Workflow Memory, AWM (2024, arXiv:2409.07429, ICML 2025)
- URL: https://arxiv.org/abs/2409.07429
- 形式化:从过去轨迹中**归纳可复用 workflow**(常用子程序),选择性提供给后续任务;正文机制:online 场景仅从被判为成功的轨迹归纳——**写入门槛=成功信号**。
- 证据:Mind2Web/WebArena,相对基线成功率大幅提升(正文:+24.6%/+51.1% 相对),跨任务/域/网站泛化。
- 局限:成功判定本身由模型/规则估计,误判会写入坏 workflow;何时淘汰过时 workflow 摘要未说明。

### 5.4 Voyager (2023, arXiv:2305.16291)
- URL: https://arxiv.org/abs/2305.16291 。OpenAlex 引用 ~204+。
- 形式化:技能库——**通过环境反馈+执行错误+自我验证迭代改进,验证通过的程序才入库**(写入门槛=self-verification);技能以代码+描述嵌入存储。
- 证据:Minecraft 独特物品 3.3x、里程碑速度 15.3x(正文数字);技能可迁移新世界。
- 局限:技能价值=能否成功执行,无相对重要性;库只增不减。

### 5.5 后续与风险
- **Mem2Evolve** (2026-04, arXiv:2604.10923, https://arxiv.org/abs/2604.10923):经验记忆与资产(工具/专家)记忆共演化,经验指导新资产创建。
- **EDGE** (2026-08, arXiv:2608.21946, https://arxiv.org/abs/2608.21946):检索到的经验只作训练时脚手架,把增益蒸馏进参数(经验记忆的"毕业机制")。
- **MARS** (2025, arXiv:2503.19271, https://arxiv.org/abs/2503.19271):反思 + 艾宾浩斯记忆优化,User/Assistant/Checker 三 agent。
- 安全面:**OEP** (2026-05, arXiv:2605.18930):局部正确但不可迁移的"干净经验"可毒化反思式自演化 agent;**EvoBreak** (2026-08, arXiv:2608.01759):单独良性的经验组合后突破安全边界;**记忆来源洗白** (2026-07, arXiv:2607.29167):LLM 巩固过程会把不可信观察改写成貌似用户历史。三者共同说明:agent 自主写入判断目前**没有可信度/来源维度**,是攻击面。

---

## 6. 早期"learning what to remember"脉络(历史定位)

- **Neural Turing Machines** (2014, arXiv:1410.5401, https://arxiv.org/abs/1410.5401):可微读写头+注意力寻址,端到端梯度学习何时写什么;信号=任务损失。证明"写入策略可学"的原点。
- **DNC** (2016, Nature, doi:10.1038/nature20101, OpenAlex 引用 1447;非 arXiv 首发):在 NTM 上加 usage 驱动的分配与时间链接——**usage-based allocation 即最早的"学习驱动的驱逐/复用"**。
- **Learning to Remember Rare Events** (2017, arXiv:1703.03129, https://arxiv.org/abs/1703.03129):终身 kNN 记忆模块,除最近邻查询外全可微,**无额外监督**端到端训练;价值定义=对稀有事件的一次性泛化。
- **MERLIN** (2018, arXiv:1803.10760, https://arxiv.org/abs/1803.10760):论证"记忆不够,还须正确的存储格式"——**存什么由无监督预测建模(压缩/预测表征)决定,而非全靠 RL 端到端**;是今天 surprise/预测误差写入线(EM-LLM/Titans)的直系前身。
- 定位:这一代方法在小规模可微设置下已把"写入即学习问题"说清,但假设梯度可穿透记忆介质;frozen LLM + 文本记忆使梯度断裂,当今 RL 记忆管理(线索 3)本质是在离散文本操作空间重做这条线,credit assignment 因此重新变难。

---

## 7. 遗忘与容量管理(decay / eviction / consolidation)

- **PA-Bench 论文** (2026-09, arXiv:2609.05767, https://arxiv.org/abs/2609.05767) —— 对空白分析关键:现有 eviction 基准把用户未来问题包含在被压缩 prompt 里(泄漏"预知");去掉后,**所有 training-free 打分策略下,顺带提到的过敏信息存活到需要它的问题时的比例 0-1%(全记忆 97%)**;结论:压缩缓存是推理复用机制而非持久层,安全关键事实需要独立可审计存储。
- **Retain or Consolidate** (2026-07, arXiv:2607.17545, https://arxiv.org/abs/2607.17545):把"何时压缩、选哪个算子(Merge/Abstract/Rewrite)"形式化为预算依赖的效用分解(对被省略证据的 coverage 效应 + 对已保留原始证据的带符号 replacement 效应)。少见的形式化决策理论视角。
- **Sleeping Agent / SWC** (2026-08, arXiv:2608.11775, https://arxiv.org/abs/2608.11775):显著性加权巩固作诊断探针,LoCoMo 全量 10 会话发现**gist 压缩帮/害取决于任务类型**(交互效应),即单一压缩策略无普适性。
- **H2-EMV** (2026-04, arXiv:2604.11306, https://arxiv.org/abs/2604.11306):机器人终身情景记忆,**通过用户交互学"什么该记"**——LM 按学到的自然语言规则估计相关性做选择性遗忘,用户对被遗忘细节的反馈更新规则;记忆 -45% 而 QA 精度保持。写入价值的监督来源=人类反馈,罕见。
- **FSFM** (2026-04, arXiv:2604.20300, https://arxiv.org/abs/2604.20300):遗忘机制分类学(被动衰减/主动删除/安全触发/自适应强化),立场文+框架。
- **NGC: Neural Garbage Collection** (2026-04, arXiv:2604.18002, https://arxiv.org/abs/2604.18002):KV cache 驱逐作为离散动作,**纯 outcome reward 端到端学会边推理边遗忘**(工作记忆层的 learned forgetting)。
- **Forget to Improve** (2026-06, arXiv:2606.25115, https://arxiv.org/abs/2606.25115):单一 net-value-per-byte 评分统治经验记忆生命周期(KEEP/共享/信任),预算即策展人;把价值、危害、字节成本放进同一标尺。
- **MemoryBank/MARS/D-MEM/GAM/RecMem**:见前述,均含 decay 或分级巩固。
- **综述**:Agent Memory Survey (2026-01, arXiv:2602.06052, https://arxiv.org/abs/2602.06052)——"2025 年数百篇记忆论文";短期记忆做经验门控、长期记忆做巩固的框架;可作领域地图引用。

---

## 空白分析:agent"自主知道什么该记"缺什么

### G1. 写入决策与未来效用在结构上脱钩,且领域已自己承认
- MemSIF 命名 DUM:"write-time salience does not reliably predict future query utility"(arXiv:2608.01742)。
- PA-Bench 实证:去掉"预知未来问题"的泄漏后 training-free 重要性打分全军覆没(0-1% vs 97%,arXiv:2609.05767)——**写入时打分在真实前瞻缺失条件下从未被证明有效**。
- RMM 的前瞻反思(arXiv:2503.08026)承认写入应为未来检索服务,但其学习信号只到检索器,未到写入。
- 含义:缺一个 operational 的"记忆价值 = 未来效用的期望"定义与在写入时可计算的估计量。

### G2. 无 ground truth 的记忆价值 → 各家奖励代理互不兼容,且现有基准区分不出好坏
- RL 系(Memory-R1/Mem-α/Fine-Mem/AgeMem)一致用下游 QA 精度当 reward——CMI-Mem 明确批评这"通过采样查询分布 + 固定 reader 给记忆定价"(arXiv:2607.20553),换查询分布或 reader 即失效。
- Nous 的消融:LoCoMo 上贝叶斯更新 vs 朴素 last-write-wins 几乎无差,**因为基准缺少矛盾/可靠性差异化证据**(arXiv:2606.22030)——评测无法暴露写入策略差异。
- TRUSTMEM 改用转移属性(coverage/preservation/faithfulness)当价值定义(arXiv:2606.25161),与 QA-reward 阵营定义不同构,无人对比两种定义谁对。

### G3. RL 写入策略的奖励稀疏与 credit assignment 有结构性障碍
- Fine-Mem 点名"final task performance as primary reward → severe reward sparsity and ineffective credit assignment"(arXiv:2601.08435)。
- Memory-R2 证明记忆写入使 rollout 环境分叉,GRPO 的同环境比较假设被破坏,轨迹级奖励对记忆操作有偏(arXiv:2605.21768)。
- AgeMem 需三阶段渐进 RL + 专用 step-wise GRPO 才能收敛(arXiv:2601.01885)。
- 含义:每条写入操作的边际价值度量(Fine-Mem 的证据回溯、CMI-Mem 的互信息)刚出现,尚无理论保证,也未在跨任务设置验证。

### G4. 启发式打分基本未经针对性消融,彼此不可比
- Generative Agents 的 1-10 importance 无校准实验;MemoryBank 遗忘曲线参数的消融摘要未说明;D-MEM/HiMem/BudgetMem/SWC 各用一套生物启发或特征启发打分,无公共基准逐维对比。MemFactory 直言领域"highly fragmented and task-specific"(arXiv:2603.29493);ICML 论文批评静态启发式"passively archived without adaptive memory valuation"(arXiv:2609.17088)。

### G5. 训练出的写入策略是任务/查询分布特定的,自主性名不副实
- Memory-R1 训练数据 152 QA 对、对话域;MemAgent 面向长文 QA;Mem-α 的泛化证据是长度泛化(30k→400k)而非任务分布泛化;CMI-Mem 以"generalizable"为卖点恰说明现状不泛化。真正 open-ended 场景(agent 自己产生目标)下的写入策略无人验证。

### G6. 写入自主性 = 新攻击面,现有价值判断无可信度/来源维度
- OEP(arXiv:2605.18930)与 EvoBreak(arXiv:2608.01759)证明反思式写入可被"干净经验"毒化;记忆来源洗白(arXiv:2607.29167)证明 LLM 巩固会抹掉低信任来源标记。"值得记"打分中没有任何一家纳入 provenance/可靠性维度(Nous 的 reliability-conditioned updating 是孤例雏形)。

### 方法分类格局速览
| 类 | 代表 | "值得记"的定义 | 可训练 | 信号 |
|---|---|---|---|---|
| 启发式打分 | Generative Agents, MemoryBank | LLM 1-10 / 时间衰减 | 否 | 无 |
| Prompt 判官 | Mem0, A-Mem, MemGPT, RMM | LLM 先验的"salient" | 否 | 无 |
| RL 操作策略 | Memory-R1, Mem-α, MEM1, AgeMem, Fine-Mem, CMI-Mem | 下游 QA 精度(+CMI/证据回溯) | 是 | outcome reward |
| Surprise 驱动 | EM-LLM, Titans, D-MEM, Nous | 预测误差/梯度/RPE | 部分 | 自监督 |
| 经验蒸馏 | Reflexion, ExpeL, AWM, Voyager | 成败信号门控 | 否 | 环境成败 |
| 遗忘/预算 | H2-EMV, NGC, Retain-or-Consolidate | 预算下净值 | 部分 | 人类反馈/outcome |
