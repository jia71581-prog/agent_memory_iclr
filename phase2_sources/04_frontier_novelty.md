# 前沿撞车检查与 novelty 地图:LLM Agent 自主决定"什么值得写入长期记忆" (2025-2026)

- 调研日期:2026-09-16。通道:arXiv API(export.arxiv.org,含 id_list 批量与 submittedDate 区间检索)+ OpenReview API2(notes/search,含 group 过滤)+ OpenAlex + arXiv e-print/PDF 全文(pdftotext)。
- 纪律:所有 arXiv ID 与摘要均来自 API 返回;venue 以 OpenReview 记录为准,查不到记录的标"仅 arXiv"或按 arXiv comment 自述标注;OpenAlex 引用数因记录分裂普遍低估,本轮判定不依赖引用数。
- 本文件是 01(系统综述)/02(选择性写入)的续篇,聚焦:撞车深挖 + 顶会格局 + 2026-06~09 最新扫描 + survey open problems + novelty 地图。

---

## 1. 撞车名单深挖(逐篇:做了什么 / 训练信号 / 与"自主判断记忆价值"的差异 / 发表状态)

### 1.1 第一梯队:RL 训练写入策略(与选题同赛道)

**Memory-R1** (arXiv:2508.19828, https://arxiv.org/abs/2508.19828)
- 做什么:Memory Manager 学 ADD/UPDATE/DELETE/NOOP 结构化操作 + Answer Agent 预筛推理,双 agent PPO/GRPO。
- 训练信号:下游 QA outcome reward,仅 152 训练 QA 对;经 Answer Agent 间接传导。
- 差异:学的是"给定候选事实选哪个操作",不学"这条信息的未来价值是多少";价值定义=采样查询分布上的答对率,无显式效用估计量、无校准。
- 发表状态:**ACL (1) 2026 已发表**(OpenReview dblp 记录 https://openreview.net/forum?id=inJIfKNFuE)。

**MEM1** (arXiv:2506.15841, https://arxiv.org/abs/2506.15841)
- 做什么:端到端 RL 学恒定大小内部推理状态更新,每轮整合旧状态+新观察、策略性丢弃冗余;记忆=推理状态本身。
- 训练信号:多轮任务成绩(检索 QA/web QA/webshop),合成组合环境。
- 差异:面向任务内工作记忆而非持久长期记忆库;"值得记"完全内隐,无显式条目、无可解释价值。
- 发表状态:**ICLR 2026 Poster**(https://openreview.net/forum?id=XY8AaxDSLb;另有 MTI-LLM @ NeurIPS 2025 Workshop Oral 版本)。注意:前序线索称"ICLR'26"属实,但为 Poster 非 Oral。

**MemAgent** (arXiv:2507.02259, https://arxiv.org/abs/2507.02259)
- 做什么:分段读长文,固定长度文本记忆 overwrite 策略,multi-conv DAPO 端到端训练;8K 训练外推 3.5M token。
- 训练信号:长文 QA 结果奖励。
- 差异:长文档工作记忆(单任务内"保留什么"),不是跨会话长期记忆,不涉及写入价值的显式建模。
- 发表状态:**ICLR 2026 Oral 双重确认**(arXiv comment + journal-ref "ICLR 2026" + OpenReview https://openreview.net/forum?id=k5nIOvYGCL)。是本方向 venue 最强锚点。

**Mem-α** (arXiv:2509.25911, https://arxiv.org/abs/2509.25911)
- 做什么:RL 训练 agent 操作 core/episodic/semantic 三层记忆+多工具;动机原文"language models may lack the ability to determine which information to store";专用训练集;30k 训练泛化 400k+。
- 训练信号:对完整交互历史的下游 QA 精度。
- 差异:最接近同题("学习记忆构建"),但价值仍=合成查询分布上的 QA 期望;无独立的记忆价值定义,泛化证据是长度而非查询分布。
- 发表状态:**ICLR 2026 被拒**(venueid ICLR.cc/2026/Conference/Rejected_Submission, https://openreview.net/forum?id=dm42omwep1);当前仅 arXiv。其验证集已成为 Fine-Mem/TRUSTMEM 的公共基准——生态影响力大于 venue。

**MemAct / Memory-as-Action** (arXiv:2510.12635, https://arxiv.org/abs/2510.12635)
- 做什么:工作记忆管理(上下文 in-place 删改)作为策略动作与任务动作联合 RL;DCPO 解决动态上下文破坏训练的问题。
- 训练信号:端到端任务奖励。
- 差异:上下文编辑层,不解决持久记忆写入价值。
- 发表状态:**ACL ARR 2026 年 1 月轮在审**(https://openreview.net/forum?id=ddGsiaISXg);v3 更新至 2026-05;接收未核实。

**AgeMem** (arXiv:2601.01885, https://arxiv.org/abs/2601.01885)
- 做什么:LTM+STM 统一进策略,记忆操作=工具动作;三阶段渐进 RL + step-wise GRPO 应对记忆操作稀疏奖励。
- 训练信号:任务结果 + 步级 GRPO 整形。
- 差异:操作执行者,不定义价值;其对标准 GRPO 失效的承认是我们引用弹药。
- 发表状态:arXiv comment 自述 **"ACL'26 SAC Highlight"**(OpenReview 未直接核到记录,按自述标注)。

**Fine-Mem** (arXiv:2601.08435, https://arxiv.org/abs/2601.08435)
- 做什么:Chunk-level Step Reward(chunk 专属辅助 QA 即时奖励)+ Evidence-Anchored Reward Attribution(按被引用证据回溯分配全局奖励)。
- 训练信号:细粒度对齐后的 QA 奖励。
- 差异:给出"单条记忆价值=作为证据被引用并答对的贡献"的工程解,但价值仍事后归因、依赖采样查询;无写入时的前瞻估计。
- 发表状态:**ACL (1) 2026 已发表**(https://openreview.net/forum?id=3RyP3XTX1I)。

**Memory-R2** (arXiv:2605.21768, https://arxiv.org/abs/2605.21768)
- 做什么:指出记忆写入使 rollout 环境分叉、GRPO 同环境假设被破坏;LoGo-GRPO 局部 rerollout(同一中间记忆状态比较不同操作)+ 全局轨迹奖励;共享参数 co-learning + 渐进课程 8→16→32 会话。
- 训练信号:局部+全局组相对奖励。
- 差异:修优化器,不修价值定义;其"局部 rerollout=同状态下操作的反事实比较"已隐式逼近操作级价值估计,是 (a) 的近邻。
- 发表状态:OpenReview 搜索只见匿名评审残迹,决定不可读;**仅 arXiv/未核实**。

**MemBuilder** (arXiv:2601.05488, https://arxiv.org/abs/2601.05488):会话级合成问题稠密奖励+组件贡献加权梯度;4B 超闭源基线。发表状态:OpenReview 无记录,**仅 arXiv**。
**DeltaMem** (arXiv:2604.01560, https://arxiv.org/abs/2604.01560):合成对话+操作级更新标签,Memory-based Levenshtein Distance 奖励——价值=与标注记忆状态的编辑距离,是"监督式"价值定义(依赖合成标签而非效用)。comment "preprint, under review";**仅 arXiv**。注意撞名:OpenReview 另有两篇无关 DeltaMem(ARR 2026-03/05,分别为 guarded delta patches 与 residual trees 经验记忆)。

### 1.2 第二梯队:价值定义/奖励设计的直接竞争者

**CMI-Mem** (arXiv:2607.20553, https://arxiv.org/abs/2607.20553)
- 做什么:混合奖励=extrinsic QA + intrinsic 条件互信息(新输入相对当前记忆状态的信息贡献,不 conditioning 采样查询);明确批评 QA reward "values memory through a sampled query distribution and a fixed reader"。开源 4B checkpoint。
- 差异:CMI 是"信息量"代理而非"效用"——它假设信息量≈价值,不建模未来任务分布;这是与 (a) 的关键区别面。占据 (b) 的"内在奖励"一格。
- 发表状态:OpenReview 无记录,**仅 arXiv**。

**AdaMem** (arXiv:2606.21144, https://arxiv.org/abs/2606.21144)
- 做什么:preference-conditioned write control:自然语言 Memory Policy 按交互情境定义"什么值得写",周期反馈更新;AdaMem-Bench;Mem0 80.0→84.35%。核心发现:**模型难以把明确的策略转成可靠的选择性写入行为(policy execution 是瓶颈)**。
- 差异:价值=用户偏好的自然语言描述,非学习到的效用估计;其失败发现是我们动机弹药。
- 发表状态:OpenReview 无本篇记录,**仅 arXiv**。撞名警告:**AdaMEM (arXiv:2606.05684) 是另一篇、已中 ICML 2026**(test-time adaptive memory,检索/策略合成侧,https://openreview.net/forum?id=RzlmkviaNy);另见 CoRR 2026 记录 "AdaMem: Adaptive User-Centric Memory"(第三个同名)。写作时必须显式消歧。

**MemSIF** (arXiv:2608.01742, https://arxiv.org/abs/2608.01742):命名 DUM(写入时显著性≠未来查询效用)与 TSM;结构化交互→双轨事实记忆缓解;不可训练。comment "Submitted to AAAI 2027";**仅 arXiv**。DUM 命名是 (a) 动机的直接引文。
**PA-Bench 论文** (arXiv:2609.05767, https://arxiv.org/abs/2609.05767):去掉"未来问题泄漏"后 training-free 打分全军覆没(过敏事实存活 0-1% vs 全记忆 97%);结论:压缩缓存非持久层。**已接收 ACM HumanSys 2026**(arXiv comment)。
**Nous** (arXiv:2606.22030, https://arxiv.org/abs/2606.22030):贝叶斯信念记忆消融显示 LoCoMo 上无增益(基准缺矛盾证据);reliability-conditioned updating + provenance-capped 投毒防御;并量化 token-F1 与 LLM-judge 27.5 分分歧。独立作者,**仅 arXiv**。
**TRUSTMEM** (arXiv:2606.25161, https://arxiv.org/abs/2606.25161):Memory Transition Verifier(coverage/preservation/faithfulness)+ 同状态候选更新偏好对 RL;HaluMem 抽取 +12.14 F1,转移级 omission/corruption/hallucination -40/-79/-50%。**仅 arXiv**。验证的是"更新转移对源文本的忠实性",非对世界的真实性。
**ICML: Interactive Memory Learning** (arXiv:2609.17088, https://arxiv.org/abs/2609.17088):Planner 选择性编码高价值信息 + Trigger 检索,延迟奖励把未来反馈传回早期存储决策,在线 RL 共演化。2026-09-15 刚挂出,**仅 arXiv**。注意其延迟奖励机制与 (a) 有一步之遥,但价值仍=对话响应质量。

### 1.3 本轮新发现的强撞车(前两轮未覆盖)

**A-MAC: Adaptive Memory Admission Control** (arXiv:2603.04549, https://arxiv.org/abs/2603.04549)
- 把记忆准入建为结构化决策:**把记忆价值分解为五个可解释因子——future utility、factual confidence、semantic novelty、temporal recency、content type prior**;规则特征+单次 LLM 效用评估+交叉验证学准入策略;LoCoMo F1 0.583。
- 威胁:显式使用了 "future utility" 与 "factual confidence" 词汇并组成准入决策——(a) 与 (c) 的词面已被占。但其 future utility 是 LLM 一次性打分,非学习到的期望估计量,无校准、无与未来查询的闭环。
- 发表状态:**ICLR 2026 Workshop MemAgents**(https://openreview.net/forum?id=mmdqUrEY24);主会记录未见。

**Towards a Formal Definition of Agent Memory** (Hongyao Tang, arXiv:2608.11654, https://arxiv.org/abs/2608.11654)
- 记忆=basis、知识=span、可答性=覆盖问题;**最优记忆=容量约束下期望覆盖最大化**;含噪时写入策略须推断所存内容的真伪(coverage vs precision);把持续记忆构建形式化为 **sequential MDP:记忆=状态、写入=动作、查询时结算的效用=延迟奖励**;在 Odyssey 上实例化;"把构建与学习 agent memory 的 open problems 重写为具体研究问题"。
- 威胁:候选 (a) 的形式化框架已被此文占据(2026-08-12)。它是理论/立场文,无学习系统、无真实基准;但任何 (a) 类论文必须引用并区分。
- 发表状态:**仅 arXiv**(单一作者)。

**SP-KV: Self-Pruned Key-Value Attention——Learning When to Write by Predicting Future Utility** (arXiv:2605.14037, https://arxiv.org/abs/2605.14037, Meta)
- 轻量 utility predictor 给每个 KV 对打分,预测超阈值才写入长期 KV cache;与 LLM 联合端到端训练,信号=next-token loss;KV 3-10x 压缩。
- 威胁:标题即"learning when to write by predicting future utility"——(a) 的标题空间被占。但层级是 KV cache/架构内,不是显式 agent 记忆条目;效用=下一 token 预测,非任务效用。
- 发表状态:**仅 arXiv**。

**TTCD: Learning What to Remember——Test-Time Context Distillation** (arXiv:2608.01672, https://arxiv.org/abs/2608.01672)
- 长窗口 teacher 监督短窗口 student 的 fast weights,hidden-state 差异=稠密自监督信号,引导记住"对未来 token 预测关键"的信息;批评现有 TTT "不考虑保留信息的 future utility"。
- 威胁:又一个占据 "learning what to remember"+"future utility" 词面的架构级工作。**仅 arXiv**。

**MemReader** (arXiv:2604.07877, https://arxiv.org/abs/2604.07877, MemOS 团队)
- GRPO 训练的主动抽取器(4B):ReAct 式显式评估 information value/reference ambiguity/completeness 后决定写/缓/检索/弃;LOCOMO/LongMemEval/HaluMem SOTA;已并入 MemOS 生产部署。
- 威胁:「用 RL 训练一个在写入前显式评估信息价值的 agent」已被做出来并部署。差异:价值评估仍是模型内隐推理,无显式效用目标或校准;面向对话抽取。**仅 arXiv**。

**Grounding Agent Memory: Environment-Probing Curation** (arXiv:2609.11060, https://arxiv.org/abs/2609.11060, 2026-09-10)
- 给异步 curator agent 最小权限只读环境工具,在写入前 **check/scope/refresh 候选记忆**(对环境验证);GitHub Copilot 生产化 harness;CLBench pass 39%→73%。
- 威胁:候选 (c) 的核心动作——"写入前对环境验证"——已被此文在企业 agent 场景做出(6 天前)。差异:training-free、程序性/企业知识、curator 工程而非学习到的验证门。

**MemGuard** (arXiv:2608.21867, https://arxiv.org/abs/2608.21867)
- 命名两大失败模式:unreliable admission(失败轨迹/偶然成功混入)与 memory drift;**把 verifier 输出持久化为记忆条目的生命周期元数据**(reward/confidence/label/uncertainty),在准入、检索、冲突消解、归档全程复用;Terminal-Bench/SWE-Bench/WebArena/Mind2Web 16 设置全胜(超 ReasoningBank)。
- 威胁:(c) 的"验证信号进入写入决策"在经验记忆上已被占。差异:verifier 是打分器非环境执行验证;无训练。**仅 arXiv**。

**GovMem: When Not to Write Memory** (arXiv:2607.02579, https://arxiv.org/abs/2607.02579)
- 把"何时拒绝写入"建为 writepath governance:依赖感知的支持度(相同断言可能同源复制)、反证检索、作用域,输出 promote/reject/needs-review;false promotion 0.597→0.040;自认保守、审查负担高。已接收 **mlise 2026**(arXiv comment)。

**LUCID: Memory Reward Inflation** (arXiv:2608.00017, https://arxiv.org/abs/2608.00017)
- 把存储分数看作隐式非参数策略的 proxy reward,证明 **Echo Gap**:LLM 自评的记忆价值系统性膨胀且错误与自评偏差相关、经记忆复利放大;证明 **Error-Independence Assumption 是纠正膨胀的必要条件**(可用信号必须既跟踪真值又与记忆偏差去相关);LUCID 去膨胀算法在 BIRD +2.9。
- 意义:对"LLM 判官定价记忆"的根本性打击,同时给 (a)(校准)与 (c)(需要误差独立的验证信号)提供理论依据。**仅 arXiv**。

**Manufactured Confidence** (arXiv:2606.29279, https://arxiv.org/abs/2606.29279):巩固把犹豫传闻改写成确信事实并被当真执行;agent 只响应措辞置信度而非来源;被动 "unverified" 标签无效。**仅 arXiv**。(c) 的失败模式证据。
**AuthMem-Bench** (arXiv:2608.01679, https://arxiv.org/abs/2608.01679):authority collapse——巩固保留断言却抹掉来源授权约束;49 配置中 48 个塌缩;自动 authority 标签把未授权行动 16.9%→0.0%。**仅 arXiv**。
**WritePolicyBench** (arXiv:2602.02574, https://arxiv.org/abs/2602.02574):字节预算+漂移流下写入策略基准(动作接口+成本模型+双指标)。单作者、10 页;**仅 arXiv**。基准命名空间已被占。
**PREPING** (arXiv:2605.13880, https://arxiv.org/abs/2605.13880, KAIST):任务前记忆构建(合成练习+Validator 决定何者可入库);证明无控制的合成轨迹会让记忆退化。**仅 arXiv**(comment: Preprint)。
**BEHEMOTH/CluE** (arXiv:2604.11610, https://arxiv.org/abs/2604.11610, USC):形式化异质记忆抽取,18 数据集+下游效用指标;**没有单一静态抽取策略能跨任务类别占优**,现有自演化 prompt 优化在异质分布下退化。**仅 arXiv**。(b) 的直接实证。
**MemSkill** (arXiv:2602.02474, https://arxiv.org/abs/2602.02474):把记忆操作重构为可学习/可演化的 skills(controller 选择+designer 复盘演化)。**仅 arXiv**。
**SelfMem** (arXiv:2607.03726, https://arxiv.org/abs/2607.03726):给 agent 记忆工具+反馈信号让其自我优化记忆策略;BEAM 100K-1M 全面领先。**仅 arXiv**。
**EPIC** (arXiv:2605.18271, https://arxiv.org/abs/2605.18271):**ICML 2026 已接收**(arXiv comment);端上预算下"存什么"=偏好相关信息,索引内存 -2404x。
**Gated-Memory Routing** (arXiv:2609.00237, https://arxiv.org/abs/2609.00237):**EMNLP 2026 main 已接收**(arXiv comment);学习到的 Memory Write Gate 只提交非冗余推理步(多智能体编排状态,非持久记忆)。
**MERIT** (arXiv:2609.05441, https://arxiv.org/abs/2609.05441):成本核算下记忆的边际效用评测;换记忆实现可移任务成功率至多 60 点;更新事实上嵌入检索不可预测地崩溃。**仅 arXiv**。
**Ground Truth First / Veracium** (arXiv:2607.21962, https://arxiv.org/abs/2607.21962):先生成带真值区间/来源渠道的事实脚本再渲染对话;9 周量程下架构排名反转;**写入阶段质量强预测下游(弱写入事实失败率 24% vs 2%)**。**仅 arXiv**。
**What Eviction Destroys** (arXiv:2609.08279, https://arxiv.org/abs/2609.08279):restore-counterfactual 审计;8k 预算下四种驱逐策略不可逆错误占比 1.00;匹配精度下 LLM-importance 与随机驱逐无可检差异。**仅 arXiv**。
**RoMeRL** (arXiv:2608.02508, https://arxiv.org/abs/2608.02508):降阶效用状态解决轨迹索引效用的反馈稀释与 memory-reward trap;理论+ALFWorld/LifelongAgentBench。**仅 arXiv**。
**Rate-Distortion 视角** (arXiv:2607.08032, https://arxiv.org/abs/2607.08032):统一 KV/prompt/架构态/agent 记忆为"预算下保留什么信息"的率失真问题;跨层通病=用注意力幅度或近期性当保留信号,"在查询已知前不可逆地丢弃后来需要的信息";指出重复压缩无基准。**仅 arXiv**。理论近邻+弹药。
**Neural Paging** (arXiv:2603.02228, https://arxiv.org/abs/2603.02228):Context Paging Problem,可微 Page Controller 逼近"Semantic Belady 最优"(保留高未来效用 token),复杂度与鲁棒界。合成验证为主。**仅 arXiv**。
**Σ-Mem** (arXiv:2607.27958, https://arxiv.org/abs/2607.27958):在线可靠性记忆(记"哪个 peer 在什么条件下可信"而非内容),谱有界更新。**仅 arXiv**。
**当稳约束未被复核** (arXiv:2608.25553, https://arxiv.org/abs/2608.25553):预算内验证分配失败——16 个模型对"读起来已定论"的约束仅 1/5 会复核来源,77% 做出过时决策。**仅 arXiv**。(c) 的读侧对偶证据。
**MemLens** (arXiv:2607.25992, https://arxiv.org/abs/2607.25992):Shapley 式记忆估值+价值感知存储的交互分析系统(demo 类)。**仅 arXiv**。
**Weighted Memory Tree** (arXiv:2608.20631, https://arxiv.org/abs/2608.20631):层级记忆+动态保留分(事件更新+选择衰减);GAIA-Text +9.97pp。**仅 arXiv**。
**SP-Mem** (arXiv:2608.16551, https://arxiv.org/abs/2608.16551):"What to Remember, What to Reveal"——隐私感知写入(敏感值分离存储+按需披露)。**仅 arXiv**。占据了该标题句式的隐私变体。

---

## 2. 顶会格局(OpenReview 核实)

### ICLR 2026(记忆相关 accepted,search+group 过滤,非穷尽)
| 论文 | 级别 | forum |
|---|---|---|
| MemAgent(RL 覆写记忆) | **Oral** | k5nIOvYGCL |
| MEM1(恒定状态 RL) | Poster | XY8AaxDSLb |
| LightMem | Poster | dyJ0GWpjJB |
| MemGen(生成式潜在记忆,trigger+weaver,arXiv:2509.24704) | Poster | vI56m4Iu4e |
| MemGAS(多粒度记忆关联/选择,arXiv:2505.19549) | Poster | i2yIvZARnG |
| Beyond a Million Tokens(LTM 基准) | Poster | y59hf5lrMn |
| 多模态 agent LTM(Seeing, Listening, Remembering) | Poster | PMz29A7Muq |
| Mem-α | **Rejected** | dm42omwep1 |
| MemoryBench(记忆+持续学习基准) | Rejected | wU4Tjlzg3h |

另有 **ICLR 2026 Workshop MemAgents**(整个 workshop 专做 agent 记忆,ICLR.cc/2026/Workshop/MemAgent):A-MAC、TierMem(From Lossy to Verified: provenance-aware tiered memory + verified write)、ALMA(元学习记忆设计,Oral)、Memory Transplants(架构 vs 内容迁移解耦,code→math shift)、INFMEM(System-2 记忆控制,Oral)、AMA-Bench(Oral)、SimpleMem(Oral)、Memory Injection Attacks 等——领域密度信号:该方向已有专属 workshop,竞争极稠密。

### NeurIPS 2025(记忆相关 accepted,非穷尽)
- G-Memory(多智能体层级记忆)**spotlight**;A-Mem poster;Titans(测试时记忆)poster;Memory Mosaics at scale **oral**;Memory Injection Attacks poster;Agentic Plan Caching poster;3DLLM-Mem poster。

### 其他 venue(本轮核实)
- **ACL 2026 main**:Memory-R1、Fine-Mem(均 OpenReview dblp 记录);AgeMem 自述 ACL'26 SAC Highlight。
- **ICML 2026**:AdaMEM(test-time adaptive memory,2606.05684)、EPIC(preference-aligned on-device 记忆构建)。
- **EMNLP 2026 main**:Gated-Memory Routing(learned write gate,多智能体编排)。
- **TMLR 2026-07**:Agent Memory in the Second Half 综述(2602.06052,PDF 页眉 "Published in Transactions on Machine Learning Research (07/2026)")。
- **NeurIPS 2026**:OpenReview 尚无公开 accepted 记录(截至 2026-09-16 在审中)——Memory-R2、CMI-Mem、Dual-Layer、MemAct 等 2026 春夏 preprint 很可能在此轮或 ICLR 2027 出现,**ICLR 2027 投稿季与我们同期,撞车风险在投稿池内而非已发表文献中**。

### 结论:venue 声明核实表
| 论文 | 声明 | 核实结果 |
|---|---|---|
| MemAgent | ICLR 2026 Oral | ✅ 确认(OpenReview+journal-ref) |
| MEM1 | ICLR 2026 | ✅ Poster 确认 |
| Memory-R1 | ACL 2026 | ✅ ACL (1) 2026 |
| Fine-Mem | — | ✅ ACL (1) 2026 |
| Mem-α | (投 ICLR 2026) | ❌ **被拒**,仅 arXiv |
| MemAct | — | ARR 2026-01 在审,接收未核实 |
| AgeMem | ACL'26 SAC Highlight | 自述,未独立核实 |
| CMI-Mem / Memory-R2 / AdaMem / MemBuilder / DeltaMem / MemSIF / Nous / TRUSTMEM / Dual-Layer / ICML(2609.17088) | — | 仅 arXiv |
| PA-Bench | — | ACM HumanSys 2026(自述 comment) |

---

## 3. Survey open problems(引用弹药)

### 3.1 Agent Memory in the Second Half(arXiv:2602.06052,TMLR 07/2026)——六大 open challenges(§9,全文核实)
1. **持续学习/自演化**(§9.1):"existing systems largely rely on heuristic memory controllers, leaving the internal coupling between memory evolution and reasoning behavior poorly understood";跨任务层面"current approaches primarily emphasize inference-time reuse rather than principled consolidation and generalization"。
2. 多人-多智能体记忆组织(§9.2):协作记忆、治理(所有权/责任/人类纠正如何处理)。
3. **基础设施与效率**(§9.3):终局愿景=记忆成为"core, learned subsystem",需要"adaptive memory controllers that dynamically allocate, compress, and retire memory units based on task relevance, **uncertainty estimates, and long-term utility**"——(a) 的直接背书。
4. **终身个性化+可信记忆**(§9.4):点名需要"**learned memory controllers that regulate when to write, compress, or overwrite** user information";staleness/概念漂移/credit assignment;投毒与可审计性为一等公民。
5. **世界模型记忆**(§9.5):"**proactive memory planning, where agents simulate the long-term consequences of storing, compressing, or forgetting information before committing updates**;memory operations become internal actions optimized jointly with external decision-making"——(a) 的最强 open-problem 引文。
6. **真实世界评测**(§9.6):现有基准"implicitly assume stationary user intent";失败模式(stale preference reuse、错误覆写、敏感信息不安全保留)不被度量;呼吁 provenance tracking、contradiction resolution、资源-效用权衡入指标。
- 另:§3.2.3 "Open problems include how agents should define episode boundaries, regulate the influence of episodic recall, and manage long-term retention";§引言处 "memory management itself is becoming a trainable, self-evolving capability"。

### 3.2 From Human Memory to AI Memory(arXiv:2504.15965)——§5 open problems(全文核实)
- 单模态→多模态;静态→流式记忆;特定→综合记忆;独占→共享记忆;个体→集体隐私;**规则演化→自动演化**("this evolutionary process often depends on manually crafted rules and heuristic adjustments … inherently limits flexibility, scalability, and efficiency")——学习式写入策略的动机引文。

### 3.3 From Storage to Experience(arXiv:2605.06716)——Future Directions(LaTeX 源核实)
- **Active Memory Perception**(记忆机制应自主评估任务是否需要引入记忆、引入哪类——按需资源);Working Memory 组织(自适应剪枝);**Benchmark for Experience**(经验阶段抽象/泛化能力的评测严重不足);分布式共享记忆;多模态记忆。

### 3.4 领域自我批评(非 survey 但同功能)
- Rate-Distortion 视角(2607.08032):跨层通病=保留信号是注意力幅度/近期性,"在查询已知前不可逆丢弃";重复压缩无人测。
- LUCID(2608.00017):自评价值膨胀 + Error-Independence 必要性定理。
- Tang(2608.11654):把"构建与学习记忆"的 open problems 重写为具体研究问题(MDP 形式化)。

---

## 4. Novelty 地图

### 4.1 已占据的点(占据者按 venue 强度排序)
| 点 | 占据者 | 状态 |
|---|---|---|
| 写入操作可训练化(RL 操作策略) | MemAgent(ICLR'26 Oral)、MEM1(ICLR'26)、Memory-R1(ACL'26)、Mem-α、AgeMem、MemAct、DeltaMem、MemBuilder | **完全占满**,做"RL 学 ADD/DELETE"已无 novelty |
| 奖励稀疏/credit assignment 修补 | Fine-Mem(ACL'26)、Memory-R2(LoGo-GRPO)、AgeMem(step-wise GRPO)、MemBuilder、RoMeRL | 占满;只剩理论保证 |
| 查询无关的内在价值代理 | CMI-Mem(CMI)、TTCD(自监督 future utility)、SP-KV(next-token 训练的 utility predictor)、Neural Paging(Semantic Belady) | 主要格已占;显式记忆条目级+任务效用级尚缺 |
| 可解释多因子准入打分 | A-MAC(五因子含 future utility,MemAgents WS)、Selective Memory 写门(2603.15994)、Weighted Memory Tree、D-MEM、BudgetMem | 占据(非学习层面) |
| 学习式主动抽取/策略演化 | MemReader(GRPO,生产部署)、MemSkill、CluE、AdaMem、SelfMem、ALMA(WS Oral)、ICML(2609.17088) | 占据 |
| 记忆价值的形式化理论 | Tang 2608.11654(coverage-MDP)、Rate-Distortion 2607.08032、WritePolicyBench | 框架层已有占位,但均无学习系统 |
| 经验/程序记忆的验证准入 | MemGuard(verifier 元数据全生命周期)、GovMem、Grounding Agent Memory(环境探测 curation)、PREPING(Validator 门)、TierMem(WS)、Voyager(源头) | **大部分占据** |
| 来源/权限保真 | AuthMem-Bench、Manufactured Confidence、Nous(provenance-capped)、Governed Persistent Memory 系 | 问题已充分命名,防御=元数据方案 |
| 写入策略评测批判 | PA-Bench(泄漏)、MERIT(边际效用/成本)、What Eviction Destroys(不可逆性)、Ground Truth First(纵向)、Nous(判分分歧)、BEHEMOTH(异质性) | 密集占据;单做基准已难 |

### 4.2 空白点(附证据)
**G-A. 显式记忆条目级的"期望未来效用"可学习估计量(经校准)**——收窄后仍空
- 谁说是 open problem:TMLR survey §9.5 "simulate the long-term consequences of storing … before committing"、§9.3 "retire memory units based on … uncertainty estimates and long-term utility"、§9.4 "learned memory controllers that regulate when to write";MemSIF 的 DUM 命名;Rate-Distortion 文"在查询已知前丢弃"通病。
- 为什么现有方法够不到:Tang 只给形式化无学习系统;SP-KV/TTCD 的效用=next-token、层级在 KV/fast-weights,非显式条目、非任务效用;RL-QA 系(Mem-α 等)的效用=采样查询分布的事后期望,无写入时前瞻估计;A-MAC 的 future utility=LLM 单次打分,无校准;LUCID 证明 LLM 自评价值系统性膨胀且需误差独立信号——**"写入时可计算、有校准保证、以任务效用为语义的条目级价值估计量+以此为目标的训练"仍无人交付**。
- 风险:必须引用并区分 Tang/SP-KV/TTCD/A-MAC/Fine-Mem(证据回溯是事后版)/Memory-R2(局部 rerollout 是反事实近似);词面 "predicting future utility"、"learning what to remember" 均已被占,需换表述强调"explicit memory + calibrated + task-utility"。

**G-B. 写入决策与查询分布解耦的泛化**——半空白
- 证据:CMI-Mem 亲口批评 QA-reward "values memory through a sampled query distribution and a fixed reader"(其 CMI 是对策但只到信息量层);BEHEMOTH 实证无单一静态抽取策略跨任务占优、自演化框架在异质分布下退化;PA-Bench 证明含未来问题泄漏的评测虚高;Nous 证明基准缺矛盾证据时写入策略差异不可见;MemAgents WS 的 Memory Transplants 证明跨域迁移中架构与内容贡献被混淆。
- 现有覆盖:CMI-Mem 声称跨场景迁移(已占"内在奖励→迁移"一格);Mem-α 的泛化证据只有长度。
- 剩余空白:**以"训练查询分布 A / 评测不相交分布 B"为核心协议的写入策略泛化研究、分布鲁棒(worst-case 查询分布)写入目标、以及效用估计量在分布移位下的校准**。单独成文偏薄,作为 G-A 的评测协议与理论包装最强。

**G-C. 事实型(declarative)记忆的验证门控**——大幅收窄,残余仍在
- 已被占的部分:环境可验证的经验/程序记忆准入(MemGuard、GovMem、Grounding Agent Memory、PREPING、TierMem、Voyager 源头);转移一致性验证(TRUSTMEM,验证对源对话的忠实,非对世界的真);来源治理(AuthMem-Bench、provenance 系)。
- 仍空的证据:Manufactured Confidence 证明被动 "unverified" 标签无效、主动不信任指令过度回避——即**元数据方案已被证不足,需要写入前的主动验证**;Grounding Agent Memory 是 training-free curator 且限企业程序知识;LUCID 定理要求验证信号与写入者偏差误差独立——现有验证器(LLM 判官)不满足;Nous 的 reliability 估计自认可被操纵。
- 残余空白:**可训练的事实验证门(用环境交互/工具调用产生误差独立的验证信号),输出校准置信并作为写入决策与后续信任的一部分,在含矛盾/不可靠证据的基准上验证**。注意与 2609.11060(6 天前)正面撞车风险最高,写作须以"learned + calibrated + declarative facts + error-independent signal"四点划界。

**G-D.(顺带发现的额外空白)写入价值的因果/反事实度量作为训练信号**
- What Eviction Destroys 的 restore-counterfactual、Memory-R2 的局部 rerollout、Fine-Mem 的证据回溯、MERIT 的边际效用,都是**评测或优化技巧层面的反事实**;没有工作把"条目的反事实边际效用"直接做成写入策略的训练目标并给出方差/偏差分析。这是 G-A 的一个具体可执行化路径。

### 4.3 三个候选切入点的存活判定
- **(a) 记忆价值=未来效用期望的可操作定义与训练目标:存活,但已被三面包围**(理论:Tang;架构:SP-KV/TTCD;启发式:A-MAC)。可辩护的组合=显式记忆条目 + 任务效用语义 + 写入时可计算 + 校准(引 LUCID)+ 分布移位评测(并入 b)。必须在相关工作里正面处理 5+ 篇近邻,novelty 论证要靠"组合+校准保证",不能靠概念首创。
- **(b) 写入与查询分布解耦:降级为 (a) 的评测协议/理论组件**。CMI-Mem 已占内在奖励格;独立成文证据链(BEHEMOTH/PA-Bench/Nous)反而说明这是评测问题,做协议比做方法站得住。
- **(c) 事实记忆的验证门控:窗口正在关闭**(Grounding Agent Memory 2026-09-10、MemGuard、GovMem、TRUSTMEM 半年内密集出现)。残余空白=learned + calibrated + 误差独立验证信号 + declarative facts;若采用,必须以 LUCID 的 EIA 定理为理论骨架、以 Manufactured Confidence/AuthMem-Bench 为失败模式动机,并抢在 NeurIPS 2026 放榜(这批 preprint 可能升级 venue)前挂 arXiv。

### 4.4 战术备注
- 同名地雷:AdaMem×3(2606.21144 / ICML'26 的 2606.05684 / CoRR 2026 user-centric)、DeltaMem×3、"Learning What to Remember"(TTCD 2608.01672 与 AdaMem 副标题)、"What to Remember"(SP-Mem 隐私版)。标题避开这些词面。
- ICLR 2027 与我们同期投稿的隐形对手:Memory-R2、CMI-Mem、Dual-Layer、MemAct、ICML(2609.17088)等大概率在同一池;NeurIPS 2026 放榜在即,投稿前需复查这批的 venue 升级。
- 基准选型:LoCoMo/LongMemEval 已被多篇证明区分度不足(Nous、2602.19320、2608.12888);建议主战场用 Mem-α 验证集 + MemoryAgentBench(Fine-Mem/TRUSTMEM 先例)+ 至少一个含矛盾/泄漏控制的基准(自建或 PA-Bench/Ground Truth First 思路),并报告成本核算(MERIT 先例)。
