# 03. LLM Agent 长期记忆 Benchmark 与评测协议盘点

日期: 2026-09-16。检索通道: arXiv API (export.arxiv.org, 含 PDF 全文抽取) + OpenAlex + OpenReview。
纪律: 所有 arXiv ID 均来自 API 返回并经标题/摘要核对; 无法核实的条目显式标注"未核实"。
tavily/WebSearch/一般网页不可用, 因此 GitHub 仓库规模、博客与 leaderboard 层面的信息一律标"未核实"。

---

## 1. 主要 benchmark 对比表

| Benchmark | arXiv / venue | 年份 | 测什么 | 规模与构造 | 协议与指标 | 已知问题 |
|---|---|---|---|---|---|---|
| MSC (Multi-Session Chat) | [2107.07567](https://arxiv.org/abs/2107.07567), ACL 2022 (10.18653/v1/2022.acl-long.356) | 2021 | 多会话开放域对话一致性 | 真人众包, 每对话 5 会话、每会话约 12 条消息, 标注者维持 persona | 回复质量自动指标 + 人评 | 会话极短, 现代 LLM 全上下文即可覆盖; 现只作为 DMR 等衍生任务的底料 |
| DMR (MemGPT) | [2310.08560](https://arxiv.org/abs/2310.08560), arXiv (ICLR 2024 投稿 Submission8071 存在, 录取结果未核实) | 2023 | 跨会话精确回忆(单问) | 基于 MSC, 用另一个 LLM 生成"只能靠前 5 会话回答"的 QA 对; Zep 论文口径: 500 段对话 x 5 会话 x ≤12 消息 | 对 gold answer 判分: LLM-judge 一致性 + ROUGE-L recall | 每段仅约 60 条消息, 全上下文即可解; 单跳事实检索; 问题措辞含糊 (Zep 论文 §4.2 明确批评, 见 §3.3) |
| LoCoMo | [2402.17753](https://arxiv.org/abs/2402.17753), ACL 2024 (10.18653/v1/2024.acl-long.747) | 2024 | 超长多会话对话记忆 (QA + 事件摘要 + 多模态对话生成) | LLM 生成(persona + 时间事件图) + 人工校对; 平均 300 turns / 9K tokens, 最多 35 会话; QA 共 7,512 题 5 类 (单跳 2,705 / 多跳 1,104 / 时间 1,547 / 开放域 285 / 对抗 1,871); 社区常用公开子集为 10 段对话 1,540 题 (第三方论文 2605.04897 口径) | 原文 F1/BLEU; 社区通行 LLM-as-a-Judge (J) | 争议最多: 写入策略区分度不足、judge 与 F1 严重分叉、类别口径不一 (adversarial 常被剔除)、被指饱和; 详见 §3 |
| PerLTQA | [2402.16288](https://arxiv.org/abs/2402.16288), SIGHAN@ACL 2024 (10.18653/v1/2024.sighan-1.18) | 2024 | 个人语义+情景记忆 QA (含记忆分类/检索/合成三子任务) | 合成: 30 个虚构人物, 8,593 题, 覆盖世界知识/档案/社会关系/事件/对话 | 分类准确率、检索指标、QA 打分 | 全合成人物; 引用量低; 近期 memory 系统论文很少用 |
| GoodAI LTM Benchmark | [2409.20222](https://arxiv.org/abs/2409.20222), NeurIPS 2024 Datasets & Benchmarks poster (10.52202/079017-1347) | 2024 | 动态单流交互中的 LTM + 持续学习 + 信息整合 | 程序化动态生成: 单个超长 user-agent 交互, 多任务并发交织、频繁上下文切换 | 任务完成式自动判分 (动态 needle 类) | 任务为脚本合成; 社区采用度低 (OpenAlex 引用 3); GitHub 版本演进未核实 |
| LongMemEval | [2410.10813](https://arxiv.org/abs/2410.10813), ICLR 2025 poster | 2024 | 聊天助手五种记忆能力: 信息抽取、跨会话推理、时间推理、知识更新、拒答(abstention) | 500 道人工精修问题嵌入可自由加长的合成聊天史; S 版约 115k tokens/题, M 版 500 会话约 1.5M tokens | LLM-judge 的 QA 准确率; 拆 indexing/retrieval/reading 三阶段分析 | 目前学界口碑最好; 但仍是端到端 QA, 且 MemDelta (2606.29914) 证明其结论对嵌入模型/底座选择高度敏感 |
| Zep 评测 (DMR + LME) | [2501.13956](https://arxiv.org/abs/2501.13956), arXiv | 2025 | 系统论文, 非新基准; 用 DMR + LongMemEval | 同上 | DMR 94.8% (gpt-4-turbo) / 98.2% (gpt-4o-mini); LME 准确率 +18.5%, 延迟 -90% | 自评; 对 DMR 的批评成为公案素材; 未在 LoCoMo 上自评 |
| Mem0 评测 | [2504.19413](https://arxiv.org/abs/2504.19413), arXiv | 2025 | 系统论文; LoCoMo 全量 4 类题 | LoCoMo (剔除 adversarial 类, 理由: 无 ground truth) | F1/BLEU-1/LLM-as-a-Judge(J), J 跑 10 次取均值; 另报 p50/p95 延迟与 token 成本 (p95 延迟 -91%, token -90% vs 全上下文) | 其对 Zep 的配置被 Zep 方公开质疑 (博客层面, 未核实); 见 §3.4 |
| MemBench | [2506.21605](https://arxiv.org/abs/2506.21605), ACL 2025 Findings (10.18653/v1/2025.findings-acl.989) | 2025 | 事实性记忆 + 反思性记忆; 参与/旁观两种交互情景 | 合成数据集 (论文自建, 规模细节未核实) | 有效性 + 效率 + 容量多维指标 | 与同名 membench (扩散模型/硬件) 重名易混; 社区采用度一般 |
| LifelongAgentBench | [2505.11942](https://arxiv.org/abs/2505.11942), arXiv, ICLR 2026 在投 (OpenReview 可见 submitted) | 2025 | Agent 终身学习/经验累积 (非对话) | Database / OS / Knowledge Graph 三个交互环境, 技能锚定、任务间有依赖; 自动标签验证 | 任务成功率; 发现朴素 experience replay 基本无效, 提出 group self-consistency | 环境偏结构化; 非记忆系统专用 |
| MemoryAgentBench | [2507.05257](https://arxiv.org/abs/2507.05257) (v4), venue 未核实 | 2025 | 四能力: 精确检索、test-time learning、长程理解、选择性遗忘 | 把既有长上下文数据集改造成增量多轮喂入 + 新构数据 | 分能力准确率 | ReFind (2608.12888) 显示在其上"无结构的 agentic BM25 检索"可打平/超过结构化记忆系统, 提示该基准可被检索侧捷径攻破 |
| HaluMem | [2511.03506](https://arxiv.org/abs/2511.03506), arXiv | 2025 | 操作级记忆幻觉: 记忆抽取 / 记忆更新 / 记忆 QA 三任务 | HaluMem-Medium / -Long 两套人机交互数据, 约 15k memory points + 3.5k 题 | 各操作阶段的幻觉率(捏造/错误/冲突/遗漏), 非端到端 | 少数提供"写入内容对错"操作级 GT 的基准之一 |
| PA-Bench | [2609.05767](https://arxiv.org/abs/2609.05767), HumanSys'26 (10.1145/3842436.3843817) | 2026 | KV-cache 压缩/驱逐下个人助手安全关键事实的存续 | 自建 100 段 span 标注的助手对话, 顺带提及的安全关键事实(过敏/用药) | query-blind 协议: 压缩时看不到未来问题; 20% budget 下全部免训练驱逐策略把事实块保留率打到 0-1% (全量记忆 97%) | 规模小(100 段); 定位是审计式研究, 不是排行榜; 结论: 压缩 cache 是推理复用机制而非持久层 |

注: PA-Bench 任务书中给的 ID 2609.05767 经 API 核实确为其来源论文 (标题是 "Will My Assistant Remember My Allergy?...", PA-Bench 是文中构造的数据集名)。

---

## 2. 2025-2026 新基准 (arXiv API 检索 "memory+benchmark", "LoCoMo", "LongMemEval" 引用面)

按用途分组, 全部 ID 来自 API 返回:

**Agent 任务耦合型 (记忆要改变行动, 不只是答题)**
- MemoryArena [2602.16313](https://arxiv.org/abs/2602.16313): 多会话相互依赖的 agentic 任务 gym (web 导航/偏好约束规划/递进信息搜索/形式推理); 明确指出在 LoCoMo 上近饱和的 agent 在其上表现差。
- MERIT [2609.05441](https://arxiv.org/abs/2609.05441): 工具型 agent 的记忆边际效用 + 全程 token/美元计费; 自动泄漏检查保证任务确实依赖跨 episode 事实; updated-fact 阶梯上嵌入检索不稳定 (0.30-0.95 跨模型)。
- Mem2ActBench [2601.19935](https://arxiv.org/abs/2601.19935): 记忆能否落到工具选择与参数 grounding; 合成 2,029 会话。
- LongMemEval-V2 [2605.12493](https://arxiv.org/abs/2605.12493), SCALE@ICML2026 oral (OpenReview): 面向 web 环境经验内化, 451 道人工题, 历史轨迹最长 115M tokens, 五能力(静态状态/动态跟踪/工作流/环境坑/前提意识)。
- LifelongAgentBench (见上表)。EvoMemBench [2605.18421](https://arxiv.org/abs/2605.18421): 记忆scope(episode内/跨episode) x 内容(知识/执行)双轴, 统一协议比较 15 种记忆方法, 结论是长上下文基线仍很强。

**对话记忆升级型**
- RHELM [2605.31086](https://arxiv.org/abs/2605.31086): 异构数据流(文档/邮件)+演化 persona, 27 种记忆特征映射。
- DynamicMem [2606.22877](https://arxiv.org/abs/2606.22877): 合成 15 个月、人均 2.2M tokens、16 个 App 的多源行为流, 档案随时间漂移。
- Memora [2604.20006](https://arxiv.org/abs/2604.20006): 记忆/推理/推荐三任务 + FAMA 指标 (对使用已失效记忆做惩罚)。
- AMemGym [2603.01966](https://arxiv.org/abs/2603.01966): 交互式 on-policy 评测环境, 直接批评现有基准依赖静态 off-policy 数据。
- LoCoMo-Plus [2602.10715](https://arxiv.org/abs/2602.10715) (OpenReview 显示 ACL 2026): 认知记忆/隐式约束, cue-trigger 语义断连; 指出字符串匹配指标失配。
- LOCOMO-CONV [2609.03467](https://arxiv.org/abs/2609.03467): 从 LoCoMo 派生对话式(非 QA 探测)查询, 显示 QA 式评测掩盖了 implicit/composed 查询上的检索缺口。
- AgentMemBench [2608.00009](https://arxiv.org/abs/2608.00009): 五种记忆策略在 LoCoMo/MultiDoc2Dial/MSC 上统一复评, 指标含 Recall@k/MRR/nDCG/F1/faithfulness/Memory Footprint/延迟。
- 题录级(摘要未逐一核读): SocialMemBench [2605.17789], GroupMemBench [2605.14498](多方对话), MemTrapBench [2608.20202](认知陷阱), MemToC [2608.26295](记忆-工具冲突), UTILMEM [2608.30508](证据利用), ClinTraceBench [2609.01111](纵向临床), H2HMem [2606.09461](多模态), MemArena [2608.02613](端侧), RUMBA [2607.21447](俄语)。

**协议审计/元评测型 (对本项目最重要)**
- MemDelta [2606.29914](https://arxiv.org/abs/2606.29914): LongMemEval-S 上单变量控制; 换嵌入模型 +6.2pp; Mem0 赢 MiniLM-RAG 却输 cloud-RAG; 建议固定嵌入、按模型族分层、报告写路径成本。
- TIAP / "Same Ranking, Different Winner" [2605.24060](https://arxiv.org/abs/2605.24060): 评分对象(Raw/Source/Canonical)不同即可翻转 LoCoMo 与 LongMemEval-S 上 83.4-94.0% 查询的 nDCG 结论。
- ReFind [2608.12888](https://arxiv.org/abs/2608.12888): 在 MemoryAgentBench 设定下, 无结构 agentic 词法检索 58.2 平均准确率超过 HippoRAG 2 (53.2), 质疑"结构本身"的贡献。
- Total Recall at What Cost [2608.11879](https://arxiv.org/abs/2608.11879): 用 665 道 LoCoMo 题配对测服务成本; 记忆系统成本不可由对话长度预测; 有的系统 400 turns 内永不回本; 无系统在精度与成本上双赢。
- Beyond Memory Leaderboards [2607.16848](https://arxiv.org/abs/2607.16848): 科研全文记忆 (PAIM/PTr); 控制检索预算后 Graphiti 的领先消失; 强调"榜单离开完整协议不可解释"。
- MemTrace [2605.28732](https://arxiv.org/abs/2605.28732): 记忆演化图 + 错误归因 (MemTraceBench)。
- What Eviction Destroys [2609.08279](https://arxiv.org/abs/2609.08279): restore-counterfactual 把错误分为可恢复/不可逆; 8k budget 下四种驱逐策略不可逆率 1.00。
- Storage Is Not Memory [2605.04897](https://arxiv.org/abs/2605.04897): 第三方在 LoCoMo 10 对话/1,540 题、统一答案模型下复测: Mem0 61.4%, Supermemory 65.4%, Zep 约 71%, EverMemOS 94.5%, 自家 93.0%。(厂商间数字弥散的独立证据)

---

## 3. 已知问题与争议 (重点核实项)

### 3.1 LoCoMo 区分不出写入策略优劣: Nous 消融 (已核实)
[2606.22030](https://arxiv.org/abs/2606.22030) "When Does Belief-Based Agent Memory Help?" (v2, 2026-07-16)。"Nous" 是文中提出的信念记忆架构名 (逐实体-属性的类别分布 + 贝叶斯更新)。经全文核实:
- 在 LoCoMo (3 段对话子集) 上做受控消融: 把贝叶斯信念更新整个换成 naive last-write-wins, 准确率不降。归因: "ordinary conversational QA rarely presents contradictory, low-reliability evidence", 即 LoCoMo 题目几乎不包含矛盾/不同可信度的证据, 因此任何写入/更新策略都测不出差别。
- 换到其自建的 contradiction/staleness 基准并提供逐观测可靠性信号后, 信念更新对 last-write-wins 优势 100 vs 67, 对 LLM-over-raw-memory 100 vs 90。
- 同文档记录: 同一批输出上 strict token-F1 与宽松 LLM-judge 宏平均差 27.5 分 (32.45 vs 59.97), 明确警告 LoCoMo 排行榜跨论文不可比。
含义: 用 LoCoMo 做"写入策略"消融属于协议错配; 需要含矛盾/更新/可靠性差异的数据 (LongMemEval 的 knowledge-update 子类、MemOps、HaluMem-update 更合适)。

### 3.2 LoCoMo 其他质量问题 (arXiv 可核实部分)
- 类别口径不一: 官方 7,512 题含 adversarial 1,871 题 (24.9%); Mem0 论文明确剔除 adversarial (理由: 无 ground truth), 社区又常用 10 对话/1,540 题子集 (2605.04897), Total Recall 用 665 题 (2608.11879)。同名"LoCoMo 成绩"实际对应不同题集。
- 饱和信号: MemoryArena (2602.16313) 称 agent 在 LoCoMo 类基准"near-saturated"; Hindsight Memory-PRM (2608.29605) 在 held-out LoCoMo 上 8B 模型即达 77.5%。
- judge/指标不稳定: 见 3.1 的 27.5 分差; TIAP (2605.24060) 证明换评分对象即可翻转结论; LOCOMO-CONV (2609.03467) 显示 QA 探测式协议高估检索能力。
- 数据本身为 LLM 生成+人工校对 (原文自述), 注意: 关于其标注错误率的具体第三方审计 (社区博客盛传), 本环境查不到 arXiv 来源, 标 未核实。

### 3.3 Zep 对 DMR 的批评 (已核实, Zep 论文 §4.2 原文)
Zep (2501.13956) 引用并批评 MemGPT 的 DMR: (1) 每对话仅约 60 条消息, 轻松塞进现代上下文窗口, 全上下文基线本身就很高 (gpt-4o-mini 全上下文 98.0%); (2) 只有单轮事实检索题, 测不了复杂记忆理解; (3) 许多问题措辞含糊, 如 "favorite drink to relax with" 在对话中从未被如此表述; (4) 不代表企业真实用例。因此 Zep 转投 LongMemEval。

### 3.4 Mem0 与 Zep 互相质疑的公案 (部分核实)
arXiv 上可核实的事实链:
- Mem0 论文 (2504.19413, 2025-04) 在 LoCoMo 上评了 Zep 平台版, 自述保留了时间戳; 结论: 总体上 "LangMem and Zep both score around 8% relatively less against our models on J score", 但 open-domain 类 Zep J=76.60 反超 Mem0g 的 75.71; 表 2 总体 J 一列中 Mem0=66.88, Mem0g=68.44 (Zep 行数值因 PDF 表格解析对位问题, 精确值未核实)。
- Zep 论文 (2501.13956, 2025-01, 早于 Mem0 论文) 从未在 LoCoMo 上自评, 只用 DMR + LongMemEval; 双方论文因此没有共同基准上的直接对账。
- 双方在 arXiv 上均无针对对方的正式回应论文 (检索 abs:"Zep" AND abs:"Mem0" 的 8 篇命中全为第三方引用, 无争论文)。
- 公案的核心交锋 (Zep 方博客指控 Mem0 错误配置 Zep 的写入等待/检索参数并质疑 LoCoMo 本身质量; Mem0 方回应) 发生在博客/GitHub 层面: 本环境无法访问, 未核实。
- 第三方旁证: Storage Is Not Memory (2605.04897) 统一答案模型复测得 Mem0 61.4% vs Zep 约 71%, 与 Mem0 论文"Zep 低约 8%"方向相反, 支持"配置与协议主导结论"的判断; MemDelta (2606.29914) 进一步显示 Mem0 对 cloud-RAG 仅在 2/6 题型打平且成本 50 倍。
结论: 该公案在学术语料中的可靠教训是协议层的: 评厂商系统必须报告写入等待/检索参数/嵌入模型/评分对象, 且要有第三方统一复测。

### 3.5 判分器与协议的系统性不稳定 (跨基准)
- token-F1 vs LLM-judge 宏平均差 27.5 分 (2606.22030)。
- 评分对象 (raw turn vs 派生记忆) 变更翻转 83-94% 查询的检索指标 (2605.24060)。
- 换嵌入模型 +6.2pp, 换底座模型族可让 full-context 与 RAG 排名互换 (2606.29914)。
- 检索预算不控制则图结构系统虚假领先 (2607.16848)。

---

## 4. 被 memory 论文借用的 agent 任务基准

- **ALFWorld**: Memp (2508.06433) 用 TravelPlanner+ALFWorld 评程序性记忆的 Build/Retrieve/Update 策略; ExpeL (2308.10144) 经验学习 (具体任务列表在正文, 摘要未列全, 细节未核实); 2026 年大量 self-evolving/skill 论文 (如 2609.02217, 2608.25500) 摘要提及 ALFWorld。用法: 跨任务经验/技能复用, 指标是任务成功率的提升曲线, 记忆只是自变量。
- **WebArena / Mind2Web**: Agent Workflow Memory (2409.07429) 在两者上做 workflow 归纳复用 (1000+ 任务); LongMemEval-V2 (2605.12493) 直接把 web agent 轨迹变成记忆评测语料; MemoryArena 含 web 导航。用法: 在线/离线 workflow 记忆, 指标为成功率与步数效率。
- **SWE-bench 类**: DreamBench-SWE (2608.20664) 专门构造多会话软件任务, 后续任务依赖前会话不可推断证据, 用隐藏可执行 oracle 判分 (预注册协议, 对 Mem0 托管配置 97/180 vs 无记忆 21/180); SWE-MeM (2606.28434, 题录级); 2608.06811 用情景记忆做 issue 修复。用法: 可执行判分回避 LLM-judge, 是记忆评测里最硬的 oracle。
- 共同缺陷: 这些基准原生不含记忆 GT, 记忆贡献只能靠"有/无记忆"或"换记忆实现"的差分来度量, 且 episode 划分是论文自定的。

---

## 5. 评测协议缺口: 能否评"写入决策质量"本身

### 5.1 现状判定
截至 2026-09, **主流基准 (LoCoMo/LongMemEval/MemBench/MemoryAgentBench 等) 都不直接评写入决策**: 它们只看端到端 QA, 写入策略的差别被检索器和判分器噪声淹没 (3.1 的 Nous 消融是直接证据)。但 2026 年已出现一批操作级供给, 缺口正在收窄:

### 5.2 "该记/不该记"的 ground truth: 已有但零散
- MemOps (2607.12893): 把记忆重构为显式生命周期操作 (remember/forget/update/reflect), 提供 **gold operation traces** (trigger/target/scope/state transition/evidence) + 六类操作级探针。这是目前最接近"写入决策 GT"的公开基准。
- HaluMem (2511.03506): 记忆抽取与更新两个操作阶段的对错标注 (约 15k memory points)。
- DeltaMem (2604.01560): 合成对话 + **操作级记忆更新标签**, 并用 Memory-based Levenshtein Distance 做更新奖励 (训练用, 也可借作评测)。
- AdaMem/AdaMem-Bench (2606.21144): 偏好条件化写入控制; GT 是"每个交互情境下用户想让 agent 记什么"; 报了 QA +4.35pp 同时持久记忆量 -9.27%, 是少见的"写得更少更好"的量化。
- MemoryRewardBench (2601.11969): 测 reward model 能否判别记忆管理好坏 (10 种管理模式, 8K-128K)。
- Fine-Mem (2601.08435) 与 Hindsight Memory-PRM (2608.29605): 不给人工 GT, 而用 chunk 级辅助 QA / 检索命中+引用+删除重答干预 构造逐操作 credit, 属于"用因果干预代替标注"的路线。
- TrustMem (2606.25161): 用 Memory Transition Verifier 按 coverage/preservation/faithfulness 三轴评每次记忆状态转移。
- 遗忘/驱逐侧: PA-Bench (2609.05767) 的 query-blind 驱逐协议, What Eviction Destroys (2609.08279) 的 restore-counterfactual (不可逆 vs 可恢复), Memora 的 FAMA (2604.20006, 罚用失效记忆), ForgetBench (2607.26455, 参数记忆侧, 题录级)。
- 结论: "该记/不该记"的显式 GT 存在于 MemOps/HaluMem/DeltaMem/AdaMem-Bench, 但相互口径不通, 且都偏对话域; 工具/代码域只有 DreamBench-SWE 的可执行 oracle 间接覆盖。

### 5.3 记忆成本是否入指标: 少数纳入, 无统一口径
- 已纳入者: Mem0 论文 (p50/p95 延迟 + token 成本), MERIT (逐操作 token/美元全记账), Total Recall (成本-精度配对 + break-even 分析), AgentMemBench (Memory Footprint + Latency), AdaMem (持久记忆量), Beyond Memory Leaderboards (检索预算控制), Hindsight-PRM (报告 1/8 上下文用量), 题录级: MemoryCPT (2608.04843), BudgetBench (2609.13149), LIMBO (2609.14138)。
- 但 LoCoMo/LongMemEval/MemoryAgentBench 官方协议均不要求报成本; Total Recall 证明成本不可由对话长度外推, 必须实测。

### 5.4 对 ICLR 实验设计的落点
1. 主端到端榜: LongMemEval-S (ICLR 2025 出处硬、五能力含 knowledge-update 与 abstention), 辅以 MemoryAgentBench (四能力覆盖含选择性遗忘)。LoCoMo 只作 legacy 对照, 必须写明子集口径+评分对象+judge, 且不要在其上做写入消融。
- 写入决策主张的直接证据: MemOps (操作 trace) + HaluMem (抽取/更新幻觉) + 自建矛盾/可靠性对照 (仿 2606.22030 协议) 三件套; 若做驱逐/遗忘, 用 restore-counterfactual (2609.08279) 与 query-blind (2609.05767) 协议。
2. 成本轴: 照 MERIT/Total Recall 报逐操作 token + 存储量 + break-even; 照 MemDelta 固定嵌入与底座、做单变量对照。
3. 行动耦合 (可选第二实验): MemoryArena 或 LongMemEval-V2; 代码域用 DreamBench-SWE 的可执行 oracle 思路。
4. 协议卫生 (审稿人红线): 报告评分对象 (TIAP), 同时给 F1 与 judge 并给分歧率 (Nous 27.5 分教训), 第三方基线数字不可直接引用不同配置的原论文值 (Mem0-Zep 公案教训)。

---

## 6. 未核实清单
- Zep 与 Mem0 的博客交锋原文与具体指控参数 (无网页通道)。
- LoCoMo 标注错误率的第三方逐条审计 (仅社区传闻, arXiv 无来源)。
- MemGPT ICLR 2024 投稿的最终结果 (OpenReview 仅见 Submission8071 的评论存在)。
- MemoryAgentBench 的正式 venue; GoodAI benchmark 的 GitHub 版本演进; MemBench 数据集具体规模。
- SWE-MeM、BudgetBench、MemoryCPT、LIMBO、SocialMemBench、GroupMemBench、MemTrapBench、MemToC、UTILMEM、ClinTraceBench、ForgetBench、MemEvoBench [2604.15774]、H2HMem、MemArena、RUMBA: 仅核实标题与 ID, 未读摘要全文。
