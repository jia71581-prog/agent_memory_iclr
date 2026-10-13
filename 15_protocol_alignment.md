# 15. LongMemEval 实验协议对齐调研(QC-Lifecycle 论文用)

日期:2026-09-17。来源全部为 arXiv PDF 原文(pdftotext 抽取)+ HF mirror API;逐条标注出处。
检索通道:arXiv API + arxiv.org PDF 直连 + hf-mirror.com(huggingface.co 本机直连超时,mirror 可用)。
tavily/WebSearch/GitHub 未使用(按任务约束)。

---

## 0. 结论速览

1. **文献中不存在统一口径的 LongMemEval-S 榜单**。判分模型(GPT-4o / GPT-4o-mini / Llama-3.3-70B / 同族自评)、答题模型、数据版本(原版 vs cleaned)、子集(500 / 470 / 78)四个维度全在漂移;跨论文数字只能做"量级参考",不能并入同一张表直接排序。
2. **cleaned 版已被至少一篇 arXiv 论文显式采用并 pin 了 hash**(What-Eviction-Destroys, 2609.08279),且该 hash 与本项目 data/ 下文件完全一致(sha256 d6f21ea9...c3a442)。其余论文均未声明版本;2025-09-19 之后从 HF 下载的一律默认拿到 cleaned 版(原版已标 deprecated),但论文文本仍写 "~115k tokens"。
3. **KU/TR 是公认的两个难类**,失败模式已被多篇论文分解:KU=旧值干扰(stale-fact),TR=时间范围检索失败+时间戳元数据推理。当前最好系统 KU≈97%、TR≈87%(GPT-4.1-mini,LycheeMemory 自报);全上下文基线 TR 仅 34-48%。
4. **无人用 60 题分层样本报主表数字**。最接近的先例是 MemDelta(n=100-500 按成本分格,报 effective n+bootstrap CI)和 RD-Forget(只用 78 题 KU 子集+oracle 设置)。建议:主表跑全量 500(cleaned),60 题样本只用于开发/消融并报 CI。

---

## 1. SOTA 数字表(LongMemEval,按论文分组)

### 1.1 LongMemEval 原文(2410.10813, ICLR 2025)

协议:判分 = gpt-4o-2024-08-06,官方分题型 prompt(temp-reasoning 容忍 off-by-one 天数;KU 只要含更新后答案即对;SSP 按 rubric),与人类专家一致率 >97%(30 题/类抽检,SSP 与 abstention 两类最低 0.90)。

| 设置 | 模型 | 变体 | Overall Acc |
|---|---|---|---|
| Oracle(仅证据 session) | GPT-4o | oracle | 0.870(无 CoN)/ 0.924(CoN+JSON) |
| 全上下文 | GPT-4o | S | 0.606(无 CoN)/ 0.640(CoN) |
| 全上下文 | Llama 3.1 70B | S | 0.334 / 0.286(CoN) |
| 全上下文 | Llama 3.1 8B | S | 0.454 / 0.420(CoN) |
| 全上下文 | Phi-3 14B / Phi-3.5 4B | S | 0.380 / 0.342(无 CoN) |
| 商业系统(97 题短历史子集) | ChatGPT(GPT-4o) | 简化 | 0.5773;Coze(GPT-4o)0.3299 |

原文自己的记忆管线优化结果全部在 **-M 变体**上报(Table 3:session+fact 键扩展 GPT-4o Top-10 = 0.700-0.720;时间感知查询扩展提升 TR 检索 recall 6.8-11.3%),没有在 -S 上报完整管线主表。

商业系统人评分题型(原文 Table 7,97 题子集,非 -S 全量):ChatGPT(GPT-4o)IE 0.688 / MR 0.441 / KU 0.833 / TR 0.435;ChatGPT(GPT-4o-mini)IE 1.000 / MR 0.647 / KU 0.667 / TR 0.652。

### 1.2 Zep(2501.13956, arXiv 2025-01)

协议:LongMemEval-S 原版(实验 2024-12~2025-01,早于 cleaned);判分 = GPT-4o + 官方分题型 prompt;答题模型 gpt-4o-mini-2024-07-18 与 gpt-4o-2024-11-20;检索 top-10 节点+边。表内只列六类(未提 abstention 30 题如何处理,**未核实**)。

| 方法 | 答题模型 | Overall | SSU | SSA | SSP | MS | KU | TR |
|---|---|---|---|---|---|---|---|---|
| Full-context | gpt-4o-mini | 55.4% | 81.4 | 81.8 | 30.0 | 40.6 | 76.9 | 36.5 |
| Zep | gpt-4o-mini | 63.8% | 92.9 | 75.0 | 53.3 | 47.4 | 74.4 | 54.1 |
| Full-context | gpt-4o | 60.2% | 81.4 | 94.6 | 20.0 | 44.3 | 78.2 | 45.1 |
| Zep | gpt-4o | 71.2% | 92.9 | 80.4 | 56.7 | 57.9 | 83.3 | 62.4 |

交叉验证:Zep 的 gpt-4o 全上下文 60.2% 与原文 0.606 吻合(同判分模型、同原版数据)。
Zep 附带延迟:全上下文 28.9-31.3s vs Zep 2.6-3.2s(-90%),上下文 115k→1.6k tokens。

### 1.3 Mem0(2504.19413)、MemoryOS(2506.06326)、SeCom(2502.05589)

三篇**均未在 LongMemEval 上自评**(PDF 全文 0 次提及可用作评测的 LongMemEval 段落):
- Mem0:只有 LoCoMo(LLM-judge J 指标)。
- MemoryOS:只有 LoCoMo + GVD(F1/BLEU-1 为主)。
- SeCom:LoCoMo + Long-MT-Bench+。
它们的 LongMemEval-S 数字全部来自**第三方复现**(见 1.4-1.6),引用时必须注明"复现值,非自报"。

### 1.4 LightMem(2510.18866, ICLR 2026)

协议:LongMemEval-S(v4 更新于 2025-10 后,版本未声明;正文写 500 题);判分 = **GPT-4o-mini** + 官方原版分题型 prompt(附录 E.1 与原文 prompt 逐字一致);答题/记忆模型同族(GPT-4o-mini、Qwen3-30B-A3B-Instruct-2507、GLM-4.6);检索条数各法一致。只报 Overall ACC,无分题型。

| 方法 | GPT-4o-mini | Qwen3-30B | GLM-4.6 |
|---|---|---|---|
| FullText | 56.80 | 54.80 | 36.71 |
| NaiveRAG | 61.00 | 60.80 | 73.20 |
| LangMem | 37.20 | 50.80 | 49.20 |
| A-MEM | 62.60 | 65.20 | 70.60 |
| MemoryOS(复现) | 44.80 | 49.60 | — |
| Mem0(复现) | 53.61 | 39.51 | — |
| LightMem(最好配置) | **68.64**(r=0.7,th=512) | **70.20**(r=0.6,th=768) | **73.20**(r=0.6) |

注意 FullText gpt-4o-mini 56.80 vs Zep 55.4 vs MemDelta 49.8:同名基线三篇论文差 7pp,判分模型(4o vs 4o-mini)与 prompt 组装差异即可造成此量级。

### 1.5 LycheeMemory V2(2608.12990, arXiv 2026-08)

协议:LongMemEval-S 500 题(版本未声明;正文仍写 ~115K tokens);判分 = **GPT-4o-mini** + 官方分题型 yes/no prompt(10-token 输出上限);答题模型 GPT-4.1-Mini / GPT-4o-Mini;同一 backbone 下所有方法同判分。Overall 为 500 题加权。

GPT-4.1-Mini 组(Table 2 全文照录):

| 方法 | SSU | SSA | SSP | MS | KU | TR | Overall |
|---|---|---|---|---|---|---|---|
| Full Context | 91.43 | 100.00 | 56.67 | 53.38 | 75.64 | 48.12 | 66.20 |
| Naive RAG | 85.71 | 82.36 | 83.33 | 63.91 | 75.64 | 45.86 | 67.20 |
| MemOS(复现) | 90.00 | 64.29 | 50.00 | 54.14 | 70.51 | 63.91 | 65.20 |
| MemoryOS(复现) | 94.29 | 89.29 | 100.00 | 67.67 | 80.77 | 54.89 | 74.40 |
| Mem0(复现) | 95.71 | 50.00 | 90.00 | 69.92 | 74.36 | 62.41 | 71.20 |
| A-Mem(复现) | 95.71 | 100.00 | 63.33 | 61.65 | 82.05 | 52.63 | 71.60 |
| Nemori(复现) | 90.00 | 92.90 | 86.70 | 55.60 | 79.50 | 72.20 | 74.60 |
| LightMem(复现) | 90.00 | 23.21 | 76.67 | 51.13 | 88.46 | 84.21 | 69.60 |
| TiMem(复现) | 92.86 | 78.57 | 73.33 | 66.92 | 79.49 | 72.93 | 75.80 |
| **LycheeMemory** | **100.00** | 98.21 | 90.00 | **87.97** | **97.44** | **87.22** | **92.20** |

GPT-4o-Mini 组:Full Context 45.60;Naive RAG 59.40;MemOS 77.80;MemoryOS 67.80;Mem0 64.40;A-Mem 63.20;Nemori 64.20;LightMem 69.81;TiMem 76.88;LycheeMemory **78.80**(SSU 97.14 / SSA 96.43 / SSP 70.00 / MS 72.93 / KU 74.36 / TR 72.18)。

注意:同一 LycheeMemory 换 4.1-mini→4o-mini,Overall 掉 13.4pp,KU 掉 23pp——答题模型对分题型数字的影响与方法本身同量级。

### 1.6 CueMem(2609.12354, arXiv 2026-09)

协议:LongMemEval-S 500 题(版本未声明);**答题+记忆抽取+判分全部用 Llama-3.3-70B-Instruct**(自评自判);嵌入 all-MiniLM-L6-v2;LLM-as-judge 是单一通用 prompt(非官方分题型 prompt,**与其余论文均不可比**)。

| 方法 | SSU | SSA | SSP | MS | TR | KU | Overall |
|---|---|---|---|---|---|---|---|
| Naive RAG | 78.57 | 100.00 | 26.67 | 48.12 | 54.14 | 75.64 | 62.80 |
| MemoryOS(复现) | 88.57 | 62.50 | 36.67 | 60.90 | 57.89 | 74.36 | 64.80 |
| Mem0(复现) | 74.29 | 12.50 | 20.00 | 45.86 | 49.62 | 71.79 | 49.60 |
| A-Mem(复现) | 94.29 | 98.21 | 26.67 | 55.64 | 51.13 | 78.21 | 66.40 |
| LightMem(复现) | 91.43 | 37.50 | 33.33 | 75.94 | 79.70 | 84.62 | 73.60 |
| **CueMem** | 94.29 | 91.07 | 40.00 | 60.90 | 73.68 | **87.18** | **75.20** |

### 1.7 RD-Forget(2609.10263, arXiv 2026-09)

协议:**只用 LongMemEval 的 oracle 数据中 78 题 knowledge-update 子集**(LME-KU),非 -S 全量;判分 = 与答题同族模型(Qwen3.5-flash / GPT-5.6-Luna / MiniMax-M2.5 / Kimi-K2.5,论文自认"within-protocol 而非独立外判");binary semantic judgment。

LME-KU 列(Table 1):RD-Forget 74.36(Qwen)/ **93.59**(Luna)/ 87.18(MiniMax)/ 76.92(Kimi);ACE 60.26/83.33/61.54/70.51;ReasoningBank 60.26/71.79/51.28/62.82。
消融(Luna,Table 2):matched Full 93.59;**−FORGET(保留旧版本为活跃)60.26,掉 33.33pp**;−QC 82.05;−RESCUE 91.03;−SLOT 89.74。

### 1.8 MemDelta(2606.29914, arXiv 2026-06)——协议敏感性对照研究

协议:LongMemEval-S 500 题(版本未声明);判分 = GPT-4o-mini 固定二元 prompt;paired 设计(McNemar + bootstrap CI);嵌入模型单变量控制。

| 策略 | n | Acc | 95% CI |
|---|---|---|---|
| S0 无记忆 | 500 | 2.2% | [1.0, 3.6] |
| 随机检索 | 500 | 3.2% | [1.8, 4.8] |
| S2 agent 自管理 scratchpad | 100 | 42.0% | [32.0, 52.0] |
| S4 verbatim RAG(MiniLM) | 500 | 47.2% | [42.8, 51.4] |
| S1 全上下文(GPT-4o-mini) | 500 | 49.8% | [45.6, 54.0] |
| S4b verbatim RAG(cloud embedding) | 500 | 53.4% | [49.0, 57.8] |

关键敏感性结论:仅换嵌入模型 +6.2pp(p=0.004);Mem0 对 MiniLM-RAG +11pp、对 cloud-RAG −1.2pp(88 题匹配子集上 72.7 vs 73.9,p=1.0,成本 50×);换底座翻转结论(Sonnet 全上下文 14.0% 因拒答 63%,RAG 44.7%;Gemini 反之 +14pp 偏全上下文)。分题型嵌入消融(n=500,GPT-4o-mini):SSU 75.7→87.1;TR 30.1→40.6;MS 24.1→35.3;KU 62.8→62.8(嵌入不敏感);SSP 40.0→30.0。

### 1.9 What-Eviction-Destroys(2609.08279, COLM 2026 WS)——cleaned 版直接先例

协议:**显式使用 longmemeval_s_cleaned.json,sha256 d6f21ea9...c3a442,470 题(剔除 30 abstention),~102k tokens/history(o200k tokenizer),并注明"the deprecated original release differs"**;读者+判官均 GPT-4o-mini(temp 0),鲁棒性读者 GPT-5.4-mini,独立复判 GPT-5.5(κ=0.90,一致率 95.7%);3 seeds,question-cluster bootstrap 10k + Holm-Bonferroni。该文报的是 eviction 策略的可恢复/不可逆错误分解,不是 SOTA 准确率主表,但其协议(pin hash、判官校准、预注册假设族)是我们论文可以直接对齐的最严格先例。

### 1.10 不在 LongMemEval 上评的竞品

MemoryLACE(2609.03201):BEAM + StructMemEval,无 LME 数字。与 QC-Lifecycle 对比时只能引方法,不能引数字。

---

## 2. 口径可比性判定

**可比组(需全部满足:同数据版本+同判分模型与 prompt+同答题模型)**:
- 组 A(原版, GPT-4o 判, 官方 prompt):原文 pilot ↔ Zep。gpt-4o 全上下文 0.606 vs 60.2% 互证。
- 组 B(GPT-4o-mini 判, 官方分题型 prompt, GPT-4o-mini 答):LightMem 自报 ↔ LycheeMemory 表(其 LightMem 复现 69.81 与自报 68.64 相差 1.2pp,基本吻合)。版本存疑(见 §3)但内部一致。

**不可比,只能作量级参考**:
- CueMem 全表(Llama-3.3-70B 自评自判 + 非官方 judge prompt)。
- RD-Forget(oracle 设置 + KU 子集 + 同族自判)——数字与 -S 全量完全不同口径。
- MemDelta 与其它论文之间(判分 prompt 是自定单一 prompt,非官方分题型版;但其内部 paired 比较自洽)。
- 任何 Mem0/MemoryOS/A-Mem/LightMem 的"复现值"跨论文比较:LightMem 复现值在三篇论文里分别为 69.60(4.1-mini)、69.81(4o-mini)、73.60(Llama70B);Mem0 为 71.20 / 64.40 / 49.60 / 53.61——同一系统跨协议波动 >20pp。

**我们论文的口径声明模板(建议放实验设置节)**:
> We evaluate on LongMemEval-S *cleaned* (HF `xiaowu0162/longmemeval-cleaned`, `longmemeval_s_cleaned.json`, sha256 `d6f21e...c3a442`, released 2025-09-19), which removes noisy history sessions from the original release; numbers are therefore not directly comparable to results reported on the deprecated original version (e.g., Zep, and all pre-2025-09 work). Answers are generated by <model, dated snapshot>, judged by <judge model> with the official per-type prompts of Wu et al. (2025). We re-run all baselines under this identical stack (same embeddings, same retrieval budget, same judge), following the controlled-comparison recommendations of MemDelta (2606.29914).

---

## 3. cleaned 版口径事实

- HF `xiaowu0162/longmemeval-cleaned`:创建 2025-09-19,原 `xiaowu0162/longmemeval` 同日标 deprecated。官方描述:"removes noisy history sessions that interfere with the answer correctness";逐 session 处理明细在作者提供的 Google Sheet(未抓取,**未核实**)。
- 三个 split:longmemeval_oracle(15.4MB)/ longmemeval_s_cleaned(277MB)/ longmemeval_m_cleaned(2.74GB)。
- 本地核验(data/longmemeval_s_cleaned.json):**仍是 500 题,题型分布与原版一致**(SSU 70 / SSA 56 / SSP 30 / MS 133 / KU 78 / TR 133;其中 30 题带 `_abs` 后缀为 abstention)。清洗只动 haystack(去噪声 session),不动题目与答案;历史长度从 ~115k 降到 ~102k tokens(2609.08279 用 o200k 计)。
- 因此:**题目可比、准确率不可比**——去掉的正是"干扰答案正确性"的 session,同一系统在 cleaned 上的分数只会 ≥ 原版趋势,任何与原版数字(Zep、原文 pilot)的直接对比都会高估我们的方法。
- 文献采用情况:显式声明 cleaned 并 pin hash 的仅 2609.08279 一篇(其 hash 与我们本地文件**逐位一致**)。LightMem(v4)、LycheeMemory、CueMem、MemDelta 均未声明版本;其中 2025-09-19 后从 HF 拉数据的论文事实上只能拿到 cleaned 版,但正文仍沿用 "~115K tokens" 的原版描述——版本混用是当前文献的普遍暗伤,我们在论文里点出这一点本身即是贡献。
- 另有第二种"cleaned":`kellyhongg/cleaned-longmemeval-s`(Chroma "Context Rot" 报告用,删了歧义题+修 oracle 版),与官方 cleaned **不是同一个东西**,引用时勿混。

---

## 4. knowledge-update 与 temporal-reasoning 难点分析

### 4.1 knowledge-update(78 题)

失败模式(多源交叉):
1. **旧值干扰 / stale-fact**:RD-Forget 消融最硬——把"识别出的旧版本保持活跃"(−FORGET)在 oracle KU 上从 93.59 掉到 60.26(−33.3pp),是所有组件中最大跌幅;证明 KU 的主瓶颈是"读侧抑制被取代的值",不是找不到新值。
2. **检索只召回更新后知识、漏掉更新前信息**:LongMemEval 原文附录 E.5:reader 答对而检索"判失败"的 ~10% 案例几乎全是 KU 类(严格检索判据要求同时召回 pre-update 证据)。
3. **写侧压缩覆盖**:原文对 ChatGPT 的人工分析——先正确记录,随历史增长在压缩时改写/丢失早期信息;MemDelta S2 scratchpad 在多 session 类塌到 3.3% 同理。
4. **嵌入不敏感**:MemDelta 分题型消融中 KU 是唯一换嵌入完全不动的类(62.8→62.8)——KU 瓶颈在读/写逻辑而非向量检索质量,对我们 lifecycle 论文是有利论据。

强弱与天花板:全上下文 KU 72-78%(Zep 76.9/78.2;LycheeMemory Full Context 75.64;MemDelta S1 71.8);普通 RAG/记忆系统 70-82%;LightMem 88.46(复现值)、CueMem 87.18、LycheeMemory **97.44**(4.1-mini,当前最高);oracle 设置下 RD-Forget 93.59。天花板可视为 ~95%+(4.1-mini 级答题模型),4o-mini 级答题模型普遍卡在 74-88%。注意 Zep 在 gpt-4o-mini 上 KU 反而低于全上下文(74.4 vs 76.9)——图结构对弱模型的时间性数据反而是负担(Zep 原文自述)。

### 4.2 temporal-reasoning(133 题)

失败模式:
1. **时间无关检索天然失败**:原文 §5.4——查询含时间引用时朴素稠密检索差;时间感知索引+查询扩展提升 recall 6.8%(session 值)-11.3%(round 值)。
2. **时间范围抽取依赖强模型**:Llama 3.1 8B 做时间范围抽取时大量幻觉/漏检(附录 E.4 有 false positive 实例),弱模型套时间过滤反而伤害检索。
3. **双通道时间信息**:题目同时考 utterance 内显式时间和 session 时间戳元数据;检索器不索引 metadata 时 TR 必然塌(Zep 全上下文 gpt-4o-mini TR 36.5%,MemDelta S1 34.6%)。
4. **读侧排序**:原文默认把检索结果按时间戳排序喂 reader,以维持时间一致性——不做这步的复现值会更低。

强弱与天花板:全上下文 34.6-48.1%(TR 是全上下文最差类,长上下文时间聚合是 LLM 硬伤);朴素 RAG 45-58%;时间感知系统显著强:Zep(时间感知图)62.4%,Nemori 72.2,TiMem 72.9,LightMem 84.21(复现值),LycheeMemory **87.22**。天花板 ~85-90%(4.1-mini 级);4o-mini 级最好 ~72。TR 是"方法差异区分度最大"的类(全上下文→最好系统差 ~40pp),适合作为我们方法的展示位;KU 区分度对 lifecycle 类方法最有针对性。

---

## 5. 抽样口径:文献先例与建议

文献用量盘点:
- 全量 500:Zep、LightMem、LycheeMemory、CueMem、MemDelta 主格。
- 470(剔 abstention):2609.08279(理由:abstention 无 gold location 可 restore)。
- 子集报数先例:**MemDelta** 因成本对部分格用 n=100-300(Sonnet 300、Gemini 100、self-memory 100),做法是:报每格 effective n、bootstrap 95% CI、McNemar 配对检验,并声明子集按源顺序(题型有序)非随机;**RD-Forget** 只用 78 题 KU 子集(且是 oracle 数据);原文商业系统 pilot 用 97 题随机子集。
- **无任何论文用 ~60 题分层样本报主表**;也无人用"每题型等额抽样"。

对我们 60 题(每型 10 题)方案的定量判断:
- 等额抽样使 SSP(总量 30)占样本 1/6 而占真实分布 6%,MS/TR(各 133)被压到 1/13——**等权平均的 overall 与文献任何 overall 都不可比**;即便按真实分布重加权,n=60 的 overall 95% CI 约 ±12pp,分题型 n=10 的 CI 约 ±30pp,无法支撑"我们在 KU/TR 上更强"这类论文主张(MemDelta n=100 格的 CI 已达 ±10pp)。
- 成本核算:全量 500 题 × ~102k tokens ≈ 51M 入 tokens/系统臂;以 GPT-4o-mini 计费量级完全可承受(MemDelta、LightMem 均以 mini 级模型跑全量)。

**建议(定稿口径)**:
1. 论文主表**必须跑全量 500(cleaned)**,报 overall + 六类分题型;abstention 30 题按官方 abstention prompt 单独判并单列(或学 2609.08279 剔除并声明理由,若我们的 lifecycle 机制不涉拒答)。
2. n=60 分层样本降级为**开发集/消融快速迭代**用途;若消融必须以 60 题报数,报 bootstrap CI + 配对检验(McNemar),并在附录声明抽样种子与题号清单(预注册式冻结,符合本项目既有审计习惯)。
3. 所有基线在我们自己的固定栈下重跑(同嵌入、同检索预算、同判官),引用他文数字只放"文献报告值"独立表格并标注协议四元组(版本/答题/判官/子集);这直接采纳 MemDelta 的三条建议(fix embeddings、按模型族分层、报写路径成本)——顺带把写侧 token 成本报上,与 LightMem/LycheeMemory 的成本表可比。
4. 判官选择:GPT-4o-mini + 官方分题型 prompt 是 2025-10 之后事实标准(LightMem/LycheeMemory/MemDelta/2609.08279 四篇一致);比 GPT-4o 判便宜且有 2609.08279 的校准先例(GPT-5.5 独立复判 κ=0.90)可引。建议跟随,并对 ≥40 题做一次判官自一致+强模型复判校准。

---

## 6. 未核实清单

- cleaned 版逐 session 清洗明细(作者 Google Sheet)——未抓取。
- Zep 论文 overall 是否含 30 题 abstention——正文只列六类,未说明。
- LightMem/LycheeMemory/CueMem/MemDelta 实际用的是原版还是 cleaned 文件——论文均未声明,无法核实(仅能按下载时间推断)。
- LycheeMemory 表中 Nemori(Nan et al., 2025)与 TiMem(Li et al., 2026b)的自报原文数字——本轮未下载其原文,表中数字均为 LycheeMemory 复现值。
- MemGPT 在 LongMemEval 上的成绩:Zep 尝试评测失败(无法注入既有历史),文献中无数字。

## 7. 本轮下载的 PDF(/tmp/lme_pdfs/)

2410.10813(LongMemEval)、2501.13956(Zep)、2504.19413(Mem0)、2502.05589(SeCom)、2506.06326(MemoryOS)、2510.18866(LightMem v4)、2608.12990(LycheeMemory V2)、2609.12354(CueMem)、2609.10263(RD-Forget)、2606.29914(MemDelta)、2609.03201(MemoryLACE)、2609.08279(What-Eviction-Destroys)。
