# Training-Free 验证门控方案（用户约束：无训练资源）

调研日期：2026-09-16。本方案响应用户明确约束："原则上不想做任何训练，4 张 4090 训练不了 RL"，完全去掉可训练组件，改为规则+校准的 training-free 路线。

---

## 执行摘要

**用户约束生效**：无训练资源（4 张 4090 不足以训 RL），原定稿方案（learned 门 + 反事实标签生产 + 本地 4B LoRA 训练）不可行。

**方案调整**：保留误差独立验证信号（5 工具）+ 校准（温度标定+CRC）+ 污染定理（理论对比点），**去掉 learned 门**（换成手写规则聚合证据特征）+ 去掉反事实标签生产。

**差异化缩减到三点**（learned 消失）：
1. ~~learned（可训练）~~ ❌ 已去除
2. **calibrated**（校准）：温度标定 + conformal risk control 给出 FDR≤α 保证
3. **误差独立验证信号**：信号来自环境工具交互（检索 oracle、DB、计算验证、来源回查、记忆交叉核对），不是 LLM judge
4. **declarative facts**（事实型记忆）

**与竞品的区别只剩**：
- vs **Grounding Agent Memory（2609.11060，6 天前）**：他们二值决策（check/scope/refresh），我们概率+校准（FDR≤α 保证）。
- vs **MemGuard（2608.21867）**：他们无校准，我们有 CRC 校准。
- vs **GovMem（2607.02579）**：他们规则治理，我们规则+校准+理论（污染定理）。

**Novelty 风险判定（如实告知）**：
- evaluation-first 设计的 novelty 攻击原话："Calibration (temperature scaling + conformal prediction) is standard ML engineering, not a research contribution"（verdict: serious）。
- 审稿人很可能不买账："你只是给 Grounding Agent Memory 加了一层校准，这是工程非科研贡献；污染定理是 LUCID 重新包装"。
- **此方案能发表的概率 < 30%**，主要依赖：①污染定理（虽被攻击为"LUCID 代数包装"，但动力系统界与 ρ-相关门 FDR 下界仍独立）；②实证对照（长时程污染曲线，LLM-judge 门在相关地板平台化、本方法收敛）；③自建基准的压力协议设计（矛盾/可靠性分层/时间性更新，Nous/PA-Bench 式）。
- **最可能结局**：被拒或只能发 workshop（ICLR MemAgents WS 级别），无法进主会。

---

## 1. 核心主张（降级版，去掉 learned）

候选事实写入长期记忆前，一个**基于规则但经校准**的验证门，通过环境交互的误差独立信号（工具调用：检索 oracle、结构化 DB、计算验证、来源回查、记忆交叉核对）产生可靠性估计，用 conformal risk control 给出 FDR≤α 保证，并证明误差独立门 vs LLM-judge 相关门在长时程污染动力学上的理论界。

---

## 2. 方法（去掉训练部分）

### 2.1 验证信号生成（与定稿相同，5 工具）

环境提供 5 个验证工具（证据全部是程序化/环境交互产物，不含 LLM judge）：
1. **检索器**：对环境文档库做 BM25（+可选本地嵌入）检索，证据特征 = 命中片段与候选的匹配分、支持/反驳片段计数。
2. **结构化数据库**：世界状态 SQL 查询，证据 = 精确匹配/矛盾/查无（覆盖率 ρ_db 受控）。
3. **计算器/Python**：数值型与派生事实重算，证据 = 数值相等性。
4. **来源回查**：调出候选事实的原始话轮，程序化比对措辞置信度漂移（hedge 词消失、量词强化）——针对 Manufactured Confidence（2606.29279）与来源洗白（2607.29167）。
5. **记忆交叉核对**：与现有高置信条目的矛盾检测（内部工具，近零成本）。

### 2.2 规则聚合（替代 learned 门）

**手写规则将证据特征聚合为验证置信 v(c)**：
```
v(c) = weighted_vote({
  w_db    * indicator(DB 精确匹配),
  w_match * norm(检索匹配分),
  w_calc  * indicator(计算验证通过),
  w_drift * (1 - 来源置信度漂移),
  w_cons  * (1 - 矛盾计数 / 记忆库规模)
})
```
权重 {w_db, w_match, w_calc, w_drift, w_cons} 在小 held-out 集上网格搜索（非训练，只是超参调优），目标 = held-out 集上 F1 最大化。

**写入者自评 s(c)**：deepseek-v4-pro 的 verbalized confidence（k=5 自一致采样）。

**融合决策**：q(c) = σ( logit v(c) + λ·logit s(c) − logit π_0 )，λ 在网格搜索中与权重一起调（或固定为 0 退化为纯验证门）。

### 2.3 校准（conformal risk control，与定稿相同）

温度标定 + CRC 给出 **FDR≤α 保证**：
- 在 held-out 校准集上用 CRC 选 τ_α，使得 E[被写入条目中假条目比例] ≤ α（标准 conformal risk control）。
- 分布移位下报告违约幅度（ECE 在训域 A / 测域 B），承认 CRC 保证限同分布。

### 2.4 数据（Ground-Truth-First 协议，去掉反事实标签）

**脚本先行协议**（仿 Veracium 2607.21962 + Nous 2606.22030）：
1. 生成事实脚本：实体/属性/真值/有效期/来源通道（带可靠性参数）/注入矛盾与过时更新。
2. deepseek-flash 渲染成多会话交互流与可查询的合成世界（SQLite 事实库 + 工具 API）。
3. 每条候选事实与脚本比对得到真值标签 y。

**去掉反事实标签生产**（原需 5M API 调用）：只用脚本真值做监督，超参调优在 held-out 集上网格搜索规则权重，目标 F1。

**规模**：约 5K-10K 候选（比原方案的 10K-20K 可更小，因为无需训练，只需足够的网格搜索验证集）。

---

## 3. 理论部件（与定稿相同，污染定理）

### Prop 2'（污染动力学界，简化版）

- **自评/相关门**：污染率 p_t 发散或平台在高值 p* = μb_0/(λ_c−μκ)。
- **EI 门**：p_t 几何收敛到 p* = μα/λ_c（α 由 CRC 控制）。
- **相关判官下界**：LLM-judge 门（高 ρ）**原理上**不能把污染压到相关地板之下。

**用途**：related work 的理论对比——在讨论"为什么 LLM-judge 门不够"时引用推论，主实验以实证对照（污染曲线）为证据。

---

## 4. 实验设计（简化版）

### 4.1 基准

| 用途 | 基准 | 说明 |
|---|---|---|
| 主战场 | 自建脚本先行基准 | 5K-10K 候选；矛盾/可靠性分层/时间性更新；训/测查询分布不交 |
| 外部榜（可选） | LongMemEval-S 子集 | **需用户下载**；**响应用户"数据集太大先只做一个"**：只跑 knowledge-update 子类 |
| legacy 对照 | LoCoMo（写明子集口径） | 绝不用于写入消融 |

### 4.2 基线（对等输入）

1. **全量存储+检索**（cloud-embedding + rerank + 128k 模型）+ **预算扫描**（10%/25%/50%/75%/100% × FIFO/LRU/RANDOM）。
2. **LLM-judge 门**：deepseek-v4-pro + **相同工具证据特征**（只是把规则换成 prompt）。
3. **A-MAC 多因子**：无校准版。
4. **被动 "unverified" 标签**：Manufactured Confidence 失败基线。
5. **未校准规则门**：同方法但去 CRC，裸阈值。

### 4.3 杀手实验

**长时程污染压力测试**：注入来源洗白+置信改写，画四条污染曲线 p_t（无门/被动标签/LLM-judge 门/本方法），叠加理论预测（若拟合不佳承认理论限 toy model）。

### 4.4 消融

- 去校准：裸阈值 vs 温度标定 vs CRC（α 扫描 → FDR 覆盖曲线）。
- 验证信号换成 LLM-judge：检验 ρ-FDR 下界。
- 纯 v / 纯 s / 融合：λ 扫描。
- 分布移位：ECE 在训域 A / 测域 B（bins≥30 样本，bootstrapped CI）。

### 4.5 协议卫生

- 同报 token-F1 + LLM-judge + 分歧率。
- MERIT/Total Recall 口径的成本核算（**规则版成本更低**：无训练成本，只有推理+工具调用）。
- 固定嵌入与 judge 模型（MemDelta 教训）。

---

## 5. Novelty 防线（弱化版，如实告知）

**四点交集缩减到三点**（learned 消失）：

1. ~~learned（可训练）~~ ❌ **已去除** — 这是与 Grounding Agent Memory/MemGuard/GovMem 的最强差异化，现在消失了。

2. **calibrated**：温度标定 + CRC 给出 FDR≤α 保证 — **但 evaluation-first 攻击已判 serious："Calibration is standard ML engineering, not a research contribution"**。我们的辩护 = CRC 应用到记忆门控的首例 + 分布移位下的校准曲线（但审稿人可能不买账）。

3. **误差独立验证信号** — 仍成立，但 Grounding Agent Memory 已做环境探测验证（虽是 training-free，但"验证信号来自环境工具"这个想法已被占）。我们的差异 = 他们 check/scope/refresh 是二值，我们产生概率+校准；但这差异可能不够大。

4. **declarative facts** — 仍成立。

**与竞品的实际差异**：
- vs Grounding Agent Memory：二值决策 vs 概率+校准。
- vs MemGuard：无校准 vs 有 CRC。
- vs GovMem：规则治理 vs 规则+校准+污染定理。

**审稿人最可能的反应**：
- "你只是把 Grounding Agent Memory 的二值决策改成概率，然后套了标准 conformal calibration，这是工程改进非科研贡献。"
- "污染定理是 LUCID 的代数重新包装（theory-first 攻击原话），不是新理论。"
- "规则聚合（手写权重）相比 MemGuard/GovMem 没有本质创新。"

**能发表的概率 < 30%**，最可能结局 = 被拒或 workshop（ICLR MemAgents WS），无法进主会。

---

## 6. 风险（如实告知）

### 6.1 Novelty 风险（最大生存威胁）

- **Grounding Agent Memory（6 天前）**：正面碰撞且我们无 learned 差异化。
- **Calibration 不被认可为贡献**：evaluation-first 攻击已判 serious。
- **污染定理被认为是 LUCID 包装**：theory-first 攻击已判 serious。
- **组合 novelty 不足**：规则+校准+理论 的组合，审稿人可能认为每个单点都不够新，组合也不够。

### 6.2 实验风险

- **误差独立性若实测 ρ̂ 不够小**：推论钝化（与定稿相同）。
- **线性模型假设偏离**：理论界失效（与定稿相同）。
- **合成基准真实性质疑**：外部效度弱（与定稿相同，但现在外部基准只做一个子集更弱）。

### 6.3 用户约束

- **无训练资源**：4 张 4090 不足训 RL，learned 方向完全不可行。
- **数据集太大**：只做 LongMemEval-S 的 knowledge-update 子类（响应用户"数据集太大先只做一个"）。

---

## 7. 时间线（缩短到 8 周，去掉训练）

| 周 | 里程碑 | 关键路径 |
|---|---|---|
| W1 | 脚本先行数据管线 + 合成世界 | deepseek-flash 渲染 5K-10K 候选 |
| W2 | 全部基线跑通 | 全量存储+检索、LLM-judge、A-MAC、未校准规则 |
| W3 | 规则权重网格搜索 + 校准 | held-out 集上调权重、温度标定+CRC |
| W4 | 主表 + 消融 | FDR 覆盖曲线、分布移位 ECE |
| W5 | 污染压力测试 | 长时程四条曲线、理论拟合 |
| W6 | LongMemEval-S 子集（可选） | 若用户可下载，只跑 knowledge-update |
| W7 | 成稿 + arXiv | 初稿→review→定稿；**W7 末必须挂 arXiv** |
| W8 | 投稿 | ICML 2027 或 NeurIPS 2027（但预期被拒） |

**关键路径总时长**：8 周（比原 12 周缩短 4 周，因为去掉 W4-6 的反事实标签+门训练）。

---

## 8. 我的建议（如实告知）

你的约束"原则上不想做任何训练，4 张 4090 训练不了 RL"是合理的工程判断，但它让这个选题的 novelty **几乎全部坍塌**。

**最诚实的评估**：
- 此 training-free 方案能发 ICLR/ICML/NeurIPS 主会的概率 < 30%。
- 最可能结局 = 被拒，或只能发 workshop（ICLR MemAgents WS）。
- 唯一的对冲 = 污染定理（虽被攻击为"LUCID 包装"，但动力系统界仍独立）+ 实证污染曲线对照（LLM-judge 门平台化 vs 本方法收敛）。

**建议两个选项**：

**选项 A（现实路线）**：接受此方案，目标降级为 workshop 或 arXiv-only，8 周完成，作为技术报告而非主会论文。承认 novelty 不足，但工程上可跑通、有理论对比点、实证对照清晰。

**选项 B（放弃此选题）**：agent memory 的验证门控方向在无训练约束下 novelty 站不住（Grounding Agent Memory 6 天前已占，learned 是唯一差异化且已去除）。建议：①彻底换选题（非 agent memory 方向）；②或回到调研报告 00 的原主方案"条目级校准效用写入门"，但那也需要训练（反事实标签+效用估计器）。

我需要你确认：接受选项 A（training-free，目标降级为 workshop/arXiv），还是选项 B（换选题或回原主方案但接受需要训练）？
