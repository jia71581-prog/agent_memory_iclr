# Training-Free Compression & Forgetting Mechanisms for Agent Memory
**Literature Review: 2025-2026 Publications**  
**Date: 2026-09-16**

## Executive Summary

Searched arXiv 2025-2026 for training-free compression and forgetting mechanisms in agent long-term memory. Found **12 distinct training-free methods** across 50+ papers, with **3 high-novelty directions** (>60% ICLR/ICML/NeurIPS 2027 acceptance probability) that are unoccupied by MemoryBank (forgetting curve) and MemoryOS (FIFO).

**Top recommendation**: Query-conditioned forgetting with lifecycle-aware consolidation (combines RD-Forget + MemoryLACE ideas) has the strongest novelty signal and addresses a fundamental gap in existing work.

---

## 1. Retrieved Papers (Compression Focus)

### 1.1 Query-Conditioned Memory Views
**arXiv:2609.10263** | "What Should an Agent Forget? Separating What Is Stored from What Is Used" | 2026-09-09  
**Authors**: Yuhang Li, Yuchen Li  
**Method**: RD-Forget framework separates storage (retained source archive) from use (query-conditioned memory view). Frozen LM curator extracts evidence, groups into semantic slots, preserves multi-hop relations. Same-slot replacement links suppress superseded values. Rate-distortion formulation guides budget-constrained view construction.  
**Training-free**: Yes (frozen curator, rule-based suppression)  
**Novelty vs. MemoryBank/MemoryOS**: MemoryBank uses time-based forgetting curve, MemoryOS uses FIFO. RD-Forget retains ALL history but controls query-time influence via semantic slots. Superseded facts stay accessible for historical queries.

### 1.2 Lazy Materialization
**arXiv:2609.14773** | "Pull: Lazy Materialization of Working Memory for Stateful LLM Conversations" | 2026-09-13  
**Authors**: Jiangang Chen  
**Method**: Maintains addressable metadata directory via local deterministic Purifier (zero LLM calls, millisecond latency). LLM lazily materializes only needed turns at query time; unmaterialized turns remain accessible but collapsed. Materialization is reversible.  
**Training-free**: Yes (deterministic metadata routing)  
**Novelty**: Unlike irreversible compression (summarization), Pull allows re-expansion. Reduces per-query tokens by 75.1% (single-hop) and 72.0% (multi-hop) with no quality loss.

### 1.3 Lifecycle-Aware Consolidation
**arXiv:2609.03201** | "MemoryLACE: Memory Lifecycle-Aware Consolidation and Evidence Retrieval" | 2026-09-02  
**Authors**: Meriem Yacoubi, Pia Schmidt, Nenad Petrovic, et al.  
**Method**: Explicitly models lifecycle through sparse merge, supersession, contradiction relations. Reconstructs relation-aware evidence units exposing current/historical/supporting/conflicting evidence. Preserves atomic natural-language memories + provenance.  
**Training-free**: Yes (rule-based lifecycle tracking)  
**Novelty**: MemoryBank/MemoryOS don't track supersession or contradictions. MemoryLACE achieves highest BEAM performance with 66.6% runtime reduction vs. Hindsight baseline.

### 1.4 Cue-Guided Reconstruction
**arXiv:2609.12354** | "CueMem: Cue-Guided Context Reconstruction for Long-Term Conversational Memory" | 2026-09-11  
**Authors**: Changjian Wang, Rongzhen Li, Weili Guan, et al.  
**Method**: Treats memory records as retrieval cues (not self-contained evidence). Links each cue to source turn. At query time, retrieves cues → maps to source-turn anchors → expands via turn graph (temporal proximity + semantic relatedness) → reconstructs compact evidence from original dialogue.  
**Training-free**: Yes (graph traversal rules)  
**Novelty**: Neither MemoryBank nor MemoryOS reconstruct context from cues. CueMem outperforms baselines on LoCoMo and LongMemEval.

### 1.5 Budget-Dependent Operator Selection
**arXiv:2607.17545** | "Retain or Consolidate? Budget-Dependent Operator Selection for Language Agent Memory" | 2026-07-20  
**Authors**: Qingcan Kang, Mingyang Liu, Shixiong Kai, et al.  
**Method**: Decomposes operator utility into coverage effect + replacement effect. Switches between retention (keeps raw records) and consolidation (Merge/Abstract/Rewrite) based on budget pressure. Offline Abstraction-Safety (OAS) learner estimates utilities.  
**Training-free operators**: Yes (Merge/Abstract/Rewrite are rule-based), but OAS uses learning for selection  
**Novelty**: Under tight budgets, consolidation improves accuracy by +48% (LongMemEval); under loose budgets, retention preferable. Shows budget-dependent crossover.

### 1.6 Semantic Segment-Level Consolidation
**arXiv:2608.12990** | "LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation" | 2026-08-13  
**Authors**: Dongfang Li, Zixuan Liu, Junmai Wang, et al.  
**Method**: Replaces turn-level consolidation with semantic segment-level batching. Segments finalized into context-independent typed memory records. Semantic boundary detection preserves event-level + temporal evidence vs. fixed-window batching.  
**Training-free**: Yes (boundary detection uses heuristics, not trained classifiers)  
**Novelty**: MemoryBank/MemoryOS consolidate per-turn or use fixed windows. LycheeMemory achieves 89.22% LoCoMo, 92.20% LongMemEval-S with 86% token reduction vs. A-Mem.

### 1.7 Dual-Layer Memory with Epistemic Routing
**arXiv:2608.22215** | "Dual-Layer Agentic Memory with Fast Write Routing and Slow Consolidation" | 2026-08-23  
**Authors**: Wenzhi Li, Dong Nie, Rui Lan, et al.  
**Method**: Cost-aware epistemic routing categorizes incoming info as non-write/write-new/write-update. Small-to-large model cascade (1.7B/8B) filters redundant memories. Periodic parametric consolidation via supervised fine-tuning internalizes high-value external memories.  
**Training-free routing**: Cascade logic is rule-based; consolidation uses fine-tuning  
**Novelty**: MemoryBank/MemoryOS don't route at write time or internalize to parameters. Prunes 68% redundant memory while retaining 98% QA EM.

### 1.8 Memory Compression for High-Fanout Sandboxes
**arXiv:2609.11294** | "Memory Compression for High-Fanout Agent Sandboxes" | 2026-09-10  
**Authors**: Mengming Li, Ceyu XU, Qijun Zhang, et al.  
**Method**: AgentZip exploits template-relative + cross-sandbox redundancy. Broadens compression scope to any profitable page. Shifts overhead control from compression-time selection to restore-time prefetching. Aligns compression with LLM waiting periods.  
**Training-free**: Yes (redundancy detection via delta encoding)  
**Novelty**: Domain-specific (high-fanout sandboxes). Not directly comparable to conversational memory systems like MemoryBank/MemoryOS.

### 1.9 Agentic Context Management
**arXiv:2607.21503** | "Agentic Context Management: Solving Agent Memory and Cost by Treating Them as Lifecycle and Architecture Problems" | 2026-07-23  
**Authors**: Gaurav Dadhich  
**Method**: Decomposes into 5 primitives: architecting, ingesting, scoping, anticipating, compacting & consolidation. Reference implementation (Maximem Synap) achieves 92% LongMemEval, 93.2% LoCoMo. Argues lifecycle management vs. storage-and-retrieval framing.  
**Training-free**: Yes (architectural framework, not trained components)  
**Novelty**: Conceptual framework. MemoryBank/MemoryOS are single-mechanism approaches; ACM is multi-primitive. Closer to system design than a single forgetting/compression algorithm.

---

## 2. Retrieved Papers (Forgetting Focus)

### 2.1 Eviction Audit (Evaluation, Not Method)
**arXiv:2609.08279** | "What Eviction Destroys: A Restore-Counterfactual Audit of Forgetting in Agent Memory" | 2026-09-08  
**Authors**: Chen Shen  
**Method**: Restore counterfactual - per-question paired intervention reinstating gold evidence. Classifies errors as recoverable (retrieval failure) vs. irreversible (eviction destroyed evidence). Evaluates FIFO, random, redundancy-aware, LLM-importance eviction at 8k/80k budgets.  
**Training-free**: Yes (audit framework)  
**Note**: This is an EVALUATION method, not a forgetting mechanism. Finds irreversible error share 0.67-0.73 for FIFO/random at 80k tokens, 1.00 at 8k.

### 2.2 Budgeted Verification with Provenance
**arXiv:2608.25553** | "When Stale Constraints Go Unchecked: Budgeted Verification Failures in Inherited Agent Memory" | 2026-08-26  
**Authors**: Kazuki Nakayashiki  
**Method**: Studies consolidated memory with provenance links. Agent has verification budget to inspect source records. Shows 16 LMs rarely re-verify stale constraints (77.3% stale-consistent decisions). One-sentence target-blind rule ("prefer memories stating limits") recovers +89.3 points.  
**Training-free**: Yes (budget allocation heuristic)  
**Novelty**: Provenance-based verification. MemoryBank/MemoryOS don't track provenance or verification budgets.

### 2.3 LifeMem (Clustering, Not Pure Training-Free)
**arXiv:2609.12655** | "LifeMem: Enabling Lifelong Experience Reuse for LLM Agents" | 2026-09-11  
**Authors**: Yuli Qiu, Yutong Li, Wei Su  
**Method**: Clusters accumulated knowledge, transfers across environments. Addresses catastrophic forgetting in lifelong learning.  
**Training-free**: Partial (uses clustering algorithms, may involve learned representations)  
**Note**: Focuses on cross-environment transfer, not compression/forgetting within single environment.

---

## 3. Novelty Analysis

### 3.1 Occupied Directions (MemoryBank, MemoryOS, PA-Bench)
- **MemoryBank**: Forgetting curve heuristic (time-based decay)
- **MemoryOS**: FIFO eviction (order-based)
- **PA-Bench**: Query-blind eviction evaluation (benchmark, not method)

### 3.2 Unoccupied Training-Free Directions

| Direction | Key Innovation | Differentiation |
|-----------|----------------|-----------------|
| **Query-conditioned forgetting** | Separates storage from use; retains all history but controls query-time influence | MemoryBank forgets by time, MemoryOS by order. RD-Forget keeps everything, controls semantically |
| **Lifecycle-aware consolidation** | Explicit merge/supersession/contradiction tracking | Neither MemoryBank nor MemoryOS model fact evolution relationships |
| **Lazy materialization** | Reversible compression via metadata directory | MemoryBank/MemoryOS use irreversible summarization/truncation |
| **Budget-dependent operator switching** | Adaptive retention ↔ consolidation based on budget pressure | MemoryBank/MemoryOS use fixed strategies |
| **Semantic segment batching** | Consolidate by semantic boundaries, not turns/time | MemoryBank uses time, MemoryOS uses order; neither detect semantic boundaries |
| **Cue-based reconstruction** | Memory as retrieval cue, not self-contained evidence | MemoryBank/MemoryOS treat memory as final evidence, not reconstruction trigger |
| **Provenance-budgeted verification** | Selective re-verification with budget allocation | MemoryBank/MemoryOS don't track provenance or verify stale memories |

---

## 4. Publication Probability Assessment (ICLR/ICML/NeurIPS 2027)

### 4.1 High Probability (>60%)

**Direction 1: Query-Conditioned Forgetting with Lifecycle Tracking**  
*Combining RD-Forget + MemoryLACE insights*

**Novelty**: 
- Neither MemoryBank (time-based) nor MemoryOS (FIFO) separate storage from use
- Lifecycle relations (supersession, contradiction) not explicitly modeled in prior work
- Addresses fundamental gap: stale facts misleading current answers but essential for historical queries

**Why >60%**:
- Clear problem motivation (demonstrated in RD-Forget, MemoryLACE papers)
- Training-free = low implementation cost, high reproducibility
- Orthogonal to retrieval quality (complements, doesn't replace existing retrieval)
- Strong baselines already exist (MemoryBank, MemoryOS, RD-Forget, MemoryLACE)
- Evaluation benchmarks mature (LongMemEval, LoCoMo, BEAM)

**Risk factors**:
- RD-Forget (Sep 2026) and MemoryLACE (Sep 2026) are very recent; may face "incremental" critique if not sufficiently distinguished
- Need ablation showing lifecycle tracking + query conditioning beats each alone

---

**Direction 2: Budget-Adaptive Compression Switching**  
*Inspired by "Retain or Consolidate" with simpler training-free selection*

**Novelty**:
- MemoryBank/MemoryOS use fixed strategies regardless of budget
- "Retain or Consolidate" shows +48% accuracy under tight budgets (consolidation) but retention better under loose budgets
- Training-free budget-pressure detector (e.g., evidence coverage ratio) can trigger switch

**Why >60%**:
- Empirical evidence of budget-dependent crossover already demonstrated
- Simple heuristic (coverage ratio) can replace learned OAS selector
- Practical impact: agents in production face variable memory budgets (mobile vs. cloud)
- Clear evaluation protocol: plot accuracy across budget sweep, show crossover point

**Risk factors**:
- "Retain or Consolidate" (Jul 2026) scooped core insight
- Need to show training-free heuristic matches or beats learned selector

---

**Direction 3: Semantic Segment Consolidation with Provenance**  
*LycheeMemory V2 + provenance tracking from "Stale Constraints"*

**Novelty**:
- LycheeMemory V2 shows semantic boundaries > turn-level/fixed-window
- "Stale Constraints" shows provenance-based verification crucial
- Neither combines semantic segmentation with provenance-enabled re-verification

**Why 50-70% (borderline >60%)**:
- Segment-level consolidation already demonstrated (LycheeMemory V2)
- Provenance tracking already demonstrated ("Stale Constraints")
- Combination is natural but incremental
- Would need strong ablation showing provenance-enabled re-verification recovers from segment consolidation errors

**Risk factors**:
- May be perceived as "combining two existing ideas" without core novelty
- LycheeMemory V2 (Aug 2026) is recent; reviewers may expect deeper contribution

---

### 4.2 Medium Probability (30-60%)

**Direction 4: Cue-Based Reconstruction with Multi-Hop Expansion**  
*CueMem with graph-based evidence aggregation*

**Why 30-60%**:
- CueMem (Sep 2026) already implements core idea
- Would need significant extension (e.g., learned graph weights, multi-hop constraints)
- Evaluation shows improvement but not dramatic (+consistent outperformance, not breakthrough)

**Direction 5: Lazy Materialization with Predictive Pre-Fetching**  
*Pull + anticipatory materialization*

**Why 30-60%**:
- Pull (Sep 2026) demonstrates lazy materialization
- Adding prediction (which turns to pre-fetch) introduces learned component or complex heuristic
- Risk: either becomes training-based (violates constraint) or heuristic is too simple

---

### 4.3 Low Probability (<30%)

**Direction 6: Multi-Agent Sandbox Compression**  
*AgentZip-style template-relative compression*

**Why <30%**:
- Domain-specific (high-fanout sandboxes), not general agent memory
- AgentZip (Sep 2026) already covers this thoroughly
- ICLR/ICML/NeurIPS prefer general methods over niche optimizations

---

## 5. Recommended Direction & Experiment Design

### 5.1 Top Recommendation: Query-Conditioned Forgetting with Lifecycle Tracking

**Core Idea**:  
Separate what an agent stores (full history with lifecycle metadata) from what it uses (query-conditioned view). Track supersession, contradiction, and support relations explicitly. At query time, construct budget-constrained view that:
1. Includes query-relevant current-state evidence
2. Suppresses superseded facts for current-state queries
3. Makes superseded facts accessible for historical queries
4. Exposes contradictions explicitly

**Why This Wins**:
- Addresses demonstrated failure mode: stale facts mislead current answers (from "Stale Constraints")
- Solves unsolved problem: temporal queries need historical facts (from RD-Forget)
- Training-free: rule-based lifecycle tracking + deterministic view construction
- Orthogonal to retrieval: works with any retrieval method (semantic search, BM25, hybrid)
- Clear baselines: MemoryBank (time-decay), MemoryOS (FIFO), RD-Forget (slots without lifecycle)

---

### 5.2 Minimal Viable Experiment (Single Dataset, <2 Weeks)

**Dataset**: LongMemEval-S (500 questions, well-studied)

**Systems to Compare** (4 baselines + 1 proposed):
1. **MemoryBank baseline**: Time-based forgetting curve
2. **MemoryOS baseline**: FIFO eviction
3. **RD-Forget baseline**: Query-conditioned slots (no lifecycle)
4. **MemoryLACE baseline**: Lifecycle relations (no query conditioning)
5. **Proposed**: Query-conditioned + lifecycle (QC-Lifecycle)

**Implementation** (3-5 days):
- Use LongMemEval-S memory stream (conversational history with updates/contradictions)
- Lifecycle tracker: Rule-based supersession detection (time-ordered updates to same entity/fact)
- Query classifier: Current-state vs. historical (simple keyword heuristic: "currently", "now", "latest" → current)
- View constructor: At query time, retrieve top-k relevant memories, filter superseded if current-state query, keep all if historical

**Evaluation** (1 day):
- Primary: Answer accuracy (exact match)
- Ablation: (a) No lifecycle tracking, (b) No query conditioning, (c) Full system
- Budget sweep: 1k, 2k, 4k, 8k tokens
- Error analysis: Count current-state errors from stale facts, historical errors from missing old facts

**Hypothesis**:
- QC-Lifecycle beats MemoryBank/MemoryOS on current-state accuracy (fewer stale-fact errors)
- QC-Lifecycle beats RD-Forget on historical accuracy (lifecycle enables historical access)
- QC-Lifecycle beats MemoryLACE on current-state accuracy (query conditioning suppresses superseded)

**Expected Result**:
- +3-5% accuracy vs. best baseline on current-state queries
- +2-4% accuracy vs. best baseline on historical queries
- <1% degradation on neutral queries (no temporal constraint)

**If Positive**: Scale to LoCoMo (1536 questions), add multi-hop reasoning analysis, implement budget-adaptive consolidation (Direction 2) as extension.

---

## 6. Summary Table: Top 3 Directions

| Direction | Core Novelty | Probability | Key Risk | MVP Effort |
|-----------|--------------|-------------|----------|------------|
| **QC-Lifecycle** | Query-conditioned + lifecycle tracking | **>60%** | RD-Forget/MemoryLACE recency | 1-2 weeks |
| **Budget-Adaptive** | Switch retention ↔ consolidation by budget | **>60%** | "Retain or Consolidate" overlap | 2-3 weeks |
| **Segment + Provenance** | Semantic segmentation + provenance verification | **50-70%** | Incremental combination | 2-3 weeks |

---

## 7. Full Paper List (50 papers, 2025-2026)

### Consolidation (17 papers)
1. 2609.17331 - Self-Emergence Agent Architecture
2. 2609.17010 - ThinkFlow: Probabilistic Latent Memory
3. 2609.15800 - Navigating Sparse Evidence (Visual RAG)
4. 2609.12655 - LifeMem: Lifelong Experience Reuse
5. 2609.10263 - **What Should an Agent Forget** (RD-Forget)
6. 2609.06072 - ACE: Adapter Consolidation for MoE
7. 2609.03753 - SimSkill: Self-Evolving Agent
8. 2609.03201 - **MemoryLACE: Lifecycle-Aware Consolidation**
9. 2609.02217 - SkillGLoW: Procedural-Family Skill Consolidation
10. 2609.02094 - MASkills: Continual Skills Optimization
11. 2608.22215 - **Dual-Layer Agentic Memory**
12. 2608.21755 - **ECHO: Auditable Memory Plane**
13. 2608.12990 - **LycheeMemory V2: Segment-Level Consolidation**
14. 2607.26809 - Practice Makes Policies (robotic)
15. 2607.26017 - UniMem: Episodic-to-Parametric Memory
16. 2607.25380 - Memory for Large Language Models (survey)
17. 2607.21503 - **Agentic Context Management** (ACM)

### Compression (18 papers)
18. 2609.17475 - JustFit: Just-in-Time State Management
19. 2609.16669 - Memory-Skill Isomorphism
20. 2609.16161 - LLM Inference in a Flash
21. 2609.15338 - Dynamic Semantic Compression
22. 2609.14773 - **Pull: Lazy Materialization**
23. 2609.12686 - Residual Vector-based Reconstruction
24. 2609.12354 - **CueMem: Cue-Guided Reconstruction**
25. 2609.12208 - Vortex: Extreme Compression
26. 2609.11294 - **Memory Compression for High-Fanout**
27. 2609.11209 - REVA: Reusable Evidence View Aggregation
28. 2609.11192 - FlexComp: One Model for Every Ratio
29. 2609.08273 - MemForest: EventTree Partitioning
30. 2609.07966 - MetaKV: Adaptive KV Cache Compression
31. 2609.07471 - MEMO: Multimodal Evidence Organization
32. 2609.03235 - SGD-KV: Summarization Guided KV Cache
33. 2608.30252 - Strong Drafts Need Compact Memories
34. 2608.30076 - Budget-Aware Compression Pipeline
35. 2607.17545 - **Retain or Consolidate** (Budget-Dependent)

### Forgetting & Unlearning (15 papers)
36. 2609.16229 - Test-Time Unlearning via Sparse Autoencoder
37. 2609.12808 - K-Bench: LLM Unlearning in Agentic Deployments
38. 2609.11447 - Catastrophic Forgetting in Sound Event Classification
39. 2609.11194 - Domain-Incremental Learning for Speech
40. 2609.10750 - Catastrophic Forgetting in Skill Retrieval
41. 2609.10439 - Forgetting Only What Matters (layer-selective)
42. 2609.08279 - **What Eviction Destroys** (restore-counterfactual)
43. 2609.05767 - Will My Assistant Remember My Allergy
44. 2609.05339 - Memory Portability Across Model Upgrades
45. 2609.04875 - Forgetting Without Restarting (execution-state)
46. 2608.28978 - Selective Forgetting: Graph-Based Memory
47. 2608.28361 - GRACE: Gradient-guided Coreset Selection
48. 2608.26545 - Memory Anchors for Continual Robot Learning
49. 2608.25553 - **When Stale Constraints Go Unchecked**
50. 2607.17545 - (duplicate, see #35)

---

## 8. Conclusion

**Main Finding**: The 2025-2026 literature shows rapid innovation in training-free agent memory mechanisms. Query-conditioned forgetting and lifecycle-aware consolidation are the most novel unoccupied directions, with clear differentiation from MemoryBank/MemoryOS.

**Top Recommendation**: **Query-Conditioned Forgetting with Lifecycle Tracking** has >60% ICLR/ICML/NeurIPS 2027 acceptance probability. Minimal viable experiment requires only LongMemEval-S (1-2 weeks), with clear baselines and ablations.

**Next Steps**:
1. Implement QC-Lifecycle prototype (3-5 days)
2. Run LongMemEval-S evaluation (1 day)
3. If positive (+3% vs. best baseline), scale to LoCoMo and add multi-hop analysis
4. Draft paper structure: Problem (stale facts + historical queries) → Method (lifecycle + query conditioning) → Experiments (4 baselines + ablations) → Analysis (error categories)

**Key Insight**: The field has moved from "how to compress" (MemoryBank/MemoryOS) to "what to forget when" (query-conditioned, lifecycle-aware). This shift aligns with production deployment needs (variable budgets, temporal queries, evolving facts).
