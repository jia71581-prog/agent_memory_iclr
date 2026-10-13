# Agent Memory Representation Learning: Training-Free Approaches
**Literature Review: 2025-2026**  
**Date: 2026-09-16**

## Executive Summary

Identified 15 training-free representation learning approaches for agent memory published in 2025-2026. Three high-potential directions emerge: (1) **probabilistic latent memory** that encodes behavioral patterns beyond text, (2) **hierarchical consolidation** with lifecycle-aware temporal dynamics, and (3) **atomic memory with semantic clustering**. The strongest novelty candidate is **probabilistic latent representations** (ThinkFlow), with estimated 30-60% acceptance probability at ICLR/ICML/NeurIPS 2027.

---

## 1. Retrieved Papers (2025-2026)

### A. Probabilistic & Latent Representations

**[1] ThinkFlow: Self-Evolving Probabilistic Latent Memory (arXiv:2609.17010v1, 2026-09-15)**
- **Venue**: Not yet published
- **Description**: Uses probabilistic latent space to encode behavioral patterns and emotional shifts that explicit textual memory loses. Self-evolving post-deployment.
- **Training-free**: ✓ (uses pre-trained embeddings, no RL/neural training)
- **Non-graph**: ✓ (latent vector space, not graph structure)

**[2] Self-Emergence Agent Architecture (arXiv:2609.17331v1, 2026-09-15)**
- **Venue**: Not yet published
- **Description**: Behavior-Inertia HMM with social-contrastive self-modeling. Uses HMM for personality consistency.
- **Training-free**: ✓ (HMM inference only)
- **Non-graph**: ✓ (probabilistic state space)

### B. Atomic Memory with Semantic Organization

**[3] ROAM: Robust Organization of Atomic Memories (arXiv:2609.09778v1, 2026-09-09)**
- **Venue**: Not yet published
- **Description**: Fine-grained atomic memories organized through semantic relations (not graph edges). Handles redundancy and conflicts.
- **Training-free**: ✓ (semantic similarity via embeddings)
- **Non-graph**: ✓ (semantic relations ≠ knowledge graph)

**[4] Compact-Memory LLM Agents (arXiv:2609.04915v1, 2026-09-04)**
- **Venue**: Not yet published
- **Description**: Online max-member clustering for atomic memory with atom-aware packing. Quality-token trade-off optimization.
- **Training-free**: ✓ (online clustering algorithm)
- **Non-graph**: ✓ (clustering, not graph)

**[5] AtomRec: Evolving Atomic Memory (arXiv:2609.04882v1, 2026-09-04)**
- **Venue**: Not yet published
- **Description**: Atomic memory for recommendation with evolving semantic organization (vs coarse summaries + collaborative graphs).
- **Training-free**: ✓ (embedding-based organization)
- **Non-graph**: ✓ (atomic units with semantic links)

### C. Memory Consolidation & Lifecycle Management

**[6] MemoryLACE: Lifecycle-Aware Consolidation (arXiv:2609.03201v1, 2026-09-02)**
- **Venue**: Not yet published
- **Description**: Distinguishes repeated evidence, historical states, updates, contradictions through lifecycle tracking. Explicit temporal relationships.
- **Training-free**: ✓ (rule-based consolidation)
- **Non-graph**: ✓ (lifecycle metadata, not graph structure)

**[7] CueMem: Cue-Guided Context Reconstruction (arXiv:2609.12354v1, 2026-09-11)**
- **Venue**: Not yet published
- **Description**: Cue-based reconstruction motivated by cognitive reconsolidation. Avoids full-history cost and compression loss.
- **Training-free**: ✓ (retrieval + reconstruction via LLM, no training)
- **Non-graph**: ✓ (cue-memory mapping)

**[8] Dual-Layer Agentic Memory (arXiv:2608.22215v2, 2026-08-23)**
- **Venue**: Not yet published
- **Description**: Fast write routing + slow consolidation. Two-tier temporal dynamics.
- **Training-free**: ✓ (routing + consolidation algorithms)
- **Non-graph**: ✓ (dual-layer hierarchy)

**[9] ECHO: Cognitively Inspired Memory Plane (arXiv:2608.21755v1, 2026-08-22)**
- **Venue**: Not yet published
- **Description**: Episodic encoding, consolidation, contextual reinstatement inspired by cognitive neuroscience. Auditable provenance.
- **Training-free**: ✓ (cognitive-inspired algorithms)
- **Non-graph**: ✓ (episodic memory model)

**[10] LycheeMemory V2 (arXiv:2608.12990v1, 2026-08-13)**
- **Venue**: Not yet published
- **Description**: Semantic segment-level consolidation (vs eager per-interaction consolidation). Lazy consolidation reduces cost.
- **Training-free**: ✓ (segment extraction + merging)
- **Non-graph**: ✓ (segment hierarchy)

**[11] Fortunate Recall: Ontology-Driven Lifecycle (arXiv:2609.10413v1, 2026-09-09)**
- **Venue**: Not yet published
- **Description**: Ontology-driven memory lifecycle management. Behavioral-type-conditioned persistence rates.
- **Training-free**: ✓ (ontology rules)
- **Non-graph**: Partial (ontology = lightweight semantic schema, not full graph)

### D. Hierarchical & Multi-Level Structures

**[12] CoMem: Collective-Individual Memory Synergy (arXiv:2609.15009v1, 2026-09-14)**
- **Venue**: Not yet published
- **Description**: Hierarchical collective vs individual memory for multi-agent systems. Structured vs flat memory.
- **Training-free**: ✓ (hierarchical organization)
- **Non-graph**: ✓ (two-level hierarchy)

**[13] MemForest: EventTree Partitioning (arXiv:2609.08273v1, 2026-09-08)**
- **Venue**: Not yet published
- **Description**: EventTree partitioning + progressive merging. Tree-based (not graph-based) organization.
- **Training-free**: ✓ (tree partitioning algorithm)
- **Non-graph**: ✓ (tree, not graph)

**[14] MEMO: Multimodal Evidence Memory (arXiv:2609.07471v1, 2026-09-07)**
- **Venue**: Not yet published
- **Description**: Hierarchical organization for multimodal evidence (text, image, video). Addresses trajectory accumulation vs context capacity tension.
- **Training-free**: ✓ (hierarchical indexing)
- **Non-graph**: ✓ (hierarchical multimodal index)

### E. Specialized Retrieval & Forgetting

**[15] RD-Forget: Separating Storage from Use (arXiv:2609.10263v1, 2026-09-09)**
- **Venue**: Not yet published
- **Description**: Training-free framework that separates what is stored (for historical queries) from what is retrieved (for current-state answers).
- **Training-free**: ✓ (retrieval filtering)
- **Non-graph**: ✓ (temporal validity metadata)

**[16] MemFuse: Multi-Source Memory Fusion (arXiv:2608.18704v1, 2026-08-19)**
- **Venue**: Not yet published
- **Description**: Fusion from fragmented observations across applications and devices. Cross-source consolidation.
- **Training-free**: ✓ (fusion algorithm)
- **Non-graph**: ✓ (multi-source index)

---

## 2. Novelty Analysis (vs HippoRAG & G-Memory)

### Already-Occupied Territory (EXCLUDED)
- **HippoRAG (NeurIPS 2024)**: Hippocampus-inspired graph memory with personalized PageRank retrieval
- **HippoRAG 2 (ICML 2025)**: Improved graph indexing with entity linking
- **G-Memory (NeurIPS 2025)**: Graph memory structure for multi-agent collaboration

All three use **explicit graph structures** (nodes = entities/memories, edges = relations).

### Distinct Non-Graph Approaches

| Paper | Key Differentiation from Graph Methods | Novelty Level |
|-------|----------------------------------------|---------------|
| **ThinkFlow** | Probabilistic latent space encodes implicit behavioral patterns text cannot capture | **High** |
| ROAM | Atomic granularity + semantic relations (not graph edges) | Medium |
| Compact-Memory | Online clustering optimizes quality-token trade-off | Medium |
| MemoryLACE | Lifecycle metadata (repeated/historical/updated/contradicted) | Medium-High |
| CueMem | Cue-based reconstruction (cognitive reconsolidation) | Medium |
| Dual-Layer | Fast-slow consolidation mimics System 1/2 | Medium |
| ECHO | Episodic encoding + contextual reinstatement (neuroscience) | Medium |
| MemForest | Tree partitioning (EventTree ≠ graph) | Low-Medium |
| CoMem | Collective-individual hierarchy for multi-agent | Low-Medium |

---

## 3. Top Candidate: Probabilistic Latent Memory (ThinkFlow)

### Core Innovation
**Problem**: Explicit textual memory suffers information bottleneck—loses behavioral patterns, emotional dynamics, implicit preferences.  
**Solution**: Encode memories in probabilistic latent space where:
1. Distributions capture uncertainty and soft patterns
2. Self-evolution updates representations without retraining
3. Behavioral inference operates over latent variables (not text retrieval)

### Why It's Different from Graphs
- **HippoRAG/G-Memory**: Explicit symbolic structures (entity nodes + relation edges)
- **ThinkFlow**: Continuous latent representations (probabilistic distributions)

The distinction mirrors **symbolic AI vs neural representations**—graphs are discrete and explicit, latent memory is continuous and implicit.

### Theoretical Grounding
- Draws from cognitive science: episodic memory encoding is not purely symbolic
- Probabilistic modeling handles uncertainty in long-term interactions
- Self-evolution mechanism addresses deployment-time adaptation (no retraining)

### Weaknesses
1. **Interpretability**: Latent representations harder to audit than graph structures
2. **Provenance**: Harder to trace "why" a memory influenced a decision
3. **Compositionality**: Graphs excel at multi-hop reasoning; latent spaces may not

### Related Work Gaps
- **MemoryStream (Park et al., 2023)**: Textual reflection hierarchy
- **HippoRAG (2024)**: Graph with retrieval
- **Vector RAG**: Flat embedding retrieval

None encode **probabilistic behavioral distributions** in latent space.

---

## 4. Publication Probability Assessment

### ThinkFlow (Probabilistic Latent Memory)
**Estimated Probability: 30-60%**

**Strengths (+):**
- Novel representation (latent vs graph vs text)
- Addresses real bottleneck (behavioral patterns lost in text)
- Self-evolution is timely (agents need post-deployment adaptation)
- Strong theoretical grounding (cognitive science + probabilistic modeling)

**Risks (-):**
- Empirical validation: needs strong baselines across multiple benchmarks
- MemoryBench, MemoryArena require explicit recall—may favor symbolic methods
- "Probabilistic latent memory" is abstract; needs concrete use cases
- Interpretability concerns for real deployments

**Comparable Acceptances:**
- NeurIPS 2024: ~26% acceptance rate
- ICML 2025: ~28% acceptance rate
- ICLR 2027: Estimated ~27% (if historical trend holds)

**Conditional on**: (1) Strong empirical results showing latent encoding improves over text/graph on behavior-dependent tasks, (2) Ablations isolating latent representation from self-evolution, (3) At least one real-world deployment case study.

### Other Candidates
- **MemoryLACE** (Lifecycle-Aware): 30-50% (lifecycle management is incremental but valuable)
- **ROAM** (Atomic + Semantic): 20-40% (atomic memory is known; semantic relations are incremental)
- **CueMem** (Cue-Guided): 20-35% (cognitive inspiration is good but similar to prior work)

---

## 5. Minimal Viable Experiment (ThinkFlow-Style Approach)

### Constraint: "Dataset too large, do one experiment"

**Setup:**
- **Task**: Long-term conversational agent (100+ turns)
- **Dataset**: Persona-Chat extended sessions OR custom collected data (50 sessions × 100 turns)
- **Metric**: Behavioral consistency + preference recall

**Compared Systems:**
1. **Baseline 1**: Flat vector RAG (text embeddings)
2. **Baseline 2**: HippoRAG-style graph memory
3. **Proposed**: Probabilistic latent memory (encode user behavior as latent distribution)

**Implementation:**
- Use pre-trained sentence encoder (e.g., MPNet)
- Encode each memory as: `p(z | memory_text)` where z ~ N(μ, Σ)
- Update distribution online: Bayesian update or exponential moving average
- Retrieval: KL divergence in latent space (vs cosine similarity in text space)

**Evaluation:**
- **Behavioral consistency**: Agent responses align with user's implicit preferences (annotated)
- **Preference recall**: Multi-choice questions about user's preferences (50 questions per session)
- **Efficiency**: Memory size vs performance trade-off

**Why This Works:**
- Single dataset = 50 sessions × 100 turns = 5000 interactions (tractable)
- Behavioral annotation can be crowdsourced (binary: consistent/inconsistent)
- Latent space encodes soft patterns text misses (hypothesis)

**Expected Outcome:**
If latent memory outperforms text RAG on behavioral metrics by >10% AND shows better quality-size trade-off, the novelty claim is validated.

---

## 6. Recommended Direction

**Primary Recommendation: Probabilistic Latent Memory (ThinkFlow-style)**

**Rationale:**
1. **Novelty**: Clear differentiation from graph-based (HippoRAG/G-Memory) and text-based (RAG) approaches
2. **Timeliness**: Self-evolution and behavioral encoding are hot topics in 2026
3. **Feasibility**: Can implement with existing embeddings (training-free constraint satisfied)
4. **Story**: "Text loses behavioral patterns → latent distributions capture them" is compelling

**Alternative (if latent is too abstract):**
**MemoryLACE** (Lifecycle-Aware Consolidation)—more concrete, easier empirical validation, addresses real pain point (stale memories), but lower ceiling on novelty.

---

## 7. Search Methodology Notes

**APIs Used:**
- arXiv API (direct connection, --noproxy)
- OpenAlex API (attempted, no results)

**Search Terms:**
- "agent memory representation"
- "semantic memory agent"
- "latent memory agent"
- "episodic semantic memory"
- "hierarchical memory agent"
- "consolidation memory agent"
- "probabilistic latent memory agent"
- "clustering memory retrieval"
- "atomic memory agent"

**Date Range:** 2025-01-01 to 2026-09-16

**Total Papers Reviewed:** ~50 abstracts, 16 selected as training-free + non-graph

---

## Appendix: Papers Excluded (Graph-Based or Requires Training)

- **Agent Zero Memory (2608.29606)**: Includes knowledge graph (multi-structure = graph + vector + fact store)
- **Diachronic Hypergraphs (2608.29678)**: Hypergraph structure (excluded: graph-based)
- **GLAM (2609.14561)**: Latent world model for navigation (requires training world model)
- Various papers on KV cache optimization (not memory representation)

---

**Conclusion:** Probabilistic latent memory is the strongest training-free, non-graph novelty candidate with 30-60% estimated acceptance probability at top-tier ML venues, contingent on strong empirical validation showing behavioral encoding advantages over symbolic and text-based baselines.
