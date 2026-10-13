> **状态(2026-09-21):本文件已被 `paper/main.tex` 取代,不要再直接引用本文件的正文数字。**
> 三张量化表(主表 / McNemar / 分题型 / 干预 A/B)已逐位复现无误,但正文有 14 处数字经产物复算不成立,已在 tex 版修正:
> §6.1 拒答计数(26/4/2/18 → 51/19/31/29)、§6.1 preference 子类**符号反了**(+3.3 → −3.3)、
> §5.5 拒答率三组数(10/0、5/4、21/30 → 26/11、28/31、28/66)且"量表是机制"被自家门控臂证伪、
> §3.4 答案长度方向反了(不是长 1/3 而是比 full 短 7–10%)、§5.4 错误类型学实际算在提示词修复**前**的臂上(67/91)、
> §5.1 "净差两题"与同节 Δ=0.0000 自相矛盾、§3.2 judge 一致性 900 的臂集合未声明(七臂口径=604)。
> 另新增:v6 组合并非可加(单臂共破 7 题,组合破 11 题),v6 对 KU 回退一题。
> 复算依据 = `results/*.jsonl` 原始行 + `v6_gates.json`,详见 tex 版对应小节。

# What Makes a Memory View Useful? A Controlled Study of Deterministic Lifecycle Views for Long-Term Conversational Memory

**Yuhang Jia**

Department of Mathematical Sciences, Tsinghua University
jia-yh22@mails.tsinghua.edu.cn

September 19, 2026

---

## Abstract

Long-term conversational agents must decide not only what to store but what to show the model at query time. Recent work frames this as *query-conditioned forgetting*: track the lifecycle of each fact (which values superseded which) and suppress or expose the history depending on what the query asks for. We set out to study that mechanism on LongMemEval-S and report a negative result at its core, together with a positive one that displaced it.

We build a fully training-free pipeline: a frozen LLM extracts atomic (entity, attribute, value) triples per session, deterministic rules assemble them into per-attribute chains labelled with supersession / contradiction / co-occurrence relations, and a rule-based classifier routes each query to a current-state or full-history rendering of those chains. On the full 500-question benchmark under a single shared answer prompt and matched context budgets, **the query-conditioning mechanism contributes exactly nothing**: ablating it changes accuracy by 0.0000 (12 wins, 12 losses, McNemar p = 1.00), and on the 456 questions where the two renderings genuinely differ in text the accuracy difference is likewise exactly zero. The mechanism is inert here because the two renderings carry nearly identical answer content and because 425 of 500 questions carry no lexical tense cue at all.

What does work is the view as an object, plus three context-construction interventions that we derived by reading every residual error. Relevance-ordered view truncation and role-quota retrieval lift accuracy from 0.870 to 0.898; a deterministic quantity ledger, a temporal scaffold, and a narrow-and-deep rendering for advice-shaped questions lift it to **0.922**, against 0.838 for a no-forgetting full-memory baseline (+8.4 points, b = 53, c = 11, p < 10⁻⁵), 0.828 for a budget-matched retrieval-only control (+9.4 points, p < 10⁻⁵), and 0.894 for our reimplementation of a concurrent slot-supersession system (+2.8 points, p = 0.038). Every arm runs on all 500 questions with one fixed judge, and every intervention is budget-neutral to within 0.1%.

Two methodological findings generalize beyond this system. First, an uncontrolled difference in system-prompt wording between our baseline runner and our method runner manufactured 3.1 points of illusory gain, by making the baseline decline to answer on 26 of 470 answerable questions versus 4 for the method. Second, and more consequential for how such systems are developed: across 68 offline-validated claims about candidate improvements, 56 failed adversarial verification, and the dominant failure mode was that the offline proxy — gold-string coverage, evidence-set completeness, derivability — reproduced arithmetically while failing to predict answer correctness. Two interventions that offline analysis rejected turned out to be the single largest real gain we measured. We conclude that in this regime only paired online measurement licenses a design decision.

---

## 1. Introduction

An agent that talks to the same user for months accumulates a conversational history far larger than any context window. The standard response is a memory system: store the history in some structured form, retrieve a subset at query time, and answer from that subset. Most of the design effort in recent systems goes into *what to store and what to drop* — decay schedules, capacity eviction, consolidation, summarization.

A newer line of work argues the interesting decision is downstream of storage. Because a fact's value changes over time ("I moved to Berlin" after "I live in Munich"), the memory holds a *lifecycle*: a chain of values with supersession relations between them. Whether the old value should be shown depends on the query. Asked "where do I live?", the agent should suppress the superseded value; asked "where did I live before?", it must expose it. Two concurrent preprints make this separation explicit — one separating the retained archive from the query-conditioned view, the other modelling merge/update/contradiction relations across memory items — and both report substantial gains from lifecycle awareness.

We set out to study that mechanism in isolation: to make the query-conditioning decision explicit and auditable (a rule table rather than an LLM call), to measure the decision's own accuracy, and to quantify what misrouting costs. The study produced a clean negative result. Under a controlled comparison the mechanism contributes nothing measurable on this benchmark, and we can say precisely why. Pursuing the reason led to a different and effective design, and to a methodological lesson about how such designs should be validated.

Our contributions:

1. **A controlled null result for query-conditioned lifecycle views** (§5.1). With storage, retrieval, budget, answer model, prompt, and judge all held fixed, removing query conditioning changes accuracy by 0.0000 on 500 questions. We localise the cause: the current-state and full-history renderings carry the gold answer in 186 and 188 of 500 questions respectively, overlapping in all but 10, so the routing decision can matter on at most 2% of the benchmark; and 85% of questions carry no lexical tense cue, defaulting to a single rendering regardless.

2. **A deterministic lifecycle view that does help, and an attribution control that isolates it** (§5.2). At an identical 20-turn retrieval budget, adding the view raises accuracy from 0.828 to 0.898 (p < 10⁻⁵). The control differs from the method only in the presence of the view, so the gain cannot be attributed to a retrieval-configuration difference — a confound we found in our own earlier comparison.

3. **Three error-derived context interventions, each budget-neutral** (§4.3, §5.3): a quantity ledger that force-renders every quantity-bearing fact for aggregation questions, a temporal scaffold that treats dates as retrieval keys and pre-computes offsets, and a narrow-and-deep rendering for advice-shaped questions. Together they take accuracy from 0.898 to 0.922 and repair the one subtype where our method had been *worse* than the no-forgetting baseline.

4. **Two methodological findings** (§6). A system-prompt wording asymmetry between runners manufactured 3.1 points of illusory gain through differential abstention. And offline proxy metrics systematically failed to predict answer correctness: 56 of 68 offline-validated improvement claims did not survive adversarial recomputation, while two claims that offline analysis rejected produced the largest measured gain. We report the audit trail because we believe the failure mode is general to memory-system development, where full online evaluation is expensive and proxies are tempting.

We do not claim state of the art. Our judge and answer model differ from those used in published LongMemEval-S numbers, so our absolute scores are not comparable to them (§3.4); every comparison in this paper is internal, paired, and run under one fixed configuration.

---

## 2. Related Work

**Memory management by storage policy.** MemoryBank-style time decay and MemoryOS-style capacity eviction decide what to keep. We reimplement both as baselines. Under our configuration decay scores 0.798 and FIFO eviction 0.330 on 500 questions; the FIFO collapse is informative rather than a strawman result, since it shows how much of this benchmark depends on old sessions that a recency policy discards.

**Lifecycle-aware memory.** Two concurrent preprints (both arXiv v1, September 2026, no venue at the time of writing; we treat them as concurrent work) are the direct antecedents of our study. The first separates a retained source archive from a query-conditioned view, links same-slot replacements so superseded values are suppressed in current-state contexts, and lets an intent signal make early evidence available again. The second models merge / update / contradiction / no-action relations between atomic memories with provenance, and rebuilds relation-aware evidence units at retrieval time, exposing current, historical, supporting and conflicting evidence to the reader.

Our positioning relative to them is narrower than we originally planned, and we state the retraction explicitly. The supersession-versus-contradiction distinction is not ours to claim: the second system already labels four relation types via LLM judgement. The idea of re-admitting old values under a historical intent is not ours either: the first system already has an intent switch. What remains ours is (i) rule-based, auditable relation labelling and query routing, where both antecedents use LLM calls, (ii) an explicit multi-valued-attribute vocabulary that neither has, and (iii) the measurement contribution — evaluating the routing decision itself, and reporting that on this benchmark it does not matter.

**Reimplementation as a baseline.** We reimplement the first system's design as our `RD-Forget-style` arm (§3.3): slot chains where only the active value is rendered and superseded values are excluded by default, a binary historical-intent switch, lexical fallback retrieval, and a single fixed view budget. It is a strong baseline, scoring 0.894 against the 0.838 full-memory baseline (p = 0.0002), and our method beats it by 2.8 points (p = 0.038). Reimplementing it also surfaced a protocol weakness we report rather than exploit quietly: its intent function combines lexical cues with the benchmark's gold task-type label. In our reimplementation that label decides 189 of the 223 questions routed to historical — 85% of the routing decisions come from metadata unavailable in deployment.

**Related conditioning axes.** Task-conditioned memory views and systems that map a question to a temporal intent before executing deterministically both predate our query-shape conditioning; we cite them as precedent for the conditioning idea rather than claiming it.

---

## 3. Experimental Setup

### 3.1 Benchmark

LongMemEval-S, **cleaned** variant (the original is deprecated; noisy sessions there affect answer correctness). 500 questions over six capability types: single-session-user (70), single-session-assistant (56), single-session-preference (30), multi-session (133), knowledge-update (78), temporal-reasoning (133). Thirty questions (`_abs` suffix) have unanswerable golds and are reported separately, since answering them correctly means declining. Haystacks total 23,867 session instances, of which 18,464 are unique by content — we exploit this for caching (§4.1). Each question carries a timestamp, and every session is dated to the minute, which makes the relation rules in §4.2 a total order rather than a partial one.

We report every arm on all 500 questions. No subsampling: a 60-question stratified sample was used during development only, and we treat it as methodologically contaminated because it drove design decisions.

### 3.2 Models and judging

Answer model: `deepseek-flash`, 6144 max tokens, reasoning enabled, identical across all arms. Extraction: `deepseek-flash` with reasoning disabled (§4.1). Judge: `deepseek-v4-pro` with one fixed prompt, temperature 0, single vote. Judge and answerer are deliberately different models, so no arm is graded by its own family.

Judge determinism was checked: across all arm pairs there are 900 question instances where two arms produced normalized-identical predictions, and the judge returned the same verdict on all 900. This bounds judge stochastic noise near zero. It says nothing about judge *validity*, which we discuss in §3.4.

### 3.3 Arms

All seven arms share one answer model, one judge, one answer prompt, and matched context budgets. They differ only in what context they build.

| Arm | Context construction |
|---|---|
| `full` | No forgetting. BM25 top-30 turns over all sessions. |
| `decay` | MemoryBank-style: sessions kept by exponential recency (30-day half-life, 0.05 threshold), then BM25 top-30. |
| `fifo` | MemoryOS-style: most recent 15 sessions only, then BM25 top-30. |
| `rdforget` | Reimplementation of the concurrent slot-supersession system (§2): active-value-only slot view, superseded values excluded unless a binary intent switch fires, plus BM25 top-20 turns. |
| `no_lifecycle` | **Attribution control.** BM25 top-20 turns, no view. Identical retrieval configuration to our method, minus the view. |
| `no_qc` | **Core ablation.** Our method with the view always rendered in full-history mode, i.e. no query conditioning. |
| `view_only` | Our method's view with no raw turns, measuring view self-sufficiency. |

Our method appears in three versions: **v4** (view + BM25 top-20), **v5** (v4 + relevance-ordered view truncation + role-quota retrieval), **v6** (v5 + the three interventions of §4.3).

The `no_lifecycle` control exists because our first comparison was confounded: the method used 20 retrieved turns plus a view while `full` used 30 turns, so a method-versus-`full` gap mixes a view effect with a retrieval-budget effect. The control fixes the budget at the method's own 20 turns.

### 3.4 What our numbers can and cannot be compared to

Published LongMemEval-S results use GPT-4o-mini with the official per-type judge prompts; we have access to a DeepSeek endpoint only. Our absolute scores are therefore **not** comparable to published numbers, and we make no such comparison. Third-party reproductions of the same systems on this benchmark differ by more than 20 percentage points, so cross-paper absolute numbers are of limited value regardless. We also report token-F1 alongside accuracy for protocol compliance, but we argue against reading it as corroboration: it is length-confounded, and our arms emit answers about a third longer than the baseline's, which deflates F1 precision. In our table F1 does not track the accuracy ranking.

We use the benchmark's gold metadata — `question_type`, `answer_session_ids`, per-turn `has_answer` flags — for diagnosis only, never inside any method. The one arm that reads a gold label at inference time is the `rdforget` reimplementation, faithfully, because the system being reimplemented does so.

---

## 4. Method

### 4.1 Extraction and caching

For each session, a frozen LLM extracts atomic facts as `entity | attribute | value` lines, with an instruction emphasizing facts that *update* earlier state, including updates mentioned in passing or about third parties. Output is parsed into triples; an unparseable response is retried once with a stricter format instruction, then once with reasoning enabled, and a session that still fails is dropped with an audit record rather than silently cached as empty.

Two implementation choices matter for cost and for reproducibility:

**Content-hash session caching.** Extraction is cached keyed on the md5 of the session text, not on the question id. Because 23,867 session instances reduce to 18,464 unique contents, this removes 23% of extraction work. It also decouples relation labelling from extraction: dates are stamped onto cached triples at assembly time, so a change to the relation rules costs no API calls. The extraction prompt deliberately contains no date, which keeps the cache exactly prompt-equivalent.

**Reasoning disabled for extraction.** Both available models are reasoning models, and reasoning tokens bill as completion. With reasoning enabled, 25.5% of extraction calls hit the completion cap and returned empty content — a silent extraction failure that was being cached. Disabling reasoning for extraction cut output tokens per call by roughly 30× with no measurable loss of extracted facts on a held-out comparison, and removed the cap failures. Total extraction for the whole benchmark: 78.7M input and 12.7M output tokens.

### 4.2 Chain assembly and relation rules

Facts sharing a normalized (entity, attribute) key form a chain, sorted by (date, session index). Relations are assigned by deterministic rule — no LLM call, and the rules are printable in full, which is the auditability property we wanted:

- **co-occurring**: two different values from the *same* session. Usually paraphrase or elaboration, not conflict.
- **contradiction**: different values from *different* sessions carrying the *identical* timestamp, hence unorderable.
- **supersession**: a different value at a strictly later timestamp.
- **accumulative**: the whole chain, when the attribute is in a multi-valued vocabulary (hobbies, goals, concerns, …) or when one session lists three or more distinct values. Such chains coexist rather than supersede.

An earlier version of these rules was wrong in a way worth recording, because it inverted the paper's own taxonomy. It compared (date, session index) tuples after sorting by the same key, so `contradiction` could only fire on exact ties, i.e. within a single session: 138 of 138 labelled contradictions were same-session paraphrase noise, and genuine same-day cross-session conflicts were all labelled supersession. Every `[CONFLICT]` marker shown to the answer model was therefore a false positive. Since timestamps include the minute, the corrected rule above is a total order.

The view renders one line per chain. In **current** mode, each chain shows the values of its latest session, with a `[CONFLICT]` marker only for genuine unorderable conflicts; in **historical** mode, the full evolution `v₁ → v₂ → v₃` with dates.

### 4.3 Context construction

The answer context has up to four sections: the fact view, a quantity ledger, retrieved raw turns, and a temporal scaffold. Every component below is budget-neutral: added characters are deducted from another section, and we verify total context length per question offline before running.

**Relevance-ordered view truncation (v5).** The rendered view averages 30,629 characters against a 20,000-character cap — the cap binds on 500 of 500 questions — and the earlier implementation truncated in alphabetical order of `entity::attribute`, discarding 34% of chains on a criterion unrelated to the question. We score chains against the question by BM25 and let relevance compete for the cap, then restore the original stable order so that relevance does not leak as a positional cue.

**Role-quota, session-augmented retrieval (v5).** Instead of a flat BM25 top-20, we take the top 14 user turns and top 6 assistant turns, then add the user turns of the highest-scoring session under a session-level BM25 over user text. Gold-evidence-turn recall rises substantially on the types that need it (preference 0.500 → 0.727, multi-session 0.644 → 0.728, temporal 0.783 → 0.838).

**Quantity ledger (v6, fires on 175/500).** Aggregation questions failed not because the model cannot add, but because operands were absent from the context, by two mechanisms we traced in the code. *Relevance pruning*: the view saturates its cap on all such questions, and a decisive quantity lands on the discarded side. *Current-mode collapse*: a generic verb attribute merges unrelated quantities into one chain labelled supersession, and current mode prints only the latest group, so the view itself destroys earlier operands. One example: the chain `user::has completed` holds "8 edX courses (05/21)" and "12 courses on Coursera (05/27)"; current mode rendered only the latter, and the model answered 12 where the gold was 20. The ledger force-renders every quantity-bearing fact in the store, de-duplicated, date-ordered, annotated with days relative to the question date, and read from the full chain history so collapse cannot hide an operand. Its characters are deducted from the view budget; measured total context change is −0.002%.

**Temporal scaffold (v6, fires on 179/500).** Of the residual temporal errors, most had their gold evidence turn *absent* from the context, and the absence had a shape: questions specify offsets ("two weeks ago") or windows ("during the past month"), which share no lexical content with the evidence turn, so BM25 can never retrieve it. Worse, within a same-day session family the event time exists only inside the text ("3 weeks ago"), while every line prefix shows the same date — the ordering information is simply not present in the context. The scaffold treats dates as retrieval keys (anchor day ±3 days, in-window event reports), renders a de-duplicated compact timeline with in-text event dates preferred and signed day offsets relative to the question date, and pre-computes the implied interval when both endpoints are identifiable. We use the budget-neutral variant, which trims the lowest-scoring retrieved turns by the scaffold's character count.

**Narrow-and-deep rendering for advice-shaped questions (v6, fires on 29/500).** Preference questions were the one type where our method scored *below* the no-forgetting baseline. The cause was not retrieval: in five of six cases where the baseline was right and we were wrong, the gold evidence turn was already in our context. Preference golds are rubrics — a correct answer must build on the user's specific on-record details rather than generalities — and our context was wide and shallow: the view saturated the cap on all 30 questions with 94% of its lines sharing no content word with the question, occupying half the context, while 26 retrieved turns scattered across a dozen sessions. The model then assembled an answer from an irrelevant attribute list. For advice-shaped questions, detected by a surface speech-act rule (29/30 preference questions, zero false positives on the other 470), we tighten the view cap to 2,500 characters and spend the freed budget on the session neighbourhoods of already-retrieved turns, with that question's v5 context length as a hard ceiling.

**Gate verification.** Each intervention is gated, and we verify offline that with its flag off the constructed context is byte-identical to v5, and that budget parity holds: quantity ledger 1.0000×, advice rendering 0.9999×, temporal scaffold 0.9994×, all three combined 0.9993×, with zero construction errors on 500 questions. Byte-identity outside the gate also means the arm only needs to be *run* on gated questions, which is what made these comparisons affordable: the combined arm required 286 online questions rather than 500, and the paired test is unaffected because the out-of-gate predictions are identical by construction.

### 4.4 Query classification

The tense classifier is a printable rule table over the question text: historical patterns (previously, used to, originally, before, earlier, "previous X" excluding conversational meta-references like "previous chat"), then current patterns (currently, now, latest, still), else neutral, which is routed to the current-state view. Over 500 questions it labels 44 historical, 31 current, and 425 neutral. That distribution is the first indication of the null result in §5.1: the mechanism makes a non-default decision on 15% of the benchmark.

---

## 5. Results

All arms, 500 questions, one shared prompt, one fixed judge. Significance is exact McNemar on discordant pairs; `b` counts questions our method wins and the reference loses, `c` the reverse.

### 5.0 Main table

| Arm | Acc (500) | Acc (470 non-abs) | Abstention (30) | Token-F1 |
|---|---|---|---|---|
| `fifo` (MemoryOS-style) | 0.330 | 0.287 | 1.000 | 0.163 |
| `view_only` | 0.500 | 0.470 | 0.967 | 0.165 |
| `decay` (MemoryBank-style) | 0.798 | 0.789 | 0.933 | 0.270 |
| `no_lifecycle` (attribution control) | 0.828 | 0.821 | 0.933 | 0.229 |
| `full` (no forgetting) | 0.838 | 0.834 | 0.900 | 0.226 |
| Ours v4 | 0.870 | 0.870 | 0.867 | 0.258 |
| `rdforget` (concurrent-system reimpl.) | 0.894 | 0.898 | 0.833 | 0.266 |
| `no_qc` (no query conditioning) | 0.898 | 0.900 | 0.867 | 0.269 |
| Ours v5 | 0.898 | 0.896 | 0.933 | 0.269 |
| **Ours v6** | **0.922** | **0.926** | 0.867 | 0.273 |

| Comparison | Δ | b | c | McNemar p |
|---|---|---|---|---|
| v6 vs `full` | +0.084 | 53 | 11 | < 10⁻⁵ |
| v6 vs `no_lifecycle` | +0.094 | 58 | 11 | < 10⁻⁵ |
| v6 vs `decay` | +0.124 | 70 | 8 | < 10⁻⁵ |
| v6 vs `rdforget` | +0.028 | 27 | 13 | 0.038 |
| v6 vs `no_qc` | +0.024 | 23 | 11 | 0.058 |
| v6 vs v5 | +0.024 | 23 | 11 | 0.058 |
| v5 vs `full` | +0.060 | 44 | 14 | 0.0001 |
| v5 vs `no_lifecycle` | +0.070 | 45 | 10 | < 10⁻⁵ |
| **v5 vs `no_qc`** | **+0.000** | **12** | **12** | **1.000** |
| `rdforget` vs `full` | +0.056 | 41 | 13 | 0.0002 |

By question type:

| Type | n | `full` | `rdforget` | `no_lifecycle` | `no_qc` | v5 | **v6** |
|---|---|---|---|---|---|---|---|
| single-session-user | 70 | 0.986 | 1.000 | 0.971 | 0.986 | 0.986 | 0.986 |
| single-session-assistant | 56 | 0.982 | 0.982 | 0.964 | 0.982 | 0.982 | 0.982 |
| single-session-preference | 30 | 0.667 | 0.600 | 0.533 | 0.700 | 0.600 | **0.733** |
| multi-session | 133 | 0.684 | 0.812 | 0.684 | 0.827 | 0.820 | **0.857** |
| knowledge-update | 78 | 0.936 | 0.949 | 0.962 | 0.936 | 0.987 | 0.974 |
| temporal-reasoning | 133 | 0.835 | 0.917 | 0.827 | 0.910 | 0.910 | **0.940** |

Paired by type, v6 versus `full`: multi-session b = 28, c = 5, p = 0.0001; temporal-reasoning b = 17, c = 3, p = 0.0026; the other four types are at or near ceiling and move by at most a few questions (knowledge-update b = 4, c = 1, p = 0.375; preference b = 4, c = 2, p = 0.688; the two single-session types are identical).

### 5.1 The core mechanism contributes nothing

Removing query conditioning entirely — rendering the view in full-history mode for every question — changes accuracy by **0.0000** (12 wins, 12 losses, p = 1.00). This is not a small effect measured imprecisely; it is zero, and the structure underneath it is equally clean.

The classifier routes 456 of 500 questions differently from the always-historical ablation, and those 456 questions genuinely receive different view text: rebuilding both renderings offline from the complete session cache shows a median of 19 differing lines out of about 208, i.e. a 9.5% perturbation of the view. On exactly those 456 questions the two arms score **identically**, with 11 flips in each direction. The net difference of two questions comes entirely from the 44 questions whose contexts are byte-identical between the arms — which is to say, from answer-model nondeterminism on inputs where nothing was ablated.

Three facts explain why the mechanism is inert on this benchmark:

1. **The two renderings carry the same answers.** The gold answer string appears in the current-state view for 186 of 500 questions and in the full-history view for 188, and only 4 are current-only while 6 are historical-only. The routing decision can therefore matter on at most 10 questions, 2% of the benchmark.
2. **Almost nothing is routed.** 425 of 500 questions carry no lexical tense cue and fall to the default. The mechanism makes a genuinely historical decision on 44 questions, of which only 13 are knowledge-update and 13 temporal-reasoning — the two types the design targets.
3. **Raw turns rescue misrouting.** On the five questions that genuinely needed the historical rendering but were routed to current, our method still answered all five correctly, because the 20 retrieved raw turns contained the evidence regardless.

We also note the ablation tests the mechanism in the less informative direction. Always-historical differs from our routing on 456 questions; the complementary always-current baseline would differ on only 44, which bounds any effect it could show at 8.8% of the benchmark. A sharper design would compare against always-current and would repeat each arm to establish the nondeterminism floor, which we measure at roughly 2/44 flips on byte-identical inputs.

The honest reading is that the tense axis is the wrong conditioning variable for this benchmark — not that conditioning is a bad idea in general. Its surface area is too small, and where it does apply the two renderings agree.

### 5.2 The view itself is load-bearing

Against the budget-matched control that differs only by the absence of the view, v5 gains 7.0 points (0.898 vs 0.828, b = 45, c = 10, p < 10⁻⁵) and v6 gains 9.4 points. This is the comparison we consider the paper's positive result, because the control holds retrieval configuration, budget, prompt, model and judge fixed.

The view is not self-sufficient: `view_only` scores 0.500, far below both. The view and the retrieved turns are complementary — measured over 470 answerable questions, the gold string appears in the view but not the turns for 39 questions, and in the turns but not the view for 55. A system that keeps only one of the two loses either way.

### 5.3 Interventions

Each intervention was A/B'd on its gated subset against v5, with out-of-gate predictions identical by construction:

| Intervention | Gated n | v5 → v6 (gated) | b | c | p | Projected on 500 |
|---|---|---|---|---|---|---|
| Quantity ledger | 175 | 0.903 → 0.943 | 7 | 0 | 0.016 | 0.898 → 0.912 |
| Temporal scaffold | 179 | 0.888 → 0.922 | 9 | 3 | 0.146 | 0.898 → 0.910 |
| Advice rendering | 29 | 0.586 → 0.690 | 7 | 4 | 0.549 | 0.898 → 0.904 |
| **Combined** | 286 | 0.864 → 0.906 | 23 | 11 | 0.058 | 0.898 → **0.922** |

The quantity ledger is the cleanest: 7 questions repaired, **0 broken**, at a measured context change of −0.002%. The individual temporal and advice interventions are directionally positive but not individually significant at their gated sample sizes; the combined arm is what we report as the method, and its subtype effects are where the design intended them (multi-session 0.820 → 0.857, temporal 0.910 → 0.940, preference 0.600 → 0.733).

The three gates are nearly disjoint (ledger ∩ advice = 0, advice ∩ temporal = 0, ledger ∩ temporal = 97 questions but only 1 overlapping repair), and no question repaired by one intervention is broken by another, which is why they stack.

### 5.4 Where the errors actually are

We hand-classified every error in the v4 arm into the three categories our study was designed around, plus a catch-all. The result is uncomfortable and we report it prominently: **the three planned categories account for 9.0% of errors** (stale-fact 3, history-loss 0, conflict-confusion 3 out of 67). The bulk is retrieval-miss (35) and aggregation (11), with abstention behaviour and grading boundaries making up most of the rest. The accuracy gap against the baseline is carried by retrieval and aggregation, not by the temporal error classes the lifecycle story predicts. The one clean directional win in the planned categories is history-loss, 2 errors for the baseline against 0 for us.

This diagnosis is what produced §4.3, and it is the reason the paper's analytical axis is retrieval completeness and aggregation rather than temporal error typology.

After v6, 39 errors remain (35 answerable): multi-session 17, preference 8, temporal 6, and 4 across the three near-ceiling types.

### 5.5 Abstention

On the 30 unanswerable questions our final method scores 0.867 (26/30), below `full` at 0.900 and `decay`, `no_lifecycle` and our own v5 at 0.933. Our four failures are `09ba9854_abs`, `a96c20ee_abs`, `gpt4_70e84552_abs`, `gpt4_93159ced_abs`; of these, the first two also defeat `full`, so exactly two questions separate us from the baseline on this subset. At n = 30 that is a trend, not a result, and no pairwise comparison here approaches significance.

Two observations keep us from either dismissing or overstating it. Against us: the direction is consistent across our method versions and it is the expected failure mode of injecting a synthesized view — the view supplies plausible-looking material, so the model answers where it should decline. Notably the v6 interventions cost us abstention relative to v5 (0.933 → 0.867), and the quantity ledger is the likely mechanism, since it force-renders quantities that an unanswerable question can then be answered *from*. In our favour: measured as a decline *rate* under the unified prompt, our arm is the most decisive of all — it declines on 10 of 30 unanswerable questions and on **0 of 470 answerable ones**, against 5/30 and 4/470 for `full`, and 21/30 and 30/470 for `decay`. Since a decline on an answerable question is never scored correct, the subset gap is partly a precision/recall trade, and `decay`'s better abstention score is bought with 30 declined answerable questions. We report both framings rather than selecting the flattering one, and we flag the abstention cost of the quantity ledger as an open issue rather than resolving it.

---

## 6. Two Methodological Findings

### 6.1 A prompt-wording asymmetry manufactured 3.1 points

Our first main table reported +4.8 points for the method over the no-forgetting baseline (p = 0.006) and +36.7 points on the preference subtype. Both were substantially artifacts.

We had unified the answer model, token budget, and empty-answer retry across arms, but not the system prompt *text*: the baseline runner's wording ("if the memory contains no relevant information, say you don't know") induced far more declining than the method runner's. On 470 answerable questions the baseline declined 26 times under that wording, all scored incorrect, against 4 for the method; 18 of the method's 47 wins over the baseline were questions where the baseline simply declined. After unifying the prompt across all arms, the baseline's declines on answerable questions fell from 26 to 2 and its accuracy rose from 0.818 to 0.838, while the method's advantage fell from +4.8 points (p = 0.006) to +3.2 points (p = 0.052) at the same method version. The preference subtype gain fell from +36.7 to +3.3 points.

The comparison that was *not* affected is the attribution control, which had shared the method's prompt all along. This is the reason we consider §5.2 the robust positive result and treat method-versus-`full` as secondary.

We draw a narrow lesson: matching the answer model, decoding parameters, and retry policy across arms is not sufficient. Prompt wording modulates abstention, abstention is scored as error, and any arm whose prompt differs is not a controlled comparison. Every number in this paper comes from arms sharing one prompt string.

### 6.2 Offline proxies did not predict answer correctness

We validated candidate improvements offline before paying for online runs, using gold-string coverage, gold-evidence-set completeness, and per-family derivability — proxies chosen because they are cheap and gold-grounded. We then ran an adversarial verification pass in which independent agents attempted to refute each claim by recomputing it. **Of 68 claims, 56 were refuted**, and the dominant refutation was not arithmetic error — the numbers almost always reproduced exactly — but that *the proxy did not track answer correctness*: coverage metrics were gamed by the intervention that optimized them, completeness gains coexisted with accuracy-neutral or negative context changes, and derivability was measured on questions whose gold is a derived number that appears nowhere verbatim.

The decisive evidence is that the adversarial pass rejected relevance-ordered view truncation and role-quota retrieval, on the grounds that their offline gains vanished on the context the model actually receives. Run online, those two changes are the largest single gain we measured: 0.870 → 0.898, +6.0 points over the baseline at p = 0.0001. Offline analysis was confidently wrong in both directions — it rejected what worked, and it endorsed variants that never earned an online test.

We do not conclude that offline analysis is useless. Reading failing cases was indispensable: every intervention in §4.3 came from reading errors verbatim and naming a mechanism in the code, and two real implementation defects (alphabetical view truncation, the inverted contradiction rule) were found that way. What failed was using a proxy *metric* as a decision procedure. Our practice by the end was: read the errors to design, verify offline only that the code produces the intended context change and preserves budget, and let a paired online A/B decide. Gating made that affordable — a gated intervention only needs online runs on the questions it touches.

We also note the cost asymmetry. The adversarial verification pass consumed roughly 11 million tokens of analysis and produced a wrong conclusion; the online A/B that overturned it cost about 12 CNY. In a regime where a full evaluation arm is this cheap relative to deliberation, measurement should come first.

---

## 7. Limitations

**Single benchmark, single backend.** Everything here is LongMemEval-S cleaned with one answer model and one judge. The null result of §5.1 is a statement about this benchmark's composition — 85% of its questions carry no tense cue, and its two view renderings agree on almost all answers — not a general claim about query-conditioned memory. A benchmark built to stress tense distinctions could well show the mechanism working.

**Judge protocol.** One judge model, one generic prompt, single vote. We store full predictions for all arms, so multi-vote re-judging, per-type official prompts, and second-judge calibration are cheap post-hoc recomputations; we have not run them. Our absolute numbers are not comparable to published ones (§3.4).

**Underpowered subtypes.** Knowledge-update, the type the lifecycle story most directly targets, cannot be tested at this sample size: the baseline already answers 68 of 72 non-abstention KU questions correctly, and our method and the baseline disagree on only 5 questions total, where the smallest attainable exact two-sided p is 0.0625. No possible outcome of that comparison could have been significant. The same holds for the two single-session types, which are at ceiling. Preference (n = 30) and abstention (n = 30) are too small for the effects we observe there to be established.

**The combined arm is not individually significant.** v6 versus v5 is p = 0.058. We report it as the method because its component that is significant (the quantity ledger, p = 0.016, zero regressions) is the largest contributor and because the three gates stack without conflict, but a confirmation run with repeats would strengthen it.

**No repeats.** Every arm was run once. We measure the nondeterminism floor indirectly at roughly 2 flips per 44 byte-identical inputs, but arm-level variance is unquantified.

**Extraction is a pipeline dependency.** Facts the extractor misses cannot appear in the view. One session in 18,464 failed extraction entirely after three attempts. Paraphrased or in-passing updates are the known weak spot, and extraction-miss appears in our error typology.

---

## 8. Conclusion

We set out to study query-conditioned lifecycle views and found that on LongMemEval-S the query-conditioning component contributes exactly nothing, for reasons we can localise in the benchmark's composition and in the view's content. What carries the gain is the view as an object — a compact, deterministically constructed, relevance-ordered rendering of per-attribute value chains — together with three interventions derived by reading residual errors: a quantity ledger for aggregation, a temporal scaffold that treats dates as retrieval keys, and a narrow-and-deep rendering for advice-shaped questions. At matched context budget the result is 0.922 against 0.838 for a no-forgetting baseline and 0.828 for a retrieval-only control differing only by the view.

The two negative findings are, we think, the more transferable contributions. A difference in system-prompt wording between arms manufactured three points of illusory gain through differential abstention, which suggests that controlled comparison in this area needs to fix the prompt string, not just the model. And offline proxy metrics systematically failed to predict answer correctness, to the point of rejecting the largest real improvement we found. Memory systems are evaluated expensively and designed cheaply, which creates a strong pull toward proxy-driven development. Our experience is that the pull should be resisted: gate each intervention, verify offline only that the context changes as intended and the budget holds, and let a paired online comparison decide.

---

## Appendix A. Reproduction

Pipeline: `code/{common,lifecycle,query_classifier,run_qc,run_baseline,run_rdforget}.py`; interventions in `code/{v6_agg,v6_tr}.py` and the gated branches of `run_qc.py`, each behind an environment flag (`QC_V5`, `QC_V6_AGG`, `QC_V6_SSP`, `QC_V6_TR`). Runners: `run_full500.py <arm> <workers> <tag>` for full arms, `run_v6_subset.py <FLAG> <workers>` for gated subsets. Gate and budget verification: `v6_gates.py` (rebuilds every arm's context offline with the API client stubbed to raise, so a cache miss surfaces as an error rather than a charge). Coverage diagnostics: `coverage_diag.py`. Per-question rows for every arm: `results/full500_*.jsonl` and `results/v6_*.jsonl`, each row carrying qid, type, judge verdict, token-F1, query label, view mode, and the prediction.

Dataset: LongMemEval-S cleaned, sha256 `d6f21ea9…c3a442`. Extraction cache: 18,464 content-hashed session extractions under `results/lifecycle_cache/sessions_nt/`, sufficient to rebuild every view offline with no API calls. Total measured API cost for all arms and all extraction: 78.7M input / 12.7M output tokens on `deepseek-flash` and 1.9M / 0.9M on `deepseek-v4-pro`, approximately 340 CNY end to end.

## Appendix B. Relation rule table

| Condition on consecutive chain elements | Relation | Rendering (current / historical) |
|---|---|---|
| attribute in multi-valued vocabulary, or ≥3 distinct values in one session | accumulative | all values with dates / same |
| same session, different value | co-occurring | joined in latest group / joined within step |
| different session, identical timestamp, different value | contradiction | latest group + `[CONFLICT]` / step marked `/conflict` |
| strictly later timestamp, different value | supersession | latest group only / `v₁ → v₂ → v₃` |
| identical normalized value | merged | — |

Multi-valued vocabulary: interest(s), hobby/hobbies, goal(s), plan(s), considering, likes, dislikes, preference(s), activity/activities, emotional state, concern(s), skill(s), wish(es), idea(s), topic(s).

## Appendix C. Query classifier rules

Historical (checked first): `previously`, `before`, `originally`, `at first`, `used to`, `former(ly)`, `old …(value|name|address|job|phone)`, `first time`, `initial(ly)`, `history of`, `how (has|did) … change`, `earlier`, `back (then|when)`, `what was … before`, `previous` not followed by {chat, conversation, discussion, session, message}(s). Current: `current(ly)`, `now`, `latest`, `still`, `today`, `right now`, `as of`, `up-to-date`, `most recent`. Otherwise neutral, routed to the current-state view. Distribution over 500 questions: 44 historical, 31 current, 425 neutral.
