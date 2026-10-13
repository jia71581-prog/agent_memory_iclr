# Error typology: QC-Lifecycle vs. no-forgetting baseline (LongMemEval-S cleaned, n=500)

Analysis target: plan section 2, item 3 (the paper's analytical contribution). Every error in
`full500_qc.jsonl` (67) and `full500_full.jsonl` (91) was classified by hand-reading the question,
the gold answer, the gold-evidence turns and the stored prediction. No LLM was used for
classification. Labels are frozen in `error_typology_labels.json`; the tabulation is reproducible
with `python3 _error_typology.py`.

Answer model deepseek-flash, judge deepseek-v4-pro (n=1, fixed prompt). qc arm = lifecycle view +
query-conditioned mode + BM25 top-20 raw turns; full arm = BM25 top-30 raw turns, no forgetting.

---

## 1. Category definitions actually applied

The plan names three categories plus a catch-all. Two extra rules were needed to make the labels
auditable rather than impressionistic, because the naive reading of the definitions silently
absorbs plain retrieval failures into the lifecycle categories.

| code | category | operational test |
|---|---|---|
| **S** | stale-fact | the answer asserts value `v_old` for an attribute whose newer value `v_new` is the gold, **and `v_new` was literally present in that arm's context** |
| **H** | history-loss | the question asks for a pre-update state of an attribute that *does* have a later value, and the answer returns only the later value or "I don't know" |
| **C** | conflict-confusion | several values for one attribute coexist legitimately (multi-valued attribute) or as a reliability conflict (user statement vs. assistant-generated figure), and the answer picks the wrong one, drops a still-valid one by false supersession, or refuses because it cannot resolve them |
| o1 | other: retrieval-miss | the required value never reached this arm's context |
| o2a | other: over-abstention | the required value **was** in context; the answer said "I don't know" or went generic |
| o2b | other: under-abstention | gold is "the information provided is not enough"; the answer supplied a value anyway |
| o3 | other: aggregation | all needed values in context; the error is counting, summing or date arithmetic |
| o4 | other: grading-boundary | the answer is substantively correct on the retrieved facts, or the gold itself is unsupported by the haystack |
| o5 | other: extraction-miss | empty prediction |

**Rule 1 (availability discipline).** S and C are only assigned when the competing value is verifiably
in the arm's reconstructed context. Every S and C label below was checked by rebuilding that arm's
exact context with `_recon_ctx.py` (BM25 is deterministic; the lifecycle view is rebuilt from the
on-disk extraction cache, 0 cache misses) and string-searching for both the older and the newer
value. This check moved one candidate out of S: `gpt4_2f56ae70` ("which streaming service did I
start using most recently", gold Disney+, full answered Apple TV+) *looks* like a textbook stale-fact
error, but "Disney+" is absent from the full arm's top-30 — it is a retrieval miss, o1. The
token-overlap heuristic in `_err_dossier.json` (`gold_tok_in_raw30 = 1.0`) was a false positive there.

**Rule 2 (H requires a lifecycle).** A past-referring question whose answer was simply never
retrieved, and whose attribute has no later competing value, is o1, not H. Without this rule almost
every temporal-reasoning retrieval miss would be counted as "history-loss" and the category would
become meaningless (it would have absorbed ~20 further cases per arm).

Mechanism (evidence coverage) is reported **separately** from category, so a category count is never
read as a causal claim.

---

## 2. Counts per arm, and the shift

| category | qc (n=67) | full (n=91) | shift (qc − full) |
|---|---|---|---|
| stale-fact | **3 (4.5%)** | **2 (2.2%)** | +1 (+2.3pp) |
| history-loss | **0 (0.0%)** | **2 (2.2%)** | −2 (−2.2pp) |
| conflict-confusion | **3 (4.5%)** | **4 (4.4%)** | −1 (+0.1pp) |
| other: retrieval-miss | 35 (52.2%) | 57 (62.6%) | −22 (−10.4pp) |
| other: over-abstention | 4 (6.0%) | 9 (9.9%) | −5 (−3.9pp) |
| other: under-abstention | 4 (6.0%) | 2 (2.2%) | +2 (+3.8pp) |
| other: aggregation | 11 (16.4%) | 5 (5.5%) | +6 (+10.9pp) |
| other: grading-boundary | 6 (9.0%) | 9 (9.9%) | −3 (−0.9pp) |
| other: extraction-miss | 1 (1.5%) | 1 (1.1%) | 0 |

**Lifecycle-attributable errors (S+H+C): qc 6/67 = 9.0%, full 8/91 = 8.8%.** As a fraction of all 500
questions: qc 1.2%, full 1.6%. The three planned categories account for a net difference of **2
questions** out of the 24-question accuracy gap (0.866 vs. 0.818). The gap is carried by
retrieval-miss (−22) and over-abstention (−5), partly given back as aggregation (+6).

The composition *proportions* barely move: the qc arm is not a baseline with the temporal errors
removed, it is a baseline with fewer errors of nearly the same shape. The one clean directional
result is history-loss: 2 → 0.

### Mechanism split (was the gold evidence in that arm's raw context?)

| category | qc | full |
|---|---|---|
| stale-fact | complete=3 | complete=1, partial=1 |
| history-loss | — | partial=2 |
| conflict-confusion | partial=2, none=1 | complete=2, partial=1, none=1 |
| retrieval-miss | partial=22, none=13 | partial=34, none=23 |
| over-abstention | complete=3, partial=1 | complete=7, partial=2 |
| aggregation | complete=5, partial=5, none=1 | complete=3, partial=2 |

All 3 qc stale-fact errors had **complete** gold-evidence coverage: the newer value was in front of
the model and it chose the older one. Both history-loss errors were **partial** coverage — the
pre-update value never reached the context — so they are lifecycle errors in shape but retrieval
errors in mechanism, and a lifecycle view alone would not be guaranteed to fix them.

### Composition by question type

| question type | qc | full |
|---|---|---|
| knowledge-update | 4: S:3 o2b:1 | 4: S:2 H:1 o2a:1 |
| multi-session | 32: C:2 o1:18 o2a:1 o2b:2 o3:8 o4:1 | 43: C:2 o1:31 o2a:3 o2b:2 o3:5 |
| temporal-reasoning | 19: C:1 o1:11 o2b:1 o3:3 o4:2 o5:1 | 21: H:1 C:1 o1:16 o4:2 o5:1 |
| single-session-preference | 9: o1:5 o2a:1 o4:3 | 20: o1:9 o2a:4 o4:7 |
| single-session-user | 2: o2a:2 | 1: o2a:1 |
| single-session-assistant | 1: o1:1 | 2: C:1 o1:1 |

All stale-fact errors in both arms sit in knowledge-update. Conflict-confusion is spread across
multi-session, temporal-reasoning and single-session-assistant, i.e. it is not a KU-only phenomenon —
useful for the paper, since it means the contradiction machinery is exercised outside the subclass
that was explicitly built for supersession.

---

## 3. Verbatim cases

### 3.1 stale-fact

**S-1 `852ce960` (knowledge-update) — both arms wrong. The paper's motivating example.**

- Q: *"What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?"*
- Gold: `$400,000`
- Evidence 2023/08/11: *"I'm actually buying a $325,000 house, and I got pre-approved for $350,000 from Wells Fargo."*
- Evidence 2023/11/30: *"...remember when I got pre-approved for $400,000 from Wells Fargo?"*
- qc pred: *"You were pre-approved for **$350,000** from Wells Fargo. (A later message mentioned $400,000, but the original recorded pre-approval amount is **$350,000**.)"*
- full pred: *"Based on the retrieved memory, you said you were pre-approved for **$350,000** from Wells Fargo. (You later referenced $400,000, but the original pre-approval amount stated was $350,000.)"*
- Verified: both `350,000` and `400,000` present in both arms' raw context; `350,000` also in the qc lifecycle view.
- Why stale-fact: the answer asserts the superseded value while explicitly seeing the newer one. Both
  arms treat the update as an unreliable *contradiction* to be discounted ("the original recorded
  amount") rather than as a *supersession* to be adopted. This is exactly the
  supersession-vs-contradiction distinction the paper claims as its own, and the method currently
  fails it.

**S-2 `07741c45` (knowledge-update) — qc wrong, full right.**

- Q: *"Where do I **currently** keep my old sneakers?"*  Gold: `in a shoe rack in my closet`
- Evidence 2023/05/25: *"...I've been keeping them under my bed for storage, and they're starting to smell."*
- Evidence 2023/05/29: *"...I'm looking forward to get rid of some of my old sneakers in a shoe rack in it [my closet]..."*
- qc pred (qlabel=current, mode=current): *"You currently keep your old sneakers **under your bed** for storage. You've also mentioned planning to move them to a shoe rack in your closet."*
- Verified: both values in the qc raw context.
- Why stale-fact: the query classifier correctly labelled this `current` and the view was rendered in
  current mode, yet the answer still fronted the superseded location and demoted the newer one to a
  "plan". The strong baseline answered it correctly.

**S-3 `69fee5aa` (knowledge-update) — qc wrong; full **and** qc_no_lifecycle right, i.e. view-induced.**

- Q: *"How many pre-1920 American coins do I have in my collection?"*  Gold: `38`
- Evidence 2023/05/27: *"I have a total of 37 coins in that collection."*
- Evidence 2023/05/29: *"I just added a new coin to my collection of pre-1920 American coins — a 1915-S Barber quarter."*
- qc pred: *"You have **37** pre-1920 American coins in your collection."*
- Verified: both evidence turns in the qc raw context.
- Why stale-fact: an *incremental* update (+1) rather than a value replacement. The lifecycle chain
  keeps `count = 37` as the current slot value and the later "added a new coin" event does not
  update it, so the view actively presented the stale count as current. The k=20 BM25-only control
  answered correctly.

**S-4 `41698283` (knowledge-update) — full wrong, qc right (the method's win in this category).**

- Q: *"What type of camera lens did I purchase **most recently**?"*  Gold: `a 70-200mm zoom lens`
- Evidence 2023/03/11: *"...I recently got a new 50mm prime lens..."*; 2023/08/30: *"...I've been getting some great shots with my new 70-200mm zoom lens lately."*
- full pred: *"You most recently purchased a **50mm prime lens**."*
- Verified: both `50mm` and `70-200` in the full arm's top-30.

### 3.2 history-loss (only 2 exist in total; both in the full arm)

**H-1 `0977f2af` (knowledge-update) — full wrong, qc right.**

- Q: *"What new kitchen gadget did I invest in **before** getting the Air Fryer?"*  Gold: `Instant Pot`
- full pred: *"I don't know. The retrieved memory mentions getting an Air Fryer, but it doesn't mention any other new kitchen gadget purchased before that."*
- Verified: `Air Fryer` present in the full arm's context, `Instant Pot` **absent**.
- Why history-loss: canonical shape — a pre-update query on a supersession chain, answered with the
  current value plus "I don't know" for the earlier one. Mechanism is a retrieval miss of the
  pre-update evidence, which is the failure mode the plan predicted for no-forgetting + similarity
  retrieval (the newer mention out-competes the older one lexically).

**H-2 `gpt4_93159ced` (temporal-reasoning) — full wrong, qc right.**

- Q: *"How long have I been working **before** I started my current job at NovaTech?"*  Gold: `4 years and 9 months`
- full pred: *"The memory doesn't specify how long you worked before starting at NovaTech. It only says you've been at NovaTech for about 4 years and 3 months and have been in the field since graduating from UC Berkeley."*
- Verified: `4 years and 3 months` (current value) in context; `9 years` (the total needed for the
  subtraction) **absent**.
- Why history-loss: the answer returns only the current-role duration and declines the earlier span.

The qc arm has **zero** history-loss errors. This is the cleanest positive result in the typology and
it lines up with the mechanism the paper claims (the view keeps the full chain reachable), but note
n=2 on the baseline side: it is a 2-question effect and cannot carry a headline claim on its own.

### 3.3 conflict-confusion

**C-1 `46a3abf7` (multi-session) — both arms wrong; false supersession of a still-valid value.**

- Q: *"How many tanks do I **currently** have, including the one I set up for my friend's kid?"*  Gold: `3`
- Evidence: 5-gallon betta tank (Finley), 20-gallon community tank "Amazonia", plus the 1-gallon set up for a friend's kid.
- full pred: *"...you currently have **2 tanks**: your 20-gallon community tank "Amazonia" and the 1-gallon tank you set up for your friend's kid. **The 5-gallon was your old tank**, and the quarantine tank was only being considered."*
- qc pred: *"You currently have **2 tanks**: — Your **20-gallon freshwater community tank "Amazonia"** — The **1-gallon tank** you set up for your friend's kid."*
- Verified: `5-gallon` present in the qc raw context **and in the qc lifecycle view**, and in the full raw context.
- Why conflict-confusion: "tank" is a multi-valued attribute. Both arms collapsed it to a
  single-valued slot and treated the newer 20-gallon as replacing the 5-gallon; the full arm says so
  explicitly. This is the multi-valued-attribute hazard the plan flags as uniquely ours (section 2b:
  "多值属性词表") — and the current lifecycle view does not prevent it.

**C-2 `gpt4_f420262d` (temporal-reasoning) — qc wrong; full **and** qc_no_lifecycle right, i.e. view-induced.**

- Q: *"What was the airline that I flied with on Valentine's day?"*  Gold: `American Airlines`
- qc pred: *"You flew with **JetBlue** on Valentine's Day — flight **B6 1018** from **Boston (BOS) to Fort Lauderdale (FLL)**, departing at **7:00 AM**."*
- Verified: `JetBlue` and `B6 1018` appear **only in the lifecycle view**, as
  `[2023/02/14 (Tue) 20:47] user | booked flight = JetBlue B6 1018 BOS to FLL, 7:00 AM departure`.
  `American Airlines` is present in the raw turns.
- Why conflict-confusion: two values compete for "airline on 2/14". The view rendered a *booked/planned*
  flight as a date-stamped fact and the model preferred the view's clean slot over the raw turn. The
  same view also contains `[2022/11/17] user | American Airlines flight date = February 10th`, where the
  session date and the in-content date disagree. Concrete evidence that slot rendering can manufacture
  a confident wrong current value.

**C-3 `09ba9854` (multi-session) — full wrong, qc right; reliability conflict.**

- Q: *"How much will I save by taking the train from the airport to my hotel instead of a taxi?"*  Gold: `$50`
- Evidence (user): *"...taking the train ... my friend told me it's only around $10."*; *"...a taxi ... would cost around $60."*
- full pred: *"...you'd save roughly **$30–$40**: the taxi was about **$60**, while a train/subway option to your hotel was **estimated at $20–$30**. If you only count the cheapest train to Shinjuku Station (~$9–$11), the savings would look larger, but that doesn't include getting from the station to your hotel."*
- Verified: both user statements (`only around $10`, `around $60`) in the full arm's context; the
  `$20–$30` figure is an **assistant-generated estimate** from an earlier turn.
- Why conflict-confusion: a reliability conflict, not a temporal one. Two values coexist for the same
  attribute and the answer preferred the assistant's own earlier guess over the user's stated figure.

Two further conflict-confusion cases worth citing:

- `561fabcd` (single-session-assistant, full only): gold `Fissionator`; pred *"The memory doesn't
  explicitly record a final agreed name. It shows the assistant suggested **"Radialisk"** and later
  listed **"Fissionator"** among other options..."* — `Fissionator` verified present in context;
  several candidate values coexist and the answer refuses rather than resolving.
- `gpt4_fe651585` (temporal-reasoning, full only): gold `Alex`; pred *"Rachel became a parent first.
  Her twins, Jackson and Julia, were about to celebrate their first birthday as of March 2023, while
  Alex adopted a baby girl from China in January 2023."* The "first birthday" claim comes from an
  **assistant** turn in the haystack (*"It seems like just yesterday they were born, and now they're
  already celebrating their first birthday!"*), which the user never confirms. Same reliability-conflict
  mechanism as C-3; a reviewer could reasonably call this a defective gold instead, so it is flagged.
- `1a8a66a6` (multi-session): qc = conflict-confusion, full = retrieval-miss. Gold `2`
  (New Yorker + Architectural Digest; Forbes cancelled). Both answered `1`. Verified:
  `Architectural Digest` is **absent from both arms' raw turns but present in the qc lifecycle view** —
  so for the qc arm the value was available and dropped (multi-valued collapse), while for the full
  arm it never arrived. Same question, different category per arm.

### 3.4 the "other" bucket, with examples

- **o1 retrieval-miss (35 qc / 57 full — the dominant class).** `157a136e`: gold `43`
  (grandma 75 − user 32); full pred *"I don't know... it doesn't give your age"* with 0/3 evidence
  turns retrieved. `a3838d2b`: gold `4` charity events before Run for the Cure; full answered `2`,
  two of the four events unretrieved. `gpt4_e061b84f`: full pred is the single sentence *"I don't know."*
- **o2a over-abstention (4 qc / 9 full) — required value in context, answer refuses.** `73d42213`:
  gold `9:00 AM`; both "left home at 7 AM" and "took me two hours to get to the clinic" verified in
  the full context; pred *"I don't know... it doesn't specify what time you reached the clinic."*
  `a2f3aa27`: gold `1300`; pred *"I don't know your current Instagram follower count. The last update
  in memory was from 2023/05/25, when you mentioned being close to 1,300."* `37f165cf`: gold `856`;
  both `440 pages` and `416-page` verified in both arms' contexts; both arms declined.
- **o2b under-abstention (4 qc / 2 full) — gold is "not enough information".** `a96c20ee_abs`: full
  pred *"Harvard University."*, qc pred *"You presented your poster at **Harvard University**..."*.
  `031748ae_abs`: qc pred *"When you first mentioned your new role (as Senior Software Engineer, not
  Software Engineer Manager), you said you led **4 engineers**. Later, you mentioned leading **5
  engineers**."* — the false premise is detected and then answered anyway. The qc arm is worse here
  (4 vs 2): higher answer coverage costs abstention discipline.
- **o3 aggregation (11 qc / 5 full).** `9ee3ecd6`: gold `100`; both values (`200 points` current,
  `300 points` threshold) verified in context; both arms answered `300`. `6e984301`: qc imported
  "about 6 weeks" from an unrelated session (*"I signed up for the recreational volleyball league
  about 6 weeks ago"*, verified present) to answer a sculpting-class question. `d851d5ba`: qc
  answered *"over $8,500"* against gold `$3,750`; the spurious `$5,000` "music education" addend is
  present **only in the lifecycle view**.
- **o4 grading-boundary (6 qc / 9 full).** Three golds look defective: `370a8ff4` (gold `15` weeks
  between 2023/01/19 and 2023/04/10; the true interval is 81 days = 11.6 weeks, and both arms
  answered "about 11 weeks and 4 days" — arithmetically correct, judged wrong); `71017277` (question
  asks who gave "a piece of jewelry"; the only matching haystack event is a **crystal chandelier**
  from the aunt, and the qc arm surfaced exactly that while declining the "jewelry" framing);
  `80ec1f4f_abs` (gold `0. You did not mention visiting any museum in December`; qc pred *"I don't
  have any record of museum or gallery visits in December..."* — semantically the gold, judged wrong).
  The rest are single-session-preference answers that are well grounded but miss one gold-named
  detail, e.g. `fca70973` and `b0479f84`.
- **o5 extraction-miss (1 each).** `gpt4_7abb270c` has an **empty** `pred` in the qc, full and
  qc_no_lifecycle arms (0 in decay/fifo/no_qc/view_only). Single case, but it is a silent
  pipeline failure, not a model error, and it should be either re-run or footnoted.

---

## 4. Cross-cutting mechanisms (flags, not categories)

| flag | qc | full |
|---|---|---|
| view-induced wrong value | **5** (`gpt4_f420262d`, `d851d5ba`, `88432d0a`, `eac54add`, `gpt4_2f8be40d`) | 0 (arm has no view) |
| distractor substitution (value imported from an unrelated session) | 7 | 3 |
| confabulated date anchor (date appears nowhere in the haystack) | 3 (`8c18457d`, `9a707b81`, `gpt4_65aabe59`) | 0 |
| disputed gold | `370a8ff4`, `71017277`, `80ec1f4f_abs` | same |

`8c18457d` is worth a footnote: the correct anchor `3/8` **is** in the qc context, and the answer
nonetheless asserted "the 10th of last month (February 10)" — a string that appears nowhere in that
question's haystack.

---

## 5. Threats to the paper's claims (read before writing the results section)

1. **The typology does not explain the headline gain.** S+H+C is 9.0% of qc errors and 8.8% of full
   errors; the net lifecycle-attributable difference is 2 questions out of a 24-question gap. The
   analytical contribution is still publishable — nobody else reports error composition — but the
   honest headline is "the three temporal error classes are a small and *stable* share of errors;
   the accuracy difference is a retrieval-coverage and abstention effect", not "we removed the
   temporal errors".
2. **qc and full do not use the same retrieval budget.** qc reads 20 raw turns, full reads 30. Of the
   23 qc-only errors, **8 have strictly better gold-evidence coverage at k=30 than at k=20** — i.e.
   about a third of the method's unique errors are attributable to the smaller raw budget, not to the
   view. Conversely the method wins overall while handicapped. The clean equal-budget comparison is
   **qc vs. qc_no_lifecycle** (both k=20, 67 vs. 97 errors); qc vs. full mixes the view effect with
   a k effect and should be labelled as such in the paper.
3. **The view both fixes and breaks.** Against the equal-k control, the view fixes 43 questions and
   breaks 13. Two of the 13 breaks are in the flagship categories: `69fee5aa` (stale-fact, the chain
   pinned a stale count) and `gpt4_f420262d` (conflict-confusion, the view rendered a planned flight
   as the fact). A "restore-counterfactual" style audit that only counts fixes will overstate the view.
4. **The method is not better on stale-fact.** qc 3 vs. full 2, and two of the three qc stale-fact
   errors (`07741c45`, `69fee5aa`) are questions the strong baseline answered correctly. Any claim
   of the form "our lifecycle view reduces stale-fact errors" is contradicted by this data.
5. **Where the gain actually lives.** knowledge-update: qc 74/78 = full 74/78 (identical; vs.
   qc_no_lifecycle 72/78). The gain is in multi-session (101/133 vs. 90/133) and
   single-session-preference (21/30 vs. 10/30). ssp is the softest-graded subclass — 7 of the full
   arm's 20 ssp errors are grading-boundary — so a 30-question subclass supplies ~11 of the 24-question
   gap. Report the subclass table, not just the total.
6. **The dominant mechanism is abstention suppression.** Abstention-style predictions: full 67/500
   (13.4%), qc_no_lifecycle 49/500, qc 30/500 (6.0%). Abstentions make up **44.0%** of the full arm's
   errors but only **13.4%** of the qc arm's. The view mostly buys answer coverage — a real and
   defensible contribution, but it should be named, and it comes with the o2b regression (under-
   abstention 4 vs. 2) which a reviewer will find in the `_abs` subset.
7. **Query conditioning is close to inert on this benchmark.** The classifier labels 425/500 (85%)
   questions `neutral` → current mode; only 44 `historical` and 31 `current` fire. qc (67 errors) vs.
   qc_no_qc (always historical, 69 errors) differ by 2 net errors over a 24-question symmetric
   difference. The misclassification-transmission analysis promised in the plan will have very little
   signal at this label distribution; consider reporting the classifier's firing rate honestly and
   moving the query-conditioning claim to the LoCoMo/budget-sweep setting where it can be stressed.
8. **Judge and gold hygiene.** 15 grading-boundary errors across the two arms, of which 3 are
   probably wrong golds (§3.4). At n=500 these are ~1.2pp per arm of accuracy and they are *not*
   symmetric between arms (6 qc vs. 9 full), so they slightly flatter the method. Recommend either
   reporting accuracy with these 3 items excluded as a sensitivity check, or a second judge pass on
   the 15.

---

## 6. Reproduction

- labels: `/root/agent_memory_iclr/results/error_typology_labels.json`
- tabulation: `python3 /root/agent_memory_iclr/results/_error_typology.py`
- context reconstruction used for every S/C/o2a verification: `_recon_ctx.py`
  (`qc_context`, `full_context`); deterministic, 0 extraction-cache misses
- per-error dossier (question, gold, evidence, both preds, coverage): `_err_dossier.json`
- per-evidence-turn retrieval flags at k=20 and k=30: `_ev_hits.json`
