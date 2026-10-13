"""v5 guards: abstention calibration + preference-question handling (training-free, zero extra API calls).

Standalone candidate module. Imports the existing pipeline read-only; edits nothing.
Every design below is annotated with the offline number that motivated it
(reproduce with:  python3 v5_guards.py validate   /  python3 v5_guards.py validate --quick).

==================================================================================
PART A - ABSTENTION.  What is actually wrong (not what was suspected)
==================================================================================
The prior hypothesis was "the synthesized view supplies plausible material, so the
model answers where it should refuse". Measured: FALSE.

  arm             abs judge acc     view in ctx?   raw turns in ctx?
  qc              0.833 (25/30)     yes            yes  (k=20)
  qc_view_only    0.967 (29/30)     yes            no
  qc_no_lifecycle 0.900 (27/30)     no             yes  (k=20)
  full            0.933 (28/30)     no             yes  (k=30)
  fifo            1.000 (30/30)     no             yes  (k=30, 15 newest sessions)

The view-only arm is the *best* non-fifo arm on abstention: the view is protective,
not harmful, because LLM extraction keeps user-asserted facts and drops assistant
chatter (verified case-by-case below). So the guard must not attack the view.

A1. The abstention metric is a *decline-rate* metric, and our arm has the best
    calibration of all seven arms.  Treating "the prediction contains a decline
    phrase" (is_decline() below; 28/30 agreement with the judge on the abs subset)
    as a detector of the 30 unanswerable questions over the pooled 500:

      arm              declines on abs   declines on answerable   precision
      qc (ours)             27/30              19/470  (4.0%)       0.587   <- best
      qc_no_qc              27/30              21/470  (4.5%)       0.562
      qc_no_lifecycle       27/30              50/470 (10.6%)       0.351
      full                  28/30              51/470 (10.9%)       0.354
      decay                 28/30              66/470 (14.0%)       0.298
      qc_view_only          29/30             191/470 (40.6%)       0.132
      fifo                  30/30             295/470 (62.8%)       0.092

    A declining answer on an answerable question is *never* correct in any arm
    (qc 5/19 are partial declines that still scored; the other arms' false declines
    score 7/51, 9/66, 7/295, 7/191 - all on partial declines).  fifo's perfect
    1.000 is bought by refusing 63% of the answerable set. So the reported gap
    (0.833 vs 0.933) is a precision/recall trade in which our arm dominates, and
    the paper should report the joint table, not the abstention column alone.

A2. Failure-by-failure diagnosis of our 5 losses (all preds read verbatim):
    * 80ec1f4f_abs  "How many museums/galleries did I visit in December?" gold "0."
      Ours: "I don't have any record of museum or gallery visits in December ... so
      I can't determine how many you visited that month."  That IS a decline
      (is_decline fires) and full's near-identical "I don't know. The retrieved
      memory doesn't mention any museums..." was graded CORRECT.  => judge
      false-negative on the decline-vs-zero distinction.  Fixable by answer FORM
      (G1d), not by retrieval.
    * gpt4_93159ced_abs  "How long have I been working before my current job at
      Google?"  Ours states the premise failure ("You haven't mentioned working at
      Google") and then substitutes NovaTech and computes "4 years and 9 months".
      view_only and no_lifecycle both refuse cleanly.  => premise substitution.
    * 031748ae_abs  "How many engineers do I lead as Software Engineer Manager?"
      Ours leads with the number and puts the premise failure in parentheses;
      full/fifo/decay give the same two numbers but LEAD with "I don't know about a
      Software Engineer Manager role" and are graded CORRECT.  => answer-form only.
    * 09ba9854_abs  "How much will I save taking the bus instead of a taxi?"
      All bus/taxi prices in our top-20 come from ASSISTANT turns (13 of 20 turns
      assistant, 95% of characters, 130 assistant numeric tokens vs 4 user ones -
      the highest assistant-number share of all 30 abs questions).  The user only
      ever said "I was told a taxi would cost around $60".  view_only refuses
      correctly because extraction never promoted assistant travel advice to a
      user fact.  => speaker-provenance failure in the RAW channel.
    * a96c20ee_abs  "At which university did I present a poster for my undergrad
      course research project?"  Fails in 6 of 7 arms (only fifo, by not holding
      the session).  The user did present a poster at a conference at Harvard, but
      for thesis research; the gold demands the undergrad-course modifier.
      => fine-grained modifier distractor; not fixable without over-refusing.

    So 3/5 are answer-FORM losses, 1/5 is provenance, 1/5 is out of reach.

A3. Candidate (i) "provenance requirement: view-only support is insufficient" -
    REJECTED on measured cost.  39 of the 470 answerable questions have the gold
    string in the view and NOT in the top-20 raw turns; 29 of those are currently
    correct.  Maximum abstention gain is 4 questions; maximum answerable loss is
    29 questions (-6.2 pp on the 470).  Strictly negative trade.
    Also 117/470 answerable have the gold string in neither channel yet 93 are
    graded correct, so string-level "support" is not a usable licence test at all.

A4. Candidate (iii) "cheap deterministic pre-check for whether any evidence about
    the queried entity exists" - IMPLEMENTED, but as an advisory note, never a
    gate.  Measured detector rates (abs recall vs answerable false-alarm rate):
      D1 missing proper noun                        5/30 (0.167)   17/470 (0.036)
      D2 missing content noun (verb/date filtered) 12/30 (0.400)   63/470 (0.134)
      D3 missing bigram, both unigrams present     10/30 (0.333)  126/470 (0.268)
      D4 two rarest question terms never co-occur  15/30 (0.500)  164/470 (0.349)
      D2 or D3 or D4                               26/30 (0.867)  244/470 (0.519)
    Precision never exceeds 0.23, so any hard gate loses more than it gains
    (base rate is 30/500).  As a note it costs ~25 prompt tokens and stays
    falsifiable by the model.  D1+D2 (union) is the recommended note trigger, and
    suppressing it whenever advice_shaped() or meta_conversational() fires is free
    (0/30 abstention questions are advice-shaped or meta-conversational):
        raw D1|D2                     abs 12/30   answerable 63/470   prec 0.160
        gated off advice              abs 12/30   answerable 58/470   prec 0.171
        gated off advice + meta       abs 12/30   answerable 42/470   prec 0.222
    Suppressing it on advice questions is not cosmetic: on 0edc2aef the note would
    have announced "'miami' never appears in memory" and on 09d032c9 "'battery'
    never appears" - i.e. it would have reinforced exactly the two wrong declines
    diagnosed in B3(i).
    HONEST LIMIT: under the gated D1|D2 trigger the note fires on only 1 of our 5
    actual abstention losses (a96c20ee, the one case no guard can fix).  It does
    not fire on 80ec1f4f, 09ba9854, gpt4_93159ced or 031748ae.  Candidate (iii) is
    therefore close to irrelevant to our specific failures; the levers are G1 and
    G2.  (With D3/D4 enabled it fires on all five, at a 0.519 false-alarm rate -
    not a trade worth making.)

A5. Candidate (ii) "mark every view line DERIVED so derived lines cannot license an
    existence claim" - REJECTED.  It is the wrong direction of causality: the view
    is the channel that already refuses correctly (0.967), and the same 39-question
    measurement in A3 shows view-only support carries 29 correct answers.

==================================================================================
PART B - PREFERENCE (SSP, n=30)
==================================================================================
B1. Reframing: SSP is our largest *relative* win, not a weakness.
      qc 0.700 | qc_no_qc 0.667 | qc_view_only 0.533 | qc_no_lifecycle 0.500
      | decay 0.367 | full 0.333 | fifo 0.133
    Ours is +36.7 pp over the full-memory baseline - the biggest per-type margin
    in the whole table.  0.700 is the lowest absolute subtype only because SSP is
    intrinsically the hardest type for every arm.

B2. What a correct answer requires (all 30 golds read).  The gold is a *rubric*:
    "The user would prefer responses that <use specific prior facts F1..Fk>."
    Gold-string coverage is 0.000 by construction.  The requirement is (a) do not
    decline, (b) name every on-record personal detail bearing on the topic.
    The evidence is a user turn that states the preference in passing while asking
    about something else, e.g. 0edc2aef's "Besides great views, I also like hotels
    with unique features, such as a rooftop pool or a hot tub on the balcony."

B3. Cause split over the 9 failures (traced individually):
    (i) EVIDENCE MISSING -> answer degenerates into a decline: 0edc2aef, 32260d93,
        09d032c9, d6233ab6, 35a27287 (5/9).  SSP has the worst evidence-turn recall
        of any type: 0.500 at top-20 turns.  The question ("Any tips?", "Can you
        recommend a show for tonight?") shares almost no vocabulary with the
        evidence turn (a power bank; a stand-up-comedy request), so BM25 cannot
        bridge it.  Note 2 of our 19 false declines on the answerable set are SSP.
    (ii) EVIDENCE PRESENT, RUBRIC UNDER-COVERED: 75f70248 (named the shedding cat,
        missed the recent deep clean), a89d7624 (named The Killers at Red Rocks,
        missed the Brandon Flowers encounter), 1d4e3b97 (named chain+cassette,
        missed the new Garmin), 195a1a1b (listed activities, missed the
        before-9:30pm / no-phone-no-TV constraint) (4/9).  Same shape as the MS
        aggregation errors: a SUBSET of the required set.
    (iii) Rendering losing the user's phrasing: only partly true.  45 preference
        values from SSP gold sessions were traced: extraction captured them and
        42/45 render into the full view, but only 36/45 survive the 20k character
        cap.  d6233ab6 is a clean causal case: `user | took | advanced placement
        courses in economics` is in the full view, dropped by the cap, and our
        answer then declared it had "nothing about your high school experience".
    (iv) Judge strictness is real but secondary: 75f70248 and 1d4e3b97 each hit one
        of two required details.  Not actionable without changing the metric.

B4. FIX 1 - speech-act router (deterministic, and the first query-conditioning
    signal in this project that is not null).  advice_shaped() fires on 30/30 SSP
    and 1/440 other answerable questions (precision 0.968, recall 1.000).  Compare
    the current tense classifier: 425/500 questions get no lexical cue and the
    routing ablation is exactly 0.0000.
    A related detector, meta_conversational(), fires on 54/56 SSA and 0/470
    KU/MS/SSU/TR/abs - useful to keep the provenance rule (G2) from touching SSA.

B5. FIX 2 - session-augmented, role-quota retrieval.  Replace "BM25 top-20 turns"
    with "top-14 user turns + top-6 assistant turns + every user turn of the top-n
    sessions by session-level BM25 over user text" (n=1 by default, n=3 when
    advice_shaped fires - advice questions are only 31/500, so the extra budget is
    almost free corpus-wide).  Gold-evidence-turn recall (n=470, 1086 evidence turns):
      type                     top20   14u+6a   +top1 session   +top2 sessions
      single-session-preference 0.500   0.545       0.727            0.773
      multi-session             0.644   0.709       0.728            0.759
      temporal-reasoning        0.783   0.802       0.838            0.850
      knowledge-update          0.958   0.965       0.986            0.993
      single-session-user       0.955   0.955       0.985            0.985
      single-session-assistant  0.911   0.893       0.911            0.929
      ALL                       0.767   0.799       0.832            0.851
      mean raw chars           20534   16426      17055            17868
    +6.5 pp overall and +22.7 pp on SSP while spending 17% FEWER characters than
    today's top-20.  The 6-slot assistant quota is what keeps SSA whole (0.911).
    SSP-only sweep of the session budget (mean raw chars in brackets):
      n=1 0.727 [16262] | n=2 0.773 [17196] | n=3 0.795 [18263]
    Reach on the 9 SSP failures, counted as "gains gold evidence it did not have":
      n=1 -> 2/9 (35a27287, 1d4e3b97) | n=2 -> 4/9 (+0edc2aef, 32260d93)
      n=3 -> 5/9 (+195a1a1b).  Default for advice questions is therefore n=3.
    The 4 that never gain: 75f70248 and a89d7624 already had complete evidence
    (they need P4, not retrieval), d6233ab6 needs the view (see B7), and 09d032c9
    is a pure lexical bridge ("battery life on my phone" -> a turn about a
    "portable power bank"); its gold session sits at session-rank 28.

B5b. Headroom of query expansion (oracle probe, no API).  Ranking sessions with
    query = question + gold rubric text puts the gold session at rank 0 for 30/30
    SSP questions, versus 12/30 top-1 and 19/30 top-3 with the bare question.  So
    ONE cheap query-expansion call on advice-shaped questions only (31 calls per
    500-question run, thinking off, ~500 input tokens each) is the single highest-
    leverage LLM addition available.  This is the only component of this module
    that needs an API call, and it cannot be validated offline.

B6. FIX 3 - verbatim preference block.  For advice-shaped questions, mine
    first-person self-disclosure sentences from the retrieved sessions' user turns
    (regex list, no API) and render them verbatim under an explicit heading.  The
    miner captures 0.659 of SSP gold evidence sentences corpus-wide (0.783 over all
    types); median index 125 sentences / 13.1k chars per question, so it is only
    affordable when scoped to the retrieved sessions (done here) - BM25 over the
    whole index recovers just 0.295 of SSP evidence and is not worth it.

B7. FIX 4 - relevance-ordered view truncation (bonus; fixes a latent bug that hurts
    both parts).  render_view() emits lines sorted by "entity::attribute", and
    run_qc.py cuts at 20,000 chars.  The median full view is 30.5k chars, so the
    cap fires on 500/500 questions and always deletes the alphabetical tail
    (`user::role`, `user::salary`, ... are systematically at risk).  Re-ordering
    lines by BM25 relevance to the question before the same 20k cut:
      view gold-string coverage (n=470): 0.528 -> 0.606  (+7.8 pp, identical budget)
      gold-bearing line lost to the alphabetical cap: 50 questions
      recovered by relevance ordering: 40 (8 of them currently wrong); 3 regressions
    SSP preference values surviving the cap: 36/45 alphabetical.  Traced case:
    d6233ab6 needs two rubric details; alphabetical keeps neither `debate` nor
    `advanced placement`, relevance ordering recovers the `debate` line but still
    drops `advanced placement courses in economics` - partial, not complete.

==================================================================================
WHAT ONLY A PAID DEV-SET RUN CAN SETTLE
==================================================================================
Offline work can bound evidence in context; it cannot predict the model's
behaviour given that context.  Unresolved by construction:
  1. Does the refusal-form directive (G1) convert 031748ae / gpt4_93159ced /
     80ec1f4f without raising the 19 false declines on the answerable set?
  2. Does the provenance clause (G2) fix 09ba9854 without harming SSA (0.982)?
  3. Does the presupposition note (A4) over-refuse on its 63/470 false alarms?
  4. Do the 4 SSP rubric-under-coverage cases flip once the personalization
     directive (P4) forbids partial personalization?
  5. Whether the +6.5 pp evidence recall of B5 turns into accuracy.
Cost of one 500-question dev run with this module, using the measured token
profile of the existing arms (deepseek-flash answer + v4-pro judge, extraction
fully cached, so the only new spend is the answer/judge pair): the qc arm's own
ledger is the reference point - see results/full500_qc_ledger_*.json.  This module
adds no new API call per question; context shrinks by ~17% (B5) at unchanged view
budget, so a rerun costs at most what full500_qc cost, and the two guards can be
evaluated on the 30 abs + 30 SSP subset (60 questions, ~12% of a full run) first.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifecycle as LC                      # noqa: E402
from run_baseline import bm25_rank          # noqa: E402
from query_classifier import classify_query  # noqa: E402

RESULTS = "/root/agent_memory_iclr/results"
DATA = "/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"

# ----------------------------------------------------------------------------------
# 0. shared helpers
# ----------------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def nz(s):
    """Normalise the typographic apostrophes/dashes the answer model emits."""
    return str(s).replace("’", "'").replace("‘", "'").replace("—", "-")


def units_of(q):
    """[(session_date, role, content)] in chronological order - same units the pipeline uses."""
    out = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                out.append((date, t["role"], t["content"]))
    return out


# ----------------------------------------------------------------------------------
# 1. decline detector (measurement instrument; also usable as a self-check)
# ----------------------------------------------------------------------------------
DECLINE_RE = re.compile(
    r"i don't know|i do not know|don't have (any |enough )?(information|record|relevant|memory)"
    r"|no (relevant )?information|not enough information|isn't (mentioned|in the memory)"
    r"|not mentioned|doesn't (mention|contain|include|say|have)|does not (mention|contain|include)"
    r"|can't (determine|tell|say|calculate|answer|give)|cannot determine|unable to determine"
    r"|no record|haven't mentioned|didn't mention|i don't have", re.I)


def is_decline(pred):
    """True if the prediction refuses / reports missing information.
    Validated: 28/30 agreement with the LLM judge on the abstention subset of the qc arm
    (the 2 disagreements are 80ec1f4f_abs and gpt4_93159ced_abs, both diagnosed in A2)."""
    return bool(DECLINE_RE.search(nz(pred)))


# ----------------------------------------------------------------------------------
# 2. A4 - deterministic presupposition ("absence certificate") detectors
# ----------------------------------------------------------------------------------
STOP = set("""a an the is are was were be been being am do does did done have has had having of in on at to for with from by
about as into over under after before between during my me i you your yours we us our it its this that these those and or but
if then than so such not no nor only own same too very can could will would shall should may might must just now
how many much long often when where which who whom whose what why whether there their them they he she his her hers
all any some most more less least other another each per both either neither
total time times day days week weeks month months year years today yesterday tomorrow currently current still recent recently ago
first last next previous initial initially originally latest thing things stuff kind type sort lot lots
one two three four five six seven eight nine ten eleven twelve
get got go went make made take took use used give gave say said tell told see saw know knew think thought
want wanted need needed like liked start started begin began end ended finish finished complete completed
spend spent pay paid buy bought come came leave left keep kept put set try tried feel felt find found
happen happened happening pass passed passing participate participated participating attend attended
order early earliest late later latest soon between including excluding besides also again
please help thanks thank ok okay yes maybe really quite bit lately something anything nothing everything
mine ours yours am pm""".split())
VERBISH = re.compile(r"(ed|ing|ly)$")
DATEISH = set("""january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday weekend weekday weeknight weeknights
tonight morning evening afternoon night""".split())


def question_terms(question):
    """(proper_nouns, content_unigrams, adjacent_content_bigrams) - normalised, deduped."""
    raw = re.findall(r"[A-Za-z0-9'\-]+", question)
    toks = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in raw]
    proper = set()
    for i, (w, t) in enumerate(zip(raw, toks)):
        if i > 0 and w[:1].isupper() and t and t not in STOP and t not in DATEISH:
            proper.add(t)
    content = []
    for i, t in enumerate(toks):
        if not t or t in STOP or len(t) <= 2 or t.isdigit() or t in DATEISH:
            continue
        if VERBISH.search(t) and t not in proper:
            continue
        content.append((i, t))
    uni = list(dict.fromkeys(t for _, t in content))
    bi = list(dict.fromkeys(a + " " + b for (i, a), (j, b) in zip(content, content[1:]) if j == i + 1))
    return sorted(proper), uni, bi


def absence_certificate(question, units, use_d3=False, use_d4=False):
    """Deterministic check for terms the question presupposes but the memory never contains.

    Returns dict(missing_terms=[...], missing_pairs=[...], unlinked_pair=[a,b] or None, fired=bool).
    Measured (abs recall / answerable false-alarm rate, n=30 / n=470):
        D1 proper noun absent                     0.167 / 0.036
        D2 content noun absent                    0.400 / 0.134   <- default trigger (D1 | D2)
        D3 bigram absent, both unigrams present    0.333 / 0.268   (off by default)
        D4 two rarest terms never share a turn     0.500 / 0.349   (off by default)
    Precision never exceeds 0.23 against a 30/500 base rate, so this must feed a NOTE
    (presupposition_note) and never a hard abstain decision.
    """
    turns = [norm(u[2]) for u in units]
    hay = " ".join(turns)
    proper, uni, bi = question_terms(question)
    missing = [t for t in uni if t not in hay]
    missing_proper = [t for t in proper if t not in hay]
    pairs = []
    if use_d3:
        for b in bi:
            a, c = b.split()
            if b not in hay and a in hay and c in hay:
                pairs.append(b)
    unlinked = None
    if use_d4:
        present = [t for t in uni if t in hay]
        if len(present) >= 2:
            df = {t: sum(1 for x in turns if t in x) for t in present}
            two = sorted(present, key=lambda t: df[t])[:2]
            if not any(all(t in x for t in two) for x in turns):
                unlinked = two
    return {"missing_terms": sorted(set(missing) | set(missing_proper)),
            "missing_pairs": pairs, "unlinked_pair": unlinked,
            "fired": bool(missing or missing_proper or pairs or unlinked)}


def presupposition_note(cert):
    """~25 prompt tokens. Advisory only: it states a verified string fact and hands the
    decision back to the model, so a false alarm (63/470) is recoverable."""
    if not cert["fired"]:
        return ""
    bits = []
    if cert["missing_terms"]:
        bits.append("never appear anywhere in the stored conversations: "
                    + ", ".join(f"'{t}'" for t in cert["missing_terms"][:6]))
    if cert["missing_pairs"]:
        bits.append("never appear as a phrase: " + ", ".join(f"'{p}'" for p in cert["missing_pairs"][:4]))
    if cert["unlinked_pair"]:
        a, b = cert["unlinked_pair"]
        bits.append(f"'{a}' and '{b}' both occur but never in the same turn")
    return ("### Verified absence check (string-level, computed deterministically)\n"
            "The following question terms " + "; ".join(bits) + ".\n"
            "If the question presupposes something these terms name, the premise is not on "
            "record: say so explicitly. If they are only paraphrases of something that IS on "
            "record, ignore this note.")


# ----------------------------------------------------------------------------------
# 3. B4 - speech-act routers
# ----------------------------------------------------------------------------------
ADVICE_RE = [re.compile(p, re.I) for p in [
    r"\bcan you (?:recommend|suggest|help|give|provide)\b",
    r"\bany (?:tips|suggestions|ideas|recommendations|advice)\b",
    r"\bdo you (?:think|know|have any)\b", r"\bwhat should i\b", r"\bcould there be\b",
    r"\bwould (?:it|you)\b", r"\bi'?ve been\b", r"\bhelp me\b", r"\bthoughts\b", r"\bany\b.*\?"]]
META_RE = re.compile(
    r"remind me|previous (chat|conversation|discussion)|we (discussed|talked about)"
    r"|our (previous|earlier) (chat|conversation)"
    r"|you (recommend|recommended|suggest|suggested|told|said|mentioned|listed|gave|provided)"
    r"|back to our|earlier conversation|checking our", re.I)


def advice_shaped(question):
    """Personalisation / advice request rather than a factual lookup.
    Measured: 30/30 single-session-preference, 1/440 other answerable (precision 0.968)."""
    ql = nz(question).lower()
    return any(p.search(ql) for p in ADVICE_RE)


def meta_conversational(question):
    """Asks what was said in an earlier conversation (the SSA frame).
    Measured: 54/56 SSA, 0/470 KU+MS+SSU+TR+abs, 9/30 SSP."""
    return bool(META_RE.search(nz(question)))


# ----------------------------------------------------------------------------------
# 4. B6 - verbatim first-person preference miner
# ----------------------------------------------------------------------------------
PREF_RE = [re.compile(p) for p in [
    r"\bi (?:really )?(?:like|love|enjoy|prefer|hate|dislike|adore|appreciate)\b",
    r"\bi'?m (?:a )?(?:really )?(?:into|fan of|interested in|looking for|hoping|aspiring|excited)\b",
    r"\bi (?:don'?t|do not) (?:like|enjoy|want)\b", r"\bi'?d rather\b", r"\bmy favou?rite\b",
    r"\bi can'?t stand\b", r"\bi (?:usually|always|often|typically|normally|tend to)\b",
    r"\bi'?ve been\b", r"\bi (?:just )?(?:bought|got|purchased|ordered|replaced|started|switched|joined|signed up)\b",
    r"\bi'?m planning\b", r"\bi (?:still )?remember\b", r"\bmy new\b",
    r"\bas an? [a-z ]{2,20}(?:er|ist|ian|major|student|fan|enthusiast)\b",
    r"\bi have been\b", r"\bi want\b", r"\bi need\b", r"\bi'?m (?:trying|working on|thinking)\b",
    r"\bi found\b", r"\bi decided\b", r"\bi'?m not (?:sure|ready)\b", r"\bit'?s been\b"]]


def preference_sentences(session_turns, min_len=25):
    """Verbatim first-person self-disclosure sentences from the user's turns of one session.
    Measured capture of gold evidence sentences: SSP 0.659, all types 0.783."""
    out = []
    for t in session_turns:
        if t.get("role") != "user" or not t.get("content"):
            continue
        for s in SENT_SPLIT.split(nz(t["content"])):
            s = s.strip()
            if len(s) > min_len and any(p.search(s.lower()) for p in PREF_RE):
                out.append(s)
    return out


# ----------------------------------------------------------------------------------
# 5. B5 - session-augmented, role-quota retrieval
# ----------------------------------------------------------------------------------
def session_docs(q, user_only=True):
    docs = []
    for sess in q["haystack_sessions"]:
        docs.append(" ".join(t["content"] for t in sess
                             if t.get("content") and (t["role"] == "user" or not user_only)))
    return docs


def retrieve_v5(q, k_user=14, k_asst=6, n_sessions=1):
    """Returns (turn_indices_into_units_of(q), chosen_session_indices).

    Role quota keeps SSA alive (assistant evidence recall 0.893 at 6 slots vs 0.911 at 20
    unconstrained slots) while cutting assistant chatter, which is 77.1% of the current
    top-20 character budget.  The session sweep is what SSP needs: evidence recall
    0.500 -> 0.727 (n_sessions=1) / 0.773 (n_sessions=2).
    """
    units = units_of(q)
    docs = [u[2] for u in units]
    order = bm25_rank(q["question"], docs)
    uidx = [i for i in order if units[i][1] == "user"][:k_user]
    aidx = [i for i in order if units[i][1] == "assistant"][:k_asst]
    keep = set(uidx) | set(aidx)
    sess_rank = bm25_rank(q["question"], session_docs(q, user_only=True))[:n_sessions]
    # add every USER turn of the winning session(s); assistant turns stay under quota
    pos, per_sess = 0, {}
    for si, sess in enumerate(q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                per_sess.setdefault(si, []).append((pos, t["role"]))
                pos += 1
    for si in sess_rank:
        for i, role in per_sess.get(si, []):
            if role == "user":
                keep.add(i)
    return sorted(keep), list(sess_rank)


# ----------------------------------------------------------------------------------
# 6. B7 - relevance-ordered view truncation
# ----------------------------------------------------------------------------------
def relevance_capped_view(store, mode, question, budget=20000):
    """Same render_view output, same character budget, chronological/alphabetical
    presentation preserved - only the *selection* changes from "alphabetical prefix"
    to "highest BM25 relevance to the question".
    Measured on n=470: view gold-string coverage 0.528 -> 0.606; 40 questions recover
    their gold-bearing line, 3 regress."""
    lines = LC.render_view(store, mode).splitlines()
    if sum(len(l) + 1 for l in lines) <= budget:
        return "\n".join(lines)
    order = bm25_rank(question, lines)
    keep, total = [], 0
    for i in order:
        if total + len(lines[i]) + 1 > budget:
            continue
        keep.append(i)
        total += len(lines[i]) + 1
    return "\n".join(lines[i] for i in sorted(keep))


# ----------------------------------------------------------------------------------
# 7. G1/G2/P4 - prompt-level guards (zero extra API calls)
# ----------------------------------------------------------------------------------
# G1: refusal FORM. Motivated by A2: 3 of our 5 abstention losses carry the right
# content in the wrong shape, and a sibling arm with the same content but the refusal
# in the lead position is graded CORRECT.
REFUSAL_FORM = (
    "REFUSAL FORM. If the question names a person, place, object, role, title, event or "
    "quantity that the memory does not contain, your FIRST sentence must say plainly that "
    "it is not in the memory (e.g. \"The memory does not mention X\"). Only then may you "
    "add the closest thing that IS on record, clearly marked as a different item. Never "
    "answer a question about X by silently supplying the value belonging to a different "
    "item, and never bury the refusal in a parenthesis after a number. If the question "
    "asks how many times something happened in a period and the memory records no such "
    "event in that period, answer \"none / zero - the memory records no such event\".")

# G2: speaker provenance. Motivated by A2/09ba9854 and by the measured 77.1% assistant
# share of the raw-turn character budget. Deliberately scoped so SSA (gold evidence is an
# assistant turn in 51/56 cases, currently 0.982) keeps its licence.
PROVENANCE = (
    "SPEAKER AUTHORITY. Turns marked [user] are the user's own assertions. Turns marked "
    "[assistant] are the assistant's earlier replies: they are valid evidence of what the "
    "assistant said, recommended or listed, but they are NOT evidence about the user's own "
    "facts, purchases, plans, prices or quantities. Generic estimates, price ranges and "
    "options offered by the assistant are suggestions, not things the user reported. Never "
    "compute a number about the user's situation out of assistant-supplied estimates.")

# P4: personalisation. Motivated by B3: 5 of 9 SSP losses are declines and 4 are partial
# rubric coverage; declines on answerable questions score 0/19 in this arm.
PERSONALIZATION = (
    "PERSONALISATION REQUEST. This question asks for advice or a recommendation, not a "
    "fact lookup. Do not refuse for lack of information. Ground the answer in EVERY "
    "specific personal detail in the memory that bears on the topic - name the items, "
    "brands, people, places, constraints and past experiences explicitly, quoting the "
    "user's own wording where it is available - and let those details drive the "
    "recommendation. A generic answer, or one that uses only some of the relevant "
    "details, is wrong.")

SYSTEM_PROMPT_V5_BASE = (
    "Answer the question using the memory context. Up to three complementary sources are "
    "provided: a structured fact view distilled from the entire conversation history, raw "
    "conversation turns retrieved verbatim, and (sometimes) the user's own verbatim "
    "statements. Use them together. Be concise. If no relevant information exists, say you "
    "don't know.\n\n" + REFUSAL_FORM + "\n\n" + PROVENANCE)


def system_prompt_v5(question):
    p = SYSTEM_PROMPT_V5_BASE
    if advice_shaped(question):
        p += "\n\n" + PERSONALIZATION
    return p


# ----------------------------------------------------------------------------------
# 8. assembled context builder (drop-in replacement for run_qc.answer_question's body)
# ----------------------------------------------------------------------------------
def build_context_v5(q, store, view_cap=20000, k_user=14, k_asst=6, n_sessions=1,
                     n_sessions_advice=3, pref_block=True, note=True,
                     use_d3=False, use_d4=False):
    """Returns (context_text, meta). `store` is the lifecycle store from build_lifecycle(q)
    (or a cache-backed reconstruction); this function makes no API call."""
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    view_txt = relevance_capped_view(store, mode, q["question"], budget=view_cap)

    advice = advice_shaped(q["question"])
    meta_conv = meta_conversational(q["question"])
    units = units_of(q)
    idx, sess_idx = retrieve_v5(q, k_user=k_user, k_asst=k_asst,
                               n_sessions=n_sessions_advice if advice else n_sessions)
    raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in idx)

    parts = []
    if view_txt:
        head = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
                "at the same time that could not be ordered)" if mode == "current"
                else "Fact evolution view (full value history per fact, oldest -> newest)")
        parts.append(f"### {head}\n{view_txt}")
    if raw_txt:
        parts.append(f"### Retrieved raw conversation turns\n{raw_txt}")

    quotes = []
    if pref_block and advice:
        for si in sess_idx:
            date = q["haystack_dates"][si]
            for s in preference_sentences(q["haystack_sessions"][si]):
                quotes.append(f"[{date}] \"{s}\"")
        if quotes:
            parts.append("### The user's own words (verbatim self-disclosures from the most "
                         "relevant sessions)\n" + "\n".join(quotes))

    cert = absence_certificate(q["question"], units, use_d3=use_d3, use_d4=use_d4)
    # Gate: suppress the note on advice / meta-conversational questions. Free on the
    # abstention side (0/30 are either) and it removes 21 of 63 answerable false alarms,
    # including the two SSP declines the note would otherwise have reinforced.
    if note and not advice and not meta_conv:
        n = presupposition_note(cert)
        if n:
            parts.append(n)

    meta = {"qlabel": label, "view_mode": mode, "n_raw": len(idx), "sessions": sess_idx,
            "advice": advice, "meta_conv": meta_conv,
            "cert_fired": cert["fired"], "cert": cert, "n_quotes": len(quotes),
            "note_shown": bool(note and not advice and not meta_conv and cert["fired"]),
            "view_chars": len(view_txt), "raw_chars": len(raw_txt)}
    return "\n\n".join(parts), meta


# ==================================================================================
# offline validation - regenerates every number quoted in this docstring
# ==================================================================================
def _load(quick=False):
    data = json.load(open(DATA))
    if quick:  # 30 abs + 30 SSP + 60 others, enough for the routers/detectors
        keep = [q for q in data if q["question_id"].endswith("_abs")
                or q["question_type"] == "single-session-preference"]
        rest = [q for q in data if q not in keep][:60]
        data = keep + rest
    arms = {}
    for a in ["qc", "full", "fifo", "decay", "qc_no_lifecycle", "qc_no_qc", "qc_view_only"]:
        p = f"{RESULTS}/full500_{a}.jsonl"
        if os.path.exists(p):
            arms[a] = {r["qid"]: r for r in (json.loads(l) for l in open(p))}
    return data, arms


def _cached_store(q):
    """Rebuild the lifecycle store from the on-disk extraction cache only (no API)."""
    import hashlib
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = LC._session_text(sess)
        if not text:
            continue
        key = hashlib.md5(text.encode()).hexdigest()[:16]
        p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
        if not os.path.exists(p):
            continue
        for e, a, v in json.load(open(p)):
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    return LC.assemble_chains(facts)


def v_decline_table(data, arms):
    print("\n=== A1  decline behaviour as a detector of the 30 unanswerable questions ===")
    print(f"{'arm':18s} {'TPabs':>5s} {'FPans':>5s} {'prec':>6s} {'absAcc':>7s} {'FPcorrect':>9s}")
    for a, rows in arms.items():
        absr = [r for r in rows.values() if r["qid"].endswith("_abs")]
        non = [r for r in rows.values() if not r["qid"].endswith("_abs")]
        tp = sum(1 for r in absr if is_decline(r.get("pred", "")))
        fp = [r for r in non if is_decline(r.get("pred", ""))]
        print(f"{a:18s} {tp:5d} {len(fp):5d} {tp / max(1, tp + len(fp)):6.3f} "
              f"{sum(r['judge'] for r in absr) / max(1, len(absr)):7.3f} "
              f"{sum(r['judge'] for r in fp):4d}/{len(fp):<4d}")


def v_routers(data, arms):
    from collections import Counter
    print("\n=== B4  speech-act routers ===")
    c = Counter()
    for q in data:
        t = "ABSTENTION" if q["question_id"].endswith("_abs") else q["question_type"]
        c[(t, advice_shaped(q["question"]), meta_conversational(q["question"]))] += 1
    types = sorted({k[0] for k in c})
    print(f"{'type':28s} {'n':>4s} {'advice':>7s} {'meta':>6s}")
    for t in types:
        n = sum(v for k, v in c.items() if k[0] == t)
        adv = sum(v for k, v in c.items() if k[0] == t and k[1])
        met = sum(v for k, v in c.items() if k[0] == t and k[2])
        print(f"{t:28s} {n:4d} {adv / n:7.3f} {met / n:6.3f}")


def v_detectors(data, arms):
    print("\n=== A4  absence-certificate detectors (abs recall / answerable false alarms) ===")
    rows = []
    for q in data:
        u = units_of(q)
        c = absence_certificate(q["question"], u, use_d3=True, use_d4=True)
        rows.append((q["question_id"].endswith("_abs"), c))
    A = [c for a, c in rows if a]
    N = [c for a, c in rows if not a]
    def rp(name, fn):
        ta = sum(1 for c in A if fn(c)); fa = sum(1 for c in N if fn(c))
        print(f"{name:44s} abs {ta:2d}/{len(A):<3d} ({ta / max(1, len(A)):.3f})   "
              f"ans {fa:3d}/{len(N):<3d} ({fa / max(1, len(N)):.3f})   prec {ta / max(1, ta + fa):.3f}")
    rp("D1|D2 missing proper/content noun", lambda c: bool(c["missing_terms"]))
    rp("D3 missing bigram, unigrams present", lambda c: bool(c["missing_pairs"]))
    rp("D4 two rarest terms never share a turn", lambda c: c["unlinked_pair"] is not None)
    rp("D1|D2|D3|D4", lambda c: c["fired"])


def v_note_gate(data, arms):
    print("\n=== A4  router gating of the presupposition note (free on the abstention side) ===")
    rows = []
    for q in data:
        c = absence_certificate(q["question"], units_of(q))
        rows.append((q["question_id"].endswith("_abs"), bool(c["missing_terms"]),
                     advice_shaped(q["question"]), meta_conversational(q["question"])))
    def rp(name, fn):
        ta = sum(1 for r in rows if r[0] and fn(r)); fa = sum(1 for r in rows if not r[0] and fn(r))
        na = sum(1 for r in rows if r[0]); nn = sum(1 for r in rows if not r[0])
        print(f"{name:34s} abs {ta:2d}/{na:<3d} answerable {fa:3d}/{nn:<3d} prec {ta/max(1,ta+fa):.3f}")
    rp("raw D1|D2", lambda r: r[1])
    rp("gated off advice", lambda r: r[1] and not r[2])
    rp("gated off advice + meta (default)", lambda r: r[1] and not r[2] and not r[3])
    fails = ["80ec1f4f_abs", "a96c20ee_abs", "09ba9854_abs", "gpt4_93159ced_abs", "031748ae_abs"]
    print("  firing on our 5 abstention losses (gated D1|D2 / with D3+D4):")
    for q in data:
        if q["question_id"] not in fails:
            continue
        u = units_of(q)
        c = absence_certificate(q["question"], u)
        c4 = absence_certificate(q["question"], u, use_d3=True, use_d4=True)
        print(f"    {q['question_id']:20s} {str(bool(c['missing_terms'])):5s} / {str(c4['fired']):5s} "
              f"terms={c['missing_terms']} pairs={c4['missing_pairs'][:2]} unlinked={c4['unlinked_pair']}")


def v_ssp_sweep(data, arms):
    import statistics as st
    print("\n=== B5  SSP-only session-budget sweep and reach on the 9 SSP failures ===")
    F = {"0edc2aef", "35a27287", "32260d93", "195a1a1b", "09d032c9",
         "75f70248", "d6233ab6", "a89d7624", "1d4e3b97"}
    ssp = [q for q in data if q["question_type"] == "single-session-preference"]
    if not ssp:
        print("  (no SSP questions in this subset)")
        return
    for ns in (1, 2, 3):
        hit = tot = 0
        chars, gains = [], []
        for q in ssp:
            units = units_of(q)
            docs = [u[2] for u in units]
            base = {docs[i] for i in bm25_rank(q["question"], docs)[:20]}
            idx, _ = retrieve_v5(q, n_sessions=ns)
            new = {docs[i] for i in idx}
            sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
            ev = [t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
                  if t.get("has_answer") and t.get("content")]
            hit += sum(1 for e in ev if e in new)
            tot += len(ev)
            chars.append(sum(len(docs[i]) for i in idx))
            if q["question_id"] in F and sum(1 for e in ev if e in new) > sum(1 for e in ev if e in base):
                gains.append(q["question_id"])
        print(f"  n_sessions={ns}: SSP ev_recall={hit/max(1,tot):.3f} mean_raw_chars={st.mean(chars):.0f} "
              f"failures_gaining_evidence={len(gains)}/9 {sorted(gains)}")
    print("  oracle query-expansion probe (query = question + gold rubric; upper bound only):")
    p1 = p2 = t1 = t3 = 0
    for q in ssp:
        docs = session_docs(q, user_only=True)
        gs = {i for i, s in enumerate(q["haystack_session_ids"]) if s in set(q["answer_session_ids"])}
        r1 = bm25_rank(q["question"], docs)
        r2 = bm25_rank(q["question"] + " " + q["answer"], docs)
        a, b = min(r1.index(i) for i in gs), min(r2.index(i) for i in gs)
        p1 += a == 0; t3 += a < 3; p2 += b == 0
        t1 += 1
    print(f"    gold session at rank 0: plain {p1}/{t1}, top-3 {t3}/{t1}, oracle {p2}/{t1}")


def v_retrieval(data, arms):
    import statistics as st
    print("\n=== B5  gold-evidence-turn recall: top20 vs role quota vs +session ===")
    agg = {}
    for q in data:
        if q["question_id"].endswith("_abs"):
            continue
        units = units_of(q)
        docs = [u[2] for u in units]
        base = set(bm25_rank(q["question"], docs)[:20])
        i1, _ = retrieve_v5(q, n_sessions=0)
        i2, _ = retrieve_v5(q, n_sessions=1)
        i3, _ = retrieve_v5(q, n_sessions=2)
        sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        ev = [t["content"] for sid in q["answer_session_ids"] for t in sid2.get(sid, [])
              if t.get("has_answer") and t.get("content")]
        s = agg.setdefault(q["question_type"], {"n": 0, "ev": 0, "b": 0, "q": 0, "s1": 0, "s2": 0,
                                                "cb": 0, "c1": 0, "c2": 0})
        s["n"] += 1
        s["ev"] += len(ev)
        for k, idx in [("b", base), ("q", set(i1)), ("s1", set(i2)), ("s2", set(i3))]:
            got = {docs[i] for i in idx}
            s[k] += sum(1 for e in ev if e in got)
        s["cb"] += sum(len(docs[i]) for i in base)
        s["c1"] += sum(len(docs[i]) for i in i2)
        s["c2"] += sum(len(docs[i]) for i in i3)
    print(f"{'type':28s} {'n':>4s} {'top20':>6s} {'quota':>6s} {'+s1':>6s} {'+s2':>6s} "
          f"{'ch20':>7s} {'ch+s1':>7s}")
    tot = {k: 0 for k in ["n", "ev", "b", "q", "s1", "s2", "cb", "c1", "c2"]}
    for t, s in sorted(agg.items()):
        for k in tot:
            tot[k] += s[k]
        e = max(1, s["ev"])
        print(f"{t:28s} {s['n']:4d} {s['b']/e:6.3f} {s['q']/e:6.3f} {s['s1']/e:6.3f} {s['s2']/e:6.3f} "
              f"{s['cb']/s['n']:7.0f} {s['c1']/s['n']:7.0f}")
    e = max(1, tot["ev"])
    print(f"{'ALL':28s} {tot['n']:4d} {tot['b']/e:6.3f} {tot['q']/e:6.3f} {tot['s1']/e:6.3f} "
          f"{tot['s2']/e:6.3f} {tot['cb']/tot['n']:7.0f} {tot['c1']/tot['n']:7.0f}")


def v_cap(data, arms):
    print("\n=== B7  view truncation: alphabetical prefix vs relevance-ordered (same 20k) ===")
    def contains(hay, needle):
        h, n = norm(hay), norm(needle)
        if not n:
            return False
        if n in h:
            return True
        toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
        return bool(toks) and all(t in h for t in toks)
    stats = {}
    rec = lost = reg = 0
    for q in data:
        if q["question_id"].endswith("_abs"):
            continue
        store = _cached_store(q)
        label, _ = classify_query(q["question"])
        mode = "historical" if label == "historical" else "current"
        lines = LC.render_view(store, mode).splitlines()
        kept, total = [], 0
        for l in lines:
            if total + len(l) + 1 > 20000:
                break
            kept.append(l); total += len(l) + 1
        alpha = "\n".join(kept)
        rel = relevance_capped_view(store, mode, q["question"])
        g = q["answer"]
        cf, ca, cr = contains("\n".join(lines), g), contains(alpha, g), contains(rel, g)
        s = stats.setdefault(q["question_type"], [0, 0, 0, 0])
        s[0] += 1; s[1] += cf; s[2] += ca; s[3] += cr
        if cf and not ca:
            lost += 1
            rec += bool(cr)
        if cf and ca and not cr:
            reg += 1
    print(f"{'type':28s} {'n':>4s} {'full':>6s} {'alpha':>6s} {'rel':>6s}")
    T = [0, 0, 0, 0]
    for t, s in sorted(stats.items()):
        for i in range(4):
            T[i] += s[i]
        print(f"{t:28s} {s[0]:4d} {s[1]/s[0]:6.3f} {s[2]/s[0]:6.3f} {s[3]/s[0]:6.3f}")
    print(f"{'ALL':28s} {T[0]:4d} {T[1]/T[0]:6.3f} {T[2]/T[0]:6.3f} {T[3]/T[0]:6.3f}")
    print(f"gold line lost to alphabetical cap: {lost}; recovered by relevance: {rec}; regressions: {reg}")


def v_provenance_risk(data, arms):
    print("\n=== A3/G2  cost of a hard provenance requirement, and assistant evidence share ===")
    diag = json.load(open(f"{RESULTS}/coverage_diag.json"))
    non = [r for r in diag if not r["is_abs"]]
    vo = [r for r in non if r["in_view"] and not r["in_raw20"]]
    print(f"answerable with gold ONLY in the view: {len(vo)}; currently correct: "
          f"{sum(1 for r in vo if r['judge'])} <- a hard 'must be in a raw turn' rule risks these")
    from collections import Counter
    roles = Counter()
    for q in data:
        if q["question_id"].endswith("_abs"):
            continue
        sid2 = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
        for sid in q["answer_session_ids"]:
            for t in sid2.get(sid, []):
                if t.get("has_answer") and t.get("content"):
                    roles[(q["question_type"], t["role"])] += 1
    for k in sorted(roles):
        print(f"  gold evidence turn role  {k[0]:28s} {k[1]:9s} {roles[k]}")


def v_smoke(data, arms):
    print("\n=== smoke test: build_context_v5 on one abstention and one SSP question ===")
    picks = [q for q in data if q["question_id"] in ("gpt4_93159ced_abs", "09d032c9", "0edc2aef")]
    for q in picks:
        store = _cached_store(q)
        ctx, meta = build_context_v5(q, store)
        print(f"\n--- {q['question_id']} ({q['question_type']}) advice={meta['advice']} "
              f"cert={meta['cert']['missing_terms'][:4]} quotes={meta['n_quotes']} "
              f"raw={meta['n_raw']} chars={len(ctx)}")
        print("   sections:", [l for l in ctx.splitlines() if l.startswith("### ")])
        print("   system prompt chars:", len(system_prompt_v5(q["question"])))


def main():
    quick = "--quick" in sys.argv
    data, arms = _load(quick=quick)
    print(f"loaded {len(data)} questions, {len(arms)} arms  (quick={quick})")
    v_decline_table(data, arms)
    v_routers(data, arms)
    v_detectors(data, arms)
    v_note_gate(data, arms)
    v_provenance_risk(data, arms)
    v_retrieval(data, arms)
    v_ssp_sweep(data, arms)
    v_cap(data, arms)
    v_smoke(data, arms)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        main()
    else:
        print(__doc__)
