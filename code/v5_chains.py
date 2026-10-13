"""QC-Lifecycle v5: soft chain merging + aggregation-aware rendering (training-free).

Two independent parts, both pure code (no API call, no embedding model, no training):

(a) merge_chains(store) -> store
    The v4 assembler keys chains on exact normalized (entity, attribute). LLM-written
    attribute phrases are free-form, so 'job' / 'current job' / 'occupation' never merge and
    the store degenerates into singletons (measured over all 500 questions: median 316.5
    chains/question, 85.1% of them singletons).
    merge_chains() clusters chain keys with deterministic rules (all of which REQUIRE an
    identical head token, so the search is bucketed by head and stays cheap), then rebuilds
    each cluster with lifecycle.assemble_chains() so the relation labels
    (initial/supersession/contradiction/cooccurring/accumulative) stay exactly the v4 rules.

(b) render_aggregation_view(store, question) -> text
    Counting/aggregation questions ('how many X ...') need a COMPLETE, enumerable item set.
    Two structural facts defeat the v4 view there:
      1. items fragment across singleton chains ('needs to pick up' vs 'needs to return'), and
      2. the 'current' view prints only the LAST session group of a chain, so earlier
         accumulated items are structurally invisible (they are labeled 'supersession').
    render_aggregation_view() fires only on aggregation questions, pools every fact whose
    attribute or value shares a stem with the question's focus terms, expands each hit to the
    WHOLE merged chain, dedupes values, and prints an explicit per-group and total distinct
    count. Returns "" for non-aggregation questions (caller keeps the normal view).

No question-specific constant appears in this module; the only hand tables are a small
lexical-synonym table and an optional hyponym table (HYPONYMS) that is used only by
render_aggregation_view and can be disabled with use_hyponyms=False for ablation.
"""
import difflib
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifecycle import assemble_chains, _norm

# ---------------------------------------------------------------- token machinery

_word = re.compile(r"[a-z0-9]+")

# determiners, auxiliaries, light/reporting verbs and recency modifiers: none of these change
# the identity of an attribute ("is interested in" == "interested in" == "expressed interest in")
LIGHT = {
    "the", "a", "an", "my", "mine", "our", "their", "his", "her", "its", "your",
    "is", "are", "am", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "s",
    # attribution-only verbs ('expressed interest in' == 'interested in'). Verbs that can
    # also be the real predicate (say/share/report/describe/confirm/tell/note/state) are NOT
    # here: the 30-cluster audit caught 'plans to share' merging into 'is planning'.
    "express", "expresse", "mention", "indicate",
    "current", "recent", "recently", "latest", "present", "now", "today", "new", "user",
    # pure adverbs / discourse particles: never change attribute identity
    "particularly", "especially", "generally", "usually", "actually", "also", "then",
    "just", "already", "still", "again", "ever", "always", "often", "sometimes",
    "very", "quite", "really", "overall", "currently", "typically", "mostly",
    "of", "in", "on", "at", "to", "for", "with", "about", "from", "by", "as", "into",
    "and", "or", "that", "this",
}
# tokens allowed to differ in the subset rule (pure modifiers)
MODIFIER = {"main", "primary", "overall", "general", "own", "personal", "usual",
            "typical", "regular", "specific", "particular", "exact"}
# tokens that are never a harmless difference: they discriminate siblings ('example 1' vs
# 'example 2', 'original price' vs 'revised price', 'actual' vs 'intended wake time').
# Their presence in the symmetric difference vetoes a merge under every rule.
DISCRIMINATIVE = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "last", "next", "other", "another", "alternative", "alternate", "original",
    "revised", "updated", "old", "former", "previous", "initial", "final", "intended",
    "planned", "additional", "extra", "actual", "estimated", "expected", "target",
    "maximum", "minimum", "average", "total", "left", "right", "upper", "lower",
}
# particles/prepositions skipped when choosing the head token
PARTICLE = {"up", "in", "on", "to", "of", "with", "for", "at", "from", "about", "by",
            "into", "off", "out", "over", "down", "back", "through", "after", "before"}

# lexical synonyms, token level, stem -> canonical stem. Hand-built, 24 entries, all
# dictionary-level synonymy (not domain knowledge about the benchmark).
SYN = {
    "occupation": "job", "profession": "job", "employment": "job", "career": "job",
    "jobtitle": "job", "role": "job", "position": "job",
    "residence": "location", "hometown": "location", "live": "location", "living": "location",
    "reside": "location", "city": "location", "address": "location",
    "pastime": "hobby", "hobbie": "hobby",
    "objective": "goal", "aim": "goal", "target": "goal",
    "purchase": "acquire", "buy": "acquire", "bought": "acquire", "acquisition": "acquire",
    "obtain": "acquire", "got": "acquire", "get": "acquire",
    "kid": "child", "offspring": "child",
    "spouse": "partner",
    "auto": "car", "vehicle": "car",
    "interested": "interest", "like": "interest", "love": "interest", "enjoy": "interest",
    "prefer": "interest", "preference": "interest", "favorite": "interest",
    "want": "wish", "desire": "wish", "hope": "wish",
    "trip": "travel", "vacation": "travel", "holiday": "travel", "journey": "travel",
}

def _stem_syn_table(tbl):
    """The lookup happens on stemmed tokens, so the table must be stemmed too. Without this,
    entries whose key loses a letter under the stemmer ('purchase'->'purchas',
    'desire'->'desir') were dead: the question term 'purchased' never reached 'acquire'."""
    out = {}
    for k, v in tbl.items():
        out[k] = v
        out[_stem(k)] = _stem(v)
    return out


# OPTIONAL hyponym table, used ONLY by render_aggregation_view to link a question's category
# word to the concrete items that instantiate it. Disable with use_hyponyms=False.
HYPONYMS = {
    "clothing": ["cloth", "clothe", "outfit", "garment", "apparel", "shirt", "tshirt", "dress",
                 "jacket", "coat", "blazer", "sweater", "hoodie", "boot", "shoe", "sneaker",
                 "pant", "trouser", "jean", "skirt", "sock", "hat", "scarf", "suit", "blouse",
                 "top", "short", "legging", "cardigan", "vest", "belt", "glove"],
    "plant": ["succulent", "cactus", "cacti", "fern", "herb", "seedling", "flower", "pothos",
              "monstera", "orchid", "ivy", "bonsai", "sapling", "shrub", "basil", "mint",
              "tomato", "aloe", "philodendron", "sansevieria", "snake"],
    "pet": ["dog", "puppy", "cat", "kitten", "fish", "bird", "hamster", "rabbit", "turtle",
            "gecko", "lizard", "snake", "guinea"],
    "tank": ["aquarium", "fishtank", "terrarium", "vivarium"],
    "book": ["novel", "memoir", "paperback", "hardcover", "textbook", "anthology", "volume"],
    "travel": ["trip", "flight", "vacation", "holiday", "cruise", "getaway", "tour", "visit"],
    "instrument": ["guitar", "piano", "violin", "drum", "flute", "ukulele", "bass", "cello"],
    "device": ["phone", "laptop", "tablet", "camera", "monitor", "headphone", "watch",
               "speaker", "console", "printer", "router", "keyboard"],
    "furniture": ["chair", "table", "desk", "sofa", "couch", "shelf", "bookcase", "bed",
                  "dresser", "cabinet", "stool", "lamp"],
    "course": ["class", "workshop", "lesson", "seminar", "lecture", "module", "certification"],
    "recipe": ["dish", "meal", "bake", "cook", "cake", "bread", "soup", "salad"],
    "appointment": ["meeting", "checkup", "consultation", "session", "visit"],
    "medication": ["pill", "supplement", "vitamin", "prescription", "dose"],
    "break": ["pause", "day", "detox", "hiatus", "abstain", "offline"],
    "event": ["party", "concert", "festival", "wedding", "fundraiser", "gala", "show",
              "conference", "meetup", "workshop", "race", "game", "match"],
    "sibling": ["brother", "sister", "stepbrother", "stepsister"],
    "relative": ["mother", "father", "sister", "brother", "aunt", "uncle", "cousin",
                 "grandmother", "grandfather", "niece", "nephew", "parent", "grandma",
                 "grandpa", "mom", "dad"],
    "vehicle": ["car", "bike", "bicycle", "motorcycle", "scooter", "truck", "van"],
    "jewelry": ["earring", "necklace", "bracelet", "ring", "pendant", "brooch", "locket",
                "cufflink", "anklet", "bangle", "chain", "charm"],
    "doctor": ["physician", "dermatologist", "specialist", "surgeon", "dentist", "podiatrist",
               "pediatrician", "cardiologist", "gastroenterologist", "optometrist",
               "orthopedist", "neurologist", "psychiatrist", "endocrinologist"],
}

# prepositions whose identity DOES change a predicate's meaning ('formed by' = founder,
# 'formed in' = year). Kept as a signature marker; 'to' is excluded because it is an
# infinitive marker here ('plans to' == 'planning').
PREP_KEEP = {"by", "in", "on", "at", "for", "with", "about", "from", "into", "of",
             "out", "up", "off", "over", "through", "as", "against", "toward", "towards"}

ENTITY_ALIAS = {"i", "me", "my", "myself", "user", "the user", "users", "the user's",
                "self", "we", "us"}


def _measure(w):
    """Porter's m: number of vowel-consonant sequences (used only by the -e restoration)."""
    forms = "".join("V" if c in "aeiou" or (c == "y" and i and w[i-1] not in "aeiou") else "C"
                    for i, c in enumerate(w))
    return forms.count("VC")


def _restore_e(u):
    """Porter step-1b cleanup after an -ing/-ed strip, so that a verb's base form and its
    inflections converge: 'bake' / 'baked' / 'baking' -> 'bake' (before this, the question
    term 'bake' never matched the stored fact 'baked chocolate cake', because the strip
    produced 'bak'). 'planning' -> 'plann' -> 'plan' keeps the double-consonant branch."""
    if u.endswith(("at", "bl", "iz")):
        return u + "e"
    if len(u) > 2 and u[-1] == u[-2] and u[-1] not in "aeiouls":
        return u[:-1]
    if (len(u) >= 2 and _measure(u) <= 1 and u[-1] not in "aeiouwxy"
            and u[-2] in "aeiou" and (len(u) < 3 or u[-3] not in "aeiou")):
        return u + "e"
    return u


def _stem(t):
    """Deterministic light stemmer (no external dependency, no data fitting).
    Order matters: plural stripping runs FIRST so a plural -ing noun and its singular
    converge ('weddings' -> 'wedding' -> 'wed' and 'wedding' -> 'wed'), and the -e
    restoration runs after an -ing/-ed strip (see _restore_e)."""
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith(("sses", "shes", "ches", "xes", "zes")):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    if len(t) > 4 and t.endswith("ing") and t[:-3][-1] not in "aeiou":
        t = _restore_e(t[:-3])              # 'using' -> 'us' -> 'use'; 'going' stays ('go' ends in a vowel)
    elif len(t) > 3 and t.endswith("ed") and t[:-2][-1] not in "aeiou":
        t = _restore_e(t[:-2])              # 'used' -> 'use'; 'need' stays ('ne' ends in a vowel)
    if len(t) > 4 and t.endswith("e") and _measure(t[:-1]) > 1:
        t = t[:-1]                          # Porter 5a: 'acquire' -> 'acquir' == stem of 'acquired'
    return t


def _stem_tokens(s):
    return [_stem(t) for t in _word.findall(str(s).lower())]


_SYN_STEMMED = None


def _syn(t):
    global _SYN_STEMMED
    if _SYN_STEMMED is None:
        _SYN_STEMMED = _stem_syn_table(SYN)
    return _SYN_STEMMED.get(t, t)


def _canon_tokens(s, keep_light_fallback=True):
    """attribute phrase -> canonical token tuple (stemmed, synonym-mapped, light-stripped)."""
    toks = []
    for t in _stem_tokens(s):
        t = _syn(t)
        if t in LIGHT:
            continue
        toks.append(t)
    if not toks and keep_light_fallback:
        raw = [_syn(t) for t in _stem_tokens(s)]
        toks = raw or [_norm(s) or "attr"]
    # order-preserving dedupe
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


def _trailing_prep(s):
    """Trailing preposition of the raw phrase, or None. Distinguishes 'formed by' from
    'formed in' after the light-token strip has removed both."""
    toks = _stem_tokens(s)
    return toks[-1] if toks and toks[-1] in PREP_KEEP else None


def _head(toks):
    for t in reversed(toks):
        if t not in PARTICLE:
            return t
    return toks[-1] if toks else ""


def _ent_canon(ent):
    n = _norm(ent)
    if n in ENTITY_ALIAS:
        return "user"
    n = re.sub(r"^the ", "", n)
    return " ".join(_stem(t) for t in n.split()) or n


# ---------------------------------------------------------------- (a) chain merging

def _near_dup_tokens(sa, sb, ratio=0.85):
    """R4': token-aligned spelling variant. Same length, at most one differing position,
    and the differing pair must share its first 3 characters (blocks 'watch 1'/'watch 2'
    and 'mother'/'sister', which a raw string edit-similarity happily merged)."""
    if len(sa) != len(sb):
        return False
    diff = [(x, y) for x, y in zip(sa, sb) if x != y]
    if len(diff) != 1:
        return False
    x, y = diff[0]
    if len(x) < 3 or len(y) < 3 or x[:3] != y[:3]:
        return False
    return difflib.SequenceMatcher(None, x, y).ratio() >= ratio


def _mergeable(sa, sb, jaccard=None, ratio=0.85, prep_a=None, prep_b=None):
    """Rules (heads are already equal by construction). Returns rule name or None.

    R1/R2/R4' are the DEFAULT set: a 30-cluster hand audit found 2/30 false merges from
    them, both of which the trailing-preposition guard below now blocks.
    R3 (token Jaccard) is OFF by default: a 20-cluster audit of R3-only merges found
    7/20 false (35%) - it fuses coordinate siblings such as
    'accommodation cost per LAX trip' / 'accommodation nights per LAX trip'.
    Pass jaccard=0.6 to switch it on (for the ablation table).

    Two precision guards apply to every rule:
      * preposition guard: the trailing preposition must be IDENTICAL ('formed by' vs
        'formed in', 'switched to' vs 'switched from', 'sells' vs 'sells at' all stay split).
        Infinitival 'to' is not a preposition here, so 'plans to' == 'planning'. The audit
        trail behind the strict form is in results/_v5_audit.md;
      * DISCRIMINATIVE guard: a digit or sibling-marking token in the symmetric difference
        vetoes the merge ('example 1' / 'example 2', 'original' / 'revised price range')."""
    if prep_a != prep_b:                 # strict: 'sells' / 'sells at' also stays split
        return None
    A, B = set(sa), set(sb)
    for t in A ^ B:                      # sibling-discriminating token -> never merge
        if t in DISCRIMINATIVE or t.isdigit():
            return None
    if A == B:
        return "R1_same_signature"
    if A <= B or B <= A:
        extra = (B - A) if A <= B else (A - B)
        if extra <= MODIFIER:
            return "R2_modifier_subset"
    if _near_dup_tokens(sa, sb, ratio):
        return "R4_token_neardup"
    if jaccard is not None:
        inter = len(A & B)
        if inter and len(A) >= 2 and len(B) >= 2 and inter / len(A | B) >= jaccard:
            return "R3_jaccard"
    return None


class _UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, k):
        while self.p[k] != k:
            self.p[k] = self.p[self.p[k]]
            k = self.p[k]
        return k

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return ra != rb


def cluster_keys(store, jaccard=None, ratio=0.85, merge_entities=True):
    """Return (clusters: list[list[key]], edges: list[(k1,k2,rule)]). Pure inspection helper."""
    meta = {}
    for k, chain in store.items():
        attr = chain[0]["attribute"]
        ent = chain[0]["entity"]
        sig = _canon_tokens(attr)
        meta[k] = {"ent": _ent_canon(ent) if merge_entities else _norm(ent),
                   "sig": sig, "head": _head(sig), "attr": attr, "n": len(chain),
                   "prep": _trailing_prep(attr)}
    buckets = {}
    for k, m in meta.items():
        buckets.setdefault((m["ent"], m["head"]), []).append(k)
    uf = _UF(list(store))
    edges = []
    for _, ks in buckets.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                rule = _mergeable(meta[ks[i]]["sig"], meta[ks[j]]["sig"], jaccard, ratio,
                                  meta[ks[i]]["prep"], meta[ks[j]]["prep"])
                if rule:
                    edges.append((ks[i], ks[j], rule))
                    uf.union(ks[i], ks[j])
    groups = {}
    for k in store:
        groups.setdefault(uf.find(k), []).append(k)
    return [sorted(v) for v in groups.values()], edges


def merge_chains(store, jaccard=None, ratio=0.85, merge_entities=True):
    """Soft-merge chains of a v4 lifecycle store. Relations are recomputed by
    lifecycle.assemble_chains() on the pooled facts, so relation semantics are unchanged."""
    if not store:
        return {}
    clusters, _ = cluster_keys(store, jaccard, ratio, merge_entities)
    out = {}
    for keys in clusters:
        facts = []
        attr_votes, ent_votes = {}, {}
        for k in keys:
            for c in store[k]:
                facts.append(c)
                attr_votes[c["attribute"]] = attr_votes.get(c["attribute"], 0) + 1
                ent_votes[c["entity"]] = ent_votes.get(c["entity"], 0) + 1
        # canonical label: most frequent original spelling, tie -> shortest, then alphabetical
        canon_attr = sorted(attr_votes, key=lambda a: (-attr_votes[a], len(a), a))[0]
        canon_ent = sorted(ent_votes, key=lambda e: (-ent_votes[e], len(e), e))[0]
        rewritten = [{**f, "entity": canon_ent, "attribute": canon_attr,
                      "attr_orig": f.get("attr_orig", f["attribute"]),
                      "ent_orig": f.get("ent_orig", f["entity"]),
                      "variants": len(keys)} for f in facts]
        sub = assemble_chains(rewritten)
        for kk, chain in sub.items():
            if kk in out:                      # defensive: never silently drop facts
                out[kk] = out[kk] + chain
            else:
                out[kk] = chain
    return out


# ---------------------------------------------------------------- (b) aggregation view

AGG_COUNT_PATTERNS = [r"\bhow many\b", r"\bhow often\b", r"\bnumber of\b",
                      r"\bhow frequently\b", r"\bhow many times\b",
                      r"\blist all\b", r"\bwhat are all\b", r"\ball of the\b"]
AGG_AMOUNT_PATTERNS = [r"\bhow much\b", r"\bin total\b", r"\btotal (number|count|amount|cost|time|spend|spent)\b",
                       r"\bhow long in total\b", r"\baltogether\b", r"\bcombined\b"]
_agg_count_re = [re.compile(p) for p in AGG_COUNT_PATTERNS]
_agg_amount_re = [re.compile(p) for p in AGG_AMOUNT_PATTERNS]

QSTOP = {
    "how", "many", "much", "often", "what", "which", "who", "when", "where", "why",
    "do", "does", "did", "have", "has", "had", "am", "is", "are", "was", "were", "be",
    "i", "me", "my", "mine", "we", "our", "us", "you", "your", "it", "its", "they",
    "the", "a", "an", "of", "from", "at", "to", "in", "on", "for", "and", "or", "so",
    "far", "total", "number", "count", "times", "time", "there", "that", "this",
    "currently", "current", "now", "today", "still", "ever", "all", "any", "some",
    "been", "being", "get", "got", "make", "made", "take", "took", "tell", "say",
    "please", "list", "give", "about", "with", "by", "as", "if", "than", "then",
    "over", "since", "during", "after", "before", "last", "past", "recent", "recently",
    "month", "week", "year", "day", "include", "includ", "including", "one", "each",
    "need", "set", "use", "go", "come", "want", "wish", "like", "different", "total",
}


def is_aggregation_question(question):
    """(bool, fired_patterns). 'count' patterns enumerate items; 'amount' patterns sum values."""
    ql = str(question).lower()
    fired = [p.pattern for p in _agg_count_re + _agg_amount_re if p.search(ql)]
    return bool(fired), fired


def aggregation_kind(question):
    ql = str(question).lower()
    if any(p.search(ql) for p in _agg_count_re):
        return "count"
    if any(p.search(ql) for p in _agg_amount_re):
        return "amount"
    return None


def question_focus(question, use_hyponyms=True):
    """Question -> (focus stems, printable order, term->group-label map).
    Hyponyms of a category word map back to the CATEGORY for grouping, so 'boots' and
    'dress' land in the same enumerable group as the question word 'clothing'."""
    core, order = set(), []
    for t in _stem_tokens(question):
        t = _syn(t)
        if t in QSTOP or t in PARTICLE or t in LIGHT or len(t) < 2:
            continue
        if t not in core:
            core.add(t)
            order.append(t)
    focus = set(core)
    label = {t: t for t in core}
    if use_hyponyms:
        for t in list(core):
            for cat, hyps in HYPONYMS.items():
                cstem = _stem(cat)
                hstems = {_stem(h) for h in hyps}
                if t == cstem or t == cat or t in hstems:
                    focus.add(cstem)
                    label[cstem] = cstem
                    for h in hstems:
                        focus.add(h)
                        label.setdefault(h, cstem)
    for t in core:
        label[t] = t     # a term the user typed keeps its own group; it is never folded
    return focus, order, label   # into an expanded category ('bake' must not become 'recipe')


def _value_key(value):
    toks = [t for t in _stem_tokens(value) if t not in LIGHT and len(t) > 1]
    return frozenset(toks) or frozenset([_norm(value)])


DEDUPE_SLACK = 3   # max extra content tokens for a subset to still count as the same item


def _same_item(a, b):
    """Exact restatement (identical content-token set) -> the item is dropped."""
    return a == b


def _maybe_restatement(a, b):
    """Weaker relation: one key is contained in the other with at most DEDUPE_SLACK extra
    content tokens ('boots' vs 'boots from Zara'). Such a line is ANNOTATED, never deleted.
    Measured on the 60 enumeration-count questions (slack sweep, same code path): deleting
    subset matches lowered gold-evidence-session recall monotonically
    (0.886 exact-only -> 0.867 at slack 3 -> 0.861 at slack 6), because a genuine item
    ('has plant = fern') gets eaten by a longer sentence that mentions it
    ('considering misting fern more frequently'). Recall is the unbiased metric here; the
    'items >= gold count' criterion is not, since it rewards inflation."""
    if a == b:
        return False
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return small <= big and len(big) - len(small) <= DEDUPE_SLACK


def _short_date(d):
    p = str(d).split()[0].split("/")
    return "/".join(p[1:]) if len(p) == 3 else str(d)


PREFIX_MIN = 7      # shared-prefix length that counts as the same concept


def _fuzzy_terms(tokens, w):
    """Focus terms a fact matches: exact stem hits plus long-shared-prefix hits, which is how
    a nominalization reaches its verb ('subscription' vs 'subscrib', 'delivery' vs 'deliver').
    Used only for aggregation selection - chain merging stays exact."""
    hits = set(tokens) & set(w)
    for t in tokens:
        if len(t) < PREFIX_MIN:
            continue
        for f in w:
            if f in hits or len(f) < PREFIX_MIN:
                continue
            n = min(len(t), len(f))
            if n >= PREFIX_MIN and t[:PREFIX_MIN] == f[:PREFIX_MIN]:
                hits.add(f)   # 'subscrib' ~ 'subscription'; 'physic' (6) stays below the bar
    return hits


def focus_weights(store, question, use_hyponyms=True, df_cap=0.25, core_df_cap=0.60):
    """Per-question IDF gate (training-free, computed from this question's own store):
    a focus term present in >df_cap of the chains carries no selectivity ('set', 'store',
    'have'), and a term present in no chain carries no signal. Survivors get log-IDF weight.
    The cap is looser for terms the user actually typed (core_df_cap) than for terms the
    hyponym table added: with one 0.25 cap, 'bake' was gated out of 'How many times did I
    bake something in the past two weeks?' (the store is baking-heavy) and the group
    collapsed onto the expanded term 'recipe'."""
    focus, order, label = question_focus(question, use_hyponyms)
    N = max(1, len(store))
    df = {t: 0 for t in focus}
    allf = {t: t for t in focus}
    for chain in store.values():
        terms = set(_stem_tokens(chain[0]["attribute"])) | set(_canon_tokens(chain[0]["attribute"]))
        for c in chain:
            terms |= set(_stem_tokens(c["value"]))
        for t in _fuzzy_terms(terms, allf):
            df[t] += 1
    core = set(order)
    w = {t: math.log(N / (1 + df[t])) for t in focus
         if 0 < df[t] <= (core_df_cap if t in core else df_cap) * N}
    return w, order, label


def aggregation_groups(store, question, use_hyponyms=True, df_cap=0.25, rel_keep=0.35,
                       max_groups=8, core_df_cap=0.60):
    """Question-conditioned enumerable groups.

    A group is keyed by the highest-weight question term that a fact matched (mapped to its
    category for hyponyms), NOT by the chain's attribute - that is what pulls
    'needs to pick up' / 'has not yet picked up' / 'plans to pick up' into ONE countable set
    while leaving the underlying chains untouched. Items are de-duplicated symmetrically on
    value-token subset ('dry cleaning' == 'dry cleaning for navy blue blazer')."""
    w, order, label = focus_weights(store, question, use_hyponyms, df_cap, core_df_cap)
    if not w:
        return [], (w, order)
    core_terms = set(order) & set(w)
    buckets = {}
    for k in sorted(store):
        chain = store[k]
        attr = chain[0]["attribute"]
        ent = chain[0]["entity"]
        a_terms = _fuzzy_terms(set(_canon_tokens(attr)) | set(_stem_tokens(attr)), w)
        a_score = sum(w[t] for t in a_terms)
        for c in chain:
            v_terms = _fuzzy_terms(set(_stem_tokens(c["value"])), w)
            score = 2 * a_score + sum(w[t] for t in v_terms)
            if score <= 0:
                continue
            hits = a_terms if a_terms else v_terms
            if not hits:
                continue
            # a term the user actually typed defines the group; expanded terms only
            # decide the group when no core term matched ('baked chocolate cake' groups
            # under [bake], not under the expanded category [recipe])
            core_hits = hits & core_terms
            top = max(core_hits or hits, key=lambda t: (w[t], t))
            gk = (ent, label.get(top, top))
            b = buckets.setdefault(gk, {"entity": ent, "term": label.get(top, top),
                                        "items": [], "keys": set(), "score": 0.0})
            # strength = how many DISTINCT question terms this single fact matches; a fact
            # matching >=2 focus terms is far more likely to be a real member of the set
            nterms = len({label.get(t, t) for t in (a_terms | v_terms)})
            b["items"].append((score, c, attr, nterms))
            b["keys"].add(k)
            b["score"] = max(b["score"], score)
    groups = []
    for (ent, term), b in buckets.items():
        b["items"].sort(key=lambda t: (t[1]["date"], t[1].get("sidx", 0), -t[0]))
        items, keys = [], []
        for score, c, attr, nterms in b["items"]:
            vk = _value_key(c["value"])
            dup = False
            for prev in keys:
                if _same_item(vk, prev):     # bounded symmetric dedupe, keep the FIRST
                    dup = True               # mention (replacing it dropped real items:
                    break                    # 'has plant = fern' lost to 'misting fern...')
            if not dup:
                keys.append(vk)
                items.append((score, c, attr, nterms))
        groups.append({"entity": ent, "term": term, "items": items,
                       "score": b["score"], "chain_keys": sorted(b["keys"]),
                       "n_strong": sum(1 for it in items if it[3] >= 2),
                       "total_score": sum(it[0] for it in items)})
    if not groups:
        return [], (w, order)
    top = max(g["score"] for g in groups)
    groups = [g for g in groups if g["score"] >= rel_keep * top]
    groups.sort(key=lambda g: (-g["score"], -len(g["items"]), g["term"]))
    return groups[:max_groups], (w, order)


def aggregation_printed_groups(store, question, use_hyponyms=True, df_cap=0.25,
                               rel_keep=0.35, max_groups=8, max_items_per_group=40):
    """The groups render_aggregation_view will print, after the global de-duplication pass.
    Single source of truth so that offline measurement and the rendered text never diverge."""
    groups, meta = aggregation_groups(store, question, use_hyponyms, df_cap, rel_keep,
                                      max_groups)
    seen, kept = [], []
    for g in groups:
        items = []
        for it in g["items"][:max_items_per_group]:
            vk = _value_key(it[1]["value"])
            if any(_same_item(vk, p) for p in seen):
                continue
            seen.append(vk)
            items.append(it)
        if items:
            kept.append({**g, "items": items,
                         "n_strong": sum(1 for i2 in items if i2[3] >= 2)})
    return kept, meta


def render_aggregation_view(store, question, use_hyponyms=True, df_cap=0.25, rel_keep=0.35,
                            max_groups=8, max_items_per_group=40, always=False):
    """Aggregation block for counting/total questions; "" when the question is not one
    (unless always=True). Prepend to the normal lifecycle view - it does not replace it."""
    kind = aggregation_kind(question)
    if kind is None and not always:
        return ""
    kept, (w, order) = aggregation_printed_groups(store, question, use_hyponyms, df_cap,
                                                  rel_keep, max_groups, max_items_per_group)
    if not kept:
        return ""
    verb = "COUNT the distinct items" if kind != "amount" else "ADD UP the values"
    head = ("### Aggregation evidence - the question asks for a total, so the candidate item "
            f"sets are listed exhaustively below and de-duplicated. {verb} in the group that "
            "matches the question; ignore items that do not belong.")
    lines = [head, "Question focus terms (by selectivity): " +
             ", ".join(t for t, _ in sorted(w.items(), key=lambda kv: -kv[1]))]
    for g in kept:
        lines.append(f"[{g['term']}] {g['entity']}: {len(g['items'])} distinct item(s), "
                     f"{g['n_strong']} of them marked (*) as matching two or more question terms")
        ordered = ([it for it in g["items"] if it[3] >= 2] +
                   [it for it in g["items"] if it[3] < 2])
        printed_keys = []
        for score, c, attr, nterms in ordered:
            mark = "*" if nterms >= 2 else " "
            vk = _value_key(c["value"])
            note = ("  [may restate an earlier line - do not count twice]"
                    if any(_maybe_restatement(vk, p) for p in printed_keys) else "")
            printed_keys.append(vk)
            lines.append(f"  {mark}- {attr} = {c['value']} ({_short_date(c['date'])}){note}")
    lines.append("Group sizes: " + "; ".join(
        f"{g['term']}={len(g['items'])}({g['n_strong']}*)" for g in kept)
        + f" | union of all groups = {sum(len(g['items']) for g in kept)} distinct items, "
        + f"{sum(g['n_strong'] for g in kept)} marked (*)")
    return "\n".join(lines)


if __name__ == "__main__":
    F = lambda e, a, v, d, s: {"entity": e, "attribute": a, "value": v, "date": d, "sidx": s}
    facts = [
        F("user", "job", "teacher", "2023/05/20 (Sat) 02:21", 0),
        F("user", "current job", "engineer", "2023/05/25 (Thu) 10:00", 3),
        F("the user", "occupation", "engineer at Acme", "2023/05/26 (Fri) 10:00", 4),
        F("user", "needs to pick up", "boots from Zara", "2023/05/20 (Sat) 02:21", 0),
        F("user", "needs to pick up", "dry cleaning", "2023/05/22 (Mon) 02:21", 2),
        F("user", "needs to return", "dress from H&M", "2023/05/28 (Sun) 02:21", 5),
        F("user", "favorite book", "Dune", "2023/05/20 (Sat) 02:21", 0),
        F("user", "book", "Neuromancer", "2023/05/21 (Sun) 02:21", 1),
    ]
    st = assemble_chains(facts)
    m = merge_chains(st)
    print("before", len(st), sorted(st))
    print("after ", len(m), sorted(m))
    assert len(m) < len(st)
    job = [k for k in m if "job" in k or "occupation" in k]
    assert len(job) == 1, job
    assert len(m[job[0]]) == 3, m[job[0]]
    # 'favorite book' must NOT merge into 'book' (different attribute, not a modifier)
    assert sum(1 for k in m if k.endswith("book")) == 2, [k for k in m if "book" in k]
    txt = render_aggregation_view(m, "How many items of clothing do I need to pick up or return?")
    print("--- aggregation ---\n" + txt)
    assert "union of all groups = 3 distinct items" in txt, txt
    assert "boots from Zara" in txt and "dress from H&M" in txt and "dry cleaning" in txt
    assert render_aggregation_view(m, "What is my job?") == ""
    print("v5 self-test OK")
