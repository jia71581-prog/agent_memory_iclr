"""INDEPENDENT adversarial recomputation of the ranked-vs-alphabetical view compression curve.

Recomputes from the dataset + session extraction cache (no API). Adds controls the original
sweep did not have:
  - strict normalized-substring gold match vs the loose all-tokens-scattered fallback
  - an alphabetical baseline that uses the SAME greedy packing as rank_view ("continue",
    not "break"), to separate packing from ranking
  - a random-order control (seeded by qid) at each cap: how much of the "ranked" gain is
    just "not-alphabetical"
  - gold length / numeric flags so degenerate containment can be quarantined
"""
import hashlib
import json
import os
import random
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
CAPS = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 10 ** 9]
DATA = None

_NORM = re.compile(r"[^a-z0-9]+")
STOPW = ("the", "and", "for", "was", "were", "you", "your")


def norm(s):
    return _NORM.sub(" ", str(s).lower()).strip()


def match_kind(hay, needle):
    """Returns (loose_hit, strict_hit). strict = normalized substring. loose = the metric the
    design agent used (coverage_diag.contains): substring OR all content tokens present anywhere."""
    h, n = norm(hay), norm(needle)
    if not n:
        return False, False
    if n in h:
        return True, True
    toks = [t for t in n.split() if len(t) > 2 and t not in STOPW]
    if not toks:
        return False, False
    return all(t in h for t in toks), False


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pack_lines(lines, cap, mode="break"):
    kept, total = [], 0
    for line in lines:
        if total + len(line) + 1 > cap:
            if mode == "break":
                break
            continue
        kept.append(line)
        total += len(line) + 1
    return kept, total


def sweep(qi):
    q = DATA[qi]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    facts, miss = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        cf = cached_facts(sess)
        if cf is None:
            miss = True
            cf = []
        for e, a, v in cf:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)
    full = render_view(store, mode)
    full_lines = full.splitlines()
    scored = V5.score_chains(store, q["question"])
    keys_by_score = [k for k, _s, _n in scored]
    line_of = {k: V5._render_chain_line(store[k], mode) for k in store}

    rng = random.Random(hashlib.md5(q["question_id"].encode()).hexdigest()[:8])
    rand_keys = sorted(store)
    rng.shuffle(rand_keys)

    gold = q["answer"]
    out = {"qid": q["question_id"], "type": q["question_type"],
           "is_abs": q["question_id"].endswith("_abs"),
           "full_chars": len(full), "full_lines": len(full_lines),
           "cache_missing": miss, "gold_len": len(str(gold)),
           "gold_numeric": bool(re.fullmatch(r"[\d.,]+", str(gold).strip())),
           "n_chains": len(store)}
    for c in CAPS:
        # a) deployed alphabetical: first-fit STOP (identical to run_qc.py)
        a_lines, _ = pack_lines(full_lines, c, "break")
        a = "\n".join(a_lines)
        # b) alphabetical with the same greedy packing rank_view uses
        ap_lines, _ = pack_lines(full_lines, c, "continue")
        ap = "\n".join(ap_lines)
        # c) ranked (module under test), re-derived here rather than calling rank_view
        r_keys, tot = [], 0
        for k in keys_by_score:
            ln = line_of[k]
            if tot + len(ln) + 1 > c:
                continue
            r_keys.append(k)
            tot += len(ln) + 1
        rv = "\n".join(line_of[k] for k in sorted(r_keys))
        # d) random-order control, same greedy packing
        d_keys, tot = [], 0
        for k in rand_keys:
            ln = line_of[k]
            if tot + len(ln) + 1 > c:
                continue
            d_keys.append(k)
            tot += len(ln) + 1
        dv = "\n".join(line_of[k] for k in sorted(d_keys))
        # cross-check against the module's own rank_view
        rv_mod, _ = V5.rank_view(store, q["question"], mode, cap=c, scored=scored)
        for tag, txt, nl in [("alpha", a, len(a_lines)), ("alphapack", ap, len(ap_lines)),
                             ("rank", rv, len(r_keys)), ("randv", dv, len(d_keys))]:
            lo, st_ = match_kind(txt, gold)
            out.setdefault(tag, {})[str(c)] = [lo, st_, len(txt), nl]
        out.setdefault("rank_matches_module", {})[str(c)] = (rv_mod == rv)
    return out


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    with Pool(30) as p:
        rows = p.map(sweep, range(len(DATA)), chunksize=4)
    json.dump(rows, open(f"{R}/_verify_viewsweep.json", "w"))
    print("wrote", len(rows))


if __name__ == "__main__":
    main()
