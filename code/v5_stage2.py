"""Stage-2 offline analysis: view-compression curve, strict budget matching, exact McNemar.
No API. Reads results/prov_v5_offline.json; the compression curve re-renders views from the cache."""
import hashlib
import json
import math
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset
from coverage_diag import contains
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"
CAPS = [4000, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 10 ** 9]
DATA = None


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def alpha_cap(full, cap):
    kept, total = [], 0
    for line in full.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def sweep(qi):
    q = DATA[qi]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for e, a, v in (cached_facts(sess) or []):
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)
    full = render_view(store, mode)
    scored = V5.score_chains(store, q["question"])
    out = {"qid": q["question_id"], "type": q["question_type"], "is_abs": q["question_id"].endswith("_abs"),
           "full_chars": len(full), "alpha": {}, "rank": {}}
    for c in CAPS:
        a = alpha_cap(full, c)
        rv, _ = V5.rank_view(store, q["question"], mode, cap=c, scored=scored)
        out["alpha"][str(c)] = [contains(a, q["answer"]), len(a), len(a.splitlines())]
        out["rank"][str(c)] = [contains(rv, q["answer"]), len(rv), len(rv.splitlines())]
    return out


def mcnemar(pairs):
    """Exact two-sided McNemar on paired binaries [(a,b)]: b=candidate wins, c=baseline wins."""
    b = sum(1 for x, y in pairs if y and not x)
    c = sum(1 for x, y in pairs if x and not y)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def main():
    global DATA
    DATA = load_dataset("s_cleaned")
    if not os.path.exists(f"{R}/prov_v5_viewsweep.json"):
        with Pool(30) as p:
            rows = p.map(sweep, range(len(DATA)), chunksize=4)
        json.dump(rows, open(f"{R}/prov_v5_viewsweep.json", "w"))
    rows = json.load(open(f"{R}/prov_v5_viewsweep.json"))
    nab = [r for r in rows if not r["is_abs"]]
    print("== A. view rendering: gold-string coverage vs char cap (n=470 non-abs) ==")
    print("full un-capped view: mean %.0f chars" % (sum(r["full_chars"] for r in nab) / len(nab)))
    print(f"{'cap':>9s} {'alpha_cov':>9s} {'rank_cov':>9s} {'delta':>7s} {'alphaChars':>10s} {'rankLines':>9s}")
    for c in CAPS:
        k = str(c)
        a = sum(r["alpha"][k][0] for r in nab) / len(nab)
        v = sum(r["rank"][k][0] for r in nab) / len(nab)
        print(f"{(c if c < 10**9 else 'none'):>9} {a:9.4f} {v:9.4f} {v-a:+7.4f} "
              f"{sum(r['alpha'][k][1] for r in nab)/len(nab):10.0f} "
              f"{sum(r['rank'][k][2] for r in nab)/len(nab):9.1f}")
    dep = sum(r["alpha"]["20000"][0] for r in nab) / len(nab)
    for c in CAPS:
        if sum(r["rank"][str(c)][0] for r in nab) / len(nab) >= dep:
            print(f"ranked view reaches the deployed 20k alphabetical coverage ({dep:.4f}) "
                  f"at a cap of {c} chars ({c/20000:.0%} of the budget)")
            break

    off = json.load(open(f"{R}/prov_v5_offline.json"))
    cov = {r["qid"]: r for r in json.load(open(f"{R}/coverage_diag.json"))}
    NAB = [r for r in off if not r["is_abs"]]
    print("\n== B. strictly comparable containment (content-only raw text, as coverage_diag) ==")
    print(f"{'variant':13s} {'inView':>7s} {'inRaw_c':>8s} {'inCtx_c':>8s} {'inCtx_dec':>9s} {'totChars':>9s} {'ratio':>6s}")
    bt = sum(r["v"]["baseline"]["view_chars"] + r["v"]["baseline"]["raw_chars"] for r in NAB) / len(NAB)
    for n in ["baseline", "rankview", "prov_hybrid6", "v5", "realloc_bm_a", "realloc_bm",
              "v5_final", "v5f_bm", "bm25_k30", "bm25_k50"]:
        t = sum(r["v"][n]["view_chars"] + r["v"][n]["raw_chars"] for r in NAB) / len(NAB)
        g = lambda k: sum(r["v"][n][k] for r in NAB) / len(NAB)
        print(f"{n:13s} {g('in_view'):7.4f} {g('in_raw_c'):8.4f} {g('in_ctx_c'):8.4f} "
              f"{g('in_ctx'):9.4f} {t:9.0f} {t/bt:6.3f}")

    print("\n== C. exact McNemar vs baseline (paired, n=470 non-abs) ==")
    print(f"{'variant':13s} {'metric':12s} {'base':>6s} {'cand':>6s} {'+':>4s} {'-':>4s} {'p':>8s}")
    metrics = [("allEvTurns", lambda r, n: r["v"][n]["ev_turns_hit"] == r["n_ev_turns"]),
               ("gold_in_ctx", lambda r, n: r["v"][n]["in_ctx_c"]),
               ("gold_in_view", lambda r, n: r["v"][n]["in_view"])]
    for n in ["rankview", "prov_hybrid6", "v5", "realloc_bm", "v5_final", "v5f_bm", "divcap2", "bm25_k50"]:
        for mn, fn in metrics:
            pairs = [(fn(r, "baseline"), fn(r, n)) for r in NAB]
            b, c, p = mcnemar(pairs)
            print(f"{n:13s} {mn:12s} {sum(x for x, _ in pairs)/len(pairs):6.3f} "
                  f"{sum(y for _, y in pairs)/len(pairs):6.3f} {b:4d} {c:4d} {p:8.4f}")

    print("\n== D. MS-only and counting-only, matched budget ==")
    MS = [r for r in NAB if r["type"] == "multi-session"]
    CN = [r for r in NAB if r["is_count"] and r["gold_num"] is not None]
    for n in ["baseline", "rankview", "prov_hybrid6", "v5", "realloc_bm", "v5_final", "v5f_bm"]:
        ms = sum(r["v"][n]["ev_turns_hit"] == r["n_ev_turns"] for r in MS) / len(MS)
        b, c, p = mcnemar([(r["v"]["baseline"]["ev_turns_hit"] == r["n_ev_turns"],
                            r["v"][n]["ev_turns_hit"] == r["n_ev_turns"]) for r in MS])
        cagg = sum(r["v"][n]["chain_agg_in_view"] >= r["gold_num"] for r in CN) / len(CN)
        cagg_all = sum(r["v"][n]["chain_agg"] >= r["gold_num"] for r in CN) / len(CN)
        print(f"{n:13s} MS allEvTurns {ms:.3f} (+{b} -{c} p={p:.4f})  "
              f"chainProxy>=gold in-view {cagg:.3f} / store {cagg_all:.3f}")

    print("\n== E. abstention distractor proxy (n=30; no offline metric can settle abstention) ==")
    print("proxy = raw turns carrying >=2 question content terms; more topical-but-irrelevant")
    print("material is the suspected cause of ours being the worst abstention arm.")
    AB = [r for r in off if r["is_abs"]]
    for n in ["baseline", "rankview", "v5", "v5_final", "v5f_bm", "bm25_k50"]:
        g = lambda k: sum(r["v"][n][k] for r in AB) / len(AB)
        print(f"{n:13s} raw_chars {g('raw_chars'):6.0f} nTurns {g('n_turns'):5.2f} "
              f"turnSessions {g('n_turn_sessions'):5.2f} viewChars {g('view_chars'):6.0f}")


if __name__ == "__main__":
    main()
