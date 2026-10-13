"""Adversarial battery run on the design agent's EXACT 46-question subset and its own
token-OR pool definition (v5_aggmech.py), plus the realizable-cluster and item-level-recall
counter-measurements. Offline."""
import hashlib, json, os, re, statistics as st, sys
from collections import Counter
from multiprocessing import Pool
from scipy.stats import fisher_exact
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import assemble_chains, SESSION_CACHE_DIR, _session_text, _norm
from v5_chains import merge_chains, aggregation_printed_groups

R = "/root/agent_memory_iclr/results"
ROWS = {r["qid"]: r for r in json.load(open(f"{R}/_v5_shape_rows.json"))}
V = {r["qid"]: r for r in json.load(open(f"{R}/_v5_offline_val.json"))}
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20}
STOP = set("how many much did i do have the a an of in on to my me for and or is are was were what "
           "different total times number all across including both currently now got get".split())
W = re.compile(r"[a-z0-9']+")
CNT = re.compile(r"^how many\b")
LIGHT = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "my", "i",
         "is", "was", "are", "were", "at", "it", "that", "this", "user"}


def gold_int(g):
    g = g.lower().replace(",", "")
    m = re.match(r"^\s*\$?(\d+)\b", g)
    if m:
        return int(m.group(1))
    for w, v in NUMW.items():
        if re.match(rf"^\s*{w}\b", g):
            return v
    return None


def qtok(s):
    return {t for t in W.findall(s.lower()) if t not in STOP and len(t) > 2}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def work(q):
    qid = q["question_id"]
    r = ROWS[qid]
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        t = _session_text(sess)
        p = os.path.join(SESSION_CACHE_DIR, f"{hashlib.md5(t.encode()).hexdigest()[:16]}.json")
        if not (t and os.path.exists(p)):
            continue
        for e, a, v in json.load(open(p)):
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    gold = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    ev_norm = norm(" ".join(t["content"] for si, s in enumerate(q["haystack_sessions"])
                            if si in gold for t in s if t.get("has_answer") and t.get("content")))
    qt = qtok(r["q"])
    pool = [f for f in facts if qt & (qtok(f["attribute"]) | qtok(f["entity"]) | qtok(f["value"]))]

    def grounded(val):
        toks = [t for t in norm(val).split() if t not in LIGHT and len(t) > 2]
        return bool(toks) and all(t in ev_norm for t in toks)

    pool_distinct = len({_norm(f["value"]) for f in pool})
    pool_gold = len({_norm(f["value"]) for f in pool if f["sidx"] in gold})
    pool_grounded = len({_norm(f["value"]) for f in pool if grounded(f["value"])})
    pool_gold_grounded = len({_norm(f["value"]) for f in pool if f["sidx"] in gold and grounded(f["value"])})
    # realizable clusters from the shipped renderer
    store5 = merge_chains(assemble_chains(facts))
    kept, _ = aggregation_printed_groups(store5, r["q"])
    sizes = [len({norm(it[1]["value"]) for it in g["items"]}) for g in kept]
    top = sizes[0] if sizes else 0
    mx = max(sizes) if sizes else 0
    union = len({norm(it[1]["value"]) for g in kept for it in g["items"]})
    union_gr = len({norm(it[1]["value"]) for g in kept for it in g["items"] if grounded(it[1]["value"])})
    return {"qid": qid, "pool": len(pool), "pool_distinct": pool_distinct, "pool_gold": pool_gold,
            "pool_grounded": pool_grounded, "pool_gold_grounded": pool_gold_grounded,
            "cl_top": top, "cl_max": mx, "cl_union": union, "cl_union_grounded": union_gr,
            "n_ev": V[qid]["n_ev"]}


if __name__ == "__main__":
    data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
    targets = [qid for qid, r in ROWS.items()
               if r["shape"] == "AGG.enum" and not r["is_abs"]
               and gold_int(r["gold"]) is not None and gold_int(r["gold"]) <= 12
               and CNT.match(r["q"].lower())]
    print("reproduced subset size:", len(targets))
    with Pool(16) as p:
        res = {x["qid"]: x for x in p.map(work, [data[t] for t in targets], chunksize=1)}
    ARM = {}
    for a in ("full500_qc", "full500_fifo", "full500_decay", "full500_qc_no_lifecycle",
              "full500_full"):
        ARM[a] = {json.loads(l)["qid"]: json.loads(l)["judge"] for l in open(f"{R}/{a}.jsonl")
                  if "judge" in json.loads(l)}
    rows = []
    for qid in targets:
        r = ROWS[qid]
        x = res[qid]
        rows.append({**x, "N": gold_int(r["gold"]), "judge": r["judge"], "q": r["q"],
                     "type": r["type"], "gold": r["gold"]})
    json.dump(rows, open(f"{R}/_adv6_agg46.json", "w"), ensure_ascii=False)
    n = len(rows)
    print(f"\nmean N={st.mean(r['N'] for r in rows):.1f} mean n_ev={st.mean(r['n_ev'] for r in rows):.1f} "
          f"mean pool={st.mean(r['pool'] for r in rows):.1f} mean pool_distinct={st.mean(r['pool_distinct'] for r in rows):.1f}")
    print("\n=== 'has >= N candidates' under progressively honest definitions (n=46) ===")
    for f, lab in [("pool_distinct", "token-OR pool, all sessions (claim's metric)"),
                   ("pool_gold", "  same pool, values from gold ev sessions only"),
                   ("pool_grounded", "  same pool, values grounded in has_answer turns"),
                   ("pool_gold_grounded", "  gold-session AND grounded"),
                   ("cl_union", "renderable: union of printed clusters"),
                   ("cl_max", "renderable: largest printed cluster"),
                   ("cl_top", "renderable: top-scored printed cluster"),
                   ("cl_union_grounded", "renderable union, grounded values only")]:
        ok = sum(1 for r in rows if r[f] >= r["N"])
        print(f"  {lab:52s} {ok:2d}/{n} = {ok/n:.3f}   mean={st.mean(r[f] for r in rows):5.2f}")
        acc_ok = [r for r in rows if r[f] >= r["N"]]
        acc_no = [r for r in rows if r[f] < r["N"]]
        a = sum(1 for r in acc_ok if r["judge"]); b = sum(1 for r in acc_no if r["judge"])
        p = fisher_exact([[a, len(acc_ok) - a], [b, len(acc_no) - b]])[1] if acc_no and acc_ok else float("nan")
        print(f"      acc|satisfied {a}/{len(acc_ok)}={a/max(1,len(acc_ok)):.3f}  "
              f"acc|not {b}/{len(acc_no)}={b/max(1,len(acc_no)):.3f}  fisher p={p:.4g}")

    print("\n=== the 0.917 vs 0.591 split: confound audit (n=46) ===")
    hi = [r for r in rows if r["N"] > r["n_ev"]]
    lo = [r for r in rows if r["N"] <= r["n_ev"]]
    for a in ARM:
        ah = sum(1 for r in hi if ARM[a][r["qid"]]); al = sum(1 for r in lo if ARM[a][r["qid"]])
        print(f"  {a:24s} N>n_ev {ah}/{len(hi)}={ah/len(hi):.3f}  N<=n_ev {al}/{len(lo)}={al/len(lo):.3f}  "
              f"p={fisher_exact([[ah,len(hi)-ah],[al,len(lo)-al]])[1]:.4g}")
    print("  type composition  hi:", dict(Counter(r["type"] for r in hi)))
    print("  type composition  lo:", dict(Counter(r["type"] for r in lo)))
    print("  n_ev strata (qc arm):")
    for k in sorted({r["n_ev"] for r in rows}):
        s = [r for r in rows if r["n_ev"] == k]
        h2 = [r for r in s if r["N"] > k]; l2 = [r for r in s if r["N"] <= k]
        print(f"    n_ev={k}: N>n_ev {sum(1 for r in h2 if r['judge'])}/{len(h2):<2d}  "
              f"N<=n_ev {sum(1 for r in l2 if r['judge'])}/{len(l2):<2d}  types={dict(Counter(r['type'] for r in s))}")
    for t in sorted({r["type"] for r in rows}):
        s = [r for r in rows if r["type"] == t]
        h2 = [r for r in s if r["N"] > r["n_ev"]]; l2 = [r for r in s if r["N"] <= r["n_ev"]]
        if h2 and l2:
            a = sum(1 for r in h2 if r["judge"]); b = sum(1 for r in l2 if r["judge"])
            print(f"    within {t}: N>n_ev {a}/{len(h2)}  N<=n_ev {b}/{len(l2)}  "
                  f"p={fisher_exact([[a,len(h2)-a],[b,len(l2)-b]])[1]:.4g}")
        else:
            print(f"    within {t}: only one half populated (hi {len(h2)}, lo {len(l2)}) -> no contrast")
    eh = [r for r in hi if not r["judge"]]; el = [r for r in lo if not r["judge"]]
    print(f"  ERROR LOCALISATION: {len(eh)+len(el)} errors total; {len(eh)} in the N>n_ev half, "
          f"{len(el)} ({len(el)/(len(eh)+len(el)):.0%}) in the N<=n_ev half where one item per gold session suffices")
