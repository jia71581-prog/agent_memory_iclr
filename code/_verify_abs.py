"""Adversarial recompute: content-level churn of v5 configs on the 30 abstention questions."""
import hashlib, json, os, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import assemble_chains, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
import v5_retrieval as V5

R = "/root/agent_memory_iclr/results"


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


data = load_dataset("s_cleaned")
ABQ = [q for q in data if q["question_id"].endswith("_abs")]
print("abstention questions:", len(ABQ), flush=True)

out = []
for q in ABQ:
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    facts, missing = [], False
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tri = cached_facts(sess)
        if tri is None:
            missing = True
            continue
        for e, a, v in tri:
            facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
    store = assemble_chains(facts)
    units = V5.build_units(q)
    bm = V5.bm25_scores(q["question"], [u[3] for u in units])

    res = {}
    base_total = None
    for name, cfg in [("baseline", "baseline"), ("v5_drop_in", "v5_drop_in"),
                      ("rankview", "rankview"), ("v5_realloc", "v5_realloc"),
                      ("v5_realloc_m", "v5_realloc")]:
        rc = None
        if name == "v5_realloc_m":
            v0, _r0, _d0 = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                            units=units, bm_scores=bm)
            rc = max(0, base_total - len(v0))
        vt, rt, dbg = V5.build_context(q, store, q["question"], config=cfg, mode=mode,
                                       units=units, bm_scores=bm, raw_char_cap=rc)
        if name == "baseline":
            base_total = len(vt) + len(rt)
        res[name] = {"view": vt, "turns": set(dbg["turn_idx"]), "prov": list(dbg["prov_idx"]),
                     "view_chars": len(vt), "raw_chars": len(rt),
                     "sess": set(dbg["turn_sessions"]), "n_chains": len(store)}
    out.append({"qid": q["question_id"], "missing": missing, "res": res, "mode": mode,
                "n_units": len(units), "n_sess": len(q["haystack_sessions"])})
    print(".", end="", flush=True)
print()


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


print("\n%-20s %7s %7s %7s %7s %7s" % ("pair (vs baseline)", "turnJac", "newTurn", "sessJac", "viewJac", "viewLineKeep"))
for name in ["rankview", "v5_drop_in", "v5_realloc", "v5_realloc_m"]:
    tj = sj = vj = nt = keep = 0.0
    for r in out:
        b, v = r["res"]["baseline"], r["res"][name]
        tj += jac(b["turns"], v["turns"])
        nt += len(v["turns"] - b["turns"])
        sj += jac(b["sess"], v["sess"])
        bl, vl = set(b["view"].splitlines()), set(v["view"].splitlines())
        vj += jac(bl, vl)
        keep += len(bl & vl) / max(1, len(bl))
    n = len(out)
    print("%-20s %7.3f %7.2f %7.3f %7.3f %7.3f" % (name, tj/n, nt/n, sj/n, vj/n, keep/n))

json.dump([{"qid": r["qid"], "missing": r["missing"], "mode": r["mode"],
            "n_units": r["n_units"], "n_sess": r["n_sess"],
            "res": {k: {"view_chars": v["view_chars"], "raw_chars": v["raw_chars"],
                        "turns": sorted(v["turns"]), "prov": v["prov"],
                        "sess": sorted(v["sess"]), "n_chains": v["n_chains"]}
                    for k, v in r["res"].items()}} for r in out],
          open(f"{R}/_verify_abs.json", "w"))
print("saved _verify_abs.json")
