"""(a) failure-anatomy of the 19 TR questions the deployed pipeline gets wrong;
(b) re-test the 'null' components (sweep, 3 weights) in the SHIPPED config, not the
    stale config the reported JSONs were measured in. Zero API."""
import json, os, sys, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_temporal as V
from common import load_dataset
from v5_temporal_measure import analyse
from v5_temporal_val import build_store_cached

R = "/root/agent_memory_iclr/results"
D = json.load(open(f"{R}/_ver_repro.json"))
FIN, OFF = D["per_q"]["final"], D["per_q"]["no_intext"]
J = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
tr = [k for k in FIN]
bad = [k for k in tr if not J[k]["judge"]]
print(f"deployed qc TR wrong: {len(bad)}/{len(tr)}")
print("family of the failures:", Counter(FIN[k]["fam"] for k in bad))
dbad = [k for k in bad if FIN[k]["fam"] in ("delta_between", "delta_since")]
print(f"  delta failures n={len(dbad)}: dTL already True on {sum(bool(FIN[k]['dTL']) for k in dbad)}"
      f", flips ON->OFF on {sum(1 for k in dbad if FIN[k]['dTL'] and not OFF[k]['dTL'])}")
obad = [k for k in bad if FIN[k]["fam"] == "ordering"]
print(f"  ordering failures n={len(obad)}: scaffold ordering verdict "
      + str(Counter(str(FIN[k]['ordc']) for k in obad)))
print(f"  dates_present==1.0 on {sum(1 for k in bad if FIN[k]['dpres'] >= 0.999)}/{len(bad)} failures")
print("  failures where in-text resolution changes ANY of dTL/ordc:",
      sum(1 for k in bad if (bool(FIN[k]['dTL']) != bool(OFF[k]['dTL'])
                             or bool(FIN[k]['ordc']) != bool(OFF[k]['ordc']))))

# ---------- (b) null components re-tested in shipped config
data = [q for q in load_dataset("s_cleaned") if q["question_type"] == "temporal-reasoning"]
stores = {q["question_id"]: build_store_cached(q)[0] for q in data}


def run(tag, **kw):
    for k, v in kw.items():
        setattr(V, k, v)
    rows = [analyse(q, stores[q["question_id"]])[0] for q in data]
    dg = [r for r in rows if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
    o = [r for r in rows if r["family"] == "ordering"]
    oi = [r for r in o if r.get("ordering_item_correct") is not None]
    al = [r for r in rows if r["family"] == "anchor_lookup"]
    m = {"deltaTL": sum(r["gold_derivable_from_timeline"] for r in dg) / len(dg),
         "ordComp": len(oi) / len(o),
         "ordOK": sum(bool(r.get("ordering_item_correct")) for r in o) / len(o),
         "anchor_hit": sum(bool(r.get("anchor_hit")) for r in al) / len(al),
         "datesPres": st.mean([r["dates_present"] for r in rows]),
         "chars": st.median([r["chars"] for r in rows])}
    sig = tuple(sorted((r["qid"], bool(r["gold_derivable_from_timeline"]),
                        str(r.get("ordering_item_correct")), r["chars"]) for r in rows))
    print(f"{tag:38s} " + "  ".join(f"{k}={v:.4g}" for k, v in m.items()), flush=True)
    return m, hash(sig)


base = dict(USE_STEM=True, USE_QFORM_STOP=False, USE_SWEEP=False,
            W_CUE=1.0, W_ANCHOR=1.0, W_INTENT=1.0)
m0, h0 = run("SHIPPED", **base)
m1, h1 = run("shipped + predicate sweep ON", **dict(base, USE_SWEEP=True))
print("  sweep bit-identical to shipped:", h1 == h0)
m2, h2 = run("shipped + qform stoplist ON", **dict(base, USE_QFORM_STOP=True))
print("  qform bit-identical:", h2 == h0)
for name, c, a, i in [("cue 1.3", 1.3, 1.0, 1.0), ("cue+anchor 1.3/1.5", 1.3, 1.5, 1.0),
                      ("cue+anchor+intent 1.3/1.5/0.4", 1.3, 1.5, 0.4),
                      ("anchor heavy 1.3/2.0/0.4", 1.3, 2.0, 0.4),
                      ("intent only 0.4", 1.0, 1.0, 0.4)]:
    m, h = run(f"shipped + {name}", **dict(base, W_CUE=c, W_ANCHOR=a, W_INTENT=i))
    print(f"  bit-identical to shipped: {h == h0}")
