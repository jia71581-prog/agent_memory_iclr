"""Adversarial part 2: does the offline proxy predict correctness? regression risk? context cost?"""
import json, math, random, statistics as st, sys, glob, re
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V
from common import load_dataset

R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_ver3_v5t_rows.json"))
rmap = {r["qid"]: r for r in rows}
allrows = json.load(open(f"{R}/v5_temporal_all.json"))

# ---- context / size claim
ch = [r["chars"] for r in allrows]
print("=== SIZE (all 500) ===")
print(f"n={len(ch)} median={st.median(ch)} mean={round(st.mean(ch))} "
      f"p90={sorted(ch)[int(.9*len(ch))]} max={max(ch)}")
tr = [r["chars"] for r in allrows if r.get("type") == "temporal-reasoning"]
print(f"TR n={len(tr)} median={st.median(tr)} mean={round(st.mean(tr))}")

# ---- judge labels from the live pipeline
qc = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{R}/full500_qc.jsonl")}
data = {q["question_id"]: q for q in load_dataset("s_cleaned")}

def jd(qid):
    r = qc.get(qid)
    return None if r is None else bool(r["judge"])

print("\n=== TR accuracy in the live qc run ===")
trq = [r for r in rows]
lab = [(r["qid"], jd(r["qid"])) for r in trq]
known = [x for x in lab if x[1] is not None]
print(f"TR n={len(trq)} with judge={len(known)} acc={sum(v for _,v in known)/len(known):.3f}")
absten = [qid for qid, _ in known if qid.endswith("_abs")]
print(f"abstention qids among TR: {len(absten)}")

print("\n=== Does the delta proxy discriminate correct vs wrong ANSWERS? ===")
dg = [r for r in rows if r["family"] in ("delta_between", "delta_since") and r["gold_vals"]]
for name, sel in [("live judge CORRECT", True), ("live judge WRONG", False)]:
    g = [r for r in dg if jd(r["qid"]) is sel]
    if not g:
        continue
    print(f"{name:20s} n={len(g):3d} derivTL={sum(r['gold_derivable_from_timeline'] for r in g)/len(g):.3f} "
          f"literal={sum(r['gold_literal_in_arith'] for r in g)/len(g):.3f} "
          f"derivEvid={sum(r['gold_derivable_from_evidence'] for r in g)/len(g):.3f}")

print("\n=== ORDERING: regression risk (scaffold asserts an order under 'trust these numbers') ===")
o = [r for r in rows if r["family"] == "ordering"]
oi = [r for r in o if r.get("ordering_item_correct") is not None]
print(f"ordering n={len(o)} computable={len(oi)} correct={sum(r['ordering_item_correct'] for r in oi)}")
bi = [r for r in oi if r["ordering_n_items"] == 2]
k = sum(r["ordering_item_correct"] for r in bi)
n = len(bi)
p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
print(f"binary subfamily {k}/{n}={k/n:.3f}; one-sided binomial vs 0.5: p={p:.3f}")
kk = sum(r["ordering_item_correct"] for r in oi); nn = len(oi)
p2 = sum(math.comb(nn, i) for i in range(kk, nn + 1)) / 2 ** nn
print(f"all computable {kk}/{nn}={kk/nn:.3f}; vs 0.5: p={p2:.3f}")
print(f"of ALL ordering questions, scaffold order correct = {kk}/{len(o)} = {kk/len(o):.3f}")

live_ok = [r for r in o if jd(r["qid"]) is True]
print(f"\nordering questions the LIVE pipeline already answers correctly: {len(live_ok)}/{len(o)}")
bad = [r for r in live_ok if r.get("ordering_item_correct") is False]
noncomp = [r for r in live_ok if r.get("ordering_item_correct") is None]
print(f"  of those, scaffold would assert a WRONG order: {len(bad)}  -> {[r['qid'] for r in bad]}")
print(f"  of those, scaffold cannot place all named items: {len(noncomp)}")
gain = [r for r in o if jd(r["qid"]) is False and r.get("ordering_item_correct") is True]
print(f"ordering questions live-WRONG where scaffold order is right (upside): {len(gain)}")
print(f"  -> naive net on ordering: +{len(gain)} / -{len(bad)} (at-risk)")

print("\n=== ANCHOR_LOOKUP null control ===")
al = [r for r in rows if r["family"] == "anchor_lookup"]
print(f"n={len(al)} anchor_hit(<=3d)={sum(bool(r.get('anchor_hit')) for r in al)/len(al):.3f} "
      f"day_has_gold={sum(bool(r.get('anchor_day_has_gold')) for r in al)/len(al):.3f} "
      f"stem={sum(bool(r.get('anchor_day_has_gold_stem')) for r in al)/len(al):.3f}")
# null: a random haystack session date; how often within 3d of the gold evidence date?
nulls = []
for r in al:
    q = data[r["qid"]]
    ds = [V.day0(V.parse_date(x)) for x in q["haystack_dates"] if V.parse_date(x)]
    tgts = set()
    sid2 = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
    for sid in q.get("answer_session_ids") or []:
        d = V.parse_date(sid2.get(sid, ""))
        if d:
            tgts.add(V.day0(d))
    if not tgts or not ds:
        continue
    hit = sum(any(abs((d - t).days) <= 3 for t in tgts) for d in ds) / len(ds)
    nulls.append(hit)
print(f"NULL (uniform random haystack session date within 3d of an evidence date): {st.mean(nulls):.3f}")
span = []
for r in al:
    q = data[r["qid"]]
    ds = sorted(V.day0(V.parse_date(x)) for x in q["haystack_dates"] if V.parse_date(x))
    span.append((ds[-1] - ds[0]).days)
print(f"haystack date span days: median={st.median(span)} n_sessions median="
      f"{st.median([len(data[r['qid']]['haystack_dates']) for r in al])}")

print("\n=== anchor_lookup: live-correct at risk from a wrong anchor ===")
for r in al:
    if r.get("anchor_err_days") not in (0, None) and r["anchor_err_days"] > 3:
        print(f"  {r['qid']} err={r['anchor_err_days']}d live_judge={jd(r['qid'])} gold={r['gold'][:60]}")
