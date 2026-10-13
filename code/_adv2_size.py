"""ADVERSARIAL VERIFIER #2 step 1: recompute the reported cost/size numbers independently.
Claim: 0 API calls; 26 ms/question; +1766 chars median (~442 tok) on 227/500; +8.83% context on
those; +4.50% mean over 500; cap binds 468/500; merged view fits 67.6% of store vs 66.0%."""
import json
import pickle
import statistics as st
import sys
import time

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view
from query_classifier import classify_query
from v5_chains import merge_chains, render_aggregation_view, aggregation_kind

CAP = 20000
D = pickle.load(open("/root/agent_memory_iclr/results/_adv2_facts.pkl", "rb"))


def capped(txt, cap=CAP):
    if len(txt) <= cap:
        return txt, False
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), True


rows = []
t_merge = t_agg = t_view = 0.0
for x in D:
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"
    s4 = assemble_chains(x["facts"])
    t0 = time.perf_counter()
    s5 = merge_chains(s4)
    t_merge += time.perf_counter() - t0
    t0 = time.perf_counter()
    v4f = render_view(s4, mode)
    v5f = render_view(s5, mode)
    t_view += time.perf_counter() - t0
    t0 = time.perf_counter()
    blk = render_aggregation_view(s5, x["question"])
    t_agg += time.perf_counter() - t0
    v4, tr4 = capped(v4f)
    v5, tr5 = capped(v5f)
    rows.append(dict(
        qid=x["qid"], type=x["type"], mode=mode,
        kind=aggregation_kind(x["question"]) or "",
        fired=bool(blk), blk=len(blk),
        n4=len(s4), n5=len(s5),
        v4f=len(v4f), v5f=len(v5f),
        v4=len(v4), v5=len(v5), tr4=tr4, tr5=tr5,
        lines4f=v4f.count("\n") + 1, lines5f=v5f.count("\n") + 1,
        lines4=v4.count("\n") + 1, lines5=v5.count("\n") + 1,
        v5tot=len(v5) + (len(blk) + 2 if blk else 0),
    ))

json.dump(rows, open("/root/agent_memory_iclr/results/_adv2_size.json", "w"))
fired = [r for r in rows if r["fired"]]
n = len(rows)
print(f"n={n}  aggregation block fires on {len(fired)}")
print(f"aggregation_kind non-null (question matches a pattern): "
      f"{sum(1 for r in rows if r['kind'])}  -> block empty on "
      f"{sum(1 for r in rows if r['kind'] and not r['fired'])} of them")
print(f"\nblock chars on the {len(fired)} firing questions: median {st.median([r['blk'] for r in fired]):.0f} "
      f"mean {st.mean([r['blk'] for r in fired]):.0f} max {max(r['blk'] for r in fired)}")
print(f"  approx tokens (chars/4): median {st.median([r['blk'] for r in fired])/4:.0f}")

# context growth, matching the design agent's definition: capped view + block
def pct(sub, a="v4", b="v5tot"):
    A, B = st.mean([r[a] for r in sub]), st.mean([r[b] for r in sub])
    return A, B, (B - A) / A
print("\n--- context (capped view v4 vs capped view v5 + uncapped block) ---")
for nm, sub in [("ALL 500", rows), (f"fires ({len(fired)})", fired),
                (f"no block ({n-len(fired)})", [r for r in rows if not r['fired']])]:
    A, B, p = pct(sub)
    mA, mB = st.median([r['v4'] for r in sub]), st.median([r['v5tot'] for r in sub])
    print(f"{nm:20s} mean {A:8.0f} -> {B:8.0f} ({p:+.2%})   median {mA:8.0f} -> {mB:8.0f} "
      f"({(mB-mA)/mA:+.2%})  median delta {mB-mA:+.0f} chars")
# per-question relative growth (the '+8.83%' could also be a per-question mean)
import statistics
rel_f = [(r["v5tot"] - r["v4"]) / r["v4"] for r in fired]
rel_all = [(r["v5tot"] - r["v4"]) / r["v4"] for r in rows]
print(f"per-question relative growth: fired mean {st.mean(rel_f):+.2%} median {st.median(rel_f):+.2%};"
      f" all500 mean {st.mean(rel_all):+.2%}")
print(f"median per-question delta chars on fired: {st.median([r['v5tot']-r['v4'] for r in fired]):+.0f}")

print("\n--- cap binding ---")
print(f"v4 uncapped view > CAP: {sum(r['tr4'] for r in rows)}/500 ; v5 view > CAP: {sum(r['tr5'] for r in rows)}/500")
print(f"v4 capped len >= CAP-200: {sum(r['v4']>=CAP-200 for r in rows)}/500 ; "
      f"v5: {sum(r['v5']>=CAP-200 for r in rows)}/500")

print("\n--- store fit under the cap (candidate definitions of '67.6% vs 66.0%') ---")
d1b = st.mean([min(1.0, r['lines4']/r['lines4f']) for r in rows])
d1a = st.mean([min(1.0, r['lines5']/r['lines5f']) for r in rows])
print(f"(A) mean per-q fraction of view LINES inside cap: v4 {d1b:.4f} -> v5 {d1a:.4f}")
d2b = sum(r['lines4'] for r in rows)/sum(r['lines4f'] for r in rows)
d2a = sum(r['lines5'] for r in rows)/sum(r['lines5f'] for r in rows)
print(f"(B) pooled lines inside cap / total lines:        v4 {d2b:.4f} -> v5 {d2a:.4f}")
d3b = st.mean([min(1.0, r['v4']/r['v4f']) for r in rows])
d3a = st.mean([min(1.0, r['v5']/r['v5f']) for r in rows])
print(f"(C) mean per-q fraction of view CHARS inside cap: v4 {d3b:.4f} -> v5 {d3a:.4f}")
d4b = st.mean([min(1.0, r['lines4']/r['n4']) for r in rows])
d4a = st.mean([min(1.0, r['lines5']/r['n5']) for r in rows])
print(f"(D) mean per-q lines-in-cap / n_chains:           v4 {d4b:.4f} -> v5 {d4a:.4f}")

print("\n--- chain counts ---")
print(f"chains median v4 {st.median([r['n4'] for r in rows]):.1f} -> v5 {st.median([r['n5'] for r in rows]):.1f} "
      f"({st.mean([(r['n5']-r['n4'])/r['n4'] for r in rows]):+.2%} per-q mean)")

print("\n--- CPU time (single process, no API) ---")
print(f"merge_chains       {t_merge/n*1000:7.1f} ms/question")
print(f"render_aggregation {t_agg/n*1000:7.1f} ms/question   (on the {len(fired)} firing: "
      f"{t_agg/max(1,len(fired))*1000:7.1f} ms)")
print(f"render_view x2     {t_view/n*1000:7.1f} ms/question")
print(f"TOTAL v5 added     {(t_merge+t_agg)/n*1000:7.1f} ms/question")
