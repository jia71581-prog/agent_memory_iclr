"""Test 4: audit the A3 cost figure. coverage_diag.json measures in_view on the UNCAPPED view
and with a bag-of-tokens matcher. Recompute under (a) their definition, (b) capped view, (c) strict substring."""
import json, os, re, sys, hashlib
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from run_baseline import bm25_rank
from query_classifier import classify_query
from _zver_core import load, cached_extract
from coverage_diag import contains, norm

RES = "/root/agent_memory_iclr/results"
diag = json.load(open(f"{RES}/coverage_diag.json"))
qc = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{RES}/full500_qc.jsonl")}
D = {q["question_id"]: q for q in load()}

non = [r for r in diag if not r["is_abs"]]
vo = [r for r in non if r["in_view"] and not r["in_raw20"]]
nei = [r for r in non if not r["in_view"] and not r["in_raw20"]]
print("=== reproduce A3 exactly as the module computes it (coverage_diag.json, UNCAPPED view) ===")
print("answerable n=%d" % len(non))
print("gold only in view : %d ; currently correct %d" % (len(vo), sum(r["judge"] for r in vo)))
print("gold in neither   : %d ; graded correct   %d" % (len(nei), sum(r["judge"] for r in nei)))
print("view coverage %.3f | raw20 coverage %.3f" % (
    sum(r["in_view"] for r in non) / len(non), sum(r["in_raw20"] for r in non) / len(non)))

print("\n=== same questions, but in_view measured on the view the pipeline ACTUALLY showed (20k cap) ===")
LC.extract_session_facts = cached_extract
rows = []
for i, r in enumerate(non):
    q = D[r["qid"]]
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    full = LC.render_view(LC.build_lifecycle(q, None), mode)
    view = full
    if len(view) > 20000:
        kept, tot = [], 0
        for l in view.splitlines():
            if tot + len(l) + 1 > 20000:
                break
            kept.append(l); tot += len(l) + 1
        view = "\n".join(kept)
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order = bm25_rank(q["question"], [u[2] for u in units])
    raw20 = " ".join(units[j][2] for j in order[:20])
    gold = q["answer"]
    rows.append({"qid": r["qid"], "judge": r["judge"],
                 "in_view_full_loose": contains(full, gold),
                 "in_view_cap_loose": contains(view, gold),
                 "in_view_cap_strict": norm(gold) in norm(view),
                 "in_view_full_strict": norm(gold) in norm(full),
                 "in_raw20_loose": contains(raw20, gold),
                 "in_raw20_strict": norm(gold) in norm(raw20)})
    if i % 100 == 0:
        print("  ..%d" % i, flush=True)
json.dump(rows, open(f"{RES}/_zver_a3.json", "w"))

def rep(vk, rk, name):
    v = [r for r in rows if r[vk] and not r[rk]]
    n = [r for r in rows if not r[vk] and not r[rk]]
    print("%-42s view-only %3d (correct %3d) | neither %3d (correct %3d)" % (
        name, len(v), sum(r["judge"] for r in v), len(n), sum(r["judge"] for r in n)))

rep("in_view_full_loose", "in_raw20_loose", "UNCAPPED view, loose matcher (module's A3)")
rep("in_view_cap_loose", "in_raw20_loose", "CAPPED view (as shown), loose matcher")
rep("in_view_full_strict", "in_raw20_strict", "UNCAPPED view, strict substring")
rep("in_view_cap_strict", "in_raw20_strict", "CAPPED view (as shown), strict substring")

vo_full = {r["qid"] for r in rows if r["in_view_full_loose"] and not r["in_raw20_loose"]}
vo_cap = {r["qid"] for r in rows if r["in_view_cap_loose"] and not r["in_raw20_loose"]}
print("\nof the module's %d 'view-only support' questions, %d lose that support once the 20k cap is applied"
      % (len(vo_full), len(vo_full - vo_cap)))

print("\n=== is the loose matcher a bag-of-words artifact? ===")
loose_only = [r for r in rows if r["in_view_full_loose"] and not r["in_view_full_strict"]]
print("in_view_full: loose=%d strict=%d -> %d (%.0f%%) are bag-of-token matches only"
      % (sum(r["in_view_full_loose"] for r in rows), sum(r["in_view_full_strict"] for r in rows),
         len(loose_only), 100 * len(loose_only) / max(1, sum(r["in_view_full_loose"] for r in rows))))
print("examples of bag-of-token-only 'support' (gold | matched anywhere in the 30k view):")
for r in loose_only[:8]:
    print("   %-22s gold=%r" % (r["qid"], D[r["qid"]]["answer"][:70]))
