"""Offline reconstruction of the two arms' contexts from the frozen lifecycle cache.
No API. Tests (a) byte-identical premise for the 44 same-mode questions,
(b) how many of the 456 routed-different questions actually get a DIFFERENT view body.
"""
import json, os, sys, hashlib, math, re
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _session_text
from query_classifier import classify_query
from run_baseline import bm25_rank

CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
VIEW_CHAR_CAP = 20000
R = "/root/agent_memory_iclr/results/"

def load(tag):
    return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{tag}.jsonl"))}
qc, nq = load("qc"), load("qc_no_qc")

data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
print("dataset n:", len(data))

def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return [], True
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(CACHE, f"{key}.json")
    if not os.path.exists(p):
        return [], False
    with open(p) as f:
        return json.load(f), True

def cap(view_txt):
    if len(view_txt) <= VIEW_CHAR_CAP:
        return view_txt
    kept, total = [], 0
    for line in view_txt.splitlines():
        if total + len(line) + 1 > VIEW_CHAR_CAP:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)

HEAD_CUR = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
            "at the same time that could not be ordered)")
HEAD_HIST = "Fact evolution view (full value history per fact, oldest -> newest)"
SYS = "x"

miss = 0
rec = {}
for i, q in enumerate(data):
    facts = []
    ok_all = True
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        trips, ok = cached_facts(sess)
        ok_all &= ok
        for ent, attr, val in trips:
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    if not ok_all:
        miss += 1
    store = assemble_chains(facts)
    vc = cap(render_view(store, "current"))
    vh = cap(render_view(store, "historical"))
    label, _ = classify_query(q["question"])
    rec[q["question_id"]] = {"label": label, "vc": vc, "vh": vh,
                             "nchain": len(store),
                             "multi": sum(1 for k in store if len({(c['date'], c.get('sidx')) for c in store[k]}) > 1)}
    if (i + 1) % 100 == 0:
        print("  built", i + 1, flush=True)

print("questions with any uncached session:", miss)

qids = list(rec)
print("\n== my classifier reproduces stored qlabel:",
      all(rec[q]["label"] == qc[q]["qlabel"] for q in qids))
print("my label dist:", Counter(rec[q]["label"] for q in qids))

diff = [q for q in qids if qc[q]["view_mode"] != nq[q]["view_mode"]]
same = [q for q in qids if qc[q]["view_mode"] == nq[q]["view_mode"]]

print("\n== (a) same-mode bucket: is the context byte-identical? ==")
print(f"n={len(same)}; both arms render mode=historical -> view text identical: ",
      all(rec[q]["vh"] == rec[q]["vh"] for q in same))
print("  (tautological given one frozen store; store frozen is verified by 0 extract calls in both ledgers"
      " and last cache write 16:24:24 < qc 16:32 < no_qc 16:42)")

print("\n== (b) routed-different bucket: does the VIEW BODY actually differ? ==")
body_same = [q for q in diff if rec[q]["vc"] == rec[q]["vh"]]
body_diff = [q for q in diff if rec[q]["vc"] != rec[q]["vh"]]
print(f"n=456 -> view body identical in {len(body_same)} ({len(body_same)/len(diff):.3f}), "
      f"differs in {len(body_diff)} ({len(body_diff)/len(diff):.3f})")
ratio = [len(rec[q]["vh"]) / max(1, len(rec[q]["vc"])) for q in body_diff]
print("  mean len(hist)/len(cur) among differing:", round(sum(ratio) / len(ratio), 3))
print("  mean #chains:", round(sum(rec[q]["nchain"] for q in diff) / len(diff), 1),
      " mean #multi-session chains:", round(sum(rec[q]["multi"] for q in diff) / len(diff), 1))

def exact_two_sided(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    obs = math.comb(n, n01) / 2 ** n
    return min(1.0, sum(math.comb(n, i) / 2 ** n for i in range(n + 1)
                        if math.comb(n, i) / 2 ** n <= obs * (1 + 1e-12)))

def report(name, ids):
    if not ids:
        print(f"[{name}] empty"); return
    a = sum(qc[q]["judge"] for q in ids); b = sum(nq[q]["judge"] for q in ids)
    n01 = sum(1 for q in ids if qc[q]["judge"] and not nq[q]["judge"])
    n10 = sum(1 for q in ids if nq[q]["judge"] and not qc[q]["judge"])
    print(f"[{name}] n={len(ids)} qc={a} ({a/len(ids):.4f}) no_qc={b} ({b/len(ids):.4f}) "
          f"delta={(a-b)/len(ids):+.4f} ({a-b:+d}q) flips {n01}/{n10} p={exact_two_sided(n01,n10):.4f}")

print("\n== sharpened ablation: only where the view body truly differs ==")
report("VIEW-BODY-DIFFERS", body_diff)
report("VIEW-BODY-IDENTICAL (header-only diff)", body_same)
report("SAME-MODE (fully identical prompt)", same)
print("\nper question type, within VIEW-BODY-DIFFERS:")
for t in sorted({qc[q]["type"] for q in body_diff}):
    report("   " + t, [q for q in body_diff if qc[q]["type"] == t])
print("\nper question type, within SAME-MODE:")
for t in sorted({qc[q]["type"] for q in same}):
    report("   " + t, [q for q in same if qc[q]["type"] == t])

json.dump({q: {"label": rec[q]["label"], "body_differs": rec[q]["vc"] != rec[q]["vh"],
               "nchain": rec[q]["nchain"], "multi": rec[q]["multi"]} for q in qids},
          open("/root/agent_memory_iclr/audit/verify_viewrecon.json", "w"))
print("\nsaved /root/agent_memory_iclr/audit/verify_viewrecon.json")
