"""Reconstruct the exact context both arms saw, offline, from the frozen extraction cache.
Tests: (1) is same-view_mode really byte-identical prompt?
       (2) are some routed-different questions ALSO byte-identical (current view == historical view)?
"""
import json, os, sys, hashlib, re
from collections import Counter
sys.path.insert(0, "/root/agent_memory_iclr/code")
sys.path.insert(0, "/root/agent_memory_iclr/results")
import lifecycle as LC
from run_baseline import bm25_rank
from query_classifier import classify_query
from common import load_dataset

VIEW_CHAR_CAP = 20000
TOP_K = 20
miss = [0]

def extract_cached(sess):
    text = LC._session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(LC.SESSION_CACHE_DIR, f"{key}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    miss[0] += 1
    return None            # signal miss explicitly

def build_store(q):
    facts, m = [], 0
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        tr = extract_cached(sess)
        if tr is None:
            m += 1
            continue
        for ent, attr, val in tr:
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    return LC.assemble_chains(facts), m

def cap(v):
    if len(v) <= VIEW_CHAR_CAP:
        return v
    kept, tot = [], 0
    for line in v.splitlines():
        if tot + len(line) + 1 > VIEW_CHAR_CAP:
            break
        kept.append(line); tot += len(line) + 1
    return "\n".join(kept)

def raw_of(q):
    units = []
    for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
        for t in sess:
            if t.get("content"):
                units.append((date, t["role"], t["content"]))
    order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:TOP_K])
    return "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)

HEAD_C = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
          "at the same time that could not be ordered)")
HEAD_H = "Fact evolution view (full value history per fact, oldest -> newest)"

def ctx(view, mode, raw):
    parts = []
    if view:
        parts.append(f"### {HEAD_C if mode=='current' else HEAD_H}\n{view}")
    if raw:
        parts.append(f"### Retrieved raw conversation turns\n{raw}")
    return "\n\n".join(parts)

data = load_dataset("s_cleaned")
R = "/root/agent_memory_iclr/results/"
def load(t):
    d = {}
    for line in open(R + f"full500_{t}.jsonl"):
        r = json.loads(line)
        if "judge" in r:
            d[r["qid"]] = r
    return d
qc, nq = load("qc"), load("qc_no_qc")

out = {}
sess_miss_q = 0
for q in data:
    qid = q["question_id"]
    if qid not in qc:
        continue
    label, _ = classify_query(q["question"])
    mode_qc = "historical" if label == "historical" else "current"
    store, m = build_store(q)
    if m:
        sess_miss_q += 1
    v_qc = cap(LC.render_view(store, mode_qc))
    v_nq = cap(LC.render_view(store, "historical"))
    raw = raw_of(q)
    c_qc = ctx(v_qc, mode_qc, raw)
    c_nq = ctx(v_nq, "historical", raw)
    out[qid] = {
        "label": label, "mode_qc": mode_qc,
        "mode_logged_qc": qc[qid]["view_mode"], "mode_logged_nq": nq[qid]["view_mode"],
        "view_identical": v_qc == v_nq,
        "ctx_identical": c_qc == c_nq,
        "len_qc": len(c_qc), "len_nq": len(c_nq),
        "view_lines": len(v_qc.splitlines()),
        "n_chains": len(store),
        "n_multigroup": sum(1 for k in store if len(LC._group_by_session(store[k])) > 1
                            and store[k][0]["relation"] != "accumulative"),
        "sess_miss": m,
    }
json.dump(out, open("/root/agent_memory_iclr/audit/v_nd_ctx.json", "w"), indent=0)

print(f"cache misses (session-level): {miss[0]}, questions with any miss: {sess_miss_q}")
# sanity: logged mode matches recomputed classifier mode
bad = [k for k, v in out.items() if v["mode_qc"] != v["mode_logged_qc"] or v["mode_logged_nq"] != "historical"]
print("logged-vs-recomputed mode mismatches:", len(bad))

same = [k for k, v in out.items() if v["mode_logged_qc"] == v["mode_logged_nq"]]
diff = [k for k, v in out.items() if v["mode_logged_qc"] != v["mode_logged_nq"]]
print(f"\nstrata: same_mode={len(same)} diff_mode={len(diff)}")
print("  same-mode: ctx byte-identical:", sum(out[k]["ctx_identical"] for k in same), "/", len(same))
print("  diff-mode: ctx byte-identical:", sum(out[k]["ctx_identical"] for k in diff), "/", len(diff))
print("  diff-mode: view byte-identical:", sum(out[k]["view_identical"] for k in diff), "/", len(diff))

# how much do the views actually differ on the routed stratum?
import difflib
dl = [(out[k]["len_qc"], out[k]["len_nq"]) for k in diff]
print(f"  diff-mode mean ctx chars: qc={sum(a for a,b in dl)/len(dl):.0f} nq={sum(b for a,b in dl)/len(dl):.0f}")
eq = [k for k in diff if out[k]["ctx_identical"]]
print(f"\n  routed-different questions whose prompt was in fact IDENTICAL: {len(eq)}")
if eq:
    jf = sum(1 for k in eq if qc[k]["judge"] != nq[k]["judge"])
    ex = sum(1 for k in eq if qc[k]["pred"].strip() == nq[k]["pred"].strip())
    print(f"    judge flips among them: {jf}/{len(eq)}   exact pred match {ex}/{len(eq)}")
