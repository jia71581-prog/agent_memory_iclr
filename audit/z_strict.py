"""Robustness of the 221/186/35 triple to the gold matcher: substring vs strict token-sequence."""
import hashlib, json, os, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
common.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked"))
import lifecycle; lifecycle.chat = common.chat
from lifecycle import assemble_chains, render_view
from collections import Counter

CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
CAP = 20000
R = "/root/agent_memory_iclr/results/"
qc = {r["qid"]: r for r in (json.loads(l) for l in open(R + "full500_qc.jsonl")) if "judge" in r}
vo = {r["qid"]: r for r in (json.loads(l) for l in open(R + "full500_qc_view_only.jsonl")) if "judge" in r}
byid = {q["question_id"]: q for q in common.load_dataset("s_cleaned")}
tok = lambda s: re.findall(r"[a-z0-9]+", str(s).lower())


def cap(txt):
    if len(txt) <= CAP: return txt
    kept, t = [], 0
    for line in txt.splitlines():
        if t + len(line) + 1 > CAP: break
        kept.append(line); t += len(line) + 1
    return "\n".join(kept)


def subseq(g, v):
    """contiguous token-subsequence containment (word-boundary safe)"""
    if not g: return False
    L = len(g)
    return any(v[i:i + L] == g for i in range(len(v) - L + 1))


s = Counter()
lost_types = Counter()
for qid, r in qc.items():
    q = byid[qid]
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = "\n".join(f"[{t['role']}] {t['content']}" for t in sess if t.get("content"))[:24000]
        if not text: continue
        p = os.path.join(CACHE, hashlib.md5(text.encode()).hexdigest()[:16] + ".json")
        for ent, attr, val in json.load(open(p)):
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    store = assemble_chains(facts)
    full = render_view(store, r["view_mode"])
    g = tok(q["answer"])
    a = subseq(g, tok(full)); b = subseq(g, tok(cap(full)))
    s["n"] += 1; s["full"] += a; s["cap"] += b
    if a and not b:
        s["lost"] += 1; lost_types[r["type"]] += 1
        s["lost_qc_ok"] += r["judge"]; s["lost_vo_ok"] += vo[qid]["judge"]

print("STRICT token-subsequence matcher:")
print("  gold in uncapped: %d/%d (%.3f)" % (s["full"], s["n"], s["full"] / s["n"]))
print("  gold in capped  : %d/%d (%.3f)" % (s["cap"], s["n"], s["cap"] / s["n"]))
print("  destroyed by cap: %d (%.3f)" % (s["lost"], s["lost"] / s["n"]))
print("  of those, qc correct %d/%d, view_only correct %d/%d" % (s["lost_qc_ok"], s["lost"], s["lost_vo_ok"], s["lost"]))
print("  lost type mix:", dict(lost_types))
