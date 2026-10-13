import hashlib, json, os, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import common
common.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked"))
import lifecycle; lifecycle.chat = common.chat
from lifecycle import assemble_chains, render_view

CACHE = "/root/agent_memory_iclr/results/lifecycle_cache/sessions_nt"
recs = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/audit/z_recs.json"))}
lost = [r for r in recs.values() if r["gold_in_full"] and not r["gold_in_cap"]]
data = common.load_dataset("s_cleaned")
byid = {q["question_id"]: q for q in data}
nrm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()

print("abstention qids in dataset:", sum(1 for q in data if q["question_id"].endswith("_abs")))
print("abstention among lost:", sum(1 for r in lost if r["qid"].endswith("_abs")))
print("abstention with gold_in_full:", sum(1 for r in recs.values() if r["qid"].endswith("_abs") and r["gold_in_full"]))
print()

targets = sorted(lost, key=lambda r: r["gold_len"])[:9] + sorted(lost, key=lambda r: -r["gold_len"])[:4]
for r in targets:
    q = byid[r["qid"]]
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        text = "\n".join(f"[{t['role']}] {t['content']}" for t in sess if t.get("content"))[:24000]
        if not text: continue
        p = os.path.join(CACHE, hashlib.md5(text.encode()).hexdigest()[:16] + ".json")
        for ent, attr, val in json.load(open(p)):
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    store = assemble_chains(facts)
    lines = render_view(store, r["mode"]).split("\n")
    g = nrm(q["answer"])
    hits = [ln for ln in lines if g in nrm(ln)]
    print("=" * 100)
    print(r["qid"], r["type"], "mode=", r["mode"])
    print("  Q   :", q["question"][:170])
    print("  GOLD:", repr(q["answer"])[:120])
    for h in hits[:3]:
        print("  HIT :", h[:190])
