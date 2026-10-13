"""Offline v5-vs-v6 context comparison over all fired questions (cache-only, zero API)."""
import hashlib, json, os, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from lifecycle import assemble_chains, render_view, _session_text
from query_classifier import classify_query
import run_qc, v6_agg

CACHE = LC.SESSION_CACHE_DIR
misses = [0]

def cached(sess):
    t = _session_text(sess)
    if not t:
        return []
    p = os.path.join(CACHE, hashlib.md5(t.encode()).hexdigest()[:16] + ".json")
    if not os.path.exists(p):
        misses[0] += 1
        return []
    with open(p) as f:
        return json.load(f)

def store_of(q):
    facts = []
    for si, (d, s) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for e, a, v in cached(s):
            facts.append({"entity": e, "attribute": a, "value": v, "date": d, "sidx": si})
    return assemble_chains(facts)

def contexts(q):
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = store_of(q)
    v5_view = run_qc._rank_view_lines(store, mode, q["question"], run_qc.VIEW_CHAR_CAP)
    units, sel = run_qc._select_turns_v5(q, q["question"], run_qc.TOP_K)
    raw = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sel)
    v6_view, ledger = v6_agg.build_v6_sections(store, mode, q["question"], q["question_date"],
                                               run_qc.VIEW_CHAR_CAP, 6000)
    return dict(mode=mode, store=store, v5_view=v5_view, v6_view=v6_view, ledger=ledger, raw=raw)

def kept_chains(store, mode, view):
    lines = set(view.splitlines())
    return sum(1 for k, ch in store.items() if render_view({k: ch}, mode) in lines)

if __name__ == "__main__":
    rows = {json.loads(l)["qid"]: json.loads(l) for l in open("/root/agent_memory_iclr/results/full500_qc_v5.jsonl")}
    data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    want = set(sys.argv[1:]) or None
    out, agg = {}, []
    for q in data:
        qid = q["question_id"]
        r = rows.get(qid)
        if not r:
            continue
        fired = v6_agg.is_agg_question(q["question"])
        if want is not None and qid not in want:
            continue
        if want is None and not fired:
            continue
        c = contexts(q)
        v5_chars = len(c["v5_view"]) + len(c["raw"])
        v6_chars = len(c["v6_view"]) + len(c["ledger"]) + len(c["raw"])
        rec = dict(qid=qid, type=r["type"], judge=r["judge"], fired=fired,
                   question=q["question"], gold=q["answer"], qdate=q["question_date"],
                   n_chains=len(c["store"]), mode=c["mode"],
                   v5_view_chars=len(c["v5_view"]), v6_view_chars=len(c["v6_view"]),
                   ledger_chars=len(c["ledger"]), ledger_lines=len(c["ledger"].splitlines()) if c["ledger"] else 0,
                   raw_chars=len(c["raw"]), v5_chars=v5_chars, v6_chars=v6_chars,
                   v5_kept=kept_chains(c["store"], c["mode"], c["v5_view"]),
                   v6_kept=kept_chains(c["store"], c["mode"], c["v6_view"]))
        agg.append(rec)
        out[qid] = dict(rec, v5_view=c["v5_view"], v6_view=c["v6_view"], ledger=c["ledger"], raw=c["raw"])
        print(f"{qid:16s} {r['type'][:18]:18s} fired={fired} judge={'Y' if r['judge'] else 'N'} "
              f"chains={len(c['store'])} ledger={len(c['ledger'])}c/{rec['ledger_lines']}L "
              f"view {len(c['v5_view'])}->{len(c['v6_view'])} kept {rec['v5_kept']}->{rec['v6_kept']} "
              f"ctx {v5_chars}->{v6_chars} ({100*(v6_chars/max(1,v5_chars)-1):+.2f}%)", flush=True)
    json.dump(out, open("/root/agent_memory_iclr/results/_v6_cmp.json", "w"), ensure_ascii=False)
    json.dump(agg, open("/root/agent_memory_iclr/results/_v6_cmp_agg.json", "w"), ensure_ascii=False, indent=1)
    print("cache misses:", misses[0], "n:", len(agg))
