"""Reconstruct v5 context for the 24 MS failures from cached extractions only (no API)."""
import hashlib, json, os, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
import lifecycle as LC
from lifecycle import assemble_chains, render_view, _session_text
from query_classifier import classify_query
import run_qc

CACHE = LC.SESSION_CACHE_DIR
miss = [0]

def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(CACHE, f"{key}.json")
    if not os.path.exists(p):
        miss[0] += 1
        return []
    with open(p) as f:
        return json.load(f)

def build_store(q):
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for ent, attr, val in cached_facts(sess):
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    return assemble_chains(facts)

def v5_context(q):
    label, _ = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    store = build_store(q)
    view = run_qc._rank_view_lines(store, mode, q["question"], run_qc.VIEW_CHAR_CAP)
    units, sel = run_qc._select_turns_v5(q, q["question"], run_qc.TOP_K)
    raw = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sel)
    return label, mode, store, view, raw, units, sel

if __name__ == "__main__":
    errs = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/_v6_ms_errors.json"))}
    data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
    out = {}
    for q in data:
        if q["question_id"] not in errs:
            continue
        label, mode, store, view, raw, units, sel = v5_context(q)
        # gold evidence turns
        gold_turns = []
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            for t in sess:
                if t.get("has_answer"):
                    gold_turns.append({"sidx": si, "date": date, "role": t["role"], "content": t.get("content", "")})
        out[q["question_id"]] = {
            "question": q["question"], "gold": q["answer"], "qdate": q["question_date"],
            "label": label, "mode": mode, "n_store": len(store),
            "view": view, "raw": raw, "gold_turns": gold_turns,
            "ctx_chars": len(view) + len(raw),
            "store": {k: [{kk: vv for kk, vv in c.items()} for c in v] for k, v in store.items()},
        }
        print(q["question_id"], "store", len(store), "viewchars", len(view), "rawchars", len(raw), "gold_turns", len(gold_turns), flush=True)
    print("cache misses:", miss[0])
    json.dump(out, open("/root/agent_memory_iclr/results/_v6_ctx.json", "w"), ensure_ascii=False)
