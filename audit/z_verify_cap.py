"""Independent recomputation of the VIEW_CHAR_CAP claim. Written from scratch by the verifier.

Does NOT reuse a6_cap.py / a7 / a11 logic. Own cache reader, own cap implementation,
own gold matcher, own per-type denominators.
"""
import hashlib, json, os, pickle, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
# hard-block any API call: everything must come from the on-disk extraction cache
import common
common.chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API BLOCKED"))
import lifecycle
lifecycle.chat = common.chat
from lifecycle import assemble_chains, render_view, _group_by_session

R = "/root/agent_memory_iclr/results/"
CACHE = R + "lifecycle_cache/sessions_nt"
CAP = 20000


def rows(tag):
    return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{tag}.jsonl")) if "judge" in r}


def sess_key(sess):
    """Reimplementation of lifecycle._session_text + md5 key (must match production)."""
    text = "\n".join(f"[{t['role']}] {t['content']}" for t in sess if t.get("content"))
    text = text[:24000]
    if not text:
        return None, ""
    return hashlib.md5(text.encode()).hexdigest()[:16], text


def build_store(q, miss):
    facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        k, txt = sess_key(sess)
        if k is None:
            continue
        p = os.path.join(CACHE, k + ".json")
        if not os.path.exists(p):
            miss.append(k)
            continue
        with open(p) as f:
            triples = json.load(f)
        for ent, attr, val in triples:
            facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    return assemble_chains(facts), len(facts)


def truncate(txt):
    """Own implementation of run_qc.py's line-boundary cap."""
    if len(txt) <= CAP:
        return txt, False
    kept, tot = [], 0
    for line in txt.split("\n"):
        if tot + len(line) + 1 > CAP:
            break
        kept.append(line)
        tot += len(line) + 1
    return "\n".join(kept), True


def nrm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def main():
    qc, nq, vo, nl = rows("qc"), rows("qc_no_qc"), rows("qc_view_only"), rows("qc_no_lifecycle")
    print("arm row counts:", {k: len(v) for k, v in
                              [("qc", qc), ("no_qc", nq), ("view_only", vo), ("no_lifecycle", nl)]})
    data = common.load_dataset("s_cleaned")
    byid = {q["question_id"]: q for q in data}
    print("dataset questions:", len(data), " type mix:", dict(Counter(q["question_type"] for q in data)))

    miss = []
    recs = []
    for qid, r in qc.items():
        q = byid[qid]
        store, nfacts = build_store(q, miss)
        keys = sorted(store)
        cur_full = render_view(store, "current")
        hist_full = render_view(store, "historical")
        # one line per chain invariant (needed to read "chains surviving" off line counts)
        assert len(cur_full.split("\n")) == len(keys) or not keys, (qid, len(keys))
        cur_cap, cur_binds = truncate(cur_full)
        hist_cap, hist_binds = truncate(hist_full)
        n_cur_lines = len(cur_cap.split("\n")) if cur_cap else 0
        n_hist_lines = len(hist_cap.split("\n")) if hist_cap else 0
        multi = {k for k in keys if len(_group_by_session(store[k])) > 1}
        kept_keys_cur = set(keys[:n_cur_lines])
        mode = r["view_mode"]
        full_v = nrm(cur_full if mode == "current" else hist_full)
        cap_v = nrm(cur_cap if mode == "current" else hist_cap)
        g = nrm(q["answer"])
        # where in the sorted key order does the gold-bearing line sit?
        gold_line_ranks = []
        if g:
            lines = (cur_full if mode == "current" else hist_full).split("\n")
            for i, ln in enumerate(lines):
                if g and g in nrm(ln):
                    gold_line_ranks.append(i)
        recs.append({
            "qid": qid, "type": r["type"], "mode": mode, "qlabel": r["qlabel"],
            "n_facts": nfacts, "n_chains": len(keys),
            "cur_full_chars": len(cur_full), "hist_full_chars": len(hist_full),
            "cur_cap_chars": len(cur_cap), "hist_cap_chars": len(hist_cap),
            "cur_binds": cur_binds, "hist_binds": hist_binds,
            "kept_cur": n_cur_lines, "kept_hist": n_hist_lines,
            "n_multi": len(multi), "multi_kept": len(multi & kept_keys_cur),
            "gold": g, "gold_len": len(g),
            "gold_in_full": bool(g) and g in full_v,
            "gold_in_cap": bool(g) and g in cap_v,
            "gold_ranks": gold_line_ranks,
            "judge_qc": r["judge"], "judge_nq": nq[qid]["judge"], "judge_vo": vo[qid]["judge"],
            "judge_nl": nl[qid]["judge"],
        })
    print("cache misses:", len(miss))
    with open("/root/agent_memory_iclr/audit/z_recs.json", "w") as f:
        json.dump(recs, f)
    print("saved", len(recs), "records")


if __name__ == "__main__":
    main()
