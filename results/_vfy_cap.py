"""Adversarial re-verification of the VIEW_CHAR_CAP truncation claim. Offline only."""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from lifecycle import assemble_chains, render_view, _session_text, SESSION_CACHE_DIR
from query_classifier import classify_query
from run_baseline import bm25_rank

R = "/root/agent_memory_iclr/results"
CAP = 20000


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in ("the", "and", "for", "was", "were", "you", "your")]
    if not toks:
        return False
    return all(t in h for t in toks)


def cached_facts(sess):
    text = _session_text(sess)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def truncate_like_run_qc(view_txt):
    """Byte-identical copy of run_qc.py's truncation block."""
    if len(view_txt) <= CAP:
        return view_txt
    kept, total = [], 0
    for line in view_txt.splitlines():
        if total + len(line) + 1 > CAP:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def main():
    data = {q["question_id"]: q for q in load_dataset("s_cleaned")}
    qc = {}
    for line in open(f"{R}/full500_qc.jsonl"):
        r = json.loads(line)
        if "judge" in r:
            qc[r["qid"]] = r

    rows = []
    for i, (qid, res) in enumerate(qc.items()):
        q = data[qid]
        gold = q["answer"]
        label, _ = classify_query(q["question"])
        mode = "historical" if label == "historical" else "current"

        facts, missing = [], False
        for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
            tri = cached_facts(sess)
            if tri is None:
                missing = True
                continue
            for e, a, v in tri:
                facts.append({"entity": e, "attribute": a, "value": v, "date": date, "sidx": si})
        store = assemble_chains(facts)
        view_full = render_view(store, mode)
        view_cap = truncate_like_run_qc(view_full)

        units = []
        for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
            for t in sess:
                if t.get("content"):
                    units.append((date, t["role"], t["content"]))
        order = bm25_rank(q["question"], [u[2] for u in units])
        # exactly as run_qc assembles it
        raw_sent = "\n".join(f"[{units[j][0]}] [{units[j][1]}] {units[j][2]}" for j in sorted(order[:20]))
        raw20_diag = " ".join(units[j][2] for j in order[:20])
        raw30_diag = " ".join(units[j][2] for j in order[:30])
        hay = " ".join(u[2] for u in units)

        nl_full = len(view_full.splitlines())
        nl_cap = len(view_cap.splitlines())
        rows.append({
            "qid": qid, "type": q["question_type"], "judge": res["judge"], "f1": res.get("f1"),
            "qlabel": label, "mode": mode, "is_abs": qid.endswith("_abs"),
            "cache_missing": missing, "n_chains": len(store),
            "view_full_chars": len(view_full), "view_cap_chars": len(view_cap),
            "view_full_lines": nl_full, "view_cap_lines": nl_cap,
            "raw_sent_chars": len(raw_sent),
            "in_view_full": contains(view_full, gold),
            "in_view_cap": contains(view_cap, gold),
            "in_raw20_sent": contains(raw_sent, gold),
            "in_raw20_diag": contains(raw20_diag, gold),
            "in_raw30_diag": contains(raw30_diag, gold),
            "in_hay": contains(hay, gold),
        })
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(qc)}", flush=True)

    with open(f"{R}/_vfy_cap.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("saved", len(rows))


if __name__ == "__main__":
    main()
