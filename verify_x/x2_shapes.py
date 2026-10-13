"""Independent recomputation #2: TEMP.anchor resolution hit, TEMP.offset bound,
RECALL.assistant context saving, evidence-present error count, ceiling arithmetic."""
import json, os, re, sys
from collections import defaultdict, Counter
from datetime import datetime
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape, resolve_relative_dates, parse_qdate

R = "/root/agent_memory_iclr/results"
OUT = "/root/agent_memory_iclr/verify_x"

judge = {}
for line in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(line)
    if "judge" in d:
        judge[d["qid"]] = d["judge"]

STOP = {"the", "and", "for", "was", "were", "you", "your", "with", "that", "this",
        "from", "have", "has", "had", "are", "not"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def contains(hay, needle):
    h, n = norm(hay), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    toks = [t for t in n.split() if len(t) > 2 and t not in STOP]
    if not toks:
        return False
    return all(t in h for t in toks)


DATE_IN_TEXT = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", re.I)


def work(q):
    qid = q["question_id"]
    shape, fired = classify_shape(q["question"])
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    ev_dates = [q["haystack_dates"][i] for i in sorted(ev)]
    gold = str(q["answer"])
    hay_all = " ".join(t.get("content", "") for s in q["haystack_sessions"] for t in s)
    ev_txt = " ".join(t.get("content", "") for i in sorted(ev) for t in q["haystack_sessions"][i])
    ev_ha_txt = " ".join(t.get("content", "") for i in sorted(ev) for t in q["haystack_sessions"][i]
                         if t.get("has_answer"))
    assist_chars = sum(len(t.get("content", "")) for s in q["haystack_sessions"] for t in s
                       if t.get("role") == "assistant")
    all_chars = sum(len(t.get("content", "")) for s in q["haystack_sessions"] for t in s)

    # ---- TEMP.anchor: resolve relative dates, does resolved date land on an evidence session date?
    res = resolve_relative_dates(q["question"], q["question_date"])
    anchor_hit = None
    anchor_hit_tol = None
    anchor_best = None
    if res:
        qd = parse_qdate(q["question_date"])
        hits, hits_tol, best = [], [], []
        for expr, dt, tol in res:
            ds = [abs((datetime.strptime(d.split(" (")[0], "%Y/%m/%d") - dt).days) for d in ev_dates]
            m = min(ds) if ds else 999
            best.append(m)
            hits.append(m <= tol)
            hits_tol.append(m <= 3)
        anchor_hit = any(hits)
        anchor_hit_tol = any(hits_tol)
        anchor_best = min(best) if best else None

    # ---- TEMP.offset: can gold number be derived by pure date arithmetic on evidence session dates?
    gold_nums = [int(x) for x in re.findall(r"\b\d+\b", gold)][:4]
    dts = [datetime.strptime(d.split(" (")[0], "%Y/%m/%d") for d in ev_dates]
    qd = parse_qdate(q["question_date"])
    cand = set()
    for a in dts + [qd]:
        for b in dts + [qd]:
            dd = abs((a - b).days)
            cand.update({dd, round(dd / 7), round(dd / 30), round(dd / 365),
                         dd // 7, dd // 30, dd // 365})
    offset_derivable = bool(gold_nums) and any(n in cand for n in gold_nums)
    # does the evidence text contain an explicit in-text date?
    ev_has_date = bool(DATE_IN_TEXT.search(ev_txt))

    return {
        "qid": qid, "shape": shape, "type": q["question_type"], "is_abs": qid.endswith("_abs"),
        "judge": judge.get(qid), "gold": gold, "question": q["question"],
        "qdate": q["question_date"], "ev_dates": ev_dates, "n_ev": len(ev),
        "n_res": len(res), "anchor_hit": anchor_hit, "anchor_hit_tol3": anchor_hit_tol,
        "anchor_best_days": anchor_best,
        "offset_derivable": offset_derivable, "ev_has_date": ev_has_date,
        "gold_in_hay": contains(hay_all, gold),
        "gold_in_ev": contains(ev_txt, gold),
        "gold_in_ev_ha": contains(ev_ha_txt, gold),
        "assist_chars": assist_chars, "all_chars": all_chars,
        "n_sess": len(q["haystack_sessions"]),
    }


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = p.map(work, data, chunksize=2)
    json.dump(rows, open(f"{OUT}/x2_shapes.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
