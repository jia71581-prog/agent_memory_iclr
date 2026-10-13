"""Independent recomputation #4: does the AGG.enum / AGG.latest_count split map onto
sum vs never-sum ground truth?  And what direction are the 26 AGG errors?"""
import json, os, re, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset
from run_baseline import bm25_rank
from v5_shape import classify_shape

R = "/root/agent_memory_iclr/results"
OUT = "/root/agent_memory_iclr/verify_x"
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20}


def gold_int(g):
    g = str(g).lower().replace(",", "").replace("$", "")
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\b", g)
    if m:
        return float(m.group(1))
    for w, v in NUMW.items():
        if re.match(rf"^\s*{w}\b", g):
            return float(v)
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", g)
    return float(m.group(1)) if m else None


def work(q):
    shape, _ = classify_shape(q["question"])
    if not shape.startswith("AGG."):
        return None
    sid2idx = {s: i for i, s in enumerate(q["haystack_session_ids"])}
    ev = {sid2idx[s] for s in q["answer_session_ids"] if s in sid2idx}
    gn = gold_int(q["answer"])
    # numbers literally present in evidence turns (has_answer preferred, else whole gold session)
    ha_txt, ev_txt = [], []
    for i in sorted(ev):
        for t in q["haystack_sessions"][i]:
            c = t.get("content", "")
            ev_txt.append(c)
            if t.get("has_answer"):
                ha_txt.append(c)
    def nums(txts):
        out = set()
        for c in txts:
            for m in re.findall(r"\b\d+(?:\.\d+)?\b", c.replace(",", "")):
                out.add(float(m))
        return out
    ha_nums, ev_nums = nums(ha_txt), nums(ev_txt)
    return {
        "qid": q["question_id"], "shape": shape, "type": q["question_type"],
        "is_abs": q["question_id"].endswith("_abs"),
        "gold": str(q["answer"]), "gold_num": gn, "n_ev": len(ev),
        "gold_in_ev_nums": (gn in ev_nums) if gn is not None else None,
        "gold_in_ha_nums": (gn in ha_nums) if gn is not None else None,
        "question": q["question"],
    }


if __name__ == "__main__":
    data = load_dataset("s_cleaned")
    with Pool(16) as p:
        rows = [r for r in p.map(work, data, chunksize=2) if r]
    QC = {}
    for l in open(f"{R}/full500_qc.jsonl"):
        d = json.loads(l)
        QC[d["qid"]] = d
    for r in rows:
        r["judge"] = QC[r["qid"]].get("judge")
        r["pred"] = QC[r["qid"]].get("pred", "")
    json.dump(rows, open(f"{OUT}/x4_agg.json", "w"), ensure_ascii=False)
    print("saved", len(rows))
