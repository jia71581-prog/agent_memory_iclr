"""Evidence-complete AGG.enum error audit: are the errors uniformly under-counts (a SUBSET)?"""
import json, re, statistics as st
from collections import Counter
R = "/root/agent_memory_iclr/results"
rows = json.load(open(f"{R}/_adv6_rows3.json"))
QC = {}
for l in open(f"{R}/full500_qc.jsonl"):
    d = json.loads(l)
    if "judge" in d:
        QC[d["qid"]] = d
for r in rows:
    r["judge"] = QC[r["qid"]]["judge"]
    r["pred"] = QC[r["qid"]].get("pred", "") or ""

WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def pred_numbers(p):
    s = p.lower().replace(",", "")
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}\b", s):
            nums.append(float(v))
    return nums


def first_pred_num(p):
    # bolded number first (the pipeline bolds the answer), else first number
    m = re.search(r"\*\*[^*]*?(-?\d+(?:\.\d+)?)", p.replace(",", ""))
    if m:
        return float(m.group(1))
    n = pred_numbers(p)
    return n[0] if n else None


AG = [r for r in rows if r["shape"] == "AGG.enum"]
print(f"AGG.enum shape questions: {len(AG)}; judged wrong: {sum(1 for r in AG if not r['judge'])}")
for lab, cond in [("gold_sess_hit==n_ev (cluster covers every gold session)",
                   lambda r: r["gold_sess_hit"] == r["n_ev"]),
                  ("union items from gold sessions >= N", lambda r: r["u_gold_sess_distinct"] >= (r["gold_num"] or 1e9)),
                  ("union grounded >= N", lambda r: r["u_grounded_evturn"] >= (r["gold_num"] or 1e9))]:
    err = [r for r in AG if not r["judge"] and cond(r)]
    print(f"\n--- evidence-complete def: {lab}: {len(err)} errors")
    under = over = neither = nonum = 0
    for r in sorted(err, key=lambda r: -(r["gold_num"] or 0)):
        pn = first_pred_num(r["pred"])
        g = r["gold_num"]
        if g is None or pn is None:
            tag = "NO-NUM"; nonum += 1
        elif pn < g:
            tag = "UNDER"; under += 1
        elif pn > g:
            tag = "OVER"; over += 1
        else:
            tag = "EQUAL"; neither += 1
        allp = pred_numbers(r["pred"])
        anyeq = (g is not None and g in allp)
        print(f"   {r['qid']:16s} {r['type'][:12]:12s} gold={str(r['gold'])[:26]:26s} predfirst={pn} {tag:6s} "
              f"goldInPredNums={anyeq} N/n_ev={g}/{r['n_ev']}  {r['question'][:60]}")
    print(f"   -> UNDER {under}  OVER {over}  EQUAL {neither}  NO-NUM {nonum}")
