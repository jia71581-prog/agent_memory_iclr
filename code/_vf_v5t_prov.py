"""ADVERSARIAL: does the RESIDUAL numeric-date false-positive rate leak into the rendered
scaffold that would go into the prompt? Render all 500 timelines (store=None, zero API) and
count lines whose event date was inferred from a numeric-date match, then classify each
provenance token as plausibly-a-date or not.
"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_temporal as V

D = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
PROV = re.compile(r'date inferred from "([^"]+)"')
NUMTOK = re.compile(r"^\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?$")

rows = []
tot_lines = 0
numprov = []
for i, q in enumerate(D):
    try:
        t = V.render_timeline(q, None, q["question"])
    except Exception as e:
        rows.append({"qid": q["question_id"], "err": repr(e)[:120]})
        continue
    for ln in t.splitlines():
        if ln.startswith("["):
            tot_lines += 1
        m = PROV.search(ln)
        if m and NUMTOK.match(m.group(1).strip()):
            numprov.append({"qid": q["question_id"], "tok": m.group(1),
                            "line": re.sub(r"\s+", " ", ln)[:230]})
    if i % 100 == 0:
        print("…", i, file=sys.stderr, flush=True)

summ = {"questions": len(D), "timeline_lines_total": tot_lines,
        "lines_dated_by_numeric_rule": len(numprov),
        "questions_affected": len({x["qid"] for x in numprov}),
        "token_hist": Counter(x["tok"] for x in numprov).most_common(30),
        "errors": [r for r in rows if "err" in r][:10]}
json.dump({"summary": summ, "lines": numprov},
          open("/root/agent_memory_iclr/results/_vf_v5t_prov.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "token_hist"}, indent=1))
print(summ["token_hist"])
for x in numprov[:40]:
    print("-", x["qid"], f'[{x["tok"]}]', x["line"][:170])
