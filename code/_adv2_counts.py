"""ADVERSARIAL VERIFIER #2 step 3: (i) is the store-fit metric predictive of correctness?
(ii) do the block's printed counts agree with the gold number? (iii) is the 'gold answer in
block' proxy contaminated by derived numbers?"""
import json
import pickle
import re
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _norm
from query_classifier import classify_query
from v5_chains import (merge_chains, render_aggregation_view, aggregation_kind,
                       aggregation_printed_groups)

CAP = 20000
D = pickle.load(open("/root/agent_memory_iclr/results/_adv2_facts.pkl", "rb"))
JQC = {json.loads(l)["qid"]: json.loads(l) for l in
       open("/root/agent_memory_iclr/results/full500_qc.jsonl")}
SIZE = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/_adv2_size.json"))}
LOSS = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/_adv2_loss.json"))}
COV = {r["qid"]: r for r in json.load(open("/root/agent_memory_iclr/results/coverage_diag.json"))}

WORDNUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def gold_num(a):
    s = str(a).strip().lower()
    m = re.fullmatch(r"(\d+)", s)
    if m:
        return int(m.group(1))
    if s in WORDNUM:
        return WORDNUM[s]
    m = re.match(r"^(\d+)\b", s)          # '4 items', '3 times'
    if m:
        return int(m.group(1))
    for w, v in WORDNUM.items():
        if re.match(rf"^{w}\b", s):
            return v
    return None


out = []
for x in D:
    kind = aggregation_kind(x["question"])
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"
    r = dict(qid=x["qid"], type=x["type"], kind=kind or "", question=x["question"],
             answer=str(x["answer"]), judge=str(JQC[x["qid"]].get("judge")) == "True",
             gnum=gold_num(x["answer"]), fired=SIZE[x["qid"]]["fired"])
    if r["fired"]:
        s5 = merge_chains(assemble_chains(x["facts"]))
        kept, meta = aggregation_printed_groups(s5, x["question"])
        r["groups"] = [(g["term"], len(g["items"]), g["n_strong"]) for g in kept]
        r["union"] = sum(len(g["items"]) for g in kept)
        r["strong_union"] = sum(g["n_strong"] for g in kept)
        r["top"] = len(kept[0]["items"]) if kept else 0
        r["top_strong"] = kept[0]["n_strong"] if kept else 0
    out.append(r)
json.dump(out, open("/root/agent_memory_iclr/results/_adv2_counts.json", "w"))

n = len(out)
fired = [r for r in out if r["fired"]]
print(f"n={n}; block fires {len(fired)}")
print(f"v4 pipeline judge acc: all {st.mean([r['judge'] for r in out]):.3f}; "
      f"firing {st.mean([r['judge'] for r in fired]):.3f}; "
      f"non-firing {st.mean([r['judge'] for r in out if not r['fired']]):.3f}")
byt = {}
for r in fired:
    s = byt.setdefault(r["type"], [0, 0])
    s[0] += 1
    s[1] += r["judge"]
print("firing questions by type: " + "; ".join(f"{t}={v[0]}(acc {v[1]/v[0]:.2f})" for t, v in sorted(byt.items())))

print("\n=== (i) is 'fraction of store inside the cap' predictive of correctness? ===")
fit = {q: min(1.0, SIZE[q]["lines4"] / SIZE[q]["lines4f"]) for q in SIZE}
jd = {r["qid"]: r["judge"] for r in out}
lo = [q for q in fit if fit[q] < st.median(fit.values())]
hi = [q for q in fit if fit[q] >= st.median(fit.values())]
print(f"judge acc, store-fit BELOW median ({len(lo)}): {st.mean([jd[q] for q in lo]):.3f}")
print(f"judge acc, store-fit ABOVE median ({len(hi)}): {st.mean([jd[q] for q in hi]):.3f}")
# point-biserial
xs = [fit[q] for q in fit]
ys = [1.0 * jd[q] for q in fit]
mx, my = st.mean(xs), st.mean(ys)
num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
print(f"corr(store-fit, judge) = {num/den:+.4f}  over 500 questions")
# and the thing store-fit is supposed to buy:
print(f"gold ANSWER string in capped view: v4 {sum(LOSS[q]['a4c'] for q in LOSS)} -> "
      f"v5 {sum(LOSS[q]['a5c'] for q in LOSS)}  (the +1.6pp fit does not move this)")
iv = [q for q in COV if COV[q]["in_view"]]
niv = [q for q in COV if not COV[q]["in_view"]]
print(f"coverage_diag in_view: {len(iv)}/500, judge acc {st.mean([jd[q] for q in iv]):.3f} "
      f"vs not-in-view {len(niv)} acc {st.mean([jd[q] for q in niv]):.3f}")

print("\n=== (ii) do the block's printed counts agree with the gold number? ===")
cnt = [r for r in fired if r["kind"] == "count" and r["gnum"] is not None]
print(f"firing 'count' questions with a numeric gold answer: {len(cnt)}")
eq_union = sum(1 for r in cnt if r["union"] == r["gnum"])
eq_top = sum(1 for r in cnt if r["top"] == r["gnum"])
eq_any = sum(1 for r in cnt if r["gnum"] in [g[1] for g in r["groups"]] + [r["union"], r["strong_union"], r["top_strong"]])
print(f"  union == gold: {eq_union}/{len(cnt)} ({eq_union/len(cnt):.1%})")
print(f"  top-group size == gold: {eq_top}/{len(cnt)} ({eq_top/len(cnt):.1%})")
print(f"  ANY printed count (per-group / union / starred) == gold: {eq_any}/{len(cnt)} ({eq_any/len(cnt):.1%})")
print(f"  median gold {st.median([r['gnum'] for r in cnt]):.1f} vs median union "
      f"{st.median([r['union'] for r in cnt]):.1f} (median top group {st.median([r['top'] for r in cnt]):.1f})")
over = sum(1 for r in cnt if r["union"] > r["gnum"])
print(f"  union OVER-counts gold on {over}/{len(cnt)} ({over/len(cnt):.1%}); "
      f"median overshoot {st.median([r['union']-r['gnum'] for r in cnt]):+.1f}, "
      f"median |top - gold| {st.median([abs(r['top']-r['gnum']) for r in cnt]):.1f}")
print(f"  currently CORRECT among these {len(cnt)}: {sum(r['judge'] for r in cnt)} "
      f"({st.mean([r['judge'] for r in cnt]):.3f}) -> questions where a wrong printed count could regress")
for r in sorted(cnt, key=lambda r: -abs(r["union"] - r["gnum"]))[:12]:
    print(f"    gold={r['gnum']:>3} union={r['union']:>3} top={r['top']:>3} judge={'Y' if r['judge'] else 'N'} "
          f"groups={r['groups'][:3]} | {r['question'][:78]}")

print("\n=== (iii) is the 'gold answer inside the block' proxy contaminated? ===")
short = [r for r in fired if len(re.sub(r'[^a-z0-9]', '', str(r['answer']).lower())) <= 2]
print(f"firing questions whose gold answer normalizes to <=2 chars (e.g. a bare count): {len(short)}/{len(fired)}")
print(f"firing questions with a numeric gold answer at all: {sum(1 for r in fired if r['gnum'] is not None)}/{len(fired)}")
hb = [q for q in LOSS if LOSS[q]["fired"] and LOSS[q]["ablk"]]
hb_num = [q for q in hb if next(r for r in out if r['qid'] == q)["gnum"] is not None]
print(f"of the {len(hb)} firing questions where the gold answer 'appears' in the block, "
      f"{len(hb_num)} have a numeric gold answer (substring of any date/number in the block)")
