"""ADVERSARIAL VERIFIER #2 step 2: does the merge delete information, and what does a
MATCHED-budget integration cost? All offline."""
import json
import pickle
import re
import statistics as st
import sys

sys.path.insert(0, "/root/agent_memory_iclr/code")
from lifecycle import assemble_chains, render_view, _norm
from query_classifier import classify_query
from v5_chains import merge_chains, render_aggregation_view, aggregation_kind

CAP = 20000
D = pickle.load(open("/root/agent_memory_iclr/results/_adv2_facts.pkl", "rb"))
JQC = {json.loads(l)["qid"]: json.loads(l) for l in
       open("/root/agent_memory_iclr/results/full500_qc.jsonl")}


def capped(txt, cap=CAP):
    if len(txt) <= cap:
        return txt
    kept, total = [], 0
    for line in txt.splitlines():
        if total + len(line) + 1 > cap:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept)


def nrm(s):
    return re.sub(r"\s+", " ", _norm(str(s))).strip()


rows = []
for x in D:
    label, _ = classify_query(x["question"])
    mode = "historical" if label == "historical" else "current"
    s4 = assemble_chains(x["facts"])
    s5 = merge_chains(s4)
    v4f, v5f = render_view(s4, mode), render_view(s5, mode)
    blk = render_aggregation_view(s5, x["question"])
    v4c, v5c = capped(v4f), capped(v5f)
    # MATCHED budget: block占用同一 20k 预算, block 放最前(设计者建议的接线)
    matched = capped((blk + "\n\n" + v5f) if blk else v5f)

    gs = set(x["gold_sidx"])
    gvals = sorted({f["value"] for f in x["facts"] if f["sidx"] in gs})
    gn = [nrm(v) for v in gvals if nrm(v)]

    def hit(txt):
        t = nrm(txt)
        return sum(1 for v in gn if v in t)

    ans = nrm(x["answer"])
    rows.append(dict(
        qid=x["qid"], type=x["type"], mode=mode, fired=bool(blk),
        kind=aggregation_kind(x["question"]) or "",
        judge=str(JQC[x["qid"]].get("judge")) == "True",
        g=len(gn),
        # uncapped: isolates merge-caused DELETION from cap effects
        h4f=hit(v4f), h5f=hit(v5f),
        # capped
        h4c=hit(v4c), h5c=hit(v5c), hm=hit(matched), hblk=hit(blk),
        a4f=bool(ans) and ans in nrm(v4f), a5f=bool(ans) and ans in nrm(v5f),
        a4c=bool(ans) and ans in nrm(v4c), a5c=bool(ans) and ans in nrm(v5c),
        am=bool(ans) and ans in nrm(matched),
        ablk=bool(ans) and ans in nrm(blk),
        nf4=sum(len(c) for c in s4.values()), nf5=sum(len(c) for c in s5.values()),
        lines4c=v4c.count("\n") + 1, lines5c=v5c.count("\n") + 1,
        linesm=matched.count("\n") + 1, lines5f=v5f.count("\n") + 1,
        len4c=len(v4c), len5c=len(v5c), lenm=len(matched), lenblk=len(blk),
    ))
json.dump(rows, open("/root/agent_memory_iclr/results/_adv2_loss.json", "w"))

n = len(rows)
G = sum(r["g"] for r in rows)
print(f"n={n}  total gold-evidence values {G}")

print("\n=== (A) merge-induced deletion, UNCAPPED view (cap effects excluded) ===")
print(f"facts retained after merge: {sum(r['nf5'] for r in rows)} / {sum(r['nf4'] for r in rows)} "
      f"({sum(r['nf5'] for r in rows)/sum(r['nf4'] for r in rows):.4%})")
print(f"gold-value hits: v4 {sum(r['h4f'] for r in rows)} -> v5 {sum(r['h5f'] for r in rows)} "
      f"({(sum(r['h5f'] for r in rows)-sum(r['h4f'] for r in rows))/sum(r['h4f'] for r in rows):+.3%})")
lose = [r for r in rows if r["h5f"] < r["h4f"]]
gain = [r for r in rows if r["h5f"] > r["h4f"]]
print(f"  questions losing gold values: {len(lose)}  gaining: {len(gain)}  unchanged: {n-len(lose)-len(gain)}")
print(f"gold ANSWER string present: v4 {sum(r['a4f'] for r in rows)} -> v5 {sum(r['a5f'] for r in rows)}")
al = [r for r in rows if r["a4f"] and not r["a5f"]]
ag = [r for r in rows if r["a5f"] and not r["a4f"]]
print(f"  answer lost by merge: {len(al)}  gained: {len(ag)}")
print(f"  lost qids (first 15): {[r['qid'] for r in al][:15]}")
print(f"  of those lost, currently judged CORRECT by the v4 pipeline: {sum(r['judge'] for r in al)}")

print("\n=== (B) capped view, unmatched budget (as reported) ===")
print(f"gold-value hits: v4 {sum(r['h4c'] for r in rows)} -> v5 {sum(r['h5c'] for r in rows)} "
      f"({(sum(r['h5c'] for r in rows)-sum(r['h4c'] for r in rows))/sum(r['h4c'] for r in rows):+.3%})")
print(f"gold ANSWER string in capped view: v4 {sum(r['a4c'] for r in rows)} -> v5 {sum(r['a5c'] for r in rows)}")
print(f"mean per-q gold recall: {st.mean([r['h4c']/r['g'] for r in rows if r['g']]):.4f} -> "
      f"{st.mean([r['h5c']/r['g'] for r in rows if r['g']]):.4f}")

fired = [r for r in rows if r["fired"]]
print(f"\n=== (C) MATCHED budget (block+view share the 20k cap), {len(fired)} firing questions ===")
print(f"chars: v4 {st.median([r['len4c'] for r in fired]):.0f} -> matched {st.median([r['lenm'] for r in fired]):.0f}")
print(f"view lines kept: v4 {st.median([r['lines4c'] for r in fired]):.0f} -> "
      f"v5 uncapped-budget {st.median([r['lines5c'] for r in fired]):.0f} -> "
      f"matched {st.median([r['linesm'] for r in fired]):.0f}")
print(f"gold-value hits on firing qs: v4 {sum(r['h4c'] for r in fired)} -> "
      f"v5 unmatched {sum(r['h5c'] for r in fired)} -> matched {sum(r['hm'] for r in fired)}")
print(f"gold ANSWER string on firing qs: v4 {sum(r['a4c'] for r in fired)} -> "
      f"v5 unmatched {sum(r['a5c'] for r in fired)} -> matched {sum(r['am'] for r in fired)}")
print(f"  matched-budget losers (had answer in v4 capped, lost it): "
      f"{[r['qid'] for r in fired if r['a4c'] and not r['am']]}")
print(f"gold answer inside the BLOCK alone: {sum(r['ablk'] for r in fired)}/{len(fired)}")
print(f"gold-evidence values inside the BLOCK alone: {sum(r['hblk'] for r in fired)}"
      f" of {sum(r['g'] for r in fired)}")
