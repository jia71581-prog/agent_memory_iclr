import json
from collections import Counter
recs=json.load(open("/root/agent_memory_iclr/audit/gold_in_view.json"))
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc=load("qc")
c=Counter()
for r in recs:
    c[(r['gold_in_cur'],r['gold_in_hist'])]+=1
print("gold-in-(current, historical) capped views over 500:")
for k,v in sorted(c.items()): print("  cur=%s hist=%s : %d"%(k[0],k[1],v))
only_cur=[r for r in recs if r['gold_in_cur'] and not r['gold_in_hist']]
only_hist=[r for r in recs if r['gold_in_hist'] and not r['gold_in_cur']]
print(f"\nquestions where ONLY current view carries gold   : {len(only_cur)} ({len(only_cur)/500:.3f})")
print(f"questions where ONLY historical view carries gold: {len(only_hist)} ({len(only_hist)/500:.3f})")
print(f"=> upper bound on questions where mode choice can matter at all: {len(only_cur)+len(only_hist)} = {(len(only_cur)+len(only_hist))/500:.3f}")
print("\nknowledge-update only (the designed win case):")
ku=[r for r in recs if r['type']=='knowledge-update']
print(f"  n={len(ku)}  gold in cur={sum(r['gold_in_cur'] for r in ku)} gold in hist={sum(r['gold_in_hist'] for r in ku)} "
      f"only-cur={sum(1 for r in ku if r['gold_in_cur'] and not r['gold_in_hist'])} only-hist={sum(1 for r in ku if r['gold_in_hist'] and not r['gold_in_cur'])}")
print("\nrouting of the 'only-current-carries-gold' questions (did the classifier route them to current?):")
print("  ",Counter((r['qlabel'],r['mode_qc']) for r in only_cur))
print("routing of the 'only-historical-carries-gold' questions (should be historical):")
print("  ",Counter((r['qlabel'],r['mode_qc']) for r in only_hist))
mis=[r for r in only_hist if r['mode_qc']=='current']
print(f"  MISROUTED (needed historical, got current): {len(mis)}; qc judged correct on them: {sum(r['judge_qc'] for r in mis)}/{len(mis)} (raw turns rescue)")
