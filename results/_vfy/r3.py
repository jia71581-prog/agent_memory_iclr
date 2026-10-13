import json,sys,re
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline, DECLINE_RE, nz
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}
# which sub-pattern fires, on qc's 19 answerable "false declines"
alts=[p for p in DECLINE_RE.pattern.split('|')]
print("=== qc: all 19 answerable declines, matched span + judge ===")
for r in D['qc'].values():
    if r['qid'].endswith('_abs') or not is_decline(r['pred']): continue
    m=DECLINE_RE.search(nz(r['pred']))
    t=nz(r['pred']); i=m.start()
    ctx=t[max(0,i-70):i+60].replace('\n',' ')
    print(f"  judge={int(r['judge'])} {r['qid']:18s} match={m.group(0)!r:28s} pos={i:4d}/{len(t):4d}  ...{ctx}...")
