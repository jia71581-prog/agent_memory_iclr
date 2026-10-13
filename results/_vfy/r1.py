import json,glob,os,re,sys
sys.path.insert(0,'/root/agent_memory_iclr/code')
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}
from v5_guards import is_decline, DECLINE_RE

print("=== reproduce claimed table (their is_decline) ===")
print(f"{'arm':16s} {'TPabs':>6s} {'FPans':>7s} {'rate':>7s} {'prec':>6s} {'absAcc':>7s} {'ansAcc':>7s} {'FPcorr':>8s}")
for a in ARMS:
    rows=list(D[a].values())
    ab=[r for r in rows if r['qid'].endswith('_abs')]
    an=[r for r in rows if not r['qid'].endswith('_abs')]
    tp=[r for r in ab if is_decline(r['pred'])]
    fp=[r for r in an if is_decline(r['pred'])]
    print(f"{a:16s} {len(tp):3d}/{len(ab):<2d} {len(fp):4d}/{len(an):<3d} {len(fp)/len(an):7.3f} "
          f"{len(tp)/max(1,len(tp)+len(fp)):6.3f} {sum(r['judge'] for r in ab)/len(ab):7.3f} "
          f"{sum(r['judge'] for r in an)/len(an):7.3f} {sum(r['judge'] for r in fp):3d}/{len(fp):<4d}")

print("\n=== detector-vs-judge agreement on abs subset (per arm) ===")
for a in ARMS:
    ab=[r for r in D[a].values() if r['qid'].endswith('_abs')]
    agree=sum(1 for r in ab if is_decline(r['pred'])==bool(r['judge']))
    print(f"  {a:16s} agree {agree}/30   disagreements: "
          + ", ".join(r['qid']+("(dec,judgeF)" if is_decline(r['pred']) else "(nodec,judgeT)")
                      for r in ab if is_decline(r['pred'])!=bool(r['judge'])))
