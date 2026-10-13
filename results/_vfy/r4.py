import json,sys,re,itertools,math
sys.path.insert(0,'/root/agent_memory_iclr/code')
from v5_guards import is_decline, DECLINE_RE, nz
R='/root/agent_memory_iclr/results/'
ARMS=['qc','qc_no_qc','qc_no_lifecycle','full','decay','qc_view_only','fifo']
D={a:{r['qid']:r for r in map(json.loads,open(R+'full500_%s.jsonl'%a))} for a in ARMS}

SENT=re.compile(r'(?<=[.!?])\s+')
NARROW=re.compile(r"i don't know|i do not know|i don't have (any |enough )?(information|record|relevant|memory|data)"
                  r"|no record|not enough information|can(no|')t determine|unable to determine", re.I)
BROAD=re.compile(DECLINE_RE.pattern + r"|no mention|isn't specified|is not specified|doesn't specify|does not specify"
                 r"|not specified|unfortunately|i'm not able to|i am unable|no data|nothing in (the |my )?(memory|context|record)", re.I)
def lead(pred, rx=DECLINE_RE):
    t=nz(pred); first=SENT.split(t.strip())[0] if t.strip() else ""
    return bool(rx.search(first))

VARIANTS={
 'V1 theirs (anywhere)'   : lambda r: is_decline(r['pred']),
 'V2 leading sentence'    : lambda r: lead(r['pred']),
 'V3 narrow regex'        : lambda r: bool(NARROW.search(nz(r['pred']))),
 'V4 broad regex'         : lambda r: bool(BROAD.search(nz(r['pred']))),
 'V5 leading+narrow'      : lambda r: lead(r['pred'], NARROW),
}
for name,fn in VARIANTS.items():
    print("="*92); print(name)
    print(f"  {'arm':16s} {'TPabs':>6s} {'FPans':>8s} {'rate':>6s} {'prec':>6s} | harm-adj: {'TP*':>4s} {'FP*':>5s} {'prec*':>6s}")
    rk=[]
    for a in ARMS:
        rows=list(D[a].values())
        ab=[r for r in rows if r['qid'].endswith('_abs')]; an=[r for r in rows if not r['qid'].endswith('_abs')]
        tp=[r for r in ab if fn(r)]; fp=[r for r in an if fn(r)]
        tps=[r for r in tp if r['judge']]; fps=[r for r in fp if not r['judge']]
        p=len(tp)/max(1,len(tp)+len(fp)); ps=len(tps)/max(1,len(tps)+len(fps))
        rk.append((a,p,ps))
        print(f"  {a:16s} {len(tp):3d}/{len(ab):<2d} {len(fp):4d}/{len(an):<3d} {len(fp)/len(an):6.3f} {p:6.3f} | "
              f"{len(tps):4d} {len(fps):5d} {ps:6.3f}")
    print("   rank by prec :", " > ".join(a for a,_,_ in sorted(rk,key=lambda x:-x[1])))
    print("   rank by prec*:", " > ".join(a for a,_,_ in sorted(rk,key=lambda x:-x[2])))
