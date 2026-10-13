"""Sweep plausible operationalizations to see which reproduces the stated numbers
view 0.278 / top20 0.320 / top30 0.386 / view+top20 0.485, loose 0.396 vs 0.611."""
import pickle, re, itertools
from statistics import mean
rows = pickle.load(open("/root/agent_memory_iclr/audit/v_h3_ssp.pkl", "rb"))
TOKS = {
    "alnum": re.compile(r"[a-z0-9]+"),
    "bm25": re.compile(r"[a-z0-9']+"),
    "alpha": re.compile(r"[a-z]+"),
}
STOP_SMALL = set("a an the of in on at to for with from by as is are was were be and or".split())
STOP_BIG = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())
STOPS = {"none": set(), "small": STOP_SMALL, "big": STOP_BIG}

target = {"view": 0.278, "raw20": 0.320, "raw30": 0.386, "view+raw20": 0.485}
best = []
for tname, TK in TOKS.items():
    tok = lambda s, TK=TK: TK.findall(str(s).lower())
    for sname, ST in STOPS.items():
        for thr in (0.02, 0.025, 0.01, 0.05):
            for minlen in (1, 3):
                for agg in ("macro", "micro"):
                    num = {k: 0 for k in target}; den = 0
                    per = {k: [] for k in target}
                    for r in rows:
                        turns = r["turns"]; n = len(turns)
                        df = {}
                        for t in turns:
                            for w in set(tok(t)): df[w] = df.get(w, 0) + 1
                        g = set(tok(r["answer"])) - set(tok(r["question"]))
                        g = {w for w in g if w not in ST and len(w) >= minlen}
                        D = {w for w in g if df.get(w, 0) / max(1, n) <= thr}
                        if not D: continue
                        den += len(D)
                        ctx = {"view": r["view"], "raw20": r["raw20"], "raw30": r["raw30"],
                               "view+raw20": r["view"] + "\n" + r["raw20"]}
                        for k, c in ctx.items():
                            h = len(D & set(tok(c)))
                            num[k] += h; per[k].append(h / len(D))
                    vals = {k: (num[k] / den if agg == "micro" else mean(per[k])) for k in target}
                    err = sum(abs(vals[k] - target[k]) for k in target)
                    best.append((err, tname, sname, thr, minlen, agg, vals))
best.sort(key=lambda x: x[0])
print("closest operationalizations to the stated coverage quadruple:")
for e, tn, sn, thr, ml, agg, v in best[:8]:
    print(f"  L1err={e:.3f} tok={tn:5s} stop={sn:5s} df<={thr} minlen={ml} agg={agg}  "
          f"view={v['view']:.3f} r20={v['raw20']:.3f} r30={v['raw30']:.3f} v+r20={v['view+raw20']:.3f}")

print("\nloose content-word recall variants (target view 0.396 vs raw30 0.611):")
for tname, TK in TOKS.items():
    tok = lambda s, TK=TK: TK.findall(str(s).lower())
    for sname, ST in STOPS.items():
        for rmq in (False, True):
            for agg in ("macro", "micro"):
                nv = nr = den = 0; pv, pr = [], []
                for r in rows:
                    g = {w for w in tok(r["answer"]) if w not in ST and len(w) > 1}
                    if rmq: g -= set(tok(r["question"]))
                    if not g: continue
                    den += len(g)
                    hv = len(g & set(tok(r["view"]))); hr = len(g & set(tok(r["raw30"])))
                    nv += hv; nr += hr; pv.append(hv / len(g)); pr.append(hr / len(g))
                V = nv / den if agg == "micro" else mean(pv)
                Rv = nr / den if agg == "micro" else mean(pr)
                if abs(V - 0.396) + abs(Rv - 0.611) < 0.06:
                    print(f"  tok={tname:5s} stop={sname:5s} rm_question={rmq} agg={agg}: view={V:.3f} raw30={Rv:.3f}")
