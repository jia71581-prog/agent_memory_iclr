"""Independent recomputation of the H3 coverage numbers on the SSP-30 subset."""
import pickle, re, json, math, sys
from statistics import mean

rows = pickle.load(open("/root/agent_memory_iclr/audit/v_h3_ssp.pkl", "rb"))
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())

STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just don should now about into over after before again further once here there also would could may might must shall s t d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn""".split())

def cov(terms, ctx_tokens):
    if not terms: return None
    return len(terms & ctx_tokens) / len(terms)

def build(df_thresh=0.02, use_stop=True, min_len=1):
    out = []
    for r in rows:
        turns = r["turns"]; n = len(turns)
        turn_tok = [set(tok(t)) for t in turns]
        df = {}
        for st in turn_tok:
            for w in st:
                df[w] = df.get(w, 0) + 1
        g = set(tok(r["answer"])); qs = set(tok(r["question"]))
        cand = g - qs
        if use_stop: cand = {w for w in cand if w not in STOP}
        if min_len > 1: cand = {w for w in cand if len(w) >= min_len}
        distinct = {w for w in cand if df.get(w, 0) / max(1, n) <= df_thresh}
        out.append((r, distinct, cand, df, n))
    return out

def report(df_thresh=0.02, use_stop=True, min_len=1, label=""):
    dat = build(df_thresh, use_stop, min_len)
    cols = {"view": lambda r: r["view"], "raw20": lambda r: r["raw20"], "raw30": lambda r: r["raw30"],
            "view+raw20": lambda r: r["view"] + "\n" + r["raw20"],
            "view_hist": lambda r: r["view_hist"]}
    per = {k: [] for k in cols}
    nemp = 0
    sizes = []
    for r, distinct, cand, df, n in dat:
        if not distinct:
            nemp += 1
            continue
        sizes.append(len(distinct))
        for k, f in cols.items():
            per[k].append(cov(distinct, set(tok(f(r)))))
    print(f"--- {label} df<={df_thresh} stop={use_stop} minlen={min_len} | questions with >=1 distinctive term: {len(sizes)}/30 (empty:{nemp}), mean #terms={mean(sizes):.2f}")
    for k in ["view", "raw20", "raw30", "view+raw20", "view_hist"]:
        v = per[k]
        print(f"    {k:11s} mean={mean(v):.4f}  (n={len(v)})")
    return per, dat

print("=== main spec (loose stopword list, df<=2% of turns, exclude question words) ===")
per, dat = report(0.02, True, 1, "main")
print()
print("=== variants ===")
for dt in (0.02, 0.05, 0.01):
    for us in (True, False):
        report(dt, us, 1, "v")
print()
# length / density
print("=== context sizes (chars / tokens) ===")
for k, f in [("view", lambda r: r["view"]), ("raw20", lambda r: r["raw20"]), ("raw30", lambda r: r["raw30"])]:
    ch = [len(f(r)) for r in rows]; tk = [len(tok(f(r))) for r in rows]
    print(f"  {k:6s} chars mean={mean(ch):8.0f} median={sorted(ch)[len(ch)//2]:8.0f}   tokens mean={mean(tk):7.0f}")
print()
# verbatim gold containment
def nrm(s): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()
print("=== gold answer string containment (normalized substring) ===")
for k, f in [("view", lambda r: r["view"]), ("raw20", lambda r: r["raw20"]), ("raw30", lambda r: r["raw30"])]:
    c = sum(1 for r in rows if nrm(r["answer"]) and nrm(r["answer"]) in nrm(f(r)))
    print(f"  {k:6s} {c}/30 = {c/30:.3f}")
json.dump({"per": {k: v for k, v in per.items()}}, open("/root/agent_memory_iclr/audit/v_h3_cov.json", "w"))
