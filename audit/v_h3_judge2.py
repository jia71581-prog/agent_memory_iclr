"""Which definition of 'gold-term coverage of the answer' gives 0.245 (qc) / 0.242 (full),
and how does the LRT behave across those definitions and across scopes?"""
import json, re, math, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset, token_f1

R = "/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{t}.jsonl"))}
arms = {t: load(t) for t in ["qc", "full"]}
data = load_dataset("s_cleaned")
byid = {q["question_id"]: q for q in data}
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())

def specs(q, pred):
    g_all = set(tok(q["answer"])); p = set(tok(pred))
    g_cs = {w for w in g_all if w not in STOP}
    g_dq = g_cs - set(tok(q["question"]))
    out = {}
    out["recall_all"] = len(g_all & p) / max(1, len(g_all))
    out["recall_content"] = len(g_cs & p) / max(1, len(g_cs))
    out["recall_content_minusQ"] = len(g_dq & p) / max(1, len(g_dq)) if g_dq else np.nan
    out["precision_content"] = len(g_cs & p) / max(1, len({w for w in p if w not in STOP}))
    out["jaccard"] = len(g_cs & p) / max(1, len(g_cs | {w for w in p if w not in STOP}))
    out["token_f1"] = token_f1(pred, q["answer"])
    return out

names = ["recall_all", "recall_content", "recall_content_minusQ", "precision_content", "jaccard", "token_f1"]
print("== mean over judge=CORRECT rows (all 500 questions per arm) ==")
vals = {t: {k: [] for k in names} for t in arms}
allrows = {t: [] for t in arms}
for t in arms:
    for qid, r in arms[t].items():
        s = specs(byid[qid], r["pred"])
        allrows[t].append((qid, r, s))
        if r["judge"]:
            for k in names:
                if not (isinstance(s[k], float) and math.isnan(s[k])): vals[t][k].append(s[k])
for k in names:
    print(f"  {k:22s} qc={np.mean(vals['qc'][k]):.4f}  full={np.mean(vals['full'][k]):.4f}   "
          f"(target 0.245 / 0.242)")

print("\n== same, restricted to the SSP-30 ==")
for k in names:
    a = [s[k] for qid, r, s in allrows["qc"] if r["judge"] and byid[qid]["question_type"] == "single-session-preference"]
    b = [s[k] for qid, r, s in allrows["full"] if r["judge"] and byid[qid]["question_type"] == "single-session-preference"]
    a = [x for x in a if not math.isnan(x)]; b = [x for x in b if not math.isnan(x)]
    print(f"  {k:22s} qc={np.mean(a):.4f} (n={len(a)})  full={np.mean(b):.4f} (n={len(b)})")

def logit_fit(X, y, iters=300):
    X = np.hstack([np.ones((len(X), 1)), X]); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); Wd = p * (1 - p) + 1e-9
        g = X.T @ (y - p); H = -(X.T * Wd) @ X
        try: step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError: break
        b = b + step
        if np.max(np.abs(step)) < 1e-11: break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    se = np.sqrt(np.diag(np.linalg.inv((X.T * (p * (1 - p))) @ X)))
    return b, ll, se

print("\n== LRT for the arm indicator, every coverage definition, scope = all 500 x 2 arms ==")
for k in names:
    X, y, ind = [], [], []
    for t in arms:
        for qid, r, s in allrows[t]:
            v = s[k]
            if isinstance(v, float) and math.isnan(v): continue
            X.append([v, math.log(1 + len(tok(r["pred"])))]); y.append(1.0 if r["judge"] else 0.0)
            ind.append(1.0 if t == "qc" else 0.0)
    X = np.array(X); y = np.array(y); ind = np.array(ind).reshape(-1, 1)
    _, ll0, _ = logit_fit(X, y)
    b1, ll1, se1 = logit_fit(np.hstack([X, ind]), y)
    chi2 = 2 * (ll1 - ll0); p = 1 - stats.chi2.cdf(chi2, 1)
    print(f"  {k:22s} chi2={chi2:6.3f} p={p:.4f}  armOR={math.exp(b1[-1]):.3f} "
          f"CI=[{math.exp(b1[-1]-1.96*se1[-1]):.3f},{math.exp(b1[-1]+1.96*se1[-1]):.3f}] n={len(y)}")

print("\n== same LRT restricted to SSP-30 (n=60) ==")
for k in ["recall_content", "token_f1"]:
    X, y, ind = [], [], []
    for t in arms:
        for qid, r, s in allrows[t]:
            if byid[qid]["question_type"] != "single-session-preference": continue
            v = s[k]
            if isinstance(v, float) and math.isnan(v): continue
            X.append([v, math.log(1 + len(tok(r["pred"])))]); y.append(1.0 if r["judge"] else 0.0)
            ind.append(1.0 if t == "qc" else 0.0)
    X = np.array(X); y = np.array(y); ind = np.array(ind).reshape(-1, 1)
    _, ll0, _ = logit_fit(X, y)
    b1, ll1, se1 = logit_fit(np.hstack([X, ind]), y)
    chi2 = 2 * (ll1 - ll0); p = 1 - stats.chi2.cdf(chi2, 1)
    print(f"  {k:22s} chi2={chi2:6.3f} p={p:.4f}  armOR={math.exp(b1[-1]):.3f} "
          f"CI=[{math.exp(b1[-1]-1.96*se1[-1]):.3f},{math.exp(b1[-1]+1.96*se1[-1]):.3f}] n={len(y)}")

# how many CORRECT answers have low gold-term coverage (i.e. judge passing thin answers) per arm
print("\n== distribution check: judge=CORRECT rows with recall_content < 0.5 ==")
for t in arms:
    v = [s["recall_content"] for qid, r, s in allrows[t] if r["judge"]]
    print(f"  {t:5s} {sum(1 for x in v if x < 0.5)}/{len(v)} = {sum(1 for x in v if x < 0.5)/len(v):.3f}")
