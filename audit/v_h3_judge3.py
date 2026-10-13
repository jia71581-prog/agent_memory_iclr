"""Last candidate definition for the judge sub-claim: coverage of DISTINCTIVE gold terms
(same df<=2% rule as the context analysis) inside the prediction, qc vs full, all 500."""
import json, re, math, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset

R = "/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{t}.jsonl"))}
arms = {t: load(t) for t in ["qc", "full"]}
data = load_dataset("s_cleaned")
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())

rowsD = {}
for i, q in enumerate(data):
    n = 0; df = {}
    for sess in q["haystack_sessions"]:
        for t in sess:
            if not t.get("content"): continue
            n += 1
            for w in set(tok(t["content"])): df[w] = df.get(w, 0) + 1
    g = {w for w in set(tok(q["answer"])) - set(tok(q["question"])) if w not in STOP}
    rowsD[q["question_id"]] = {w for w in g if df.get(w, 0) / max(1, n) <= 0.02}
    if (i + 1) % 100 == 0: print("  ...", i + 1, flush=True)

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

print("\n== coverage of DISTINCTIVE gold terms inside the prediction ==")
for t in arms:
    v = [len(rowsD[qid] & set(tok(r["pred"]))) / len(rowsD[qid])
         for qid, r in arms[t].items() if r["judge"] and rowsD[qid]]
    a = [len(rowsD[qid] & set(tok(r["pred"]))) / len(rowsD[qid])
         for qid, r in arms[t].items() if rowsD[qid]]
    print(f"  {t:5s} judge=CORRECT mean={np.mean(v):.4f} (n={len(v)})   all rows mean={np.mean(a):.4f} (n={len(a)})")

X, y, ind = [], [], []
for t in arms:
    for qid, r in arms[t].items():
        if not rowsD[qid]: continue
        c = len(rowsD[qid] & set(tok(r["pred"]))) / len(rowsD[qid])
        X.append([c, math.log(1 + len(tok(r["pred"])))]); y.append(1.0 if r["judge"] else 0.0)
        ind.append(1.0 if t == "qc" else 0.0)
X = np.array(X); y = np.array(y); ind = np.array(ind).reshape(-1, 1)
_, ll0, _ = logit_fit(X, y)
b1, ll1, se1 = logit_fit(np.hstack([X, ind]), y)
chi2 = 2 * (ll1 - ll0); p = 1 - stats.chi2.cdf(chi2, 1)
print(f"  LRT arm indicator: chi2={chi2:.3f} p={p:.4f} OR={math.exp(b1[-1]):.3f} "
      f"CI=[{math.exp(b1[-1]-1.96*se1[-1]):.3f},{math.exp(b1[-1]+1.96*se1[-1]):.3f}] n={len(y)}")
