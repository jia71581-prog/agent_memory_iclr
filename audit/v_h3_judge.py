"""H3 part 4: the judge 'equal content bar' claim.
mean gold-term coverage of judge=CORRECT answers, qc vs full, and the LRT on an arm indicator."""
import json, re, math
import numpy as np
from scipy import stats
import sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from common import load_dataset

R = "/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R + f"full500_{t}.jsonl"))}
arms = {t: load(t) for t in ["qc", "full", "qc_no_qc", "qc_view_only", "qc_no_lifecycle", "decay", "fifo"]}
byid = {q["question_id"]: q for q in load_dataset("s_cleaned")}
W = re.compile(r"[a-z0-9]+")
def tok(s): return W.findall(str(s).lower())
STOP = set("""a an the and or but if then than that this these those of in on at to for with from by as is are was were be been being am do does did done have has had having i me my mine you your yours he him his she her hers it its we us our ours they them their theirs what which who whom when where why how all any both each few more most other some such no nor not only own same so too very can will just should now about into over after before again further once here there also would could may might must shall""".split())

def gold_terms(q, drop_q=True, drop_stop=True):
    g = set(tok(q["answer"]))
    if drop_q: g -= set(tok(q["question"]))
    if drop_stop: g = {w for w in g if w not in STOP}
    return g

def cov_pred(q, pred, **kw):
    g = gold_terms(q, **kw)
    if not g: return None
    return len(g & set(tok(pred))) / len(g)

print("== mean gold-term coverage of the PREDICTION, among judge=CORRECT rows ==")
for spec in [dict(drop_q=True, drop_stop=True), dict(drop_q=False, drop_stop=True),
             dict(drop_q=False, drop_stop=False), dict(drop_q=True, drop_stop=False)]:
    line = f"  spec drop_q={spec['drop_q']} drop_stop={spec['drop_stop']}: "
    for t in ["qc", "full"]:
        v = [cov_pred(byid[qid], r["pred"], **spec) for qid, r in arms[t].items() if r["judge"]]
        v = [x for x in v if x is not None]
        line += f"{t}={np.mean(v):.4f} (n={len(v)})  "
    print(line)

# main spec for the model
SPEC = dict(drop_q=False, drop_stop=True)
def build(armlist):
    X, y, arm = [], [], []
    for t in armlist:
        for qid, r in arms[t].items():
            c = cov_pred(byid[qid], r["pred"], **SPEC)
            if c is None: continue
            X.append([c, math.log(1 + len(tok(r["pred"])))])
            y.append(1.0 if r["judge"] else 0.0)
            arm.append(t)
    return np.array(X), np.array(y), np.array(arm)

def logit_fit(X, y, iters=200):
    X = np.hstack([np.ones((len(X), 1)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        Wd = p * (1 - p) + 1e-9
        g = X.T @ (y - p)
        H = -(X.T * Wd) @ X
        try: step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError: break
        b = b + step
        if np.max(np.abs(step)) < 1e-10: break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    cov = np.linalg.inv((X.T * (p * (1 - p))) @ X)
    return b, ll, np.sqrt(np.diag(cov))

for armset, famfun, label in [
    (["qc", "full"], lambda a: (a == "qc").astype(float), "qc vs full (n=1000)"),
    (["qc", "qc_no_qc", "qc_view_only", "qc_no_lifecycle", "full", "decay", "fifo"],
     lambda a: np.isin(a, ["qc", "qc_no_qc", "qc_view_only", "qc_no_lifecycle"]).astype(float),
     "qc-family vs baseline-family (n=3500)"),
]:
    X, y, arm = build(armset)
    b0, ll0, se0 = logit_fit(X, y)
    ind = famfun(arm).reshape(-1, 1)
    b1, ll1, se1 = logit_fit(np.hstack([X, ind]), y)
    chi2 = 2 * (ll1 - ll0); p = 1 - stats.chi2.cdf(chi2, 1)
    coef = b1[-1]; se = se1[-1]
    print(f"\n== LRT, {label} ==")
    print(f"  judge ~ coverage + log(len)           ll={ll0:.3f}")
    print(f"  judge ~ coverage + log(len) + armfam  ll={ll1:.3f}")
    print(f"  chi2={chi2:.3f}  p={p:.4f}   arm coef={coef:+.4f} se={se:.4f} "
          f"OR={math.exp(coef):.3f} 95%CI OR=[{math.exp(coef-1.96*se):.3f}, {math.exp(coef+1.96*se):.3f}]")
    print(f"  raw acc: " + ", ".join(f"{t}={np.mean([1 if arms[t][q]['judge'] else 0 for q in arms[t]]):.3f}" for t in armset))
