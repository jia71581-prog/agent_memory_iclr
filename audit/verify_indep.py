"""Independent adversarial recomputation of the qc vs no_qc routed-subset claim.
No imports from audit/*.py. Own F1, own exact tests.
"""
import json, math, re, os, hashlib, itertools, random
from collections import Counter

R = "/root/agent_memory_iclr/results/"

def load(tag):
    rows = [json.loads(l) for l in open(R + f"full500_{tag}.jsonl")]
    return rows

qc_rows = load("qc")
nq_rows = load("qc_no_qc")

print("== integrity ==")
print("rows:", len(qc_rows), len(nq_rows))
print("error rows qc:", sum(1 for r in qc_rows if "error" in r),
      " no_qc:", sum(1 for r in nq_rows if "error" in r))
print("dup qids qc:", len(qc_rows) - len(set(r["qid"] for r in qc_rows)),
      " no_qc:", len(nq_rows) - len(set(r["qid"] for r in nq_rows)))
qc = {r["qid"]: r for r in qc_rows}
nq = {r["qid"]: r for r in nq_rows}
print("qid sets equal:", set(qc) == set(nq))
print("row order identical:", [r["qid"] for r in qc_rows] == [r["qid"] for r in nq_rows])
qids = [r["qid"] for r in qc_rows]
print("qlabel identical across arms:", all(qc[q]["qlabel"] == nq[q]["qlabel"] for q in qids))
print("type identical across arms:", all(qc[q]["type"] == nq[q]["type"] for q in qids))
print("judge dtype:", Counter(type(qc[q]["judge"]).__name__ for q in qids))

# dataset order check
data = json.load(open("/root/agent_memory_iclr/data/longmemeval_s_cleaned.json"))
print("dataset n:", len(data), "order matches jsonl:", [d["question_id"] for d in data] == qids)
gold = {d["question_id"]: d for d in data}

print("\n== view_mode structure ==")
print("qc view_mode:", Counter(qc[q]["view_mode"] for q in qids))
print("no_qc view_mode:", Counter(nq[q]["view_mode"] for q in qids))
print("qc (qlabel->view_mode):", Counter((qc[q]["qlabel"], qc[q]["view_mode"]) for q in qids))
print("no_qc (qlabel->view_mode):", Counter((nq[q]["qlabel"], nq[q]["view_mode"]) for q in qids))

diff = [q for q in qids if qc[q]["view_mode"] != nq[q]["view_mode"]]
same = [q for q in qids if qc[q]["view_mode"] == nq[q]["view_mode"]]
print("n diff:", len(diff), " n same:", len(same))
print("all diff are qc=current/nq=historical:",
      all(qc[q]["view_mode"] == "current" and nq[q]["view_mode"] == "historical" for q in diff))
print("all same are historical/historical:",
      all(qc[q]["view_mode"] == "historical" == nq[q]["view_mode"] for q in same))
print("diff qlabel:", Counter(qc[q]["qlabel"] for q in diff))
print("same qlabel:", Counter(qc[q]["qlabel"] for q in same))

# ---------- own exact tests ----------
def exact_binom_two_sided(n01, n10):
    """Exact two-sided sign/McNemar test, p=0.5 null; sum of all outcomes <= as-likely."""
    n = n01 + n10
    if n == 0:
        return 1.0
    obs = math.comb(n, n01) / 2 ** n
    p = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) / 2 ** n
        if pi <= obs * (1 + 1e-12):
            p += pi
    return min(1.0, p)

def acc(arm, ids):
    return sum(1 for q in ids if arm[q]["judge"]), sum(1 for q in ids if arm[q]["judge"]) / len(ids)

def flips(ids):
    n01 = sum(1 for q in ids if qc[q]["judge"] and not nq[q]["judge"])   # qc right, nq wrong
    n10 = sum(1 for q in ids if nq[q]["judge"] and not qc[q]["judge"])
    both = sum(1 for q in ids if qc[q]["judge"] and nq[q]["judge"])
    neither = sum(1 for q in ids if not qc[q]["judge"] and not nq[q]["judge"])
    return n01, n10, both, neither

print("\n== accuracy / paired tests ==")
for name, ids in [("ALL", qids), ("ROUTED-DIFFERENT", diff), ("IDENTICAL-MODE", same)]:
    ca, aa = acc(qc, ids); cb, ab = acc(nq, ids)
    n01, n10, both, neither = flips(ids)
    p = exact_binom_two_sided(n01, n10)
    fa = sum(qc[q]["f1"] for q in ids) / len(ids)
    fb = sum(nq[q]["f1"] for q in ids) / len(ids)
    print(f"[{name}] n={len(ids)}  qc={ca} ({aa:.4f})  no_qc={cb} ({ab:.4f})  delta={aa-ab:+.4f} ({ca-cb:+d} q)")
    print(f"   2x2: both={both} neither={neither} n01(qc+/nq-)={n01} n10(qc-/nq+)={n10}  exact p={p:.4f}")
    print(f"   token-F1 qc={fa:.4f} no_qc={fb:.4f} delta={fa-fb:+.4f}")

# recompute token-F1 from pred+gold with my own implementation
W = re.compile(r"[a-z0-9]+")
def toks(s): return [t for t in W.findall(str(s).lower()) if t not in ("a", "an", "the")]
def f1(pred, g):
    p, gg = toks(pred), toks(g)
    if not p or not gg: return 0.0
    pc, gc = Counter(p), Counter(gg)
    ov = sum(min(c, gc[t]) for t, c in pc.items())
    if not ov: return 0.0
    pr, rc = ov / len(p), ov / len(gg)
    return 2 * pr * rc / (pr + rc)

print("\n== recomputed token-F1 from pred vs stored f1 ==")
mx = 0; nbad = 0
for q in qids:
    for arm in (qc, nq):
        r = f1(arm[q]["pred"], gold[q]["answer"])
        d = abs(r - arm[q]["f1"])
        mx = max(mx, d)
        if d > 0.0006: nbad += 1
print("max |recomputed - stored| f1:", round(mx, 5), " rows off by >0.0006:", nbad, "(stored rounded to 3dp)")

for name, ids in [("ROUTED-DIFFERENT", diff), ("IDENTICAL-MODE", same)]:
    fa = sum(f1(qc[q]["pred"], gold[q]["answer"]) for q in ids) / len(ids)
    fb = sum(f1(nq[q]["pred"], gold[q]["answer"]) for q in ids) / len(ids)
    ds = [f1(qc[q]["pred"], gold[q]["answer"]) - f1(nq[q]["pred"], gold[q]["answer"]) for q in ids]
    m = sum(ds) / len(ds)
    sd = (sum((x - m) ** 2 for x in ds) / (len(ds) - 1)) ** 0.5
    se = sd / len(ds) ** 0.5
    print(f"[{name}] my token-F1 qc={fa:.4f} no_qc={fb:.4f} delta={m:+.4f} sd={sd:.4f} se={se:.4f} t={m/se:+.2f}")

# ---------- the 2 identical-mode flips ----------
print("\n== identical-mode discordant pairs ==")
for q in same:
    if qc[q]["judge"] != nq[q]["judge"]:
        print(f"  qid={q} type={qc[q]['type']} qlabel={qc[q]['qlabel']} qc.judge={qc[q]['judge']} nq.judge={nq[q]['judge']}")
        print(f"     qc.f1={qc[q]['f1']} nq.f1={nq[q]['f1']}  pred_equal={qc[q]['pred'].strip()==nq[q]['pred'].strip()}")
        print(f"     gold={gold[q]['answer'][:120]!r}")
        print(f"     qc.pred={qc[q]['pred'][:200]!r}")
        print(f"     nq.pred={nq[q]['pred'][:200]!r}")

print("\n== pred determinism, per subset ==")
for name, ids in [("ROUTED-DIFFERENT", diff), ("IDENTICAL-MODE", same)]:
    ex = sum(1 for q in ids if qc[q]["pred"].strip() == nq[q]["pred"].strip())
    print(f"  [{name}] n={len(ids)} exact pred match={ex} ({ex/len(ids):.3f})")

# ---------- CI / power on the routed-different null ----------
print("\n== power / interval for the routed-different null (n=456) ==")
n01, n10, both, neither = flips(diff)
nd = n01 + n10
# exact CI on the discordant proportion -> CI on delta (Wald on paired diff)
ds = [(1 if qc[q]["judge"] else 0) - (1 if nq[q]["judge"] else 0) for q in diff]
m = sum(ds) / len(ds)
sd = (sum((x - m) ** 2 for x in ds) / (len(ds) - 1)) ** 0.5
se = sd / len(ds) ** 0.5
print(f"  discordant={nd} (n01={n01}, n10={n10})  paired mean delta={m:+.5f} se={se:.5f}")
print(f"  95% CI on accuracy delta ~ [{m-1.96*se:+.4f}, {m+1.96*se:+.4f}]  (i.e. +/-{1.96*se*len(diff):.1f} questions)")
# smallest true effect detectable at 80% power with 22 discordant pairs
print(f"  with {nd} discordant pairs, an 80%-power detectable |delta| is roughly "
      f"{1.96*se:.4f} (>= {1.96*se*len(diff):.0f} questions)")

# ---------- verify byte-identical-context premise from the ledgers ----------
print("\n== prompt-token comparison per question (ledger alignment) ==")
def per_q_prompt_tokens(path):
    rows = json.load(open(path))["rows"]
    out, cur = [], None
    for r in rows:
        if r["op"].startswith("answer/qc") and not r["op"].endswith("-retry"):
            if cur is not None: out.append(cur)
            cur = [r["prompt_tokens"]]
        elif r["op"] == "answer/qc-retry":
            cur.append(r["prompt_tokens"])
    if cur is not None: out.append(cur)
    return out

a = per_q_prompt_tokens(R + "full500_qc_ledger_1789633940.json")
b = per_q_prompt_tokens(R + "full500_qc_no_qc_ledger_1789634556.json")
print("blocks:", len(a), len(b))
if len(a) == len(b) == len(qids):
    eq_same = eq_diff = 0
    dtok_same, dtok_diff = [], []
    for q, ta, tb in zip(qids, a, b):
        d = ta[0] - tb[0]
        if qc[q]["view_mode"] == nq[q]["view_mode"]:
            dtok_same.append(d); eq_same += (d == 0)
        else:
            dtok_diff.append(d); eq_diff += (d == 0)
    print(f"  identical-mode  n={len(dtok_same)}: prompt_tokens equal in {eq_same} "
          f"({eq_same/len(dtok_same):.3f}); max|d|={max(abs(x) for x in dtok_same)}")
    print(f"  routed-diff     n={len(dtok_diff)}: prompt_tokens equal in {eq_diff} "
          f"({eq_diff/len(dtok_diff):.3f}); mean d={sum(dtok_diff)/len(dtok_diff):+.1f}, "
          f"median|d|={sorted(abs(x) for x in dtok_diff)[len(dtok_diff)//2]}")
