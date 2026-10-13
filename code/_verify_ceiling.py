"""ADVERSARIAL VERIFIER step 4: is the 29.9% 'ceiling for ANY head-bucketed rule' robust,
and does 'unique head' really mean 'genuinely distinct predicate'?

The head is computed AFTER v5_chains' own 24-entry hand synonym table SYN. So 'occupation'
is non-unique only because somebody typed occupation->job into that table. Test how much the
ceiling moves under mechanical (non-hand-picked) changes to the head normalisation."""
import pickle
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/root/agent_memory_iclr/code")
import v5_chains as V
from lifecycle import assemble_chains

MINE = pickle.load(open("/root/agent_memory_iclr/results/_verify_facts.pkl", "rb"))
stores = {}
for x in MINE:
    stores[x["qid"]] = assemble_chains(x["facts"])
base = st.median([len(s) for s in stores.values()])
print(f"v4 chains median {base}")


def head_of(attr):
    return V._head(V._canon_tokens(attr))


def ceiling(headfn, entfn=V._ent_canon):
    ns, uq = [], []
    for s in stores.values():
        h = Counter()
        for ch in s.values():
            h[(entfn(ch[0]["entity"]), headfn(ch[0]["attribute"]))] += 1
        ns.append(len(h))
        uq.append(sum(1 for v in h.values() if v == 1) / len(s))
    m = st.median(ns)
    return m, 1 - m / base, st.median(uq)


# ---- variant 1: exactly as claimed
m, r, u = ceiling(head_of)
print(f"[as claimed        ] ceiling median {m:.1f}  reduction {r:.4f}  unique-head {u:.4f}")

# ---- variant 2: SYN table disabled (how much of the merging is the 24 hand entries?)
saved = V._SYN_STEMMED
V._SYN_STEMMED = {}
m2, r2, u2 = ceiling(head_of)
V._SYN_STEMMED = saved
print(f"[SYN table OFF     ] ceiling median {m2:.1f}  reduction {r2:.4f}  unique-head {u2:.4f}")

# ---- variant 3: purely MECHANICAL head collapse - heads sharing a 5-char prefix in the same
# question are one bucket (catches stemmer misses: 'subscription'/'subscribe',
# 'organisation'/'organise'; no semantics, no hand table, no data fitting)
def ceiling_prefix(P):
    ns, uq = [], []
    for s in stores.values():
        raw = defaultdict(list)
        for ch in s.values():
            raw[(V._ent_canon(ch[0]["entity"]), head_of(ch[0]["attribute"]))].append(1)
        # union heads of the same entity sharing a P-char prefix
        keys = sorted(raw)
        parent = {k: k for k in keys}
        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k
        byent = defaultdict(list)
        for k in keys:
            byent[k[0]].append(k)
        for ent, ks in byent.items():
            pref = defaultdict(list)
            for k in ks:
                if len(k[1]) >= P:
                    pref[k[1][:P]].append(k)
            for _, grp in pref.items():
                for k in grp[1:]:
                    a, b = find(grp[0]), find(k)
                    if a != b:
                        parent[b] = a
        buck = Counter()
        for k, v in raw.items():
            buck[find(k)] += len(v)
        ns.append(len(buck))
        uq.append(sum(1 for v in buck.values() if v == 1) / len(s))
    m = st.median(ns)
    return m, 1 - m / base, st.median(uq)


for P in (6, 5, 4):
    m3, r3, u3 = ceiling_prefix(P)
    print(f"[+{P}-char prefix    ] ceiling median {m3:.1f}  reduction {r3:.4f}  unique-head {u3:.4f}")

# ---- variant 4: ignore the head entirely, bucket on ANY shared content stem with the same
# entity (transitive) - the true ceiling of lexical (non-embedding) chain merging
def ceiling_anyshared():
    ns = []
    for s in stores.values():
        keys = list(s)
        parent = {k: k for k in keys}
        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k
        bytok = defaultdict(list)
        for k in keys:
            e = V._ent_canon(s[k][0]["entity"])
            for t in set(V._canon_tokens(s[k][0]["attribute"])):
                bytok[(e, t)].append(k)
        for _, grp in bytok.items():
            for k in grp[1:]:
                a, b = find(grp[0]), find(k)
                if a != b:
                    parent[b] = a
        ns.append(len({find(k) for k in keys}))
    m = st.median(ns)
    return m, 1 - m / base


m4, r4 = ceiling_anyshared()
print(f"[any shared stem   ] ceiling median {m4:.1f}  reduction {r4:.4f}   (over-merges; upper bound)")

# ---- how 'genuinely distinct' are the unique heads? count unique heads that have a near
# neighbour head in the SAME question (edit distance <=2 or 5-char prefix)
def edist(a, b):
    if abs(len(a) - len(b)) > 2:
        return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


shares = []
for s in stores.values():
    h = Counter()
    for ch in s.values():
        h[(V._ent_canon(ch[0]["entity"]), head_of(ch[0]["attribute"]))] += 1
    uniq = [k for k, v in h.items() if v == 1]
    allh = list(h)
    near = 0
    for k in uniq:
        for k2 in allh:
            if k2 == k or k2[0] != k[0]:
                continue
            if (len(k[1]) >= 5 and len(k2[1]) >= 5 and k[1][:5] == k2[1][:5]) or \
               (len(k[1]) >= 4 and edist(k[1], k2[1]) <= 2):
                near += 1
                break
    shares.append(near / max(1, len(uniq)))
print(f"\nshare of UNIQUE heads that have a lexically near-identical sibling head in the same "
      f"question: median {st.median(shares):.4f} mean {st.mean(shares):.4f}")

# ---- most common heads across the corpus: how free-form is the predicate space really?
allheads = Counter()
for s in stores.values():
    for ch in s.values():
        allheads[head_of(ch[0]["attribute"])] += 1
tot = sum(allheads.values())
print(f"\ndistinct heads corpus-wide {len(allheads)} over {tot} chains")
print("top 25 heads:", ", ".join(f"{h}:{c}" for h, c in allheads.most_common(25)))
cum = 0
for i, (h, c) in enumerate(allheads.most_common(), 1):
    cum += c
    if cum / tot >= 0.5:
        print(f"top {i} heads cover 50% of all chains")
        break
