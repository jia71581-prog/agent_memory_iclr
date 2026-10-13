import json, re, statistics as st
from collections import Counter

recs = json.load(open("/root/agent_memory_iclr/audit/z_recs.json"))
lost = [r for r in recs if r["gold_in_full"] and not r["gold_in_cap"]]

print("== per-type conditional rate (confound check) ==")
den = Counter(r["type"] for r in recs)
contained = Counter(r["type"] for r in recs if r["gold_in_full"])
lc = Counter(r["type"] for r in lost)
print("%-26s  n   gold_in_uncapped   lost/n    lost/contained" % "type")
for t in sorted(den):
    c = contained[t]
    print("%-26s %3d   %3d (%.3f)        %.3f     %s" % (
        t, den[t], c, c / den[t], lc[t] / den[t],
        ("%d/%d = %.3f" % (lc[t], c, lc[t] / c)) if c else "-"))

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return c - h, c + h

print("\n35/500 Wilson 95%% CI: [%.3f, %.3f]" % wilson(35, 500))
print("221/500 -> 186/500 : McNemar-style, discordant pairs b=%d c=%d (deterministic, one-directional)"
      % (len(lost), sum(1 for r in recs if r["gold_in_cap"] and not r["gold_in_full"])))
