"""Test 2: mechanism audit for 09ba9854 -- is the view's 'protection' extraction-provenance or truncation?"""
import json, re, sys
sys.path.insert(0, "/root/agent_memory_iclr/code")
from _zver_core import build_ctx, load, MISS

D = {q["question_id"]: q for q in load()}
q = D["09ba9854_abs"]
c = build_ctx(q)
print("cache misses:", len(MISS))
print("mode:", c["mode"], "label:", c["label"])
print("view chars capped=%d full=%d  (cap fired: %s)" % (len(c["view"]), len(c["view_full"]), len(c["view_full"]) > 20000))
vf = c["view_full"].splitlines()
vc = c["view"].splitlines()
print("view lines: capped=%d full=%d  -> %d lines DELETED by the 20k cap" % (len(vc), len(vf), len(vf) - len(vc)))

KEY = re.compile(r"bus|taxi|fare|limousine|cab|narita express|\bnex\b|train|ticket|yen|¥|\$|cost|price|save|savings", re.I)
print("\n--- KEPT view lines matching bus/taxi/fare/price vocabulary ---")
for l in vc:
    if KEY.search(l):
        print("  KEPT :", l[:200])
print("\n--- DELETED-BY-CAP view lines matching same vocabulary ---")
ndel = 0
for l in vf[len(vc):]:
    if KEY.search(l):
        ndel += 1
        print("  CUT  :", l[:200])
print("  (%d cut lines match)" % ndel)

NARROW = re.compile(r"\bbus\b|\btaxi\b|\bfare\b|limousine|\bcab\b", re.I)
print("\nnarrow bus/taxi/fare/limousine matches: kept=%d cut=%d" % (
    sum(1 for l in vc if NARROW.search(l)), sum(1 for l in vf[len(vc):] if NARROW.search(l))))

# who said the bus/taxi numbers in the raw haystack?
print("\n--- speaker provenance of bus/taxi/fare mentions in the FULL haystack ---")
cnt = {"user": 0, "assistant": 0}
ex = {"user": [], "assistant": []}
for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
    for t in sess:
        ct = t.get("content") or ""
        if NARROW.search(ct):
            cnt[t["role"]] = cnt.get(t["role"], 0) + 1
            if len(ex[t["role"]]) < 6:
                ex[t["role"]].append((date, ct[:260]))
print("turns mentioning bus/taxi/fare:", cnt)
for role in ("user", "assistant"):
    print(" ", role.upper())
    for d, s in ex[role]:
        print("    [%s] %s" % (d, s.replace("\n", " ")))

# retrieved top-20 composition
ru = c["raw_units"]
import collections
print("\ntop-20 retrieved roles:", collections.Counter(r[1] for r in ru))
chars = collections.Counter()
for r in ru:
    chars[r[1]] += len(r[2])
tot = sum(chars.values())
print("retrieved chars by role:", dict(chars), "assistant share=%.3f" % (chars["assistant"] / tot))
NUM = re.compile(r"\d[\d,\.]*")
nums = collections.Counter()
for r in ru:
    nums[r[1]] += len(NUM.findall(r[2]))
print("numeric tokens by role:", dict(nums))
