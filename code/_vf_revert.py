"""ADVERSARIAL: revert each of the 4 claimed v5_temporal.py fixes independently.

For each variant: (a) does the module self-test FAIL (i.e. is the regression test real)?
(b) does the claimed wrong behaviour actually appear on the REAL corpus sentence?
Offline, no API.
"""
import os
import re
import shutil
import subprocess
import sys

SRC = "/root/agent_memory_iclr/code/v5_temporal.py"
TMP = "/tmp/v5t_variants"
os.makedirs(TMP, exist_ok=True)
base = open(SRC).read()

VARIANTS = {}

# --- bug 1: MONTHS three-letter aliases off by one -------------------------
VARIANTS["bug1_months_offby1"] = (
    "MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})",
    "MONTHS.update({m[:3]: i + 1 for m, i in list(MONTHS.items())})",
)

# --- bug 2: numeric-date rule with no following-noun / quantifier guard ----
g2 = '''        if re.search(r"^\\s*(?:more\\s+)?(episodes?|times?|days?|hours?|minutes?|weeks?|months?|years?|"
                     r"people|cups?|tsp|tbsp|miles?|km|kg|lbs?|oz|sets?|reps?|pages?|chapters?|"
                     r"servings?|glasses|bottles?|nights?|sessions?)\\b", after):
            continue                                  # "2-3 episodes" / "1/2 cup" 不是日期
        if re.search(r"\\b(at least|about|around|roughly|maybe|between|some|every|per)\\s*$", before):
            continue
'''
assert g2 in base, "guard-2 block not found verbatim"
VARIANTS["bug2_no_episode_guard"] = (g2, "")

# --- bug 3: explicit date after session date rolled back a year -----------
g3 = '''        conf, kind = "high", "explicit"
        if not m.group(3) and dt > ref + timedelta(days=1):
            # 会话日之后的显式日期 = 计划中的未来事件(机票/面试),不能当过去事件定时
            conf, kind = "med", "explicit-future"
        add(m.group(0), dt, conf, kind)'''
assert g3 in base
VARIANTS["bug3_rollback_year"] = (g3, '''        conf, kind = "high", "explicit"
        if not m.group(3) and dt > ref + timedelta(days=1):
            try:
                dt = datetime(yr - 1, mo, dy)
            except ValueError:
                pass
        add(m.group(0), dt, conf, kind)''')

# --- bug 4: anchor-day hatch without the "sentence has its own date" test --
g4 = '''                default_dated = prov in ("", "today")
                if abs((ev - anchor_dt).days) > window and not (
                        same_session and default_dated and ANCHOR_RE.search(sent)):'''
assert g4 in base
VARIANTS["bug4_hatch_no_owndate"] = (g4, '''                default_dated = prov in ("", "today")
                if abs((ev - anchor_dt).days) > window and not (
                        same_session and ANCHOR_RE.search(sent)):''')

results = {}
for name, (old, new) in VARIANTS.items():
    assert old in base, name
    path = os.path.join(TMP, "v5_temporal.py")
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    # variant dir needs the sibling modules v5_temporal imports
    open(path, "w").write(base.replace(old, new))
    p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=300,
                       cwd="/root/agent_memory_iclr/code")
    tail = (p.stdout + p.stderr).strip().splitlines()
    results[name] = (p.returncode, tail[-1] if tail else "")

print("== self-test on each reverted variant (expect nonzero exit = regression test is real) ==")
for k, (rc, msg) in results.items():
    print(f"{k:26s} exit={rc}  {msg[:160]}")

# baseline
p = subprocess.run([sys.executable, SRC], capture_output=True, text=True, timeout=300,
                   cwd="/root/agent_memory_iclr/code")
print(f"{'CURRENT (unreverted)':26s} exit={p.returncode}  {p.stdout.strip()[-120:]}")
