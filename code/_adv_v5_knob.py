"""Adversarial control 6: is the reported 0.733 -> 0.800 delta really 'the four bug fixes'?
The same edit window also introduced the core_df_cap=0.60 knob, whose own docstring says it
was set because 'bake' was gated out of question 88432d0a -- a question INSIDE the 60-question
subset the headline number is computed on.  Measure the knob's share of the delta. OFFLINE."""
import functools
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_chains as V
from _adv_v5_bugs import measure, summarize, line, HDR

_orig_groups = V.aggregation_groups


def patched(core_cap):
    @functools.wraps(_orig_groups)
    def f(store, question, use_hyponyms=True, df_cap=0.25, rel_keep=0.35, max_groups=8,
          core_df_cap=0.60):
        return _orig_groups(store, question, use_hyponyms, df_cap, rel_keep, max_groups,
                           core_cap if core_cap is not None else df_cap)
    return f


print("=== (f) how much of the reported delta is the core_df_cap knob, not a bug fix? ===")
print(HDR)
for cap, lab in [(0.60, "shipped: core_df_cap 0.60"),
                 (0.25, "core_df_cap 0.25 (= df_cap, pre-knob)"),
                 (0.40, "core_df_cap 0.40"), (0.80, "core_df_cap 0.80"),
                 (1.01, "core_df_cap 1.00 (no cap on typed terms)")]:
    V.aggregation_groups = patched(cap)
    print(line(summarize(measure(set(), "exact"), lab)))
V.aggregation_groups = _orig_groups

print("\n=== same knob sweep with ALL FOUR BUGS restored ===")
print(HDR)
for cap, lab in [(0.60, "all-4-bugs, core_df_cap 0.60"),
                 (0.25, "all-4-bugs, core_df_cap 0.25")]:
    V.aggregation_groups = patched(cap)
    print(line(summarize(measure({"i", "ii", "iii_nodbl", "iv"}, "exact"), lab)))
V.aggregation_groups = _orig_groups
