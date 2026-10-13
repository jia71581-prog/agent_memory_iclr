"""QC-Lifecycle 核心部件 1:查询时态分类器(确定性规则,可审计)。

三分类:
  current  - 问当前状态(默认;或含 currently/now/latest/still 等)
  historical - 问历史状态/变化过程(含 previously/before/originally/first/used to/history/change 等)
  neutral  - 单会话事实型,无时态敏感(检索命中哪层用哪层)
规则表显式列出,论文中整表展示(vs RD-Forget 的 LLM 黑盒 intent 判断)。
"""
import re

HISTORICAL_PATTERNS = [
    r"\bpreviously\b", r"\bbefore\b", r"\boriginally\b", r"\bat first\b",
    r"\bused to\b", r"\bformer(ly)?\b", r"\bold\b.*\b(value|name|address|job|phone)\b",
    r"\bfirst time\b", r"\binitial(ly)?\b", r"\bhistory of\b", r"\bhow (has|did) .* change",
    r"\bearlier\b", r"\bback (then|when)\b", r"\bwhat was\b.*\bbefore\b",
    # previous X = 历史指称,但 "previous chat/conversation/..." 是会话元指称(单会话题),排除
    r"\bprevious\b(?!\s+(chat|chats|conversation|conversations|discussion|discussions|session|sessions|message|messages))",
]
CURRENT_PATTERNS = [
    r"\bcurrent(ly)?\b", r"\bnow\b", r"\blatest\b", r"\bstill\b", r"\btoday\b",
    r"\bright now\b", r"\bas of\b", r"\bup[- ]to[- ]date\b", r"\bmost recent\b",
]
# 时间演化敏感题型(LongMemEval 标签,仅用于评测分桶,不进分类器决策)
_hist_re = [re.compile(p) for p in HISTORICAL_PATTERNS]
_curr_re = [re.compile(p) for p in CURRENT_PATTERNS]


def classify_query(question):
    """返回 (label, fired_rules)。优先级: historical > current > neutral(带默认current视图)。
    设计依据:historical 词面是强信号;current 词面次之;
    无词面命中的问题默认按 current 处理(用户问事实默认问现状),
    但 neutral 标签保留给消融分析。"""
    ql = question.lower()
    fired = [p.pattern for p in _hist_re if p.search(ql)]
    if fired:
        return "historical", fired
    fired = [p.pattern for p in _curr_re if p.search(ql)]
    if fired:
        return "current", fired
    return "neutral", []


if __name__ == "__main__":
    import json
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import load_dataset
    data = load_dataset("s_cleaned")
    from collections import Counter
    dist = Counter()
    for q in data:
        label, _ = classify_query(q["question"])
        dist[(q["question_type"], label)] += 1
    for (t, l), c in sorted(dist.items()):
        print(f"{t:28s} -> {l:10s} {c}")
