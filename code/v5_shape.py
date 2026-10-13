"""QC-Lifecycle v5:question-SHAPE 条件化(替换已证 null 的 current/historical 时态轴)。

设计动机(实测):
  时态轴 null(p=0.84,456 题上 delta=0.0000),因为 (a) current/historical 两视图
  答案含量几乎相同(gold 在 current 186/500,historical 188/500),(b) 425/500 题无时态词面。
  时态只能决定"取链的哪一段",无法决定"要算什么"。SHAPE 决定后者,词面覆盖面大得多。

轴的重构:SHAPE(要算什么)= 主轴;TENSE(取哪一段)降级为 modifier。

========================= 规则表(论文原样打印)=========================
优先级从上到下,第一个命中即返回。每条规则给出 rule_id 和触发词面。

 S1  RECALL.assistant  上一轮助手输出的复述请求
       R1a remind_me        "remind me"
       R1b you_uttered      "you (mentioned|said|recommended|suggested|provided|told me|gave me)"
       R1c metaref_ours     "our (previous|earlier|prior|last) ..." / "(previous|earlier) (chat|conversation|discussion)"
       R1d we_discussed     "we (discussed|talked about)" / "going back to our" / "looking back at our"
 S2  PREF.advice       建议/推荐请求(gold 是 rubric 而非事实串)
       R2a any_tips         "any (tips|advice|suggestions|ideas|recommendations)"
       R2b can_you_rec      "can you (recommend|suggest)" / "do you have any (ideas|tips)"
       R2c what_should_i    "what should i" / "what do you think" / "do you think it would be"
       R2d rec_noun         "recommend(ations)" / "suggest(ions)" 出现且无 wh-事实疑问词
 S3  TEMP.offset       两锚点间/距今的时长算术
       R3a howmany_unit_ago "how (many|long) <unit> ago"
       R3b passed_between   "<unit> (passed|had passed) (between|since)" / "how many <unit> passed"
       R3c how_long_take    "how long (did|does) it take" / "how many <unit> did it take"
       R3d how_long_been    "how long (have|had) i been ... (when|before)"
       R3e n_unit_before    "how many <unit> before" / "how long before"
 S4  TEMP.order        次序/最近性选择(需要按时间排序的候选表)
       R4a happened_first   "happened first" / "which|who|what ... first"
       R4b order_of         "(order|sequence) of" / "from earliest to latest" / "first to last"
       R4c most_recent      "most recent(ly)" / "latest" 作选择用
 S5  AGG.arith         少数具名操作数上的闭式算术
       R5a compare_more     "how much (more|less|faster|earlier|older|cheaper|higher)" / "compared to" / "difference (in|between)"
       R5b total_of_and     "(total|average|combined|sum)" 且句中有 "X and Y" 具名并列
       R5c percent          "what percentage" / "percentage of" / "what percent"
       R5d age_arith        "how old (was|am|will) i" / "how many years older"
 S6  AGG.enum          开放式枚举计数(需要完整证据集)
       R6a count_window     计数疑问词 + 时间窗("in the last/past N ...", "this year", "in march", "in total")
       R6b count_episodic   计数疑问词 + 情节性动词("did i (attend|buy|visit|spend|acquire|...)", "have i (tried|completed|...)")
       R6c count_union      计数疑问词 + 并集词("including", "both", "combined", "in total", "altogether", "each")
 S7  AGG.latest_count  静态量的最新报值("how many X do I (currently) have/own")
       S7 = 计数疑问词 + 静态现在时(have/own/are there/is on ...) 且无 S6 的窗口/情节/并集线索
 S8  TEMP.anchor       相对日期锚定的事实查找(先解析日期,再查该日附近)
       S8 = 相对日期点表达("N (days|weeks|months) ago", "last saturday", "the past weekend", "a month ago")
            且无量化疑问词
 S9  LOOKUP.rate       频率/速率查找("how often", "how many X per week/a day/in a typical week")
 S10 LOOKUP.*          默认单事实查找;TENSE 作 modifier:
       .latest   有 "current(ly)|now|still|as of|up-to-date" 词面
       .history  有 "previously|used to|originally|before|initial(ly)|former" 词面
       .plain    无时态词面(默认)
=======================================================================

每个 shape 声明一个 rendering 策略(SHAPE_STRATEGY),由 parent 接线;
本模块只做分类 + 提供确定性的日期解析原语(TEMP.* 策略要用)。
无 API 调用,无训练,纯规则。
"""
import re
from datetime import datetime, timedelta

# ---------------------------------------------------------------- 词面原子
UNIT = r"(?:seconds?|minutes?|hours?|days?|nights?|weeks?|months?|years?)"
COUNT_WH = r"(?:how many|how much|what is the (?:total|number|average|combined)|what's the (?:total|number)|total number of|how far|how long)"

# 计数/量化疑问词(S5-S7 的入口条件)
QUANT = [
    ("Q1", r"\bhow many\b"),
    ("Q2", r"\bhow much\b"),
    ("Q3", r"\bwhat (?:is|was) the (?:total|average|number|combined|minimum|maximum|difference)\b"),
    ("Q4", r"\btotal (?:number|amount|cost|distance|weight|time|price|value)\b"),
    ("Q5", r"\bwhat percentage\b|\bwhat percent\b"),
    ("Q6", r"\bhow old\b"),
    ("Q7", r"\bhow far\b"),
]

# 时间窗(限定一个区间内的全部情节)
WINDOW = [
    ("W1", r"\bin the (?:last|past|previous) (?:\w+ )?" + UNIT + r"\b"),
    ("W2", r"\b(?:this|last|past) (?:year|month|week|weekend)\b"),
    ("W3", r"\bin (?:january|february|march|april|may|june|july|august|september|october|november|december)\b"),
    ("W4", r"\bin total\b|\bin all\b|\baltogether\b"),
    ("W5", r"\bsince (?:i|the|my|starting)\b"),
    ("W6", r"\bso far\b|\bto date\b"),
    ("W7", r"\bfrom \w+ to \w+\b"),
    ("W8", r"\b(?:each|every) (?:day|week|month|year)\b"),
    ("W9", r"\brecently\b|\bthese days\b"),
]

# 情节性动词(过去/现在完成:发生过的可枚举事件)
EPISODIC = [
    ("E1", r"\bdid i\b"),
    ("E2", r"\bhave i (?:\w+ )?(?:tried|completed|attended|visited|bought|purchased|read|watched|made|done|spent|played|written|worked|used|led|acquired|raised|earned|listened|ridden|worn|gone|been to)\b"),
    ("E3", r"\bhave i (?:ever )?\w+ed\b"),
    ("E4", r"\bdid it take\b"),
    ("E5", r"\bi (?:attended|visited|bought|spent|made|took|went|had)\b"),
]

# 并集/合计线索
UNION = [
    ("U1", r"\bincluding\b|\bincludes\b"),
    ("U2", r"\bboth\b"),
    ("U3", r"\bcombined\b|\bin total\b|\baltogether\b|\btotal\b"),
    ("U4", r"\bacross\b"),
    ("U5", r"\bor\b.*\bin (?:the )?(?:last|past)\b"),
]

# 静态现在时(状态量,不是情节序列)
STATIVE = [
    ("T1", r"\bdo i (?:currently )?(?:have|own|use|need|possess)\b"),
    ("T2", r"\bare there\b|\bis there\b"),
    ("T3", r"\bdo i (?:have|need) left\b|\bdo i have left\b"),
    ("T4", r"\bare (?:currently )?(?:on|in)\b"),
    ("T5", r"\bam i\b"),
]

# 相对日期"点"表达(锚定单个日期,而非区间)
REL_POINT = [
    ("P1", r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|a couple of|a few|several) " + UNIT + r" ago\b"),
    ("P2", r"\b(?:last|this|past) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    ("P3", r"\b(?:last|the past|this) (?:weekend|night|week|month)\b"),
    ("P4", r"\byesterday\b|\btoday\b"),
    ("P5", r"\bon (?:valentine's|christmas|new year's|thanksgiving|easter)\b"),
    ("P6", r"\bthe (?:\w+ )?two " + UNIT + r" ago\b"),
]

RULES = {
    # S1 RECALL.assistant
    "R1a": r"\bremind me\b",
    "R1b": r"\byou (?:mentioned|said|recommended|suggested|provided|told me|gave me|listed|described)\b",
    "R1c": r"\bour (?:previous|earlier|prior|last)\b|\b(?:previous|earlier) (?:chat|conversation|discussion|talk)\b",
    "R1d": r"\bwe (?:discussed|talked about)\b|\b(?:going|looking|getting) back (?:to|at|through) (?:our|the|my)\b|\bfollow(?:ing)? up on our\b|\bcheck(?:ing)? back on our\b",
    # S2 PREF.advice
    "R2a": r"\bany (?:tips|advice|suggestions|ideas|recommendations|thoughts)\b",
    "R2b": r"\bcan you (?:recommend|suggest)\b|\bdo you have any (?:ideas|tips|suggestions|advice)\b|\bcould you (?:recommend|suggest)\b",
    "R2c": r"\bwhat should i\b|\bwhat do you think\b|\bdo you think (?:it|i|there)\b|\bwhat would you\b",
    "R2d": r"\brecommend(?:ations?)?\b|\bsuggest(?:ions?)?\b|\badvice\b",
    # S3 TEMP.offset
    "R3a": r"\bhow (?:many|long) " + UNIT + r"?\s*ago\b|\bhow many " + UNIT + r" ago\b",
    "R3b": r"\b" + UNIT + r" (?:passed|had passed|have passed|has passed)\b|\bhow many " + UNIT + r" (?:passed|had passed|have passed|has|have)\b|\bpassed (?:between|since)\b",
    "R3c": r"\bhow long (?:did|does|will) it take\b|\bhow many " + UNIT + r" did it take\b|\bdid it take (?:for|me)\b",
    "R3d": r"\bhow long (?:have|had) i been\b|\bhow many " + UNIT + r" (?:have|had) i been\b|\bhow long (?:have|had) i\b",
    "R3e": r"\bhow many " + UNIT + r" before\b|\bhow long before\b|\bhow long did i \w+ before\b|\bhow many " + UNIT + r" (?:since|between)\b|\bhow (?:many|much) " + UNIT + r" in total do i spent\b",
    # S4 TEMP.order
    "R4a": r"\bhappened first\b|\b(?:which|who|what) [^?]*\bfirst\b|\bdid i [^?]*\bfirst\b|\b(?:first|second|third) among\b",
    "R4b": r"\b(?:order|sequence) of\b|\bfrom (?:earliest|first) to (?:latest|last)\b|\bin (?:chronological )?order\b|\bstarting from the earliest\b",
    "R4c": r"\bmost recent(?:ly)?\b|\bwhich [^?]*\blatest\b",
    # S5 AGG.arith
    "R5a": r"\bhow much (?:more|less|faster|slower|earlier|later|older|younger|cheaper|higher|lower)\b|\bcompared to\b|\bdifference (?:in|between)\b|\bhow many " + UNIT + r" (?:more|less|older|younger)\b|\bhow much (?:will i save|did i save|have i saved)\b|\bexceed\b|\bmore than my\b",
    "R5b": r"\b(?:total|average|combined|sum)\b",
    "R5c": r"\bwhat percentage\b|\bwhat percent\b|\bpercentage (?:of|discount|increase)\b",
    "R5d": r"\bhow old (?:was|am|will) i\b|\bhow many years (?:older|younger|will i be)\b|\baverage age\b",
    # S9 LOOKUP.rate
    "R9a": r"\bhow often\b",
    "R9b": r"\bin a typical (?:week|day|month)\b|\bper (?:week|day|month)\b|\ba (?:week|day) do i\b|\bdays a week\b|\bevery day\b|\bin a day\b",
    # S10 modifiers
    "M_latest": r"\bcurrent(?:ly)?\b|\bnow\b|\bstill\b|\bas of\b|\bup[- ]to[- ]date\b|\btoday\b",
    # "previous X" 里 X=chat/conversation 是会话元指称,已由 S1 吸收;此处排除以免误判
    "M_history": r"\bpreviously\b|\bused to\b|\boriginally\b|\binitial(?:ly)?\b|\bformer(?:ly)?\b|\bat first\b|\bbefore\b|\bprevious\b(?!\s+(?:chat|conversation|discussion|message)s?)",
}

_C = {k: re.compile(v) for k, v in RULES.items()}
_CQ = [(k, re.compile(v)) for k, v in QUANT]
_CW = [(k, re.compile(v)) for k, v in WINDOW]
_CE = [(k, re.compile(v)) for k, v in EPISODIC]
_CU = [(k, re.compile(v)) for k, v in UNION]
_CT = [(k, re.compile(v)) for k, v in STATIVE]
_CP = [(k, re.compile(v)) for k, v in REL_POINT]

# wh-事实疑问词(用于 R2d 的排除:含事实疑问词说明在问事实,不是求建议)
_WH_FACT = re.compile(r"\bwhat (?:was|is|were) the (?:name|colou?r|price|number|title|address|phone)\b|\bwhich (?:one|of)\b|\bwhat did (?:i|you)\b|\bwhat (?:was|were) (?:my|the)\b")

SHAPES = ["RECALL.assistant", "PREF.advice", "TEMP.offset", "TEMP.order", "TEMP.anchor",
          "AGG.arith", "AGG.enum", "AGG.latest_count", "LOOKUP.rate",
          "LOOKUP.latest", "LOOKUP.history", "LOOKUP.plain"]

# 粗粒度族(论文正文用 5 族;细粒度用于消融)
FAMILY = {
    "RECALL.assistant": "RECALL", "PREF.advice": "PREFERENCE",
    "TEMP.offset": "TEMPORAL", "TEMP.order": "TEMPORAL", "TEMP.anchor": "TEMPORAL",
    "AGG.arith": "AGGREGATE", "AGG.enum": "AGGREGATE", "AGG.latest_count": "AGGREGATE",
    "LOOKUP.rate": "LOOKUP", "LOOKUP.latest": "LOOKUP", "LOOKUP.history": "LOOKUP",
    "LOOKUP.plain": "LOOKUP",
}

# 每 shape 的 rendering 策略声明(parent 接线用;本模块不执行)
SHAPE_STRATEGY = {
    "AGG.enum":         "count-annotated item clusters: 会话覆盖式检索(每会话最高分轮轮询)+ 按 (entity,attribute-cluster) 聚合并显式编号计数",
    "AGG.arith":        "operand-slot view: 从问句抽出具名操作数(and/compared-to 并列),每个操作数单独检索一段证据,显式列出后再算",
    "AGG.latest_count": "latest-value view: 该量的最新报值 + 全部历史报值(标 stale),禁止把历史值相加",
    "TEMP.offset":      "offset timeline: 每候选事件渲染 [绝对日期 | 距今 N 天 / N 周],算术在渲染期完成,模型只做选择",
    "TEMP.order":       "sorted timeline: 候选事件按日期升序编号,仅给日期+事件短语",
    "TEMP.anchor":      "date-resolved window: 把问句相对日期解析成绝对日期,优先给该日期 ±3 天窗口内的原文轮",
    "RECALL.assistant": "assistant-turn view: 检索限定在 assistant 轮,原文逐字给出(不做事实抽取)",
    "PREF.advice":      "rubric view: 用户偏好/约束型事实(preference/constraint 属性链)逐字给出",
    "LOOKUP.rate":      "latest-value view(rate 属性)",
    "LOOKUP.latest":    "latest-value view",
    "LOOKUP.history":   "full-chain evolution view",
    "LOOKUP.plain":     "latest-value view + BM25 原文轮(现状默认)",
}


def _hits(compiled, ql):
    return [k for k, c in compiled if c.search(ql)]


def classify_shape(question):
    """返回 (shape, fired_rules)。shape 形如 'AGG.enum';fired_rules 是命中的 rule_id 列表。
    优先级 = 上面规则表的 S1..S10 顺序,第一个命中即返回(确定性,可审计)。"""
    ql = question.lower()
    f = []

    # S1 RECALL.assistant
    f = [k for k in ("R1a", "R1b", "R1c", "R1d") if _C[k].search(ql)]
    if f:
        return "RECALL.assistant", f

    # S2 PREF.advice
    f = [k for k in ("R2a", "R2b", "R2c") if _C[k].search(ql)]
    if f:
        return "PREF.advice", f
    if _C["R2d"].search(ql) and not _WH_FACT.search(ql):
        return "PREF.advice", ["R2d"]

    # S3 TEMP.offset
    f = [k for k in ("R3a", "R3b", "R3c", "R3d", "R3e") if _C[k].search(ql)]
    if f:
        return "TEMP.offset", f

    # S4 TEMP.order
    f = [k for k in ("R4a", "R4b", "R4c") if _C[k].search(ql)]
    if f:
        return "TEMP.order", f

    quant = _hits(_CQ, ql)
    win = _hits(_CW, ql)
    epi = _hits(_CE, ql)
    uni = _hits(_CU, ql)
    sta = _hits(_CT, ql)

    # S5 AGG.arith
    if quant:
        f = [k for k in ("R5a", "R5c", "R5d") if _C[k].search(ql)]
        if f:
            return "AGG.arith", quant + f
        # total/average/combined + 具名并列 X and Y
        if _C["R5b"].search(ql) and re.search(r"\w+ and (?:my |the |a )?\w+", ql):
            return "AGG.arith", quant + ["R5b"]

    # S6 AGG.enum
    if quant and (win or epi or uni):
        return "AGG.enum", quant + win + epi + uni

    # S7 AGG.latest_count
    if quant and sta:
        return "AGG.latest_count", quant + sta

    # S8 TEMP.anchor(无量化疑问词的相对日期锚定查找)
    pnt = _hits(_CP, ql)
    if pnt and not quant:
        return "TEMP.anchor", pnt

    # S9 LOOKUP.rate
    f = [k for k in ("R9a", "R9b") if _C[k].search(ql)]
    if f:
        return "LOOKUP.rate", f

    # S10 LOOKUP.* (tense 降级为 modifier;两者同时命中时 latest 优先 ——
    # "previous/before" 常只是用来指称实体("我上一家公司的同事"),而 "currently" 直接
    # 修饰被问的状态本身)
    if _C["M_latest"].search(ql):
        return "LOOKUP.latest", ["M_latest"] + quant
    if _C["M_history"].search(ql):
        return "LOOKUP.history", ["M_history"] + quant
    return "LOOKUP.plain", quant


# ------------------------------------------------- TEMP.* 策略用的日期原语(纯规则)
_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
         "a couple of": 2, "a few": 3, "several": 3}
_DOW = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
        "saturday": 5, "sunday": 6}
_UNIT_DAYS = {"day": 1, "night": 1, "week": 7, "month": 30, "year": 365}


def parse_qdate(question_date):
    return datetime.strptime(question_date.split(" (")[0], "%Y/%m/%d")


def resolve_relative_dates(question, question_date):
    """把问句里的相对日期表达解析成绝对日期(确定性,无 API)。
    返回 [(表达原文, 解析出的 date, 容差天数)]。month 按 30 天近似,容差随之放宽。"""
    ql = question.lower()
    qd = parse_qdate(question_date)
    out = []
    for m in re.finditer(r"\b(\d+|a couple of|a few|several|" + "|".join(_NUMW) + r")\s+(day|night|week|month|year)s?\s+ago\b", ql):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMW[m.group(1)]
        u = m.group(2)
        out.append((m.group(0), qd - timedelta(days=n * _UNIT_DAYS[u]), 3 if u in ("day", "night") else (4 if u == "week" else 10)))
    for m in re.finditer(r"\b(?:last|this|past)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", ql):
        target = _DOW[m.group(1)]
        back = (qd.weekday() - target) % 7
        back = back or 7
        out.append((m.group(0), qd - timedelta(days=back), 2))
    if re.search(r"\b(?:last|the past|this)\s+weekend\b", ql):
        back = (qd.weekday() - 5) % 7 or 7
        out.append(("weekend", qd - timedelta(days=back), 2))
    if re.search(r"\byesterday\b", ql):
        out.append(("yesterday", qd - timedelta(days=1), 1))
    return out


def days_between(date_a, date_b):
    """两个 haystack 日期串之间的天数(绝对值)。"""
    a = datetime.strptime(date_a.split(" (")[0], "%Y/%m/%d")
    b = datetime.strptime(date_b.split(" (")[0], "%Y/%m/%d")
    return abs((a - b).days)


def print_rule_table():
    print(__doc__.split("=========================")[1])
    for s in SHAPES:
        print(f"{s:20s} {FAMILY[s]:11s} {SHAPE_STRATEGY[s]}")


if __name__ == "__main__":
    # 单元测试:每条规则至少一个真实数据集问句
    CASES = [
        ("Can you remind me of the name of that romantic Italian restaurant in Rome you recommended for dinner?", "RECALL.assistant"),
        ("I was going through our previous conversation about The Library of Babel, and I wanted to confirm - what did Borges say about the Library?", "RECALL.assistant"),
        ("I think we discussed work from home jobs for seniors earlier. Can you remind me what was the 7th job?", "RECALL.assistant"),
        ("Can you suggest a hotel for my upcoming trip to Miami?", "PREF.advice"),
        ("I've been sneezing quite a bit lately. Do you think it might be my living room?", "PREF.advice"),
        ("My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?", "PREF.advice"),
        ("How many days ago did I buy a smoker?", "TEMP.offset"),
        ("How many days passed between the day I fixed my mountain bike and the day I upgraded my pedals?", "TEMP.offset"),
        ("How many weeks had passed since I recovered from the flu when I went on my 10th jog outdoors?", "TEMP.offset"),
        ("How long did it take me to finish 'The Nightingale'?", "TEMP.offset"),
        ("Which event happened first, my cousin's wedding or Michael's engagement party?", "TEMP.order"),
        ("What is the order of the six museums I visited from earliest to latest?", "TEMP.order"),
        ("Which streaming service did I start using most recently?", "TEMP.order"),
        ("What is the total cost of the car cover and detailing spray I purchased?", "AGG.arith"),
        ("How much more did I spend on accommodations per night in Hawaii compared to Tokyo?", "AGG.arith"),
        ("What is the average age of me, my parents, and my grandparents?", "AGG.arith"),
        ("How many plants did I acquire in the last month?", "AGG.enum"),
        ("How many days did I take social media breaks in total?", "AGG.enum"),
        ("How many tanks do I currently have, including the one I set up for my friend's kid?", "AGG.enum"),
        ("How many bikes do I currently own?", "AGG.latest_count"),
        ("I received a piece of jewelry last Saturday from whom?", "TEMP.anchor"),
        ("What charity event did I participate in a month ago?", "TEMP.anchor"),
        ("How often do I attend yoga classes to help with my anxiety?", "LOOKUP.rate"),
        ("What was my previous occupation?", "LOOKUP.history"),
        ("What company is Rachel, an old colleague from my previous company, currently working at?", "LOOKUP.latest"),
        ("What degree did I graduate with?", "LOOKUP.plain"),
        ("What breed is my dog?", "LOOKUP.plain"),
    ]
    bad = 0
    for q, want in CASES:
        got, fired = classify_shape(q)
        ok = got == want
        bad += not ok
        print(f"{'ok ' if ok else 'FAIL'} {got:18s} (want {want:18s}) {fired} | {q[:70]}")
    # 日期原语测试
    r = resolve_relative_dates("I received a piece of jewelry last Saturday from whom?", "2023/06/02 (Fri) 10:00")
    assert r and r[0][1].strftime("%Y/%m/%d") == "2023/05/27", r
    r2 = resolve_relative_dates("What was the milestone I mentioned four weeks ago?", "2023/06/02 (Fri) 10:00")
    assert r2 and r2[0][1].strftime("%Y/%m/%d") == "2023/05/05", r2
    assert days_between("2023/05/20 (Sat) 02:21", "2023/05/27 (Sat) 02:21") == 7
    print(f"\n{len(CASES)-bad}/{len(CASES)} unit cases pass; date primitives OK")
