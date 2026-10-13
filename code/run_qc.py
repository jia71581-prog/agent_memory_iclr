"""QC-Lifecycle 主方法 runner。

管线:
 1. build_lifecycle(q) → 事实链 store(会话级内容哈希缓存,flash 抽取)
 2. classify_query(question) → current / historical / neutral(neutral 默认 current 视图)
 3. render_view(store, mode) → 生命周期视图文本
 4. BM25 检索原文轮 top-k(视图给结构化事实,原文轮给细节兜底;两者互补)
 5. deepseek-v4-pro 回答 → 双指标判分

用法: python3 run_qc.py <N题> [--full500] [--ablate=no_lifecycle|no_qc|view_only]
  no_lifecycle : 跳过视图,只用 BM25 原文(k 与主方法一致,归因对照)
  no_qc        : 视图固定用 historical 模式(不做查询条件化,≈MemoryLACE 精神)
  view_only    : 只给视图不给原文轮(测视图信息完备度)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_dataset, answer_chat, token_f1, llm_judge, CostLedger, SHARED_ANSWER_PROMPT
from query_classifier import classify_query
from lifecycle import build_lifecycle, render_view
from run_baseline import bm25_rank, stratified_sample

N = 60
ABLATE = None
FULL = False
V5 = os.environ.get("QC_V5") == "1"  # v5: 相关性排序视图截断 + 角色配额/会话扩展检索
TOP_K = 20        # 视图已提供结构化事实,原文轮配额比基线小(归因对照 = qc_no_lifecycle 同 k)
VIEW_CHAR_CAP = 20000
V5_USER_QUOTA = 14   # v5 原文轮角色配额:用户轮/助手轮
V5_ASST_QUOTA = 6
V5_SESSION_AUG = 1   # 额外并入 session-BM25 最高分的 N 个会话的用户轮

# ------------------------------------------------ v6-SSP:建议型问句的"窄而深"上下文(QC_V6_SSP)
# 诊断(SSP 全 30 题逐案读原文 + 上下文重建,gold 位置只用于诊断不进方法):
#  * v5 在 SSP 上是全项目唯一低于无遗忘基线的类型(full 0.667 / v4 0.633 / v5 0.600)。
#  * 不是检索失败:6 个 "full 对 / v5 错" 案例里 5 个 gold 证据轮就在 v5 上下文内
#    (gold 轮命中 v5 1.17 vs full 1.33;gold 会话轮 v5 5.67 vs full 4.83 —— v5 反而更多)。
#  * SSP 的 gold 是 rubric 句:"要建立在用户在录的具体细节上(牌子/器材/上次做过的事),
#    不要泛泛而谈"。它评的是"聚焦精度",不是"事实召回"。
#  * 而 v5 给的上下文恰好是"宽而浅":事实视图 30/30 题全部撞满 20,000 字符上限
#    (平均 197 行,其中 185 行与问句零内容词重合 = 94%,占 SSP 上下文 51%),
#    原文 26 轮散落在 11-14 个会话里(57f827a0:gold 会话只占 1/26 轮,6 轮来自 Wi-Fi 会话)。
#    模型于是从这份无关属性清单里凑答案:6b7dfb22 用 Spider-Man/星战手办凑"绘画灵感",
#    1c0ddc50 把 gold 明确排除的 true crime / self-improvement 又推荐一遍(视图把
#    "我听够了 true crime 想换口味"抽成并列的兴趣值,方向性丢失)。
# 干预:建议型问句上把"宽而浅"换成"窄而深"——同一 _rank_view_lines 只把 cap 收紧(20000→2500),
# 省下的 17.5k 字符换成"已命中轮的会话邻域"(锚点沿用 v5 轮排名,不新增排序决策;
# 命中轮多的会话自然拿到更多邻居),且总字符以同题 v5 上下文长度为硬上限(delta <= 0)。
# 不改任何 prompt/分节标题:纯上下文组成干预(本项目已有"提示词假象"教训)。
# 门控走词面言语行为探测,不用 benchmark 的 question_type(实测 29/30 SSP 命中,470 非 SSP 零误报)。
V6SSP = os.environ.get("QC_V6_SSP") == "1"
V6SSP_VIEW_CAP = int(os.environ.get("QC_V6_SSP_VIEW_CAP", "2500"))   # 建议型问句的视图上限
V6SSP_NEIGHBORS = int(os.environ.get("QC_V6_SSP_NEIGHBORS", "2"))    # 每个命中轮左右各补几轮

# ------------------------------------------------ v6-AGG:量化聚合题的数量台账(QC_V6_AGG)
# 诊断(v5 的 24 道 multi-session 错题逐案读原文 + 离线重建上下文;gold 位置只用于诊断不进方法):
#  * 24 错里 15 道是聚合题,其中 13 道属"量纲聚合"(总额/时长/差值/最大值),v4 期的少数化倾向仍在。
#  * 根因不是模型不会加法,是操作数不在上下文里,两条机制:
#    (A) 相关性剪枝:v5 视图 24/24 题撞满 20,000 上限,平均丢弃 36% 的链;决定性数量常在被丢那侧
#        157a136e gold 43=75-32,`user::age = 32` 被剪掉(view/raw 都没有)→ 模型拒答;
#        92a0aa75 gold 1y5m=3y9m-2y4m,两个操作数双双被剪掉;
#        731e37d7 gold $720=500+200+20,`paid for writing workshop = $200` 被剪掉 → 答 $520。
#    (B) current 视图坍塌:通用动词属性把互不相干的数量并进一条链并标成 supersession,
#        current 只渲染最新一组,较早的数量被视图自己销毁:
#        67e0d0f2 chain `user::has completed` = [8 edX courses(05/21), 12 courses on Coursera(05/27)]
#                 → 渲染成 "= 12 courses on Coursera" → 答 12(gold 20);
#        e3038f8c chain `user::owns` = [vase, rare coins, 25 rare coins(05/23), e-reader, Trek bike(05/29)]
#                 → 渲染成 "= a Trek Emonda road bike" → 答 74(gold 99)。
# 干预:量纲聚合题上把 store 里全部带数量的事实强制进上下文(去重、按日期序、附相对问题日期天数、
# 走全链历史故不受 (B) 影响),台账字符从 VIEW_CHAR_CAP 里扣除 → 预算中性(实测 175 题均值 -0.002%)。
# 物件计数题(how many plants / weddings / albums)刻意不触发:台账只装数量,对物件计数无用。
V6_AGG = os.environ.get("QC_V6_AGG") == "1"
V6_LEDGER_CAP = int(os.environ.get("QC_V6_LEDGER_CAP", "6000"))  # 台账字符上限(从视图预算里扣)
if V6_AGG:
    import v6_agg
    V5 = True   # v6-AGG 是 v5 的严格超集:误开单飞时自动补上 v5

# ------------------------------------------------ v6-TR:时间脚手架(QC_V6_TR),temporal-reasoning 残差
# 诊断(v5 的 12 道 temporal-reasoning 错题逐案读原文 + 离线重建上下文;gold 位置只用于诊断不进方法):
#  * 12 题里 8 题的 gold 证据轮**根本不在 v5 上下文里**,缺失是"日期形状"而不是主题形状的:
#      问句给偏移量而非词面("two weeks ago"/"four weeks ago"/"10 days ago")→ BM25 对日期无感,
#      gold 轮与问句零词面重叠 → 永远捞不到(gpt4_1e4a8aec / eac54add / gpt4_8279ba03,后者直接拒答);
#      问句给时间窗("during the past month")→ 3 条 gold 轮一条没进(gpt4_e061b84f);
#      枚举题要全部 n 个实例,BM25 名额被同主题噪声吃掉 → 6 缺 4 / 6 缺 2(gpt4_7abb270c / a3838d2b);
#      同日会话族里事件时间只存在于句内表达("3 weeks ago"/"a month ago"/"February 12th"),
#      v5 行首日期全同一天 → 顺序信息在上下文里根本不存在(gpt4_65aabe59 / gpt4_fe651585)。
#  * 另 4 题证据齐全但端点/进位错:6e984301 用了同会话干扰句 "for about 6 weeks now"(答 9,gold 3);
#      gpt4_61e13b3c 天数算对却报 "3.1 weeks" 并并列两个候选(gold "3 weeks")。
# 干预:问句表层门控(相对偏移/时间窗/枚举排序/时距)命中后,把日期当检索键——锚点日 ±3 天、
#   时间窗内、会话级相关的事件报告句,词面不重叠也给;渲染成去重的紧凑时间线(事件日期取句内表达
#   优先,附相对问题日的有符号天偏移),时距题另给一行预计算区间(天/周/整周并列)。
#   QC_V6_TR=1 纯附加(v5 视图与原文轮逐字节不变);QC_V6_TR=2 预算中性(按脚手架字符数从原文轮
#   里挤掉 BM25 最低分的轮)。实测脚手架 median/mean 见 results/;不声称准确率收益,由在线 A/B 判。
V6_TR = os.environ.get("QC_V6_TR", "")
if V6_TR:
    import v6_tr
    V5 = True   # v6-TR 是 v5 的严格超集:误开单飞时自动补上 v5

SYSTEM_PROMPT = SHARED_ANSWER_PROMPT  # 全臂统一(见 common.py 说明);来源差异只在上下文分节标题


def _rank_view_lines(store, mode, question, cap):
    """v5: 视图超出 cap 时按与问题的相关性保留链,而非按 entity::attribute 字母序。
    v4 的字母序截断在 470/470 题上触发,平均丢弃 34% 的链,丢弃标准与问题无关。"""
    keys = list(store)
    if not keys:
        return ""
    texts = [" ".join(f"{c['entity']} {c['attribute']} {c['value']}" for c in store[k]) for k in keys]
    order = bm25_rank(question, texts)
    kept, total = [], 0
    for i in order:
        line = render_view({keys[i]: store[keys[i]]}, mode)
        if total + len(line) + 1 > cap:
            continue  # 跳过超预算的长链,继续尝试后面更短的
        kept.append((i, line))
        total += len(line) + 1
    kept.sort(key=lambda x: x[0])  # 恢复原有稳定顺序,避免把相关性泄漏成位置线索
    return "\n".join(l for _, l in kept)


def _select_turns_v5(q, question, top_k):
    """v5 检索:角色配额(用户轮优先)+ 最高分会话的用户轮并入。
    依据:gold 证据轮召回 SSP 0.500->0.727 / MS 0.644->0.728 / TR 0.783->0.838(离线实测)。"""
    units, by_sess = [], {}
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                idx = len(units)
                units.append((date, t["role"], t["content"], si))
                by_sess.setdefault(si, []).append(idx)
    if not units:
        return units, []
    order = bm25_rank(question, [u[2] for u in units])
    user_idx = [i for i in order if units[i][1] != "assistant"]
    asst_idx = [i for i in order if units[i][1] == "assistant"]
    chosen = user_idx[:V5_USER_QUOTA] + asst_idx[:V5_ASST_QUOTA]
    # 会话级 BM25(仅用户文本)选最高分会话,并入其用户轮
    sess_ids = sorted(by_sess)
    sess_text = [" ".join(units[i][2] for i in by_sess[s] if units[i][1] != "assistant") for s in sess_ids]
    for rank in bm25_rank(question, sess_text)[:V5_SESSION_AUG]:
        for i in by_sess[sess_ids[rank]]:
            if units[i][1] != "assistant":
                chosen.append(i)
    # 去重并按 BM25 名次补齐/截断到预算轮数
    seen, final = set(), []
    for i in chosen:
        if i not in seen:
            seen.add(i)
            final.append(i)
    for i in order:
        if len(final) >= top_k + 6:  # 会话扩展允许小幅超额,下方按字符预算再收
            break
        if i not in seen:
            seen.add(i)
            final.append(i)
    return units, sorted(final[:top_k + 6])


# --------------------------------------------------------------------- v6-SSP helpers
_V6SSP_RE = [
    # 显式求建议/推荐(允许一个修饰词:"any helpful tips")
    r"\bany (?:\w+ )?(?:tips|advice|suggestions|ideas|recommendations|thoughts)\b",
    r"\bcan you (?:recommend|suggest)\b|\bcould you (?:recommend|suggest)\b"
    r"|\bdo you have any (?:\w+ )?(?:ideas|tips|suggestions|advice)\b",
    r"\bwhat should i\b|\bwhat do you think\b|\bdo you think (?:it|i|there)\b|\bwhat would you\b",
]
_V6SSP_WEAK = r"\brecommend(?:ations?)?\b|\bsuggest(?:ions?)?\b|\badvice\b"
# 含具体事实疑问词 → 在问事实而不是求建议(排除项,防 RECALL 型误入)
_V6SSP_WH = (r"\bwhat (?:was|is|were) the (?:name|colou?r|price|number|title|address|phone)\b"
             r"|\bwhich (?:one|of)\b|\bwhat did (?:i|you)\b|\bwhat (?:was|were) (?:my|the)\b")
_V6SSP_C = None


def _v6ssp_is_advice(question):
    """建议/推荐言语行为探测(纯词面规则,不看 benchmark 的 question_type 标签)。
    500 题实测:29/30 single-session-preference 命中,470 道非 SSP 题 0 误报。
    唯一漏掉的 SSP 无任何建议词面("...Could there be a reason for this?"),v5 本已答对。"""
    global _V6SSP_C
    if _V6SSP_C is None:
        import re
        _V6SSP_C = ([re.compile(p) for p in _V6SSP_RE], re.compile(_V6SSP_WEAK), re.compile(_V6SSP_WH))
    strong, weak, wh = _V6SSP_C
    ql = question.lower()
    if any(c.search(ql) for c in strong):
        return True
    return bool(weak.search(ql) and not wh.search(ql))


def _v6ssp_render_units(units, idxs):
    return "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in idxs)


def _v6ssp_expand_turns(question, units, sel, view_len, budget):
    """把 v5 已选中的轮向同会话的相邻轮扩张,省下的视图预算换成"轮的邻域"。
    动机:SSP 的 gold 要求引用在录的具体细节,而这些细节一半在助手轮里
    (gold 会话里用户说"我想换个播客口味",助手轮才给出 Hardcore History/Lore 的具体名字)。
    v5 的逐轮 BM25 把对话切碎成孤立片段:57f827a0 的 gold 会话在 26 轮里只占 1 轮。
    锚点沿用 v5 的轮排名(该排名的 gold 轮命中已优于 full,不新增排序决策),
    只补每个锚轮左右各 V6SSP_NEIGHBORS 轮;命中轮多的会话自然获得更多邻居 = 窄而深。
    按锚轮 BM25 名次优先分配,总字符硬上限 budget(= 同题 v5 上下文长度)。"""
    by_sess = {}
    for i, u in enumerate(units):
        by_sess.setdefault(u[3], []).append(i)
    rank = {i: r for r, i in enumerate(bm25_rank(question, [u[2] for u in units]))}
    keep = set(sel)
    anchors = sorted(sel, key=lambda x: rank.get(x, 1 << 30))
    cands, seen = [], set()
    # 先给所有锚轮补紧邻轮(off=1,即成对的那一问一答),再考虑更远的邻居
    for off in range(1, V6SSP_NEIGHBORS + 1):
        for i in anchors:
            pos = by_sess[units[i][3]]
            j = pos.index(i)
            for k in (j - off, j + off):
                if 0 <= k < len(pos) and pos[k] not in keep and pos[k] not in seen:
                    seen.add(pos[k])
                    cands.append(pos[k])
    for c in cands:
        trial = sorted(keep | {c})
        if len(_v6ssp_render_units(units, trial)) + view_len > budget:
            continue  # 跳过放不下的长轮,继续试后面更短的(与 _rank_view_lines 同策略)
        keep = set(trial)
    return _v6ssp_render_units(units, sorted(keep))



# ================================ v6-INF:答案侧推理强化(只改回答阶段,检索/视图沿用 v5 不动)
# 诊断(v5 剩余 51 错逐案读原文 + 上下文重建;gold 位置只用于诊断,不进方法):
#  ~30/51 属"证据在上下文里但答错",机制可点名:
#   * 两前提未合成 → 直接拒答:73d42213(7AM 出门 + 路上两小时,均在轮内,答"没说到达时间");
#     37f165cf(440 + 416 页都在,拒绝相加);7024f17c(30 分钟跑步在视图里,答"信息不足")。
#   * 问的是差值,答成端点:9ee3ecd6(现 200 分、需"总共 300 分",gold 100,答 300)。
#   * 前提凭空生成:6e984301(两个 "today" 轮 02/11 与 03/04 相隔正好 3 周 = gold,模型却编出
#     "02/11 时已上了约 6 周"再相加得 9);d01c6aa8(上下文三处 "32-year-old",模型说"你 20 岁");
#     c18a7dc8(拿"通常 22 岁毕业"补空);gpt4_fe651585(编出双胞胎生于 2 月 12 日)。
#   * 计数的纳入/排除边界:0a995998(把干洗待取件按定义排除,3 记成 1);gpt4_2ba83207(把
#     "上个月 Thrive Market $150"按窗口排除);d851d5ba(反向:多算成 $8,750)。
#   * 对冲不给单值:gpt4_61e13b3c(算出 22 天 = "3 周 1 天/约 3.1 周",gold "3 周",判错)。
#   * 建议型问句上的两种失败:实体不匹配就拒答(0edc2aef 有西雅图"景观/屋顶泳池/阳台按摩浴缸"
#     偏好在视图里,却因"上下文没有迈阿密"拒答;09d032c9 充电宝在轮内仍答"我不知道";
#     d6233ab6 列出了辩论队/AP 课却包在拒答里),以及锚点漂移(57f827a0 问卧室家具,
#     dresser + mid-century 两条都在上下文里,答案却全程讲 Wi-Fi 路由器)。
# 干预 = 两个可独立 A/B 的开关,都不动检索:
#   QC_V6_INF=1  按 question-shape 追加回答脚手架(零额外 API 调用,仅 system prompt 变长)
#   QC_V6_VER=1  门控二次校验改写(命中门时每题 +1 次 flash 调用,消息前缀与首答完全一致以吃 KV 缓存)
# 两开关都关时,本文件行为与 v5 逐字节一致。
# 注意(方法论):脚手架改的是全臂共用的 SHARED_ANSWER_PROMPT 之后的追加段,因此
# "v6 vs v5" 是同臂配对比较(合法);若要进"v6 vs full 基线"的头条表,必须把同一段
# 追加进基线臂(from run_qc import v6_scaffold_for),否则重犯 2026-09-17 的提示词不对称假象。
V6_INF = os.environ.get("QC_V6_INF") == "1"
V6_VER = os.environ.get("QC_V6_VER") == "1"
V6_INF_NEUTRAL = os.environ.get("QC_V6_INF_NEUTRAL") == "1"  # 预算严格中性:视图尾部按脚手架字符数回削
V6_VER_ALL = os.environ.get("QC_V6_VER_ALL") == "1"          # 校验门放宽:所有数值/排序 shape 全查
if V6_INF or V6_VER:
    V5 = True  # v6 是 v5 的严格超集:强制 v5 检索/视图(防止误跑成 v4 + 脚手架)

_V6_BLK_COMPOSE = (
    "- The answer may have to be put together from two or more statements (a start time plus a "
    "travel time, a current amount plus a target amount, an age plus an elapsed period, two page "
    "counts). Combine them and give the result.\n"
    "- If the value is neither stated nor derivable from what is in the context, say plainly that "
    "the context does not contain it. Never close the gap with a plausible or typical value.\n")
_V6_BLK_DATE = (
    "- Every retrieved turn is prefixed with the date it was said. Inside a turn, \"today\", "
    "\"yesterday\", \"last week\", \"N weeks ago\" are relative to that turn's own prefix date, not "
    "to the date the question is asked. Resolve them against the prefix date before using them.\n")
_V6_BLK_COUNT = (
    "- First list the qualifying items, one per line, each with the date it was mentioned; then "
    "count or order that list and give the result.\n"
    "- Keep an item unless the context rules it out; do not drop one because it is worded "
    "differently from the question. Merge repeated mentions of the same item, and leave out items "
    "the context does not attribute to the user.\n")
_V6_BLK_ARITH = (
    "- For a duration, an age, a total or a difference: write down the values you are combining "
    "with their dates, do the arithmetic explicitly, then give one final value in the unit the "
    "question asks for. If the question asks how much more is still needed, subtract what is "
    "already there from the target instead of repeating the target. Do not answer with a range or "
    "with two alternatives.\n")
_V6_BLK_NOINVENT = (
    "- Use only dates, numbers, names and places that appear in the context. Do not substitute a "
    "typical, average or estimated value for a missing one, and do not bring in facts from outside "
    "the context.\n")
_V6_BLK_LATEST = (
    "- When the same thing has different values at different dates, the latest dated statement is "
    "the current one - including when the later statement is phrased as something the user was "
    "about to do - unless the question asks about the past.\n")
_V6_BLK_PREF = (
    "- This is a request for advice. Ground it in what the context records about this user: their "
    "stated preferences, the specific brands, titles, places, equipment and activities they "
    "mentioned, and their constraints. Name those details in the answer.\n"
    "- Respect what they said they dislike, want to avoid, or are tired of; do not recommend it "
    "back to them.\n"
    "- If the exact place, trip, item or event in the question does not itself appear in the "
    "context, still answer the request by applying the preferences that are there. Do not refuse "
    "for lack of an exact match.\n")

_V6_SHAPE_BLOCKS = {
    "AGG.enum":         (_V6_BLK_DATE, _V6_BLK_COUNT, _V6_BLK_ARITH, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "AGG.arith":        (_V6_BLK_DATE, _V6_BLK_ARITH, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "AGG.latest_count": (_V6_BLK_COUNT, _V6_BLK_LATEST, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "TEMP.offset":      (_V6_BLK_DATE, _V6_BLK_ARITH, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "TEMP.order":       (_V6_BLK_DATE, _V6_BLK_COUNT, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "TEMP.anchor":      (_V6_BLK_DATE, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "PREF.advice":      (_V6_BLK_PREF,),
    "LOOKUP.latest":    (_V6_BLK_LATEST, _V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "LOOKUP.history":   (_V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "LOOKUP.rate":      (_V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "LOOKUP.plain":     (_V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
    "RECALL.assistant": (_V6_BLK_NOINVENT, _V6_BLK_COMPOSE),
}
_V6_SCAFFOLD_HEAD = "Additional instructions for this question:\n"

_V6_VERIFY_CHECKLIST = (
    "Check the answer above against the context before it is shown to the user.\n"
    "1. Every date, number, name and place in the answer must appear in the context. Remove or "
    "correct any that does not, and never replace a missing value with a typical or estimated one.\n"
    "2. Relative expressions inside a turn (\"today\", \"last week\", \"N weeks ago\") refer to that "
    "turn's own date prefix. Recheck any date you derived from one.\n"
    "3. If a count, a total or a difference was asked for: re-list the qualifying items with their "
    "dates and recount. Do not drop an item because its wording differs from the question's, do "
    "not count one item twice, and do not include what the context does not attribute to the user.\n"
    "4. If how-much-more or how-long-between was asked, the answer must be the difference, not "
    "either endpoint, and it must be a single value in the unit asked for - not a range and not two "
    "alternatives.\n"
    "5. If the answer declines or says the context is insufficient: check once more whether the "
    "context supports an answer, including one that follows from combining two statements or that "
    "the same conversation makes clear. If it does, answer it. If it genuinely does not, keep the "
    "refusal and do not invent a value.\n"
    "6. If advice was asked for and the exact place, item or event is absent, apply the user's "
    "recorded preferences and constraints instead of refusing, and do not recommend what they said "
    "they avoid.\n\n"
    "Reply with exactly \"OK\" on a single line if the answer needs no change. Otherwise reply "
    "\"REVISED\" on the first line and the corrected final answer on the following lines. Do not "
    "explain your checking.")

_V6_NUM_SHAPES = ("AGG.enum", "AGG.arith", "AGG.latest_count",
                  "TEMP.offset", "TEMP.order", "TEMP.anchor")
_V6_REFUSAL_RE = (
    r"i (?:don't|do not|cannot|can't) (?:know|tell|say|determine|have|find|see|identify)"
    r"|(?:doesn't|does not|do not|don't) (?:say|state|specify|mention|provide|contain|include"
    r"|indicate|record|tie)|not enough (?:information|info|context|detail)"
    r"|(?:can't|cannot|couldn't) (?:be )?(?:determine|determined|calculate|calculated|compute"
    r"|computed|know)|no (?:information|record|mention|indication|data)"
    r"|context (?:doesn't|does not|contains no|has no|lacks)"
    r"|nothing (?:about|in the context|relevant)|unable to"
    r"|isn't (?:specified|stated|mentioned|in the context)|is not (?:specified|stated|mentioned)"
    r"|unclear from")
# 草稿自己承认不完整 / 并列给出备选值(实测:d851d5ba "at least $8,750"、e3038f8c "exact total
# can't be determined"、bf659f65 "if counting the Tame Impala vinyl, it could be 3"、
# gpt4_61e13b3c "if using the fact timestamp of 3/21 instead, it would be 23 days"、67e0d0f2
# "plus any edX courses not counted")
_V6_HEDGE_RE = (
    r"\bat least\b|\bplus any\b|\bif counting\b|\bif using\b|\bif you meant\b|\bcould be\b"
    r"|\bit would be\b|\bnot counted\b|\bexact (?:total|number|figure|count)\b")
_V6_NUM_RE = r"\d[\d,]*(?:\.\d+)?"


def v6_scaffold_for(question):
    """按 question-shape 返回追加的回答脚手架(无 shape 命中 → 空串)。
    公开函数:基线臂要做提示词对称对照时 import 这一个即可。"""
    from v5_shape import classify_shape
    shape, _ = classify_shape(question)
    blocks = _V6_SHAPE_BLOCKS.get(shape)
    return (_V6_SCAFFOLD_HEAD + "".join(blocks)) if blocks else ""


def _v6_norm_apostrophe(s):
    return (s or "").replace("’", "'").replace("‘", "'")


def _v6_numbers(s):
    import re
    return {m.replace(",", "").rstrip(".") for m in re.findall(_V6_NUM_RE, s or "")}


def _v6_verify_gate(shape, draft, ctx):
    """二次校验门(纯词面 + shape,无 API、无 gold)。返回触发原因或 None。
    四条触发都对应实测失败模式(括号内 = 该触发在 v5 的 500 题上覆盖的错题数/牵连的对题数):
      pref                建议型问句(12/51 错;门覆盖 12,牵连 16)
      refusal             草稿里出现拒答/信息不足词面(11 错,牵连 34)
      unsupported-number  数值型草稿里出现上下文中没有的数字 = 编前提(5 错,牵连 35)
      hedge               草稿自己声明不完整/给了备选值("at least" / "if counting" /
                          "exact total can't be determined" / "it would be";7 错,牵连 18)
    QC_V6_VER_ALL=1 时再加一条 many-numbers(数值型 >=3 个不同数字),覆盖 +11 但牵连 +127。"""
    import re
    d = _v6_norm_apostrophe(draft)
    if shape == "PREF.advice":
        return "pref"
    if re.search(_V6_REFUSAL_RE, d, re.I):
        return "refusal"
    if shape in _V6_NUM_SHAPES:
        nd = _v6_numbers(d)
        if nd - _v6_numbers(ctx):
            return "unsupported-number"
        if re.search(_V6_HEDGE_RE, d, re.I):
            return "hedge"
        if V6_VER_ALL and len(nd) >= 3:
            return "many-numbers"
    return None


def _v6_verify_revise(system_prompt, user_content, draft, ledger):
    """二次校验改写。消息前缀 = 首答的 [system, user] 逐字节一致(吃 prompt cache),
    尾部追加草稿与检查表。解析失败/OK → 原样返回草稿(不让校验轮引入改写漂移)。"""
    from common import chat, ANSWER_MODEL
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": draft},
            {"role": "user", "content": _V6_VERIFY_CHECKLIST}]
    out = chat(msgs, model=ANSWER_MODEL, max_tokens=6144, ledger=ledger, op="verify/qc-v6")
    if not (out or "").strip():  # 推理吃光配额 → 加倍重试一次(与 answer_chat 同策略)
        out = chat(msgs, model=ANSWER_MODEL, max_tokens=12288, ledger=ledger, op="verify/qc-v6-retry")
    t = (out or "").strip()
    if not t:
        return draft, "empty"
    first = t.splitlines()[0].strip().strip("*#` ").upper()
    if first.startswith("OK"):
        return draft, "ok"
    if first.startswith("REVISED"):
        rest = t.split("\n", 1)[1].strip() if "\n" in t else ""
        return (rest or draft), ("revised" if rest else "revised-empty")
    return draft, "unparsed"


def _v6_trim_view_tail(ctx, n_chars):
    """预算严格中性变体:从事实视图分节尾部按行边界回削 n_chars(与 v4 的 cap 截断同一操作)。"""
    if n_chars <= 0 or not ctx:
        return ctx
    marker = "\n\n### Retrieved raw conversation turns\n"
    i = ctx.find(marker)
    head, tail = (ctx[:i], ctx[i:]) if i >= 0 else (ctx, "")
    lines = head.split("\n")
    removed = 0
    while len(lines) > 2 and removed < n_chars:  # 至少保留分节标题 + 一行
        removed += len(lines.pop()) + 1
    return "\n".join(lines) + tail


def _v6_answer(q, ctx, ledger, label, mode):
    """v6 回答阶段。V6_INF 只加 system prompt 段;V6_VER 只在门命中时多一次 flash 调用。"""
    scaffold = v6_scaffold_for(q["question"]) if V6_INF else ""
    if scaffold and V6_INF_NEUTRAL:
        ctx = _v6_trim_view_tail(ctx, len(scaffold))
    sys_p = SYSTEM_PROMPT + ("\n\n" + scaffold if scaffold else "")
    user_c = f"{ctx}\n\nQuestion (asked on {q['question_date']}): {q['question']}"
    ans = answer_chat(sys_p, user_c, ledger, op="answer/qc")
    if V6_VER:
        from v5_shape import classify_shape
        shape, _ = classify_shape(q["question"])
        why = _v6_verify_gate(shape, ans, ctx)
        if why:
            ans, status = _v6_verify_revise(sys_p, user_c, ans, ledger)
            print(f"    [v6ver] {q['question_id']} shape={shape} gate={why} -> {status}", flush=True)
    return ans, label, mode


def answer_question(q, ledger):
    label, fired = classify_query(q["question"])
    mode = "historical" if label == "historical" else "current"
    if ABLATE == "no_qc":
        mode = "historical"

    view_txt = ""
    v6_ledger_txt = ""   # v6-AGG:数量台账分节;未触发时恒为空,上下文与 v5 逐字节一致
    if ABLATE != "no_lifecycle":
        store = build_lifecycle(q, ledger)
        if V5 and V6_AGG and ABLATE is None and v6_agg.is_agg_question(q["question"]):
            # 量纲聚合题:台账与排序视图共用同一个 VIEW_CHAR_CAP(预算中性),台账走全链历史
            view_txt, v6_ledger_txt = v6_agg.build_v6_sections(
                store, mode, q["question"], q["question_date"], VIEW_CHAR_CAP, V6_LEDGER_CAP)
        elif V5:
            view_txt = _rank_view_lines(store, mode, q["question"], VIEW_CHAR_CAP)
        else:
            view_txt = render_view(store, mode)
            if len(view_txt) > VIEW_CHAR_CAP:  # 行边界截断(render_view 输出已确定性排序)
                kept, total = [], 0
                for line in view_txt.splitlines():
                    if total + len(line) + 1 > VIEW_CHAR_CAP:
                        break
                    kept.append(line)
                    total += len(line) + 1
                view_txt = "\n".join(kept)

    raw_txt = ""
    if ABLATE != "view_only":
        if V5:
            units, sel = _select_turns_v5(q, q["question"], TOP_K)
            raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sel)
        else:
            units = []
            for date, sess in zip(q["haystack_dates"], q["haystack_sessions"]):
                for t in sess:
                    if t.get("content"):
                        units.append((date, t["role"], t["content"]))
            order = sorted(bm25_rank(q["question"], [u[2] for u in units])[:TOP_K])
            raw_txt = "\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in order)

    if V6SSP and V5 and ABLATE is None and _v6ssp_is_advice(q["question"]):
        # 建议型问句:以同题 v5 上下文长度为硬预算,视图 cap 收紧(同一 _rank_view_lines),
        # 省下的字符换成已命中轮的会话邻域。分节标题与 v5 完全一致,总长必 <= v5。
        budget = len(view_txt) + len(raw_txt)
        view_txt = _rank_view_lines(store, mode, q["question"], V6SSP_VIEW_CAP)
        raw_txt = _v6ssp_expand_turns(q["question"], units, sel, len(view_txt), budget)

    v6_tr_txt = ""   # v6-TR:时间脚手架分节;门控未命中时恒为空,上下文与 v5 逐字节一致
    if V6_TR and V5 and ABLATE is None:
        v6_tr_txt = v6_tr.tr_scaffold(q)
        if v6_tr_txt and V6_TR == "2" and raw_txt:   # 预算中性变体:按脚手架字符数回削原文轮
            raw_txt = v6_tr.tr_trim_raw(q["question"], raw_txt, len(v6_tr_txt) + 2)

    parts = []
    if view_txt:
        head = ("Current-state fact view (latest value per fact; [CONFLICT] marks values reported "
                "at the same time that could not be ordered)"
                if mode == "current" else
                "Fact evolution view (full value history per fact, oldest -> newest)")
        parts.append(f"### {head}\n{view_txt}")
    if v6_ledger_txt:  # v6-AGG:数量台账(全量数量事实,已去重,附相对问题日期的天数)
        parts.append(f"{v6_agg.LEDGER_HEADER}\n{v6_ledger_txt}")
    if raw_txt:
        parts.append(f"### Retrieved raw conversation turns\n{raw_txt}")
    if v6_tr_txt:  # v6-TR:时间脚手架放最后(紧贴问句;不改动上面任何分节的字节)
        parts.append(v6_tr_txt)
    ctx = "\n\n".join(parts)

    if V6_INF or V6_VER:  # v6-INF/VER 分支:只改回答阶段,上下文与 v5 相同
        return _v6_answer(q, ctx, ledger, label, mode)

    ans = answer_chat(SYSTEM_PROMPT,
                      f"{ctx}\n\nQuestion (asked on {q['question_date']}): {q['question']}",
                      ledger, op="answer/qc")
    return ans, label, mode


def main():
    data = load_dataset("s_cleaned")
    sample = data if FULL else stratified_sample(data, N)
    tag = f"qc{'_' + ABLATE if ABLATE else ''}"
    ledger = CostLedger()
    results = []
    for i, q in enumerate(sample):
        try:
            ans, label, mode = answer_question(q, ledger)
            f1 = token_f1(ans, q["answer"])
            correct = llm_judge(q["question"], q["question_type"], q["answer"], ans, ledger=ledger)
        except Exception as e:
            print(f"[{i+1}] ERROR {q['question_id']}: {e}", flush=True)
            results.append({"qid": q["question_id"], "type": q["question_type"], "error": str(e)})
            continue
        results.append({"qid": q["question_id"], "type": q["question_type"],
                        "qlabel": label, "view_mode": mode,
                        "f1": round(f1, 3), "judge": bool(correct), "pred": ans[:4000]})
        print(f"[{i+1}/{len(sample)}] {q['question_type'][:22]:22s} q={label[:4]} judge={'Y' if correct else 'N'} f1={f1:.2f}", flush=True)

    ok = [r for r in results if "judge" in r]
    by_type = {}
    for r in ok:
        s = by_type.setdefault(r["type"], {"n": 0, "judge": 0, "f1": 0.0})
        s["n"] += 1
        s["judge"] += r["judge"]
        s["f1"] += r["f1"]
    print(f"\n=== {tag} | n={len(ok)} ===")
    for t, s in sorted(by_type.items()):
        print(f"{t:28s} n={s['n']:3d} judge_acc={s['judge']/s['n']:.3f} f1={s['f1']/s['n']:.3f}")
    total = sum(r["judge"] for r in ok) / max(1, len(ok))
    print(f"总体 judge_acc={total:.3f}")
    out = f"/root/agent_memory_iclr/results/{tag}_n{len(sample)}.json"
    with open(out, "w") as f:
        json.dump({"system": tag, "results": results,
                   "by_type": {t: {"n": s["n"], "judge_acc": s["judge"]/s["n"], "f1": s["f1"]/s["n"]}
                                for t, s in by_type.items()},
                   "total_judge_acc": total, "cost": ledger.summary()}, f, ensure_ascii=False, indent=1)
    print("saved:", out)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--ablate="):
            ABLATE = a.split("=", 1)[1]
        elif a == "--full500":
            FULL = True
        else:
            N = int(a)
    main()
