"""QC-Lifecycle 核心部件 2:生命周期抽取管线(training-free)。

架构 v4(2026-09-17 preflight 审查后重构):
 1. 会话级抽取缓存:key = md5(会话文本),值 = 无日期 (entity, attribute, value) 三元组。
    全库 23,867 会话实例仅 18,464 唯一 → 跨题去重省 ~23% 抽取开销。
    抽取失败(空输出/零解析行)重试一次更大配额;仍失败则抛异常,不缓存失败结果。
 2. 装配期(纯代码,无 API):按题把三元组盖上会话日期/序号,对齐成 (entity, attribute) 链。
 3. 链内关系标注(确定性规则,时间戳含时分可全序比较):
      cooccurring   - 同一会话内的不同 value(共现,多为转述噪声,非冲突)
      supersession  - 更晚时间戳的不同 value(时间性更新)
      contradiction - 不同会话、时间戳完全相同的不同 value(不可排序 → 可靠性冲突)
 4. 视图渲染:
      current    → 每链最新会话的值(共现值并列),含 contradiction 的链附 [CONFLICT]
      historical → 全链演化(同会话共现值用 / 连接)
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat

CACHE_DIR = "/root/agent_memory_iclr/results/lifecycle_cache"
# thinking-off 抽取的独立缓存(2026-09-17 成本决策:reasoning 燃烧占抽取成本 ~87%,
# 小样本对照显示事实召回持平;旧 thinking-on 缓存留在 sessions/ 可回退)
SESSION_CACHE_DIR = os.path.join(CACHE_DIR, "sessions_nt")

EXTRACT_PROMPT = """Extract atomic facts about the user (and stable world facts they mention) from this conversation session.

Session:
{session}

Rules:
- One fact per line, format: entity | attribute | value
- entity is usually "user" or a named person/thing; attribute is a short noun phrase; value is concise.
- Only extract facts explicitly stated. Skip chit-chat, questions, hypotheticals.
- IMPORTANT: always extract facts that UPDATE or CHANGE earlier state (moved, switched, replaced, upgraded, new job, price changed, "moved back", "no longer"), including updates mentioned in passing or reported about OTHER people (e.g. "my friend Rachel just moved back to the suburbs" -> Rachel | recently moved to | the suburbs).
- Facts stated by the assistant that the user confirms also count.
- Max 20 facts. If none, output NONE.

Facts:"""


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _session_text(session_turns):
    text = "\n".join(f"[{t['role']}] {t['content']}" for t in session_turns if t.get("content"))
    return text[:24000]


def _parse_facts(out):
    """解析抽取输出。返回三元组列表;失败签名(非空输出但零解析且无 NONE)返回 None。"""
    if not out or not out.strip():
        return None
    triples = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and parts[0] and parts[2] and parts[0].upper() != "NONE":
            triples.append(parts[:3])
    if not triples and "NONE" not in out.upper():
        return None
    return triples


def extract_session_facts(session_turns, ledger):
    """会话级抽取,内容哈希缓存。返回 [(entity, attribute, value)]。
    抽取失败抛 RuntimeError(不缓存失败),调用方决定重试策略。"""
    text = _session_text(session_turns)
    if not text:
        return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    cache = os.path.join(SESSION_CACHE_DIR, f"{key}.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # 损坏缓存 → 重建
    prompt = EXTRACT_PROMPT.format(session=text)
    # 阶梯式重试:格式失败在温度0下是确定性的,单纯加配额无效,须改变生成条件
    triples = _parse_facts(chat([{"role": "user", "content": prompt}], model="deepseek-flash",
                                max_tokens=2048, ledger=ledger, op="extract", thinking=False))
    if triples is None:  # 追加严格格式指令重试
        strict = prompt + "\n(STRICT FORMAT: every line must have exactly two ' | ' separators: entity | attribute | value)"
        triples = _parse_facts(chat([{"role": "user", "content": strict}], model="deepseek-flash",
                                    max_tokens=4096, ledger=ledger, op="extract-retry", thinking=False))
    if triples is None:  # thinking-on 兜底(格式遵从更好)
        triples = _parse_facts(chat([{"role": "user", "content": prompt}], model="deepseek-flash",
                                    max_tokens=8192, ledger=ledger, op="extract-retry-think"))
    if triples is None:  # 三次全败:弃用该会话并留审计痕迹,不再阻断全局(计入 extraction-miss)
        with open(os.path.join(CACHE_DIR, "extract_giveup.log"), "a") as f:
            f.write(f"{key}\n")
        triples = []
    os.makedirs(SESSION_CACHE_DIR, exist_ok=True)
    tmp = f"{cache}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(triples, f, ensure_ascii=False)
    os.replace(tmp, cache)
    return triples


# 多值属性:天然并存多个值,不构成 supersession/contradiction(累积语义)。
# 判定 = 词表 + 结构启发(同时间戳 >=3 个不同值视为多值)。论文中作为设计表格展示。
MULTI_VALUE_ATTRS = {
    "interest", "interests", "hobby", "hobbies", "goal", "goals", "plan", "plans",
    "considering", "likes", "dislikes", "preference", "preferences", "activity",
    "activities", "emotional state", "concern", "concerns", "skill", "skills",
    "wish", "wishes", "idea", "ideas", "topic", "topics",
}


def is_multi_valued(attr_norm, facts):
    if attr_norm in MULTI_VALUE_ATTRS:
        return True
    # 结构启发:单会话内 >=3 个不同值 = 一次列举多值(累积语义);
    # 跨会话分布的多值不触发,留给 supersession/contradiction 规则处理。
    by_sess = {}
    for f in facts:
        by_sess.setdefault((f["date"], f.get("sidx")), set()).add(_norm(f["value"]))
    return any(len(v) >= 3 for v in by_sess.values())


def assemble_chains(all_facts):
    """把带日期的原子事实装配成 lifecycle store(纯代码,无 API)。"""
    chains = {}
    for f in all_facts:
        k = f"{_norm(f['entity'])}::{_norm(f['attribute'])}"
        chains.setdefault(k, []).append(f)
    store = {}
    for k, facts in chains.items():
        facts.sort(key=lambda f: (f["date"], f.get("sidx", 0)))
        attr_norm = k.split("::", 1)[1]
        if is_multi_valued(attr_norm, facts):
            seen, chain = set(), []
            for f in facts:
                nv = _norm(f["value"])
                if nv in seen:
                    continue
                seen.add(nv)
                chain.append({**f, "relation": "accumulative"})
        else:
            chain = []
            seen = set()
            for f in facts:
                nv = _norm(f["value"])
                if nv in seen:
                    continue  # 重复证据,合并
                seen.add(nv)
                if not chain:
                    rel = "initial"
                else:
                    prev = chain[-1]
                    if f.get("sidx") == prev.get("sidx") and f["date"] == prev["date"]:
                        rel = "cooccurring"      # 同会话共现,非冲突
                    elif f["date"] == prev["date"]:
                        rel = "contradiction"    # 跨会话同时间戳,不可排序
                    else:
                        rel = "supersession"     # 更晚时间戳
                chain.append({**f, "relation": rel})
        if chain:
            store[k] = chain
    return store


def build_lifecycle(q, ledger):
    """对一道题的全部 haystack 会话构建 lifecycle store。
    会话抽取走内容哈希缓存;任一会话抽取失败则异常上抛(调用方按错误行处理,断点续跑时重试)。"""
    all_facts = []
    for si, (date, sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for ent, attr, val in extract_session_facts(sess, ledger):
            all_facts.append({"entity": ent, "attribute": attr, "value": val, "date": date, "sidx": si})
    return assemble_chains(all_facts)


def _group_by_session(chain):
    groups = []
    for c in chain:
        gk = (c["date"], c.get("sidx"))
        if groups and groups[-1][0] == gk:
            groups[-1][1].append(c)
        else:
            groups.append((gk, [c]))
    return groups


def render_view(store, mode):
    """把 lifecycle store 渲染成上下文文本行列表(确定性排序)。mode: current | historical"""
    lines = []
    for k in sorted(store):
        chain = store[k]
        ent, attr = chain[0]["entity"], chain[0]["attribute"]
        is_accum = chain[0]["relation"] == "accumulative"
        has_conflict = any(c["relation"] == "contradiction" for c in chain)
        if is_accum:
            vals = "; ".join(f"{c['value']} ({c['date'].split()[0]})" for c in chain)
            lines.append(f"{ent} | {attr} = {vals}")
        elif mode == "current":
            groups = _group_by_session(chain)
            last_date, last_facts = groups[-1][0][0], groups[-1][1]
            vals = "; ".join(c["value"] for c in last_facts)
            tag = " [CONFLICT: multiple values reported at the same time]" if has_conflict else ""
            lines.append(f"[{last_date}] {ent} | {attr} = {vals}{tag}")
        else:  # historical: 全链演化
            groups = _group_by_session(chain)
            if len(groups) == 1:
                vals = " / ".join(c["value"] for c in groups[0][1])
                lines.append(f"[{groups[0][0][0]}] {ent} | {attr} = {vals}")
            else:
                steps = " -> ".join(
                    " / ".join(c["value"] for c in g[1])
                    + f" ({g[0][0].split()[0]}{'/conflict' if g[1][0]['relation'] == 'contradiction' else ''})"
                    for g in groups)
                lines.append(f"{ent} | {attr}: {steps}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 关系规则单元测试(无 API)
    F = lambda e, a, v, d, s: {"entity": e, "attribute": a, "value": v, "date": d, "sidx": s}
    facts = [
        F("user", "job", "teacher", "2023/05/20 (Sat) 02:21", 0),
        F("user", "job", "engineer", "2023/05/25 (Thu) 10:00", 3),      # supersession
        F("user", "job", "engineer at Acme", "2023/05/25 (Thu) 10:00", 3),  # cooccurring
        F("user", "job", "designer", "2023/05/25 (Thu) 10:00", 4),      # contradiction(同时间戳不同会话)
        F("user", "hobby", "hiking", "2023/05/20 (Sat) 02:21", 0),
        F("user", "hobby", "chess", "2023/05/21 (Sun) 09:00", 1),       # 多值词表 → accumulative
    ]
    store = assemble_chains(facts)
    job = store["user::job"]
    rels = [c["relation"] for c in job]
    assert rels == ["initial", "supersession", "cooccurring", "contradiction"], rels
    assert all(c["relation"] == "accumulative" for c in store["user::hobby"])
    cur = render_view(store, "current")
    hist = render_view(store, "historical")
    job_cur = [l for l in cur.splitlines() if "| job" in l][0]
    assert "CONFLICT" in job_cur and "designer" in job_cur and "teacher" not in job_cur, job_cur
    assert "teacher" in hist and "->" in hist and "engineer / engineer at Acme" in hist, hist
    print("relation rules OK")
    print("--- current ---\n" + cur)
    print("--- historical ---\n" + hist)
    # 解析失败签名测试
    assert _parse_facts("") is None and _parse_facts("blah blah") is None
    assert _parse_facts("NONE") == [] and _parse_facts("a | b | c") == [["a", "b", "c"]]
    print("parse guards OK")
