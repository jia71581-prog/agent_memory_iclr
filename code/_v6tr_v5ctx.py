"""Offline reproduction of the v5 context for a question. Cache-only: never calls the API."""
import hashlib, json, os, sys
sys.path.insert(0,'/root/agent_memory_iclr/code')
import lifecycle as L
from run_baseline import bm25_rank

def facts_cached(sess):
    text = L._session_text(sess)
    if not text: return []
    key = hashlib.md5(text.encode()).hexdigest()[:16]
    p = os.path.join(L.SESSION_CACHE_DIR, f"{key}.json")
    if not os.path.exists(p):
        raise RuntimeError("CACHE MISS "+key)
    with open(p) as f: return json.load(f)

def store_of(q):
    all_facts=[]
    for si,(date,sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for ent,attr,val in facts_cached(sess):
            all_facts.append({"entity":ent,"attribute":attr,"value":val,"date":date,"sidx":si})
    return L.assemble_chains(all_facts)

VIEW_CHAR_CAP=20000; TOP_K=20; V5_USER_QUOTA=14; V5_ASST_QUOTA=6; V5_SESSION_AUG=1

def rank_view_lines(store, mode, question, cap=VIEW_CHAR_CAP):
    keys=list(store)
    if not keys: return ""
    texts=[" ".join(f"{c['entity']} {c['attribute']} {c['value']}" for c in store[k]) for k in keys]
    order=bm25_rank(question, texts)
    kept,total=[],0
    for i in order:
        line=L.render_view({keys[i]:store[keys[i]]}, mode)
        if total+len(line)+1>cap: continue
        kept.append((i,line)); total+=len(line)+1
    kept.sort(key=lambda x:x[0])
    return "\n".join(l for _,l in kept)

def select_turns_v5(q, question, top_k=TOP_K):
    units,by_sess=[],{}
    for si,(date,sess) in enumerate(zip(q["haystack_dates"], q["haystack_sessions"])):
        for t in sess:
            if t.get("content"):
                idx=len(units); units.append((date,t["role"],t["content"],si)); by_sess.setdefault(si,[]).append(idx)
    if not units: return units,[]
    order=bm25_rank(question,[u[2] for u in units])
    user_idx=[i for i in order if units[i][1]!="assistant"]
    asst_idx=[i for i in order if units[i][1]=="assistant"]
    chosen=user_idx[:V5_USER_QUOTA]+asst_idx[:V5_ASST_QUOTA]
    sess_ids=sorted(by_sess)
    sess_text=[" ".join(units[i][2] for i in by_sess[s] if units[i][1]!="assistant") for s in sess_ids]
    for rank in bm25_rank(question,sess_text)[:V5_SESSION_AUG]:
        for i in by_sess[sess_ids[rank]]:
            if units[i][1]!="assistant": chosen.append(i)
    seen,final=set(),[]
    for i in chosen:
        if i not in seen: seen.add(i); final.append(i)
    for i in order:
        if len(final)>=top_k+6: break
        if i not in seen: seen.add(i); final.append(i)
    return units, sorted(final[:top_k+6])

def v5_context(q):
    from query_classifier import classify_query
    label,fired=classify_query(q["question"])
    mode="historical" if label=="historical" else "current"
    store=store_of(q)
    view=rank_view_lines(store,mode,q["question"])
    units,sel=select_turns_v5(q,q["question"])
    raw="\n".join(f"[{units[i][0]}] [{units[i][1]}] {units[i][2]}" for i in sel)
    parts=[]
    if view:
        head=("Current-state fact view (latest value per fact; [CONFLICT] marks values reported at the same time that could not be ordered)" if mode=="current" else "Fact evolution view (full value history per fact, oldest -> newest)")
        parts.append(f"### {head}\n{view}")
    if raw: parts.append(f"### Retrieved raw conversation turns\n{raw}")
    return "\n\n".join(parts), store, units, sel, mode
