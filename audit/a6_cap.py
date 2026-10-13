"""How much of the view survives the 20k cap, and how much do the two capped views really differ?"""
import json, sys
sys.path.insert(0,"/root/agent_memory_iclr/code")
import common; common.chat=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("no api"))
import lifecycle; lifecycle.chat=common.chat
from lifecycle import build_lifecycle, render_view, _group_by_session
from common import load_dataset
R="/root/agent_memory_iclr/results/"
def load(t): return {r["qid"]: r for r in (json.loads(l) for l in open(R+f"full500_{t}.jsonl"))}
qc,nq=load("qc"),load("qc_no_qc")
byid={q["question_id"]:q for q in load_dataset("s_cleaned")}
CAP=20000
def cap_lines(txt):
    kept,tot=[],0
    for line in txt.splitlines():
        if tot+len(line)+1>CAP: break
        kept.append(line); tot+=len(line)+1
    return kept
out=[]
for qid,r in qc.items():
    store=build_lifecycle(byid[qid],None)
    cur_all=render_view(store,"current").splitlines()
    hist_all=render_view(store,"historical").splitlines()
    cur=cap_lines("\n".join(cur_all)); hist=cap_lines("\n".join(hist_all))
    # chains that can differ between modes = multi-session-group chains
    keys=sorted(store)
    multi_keys={k for k in keys if len(_group_by_session(store[k]))>1}
    kept_keys=set(keys[:len(cur)])          # render is 1 line per chain, sorted(store) order
    kept_keys_h=set(keys[:len(hist)])
    diff_lines=sum(1 for a,b in zip(cur,hist) if a!=b)
    out.append({"qid":qid,"type":r["type"],"qlabel":r["qlabel"],
      "n_chains":len(keys),"kept_cur":len(cur),"kept_hist":len(hist),
      "n_multi":len(multi_keys),"multi_kept":len(multi_keys&kept_keys),
      "diff_lines_in_prefix":diff_lines,
      "judge_qc":r["judge"],"judge_nq":nq[qid]["judge"]})
json.dump(out,open("/root/agent_memory_iclr/audit/cap_stats.json","w"))
n=len(out)
f=lambda k: sum(r[k] for r in out)/n
print(f"questions: {n}")
print(f"mean chains built per question      : {f('n_chains'):.1f}")
print(f"mean chains SURVIVING 20k cap (cur) : {f('kept_cur'):.1f}  ({f('kept_cur')/f('n_chains')*100:.1f}% of chains)")
print(f"mean chains SURVIVING 20k cap (hist): {f('kept_hist'):.1f}")
print(f"mean multi-session chains built     : {f('n_multi'):.1f}")
print(f"mean multi-session chains SURVIVING : {f('multi_kept'):.1f}  ({f('multi_kept')/max(1e-9,f('n_multi'))*100:.1f}% of them)")
print(f"mean lines that DIFFER between the two capped views: {f('diff_lines_in_prefix'):.2f}")
z=sum(1 for r in out if r['diff_lines_in_prefix']==0)
print(f"questions where the two capped views differ in 0 lines: {z}/{n}")
import statistics as st
dl=[r['diff_lines_in_prefix'] for r in out]
print("diff-line distribution: min",min(dl),"p25",sorted(dl)[n//4],"median",st.median(dl),"p75",sorted(dl)[3*n//4],"max",max(dl))
print("mean fraction of capped view lines that differ: %.4f"%(sum(r['diff_lines_in_prefix']/max(1,r['kept_cur']) for r in out)/n))
