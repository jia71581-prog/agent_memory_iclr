"""Decisive test: for the AGG.enum/latest_count 'evidence-session-complete' errors, is the
operand the model said was missing actually present anywhere in the real qc context?"""
import json,sys,re
sys.path.insert(0,"/root/agent_memory_iclr/code"); sys.path.insert(0,"/root/agent_memory_iclr/results")
import _recon_ctx as RC
from common import load_dataset
data={q['question_id']:q for q in load_dataset("s_cleaned")}
PROBE={
 'e3038f8c':['25 rare coins','rare coins','first edition','5 rare books','rare books'],
 '67e0d0f2':['8 edX','edX courses','previous 8'],
 'f35224e0':['My Favorite Murder','favorite murder'],
 '80ec1f4f':['Modern Art Museum','Art Cube','nieces','niece'],
 '6cb6f249':['10-day','10 day','ten-day','social media break'],
 '7024f17c':['30-minute jog','30-min','jog'],
 'd682f1a2':['weekends have b','meal kit','delivery'],
 'bf659f65':['Red Rocks','Whiskey Wanderers','vinyl'],
 'gpt4_1e4a8aec':['tomato saplings','12 new tomato','saplings'],
}
for qid,pr in PROBE.items():
    q=data[qid]
    label,mode,view,raw_idx=RC.qc_context(q)
    units=RC.units_of(q); raw=" ".join(units[i][2] for i in raw_idx)
    hay=" ".join(u[2] for u in units)
    print('='*90); print(qid,'gold=',str(q['answer'])[:60],'| view',len(view),'raw',len(raw))
    for p in pr:
        f=lambda s: p.lower() in s.lower()
        print(f'   "{p}": view={f(view)} raw20={f(raw)} haystack={f(hay)}')
