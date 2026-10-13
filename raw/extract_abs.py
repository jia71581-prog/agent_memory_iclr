#!/usr/bin/env python3
import json

data = json.load(open('/root/agent_memoir_tmp.json'))
# Build id->entry map (strip version)
idx = {}
for entries in data.values():
    if isinstance(entries, dict):
        continue
    for e in entries:
        base = e['id'].split('v')[0]
        idx[base] = e

wanted = [
 '2310.08560','2304.03442','2303.11366','2305.10250','2502.12110','2504.19413',
 '2405.14831','2502.14802','2402.04624','2502.00592','2507.02259','2507.03724',
 '2501.13956','2305.16291','2308.10144','2409.07429',
 '2506.06326','2507.07957','2506.07398','2508.06433','2510.18866','2508.19828',
 '2509.25911','2607.26760','2606.21144','2606.19847','2604.01560','2601.05488',
 '2602.19320','2404.13501','2602.06052','2605.06716','2608.22215','2608.15008',
]
out = []
for w in wanted:
    e = idx.get(w)
    if not e:
        out.append(f'### {w}\nNOT FOUND IN CACHE\n')
        continue
    out.append(f"### {w} | {e['published']} | {e['title']}\nAuthors: {', '.join(e['authors'])}\nAbstract: {e['summary']}\n")
with open('/root/agent_memory_iclr/raw/abstracts.md','w') as f:
    f.write('\n'.join(out))
print('written', len(out), 'entries; missing:', [w for w in wanted if w not in idx])
