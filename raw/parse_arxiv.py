#!/usr/bin/env python3
import xml.etree.ElementTree as ET, glob, os, json

NS = {'a': 'http://www.w3.org/2005/Atom'}
out = {}
for f in sorted(glob.glob('/root/agent_memory_iclr/raw/arxiv_*.xml')):
    name = os.path.basename(f)[6:-4]
    try:
        root = ET.parse(f).getroot()
    except Exception as e:
        out[name] = {'error': str(e)}
        continue
    entries = []
    for e in root.findall('a:entry', NS):
        aid = e.find('a:id', NS).text.strip()
        title = ' '.join(e.find('a:title', NS).text.split())
        pub = e.find('a:published', NS).text[:10]
        upd = e.find('a:updated', NS).text[:10]
        summ = ' '.join(e.find('a:summary', NS).text.split())
        authors = [x.find('a:name', NS).text for x in e.findall('a:author', NS)]
        entries.append({'id': aid.replace('http://arxiv.org/abs/',''), 'title': title,
                        'published': pub, 'updated': upd,
                        'authors': authors[:4] + (['et al.'] if len(authors) > 4 else []),
                        'summary': summ})
    out[name] = entries

with open('/root/agent_memoir_tmp.json', 'w') as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

# Print compact overview: per query, list id | date | title
for name, entries in out.items():
    print(f'=== {name} ===')
    if isinstance(entries, dict):
        print(' ERROR', entries); continue
    if not entries:
        print(' (no results)')
    for x in entries:
        print(f"  {x['id']} | {x['published']} | {x['title'][:110]}")
