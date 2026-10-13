#!/usr/bin/env python3
import subprocess, json, time, urllib.parse

terms = {
 'Reflexion': 'Reflexion: Language Agents with Verbal Reinforcement Learning',
 'HippoRAG': 'HippoRAG: Neurobiologically Inspired Long-Term Memory',
 'HippoRAG2': 'From RAG to Memory: Non-Parametric Continual Learning',
 'Voyager': 'Voyager: An Open-Ended Embodied Agent',
 'MemGPT': 'MemGPT: Towards LLMs as Operating Systems',
 'MemoryLLM': 'MEMORYLLM: Towards Self-Updatable Large Language Models',
 'M+': 'M+: Extending MemoryLLM with Scalable Long-Term Memory',
 'A-MEM': 'A-MEM: Agentic Memory for LLM Agents',
 'AWM': 'Agent Workflow Memory',
 'Mem0': 'Mem0: Building Production-Ready AI Agents',
 'MemAgent': 'MemAgent: Reshaping Long-Context LLM',
 'GenAgents': 'Generative Agents: Interactive Simulacra of Human Behavior',
 'G-Memory': 'G-Memory: Tracing Hierarchical Memory',
 'Memp': 'Memp: Exploring Agent Procedural Memory',
}

def search(api, term):
    q = urllib.parse.quote(term)
    url = f'https://{api}/notes/search?term={q}&limit=5'
    r = subprocess.run(['curl','-sS','--noproxy','*',url], capture_output=True, text=True, timeout=60)
    try:
        return json.loads(r.stdout).get('notes', [])
    except Exception:
        return []

for key, t in terms.items():
    print(f'== {key} ==')
    for api in ('api2.openreview.net','api.openreview.net'):
        notes = search(api, t)
        for n in notes[:4]:
            c = n.get('content', {})
            def val(x):
                v = c.get(x)
                if isinstance(v, dict): v = v.get('value')
                return v
            title = (val('title') or '')[:70]
            venue = val('venue') or ''
            venueid = val('venueid') or ''
            if t.split(':')[0].lower() in (title or '').lower():
                print(f'  [{api}] {title} | venue={venue} | venueid={venueid}')
        time.sleep(1)
