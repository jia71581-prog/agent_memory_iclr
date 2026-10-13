#!/usr/bin/env python3
import subprocess, json, time, urllib.parse

titles = {
 'MemGPT': 'MemGPT Towards LLMs as Operating Systems',
 'GenAgents': 'Generative Agents Interactive Simulacra of Human Behavior',
 'Reflexion': 'Reflexion Language Agents with Verbal Reinforcement Learning',
 'MemoryBank': 'MemoryBank Enhancing Large Language Models with Long-Term Memory',
 'A-MEM': 'A-MEM Agentic Memory for LLM Agents',
 'Mem0': 'Mem0 Building Production-Ready AI Agents with Scalable Long-Term Memory',
 'HippoRAG': 'HippoRAG Neurobiologically Inspired Long-Term Memory for Large Language Models',
 'HippoRAG2': 'From RAG to Memory Non-Parametric Continual Learning for Large Language Models',
 'MemoryLLM': 'MemoryLLM Towards Self-Updatable Large Language Models',
 'M+': 'Extending MemoryLLM with Scalable Long-Term Memory',
 'MemAgent': 'MemAgent Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent',
 'MemOS': 'MemOS A Memory OS for AI System',
 'Zep': 'Zep A Temporal Knowledge Graph Architecture for Agent Memory',
 'Voyager': 'Voyager An Open-Ended Embodied Agent with Large Language Models',
 'ExpeL': 'ExpeL LLM Agents Are Experiential Learners',
 'AWM': 'Agent Workflow Memory',
 'MemoryOS': 'Memory OS of AI Agent',
 'MIRIX': 'MIRIX Multi-Agent Memory System for LLM-Based Agents',
 'G-Memory': 'G-Memory Tracing Hierarchical Memory for Multi-Agent Systems',
 'Memp': 'Memp Exploring Agent Procedural Memory',
 'Memory-R1': 'Memory-R1 Enhancing Large Language Model Agents to Manage and Utilize Memories',
 'Mem-alpha': 'Learning Memory Construction via Reinforcement Learning',
 'Survey2024': 'A Survey on the Memory Mechanism of Large Language Model based Agents',
}

for key, t in titles.items():
    q = urllib.parse.quote(t)
    url = f'https://api.openalex.org/works?search={q}&per-page=3&select=title,publication_year,cited_by_count,primary_location,locations'
    try:
        r = subprocess.run(['curl','-sS','--noproxy','*',url], capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout)
    except Exception as e:
        print(f'{key}: ERROR {e}')
        continue
    print(f'== {key} ==')
    for w in d.get('results', [])[:3]:
        venues = set()
        pl = w.get('primary_location') or {}
        src = (pl.get('source') or {})
        if src.get('display_name'):
            venues.add(src['display_name'])
        for loc in (w.get('locations') or []):
            s = (loc.get('source') or {})
            if s.get('display_name'):
                venues.add(s['display_name'])
        print(f"  {w.get('publication_year')} | cites={w.get('cited_by_count')} | {w.get('title','')[:80]} | venues={sorted(venues)}")
    time.sleep(1)
