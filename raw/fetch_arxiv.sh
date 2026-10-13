#!/bin/bash
# Fetch arXiv API results for each target system. 3s sleep between requests.
cd /root/agent_memory_iclr/raw

fetch() {
  name="$1"; query="$2"; n="$3"; sort="$4"
  if [ -z "$sort" ]; then sort="relevance"; fi
  curl -sS --noproxy '*' -G 'https://export.arxiv.org/api/query' \
    --data-urlencode "search_query=$query" \
    --data-urlencode "max_results=$n" \
    --data-urlencode "sortBy=$sort" \
    --data-urlencode "sortOrder=descending" \
    -o "arxiv_${name}.xml"
  echo "$name: $(wc -c < arxiv_${name}.xml) bytes"
  sleep 3
}

fetch memgpt      'all:"MemGPT"' 5
fetch genagents   'ti:"Generative Agents"' 8
fetch reflexion   'ti:"Reflexion"' 5
fetch memorybank  'all:"MemoryBank"' 5
fetch amem        'all:"agentic memory"' 10
fetch mem0        'ti:"Mem0"' 5
fetch hipporag    'all:"HippoRAG"' 8
fetch memoryllm   'all:"MemoryLLM"' 8
fetch memagent    'ti:"MemAgent"' 5
fetch memos       'ti:"MemOS"' 8
fetch zep         'ti:"Zep"' 5
fetch voyager     'ti:"Voyager" AND abs:"large language"' 5
fetch expel       'ti:"ExpeL"' 5
fetch awm         'ti:"Agent Workflow Memory"' 5
fetch recent_am   'abs:"agent memory" AND cat:cs.CL' 30 submittedDate
fetch recent_ltm  'abs:"long-term memory" AND abs:"agent" AND cat:cs.CL' 30 submittedDate
echo DONE
