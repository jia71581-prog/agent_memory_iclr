#!/bin/bash
cd /root/agent_memory_iclr/raw
fetch() {
  name="$1"; query="$2"; n="$3"
  curl -sS --noproxy '*' -G 'https://export.arxiv.org/api/query' \
    --data-urlencode "search_query=$query" \
    --data-urlencode "max_results=$n" \
    -o "arxiv_${name}.xml"
  echo "$name: $(wc -c < arxiv_${name}.xml) bytes"
  sleep 3
}
fetch genagents2  'ti:"Interactive Simulacra"' 3
fetch reflexion2  'all:"Reflexion" AND all:"language agents"' 5
fetch voyager2    'ti:"An Open-Ended Embodied Agent"' 3
fetch memoryos    'ti:"Memory OS"' 5
fetch mirix       'ti:"MIRIX"' 3
fetch gmemory     'ti:"G-Memory"' 3
fetch memp        'ti:"Memp"' 5
fetch lightmem    'ti:"LightMem"' 3
fetch memoryr1    'ti:"Memory-R1"' 3
fetch memalpha    'all:"memory construction" AND all:"reinforcement learning" AND cat:cs.CL' 8
fetch survey      'ti:"survey" AND ti:"memory" AND abs:"agents"' 10
echo DONE
