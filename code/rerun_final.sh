#!/bin/bash
# 修复版(同日sidx排序+空回答重试)全链路复跑:重建缓存 → 主方法+两消融
cd /root/agent_memory_iclr/code
python3 precompute_lifecycle.py 60 12 > ../results/log_precompute_v2.log 2>&1
python3 run_qc.py 60 > ../results/log_qc_v2.log 2>&1
python3 run_qc.py 60 --ablate=no_qc > ../results/log_qc_noqc_v2.log 2>&1
python3 run_qc.py 60 --ablate=view_only > ../results/log_qc_viewonly_v2.log 2>&1
echo "V2 ALL DONE"
