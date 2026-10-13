#!/bin/bash
# 预计算缓存就绪后:主方法 + 三消融 串行跑(共享缓存,答题+判分是主要耗时)
cd /root/agent_memory_iclr/code
python3 run_qc.py 60 > ../results/log_qc_n60.log 2>&1
python3 run_qc.py 60 --ablate=no_qc > ../results/log_qc_noqc_n60.log 2>&1
python3 run_qc.py 60 --ablate=view_only > ../results/log_qc_viewonly_n60.log 2>&1
echo "ALL QC RUNS DONE"
