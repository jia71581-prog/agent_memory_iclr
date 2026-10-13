#!/bin/bash
# 全量 500 主表批跑序列(KU v4 门通过后启动)
# 阶段1: 会话级缓存预热(断点续跑) + 三基线与 no_lifecycle 对照并行(均不依赖缓存)
# 阶段2: QC 臂(依赖缓存,串行防 API 过载)
set -e
cd /root/agent_memory_iclr/code

python3 precompute_sessions.py all 16 > ../results/log_precompute_full500.log 2>&1 &
PRE=$!
python3 run_full500.py full 8 > ../results/log_f500_full.log 2>&1 &
python3 run_full500.py decay 8 > ../results/log_f500_decay.log 2>&1 &
python3 run_full500.py fifo 8 > ../results/log_f500_fifo.log 2>&1 &
python3 run_full500.py qc_no_lifecycle 8 > ../results/log_f500_nolc.log 2>&1 &

wait $PRE || { echo "PRECOMPUTE FAILED - rerun launch_full500.sh (断点续跑)"; exit 1; }
grep -q "PRECOMPUTE SESSIONS DONE ok" ../results/log_precompute_full500.log
wait

python3 run_full500.py qc 8 > ../results/log_f500_qc.log 2>&1
python3 run_full500.py qc_no_qc 8 > ../results/log_f500_noqc.log 2>&1
python3 run_full500.py qc_view_only 8 > ../results/log_f500_viewonly.log 2>&1
echo "FULL500 ALL DONE"
