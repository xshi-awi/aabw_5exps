#!/bin/bash
# 批量提交所有5个实验的WMT计算 (每个实验4个分段)

echo "=========================================="
echo "批量提交 5×4=20 个WMT分段作业"
echo "=========================================="

for exp in pi mh lig lgm mis; do
    echo ">>> 提交 ${exp^^} (4个分段)..."
    cd /work/ba0989/a270064/bb1029/aabw_5exps/$exp
    bash slurm_wmt_100years.sh
    echo ""
    sleep 2
done

echo "完成! 查看作业: squeue -u \$USER"
