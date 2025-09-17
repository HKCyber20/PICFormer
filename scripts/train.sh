#!/bin/bash

# PICFormer训练脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 创建必要的目录
mkdir -p checkpoint
mkdir -p logs
mkdir -p runs

# 开始训练
echo "开始PICFormer训练..."
python src/train.py \
    --config config/config.yaml \
    --checkpoint checkpoint \
    --resume ""

echo "训练完成！" 