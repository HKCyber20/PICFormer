#!/bin/bash

# PICFormer评估脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 检查模型文件是否存在
if [ ! -f "checkpoint/best.pth" ]; then
    echo "错误: 找不到模型文件 checkpoint/best.pth"
    echo "请先运行训练脚本或确保模型文件存在"
    exit 1
fi

# 开始评估
echo "开始PICFormer评估..."
python src/evaluate.py \
    --config config/config.yaml \
    --checkpoint checkpoint

echo "评估完成！结果保存在 checkpoint/evaluation_results.json" 