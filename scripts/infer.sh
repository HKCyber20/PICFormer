#!/bin/bash

# PICFormer推理脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <输入文件路径> [输出文件路径]"
    echo "示例: $0 data/input_poses.npy output/results.npz"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=${2:-"output_results.npz"}

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 找不到输入文件 $INPUT_FILE"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "checkpoint/best.pth" ]; then
    echo "错误: 找不到模型文件 checkpoint/best.pth"
    echo "请先运行训练脚本或确保模型文件存在"
    exit 1
fi

# 创建输出目录
mkdir -p $(dirname "$OUTPUT_FILE")

# 开始推理
echo "开始PICFormer推理..."
echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"

python src/infer.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --checkpoint checkpoint/best.pth

echo "推理完成！结果保存在 $OUTPUT_FILE" 