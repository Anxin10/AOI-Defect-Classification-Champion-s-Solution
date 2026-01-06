#!/bin/bash

# AOI Champion Pipeline Automation Script
# Usage: ./run_pipeline.sh

set -e # Exit immediately if a command exits with a non-zero status

echo "=========================================================="
echo "🚀 Starting AOI Champion's Pipeline"
echo "=========================================================="

# Check if data exists
if [ ! -d "data" ]; then
    echo "❌ Error: 'data' directory not found. Please unzip aoi_data.zip into ./data/"
    exit 1
fi

echo "Step 1: Training Teacher Models (The Godly Trio)"
echo "----------------------------------------------------------"

echo "[1/3] Training ConvNeXt V2 Large..."
python train_teacher.py --model convnext

echo "[2/3] Training EVA-02 Large..."
python train_teacher.py --model eva02

echo "[3/3] Training Swin Transformer V2..."
python train_teacher.py --model swinv2

echo "✅ Step 1 Completed: All Teacher Models Trained."
echo "=========================================================="

echo "Step 2: Generating Pseudo Labels (Ensemble & TTA)"
echo "----------------------------------------------------------"
python inference_pseudo.py

echo "✅ Step 2 Completed: Pseudo labels generated at data/train_pseudo.csv"
echo "=========================================================="

echo "Step 3: Training Student Model (Noisy Student)"
echo "----------------------------------------------------------"
echo "Retraining Student Model (ConvNeXt V2) on Expanded Dataset..."
python train_student.py --model convnext

echo "✅ Step 3 Completed: Student Model Trained."
echo "=========================================================="

echo "🏆 Pipeline Finished Successfully!"
echo "Final student weights are in outputs/models/"
