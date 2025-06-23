#!/bin/bash

# Script para entrenar solo ResNet-18 (2D y 3D)

echo "🧠 Entrenando solo ResNet-18 en todas las tareas binarias"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./batch_results_resnet18_${TIMESTAMP}"

python batch_train_binary_classification.py \
    --n_runs 15 \
    --epochs 50 \
    --gpu 0 \
    --models resnet18 \
    --results_dir $RESULTS_DIR

echo "✅ ResNet-18 completado! Resultados en: $RESULTS_DIR"
