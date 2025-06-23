#!/bin/bash

# Script para entrenar solo ViT (2D y 3D)

echo "🤖 Entrenando solo ViT en todas las tareas binarias"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./batch_results_vit_${TIMESTAMP}"

python batch_train_binary_classification.py \
    --n_runs 15 \
    --epochs 50 \
    --gpu 0 \
    --models vit \
    --results_dir $RESULTS_DIR

echo "✅ ViT completado! Resultados en: $RESULTS_DIR"
