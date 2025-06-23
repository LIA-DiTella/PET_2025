#!/bin/bash

# Script para entrenar solo InceptionV3 (2D y 3D)

echo "🏗️ Entrenando solo InceptionV3 en todas las tareas binarias"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./batch_results_inceptionv3_${TIMESTAMP}"

python batch_train_binary_classification.py \
    --n_runs 15 \
    --epochs 50 \
    --gpu 0 \
    --models inceptionv3 \
    --results_dir $RESULTS_DIR

echo "✅ InceptionV3 completado! Resultados en: $RESULTS_DIR"
