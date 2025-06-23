#!/bin/bash

# Script para ejecutar entrenamiento en lote de todas las tareas de clasificación binaria
# Usando búsqueda aleatoria de hiperparámetros

echo "🚀 Iniciando entrenamiento en lote para clasificación binaria"
echo "=================================================="

# Configuración por defecto
N_RUNS=10        # Iteraciones de búsqueda aleatoria por tarea
EPOCHS=50        # Épocas por entrenamiento
GPU_ID=0         # ID de GPU (cambiar según disponibilidad)

# Crear directorio para resultados con timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./batch_results_${TIMESTAMP}"

echo "📁 Directorio de resultados: $RESULTS_DIR"
echo "🔄 Iteraciones por tarea: $N_RUNS"
echo "📊 Épocas por entrenamiento: $EPOCHS"
echo "🖥️  GPU ID: $GPU_ID"
echo "=================================================="

# Ejecutar entrenamiento completo
python batch_train_binary_classification.py \
    --n_runs $N_RUNS \
    --epochs $EPOCHS \
    --gpu $GPU_ID \
    --results_dir $RESULTS_DIR

echo "=================================================="
echo "✅ Entrenamiento completado!"
echo "📊 Revisa los resultados en: $RESULTS_DIR"
