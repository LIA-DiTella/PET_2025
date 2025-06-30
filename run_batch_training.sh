#!/bin/bash

# Script para ejecutar entrenamiento en lote de todas las tareas de clasificación binaria
# Usando búsqueda aleatoria de hiperparámetros

echo "🚀 Iniciando entrenamiento en lote para clasificación binaria"
echo "=================================================="

# Configuración por defecto
N_RUNS=10        # Iteraciones de búsqueda aleatoria por tarea
EPOCHS=50        # Épocas por entrenamiento
GPU_ID=0         # ID de GPU (cambiar según disponibilidad)
USE_OTSU=true    # Usar umbralizado de Otsu por defecto

# Crear directorio para resultados con timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="./batch_results_${TIMESTAMP}"

echo "📁 Directorio de resultados: $RESULTS_DIR"
echo "🔄 Iteraciones por tarea: $N_RUNS"
echo "📊 Épocas por entrenamiento: $EPOCHS"
echo "🎯 Umbralizado de Otsu: $([ "$USE_OTSU" = true ] && echo "✅ Habilitado" || echo "❌ Deshabilitado")"
echo "🖥️  GPU ID: $GPU_ID"
echo "=================================================="

# Ejecutar entrenamiento completo
if [ "$USE_OTSU" = true ]; then
    python batch_train_binary_classification.py \
        --n_runs $N_RUNS \
        --epochs $EPOCHS \
        --gpu $GPU_ID \
        --results_dir $RESULTS_DIR \
        --use_otsu_masking
else
    python batch_train_binary_classification.py \
        --n_runs $N_RUNS \
        --epochs $EPOCHS \
        --gpu $GPU_ID \
        --results_dir $RESULTS_DIR \
        --no_otsu_masking
fi

echo "=================================================="
echo "✅ Entrenamiento completado!"
echo "📊 Revisa los resultados en: $RESULTS_DIR"

# Ejemplos de uso adicionales:
# 
# 1. Entrenar solo modelos específicos con Otsu:
# python batch_train_binary_classification.py --models swin_transformer vit --n_runs 2 --epochs 20
#
# 2. Entrenar sin Otsu masking:
# python batch_train_binary_classification.py --n_runs 3 --epochs 30 --no_otsu_masking
#
# 3. Solo entrenar en 2D:
# python batch_train_binary_classification.py --dimensions 2d --n_runs 2 --epochs 20
#
# 4. Comparación con y sin Otsu:
# python batch_train_binary_classification.py --models resnet18 --results_dir ./results_with_otsu
# python batch_train_binary_classification.py --models resnet18 --no_otsu_masking --results_dir ./results_without_otsu
