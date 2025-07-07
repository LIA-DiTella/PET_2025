# Guía de ejecución para evaluación multi-dataset

## Entorno de trabajo

- **Directorio base**: `/home/ipardo/storage1/PET_2025/`
- **Archivos de configuración**: El script usa las rutas actualizadas automáticamente

## Pasos para ejecutar

1. **Navegar al directorio del proyecto**:
   ```bash
   cd /home/ipardo/storage1/PET_2025/
   ```

2. **Verificar que todos los archivos necesarios existen**:
   ```bash
   python verify_paths.py
   ```

3. **Ejecutar la evaluación**:
   ```bash
   python batch_evaluate_best_models.py --experiments_dir ./experiments --output_dir ./best_models_evaluation
   ```

## Estructura de archivos verificada

✅ **FLENI100**:
- Imágenes: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/converted_niftis/`
- Metadatos: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/fleni100.csv`

✅ **FLENI600**:
- Imágenes: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/converted_niftis/`
- Metadatos: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/fleni600.csv`

## Resultados esperados

El script debería:
1. Encontrar automáticamente todos los experimentos válidos
2. Seleccionar el mejor modelo para cada combinación (modelo, dimensión)
3. Evaluar cada modelo en ADNI, FLENI100 y FLENI600
4. Generar reportes en `./best_models_evaluation/`

## Solución de problemas

Si encuentras errores sobre archivos no encontrados:
1. Ejecuta `python verify_paths.py` para identificar archivos faltantes
2. Verifica que estás ejecutando desde `/home/ipardo/storage1/PET_2025/`
3. Confirma que los archivos CSV tienen las columnas correctas (`Subject`, `Group`)

## Archivos de salida

- `evaluation_report.csv`: Resumen consolidado de métricas
- `complete_evaluation_results.json`: Resultados detallados en formato JSON
- Subdirectorios por modelo con gráficos y métricas individuales
