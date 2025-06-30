# Implementación de Umbralizado de Otsu para Imágenes PET

## Resumen de Cambios

### 1. Funciones Añadidas en `data/dataset.py`

- **`otsu_threshold(image)`**: Implementa el algoritmo de umbralizado de Otsu
  - Calcula automáticamente el threshold óptimo que maximiza la varianza entre clases
  - Separa el tejido cerebral del fondo en imágenes PET
  - Retorna el valor del threshold y una máscara binaria

- **`apply_brain_mask(image, use_otsu=True, min_threshold_percentile=5)`**: Aplica máscara cerebral
  - Opción de usar Otsu o threshold basado en percentil
  - Elimina el fondo de las imágenes conservando solo tejido cerebral

### 2. Integración en PETDataset

- **Nuevo parámetro**: `use_otsu_masking=True` en el constructor
- **Aplicación automática**: Se aplica en `process_image()` antes de la normalización z-score
- **Configuración**: Controlable desde archivos YAML con `use_otsu_masking: true/false`

### 3. Configuraciones Actualizadas

- `configs/swin_t_2d_adni_cnad_server.yaml`: Añadido `use_otsu_masking: true`
- `configs/swin3d_t_3d_adni_cnad_server.yaml`: Añadido `use_otsu_masking: true`

### 4. Script de Prueba

- **`test_otsu_masking.py`**: Script para probar y visualizar el umbralizado de Otsu
  - Muestra imagen original, máscara, y resultado
  - Compara diferentes métodos de enmascaramiento
  - Genera estadísticas de conservación de vóxeles

## Ventajas del Umbralizado de Otsu

1. **Automático**: No requiere ajuste manual de parámetros
2. **Adaptativo**: Se ajusta a las características específicas de cada imagen
3. **Robusto**: Funciona bien con imágenes PET que tienen distribución bimodal
4. **Conserva información relevante**: Elimina fondo mientras preserva tejido cerebral

## Uso

### En Configuración YAML
```yaml
data:
  use_otsu_masking: true  # Habilitar umbralizado de Otsu
  # ... otros parámetros
```

### En Código Python
```python
from data.dataset import PETDataset, get_data_loaders

# El dataset aplicará automáticamente Otsu si está configurado
train_loader, val_loader, test_loader = get_data_loaders(config)
```

### Para Probar Manualmente
```bash
python test_otsu_masking.py
```

## Impacto Esperado

1. **Mejor calidad de datos**: Eliminación del ruido de fondo
2. **Mejores métricas**: Enfoque en señales cerebrales relevantes
3. **Entrenamiento más eficiente**: Menos datos irrelevantes para procesar
4. **Consistencia**: Método estandarizado para todas las imágenes

## Parámetros de Control

- `use_otsu_masking`: Boolean para habilitar/deshabilitar
- `min_threshold_percentile`: Percentil alternativo si no se usa Otsu
- Compatible con todas las arquitecturas (ResNet, ViT, Swin Transformer)
- Funciona tanto en 2D como 3D

La implementación mantiene toda la funcionalidad existente mientras añade esta capacidad de preprocesamiento avanzado que es especialmente útil para imágenes médicas como PET scans.
