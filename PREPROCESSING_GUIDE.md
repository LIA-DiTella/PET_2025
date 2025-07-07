# Ejemplo de uso de las nuevas funciones de preprocesamiento

## Uso de las funciones añadidas desde hugo_preprocess.py

Este documento muestra cómo usar las nuevas funciones `clipped_zoom` y `get_indices_to_be_deleted` 
que se han integrado al dataset.

### 1. Función `clipped_zoom`

Esta función aplica zoom con recorte controlado manteniendo las dimensiones originales.
Se puede usar directamente en el código de preprocesamiento:

```python
import numpy as np
from data.dataset import clipped_zoom

# Ejemplo de uso
imagen = np.random.rand(128, 128)  # Imagen de ejemplo

# Zoom in (acercar)
imagen_zoom_in = clipped_zoom(imagen, zoom_factor=1.2)

# Zoom out (alejar)
imagen_zoom_out = clipped_zoom(imagen, zoom_factor=0.8)
```

### 2. Función `get_indices_to_be_deleted`

Esta función identifica cortes de baja calidad basados en superficie mínima.

```python
import numpy as np
from data.dataset import get_indices_to_be_deleted

# Imagen 3D de ejemplo
imagen_3d = np.random.rand(128, 128, 64)

# Identificar cortes a eliminar
indices_eliminar = get_indices_to_be_deleted(
    imagen_3d,
    min_slice_surface=100.0 * 100.0,  # Superficie mínima en mm²
    pixel_surface=1.5,                # Superficie por píxel
    min_slice_surface_threshold=0.1   # Umbral de intensidad
)

print(f"Cortes a eliminar: {indices_eliminar}")

# Eliminar cortes de baja calidad
imagen_filtrada = np.delete(imagen_3d, indices_eliminar, axis=2)
```

### 3. Integración en el Dataset

Las nuevas funciones se pueden activar a través de la configuración del dataset:

```python
# En el archivo de configuración (YAML)
data:
  # ... otras configuraciones ...
  
  # Activar filtrado por superficie
  use_surface_filtering: true
  min_slice_surface: 10000.0      # 100.0 * 100.0 mm²
  pixel_surface: 1.5              # mm² por píxel
  min_slice_surface_threshold: 0.0 # Umbral de intensidad mínima
```

O directamente al crear el dataset:

```python
from data.dataset import PETDataset

dataset = PETDataset(
    data_dir="./data",
    csv_path="./metadata.csv",
    # ... otros parámetros ...
    use_surface_filtering=True,
    min_slice_surface=100.0 * 100.0,
    pixel_surface=1.5,
    min_slice_surface_threshold=0.0
)
```

### 4. Parámetros de configuración

- `use_surface_filtering`: Activa/desactiva el filtrado por superficie
- `min_slice_surface`: Superficie mínima requerida para conservar un corte (mm²)
- `pixel_surface`: Superficie por píxel (típicamente 1.5 mm²)
- `min_slice_surface_threshold`: Umbral de intensidad mínima para considerar un píxel

### 5. Beneficios

- **Filtrado automático**: Elimina cortes con poca información cerebral
- **Mejor calidad**: Mejora la calidad del dataset eliminando cortes ruidosos
- **Compatibilidad**: Se integra perfectamente con el pipeline existente
- **Configurabilidad**: Todos los parámetros son configurables según el dataset

### 6. Consideraciones

- El filtrado por superficie está desactivado por defecto (`use_surface_filtering=False`)
- Los valores por defecto están optimizados para imágenes PET estándar
- Se requiere al menos 16 cortes válidos para que la imagen sea procesada
- Los parámetros pueden necesitar ajuste según el tipo específico de imágenes
