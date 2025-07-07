# Configuración ejemplo para múltiples datasets

## Estructura de archivos requerida

Para que el script funcione correctamente, necesitas la siguiente estructura de archivos:

### ADNI (dataset principal)

```text
/home/ipardo/storage1/PET_2025/data/metadata/
├── adni_test.csv
├── adni_train.csv
└── adni_val.csv
```

### FLENI100

```text
/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/
├── converted_niftis/          (archivos *.nii.gz)
├── fleni100.csv
└── __MACOSX/
```

### FLENI600

```text
/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/
├── converted_niftis/          (archivos *.nii.gz)
├── fleni600.csv
└── __MACOSX/
```

## Rutas actualizadas en el script

Las rutas han sido actualizadas para coincidir con la estructura real:

- **FLENI100 data_dir**: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/converted_niftis/`
- **FLENI100 test_csv**: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/fleni100.csv`
- **FLENI600 data_dir**: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/converted_niftis/`
- **FLENI600 test_csv**: `/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/fleni600.csv`

## Formato de archivos CSV

Los archivos CSV deben tener al menos estas columnas:
- `Subject` o `PTID`: ID del sujeto
- `Group` o `DX`: Diagnóstico (CN, AD, MCI, etc.)

Ejemplo de `fleni_test.csv`:
```csv
Subject,Group
FLENI_001,CN
FLENI_002,AD
FLENI_003,CN
...
```

## Correcciones aplicadas

1. **Método de evaluación**: Cambiado de `evaluate_model` a `evaluate_on_multiple_datasets`
2. **Configuración de datasets**: Añadidos todos los parámetros necesarios para FLENI100 y FLENI600
3. **Archivos CSV de test**: Especificadas las rutas correctas a los archivos de metadatos
4. **Mapeo de nombres**: Configurado el mapeo de claves de configuración a nombres de datasets

## Para ejecutar el script

```bash
python batch_evaluate_best_models.py --experiments_dir ./experiments --output_dir ./best_models_evaluation
```

Asegúrate de que:
1. Los archivos CSV de test existan en las rutas especificadas
2. Los directorios de datos contengan las imágenes NIfTI
3. Los experimentos tengan checkpoints válidos
