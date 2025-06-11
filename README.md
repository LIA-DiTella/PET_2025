# Proyecto de Clasificación PET para Alzheimer

Este repositorio contiene el código para entrenar y evaluar modelos de deep learning para la clasificación de imágenes PET cerebrales en el contexto del diagnóstico de Alzheimer. El proyecto soporta múltiples arquitecturas de modelos (ResNet-18, InceptionNetV3, ViT) tanto para imágenes 2D como para volúmenes 3D completos.

## Estructura del Proyecto

```
├── configs/                      # Archivos de configuración para modelos
│   ├── inceptionv3_2d_adni_cnad.yaml
│   ├── resnet18_2d_adni_cnad.yaml
│   ├── resnet18_3d_adni_cnmciad.yaml
│   └── vit_3d_adni_cnmciad.yaml
├── data/                         # Código para manejo de datos
│   └── dataset.py                # Implementación de datasets
├── models/                       # Implementaciones de modelos
│   ├── inceptionv3/              # Modelo InceptionNetV3
│   │   ├── inceptionv3_2d.py     # Implementación 2D
│   │   └── inceptionv3_3d.py     # Implementación 3D
│   ├── resnet18/                 # Modelo ResNet-18
│   │   ├── resnet18_2d.py        # Implementación 2D
│   │   └── resnet18_3d.py        # Implementación 3D
│   └── vit/                      # Vision Transformer
│       ├── vit_2d.py             # Implementación 2D
│       └── vit_3d.py             # Implementación 3D
├── utils/                        # Utilidades
│   ├── config_utils.py           # Manejo de configuración
│   ├── evaluation_utils.py       # Métricas y evaluación
│   └── trainer.py                # Entrenador de modelos
├── train.py                      # Script de entrenamiento
├── evaluate.py                   # Script de evaluación
└── predict.py                    # Script para predicciones
```

## Modelos Implementados

- **ResNet-18**: Arquitectura CNN clásica adaptada para imágenes médicas.
- **InceptionNetV3**: Modelo más profundo con módulos inception para capturar características a múltiples escalas.
- **Vision Transformer (ViT)**: Modelo basado en transformers para procesamiento de imágenes, capaz de capturar relaciones de largo alcance.

Cada modelo está implementado tanto en versión 2D (para trabajar con cortes) como en versión 3D (para procesar volúmenes completos).

## Características Principales

- Soporte para datos ADNI y FLENI (100 y 600)
- Clasificación binaria (CN/AD) y multiclase (CN/MCI/AD)
- Procesamiento de imágenes 2D y volúmenes 3D
- Inicialización con pesos preentrenados en ImageNet
- Configuración flexible mediante archivos YAML
- Entrenamiento con early stopping y schedulers

## Uso Básico

### Entrenamiento

Para entrenar un modelo:

```bash
python train.py --config configs/resnet18_2d_adni_cnad.yaml
```

### Evaluación

Para evaluar un modelo entrenado:

```bash
python evaluate.py --config configs/resnet18_2d_adni_cnad.yaml --checkpoint path/to/model_checkpoint.pth
```

### Predicción

Para realizar predicciones con un modelo entrenado:

```bash
python predict.py --config configs/resnet18_2d_adni_cnad.yaml --checkpoint path/to/model_checkpoint.pth --input path/to/pet_scan.nii.gz
```

## Requisitos

- PyTorch 1.9+
- torchvision
- nibabel
- scikit-image
- pandas
- matplotlib
- pyyaml
- tqdm

## Cómo Contribuir

1. Clona el repositorio
2. Crea una nueva rama para tu funcionalidad
3. Realiza tus cambios
4. Envía un pull request
