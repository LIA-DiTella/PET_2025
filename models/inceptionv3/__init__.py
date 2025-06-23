"""Módulo de modelos InceptionV3 para clasificación de imágenes médicas.

Contiene implementaciones para:
- InceptionV3 2D: Basado en torchvision, preentrenado con ImageNet
- InceptionV3 3D (I3D): Basado en pytorch-i3d, preentrenado con Kinetics-400
"""

from .inceptionv3_2d import InceptionV3_2D, get_inceptionv3_2d
from .inceptionv3_3d import InceptionV3_3D, get_inceptionv3_3d

__all__ = [
    "InceptionV3_2D",
    "get_inceptionv3_2d",
    "InceptionV3_3D",
    "get_inceptionv3_3d",
]
