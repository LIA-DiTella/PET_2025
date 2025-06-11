import torch
import torch.nn.functional as F
from torch import nn


class BasicConv3d(nn.Module):
    """Bloque básico de convolución 3D para InceptionV3."""

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionModule3D(nn.Module):
    """Módulo Inception para volumetrías 3D (adaptación de InceptionV3)."""

    def __init__(self, in_channels, out1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool) -> None:
        super().__init__()

        # Rama 1x1x1
        self.branch1 = BasicConv3d(in_channels, out1x1, kernel_size=1)

        # Rama 3x3x3
        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels, red_3x3, kernel_size=1),
            BasicConv3d(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        # Rama 5x5x5 (implementada como dos conv 3x3x3 consecutivas)
        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels, red_5x5, kernel_size=1),
            BasicConv3d(red_5x5, out_5x5, kernel_size=3, padding=1),
            BasicConv3d(out_5x5, out_5x5, kernel_size=3, padding=1),
        )

        # Rama pool
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BasicConv3d(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionV3_3D(nn.Module):
    """Adaptación de InceptionNetV3 para volumetrías 3D (PET scans)."""

    def __init__(self, num_classes=2) -> None:
        super().__init__()

        # Entrada: adaptar para manejar 1 canal (PET scans)
        self.conv1 = BasicConv3d(1, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = BasicConv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)

        self.conv4 = BasicConv3d(64, 80, kernel_size=3, stride=1, padding=0)
        self.conv5 = BasicConv3d(80, 192, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)

        # Módulos Inception
        self.inception1 = InceptionModule3D(192, 64, 48, 64, 64, 96, 32)
        self.inception2 = InceptionModule3D(256, 64, 48, 64, 64, 96, 64)
        self.inception3 = InceptionModule3D(288, 64, 48, 64, 64, 96, 64)

        # Reducción de dimensionalidad
        self.reduction1 = nn.Sequential(
            BasicConv3d(288, 384, kernel_size=3, stride=2, padding=0),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=0),
        )

        self.inception4 = InceptionModule3D(384, 192, 128, 192, 128, 192, 192)
        self.inception5 = InceptionModule3D(768, 192, 160, 192, 160, 192, 192)

        # Global Average Pooling y clasificador
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        # Entrada
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        # Módulos Inception
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        # Reducción
        x = self.reduction1(x)

        x = self.inception4(x)
        x = self.inception5(x)

        # Clasificación
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def get_inceptionv3_3d(config):
    """Función para instanciar un modelo InceptionV3 3D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        InceptionV3_3D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD

    return InceptionV3_3D(num_classes=num_classes)
