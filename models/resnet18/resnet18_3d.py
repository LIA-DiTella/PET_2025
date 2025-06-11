import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn


class Conv3DBlock(nn.Module):
    """Bloque de convolución 3D básico para ResNet-18 3D."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class ResidualBlock3D(nn.Module):
    """Bloque residual para ResNet-18 3D."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet18_3D(nn.Module):
    """Implementación de ResNet-18 para volumetrías 3D (PET scans)."""

    def __init__(self, num_classes=2, pretrained=False, in_channels=1) -> None:
        super().__init__()

        # Capas iniciales
        self.in_channels = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Bloques residuales
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Clasificación
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Inicialización de pesos
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            # Los pesos se cargarán desde el modelo preentrenado
            pass

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock3D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_resnet18_3d(config):
    """Función para instanciar un modelo ResNet-18 3D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        ResNet18_3D: Modelo instanciado con pesos preentrenados si se especifica

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)  # Por defecto cargar pesos preentrenados
    in_channels = config.get("in_channels", 1)  # Por defecto 1 canal, pero configurable
    
    model = ResNet18_3D(num_classes=num_classes, pretrained=pretrained, in_channels=in_channels)
    
    if pretrained:
        # Descargar pesos preentrenados desde Hugging Face
        weights_path = hf_hub_download(
            repo_id="TencentMedicalNet/MedicalNet-Resnet18",
            filename="resnet_18_23dataset.pth",
            cache_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".cache")
        )
        
        # Cargar pesos preentrenados
        pretrained_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        
        # Filtrar pesos relevantes (eliminar la capa FC que no coincide)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        
        # Si el número de canales de entrada es distinto de 1, no podemos usar los pesos de la primera capa directamente
        if in_channels != 1 and 'conv1.weight' in pretrained_dict:
            # Expandir los pesos de la primera capa convolucional para manejar múltiples canales
            # Técnica común: duplicar los pesos existentes a lo largo del canal de entrada
            pretrained_conv1 = pretrained_dict['conv1.weight']  # [64, 1, 7, 7, 7]
            if pretrained_conv1.size(1) == 1:
                # Expandir pesos replicando el canal único a todos los canales de entrada
                new_conv1 = pretrained_conv1.repeat(1, in_channels, 1, 1, 1)
                # Normalizar para mantener la magnitud de activación similar
                new_conv1 = new_conv1 / in_channels
                pretrained_dict['conv1.weight'] = new_conv1
                print(f"Primera capa convolucional adaptada de 1 a {in_channels} canales")
        
        # Actualizar los pesos del modelo
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print("Modelo ResNet-18 3D cargado con pesos preentrenados de MedicalNet")
    
    return model
