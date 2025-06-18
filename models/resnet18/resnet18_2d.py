from torch import nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extract=True) -> None:
    """Configura requires_grad=False para los parámetros si feature_extract=True.

    Args:
        model: Modelo PyTorch
        feature_extract: Si es True, congela todos los parámetros excepto los últimos

    """
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


class ResNet18_2D(nn.Module):
    """Modelo ResNet-18 adaptado para imágenes médicas 2D.
    Útil para clasificación CN/AD o CN/MCI/AD.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        feature_extract=False,
        dropout_rate=0.6,
    ) -> None:
        """Inicializa el modelo ResNet-18 para clasificación de imágenes 2D.

        Args:
            num_classes (int): Número de clases para clasificación (2 para CN/AD, 3 para CN/MCI/AD)
            pretrained (bool): Si se debe cargar pesos preentrenados de ImageNet
            feature_extract (bool): Si es True, solo se entrenan los parámetros de las capas nuevas
            dropout_rate (float): Tasa de dropout para las capas finales

        """
        super().__init__()

        # Cargar modelo pre-entrenado
        self.model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Congelar parámetros si feature_extract=True
        # set_parameter_requires_grad(self.model, feature_extract)
        set_parameter_requires_grad(self.model, feature_extract=feature_extract)

        # Modificar la primera capa convolucional para aceptar imágenes de 1 canal (PET scans)
        # original_weight = self.model.conv1.weight.data
        # if pretrained:
        #    Inicializar con el promedio de los canales RGB
        #    self.model.conv1.weight.data = torch.mean(original_weight, dim=1, keepdim=True)

        # Modificar la capa de clasificación final
        in_features = self.model.fc.in_features
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(in_features, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, num_classes),
        #     # nn.Linear(in_features, num_classes),
        # )
        self.model.dropout = nn.Dropout(dropout_rate)
        self.model.fc = nn.Sequential(
            # nn.Dropout(dropout_rate),
            nn.Linear(in_features, 1024),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(in_features, num_classes),
        #     nn.Softmax(dim=1)  # Asegurar salida de probabilidades
        # )

    def forward(self, x):
        """Forward pass del modelo.

        Args:
            x (torch.Tensor): Tensor de entrada de forma [batch_size, 1, H, W]

        Returns:
            torch.Tensor: Logits de salida de forma [batch_size, num_classes]

        """
        # print(f"Input shape: {x.shape}")
        # if x.dim() == 3:
        #     # Add 3 channels (copied from the single channel)
        #     x = torch.stack([x] * 3, dim=1)
        # elif x.dim() == 4 and x.size(1) == 1:
        #     # Add 3 channels (copied from the single channel)
        #     x = x.repeat(1, 3, 1, 1)
        # elif x.dim() != 4 or x.size(1) != 3:
        #     raise ValueError("Input tensor must be of shape [batch_size, 1, H, W] or [batch_size, 3, H, W]")
        # print(f"Modified input shape: {x.shape}")
        out = self.model(x)
        # print(f"Output shape: {out.shape}")
        return out


def get_resnet18_2d(config):
    """Función para instanciar un modelo ResNet-18 2D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        ResNet18_2D: Modelo instanciado

    """
    # Obtener número de clases de configuración
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)
    feature_extract = config.get("feature_extract", False)
    dropout_rate = config.get("dropout_rate", 0.6)

    return ResNet18_2D(
        num_classes=num_classes,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate,
    )
