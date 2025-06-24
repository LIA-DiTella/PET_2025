# import torch
# import torch.nn.functional as F
from torch import nn
from torchvision.models.video import mvit_v1_b, mvit_v2_s

variants_enum = {
    "mvit_v1_b": mvit_v1_b,
    "mvit_v2_s": mvit_v2_s,
}

weights_enum = {
    "mvit_v1_b": "MViT_V1_B_Weights",
    "mvit_v2_s": "MViT_V2_S_Weights",
}


def set_parameter_requires_grad(model, feature_extract) -> None:
    """Configura requires_grad=False para los parámetros si feature_extract=True.

    Args:
        model: Modelo PyTorch
        feature_extract: Si es True, congela todos los parámetros excepto los últimos

    """
    if feature_extract:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head")


def set_dropout(model, dropout_rate):
    """Configura la tasa de dropout en el modelo.

    Args:
        model: Modelo PyTorch
        dropout_rate (float): Tasa de dropout a aplicar

    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate


class ViT_3D(nn.Module):
    """Modelo Vision Transformer 2D adaptado para imágenes médicas.

    Útil para clasificación CN/AD o CN/MCI/AD.
    """

    def __init__(
        self,
        variant="mvit_v1_b",
        num_classes=2,
        pretrained=True,
        feature_extract=False,
        dropout_rate=0.4,
    ) -> None:
        """Inicializa el modelo ViT 2D para clasificación de imágenes.

        Args:
            img_size (int): Tamaño de la imagen de entrada
            patch_size (int): Tamaño del parche
            in_channels (int): Número de canales de entrada (1 para PET scans)
            num_classes (int): Número de clases para clasificación
            embed_dim (int): Dimensión del embedding
            depth (int): Profundidad del modelo
            num_heads (int): Número de cabezas en el mecanismo de atención

        """
        super().__init__()

        self.model = variants_enum[variant](
            weights=weights_enum[variant] if pretrained else None,
        )

        in_features = self.model.head.in_features

        self.model.head = nn.Sequential(
            nn.Linear(in_features, num_classes), nn.Softmax(dim=1)
        )

        set_dropout(self.model, dropout_rate)

        set_parameter_requires_grad(self.model, feature_extract)

    def forward(self, x):
        """Propagación hacia adelante del modelo."""
        out = self.model(x)
        return out


def get_vit_3d(config):
    """Función para instanciar un modelo ViT 2D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        ViT_3D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)
    feature_extract = config.get("feature_extract", False)
    dropout_rate = config.get("dropout_rate", 0.4)
    variant = config.get("variant", "mvit_v1_b")

    return ViT_3D(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate,
    )
