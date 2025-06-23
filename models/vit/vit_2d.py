# import torch
# import torch.nn.functional as F
from torch import nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

variants_enum = {
    "vit_b_16": vit_b_16,
    "vit_b_32": vit_b_32,  # No hay vit_b_32 en torchvision, usar vit_b_16
    "vit_l_16": vit_l_16,
    "vit_l_32": vit_l_32,
}


def set_parameter_requires_grad(model, feature_extract) -> None:
    """Configura requires_grad=False para los parámetros si feature_extract=True.

    Args:
        model: Modelo PyTorch
        feature_extract: Si es True, congela todos los parámetros excepto los últimos

    """
    if feature_extract:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("heads.head")


def set_dropout(model, dropout_rate):
    """Configura la tasa de dropout en el modelo.

    Args:
        model: Modelo PyTorch
        dropout_rate (float): Tasa de dropout a aplicar

    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate


class ViT_2D(nn.Module):
    """Modelo Vision Transformer 2D adaptado para imágenes médicas.

    Útil para clasificación CN/AD o CN/MCI/AD.
    """

    def __init__(
        self,
        variant="vit_b_16",
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
            weights="IMAGENET1K_V1" if pretrained else None,
        )

        in_features = self.model.heads.head.in_features

        self.model.heads.head = nn.Sequential(
            nn.Linear(in_features, num_classes), nn.Softmax(dim=1)
        )

        set_dropout(self.model, dropout_rate)

        set_parameter_requires_grad(self.model, feature_extract)

    def forward(self, x):
        """Propagación hacia adelante del modelo."""
        out = self.model(x)
        return out


def get_vit_2d(config):
    """Función para instanciar un modelo ViT 2D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        ViT_2D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)
    feature_extract = config.get("feature_extract", False)
    dropout_rate = config.get("dropout_rate", 0.4)
    variant = config.get("variant", "vit_b_16")

    return ViT_2D(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate,
    )
