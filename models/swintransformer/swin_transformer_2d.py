from torch import nn
from torchvision.models import (
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    Swin_V2_B_Weights,
    Swin_V2_S_Weights,
    Swin_V2_T_Weights,
    swin_b,
    swin_s,
    swin_t,
    swin_v2_b,
    swin_v2_s,
    swin_v2_t,
)

variants = {
    "swin_t": (Swin_T_Weights.IMAGENET1K_V1, swin_t),
    "swin_s": (Swin_S_Weights.IMAGENET1K_V1, swin_s),
    "swin_b": (Swin_B_Weights.IMAGENET1K_V1, swin_b),
    "swin_v2_t": (Swin_V2_T_Weights.IMAGENET1K_V1, swin_v2_t),
    "swin_v2_s": (Swin_V2_S_Weights.IMAGENET1K_V1, swin_v2_s),
    "swin_v2_b": (Swin_V2_B_Weights.IMAGENET1K_V1, swin_v2_b),
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


class SwinTransformer_2D(nn.Module):
    """Modelo Swin Transformer 2D adaptado para imágenes médicas.

    Útil para clasificación CN/AD o CN/MCI/AD usando Swin Transformer.
    """

    def __init__(
        self,
        variant="swin_t",
        num_classes=2,
        pretrained=True,
        feature_extract=False,
        dropout_rate=0.4,
    ) -> None:
        """Inicializa el modelo Swin Transformer 2D para clasificación de imágenes.

        Args:
            variant (str): Variante del modelo Swin ('swin_t', 'swin_s', 'swin_b', etc.)
            num_classes (int): Número de clases para clasificación
            pretrained (bool): Si usar pesos preentrenados en ImageNet
            feature_extract (bool): Si congelar las capas del feature extractor
            dropout_rate (float): Tasa de dropout a aplicar

        """
        super().__init__()

        if variant not in variants:
            raise ValueError(f"Variant {variant} not supported. Available: {list(variants.keys())}")

        weights, model_fn = variants[variant]

        self.model = model_fn(
            weights=weights if pretrained else None,
        )

        # Obtener el número de features de entrada del clasificador
        in_features = self.model.head.in_features

        # Reemplazar el clasificador final
        self.model.head = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes), nn.Softmax(dim=1)
        )

        # Configurar dropout en otras partes del modelo
        set_dropout(self.model, dropout_rate)

        # Configurar congelamiento de parámetros si es necesario
        set_parameter_requires_grad(self.model, feature_extract)

    def forward(self, x):
        """Propagación hacia adelante del modelo."""
        return self.model(x)


def get_swin_transformer_2d(config):
    """Función para instanciar un modelo Swin Transformer 2D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        SwinTransformer_2D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)
    feature_extract = config.get("feature_extract", False)
    dropout_rate = config.get("dropout_rate", 0.4)
    variant = config.get("variant", "swin_t")

    return SwinTransformer_2D(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate,
    )
