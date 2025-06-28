from torch import nn
from torchvision.models.video import (
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)

variants = {
    "swin3d_t": (Swin3D_T_Weights.KINETICS400_V1, swin3d_t),
    "swin3d_s": (Swin3D_S_Weights.KINETICS400_V1, swin3d_s),
    "swin3d_b": (Swin3D_B_Weights.KINETICS400_V1, swin3d_b),
    "swin3d_b_22k": (Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1, swin3d_b),
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


class SwinTransformer_3D(nn.Module):
    """Modelo Swin Transformer 3D adaptado para datos volumétricos médicos.

    Útil para clasificación CN/AD o CN/MCI/AD usando volúmenes 3D completos.
    Basado en Video Swin Transformer de torchvision adaptado para datos médicos.
    """

    def __init__(
        self,
        variant="swin3d_t",
        num_classes=2,
        pretrained=True,
        feature_extract=False,
        dropout_rate=0.4,
    ) -> None:
        """Inicializa el modelo Swin Transformer 3D para clasificación volumétrica.

        Args:
            variant (str): Variante del modelo ('swin3d_t', 'swin3d_s', 'swin3d_b', 'swin3d_b_22k')
            num_classes (int): Número de clases para clasificación
            pretrained (bool): Si usar pesos preentrenados en Kinetics-400
            feature_extract (bool): Si congelar las capas del feature extractor
            dropout_rate (float): Tasa de dropout a aplicar

        """
        super().__init__()

        if variant not in variants:
            raise ValueError(f"Variant {variant} not supported. Available: {list(variants.keys())}")

        weights, model_fn = variants[variant]

        # Cargar el modelo base
        self.model = model_fn(
            weights=weights if pretrained else None,
        )

        # Obtener el número de features de entrada del clasificador
        in_features = self.model.head.in_features

        # Reemplazar el clasificador final
        self.model.head = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes), nn.Softmax(dim=1)
        )

        # Configurar dropout en otras partes del modelo si es necesario
        set_dropout(self.model, dropout_rate)

        # Configurar congelamiento de parámetros si es necesario
        set_parameter_requires_grad(self.model, feature_extract)

    def forward(self, x):
        """Propagación hacia adelante del modelo.

        Args:
            x (Tensor): Tensor de entrada con forma (B, C, T, H, W)
                       donde T es la profundidad temporal/axial del volumen

        Returns:
            Tensor: Logits de clasificación con forma (B, num_classes)
        """
        return self.model(x)


def get_swin_transformer_3d(config):
    """Función para instanciar un modelo Swin Transformer 3D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        SwinTransformer_3D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)
    feature_extract = config.get("feature_extract", False)
    dropout_rate = config.get("dropout_rate", 0.4)
    variant = config.get("variant", "swin3d_t")

    return SwinTransformer_3D(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate,
    )
