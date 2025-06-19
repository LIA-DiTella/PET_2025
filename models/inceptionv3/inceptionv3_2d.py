import torch
from torch import nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extract) -> None:
    """Configura requires_grad=False para los parámetros si feature_extract=True.

    Args:
        model: Modelo PyTorch
        feature_extract: Si es True, congela todos los parámetros excepto los últimos

    """
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


class InceptionV3_2D(nn.Module):
    """Modelo InceptionV3 adaptado para imágenes médicas 2D.
    Útil para clasificación CN/AD o CN/MCI/AD.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        feature_extract=False,
        dropout_rate=0.4,
        aux_enabled=True,
    ) -> None:
        """Inicializa el modelo InceptionNetV3 para clasificación de imágenes 2D.

        Args:
            num_classes (int): Número de clases para clasificación (2 para CN/AD, 3 para CN/MCI/AD)
            pretrained (bool): Si se debe cargar pesos preentrenados de ImageNet
            feature_extract (bool): Si es True, solo se entrenan los parámetros de las capas nuevas
            dropout_rate (float): Tasa de dropout para las capas finales
            aux_enabled (bool): Si se habilita la salida auxiliar

        """
        super().__init__()

        # Cargar modelo pre-entrenado
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=aux_enabled)

        # Congelar parámetros si feature_extract=True
        set_parameter_requires_grad(self.model, feature_extract)

        # # Modificar la primera capa convolucional para aceptar imágenes de 1 canal (PET scans)
        # # Guardamos los pesos originales para inicializar el primer canal
        # if pretrained:
        #     original_conv = self.model.Conv2d_1a_3x3.conv
        #     original_weights = original_conv.weight.data

        #     # Creamos una nueva capa con un canal de entrada
        #     self.model.Conv2d_1a_3x3.conv = nn.Conv2d(
        #         1,
        #         32,
        #         kernel_size=3,
        #         stride=2,
        #         padding=0,
        #         bias=False,
        #     )

        #     # Inicializamos con el promedio de los tres canales originales
        #     self.model.Conv2d_1a_3x3.conv.weight.data = torch.mean(
        #         original_weights,
        #         dim=1,
        #         keepdim=True,
        #     )
        # else:
        #     self.model.Conv2d_1a_3x3.conv = nn.Conv2d(
        #         1,
        #         32,
        #         kernel_size=3,
        #         stride=2,
        #         padding=0,
        #         bias=False,
        #     )

        # Modificar las capas de clasificación final
        in_features = self.model.fc.in_features
        self.model.dropout = nn.Dropout(dropout_rate)
        
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1)  # Asegurar salida de probabilidades
        )

        # Modificar la capa auxiliar si existe
        if aux_enabled and hasattr(self.model, "AuxLogits"):
            in_features_aux = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    def forward(self, x):
        """Forward pass del modelo.

        Args:
            x (torch.Tensor): Tensor de entrada de forma [batch_size, 1, H, W]

        Returns:
            torch.Tensor: Logits de salida

        """
        # InceptionV3 espera imágenes de al menos 299x299
        # Importante asegurar que la entrada tenga las dimensiones correctas

        # El tamaño mínimo para InceptionV3 es 299x299
        if x.size(-1) < 299 or x.size(-2) < 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        if self.training and self.model.aux_logits:
            main_output, aux_output = self.model(x)
            return main_output, aux_output
        # Desactivamos aux_logits durante la evaluación
        prev_aux = self.model.aux_logits
        self.model.aux_logits = False
        output = self.model(x)
        self.model.aux_logits = prev_aux
        return output


def get_inceptionv3_2d(config):
    """Función para instanciar un modelo InceptionV3 2D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        InceptionV3_2D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)

    return InceptionV3_2D(num_classes=num_classes, pretrained=pretrained)
