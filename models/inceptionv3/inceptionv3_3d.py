import os
import sys

import torch
from torch import nn

# Agregar el submódulo pytorch-i3d al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "external", "pytorch-i3d"))

try:
    from pytorch_i3d import InceptionI3d
except ImportError as err:
    raise ImportError(
        "No se pudo importar pytorch_i3d. Asegúrate de que el submódulo esté inicializado correctamente."
    ) from err


def set_parameter_requires_grad(model, feature_extract) -> None:
    """Configura requires_grad=False para los parámetros si feature_extract=True.

    Args:
        model: Modelo PyTorch
        feature_extract: Si es True, congela todos los parámetros excepto los últimos

    """
    if feature_extract:
        for name, param in model.named_parameters():
            # Congelar todos los parámetros excepto los de la capa de clasificación final (logits)
            param.requires_grad = name.startswith("logits") or "logits" in name


class InceptionV3_3D(nn.Module):
    """Modelo InceptionV3 3D (I3D) adaptado para imágenes médicas.
    Basado en pytorch-i3d de piergiaj/pytorch-i3d.
    Útil para clasificación CN/AD o CN/MCI/AD.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        feature_extract=False,
        dropout_rate=0.5,
        weights_path=None,
        in_channels=3,
    ) -> None:
        """Inicializa el modelo InceptionV3 3D para clasificación de imágenes 3D.

        Args:
            num_classes (int): Número de clases para clasificación (2 para CN/AD, 3 para CN/MCI/AD)
            pretrained (bool): Si se debe cargar pesos preentrenados de Kinetics-400
            feature_extract (bool): Si es True, solo se entrenan los parámetros de las capas nuevas
            dropout_rate (float): Tasa de dropout para las capas finales
            weights_path (str): Ruta personalizada a los pesos preentrenados
            in_channels (int): Número de canales de entrada (3 para RGB, 1 para escala de grises/PET)

        """
        super().__init__()

        # Inicializar modelo I3D con configuración base
        self.model = InceptionI3d(
            num_classes=400,  # Kinetics-400 tiene 400 clases
            spatial_squeeze=True,
            final_endpoint="Logits",
            in_channels=in_channels,
            dropout_keep_prob=1 - dropout_rate,  # I3D usa keep_prob, no drop_prob
        )

        # Cargar pesos preentrenados si están disponibles
        if pretrained:
            self._load_pretrained_weights(weights_path)

        # Congelar parámetros si feature_extract=True
        set_parameter_requires_grad(self.model, feature_extract)

        # Reemplazar la capa de clasificación final para nuestro número de clases
        self.model.replace_logits(num_classes)

        # Agregar softmax para obtener probabilidades
        self.softmax = nn.Softmax(dim=1)

    def _load_pretrained_weights(self, weights_path=None):
        """Carga pesos preentrenados de Kinetics-400.

        Args:
            weights_path (str): Ruta personalizada a los pesos. Si es None, busca en ubicaciones estándar.

        """
        if weights_path is None:
            # Buscar pesos en ubicaciones estándar
            possible_paths = [
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "external",
                    "pytorch-i3d",
                    "models",
                    "rgb_imagenet.pt",
                ),
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "external",
                    "pytorch-i3d",
                    "models",
                    "rgb_kinetics.pt",
                ),
                os.path.join(os.path.dirname(__file__), "..", "..", "weights", "rgb_imagenet.pt"),
                os.path.join(os.path.dirname(__file__), "..", "..", "weights", "rgb_kinetics.pt"),
            ]

            weights_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    weights_path = path
                    break

        if weights_path and os.path.exists(weights_path):
            try:
                print(f"Cargando pesos preentrenados desde: {weights_path}")
                self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                print("Pesos preentrenados cargados exitosamente")
            except Exception as e:
                print(f"Error al cargar pesos preentrenados: {e}")
                print("Continuando sin pesos preentrenados...")
        else:
            print("No se encontraron pesos preentrenados. Continuando sin ellos...")
            print("Para descargar pesos preentrenados, visita:")
            print("https://github.com/piergiaj/pytorch-i3d")

    def forward(self, x):
        """Forward pass del modelo.

        Args:
            x (torch.Tensor): Tensor de entrada de forma [batch_size, channels, depth, height, width]

        Returns:
            torch.Tensor: Salida con probabilidades de forma [batch_size, num_classes]

        """
        # El modelo I3D espera entrada de forma [batch_size, channels, frames, height, width]
        # donde frames es la dimensión temporal (depth en nuestro caso)

        # Dimensiones mínimas requeridas por I3D para avg_pool (kernel [2, 7, 7])
        if ((x.ndim == 5) and (x.shape[1:] == (3, 224, 224, 16))) or (
            (x.ndim == 4) and (x.shape == (3, 224, 224, 16))
        ):
            if x.ndim == 4:
                # reshape from (3, 224, 224, 16) to (3, 16, 224, 224)
                x = x.permute(0, 3, 1, 2)
            elif x.ndim == 5:
                # reshape from (batch_size, 3, 224, 224, 16) to (batch_size, 3, 16, 224, 224)
                x = x.permute(0, 1, 4, 2, 3)

        if not (
            ((x.ndim == 5) and (x.shape[1:] == (3, 16, 224, 224)))
            or ((x.ndim == 4) and (x.shape == (3, 16, 224, 224)))
        ):
            raise ValueError(
                "El tensor de entrada debe tener forma [batch_size, channels, depth, height, width] "
                "o [depth, height, width] "
                "donde depth es la dimensión temporal (frames)."
                f" Recibido: {x.shape}"
            )

        min_depth = 16  # Para mejor rendimiento temporal
        min_height = 7  # Mínimo para el kernel de altura
        min_width = 7  # Mínimo para el kernel de ancho

        # Asegurar dimensión temporal (depth) mínima
        if x.size(2) < min_depth:
            # Repetir frames para llegar al mínimo
            repeat_factor = (min_depth + x.size(2) - 1) // x.size(2)  # Ceil division
            x = x.repeat(1, 1, repeat_factor, 1, 1)[:, :, :min_depth]

        # Asegurar dimensiones espaciales mínimas usando interpolación
        batch_size, channels, depth, height, width = x.shape

        if height < min_height or width < min_width:
            # Calcular nuevas dimensiones manteniendo proporciones cuando sea posible
            new_height = max(height, min_height)
            new_width = max(width, min_width)

            # Usar interpolación trilineal para redimensionar
            x = torch.nn.functional.interpolate(
                x, size=(depth, new_height, new_width), mode="trilinear", align_corners=False
            )

        # Forward pass a través del modelo I3D
        logits = self.model(x)

        # Aplicar softmax para obtener probabilidades
        output = self.softmax(logits).squeeze(2)

        return output


def get_inceptionv3_3d(config):
    """Función para instanciar un modelo InceptionV3 3D con configuración específica.

    Args:
        config (dict): Diccionario con parámetros de configuración

    Returns:
        InceptionV3_3D: Modelo instanciado

    """
    num_classes = config.get("num_classes", 2)  # Por defecto binario CN/AD
    pretrained = config.get("pretrained", True)
    feature_extract = config.get("feature_extract", False)
    dropout_rate = config.get("dropout_rate", 0.5)
    weights_path = config.get("weights_path", None)
    in_channels = config.get("in_channels", 3)

    return InceptionV3_3D(
        num_classes=num_classes,
        pretrained=pretrained,
        feature_extract=feature_extract,
        dropout_rate=dropout_rate,
        weights_path=weights_path,
        in_channels=in_channels,
    )
