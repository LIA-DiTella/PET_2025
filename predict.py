import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos propios
from models.inceptionv3.inceptionv3_2d import get_inceptionv3_2d
from models.inceptionv3.inceptionv3_3d import get_inceptionv3_3d
from models.resnet18.resnet18_2d import get_resnet18_2d
from models.resnet18.resnet18_3d import get_resnet18_3d
from models.vit.vit_2d import get_vit_2d
from models.vit.vit_3d import get_vit_3d
from utils.config_utils import load_config


def get_model(model_name, dimension, config):
    """Crea una instancia del modelo según los parámetros.

    Args:
        model_name (str): Nombre del modelo (resnet18, inceptionv3, vit)
        dimension (str): Dimensión del modelo (2d, 3d)
        config (dict): Configuración del modelo

    Returns:
        nn.Module: Modelo instanciado

    """
    if model_name == "resnet18":
        if dimension == "2d":
            return get_resnet18_2d(config)
        # 3d
        return get_resnet18_3d(config)

    if model_name == "inceptionv3":
        if dimension == "2d":
            return get_inceptionv3_2d(config)
        # 3d
        return get_inceptionv3_3d(config)

    if model_name == "vit":
        if dimension == "2d":
            return get_vit_2d(config)
        # 3d
        return get_vit_3d(config)

    msg = f"Modelo {model_name} no soportado"
    raise ValueError(msg)


def load_volume_3d(file_path, target_shape=None):
    """Carga un volumen 3D desde un archivo NIfTI.

    Args:
        file_path (str): Ruta al archivo NIfTI
        target_shape (tuple, optional): Forma objetivo para redimensionar

    Returns:
        torch.Tensor: Tensor del volumen procesado [1, 1, D, H, W]

    """
    # Cargar volumen
    img = nib.load(file_path)
    img_data = img.get_fdata()

    # Redimensionar si es necesario
    if target_shape is not None:
        img_data = resize(img_data, target_shape, order=1, preserve_range=True, anti_aliasing=True)

    # Normalizar (valores entre 0 y 1)
    img_min, img_max = img_data.min(), img_data.max()
    if img_max > img_min:
        img_data = (img_data - img_min) / (img_max - img_min)

    # Convertir a tensor
    return torch.from_numpy(img_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


def load_slice_2d(file_path, slice_idx=None):
    """Carga un corte 2D desde un archivo NIfTI.

    Args:
        file_path (str): Ruta al archivo NIfTI
        slice_idx (int, optional): Índice del corte a extraer (si es None, usa el corte central)

    Returns:
        torch.Tensor: Tensor del corte procesado [1, 1, H, W]

    """
    # Cargar volumen
    img = nib.load(file_path)
    img_data = img.get_fdata()

    # Seleccionar corte
    if slice_idx is None:
        slice_idx = img_data.shape[2] // 2  # Corte central

    slice_data = img_data[:, :, slice_idx]

    # Normalizar (valores entre 0 y 1)
    slice_min, slice_max = slice_data.min(), slice_data.max()
    if slice_max > slice_min:
        slice_data = (slice_data - slice_min) / (slice_max - slice_min)

    # Convertir a tensor
    return torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def predict_single_image(model, image_path, config):
    """Realiza una predicción para una sola imagen.

    Args:
        model: Modelo pre-entrenado
        image_path (str): Ruta a la imagen
        config (dict): Configuración

    Returns:
        tuple: (class_id, probabilities)

    """
    dimension = config.get("model", {}).get("dimension", "2d")

    if dimension == "3d":
        input_tensor = load_volume_3d(image_path)
    else:  # 2d
        slice_idx = None  # Usa el corte central
        input_tensor = load_slice_2d(image_path, slice_idx)

    # Normalización adicional si es necesario
    input_tensor = (input_tensor - 0.5) / 0.5  # Normalizar a [-1, 1]

    # Preparar para el modelo
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Realizar predicción
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)

    return predicted_class, probabilities


def predict_batch(model_path, config_path, image_paths, output_file=None, gpu_id=None):
    """Realiza predicciones para un lote de imágenes.

    Args:
        model_path (str): Ruta al checkpoint del modelo
        config_path (str): Ruta al archivo de configuración YAML
        image_paths (list): Lista de rutas a las imágenes
        output_file (str, optional): Archivo para guardar resultados
        gpu_id (int, optional): ID de GPU a usar

    Returns:
        list: Lista de resultados [(ruta, clase, probabilidades), ...]

    """
    # Cargar configuración
    config = load_config(config_path)

    # Configurar dispositivo
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtener parámetros del modelo
    model_config = config.get("model", {})
    model_name = model_config.get("name", "resnet18").lower()
    dimension = model_config.get("dimension", "2d").lower()

    # Crear modelo
    num_classes = len(config.get("data", {}).get("classes", "CN_AD").split("_"))
    model_config["num_classes"] = num_classes
    model = get_model(model_name, dimension, model_config).to(device)

    # Cargar pesos del modelo
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Obtener nombres de las clases
    class_names = config.get("data", {}).get("classes", "CN_AD").split("_")

    # Realizar predicciones para cada imagen
    results = []

    for image_path in image_paths:
        if not os.path.exists(image_path):
            continue

        try:
            predicted_class, probabilities = predict_single_image(model, image_path, config)

            # Mapear ID a nombre de clase
            class_label = class_names[predicted_class]

            # Formatear probabilidades
            prob_str = ", ".join(
                [f"{class_names[i]}: {prob:.4f}" for i, prob in enumerate(probabilities)],
            )

            result = {
                "path": image_path,
                "predicted_class": class_label,
                "probabilities": probabilities,
                "prob_str": prob_str,
            }

            results.append(result)

        except Exception:
            pass

    # Guardar resultados
    if output_file:
        with open(output_file, "w") as f:
            f.write("file_path,predicted_class,probabilities\n")
            for result in results:
                f.write(f"{result['path']},{result['predicted_class']},{result['prob_str']}\n")

    return results


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Predicción con modelos de clasificación de PET")
    parser.add_argument("--model", type=str, required=True, help="Ruta al checkpoint del modelo")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        nargs="+",
        help="Ruta(s) a las imágenes de entrada (pueden ser múltiples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo para guardar resultados (CSV)",
    )
    parser.add_argument("--gpu", type=int, default=None, help="ID de GPU a usar")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = predict_batch(args.model, args.config, args.input, args.output, args.gpu)
