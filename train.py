"""Script de entrenamiento para modelos de clasificación PET."""

import argparse
import os
import sys

import torch

# Importar módulos propios
from data.dataset import get_data_loaders
from models.inceptionv3.inceptionv3_2d import get_inceptionv3_2d
from models.inceptionv3.inceptionv3_3d import get_inceptionv3_3d
from models.resnet18.resnet18_2d import get_resnet18_2d
from models.resnet18.resnet18_3d import get_resnet18_3d
from models.vit.vit_2d import get_vit_2d
from models.vit.vit_3d import get_vit_3d
from utils.config_utils import create_experiment_dir, load_config
from utils.trainer import Trainer

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def train_model(config_path, gpu_id=None):
    """Entrena un modelo según la configuración proporcionada.

    Args:
        config_path (str): Ruta al archivo de configuración YAML
        gpu_id (int, optional): ID de GPU a usar

    Returns:
        dict: Métricas de evaluación

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

    # Crear directorio para el experimento
    exp_base_dir = config.get("experiment", {}).get("base_dir", "./experiments")
    dataset_name = config.get("data", {}).get("dataset_name", "ADNI")
    classes = config.get("data", {}).get("classes", "CN_AD")
    exp_name = config.get("experiment", {}).get("name", None)

    exp_dir = create_experiment_dir(
        exp_base_dir,
        model_name,
        dimension,
        dataset_name,
        classes,
        exp_name,
    )

    # Crear modelo
    num_classes = len(classes.split("_"))
    model_config["num_classes"] = num_classes
    model = get_model(model_name, dimension, model_config)

    # Obtener data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Crear entrenador
    trainer = Trainer(
        model, config, device, exp_dir, train_loader, val_loader, test_loader, num_classes
    )

    # Entrenar modelo
    training_results = trainer.train(train_loader, val_loader)

    # Evaluar en conjunto de prueba
    test_metrics = trainer.evaluate(test_loader, save_results=True) if test_loader else None

    # Cerrar wandb
    trainer.close_wandb()

    return {"training": training_results, "evaluation": test_metrics, "exp_dir": exp_dir}


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos para clasificación de PET",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="ID de GPU a usar (por defecto, usa cuda si está disponible)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train_model(args.config, args.gpu)
