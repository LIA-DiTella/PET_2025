"""Script de plotting de datos para clasificación de imágenes PET."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

# Importar módulos propios
from data.dataset import get_data_loaders
from utils.config_utils import load_config

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def animate_images(train_loader, interval=500):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    images = []
    # labels = []

    iterator = iter(train_loader)
    for batch_imgs, _ in iterator:
        for img in batch_imgs:
            images.append(img.numpy())
            # labels.append(label.item())

    images = np.array(images)
    # labels = np.array(labels)

    def update(frame):
        ax.clear()
        ax.imshow(images[frame].squeeze(), cmap="hot")
        ax.set_title(f"Image {frame + 1}")
        ax.axis("off")

    ani = FuncAnimation(fig, update, frames=len(images), interval=interval)

    # save the animation as a gif
    ani.save("plot.gif", writer="imagemagick", fps=2)


def plot_data(config_path, gpu_id=None):
    # Cargar configuración
    config = load_config(config_path)

    # Configurar dispositivo
    if gpu_id is not None and torch.cuda.is_available():
        torch.device(f"cuda:{gpu_id}")
    else:
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtener parámetros del modelo
    model_config = config.get("model", {})

    # Crear directorio para el experimento
    classes = config.get("data", {}).get("classes", "CN_AD")

    # Crear modelo
    num_classes = len(classes.split("_"))
    model_config["num_classes"] = num_classes

    # Obtener data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Llamar a la función de animación
    animate_images(train_loader, interval=500)


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
    results = plot_data(args.config, args.gpu)
