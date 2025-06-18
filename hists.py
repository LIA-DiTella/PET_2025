"""Script de plotting de datos para clasificación de imágenes PET."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Importar módulos propios
from data.dataset import get_data_loaders
from utils.config_utils import load_config

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    # get distribution of values in images in train_loader
    def get_images(loader):
        all_images = []
        for batch in loader:
            images, _ = batch
            all_images.append(images.numpy())
        all_images = np.concatenate(all_images, axis=0)
        return all_images

    # Obtener distribución de imágenes
    train_images = get_images(train_loader)
    val_images = get_images(val_loader)
    test_images = get_images(test_loader)

    # Crear histogramas de las imágenes
    def plot_histogram(images, title, ax):
        ax.hist(images.flatten(), bins=50, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Intensidad de píxel")
        ax.set_ylabel("Frecuencia")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_histogram(train_images, "Histograma de Imágenes de Entrenamiento", axs[0])
    plot_histogram(val_images, "Histograma de Imágenes de Validación", axs[1])
    plot_histogram(test_images, "Histograma de Imágenes de Prueba", axs[2])
    plt.tight_layout()
    plt.savefig("histograms.png")

    # Log distributions
    print("Distribución de imágenes de entrenamiento:")
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    log_train_images = np.log1p(train_images)
    log_val_images = np.log1p(val_images)
    log_test_images = np.log1p(test_images)
    plot_histogram(log_train_images, "Histograma de Imágenes de Entrenamiento (log)", axs[0])
    plot_histogram(log_val_images, "Histograma de Imágenes de Validación (log)", axs[1])
    plot_histogram(log_test_images, "Histograma de Imágenes de Prueba (log)", axs[2])
    plt.tight_layout()
    plt.savefig("histograms_log.png")

    # filter zeros from train_images
    train_images = train_images[train_images != 0]
    val_images = val_images[val_images != 0]
    test_images = test_images[test_images != 0]
    # Crear histogramas de las imágenes sin ceros
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_histogram(train_images, "Histograma de Imágenes de Entrenamiento (sin ceros)", axs[0])
    plot_histogram(val_images, "Histograma de Imágenes de Validación (sin ceros)", axs[1])
    plot_histogram(test_images, "Histograma de Imágenes de Prueba (sin ceros)", axs[2])
    plt.tight_layout()
    plt.savefig("histograms_no_zeros.png")

    # Log distributions without zeros
    print("Distribución de imágenes de entrenamiento sin ceros:")
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    log_train_images_no_zeros = np.log1p(train_images)
    log_val_images_no_zeros = np.log1p(val_images)
    log_test_images_no_zeros = np.log1p(test_images)
    plot_histogram(
        log_train_images_no_zeros,
        "Histograma de Imágenes de Entrenamiento (log, sin ceros)",
        axs[0],
    )
    plot_histogram(
        log_val_images_no_zeros, "Histograma de Imágenes de Validación (log, sin ceros)", axs[1]
    )
    plot_histogram(
        log_test_images_no_zeros, "Histograma de Imágenes de Prueba (log, sin ceros)", axs[2]
    )
    plt.tight_layout()
    plt.savefig("histograms_log_no_zeros.png")


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
