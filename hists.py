"""Script de plotting de datos para clasificación de imágenes PET."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Importar módulos propios
from data.dataset import get_data_loaders
from utils.config_utils import load_config

# Configurar estilo de seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")

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
        # Filtrar valores infinitos y NaN
        valid_images = images.flatten()
        valid_images = valid_images[np.isfinite(valid_images)]

        if len(valid_images) == 0:
            ax.text(
                0.5, 0.5, "No hay datos válidos", transform=ax.transAxes, ha="center", va="center"
            )
            ax.set_title(title)
            return

        # Crear DataFrame para seaborn
        data = pd.DataFrame({"intensity": valid_images})

        # Usar seaborn para crear histograma con KDE
        sns.histplot(data=data, x="intensity", bins=100, kde=True, ax=ax, alpha=0.7)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Intensidad de píxel", fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)

        # Añadir estadísticas como texto
        mean_val = np.mean(valid_images)
        std_val = np.std(valid_images)
        min_val = np.min(valid_images)
        max_val = np.max(valid_images)

        stats_text = (
            f"Media: {mean_val:.3f}\nDesv.Est: {std_val:.3f}\nRango: [{min_val:.3f}, {max_val:.3f}]"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=10,
        )

    # Crear figura principal con mejor estilo
    plt.style.use("default")  # Reset para usar seaborn
    sns.set_theme(style="whitegrid", palette="husl")

    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(
        "Distribución de Intensidades de Píxeles en Datasets", fontsize=16, fontweight="bold"
    )

    plot_histogram(train_images, "Imágenes de Entrenamiento", axs[0])
    plot_histogram(val_images, "Imágenes de Validación", axs[1])
    plot_histogram(test_images, "Imágenes de Prueba", axs[2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Ajustar para el título principal
    plt.savefig("histograms.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Distribuciones logarítmicas
    print("Creando distribuciones logarítmicas...")

    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(
        "Distribución de Intensidades de Píxeles (Escala Logarítmica)",
        fontsize=16,
        fontweight="bold",
    )

    # Para evitar -inf, usar log(abs(x) + 1) * sign(x) o una transformación más segura
    # Alternativa: escalar los datos al rango [0, max] antes del log
    def safe_log_transform(images):
        # Escalar al rango [0, max] donde max es el valor máximo de los datos
        images_shifted = images - np.min(images)  # Mover al rango [0, max]
        return np.log1p(images_shifted)  # log(1 + x) donde x >= 0

    log_train_images = safe_log_transform(train_images)
    log_val_images = safe_log_transform(val_images)
    log_test_images = safe_log_transform(test_images)

    plot_histogram(log_train_images, "Imágenes de Entrenamiento (log)", axs[0])
    plot_histogram(log_val_images, "Imágenes de Validación (log)", axs[1])
    plot_histogram(log_test_images, "Imágenes de Prueba (log)", axs[2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("histograms_log.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Crear visualización comparativa con seaborn
    print("Creando visualización comparativa...")

    # Preparar datos para comparación
    def prepare_comparison_data(train, val, test, max_samples=10000):
        """Preparar datos para comparación limitando el número de muestras para mejor rendimiento"""
        train_flat = train.flatten()
        val_flat = val.flatten()
        test_flat = test.flatten()

        # Filtrar valores finitos
        train_clean = train_flat[np.isfinite(train_flat)]
        val_clean = val_flat[np.isfinite(val_flat)]
        test_clean = test_flat[np.isfinite(test_flat)]

        # Submuestrear si hay demasiados datos
        if len(train_clean) > max_samples:
            train_clean = np.random.choice(train_clean, max_samples, replace=False)
        if len(val_clean) > max_samples:
            val_clean = np.random.choice(val_clean, max_samples, replace=False)
        if len(test_clean) > max_samples:
            test_clean = np.random.choice(test_clean, max_samples, replace=False)

        # Crear DataFrame
        data = pd.DataFrame(
            {
                "intensity": np.concatenate([train_clean, val_clean, test_clean]),
                "dataset": (
                    ["Entrenamiento"] * len(train_clean)
                    + ["Validación"] * len(val_clean)
                    + ["Prueba"] * len(test_clean)
                ),
            }
        )
        return data

    comparison_data = prepare_comparison_data(train_images, val_images, test_images)

    # Crear plot comparativo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histograma con overlay
    sns.histplot(data=comparison_data, x="intensity", hue="dataset", alpha=0.6, bins=50, ax=ax1)
    ax1.set_title("Distribución de Intensidades por Dataset", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Intensidad de píxel", fontsize=12)
    ax1.set_ylabel("Frecuencia", fontsize=12)

    # Boxplot comparativo
    sns.boxplot(data=comparison_data, x="dataset", y="intensity", ax=ax2)
    ax2.set_title("Distribución de Intensidades (Boxplot)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_ylabel("Intensidad de píxel", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("histograms_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Filtrar ceros de las imágenes
    print("Analizando distribuciones sin ceros...")
    train_images_no_zeros = train_images[train_images != 0]
    val_images_no_zeros = val_images[val_images != 0]
    test_images_no_zeros = test_images[test_images != 0]

    # Crear histogramas sin ceros con seaborn
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(
        "Distribución de Intensidades de Píxeles (Excluyendo Ceros)", fontsize=16, fontweight="bold"
    )

    plot_histogram(train_images_no_zeros, "Imágenes de Entrenamiento (sin ceros)", axs[0])
    plot_histogram(val_images_no_zeros, "Imágenes de Validación (sin ceros)", axs[1])
    plot_histogram(test_images_no_zeros, "Imágenes de Prueba (sin ceros)", axs[2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("histograms_no_zeros.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Distribuciones logarítmicas sin ceros
    print("Creando distribuciones logarítmicas sin ceros...")

    log_train_images_no_zeros = safe_log_transform(train_images_no_zeros)
    log_val_images_no_zeros = safe_log_transform(val_images_no_zeros)
    log_test_images_no_zeros = safe_log_transform(test_images_no_zeros)

    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(
        "Distribución de Intensidades (Escala Log, Sin Ceros)", fontsize=16, fontweight="bold"
    )

    plot_histogram(log_train_images_no_zeros, "Imágenes de Entrenamiento (log, sin ceros)", axs[0])
    plot_histogram(log_val_images_no_zeros, "Imágenes de Validación (log, sin ceros)", axs[1])
    plot_histogram(log_test_images_no_zeros, "Imágenes de Prueba (log, sin ceros)", axs[2])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig("histograms_log_no_zeros.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Análisis completado. Se generaron los siguientes archivos:")
    print("- histograms.png")
    print("- histograms_log.png")
    print("- histograms_comparison.png")
    print("- histograms_no_zeros.png")
    print("- histograms_log_no_zeros.png")

    return


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
