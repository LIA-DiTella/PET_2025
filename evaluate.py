import argparse
import os
import sys

import torch

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos propios
from data.dataset import get_data_loaders
from models.inceptionv3.inceptionv3_2d import get_inceptionv3_2d
from models.inceptionv3.inceptionv3_3d import get_inceptionv3_3d
from models.resnet18.resnet18_2d import get_resnet18_2d
from models.resnet18.resnet18_3d import get_resnet18_3d
from models.vit.vit_2d import get_vit_2d
from models.vit.vit_3d import get_vit_3d
from utils.config_utils import load_config
from utils.evaluation_utils import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_results_to_csv,
    update_results_table,
)


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


def evaluate_model(model_path, config_path, data_config=None, output_dir=None, gpu_id=None):
    """Evalúa un modelo en un conjunto de datos externo.

    Args:
        model_path (str): Ruta al checkpoint del modelo
        config_path (str): Ruta al archivo de configuración YAML
        data_config (dict, optional): Configuración del conjunto de datos a evaluar
        output_dir (str, optional): Directorio para guardar resultados
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

    # Crear modelo
    num_classes = len(config.get("data", {}).get("classes", "CN_AD").split("_"))
    model_config["num_classes"] = num_classes
    model = get_model(model_name, dimension, model_config).to(device)

    # Cargar pesos del modelo
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Si se proporciona nueva configuración de datos, actualizarla en config
    if data_config:
        config["data"] = {**config.get("data", {}), **data_config}

    # Obtener data loader para el conjunto de test
    _, _, test_loader = get_data_loaders(config)

    if test_loader is None:
        msg = "No se pudo crear el data loader de prueba. Verificar configuración."
        raise ValueError(msg)

    # Configurar directorio de salida
    if not output_dir:
        # Usar el directorio del modelo o crear uno nuevo
        exp_dir = os.path.dirname(
            os.path.dirname(model_path),
        )  # Subir dos niveles desde 'checkpoints/best_model.pth'
        output_dir = os.path.join(exp_dir, "external_evaluation")

    os.makedirs(output_dir, exist_ok=True)

    # Evaluar el modelo
    all_targets = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            scores = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    # Calcular métricas
    metrics = calculate_metrics(all_targets, all_predictions, all_scores)

    # Guardar resultados
    dataset_name = config["data"].get("dataset_name", "unknown")
    results_path = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
    save_results_to_csv(metrics, results_path)

    # Visualizar matriz de confusión
    class_names = config["data"].get("classes", "CN_AD").split("_")
    cm_path = os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=cm_path)

    # Visualizar curva ROC
    if "auc_roc" in metrics:
        roc_path = os.path.join(output_dir, f"{dataset_name}_roc_curve.png")
        plot_roc_curve(all_targets, all_scores, class_names, save_path=roc_path)

    # Actualizar tabla general de resultados
    results_table = {
        "Model": model_name,
        "Dimension": dimension,
        "Train Classes": config.get("data", {}).get("classes", "CN_AD"),
        "Eval Classes": config["data"].get("classes", "CN_AD"),
        "Dataset": dataset_name,
        "AUC ROC": metrics.get("auc_roc", "N/A"),
        "Accuracy": metrics.get("accuracy", "N/A"),
        "Sensitivity": metrics.get("sensitivity", "N/A"),
        "Specificity": metrics.get("specificity", "N/A"),
    }

    table_path = os.path.join(output_dir, "results_table.csv")
    update_results_table(results_table, table_path)

    return metrics


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Evaluación de modelos para clasificación de PET")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al checkpoint del modelo a evaluar",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directorio con datos de evaluación (sobreescribe configuración)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="CSV con metadatos para evaluación (sobreescribe configuración)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directorio para guardar resultados",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Nombre del conjunto de datos a evaluar",
    )
    parser.add_argument("--gpu", type=int, default=None, help="ID de GPU a usar")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Preparar configuración de datos si se proporcionaron argumentos que la sobreescriben
    data_config = {}
    if args.data_dir:
        data_config["data_dir"] = args.data_dir
    if args.test_csv:
        data_config["test_csv"] = args.test_csv
    if args.dataset_name:
        data_config["dataset_name"] = args.dataset_name

    data_config = data_config if data_config else None

    # Ejecutar evaluación
    metrics = evaluate_model(
        args.model,
        args.config,
        data_config=data_config,
        output_dir=args.output,
        gpu_id=args.gpu,
    )
