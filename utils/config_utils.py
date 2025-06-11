import os

import yaml


def load_config(config_path):
    """Carga la configuración desde un archivo YAML.

    Args:
        config_path (str): Ruta al archivo de configuración YAML

    Returns:
        dict: Configuración cargada

    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_config(config, model_name, dimension):
    """Obtiene la configuración específica para un modelo y dimensión.

    Args:
        config (dict): Configuración global
        model_name (str): Nombre del modelo (resnet18, inceptionv3, vit)
        dimension (str): Dimensión del modelo (2d, 3d)

    Returns:
        dict: Configuración específica del modelo

    """
    return config.get("models", {}).get(model_name, {}).get(dimension, {})


def get_data_config(config, dataset_name):
    """Obtiene la configuración específica para un conjunto de datos.

    Args:
        config (dict): Configuración global
        dataset_name (str): Nombre del conjunto de datos (ADNI, FLENI100, FLENI600)

    Returns:
        dict: Configuración específica del conjunto de datos

    """
    return config.get("datasets", {}).get(dataset_name, {})


def save_config(config, save_path) -> None:
    """Guarda la configuración en un archivo YAML.

    Args:
        config (dict): Configuración a guardar
        save_path (str): Ruta donde guardar el archivo

    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def create_experiment_dir(base_dir, model_name, dimension, dataset_name, classes, exp_name=None):
    """Crea un directorio para un experimento específico.

    Args:
        base_dir (str): Directorio base para experimentos
        model_name (str): Nombre del modelo
        dimension (str): Dimensión del modelo (2d, 3d)
        dataset_name (str): Nombre del conjunto de datos
        classes (str): Clases utilizadas (CN_AD, CN_MCI_AD)
        exp_name (str, optional): Nombre adicional del experimento

    Returns:
        str: Ruta al directorio del experimento

    """
    exp_components = [model_name, dimension, dataset_name, classes]
    if exp_name:
        exp_components.append(exp_name)

    exp_dir = os.path.join(base_dir, "_".join(exp_components))
    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir
