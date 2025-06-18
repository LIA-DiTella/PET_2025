import argparse
import copy
import random
from pathlib import Path

import yaml

from train import train_model
from utils.config_utils import load_config

from data.dataset import get_data_loaders


def random_search(config_path: str, n_runs: int = 5):
    base_config = load_config(config_path)

    # Espacios de búsqueda
    lr_list = [1e-4, 1e-5, 1e-6]
    dropout_list = [0.3, 0.5, 0.6, 0.7]
    batch_list = [16, 32, 64]
    optimizers = ["adam", "sgd"]
    feature_extract_options = [True, False]

    results = []

    for _ in range(n_runs):
        config = copy.deepcopy(base_config)

        # Muestreo aleatorio
        lr = random.choice(lr_list)
        dropout = random.choice(dropout_list)
        batch = random.choice(batch_list)
        opt = random.choice(optimizers)
        fe = random.choice(feature_extract_options)

        # Nombre del experimento
        run_name = f"rs_lr{lr}_do{dropout}_bs{batch}_{opt}_fe{fe}_{random.randint(1000, 9999)}"

        # Actualización de hiperparámetros
        config["wandb"]["name"] = run_name
        config["model"]["dropout_rate"] = dropout
        config["model"]["feature_extract"] = fe
        config["training"]["optimizer"]["name"] = opt
        config["training"]["optimizer"]["lr"] = lr
        config["training"]["epochs"] = 50
        config["data"]["batch_size"] = batch
        config["experiment"]["name"] = run_name

        # save config
        config_save_path = Path(config["experiment"]["base_dir"]) / "configs" / f"{run_name}.yaml"
        config_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
            
        config["experiment"]["config_path"] = config_save_path

        print(f"\n🔁 Ejecutando experimento: {run_name}")
        result = train_model(config_save_path, gpu_id=None, data_loaders=get_data_loaders(config))
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Search para ResNet-18 2D en PET")
    parser.add_argument("--config", type=str, required=True, help="Ruta al YAML base")
    parser.add_argument("--n", type=int, default=5, help="Cantidad de combinaciones a probar")
    args = parser.parse_args()

    all_results = random_search(args.config, n_runs=args.n)
