import argparse
import copy
import random
from pathlib import Path

import yaml

from data.dataset import get_data_loaders
from train import train_model
from utils.config_utils import load_config


def random_search(config_path: str, n_runs: int = 5, gpu_id: int = None):
    base_config = load_config(config_path)

    # Espacios de búsqueda
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6]
    dropout_list = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7]
    batch_list = [2, 4, 8, 16]
    optimizers = ["adam", "sgd"]
    """  
    if sched_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=sched_config.get("t_max", self.epochs),
            eta_min=sched_config.get("eta_min", 0),
        )

    if sched_name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=sched_config.get("mode", "min"),
            factor=sched_config.get("factor", 0.1),
            patience=sched_config.get("patience", 5),
        )

    if sched_name == "step_lr":
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=sched_config.get("step_size", 30),
            gamma=sched_config.get("gamma", 0.1),
        )
    """

    scheduler_names = ["cosine_annealing", "reduce_on_plateau", "step_lr"]
    scheduler_configs = [
        {"t_max": 50, "eta_min": 0.00001},
        {"mode": "min", "factor": 0.1, "patience": 5},
        {"step_size": 30, "gamma": 0.1},
    ]

    feature_extract_options = [True, False]

    data_loaders = get_data_loaders(base_config)
    results = []

    for _ in range(n_runs):
        config = copy.deepcopy(base_config)

        # Muestreo aleatorio
        lr = random.choice(lr_list)
        dropout = random.choice(dropout_list)
        batch = random.choice(batch_list)
        opt = random.choice(optimizers)
        fe = random.choice(feature_extract_options)
        model_name = config["model"]["name"].lower()

        # Nombre del experimento
        run_name = f"{model_name}_rs_lr{lr}_do{dropout}_bs{batch}_{opt}_fe{fe}_{random.randint(1000, 9999)}"

        # Actualización de hiperparámetros
        config["wandb"]["name"] = run_name
        config["model"]["dropout_rate"] = dropout
        config["model"]["feature_extract"] = fe
        config["training"]["optimizer"]["name"] = opt
        config["training"]["optimizer"]["lr"] = lr

        config["training"]["scheduler"]["name"] = random.choice(scheduler_names)
        sched_index = scheduler_names.index(config["training"]["scheduler"]["name"])
        for key, value in scheduler_configs[sched_index].items():
            config["training"]["scheduler"][key] = value

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
        result = train_model(config_save_path, gpu_id=gpu_id, data_loaders=data_loaders)
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Search para ResNet-18 2D en PET")
    parser.add_argument("--config", type=str, required=True, help="Ruta al YAML base")
    parser.add_argument("--n", type=int, default=5, help="Cantidad de combinaciones a probar")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="ID de GPU a usar (por defecto, usa cuda si está disponible)",
    )
    args = parser.parse_args()

    all_results = random_search(args.config, n_runs=args.n, gpu_id=args.gpu)
