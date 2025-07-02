"""
Script para entrenar automáticamente todas las tareas de clasificación binaria
usando búsqueda aleatoria de hiperparámetros en múltiples modelos y configuraciones.

Características:
- Soporte para múltiples modelos: ResNet18, InceptionV3, ViT, Swin Transformer
- Dimensiones 2D y 3D
- Búsqueda aleatoria de hiperparámetros
- Umbralizado de Otsu configurable para máscara cerebral
- Guardado automático de resultados y configuraciones
- Integración con Weights & Biases
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from data.dataset import get_data_loaders
from train import train_model
from utils.config_utils import load_config


class BinaryClassificationBatchTrainer:
    """Entrenador en lote para tareas de clasificación binaria."""

    def __init__(
        self,
        base_configs_dir: str = "./configs",
        results_dir: str = "./batch_results",
        use_otsu_masking: bool = True,
    ):
        self.base_configs_dir = Path(base_configs_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.use_otsu_masking = use_otsu_masking

        # Espacios de búsqueda para hiperparámetros
        self.hyperparams_space = {
            "lr": [1e-3, 1e-4, 1e-5, 1e-6],
            "dropout": [0.0, 0.1, 0.25, 0.5],
            "batch_size": [2, 4, 8, 16],
            "optimizers": ["adam", "sgd"],
            "schedulers": [
                {"name": "cosine_annealing", "config": {"t_max": 50, "eta_min": 0.00001}},
                {
                    "name": "reduce_on_plateau",
                    "config": {"mode": "min", "factor": 0.1, "patience": 5},
                },
                {"name": "step_lr", "config": {"step_size": 30, "gamma": 0.1}},
            ],
            "feature_extract": [True, False],
        }

        # Definir todas las tareas de clasificación binaria desde la tabla
        self.binary_tasks = self._define_binary_tasks()

    def _define_binary_tasks(self) -> List[Dict]:
        """Define todas las tareas de clasificación binaria basadas en la tabla."""
        tasks = []

        # models = ["resnet18", "inceptionv3", "vit", "swin_transformer"]
        models = ["swin_transformer", "vit"]
        dimensions = ["2d", "3d"]
        datasets = ["ADNI"]  # Solo ADNI según el request

        # Tareas binarias: CN/AD (entrenar con CN/AD, evaluar con CN/AD)
        # y CN/AD from multiclass (entrenar con CN/MCI/AD, evaluar con CN/AD)

        for model in models:
            for dim in dimensions:
                for dataset in datasets:
                    # Tarea 1: CN/AD puro (entrenar y evaluar con CN/AD)
                    tasks.append(
                        {
                            "model": model,
                            "dimension": dim,
                            "dataset": dataset,
                            "train_classes": "CN_AD",
                            "eval_classes": "CN_AD",
                            "task_type": "binary_pure",
                        }
                    )

                    # Tarea 2: CN/AD desde multiclase (entrenar con CN/MCI/AD, evaluar con CN/AD)
                    # tasks.append(
                    #     {
                    #         "model": model,
                    #         "dimension": dim,
                    #         "dataset": dataset,
                    #         "train_classes": "CN_MCI_AD",
                    #         "eval_classes": "CN_AD",
                    #         "task_type": "binary_from_multiclass",
                    #     }
                    # )

        return tasks

    def _get_base_config_path(self, model: str, dimension: str, dataset: str, classes: str) -> Path:
        """Obtiene la ruta del archivo de configuración base."""
        # Intentar encontrar un archivo de configuración base apropiado
        config_patterns = [
            f"{model}_{dimension}_{dataset.lower()}_{classes.lower()}_server.yaml",
            f"{model}_{dimension}_{dataset.lower()}_{classes.lower()}.yaml",
            f"{model}_{dimension}_{dataset.lower()}_cnad_server.yaml",
            f"{model}_{dimension}_{dataset.lower()}_cnad.yaml",
        ]

        for pattern in config_patterns:
            config_path = self.base_configs_dir / pattern
            if config_path.exists():
                return config_path

        # Si no se encuentra, usar configuración por defecto según el modelo
        default_configs = {
            "resnet18": {
                "2d": self.base_configs_dir / "resnet18_2d_adni_cnad_server.yaml",
                "3d": self.base_configs_dir / "resnet18_3d_adni_cnad_server.yaml",
            },
            "inceptionv3": {
                "2d": self.base_configs_dir / "inceptionv3_2d_adni_cnad_server.yaml",
                "3d": self.base_configs_dir / "inceptionv3_3d_adni_cnad_server.yaml",
            },
            "vit": {
                "2d": self.base_configs_dir / "vit_b_16_2d_adni_cnad_server.yaml",
                "3d": self.base_configs_dir / "vit_3d_adni_cnad_server.yaml",
            },
            "swin_transformer": {
                "2d": self.base_configs_dir / "swin_t_2d_adni_cnad_server.yaml",
                "3d": self.base_configs_dir / "swin3d_t_3d_adni_cnad_server.yaml",
            },
        }

        if model in default_configs and dimension in default_configs[model]:
            return default_configs[model][dimension]

        raise FileNotFoundError(f"No se encontró configuración base para {model}_{dimension}")

    def _sample_hyperparameters(self) -> Dict:
        """Muestrea hiperparámetros aleatorios del espacio de búsqueda."""
        scheduler = random.choice(self.hyperparams_space["schedulers"])

        return {
            "lr": random.choice(self.hyperparams_space["lr"]),
            "dropout": random.choice(self.hyperparams_space["dropout"]),
            "batch_size": random.choice(self.hyperparams_space["batch_size"]),
            "optimizer": random.choice(self.hyperparams_space["optimizers"]),
            "scheduler": scheduler,
            "feature_extract": random.choice(self.hyperparams_space["feature_extract"]),
        }

    def _create_config_for_task(
        self, task: Dict, hyperparams: Dict, epochs: int, run_id: int
    ) -> Tuple[Dict, Path]:
        """Crea una configuración específica para una tarea."""
        base_config_path = self._get_base_config_path(
            task["model"], task["dimension"], task["dataset"], task["train_classes"]
        )

        config = load_config(str(base_config_path))
        config = copy.deepcopy(config)

        # Actualizar configuración del modelo
        config["model"]["name"] = task["model"]
        config["model"]["dimension"] = task["dimension"]
        config["model"]["dropout_rate"] = hyperparams["dropout"]
        config["model"]["feature_extract"] = hyperparams["feature_extract"]

        # Actualizar configuración de datos
        config["data"]["dataset_name"] = task["dataset"]
        config["data"]["classes"] = task["train_classes"]
        config["data"]["dimension"] = task["dimension"]
        config["data"]["batch_size"] = hyperparams["batch_size"]
        config["data"]["use_otsu_masking"] = self.use_otsu_masking

        # Actualizar configuración de entrenamiento
        config["training"]["epochs"] = epochs
        config["training"]["optimizer"]["name"] = hyperparams["optimizer"]
        config["training"]["optimizer"]["lr"] = hyperparams["lr"]
        config["training"]["scheduler"]["name"] = hyperparams["scheduler"]["name"]

        # Actualizar configuración del scheduler
        for key, value in hyperparams["scheduler"]["config"].items():
            config["training"]["scheduler"][key] = value

        # Nombre del experimento
        run_name = (
            f"{task['model']}_{task['dimension']}_{task['dataset']}_{task['train_classes']}_"
            f"{task['task_type']}_lr{hyperparams['lr']}_do{hyperparams['dropout']}_"
            f"bs{hyperparams['batch_size']}_{hyperparams['optimizer']}_"
            f"fe{hyperparams['feature_extract']}_run{run_id}_{random.randint(1000, 9999)}"
        )

        config["experiment"]["name"] = run_name
        config["wandb"]["name"] = run_name
        config["wandb"]["tags"] = [
            task["model"],
            task["dimension"],
            task["dataset"].lower(),
            "pet",
            "alzheimer",
            task["train_classes"].lower(),
            task["task_type"],
            "otsu_masking" if self.use_otsu_masking else "no_otsu_masking",
        ]

        # Guardar configuración
        config_save_path = self.results_dir / "configs" / f"{run_name}.yaml"
        config_save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_save_path, "w") as f:
            yaml.dump(config, f)

        config["experiment"]["config_path"] = str(config_save_path)

        return config, config_save_path

    def random_search_single_task(
        self, task: Dict, n_runs: int, epochs: int, gpu_id: int = None, data_loaders=None
    ) -> List[Dict]:
        """Ejecuta búsqueda aleatoria para una sola tarea."""
        print("\n🚀 Iniciando búsqueda aleatoria para:")
        print(f"   Modelo: {task['model']} {task['dimension']}")
        print(f"   Dataset: {task['dataset']}")
        print(f"   Clases entrenamiento: {task['train_classes']}")
        print(f"   Clases evaluación: {task['eval_classes']}")
        print(f"   Tipo: {task['task_type']}")
        print(f"   Iteraciones: {n_runs}")

        results = []

        # Cargar datos una sola vez para esta tarea
        base_config_path = self._get_base_config_path(
            task["model"], task["dimension"], task["dataset"], task["train_classes"]
        )
        base_config = load_config(str(base_config_path))
        base_config["data"]["classes"] = task["train_classes"]
        base_config["data"]["use_otsu_masking"] = self.use_otsu_masking

        if data_loaders is None:
            try:
                data_loaders = get_data_loaders(base_config)
            except Exception as e:
                print(f"❌ Error cargando datos para {task}: {e}")
                return results

        for run_id in range(n_runs):
            try:
                # Muestrear hiperparámetros
                hyperparams = self._sample_hyperparameters()

                # Crear configuración para este run
                config, config_path = self._create_config_for_task(
                    task, hyperparams, epochs, run_id
                )

                print(f"\n🔁 Ejecutando run {run_id + 1}/{n_runs}")
                print(f"   Nombre: {config['experiment']['name']}")
                print(
                    f"   Hiperparámetros: lr={hyperparams['lr']}, dropout={hyperparams['dropout']}, "
                    f"batch_size={hyperparams['batch_size']}, optimizer={hyperparams['optimizer']}"
                )

                start_time = time.time()

                # Entrenar modelo
                result = train_model(str(config_path), gpu_id=gpu_id, data_loaders=data_loaders)

                end_time = time.time()
                training_time = end_time - start_time

                # Agregar metadatos al resultado
                result["task"] = task
                result["hyperparams"] = hyperparams
                result["run_id"] = run_id
                result["training_time"] = training_time
                result["config_path"] = str(config_path)
                result["use_otsu_masking"] = self.use_otsu_masking

                results.append(result)

                print(f"✅ Completado en {training_time / 60:.1f} minutos")

            except Exception as e:
                print(f"❌ Error en run {run_id + 1}: {e}")
                continue

        return results

    def run_batch_training(
        self,
        n_runs: int,
        epochs: int,
        gpu_id: int = None,
        filter_models: List[str] = None,
        filter_dimensions: List[str] = None,
    ) -> Dict:
        """Ejecuta entrenamiento en lote para todas las tareas."""
        print("🎯 Iniciando entrenamiento en lote")
        print(f"   Total de tareas: {len(self.binary_tasks)}")
        print(f"   Iteraciones por tarea: {n_runs}")
        print(f"   Épocas por entrenamiento: {epochs}")
        print(
            f"   Umbralizado de Otsu: {'✅ Habilitado' if self.use_otsu_masking else '❌ Deshabilitado'}"
        )

        # Filtrar tareas si se especifica
        tasks_to_run = self.binary_tasks
        if filter_models:
            tasks_to_run = [t for t in tasks_to_run if t["model"] in filter_models]
        if filter_dimensions:
            tasks_to_run = [t for t in tasks_to_run if t["dimension"] in filter_dimensions]

        print(f"   Tareas a ejecutar: {len(tasks_to_run)}")

        all_results = {}
        total_start_time = time.time()

        data_loaders_2d = None
        data_loaders_3d = None

        # Cargar data loaders una sola vez para cada dimensión
        try:
            if any(t["dimension"] == "2d" for t in tasks_to_run):
                print("🔄 Cargando data loaders para 2D...")
                base_config_2d = self._get_base_config_path(
                    tasks_to_run[0]["model"],
                    "2d",
                    tasks_to_run[0]["dataset"],
                    tasks_to_run[0]["train_classes"],
                )
                base_config_2d = load_config(str(base_config_2d))
                base_config_2d["data"]["dimension"] = "2d"
                base_config_2d["data"]["classes"] = tasks_to_run[0]["train_classes"]
                base_config_2d["data"]["use_otsu_masking"] = self.use_otsu_masking
                data_loaders_2d = get_data_loaders(base_config_2d)

            if any(t["dimension"] == "3d" for t in tasks_to_run):
                print("🔄 Cargando data loaders para 3D...")
                base_config_3d = self._get_base_config_path(
                    tasks_to_run[0]["model"],
                    "3d",
                    tasks_to_run[0]["dataset"],
                    tasks_to_run[0]["train_classes"],
                )
                base_config_3d = load_config(str(base_config_3d))
                base_config_3d["data"]["dimension"] = "3d"
                base_config_3d["data"]["classes"] = tasks_to_run[0]["train_classes"]
                base_config_3d["data"]["use_otsu_masking"] = self.use_otsu_masking
                data_loaders_3d = get_data_loaders(base_config_3d)
        except Exception as e:
            print(f"❌ Error cargando data loaders: {e}")
            return {}

        for i, task in enumerate(tasks_to_run):
            task_key = f"{task['model']}_{task['dimension']}_{task['dataset']}_{task['train_classes']}_{task['task_type']}"

            print(f"\n{'=' * 80}")
            print(f"Tarea {i + 1}/{len(tasks_to_run)}: {task_key}")
            print(f"{'=' * 80}")

            task_results = self.random_search_single_task(
                task,
                n_runs,
                epochs,
                gpu_id,
                (data_loaders_2d if task["dimension"] == "2d" else data_loaders_3d),
            )
            all_results[task_key] = task_results

            # Guardar resultados intermedios
            self._save_intermediate_results(all_results)

        total_time = time.time() - total_start_time
        print(f"\n🎉 Entrenamiento en lote completado en {total_time / 3600:.1f} horas")

        # Guardar resultados finales
        self._save_final_results(all_results)

        return all_results

    def cross_evaluate_trained_models(
        self,
        target_datasets: List[str] = None,
        target_classes_list: List[str] = None,
        gpu_id: int = None,
        use_best_models_only: bool = True,
    ) -> Dict:
        """Evalúa modelos entrenados en diferentes conjuntos de datos (evaluación cruzada).

        Args:
            target_datasets: Lista de datasets objetivo para evaluación
            target_classes_list: Lista de configuraciones de clases para evaluación
            gpu_id: ID de GPU a usar
            use_best_models_only: Si True, usa solo los mejores modelos de cada tarea

        Returns:
            Diccionario con resultados de evaluación cruzada
        """
        # Importar evaluador
        try:
            from model_evaluator import ModelEvaluator
        except ImportError:
            print(
                "❌ No se pudo importar ModelEvaluator. Asegúrate de que model_evaluator.py esté disponible."
            )
            return {}

        if target_datasets is None:
            target_datasets = ["ADNI"]

        if target_classes_list is None:
            target_classes_list = ["CN_AD", "CN_MCI_AD"]

        # Configurar dispositivo
        if gpu_id is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        evaluator = ModelEvaluator(device=device)

        print("\n🔄 Iniciando evaluación cruzada de modelos entrenados")
        print(f"   Datasets objetivo: {target_datasets}")
        print(f"   Configuraciones de clases: {target_classes_list}")

        # Buscar experimentos entrenados
        experiments_dir = Path("./experiments")
        if not experiments_dir.exists():
            print("❌ No se encontró directorio de experimentos")
            return {}

        all_cross_eval_results = {}

        # Buscar experimentos que coincidan con las tareas entrenadas
        for task_key, task_results in self._find_trained_experiments().items():
            if not task_results:
                continue

            print(f"\n{'=' * 80}")
            print(f"Evaluación cruzada para: {task_key}")
            print(f"{'=' * 80}")

            # Seleccionar modelo a evaluar
            if use_best_models_only:
                # Usar el mejor modelo según AUC
                best_result = self._get_best_result(task_results)
                if not best_result:
                    print(f"⚠️  No se encontró mejor modelo para {task_key}")
                    continue
                models_to_eval = [best_result]
            else:
                # Evaluar todos los modelos entrenados
                models_to_eval = task_results

            task_cross_results = {}

            for model_result in models_to_eval:
                if "config_path" not in model_result:
                    continue

                # Encontrar checkpoint del modelo
                config_path = model_result["config_path"]
                experiment_name = Path(config_path).stem
                exp_dir = experiments_dir / experiment_name

                if not exp_dir.exists():
                    print(f"⚠️  No se encontró directorio de experimento: {exp_dir}")
                    continue

                checkpoints_dir = exp_dir / "checkpoints"
                if not checkpoints_dir.exists():
                    print(f"⚠️  No se encontró directorio de checkpoints: {checkpoints_dir}")
                    continue

                # Buscar mejor checkpoint
                best_checkpoint = None
                for checkpoint_file in checkpoints_dir.glob("*.pth"):
                    if "best" in checkpoint_file.name.lower():
                        best_checkpoint = checkpoint_file
                        break

                if best_checkpoint is None:
                    checkpoints = list(checkpoints_dir.glob("*.pth"))
                    if checkpoints:
                        best_checkpoint = max(checkpoints, key=os.path.getctime)
                    else:
                        print(f"⚠️  No se encontraron checkpoints en {checkpoints_dir}")
                        continue

                print(f"🔍 Evaluando modelo: {experiment_name}")
                print(f"   Checkpoint: {best_checkpoint}")

                try:
                    # Cargar modelo
                    model, original_config = evaluator.load_model_from_checkpoint(
                        str(best_checkpoint), str(config_path)
                    )

                    # Evaluar en cada combinación de dataset y clases
                    for target_dataset in target_datasets:
                        for target_classes in target_classes_list:
                            eval_key = f"{target_dataset}_{target_classes}"

                            print(f"\n   📊 Evaluando en: {eval_key}")

                            # Crear directorio de salida
                            cross_eval_dir = (
                                self.results_dir
                                / "cross_evaluation"
                                / task_key
                                / experiment_name
                                / eval_key
                            )
                            cross_eval_dir.mkdir(parents=True, exist_ok=True)

                            # Evaluar
                            metrics = evaluator.evaluate_on_dataset(
                                model=model,
                                original_config=original_config,
                                target_dataset=target_dataset,
                                target_classes=target_classes,
                                use_otsu_masking=self.use_otsu_masking,
                                output_dir=str(cross_eval_dir),
                            )

                            # Guardar resultados
                            if eval_key not in task_cross_results:
                                task_cross_results[eval_key] = []

                            task_cross_results[eval_key].append(
                                {
                                    "experiment_name": experiment_name,
                                    "original_task": task_key,
                                    "target_evaluation": eval_key,
                                    "metrics": metrics,
                                    "model_config": model_result,
                                }
                            )

                except Exception as e:
                    print(f"❌ Error evaluando {experiment_name}: {e}")
                    continue

            all_cross_eval_results[task_key] = task_cross_results

        # Guardar resumen de evaluación cruzada
        cross_eval_summary_path = self.results_dir / "cross_evaluation_summary.json"
        with open(cross_eval_summary_path, "w") as f:
            json.dump(self._make_serializable(all_cross_eval_results), f, indent=2)

        print("\n🎉 Evaluación cruzada completada!")
        print(f"📋 Resumen guardado en: {cross_eval_summary_path}")

        return all_cross_eval_results

    def _find_trained_experiments(self) -> Dict:
        """Encuentra experimentos ya entrenados basándose en los resultados guardados."""
        results_file = self.results_dir / "intermediate_results.json"

        if not results_file.exists():
            print("⚠️  No se encontraron resultados de entrenamiento previos")
            return {}

        with open(results_file, "r") as f:
            return json.load(f)

    def _get_best_result(self, task_results: List[Dict]) -> Optional[Dict]:
        """Obtiene el mejor resultado de una tarea basándose en AUC ROC."""
        best_result = None
        best_auc = -1

        for result in task_results:
            if (
                "evaluation" in result
                and result["evaluation"]
                and "auc_roc" in result["evaluation"]
            ):
                auc = result["evaluation"]["auc_roc"]
                if isinstance(auc, (list, np.ndarray)):
                    auc = float(auc[0]) if len(auc) > 0 else -1
                else:
                    auc = float(auc)

                if auc > best_auc:
                    best_auc = auc
                    best_result = result

        return best_result

    def _make_serializable(self, obj):
        """Convierte objetos no serializables a formato JSON-compatible."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Para objetos complejos, intentar extraer solo atributos serializables
            return str(obj)
        else:
            return obj

    def _save_intermediate_results(self, results: Dict):
        """Guarda resultados intermedios."""
        results_file = self.results_dir / "intermediate_results.json"

        # Convertir resultados a formato serializable
        serializable_results = {}
        for task_key, task_results in results.items():
            serializable_results[task_key] = []
            for result in task_results:
                # Crear copia sin objetos no serializables
                clean_result = {
                    "task": self._make_serializable(result["task"]),
                    "hyperparams": self._make_serializable(result["hyperparams"]),
                    "run_id": result["run_id"],
                    "training_time": result["training_time"],
                    "config_path": result["config_path"],
                }

                # Agregar métricas de evaluación si existen
                if "evaluation" in result and result["evaluation"]:
                    clean_result["evaluation"] = self._make_serializable(result["evaluation"])

                # Agregar métricas de entrenamiento si existen
                if "training" in result and result["training"]:
                    clean_result["training"] = self._make_serializable(result["training"])

                serializable_results[task_key].append(clean_result)

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def _save_final_results(self, results: Dict):
        """Guarda resultados finales y genera resumen."""
        # Guardar resultados completos
        self._save_intermediate_results(results)

        # Generar resumen
        summary = self._generate_summary(results)
        summary_file = self.results_dir / "summary.json"

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n📊 Resultados guardados en:")
        print(f"   Completos: {self.results_dir / 'intermediate_results.json'}")
        print(f"   Resumen: {summary_file}")

    def _generate_summary(self, results: Dict) -> Dict:
        """Genera un resumen de los resultados."""
        summary = {
            "total_tasks": len(results),
            "total_runs": sum(len(task_results) for task_results in results.values()),
            "use_otsu_masking": self.use_otsu_masking,
            "tasks": {},
        }

        for task_key, task_results in results.items():
            if not task_results:
                continue

            # Encontrar mejor resultado por AUC ROC
            best_result = None
            best_auc = -1

            aucs = []
            for result in task_results:
                if (
                    "evaluation" in result
                    and result["evaluation"]
                    and "auc_roc" in result["evaluation"]
                ):
                    auc = result["evaluation"]["auc_roc"]
                    # Convertir a float si es numpy
                    if hasattr(auc, "item"):
                        auc = auc.item()
                    aucs.append(auc)
                    if auc > best_auc:
                        best_auc = auc
                        best_result = result

            if aucs:
                summary["tasks"][task_key] = {
                    "n_runs": len(task_results),
                    "best_auc": float(best_auc),
                    "mean_auc": float(sum(aucs) / len(aucs)),
                    "std_auc": float(
                        (sum((x - sum(aucs) / len(aucs)) ** 2 for x in aucs) / len(aucs)) ** 0.5
                    ),
                    "best_hyperparams": self._make_serializable(best_result["hyperparams"])
                    if best_result
                    else None,
                }

        return summary


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento en lote para tareas de clasificación binaria con búsqueda aleatoria"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Número de iteraciones de búsqueda aleatoria por tarea (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Número de épocas por entrenamiento (default: 50)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="ID de GPU a usar (default: usa cuda si está disponible)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet18", "inceptionv3", "vit", "swin_transformer"],
        help="Modelos a entrenar (default: todos)",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=["2d", "3d"],
        help="Dimensiones a entrenar (default: ambas)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./batch_results",
        help="Directorio para guardar resultados (default: ./batch_results)",
    )
    parser.add_argument(
        "--use_otsu_masking",
        action="store_true",
        default=True,
        help="Habilitar umbralizado de Otsu para máscara cerebral (default: True)",
    )
    parser.add_argument(
        "--no_otsu_masking",
        action="store_true",
        help="Deshabilitar umbralizado de Otsu",
    )
    parser.add_argument(
        "--cross_evaluate",
        action="store_true",
        help="Realizar evaluación cruzada después del entrenamiento",
    )
    parser.add_argument(
        "--cross_eval_datasets",
        nargs="+",
        default=["ADNI"],
        help="Datasets para evaluación cruzada (default: ADNI)",
    )
    parser.add_argument(
        "--cross_eval_classes",
        nargs="+",
        default=["CN_AD", "CN_MCI_AD"],
        help="Configuraciones de clases para evaluación cruzada (default: CN_AD CN_MCI_AD)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determinar si usar Otsu masking
    use_otsu = args.use_otsu_masking and not args.no_otsu_masking

    # Crear entrenador
    trainer = BinaryClassificationBatchTrainer(
        results_dir=args.results_dir, use_otsu_masking=use_otsu
    )

    # Ejecutar entrenamiento en lote
    results = trainer.run_batch_training(
        n_runs=args.n_runs,
        epochs=args.epochs,
        gpu_id=args.gpu,
        filter_models=args.models,
        filter_dimensions=args.dimensions,
    )

    print(f"\n✨ Proceso de entrenamiento completado! Revisa los resultados en {args.results_dir}")

    # Ejecutar evaluación cruzada si se solicitó
    if args.cross_evaluate:
        print("\n🔄 Iniciando evaluación cruzada...")
        cross_eval_results = trainer.cross_evaluate_trained_models(
            target_datasets=args.cross_eval_datasets,
            target_classes_list=args.cross_eval_classes,
            gpu_id=args.gpu,
            use_best_models_only=True,
        )

        if cross_eval_results:
            print("\n📊 Evaluación cruzada completada!")
            print(f"   Resultados guardados en: {args.results_dir}/cross_evaluation/")
        else:
            print("\n⚠️  No se pudo completar la evaluación cruzada")

    print(f"\n🎉 Proceso completo finalizado! Todos los resultados están en {args.results_dir}")
