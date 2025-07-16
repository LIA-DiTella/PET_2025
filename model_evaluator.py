# flake8: noqa
"""
Script para cargar modelos ya entrenados y evaluarlos en diferentes conjuntos de datos.

Características:
- Carga modelos desde checkpoints existentes
- Evaluación en conjuntos de datos diferentes al de entrenamiento
- Soporte para todos los modelos: ResNet18, InceptionV3, ViT, Swin Transformer
- Reutiliza configuraciones de experimentos anteriores
- Genera reportes detallados de evaluación cruzada
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar módulos propios
from data.dataset import get_test_data_loader
from models.inceptionv3.inceptionv3_2d import get_inceptionv3_2d
from models.inceptionv3.inceptionv3_3d import get_inceptionv3_3d
from models.resnet18.resnet18_2d import get_resnet18_2d
from models.resnet18.resnet18_3d import get_resnet18_3d
from models.swintransformer.swin_transformer_2d import get_swin_transformer_2d
from models.swintransformer.swin_transformer_3d import get_swin_transformer_3d
from models.vit.vit_2d import get_vit_2d
from models.vit.vit_3d import get_vit_3d
from utils.config_utils import load_config
from utils.evaluation_utils import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_results_to_csv,
)


class ModelEvaluator:
    """Clase para evaluar modelos preentrenados en diferentes conjuntos de datos."""

    def __init__(self, device: Optional[torch.device] = None):
        """Inicializa el evaluador de modelos.

        Args:
            device: Dispositivo para evaluación (CPU/GPU)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"🔧 Inicializando evaluador en dispositivo: {self.device}")

    def get_model(self, model_name: str, dimension: str, config: Dict) -> torch.nn.Module:
        """Crea una instancia del modelo según los parámetros.

        Args:
            model_name: Nombre del modelo (resnet18, inceptionv3, vit, swin_transformer)
            dimension: Dimensión del modelo (2d, 3d)
            config: Configuración del modelo

        Returns:
            Modelo instanciado
        """
        model_name = model_name.lower()
        dimension = dimension.lower()

        if model_name == "resnet18":
            if dimension == "2d":
                return get_resnet18_2d(config)
            return get_resnet18_3d(config)

        elif model_name == "inceptionv3":
            if dimension == "2d":
                return get_inceptionv3_2d(config)
            return get_inceptionv3_3d(config)

        elif model_name == "vit":
            if dimension == "2d":
                return get_vit_2d(config)
            return get_vit_3d(config)

        elif model_name == "swin_transformer":
            if dimension == "2d":
                return get_swin_transformer_2d(config)
            return get_swin_transformer_3d(config)

        else:
            raise ValueError(f"Modelo {model_name} no soportado")

    def load_model_from_checkpoint(
        self, checkpoint_path: str, config_path: str
    ) -> Tuple[torch.nn.Module, Dict]:
        """Carga un modelo desde checkpoint y su configuración.

        Args:
            checkpoint_path: Ruta al archivo de checkpoint
            config_path: Ruta al archivo de configuración YAML

        Returns:
            Tupla con (modelo_cargado, configuración)
        """
        print(f"📂 Cargando modelo desde: {checkpoint_path}")
        print(f"📂 Cargando configuración desde: {config_path}")

        # Cargar configuración
        config = load_config(config_path)

        # Obtener parámetros del modelo
        model_config = config.get("model", {})
        model_name = model_config.get("name", "resnet18").lower()
        dimension = model_config.get("dimension", "2d").lower()

        # Determinar número de clases
        train_classes = config.get("data", {}).get("classes", "CN_AD")
        num_classes = len(train_classes.split("_"))
        model_config["num_classes"] = num_classes

        # Crear modelo
        model = self.get_model(model_name, dimension, model_config)
        model.to(self.device)

        # Cargar pesos del modelo
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Checkpoint cargado - Época: {checkpoint.get('epoch', 'N/A')}")
            if "best_val_auc" in checkpoint:
                print(f"   AUC de validación: {checkpoint['best_val_auc']:.4f}")
        else:
            # Checkpoint solo contiene el estado del modelo
            model.load_state_dict(checkpoint)
            print("✅ Estado del modelo cargado")

        model.eval()
        return model, config

    def evaluate_on_dataset(
        self,
        model: torch.nn.Module,
        original_config: Dict,
        target_dataset: str,
        target_classes: str,
        target_dimension: Optional[str] = None,
        use_otsu_masking: Optional[bool] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """Evalúa un modelo en un conjunto de datos específico.

        Args:
            model: Modelo ya cargado
            original_config: Configuración original del modelo
            target_dataset: Nombre del dataset objetivo (ej: "ADNI", "FLENI")
            target_classes: Clases objetivo (ej: "CN_AD", "CN_MCI_AD")
            target_dimension: Dimensión objetivo (2d/3d). Si None, usa la original
            use_otsu_masking: Si usar Otsu masking. Si None, usa configuración original
            output_dir: Directorio para guardar resultados

        Returns:
            Diccionario con métricas de evaluación
        """
        print("\n🎯 Evaluando en:")
        print(f"   Dataset: {target_dataset}")
        print(f"   Clases: {target_classes}")

        # Crear configuración para el dataset objetivo
        eval_config = original_config.copy()

        # Actualizar configuración de datos
        eval_config["data"]["dataset_name"] = target_dataset
        eval_config["data"]["classes"] = target_classes

        if target_dimension is not None:
            eval_config["data"]["dimension"] = target_dimension

        if use_otsu_masking is not None:
            eval_config["data"]["use_otsu_masking"] = use_otsu_masking

        try:
            # Obtener solo el test loader específico
            test_loader = get_test_data_loader(eval_config, "data")

            if test_loader is None:
                raise ValueError("No se pudo crear el test loader")

            print(f"📊 Datos cargados - {len(test_loader.dataset)} muestras de test")

        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return {"error": str(e)}

        # Evaluar el modelo
        all_targets = []
        all_predictions = []
        all_scores = []

        print("🔍 Evaluando modelo...")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = model(data)
                scores = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

                if batch_idx % 10 == 0:
                    print(f"   Procesado: {batch_idx + 1}/{len(test_loader)} batches")

        # Calcular métricas
        try:
            metrics = calculate_metrics(all_targets, all_predictions, all_scores)
            print(f"✅ Evaluación completada - AUC: {metrics.get('auc_roc', 'N/A'):.4f}")
        except Exception as e:
            print(f"❌ Error calculando métricas: {e}")
            return {"error": f"Error calculando métricas: {str(e)}"}

        # Guardar resultados si se especifica directorio
        if output_dir:
            self._save_evaluation_results(
                metrics,
                all_targets,
                all_scores,
                target_dataset,
                target_classes,
                output_dir,
                original_config,
            )

        return metrics

    def evaluate_on_multiple_datasets(
        self, model: torch.nn.Module, original_config: Dict, output_dir: Optional[str] = None
    ) -> Dict:
        """Evalúa un modelo en múltiples configuraciones de datos definidas en el config.

        Args:
            model: Modelo ya cargado
            original_config: Configuración original que puede contener data, data2, data3, etc.
            output_dir: Directorio para guardar resultados

        Returns:
            Diccionario con resultados para cada configuración de datos
        """
        print("\n🔍 Buscando configuraciones de datos adicionales...")

        # Encontrar todas las configuraciones de datos (data, data2, data3, etc.)
        data_configs = {}
        for key, value in original_config.items():
            if key.startswith("data") and isinstance(value, dict):
                data_configs[key] = value
                print(f"   Encontrada configuración: {key} - {value.get('dataset_name', 'N/A')}")

        if not data_configs:
            print("⚠️  No se encontraron configuraciones de datos")
            return {}

        all_results = {}

        for config_name, data_config in data_configs.items():
            print(f"\n📊 Evaluando configuración: {config_name}")

            # Crear configuración temporal para esta evaluación
            temp_config = original_config.copy()
            temp_config["data"] = data_config

            dataset_name = data_config.get("dataset_name", config_name)
            classes = data_config.get("classes", "CN_AD")

            try:
                # Obtener solo el test loader para esta configuración
                test_loader = get_test_data_loader(temp_config, "data")

                if test_loader is None:
                    print(f"⚠️  No se pudo crear test loader para {config_name}")
                    continue

                print(f"   📈 Datos cargados - {len(test_loader.dataset)} muestras")

                # Evaluar el modelo
                all_targets = []
                all_predictions = []
                all_scores = []

                print(f"   🔍 Evaluando en {dataset_name}...")
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = data.to(self.device), target.to(self.device)

                        # Forward pass
                        output = model(data)
                        scores = torch.softmax(output, dim=1)
                        predictions = torch.argmax(output, dim=1)

                        all_targets.extend(target.cpu().numpy())
                        all_predictions.extend(predictions.cpu().numpy())
                        all_scores.extend(scores.cpu().numpy())

                        if batch_idx % 10 == 0 and batch_idx > 0:
                            print(f"      Procesado: {batch_idx + 1}/{len(test_loader)} batches")

                # Calcular métricas
                metrics = calculate_metrics(all_targets, all_predictions, all_scores)
                print(f"   ✅ {config_name} completado - AUC: {metrics.get('auc_roc', 'N/A'):.4f}")

                # Agregar metadatos
                metrics["config_name"] = config_name
                metrics["dataset_name"] = dataset_name
                metrics["classes"] = classes
                metrics["num_samples"] = len(all_targets)

                all_results[config_name] = metrics

                # Guardar resultados individuales si se especifica directorio
                if output_dir:
                    config_output_dir = os.path.join(output_dir, config_name)
                    self._save_evaluation_results(
                        metrics,
                        all_targets,
                        all_scores,
                        dataset_name,
                        classes,
                        config_output_dir,
                        temp_config,
                    )

            except Exception as e:
                print(f"❌ Error evaluando {config_name}: {e}")
                all_results[config_name] = {"error": str(e)}
                import traceback

                traceback.print_exc()
                continue

        # Guardar resumen comparativo
        if output_dir and all_results:
            self._save_multi_dataset_summary(all_results, output_dir, original_config)

        return all_results

    def load_all_evaluation_datasets(self, original_config: Dict) -> Dict:
        """Carga todos los datasets de evaluación al inicio.
        
        Args:
            original_config: Configuración que puede contener data, data2, data3, etc.
            
        Returns:
            Dict con test loaders pre-cargados para cada configuración
        """
        print("\n🔄 Cargando todos los datasets de evaluación...")

        # Encontrar todas las configuraciones de datos (data, data2, data3, etc.)
        data_configs = {}
        for key, value in original_config.items():
            if key.startswith("data") and isinstance(value, dict):
                data_configs[key] = value
                print(f"   📊 Encontrada configuración: {key} - {value.get('dataset_name', 'N/A')}")

        if not data_configs:
            print("⚠️  No se encontraron configuraciones de datos")
            return {}

        test_loaders_cache = {}

        for config_name, data_config in data_configs.items():
            dataset_name = data_config.get("dataset_name", config_name)

            try:
                # Crear configuración temporal para esta evaluación
                temp_config = original_config.copy()
                temp_config["data"] = data_config

                # Obtener solo el test loader para esta configuración
                print("   📈 Cargando test loader para:", dataset_name)
                test_loader = get_test_data_loader(temp_config, "data")

                if test_loader is not None:
                    test_loaders_cache[config_name] = test_loader
                    print(f"      ✅ {dataset_name}: {len(test_loader.dataset)} muestras")
                else:
                    print(f"      ❌ {dataset_name}: No se pudo crear test loader")
                    test_loaders_cache[config_name] = None

            except Exception as e:
                print(f"      ❌ Error cargando {dataset_name}: {e}")
                test_loaders_cache[config_name] = None

        loaded_count = len([k for k, v in test_loaders_cache.items() if v is not None])
        print(f"✅ Datasets de evaluación cargados: {loaded_count}/{len(data_configs)}")

        return test_loaders_cache

    def evaluate_on_multiple_datasets_optimized(
        self,
        model: torch.nn.Module,
        original_config: Dict,
        test_loaders_cache: Optional[Dict] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Evalúa un modelo en múltiples datasets usando test loaders pre-cargados.

        Args:
            model: Modelo ya cargado
            original_config: Configuración original
            test_loaders_cache: Dict con test loaders pre-cargados. Si None, se cargan
            output_dir: Directorio para guardar resultados

        Returns:
            Diccionario con resultados para cada configuración de datos
        """
        # Si no se proporcionó cache, cargar datasets
        if test_loaders_cache is None:
            test_loaders_cache = self.load_all_evaluation_datasets(original_config)

        if not test_loaders_cache:
            print("⚠️  No hay datasets disponibles para evaluación")
            return {}

        all_results = {}

        for config_name, test_loader in test_loaders_cache.items():
            if test_loader is None:
                continue

            print(f"\n📊 Evaluando configuración: {config_name}")

            # Obtener información del dataset desde la configuración original
            data_config = original_config.get(config_name, {})
            dataset_name = data_config.get("dataset_name", config_name)

            print(f"   📈 Usando test loader pre-cargado - {len(test_loader.dataset)} muestras")

            try:
                # Evaluar el modelo usando el test loader pre-cargado
                all_targets = []
                all_predictions = []
                all_scores = []

                print(f"   🔍 Evaluando en {dataset_name}...")
                start_time = time.time()

                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(test_loader):
                        data, target = data.to(self.device), target.to(self.device)

                        # Hacer predicciones
                        outputs = model(data)

                        # Aplicar softmax para obtener probabilidades
                        probabilities = torch.softmax(outputs, dim=1)

                        # Obtener predicciones (clase con mayor probabilidad)
                        _, predicted = torch.max(outputs, 1)

                        # Guardar resultados - asegurar que todos sean listas de elementos individuales
                        targets_batch = target.cpu().numpy().tolist()
                        predictions_batch = predicted.cpu().numpy().tolist()

                        all_targets.extend(targets_batch)
                        all_predictions.extend(predictions_batch)

                        # Debug: verificar longitudes cada 10 batches
                        if (batch_idx + 1) % 10 == 0:
                            print(f"      Batch {batch_idx + 1}: targets_batch={len(targets_batch)}, all_targets={len(all_targets)}, all_predictions={len(all_predictions)}")

                        # Para problemas binarios, usar probabilidad de la clase positiva
                        if probabilities.shape[1] == 2:
                            scores_batch = probabilities[:, 1].cpu().numpy().tolist()
                            all_scores.extend(scores_batch)
                        else:
                            # Para multiclase, usar probabilidad máxima
                            max_probs, _ = torch.max(probabilities, 1)
                            scores_batch = max_probs.cpu().numpy().tolist()
                            all_scores.extend(scores_batch)

                        if (batch_idx + 1) % 50 == 0:
                            print(f"      Procesados {batch_idx + 1}/{len(test_loader)} batches")

                eval_time = time.time() - start_time
                print(f"   ⏱️  Evaluación completada en {eval_time:.1f}s")

                # Calcular métricas
                metrics = calculate_metrics(
                    np.array(all_targets),
                    np.array(all_predictions),
                    np.array(all_scores)
                )

                print(f"   📊 Resultado: Accuracy={metrics.get('accuracy', 0):.4f}, AUC={metrics.get('auc_roc', 0):.4f}")

                # Guardar resultados si se especifica directorio
                if output_dir:
                    result_dir = Path(output_dir) / f"eval_{config_name}"
                    result_dir.mkdir(parents=True, exist_ok=True)

                    save_results_to_csv(
                        all_targets, all_predictions, all_scores, str(result_dir / "results.csv")
                    )

                    # Guardar gráficos si es clasificación binaria
                    # if len(set(all_targets)) == 2:
                    # TypeError: unhashable type: 'list'
                    print(all_targets)
                    if len(all_targets) <= 2:  # Asegurar que sea binaria o multiclase
                        plot_roc_curve(all_targets, all_scores, str(result_dir / "roc_curve.png"))
                        plot_confusion_matrix(
                            all_targets, all_predictions, str(result_dir / "confusion_matrix.png")
                        )

                all_results[config_name] = metrics

            except Exception as e:
                print(f"   ❌ Error evaluando {config_name}: {e}")
                all_results[config_name] = {"error": str(e)}

                # print stack trace for debugging
                import traceback
                traceback.print_exc()

        return all_results

    def _save_evaluation_results(
        self,
        metrics: Dict,
        targets: List,
        scores: List,
        dataset_name: str,
        classes: str,
        output_dir: str,
        config: Dict,
    ):
        """Guarda los resultados de evaluación."""
        os.makedirs(output_dir, exist_ok=True)

        # Guardar métricas
        results_file = os.path.join(output_dir, f"{dataset_name}_{classes}_metrics.json")
        with open(results_file, "w") as f:
            json.dump(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()},
                f,
                indent=2,
            )

        # Guardar CSV con métricas
        csv_file = os.path.join(output_dir, f"{dataset_name}_{classes}_metrics.csv")
        save_results_to_csv(metrics, csv_file)

        # Generar visualizaciones
        class_names = classes.split("_")

        # Matriz de confusión
        if "confusion_matrix" in metrics:
            cm_path = os.path.join(output_dir, f"{dataset_name}_{classes}_confusion_matrix.png")
            plot_confusion_matrix(metrics["confusion_matrix"], class_names, save_path=cm_path)

        # Curva ROC (solo para clasificación binaria)
        if len(class_names) == 2 and "auc_roc" in metrics:
            roc_path = os.path.join(output_dir, f"{dataset_name}_{classes}_roc_curve.png")
            plot_roc_curve(targets, scores, class_names, save_path=roc_path)

        print(f"💾 Resultados guardados en: {output_dir}")

    def _save_multi_dataset_summary(self, results: Dict, output_dir: str, original_config: Dict):
        """Guarda un resumen comparativo de evaluación en múltiples datasets."""
        os.makedirs(output_dir, exist_ok=True)

        # Crear resumen comparativo
        summary = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "name": original_config.get("model", {}).get("name", "unknown"),
                "dimension": original_config.get("model", {}).get("dimension", "unknown"),
                "original_training_data": original_config.get("data", {}).get(
                    "dataset_name", "unknown"
                ),
            },
            "results_summary": {},
            "detailed_results": results,
        }

        # Crear tabla comparativa
        comparison_table = []
        for config_name, metrics in results.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                row = {
                    "Configuration": config_name,
                    "Dataset": metrics.get("dataset_name", "N/A"),
                    "Classes": metrics.get("classes", "N/A"),
                    "Samples": metrics.get("num_samples", "N/A"),
                    "AUC": f"{metrics.get('auc_roc', 0):.4f}" if "auc_roc" in metrics else "N/A",
                    "Accuracy": f"{metrics.get('accuracy', 0):.4f}"
                    if "accuracy" in metrics
                    else "N/A",
                    "Sensitivity": f"{metrics.get('sensitivity', 0):.4f}"
                    if "sensitivity" in metrics
                    else "N/A",
                    "Specificity": f"{metrics.get('specificity', 0):.4f}"
                    if "specificity" in metrics
                    else "N/A",
                }
                comparison_table.append(row)

        summary["comparison_table"] = comparison_table

        # Guardar resumen en JSON
        summary_file = os.path.join(output_dir, "multi_dataset_evaluation_summary.json")
        with open(summary_file, "w") as f:
            json.dump(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summary.items()},
                f,
                indent=2,
            )

        # Guardar tabla comparativa en CSV
        if comparison_table:
            import csv

            csv_file = os.path.join(output_dir, "multi_dataset_comparison.csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=comparison_table[0].keys())
                writer.writeheader()
                writer.writerows(comparison_table)

        print("📋 Resumen comparativo guardado en:")
        print(f"   JSON: {summary_file}")
        print(f"   CSV: {csv_file}")

    def cross_evaluate_experiment(
        self,
        experiment_dir: str,
        target_datasets: List[str],
        target_classes_list: List[str],
        output_dir: str,
    ) -> Dict:
        """Evalúa un experimento completo en múltiples conjuntos de datos.

        Args:
            experiment_dir: Directorio del experimento (contiene config.yaml y checkpoints/)
            target_datasets: Lista de datasets objetivo
            target_classes_list: Lista de configuraciones de clases objetivo
            output_dir: Directorio para guardar resultados

        Returns:
            Diccionario con todos los resultados de evaluación cruzada
        """
        experiment_path = Path(experiment_dir)
        config_path = experiment_path / "config.yaml"
        checkpoints_dir = experiment_path / "checkpoints"

        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró config.yaml en {experiment_dir}")

        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No se encontró directorio checkpoints en {experiment_dir}")

        # Buscar el mejor checkpoint
        best_checkpoint = None
        for checkpoint_file in checkpoints_dir.glob("*.pth"):
            if "best" in checkpoint_file.name.lower():
                best_checkpoint = checkpoint_file
                break

        if best_checkpoint is None:
            # Usar el último checkpoint disponible
            checkpoints = list(checkpoints_dir.glob("*.pth"))
            if checkpoints:
                best_checkpoint = max(checkpoints, key=os.path.getctime)
            else:
                raise FileNotFoundError(f"No se encontraron checkpoints en {checkpoints_dir}")

        print(f"🏆 Usando checkpoint: {best_checkpoint}")

        # Cargar modelo
        model, original_config = self.load_model_from_checkpoint(
            str(best_checkpoint), str(config_path)
        )

        # Evaluar en todos los conjuntos de datos objetivo
        all_results = {}

        for target_dataset in target_datasets:
            for target_classes in target_classes_list:
                eval_key = f"{target_dataset}_{target_classes}"
                print(f"\n{'=' * 60}")
                print(f"Evaluando en: {eval_key}")
                print(f"{'=' * 60}")

                eval_output_dir = os.path.join(output_dir, eval_key)

                metrics = self.evaluate_on_dataset(
                    model=model,
                    original_config=original_config,
                    target_dataset=target_dataset,
                    target_classes=target_classes,
                    output_dir=eval_output_dir,
                )

                all_results[eval_key] = metrics

        # Guardar resumen de evaluación cruzada
        summary_path = os.path.join(output_dir, "cross_evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    k: (
                        {
                            key: (val.tolist() if isinstance(val, np.ndarray) else val)
                            for key, val in v.items()
                        }
                        if isinstance(v, dict)
                        else v
                    )
                    for k, v in all_results.items()
                },
                f,
                indent=2,
            )

        print("\n🎉 Evaluación cruzada completada!")
        print(f"📋 Resumen guardado en: {summary_path}")

        return all_results


def find_experiments(base_dir: str, pattern: str = "*") -> List[Path]:
    """Encuentra directorios de experimentos que contienen config.yaml y checkpoints/."""
    base_path = Path(base_dir)
    experiments = []

    for exp_dir in base_path.glob(pattern):
        if exp_dir.is_dir():
            config_file = exp_dir / "config.yaml"
            checkpoints_dir = exp_dir / "checkpoints"

            if config_file.exists() and checkpoints_dir.exists():
                experiments.append(exp_dir)

    return experiments


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evaluador de modelos preentrenados en diferentes conjuntos de datos"
    )

    parser.add_argument(
        "--mode",
        choices=["single", "cross", "batch", "multi_data"],
        default="single",
        help="Modo de evaluación: single (un modelo), cross (evaluación cruzada), batch (múltiples experimentos), multi_data (múltiples configuraciones de datos)",
    )

    # Argumentos para modo single
    parser.add_argument(
        "--checkpoint", type=str, help="Ruta al checkpoint del modelo (modo single)"
    )
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración (modo single)")

    # Argumentos para modo cross/batch
    parser.add_argument(
        "--experiment_dir", type=str, help="Directorio del experimento (modo cross)"
    )
    parser.add_argument(
        "--experiments_base_dir",
        type=str,
        default="./experiments",
        help="Directorio base con experimentos (modo batch)",
    )

    # Argumentos de evaluación
    parser.add_argument(
        "--target_datasets", nargs="+", default=["ADNI"], help="Datasets objetivo para evaluación"
    )
    parser.add_argument(
        "--target_classes",
        nargs="+",
        default=["CN_AD", "CN_MCI_AD"],
        help="Configuraciones de clases objetivo",
    )
    parser.add_argument(
        "--use_otsu_masking", action="store_true", help="Usar Otsu masking en evaluación"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./cross_evaluation_results",
        help="Directorio para guardar resultados",
    )
    parser.add_argument("--gpu", type=int, default=None, help="ID de GPU a usar")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configurar dispositivo
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear evaluador
    evaluator = ModelEvaluator(device=device)

    if args.mode == "single":
        if not args.checkpoint or not args.config:
            print("❌ Para modo single, se requieren --checkpoint y --config")
            sys.exit(1)

        # Cargar modelo
        model, config = evaluator.load_model_from_checkpoint(args.checkpoint, args.config)

        # Evaluar en cada combinación de dataset y clases
        for target_dataset in args.target_datasets:
            for target_classes in args.target_classes:
                output_subdir = os.path.join(args.output_dir, f"{target_dataset}_{target_classes}")
                metrics = evaluator.evaluate_on_dataset(
                    model=model,
                    original_config=config,
                    target_dataset=target_dataset,
                    target_classes=target_classes,
                    use_otsu_masking=args.use_otsu_masking if args.use_otsu_masking else None,
                    output_dir=output_subdir,
                )

    elif args.mode == "cross":
        if not args.experiment_dir:
            print("❌ Para modo cross, se requiere --experiment_dir")
            sys.exit(1)

        # Evaluación cruzada de un experimento
        results = evaluator.cross_evaluate_experiment(
            experiment_dir=args.experiment_dir,
            target_datasets=args.target_datasets,
            target_classes_list=args.target_classes,
            output_dir=args.output_dir,
        )

    elif args.mode == "multi_data":
        if not args.checkpoint or not args.config:
            print("❌ Para modo multi_data, se requieren --checkpoint y --config")
            sys.exit(1)

        # Cargar modelo
        model, config = evaluator.load_model_from_checkpoint(args.checkpoint, args.config)

        # Evaluar en todas las configuraciones de datos definidas en el config
        results = evaluator.evaluate_on_multiple_datasets(
            model=model, original_config=config, output_dir=args.output_dir
        )

        print("\n📊 Evaluación en múltiples datasets completada!")
        for config_name, metrics in results.items():
            if isinstance(metrics, dict) and "auc_roc" in metrics:
                print(f"   {config_name}: AUC = {metrics['auc_roc']:.4f}")

    elif args.mode == "batch":
        # Evaluación en lote de múltiples experimentos
        experiments = find_experiments(args.experiments_base_dir)

        if not experiments:
            print(f"❌ No se encontraron experimentos en {args.experiments_base_dir}")
            sys.exit(1)

        print(f"🔍 Encontrados {len(experiments)} experimentos")

        for i, exp_dir in enumerate(experiments):
            print(f"\n🏃 Procesando experimento {i + 1}/{len(experiments)}: {exp_dir.name}")

            exp_output_dir = os.path.join(args.output_dir, exp_dir.name)

            try:
                results = evaluator.cross_evaluate_experiment(
                    experiment_dir=str(exp_dir),
                    target_datasets=args.target_datasets,
                    target_classes_list=args.target_classes,
                    output_dir=exp_output_dir,
                )
                print(f"✅ Experimento {exp_dir.name} completado")
            except Exception as e:
                print(f"❌ Error en experimento {exp_dir.name}: {e}")
                continue

        print(f"\n🎉 Evaluación en lote completada! Resultados en: {args.output_dir}")
