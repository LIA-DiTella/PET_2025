"""
Script para evaluar automáticamente los mejor        ]

    def find_all_experiments(self) -> List[Dict]:renados en múltiples conjuntos de datos.

Este script:
1. Busca todos los experimentos válidos en el directorio de experimentos
2. Selecciona el mejor modelo para cada combinación (modelo, dimensión) basado en métricas
3. Evalúa cada mejor modelo en todos los conjuntos de prueba disponibles (ADNI, FLENI100, FLENI600)
4. Genera un reporte completo con métricas de evaluación
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import yaml

from model_evaluator import ModelEvaluator
from utils.config_utils import load_config


class BestModelMultiEvaluator:
    """Evaluador automático de mejores modelos en múltiples conjuntos de datos."""

    def __init__(
        self,
        batch_results_dir: str = "./batch_results",
        experiments_dir: str = "./experiments",
        output_dir: str = "./best_models_evaluation",
    ):
        self.batch_results_dir = Path(batch_results_dir)
        self.experiments_dir = Path(experiments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = ModelEvaluator(device=self.device)

        # Conjuntos de datos de prueba disponibles
        self.test_datasets = [
            {"name": "ADNI", "config_key": "data"},
            {"name": "FLENI100", "config_key": "data2"},
            {"name": "FLENI600", "config_key": "data3"},
        ]

    def load_batch_results(self) -> Dict:
        """Carga los resultados del entrenamiento por lotes."""
        results_file = self.batch_results_dir / "intermediate_results.json"

        if not results_file.exists():
            print(f"❌ No se encontraron resultados de entrenamiento en {results_file}")
            return {}

        with open(results_file, "r") as f:
            return json.load(f)

    def find_best_models_per_combination(self, experiments: List[Dict]) -> Dict:
        """Encuentra el mejor modelo para cada combinación (modelo, dimensión).

        Args:
            experiments: Lista de experimentos encontrados

        Returns:
            Dict con la estructura: {model_dim_key: best_experiment_info}
        """
        best_models = {}

        for experiment in experiments:
            model = experiment["model"]
            dimension = experiment["dimension"]
            metrics = experiment["metrics"]

            # Saltar si no hay métricas válidas
            if not metrics:
                continue

            # Clave para agrupar por modelo y dimensión
            model_dim_key = f"{model}_{dimension}"

            # Intentar obtener AUC de diferentes posibles nombres de columna
            auc = -1
            auc_keys = ["auc_roc", "auc", "roc_auc", "test_auc", "val_auc"]

            for key in auc_keys:
                if key in metrics and metrics[key] is not None:
                    try:
                        auc_val = float(metrics[key])
                        if auc_val > auc:
                            auc = auc_val
                    except (ValueError, TypeError):
                        continue

            # Si no encontramos AUC, usar accuracy como métrica alternativa
            if auc == -1:
                acc_keys = ["accuracy", "test_accuracy", "val_accuracy", "acc"]
                for key in acc_keys:
                    if key in metrics and metrics[key] is not None:
                        try:
                            auc = float(metrics[key])
                            break
                        except (ValueError, TypeError):
                            continue

            # Saltar si no pudimos obtener ninguna métrica útil
            if auc == -1:
                continue

            # Actualizar el mejor modelo para esta combinación si es mejor que el actual
            if model_dim_key not in best_models or auc > best_models[model_dim_key]["auc"]:
                best_models[model_dim_key] = {
                    "experiment": experiment,
                    "auc": auc,
                    "model": model,
                    "dimension": dimension,
                }

        return best_models

    def find_all_experiments(self) -> List[Dict]:
        """Busca todos los experimentos válidos en el directorio de experimentos.

        Returns:
            Lista de diccionarios con información de cada experimento
        """
        experiments = []

        if not self.experiments_dir.exists():
            print(f"❌ Directorio de experimentos no encontrado: {self.experiments_dir}")
            return experiments

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # Verificar si tiene estructura válida
            config_file = exp_dir / "config.yaml"
            checkpoint_file = exp_dir / "checkpoints" / "best_model.pth"
            metrics_file = exp_dir / "results" / "test_metrics.csv"

            if not (config_file.exists() and checkpoint_file.exists()):
                continue

            # Extraer información del nombre del experimento
            exp_name = exp_dir.name
            exp_info = self.parse_experiment_name(exp_name)

            # Leer métricas si existen
            metrics = {}
            if metrics_file.exists():
                try:
                    import pandas as pd

                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        # Tomar la última fila (mejores métricas)
                        metrics = df.iloc[-1].to_dict()
                except Exception as e:
                    print(f"⚠️  Error leyendo métricas de {exp_name}: {e}")

            experiment = {
                "name": exp_name,
                "path": exp_dir,
                "config_path": config_file,
                "checkpoint_path": checkpoint_file,
                "metrics_path": metrics_file,
                "model": exp_info.get("model", "unknown"),
                "dimension": exp_info.get("dimension", "unknown"),
                "metrics": metrics,
            }

            experiments.append(experiment)

        return experiments

    def parse_experiment_name(self, exp_name: str) -> Dict:
        """Extrae información del nombre del experimento.

        Args:
            exp_name: Nombre del directorio del experimento

        Returns:
            Dict con model, dimension, etc.
        """
        # Buscar patrones comunes
        model = "unknown"
        dimension = "unknown"

        # Intentar extraer modelo y dimensión
        if "resnet18" in exp_name.lower():
            model = "resnet18"
        elif "inceptionv3" in exp_name.lower():
            model = "inceptionv3"
        elif "vit" in exp_name.lower():
            model = "vit"
        elif "swin" in exp_name.lower():
            model = "swin"

        if "2d" in exp_name.lower():
            dimension = "2d"
        elif "3d" in exp_name.lower():
            dimension = "3d"

        return {"model": model, "dimension": dimension}

    def create_multi_eval_config(self, original_config_path: str) -> str:
        """Crea una configuración con múltiples conjuntos de datos para evaluación.

        Args:
            original_config_path: Ruta a la configuración original

        Returns:
            Ruta a la nueva configuración con múltiples datasets
        """
        original_config = load_config(original_config_path)

        # Crear nueva configuración con datasets adicionales
        multi_config = original_config.copy()

        # Mantener ADNI como data (configuración original)
        # multi_config["data"] ya está configurado

        # Configurar FLENI100 como data2
        multi_config["data2"] = {
            "dataset_name": "FLENI",
            "data_dir": "/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/converted_niftis/",
            "test_csv": "/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni100/fleni100.csv",
            "classes": original_config["data"]["classes"],
            "dimension": original_config["data"]["dimension"],
            "batch_size": original_config["data"]["batch_size"],
            "num_workers": original_config["data"].get("num_workers", 4),
            "slice_selection": original_config["data"].get("slice_selection", "uniform"),
            "num_slices": original_config["data"].get("num_slices", 16),
            "output_size": original_config["data"].get("output_size", (224, 224)),
            "channels": original_config["data"].get("channels", 3),
            "use_otsu_masking": original_config["data"].get("use_otsu_masking", True),
        }

        # Configurar FLENI600 como data3
        multi_config["data3"] = {
            "dataset_name": "FLENI",
            "data_dir": "/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/converted_niftis/",
            "test_csv": "/home/ipardo/storage1/PET_2025/data/NIFTIs/fleni600/fleni600.csv",
            "classes": original_config["data"]["classes"],
            "dimension": original_config["data"]["dimension"],
            "batch_size": original_config["data"]["batch_size"],
            "num_workers": original_config["data"].get("num_workers", 4),
            "slice_selection": original_config["data"].get("slice_selection", "uniform"),
            "num_slices": original_config["data"].get("num_slices", 16),
            "output_size": original_config["data"].get("output_size", (224, 224)),
            "channels": original_config["data"].get("channels", 3),
            "use_otsu_masking": original_config["data"].get("use_otsu_masking", True),
        }

        # Guardar configuración temporal
        temp_config_path = self.output_dir / "temp_multi_eval_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(multi_config, f)

        return str(temp_config_path)

    def load_all_evaluation_datasets_for_multi_config(self, multi_config_path: str) -> Dict:
        """Carga todos los datasets de evaluación usando configuración multi-dataset.
        
        Args:
            multi_config_path: Ruta a configuración que contiene data, data2, data3
            
        Returns:
            Dict con test loaders pre-cargados
        """
        try:
            # Cargar configuración multi-dataset
            with open(multi_config_path, 'r') as f:
                import yaml
                multi_config = yaml.safe_load(f)
            
            return self.evaluator.load_all_evaluation_datasets(multi_config)
        except Exception as e:
            print(f"❌ Error cargando datasets de evaluación: {e}")
            return {}

    def evaluate_model_on_all_datasets(self, model_info: Dict) -> Dict:
        """Evalúa un modelo en todos los conjuntos de datos disponibles.

        Args:
            model_info: Información del modelo con experimento, checkpoint, etc.

        Returns:
            Dict con resultados de evaluación en todos los datasets
        """
        experiment = model_info["experiment"]
        config_path = str(experiment["config_path"])
        checkpoint_path = str(experiment["checkpoint_path"])

        if not os.path.exists(checkpoint_path):
            print(f"❌ No se encontró checkpoint: {checkpoint_path}")
            return {}

        print(f"🔍 Evaluando {model_info['model']}_{model_info['dimension']}:")
        print(f"   Experimento: {experiment['name']}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Métrica original: {model_info['auc']:.4f}")

        # Crear configuración con múltiples datasets
        multi_config_path = self.create_multi_eval_config(config_path)

        try:
            # Cargar modelo
            model, original_config = self.evaluator.load_model_from_checkpoint(
                checkpoint_path, multi_config_path
            )

            # Evaluar en todos los datasets usando el método integrate multiple datasets
            evaluation_results = self.evaluator.evaluate_on_multiple_datasets(
                model=model,
                original_config=original_config,
                output_dir=str(
                    self.output_dir / f"{model_info['model']}_{model_info['dimension']}"
                ),
            )

            # Mapear los nombres de las claves de configuración a nombres de datasets
            config_to_dataset = {"data": "ADNI", "data2": "FLENI100", "data3": "FLENI600"}

            # Transformar las claves de los resultados
            final_results = {}
            for config_key, metrics in evaluation_results.items():
                dataset_name = config_to_dataset.get(config_key, config_key)
                final_results[dataset_name] = metrics
                if "error" not in metrics:
                    print(
                        f"   ✅ {dataset_name}: Accuracy={metrics.get('accuracy', 0):.4f}, AUC={metrics.get('auc_roc', 0):.4f}"
                    )
                else:
                    print(f"   ❌ {dataset_name}: {metrics['error']}")

            return final_results

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return {}
        finally:
            # Limpiar archivo temporal
            if os.path.exists(multi_config_path):
                os.remove(multi_config_path)

    def evaluate_model_on_all_datasets_optimized(self, model_info: Dict) -> Dict:
        """Evalúa un modelo en todos los conjuntos de datos disponibles de forma optimizada.

        Args:
            model_info: Información del modelo con experimento, checkpoint, etc.

        Returns:
            Dict con resultados de evaluación en todos los datasets
        """
        experiment = model_info["experiment"]
        config_path = str(experiment["config_path"])
        checkpoint_path = str(experiment["checkpoint_path"])

        if not os.path.exists(checkpoint_path):
            print(f"❌ No se encontró checkpoint: {checkpoint_path}")
            return {}

        print(f"🔍 Evaluando {model_info['model']}_{model_info['dimension']} (optimizado):")
        print(f"   Experimento: {experiment['name']}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Métrica original: {model_info['auc']:.4f}")

        # Crear configuración con múltiples datasets
        multi_config_path = self.create_multi_eval_config(config_path)

        try:
            # Cargar modelo
            model, original_config = self.evaluator.load_model_from_checkpoint(
                checkpoint_path, multi_config_path
            )

            # Cargar todos los datasets de evaluación una sola vez
            test_loaders_cache = self.load_all_evaluation_datasets_for_multi_config(multi_config_path)
            
            if not test_loaders_cache:
                print("❌ No se pudieron cargar los datasets de evaluación")
                return {}

            # Evaluar en todos los datasets usando el método optimizado
            evaluation_results = self.evaluator.evaluate_on_multiple_datasets_optimized(
                model=model,
                original_config=original_config,
                test_loaders_cache=test_loaders_cache,
                output_dir=str(
                    self.output_dir / f"{model_info['model']}_{model_info['dimension']}"
                ),
            )

            # Mapear los nombres de las claves de configuración a nombres de datasets
            config_to_dataset = {"data": "ADNI", "data2": "FLENI100", "data3": "FLENI600"}

            # Transformar las claves de los resultados
            final_results = {}
            for config_key, metrics in evaluation_results.items():
                dataset_name = config_to_dataset.get(config_key, config_key)
                final_results[dataset_name] = metrics
                if "error" not in metrics:
                    print(
                        f"   ✅ {dataset_name}: Accuracy={metrics.get('accuracy', 0):.4f}, AUC={metrics.get('auc_roc', 0):.4f}"
                    )
                else:
                    print(f"   ❌ {dataset_name}: {metrics['error']}")

            return final_results

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return {}
        finally:
            # Limpiar archivo temporal
            if os.path.exists(multi_config_path):
                os.remove(multi_config_path)

    def generate_evaluation_report(self, all_results: Dict) -> pd.DataFrame:
        """Genera un reporte consolidado de todas las evaluaciones.

        Args:
            all_results: Resultados de evaluación de todos los modelos

        Returns:
            DataFrame con el reporte consolidado
        """
        report_data = []

        for _model_dim_key, model_results in all_results.items():
            if "evaluations" not in model_results:
                continue

            base_info = {
                "model": model_results["model"],
                "dimension": model_results["dimension"],
                "original_auc": model_results["auc"],
            }

            # Agregar métricas para cada dataset
            for dataset_name, metrics in model_results["evaluations"].items():
                if not metrics:
                    continue

                row = base_info.copy()
                row.update(
                    {
                        "dataset": dataset_name,
                        "accuracy": metrics.get("accuracy", 0),
                        "auc_roc": metrics.get("auc_roc", 0),
                        "f1_score": metrics.get("f1_score", 0),
                        "precision": metrics.get("precision", 0),
                        "recall": metrics.get("recall", 0),
                        "specificity": metrics.get("specificity", 0),
                    }
                )

                report_data.append(row)

        return pd.DataFrame(report_data)

    def run_evaluation(self) -> Dict:
        """Ejecuta la evaluación completa de todos los mejores modelos.

        Returns:
            Dict con todos los resultados de evaluación
        """
        print("🚀 Iniciando evaluación de mejores modelos en múltiples datasets")

        # Buscar todos los experimentos
        experiments = self.find_all_experiments()
        if not experiments:
            print("❌ No se encontraron experimentos válidos")
            return {}

        print(f"🔍 Encontrados {len(experiments)} experimentos")

        # Encontrar mejores modelos por combinación
        best_models = self.find_best_models_per_combination(experiments)

        if not best_models:
            print("❌ No se encontraron modelos con métricas válidas")
            return {}

        print("📋 Mejores modelos encontrados:")
        for model_dim_key, info in best_models.items():
            print(f"   {model_dim_key}: Métrica={info['auc']:.4f}")

        # Evaluar cada mejor modelo en todos los datasets
        all_results = {}

        for model_dim_key, model_info in best_models.items():
            print(f"{'=' * 80}")
            print(f"Evaluando {model_dim_key}")
            print(f"{'=' * 80}")

            evaluations = self.evaluate_model_on_all_datasets(model_info)

            all_results[model_dim_key] = {**model_info, "evaluations": evaluations}

        # Generar reporte
        print("📊 Generando reporte consolidado...")
        report_df = self.generate_evaluation_report(all_results)

        # Guardar reporte
        report_path = self.output_dir / "evaluation_report.csv"
        report_df.to_csv(report_path, index=False)

        # Guardar resultados completos
        results_path = self.output_dir / "complete_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(self._make_serializable(all_results), f, indent=2)

        print("🎉 Evaluación completada!")
        print(f"   📋 Reporte: {report_path}")
        print(f"   📋 Resultados completos: {results_path}")

        # Mostrar resumen
        print("📈 Resumen de resultados:")
        if not report_df.empty:
            summary = (
                report_df.groupby(["model", "dimension"])
                .agg(
                    {
                        "accuracy": ["mean", "std"],
                        "auc_roc": ["mean", "std"],
                        "f1_score": ["mean", "std"],
                    }
                )
                .round(4)
            )
            print(summary)

        return all_results

    def run_optimized_evaluation(
        self,
        max_models_per_type: int = 3,
        save_detailed_results: bool = True,
    ) -> pd.DataFrame:
        """Ejecuta evaluación optimizada de mejores modelos en múltiples datasets.

        Args:
            max_models_per_type: Máximo número de mejores modelos por tipo a evaluar
            save_detailed_results: Si guardar resultados detallados

        Returns:
            DataFrame con reporte consolidado
        """
        print("🚀 Iniciando evaluación optimizada de mejores modelos")

        # Buscar experimentos válidos
        all_experiments = self.find_all_experiments()
        print(f"📊 Total de experimentos encontrados: {len(all_experiments)}")

        if not all_experiments:
            print("❌ No se encontraron experimentos válidos")
            return pd.DataFrame()

        # Encontrar mejores modelos por combinación
        best_models = self.find_best_models_per_combination(all_experiments)
        print(f"🏆 Mejores modelos por combinación: {len(best_models)}")

        # Limitar número de modelos por tipo si se especifica
        if max_models_per_type > 0:
            limited_models = {}
            for key, model_info in list(best_models.items())[:max_models_per_type * 4]:  # 4 models * dimensions
                limited_models[key] = model_info
            best_models = limited_models
            print(f"🎯 Evaluando los {len(best_models)} mejores modelos")

        all_results = {}

        # Evaluar cada mejor modelo usando el método optimizado
        for i, (model_dim_key, model_info) in enumerate(best_models.items()):
            print(f"\n{'=' * 80}")
            print(f"Evaluando modelo {i + 1}/{len(best_models)}: {model_dim_key}")
            print(f"{'=' * 80}")

            # Usar método optimizado
            evaluation_results = self.evaluate_model_on_all_datasets_optimized(model_info)

            all_results[model_dim_key] = {
                "model": model_info["model"],
                "dimension": model_info["dimension"],
                "auc": model_info["auc"],
                "experiment": model_info["experiment"],
                "evaluations": evaluation_results,
            }

        # Generar reporte final
        report_df = self.generate_evaluation_report(all_results)

        if save_detailed_results:
            # Guardar resultados detallados
            detailed_results_path = self.output_dir / "detailed_results_optimized.json"
            with open(detailed_results_path, "w") as f:
                json.dump(self._make_serializable(all_results), f, indent=2)

            # Guardar reporte como CSV
            report_path = self.output_dir / "evaluation_report_optimized.csv"
            report_df.to_csv(report_path, index=False)

            print("\n📊 Resultados guardados:")
            print(f"   📋 Reporte CSV: {report_path}")
            print(f"   📝 Detalles JSON: {detailed_results_path}")

        return report_df

    def _make_serializable(self, obj):
        """Convierte objetos no serializables a formato JSON-compatible."""
        import numpy as np
        
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
            return tuple(self._make_serializable(item) for item in obj)
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evaluación automática de mejores modelos en múltiples conjuntos de datos"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="./experiments",
        help="Directorio con experimentos y checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./best_models_evaluation",
        help="Directorio de salida para resultados de evaluación",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=1,
        help="Máximo número de mejores modelos por tipo a evaluar (0 = todos, default: 1)",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Usar evaluación optimizada con carga única de datasets",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = BestModelMultiEvaluator(
        batch_results_dir="./batch_results",  # No se usa pero mantenemos para compatibilidad
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )

    if args.optimized:
        print("🚀 Usando evaluación optimizada con carga única de datasets")
        # Usar método optimizado
        report_df = evaluator.run_optimized_evaluation(
            max_models_per_type=args.max_models,
            save_detailed_results=True,
        )
        
        # Mostrar resumen
        if not report_df.empty:
            print("\n📈 Resumen de resultados (optimizado):")
            print(f"   Total de evaluaciones: {len(report_df)}")
            print(f"   AUC promedio: {report_df['auc_roc'].mean():.4f}")
            print(f"   Accuracy promedio: {report_df['accuracy'].mean():.4f}")

            print("\n🏆 Top 5 mejores resultados por AUC:")
            top_results = report_df.nlargest(5, 'auc_roc')[['model', 'dimension', 'dataset', 'auc_roc', 'accuracy']]
            print(top_results.to_string(index=False))
        else:
            print("⚠️  No se generaron resultados para mostrar")
    else:
        print("🐌 Usando evaluación estándar (carga múltiple de datasets)")
        # Usar método estándar
        results = evaluator.run_evaluation()
        
        if results:
            print("✅ Evaluación completada con método estándar")
        else:
            print("❌ No se obtuvieron resultados")
