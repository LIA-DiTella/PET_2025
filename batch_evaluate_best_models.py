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

from data.dataset import get_test_data_loader
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
                "metrics": metrics
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
            
        return {
            "model": model,
            "dimension": dimension
        }

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
        
        # Configurar FLENI100 como data2
        multi_config["data2"] = {
            "dataset_name": "FLENI",
            "data_dir": "/mnt/data/PET_2025/data/fleni100/",
            "metadata_dir": "/mnt/data/PET_2025/data/fleni100/metadata/",
            "classes": "CN_AD",
            "dimension": original_config["data"]["dimension"],
            "batch_size": original_config["data"]["batch_size"],
            "use_otsu_masking": original_config["data"].get("use_otsu_masking", True),
        }
        
        # Configurar FLENI600 como data3
        multi_config["data3"] = {
            "dataset_name": "FLENI",
            "data_dir": "/mnt/data/PET_2025/data/fleni600/",
            "metadata_dir": "/mnt/data/PET_2025/data/fleni600/metadata/",
            "classes": "CN_AD",
            "dimension": original_config["data"]["dimension"],
            "batch_size": original_config["data"]["batch_size"],
            "use_otsu_masking": original_config["data"].get("use_otsu_masking", True),
        }
        
        # Guardar configuración temporal
        temp_config_path = self.output_dir / "temp_multi_eval_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(multi_config, f)
            
        return str(temp_config_path)

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
            
            evaluation_results = {}
            
            # Evaluar en cada conjunto de datos
            for dataset_info in self.test_datasets:
                dataset_name = dataset_info["name"]
                config_key = dataset_info["config_key"]
                
                print(f"   📊 Evaluando en {dataset_name}...")
                
                try:
                    # Obtener data loader para este dataset
                    test_loader = get_test_data_loader(original_config, config_key)
                    
                    if test_loader is None:
                        print(f"   ⚠️  No se pudo cargar {dataset_name}")
                        continue
                        
                    # Crear directorio de salida para este dataset
                    dataset_output_dir = (
                        self.output_dir
                        / f"{model_info['model']}_{model_info['dimension']}"
                        / dataset_name
                    )
                    dataset_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Evaluar modelo
                    metrics = self.evaluator.evaluate_model(
                        model=model,
                        test_loader=test_loader,
                        output_dir=str(dataset_output_dir),
                        save_plots=True,
                        save_predictions=True
                    )
                    
                    evaluation_results[dataset_name] = metrics
                    
                    print(f"   ✅ {dataset_name}: Accuracy={metrics.get('accuracy', 0):.4f}, AUC={metrics.get('auc_roc', 0):.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Error evaluando en {dataset_name}: {e}")
                    continue
                    
            return evaluation_results
            
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
        
        for model_dim_key, model_results in all_results.items():
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
                row.update({
                    "dataset": dataset_name,
                    "accuracy": metrics.get("accuracy", 0),
                    "auc_roc": metrics.get("auc_roc", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "specificity": metrics.get("specificity", 0),
                })
                
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
            print(f"{'='*80}")
            print(f"Evaluando {model_dim_key}")
            print(f"{'='*80}")
            
            evaluations = self.evaluate_model_on_all_datasets(model_info)
            
            all_results[model_dim_key] = {
                **model_info,
                "evaluations": evaluations
            }
            
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
            summary = report_df.groupby(['model', 'dimension']).agg({
                'accuracy': ['mean', 'std'],
                'auc_roc': ['mean', 'std'],
                'f1_score': ['mean', 'std']
            }).round(4)
            print(summary)
        
        return all_results

    def _make_serializable(self, obj):
        """Convierte objetos no serializables a formato JSON-compatible."""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
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
        help="Directorio con experimentos y checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./best_models_evaluation",
        help="Directorio de salida para resultados de evaluación"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    evaluator = BestModelMultiEvaluator(
        batch_results_dir="./batch_results",  # No se usa pero mantenemos para compatibilidad
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_evaluation()
