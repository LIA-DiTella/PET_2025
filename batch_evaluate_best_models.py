"""
Script para evaluar automáticamente los mejores modelos entrenados en múltiples conjuntos de datos.

Este script:
1. Encuentra todos los experimentos completados de entrenamiento por lotes
2. Selecciona el mejor modelo para cada combinación (modelo, dimensión)
3. Evalúa cada mejor modelo en todos los conjuntos de prueba disponibles (ADNI, FLENI100, FLENI600)
4. Genera un reporte complejo con métricas de evaluación
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    def find_best_models_per_combination(self, batch_results: Dict) -> Dict:
        """Encuentra el mejor modelo para cada combinación (modelo, dimensión).
        
        Args:
            batch_results: Resultados del entrenamiento por lotes
            
        Returns:
            Dict con la estructura: {model_dim_key: best_experiment_info}
        """
        best_models = {}
        
        for task_key, task_results in batch_results.items():
            if not task_results:
                continue
                
            # Extraer información de la tarea
            task_info = task_results[0].get("task", {})
            model = task_info.get("model", "unknown")
            dimension = task_info.get("dimension", "unknown")
            
            # Clave para agrupar por modelo y dimensión
            model_dim_key = f"{model}_{dimension}"
            
            # Encontrar el mejor resultado basado en AUC ROC
            best_result = None
            best_auc = -1
            
            for result in task_results:
                if "evaluation" not in result or not result["evaluation"]:
                    continue
                    
                auc = result["evaluation"].get("auc_roc", -1)
                if isinstance(auc, (list, tuple)):
                    auc = float(auc[0]) if len(auc) > 0 else -1
                else:
                    auc = float(auc) if auc is not None else -1
                    
                if auc > best_auc:
                    best_auc = auc
                    best_result = result
            
            # Actualizar el mejor modelo para esta combinación si es mejor que el actual
            if best_result and (model_dim_key not in best_models or best_auc > best_models[model_dim_key]["auc"]):
                best_models[model_dim_key] = {
                    "task_key": task_key,
                    "result": best_result,
                    "auc": best_auc,
                    "model": model,
                    "dimension": dimension,
                }
                
        return best_models

    def find_experiment_checkpoint(self, config_path: str) -> Optional[Path]:
        """Encuentra el checkpoint del mejor modelo para un experimento.
        
        Args:
            config_path: Ruta al archivo de configuración del experimento
            
        Returns:
            Path al checkpoint del mejor modelo o None si no se encuentra
        """
        if not config_path or not os.path.exists(config_path):
            return None
            
        # Extraer nombre del experimento desde el config path
        experiment_name = Path(config_path).stem
        
        # Buscar en el directorio de experimentos
        exp_dir = self.experiments_dir / experiment_name
        if not exp_dir.exists():
            print(f"⚠️  Directorio de experimento no encontrado: {exp_dir}")
            return None
            
        # Buscar checkpoint en el directorio de checkpoints
        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            print(f"⚠️  Directorio de checkpoints no encontrado: {checkpoints_dir}")
            return None
            
        # Buscar el mejor checkpoint
        best_checkpoint = checkpoints_dir / "best_model.pth"
        if best_checkpoint.exists():
            return best_checkpoint
            
        # Si no existe best_model.pth, buscar cualquier checkpoint
        checkpoints = list(checkpoints_dir.glob("*.pth"))
        if checkpoints:
            return max(checkpoints, key=os.path.getctime)
            
        return None

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
            model_info: Información del modelo con resultado, config_path, etc.
            
        Returns:
            Dict con resultados de evaluación en todos los datasets
        """
        config_path = model_info["result"].get("config_path")
        if not config_path:
            print(f"❌ No se encontró config_path para {model_info['model']}_{model_info['dimension']}")
            return {}
            
        # Encontrar checkpoint
        checkpoint_path = self.find_experiment_checkpoint(config_path)
        if not checkpoint_path:
            print(f"❌ No se encontró checkpoint para {model_info['model']}_{model_info['dimension']}")
            return {}
            
        print(f"🔍 Evaluando {model_info['model']}_{model_info['dimension']}:")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   AUC original: {model_info['auc']:.4f}")
        
        # Crear configuración con múltiples datasets
        multi_config_path = self.create_multi_eval_config(config_path)
        
        try:
            # Cargar modelo
            model, original_config = self.evaluator.load_model_from_checkpoint(
                str(checkpoint_path), multi_config_path
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
                        self.output_dir / 
                        f"{model_info['model']}_{model_info['dimension']}" / 
                        dataset_name
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
        
        # Cargar resultados del entrenamiento por lotes
        batch_results = self.load_batch_results()
        if not batch_results:
            return {}
            
        # Encontrar mejores modelos por combinación
        best_models = self.find_best_models_per_combination(batch_results)
        
        if not best_models:
            print("❌ No se encontraron modelos entrenados")
            return {}
            
        print(f"📋 Mejores modelos encontrados:")
        for model_dim_key, info in best_models.items():
            print(f"   {model_dim_key}: AUC={info['auc']:.4f}")
            
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
            
        print(f"🎉 Evaluación completada!")
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
        "--batch_results_dir",
        type=str,
        default="./batch_results",
        help="Directorio con resultados del entrenamiento por lotes"
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
        batch_results_dir=args.batch_results_dir,
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir
    )
    
    results = evaluator.run_evaluation()
