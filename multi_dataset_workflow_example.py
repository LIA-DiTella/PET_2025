#!/usr/bin/env python3
"""
Script de ejemplo completo que muestra el workflow completo:
1. Entrenar un modelo con la configuración principal (data)
2. Evaluar automáticamente en múltiples configuraciones adicionales (data2, data3, etc.)

Este ejemplo demuestra cómo usar las nuevas capacidades de evaluación
en múltiples distribuciones de datos.
"""

import argparse
from pathlib import Path

from model_evaluator import ModelEvaluator
from train import train_model


def main_workflow_example():
    """Ejemplo del workflow completo: entrenar y evaluar en múltiples datasets."""
    print("🚀 Workflow Completo: Entrenar y Evaluar en Múltiples Datasets")
    print("=" * 70)

    # 1. Configuración
    config_path = "./configs/resnet18_2d_adni_cnad_multi_eval.yaml"

    if not Path(config_path).exists():
        print(f"❌ No se encontró archivo de configuración: {config_path}")
        print("💡 Asegúrate de haber creado el archivo de configuración con múltiples datasets")
        return

    print(f"📋 Usando configuración: {config_path}")

    # 2. Entrenar modelo (opcional - solo si no existe)
    print("\n🎯 Paso 1: Verificando modelo entrenado...")

    # Buscar experimento existente
    experiments_dir = Path("./experiments")
    existing_experiments = []
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and "resnet18" in exp_dir.name.lower():
                checkpoint_path = exp_dir / "checkpoints" / "best_model.pth"
                if checkpoint_path.exists():
                    existing_experiments.append((exp_dir, checkpoint_path))

    if existing_experiments:
        # Usar experimento existente
        exp_dir, checkpoint_path = existing_experiments[0]
        exp_config_path = exp_dir / "config.yaml"
        print(f"✅ Usando modelo existente: {exp_dir.name}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Config: {exp_config_path}")
    else:
        # Entrenar nuevo modelo
        print("🔄 No se encontró modelo existente. Entrenando nuevo modelo...")
        try:
            results = train_model(config_path, gpu_id=None)
            exp_dir = Path(results["exp_dir"])
            checkpoint_path = exp_dir / "checkpoints" / "best_model.pth"
            exp_config_path = exp_dir / "config.yaml"
            print(f"✅ Modelo entrenado exitosamente en: {exp_dir}")
        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            return

    # 3. Evaluar en múltiples datasets
    print("\n🎯 Paso 2: Evaluando en múltiples configuraciones de datos...")

    # Crear evaluador
    evaluator = ModelEvaluator()

    # Cargar modelo entrenado
    print(f"📂 Cargando modelo desde: {checkpoint_path}")
    model, original_config = evaluator.load_model_from_checkpoint(
        str(checkpoint_path), str(exp_config_path)
    )

    # Para este ejemplo, vamos a usar la configuración con múltiples datasets
    # en lugar de la configuración original del experimento
    from utils.config_utils import load_config

    multi_config = load_config(config_path)

    # Evaluar en todas las configuraciones de datos
    results = evaluator.evaluate_on_multiple_datasets(
        model=model,
        original_config=multi_config,  # Usar configuración con múltiples datasets
        output_dir="./evaluation_results/multi_dataset_workflow",
    )

    # 4. Mostrar resumen de resultados
    print("\n🎯 Paso 3: Resumen de Resultados")
    print("=" * 50)

    if not results:
        print("❌ No se obtuvieron resultados")
        return

    print("📊 Resultados por configuración:")
    for config_name, metrics in results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            dataset_name = metrics.get("dataset_name", "N/A")
            auc = metrics.get("auc_roc", 0)
            acc = metrics.get("accuracy", 0)
            samples = metrics.get("num_samples", 0)
            classes = metrics.get("classes", "N/A")

            print(f"\n   📈 {config_name}:")
            print(f"      Dataset: {dataset_name}")
            print(f"      Clases: {classes}")
            print(f"      Muestras: {samples}")
            print(f"      AUC ROC: {auc:.4f}")
            print(f"      Accuracy: {acc:.4f}")

        elif isinstance(metrics, dict) and "error" in metrics:
            print(f"\n   ❌ {config_name}: Error - {metrics['error']}")

    print("\n✅ Workflow completado!")
    print(f"📁 Resultados detallados en: ./evaluation_results/multi_dataset_workflow/")
    print(
        f"📋 Resumen comparativo: ./evaluation_results/multi_dataset_workflow/multi_dataset_evaluation_summary.json"
    )


def demo_yaml_configuration():
    """Muestra cómo configurar un YAML con múltiples datasets."""
    print("\n📝 Demostración: Configuración YAML con Múltiples Datasets")
    print("=" * 60)

    yaml_example = """
# Configuración principal para entrenamiento
data:
  dataset_name: 'ADNI'
  classes: 'CN_AD'
  data_dir: './data/NIFTIs/Archivo/converted_niftis/'
  train_csv: './data/metadata/adni_train.csv'
  val_csv: './data/metadata/adni_val.csv'
  test_csv: './data/metadata/adni_test.csv'
  # ... otras configuraciones

# Configuraciones adicionales para evaluación
data2:
  dataset_name: 'fleni100'
  classes: 'CN_AD'
  data_dir: './data/NIFTIs/fleni100/converted_niftis/'
  test_csv: './data/NIFTIs/fleni100/fleni100.csv'
  # ... otras configuraciones

data3:
  dataset_name: 'fleni600'
  classes: 'CN_AD'
  data_dir: './data/NIFTIs/fleni600/converted_niftis/'
  test_csv: './data/NIFTIs/fleni600/fleni600.csv'
  # ... otras configuraciones
"""

    print("💡 Estructura del archivo YAML:")
    print(yaml_example)

    print("🔑 Puntos clave:")
    print("   • 'data' se usa para entrenamiento (requiere train_csv, val_csv, test_csv)")
    print("   • 'data2', 'data3', etc. se usan solo para evaluación (solo requieren test_csv)")
    print("   • Cada configuración puede tener diferentes datasets, clases, o parámetros")
    print("   • El evaluador automáticamente detecta todas las configuraciones data*")


def interactive_menu():
    """Menú interactivo para mostrar diferentes ejemplos."""
    while True:
        print("\n🔧 Menú de Ejemplos - Evaluación Multi-Dataset")
        print("=" * 50)
        print("1. Ejecutar workflow completo (entrenar + evaluar)")
        print("2. Solo evaluar modelo existente")
        print("3. Mostrar configuración YAML")
        print("4. Salir")

        choice = input("\nSelecciona una opción (1-4): ").strip()

        if choice == "1":
            main_workflow_example()
        elif choice == "2":
            evaluate_existing_model_example()
        elif choice == "3":
            demo_yaml_configuration()
        elif choice == "4":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Por favor selecciona 1-4.")


def evaluate_existing_model_example():
    """Ejemplo de evaluación de modelo existente en múltiples datasets."""
    print("\n🔍 Ejemplo: Evaluar Modelo Existente en Múltiples Datasets")
    print("=" * 60)

    # Buscar experimentos existentes
    experiments_dir = Path("./experiments")
    if not experiments_dir.exists():
        print("❌ No se encontró directorio de experimentos")
        return

    # Listar experimentos disponibles
    available_experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            checkpoint_path = exp_dir / "checkpoints" / "best_model.pth"
            config_path = exp_dir / "config.yaml"
            if checkpoint_path.exists() and config_path.exists():
                available_experiments.append((exp_dir, checkpoint_path, config_path))

    if not available_experiments:
        print("❌ No se encontraron experimentos válidos")
        print("💡 Ejecuta primero el entrenamiento o usa la opción 1 del menú")
        return

    print(f"📁 Experimentos disponibles:")
    for i, (exp_dir, _, _) in enumerate(available_experiments, 1):
        print(f"   {i}. {exp_dir.name}")

    # Seleccionar experimento (para este ejemplo, usar el primero)
    exp_dir, checkpoint_path, config_path = available_experiments[0]
    print(f"\n📂 Usando experimento: {exp_dir.name}")

    # Crear evaluador y cargar modelo
    evaluator = ModelEvaluator()
    model, original_config = evaluator.load_model_from_checkpoint(
        str(checkpoint_path), str(config_path)
    )

    # Cargar configuración con múltiples datasets
    multi_config_path = "./configs/resnet18_2d_adni_cnad_multi_eval.yaml"
    if Path(multi_config_path).exists():
        from utils.config_utils import load_config

        multi_config = load_config(multi_config_path)

        # Evaluar en múltiples configuraciones
        results = evaluator.evaluate_on_multiple_datasets(
            model=model,
            original_config=multi_config,
            output_dir="./evaluation_results/existing_model_multi_eval",
        )

        print("\n📊 Resultados:")
        for config_name, metrics in results.items():
            if isinstance(metrics, dict) and "auc_roc" in metrics:
                print(f"   {config_name}: AUC = {metrics['auc_roc']:.4f}")
    else:
        print(f"⚠️  No se encontró configuración multi-dataset: {multi_config_path}")
        print("💡 Usa la configuración original del modelo")


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Ejemplos completos de workflow multi-dataset")
    parser.add_argument(
        "--mode",
        choices=["workflow", "evaluate", "config", "interactive"],
        default="interactive",
        help="Modo de ejecución",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("🚀 Ejemplos de Evaluación Multi-Dataset")
    print("🔧 Nuevas capacidades para evaluar modelos en múltiples distribuciones")
    print("=" * 70)

    if args.mode == "workflow":
        main_workflow_example()
    elif args.mode == "evaluate":
        evaluate_existing_model_example()
    elif args.mode == "config":
        demo_yaml_configuration()
    elif args.mode == "interactive":
        interactive_menu()

    print("\n🎉 Para más información, consulta la documentación del ModelEvaluator")
