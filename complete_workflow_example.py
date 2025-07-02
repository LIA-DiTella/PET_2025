"""
Script de ejemplo que demuestra el workflow completo de entrenamiento y evaluación automática.

Este script muestra cómo:
1. Ejecutar entrenamiento por lotes con búsqueda aleatoria de hiperparámetros
2. Automáticamente seleccionar y evaluar los mejores modelos en múltiples conjuntos de datos
3. Generar reportes consolidados de evaluación
"""

import subprocess
import sys
from pathlib import Path


def run_complete_workflow(
    n_runs: int = 3,
    epochs: int = 30,
    models: list = None,
    dimensions: list = None,
    results_dir: str = "./complete_workflow_results",
    use_gpu: int = None,
):
    """Ejecuta el workflow completo de entrenamiento y evaluación.

    Args:
        n_runs: Número de iteraciones de búsqueda aleatoria por tarea
        epochs: Número de épocas por entrenamiento
        models: Lista de modelos a entrenar (default: todos disponibles)
        dimensions: Lista de dimensiones a usar (default: 2d y 3d)
        results_dir: Directorio para guardar todos los resultados
        use_gpu: ID de GPU a usar (default: detectar automáticamente)
    """

    print("🚀 Iniciando workflow complejo de entrenamiento y evaluación de modelos PET")
    print("=" * 80)

    # Configuración por defecto
    if models is None:
        models = ["vit", "swin_transformer"]  # Modelos más modernos

    if dimensions is None:
        dimensions = ["2d", "3d"]

    # Crear directorio de resultados
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Directorio de resultados: {results_path.absolute()}")
    print(f"🎯 Modelos: {', '.join(models)}")
    print(f"📐 Dimensiones: {', '.join(dimensions)}")
    print(f"🔁 Iteraciones por tarea: {n_runs}")
    print(f"📊 Épocas por entrenamiento: {epochs}")

    # PASO 1: Ejecutar entrenamiento por lotes con evaluación automática
    print("\n" + "=" * 80)
    print("PASO 1: Entrenamiento por lotes con búsqueda aleatoria")
    print("=" * 80)

    train_command = (
        [
            sys.executable,
            "batch_train_binary_classification.py",
            "--n_runs",
            str(n_runs),
            "--epochs",
            str(epochs),
            "--results_dir",
            str(results_path),
            "--models",
        ]
        + models
        + ["--dimensions"]
        + dimensions
        + [
            "--use_otsu_masking",
            "--auto_evaluate_best",  # Evaluación automática de mejores modelos
        ]
    )

    if use_gpu is not None:
        train_command.extend(["--gpu", str(use_gpu)])

    print(f"💻 Ejecutando: {' '.join(train_command)}")

    try:
        result = subprocess.run(train_command, check=True, capture_output=False)
        print("\n✅ Entrenamiento y evaluación automática completados!")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en entrenamiento: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Proceso interrumpido por el usuario")
        return False

    # PASO 2: Verificar resultados
    print("\n" + "=" * 80)
    print("PASO 2: Verificación de resultados")
    print("=" * 80)

    # Verificar archivos de resultados
    expected_files = [
        "intermediate_results.json",
        "summary.json",
        "best_models_evaluation/evaluation_report.csv",
        "best_models_evaluation/complete_evaluation_results.json",
    ]

    print("📋 Verificando archivos de resultados:")
    all_found = True

    for file_name in expected_files:
        file_path = results_path / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name}")
            all_found = False

    if all_found:
        print("\n🎉 Todos los archivos de resultados encontrados!")
    else:
        print("\n⚠️  Algunos archivos de resultados no se encontraron")

    # PASO 3: Mostrar resumen de resultados
    print("\n" + "=" * 80)
    print("PASO 3: Resumen de resultados")
    print("=" * 80)

    try:
        import json
        import pandas as pd

        # Cargar resumen del entrenamiento
        summary_file = results_path / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

            print(f"📈 Resumen del entrenamiento:")
            print(f"   Total de tareas: {summary.get('total_tasks', 0)}")
            print(f"   Total de entrenamientos: {summary.get('total_runs', 0)}")
            print(
                f"   Umbralizado de Otsu: {'✅' if summary.get('use_otsu_masking', False) else '❌'}"
            )

            print(f"\n🏆 Mejores resultados por tarea:")
            for task_key, task_info in summary.get("tasks", {}).items():
                print(f"   {task_key}:")
                print(f"      Mejor AUC: {task_info.get('best_auc', 0):.4f}")
                print(
                    f"      AUC promedio: {task_info.get('mean_auc', 0):.4f} ± {task_info.get('std_auc', 0):.4f}"
                )

        # Cargar reporte de evaluación
        eval_report_file = results_path / "best_models_evaluation" / "evaluation_report.csv"
        if eval_report_file.exists():
            eval_df = pd.read_csv(eval_report_file)

            print(f"\n📊 Reporte de evaluación en múltiples datasets:")
            print("=" * 60)

            if not eval_df.empty:
                # Mostrar tabla resumida
                summary_table = eval_df.pivot_table(
                    index=["model", "dimension"],
                    columns="dataset",
                    values=["accuracy", "auc_roc", "f1_score"],
                    aggfunc="mean",
                ).round(4)

                print(summary_table)

                # Mostrar estadísticas generales
                print(f"\n📈 Estadísticas generales:")
                print(f"   Modelos evaluados: {eval_df['model'].nunique()}")
                print(f"   Datasets evaluados: {eval_df['dataset'].nunique()}")
                print(
                    f"   Accuracy promedio: {eval_df['accuracy'].mean():.4f} ± {eval_df['accuracy'].std():.4f}"
                )
                print(
                    f"   AUC ROC promedio: {eval_df['auc_roc'].mean():.4f} ± {eval_df['auc_roc'].std():.4f}"
                )
                print(
                    f"   F1 Score promedio: {eval_df['f1_score'].mean():.4f} ± {eval_df['f1_score'].std():.4f}"
                )

    except Exception as e:
        print(f"⚠️  Error mostrando resumen: {e}")

    # PASO 4: Generar recomendaciones
    print("\n" + "=" * 80)
    print("PASO 4: Archivos generados y próximos pasos")
    print("=" * 80)

    print("📁 Archivos generados:")
    print(f"   📊 Resultados de entrenamiento: {results_path}/intermediate_results.json")
    print(f"   📋 Resumen: {results_path}/summary.json")
    print(
        f"   📈 Reporte de evaluación: {results_path}/best_models_evaluation/evaluation_report.csv"
    )
    print(
        f"   🔍 Resultados detallados: {results_path}/best_models_evaluation/complete_evaluation_results.json"
    )
    print(
        f"   📊 Gráficos y predicciones: {results_path}/best_models_evaluation/[modelo]/[dataset]/"
    )

    print(f"\n🔍 Próximos pasos recomendados:")
    print(f"   1. Revisar el reporte CSV para comparar modelos entre datasets")
    print(f"   2. Examinar gráficos de ROC y matrices de confusión en las subcarpetas")
    print(f"   3. Analizar predicciones detalladas para casos específicos")
    print(f"   4. Considerar fine-tuning de los mejores modelos con más épocas")

    print(f"\n🎉 Workflow completo finalizado exitosamente!")
    print(f"📁 Todos los resultados están en: {results_path.absolute()}")

    return True


def main():
    """Función principal para ejecutar el workflow con parámetros de ejemplo."""

    print("🔬 Ejemplo de workflow complejo para modelos PET de clasificación de Alzheimer")
    print("=" * 80)

    # Configuración del ejemplo
    config = {
        "n_runs": 2,  # Pocas iteraciones para el ejemplo
        "epochs": 20,  # Pocas épocas para el ejemplo
        "models": ["vit", "swin_transformer"],  # Modelos modernos
        "dimensions": ["2d"],  # Solo 2D para agilizar el ejemplo
        "results_dir": "./workflow_example_results",
        "use_gpu": 0 if input("¿Usar GPU 0? (y/n): ").lower().startswith("y") else None,
    }

    print(f"\n🎯 Configuración del ejemplo:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    confirm = input(f"\n¿Proceder con esta configuración? (y/n): ")
    if not confirm.lower().startswith("y"):
        print("❌ Cancelado por el usuario")
        return

    # Ejecutar workflow
    success = run_complete_workflow(**config)

    if success:
        print(f"\n🌟 ¡Ejemplo completado exitosamente!")
    else:
        print(f"\n💥 El ejemplo falló. Revisa los logs para más detalles.")


if __name__ == "__main__":
    main()
