#!/usr/bin/env python3
"""
Script de ejemplo para ejecutar evaluación optimizada de mejores modelos.

Este script demuestra cómo usar la nueva funcionalidad de evaluación optimizada
que carga todos los datasets una sola vez al inicio.
"""

import argparse
from batch_evaluate_best_models import BestModelMultiEvaluator


def main():
    """Función principal para ejecutar evaluación optimizada."""
    parser = argparse.ArgumentParser(
        description="Evaluación optimizada de mejores modelos en múltiples datasets"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="./experiments",
        help="Directorio con experimentos entrenados",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./best_models_evaluation_optimized",
        help="Directorio de salida para resultados",
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=1,
        help="Máximo número de mejores modelos por tipo a evaluar (0 = todos, default: 1)",
    )

    args = parser.parse_args()

    print("🚀 Iniciando evaluación optimizada de mejores modelos")
    print("=" * 60)
    print("📊 Configuración:")
    print(f"   Experimentos: {args.experiments_dir}")
    print(f"   Salida: {args.output_dir}")
    print(f"   Max modelos por tipo: {args.max_models if args.max_models > 0 else 'Todos'} (default: mejor modelo)")
    print("=" * 60)

    # Crear evaluador
    evaluator = BestModelMultiEvaluator(
        batch_results_dir="./batch_results",  # No se usa en método optimizado
        experiments_dir=args.experiments_dir,
        output_dir=args.output_dir,
    )

    print("\n🔍 Buscando experimentos disponibles...")
    
    try:
        # Ejecutar evaluación optimizada
        report_df = evaluator.run_optimized_evaluation(
            max_models_per_type=args.max_models,
            save_detailed_results=True,
        )

        if not report_df.empty:
            print("\n🎉 Evaluación completada exitosamente!")
            print("\n📊 Resumen de resultados:")
            print(f"   Total de evaluaciones: {len(report_df)}")
            print(f"   AUC promedio: {report_df['auc_roc'].mean():.4f}")
            print(f"   Accuracy promedio: {report_df['accuracy'].mean():.4f}")

            print("\n🏆 Top 5 mejores resultados por AUC:")
            top_results = report_df.nlargest(5, 'auc_roc')[
                ['model', 'dimension', 'dataset', 'auc_roc', 'accuracy']
            ]
            print(top_results.to_string(index=False))

            print(f"\n📁 Resultados detallados guardados en: {args.output_dir}")
            
            # Mostrar estadísticas por dataset
            print("\n📊 Estadísticas por dataset:")
            dataset_stats = report_df.groupby('dataset')[['auc_roc', 'accuracy']].agg(['mean', 'std'])
            print(dataset_stats.round(4))
            
            # Mostrar estadísticas por modelo
            print("\n📊 Estadísticas por modelo:")
            model_stats = report_df.groupby('model')[['auc_roc', 'accuracy']].agg(['mean', 'std'])
            print(model_stats.round(4))

        else:
            print("⚠️  No se generaron resultados para mostrar")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluación interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        raise


if __name__ == "__main__":
    main()
