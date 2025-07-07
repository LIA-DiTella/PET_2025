#!/usr/bin/env python3
"""
Script de verificación para comprobar que las rutas y archivos necesarios existen
antes de ejecutar la evaluación de múltiples datasets.
"""

import os
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Verifica si un archivo o directorio existe."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists


def main():
    """Función principal de verificación."""
    print("🔍 Verificando estructura de archivos para evaluación multi-dataset")
    print("=" * 80)

    # Directorio base del proyecto
    base_dir = "/home/ipardo/storage1/PET_2025"
    print(f"📁 Directorio base del proyecto: {base_dir}")

    # Verificar directorio base
    if not check_file_exists(base_dir, "Directorio base del proyecto"):
        print("❌ El directorio base no existe. Verifica la ruta.")
        return False

    print("\n📊 Verificando datasets:")
    print("-" * 40)

    all_good = True

    # FLENI100
    print("\n🔸 FLENI100:")
    fleni100_data_dir = f"{base_dir}/data/NIFTIs/fleni100/converted_niftis/"
    fleni100_csv = f"{base_dir}/data/NIFTIs/fleni100/fleni100.csv"

    all_good &= check_file_exists(fleni100_data_dir, "Directorio de imágenes FLENI100")
    all_good &= check_file_exists(fleni100_csv, "CSV de metadatos FLENI100")

    # Contar archivos NIfTI en FLENI100
    if os.path.exists(fleni100_data_dir):
        nifti_files = list(Path(fleni100_data_dir).glob("*.nii*"))
        print(f"   📈 Archivos NIfTI encontrados: {len(nifti_files)}")

    # FLENI600
    print("\n🔸 FLENI600:")
    fleni600_data_dir = f"{base_dir}/data/NIFTIs/fleni600/converted_niftis/"
    fleni600_csv = f"{base_dir}/data/NIFTIs/fleni600/fleni600.csv"

    all_good &= check_file_exists(fleni600_data_dir, "Directorio de imágenes FLENI600")
    all_good &= check_file_exists(fleni600_csv, "CSV de metadatos FLENI600")

    # Contar archivos NIfTI en FLENI600
    if os.path.exists(fleni600_data_dir):
        nifti_files = list(Path(fleni600_data_dir).glob("*.nii*"))
        print(f"   📈 Archivos NIfTI encontrados: {len(nifti_files)}")

    # Verificar directorio de experimentos
    print("\n🔸 Experimentos:")
    experiments_dir = f"{base_dir}/experiments"
    all_good &= check_file_exists(experiments_dir, "Directorio de experimentos")

    # Contar experimentos
    if os.path.exists(experiments_dir):
        experiments = [d for d in Path(experiments_dir).iterdir() if d.is_dir()]
        print(f"   📈 Experimentos encontrados: {len(experiments)}")

        # Verificar algunos experimentos de ejemplo
        for _i, exp_dir in enumerate(experiments[:3]):  # Solo los primeros 3
            config_file = exp_dir / "config.yaml"
            checkpoint_file = exp_dir / "checkpoints" / "best_model.pth"

            config_exists = config_file.exists()
            checkpoint_exists = checkpoint_file.exists()

            status = "✅" if (config_exists and checkpoint_exists) else "⚠️"
            print(
                f"   {status} {exp_dir.name}: config={config_exists}, checkpoint={checkpoint_exists}"
            )

    print("\n" + "=" * 80)

    if all_good:
        print("🎉 ¡Todas las verificaciones pasaron! El script debería funcionar correctamente.")
        print("\n💡 Para ejecutar la evaluación:")
        print(f"   cd {base_dir}")
        print(
            "   python batch_evaluate_best_models.py --experiments_dir ./experiments --output_dir ./best_models_evaluation"
        )
    else:
        print(
            "❌ Algunas verificaciones fallaron. Revisa los archivos faltantes antes de continuar."
        )

    return all_good


if __name__ == "__main__":
    main()
