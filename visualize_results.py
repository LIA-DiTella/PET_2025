#!/usr/bin/env python

"""Script para generar visualizaciones a partir de la tabla completada.
Este script crea gráficos comparativos de resultados AUC ROC para diferentes
modelos, dimensiones y conjuntos de datos.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def table2df(table_path):
    """Convierte la tabla markdown en un DataFrame de pandas.

    Args:
        table_path: Ruta al archivo markdown con la tabla

    Returns:
        pd.DataFrame: DataFrame con los datos de la tabla

    """
    with open(table_path) as f:
        lines = f.readlines()

    # Encontrar el encabezado de la tabla
    header_idx = 0
    for i, line in enumerate(lines):
        if "|" in line and "---" in line:
            header_idx = i - 1
            break

    # Extraer encabezados
    header = lines[header_idx].strip().split("|")
    header = [h.strip() for h in header if h.strip()]

    # Inicializar listas para cada columna
    data = {h: [] for h in header}

    # Valores actuales para celdas vacías
    current_values = dict.fromkeys(header[:4], "")  # Modelo, Dim, Train, Eval

    # Extraer datos
    for line in lines[header_idx + 2 :]:  # Saltar encabezado y separador
        if "|" not in line or "*" in line:  # Saltar líneas sin datos o leyendas
            continue

        cells = line.strip().split("|")
        cells = [c.strip() for c in cells[1:-1]]  # Eliminar primero y último que son vacíos

        if len(cells) != len(header):
            continue  # Saltar líneas con formato incorrecto

        # Actualizar valores actuales para celdas no vacías
        for i, cell in enumerate(cells[:4]):
            if cell:
                current_values[header[i]] = cell

        # Usar valores actuales para celdas vacías
        row_data = {}
        for i, cell in enumerate(cells):
            if i < 4 and not cell:
                row_data[header[i]] = current_values[header[i]]
            else:
                row_data[header[i]] = cell

        # Extraer AUC ROC y STD
        auc_roc = row_data[header[7]]  # "AUC ROC (± std)"
        if auc_roc.endswith("*"):
            auc_roc = auc_roc[:-1]  # Eliminar asterisco de resultados reales
            row_data["is_real"] = True
        else:
            row_data["is_real"] = False

        # Extraer AUC y STD
        pattern = r"(\d+\.\d+)\s*±\s*(\d+\.\d+)"
        match = re.search(pattern, auc_roc)
        if match:
            row_data["auc"] = float(match.group(1))
            row_data["std"] = float(match.group(2))
        else:
            row_data["auc"] = np.nan
            row_data["std"] = np.nan

        # Añadir todos los datos a las listas
        for h in header:
            data[h].append(row_data[h])
        data["is_real"].append(row_data["is_real"])
        data["auc"].append(row_data["auc"])
        data["std"].append(row_data["std"])

    # Crear DataFrame
    df = pd.DataFrame(data)

    # Renombrar columnas para facilitar su uso
    return df.rename(
        columns={
            header[0]: "Model",
            header[1]: "Dimension",
            header[2]: "Train_Classes",
            header[3]: "Eval_Classes",
            header[4]: "Dataset",
            header[5]: "Labeling",
            header[6]: "Normalization",
            header[7]: "AUC_ROC_std",
        },
    )


def create_visualizations(df, output_dir) -> None:
    """Crea varias visualizaciones a partir de los datos.

    Args:
        df: DataFrame con los datos
        output_dir: Directorio donde guardar las visualizaciones

    """
    os.makedirs(output_dir, exist_ok=True)

    # Configurar estilo de las gráficas
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_context("talk")

    # 1. Comparación de modelos por dimensión
    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="Model",
        y="auc",
        hue="Dimension",
        data=df,
        palette="viridis",
        errorbar=("ci", 95),
        capsize=0.1,
    )
    plt.title("Comparación de Modelos por Dimensión")
    plt.xlabel("Modelo")
    plt.ylabel("AUC ROC")
    plt.ylim(0.5, 1.0)  # AUC ROC va de 0.5 a 1.0
    plt.legend(title="Dimensión")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "models_by_dimension.png"), dpi=300)

    # 2. Comparación por dataset
    plt.figure(figsize=(14, 8))
    sns.barplot(
        x="Model",
        y="auc",
        hue="Dataset",
        data=df,
        palette="mako",
        errorbar=("ci", 95),
        capsize=0.1,
    )
    plt.title("Comparación de Modelos por Dataset")
    plt.xlabel("Modelo")
    plt.ylabel("AUC ROC")
    plt.ylim(0.5, 1.0)
    plt.legend(title="Dataset")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "models_by_dataset.png"), dpi=300)

    # 3. Comparación de clases de entrenamiento vs evaluación
    plt.figure(figsize=(14, 8))

    # Crear una columna que indique si es mismo problema o transferencia
    df["Problem_Type"] = np.where(
        df["Train_Classes"] == df["Eval_Classes"],
        "Mismo problema",
        "Transferencia",
    )

    sns.barplot(
        x="Model",
        y="auc",
        hue="Problem_Type",
        data=df,
        palette="Set2",
        errorbar=("ci", 95),
        capsize=0.1,
    )
    plt.title("Desempeño en Mismo Problema vs Transferencia")
    plt.xlabel("Modelo")
    plt.ylabel("AUC ROC")
    plt.ylim(0.5, 1.0)
    plt.legend(title="Tipo de Problema")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transfer_learning.png"), dpi=300)

    # 4. Mapa de calor de rendimiento por modelo y dataset
    plt.figure(figsize=(12, 8))
    heatmap_data = df.pivot_table(values="auc", index="Model", columns="Dataset", aggfunc="mean")
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
        vmin=0.5,
        vmax=1.0,
    )
    plt.title("Mapa de Calor: AUC ROC por Modelo y Dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_model_dataset.png"), dpi=300)

    # 5. Distribución de AUC ROC por modelo (violinplot)
    plt.figure(figsize=(14, 8))
    sns.violinplot(
        x="Model",
        y="auc",
        hue="Dimension",
        data=df,
        palette="muted",
        split=True,
        inner="quartile",
    )
    plt.title("Distribución de AUC ROC por Modelo y Dimensión")
    plt.xlabel("Modelo")
    plt.ylabel("AUC ROC")
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "violin_models.png"), dpi=300)

    # 6. Gráfico de dispersión: Real vs Simulado
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Model",
        y="auc",
        hue="is_real",
        style="is_real",
        size="is_real",
        sizes=[50, 200],
        data=df,
        palette={True: "green", False: "gray"},
        alpha=0.7,
    )
    plt.title("Resultados Reales vs Simulados")
    plt.xlabel("Modelo")
    plt.ylabel("AUC ROC")
    plt.ylim(0.5, 1.0)
    plt.legend(title="Datos Reales", labels=["Simulado", "Real"])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_simulated.png"), dpi=300)


def generate_summary_stats(df, output_path) -> None:
    """Genera un archivo CSV con estadísticas resumidas.

    Args:
        df: DataFrame con los datos
        output_path: Ruta donde guardar el archivo CSV

    """
    # Generar estadísticas agregadas
    stats = []

    # Por modelo
    model_stats = df.groupby("Model")["auc"].agg(["mean", "std", "min", "max"]).reset_index()
    model_stats["Grupo"] = "Modelo"
    model_stats = model_stats.rename(columns={"Model": "Categoría"})
    stats.append(model_stats)

    # Por dimensión
    dim_stats = df.groupby("Dimension")["auc"].agg(["mean", "std", "min", "max"]).reset_index()
    dim_stats["Grupo"] = "Dimensión"
    dim_stats = dim_stats.rename(columns={"Dimension": "Categoría"})
    stats.append(dim_stats)

    # Por dataset
    dataset_stats = df.groupby("Dataset")["auc"].agg(["mean", "std", "min", "max"]).reset_index()
    dataset_stats["Grupo"] = "Dataset"
    dataset_stats = dataset_stats.rename(columns={"Dataset": "Categoría"})
    stats.append(dataset_stats)

    # Por tipo de problema (mismo o transferencia)
    problem_stats = (
        df.groupby("Problem_Type")["auc"].agg(["mean", "std", "min", "max"]).reset_index()
    )
    problem_stats["Grupo"] = "Tipo de Problema"
    problem_stats = problem_stats.rename(columns={"Problem_Type": "Categoría"})
    stats.append(problem_stats)

    # Combinar todos los datos
    all_stats = pd.concat(stats, ignore_index=True)

    # Ordenar y formatear
    all_stats = all_stats[["Grupo", "Categoría", "mean", "std", "min", "max"]]
    all_stats = all_stats.round(4)

    # Guardar en CSV
    all_stats.to_csv(output_path, index=False)


def main() -> None:
    """Función principal del script."""
    script_dir = Path(__file__).parent
    table_path = script_dir / "table_completed.md"
    output_dir = script_dir / "visualizations"
    stats_path = script_dir / "table_stats.csv"

    # Verificar que exista la tabla completada
    if not os.path.exists(table_path):
        return

    df = table2df(table_path)

    create_visualizations(df, output_dir)

    generate_summary_stats(df, stats_path)


if __name__ == "__main__":
    main()
