import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calcula las métricas de evaluación.

    Args:
        y_true (array): Etiquetas reales
        y_pred (array): Predicciones del modelo
        y_score (array, optional): Puntuaciones/probabilidades para ROC AUC

    Returns:
        dict: Diccionario con las métricas calculadas

    """
    try:
        metrics = {}
        
        # Convertir a numpy arrays y asegurar formato correcto
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Verificar que tengan la misma longitud
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        # Asegurar que sean enteros
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        # Precisión
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()  # Convertir a lista para JSON

        # Para problemas binarios
        if len(np.unique(y_true)) == 2 and y_score is not None:
            from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
            
            # Asegurar que y_score sea numpy array
            if not isinstance(y_score, np.ndarray):
                y_score = np.array(y_score)
            
            try:
                # Si y_score es 2D (probabilidades por clase), usar columna de clase positiva
                if y_score.ndim == 2 and y_score.shape[1] == 2:
                    y_score_binary = y_score[:, 1]  # Probabilidad de clase positiva
                else:
                    y_score_binary = y_score.flatten()
                
                metrics["auc_roc"] = roc_auc_score(y_true, y_score_binary)
                metrics["f1_score"] = f1_score(y_true, y_pred)
                metrics["precision"] = precision_score(y_true, y_pred)
                metrics["recall"] = recall_score(y_true, y_pred)
                
                # Calcular sensibilidad y especificidad
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
            except Exception as e:
                print(f"Warning: Could not calculate some binary metrics: {e}")

        return metrics
        
    except Exception as e:
        return {"error": f"Error calculating metrics: {str(e)}"}


def plot_confusion_matrix(cm, class_names, save_path=None) -> None:
    """Visualiza la matriz de confusión.

    Args:
        cm (array): Matriz de confusión
        class_names (list): Nombres de las clases
        save_path (str, optional): Ruta para guardar la figura

    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Añadir etiquetas numéricas
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true, y_score, class_names, save_path=None) -> None:
    """Genera curvas ROC.

    Args:
        y_true (array): Etiquetas reales
        y_score (array): Puntuaciones de predicción
        class_names (list): Nombres de las clases
        save_path (str, optional): Ruta para guardar la figura

    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    plt.figure(figsize=(10, 8))

    # Para caso binario
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2)

    # Para caso multiclase
    else:
        # Binarizar las etiquetas
        y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

        # Calcular ROC para cada clase
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label=f"ROC curve {class_names[i]} (area = {roc_auc[i]:.2f})",
            )

        plt.plot([0, 1], [0, 1], "k--", lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_results_to_csv(y_true, y_pred, y_score, save_path) -> None:
    """Guarda los resultados en un archivo CSV.

    Args:
        y_true (array): Etiquetas reales  
        y_pred (array): Predicciones
        y_score (array): Probabilidades
        save_path (str): Ruta para guardar el CSV

    """
    # Asegurarse que el directorio existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Crear DataFrame con los resultados
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score if y_score.ndim == 1 else y_score[:, 1]  # Para binario, usar prob clase positiva
    })

    # Guardar como CSV
    df.to_csv(save_path, index=False)


def save_metrics_to_csv(results_dict, save_path) -> None:
    """Guarda métricas en un archivo CSV.

    Args:
        results_dict (dict): Diccionario con resultados
        save_path (str): Ruta para guardar el CSV

    """
    # Asegurarse que el directorio existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convertir diccionario a DataFrame
    df = pd.DataFrame([results_dict])

    # Guardar como CSV
    df.to_csv(save_path, index=False)


def update_results_table(results_dict, table_path) -> None:
    """Actualiza la tabla general de resultados.

    Args:
        results_dict (dict): Diccionario con los resultados
        table_path (str): Ruta a la tabla de resultados

    """
    # Crear tabla si no existe
    if not os.path.exists(table_path):
        df = pd.DataFrame([results_dict])
    else:
        # Cargar tabla existente y añadir nueva fila
        df = pd.read_csv(table_path)
        df = df.append(results_dict, ignore_index=True)

    # Guardar tabla actualizada
    df.to_csv(table_path, index=False)
