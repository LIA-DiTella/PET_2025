"""Script de entrenamiento para modelos de clasificación PET."""

import argparse
import os
import sys

import torch
from torch import nn
from torchvision import models

# Importar módulos propios
from data.dataset import get_data_loaders
from utils.config_utils import load_config

from matplotlib import pyplot as plt

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_model(config_path, gpu_id=None, data_loaders=None):
    """Entrena un modelo según la configuración proporcionada.

    Args:
        config_path (str): Ruta al archivo de configuración YAML
        gpu_id (int, optional): ID de GPU a usar

    Returns:
        dict: Métricas de evaluación

    """
    # Cargar configuración
    config = load_config(config_path)

    # Configurar dispositivo
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {device}")

    # Obtener parámetros del modelo
    model_config = config.get("model", {})
    # model_name = model_config.get("name", "resnet18").lower()
    # dimension = model_config.get("dimension", "2d").lower()

    # Crear directorio para el experimento
    # exp_base_dir = config.get("experiment", {}).get("base_dir", "./experiments")
    # dataset_name = config.get("data", {}).get("dataset_name", "ADNI")
    classes = config.get("data", {}).get("classes", "CN_AD")
    # exp_name = config.get("experiment", {}).get("name", None)

    # Crear modelo
    num_classes = len(classes.split("_"))
    model_config["num_classes"] = num_classes

    # Obtener data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Entrenar modelo
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    epochs = 50
    for epoch in range(epochs):
        # Métricas de entrenamiento por época
        model.train()
        epoch_train_loss = 0.0
        epoch_train_corrects = 0
        epoch_train_total = 0
        
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()

            # Calcular métricas de entrenamiento
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                # Las etiquetas son one-hot, necesitamos argmax
                true = labels.argmax(dim=1)
                batch_corrects = (pred == true).sum().item()
                
                epoch_train_corrects += batch_corrects
                epoch_train_total += labels.size(0)
                epoch_train_loss += loss_value.item() * labels.size(0)

        # Promediar métricas de entrenamiento por época
        train_loss = epoch_train_loss / epoch_train_total
        train_accuracy = epoch_train_corrects / epoch_train_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs, val_labels = val_batch
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                v_loss = loss(val_outputs, val_labels)
                
                # Obtener predicciones
                val_pred = val_outputs.argmax(dim=1)
                # Las etiquetas son one-hot, necesitamos argmax
                val_true = val_labels.argmax(dim=1)
                
                # Acumular métricas
                batch_size = val_labels.size(0)
                val_loss += v_loss.item() * batch_size
                val_corrects += (val_pred == val_true).sum().item()
                val_total += batch_size

        val_loss /= val_total
        val_accuracy = val_corrects / val_total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Debug info para el primer epoch
        if epoch == 0:
            print(f"Debug - Val total samples: {val_total}, Val corrects: {val_corrects}")
            if 'val_pred' in locals():
                print(f"Debug - Unique predictions: {torch.unique(val_pred).cpu().numpy()}")
                print(f"Debug - Unique true labels: {torch.unique(val_true).cpu().numpy()}")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Evaluación en test
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_corrects = 0
        test_total = 0
        
        for test_batch in test_loader:
            test_inputs, test_labels = test_batch
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
            t_loss = loss(test_outputs, test_labels)
            
            # Obtener predicciones
            test_pred = test_outputs.argmax(dim=1)
            # Las etiquetas son one-hot, necesitamos argmax
            test_true = test_labels.argmax(dim=1)
            
            # Acumular métricas
            batch_size = test_labels.size(0)
            test_loss += t_loss.item() * batch_size
            test_corrects += (test_pred == test_true).sum().item()
            test_total += batch_size

        test_loss /= test_total
        test_accuracy = test_corrects / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plotear resultados
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_title('Loss per Epoch')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(train_accuracies, label='Train Accuracy')
    axs[1].plot(val_accuracies, label='Validation Accuracy')
    axs[1].set_title('Accuracy per Epoch')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    
    plt.savefig("training_results.png")


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos para clasificación de PET",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo de configuración YAML",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="ID de GPU a usar (por defecto, usa cuda si está disponible)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = train_model(args.config, args.gpu)
