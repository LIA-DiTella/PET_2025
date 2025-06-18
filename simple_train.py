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
        for batch in train_loader:
            # Entrenamiento
            model.train()
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()

            model.eval()
            # ROC_AUC, Accuracy, Loss

            with torch.no_grad():
                outputs = model(inputs)
                loss_value = loss(outputs, labels)
                pred = outputs.argmax(dim=1)
                true = labels.argmax(dim=1) if labels.dim() > 1 else labels
                corrects = (pred == true).sum().item()
            
            accuracy = corrects / labels.size(0)

            train_losses.append(loss_value.item())
            train_accuracies.append(accuracy)
        
        model.eval()
        # Valid
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        for val_batch in val_loader:
            val_inputs, val_labels = val_batch
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            with torch.no_grad():
                val_outputs = model(val_inputs)
                v_loss = loss(val_outputs, val_labels)
                _, v_preds = torch.max(val_outputs, 1)
                v_corrects = (v_preds == val_labels).sum().item()
                v_total = val_labels.size(0)
            val_loss += v_loss.item() * v_total
            val_corrects += v_corrects
            val_total += v_total

        val_loss /= val_total
        val_accuracy = val_corrects / val_total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value.item()}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_corrects = 0
        test_total = 0
        for test_batch in test_loader:
            test_inputs, test_labels = test_batch
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            outputs = model(test_inputs)
            t_loss = loss(outputs, test_labels)
            _, t_preds = torch.max(outputs, 1)
            t_corrects = (t_preds == test_labels).sum().item()
            t_total = test_labels.size(0)
            test_loss += t_loss.item() * t_total
            test_corrects += t_corrects
            test_total += t_total

        test_loss /= test_total
        test_accuracy = test_corrects / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


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
