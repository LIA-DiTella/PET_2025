import os
import sys

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision import models
from tqdm import tqdm

# Importar módulos propios
from data.dataset import get_data_loaders
from utils.config_utils import load_config

# Añadir directorios al path
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

config_path = "configs/resnet18_2d_adni_cnad_server.yaml"

# Cargar configuración
config = load_config(config_path)

# Configurar dispositivo
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


# Plot images from the training set
def plot_images(data_loader, num_images=4):
    images = []
    labels = []

    # Collect images and labels from the data loader
    for i, (image, label) in enumerate(data_loader):
        if i >= num_images:
            break
        images.append(image)
        labels.append(label)
    images = torch.stack(images)
    labels = torch.stack(labels)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img = images[i].squeeze().cpu().numpy()
        # normalize the image to [0, 1] range for better visualization
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 255
        if img.ndim == 4:
            img = img[0]
        img = img[0]

        axes[i].imshow(img, cmap="viridis")
        axes[i].set_title(f"Label: {torch.argmax(labels[i]).item()} - Shape: {img.shape}")
        axes[i].axis("off")
    plt.show()


# Plot images from the training set
plot_images(train_loader)
plot_images(val_loader)
plot_images(test_loader)

# Entrenar modelo
model = models.resnet18(weights="IMAGENET1K_V1")
# model = models.resnet18(weights=None)  # No usar pesos preentrenados
# model.fc = nn.Linear(model.fc.in_features, num_classes)

model.dropout = nn.Dropout(p=0.6)  # Añadir dropout
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_classes), nn.Softmax(dim=1))

model = model.to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model.eval()
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    print(f"Labels: {labels}")
    print(f"Output: {outputs}")

    _, predicted = torch.max(outputs, 1)
    print(f"Predicted: {predicted}")
    print(f"ArgMax: {torch.argmax(outputs, dim=1)}")

    # Calcular matriz de confusión
    cm = confusion_matrix(
        torch.argmax(labels, dim=1).cpu().numpy(),
        torch.argmax(outputs, dim=1).cpu().numpy(),
        labels=list(range(num_classes)),
    )

    print(f"Confusion Matrix:\n{cm}")

    break

model.train()

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

epochs = 100

epoch_iterator = tqdm(range(epochs), desc="Training Progress")
for epoch in epoch_iterator:
    epoch_iterator.set_description(f"Epoch {epoch + 1}/{epochs}")

    corrects = 0
    total = 0

    batch_losses = []

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

        preds = outputs.argmax(dim=1)
        trues = labels.argmax(dim=1)
        batch_corrects = (preds == trues).sum().item()

        corrects += batch_corrects
        total += labels.size(0)

        batch_losses.append(loss_value.item())

    train_loss = sum(batch_losses) / len(batch_losses)
    train_accuracy = corrects / len(train_loader.dataset)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    # Valid
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    for val_batch in val_loader:
        val_inputs, val_labels = val_batch
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

        with torch.no_grad():
            outputs = model(val_inputs)
            v_loss = loss(outputs, val_labels)
            preds = outputs.argmax(dim=1)
            trues = val_labels.argmax(dim=1)
            v_corrects = (preds == trues).sum().item()

        val_corrects += v_corrects
        val_total += val_labels.size(0)
        val_loss += v_loss.item() * val_labels.size(0)

    val_loss /= val_total
    val_accuracy = val_corrects / val_total

    # save checkpoint
    if val_accuracy > max(val_accuracies, default=0):
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save(model.state_dict(), os.path.join(os.getcwd(), "checkpoints", "best_model.pth"))

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # print(
    #     f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value.item()}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    # )
    epoch_iterator.set_postfix(
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )

print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")

# Plotear resultados
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(train_losses, label="Train Loss")
axs[0].plot(val_losses, label="Validation Loss")
axs[0].set_title("Loss per Epoch")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(train_accuracies, label="Train Accuracy")
axs[1].plot(val_accuracies, label="Validation Accuracy")
axs[1].set_title("Accuracy per Epoch")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
plt.tight_layout()

plt.savefig("training_results.png")

best_model = models.resnet18(weights="IMAGENET1K_V1")
best_model.fc = nn.Sequential(nn.Linear(best_model.fc.in_features, num_classes), nn.Softmax(dim=1))
best_model = best_model.to(device)
# Cargar el mejor modelo guardado
best_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "checkpoints", "best_model.pth")))
# Evaluar el mejor modelo en el conjunto de prueba
# best_model = model

best_model.eval()
with torch.no_grad():
    test_loss = 0.0
    test_corrects = 0
    test_total = 0
    for test_batch in test_loader:
        test_inputs, test_labels = test_batch
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

        outputs = best_model(test_inputs)
        t_loss = loss(outputs, test_labels)
        preds = outputs.argmax(dim=1)
        trues = test_labels.argmax(dim=1)
        t_corrects = (preds == trues).sum().item()
        test_corrects += t_corrects
        test_total += test_labels.size(0)
        test_loss += t_loss.item() * test_labels.size(0)

    test_loss /= test_total
    test_accuracy = test_corrects / test_total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


# Confusion matrix


def plot_confusion_matrix(y_true, y_pred, classes, set_name="Test"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - " + set_name)
    plt.show()


# train
y_true = []
y_pred = []
for train_batch in train_loader:
    train_inputs, train_labels = train_batch
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

    with torch.no_grad():
        outputs = model(train_inputs)
        pred = outputs.argmax(dim=1)
        true = train_labels.argmax(dim=1) if train_labels.dim() > 1 else train_labels
        y_true.extend(true.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
# Convertir a numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
# Plotear matriz de confusión
plot_confusion_matrix(y_true, y_pred, classes.split("_"), set_name="Train")

# val
y_true = []
y_pred = []
for val_batch in val_loader:
    val_inputs, val_labels = val_batch
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

    with torch.no_grad():
        outputs = model(val_inputs)
        pred = outputs.argmax(dim=1)
        true = val_labels.argmax(dim=1) if val_labels.dim() > 1 else val_labels
        y_true.extend(true.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
# Convertir a numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
# Plotear matriz de confusión
plot_confusion_matrix(y_true, y_pred, classes.split("_"), set_name="Validation")

# Obtener predicciones del conjunto de prueba
y_true = []
y_pred = []
for test_batch in test_loader:
    test_inputs, test_labels = test_batch
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    with torch.no_grad():
        outputs = model(test_inputs)
        pred = outputs.argmax(dim=1)
        true = test_labels.argmax(dim=1) if test_labels.dim() > 1 else test_labels
        y_true.extend(true.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# Convertir a numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
# Plotear matriz de confusión
plot_confusion_matrix(y_true, y_pred, classes.split("_"), set_name="Test")

#
