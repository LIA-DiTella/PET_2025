import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from utils.config_utils import save_config

# Importar utilidades
from utils.evaluation_utils import calculate_metrics, save_results_to_csv


class Trainer:
    """Clase para entrenar modelos de clasificación de imágenes médicas."""

    def __init__(self, model, config, device=None, exp_dir=None) -> None:
        """Inicializa el entrenador.

        Args:
            model: Modelo PyTorch a entrenar
            config (dict): Configuración del entrenamiento
            device: Dispositivo para el entrenamiento (CPU/GPU)
            exp_dir (str): Directorio para guardar resultados del experimento

        """
        self.model = model
        self.config = config
        self.exp_dir = exp_dir or "./experiments"

        # Crear directorios de salida
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "results"), exist_ok=True)

        # Guardar configuración
        save_config(config, os.path.join(self.exp_dir, "config.yaml"))

        # Dispositivo
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Configuración de entrenamiento
        self.epochs = config.get("training", {}).get("epochs", 100)
        self.patience = config.get("training", {}).get("early_stopping_patience", 10)

        # Función de pérdida
        self.criterion = self._get_loss_function()

        # Optimizador
        self.optimizer = self._get_optimizer()

        # Scheduler de tasa de aprendizaje
        self.scheduler = self._get_scheduler()

        # Configuración de logging
        self._setup_logging()

    def _get_loss_function(self):
        """Configura la función de pérdida según la configuración."""
        loss_config = self.config.get("training", {}).get("loss", {})
        loss_name = loss_config.get("name", "cross_entropy")

        if loss_name == "cross_entropy":
            # Opción de pesos para clases desbalanceadas
            if loss_config.get("weighted", False):
                class_weights = torch.tensor(loss_config.get("class_weights"), device=self.device)
                return nn.CrossEntropyLoss(weight=class_weights)
            return nn.CrossEntropyLoss()

        if loss_name == "focal_loss":
            from kornia.losses import FocalLoss

            alpha = loss_config.get("alpha", 0.5)
            gamma = loss_config.get("gamma", 2.0)
            return FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

        self.logger.warning(
            f"Función de pérdida {loss_name} no reconocida. Usando CrossEntropyLoss por defecto.",
        )
        return nn.CrossEntropyLoss()

    def _get_optimizer(self):
        """Configura el optimizador según la configuración."""
        optim_config = self.config.get("training", {}).get("optimizer", {})
        optim_name = optim_config.get("name", "adam")
        lr = optim_config.get("lr", 0.001)
        weight_decay = optim_config.get("weight_decay", 0)

        if optim_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )

        if optim_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=optim_config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )

        if optim_name.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.logger.warning(f"Optimizador {optim_name} no reconocido. Usando Adam por defecto.")
        return optim.Adam(self.model.parameters(), lr=lr)

    def _get_scheduler(self):
        """Configura el scheduler según la configuración."""
        sched_config = self.config.get("training", {}).get("scheduler", {})
        sched_name = sched_config.get("name", None)

        if not sched_name:
            return None

        if sched_name == "cosine_annealing":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get("t_max", self.epochs),
                eta_min=sched_config.get("eta_min", 0),
            )

        if sched_name == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.get("mode", "min"),
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 5),
            )

        if sched_name == "step_lr":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get("step_size", 30),
                gamma=sched_config.get("gamma", 0.1),
            )

        self.logger.warning(f"Scheduler {sched_name} no reconocido. No se usará scheduler.")
        return None

    def _setup_logging(self) -> None:
        """Configura el sistema de logging."""
        self.logger = logging.getLogger("model_trainer")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # Log a consola
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # Log a archivo
            file_handler = logging.FileHandler(os.path.join(self.exp_dir, "training.log"))
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def train(self, train_loader, val_loader):
        """Entrena el modelo.

        Args:
            train_loader: DataLoader para datos de entrenamiento
            val_loader: DataLoader para datos de validación

        Returns:
            dict: Diccionario con métricas del mejor modelo

        """
        best_val_metric = 0.0  # Para guardar el mejor modelo según AUC ROC
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()

        self.logger.info(f"Iniciando entrenamiento en {self.device}")
        self.logger.info(f"Configuración: {self.config}")

        for epoch in range(1, self.epochs + 1):
            # Entrenamiento
            train_loss, train_acc = self._train_epoch(train_loader, epoch)

            # Evaluación
            if val_loader:
                val_metrics = self._evaluate(val_loader)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]
                # val_auc = val_metrics.get("auc_roc", 0.0)

                # Logging
                self.logger.info(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
                )

                # Actualizar scheduler si es ReduceLROnPlateau
                if self.scheduler and isinstance(
                    self.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
            else:
                self.logger.info(
                    f"Epoch {epoch}/{self.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}",
                )
                # Actualizar scheduler normal
                if self.scheduler and not isinstance(
                    self.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step()

            # Actualizar scheduler normal
            if self.scheduler and not isinstance(
                self.scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.scheduler.step()

            # Guardar el mejor modelo (basado en AUC ROC o precisión)
            current_metric = val_acc

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0

                # Guardar mejor modelo
                checkpoint_path = os.path.join(self.exp_dir, "checkpoints", "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": val_metrics,
                    },
                    checkpoint_path,
                )

                self.logger.info(
                    f"Nuevo mejor modelo guardado (epoch {epoch}, metric: {current_metric:.4f})",
                )
            else:
                patience_counter += 1

            # Early stopping
            if self.patience > 0 and patience_counter >= self.patience:
                self.logger.info(f"Early stopping activado después de {epoch} epochs")
                break

        # Tiempo total
        total_time = time.time() - start_time
        self.logger.info(f"Entrenamiento completado en {total_time:.2f} segundos")
        self.logger.info(f"Mejor modelo en epoch {best_epoch} con métrica {best_val_metric:.4f}")

        # Cargar el mejor modelo para evaluación final
        if val_loader:
            self._load_best_model()

        return {
            "best_epoch": best_epoch,
            "best_metric": best_val_metric,
            "training_time": total_time,
        }

    def _train_epoch(self, train_loader, epoch):
        """Entrena el modelo durante una época."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            if hasattr(self.model, "module") and hasattr(self.model.module, "aux_logits"):
                output, aux_output = self.model(data)
                loss1 = self.criterion(output, target)
                loss2 = self.criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2  # Ponderación para la salida auxiliar
            else:
                output = self.model(data)
                loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Estadísticas
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            # Actualizar barra de progreso
            pbar.set_postfix({"loss": total_loss / (batch_idx + 1), "acc": correct / total})

        # Estadísticas de la época
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate(self, data_loader):
        """Evalúa el modelo en un conjunto de datos."""
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_scores = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Calcular pérdida
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # Guardar predicciones
                scores = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        # Convertir a arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)

        # Calcular métricas
        metrics = calculate_metrics(all_targets, all_predictions, all_scores)
        metrics["loss"] = total_loss / len(data_loader)

        return metrics

    def _load_best_model(self) -> None:
        """Carga el mejor modelo guardado."""
        checkpoint_path = os.path.join(self.exp_dir, "checkpoints", "best_model.pth")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Mejor modelo cargado de {checkpoint_path}")
        else:
            self.logger.warning(f"No se encontró el mejor modelo en {checkpoint_path}")

    def evaluate(self, test_loader, save_results=True):
        """Evalúa el modelo en el conjunto de prueba.

        Args:
            test_loader: DataLoader para los datos de prueba
            save_results (bool): Si se deben guardar los resultados

        Returns:
            dict: Métricas de evaluación

        """
        metrics = self._evaluate(test_loader)

        # Logging
        self.logger.info("Evaluación final:")
        for metric_name, value in metrics.items():
            if metric_name != "confusion_matrix":
                self.logger.info(f"{metric_name}: {value}")

        # Guardar resultados
        if save_results:
            save_path = os.path.join(self.exp_dir, "results", "test_metrics.csv")
            save_results_to_csv(metrics, save_path)

        return metrics
