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

# Importar wandb para logging de experimentos
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb no está instalado. El logging de wandb está deshabilitado.")

from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


class Trainer:
    """Clase para entrenar modelos de clasificación de imágenes médicas."""

    def __init__(self, model, config, device=None, exp_dir=None, train_loader=None, val_loader=None, test_loader=None, num_classes=2):
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

        # class_weights = None

        # loss_config = self.config.get("training", {}).get("loss", {})
        # Calcular pesos de clases si no se proporcionan
        # class_counts = np.zeros(num_classes)
        # for _, targets in train_loader:
        #     for target in targets:
        #         class_counts[target.item()] += 1
        # # total_count = np.sum(class_counts)
        # class_weights = torch.tensor(
        #     total_count / (num_classes * class_counts), dtype=torch.float32
        # ).to(self.device)

        y = torch.cat([targets for _, targets in train_loader], dim=0)

        class_weights = compute_class_weight("balanced", classes=np.unique(y.numpy()), y=y.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        print(f"Calculated Class Weights: {class_weights}")

        # Función de pérdida
        self.criterion = self._get_loss_function(class_weights=class_weights)

        # Optimizador
        self.optimizer = self._get_optimizer()

        # Scheduler de tasa de aprendizaje
        self.scheduler = self._get_scheduler()

        # Configuración de logging
        self._setup_logging()

        # Configuración de wandb
        self._setup_wandb()

    def _get_loss_function(self, class_weights=None):
        """Configura la función de pérdida según la configuración."""
        loss_config = self.config.get("training", {}).get("loss", {})
        loss_name = loss_config.get("name", "cross_entropy")
        print(f"Using Class Weights: {class_weights}")

        if loss_name == "cross_entropy":
            # Opción de pesos para clases desbalanceadas
            if loss_config.get("weighted", False):
                if class_weights is None:
                    if loss_config.get("class_weights", None) is not None:
                        class_weights = torch.Tensor(loss_config["class_weights"])
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
                # betas=(0.9, 0.999),
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

    def _setup_wandb(self) -> None:
        """Configura wandb para logging de experimentos."""
        wandb_config = self.config.get("wandb", {})

        if not WANDB_AVAILABLE or not wandb_config.get("enabled", False):
            self.use_wandb = False
            return

        self.use_wandb = True

        # Inicializar wandb
        wandb.init(
            project=wandb_config.get("project", "pet-classification"),
            entity=wandb_config.get("entity", None),
            name=wandb_config.get("name", None),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            group=wandb_config.get("group", None),
            job_type=wandb_config.get("job_type", "train"),
            config=self.config,
            dir=self.exp_dir,
            resume="allow"
        )

        # Hacer seguimiento del modelo si está habilitado
        if wandb_config.get("watch_model", False):
            wandb.watch(
                self.model,
                log="all",
                log_freq=wandb_config.get("watch_freq", 100)
            )

        # Configurar frecuencia de logging
        self.log_frequency = wandb_config.get("log_frequency", 10)

        self.logger.info("Wandb configurado correctamente")

    def close_wandb(self):
        """Cierra la sesión de wandb."""
        if self.use_wandb:
            wandb.finish()
            self.logger.info("Sesión de wandb cerrada")

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
            self.model.train()
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            train_metrics = self._evaluate(train_loader)
            # train_loss = train_metrics["loss"]
            # train_acc = train_metrics["accuracy"]
            train_auc = train_metrics.get("auc_roc", 0.0)


            self.model.eval()
            # Evaluación
            val_metrics = self._evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_auc = val_metrics.get("auc_roc", 0.0)

            # Logging
            # self.logger.info(
            #     f"Epoch {epoch}/{self.epochs} - "
            #     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            #     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}",
            # )

            # self.logger.info(
            #     str({
            #         "epoch": epoch,
            #         "train_loss": train_loss,
            #         "train_accuracy": train_acc,
            #         "val_loss": val_loss,
            #         "val_accuracy": val_acc,
            #         "learning_rate": self.optimizer.param_groups[0]["lr"],
            #     })
            # )

            # Logging a wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "train_auc_roc": train_auc,
                "val_auc_roc": val_auc,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, step=epoch)

            # Actualizar scheduler si es ReduceLROnPlateau
            if self.scheduler and isinstance(
                self.scheduler,
                optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.scheduler.step(val_loss)

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
                        "val_metrics": val_metrics,
                        "train_metrics": train_metrics,
                    },
                    checkpoint_path,
                )

                self.logger.info(
                    f"Nuevo mejor modelo guardado (epoch {epoch}, metric: {current_metric:.4f})",
                )
            else:
                patience_counter += 1

            # Early stopping
            # if self.patience > 0 and patience_counter >= self.patience:
            #     self.logger.info(f"Early stopping activado después de {epoch} epochs")
            #     break

        # Tiempo total
        total_time = time.time() - start_time
        self.logger.info(f"Entrenamiento completado en {total_time:.2f} segundos")
        self.logger.info(f"Mejor modelo en epoch {best_epoch} con métrica {best_val_metric:.4f}")

        # Cargar el mejor modelo para evaluación final
        self._load_best_model()

        # Logging final
        self.logger.info("Entrenamiento finalizado")

        # Confussion matrix on Train set
        self.model.eval()
        train_metrics = self._evaluate(train_loader)
        self.logger.info("Métricas de entrenamiento:")
        for metric_name, value in train_metrics.items():
            if metric_name != "confusion_matrix":
                self.logger.info(f"{metric_name}: {value}")
        if "confusion_matrix" in train_metrics:
            self.logger.info("Matriz de confusión en el conjunto de entrenamiento:")
            self.logger.info(train_metrics["confusion_matrix"])

            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                plt.figure(figsize=(8, 6))
                sns.heatmap(train_metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
                plt.title('Matriz de Confusión - Conjunto de Entrenamiento')
                plt.ylabel('Etiqueta Real')
                plt.xlabel('Predicción')

                # Guardar y loggear a wandb
                wandb.log({"train_confusion_matrix": wandb.Image(plt)})
                plt.close()
            except ImportError:
                self.logger.warning("matplotlib y/o seaborn no están disponibles para visualizar la matriz de confusión")

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
            
            # if hasattr(self.model, "module") and hasattr(self.model.module, "aux_logits"):
            #     output, aux_output = self.model(data)
            #     loss1 = self.criterion(output, target)
            #     loss2 = self.criterion(aux_output, target)
            #     loss = loss1 + 0.4 * loss2  # Ponderación para la salida auxiliar
            # else:

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

            # Logging detallado de batch a wandb
            # if self.use_wandb and hasattr(self, 'log_frequency') and batch_idx % self.log_frequency == 0:
            #     wandb.log({
            #         "batch_loss": loss.item(),
            #         "batch_accuracy": correct / total,
            #         "batch": (epoch - 1) * len(train_loader) + batch_idx
            #     })

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
                scores = output  # Asumiendo que output ya son logits
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

        # Añadir ROC-AUC si hay más de una clase
        if len(np.unique(all_targets)) > 1:
            try:
                if len(np.unique(all_targets)) == 2:
                    # Binario: usar probabilidades de la clase positiva
                    metrics["auc_roc"] = roc_auc_score(all_targets, all_scores[:, 1])
                # else:
                    # Multiclase: usar average='macro'
                    # metrics["auc_roc"] = roc_auc_score(all_targets, all_scores, multi_class='ovr', average='macro')
            except Exception as e:
                self.logger.warning(f"No se pudo calcular ROC-AUC: {e}")
                metrics["auc_roc"] = 0.0

        return metrics

    def _load_best_model(self) -> None:
        """Carga el mejor modelo guardado."""
        checkpoint_path = os.path.join(self.exp_dir, "checkpoints", "best_model.pth")

        if os.path.exists(checkpoint_path):
            # Usar weights_only=False para cargar checkpoints propios que contienen métricas numpy
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

        # Logging a wandb
        if self.use_wandb:
            # Preparar métricas para wandb
            wandb_metrics = {}
            for metric_name, value in metrics.items():
                if metric_name != "confusion_matrix":
                    wandb_metrics[f"test_{metric_name}"] = value

            # Logging de la matriz de confusión como imagen si está disponible
            if "confusion_matrix" in metrics:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
                    plt.title('Matriz de Confusión - Conjunto de Prueba')
                    plt.ylabel('Etiqueta Real')
                    plt.xlabel('Predicción')

                    # Guardar y loggear a wandb
                    wandb_metrics["test_confusion_matrix"] = wandb.Image(plt)
                    plt.close()
                except ImportError:
                    self.logger.warning("matplotlib y/o seaborn no están disponibles para visualizar la matriz de confusión")

            wandb.log(wandb_metrics)

        # Guardar resultados
        if save_results:
            save_path = os.path.join(self.exp_dir, "results", "test_metrics.csv")
            save_results_to_csv(metrics, save_path)

        return metrics
