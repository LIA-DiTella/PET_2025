import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class PETDataset(Dataset):
    """Dataset para imágenes PET 2D (cortes)."""

    def __init__(
        self,
        data_dir,
        csv_path,
        transform=None,
        slice_selection="uniform",
        num_slices=16,
        mode="train",
        verbose=True,
        limit=None,
        is_3d=False,
        class_count=2,
    ) -> None:
        """Inicializa el dataset.

        Args:
            data_dir (str): Directorio con las imágenes
            csv_path (str): Ruta al CSV con metadatos
            transform: Transformaciones a aplicar
            slice_selection (str): Método de selección de cortes ('middle', 'all', 'uniform')
            num_slices (int): Número de cortes a seleccionar por volumen
            mode (str): Modo de operación ('train', 'val', 'test')
            limit (int, opcional): Límite de sujetos a cargar (para pruebas)

        """
        self.data_dir = data_dir
        self.transform = transform
        self.slice_selection = slice_selection
        self.num_slices = num_slices
        self.mode = mode
        self.limit = limit
        self.class_count = class_count  # CN, MCI, AD
        self.verbose = verbose
        self.is_3d = is_3d

        # Cargar metadatos
        self.metadata = pd.read_csv(csv_path)
        if self.verbose:
            print(f"CSV cargado. Filas: {len(self.metadata)}")

        # Obtener lista de imágenes y etiquetas
        self.samples = []
        self._load_dataset()

    def _label_to_index(self, label, class_count=3) -> int:
        """Convierte etiquetas de diagnóstico a índices numéricos."""
        if isinstance(label, str):
            # Normalizar label (convertir a mayúsculas y eliminar espacios)
            label = label.upper().strip()
            # Controles normales

            if self.class_count == 2:
                if label in ["CN", "SMC"]:
                    return 0
                elif label in ["AD"]:
                    return 1
                else:
                    raise ValueError(f"Etiqueta desconocida: {label}")
            elif self.class_count == 3:
                if label in ["CN", "SMC"]:
                    return 0
                elif label in ["MCI", "EMCI", "LMCI"]:
                    return 1
                elif label in ["AD"]:
                    return 2

            # Si no se reconoce, lanzar error
            else:
                raise ValueError(f"Etiqueta desconocida: {label}")
        else:
            # Si es numérico, convertirlo asegurando que esté en rango
            label_int = int(label) % self.class_count
            return label_int

    def _find_subject_file(self, subject_id):
        """Busca el archivo correspondiente a un sujeto probando diferentes patrones."""
        subject_id = str(subject_id)

        # Lista de patrones posibles para buscar el archivo
        patterns = [
            # Ruta completa
            # os.path.join(self.data_dir, subject_id),
            # Nombre de archivo con extensión
            os.path.join(self.data_dir, f"{subject_id}.nii.gz"),
            # os.path.join(self.data_dir, f"{subject_id}.nii"),
            # Subdirectorio con nombre de sujeto
            # os.path.join(self.data_dir, subject_id, f"{subject_id}.nii.gz"),
            # os.path.join(self.data_dir, subject_id, f"{subject_id}.nii"),
            # os.path.join(self.data_dir, subject_id, "pet.nii.gz"),
            # os.path.join(self.data_dir, subject_id, "pet.nii"),
        ]

        # Probar cada patrón
        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern

        # Si llegamos aquí, no se encontró el archivo
        return None

    def _load_dataset(self) -> None:
        """Carga información de las imágenes y prepara dataset."""
        # Identificar columnas relevantes
        subject_columns = ['Subject', 'subject', 'ID', 'PTID', 'path', 'filename']
        diagnosis_columns = ['Group', 'group', 'diagnosis', 'DX', 'class', 'label']

        # Encontrar la columna de sujeto
        subject_col = None
        for col in subject_columns:
            if col in self.metadata.columns:
                subject_col = col
                break

        # Si no se encontró, buscar columna con valores únicos
        if subject_col is None:
            for col in self.metadata.columns:
                if len(self.metadata[col].unique()) >= len(self.metadata) * 0.9:  # Al menos 90% de valores únicos
                    subject_col = col
                    if self.verbose:
                        print(f"Usando columna '{col}' como identificador de sujetos (valores únicos)")
                    break

        if subject_col is None:
            raise ValueError("No se pudo identificar una columna para identificar sujetos")

        # Encontrar la columna de diagnóstico
        diagnosis_col = None
        for col in diagnosis_columns:
            if col in self.metadata.columns:
                diagnosis_col = col
                break

        # Si no se encontró, buscar columna con pocos valores únicos (típico de diagnósticos)
        if diagnosis_col is None:
            for col in self.metadata.columns:
                unique_values = self.metadata[col].nunique()
                if 2 <= unique_values <= 5:  # Típico para diagnósticos
                    diagnosis_col = col
                    if self.verbose:
                        print(f"Usando columna '{col}' como diagnóstico ({unique_values} valores únicos)")
                    break

        if diagnosis_col is None:
            raise ValueError("No se pudo identificar una columna para diagnósticos")

        if self.verbose:
            print(f"Usando columna '{subject_col}' para identificar sujetos")
            print(f"Usando columna '{diagnosis_col}' para diagnósticos")
            print(f"Valores únicos en '{diagnosis_col}': {self.metadata[diagnosis_col].unique().tolist()}")

        # Procesar cada fila en el CSV
        count = 0
        # for _, row in self.metadata.iterrows():
        for index, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Cargando dataset", disable=not self.verbose):
            # Si hay un límite de sujetos y se alcanzó, detener
            if self.limit is not None and count >= self.limit:
                break

            # Obtener ID del sujeto
            subject_id = str(row[subject_col])

            # Encontrar archivo del sujeto
            img_path = self._find_subject_file(subject_id)

            if img_path is None:
                if self.verbose:
                    print(f"No se encontró archivo para el sujeto: {subject_id}")
                continue

            # Procesar etiqueta
            try:
                label = self._label_to_index(row[diagnosis_col], self.class_count)
            except Exception as e:
                if self.verbose:
                    print(f"Error al procesar etiqueta '{row[diagnosis_col]}': {e}")
                continue

            # Cargar y procesar imagen
            try:
                nifti = nib.load(img_path)
                img_data = nifti.get_fdata()

                if self.verbose and count == 0:
                    print(f"Cargado volumen de forma: {img_data.shape}")

                # if dynamic PET, seleccionar el primer volumen
                if img_data.ndim == 4:
                    img_data = img_data[:, :, :, 0]
                elif img_data.ndim != 3:
                    raise ValueError(f"Formato de imagen no soportado: {img_data.ndim} dimensiones")

                # Seleccionar cortes según el método especificado
                if self.slice_selection == "middle":
                    # Corte central y adyacentes
                    middle_idx = img_data.shape[2] // 2
                    start_idx = middle_idx - (self.num_slices // 2)
                    end_idx = start_idx + self.num_slices

                    # Asegurar índices en rango
                    start_idx = max(0, start_idx)
                    end_idx = min(img_data.shape[2], end_idx)

                    slice_indices = list(range(start_idx, end_idx))

                elif self.slice_selection == "uniform":
                    # Seleccionar cortes uniformemente distribuidos
                    # 16 imágenes igualmente separadas entre ellas a lo largo del eje axial (redondeando al slice más cercano).
                    slice_indices = np.linspace(0, img_data.shape[2] - 1, self.num_slices, dtype=int).tolist()

                elif self.slice_selection == "all":
                    # Tomar todos los cortes
                    slice_indices = list(range(0, img_data.shape[2]))

                else:
                    raise ValueError(f"Método de selección de cortes desconocido: {self.slice_selection}")

                # Crear un array con todos los cortes seleccionados de una vez
                slices_data = np.array([img_data[:, :, idx] for idx in slice_indices if 0 <= idx < img_data.shape[2]])

                # Normalizar todos los cortes de una vez (por corte individual)
                for i in range(len(slices_data)):
                    slice_min, slice_max = slices_data[i].min(), slices_data[i].max()
                    if slice_max > slice_min:
                        slices_data[i] = (slices_data[i] - slice_min) / (slice_max - slice_min)

                # Redimensionar todos los cortes a 512x512
                image = np.zeros((128, 128, len(slices_data)), dtype=np.float32)
                for i in range(len(slices_data)):
                    image[:, :, i] = resize(slices_data[i], (128, 128), anti_aliasing=True)

                if not self.is_3d:
                    # Make grid in 2D image, handling cases with fewer slices
                    if len(slices_data) == 1:
                        # Si solo hay un corte, usarlo directamente
                        image = image[:, :, 0]
                    else:
                        # Número de columnas por fila (máximo 4)
                        cols_per_row = min(4, len(slices_data))
                        rows = []

                        # Crear filas
                        for j in range(0, len(slices_data), cols_per_row):
                            # Para cada fila, obtener tantos cortes como sea posible sin exceder el límite
                            available_cols = min(cols_per_row, len(slices_data) - j)
                            row_slices = [image[:, :, j + i] for i in range(available_cols)]

                            # Si no hay suficientes para completar la fila, añadir arrays vacíos
                            while len(row_slices) < cols_per_row:
                                # Usar arrays de ceros con la misma forma que los otros cortes
                                row_slices.append(np.zeros_like(row_slices[0]))

                            # Concatenar horizontalmente para formar la fila
                            rows.append(np.concatenate(row_slices, axis=1))

                        # Concatenar verticalmente todas las filas
                        image = np.concatenate(rows, axis=0)

                # Añadir a las muestras
                self.samples.append((image, label))                    
                count += 1

            except Exception as e:
                if self.verbose:
                    print(f"Error al procesar imagen {img_path}: {e}")

                import traceback
                traceback.print_exc()

        if self.verbose:
            print(f"Total de muestras cargadas: {len(self.samples)} de {count} sujetos")

            # Análisis de distribución de clases
            if self.samples:
                labels = [label for _, label in self.samples]
                unique_labels, counts = np.unique(labels, return_counts=True)

                print("\nDistribución de clases:")
                for i in range(len(unique_labels)):
                    label_val = unique_labels[i]
                    count = counts[i]
                    if self.class_count == 2:
                        class_name = "CN" if label_val == 0 else "AD" if label_val == 1 else f"Clase {label_val}"
                    else:
                        class_name = "CN" if label_val == 0 else "MCI" if label_val == 1 else "AD" if label_val == 2 else f"Clase {label_val}"

                    print(f"  - {class_name}: {count} muestras ({count/len(labels)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        print(type(img), img.shape, label)

        if not self.is_3d:
            # Añadir dimensión de canal para 2D (C, H, W)
            img = img[np.newaxis, :, :]  # Añadir dimensión de canal (1, H, W)

        print(f"Imagen {idx}: forma {img.shape}, etiqueta {label}")

        # Convertir a tensor de PyTorch
        # img = torch.from_numpy(img).float()

        # Aplicar transformaciones si existen
        if self.transform:
            # img = self.transform(img)
            img = transforms.functional.to_tensor(img)  # Convertir a tensor
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            img = transforms.functional.resize(img, (256, 256))  # Redimensionar a 256x256
            img = transforms.functional.center_crop(img, (224, 224))  # Recortar al centro a 224x224
            img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar            

        print(img.shape, label)

        return img, label


def get_data_loaders(config):
    """Crea los data loaders para los conjuntos de entrenamiento, validación y prueba.

    Args:
        config (dict): Diccionario de configuración

    Returns:
        tuple: (train_loader, val_loader, test_loader)

    """
    data_config = config.get("data", {})
    classes = data_config.get("classes", "CN_AD").split("_")
    for class_name in classes:
        if class_name not in ["CN", "MCI", "AD", "SMC", "EMCI", "LMCI"]:
            raise ValueError(f"Clase desconocida: {class_name}. Debe ser CN, MCI, AD, SMC, EMCI o LMCI.")
    num_classes = len(classes)
    if num_classes not in [2, 3]:
        raise ValueError(f"Número de clases no soportado: {num_classes}. Debe ser 2 o 3.")
    print(f"Configuración de clases: {classes} ({num_classes} clases)")
    is_3d = data_config.get("dimension", "2d") == "3d"
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)

    # Directorios y archivos
    data_dir = data_config.get("data_dir", "./data")
    train_csv = data_config.get("train_csv")
    val_csv = data_config.get("val_csv")
    test_csv = data_config.get("test_csv")

    # Transformaciones
    train_transform = get_transforms(data_config, is_train=True, is_3d=is_3d)
    test_transform = get_transforms(data_config, is_train=False, is_3d=is_3d)
    slice_selection = data_config.get("slice_selection", "uniform")
    num_slices = data_config.get("num_slices", 16)

    train_limit = data_config.get("train_limit", None)
    val_limit = data_config.get("val_limit", None)
    test_limit = data_config.get("test_limit", None)

    train_dataset = PETDataset(
        data_dir,
        train_csv,
        transform=train_transform,
        slice_selection=slice_selection,
        num_slices=num_slices,
        mode="train",
        is_3d=is_3d,
        class_count=num_classes,
        limit=train_limit,
    )

    val_dataset = (
        PETDataset(
            data_dir,
            val_csv,
            transform=test_transform,
            slice_selection="uniform",
            num_slices=1,
            mode="val",
            is_3d=is_3d,
            class_count=num_classes,
            limit=val_limit,
        )
        if val_csv
        else None
    )

    test_dataset = (
        PETDataset(
            data_dir,
            test_csv,
            transform=test_transform,
            slice_selection="uniform",
            num_slices=1,
            mode="test",
            is_3d=is_3d,
            class_count=num_classes,
            limit=test_limit,
        )
        if test_csv
        else None
    )

    # Crear data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if val_dataset
        else None
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_dataset
        else None
    )

    return train_loader, val_loader, test_loader


def get_transforms(config, is_train=True, is_3d=False):
    """Crea transformaciones para aumentación y preprocesamiento.

    Args:
        config (dict): Diccionario de configuración
        is_train (bool): Si es para entrenamiento
        is_3d (bool): Si es para datos 3D

    Returns:
        transformación compuesta

    """

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform
