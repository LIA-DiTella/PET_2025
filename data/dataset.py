import os

# import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from nilearn import image as nli
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def make_resample(_df, column):
    ad = _df[_df[column] == "AD"]
    cn = _df[_df[column] == "CN"]
    smc = _df[_df[column] == "SMC"]
    mci = _df[_df[column] == "MCI"]
    emci = _df[_df[column] == "EMCI"]
    lmci = _df[_df[column] == "LMCI"]

    # Upsample minority class (AD)
    ad_count = len(ad)
    cn_count = len(cn) + len(smc)

    while ad_count < cn_count:
        ad = pd.concat(
            [ad, resample(ad, replace=True, n_samples=cn_count - ad_count, random_state=42)]
        )
        ad_count = len(ad)

    # Concatenar de nuevo
    # df_resampled = pd.concat([ad, cn, smc], ignore_index=True)
    df_resampled = pd.concat([ad, cn, smc, mci, emci, lmci], ignore_index=True)

    # Mezclar filas
    df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_resampled


def torch_resize_2d(image, target_size, mode="nearest"):
    """
    Resize a 2D image using PyTorch with nearest neighbor interpolation.

    Args:
        image (np.ndarray): Input image of shape (H, W)
        target_size (tuple): Target size (H, W)
        mode (str): Interpolation mode ('nearest', 'bilinear', etc.)

    Returns:
        np.ndarray: Resized image
    """
    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Resize using torch.nn.functional.interpolate
    resized_tensor = torch.nn.functional.interpolate(
        img_tensor, size=target_size, mode=mode, align_corners=False if mode != "nearest" else None
    )

    # Remove batch and channel dimensions and convert back to numpy
    resized_image = resized_tensor.squeeze(0).squeeze(0).numpy()

    return resized_image


def torch_resize_3d(image, target_size, mode="nearest"):
    """
    Resize a 3D image using PyTorch with nearest neighbor interpolation.

    Args:
        image (np.ndarray): Input image of shape (H, W, D)
        target_size (tuple): Target size (H, W, D)
        mode (str): Interpolation mode ('nearest', 'trilinear', etc.)

    Returns:
        np.ndarray: Resized image
    """
    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)

    # Resize using torch.nn.functional.interpolate
    resized_tensor = torch.nn.functional.interpolate(
        img_tensor, size=target_size, mode=mode, align_corners=False if mode != "nearest" else None
    )

    # Remove batch and channel dimensions and convert back to numpy
    resized_image = resized_tensor.squeeze(0).squeeze(0).numpy()

    return resized_image


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
        config=None,
        dynamic_all=False,
        output_size=(224, 224),
        channels=3,
        use_otsu_masking=True,
        use_surface_filtering=False,
        min_slice_surface=100.0 * 100.0,
        pixel_surface=1.5,
        min_slice_surface_threshold=0.0,
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
            use_otsu_masking (bool): Si aplicar umbralizado de Otsu para máscara cerebral
            use_surface_filtering (bool): Si aplicar filtrado por superficie mínima de cortes
            min_slice_surface (float): Superficie mínima requerida para conservar un corte
            pixel_surface (float): Superficie por píxel (mm²)
            min_slice_surface_threshold (float): Umbral mínimo de intensidad para considerar un píxel

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
        self.dynamic_all = (
            dynamic_all  # Si es True, procesa todos los cortes de un volumen dinámico
        )
        self.output_size = output_size  # Tamaño de salida para las imágenes procesadas
        self.channels = channels  # Número de canales de salida (1 para 2D, 3 para RGB)
        self.use_otsu_masking = use_otsu_masking  # Si aplicar umbralizado de Otsu

        # Parámetros para filtrado por superficie
        self.use_surface_filtering = use_surface_filtering
        self.min_slice_surface = min_slice_surface
        self.pixel_surface = pixel_surface
        self.min_slice_surface_threshold = min_slice_surface_threshold

        # Cargar metadatos
        df = pd.read_csv(csv_path)

        # upsample minority class

        self.metadata = (
            make_resample(df, "Group")
            if (mode == "train" and config and config.get("data", {}).get("resample", False))
            else df
        )  # Resample para entrenamiento si es necesario

        if self.verbose:
            print(f"CSV cargado. Filas: {len(self.metadata)}")
            print("3D" * (is_3d) + "2D" * (not is_3d))
            print(f"Modo: {self.mode}, Selección de cortes: {self.slice_selection}")
            print(
                f"Clase(s): {self.class_count} ({'CN, MCI, AD' if self.class_count == 3 else 'CN, AD'})"
            )
            print(f"Output size: {self.output_size}, Canales: {self.channels}")
            if self.use_otsu_masking:
                print("Usando umbralizado de Otsu para máscara cerebral")

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

    def process_image(self, img_data):
        # Aplicar filtrado por superficie si está habilitado
        slice_indices_to_delete = []
        if self.use_surface_filtering:
            try:
                slice_indices_to_delete = get_indices_to_be_deleted(
                    img_data,
                    min_slice_surface=self.min_slice_surface,
                    pixel_surface=self.pixel_surface,
                    min_slice_surface_threshold=self.min_slice_surface_threshold,
                )
                if self.verbose and len(slice_indices_to_delete) > 0:
                    print(
                        f"Eliminando {len(slice_indices_to_delete)} cortes por superficie insuficiente"
                    )

                # Eliminar cortes con superficie insuficiente
                img_data = np.delete(img_data, slice_indices_to_delete, axis=2)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: No se pudo aplicar filtrado por superficie: {e}")

        # Aplicar umbralizado de Otsu si está habilitado
        if self.use_otsu_masking:
            img_data = apply_brain_mask(img_data, use_otsu=True)

        # z-score normalization
        img_data = (img_data - np.mean(img_data)) / np.std(img_data)

        if self.slice_selection == "middle":
            # Corte central y adyacentes
            middle_idx = img_data.shape[2] // 2
            start_idx = middle_idx - (self.num_slices // 2)
            end_idx = start_idx + self.num_slices

            # Asegurar índices en rango
            start_idx = max(0, start_idx)
            end_idx = min(img_data.shape[2], end_idx)

            slice_indices = list(range(start_idx, end_idx))
        elif self.slice_selection == "intensity":
            # Seleccionar cortes basados en intensidad
            # num_slices basado en la intensidad de los cortes (suma de valores absolutos por corte)

            intensity_dist = np.sum(np.abs(img_data), axis=(0, 1))
            # Obtener el índice del corte con mayor intensidad
            top_indices = np.argsort(intensity_dist)[-self.num_slices :]
            # Ordenar los índices seleccionados
            slice_indices = sorted(top_indices.tolist())
        elif self.slice_selection == "uniform":
            # Seleccionar cortes uniformemente distribuidos
            # Para 3D, asegurar que siempre tengamos exactamente num_slices
            if self.is_3d:
                slice_indices = np.linspace(
                    0, img_data.shape[2] - 1, self.num_slices, dtype=int
                ).tolist()
            else:
                # Para 2D, usar el número disponible o num_slices
                actual_slices = min(self.num_slices, img_data.shape[2])
                slice_indices = np.linspace(
                    0, img_data.shape[2] - 1, actual_slices, dtype=int
                ).tolist()

        elif self.slice_selection == "all":
            # Tomar todos los cortes
            slice_indices = list(range(0, img_data.shape[2]))

        else:
            raise ValueError(f"Método de selección de cortes desconocido: {self.slice_selection}")

        # Crear un array con todos los cortes seleccionados de una vez
        slices_data = np.array(
            [img_data[:, :, idx] for idx in slice_indices if 0 <= idx < img_data.shape[2]]
        )

        # Para 3D, asegurar que siempre tengamos exactamente num_slices
        if self.is_3d and len(slices_data) < self.num_slices:
            # Rellenar con el último slice disponible si es necesario
            last_slice = (
                slices_data[-1] if len(slices_data) > 0 else np.zeros_like(img_data[:, :, 0])
            )
            while len(slices_data) < self.num_slices:
                slices_data = np.append(slices_data, [last_slice], axis=0)

        # Normalizar todos los cortes de una vez (por corte individual)
        # for i in range(len(slices_data)):
        #     slice_min, slice_max = slices_data[i].min(), slices_data[i].max()
        #     if slice_max > slice_min:
        #         slices_data[i] = (slices_data[i] - slice_min) / (slice_max - slice_min)

        if self.is_3d:
            # Para 3D, crear array con dimensiones exactas
            image = np.zeros((128, 128, self.num_slices), dtype=np.float32)
            for i in range(self.num_slices):
                if i < len(slices_data):
                    slice_img = torch_resize_2d(slices_data[i], (128, 128))
                    image[:, :, i] = slice_img
        else:
            # Para 2D, mantener el comportamiento original
            image = np.zeros((128, 128, len(slices_data)), dtype=np.float32)
            for i in range(len(slices_data)):
                slice_img = torch_resize_2d(slices_data[i], (128, 128))
                image[:, :, i] = slice_img

        if not self.is_3d:
            # Make grid in 2D image, handling cases with fewer slices
            # Número de columnas por fila
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

        return image

    def _load_dataset(self) -> None:
        """Carga información de las imágenes y prepara dataset."""
        # Identificar columnas relevantes
        subject_columns = ["Subject", "subject", "ID", "PTID", "path", "filename"]
        diagnosis_columns = ["Group", "group", "diagnosis", "DX", "class", "label"]

        # Encontrar la columna de sujeto
        subject_col = None
        for col in subject_columns:
            if col in self.metadata.columns:
                subject_col = col
                break

        # Si no se encontró, buscar columna con valores únicos
        if subject_col is None:
            for col in self.metadata.columns:
                if (
                    len(self.metadata[col].unique()) >= len(self.metadata) * 0.9
                ):  # Al menos 90% de valores únicos
                    subject_col = col
                    if self.verbose:
                        print(
                            f"Usando columna '{col}' como identificador de sujetos (valores únicos)"
                        )
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
                        print(
                            f"Usando columna '{col}' como diagnóstico ({unique_values} valores únicos)"
                        )
                    break

        if diagnosis_col is None:
            raise ValueError("No se pudo identificar una columna para diagnósticos")

        if self.verbose:
            print(f"Usando columna '{subject_col}' para identificar sujetos")
            print(f"Usando columna '{diagnosis_col}' para diagnósticos")
            print(
                f"Valores únicos en '{diagnosis_col}': {self.metadata[diagnosis_col].unique().tolist()}"
            )

        # Procesar cada fila en el CSV
        count = 0
        # for _, row in self.metadata.iterrows():
        for _index, row in tqdm(
            self.metadata.iterrows(),
            total=len(self.metadata),
            desc="Cargando dataset",
            disable=not self.verbose,
        ):
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
            except ValueError:
                continue

            # Cargar y procesar imagen
            try:
                # nifti = nib.load(img_path)
                # img_data = nifti.get_fdata()

                nifti = nli.load_img(img_path)  # Usar nilearn para cargar imágenes NIfTI
                img_data = nifti.get_fdata()
                if count == 0 and self.verbose:
                    print(f"Cargando imagen: {img_path}")
                    print(f"Dimensiones de la imagen: {img_data.shape}")
                    hdr = nifti.header
                    print(f"Header de la imagen: {hdr}")

                # if img_data.ndim == 4:
                #     TR = nifti.header["pixdim"][4]
                #     print(f"Cleaning - detrend and standardize with TR={TR}")
                #     nifti = nli.clean_img(nifti, detrend=True, standardize=True, t_r=TR)
                #     print(f"Imagen dinámica detectada: {img_path} con TR={TR}")
                #     nifti = nli.mean_img(
                #         nifti,  # Promediar a lo largo del eje temporal si es dinámico
                #         copy_header=True,
                #     )
                #     img_data = nifti.get_fdata()

                if self.verbose and count == 0:
                    print(f"Cargado volumen de forma: {img_data.shape}")

                # if dynamic PET, seleccionar el primer volumen
                if img_data.ndim != 3 and img_data.ndim != 4:
                    raise ValueError(f"Formato de imagen no soportado: {img_data.ndim} dimensiones")

                # if img_data.ndim == 4:
                #     # dynamic_index = 0
                #     # dynamic_index = img_data.shape[3] // 2  # Seleccionar el corte medio si es dinámico
                #     # img_data = img_data[:, :, :, dynamic_index]
                #     if count == 0 and self.verbose:
                #         print(f"Imagen dinámica detectada: {img_path} con {img_data.shape[3]} volúmenes")

                #     if self.dynamic_all:
                #         for i in range(img_data.shape[3]):
                #             img_data_slice = img_data[:, :, :, i]
                #             image = self.process_image(img_data_slice)
                #             # Añadir a las muestras
                #             self.samples.append((image, label))
                #             count += 1
                #     else:
                #         # img_data = np.mean(img_data, axis=3)  # Promediar a lo largo del eje temporal si es dinámico
                #         if count == 0 and self.verbose:
                #             print(f"Usando el primer volumen de la imagen dinámica: {img_path}")

                #         img_data = img_data[:, :, :, 0]  # Usar solo el primer volumen
                #         image = self.process_image(img_data)
                #         # Añadir a las muestras
                #         self.samples.append((image, label))
                #         count += 1
                # else:
                #     if count == 0 and self.verbose:
                #         print("not dynamic PET")
                #     image = self.process_image(img_data)
                #     # Añadir a las muestras
                #     self.samples.append((image, label))
                #     count += 1

                if img_data.ndim == 4:
                    # img_data = img_data[:, :, :, 0]  # Usar solo el primer volumen
                    dynamic_index = (
                        img_data.shape[3] // 2
                    )  # Seleccionar el corte medio si es dinámico
                    img_data = img_data[:, :, :, dynamic_index]  # Usar el corte medio

                    # img_data = np.mean(
                    #     img_data, axis=3
                    # )  # Promediar a lo largo del eje temporal si es dinámico

                elif img_data.ndim != 3:
                    raise ValueError(f"Formato de imagen no soportado: {img_data.ndim} dimensiones")

                image = self.process_image(img_data)
                # Aplicar transformaciones durante la carga
                image = self._apply_transforms(image)

                if self.verbose and count == 0:
                    print(f"Imagen procesada final: {image.shape}")

                ohe_label = np.zeros(self.class_count, dtype=np.float32)
                ohe_label[label] = 1.0
                label = torch.tensor(ohe_label, dtype=torch.float32)

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
                        class_name = (
                            "CN"
                            if label_val == 0
                            else "AD"
                            if label_val == 1
                            else f"Clase {label_val}"
                        )
                    else:
                        class_name = (
                            "CN"
                            if label_val == 0
                            else "MCI"
                            if label_val == 1
                            else "AD"
                            if label_val == 2
                            else f"Clase {label_val}"
                        )

                    print(f"  - {class_name}: {count} muestras ({count / len(labels) * 100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]

        return img, label

    def _apply_transforms(self, image):
        """Aplica transformaciones a la imagen durante la carga del dataset."""
        if not self.transform:
            return image

        if self.is_3d:
            # Para 3D: convertir (H, W, D) -> (C, D, H, W)
            if image.shape == (128, 128, self.num_slices):
                # Redimensionar si es necesario
                if self.output_size != (128, 128):
                    resized_image = np.zeros(
                        (self.output_size[0], self.output_size[1], self.num_slices),
                        dtype=np.float32,
                    )
                    for i in range(self.num_slices):
                        resized_image[:, :, i] = torch_resize_2d(image[:, :, i], self.output_size)
                    image = resized_image

                # Crear tensor con formato (C, D, H, W)
                # Primero transponer (H, W, D) -> (D, H, W)
                # image = image.transpose(2, 0, 1)

                image = torch.tensor(image, dtype=torch.float32)
                image = image.unsqueeze(0)  # (1, D, H, W)

                # Duplicar canales si es necesario (para RGB)
                if self.channels == 3:
                    image = image.repeat(3, 1, 1, 1)  # (3, D, H, W)

            else:
                raise ValueError(f"Dimensiones inesperadas para 3D: {image.shape}")

        else:
            # Para 2D: mantener el comportamiento original
            image = transforms.functional.to_tensor(image)  # Convertir a tensor
            image = transforms.functional.resize(
                image, self.output_size
            )  # Redimensionar a tamaño de salida

            if self.channels != 1:
                image = image.repeat(self.channels, 1, 1)
                image = transforms.functional.normalize(
                    image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

        return image


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
            raise ValueError(
                f"Clase desconocida: {class_name}. Debe ser CN, MCI, AD, SMC, EMCI o LMCI."
            )
    num_classes = len(classes)
    if num_classes not in [2, 3]:
        raise ValueError(f"Número de clases no soportado: {num_classes}. Debe ser 2 o 3.")
    print(f"Configuración de clases: {classes} ({num_classes} clases)")
    is_3d = data_config.get("dimension", "2d") == "3d"
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)

    output_size = data_config.get("output_size", (224, 224))
    channels = data_config.get("channels", 3)  # Número de canales de salida (1 para 2D, 3 para RGB)
    use_otsu_masking = data_config.get("use_otsu_masking", True)  # Umbralizado de Otsu por defecto

    # Parámetros para filtrado por superficie (del preprocesamiento de Hugo)
    use_surface_filtering = data_config.get("use_surface_filtering", False)
    min_slice_surface = data_config.get("min_slice_surface", 100.0 * 100.0)
    pixel_surface = data_config.get("pixel_surface", 1.5)
    min_slice_surface_threshold = data_config.get("min_slice_surface_threshold", 0.0)

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
        config=config,
        output_size=output_size,
        channels=channels,
        use_otsu_masking=use_otsu_masking,
        use_surface_filtering=use_surface_filtering,
        min_slice_surface=min_slice_surface,
        pixel_surface=pixel_surface,
        min_slice_surface_threshold=min_slice_surface_threshold,
    )

    val_dataset = (
        PETDataset(
            data_dir,
            val_csv,
            transform=test_transform,
            slice_selection=slice_selection,
            num_slices=num_slices,
            mode="val",
            is_3d=is_3d,
            class_count=num_classes,
            limit=val_limit,
            output_size=output_size,
            channels=channels,
            use_otsu_masking=use_otsu_masking,
            use_surface_filtering=use_surface_filtering,
            min_slice_surface=min_slice_surface,
            pixel_surface=pixel_surface,
            min_slice_surface_threshold=min_slice_surface_threshold,
        )
        if val_csv
        else None
    )

    test_dataset = (
        PETDataset(
            data_dir,
            test_csv,
            transform=test_transform,
            slice_selection=slice_selection,
            num_slices=num_slices,
            mode="test",
            is_3d=is_3d,
            class_count=num_classes,
            limit=test_limit,
            output_size=output_size,
            channels=channels,
            use_otsu_masking=use_otsu_masking,
            use_surface_filtering=use_surface_filtering,
            min_slice_surface=min_slice_surface,
            pixel_surface=pixel_surface,
            min_slice_surface_threshold=min_slice_surface_threshold,
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
            batch_size=1,
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
            batch_size=1,
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


def otsu_threshold(image):
    """
    Implementa el umbralizado de Otsu para separar automáticamente
    el tejido cerebral del fondo en imágenes PET.

    Args:
        image (np.ndarray): Imagen de entrada

    Returns:
        tuple: (threshold_value, binary_mask)
    """
    # Aplanar la imagen y remover valores NaN/inf
    flat_image = image.flatten()
    flat_image = flat_image[np.isfinite(flat_image)]

    if len(flat_image) == 0:
        return 0, np.zeros_like(image, dtype=bool)

    # Calcular histograma
    hist, bin_edges = np.histogram(flat_image, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalizar histograma para obtener probabilidades
    hist = hist.astype(float)
    hist /= hist.sum()

    # Calcular probabilidades acumuladas y medias acumuladas
    cum_prob = np.cumsum(hist)
    cum_mean = np.cumsum(bin_centers * hist)

    # Media global
    global_mean = cum_mean[-1]

    # Evitar división por cero
    cum_prob = np.where(cum_prob == 0, 1e-10, cum_prob)

    # Calcular varianza entre clases para cada posible threshold
    variance_between = np.zeros_like(cum_prob)

    for i in range(len(cum_prob)):
        if cum_prob[i] > 0 and cum_prob[i] < 1:
            # Probabilidades de cada clase
            w0 = cum_prob[i]  # Peso clase 0 (fondo)
            w1 = 1 - w0  # Peso clase 1 (primer plano)

            # Medias de cada clase
            mu0 = cum_mean[i] / w0 if w0 > 0 else 0
            mu1 = (global_mean - cum_mean[i]) / w1 if w1 > 0 else 0

            # Varianza entre clases
            variance_between[i] = w0 * w1 * (mu0 - mu1) ** 2

    # Encontrar el threshold que maximiza la varianza entre clases
    optimal_idx = np.argmax(variance_between)
    optimal_threshold = bin_centers[optimal_idx]

    # Crear máscara binaria
    binary_mask = image > optimal_threshold

    return optimal_threshold, binary_mask


def apply_brain_mask(image, use_otsu=True, min_threshold_percentile=5):
    """
    Aplica máscara cerebral para eliminar el fondo de las imágenes PET.

    Args:
        image (np.ndarray): Imagen PET 3D
        use_otsu (bool): Si usar umbralizado de Otsu
        min_threshold_percentile (float): Percentil mínimo para threshold alternativo

    Returns:
        np.ndarray: Imagen con máscara aplicada
    """
    if use_otsu:
        threshold, mask = otsu_threshold(image)
    else:
        # Threshold basado en percentil como alternativa
        threshold = np.percentile(image[image > 0], min_threshold_percentile)
        mask = image > threshold

    # Aplicar la máscara
    masked_image = image.copy()
    masked_image[~mask] = 0

    return masked_image


def clipped_zoom(img, zoom_factor, **kwargs):
    """
    Aplica zoom con recorte controlado manteniendo las dimensiones originales.

    Args:
        img (np.ndarray): Imagen de entrada con forma (H, W) o (H, W, C)
        zoom_factor (float): Factor de zoom (>1 acerca, <1 aleja)
        **kwargs: Argumentos adicionales para ndi.zoom

    Returns:
        np.ndarray: Imagen con zoom aplicado y dimensiones originales
    """
    h, w = img.shape[:2]

    # Para imágenes multicanal, no aplicar zoom al canal RGB
    # Crear tupla de factores de zoom con 1's para dimensiones después de ancho y alto
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zoom out (alejar)
    if zoom_factor < 1:
        # Caja delimitadora de la imagen alejada dentro del array de salida
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Relleno con ceros
        out = np.zeros_like(img)
        out[top : top + zh, left : left + zw] = ndi.zoom(img, zoom_tuple, **kwargs)

    # Zoom in (acercar)
    elif zoom_factor > 1:
        # Caja delimitadora de la región acercada dentro del array de entrada
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndi.zoom(img[top : top + zh, left : left + zw], zoom_tuple, **kwargs)

        # `out` podría ser ligeramente más grande que `img` debido al redondeo,
        # así que recortar píxeles extra en los bordes
        trim_top = (out.shape[0] - h) // 2
        trim_left = (out.shape[1] - w) // 2
        out = out[trim_top : trim_top + h, trim_left : trim_left + w]

    # Si zoom_factor == 1, devolver el array de entrada
    else:
        out = img
    return out


def get_indices_to_be_deleted(
    sample, min_slice_surface=None, pixel_surface=1.5, min_slice_surface_threshold=0.0
):
    """
    Devuelve un array de índices que deben ser borrados de la imagen NIfTI
    por no cumplir una superficie mínima.

    Args:
        sample: Imagen NIfTI o array numpy 3D
        min_slice_surface (float): Superficie mínima requerida para conservar un corte
        pixel_surface (float): Superficie por píxel (mm²)
        min_slice_surface_threshold (float): Umbral mínimo de intensidad para considerar un píxel

    Returns:
        list: Lista de índices de cortes a eliminar
    """
    # Chequeo de eliminar slices por superficie
    if min_slice_surface is None:
        return []

    # Obtener datos de la imagen
    if hasattr(sample, "get_fdata"):
        brain_vol_data = sample.get_fdata()
    else:
        brain_vol_data = sample

    with_more_than_100 = 0
    delete_indices = []
    if min_slice_surface <= 0.0:
        raise Exception("min_slice_surface should be > 0.0")

    for i in range(0, brain_vol_data.shape[2]):
        surface = 0.0
        slice_matrix = brain_vol_data[:, :, i]
        slice_array = slice_matrix.reshape(-1)
        surface = ((slice_array > min_slice_surface_threshold).sum()) * pixel_surface
        if surface >= min_slice_surface:
            with_more_than_100 += 1
        else:
            delete_indices.append(i)
        if with_more_than_100 < 16:
            raise Exception("No enough brain surface (" + str(with_more_than_100) + " slices)")

    return delete_indices


def get_test_data_loader(config, data_config_key="data"):
    """Crea solo un data loader de test para evaluación.

    Args:
        config (dict): Diccionario de configuración completo
        data_config_key (str): Clave de la configuración de datos a usar (ej: "data", "data2", "data3")

    Returns:
        DataLoader: Test data loader o None si no se puede crear

    """
    # Obtener configuración de datos específica
    data_config = config.get(data_config_key, {})

    if not data_config:
        print(f"⚠️  No se encontró configuración de datos para clave: {data_config_key}")
        return None

    classes = data_config.get("classes", "CN_AD").split("_")
    num_classes = len(classes)

    # Validar clases
    valid_classes = ["CN", "AD", "MCI", "SMC", "EMCI", "LMCI"]
    for class_name in classes:
        if class_name not in valid_classes:
            raise ValueError(f"Clase {class_name} no válida. Clases válidas: {valid_classes}")

    if num_classes not in [2, 3]:
        raise ValueError(f"Número de clases debe ser 2 o 3, recibido: {num_classes}")

    print(f"📊 Creando test loader para: {classes} ({num_classes} clases)")
    print(f"📊 Dataset: {data_config.get('dataset_name', 'N/A')}")

    # Parámetros del dataset
    is_3d = data_config.get("dimension", "2d") == "3d"
    batch_size = data_config.get("batch_size", 1)  # Para evaluación, típicamente batch_size=1
    num_workers = data_config.get("num_workers", 1)  # Menos workers para evaluación

    output_size = data_config.get("output_size", (224, 224))
    channels = data_config.get("channels", 3)
    use_otsu_masking = data_config.get("use_otsu_masking", True)

    # Parámetros para filtrado por superficie
    use_surface_filtering = data_config.get("use_surface_filtering", False)
    min_slice_surface = data_config.get("min_slice_surface", 100.0 * 100.0)
    pixel_surface = data_config.get("pixel_surface", 1.5)
    min_slice_surface_threshold = data_config.get("min_slice_surface_threshold", 0.0)

    # Archivos y directorios
    data_dir = data_config.get("data_dir", "./data")
    test_csv = data_config.get("test_csv")

    if not test_csv:
        print(f"⚠️  No se especificó test_csv en configuración {data_config_key}")
        return None

    # Verificar que los archivos existen
    if not os.path.exists(test_csv):
        print(f"❌ No se encontró archivo CSV: {test_csv}")
        return None

    if not os.path.exists(data_dir):
        print(f"❌ No se encontró directorio de datos: {data_dir}")
        return None

    # Transformaciones (solo para test, sin augmentación)
    test_transform = get_transforms(config, is_train=False, is_3d=is_3d)
    slice_selection = data_config.get("slice_selection", "middle")
    num_slices = data_config.get("num_slices", 16)

    # Límite de datos para testing (opcional)
    test_limit = data_config.get("test_limit", None)

    try:
        # Crear dataset de test
        test_dataset = PETDataset(
            data_dir=data_dir,
            csv_path=test_csv,
            transform=test_transform,
            slice_selection=slice_selection,
            num_slices=num_slices,
            mode="test",
            is_3d=is_3d,
            class_count=num_classes,
            limit=test_limit,
            config=config,  # Pasar config completo
            output_size=output_size,
            channels=channels,
            use_otsu_masking=use_otsu_masking,
            use_surface_filtering=use_surface_filtering,
            min_slice_surface=min_slice_surface,
            pixel_surface=pixel_surface,
            min_slice_surface_threshold=min_slice_surface_threshold,
        )

        # Crear data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle para evaluación
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"✅ Test loader creado - {len(test_dataset)} muestras")
        return test_loader

    except Exception as e:
        print(f"❌ Error creando test loader: {e}")
        return None
