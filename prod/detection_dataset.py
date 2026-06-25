from __future__ import annotations

import json
import random
import subprocess
import sys
import zipfile
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF


CARDD_VIEW_URL = "https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view"
CARDD_FILE_ID = "1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OFFICIAL_SPLIT_FILES = {
    "train": "instances_train2017.json",
    "val": "instances_val2017.json",
    "test": "instances_test2017.json",
}


# Normaliza el nombre del split para trabajar siempre con train/val/test.
def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in OFFICIAL_SPLIT_FILES:
        expected = ", ".join(sorted(OFFICIAL_SPLIT_FILES))
        raise ValueError(f"Split invalido: {split}. Esperados: {expected}")
    return normalized

# Valida el formato de `image_size` y lo convierte a tupla (alto, ancho).
def normalize_image_size(image_size):
    if image_size is None:
        return None

    if isinstance(image_size, int):
        return (image_size, image_size)

    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size debe tener alto y ancho mayores a cero")
        return (height, width)

    raise ValueError("image_size debe ser None, un int o una tupla/lista (alto, ancho)")

# Instala e importa gdown solo cuando hace falta descargar el dataset.
def ensure_gdown():
    try:
        import gdown

        return gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

        return gdown

# Descarga el ZIP de CarDD si todavía no existe un archivo válido local.
def download_cardd_zip(file_id=CARDD_FILE_ID, zip_path=None):
    zip_path = Path(zip_path) if zip_path is not None else Path("data") / "CarDD_release.zip"

    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        return zip_path

    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        zip_path.unlink()

    gdown = ensure_gdown()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(zip_path), quiet=False)

    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile("La descarga no produjo un archivo ZIP válido.")

    return zip_path

# Extrae el ZIP del dataset en la carpeta elegida.
def extract_cardd_zip(zip_path, extract_dir=None):
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir) if extract_dir is not None else zip_path.parent

    if not zip_path.exists():
        raise FileNotFoundError(f"No existe el ZIP en {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"{zip_path} no es un ZIP válido")

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir

# Busca la raíz general del dataset probando varias estructuras posibles.
def find_dataset_root(data_dir) -> Path:
    base_dir = Path(data_dir)
    candidates = [
        base_dir / "raw" / "CarDD",
        base_dir / "CarDD",
        base_dir / "CarDD_release",
        base_dir / "CarDD_release" / "CarDD_COCO",
        base_dir / "CarDD" / "CarDD_COCO",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError("No se encontro el dataset CarDD. Rutas verificadas:\n" + checked)

# Busca específicamente la raíz de CarDD_COCO, usada por el pipeline actual.
def find_coco_root(data_dir) -> Path:
    base_dir = Path(data_dir)

    candidates = [
        base_dir,
        base_dir / "raw" / "CarDD",
        base_dir / "CarDD",
        base_dir / "CarDD_release",
        base_dir / "CarDD_release" / "CarDD_COCO",
        base_dir / "CarDD" / "CarDD_COCO",
    ]

    checked_paths = []
    for candidate in candidates:
        checked_paths.append(candidate)
        if (candidate / "annotations").exists():
            return candidate

        coco_candidate = candidate / "CarDD_COCO"
        checked_paths.append(coco_candidate)
        if (coco_candidate / "annotations").exists():
            return coco_candidate

    checked = "\n".join(str(path) for path in checked_paths)
    raise FileNotFoundError("No se encontro CarDD_COCO. Rutas verificadas:\n" + checked)
# Garantiza que CarDD_COCO exista; si no está, lo descarga y lo extrae.
def ensure_cardd_dataset(data_dir, file_id=CARDD_FILE_ID, zip_filename="CarDD_release.zip") -> Path:
    data_dir = Path(data_dir)

    try:
        return find_coco_root(data_dir)
    except FileNotFoundError:
        zip_path = download_cardd_zip(file_id=file_id, zip_path=data_dir / zip_filename)
        extract_cardd_zip(zip_path, extract_dir=data_dir)
        return find_coco_root(data_dir)


class ComposeDetection:
    # Encadena transforms que reciben y devuelven `(image, target)`.
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image, target):
        for transform in self.transforms_list:
            image, target = transform(image, target)
        return image, target


class ToTensorDetection:
    # Convierte la imagen PIL a tensor sin modificar el target.
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


class RandomHorizontalFlipDetection:
    # Aplica flip horizontal y reajusta las bounding boxes en consecuencia.
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        _, _, width = image.shape
        image = torch.flip(image, dims=[2])

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            xmin = boxes[:, 0].clone()
            xmax = boxes[:, 2].clone()
            boxes[:, 0] = width - xmax
            boxes[:, 2] = width - xmin
            target["boxes"] = boxes

        return image, target


class Rotate90Detection:
    # Rota 90° en sentido horario y corrige las cajas anotadas.
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        if isinstance(image, torch.Tensor):
            _, height, _ = image.shape
            image = torch.rot90(image, k=3, dims=[1, 2])
        else:
            height = image.height
            image = image.transpose(Image.ROTATE_270)

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            xmin = boxes[:, 0].clone()
            ymin = boxes[:, 1].clone()
            xmax = boxes[:, 2].clone()
            ymax = boxes[:, 3].clone()

            boxes[:, 0] = height - ymax
            boxes[:, 1] = xmin
            boxes[:, 2] = height - ymin
            boxes[:, 3] = xmax
            target["boxes"] = boxes

        return image, target


def _clone_detection_target(target: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in target.items()
    }


class RandomObjectCropDetection:
    # Recorta alrededor de objetos objetivo y reajusta/remueve cajas fuera del crop.
    def __init__(
        self,
        p=0.5,
        target_class_ids=None,
        crop_scale_range=(2.0, 4.0),
        min_crop_size=(320, 320),
        min_visible_fraction=0.3,
        center_jitter=0.15,
    ):
        self.p = float(p)
        self.target_class_ids = set(int(class_id) for class_id in target_class_ids or [])
        self.crop_scale_range = tuple(float(value) for value in crop_scale_range)
        self.min_crop_size = normalize_image_size(min_crop_size)
        self.min_visible_fraction = float(min_visible_fraction)
        self.center_jitter = float(center_jitter)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or boxes.numel() == 0:
            return image, target

        image_width, image_height = self._image_size(image)
        candidate_indices = self._candidate_indices(labels)
        if not candidate_indices:
            candidate_indices = list(range(len(boxes)))
        if not candidate_indices:
            return image, target

        selected_index = random.choice(candidate_indices)
        selected_box = boxes[selected_index].detach().cpu()
        crop_box = self._sample_crop_box(selected_box, image_width, image_height)
        if crop_box is None:
            return image, target

        cropped_image = self._crop_image(image, crop_box)
        cropped_target = self._crop_target(target, crop_box)
        if cropped_target["boxes"].numel() == 0:
            return image, target

        return cropped_image, cropped_target

    def _candidate_indices(self, labels):
        if not self.target_class_ids:
            return list(range(len(labels)))
        return [
            index
            for index, label in enumerate(labels.detach().cpu().tolist())
            if int(label) in self.target_class_ids
        ]

    def _image_size(self, image):
        if isinstance(image, torch.Tensor):
            _, image_height, image_width = image.shape
            return int(image_width), int(image_height)
        return int(image.width), int(image.height)

    def _sample_crop_box(self, selected_box, image_width, image_height):
        x1, y1, x2, y2 = [float(value) for value in selected_box.tolist()]
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        min_height, min_width = self.min_crop_size
        min_scale, max_scale = self.crop_scale_range
        crop_scale = random.uniform(min_scale, max_scale)

        crop_width = min(float(image_width), max(float(min_width), box_width * crop_scale))
        crop_height = min(float(image_height), max(float(min_height), box_height * crop_scale))
        if crop_width <= 1 or crop_height <= 1:
            return None

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        jitter_x = random.uniform(-self.center_jitter, self.center_jitter) * crop_width
        jitter_y = random.uniform(-self.center_jitter, self.center_jitter) * crop_height
        center_x += jitter_x
        center_y += jitter_y

        crop_x1 = min(max(center_x - crop_width / 2.0, 0.0), max(float(image_width) - crop_width, 0.0))
        crop_y1 = min(max(center_y - crop_height / 2.0, 0.0), max(float(image_height) - crop_height, 0.0))
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        return (
            int(round(crop_x1)),
            int(round(crop_y1)),
            int(round(crop_x2)),
            int(round(crop_y2)),
        )

    def _crop_image(self, image, crop_box):
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        if isinstance(image, torch.Tensor):
            return image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        return image.crop(crop_box)

    def _crop_target(self, target, crop_box):
        crop_x1, crop_y1, crop_x2, crop_y2 = [float(value) for value in crop_box]
        crop_width = max(crop_x2 - crop_x1, 1.0)
        crop_height = max(crop_y2 - crop_y1, 1.0)
        cropped_target = _clone_detection_target(target)
        boxes = cropped_target["boxes"].clone()
        original_widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        original_heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        original_area = (original_widths * original_heights).clamp(min=1e-6)

        boxes[:, [0, 2]] -= crop_x1
        boxes[:, [1, 3]] -= crop_y1
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=crop_width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=crop_height)

        new_widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        new_heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        new_area = new_widths * new_heights
        visible_fraction = new_area / original_area
        keep_mask = (
            (new_widths > 1.0)
            & (new_heights > 1.0)
            & (visible_fraction >= self.min_visible_fraction)
        )

        cropped_target["boxes"] = boxes[keep_mask]
        if "labels" in cropped_target:
            cropped_target["labels"] = cropped_target["labels"][keep_mask]
        if "iscrowd" in cropped_target:
            cropped_target["iscrowd"] = cropped_target["iscrowd"][keep_mask]
        if "area" in cropped_target:
            cropped_target["area"] = new_area[keep_mask].to(dtype=torch.float32)

        return cropped_target


def resolve_target_class_ids(dataset, target_classes) -> set[int]:
    class_ids = set()
    for target_class in target_classes or []:
        if isinstance(target_class, str):
            if target_class not in dataset.class_to_idx:
                raise ValueError(f"Clase objetivo desconocida: {target_class}")
            class_ids.add(int(dataset.class_to_idx[target_class]))
        else:
            class_ids.add(int(target_class))
    return class_ids


def sample_contains_target_classes(dataset, sample_index: int, target_classes) -> bool:
    target_class_ids = resolve_target_class_ids(dataset, target_classes)
    if not target_class_ids:
        return False

    image_id = int(dataset.image_ids[int(sample_index)])
    annotations = dataset.annotations_by_image.get(image_id, [])
    for annotation in annotations:
        category_name = dataset.category_id_to_name[int(annotation["category_id"])]
        class_id = int(dataset.class_to_idx[category_name])
        if class_id in target_class_ids:
            return True
    return False


def build_oversampling_weights(
    dataset,
    target_classes=("dent", "scratch"),
    target_factor=2.5,
    base_weight=1.0,
):
    weights = []
    for sample_index in range(len(dataset)):
        has_target_class = sample_contains_target_classes(dataset, sample_index, target_classes)
        weights.append(float(target_factor) if has_target_class else float(base_weight))
    return torch.as_tensor(weights, dtype=torch.double)


def build_oversampling_sampler(
    dataset,
    target_classes=("dent", "scratch"),
    target_factor=2.5,
    num_samples=None,
    replacement=True,
):
    weights = build_oversampling_weights(
        dataset=dataset,
        target_classes=target_classes,
        target_factor=target_factor,
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples or len(weights)),
        replacement=bool(replacement),
    )


class CarDamageDetectionDataset(Dataset):
    # Dataset reusable para detección en formato compatible con torchvision.
    def __init__(
        self,
        data_dir,
        split,
        transform=None,
        image_size=None,
        resize=False,
        include_empty=False,
        model_name=None,
        annotation_file=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = normalize_split(split)
        self.transform = transform
        self.image_size = normalize_image_size(image_size)
        self.model_name = (model_name or "generic").strip().lower()
        self.include_empty = include_empty
        self.annotation_file = annotation_file
        self.resize = bool(resize)

        if self.model_name in {"fasterrcnn", "faster_r_cnn", "faster-rcnn"}:
            self.resize = bool(resize)

        if self.resize and self.image_size is None:
            raise ValueError("Si resize=True, image_size no puede ser None")

        self.coco_root = self._find_coco_root()
        self.annotation_path = self._resolve_annotation_file()
        self.annotation_data = self._load_json(self.annotation_path)
        self.images = self.annotation_data.get("images", [])
        self.image_ids = [int(image["id"]) for image in self.images]
        self.images_by_id = {int(image["id"]): image for image in self.images}
        self.annotations_by_image = self._group_annotations_by_image(
            self.annotation_data.get("annotations", [])
        )
        self.class_to_idx = self._build_class_to_idx()
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.category_id_to_name = self._load_category_id_to_name()
        self.skipped_images = {}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Toma una muestra cruda, ajusta tamaño si corresponde y aplica transforms.
        image, target = self.get_raw_sample(idx)

        image, target = self._apply_resize(image, target)

        if self.transform:
            image, target = self.transform(image, target)
        else:
            image = TF.to_tensor(image)

        return image, target

    def get_raw_sample(self, idx):
        # Devuelve la imagen PIL original junto al target de detección sin transforms.
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_path = self._image_path_from_info(image_info)

        if not image_path.exists():
            raise FileNotFoundError(f"No existe la imagen: {image_path}")

        image = Image.open(image_path).convert("RGB")
        target = self._build_target(image_id)
        return image, target

    def _find_coco_root(self) -> Path:
        return find_coco_root(self.data_dir)

    def _resolve_annotation_file(self) -> Path:
        # Prioriza un archivo de anotaciones explícito; si no, usa el oficial del split.
        if self.annotation_file is not None:
            path = Path(self.annotation_file)
            if not path.exists():
                raise FileNotFoundError(f"No existe el archivo de anotaciones: {path}")
            return path

        annotation_path = self.coco_root / "annotations" / OFFICIAL_SPLIT_FILES[self.split]
        if not annotation_path.exists():
            raise FileNotFoundError(f"No existe el archivo de anotaciones: {annotation_path}")
        return annotation_path

    def _annotation_paths_for_categories(self):
        # Reúne todas las anotaciones disponibles para construir un mapa de clases estable.
        paths = [
            self.coco_root / "annotations" / filename
            for filename in OFFICIAL_SPLIT_FILES.values()
            if (self.coco_root / "annotations" / filename).exists()
        ]

        resolved_path = self._resolve_annotation_file()
        if resolved_path not in paths:
            paths.append(resolved_path)

        return paths

    def _load_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_class_to_idx(self):
        # Crea índices consecutivos para las clases dejando `background=0`.
        categories = {}
        category_lists = [
            self._load_json(annotation_path).get("categories", [])
            for annotation_path in self._annotation_paths_for_categories()
        ]
        categories = {
            int(category["id"]): category["name"]
            for category_list in category_lists
            for category in category_list
        }

        if not categories:
            raise ValueError("No se pudieron cargar categorias desde las anotaciones COCO.")

        class_to_idx = {"background": 0}
        for idx, category_id in enumerate(sorted(categories), start=1):
            class_to_idx[categories[category_id]] = idx
        return class_to_idx

    def _load_category_id_to_name(self):
        # Recupera el nombre original de cada categoría COCO.
        category_lists = [
            self._load_json(annotation_path).get("categories", [])
            for annotation_path in self._annotation_paths_for_categories()
        ]
        category_id_to_name = {
            int(category["id"]): category["name"]
            for category_list in category_lists
            for category in category_list
        }

        if not category_id_to_name:
            raise ValueError("No se pudieron cargar categorias desde las anotaciones COCO.")

        return category_id_to_name

    def _relative_to_base(self, image_path: Path) -> str:
        try:
            return str(image_path.relative_to(self.data_dir))
        except ValueError:
            return str(image_path)

    def _group_annotations_by_image(self, annotations):
        # Agrupa las anotaciones por `image_id` para acceso rápido dentro de __getitem__.
        ordered_annotations = sorted(
            annotations,
            key=lambda annotation: int(annotation["image_id"]),
        )
        return {
            int(image_id): list(group)
            for image_id, group in groupby(
                ordered_annotations,
                key=lambda annotation: int(annotation["image_id"]),
            )
        }

    def _convert_bbox_xywh_to_xyxy(self, bbox):
        # Convierte cajas COCO `(x, y, width, height)` al formato `(xmin, ymin, xmax, ymax)`.
        x, y, width, height = bbox
        xmin = float(x)
        ymin = float(y)
        xmax = float(x + width)
        ymax = float(y + height)
        return xmin, ymin, xmax, ymax, float(width), float(height)

    def _image_path_from_info(self, image_info):
        # Reconstruye la ruta física de la imagen a partir del split y el file_name.
        filename = image_info.get("file_name")
        return self.coco_root / f"{self.split}2017" / filename

    def _build_target(self, image_id):
        # Construye el diccionario target que esperan los detectores de torchvision.
        annotations = self.annotations_by_image.get(int(image_id), [])
        converted_annotations = [
            (annotation, *self._convert_bbox_xywh_to_xyxy(annotation["bbox"]))
            for annotation in annotations
        ]
        valid_annotations = [
            (annotation, xmin, ymin, xmax, ymax, width, height)
            for annotation, xmin, ymin, xmax, ymax, width, height in converted_annotations
            if xmax > xmin and ymax > ymin
        ]

        boxes = (
            torch.tensor(
                [[xmin, ymin, xmax, ymax] for _, xmin, ymin, xmax, ymax, _, _ in valid_annotations],
                dtype=torch.float32,
            )
            if valid_annotations
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels = torch.tensor(
            [
                self.class_to_idx[self.category_id_to_name[int(annotation["category_id"])]]
                for annotation, *_ in valid_annotations
            ],
            dtype=torch.int64,
        )
        area = torch.tensor(
            [
                float(annotation.get("area", width * height))
                for annotation, _, _, _, _, width, height in valid_annotations
            ],
            dtype=torch.float32,
        )
        iscrowd = torch.tensor(
            [int(annotation.get("iscrowd", 0)) for annotation, *_ in valid_annotations],
            dtype=torch.int64,
        )

        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

    def _apply_resize(self, image, target):
        # Redimensiona imagen y reescala cajas/áreas cuando el experimento lo requiere.
        if not self.resize or self.image_size is None:
            return image, target

        target_height, target_width = self.image_size
        original_width, original_height = image.size

        if original_height == target_height and original_width == target_width:
            return image, target

        scale_x = target_width / original_width
        scale_y = target_height / original_height

        image = image.resize((target_width, target_height), Image.BILINEAR)

        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes

        if target["area"].numel() > 0:
            target["area"] = target["area"] * (scale_x * scale_y)

        return image, target


# `DataLoader` usa este helper para agrupar listas de imágenes y targets variables.
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
