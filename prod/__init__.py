# Reexporta utilidades de dataset y transforms para notebooks y scripts.
from .detection_dataset import (
    CarDamageDetectionDataset,
    ComposeDetection,
    RandomHorizontalFlipDetection,
    RandomObjectCropDetection,
    Rotate90Detection,
    ToTensorDetection,
    build_oversampling_sampler,
    build_oversampling_weights,
    collate_fn,
    download_cardd_zip,
    ensure_cardd_dataset,
    ensure_gdown,
    extract_cardd_zip,
    find_coco_root,
    find_dataset_root,
    resolve_target_class_ids,
    sample_contains_target_classes,
)
from .detection_metrics import create_map_metric, evaluate_map, extract_main_map_metrics

# Reexporta fábricas de modelos y helpers de optimización/conteo de parámetros.
from .detection_models import (
    build_fasterrcnn_variants,
    build_optimizer,
    count_total_parameters,
    count_trainable_parameters,
    create_model_from_config,
    describe_parameter_counts,
)

# Reexporta la lógica principal de entrenamiento y restauración de checkpoints.
from .detection_training import (
    evaluate_detection_loss,
    load_checkpoint,
    run_detection_experiment,
    train_one_epoch,
)

# Expone una API pública compacta para importar desde `prod` directamente.
__all__ = [
    "CarDamageDetectionDataset",
    "ComposeDetection",
    "RandomHorizontalFlipDetection",
    "RandomObjectCropDetection",
    "Rotate90Detection",
    "ToTensorDetection",
    "build_oversampling_sampler",
    "build_oversampling_weights",
    "collate_fn",
    "download_cardd_zip",
    "ensure_cardd_dataset",
    "ensure_gdown",
    "extract_cardd_zip",
    "find_coco_root",
    "find_dataset_root",
    "resolve_target_class_ids",
    "sample_contains_target_classes",
    "create_map_metric",
    "evaluate_map",
    "extract_main_map_metrics",
    "build_fasterrcnn_variants",
    "build_optimizer",
    "count_total_parameters",
    "count_trainable_parameters",
    "create_model_from_config",
    "describe_parameter_counts",
    "evaluate_detection_loss",
    "load_checkpoint",
    "run_detection_experiment",
    "train_one_epoch",
]
