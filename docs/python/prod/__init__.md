# `prod/__init__.py`

## Proposito del archivo

`prod/__init__.py` reexporta la API principal del paquete `prod` para facilitar imports mas cortos desde notebooks o scripts.

## Para que se usa

Permite hacer imports como:

```python
from prod import CarDamageDetectionDataset, create_model_from_config
```

en lugar de importar cada simbolo desde su modulo concreto.

## Simbolos reexportados

### Dataset y descarga

- `CarDamageDetectionDataset`
- `ComposeDetection`
- `RandomHorizontalFlipDetection`
- `ToTensorDetection`
- `collate_fn`
- `download_cardd_zip`
- `ensure_cardd_dataset`
- `ensure_gdown`
- `extract_cardd_zip`
- `find_coco_root`
- `find_dataset_root`

### Metricas

- `create_map_metric`
- `evaluate_map`
- `extract_main_map_metrics`

### Modelos

- `build_fasterrcnn_variants`
- `build_optimizer`
- `count_total_parameters`
- `count_trainable_parameters`
- `create_model_from_config`
- `describe_parameter_counts`

### Entrenamiento

- `evaluate_detection_loss`
- `load_checkpoint`
- `run_detection_experiment`
- `train_one_epoch`

## Variable `__all__`

La lista `__all__` explicita que simbolos forman parte de la API publica del paquete. Eso ayuda a dejar claro que componentes se consideran reutilizables desde afuera.
