# Docs

Esta carpeta centraliza la documentacion operativa y tecnica del repositorio.

## Contenido

- `setup_windows_gpu.md`: guia paso a paso para levantar el proyecto en Windows, crear `venv` e instalar PyTorch con GPU.
- `python/`: documentacion por archivo Python del repo.

## Documentacion por archivo Python

### Raiz

- `python/app.md`: estado y uso previsto de `app.py`.
- `python/utils.md`: estado y uso previsto de `utils.py`.

### `prod/`

- `python/prod/__init__.md`: API reexportada del paquete `prod`.
- `python/prod/detection_dataset.md`: utilidades de dataset, descarga, transforms y `CarDamageDetectionDataset`.
- `python/prod/detection_models.md`: factories y helpers de modelos de deteccion.
- `python/prod/detection_training.md`: loops de entrenamiento, validacion y checkpoints.
- `python/prod/detection_metrics.md`: metrica mAP y helpers de evaluacion.

## Orden sugerido de lectura

1. `../README.md`
2. `setup_windows_gpu.md`
3. `python/prod/detection_dataset.md`
4. `python/prod/detection_models.md`
5. `python/prod/detection_training.md`
6. `python/prod/detection_metrics.md`

## Alcance

Esta documentacion describe:

- para que sirve cada archivo
- que funciones y clases expone
- para que se usa cada funcion
- como encaja cada modulo en el pipeline actual

La fuente principal de verdad funcional sigue siendo `dev/01_dataset_preparation.ipynb` y `dev/02_model_training.ipynb`, pero `prod/` ya concentra la capa reusable del proyecto.
