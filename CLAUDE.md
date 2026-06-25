# CLAUDE.md

## Resumen del proyecto

Este repositorio es la base de un proyecto final de redes neuronales para deteccion de danos en autos usando el dataset CarDD.

El foco actual del repo no es una app final ni un producto terminado. El foco actual es dejar lista una base reproducible para deteccion de objetos en PyTorch:

- descarga y deteccion automatica del dataset
- lectura de anotaciones COCO
- conversion de anotaciones a registros por imagen
- definicion de un `Dataset` custom para deteccion
- definicion de transforms compatibles con bounding boxes
- construccion de `DataLoader`s para train, val y test
- base de entrenamiento para modelos de deteccion de `torchvision`
- evaluacion con `mAP@50:95` y `mAP@50`
- visualizacion y verificacion manual de batches y augmentations

Hoy, la fuente principal de verdad sigue siendo el notebook `dev/01_dataset_preparation.ipynb`, pero la implementacion reutilizable de datos, modelos, entrenamiento y metricas ya se extrajo a `prod/`.

## Estado actual

- `README.md` explica el objetivo general, el setup y como reproducir el entorno.
- `data/README.md` explica el dataset y la estructura esperada.
- `docs/` contiene la documentacion operativa y tecnica agregada.
- `app.py` esta vacio.
- `utils.py` esta vacio.
- `prod/detection_dataset.py` contiene la implementacion reutilizable del dataset de deteccion, transforms y `collate_fn`.
- `prod/detection_models.py`, `prod/detection_training.py` y `prod/detection_metrics.py` contienen la base reutilizable para entrenamiento y evaluacion.
- No hay tests automaticos versionados.
- El notebook construye `csv_manifest_df`, pero no persiste `data/train.csv`, `data/val.csv` ni `data/test.csv` a disco en el estado actual.

## Entorno recomendado

### Python

- Version recomendada: `Python 3.11`
- Tambien puede funcionar `Python 3.12`
- No se recomienda `Python 3.14` como primera opcion para este repo por compatibilidad practica de paquetes de ML

### Windows

En Windows conviene usar el launcher `py` para seleccionar la version de Python correcta.

Instalacion sugerida si falta Python 3.11:

```powershell
winget install -e --id Python.Python.3.11
```

Creacion del entorno virtual:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Instalacion base del repo:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Instalacion recomendada de PyTorch GPU:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Alternativa:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Validacion de GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

## Objetivo tecnico actual

Preparar un pipeline reproducible para deteccion de danos en vehiculos usando CarDD en formato COCO, de forma que luego sea sencillo conectar y comparar modelos como Faster R-CNN, RetinaNet o FCOS.

## Dataset usado

- Dataset: CarDD, Car Damage Detection Dataset
- Fuente oficial: `https://cardd-ustc.github.io/`
- Variante usada en este repo: `CarDD_COCO`
- Descarga directa usada por el notebook: Google Drive via `gdown`
- Clases de dano detectadas en las anotaciones actuales: `dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`

El pipeline arma tambien la clase `background` con indice `0` para compatibilidad con pipelines de deteccion.

## Estructura del repositorio

### Raiz

- `README.md`: overview del proyecto, setup local, `venv` y PyTorch GPU.
- `CLAUDE.md`: contexto general del repo, arquitectura actual y mapa funcional.
- `AGENTS.md`: guia operativa para agentes y contribuidores automatizados.
- `requirements.txt`: dependencias Python minimas del proyecto.
- `.gitignore`: excluye dataset pesado, imagenes, entornos y archivos temporales.
- `app.py`: reservado para una futura aplicacion o entrypoint; hoy esta vacio.
- `utils.py`: reservado para helpers reutilizables; hoy esta vacio.

### `dev/`

- `01_dataset_preparation.ipynb`: activo principal del repo para datos.
- `02_model_training.ipynb`: notebook base de entrenamiento para semana 3.

### `data/`

- `README.md`: explica el dataset, la descarga y la estructura esperada.
- `CarDD_release/`: copia local del dataset detectada en este workspace.
- `CarDD_release/CarDD_COCO/annotations/`: JSON COCO usados como fuente de verdad.
- `CarDD_release/CarDD_SOD/`: otra variante del dataset que hoy no se usa.

### `prod/`

- `detection_dataset.py`: dataset de deteccion, transforms, utilidades de descarga y `collate_fn`.
- `detection_models.py`: factory y variantes de modelos de deteccion.
- `detection_training.py`: entrenamiento, validacion y checkpoints.
- `detection_metrics.py`: evaluacion con mAP.
- `__init__.py`: API reexportada del paquete.

### `docs/`

- `README.md`: indice general de la documentacion.
- `setup_windows_gpu.md`: guia de entorno local en Windows.
- `python/`: documentacion por archivo Python del repo.

## Donde buscar cada cosa

### Si queres entender de que va el proyecto

- Leer `README.md`.
- Leer este archivo.

### Si queres entender el dataset y su estructura

- Leer `data/README.md`.
- Revisar `data/CarDD_release/CarDD_COCO/annotations/*.json`.

### Si queres entender la logica real del pipeline

- Leer `dev/01_dataset_preparation.ipynb`.
- Leer `prod/detection_dataset.py` para la version reutilizable.
- Leer `dev/02_model_training.ipynb`, `prod/detection_models.py`, `prod/detection_training.py` y `prod/detection_metrics.py`.

### Si queres documentacion tecnica por modulo

- Leer `docs/README.md`.
- Leer `docs/python/prod/detection_dataset.md`.
- Leer `docs/python/prod/detection_models.md`.
- Leer `docs/python/prod/detection_training.md`.
- Leer `docs/python/prod/detection_metrics.md`.

## Flujo del pipeline

### Entrada

- JSON COCO de train, val y test
- imagenes correspondientes en `train2017`, `val2017` y `test2017`

### Transformacion

- lectura de categorias, imagenes y annotations
- agrupacion por `image_id`
- conversion de boxes de `XYWH` a `XYXY`
- armado de `records` por imagen en notebook
- creacion de `target` compatible con `torchvision`

### Salida actual

- datasets de PyTorch para deteccion
- `DataLoader`s listos para entrenamiento y evaluacion
- metricas mAP
- checkpoints opcionales
- visualizaciones y debugging en notebook

## Funciones y clases principales

### Dataset y descarga

- `ensure_gdown()`: garantiza que `gdown` este instalado.
- `download_cardd_zip(...)`: descarga el ZIP del dataset desde Google Drive.
- `extract_cardd_zip(...)`: extrae el ZIP en `data/`.
- `find_dataset_root(...)`: busca el dataset en rutas esperadas.
- `find_coco_root(...)`: localiza especificamente `CarDD_COCO`.
- `ensure_cardd_dataset(...)`: combina deteccion local con descarga/extraccion automatica.
- `CarDamageDetectionDataset`: dataset principal compatible con modelos de deteccion de `torchvision`.

### Transforms

- `ComposeDetection`: encadena transforms que operan sobre `(image, target)`.
- `ToTensorDetection`: convierte la imagen a tensor.
- `RandomHorizontalFlipDetection`: aplica flip horizontal y corrige boxes.

### Modelos

- `create_model_from_config(...)`: crea un modelo desde un diccionario de configuracion.
- `build_fasterrcnn_variants(...)`: devuelve configuraciones base para tres variantes de Faster R-CNN.
- `build_optimizer(...)`: crea optimizadores `SGD`, `Adam` o `AdamW`.

### Entrenamiento

- `train_one_epoch(...)`: ejecuta una epoca de entrenamiento.
- `evaluate_detection_loss(...)`: calcula loss de validacion.
- `run_detection_experiment(...)`: orquesta entrenamiento completo, evaluacion y checkpoints.
- `load_checkpoint(...)`: restaura pesos guardados.

### Metricas

- `create_map_metric(...)`: crea el acumulador de `torchmetrics`.
- `evaluate_map(...)`: calcula mAP sobre un dataloader.
- `extract_main_map_metrics(...)`: extrae las metricas principales de un resultado ya calculado.

## Estructura de datos importante

### `record`

En el notebook, cada muestra se representa como un diccionario con:

- `image_path`
- `image_id`
- `boxes`
- `labels`
- `label_names`
- `area`
- `iscrowd`
- `split`
- `width`
- `height`

### `target`

El `target` que devuelve el dataset esta alineado con el formato comun de deteccion en PyTorch:

- `boxes`: `torch.float32`
- `labels`: `torch.int64`
- `image_id`: `torch.int64`
- `area`: `torch.float32`
- `iscrowd`: `torch.int64`

## Dependencias

`requirements.txt` hoy declara:

- `torch`
- `torchvision`
- `torchmetrics`
- `pandas`
- `numpy`
- `matplotlib`
- `pillow`
- `tqdm`
- `jupyter`
- `kaggle`
- `gdown`
- `faster-coco-eval`

`kaggle` no forma parte del flujo principal implementado hoy. `gdown` si se usa activamente para descarga automatica. `torchmetrics` y `faster-coco-eval` se usan para evaluacion de deteccion con mAP.

## Inconsistencias y huecos actuales

- `app.py` y `utils.py` existen pero no concentran logica real todavia.
- El dataset local presente en este workspace incluye anotaciones y archivos auxiliares, pero las imagenes no estan versionadas en Git.
- La logica reutilizable ya vive en `prod/`, mientras que la exploracion y verificacion visual siguen centralizadas en notebooks.

## Regla practica para cualquiera que entre al repo

Si necesitas entender algo rapido:

1. Lee `README.md` para el panorama general y el setup.
2. Lee `data/README.md` para el dataset.
3. Lee `docs/README.md` para ubicar la documentacion tecnica.
4. Lee `dev/01_dataset_preparation.ipynb` para la implementacion real de datos.
5. Asumi que `app.py` y `utils.py` todavia no son la fuente principal de verdad, y que la implementacion reutilizable actual vive en `prod/`.
