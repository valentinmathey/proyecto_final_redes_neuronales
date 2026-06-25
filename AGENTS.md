# AGENTS.md

## Proposito de este archivo

Este archivo esta pensado para agentes, asistentes y contribuidores que necesiten trabajar rapido sobre este repositorio sin asumir una arquitectura que todavia no existe.

La idea principal es simple: hoy la logica real del proyecto vive en notebooks y en los modulos reutilizables de `prod/`, no en una app Python modular completa ni en `app.py`.

## Mision del repositorio

Construir un pipeline reproducible para deteccion de danos en autos sobre CarDD usando PyTorch, empezando por la preparacion del dataset y dejando el camino listo para entrenamiento, evaluacion e inferencia futuras.

## Verdad operativa actual

- La fuente principal de verdad funcional para datos es `dev/01_dataset_preparation.ipynb`.
- El notebook base de entrenamiento es `dev/02_model_training.ipynb`.
- `README.md`, `CLAUDE.md` y `data/README.md` describen bien el objetivo, pero no reemplazan a los notebooks.
- `docs/` contiene documentacion tecnica complementaria y documentacion por modulo.
- `app.py` y `utils.py` no contienen logica util hoy.
- `prod/detection_dataset.py` contiene la implementacion reusable del dataset de deteccion.
- `prod/detection_models.py`, `prod/detection_training.py` y `prod/detection_metrics.py` contienen la base de entrenamiento y evaluacion.
- El repo esta en etapa de preparacion de datos y base experimental, no de producto final.

## Entorno recomendado para trabajar en este repo

- Preferir `Python 3.11`.
- En Windows, usar `py` como launcher.
- Crear entornos virtuales con `py -3.11 -m venv .venv`.
- No asumir que `python3` sirve en Windows, porque puede disparar el alias de Microsoft Store.
- Para GPU, la opcion documentada principal es `cu132`; `cu121` queda como alternativa.

Comandos base recomendados en Windows:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

## Orden de lectura recomendado

1. `README.md`
2. `data/README.md`
3. `docs/README.md`
4. `dev/01_dataset_preparation.ipynb`
5. `dev/02_model_training.ipynb`
6. `requirements.txt`
7. `.gitignore`

Si una duda no queda resuelta en los README, casi seguro la respuesta esta en los notebooks o en `docs/python/`.

## Mapa rapido: donde buscar cada cosa

### Proyecto y objetivo general

- `README.md`
- `CLAUDE.md`

### Setup local y PyTorch GPU

- `README.md`
- `docs/setup_windows_gpu.md`

### Dataset, estructura de carpetas y origen de los datos

- `data/README.md`
- `data/CarDD_release/CarDD_COCO/annotations/`
- `dev/01_dataset_preparation.ipynb`, seccion de descarga y deteccion del dataset

### Documentacion tecnica por archivo Python

- `docs/python/app.md`
- `docs/python/utils.md`
- `docs/python/prod/detection_dataset.md`
- `docs/python/prod/detection_models.md`
- `docs/python/prod/detection_training.md`
- `docs/python/prod/detection_metrics.md`
- `docs/python/prod/__init__.md`

### Logica de descarga automatica

- `ensure_gdown`
- `download_cardd_zip`
- `extract_cardd_zip`
- `find_dataset_root`
- `find_coco_root`
- `ensure_cardd_dataset`

### Logica de parsing de anotaciones COCO

- `dev/01_dataset_preparation.ipynb`, seccion de preparacion de anotaciones COCO para deteccion
- `prod/detection_dataset.py`, construccion de `target`

### Split train / val / test

- `dev/01_dataset_preparation.ipynb`
- `instances_train2017.json`
- `instances_val2017.json`
- `instances_test2017.json`

### Dataset y transforms para PyTorch

- `prod/detection_dataset.py`
- `ComposeDetection`
- `ToTensorDetection`
- `RandomHorizontalFlipDetection`
- `CarDamageDetectionDataset`

### DataLoaders

- `prod/detection_dataset.py`
- `collate_fn`
- `train_loader`
- `val_loader`
- `test_loader`

### Modelos, entrenamiento y metricas

- `prod/detection_models.py`
- `prod/detection_training.py`
- `prod/detection_metrics.py`
- `dev/02_model_training.ipynb`

## Estructura real del repo

### Archivos raiz

- `README.md`: explica de que va el proyecto y como levantar el entorno.
- `CLAUDE.md`: contexto tecnico amplio del repo.
- `AGENTS.md`: esta guia operativa.
- `requirements.txt`: dependencias.
- `.gitignore`: reglas para no versionar dataset pesado ni imagenes.
- `app.py`: placeholder vacio.
- `utils.py`: placeholder vacio.
- `prod/detection_dataset.py`: pipeline reusable de datos para deteccion.
- `prod/detection_models.py`: factory y configuraciones de modelos.
- `prod/detection_training.py`: entrenamiento, validacion y checkpoints.
- `prod/detection_metrics.py`: mAP para deteccion.

### Carpetas

- `dev/`: notebooks y trabajo exploratorio.
- `data/`: dataset local y documentacion de datos.
- `prod/`: codigo reusable fuera del notebook.
- `docs/`: documentacion operativa y tecnica.

## Estado del dataset en este workspace

- Existe `data/CarDD_release/`.
- Dentro existe `CarDD_COCO/` y `CarDD_SOD/`.
- El pipeline actual solo usa `CarDD_COCO`.
- En Git no deben versionarse imagenes pesadas.
- En este workspace las anotaciones COCO si estan presentes.

## Punto importante sobre las carpetas de datos

No asumas que la estructura local siempre es identica. El notebook y `prod/detection_dataset.py` contemplan varias rutas candidatas para encontrar el dataset.

Rutas que el proyecto sabe detectar:

- `data/raw/CarDD/`
- `data/CarDD/`
- `data/CarDD_release/`
- `data/CarDD_release/CarDD_COCO/`
- `data/CarDD/CarDD_COCO/`

Si una tarea falla porque no encuentra datos, primero revisar `find_dataset_root()` y `find_coco_root()` antes de cambiar rutas a mano.

## Contrato conceptual del pipeline

### Entrada

- JSON COCO de train, val y test
- imagenes correspondientes en `train2017`, `val2017` y `test2017`

### Transformacion

- lectura de categorias, imagenes y annotations
- agrupacion por `image_id`
- conversion de boxes de `XYWH` a `XYXY`
- armado de `records` por imagen en notebook
- armado de `target` compatible con `torchvision` en `prod/detection_dataset.py`

### Salida actual

- listas `train_records`, `val_records`, `test_records` dentro del notebook
- `Dataset` de PyTorch reusable desde notebook o scripts
- `DataLoader`s listos para entrenamiento y evaluacion
- metricas mAP y checkpoints opcionales

### Salida no implementada todavia

- CSVs persistidos a disco
- pipeline de entrenamiento CLI versionado
- pipeline de evaluacion formal separado
- interfaz de inferencia

## Cuando te pidan algo, donde tocar

### Si te piden explicar el proyecto

- Basate primero en `README.md`, `CLAUDE.md` y `docs/README.md`.

### Si te piden arreglar o extender la preparacion del dataset

- Toca primero `dev/01_dataset_preparation.ipynb` si el cambio es exploratorio o de flujo.
- Si la logica ya esta madura, toca o extiende `prod/detection_dataset.py` y deja el notebook como consumidor.
- Si agregas comportamiento reusable, actualiza tambien `docs/python/prod/detection_dataset.md`.

### Si te piden una funcion reusable

- No la dejes enterrada en el notebook si ya esta estabilizada.
- Por defecto, sumala a `prod/`.
- Documentala en `docs/python/`.

### Si te piden entrenamiento de modelo

- La base actual soporta Faster R-CNN, RetinaNet y FCOS desde `torchvision`.
- El notebook principal para esta etapa es `dev/02_model_training.ipynb`.
- Reutiliza `prod/detection_models.py`, `prod/detection_training.py` y `prod/detection_metrics.py`.

### Si te piden una app o script de inferencia

- No asumas que `app.py` ya tiene estructura base. Hay que diseniarla desde cero o a partir de nuevos modulos.

## Hechos importantes que un agente no debe asumir mal

- Este repo no esta centrado en clasificacion simple; esta orientado a deteccion con bounding boxes.
- El dataset de verdad usado hoy es `CarDD_COCO`, no `CarDD_SOD`.
- El notebook crea `csv_manifest_df`, pero no guarda CSVs a disco en el estado actual.
- `app.py` y `utils.py` no son la fuente de verdad actual.
- Para datos de deteccion, la implementacion reusable ahora esta en `prod/detection_dataset.py`.
- Para semana 3, las metricas principales son `mAP@50:95` y `mAP@50`, no accuracy.
- La documentacion por modulo vive en `docs/python/`.

## Checklist de trabajo seguro

1. Confirmar si la tarea impacta dataset, notebook, documentacion o modularizacion.
2. Leer la seccion relevante del notebook antes de editar nada si el cambio afecta flujo de datos.
3. No romper el contrato de `record` ni el formato de `target` si el cambio toca deteccion.
4. Mantener consistencia entre `README.md`, `data/README.md`, `CLAUDE.md`, `AGENTS.md` y `docs/`.
5. Si agregas persistencia, scripts o nuevas APIs, dejar claramente documentado donde quedan y cual es la nueva fuente de verdad.

## Formato interno que conviene preservar

### `record`

Cada muestra del notebook se representa como un diccionario con:

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

El dataset devuelve un target compatible con modelos de deteccion de `torchvision`:

- `boxes`
- `labels`
- `image_id`
- `area`
- `iscrowd`

## Comandos base

Instalacion recomendada:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Trabajo interactivo:

```powershell
jupyter notebook dev/01_dataset_preparation.ipynb
```

## Si tenes que responder rapido a "de que va este repo"

Respuesta corta correcta:

"Es un proyecto base para deteccion de danos en autos con CarDD y PyTorch. Hoy la parte implementada es la preparacion del dataset en formato COCO, un Dataset custom, transforms, DataLoaders, una base de entrenamiento y documentacion tecnica por modulo; todavia no hay una app final." 

## Regla final

Si algo parece faltar en el codigo Python plano, no concluyas enseguida que el repo esta incompleto por error. Primero revisa si esa logica ya vive en los notebooks o ya esta documentada en `docs/python/`. En este proyecto, esa es la situacion normal.
