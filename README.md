# Proyecto Final - Deteccion de danos en autos con CarDD

Repositorio base para preparar un pipeline reproducible en PyTorch usando **CarDD (Car Damage Detection Dataset)**. En el estado actual, el repo cubre preparacion del dataset para deteccion y una base de entrenamiento con fine-tuning de modelos de deteccion de `torchvision`, principalmente **Faster R-CNN**.

## Estado actual

- La fuente principal de verdad funcional sigue siendo `dev/01_dataset_preparation.ipynb`.
- El notebook de entrenamiento base es `dev/02_model_training.ipynb`.
- La logica reusable actual vive en `prod/`.
- El foco del repo hoy es deteccion con bounding boxes, no clasificacion simple.
- Todavia no hay una app final ni un pipeline CLI modular completo.

## Dataset usado

- **Dataset:** CarDD: Car Damage Detection Dataset
- **Fuente oficial:** https://cardd-ustc.github.io/
- **Formato usado en esta entrega:** `CarDD_COCO`
- **ZIP de descarga directa:** `https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view`
- **Nota:** en este repo se usa COCO como fuente principal para deteccion usando directamente los JSON COCO como fuente de verdad.

## Estructura del repo

```text
.
|-- data/
|   |-- README.md
|   `-- CarDD_release/
|-- dev/
|   |-- 01_dataset_preparation.ipynb
|   `-- 02_model_training.ipynb
|-- docs/
|   |-- README.md
|   |-- setup_windows_gpu.md
|   `-- python/
|       |-- app.md
|       |-- utils.md
|       `-- prod/
|           |-- __init__.md
|           |-- detection_dataset.md
|           |-- detection_metrics.md
|           |-- detection_models.md
|           `-- detection_training.md
|-- prod/
|   |-- __init__.py
|   |-- detection_dataset.py
|   |-- detection_metrics.py
|   |-- detection_models.py
|   `-- detection_training.py
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
`-- requirements.txt
```

## Setup recomendado en Windows

La recomendacion para este repo es usar **Python 3.11**. Aunque PyTorch soporta Python 3.10 o superior, en proyectos de ML suele ser mas seguro trabajar con `3.11` o `3.12` que con `3.14`, por compatibilidad de paquetes.

### 1. Instalar Python 3.11 si no esta disponible

Si `py -3.11` falla y solo tenes Python 3.14 instalado:

```powershell
winget install -e --id Python.Python.3.11
```

Despues cerra PowerShell y abrilo de nuevo.

Valida que el launcher de Windows vea la version correcta:

```powershell
py -0p
py -3.11 --version
```

### 2. Crear y activar el entorno virtual

Desde la raiz del repo:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Valida que quedaste dentro del entorno:

```powershell
python --version
```

En PowerShell deberias ver algo parecido a esto:

```text
(.venv) PS C:\~\proyecto_final_redes_neuronales>
```

### 3. Instalar dependencias del proyecto

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Instalar PyTorch con soporte GPU

Opcion principal recomendada:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Alternativa si necesitas una build mas conservadora:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

No ejecutes ambas. Usa una sola variante.

### 5. Validar que PyTorch detecta la GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

Si `torch.cuda.is_available()` devuelve `True`, la instalacion GPU quedo bien.

### 6. Abrir el notebook principal

```powershell
jupyter notebook dev/01_dataset_preparation.ipynb
```

Si `jupyter` no responde bien en el entorno:

```powershell
python -m pip install notebook ipykernel
python -m notebook dev/01_dataset_preparation.ipynb
```

## Notas practicas para Windows

- En Windows conviene usar `py` como launcher cuando hay varias versiones de Python instaladas.
- `python3 --version` puede invocar el alias de Microsoft Store. Para este repo, podes ignorarlo.
- Si dentro del `venv` `python --version` sigue mostrando Python 3.14, recrea el entorno con `py -3.11 -m venv .venv`.

## Reproducibilidad local

1. Clonar el repositorio.
2. Crear el entorno virtual con `Python 3.11`.
3. Instalar dependencias y PyTorch GPU.
4. Abrir `dev/01_dataset_preparation.ipynb`.
5. Ejecutar el notebook. Si el dataset no esta presente localmente, intentara descargarlo y extraerlo automaticamente.

## Google Colab

1. Subir `dev/01_dataset_preparation.ipynb` a Colab o abrirlo desde GitHub.
2. Ejecutar la celda de instalacion opcional de dependencias.
3. Ejecutar el resto del notebook.
4. El dataset se descargara automaticamente en `/content/data/` si no existe.

## Modulos principales

- `prod/detection_dataset.py`: descarga opcional, deteccion de rutas, parsing de datos COCO, `Dataset`, transforms y `collate_fn`.
- `prod/detection_models.py`: factories y configuraciones de modelos de deteccion.
- `prod/detection_training.py`: loops de entrenamiento, validacion y checkpoints.
- `prod/detection_metrics.py`: evaluacion con `mAP@50:95`, `mAP@50` y metricas relacionadas.

## Documentacion tecnica

La carpeta `docs/` centraliza la documentacion adicional del proyecto.

- `docs/README.md`: mapa general de la documentacion.
- `docs/setup_windows_gpu.md`: guia detallada de instalacion en Windows con `venv` y PyTorch GPU.
- `docs/python/`: documentacion por archivo Python.

## Notas importantes

- Las imagenes del dataset **no** se suben a GitHub.
- Se versionan notebooks, modulos Python y documentacion.
- El notebook arma un manifiesto en memoria a partir de los splits oficiales de CarDD COCO, pero hoy no persiste `data/train.csv`, `data/val.csv` y `data/test.csv` a disco.
- La base de entrenamiento actual compara variantes de `Faster R-CNN` y usa `mAP@50:95` y `mAP@50` como metricas principales de deteccion.
