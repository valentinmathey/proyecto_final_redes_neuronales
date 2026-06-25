# Setup Windows GPU

Guia recomendada para levantar este proyecto en Windows, crear un entorno virtual e instalar PyTorch con soporte GPU.

## Version de Python recomendada

Para este repo se recomienda **Python 3.11**.

Aunque PyTorch soporta Python 3.10 o superior, para proyectos de ML suele ser mejor evitar `Python 3.14` como primera opcion por compatibilidad practica de dependencias.

## 1. Instalar Python 3.11

Si `py -3.11` no existe en tu maquina:

```powershell
winget install -e --id Python.Python.3.11
```

Despues cerra PowerShell y volvelo a abrir.

Valida que el launcher detecte la version instalada:

```powershell
py -0p
py -3.11 --version
```

## 2. Crear el entorno virtual

Desde la raiz del repo:

```powershell
py -3.11 -m venv .venv
```

Si PowerShell bloquea la activacion del entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activa el entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

Valida la version activa:

```powershell
python --version
```

Si dentro del entorno aparece Python 3.14, elimina `.venv` y recrealo con `py -3.11 -m venv .venv`.

## 3. Instalar dependencias del proyecto

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Usar `python -m pip` ayuda a asegurarse de instalar dentro del entorno activo.

## 4. Instalar PyTorch con GPU

### Opcion principal: CUDA 13.2

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

### Alternativa: CUDA 12.1

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

No ejecutes las dos variantes. Elegi una sola.

## 5. Validar GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sin GPU')"
```

Salida esperada aproximada:

```text
2.x.x+cu132
13.2
True
NVIDIA ...
```

## 6. Abrir el notebook principal

```powershell
jupyter notebook dev/01_dataset_preparation.ipynb
```

Si `jupyter` no esta disponible o falla el entrypoint:

```powershell
python -m pip install notebook ipykernel
python -m notebook dev/01_dataset_preparation.ipynb
```

## Notas utiles para Windows

- Preferi `py` para seleccionar versiones de Python.
- `python3` puede invocar el alias de Microsoft Store y no sirve como referencia confiable en este contexto.
- Deberias ver `(.venv)` al inicio del prompt cuando el entorno esta activo.
