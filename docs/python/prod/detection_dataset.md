# `prod/detection_dataset.py`

## Proposito del archivo

Este modulo concentra la capa reusable de datos para deteccion.

Incluye:

- utilidades para descargar o localizar CarDD
- normalizacion de argumentos de dataset
- transforms compatibles con deteccion
- oversampling para clases objetivo
- el dataset `CarDamageDetectionDataset`
- `collate_fn` para `DataLoader`s de deteccion

## Constantes principales

- `CARDD_VIEW_URL`: URL publica del archivo en Google Drive.
- `CARDD_FILE_ID`: ID de Google Drive usado por `gdown`.
- `IMAGE_EXTENSIONS`: extensiones de imagen reconocidas.
- `OFFICIAL_SPLIT_FILES`: mapeo de split a nombre de JSON COCO oficial.

## Funciones de apoyo

### `normalize_split(split: str) -> str`

Valida y normaliza el nombre del split.

Se usa para aceptar valores como `Train` o ` train ` y convertirlos a `train`, `val` o `test`.

Lanza `ValueError` si el split no esta entre los oficiales.

### `normalize_image_size(image_size)`

Normaliza el parametro `image_size`.

Acepta:

- `None`
- un entero, que se transforma en `(size, size)`
- una tupla o lista `(alto, ancho)`

Se usa para validar el resize opcional del dataset.

### `ensure_gdown()`

Intenta importar `gdown` y, si no esta instalado, lo instala con `pip` usando el Python activo.

Se usa como helper de descarga automatica del dataset.

### `download_cardd_zip(file_id=CARDD_FILE_ID, zip_path=None)`

Descarga `CarDD_release.zip` desde Google Drive.

Comportamiento:

- si el ZIP ya existe y es valido, lo reutiliza
- si existe pero no es un ZIP valido, lo elimina
- si no existe, descarga el archivo

Devuelve la ruta final del ZIP.

### `extract_cardd_zip(zip_path, extract_dir=None)`

Extrae el ZIP del dataset en el directorio indicado.

Valida que la ruta exista y que el archivo sea realmente un ZIP.

Devuelve el directorio de extraccion.

### `find_dataset_root(data_dir) -> Path`

Busca la raiz del dataset CarDD en varias rutas conocidas.

Rutas candidatas:

- `data/raw/CarDD/`
- `data/CarDD/`
- `data/CarDD_release/`
- `data/CarDD_release/CarDD_COCO/`
- `data/CarDD/CarDD_COCO/`

Devuelve la primera ruta existente.

### `find_coco_root(data_dir) -> Path`

Busca especificamente la carpeta que contiene `annotations/` para la variante `CarDD_COCO`.

Es la funcion mas importante para localizar el dataset utilizable por el pipeline de deteccion.

### `ensure_cardd_dataset(data_dir, file_id=CARDD_FILE_ID, zip_filename="CarDD_release.zip") -> Path`

Primero intenta encontrar `CarDD_COCO` localmente. Si no lo encuentra:

1. descarga el ZIP
2. lo extrae en `data_dir`
3. vuelve a buscar `CarDD_COCO`

Devuelve la raiz COCO lista para usar.

## Transforms

### `class ComposeDetection`

Encadena una lista de transforms para deteccion.

#### `__init__(self, transforms_list)`

Guarda la lista de transforms.

#### `__call__(self, image, target)`

Aplica cada transform secuencialmente sobre el par `(image, target)`.

Se usa cuando una transform modifica la imagen y necesita mantener sincronizadas las bounding boxes.

### `class ToTensorDetection`

Transform simple para convertir la imagen a tensor sin modificar el `target`.

#### `__call__(self, image, target)`

Convierte la imagen con `torchvision.transforms.functional.to_tensor`.

### `class RandomHorizontalFlipDetection`

Aplica flip horizontal aleatorio y ajusta las coordenadas `x` de las boxes.

#### `__init__(self, p=0.5)`

Define la probabilidad de flip.

#### `__call__(self, image, target)`

Comportamiento:

- si no toca flip, devuelve imagen y target sin cambios
- si la imagen no es tensor, la convierte a tensor
- invierte la imagen horizontalmente
- recalcula `xmin` y `xmax` de cada box en funcion del ancho de la imagen

### `class RandomObjectCropDetection`

Recorta aleatoriamente alrededor de un objeto objetivo y ajusta las boxes al nuevo crop.

Uso principal:

- mejorar la resolucion efectiva de clases chicas o ambiguas
- se aplica solo en entrenamiento
- normalmente se configura con `target_class_ids` para `dent` y `scratch`

Comportamiento:

- si no aplica por probabilidad, devuelve la muestra original
- elige preferentemente una box de las clases objetivo
- recorta alrededor de esa box con contexto aleatorio
- reubica las coordenadas de todas las boxes al crop
- elimina boxes con visibilidad insuficiente
- si el crop dejara la muestra sin boxes validas, devuelve la muestra original

## Oversampling

### `resolve_target_class_ids(dataset, target_classes)`

Convierte nombres o IDs de clases objetivo al índice interno usado por el dataset.

### `sample_contains_target_classes(dataset, sample_index, target_classes)`

Indica si una imagen contiene al menos una anotación de las clases objetivo.

### `build_oversampling_weights(dataset, target_classes=("dent", "scratch"), target_factor=2.5, base_weight=1.0)`

Construye un tensor de pesos para muestreo ponderado.

Las imágenes que contienen `dent` o `scratch` reciben `target_factor`; el resto recibe `base_weight`.

### `build_oversampling_sampler(dataset, target_classes=("dent", "scratch"), target_factor=2.5, num_samples=None, replacement=True)`

Crea un `WeightedRandomSampler` listo para pasar al `DataLoader` de entrenamiento.

Cuando se usa este sampler, el `DataLoader` debe tener `shuffle=False`, porque el sampler controla el orden.

## Dataset principal

### `class CarDamageDetectionDataset(Dataset)`

Dataset compatible con modelos de deteccion de `torchvision`.

Lee un JSON COCO oficial, abre las imagenes del split correspondiente y construye un `target` con este formato:

- `boxes`
- `labels`
- `image_id`
- `area`
- `iscrowd`

### Constructor

#### `__init__(self, data_dir, split, transform=None, image_size=None, resize=False, include_empty=False, model_name=None, annotation_file=None)`

Parametros relevantes:

- `data_dir`: carpeta base donde se buscara el dataset
- `split`: `train`, `val` o `test`
- `transform`: transform opcional compatible con deteccion
- `image_size`: tamano de resize opcional
- `resize`: activa el resize previo a las transforms
- `include_empty`: hoy se guarda pero no altera el flujo actual
- `model_name`: nombre del modelo consumidor; hoy solo se normaliza
- `annotation_file`: permite usar un JSON COCO explicito en lugar del oficial del split

Durante la inicializacion el dataset:

1. localiza `CarDD_COCO`
2. resuelve el archivo de anotaciones
3. carga el JSON COCO
4. indexa imagenes y anotaciones por `image_id`
5. construye el mapeo `class_to_idx`
6. arma `category_id_to_name`

### Metodos publicos

#### `__len__(self)`

Devuelve la cantidad de imagenes del split cargado.

#### `__getitem__(self, idx)`

Obtiene una muestra lista para PyTorch.

Flujo:

1. obtiene imagen y target crudos con `get_raw_sample`
2. aplica resize opcional
3. aplica transform si existe
4. si no hay transform, convierte la imagen a tensor

Devuelve `(image, target)`.

#### `get_raw_sample(self, idx)`

Devuelve la imagen PIL original y el `target` construido antes de transforms o resize.

Es util para inspeccion o debugging visual.

### Metodos internos

#### `_find_coco_root(self) -> Path`

Delegacion directa a `find_coco_root` usando `self.data_dir`.

#### `_resolve_annotation_file(self) -> Path`

Decide que JSON COCO usar.

- si `annotation_file` fue pasado explicitamente, usa ese archivo
- si no, usa el JSON oficial del split

#### `_annotation_paths_for_categories(self)`

Construye la lista de JSONs COCO disponibles para leer categorias.

Esto permite que el dataset arme un vocabulario de clases consistente incluso si se esta leyendo un solo split.

#### `_load_json(self, path: Path)`

Lee un JSON desde disco y lo parsea con `json.loads`.

#### `_build_class_to_idx(self)`

Construye el mapeo `nombre_de_clase -> indice`.

Regla importante:

- `background` siempre queda en indice `0`
- el resto de las clases se ordena por `category_id`

#### `_load_category_id_to_name(self)`

Construye el mapeo `category_id -> nombre_de_clase` leyendo las categorias disponibles en anotaciones COCO.

#### `_relative_to_base(self, image_path: Path) -> str`

Intenta expresar una ruta como relativa a `self.data_dir`.

Hoy es un helper interno que no participa del flujo principal del dataset.

#### `_group_annotations_by_image(self, annotations)`

Agrupa la lista plana de anotaciones COCO por `image_id`.

Se usa para acceder rapido a las anotaciones de cada imagen al construir targets.

#### `_convert_bbox_xywh_to_xyxy(self, bbox)`

Convierte una box COCO de formato `XYWH` a `XYXY`.

Tambien devuelve `width` y `height` como flotantes para reutilizarlos en el calculo de area.

#### `_image_path_from_info(self, image_info)`

Construye la ruta de una imagen en funcion del split actual y del campo `file_name` del JSON COCO.

#### `_build_target(self, image_id)`

Construye el diccionario `target` para una imagen.

Responsabilidades:

- recuperar las anotaciones del `image_id`
- convertir boxes de `XYWH` a `XYXY`
- filtrar boxes degeneradas
- construir tensores `boxes`, `labels`, `area` e `iscrowd`

#### `_apply_resize(self, image, target)`

Aplica resize a la imagen y reescala `boxes` y `area` para mantener consistencia geometrica.

El resize ocurre antes de las transforms.

## Funcion para DataLoader

### `collate_fn(batch)`

Transforma un batch de deteccion en dos listas:

- lista de imagenes
- lista de targets

Es necesaria porque en deteccion cada imagen puede tener una cantidad distinta de bounding boxes.

## Como se usa en el proyecto

Este modulo es la base reusable para reemplazar logica que originalmente vivia solo en el notebook de preparacion de datos.
