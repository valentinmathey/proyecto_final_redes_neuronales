# `prod/detection_models.py`

## Proposito del archivo

Este modulo concentra la creacion y configuracion de modelos de deteccion basados en `torchvision`.

Hoy soporta:

- Faster R-CNN
- Faster R-CNN MobileNet V3 Large FPN
- Faster R-CNN MobileNet V3 Large 320 FPN
- RetinaNet
- FCOS

## Pesos preentrenados

Los modelos de este modulo se crean siempre con los pesos `DEFAULT` recomendados por `torchvision`.

Eso simplifica la interfaz del proyecto y evita exponer una configuracion que en la practica no se estaba variando entre experimentos.

## Reemplazo de cabezas de clasificacion

### `replace_fasterrcnn_predictor(model, num_classes: int)`

Reemplaza el predictor final de Faster R-CNN por uno nuevo con `num_classes`.

Se usa cuando se toma un modelo preentrenado y se adapta al numero de clases del dataset actual.

### `replace_retinanet_head(model, num_classes: int)`

Reemplaza la cabeza de clasificacion de RetinaNet preservando `in_channels` y `num_anchors` del modelo original.

### `replace_fcos_head(model, num_classes: int)`

Reemplaza la cabeza de clasificacion de FCOS preservando canales, anchors y cantidad de convoluciones.

## Registro interno de modelos

### `_MODEL_REGISTRY`

Mapa interno `nombre -> (factory, enum_de_pesos, funcion_de_reemplazo_de_head)`.

Se usa para unificar la creacion de modelos distintos bajo la misma interfaz.

## Creacion de modelos

### `_create_model(model_name: str, num_classes: int, trainable_backbone_layers: int = 3, min_size=None, max_size=None, pretrained: bool = True)`

Funcion interna que:

1. resuelve la factory correcta desde `_MODEL_REGISTRY`
2. usa los pesos `DEFAULT` del modelo cuando `pretrained=True`, o `weights=None` y `weights_backbone=None` cuando `pretrained=False`
3. crea el modelo con argumentos comunes
4. reemplaza la cabeza final para ajustarla al dataset

### `create_model_from_config(config: dict, pretrained: bool = True)`

Punto de entrada principal del modulo.

Espera un diccionario de configuracion con campos como:

- `model_name`
- `num_classes`
- `trainable_backbone_layers`
- `min_size`
- `max_size`

Valida que `model_name` sea soportado y devuelve el modelo listo para entrenar.

Para inferencia en la app, `pretrained=False` permite reconstruir la arquitectura sin descargar pesos externos antes de cargar `dev/modelo.pth`.

## Conteo de parametros

### `count_trainable_parameters(model) -> int`

Cuenta cuantos parametros tienen `requires_grad=True`.

### `count_total_parameters(model) -> int`

Cuenta el total de parametros del modelo.

### `describe_parameter_counts(model) -> dict`

Devuelve un resumen con:

- `trainable_parameters`
- `frozen_parameters`
- `total_parameters`

Es util para comparar configuraciones de fine-tuning.

## Configuraciones base

### `build_fasterrcnn_variants(num_classes: int) -> dict`

Devuelve tres configuraciones listas para experimentar con Faster R-CNN:

- `fasterrcnn_head_only`
- `fasterrcnn_partial_backbone`
- `fasterrcnn_full_backbone`

La diferencia principal entre ellas es `trainable_backbone_layers`.

Ademas del `fasterrcnn` basado en ResNet-50, el registro interno tambien soporta las variantes `fasterrcnn_mobilenet_v3_large_fpn` y `fasterrcnn_mobilenet_v3_large_320_fpn`, pensadas para experimentos mas livianos y rapidos.

## Parametros entrenables y optimizador

### `get_trainable_parameters(model) -> list`

Devuelve solo los parametros entrenables del modelo.

Es un helper usado antes de crear el optimizador.

### `build_optimizer(model, optimizer_name="sgd", lr=0.005, weight_decay=0.0005, momentum=0.9)`

Crea el optimizador para los parametros entrenables.

Opciones soportadas:

- `sgd`
- `adam`
- `adamw`

Lanza `ValueError` si recibe un nombre no soportado.

## Como se usa en el proyecto

Este modulo desacopla la definicion de modelos del notebook de entrenamiento y permite repetir experimentos con una interfaz mas estable.
