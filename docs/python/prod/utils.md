# `prod/utils.py`

## Proposito del archivo

`prod/utils.py` concentra la logica auxiliar de la app Streamlit final.

## Responsabilidades

- Resolver el checkpoint final desde `dev/modelo.pth` local o desde Google Drive.
- Cachear la carga del modelo con `@st.cache_resource`.
- Leer `dev/best_test_result.json` para ubicar el checkpoint ganador y extraer desde ahi la metadata `config`.
- Reconstruir la arquitectura indicada por esa metadata, pero cargar siempre los pesos desde `dev/modelo.pth`.
- Aplicar el mismo preprocesamiento usado en test: si `config.resize=True`, redimensiona a `config.image_size` antes de inferir y reescala las cajas al tamano original para dibujar.
- Ejecutar inferencia estandar, inferencia por tiles y verificacion `High-detail`.
- Aplicar NMS sobre detecciones tiled.
- Dibujar bounding boxes, labels y severidad sobre la imagen.
- Calcular severidad, rango de costo orientativo, cobertura simulada y tablas de presentacion.
- Cargar `dev/best_test_result.json` para mostrar metricas reales.

## Variables de entorno para despliegue

La app busca primero un checkpoint local en `dev/modelo.pth`. Si no existe, descarga desde Google Drive usando:

- `MODEL_GDRIVE_ID`
- `MODEL_GDRIVE_URL`

En Streamlit Cloud estas claves deben configurarse como secrets o variables del entorno.

## Nota tecnica

`dev/modelo.pth` es la fuente de pesos de la UI. `dev/best_test_result.json` no se usa para elegir otro archivo de pesos: su `checkpoint_path` solo se abre para leer el `config` del modelo ganador.

Los checkpoints se cargan con `torch.load(..., weights_only=False)` porque fueron guardados como payload completo de entrenamiento, no como un `state_dict` aislado.

## Uso desde la app Streamlit

`prod/app.py` decide si corre modo `Estandar` o `Tiled` y si activa `High-detail`. Este modulo ejecuta esas decisiones con las siguientes funciones:

- `run_inference`: inferencia directa sobre la imagen completa con filtrado por `score_threshold`.
- `run_inference_tiled`: divide la imagen en una grilla, ejecuta inferencia por tile y aplica NMS global.
- `verify_detections`: segunda pasada local usada por `High-detail`; confirma o descarta detecciones candidatas, pero no agrega hallazgos nuevos.
- `run_inference_two_pass`: helper que combina inferencia estandar y verificacion local.

El modo automatico de la app usa `Estandar + High-detail` con `score_threshold=0.50`. La consola tecnica puede cambiar esos valores manualmente.
