# `prod/app.py`

## Proposito del archivo

`prod/app.py` contiene la app Streamlit de inferencia visual para detectar danos vehiculares sobre imagenes cargadas por archivo o camara.

La app usa el checkpoint final, ejecuta el detector, dibuja las cajas detectadas y presenta una lectura orientativa de severidad, costos y cobertura.

## Modo automatico por defecto

Cuando la `Consola tecnica` esta apagada, la app usa siempre la misma configuracion automatica:

- `Modo de escaneo`: `Estandar`.
- `High-detail`: activado.
- `Umbral de score`: `0.50`.
- `Cuadrantes`: `2`, solo como valor interno compatible; no se usa porque el modo automatico no ejecuta `Tiled`.

Esta configuracion no depende del tamano de la imagen. Antes el flujo podia elegir `Estandar` o `Tiled` segun la resolucion; ahora el comportamiento normal queda fijo en `Estandar + High-detail` con umbral `50%`.

## Que hace cada opcion de la consola tecnica

La `Consola tecnica` permite sobrescribir manualmente el modo automatico para pruebas y comparaciones.

### Umbral de score del detector

Controla el score minimo que debe tener una prediccion para mostrarse como deteccion.

- Un valor mas bajo deja pasar mas cajas, incluyendo detecciones menos confiables.
- Un valor mas alto filtra mas fuerte y muestra solo hallazgos con mayor score.
- El modo automatico usa `0.50`.

Este umbral se aplica tanto a la primera inferencia como a la verificacion local de `High-detail`.

### Modo `Estandar`

Ejecuta el detector una vez sobre la imagen completa.

Es el modo usado por defecto en automatico. Mantiene una lectura directa de la escena y evita dividir la imagen en tiles.

### Modo `Tiled`

Divide la imagen en una grilla de cuadrantes con overlap, ejecuta inferencia en cada tile y luego aplica NMS global para reducir cajas duplicadas.

Puede ayudar cuando el dano es pequeno o queda diluido en una imagen grande, pero tambien aumenta el costo de inferencia y puede cambiar el contexto visual que recibe el modelo.

### Cuadrantes

Solo aparece cuando el modo seleccionado es `Tiled`.

Define el tamano de la grilla:

- `2`: grilla `2x2`, cuatro tiles.
- `3`: grilla `3x3`, nueve tiles.
- `4`: grilla `4x4`, dieciseis tiles.

Mas cuadrantes implican regiones mas pequenas y mas pasadas de inferencia.

### High-detail

Activa una segunda verificacion local sobre cada hallazgo encontrado en la primera pasada.

No agrega detecciones nuevas. Toma cada caja candidata, recorta una region ampliada alrededor del hallazgo y vuelve a inferir sobre ese crop. Si el modelo vuelve a detectar la misma clase dentro del recorte, el hallazgo queda confirmado; si no, se descarta.

En modo automatico esta siempre activado.

## Flujo de inferencia

1. La app carga una imagen desde archivo o camara.
2. Si la consola tecnica esta apagada, aplica `Estandar + High-detail` con umbral `0.50`.
3. Si la consola tecnica esta encendida, usa los valores seleccionados manualmente.
4. Ejecuta `run_inference` para modo `Estandar` o `run_inference_tiled` para modo `Tiled`.
5. Si `High-detail` esta activado, ejecuta `verify_detections` sobre las detecciones candidatas.
6. Ordena las detecciones, dibuja las cajas y arma el resumen visual y economico.

## Funciones principales

- `auto_select_scan_config(pil_image)`: devuelve la configuracion automatica fija: `Estandar`, grilla `2`, umbral `0.50` y `High-detail=True`.
- `render_sidebar(evaluation_result)`: renderiza la consola tecnica y devuelve la configuracion manual si esta activa.
- `run_analysis_flow(...)`: ejecuta la inferencia con la configuracion recibida y persiste el resultado en `st.session_state`.
- `render_inspection_tab(sidebar_config)`: coordina carga de imagen, seleccion de configuracion, analisis y visualizacion.

## Relacion con `prod/utils.py`

`prod/app.py` se encarga de la interfaz y del flujo Streamlit. La logica reusable de inferencia, preprocesamiento, NMS, verificacion high-detail, dibujo, severidad y costos vive en `prod/utils.py`.
