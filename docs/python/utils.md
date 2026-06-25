# `utils.py`

## Proposito del archivo

`utils.py` concentra helpers compartidos por los notebooks cuando la logica no pertenece a un modulo especifico de `prod/`.

## Estado actual

- Exporta tablas de comparacion de resultados a HTML.
- Exporta reportes HTML detallados para el mejor resultado en test.
- Persiste resultados finales de test por corrida sin pisar el baseline canonico.
- Guarda y lee corridas de experimentos en formato JSONL.
- Arma registros serializables a partir de los resultados de entrenamiento.
- Normaliza y resuelve rutas de checkpoints para que las corridas sean portables entre Windows y Linux.
- Participa en `dev/02_model_training.ipynb` y `dev/03_model_selection.ipynb`.

## Helpers disponibles

### `append_jsonl_record(path, record)`

Agrega un diccionario como una linea JSON a un archivo `.jsonl`.

Se usa para registrar cada experimento terminado sin pisar corridas anteriores.

### `load_jsonl_records(path)`

Lee un archivo `.jsonl` y devuelve una lista de diccionarios.

Si el archivo no existe, devuelve una lista vacia.

### `make_experiment_run_record(experiment, run_result, best_row)`

Construye el registro persistente de una corrida.

El `run_id` se arma con timestamp, nombre del experimento y `optimizer_name` para distinguir mejor corridas parecidas.

Incluye:

- `run_id`
- `created_at`
- `name`
- `optimizer_name`
- `trainable_backbone_layers`
- `num_epochs`
- `config`
- metricas principales de validacion
- `checkpoint_path`
- tiempos de entrenamiento
- `history`

No incluye `best_payload`, porque contiene tensores y pesos no serializables.

### `to_portable_path(path, base_dir=None)`

Convierte una ruta de archivo a un formato portable para persistencia.

Cuando `base_dir` esta definido y la ruta cae dentro de esa base, devuelve una ruta relativa en formato POSIX, por ejemplo:

- `dev/experiments/model_best.pth`

Esto evita guardar rutas absolutas dependientes de Windows o Linux dentro del manifest.

### `resolve_portable_path(path, base_dir=None, fallback_dir=None)`

Resuelve una ruta persistida de checkpoint a una ruta real del filesystem actual.

Soporta:

- rutas relativas guardadas contra el repo
- rutas POSIX normales
- rutas historicas de Windows con backslashes y drive letter

Si recibe una ruta vieja de Windows y `fallback_dir` apunta a `dev/experiments`, intenta recuperar el archivo local usando el nombre del checkpoint.

### `load_experiment_runs(path)`

Carga las corridas JSONL y las devuelve como `pandas.DataFrame`.

Si una corrida no trae `optimizer_name` como campo de primer nivel, lo reconstruye desde `config` para mantener compatibilidad con manifests anteriores.

Hace lo mismo con `trainable_backbone_layers` y `num_epochs` cuando esos campos todavia no existen como columnas propias.

### `export_results_comparison_html(results_df, output_path, title="Comparacion de resultados")`

Exporta un `DataFrame` a una tabla HTML estilizada, con contenedor responsive, encabezado sticky y formato visual mas legible para comparar corridas.

Cuando existe `training_duration_seconds`, la exportacion agrega una columna visible `duration_hms` en formato `HH:MM:SS` y prioriza esa vista compacta para la tabla comparativa.

### `export_model_comparison_html(comparison_runs, output_path, title="Comparacion modelo vs modelo", selected_run_id=None, selection_reason=None, comparison_split="val")`

Genera el HTML detallado `modelo vs modelo` usado por el notebook de entrenamiento.

El reporte muestra las corridas en orden cronologico e incluye, por prueba:

- configuracion general de la corrida
- dataset y transforms usados
- metricas globales de validacion
- mAP por clase
- curvas de loss y mAP por epoca
- curvas precision-recall por clase

Si la corrida coincide con `selected_run_id`, tambien renderiza la tabla y grafica de sensibilidad a `NMS`.

### `is_detection_test_report_complete(report)`

Valida si un JSON de `best_test_result` ya tiene todas las secciones requeridas por el nuevo reporte final:

- `summary`
- `class_metrics`
- `pr_curves`
- `dataset_diagnostics`
- `nms_sensitivity`

Se usa para decidir si una cache vieja debe recalcularse.

### `export_detection_test_report_html(report, output_path, title="Reporte final de deteccion en test")`

Genera un reporte HTML mas rico para el mejor checkpoint evaluado en test.

Incluye:

- tarjetas con metricas globales
- tabla de metricas por clase
- tabla de diagnostico del dataset
- tabla de sensibilidad a `NMS`
- una curva precision-recall SVG por clase

### `detection_test_result_paths(run_id, output_dir)`

Devuelve las rutas esperadas para los artefactos por corrida:

- `{run_id}_test_result.json`
- `{run_id}_test_result_report.html`

### `save_detection_test_result_artifacts(report, output_dir, ..., update_canonical=False)`

Guarda el JSON y HTML de test dentro de `dev/test_results/`.

Por defecto no actualiza `dev/best_test_result.json`. Solo lo hace si `update_canonical=True`.

### `archive_canonical_detection_test_result(canonical_json_path, output_dir, overwrite=False)`

Copia el baseline canonico actual a la carpeta de resultados por corrida usando su `run_id`.

Sirve para comparar corridas nuevas contra el baseline historico sin pisarlo.

### `build_detection_test_results_comparison_df(runs_manifest_path, test_results_dir)`

Lee `runs_manifest.jsonl` y todos los `*_test_result.json` guardados por corrida.

Devuelve una tabla con:

- `run_id`
- experimento
- checkpoint
- `test_map`
- `test_map_50`
- `dent_map`
- `scratch_map`
- `crack_map`
- rutas al JSON y HTML del reporte

## Relacion con el proyecto actual

La logica reusable de dataset, modelos, entrenamiento y metricas sigue en `prod/`.

`utils.py` queda reservado para helpers transversales de notebooks, especialmente persistencia y reportes livianos.
