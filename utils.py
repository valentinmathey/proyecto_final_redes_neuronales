from __future__ import annotations

import json
from html import escape
from pathlib import Path, PureWindowsPath
from datetime import timedelta

import pandas as pd


# Permite serializar valores frecuentes del proyecto al guardar JSON.
def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# Agrega una línea JSON al manifest sin pisar corridas anteriores.
def append_jsonl_record(path, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=_json_default)
        f.write("\n")
    return path


# Lee un archivo JSONL y devuelve una lista de diccionarios.
def load_jsonl_records(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Convierte una ruta absoluta a un formato portable relativo al repo cuando es posible.
def to_portable_path(path, base_dir=None) -> str:
    path = Path(path)
    if base_dir is not None:
        base_path = Path(base_dir).resolve()
        try:
            return path.resolve().relative_to(base_path).as_posix()
        except ValueError:
            pass
    if not path.is_absolute():
        return path.as_posix()
    return str(path)


# Resuelve rutas históricas de Windows o rutas relativas guardadas en manifests.
def resolve_portable_path(path, base_dir=None, fallback_dir=None) -> Path:
    raw_path = str(path)
    candidate = Path(raw_path)
    portable_candidate = Path(PureWindowsPath(raw_path).as_posix())

    # Evalúa variantes POSIX y Windows para reutilizar manifests viejos.
    candidates = []

    for item in (candidate, portable_candidate):
        if item not in candidates:
            candidates.append(item)

    if base_dir is not None:
        base_path = Path(base_dir)
        for item in (candidate, portable_candidate):
            combined = base_path / item
            if combined not in candidates:
                candidates.append(combined)

    looks_like_windows_path = (
        "\\" in raw_path
        or (len(raw_path) >= 2 and raw_path[1] == ":")
    )
    if fallback_dir is not None and looks_like_windows_path:
        filename_candidate = Path(fallback_dir) / PureWindowsPath(raw_path).name
        if filename_candidate not in candidates:
            candidates.append(filename_candidate)

    # Devuelve la primera ruta candidata que realmente exista en disco.
    for item in candidates:
        if item.exists():
            return item.resolve()

    tried_paths = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Checkpoint path not found: {raw_path}. Tried: {tried_paths}")


# Construye el registro serializable de una corrida a partir del resultado del entrenamiento.
def make_experiment_run_record(experiment: dict, run_result: dict, best_row: dict) -> dict:
    created_at = run_result["training_end_time"]
    optimizer_name = str(experiment.get("optimizer_name", "unknown")).strip().lower()
    run_id = (
        f"{created_at.replace(':', '').replace('-', '').replace('T', '_')}_"
        f"{experiment['name']}_{optimizer_name}"
    )
    return {
        "run_id": run_id,
        "created_at": created_at,
        "name": experiment["name"],
        "optimizer_name": optimizer_name,
        "trainable_backbone_layers": experiment.get("trainable_backbone_layers"),
        "num_epochs": experiment.get("num_epochs"),
        "config": experiment,
        "best_epoch": run_result["best_epoch"],
        "best_map": best_row.get("map"),
        "best_map_50": best_row.get("map_50"),
        "best_map_75": best_row.get("map_75"),
        "best_val_loss": best_row.get("val_loss"),
        "checkpoint_path": run_result["best_checkpoint_path"],
        "training_start_time": run_result["training_start_time"],
        "training_end_time": run_result["training_end_time"],
        "training_duration_seconds": run_result["training_duration_seconds"],
        "history": run_result["history"],
    }


# Carga el manifest histórico y normaliza columnas útiles para análisis en pandas.
def load_experiment_runs(path) -> pd.DataFrame:
    records = load_jsonl_records(path)
    if not records:
        return pd.DataFrame()

    normalized_records = []
    for record in records:
        normalized_record = dict(record)
        config = normalized_record.get("config") or {}
        normalized_record["optimizer_name"] = normalized_record.get(
            "optimizer_name",
            config.get("optimizer_name", "unknown"),
        )
        normalized_record["trainable_backbone_layers"] = normalized_record.get(
            "trainable_backbone_layers",
            config.get("trainable_backbone_layers"),
        )
        normalized_record["num_epochs"] = normalized_record.get(
            "num_epochs",
            config.get("num_epochs"),
        )
        normalized_records.append(normalized_record)

    return pd.DataFrame(normalized_records)


# Formatea duración en segundos como HH:MM:SS para tablas comparativas.
def _format_duration_hms(total_seconds) -> str:
    if pd.isna(total_seconds):
        return ""
    rounded_seconds = int(round(float(total_seconds)))
    return str(timedelta(seconds=rounded_seconds))


# Redondea métricas numéricas para mostrarlas de forma compacta.
def _format_metric_2_decimals(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


# Exporta una tabla HTML estilizada con el resumen de corridas registradas.
def export_results_comparison_html(
    results_df: pd.DataFrame,
    output_path,
    title: str = "Comparacion de resultados",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Trabaja sobre una copia para no mutar el DataFrame original del notebook.
    display_df = results_df.copy()
    if "training_duration_seconds" in display_df.columns:
        display_df["duration_hms"] = display_df["training_duration_seconds"].apply(_format_duration_hms)
    for metric_column in ["best_map", "best_map_50", "best_map_75", "best_val_loss"]:
        if metric_column in display_df.columns:
            display_df[metric_column] = display_df[metric_column].apply(_format_metric_2_decimals)

    # Prioriza las columnas que mejor explican la comparación experimental.
    preferred_columns = [
        "name",
        "duration_hms",
        "optimizer_name",
        "trainable_backbone_layers",
        "num_epochs",
        "best_epoch",
        "best_map",
        "best_map_50",
        "best_map_75",
        "best_val_loss",
        "checkpoint_path",
    ]
    available_columns = [column for column in preferred_columns if column in display_df.columns]
    if available_columns:
        display_df = display_df[available_columns]

    # Renombra encabezados para la versión visual del reporte.
    display_df = display_df.rename(
        columns={
            "name": "Experimento",
            "duration_hms": "Duracion",
            "optimizer_name": "Optimizer",
            "trainable_backbone_layers": "Capas backbone",
            "num_epochs": "Epocas",
            "best_epoch": "Mejor epoca",
            "best_map": "mAP",
            "best_map_50": "mAP@50",
            "best_map_75": "mAP@75",
            "best_val_loss": "Val loss",
            "checkpoint_path": "Checkpoint",
        }
    )

    table_html = display_df.to_html(index=False, classes="results-table", border=0)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f3f6f8;
            --bg-accent: #e8eff2;
            --surface: rgba(255, 255, 255, 0.88);
            --surface-strong: #ffffff;
            --border: rgba(20, 48, 64, 0.12);
            --text: #13212b;
            --muted: #5a6c78;
            --accent: #0f8b8d;
            --shadow: 0 22px 50px rgba(24, 48, 63, 0.10);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 40px 24px 56px;
            font-family: Aptos, Manrope, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 34%),
                linear-gradient(135deg, var(--bg) 0%, var(--bg-accent) 100%);
        }}

        .page {{
            max-width: 1500px;
            margin: 0 auto;
        }}

        .hero {{
            margin-bottom: 20px;
            padding: 28px 30px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.82));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }}

        h1 {{
            margin: 0 0 10px;
            font-size: clamp(1.9rem, 3vw, 2.8rem);
            line-height: 1.05;
            letter-spacing: 0;
        }}

        .subtitle {{
            margin: 0;
            color: var(--muted);
            font-size: 0.98rem;
        }}

        .table-shell {{
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: var(--shadow);
            overflow: hidden;
            backdrop-filter: blur(18px);
        }}

        .table-wrap {{
            overflow: auto;
        }}

        table.results-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--surface-strong);
        }}

        .results-table thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 14px 16px;
            text-align: left;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--muted);
            background: rgba(243, 248, 250, 0.96);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .results-table tbody td {{
            padding: 14px 16px;
            font-size: 0.95rem;
            border-bottom: 1px solid rgba(20, 48, 64, 0.08);
            vertical-align: top;
        }}

        .results-table tbody tr:nth-child(even) {{
            background: rgba(244, 249, 250, 0.72);
        }}

        .results-table tbody tr:hover {{
            background: rgba(15, 139, 141, 0.08);
        }}

        .results-table tbody tr:last-child td {{
            border-bottom: 0;
        }}

        .results-table td:nth-child(1),
        .results-table td:nth-child(2),
        .results-table td:nth-child(3),
        .results-table td:nth-child(4),
        .results-table td:nth-child(5),
        .results-table td:nth-child(11) {{
            font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
            font-size: 0.88rem;
        }}

        .results-table td:nth-child(3) {{
            color: var(--accent);
            font-weight: 700;
        }}

        .results-table td:nth-child(2),
        .results-table td:nth-child(6),
        .results-table td:nth-child(7),
        .results-table td:nth-child(8) {{
            white-space: nowrap;
        }}

        .results-table td:nth-child(11) {{
            min-width: 360px;
            color: var(--muted);
            word-break: break-all;
        }}

        .footer-note {{
            padding: 14px 18px 18px;
            color: var(--muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(250, 252, 253, 0.86), rgba(244, 248, 249, 0.95));
        }}

        @media (max-width: 900px) {{
            body {{
                padding: 20px 14px 32px;
            }}

            .hero,
            .table-shell {{
                border-radius: 14px;
            }}

            .results-table thead th,
            .results-table tbody td {{
                padding: 12px 13px;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <p class="subtitle">{len(results_df)} corrida(s) registradas en la comparacion actual.</p>
        </section>
        <section class="table-shell">
            <div class="table-wrap">
                {table_html}
            </div>
            <div class="footer-note">
                Tabla generada automaticamente a partir del manifest de corridas.
            </div>
        </section>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def is_detection_test_report_complete(report: dict | None) -> bool:
    if not isinstance(report, dict):
        return False

    required_top_level_keys = {
        "run_id",
        "checkpoint_path",
        "summary",
        "class_metrics",
        "pr_curves",
        "dataset_diagnostics",
        "nms_sensitivity",
    }
    if not required_top_level_keys.issubset(report):
        return False

    summary = report.get("summary") or {}
    required_summary_keys = {"map", "map_50", "map_75", "mar_100"}
    if not required_summary_keys.issubset(summary):
        return False

    return isinstance(report.get("pr_curves"), list) and isinstance(report.get("class_metrics"), list)


def _format_metric_4_decimals(value):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _build_info_cards_html(summary: dict) -> str:
    cards = [
        ("mAP@50:95", _format_metric_4_decimals(summary.get("map"))),
        ("mAP@50", _format_metric_4_decimals(summary.get("map_50"))),
        ("mAP@75", _format_metric_4_decimals(summary.get("map_75"))),
        ("mAR@100", _format_metric_4_decimals(summary.get("mar_100"))),
    ]
    return "".join(
        f"""
        <article class="metric-card">
            <span class="metric-label">{escape(label)}</span>
            <strong class="metric-value">{escape(value)}</strong>
        </article>
        """
        for label, value in cards
    )


def _build_comparison_metric_cards_html(summary: dict) -> str:
    cards = [
        ("mAP@50:95", _format_metric_4_decimals(summary.get("map"))),
        ("mAP@50", _format_metric_4_decimals(summary.get("map_50"))),
        ("mAP@75", _format_metric_4_decimals(summary.get("map_75"))),
    ]
    return "".join(
        f"""
        <article class="metric-card">
            <span class="metric-label">{escape(label)}</span>
            <strong class="metric-value">{escape(value)}</strong>
        </article>
        """
        for label, value in cards
    )


def _build_html_table(data, columns=None, classes="results-table"):
    frame = pd.DataFrame(data)
    if columns is not None and not frame.empty:
        frame = frame.reindex(columns=columns)
    if frame.empty:
        return '<p class="empty-state">No hay datos disponibles para esta seccion.</p>'
    return frame.to_html(index=False, classes=classes, border=0)


def _localize_class_name(class_name) -> str:
    english_name = str(class_name or "").strip()
    spanish_names = {
        "dent": "Abolladura",
        "scratch": "Rayón",
        "crack": "Grieta",
        "glass shatter": "Vidrio roto",
        "lamp broken": "Faro roto",
        "tire flat": "Neumático pinchado",
    }
    spanish_name = spanish_names.get(english_name)
    if not spanish_name:
        return english_name
    return f"{spanish_name} ({english_name})"


def _localize_class_metric_rows(class_metrics) -> list[dict]:
    rows = []
    for row in class_metrics or []:
        localized_row = dict(row)
        localized_row["class_name"] = _localize_class_name(row.get("class_name"))
        rows.append(localized_row)
    return rows


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _make_multi_series_svg(
    records,
    x_key: str,
    series_defs,
    width: int = 580,
    height: int = 290,
    y_min=None,
    y_max=None,
    x_label: str = "Epoca",
    y_label: str = "Valor",
) -> str:
    records = list(records or [])
    series_defs = list(series_defs or [])
    points_by_series = []
    x_values = []
    y_values = []

    for key, label, css_class in series_defs:
        points = []
        for record in records:
            x_value = _safe_float(record.get(x_key))
            y_value = _safe_float(record.get(key))
            if x_value is None or y_value is None:
                continue
            points.append((x_value, y_value))
            x_values.append(x_value)
            y_values.append(y_value)
        if points:
            points_by_series.append((label, css_class, points))

    if not points_by_series:
        return '<div class="empty-state">Sin puntos suficientes para graficar.</div>'

    padding_left = 54
    padding_right = 24
    padding_top = 26
    padding_bottom = 44
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    computed_y_min = min(y_values)
    computed_y_max = max(y_values)
    y_min = computed_y_min if y_min is None else float(y_min)
    y_max = computed_y_max if y_max is None else float(y_max)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    else:
        y_padding = (y_max - y_min) * 0.08
        y_min -= y_padding
        y_max += y_padding

    def scale_x(value):
        return padding_left + ((float(value) - x_min) / (x_max - x_min)) * plot_width

    def scale_y(value):
        return padding_top + (1.0 - ((float(value) - y_min) / (y_max - y_min))) * plot_height

    grid_lines = []
    tick_labels = []
    for tick_index in range(5):
        ratio = tick_index / 4
        x_value = x_min + ((x_max - x_min) * ratio)
        y_value = y_min + ((y_max - y_min) * ratio)
        x = scale_x(x_value)
        y = scale_y(y_value)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{padding_top}" x2="{x:.2f}" y2="{padding_top + plot_height}" class="grid-line" />'
        )
        grid_lines.append(
            f'<line x1="{padding_left}" y1="{y:.2f}" x2="{padding_left + plot_width}" y2="{y:.2f}" class="grid-line" />'
        )
        tick_labels.append(
            f'<text x="{x:.2f}" y="{height - 17}" class="axis-label" text-anchor="middle">{x_value:.2g}</text>'
        )
        tick_labels.append(
            f'<text x="{padding_left - 10}" y="{y + 4:.2f}" class="axis-label" text-anchor="end">{y_value:.3g}</text>'
        )

    line_paths = []
    legend_items = []
    for index, (label, css_class, points) in enumerate(points_by_series):
        polyline_points = " ".join(
            f"{scale_x(x_value):.2f},{scale_y(y_value):.2f}"
            for x_value, y_value in points
        )
        class_name = css_class or f"series-line-{index}"
        line_paths.append(f'<polyline points="{polyline_points}" class="series-line {class_name}" />')
        legend_x = padding_left + (index * 170)
        legend_items.append(
            f"""
            <g class="chart-legend-item">
                <line x1="{legend_x:.2f}" y1="16" x2="{legend_x + 24:.2f}" y2="16" class="series-line {class_name}" />
                <text x="{legend_x + 30:.2f}" y="20" class="axis-label">{escape(str(label))}</text>
            </g>
            """
        )

    return f"""
    <svg class="history-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(y_label)}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" ry="16" class="chart-bg" />
        {''.join(grid_lines)}
        <line x1="{padding_left}" y1="{padding_top + plot_height}" x2="{padding_left + plot_width}" y2="{padding_top + plot_height}" class="axis-line" />
        <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + plot_height}" class="axis-line" />
        {''.join(line_paths)}
        {''.join(tick_labels)}
        {''.join(legend_items)}
        <text x="{padding_left + (plot_width / 2):.2f}" y="{height - 4}" class="axis-title" text-anchor="middle">{escape(x_label)}</text>
        <text x="18" y="{padding_top + (plot_height / 2):.2f}" class="axis-title" text-anchor="middle" transform="rotate(-90 18 {padding_top + (plot_height / 2):.2f})">{escape(y_label)}</text>
    </svg>
    """


def _build_dataset_transform_rows(config: dict, comparison_split: str = "val") -> list[dict]:
    config = config or {}
    use_object_crop = bool(config.get("use_object_crop", False))
    oversample_target_factor = _safe_float(config.get("oversample_target_factor"))
    has_oversampling = oversample_target_factor is not None and oversample_target_factor > 1.0
    target_classes = config.get("target_classes") or []
    if isinstance(target_classes, (list, tuple)):
        target_classes_value = ", ".join(str(value) for value in target_classes) or "No aplica"
    else:
        target_classes_value = str(target_classes)

    return [
        {"item": "resize", "valor": config.get("resize")},
        {"item": "image_size", "valor": config.get("image_size")},
        {
            "item": "RandomObjectCropDetection",
            "valor": (
                f"Si, p={config.get('object_crop_probability', 0.5)}"
                if use_object_crop
                else "No"
            ),
        },
        {
            "item": "Oversampling",
            "valor": (
                f"Si, factor={oversample_target_factor}"
                if has_oversampling
                else "No"
            ),
        },
        {"item": "Clases objetivo augmentation", "valor": target_classes_value},
    ]


def _humanize_detection_model_name(model_name) -> str:
    model_name = str(model_name or "").strip()
    readable_names = {
        "fcos": "FCOS ResNet50 FPN",
        "retinanet": "RetinaNet ResNet50 FPN",
        "fasterrcnn": "Faster R-CNN ResNet50 FPN",
        "fasterrcnn_mobilenet_v3_large_fpn": "Faster R-CNN MobileNet V3 Large FPN",
        "fasterrcnn_mobilenet_v3_large_320_fpn": "Faster R-CNN MobileNet V3 Large 320 FPN",
    }
    return readable_names.get(model_name, model_name or "Modelo")


def _format_optimizer_name(optimizer_name) -> str:
    optimizer_name = str(optimizer_name or "").strip().lower()
    readable_names = {
        "sgd": "SGD",
        "adam": "Adam",
        "adamw": "AdamW",
    }
    return readable_names.get(optimizer_name, optimizer_name.upper() if optimizer_name else "Optimizer")


def _format_trainable_layers_label(trainable_layers) -> str:
    if trainable_layers is None or pd.isna(trainable_layers):
        return "capas del backbone entrenables no especificadas"
    try:
        trainable_layers = int(trainable_layers)
    except (TypeError, ValueError):
        return f"{trainable_layers} capas del backbone entrenables"
    layer_word = "capa" if trainable_layers == 1 else "capas"
    return f"{trainable_layers} {layer_word} del backbone entrenables"


def _format_target_classes_label(target_classes) -> str:
    if not target_classes:
        return ""
    if isinstance(target_classes, (list, tuple)):
        return "/".join(str(value) for value in target_classes)
    return str(target_classes)


def _build_augmentation_title_parts(config: dict) -> list[str]:
    config = config or {}
    augmentation_parts = []

    if config.get("use_object_crop", False):
        augmentation_parts.append("Object crop")

    oversample_target_factor = config.get("oversample_target_factor")
    has_oversampling = oversample_target_factor is not None and float(oversample_target_factor) > 1.0
    if has_oversampling:
        target_classes_label = _format_target_classes_label(config.get("target_classes"))
        oversampling_label = "Oversampling"
        if target_classes_label:
            oversampling_label = f"{oversampling_label} {target_classes_label}"
        augmentation_parts.append(oversampling_label)

    return augmentation_parts


def _build_comparison_run_title(run: dict) -> str:
    run = run or {}
    config = run.get("config") or {}
    model_label = _humanize_detection_model_name(config.get("model_name"))
    optimizer_label = _format_optimizer_name(run.get("optimizer_name") or config.get("optimizer_name"))
    layers_label = _format_trainable_layers_label(
        config.get("trainable_backbone_layers", run.get("trainable_backbone_layers"))
    )
    augmentations_label = " + ".join(_build_augmentation_title_parts(config))
    title_parts = [model_label, optimizer_label, layers_label]
    if augmentations_label:
        title_parts.append(augmentations_label)
    return " · ".join(title_parts)


def _make_pr_curve_svg(curve: dict, width: int = 420, height: int = 260) -> str:
    padding_left = 44
    padding_right = 18
    padding_top = 20
    padding_bottom = 34
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom
    recall_values = curve.get("recall") or []
    precision_values = curve.get("precision") or []

    if not recall_values or not precision_values:
        return '<div class="empty-state">Sin puntos suficientes para la curva PR.</div>'

    def scale_x(value):
        return padding_left + float(value) * plot_width

    def scale_y(value):
        return padding_top + (1.0 - float(value)) * plot_height

    polyline_points = " ".join(
        f"{scale_x(recall_value):.2f},{scale_y(precision_value):.2f}"
        for recall_value, precision_value in zip(recall_values, precision_values)
    )

    tick_labels = []
    grid_lines = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = scale_x(tick)
        y = scale_y(tick)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{padding_top}" x2="{x:.2f}" y2="{padding_top + plot_height}" class="grid-line" />'
        )
        grid_lines.append(
            f'<line x1="{padding_left}" y1="{y:.2f}" x2="{padding_left + plot_width}" y2="{y:.2f}" class="grid-line" />'
        )
        tick_labels.append(
            f'<text x="{x:.2f}" y="{height - 10}" class="axis-label" text-anchor="middle">{tick:.2f}</text>'
        )
        tick_labels.append(
            f'<text x="{padding_left - 10}" y="{y + 4:.2f}" class="axis-label" text-anchor="end">{tick:.2f}</text>'
        )

    return f"""
    <svg class="pr-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Curva precision-recall de {escape(curve.get('class_name', 'clase'))}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" ry="16" class="chart-bg" />
        {''.join(grid_lines)}
        <line x1="{padding_left}" y1="{padding_top + plot_height}" x2="{padding_left + plot_width}" y2="{padding_top + plot_height}" class="axis-line" />
        <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + plot_height}" class="axis-line" />
        <polyline points="{polyline_points}" class="pr-line" />
        {''.join(tick_labels)}
        <text x="{padding_left + (plot_width / 2):.2f}" y="{height - 4}" class="axis-title" text-anchor="middle">Recall</text>
        <text x="18" y="{padding_top + (plot_height / 2):.2f}" class="axis-title" text-anchor="middle" transform="rotate(-90 18 {padding_top + (plot_height / 2):.2f})">Precision</text>
    </svg>
    """


def _build_pr_cards_html(pr_curves) -> str:
    return "".join(
        f"""
        <article class="pr-card">
            <div class="pr-card-header">
                <h3>{escape(curve.get('class_name', 'Clase'))}</h3>
                <p>AP@50={_format_metric_4_decimals(curve.get('ap_50'))}</p>
            </div>
            {_make_pr_curve_svg(curve)}
        </article>
        """
        for curve in (pr_curves or [])
    )


def _build_nms_section_html(nms_sensitivity: dict) -> str:
    nms_sensitivity = nms_sensitivity or {}
    nms_results = nms_sensitivity.get("results") or []
    nms_table_rows = [
        {
            "nms_threshold": row.get("nms_threshold"),
            "map": row.get("map"),
            "map_50": row.get("map_50"),
            "map_75": row.get("map_75"),
        }
        for row in nms_results
    ]
    nms_html = _build_html_table(
        nms_table_rows,
        columns=["nms_threshold", "map", "map_50", "map_75"],
    )
    nms_chart_html = _make_multi_series_svg(
        nms_results,
        x_key="nms_threshold",
        series_defs=[("map", "mAP@50:95", "line-primary")],
        x_label="NMS threshold",
        y_label="mAP@50:95",
    )
    return f"""
    <section class="nested-section nms-highlight">
        <div class="nested-header">
            <h3>Sensibilidad a NMS del modelo seleccionado</h3>
            <p>
                score_threshold={escape(str(nms_sensitivity.get('score_threshold')))} |
                detections_per_img={escape(str(nms_sensitivity.get('detections_per_img')))} |
                baseline_nms={escape(str(nms_sensitivity.get('baseline_nms_threshold')))}
            </p>
        </div>
        <div class="nms-grid">
            <div class="table-wrap">{nms_html}</div>
            <div>{nms_chart_html}</div>
        </div>
    </section>
    """


def _build_model_comparison_summary_rows(comparison_runs: list[dict]) -> list[dict]:
    summary_rows = []
    for run_index, run in enumerate(comparison_runs or [], start=1):
        report = run.get("validation_report") or run.get("comparison_report") or {}
        summary = report.get("summary") or {}
        summary_rows.append(
            {
                "Prueba": run_index,
                "Nombre experimento": run.get("name"),
                "best_mAP": _format_metric_4_decimals(
                    run.get("best_map", summary.get("map"))
                ),
                "mAP@50": _format_metric_4_decimals(
                    run.get("best_map_50", summary.get("map_50"))
                ),
            }
        )
    return summary_rows


def _build_model_comparison_summary_html(comparison_runs: list[dict]) -> str:
    summary_table_html = _build_html_table(
        _build_model_comparison_summary_rows(comparison_runs),
        columns=["Prueba", "Nombre experimento", "best_mAP", "mAP@50"],
    )
    return f"""
    <section class="summary-section" id="tabla-resumen">
        <div class="nested-header">
            <h2>Tabla resumen</h2>
            <p>Resumen final por prueba, usando las metricas ya guardadas para cada corrida.</p>
        </div>
        <div class="table-wrap">{summary_table_html}</div>
    </section>
    """


def export_model_comparison_html(
    comparison_runs: list[dict],
    output_path,
    title: str = "Comparacion modelo vs modelo",
    selected_run_id: str | None = None,
    selection_reason: str | None = None,
    comparison_split: str = "val",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_runs = list(comparison_runs or [])
    selected_run_id = str(selected_run_id) if selected_run_id is not None else None

    run_sections = []
    for run_index, run in enumerate(comparison_runs, start=1):
        config = run.get("config") or {}
        history = run.get("history") or []
        report = run.get("validation_report") or run.get("comparison_report") or {}
        summary = report.get("summary") or {}
        class_metrics = report.get("class_metrics") or []
        pr_curves = report.get("pr_curves") or []
        nms_sensitivity = report.get("nms_sensitivity") or {}
        run_id = str(run.get("run_id", ""))
        is_selected = bool(selected_run_id and run_id == selected_run_id)
        duration_hms = _format_duration_hms(run.get("training_duration_seconds"))
        run_title = _build_comparison_run_title(run)

        run_meta_rows = [
            {"item": "Experimento", "valor": run.get("name")},
            {"item": "Modelo", "valor": config.get("model_name")},
            {"item": "Optimizer", "valor": run.get("optimizer_name") or config.get("optimizer_name")},
            {"item": "Epocas configuradas", "valor": run.get("num_epochs") or config.get("num_epochs")},
            {"item": "Mejor epoca", "valor": run.get("best_epoch")},
            {"item": "Duracion", "valor": duration_hms},
        ]
        run_meta_html = _build_html_table(run_meta_rows, columns=["item", "valor"])
        transforms_html = _build_html_table(
            _build_dataset_transform_rows(config, comparison_split=comparison_split),
            columns=["item", "valor"],
        )
        class_metrics_html = _build_html_table(
            _localize_class_metric_rows(class_metrics),
            columns=["class_id", "class_name", "map_per_class"],
        )
        loss_chart_html = _make_multi_series_svg(
            history,
            x_key="epoch",
            series_defs=[
                ("train_loss", "train_loss", "line-primary"),
                ("val_loss", "val_loss", "line-secondary"),
            ],
            x_label="Epoca",
            y_label="Loss",
        )
        map_chart_html = _make_multi_series_svg(
            history,
            x_key="epoch",
            series_defs=[
                ("map", "mAP@50:95", "line-primary"),
                ("map_50", "mAP@50", "line-secondary"),
            ],
            y_min=0.0,
            y_max=1.0,
            x_label="Epoca",
            y_label="mAP",
        )
        pr_cards_html = _build_pr_cards_html(pr_curves)
        selected_badge = '<span class="selected-badge">Modelo seleccionado</span>' if is_selected else ""
        nms_section_html = (
            _build_nms_section_html(nms_sensitivity)
            if is_selected and nms_sensitivity.get("results")
            else ""
        )

        active_class = " is-active" if run_index == 1 else ""
        selected_class = " selected-run" if is_selected else ""
        dot_active_class = " is-active" if run_index == 1 else ""
        run_sections.append(
            f"""
            <section class="run-card{active_class}{selected_class}" data-slide-index="{run_index - 1}">
                <div class="run-header">
                    <div>
                        <p class="eyebrow">Prueba {run_index}</p>
                        <h2>{escape(run_title)}</h2>
                    </div>
                    {selected_badge}
                </div>
                <div class="run-content-layout">
                    <div class="run-tables-column">
                        <div class="metric-grid">
                            {_build_comparison_metric_cards_html(summary)}
                        </div>
                        <section class="nested-section class-charts-section">
                            <div class="nested-header">
                                <h3>Curvas precision-recall por clase</h3>
                                <p>Calculadas sobre validacion a IoU=0.50, area=all y max_dets=100.</p>
                            </div>
                            <div class="pr-grid">
                                {pr_cards_html or '<p class="empty-state">No se generaron curvas precision-recall.</p>'}
                            </div>
                        </section>
                        <div class="table-pair-grid">
                            <section class="nested-section">
                                <div class="nested-header">
                                    <h3>Configuracion de la corrida</h3>
                                    <p>Arquitectura, optimizador y duracion del entrenamiento.</p>
                                </div>
                                <div class="table-wrap">{run_meta_html}</div>
                            </section>
                            <section class="nested-section">
                                <div class="nested-header">
                                    <h3>Dataset y transforms</h3>
                                    <p>Las augmentations listadas corresponden a entrenamiento.</p>
                                </div>
                                <div class="table-wrap">{transforms_html}</div>
                            </section>
                        </div>
                        <section class="nested-section">
                            <div class="nested-header">
                                <h3>mAP por clase en validacion</h3>
                                <p>Metricas por clase del checkpoint de esta prueba.</p>
                            </div>
                            <div class="table-wrap">{class_metrics_html}</div>
                        </section>
                        {nms_section_html}
                    </div>
                    <div class="run-charts-column">
                        <section class="nested-section training-charts-section">
                            <div class="nested-header">
                                <h3>Curvas de entrenamiento</h3>
                                <p>Las mismas curvas usadas en el notebook para leer la dinamica por epoca.</p>
                            </div>
                            <div class="chart-stack">
                                <article>{loss_chart_html}</article>
                                <article>{map_chart_html}</article>
                            </div>
                        </section>
                    </div>
                </div>
            </section>
            """
        )

    carousel_dots_html = "".join(
        f"""
        <button
            class="carousel-dot{' is-active' if index == 0 else ''}"
            type="button"
            data-slide-target="{index}"
            aria-label="Ir a prueba {index + 1}"
        ></button>
        """
        for index in range(len(comparison_runs))
    )
    summary_table_html = _build_model_comparison_summary_html(comparison_runs)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f7f9;
            --surface: rgba(255, 255, 255, 0.92);
            --surface-strong: #ffffff;
            --border: rgba(17, 39, 54, 0.12);
            --text: #112736;
            --muted: #5b6b77;
            --accent: #1565c0;
            --accent-2: #0f8b8d;
            --warning: #f59e0b;
            --accent-soft: rgba(21, 101, 192, 0.1);
            --shadow: 0 18px 40px rgba(17, 39, 54, 0.09);
        }}

        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            padding: 18px 12px 36px;
            font-family: Aptos, Manrope, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 32%),
                linear-gradient(135deg, #eef3f6 0%, #dfe8ed 100%);
        }}

        .page {{
            width: 100%;
            max-width: none;
            margin: 0 auto;
        }}

        .hero,
        .run-card,
        .summary-section {{
            border: 1px solid var(--border);
            border-radius: 22px;
            background: var(--surface);
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }}

        .hero {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 14px 18px;
            margin-bottom: 14px;
        }}

        .hero h1 {{
            margin: 0 0 4px;
            font-size: clamp(1.15rem, 1.8vw, 1.65rem);
        }}

        .hero p {{
            margin: 0;
            color: var(--muted);
            max-width: 980px;
            font-size: 0.88rem;
            line-height: 1.35;
        }}

        .run-card {{
            padding: 24px;
            margin-bottom: 24px;
        }}

        .summary-section {{
            padding: 24px;
            margin-top: 24px;
        }}

        .summary-section h2 {{
            margin: 0 0 4px;
            font-size: clamp(1.3rem, 2vw, 1.9rem);
        }}

        .carousel-shell {{
            margin: 0;
            padding: 0;
            border: 0;
            border-radius: 0;
            background: transparent;
            box-shadow: none;
        }}

        .carousel-controls {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }}

        .carousel-button {{
            appearance: none;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 8px 12px;
            background: var(--surface-strong);
            color: var(--text);
            font: inherit;
            font-size: 0.9rem;
            font-weight: 800;
            cursor: pointer;
            transition: transform 0.16s ease, background 0.16s ease, border-color 0.16s ease;
        }}

        .carousel-button:hover {{
            transform: translateY(-1px);
            background: rgba(21, 101, 192, 0.08);
            border-color: rgba(21, 101, 192, 0.35);
        }}

        .carousel-counter {{
            color: var(--muted);
            font-weight: 800;
            text-align: center;
            min-width: 118px;
            white-space: nowrap;
        }}

        .carousel-dots {{
            display: none;
            justify-content: center;
            gap: 8px;
            margin-top: 14px;
            flex-wrap: wrap;
        }}

        .carousel-dot {{
            width: 10px;
            height: 10px;
            padding: 0;
            border: 0;
            border-radius: 999px;
            background: rgba(17, 39, 54, 0.25);
            cursor: pointer;
        }}

        .carousel-dot.is-active {{
            width: 28px;
            background: var(--accent);
        }}

        .run-card:not(.is-active) {{
            display: none;
        }}

        .selected-run {{
            border-color: rgba(245, 158, 11, 0.55);
            box-shadow: 0 20px 48px rgba(245, 158, 11, 0.15);
        }}

        .run-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 18px;
        }}

        .eyebrow {{
            margin: 0 0 6px;
            color: var(--accent);
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-size: 0.8rem;
        }}

        .run-header h2 {{
            margin: 0;
            font-size: clamp(1.35rem, 2vw, 2rem);
        }}

        .run-subtitle {{
            margin: 6px 0 0;
            color: var(--muted);
            font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
            word-break: break-all;
        }}

        .selected-badge {{
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(245, 158, 11, 0.16);
            color: #92400e;
            font-weight: 800;
            white-space: nowrap;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 14px;
            margin: 18px 0;
        }}

        .metric-card {{
            padding: 16px;
            border-radius: 16px;
            background: var(--surface-strong);
            border: 1px solid var(--border);
        }}

        .metric-label {{
            display: block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .metric-value {{ font-size: 1.45rem; }}

        .run-content-layout {{
            display: grid;
            grid-template-columns: minmax(620px, 1.35fr) minmax(420px, 0.85fr);
            gap: 18px;
            align-items: start;
        }}

        .run-tables-column,
        .run-charts-column {{
            display: flex;
            flex-direction: column;
            gap: 16px;
            min-width: 0;
        }}

        .run-tables-column .metric-grid {{
            margin: 0;
        }}

        .run-tables-column .nested-section,
        .run-charts-column .nested-section {{
            margin-top: 0;
        }}

        .table-pair-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(260px, 1fr));
            gap: 16px;
            align-items: start;
        }}

        .two-column {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 16px;
        }}

        .nested-section {{
            margin-top: 16px;
            padding: 18px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.72);
        }}

        .nested-header {{
            margin-bottom: 12px;
        }}

        .nested-header h3 {{
            margin: 0 0 4px;
            font-size: 1.08rem;
        }}

        .nested-header p {{
            margin: 0;
            color: var(--muted);
        }}

        .table-wrap {{ overflow: auto; }}

        table.results-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--surface-strong);
            border-radius: 14px;
            overflow: hidden;
        }}

        .results-table thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 12px 14px;
            text-align: left;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            background: rgba(240, 245, 249, 0.98);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .results-table tbody td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(17, 39, 54, 0.08);
            vertical-align: top;
        }}

        .results-table tbody tr:nth-child(even) {{
            background: rgba(244, 248, 250, 0.84);
        }}

        .chart-grid,
        .nms-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 18px;
        }}

        .chart-stack {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
        }}

        .pr-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }}

        .run-tables-column .pr-grid {{
            grid-template-columns: repeat(3, minmax(170px, 1fr));
            gap: 10px;
        }}

        .class-charts-section {{
            padding: 12px;
        }}

        .class-charts-section .nested-header {{
            margin-bottom: 8px;
        }}

        .class-charts-section .nested-header h3 {{
            font-size: 0.98rem;
        }}

        .class-charts-section .nested-header p {{
            font-size: 0.82rem;
        }}

        .class-charts-section .pr-card {{
            padding: 8px;
            border-radius: 12px;
        }}

        .class-charts-section .pr-card-header {{
            margin-bottom: 4px;
            gap: 6px;
        }}

        .class-charts-section .pr-card-header h3,
        .class-charts-section .pr-card-header p {{
            font-size: 0.78rem;
        }}

        .class-charts-section .pr-chart {{
            height: 142px;
        }}

        .pr-card {{
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface-strong);
            padding: 16px;
        }}

        .pr-card-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }}

        .pr-card-header h3 {{
            margin: 0;
            font-size: 1rem;
        }}

        .pr-card-header p {{
            margin: 0;
            color: var(--muted);
            white-space: nowrap;
        }}

        .pr-chart,
        .history-chart {{
            width: 100%;
            height: auto;
        }}

        .chart-bg {{
            fill: #f7fbff;
            stroke: rgba(21, 101, 192, 0.08);
        }}

        .grid-line {{
            stroke: rgba(17, 39, 54, 0.08);
            stroke-width: 1;
        }}

        .axis-line {{
            stroke: rgba(17, 39, 54, 0.25);
            stroke-width: 1.4;
        }}

        .pr-line,
        .series-line {{
            fill: none;
            stroke-width: 3;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}

        .pr-line,
        .line-primary {{ stroke: var(--accent); }}

        .line-secondary {{ stroke: var(--accent-2); }}

        .series-line-0 {{ stroke: var(--accent); }}

        .series-line-1 {{ stroke: var(--accent-2); }}

        .axis-label,
        .axis-title {{
            fill: var(--muted);
            font-size: 11px;
        }}

        .nms-highlight {{
            border-color: rgba(245, 158, 11, 0.5);
            background: rgba(255, 251, 235, 0.72);
        }}

        .empty-state,
        .summary-note {{
            margin: 0;
            padding: 14px 16px;
            border-radius: 14px;
            background: var(--accent-soft);
            color: var(--muted);
        }}

        .summary-note {{
            margin-top: 14px;
            color: var(--text);
        }}

        @media (max-width: 900px) {{
            body {{ padding: 18px 12px 28px; }}
            .hero,
            .run-card {{ border-radius: 16px; }}
            .run-card,
            .hero {{ padding: 18px; }}
            .run-content-layout {{
                grid-template-columns: 1fr;
            }}
            .table-pair-grid,
            .run-tables-column .pr-grid {{
                grid-template-columns: 1fr;
            }}
            .hero {{
                align-items: stretch;
                flex-direction: column;
            }}
            .carousel-controls {{
                display: grid;
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <section class="carousel-shell" aria-label="Navegacion de pruebas">
                <div class="carousel-controls">
                    <button class="carousel-button" id="prev-slide" type="button">← Anterior</button>
                    <span class="carousel-counter" id="slide-counter">Prueba 1 de {len(comparison_runs)}</span>
                    <button class="carousel-button" id="next-slide" type="button">Siguiente →</button>
                </div>
                <div class="carousel-dots" aria-label="Selector de pruebas">
                    {carousel_dots_html}
                </div>
            </section>
        </section>
        {''.join(run_sections)}
        {summary_table_html}
    </div>
    <script>
        (() => {{
            const slides = Array.from(document.querySelectorAll('.run-card[data-slide-index]'));
            const dots = Array.from(document.querySelectorAll('.carousel-dot[data-slide-target]'));
            const counter = document.getElementById('slide-counter');
            const prevButton = document.getElementById('prev-slide');
            const nextButton = document.getElementById('next-slide');
            let currentSlide = 0;

            function showSlide(index) {{
                if (!slides.length) {{
                    return;
                }}
                currentSlide = (index + slides.length) % slides.length;
                slides.forEach((slide, slideIndex) => {{
                    slide.classList.toggle('is-active', slideIndex === currentSlide);
                }});
                dots.forEach((dot, dotIndex) => {{
                    dot.classList.toggle('is-active', dotIndex === currentSlide);
                    dot.setAttribute('aria-current', dotIndex === currentSlide ? 'true' : 'false');
                }});
                if (counter) {{
                    counter.textContent = `Prueba ${{currentSlide + 1}} de ${{slides.length}}`;
                }}
                window.scrollTo({{ top: 0, behavior: 'smooth' }});
            }}

            prevButton?.addEventListener('click', () => showSlide(currentSlide - 1));
            nextButton?.addEventListener('click', () => showSlide(currentSlide + 1));
            dots.forEach((dot) => {{
                dot.addEventListener('click', () => showSlide(Number(dot.dataset.slideTarget || 0)));
            }});
            document.addEventListener('keydown', (event) => {{
                if (event.key === 'ArrowLeft') {{
                    showSlide(currentSlide - 1);
                }}
                if (event.key === 'ArrowRight') {{
                    showSlide(currentSlide + 1);
                }}
            }});
            showSlide(0);
        }})();
    </script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def export_detection_test_report_html(
    report: dict,
    output_path,
    title: str = "Reporte final de deteccion en test",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = report.get("summary") or {}
    class_metrics = report.get("class_metrics") or []
    pr_curves = report.get("pr_curves") or []
    dataset_diagnostics = report.get("dataset_diagnostics") or {}
    nms_sensitivity = report.get("nms_sensitivity") or {}

    class_metrics_html = _build_html_table(
        _localize_class_metric_rows(class_metrics),
        columns=["class_id", "class_name", "map_per_class", "mar_100_per_class"],
    )
    dataset_diagnostics_html = _build_html_table(
        dataset_diagnostics.get("per_class") or [],
        columns=[
            "class_id",
            "class_name",
            "annotation_count",
            "image_count",
            "median_bbox_area",
            "median_bbox_area_pct",
            "mean_instances_per_image",
            "max_instances_per_image",
        ],
    )
    nms_results = nms_sensitivity.get("results") or []
    nms_table_rows = [
        {
            "nms_threshold": row.get("nms_threshold"),
            "map": row.get("map"),
            "map_50": row.get("map_50"),
            "map_75": row.get("map_75"),
            "mar_100": row.get("mar_100"),
        }
        for row in nms_results
    ]
    nms_html = _build_html_table(
        nms_table_rows,
        columns=["nms_threshold", "map", "map_50", "map_75", "mar_100"],
    )

    pr_cards_html = _build_pr_cards_html(pr_curves)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f7f9;
            --surface: rgba(255, 255, 255, 0.9);
            --surface-strong: #ffffff;
            --border: rgba(17, 39, 54, 0.12);
            --text: #112736;
            --muted: #5b6b77;
            --accent: #1565c0;
            --accent-soft: rgba(21, 101, 192, 0.1);
            --shadow: 0 18px 40px rgba(17, 39, 54, 0.09);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 32px 20px 48px;
            font-family: Aptos, Manrope, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 32%),
                linear-gradient(135deg, #eef3f6 0%, #dfe8ed 100%);
        }}

        .page {{
            max-width: 1500px;
            margin: 0 auto;
        }}

        .hero,
        .section-shell {{
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--surface);
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }}

        .hero {{
            padding: 28px 30px;
            margin-bottom: 20px;
        }}

        .hero h1 {{
            margin: 0 0 8px;
            font-size: clamp(1.9rem, 3vw, 2.8rem);
        }}

        .hero p {{
            margin: 0;
            color: var(--muted);
            max-width: 920px;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 22px;
        }}

        .metric-card {{
            padding: 18px 18px 16px;
            border-radius: 16px;
            background: var(--surface-strong);
            border: 1px solid var(--border);
        }}

        .metric-label {{
            display: block;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 1.5rem;
        }}

        .section-shell {{
            margin-bottom: 20px;
            overflow: hidden;
        }}

        .section-header {{
            padding: 20px 24px 10px;
        }}

        .section-header h2 {{
            margin: 0 0 6px;
            font-size: 1.3rem;
        }}

        .section-header p {{
            margin: 0;
            color: var(--muted);
        }}

        .section-body {{
            padding: 0 24px 24px;
        }}

        .table-wrap {{
            overflow: auto;
        }}

        table.results-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--surface-strong);
        }}

        .results-table thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 14px 15px;
            text-align: left;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            background: rgba(240, 245, 249, 0.98);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .results-table tbody td {{
            padding: 14px 15px;
            border-bottom: 1px solid rgba(17, 39, 54, 0.08);
            vertical-align: top;
        }}

        .results-table tbody tr:nth-child(even) {{
            background: rgba(244, 248, 250, 0.84);
        }}

        .pr-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
        }}

        .pr-card {{
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface-strong);
            padding: 18px;
        }}

        .pr-card-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }}

        .pr-card-header h3 {{
            margin: 0;
            font-size: 1.05rem;
        }}

        .pr-card-header p {{
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
            white-space: nowrap;
        }}

        .pr-chart {{
            width: 100%;
            height: auto;
        }}

        .chart-bg {{
            fill: #f7fbff;
            stroke: rgba(21, 101, 192, 0.08);
        }}

        .grid-line {{
            stroke: rgba(17, 39, 54, 0.08);
            stroke-width: 1;
        }}

        .axis-line {{
            stroke: rgba(17, 39, 54, 0.25);
            stroke-width: 1.4;
        }}

        .pr-line {{
            fill: none;
            stroke: var(--accent);
            stroke-width: 3;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}

        .axis-label,
        .axis-title {{
            fill: var(--muted);
            font-size: 11px;
        }}

        .empty-state {{
            margin: 0;
            padding: 18px;
            border-radius: 14px;
            background: var(--accent-soft);
            color: var(--muted);
        }}

        .summary-note {{
            margin-top: 14px;
            padding: 14px 16px;
            border-radius: 14px;
            background: var(--accent-soft);
            color: var(--text);
        }}

        @media (max-width: 900px) {{
            body {{
                padding: 18px 12px 28px;
            }}

            .hero,
            .section-shell {{
                border-radius: 16px;
            }}

            .section-header,
            .section-body,
            .hero {{
                padding-left: 18px;
                padding-right: 18px;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <p>
                Reporte final de test para {escape(report.get('best_experiment', 'el mejor experimento'))}.
                Resume metricas globales, diferencias por clase, sensibilidad a NMS y curvas precision-recall por clase.
            </p>
            <div class="metric-grid">
                {_build_info_cards_html(summary)}
            </div>
            <div class="summary-note">
                {escape((nms_sensitivity.get('conclusion') or 'Sin conclusion disponible para el barrido de NMS.'))}
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Metricas por clase</h2>
                <p>mAP y mAR del checkpoint final sobre el split de test.</p>
            </div>
            <div class="section-body">
                <div class="table-wrap">{class_metrics_html}</div>
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Diagnostico del dataset</h2>
                <p>
                    Split: {escape(str(dataset_diagnostics.get('split')))} |
                    imagenes: {dataset_diagnostics.get('num_images', 0)} |
                    anotaciones: {dataset_diagnostics.get('num_annotations', 0)}
                </p>
            </div>
            <div class="section-body">
                <div class="table-wrap">{dataset_diagnostics_html}</div>
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Sensibilidad a NMS</h2>
                <p>
                    score_threshold={escape(str(nms_sensitivity.get('score_threshold')))} |
                    detections_per_img={escape(str(nms_sensitivity.get('detections_per_img')))} |
                    baseline_nms={escape(str(nms_sensitivity.get('baseline_nms_threshold')))}
                </p>
            </div>
            <div class="section-body">
                <div class="table-wrap">{nms_html}</div>
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Curvas precision-recall por clase</h2>
                <p>Una curva por clase a IoU=0.50, area=all y max_dets=100.</p>
            </div>
            <div class="section-body">
                <div class="pr-grid">
                    {pr_cards_html or '<p class="empty-state">No se generaron curvas precision-recall.</p>'}
                </div>
            </div>
        </section>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def detection_test_result_paths(run_id: str, output_dir) -> dict:
    output_dir = Path(output_dir)
    safe_run_id = str(run_id).strip()
    return {
        "json": output_dir / f"{safe_run_id}_test_result.json",
        "html": output_dir / f"{safe_run_id}_test_result_report.html",
    }


def save_detection_test_result_artifacts(
    report: dict,
    output_dir,
    title: str = "Reporte final de deteccion en test",
    canonical_json_path=None,
    canonical_html_path=None,
    update_canonical: bool = False,
) -> dict:
    if not report.get("run_id"):
        raise ValueError("El reporte de test debe incluir run_id para persistir artefactos por corrida.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = detection_test_result_paths(report["run_id"], output_dir)

    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    export_detection_test_report_html(report, paths["html"], title=title)

    if update_canonical:
        if canonical_json_path is not None:
            canonical_json_path = Path(canonical_json_path)
            canonical_json_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_json_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
        if canonical_html_path is not None:
            export_detection_test_report_html(report, canonical_html_path, title=title)

    return paths


def archive_canonical_detection_test_result(
    canonical_json_path,
    output_dir,
    overwrite: bool = False,
) -> Path | None:
    canonical_json_path = Path(canonical_json_path)
    if not canonical_json_path.exists():
        return None

    report = json.loads(canonical_json_path.read_text(encoding="utf-8"))
    run_id = report.get("run_id")
    if not run_id:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archived_path = detection_test_result_paths(run_id, output_dir)["json"]
    if archived_path.exists() and not overwrite:
        return archived_path

    archived_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return archived_path


def load_detection_test_results(test_results_dir) -> list[dict]:
    test_results_dir = Path(test_results_dir)
    if not test_results_dir.exists():
        return []

    reports = []
    for result_path in sorted(test_results_dir.glob("*_test_result.json")):
        report = json.loads(result_path.read_text(encoding="utf-8"))
        report["_result_json_path"] = result_path.as_posix()
        report["_result_html_path"] = detection_test_result_paths(
            report.get("run_id", result_path.stem.replace("_test_result", "")),
            test_results_dir,
        )["html"].as_posix()
        reports.append(report)
    return reports


def build_detection_test_results_comparison_df(
    runs_manifest_path,
    test_results_dir,
) -> pd.DataFrame:
    run_records = {
        record.get("run_id"): record
        for record in load_jsonl_records(runs_manifest_path)
        if record.get("run_id")
    }
    rows = []
    for report in load_detection_test_results(test_results_dir):
        run_id = report.get("run_id")
        run_record = run_records.get(run_id, {})
        class_metrics = {
            row.get("class_name"): row
            for row in report.get("class_metrics", [])
        }
        rows.append(
            {
                "run_id": run_id,
                "experiment": report.get("best_experiment") or run_record.get("name"),
                "checkpoint_path": report.get("checkpoint_path") or run_record.get("checkpoint_path"),
                "test_map": report.get("test_map"),
                "test_map_50": report.get("test_map_50"),
                "dent_map": (class_metrics.get("dent") or {}).get("map_per_class"),
                "scratch_map": (class_metrics.get("scratch") or {}).get("map_per_class"),
                "crack_map": (class_metrics.get("crack") or {}).get("map_per_class"),
                "result_json": report.get("_result_json_path"),
                "result_html": report.get("_result_html_path"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "experiment",
                "checkpoint_path",
                "test_map",
                "test_map_50",
                "dent_map",
                "scratch_map",
                "crack_map",
                "result_json",
                "result_html",
            ]
        )

    return pd.DataFrame(rows).sort_values(by="test_map", ascending=False, na_position="last").reset_index(drop=True)
