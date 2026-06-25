from __future__ import annotations

import hashlib
import sys
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prod.utils import (  # noqa: E402
    CLASS_COLORS,
    CLASS_LABELS_ES,
    CLASS_NAMES,
    DEFAULT_LABOR_RATE_USD,
    LABOR_RATE_RANGE_USD,
    MODEL_NAME,
    build_coverage_table,
    build_detection_table,
    build_impact_scenarios,
    draw_predictions,
    estimate_impact_report,
    load_evaluation_summary,
    load_model,
    load_model_metadata,
    run_inference,
    run_inference_two_pass,
    run_inference_tiled,
    summarize_detections,
    verify_detections,
)


st.set_page_config(
    page_title="Peritaje Visual Inteligente",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SESSION_DEFAULTS = {
    "inspection_original_image": None,
    "inspection_result_image": None,
    "inspection_detections": [],
    "inspection_summary": {},
    "inspection_threshold": None,
    "inspection_mode": None,
    "inspection_error": None,
    "inspection_signature": None,
    "inspection_source": None,
}

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-start: #08121a;
            --bg-end: #0f2028;
            --panel: rgba(255, 255, 255, 0.055);
            --panel-strong: rgba(255, 255, 255, 0.09);
            --line: rgba(255, 255, 255, 0.1);
            --text: #eef6f8;
            --muted: #a5b8c2;
            --teal: #2ca6a4;
            --teal-soft: rgba(44, 166, 164, 0.16);
            --amber: #d39a44;
            --amber-soft: rgba(211, 154, 68, 0.18);
            --danger: #c5654d;
            --danger-soft: rgba(197, 101, 77, 0.18);
            --critical: #8f3d45;
            --critical-soft: rgba(143, 61, 69, 0.22);
            --success: #4ea17b;
            --success-soft: rgba(78, 161, 123, 0.18);
            --ink: #102530;
            --shadow: 0 22px 58px rgba(0, 0, 0, 0.28);
            --shadow-soft: 0 12px 30px rgba(0, 0, 0, 0.18);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(44, 166, 164, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(211, 154, 68, 0.12), transparent 25%),
                linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(10, 22, 30, 0.98), rgba(8, 17, 24, 0.98)),
                linear-gradient(135deg, rgba(44, 166, 164, 0.06), transparent 40%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: var(--text);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.055);
        }

        [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            padding: 0.42rem 0.92rem;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            background: rgba(44, 166, 164, 0.18);
            border-color: rgba(44, 166, 164, 0.28);
        }

        .hero-shell {
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.6rem 1.7rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03)),
                linear-gradient(135deg, rgba(44, 166, 164, 0.1), transparent 44%),
                radial-gradient(circle at bottom right, rgba(94, 141, 218, 0.14), transparent 28%);
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            right: -60px;
            bottom: -70px;
            background: radial-gradient(circle, rgba(44, 166, 164, 0.22), transparent 67%);
            pointer-events: none;
        }

        .hero-grid {
            display: block;
            position: relative;
            z-index: 1;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.34rem 0.72rem;
            border-radius: 999px;
            background: rgba(44, 166, 164, 0.12);
            border: 1px solid rgba(44, 166, 164, 0.22);
            color: #ccf1ef;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0.8rem 0 0.3rem 0;
            color: #ffffff;
            font-size: 2.3rem;
            line-height: 1.03;
            font-weight: 800;
        }

        .hero-subtitle {
            margin: 0 0 0.55rem 0;
            color: #deedf2;
            font-size: 1.03rem;
            font-weight: 600;
        }

        .hero-description {
            margin: 0;
            color: var(--muted);
            font-size: 0.97rem;
            max-width: 960px;
            line-height: 1.55;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hero-chip,
        .class-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #f4fbfd;
            font-size: 0.84rem;
        }

        .section-shell {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            box-shadow: var(--shadow);
        }

        .section-title {
            margin: 0;
            color: #f5fbfd;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .section-caption {
            margin: 0.2rem 0 0 0;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .empty-state {
            border: 1px dashed rgba(255, 255, 255, 0.16);
            border-radius: 22px;
            padding: 1.3rem 1.35rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.03)),
                linear-gradient(135deg, rgba(44, 166, 164, 0.08), transparent 48%);
        }

        .empty-state h3 {
            margin: 0 0 0.4rem 0;
            color: #ffffff;
            font-size: 1.18rem;
        }

        .empty-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
            gap: 1rem;
            align-items: start;
        }

        .empty-state p,
        .empty-state li {
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.94rem;
        }

        .empty-state ul {
            margin: 0.65rem 0 0 1rem;
            padding: 0;
        }

        .signal-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
        }

        .signal-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 0.85rem 0.9rem;
            background: rgba(255, 255, 255, 0.045);
        }

        .signal-card strong {
            display: block;
            margin-bottom: 0.25rem;
            color: #ffffff;
            font-size: 0.88rem;
        }

        .signal-card span {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.45;
        }

        .severity-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 92px;
            border-radius: 999px;
            padding: 0.28rem 0.66rem;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .severity-low {
            background: var(--success-soft);
            border-color: rgba(78, 161, 123, 0.24);
            color: #d9f3e7;
        }

        .severity-medium {
            background: var(--teal-soft);
            border-color: rgba(44, 166, 164, 0.24);
            color: #d9f6f5;
        }

        .severity-high {
            background: var(--amber-soft);
            border-color: rgba(211, 154, 68, 0.24);
            color: #f6e8cf;
        }

        .severity-critical {
            background: var(--critical-soft);
            border-color: rgba(143, 61, 69, 0.24);
            color: #f3dde0;
        }

        .media-frame {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 0.95rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.03));
            box-shadow: var(--shadow-soft);
        }

        .media-caption {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.65rem;
        }

        .media-caption strong {
            color: #ffffff;
            font-size: 0.95rem;
        }

        .media-caption span {
            color: var(--muted);
            font-size: 0.85rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.95rem 0 1.1rem 0;
        }

        .metric-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            padding: 0.95rem 0.95rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.03));
            box-shadow: var(--shadow-soft);
        }

        .metric-card-label {
            display: block;
            color: var(--muted);
            font-size: 0.73rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.34rem;
            font-weight: 700;
        }

        .metric-card-value {
            color: #ffffff;
            font-size: 1.38rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.32rem;
        }

        .metric-card-note {
            color: var(--muted);
            font-size: 0.81rem;
            line-height: 1.35;
        }

        .muted {
            color: var(--muted);
        }

        .small {
            font-size: 0.82rem;
        }

        .interpret-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .interpret-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            background: rgba(255, 255, 255, 0.05);
        }

        .interpret-card h4 {
            margin: 0 0 0.35rem 0;
            color: #f3fbfd;
            font-size: 0.93rem;
        }

        .interpret-card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .highlight-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 0.2rem;
        }

        .highlight-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 0.9rem 0.95rem;
            background: rgba(255, 255, 255, 0.05);
        }

        .highlight-card h4 {
            margin: 0.55rem 0 0.22rem 0;
            color: #f8fcfd;
            font-size: 0.98rem;
        }

        .highlight-card p {
            margin: 0.1rem 0;
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        .findings-stack {
            display: grid;
            gap: 0.7rem;
        }

        .story-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .story-card {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            background: rgba(255, 255, 255, 0.05);
        }

        .story-card h4 {
            margin: 0 0 0.45rem 0;
            color: #f6fbfd;
            font-size: 1rem;
        }

        .story-card p,
        .story-card li {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.54;
        }

        .story-card ul,
        .story-card ol {
            margin: 0.55rem 0 0 1rem;
            padding: 0;
        }

        .flow-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .flow-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.85rem 0.9rem;
            background: rgba(255, 255, 255, 0.05);
        }

        .flow-card strong {
            display: block;
            margin-bottom: 0.22rem;
            color: #f4fbfd;
            font-size: 0.86rem;
        }

        .flow-card span {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.42;
        }

        .sidebar-note {
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 16px;
            padding: 0.85rem 0.9rem;
            background: rgba(255, 255, 255, 0.05);
        }

        @media (max-width: 900px) {
            .empty-grid,
            .interpret-grid,
            .story-grid,
            .flow-grid,
            .metric-strip,
            .highlight-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 1.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session_defaults() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/d"
    return f"{float(value):.{digits}f}"


def format_currency_range(min_value: int, max_value: int) -> str:
    return f"USD {min_value:,} - {max_value:,}"


def format_usd(value: int | float) -> str:
    return f"USD {int(round(value)):,}"


def get_model_display_name() -> str:
    try:
        metadata = load_model_metadata()
    except Exception:
        return MODEL_NAME
    return str(metadata.get("display_name") or MODEL_NAME)


def compute_signature(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as pil_image:
            return pil_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("No se pudo abrir la imagen. Verifica que sea un archivo JPG o PNG valido.") from exc


def sort_detections(detections: list[dict]) -> list[dict]:
    return sorted(
        detections,
        key=lambda item: (
            int(item.get("severity_score", 0)),
            float(item.get("score", 0.0)),
            float(item.get("area_pct", 0.0)),
        ),
        reverse=True,
    )


def build_inspection_summary(detections: list[dict]) -> dict:
    summary = summarize_detections(detections)
    if not detections:
        return {
            **summary,
            "score_max": 0.0,
            "area_pct_total": 0.0,
            "top_damage": "Sin hallazgos",
        }

    top_detection = max(
        detections,
        key=lambda item: (
            int(item.get("severity_score", 0)),
            float(item.get("score", 0.0)),
            float(item.get("area_pct", 0.0)),
        ),
    )
    return {
        **summary,
        "score_max": max(float(det.get("score", 0.0)) for det in detections),
        "area_pct_total": sum(float(det.get("area_pct", 0.0)) for det in detections),
        "top_damage": top_detection.get("class_name_es", "Sin hallazgos"),
    }


def crop_detection_focus(
    image: Image.Image,
    box: list[int],
    padding_ratio: float = 0.18,
) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    pad_x = int((x1 - x0) * padding_ratio)
    pad_y = int((y1 - y0) * padding_ratio)
    crop_box = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )
    return image.crop(crop_box)


def build_impact_recommendation(impact_summary: dict) -> tuple[str, str]:
    policy_name = str(impact_summary.get("selected_policy", "la poliza")).lower()
    midpoint_total = float(impact_summary.get("midpoint_total", 0.0))
    out_of_pocket_mid = float(impact_summary.get("out_of_pocket_mid", 0.0))
    covered_mid = float(impact_summary.get("covered_mid", 0.0))
    deductible = int(impact_summary.get("deductible", 0))

    if midpoint_total <= 0:
        return (
            "Sin impacto economico estimado",
            "No hay un rango economico relevante para analizar con la configuracion actual.",
        )

    if covered_mid <= 0:
        return (
            "Cobertura insuficiente",
            f"Con {policy_name}, los danos detectados no entran en la cobertura simulada. La decision economica dependeria de un presupuesto particular o de otra poliza.",
        )

    if midpoint_total <= deductible:
        return (
            "Costo menor a la franquicia",
            f"El costo medio estimado ({format_usd(midpoint_total)}) queda por debajo de la franquicia ({format_usd(deductible)}), por lo que abrir un siniestro probablemente no mejore el gasto de bolsillo.",
        )

    if out_of_pocket_mid <= midpoint_total * 0.55:
        return (
            "Cobertura favorable",
            f"Con la poliza seleccionada, el gasto de bolsillo medio estimado ({format_usd(out_of_pocket_mid)}) cae sensiblemente frente al costo total ({format_usd(midpoint_total)}).",
        )

    return (
        "Comparar presupuesto y poliza",
        f"La cobertura reduce una parte del costo, pero el gasto de bolsillo medio ({format_usd(out_of_pocket_mid)}) sigue siendo material. Conviene contrastarlo con un presupuesto real del taller.",
    )


def build_impact_report_html(
    impact_report: dict,
    summary: dict,
    latest_summary: dict,
) -> str:
    source_items = "".join(
        f"<li><strong>{escape(str(item.get('source', '-')))}</strong>: "
        f"<a href=\"{escape(str(item.get('url', '#')))}\">{escape(str(item.get('url', '#')))}</a><br>"
        f"{escape(str(item.get('note', '-')))}</li>"
        for item in impact_report.get("source_rows", [])
    )
    damage_rows = "".join(
        """
        <tr>
            <td>{dano}</td>
            <td>{sev}</td>
            <td>{score}</td>
            <td>{route}</td>
            <td>{labor}</td>
            <td>{materials}</td>
            <td>{parts}</td>
            <td>{total}</td>
        </tr>
        """.format(
            dano=escape(str(row.get("Dano", "-"))),
            sev=escape(str(row.get("Severidad", "-"))),
            score=f"{float(row.get('Score detector', 0.0)):.1%}",
            route=escape(str(row.get("Ruta de reparacion", "-"))),
            labor=escape(format_currency_range(int(row.get("Mano de obra minimo", 0)), int(row.get("Mano de obra maximo", 0)))),
            materials=escape(format_currency_range(int(row.get("Materiales minimo", 0)), int(row.get("Materiales maximo", 0)))),
            parts=escape(format_currency_range(int(row.get("Repuestos minimo", 0)), int(row.get("Repuestos maximo", 0)))),
            total=escape(format_currency_range(int(row.get("Costo minimo", 0)), int(row.get("Costo maximo", 0)))),
        )
        for row in impact_report.get("rows", [])
    )
    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Reporte de impacto estimado</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 32px; color: #102530; }}
            h1, h2 {{ margin-bottom: 8px; }}
            p {{ line-height: 1.5; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
            th, td {{ border: 1px solid #d8e1e5; padding: 10px; text-align: left; vertical-align: top; }}
            th {{ background: #eef5f7; }}
            .kpi {{ display: inline-block; min-width: 180px; margin: 0 16px 16px 0; padding: 14px 16px; border: 1px solid #d8e1e5; border-radius: 12px; }}
            .muted {{ color: #52656f; }}
        </style>
    </head>
    <body>
        <h1>Peritaje Visual Inteligente</h1>
        <p class="muted">Reporte exportado desde la cabina de analisis visual.</p>
        <div class="kpi"><strong>Hallazgos</strong><br>{int(latest_summary.get("count", 0))}</div>
        <div class="kpi"><strong>Severidad maxima</strong><br>{escape(str(latest_summary.get("max_severity", "Sin hallazgos")))}</div>
        <div class="kpi"><strong>Rango total</strong><br>{escape(format_currency_range(int(summary.get("total_min", 0)), int(summary.get("total_max", 0))))}</div>
        <div class="kpi"><strong>Gasto de bolsillo</strong><br>{escape(format_usd(summary.get("out_of_pocket_mid", 0)))}</div>
        <h2>Desglose por hallazgo</h2>
        <table>
            <thead>
                <tr>
                    <th>Dano</th>
                    <th>Severidad</th>
                    <th>Score detector</th>
                    <th>Ruta de reparacion</th>
                    <th>Mano de obra</th>
                    <th>Materiales</th>
                    <th>Repuestos</th>
                    <th>Total</th>
                </tr>
            </thead>
            <tbody>{damage_rows}</tbody>
        </table>
        <h2>Fuentes publicas utilizadas</h2>
        <ul>{source_items}</ul>
    </body>
    </html>
    """


def persist_inspection_state(
    original_image: Image.Image,
    result_image: Image.Image,
    detections: list[dict],
    summary: dict,
    threshold: float,
    scan_mode: str,
    signature: str,
    source_label: str,
) -> None:
    st.session_state["inspection_original_image"] = original_image
    st.session_state["inspection_result_image"] = result_image
    st.session_state["inspection_detections"] = detections
    st.session_state["inspection_summary"] = summary
    st.session_state["inspection_threshold"] = threshold
    st.session_state["inspection_mode"] = scan_mode
    st.session_state["inspection_error"] = None
    st.session_state["inspection_signature"] = signature
    st.session_state["inspection_source"] = source_label


def render_section_header(title: str, caption: str) -> None:
    with st.container(border=False):
        st.markdown(f"#### {escape(title)}")
        st.caption(caption)


def render_metric_strip(cards: list[dict]) -> None:
    columns_per_row = 3 if len(cards) <= 3 else 6
    for start in range(0, len(cards), columns_per_row):
        row_cards = cards[start:start + columns_per_row]
        columns = st.columns(len(row_cards), gap="medium")
        for column, card in zip(columns, row_cards):
            with column:
                with st.container(border=True):
                    st.caption(card["label"])
                    st.markdown(f"### {card['value']}")
                    st.caption(card["note"])


def render_hero(evaluation_result: dict) -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-grid">
                <div class="hero-main">
                    <div class="hero-kicker">Analisis visual</div>
                    <h1 class="hero-title">Peritaje Visual Inteligente</h1>
                    <p class="hero-subtitle">Deteccion de danos vehiculares</p>
                    <p class="hero-description">
                        Lectura visual clara de hallazgos detectados sobre imagenes de vehiculos.
                    </p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def auto_select_scan_config(pil_image: Image.Image) -> tuple[str, int, float, bool]:
    """Devuelve la configuracion automatica por defecto.

    El flujo normal usa siempre lectura estandar, verificacion high-detail y
    umbral 0.50. La consola tecnica permite cambiar estos valores manualmente.
    """
    _ = pil_image
    return "Estandar", 2, 0.50, True


def render_sidebar(evaluation_result: dict) -> dict:
    summary = evaluation_result.get("summary", {})
    with st.sidebar:
        technical_console = st.toggle(
            "Consola tecnica",
            value=False,
            help="Controles avanzados de umbral y modo de escaneo. Apagado: el sistema elige todo automaticamente.",
        )
        if not technical_console:
            st.caption("Modo automatico: subi una imagen y el sistema elige la mejor configuracion.")
            return {"technical_console": False}

        st.space("small")
        st.markdown("### :material/tune: Consola tecnica")
        st.caption("Controles globales.")

        score_threshold = st.slider(
            "Umbral de score del detector",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            help="Filtra detecciones con score menor al umbral configurado.",
        )
        scan_mode = st.segmented_control(
            "Modo de escaneo",
            options=["Estandar", "Tiled"],
            default="Estandar",
            selection_mode="single",
            help="Estandar: lectura directa de la imagen completa. Tiled: divide la imagen en una grilla de cuadrantes con overlap para detectar danos en distintas zonas.",
        ) or "Estandar"

        grid_size = 2
        if scan_mode == "Tiled":
            grid_size = st.slider(
                "Cuadrantes",
                min_value=2,
                max_value=4,
                value=2,
                step=1,
                help="Tamano de la grilla. 2 → 2×2 (4 tiles), 3 → 3×3 (9 tiles), 4 → 4×4 (16 tiles).",
            )
            st.caption(f"Grilla: {grid_size}×{grid_size} = {grid_size ** 2} cuadrantes con overlap del 10%")

        high_detail = st.toggle(
            "High-detail",
            value=False,
            help="Reverifica cada hallazgo en una segunda pasada local. Funciona con Estandar y Tiled. No agrega detecciones nuevas: solo confirma o descarta las ya encontradas.",
        )

        _mode_parts = []
        if scan_mode == "Tiled":
            _mode_parts.append(f"Tiled {grid_size}×{grid_size}")
        else:
            _mode_parts.append("Estandar")
        if high_detail:
            _mode_parts.append("High-detail")
        _mode_label = " + ".join(_mode_parts)

        _base_desc = (
            f"Division en {grid_size}×{grid_size} cuadrantes con overlap y NMS global."
            if scan_mode == "Tiled"
            else "Lectura directa de la imagen completa."
        )
        _hd_note = " Segunda verificacion local activada." if high_detail else ""
        st.markdown(
            f"""
            <div class="sidebar-note">
                <strong>Modo activo</strong><br>
                {escape(_mode_label)}<br>
                <span class="small muted">
                    {escape(_base_desc + _hd_note)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.space("small")
        st.markdown("#### :material/query_stats: Ficha tecnica")
        st.caption(get_model_display_name())
        st.markdown(
            f"""
            :blue-badge[CarDD COCO] :green-badge[6 clases]  
            mAP@50:95 **{format_metric(summary.get("map"), 3)}**  
            mAP@50 **{format_metric(summary.get("map_50"), 3)}**  
            mAR@100 **{format_metric(summary.get("mar_100"), 3)}**
            """
        )

        with st.expander("Clases detectadas", icon=":material/category:"):
            class_chips = []
            for class_id, class_name in CLASS_NAMES.items():
                color = CLASS_COLORS.get(class_id, "#ffffff")
                label = CLASS_LABELS_ES.get(class_id, class_name)
                class_chips.append(
                    f'<span class="class-chip"><span style="width:10px;height:10px;border-radius:50%;display:inline-block;background:{escape(color)};"></span>{escape(label)}<span class="small muted">{escape(class_name)}</span></span>'
                )
            st.markdown(f'<div class="chip-row">{"".join(class_chips)}</div>', unsafe_allow_html=True)


    return {
        "technical_console": True,
        "score_threshold": score_threshold,
        "scan_mode": scan_mode,
        "high_detail": high_detail,
        "grid_size": grid_size,
    }


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-grid">
                <div>
                    <h3>Carga una imagen</h3>
                    <p>
                        Carga una foto del vehiculo para obtener detecciones, severidad estimada y una lectura economica orientativa.
                    </p>
                    <ul>
                        <li>Mejores casos: vidrio roto, rueda pinchada, faro roto.</li>
                        <li>Mas exigentes: rayon, abolladura y grieta pequena.</li>
                        <li>Conviene usar imagenes nitidas y con el dano visible.</li>
                    </ul>
                </div>
                <div class="signal-grid">
                    <div class="signal-card">
                        <strong>Entrada recomendada</strong>
                        <span>Imagen frontal o lateral donde el vehiculo ocupe buena parte del cuadro.</span>
                    </div>
                    <div class="signal-card">
                        <strong>Salida</strong>
                        <span>Cajas, score del detector, severidad y tabla tecnica de hallazgos.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel() -> tuple[Image.Image | None, str | None, str | None]:
    render_section_header(
        ":material/upload: Cargar imagen",
        "Arrastra una imagen JPG o PNG al panel, o usa la camara si prefieres capturarla en el momento.",
    )
    upload_tab, camera_tab = st.tabs(
        [
            ":material/upload_file: Arrastrar o subir",
            ":material/photo_camera: Camara",
        ]
    )

    image_bytes: bytes | None = None
    source_label: str | None = None

    with upload_tab:
        st.caption("Arrastra y suelta una imagen en el panel de abajo, o haz clic para buscarla en tu equipo.")
        uploaded = st.file_uploader(
            "Arrastra una imagen aqui o haz clic para subirla",
            type=["jpg", "jpeg", "png"],
            help="Formatos admitidos: JPG, JPEG y PNG. Conviene que el vehiculo ocupe buena parte del cuadro.",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            source_label = "archivo"

    with camera_tab:
        camera_photo = st.camera_input("Usar camara")
        if camera_photo is not None and image_bytes is None:
            image_bytes = camera_photo.getvalue()
            source_label = "camara"

    if image_bytes is None:
        return None, None, None

    image_signature = compute_signature(image_bytes)
    pil_image = open_image_from_bytes(image_bytes)
    return pil_image, image_signature, source_label


def render_kpi_cards(summary: dict) -> None:
    render_metric_strip(
        [
            {
                "label": "Danos detectados",
                "value": str(summary.get("count", 0)),
                "note": "Hallazgos filtrados por umbral",
            },
            {
                "label": "Severidad maxima",
                "value": str(summary.get("max_severity", "Sin hallazgos")),
                "note": "Regla por clase y area relativa",
            },
            {
                "label": "Clase dominante",
                "value": str(summary.get("dominant_class", "Sin detecciones")),
                "note": "Mayor presencia en la lectura actual",
            },
            {
                "label": "Score maximo",
                "value": f"{float(summary.get('score_max', 0.0)):.1%}",
                "note": "Confianza relativa del mejor hallazgo",
            },
        ]
    )


def render_image_comparison(original_image: Image.Image, result_image: Image.Image, mode_label: str) -> None:
    col_original, col_result = st.columns(2, gap="medium")
    with col_original:
        with st.container(border=True):
            st.markdown('<div class="media-caption"><strong>Imagen original</strong><span>Entrada visual usada por el flujo de inferencia.</span></div>', unsafe_allow_html=True)
            st.image(original_image)
    with col_result:
        with st.container(border=True):
            st.markdown(f'<div class="media-caption"><strong>Resultado anotado</strong><span>Visualizacion del detector en modo {escape(mode_label.lower())}.</span></div>', unsafe_allow_html=True)
            st.image(result_image)


def render_highlight_cards(detections: list[dict]) -> None:
    top_detections = sort_detections(detections)[:3]
    if not top_detections:
        return

    st.markdown("**Hallazgos clave**")
    columns = st.columns(len(top_detections), gap="medium")
    severity_badges = {
        "Leve": ":green-badge[Leve]",
        "Moderada": ":blue-badge[Moderada]",
        "Alta": ":orange-badge[Alta]",
        "Critica": ":red-badge[Critica]",
    }

    for column, det in zip(columns, top_detections):
        severity = det.get("severity_label", "Leve")
        with column:
            with st.container(border=True):
                st.markdown(severity_badges.get(severity, f":gray-badge[{severity}]"))
                st.markdown(f"**{det.get('class_name_es', '-')}**")
                st.caption(det.get("class_name", "-"))
                st.markdown(f"Score del detector: **{float(det.get('score', 0.0)):.1%}**")
                st.markdown(f"Area relativa: **{float(det.get('area_pct', 0.0)):.1f}%**")
                st.markdown(
                    f"Impacto orientativo: **{format_currency_range(int(det.get('cost_min', 0)), int(det.get('cost_max', 0)))}**"
                )


def render_detection_table(detections: list[dict]) -> None:
    render_section_header(
        ":material/table_chart: Hallazgos priorizados",
        "Lectura ordenada por severidad y score del detector.",
    )
    table_rows = []
    for det in sort_detections(detections):
        table_rows.append(
            {
                "Clase": det.get("class_name_es", "-"),
                "Score del detector": float(det.get("score", 0.0)),
                "Severidad": det.get("severity_label", "Leve"),
                "Area relativa": float(det.get("area_pct", 0.0)) / 100.0,
                "Rango orientativo": format_currency_range(int(det.get("cost_min", 0)), int(det.get("cost_max", 0))),
            }
        )

    styled_df = pd.DataFrame(table_rows)
    severity_colors = {
        "Leve": "background-color: rgba(78, 161, 123, 0.18); color: #d9f3e7;",
        "Moderada": "background-color: rgba(44, 166, 164, 0.18); color: #d9f6f5;",
        "Alta": "background-color: rgba(211, 154, 68, 0.18); color: #f6e8cf;",
        "Critica": "background-color: rgba(143, 61, 69, 0.22); color: #f3dde0;",
    }

    def severity_style(value: str) -> str:
        return severity_colors.get(value, "")

    styled_view = styled_df.style.map(severity_style, subset=["Severidad"])

    st.dataframe(
        styled_view,
        hide_index=True,
        column_config={
            "Score del detector": st.column_config.NumberColumn("Score del detector", format="percent"),
            "Area relativa": st.column_config.NumberColumn("Area relativa", format="percent"),
            "Rango orientativo": st.column_config.TextColumn("Rango orientativo", width="medium"),
            "Clase": st.column_config.TextColumn("Clase", width="medium"),
        },
    )
    with st.expander("Ver detalle tecnico", icon=":material/data_table:"):
        table_df = pd.DataFrame(build_detection_table(sort_detections(detections)))
        st.dataframe(table_df, hide_index=True)


def render_interpretation_panel(detections: list[dict], summary: dict) -> None:
    render_section_header(
        ":material/visibility: Interpretacion practica",
        "Lectura breve para interpretar la salida del detector.",
    )
    if not detections:
        st.warning(
            "No hubo hallazgos visibles con la configuracion actual. Eso no prueba ausencia de dano: puede ser un caso borde, una imagen de baja calidad o un caso fuera del dominio del dataset.",
            icon=":material/warning:",
        )
        st.caption("Limitacion: esta lectura no reemplaza revision profesional ni diagnostico mecanico.")
        return

    sorted_detections = sort_detections(detections)
    top_detection = sorted_detections[0]
    review_message = (
        "Conviene revisar manualmente detalles finos o lineales, porque rayones y grietas pequenas son visualmente mas exigentes."
        if any(det.get("class_name_es") in {"Rayon", "Grieta"} for det in detections)
        else "Conviene revisar que la region marcada corresponda exactamente al dano visible y no a reflejos, perspectiva o fondo."
    )
    cards = [
        (
            "Que encontro el modelo",
            f"Se detectaron {summary.get('count', 0)} hallazgo(s) y la clase dominante fue {summary.get('dominant_class', 'Sin detecciones')}.",
        ),
        (
            "Hallazgo principal",
            f"{top_detection.get('class_name_es', 'Sin hallazgos')} aparece como el hallazgo prioritario por severidad {top_detection.get('severity_label', 'Leve')} y score {float(top_detection.get('score', 0.0)):.1%}.",
        ),
        (
            "Como leer el score",
            f"El score maximo observado fue {float(summary.get('score_max', 0.0)):.1%}. Debe leerse como confianza relativa del detector, no como probabilidad calibrada.",
        ),
    ]
    columns = st.columns(len(cards), gap="medium")
    for column, (title, text) in zip(columns, cards):
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(text)
    st.caption(f"Revision manual sugerida: {review_message}")
    st.caption("Limitacion: esta salida no reemplaza revision profesional ni diagnostico mecanico.")


def render_download(result_image: Image.Image) -> None:
    buffer = BytesIO()
    result_image.save(buffer, format="PNG")
    st.download_button(
        "Descargar evidencia visual",
        data=buffer.getvalue(),
        file_name="peritaje_visual_cardd.png",
        mime="image/png",
        icon=":material/download:",
    )


def run_analysis_flow(
    pil_image: Image.Image,
    image_signature: str,
    source_label: str,
    score_threshold: float,
    scan_mode: str,
    high_detail: bool,
    grid_size: int,
) -> tuple[list[dict], Image.Image, dict]:
    mode_parts = [f"Tiled {grid_size}x{grid_size}" if scan_mode == "Tiled" else "Estandar"]
    if high_detail:
        mode_parts.append("High-detail")
    mode_label = " + ".join(mode_parts)

    analysis_signature = f"{image_signature}|{score_threshold:.2f}|{mode_label}"
    if (
        st.session_state.get("inspection_signature") == analysis_signature
        and st.session_state.get("inspection_result_image") is not None
        and st.session_state.get("inspection_error") is None
    ):
        return (
            st.session_state.get("inspection_detections", []),
            st.session_state.get("inspection_result_image"),
            st.session_state.get("inspection_summary", {}),
        )

    with st.spinner("Analizando imagen..."):
        model = load_model()
        if scan_mode == "Tiled":
            detections = run_inference_tiled(model, pil_image, score_threshold=score_threshold, grid_size=grid_size)
        else:
            detections = run_inference(model, pil_image, score_threshold=score_threshold)
        if high_detail:
            detections = verify_detections(model, pil_image, detections, score_threshold=score_threshold)

    detections = sort_detections(detections)
    result_image = draw_predictions(pil_image, detections)
    summary = build_inspection_summary(detections)
    persist_inspection_state(
        original_image=pil_image,
        result_image=result_image,
        detections=detections,
        summary=summary,
        threshold=score_threshold,
        scan_mode=mode_label,
        signature=analysis_signature,
        source_label=source_label,
    )
    return detections, result_image, summary


def render_inspection_tab(sidebar_config: dict) -> None:
    render_section_header(
        ":material/car_crash: Analizar imagen",
        "Carga una imagen, ejecuta el detector y revisa los hallazgos encontrados.",
    )
    try:
        pil_image, image_signature, source_label = render_input_panel()
    except ValueError as exc:
        st.session_state["inspection_error"] = str(exc)
        st.error(str(exc), icon=":material/error:")
        return

    if pil_image is None or image_signature is None or source_label is None:
        render_empty_state()
        if st.session_state.get("inspection_error"):
            st.caption(f"Ultimo error registrado: {st.session_state['inspection_error']}")
        return

    if sidebar_config.get("technical_console"):
        score_threshold = sidebar_config["score_threshold"]
        scan_mode = sidebar_config["scan_mode"]
        high_detail = sidebar_config["high_detail"]
        grid_size = sidebar_config["grid_size"]
    else:
        scan_mode, grid_size, score_threshold, high_detail = auto_select_scan_config(pil_image)
        st.caption("Configuracion automatica aplicada: Estandar + High-detail, umbral 50%.")

    try:
        detections, result_image, summary = run_analysis_flow(
            pil_image=pil_image,
            image_signature=image_signature,
            source_label=source_label,
            score_threshold=score_threshold,
            scan_mode=scan_mode,
            high_detail=high_detail,
            grid_size=grid_size,
        )
    except Exception as exc:
        st.session_state["inspection_error"] = str(exc)
        st.error(f"No se pudo ejecutar el analisis: {exc}", icon=":material/error:")
        return

    render_kpi_cards(summary)
    render_image_comparison(
        pil_image,
        result_image,
        scan_mode,
    )

    if detections:
        render_detection_table(detections)
        render_interpretation_panel(detections, summary)
        render_download(result_image)
    else:
        st.warning(
            f"No se detectaron danos con umbral {score_threshold:.0%}. Prueba un umbral menor, una toma mas cercana o una imagen mas alineada con el dataset.",
            icon=":material/warning:",
        )
        render_interpretation_panel(detections, summary)


def render_impact_dashboard() -> None:
    render_section_header(
        ":material/paid: Impacto estimado",
        "Lectura orientativa de costos, prioridad y cobertura basada en la ultima inspeccion persistida.",
    )
    detections = st.session_state.get("inspection_detections", [])
    summary = st.session_state.get("inspection_summary", {})

    if not detections and not summary:
        st.info(
            "Primero analiza una imagen para ver el impacto estimado de los hallazgos detectados.",
            icon=":material/info:",
        )
        return

    coverage_rows = build_coverage_table(detections)
    coverage_options = [row["Cobertura"] for row in coverage_rows]

    st.caption("Referencia economica orientativa basada en fuentes publicas. No reemplaza una cotizacion oficial.")

    with st.expander("Supuestos economicos", icon=":material/tune:"):
        control_a, control_b = st.columns(2, gap="medium")
        with control_a:
            labor_rate = st.slider(
                "Tarifa laboral estimada (USD/h)",
                min_value=LABOR_RATE_RANGE_USD[0],
                max_value=LABOR_RATE_RANGE_USD[1],
                value=DEFAULT_LABOR_RATE_USD,
                step=1,
                help="Rango base tomado de la guia AAA 2026 de tarifas laborales.",
            )
            selected_policy = st.selectbox(
                "Cobertura simulada",
                options=coverage_options,
                index=coverage_options.index("Todo riesgo") if "Todo riesgo" in coverage_options else 0,
            )
        with control_b:
            shop_profile = st.segmented_control(
                "Perfil de taller",
                options=["Economico", "Estandar", "Premium"],
                default="Estandar",
                selection_mode="single",
            )
            parts_policy = st.segmented_control(
                "Politica de repuestos",
                options=["Aftermarket", "Mixto", "OEM"],
                default="Mixto",
                selection_mode="single",
            )
            deductible = st.number_input(
                "Franquicia / deducible (USD)",
                min_value=0,
                value=500,
                step=50,
                help="Se usa para estimar el gasto de bolsillo en la cobertura seleccionada.",
            )

    impact_report = estimate_impact_report(
        detections,
        labor_rate=labor_rate,
        shop_profile=shop_profile or "Estandar",
        parts_policy=parts_policy or "Mixto",
        selected_policy=selected_policy,
        deductible=int(deductible),
    )
    impact_summary = impact_report["summary"]
    recommendation_title, recommendation_text = build_impact_recommendation(impact_summary)

    render_metric_strip(
        [
            {
                "label": "Severidad maxima",
                "value": str(summary.get("max_severity", "Sin hallazgos")),
                "note": "Mayor severidad observada",
            },
            {
                "label": "Rango total",
                "value": format_currency_range(int(impact_summary["total_min"]), int(impact_summary["total_max"])),
                "note": "Resultado con supuestos actuales",
            },
            {
                "label": "Costo medio",
                "value": f"USD {int(round(impact_summary['midpoint_total'])):,}",
                "note": "Punto medio del rango estimado",
            },
            {
                "label": "Gasto de bolsillo",
                "value": f"USD {int(round(impact_summary['out_of_pocket_mid'])):,}",
                "note": f"Escenario con {selected_policy.lower()} y franquicia",
            },
            {
                "label": "Cobertura",
                "value": selected_policy,
                "note": f"Taller {shop_profile.lower()} - repuestos {parts_policy.lower()}",
            },
        ]
    )

    with st.container(border=True):
        st.markdown(f"**{recommendation_title}**")
        st.write(recommendation_text)
        st.caption(
            "La recomendacion compara costo medio estimado, cobertura simulada y franquicia. Es una ayuda para lectura economica, no una decision contractual."
        )

    cost_rows = [
        {
            "Dano": row["Dano"],
            "Severidad": row["Severidad"],
            "Ruta de reparacion": row["Ruta de reparacion"],
            "Total": format_currency_range(int(row["Costo minimo"]), int(row["Costo maximo"])),
        }
        for row in impact_report["rows"]
    ]

    col_costs, col_coverage = st.columns(2, gap="medium")
    with col_costs:
        with st.container(border=True):
            st.markdown("**Costos por hallazgo**")
            st.caption("Cada rango se recalcula con la tarifa laboral, perfil de taller y politica de repuestos seleccionados.")
            st.dataframe(
                pd.DataFrame(cost_rows),
                hide_index=True,
                column_config={
                    "Total": st.column_config.TextColumn("Total", width="medium"),
                },
            )
    with col_coverage:
        with st.container(border=True):
            st.markdown("**Cobertura y gasto de bolsillo**")
            st.caption("La franquicia se aplica solo sobre danos cubiertos por la poliza seleccionada.")
            st.dataframe(pd.DataFrame(coverage_rows), hide_index=True)
            if impact_summary["covered_mid"] > 0:
                if impact_summary["out_of_pocket_mid"] < impact_summary["midpoint_total"]:
                    st.success(
                        f"Con {selected_policy.lower()}, el gasto estimado del usuario seria de aproximadamente USD {int(round(impact_summary['out_of_pocket_mid'])):,}, por debajo del costo medio total.",
                        icon=":material/check_circle:",
                    )
                else:
                    st.warning(
                        f"Aun con {selected_policy.lower()}, el gasto de bolsillo estimado ronda USD {int(round(impact_summary['out_of_pocket_mid'])):,}.",
                        icon=":material/warning:",
                    )
            else:
                st.warning(
                    f"La poliza {selected_policy.lower()} no cubre los hallazgos actuales segun la logica simulada.",
                    icon=":material/warning:",
                )

    if impact_report["rows"]:
        focus_options = {
            f"{index + 1}. {row['Dano']} - {row['Severidad']} - {row['Score detector']:.1%}": row
            for index, row in enumerate(impact_report["rows"])
        }
        selected_focus_label = st.selectbox(
            "Revisar un hallazgo",
            options=list(focus_options.keys()),
            help="Permite revisar un hallazgo puntual junto con su ruta de reparacion y desglose economico.",
        )
        selected_focus = focus_options[selected_focus_label]
        original_image = st.session_state.get("inspection_original_image")
        if original_image is not None:
            focused_detection = next(
                (
                    det for det in detections
                    if det.get("class_name_es") == selected_focus["Dano"]
                    and abs(float(det.get("score", 0.0)) - float(selected_focus["Score detector"])) < 1e-6
                ),
                detections[0],
            )
            crop = crop_detection_focus(
                original_image,
                focused_detection.get("box", [0, 0, original_image.width, original_image.height]),
            )
            focus_col_a, focus_col_b = st.columns([1.1, 1], gap="medium")
            with focus_col_a:
                with st.container(border=True):
                    st.markdown("**Detalle visual**")
                    st.image(crop)
            with focus_col_b:
                with st.container(border=True):
                    st.markdown("**Lectura tecnica**")
                    st.write(f"Ruta sugerida: {selected_focus['Ruta de reparacion']}")
                    st.write(
                        "Desglose estimado: "
                        f"mano de obra {format_currency_range(int(selected_focus['Mano de obra minimo']), int(selected_focus['Mano de obra maximo']))}, "
                        f"materiales {format_currency_range(int(selected_focus['Materiales minimo']), int(selected_focus['Materiales maximo']))}, "
                        f"repuestos {format_currency_range(int(selected_focus['Repuestos minimo']), int(selected_focus['Repuestos maximo']))}."
                    )
                    st.caption(
                        "El recorte mantiene contexto alrededor del box para ayudar a contrastar el dano visible con la lectura economica."
                    )
    else:
        st.info(
            "La ultima inspeccion no dejo hallazgos confirmados. El tablero economico sigue disponible para revisar metodologia y cobertura simulada.",
            icon=":material/info:",
        )

    with st.expander("Sensibilidad por perfil de taller", icon=":material/tune:"):
        st.caption("Muestra como cambia el rango total segun el perfil del taller.")
        scenario_df = pd.DataFrame(build_impact_scenarios(detections, labor_rate=labor_rate, parts_policy=parts_policy or "Mixto"))
        st.dataframe(
            scenario_df,
            hide_index=True,
            column_config={
                "Costo minimo": st.column_config.NumberColumn("Costo minimo", format="USD %d"),
                "Costo medio": st.column_config.NumberColumn("Costo medio", format="USD %d"),
                "Costo maximo": st.column_config.NumberColumn("Costo maximo", format="USD %d"),
            },
        )

    report_html = build_impact_report_html(impact_report, impact_summary, summary)
    st.download_button(
        "Descargar reporte tecnico (HTML)",
        data=report_html.encode("utf-8"),
        file_name="impacto_estimado_peritaje_visual.html",
        mime="text/html",
        icon=":material/description:",
    )

    with st.expander("Metodologia y fuentes", icon=":material/menu_book:"):
        st.markdown(
            """
            La estimacion no usa un precio fijo unico. Se arma con tres capas:

            1. Tarifa laboral base.
            2. Ruta de reparacion asociada al tipo de dano y su severidad.
            3. Benchmarks publicos para vidrio, faros, neumaticos, rayones y abolladuras.

            Cuando no existe un benchmark publico exacto para una clase del dataset, la app lo indica y usa una aproximacion basada en mano de obra y complejidad de refinish.
            """
        )
        methodology_rows = pd.DataFrame(impact_report["source_rows"])
        if not methodology_rows.empty:
            st.dataframe(
                methodology_rows.rename(
                    columns={
                        "source": "Fuente",
                        "url": "Link",
                        "note": "Como se usa",
                    }
                ),
                hide_index=True,
                column_config={
                    "Link": st.column_config.LinkColumn("Link"),
                },
            )


def render_model_metrics(evaluation_result: dict) -> None:
    render_section_header(
        ":material/query_stats: Rendimiento del modelo",
        "Metricas reales del conjunto de test.",
    )
    if not evaluation_result:
        st.error(
            "No se encontraron metricas finales. Verifica la presencia de dev/best_test_result.json para poblar esta seccion.",
            icon=":material/error:",
        )
        return

    summary = evaluation_result.get("summary", {})
    render_metric_strip(
        [
            {"label": "mAP@50:95", "value": format_metric(summary.get("map"), 4), "note": "Metrica principal de deteccion"},
            {"label": "mAP@50", "value": format_metric(summary.get("map_50"), 4), "note": "IoU menos estricto"},
            {"label": "mAP@75", "value": format_metric(summary.get("map_75"), 4), "note": "Localizacion mas exigente"},
            {"label": "mAR@100", "value": format_metric(summary.get("mar_100"), 4), "note": "Recall medio"},
        ]
    )

    class_name_map = {name: CLASS_LABELS_ES.get(class_id, name) for class_id, name in CLASS_NAMES.items()}
    class_metrics = pd.DataFrame(evaluation_result.get("class_metrics", []))
    diagnostics = pd.DataFrame((evaluation_result.get("dataset_diagnostics") or {}).get("per_class", []))

    if not class_metrics.empty:
        with st.container(border=True):
            st.markdown("**Lectura por clase**")
            st.caption("Comparacion de precision media y recall medio por tipo de dano.")
            metrics_view = class_metrics.copy()
            metrics_view["Clase"] = metrics_view["class_name"].map(class_name_map).fillna(metrics_view["class_name"])
            st.bar_chart(
                metrics_view.set_index("Clase")[["map_per_class", "mar_100_per_class"]],
                stack=False,
            )

    st.caption(
        "Estas metricas provienen del conjunto de test. Glass shatter y tire flat tienden a rendir mejor; crack, dent y scratch suelen ser mas dificiles."
    )

    nms = evaluation_result.get("nms_sensitivity") or {}
    conclusion = nms.get("conclusion")
    if conclusion or not diagnostics.empty:
        with st.expander("Detalle tecnico adicional", icon=":material/data_table:"):
            if not diagnostics.empty:
                diagnostics_view = diagnostics.copy()
                diagnostics_view["Clase"] = diagnostics_view["class_name"].map(class_name_map).fillna(diagnostics_view["class_name"])
                st.dataframe(
                    diagnostics_view[["Clase", "annotation_count", "median_bbox_area_pct"]].rename(
                        columns={
                            "annotation_count": "Anotaciones",
                            "median_bbox_area_pct": "Area mediana (%)",
                        }
                    ),
                    hide_index=True,
                )
            if conclusion:
                st.caption(f"Lectura adicional sobre NMS: {conclusion}")


def render_project_story() -> None:
    render_section_header(
        ":material/school: Sistema y alcance",
        "Resumen tecnico del sistema y sus limites operativos.",
    )
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        with st.container(border=True):
            st.markdown("**Base del sistema**")
            st.write("Dataset: CarDD COCO.")
            st.write(f"Modelo: {get_model_display_name()}.")
            st.write("Clases: dent, scratch, crack, glass shatter, lamp broken, tire flat.")
        with st.container(border=True):
            st.markdown("**Flujo**")
            st.write("Imagen -> preprocesamiento -> modelo -> postproceso -> visualizacion.")
    with col_b:
        with st.container(border=True):
            st.markdown("**Alcance y limites**")
            st.write("Funciona mejor con danos visibles y fotos similares al dominio del dataset.")
            st.write("Rayones finos, grietas pequenas y fondos complejos son mas exigentes.")
            st.write("No reemplaza revision profesional ni diagnostico mecanico.")

    latest_summary = st.session_state.get("inspection_summary", {})
    if latest_summary:
        with st.expander("Ultimo caso analizado", icon=":material/history:"):
            render_metric_strip(
                [
                    {
                        "label": "Hallazgos",
                        "value": str(latest_summary.get("count", 0)),
                        "note": "Cantidad detectada en la ultima lectura",
                    },
                    {
                        "label": "Clase dominante",
                        "value": str(latest_summary.get("dominant_class", "Sin detecciones")),
                        "note": "Mayor presencia en la imagen",
                    },
                    {
                        "label": "Severidad maxima",
                        "value": str(latest_summary.get("max_severity", "Sin hallazgos")),
                        "note": "Mayor severidad registrada",
                    },
                    {
                        "label": "Modo",
                        "value": str(st.session_state.get("inspection_mode", "n/d")),
                        "note": "Configuracion aplicada",
                    },
                ]
            )


def main() -> None:
    inject_custom_css()
    ensure_session_defaults()
    evaluation_result = load_evaluation_summary()
    sidebar_config = render_sidebar(evaluation_result)
    render_hero(evaluation_result)
    st.space("small")

    inspection_tab, performance_tab, project_tab = st.tabs(
        [
            ":material/car_crash: Analizar imagen",
            ":material/query_stats: Rendimiento del modelo",
            ":material/school: Sistema y alcance",
        ]
    )

    with inspection_tab:
        render_inspection_tab(sidebar_config)
    with performance_tab:
        render_model_metrics(evaluation_result)
    with project_tab:
        render_project_story()


if __name__ == "__main__":
    main()
