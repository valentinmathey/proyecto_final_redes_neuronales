from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gdown
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from prod.detection_models import create_model_from_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKPOINT_PATH = PROJECT_ROOT / "dev" / "modelo.pth"
TEST_RESULT_PATH = PROJECT_ROOT / "dev" / "best_test_result.json"
CACHED_CHECKPOINT_PATH = Path(tempfile.gettempdir()) / "cardd_modelo.pth"
MIN_CHECKPOINT_BYTES = 50 * 1024 * 1024

NUM_CLASSES = 7
MODEL_NAME = "Modelo final CarDD"
MODEL_DRIVE_ID_ENV = "MODEL_GDRIVE_ID"
MODEL_DRIVE_URL_ENV = "MODEL_GDRIVE_URL"

MODEL_DISPLAY_NAMES = {
    "fasterrcnn": "Faster R-CNN ResNet50 FPN",
    "fasterrcnn_mobilenet_v3_large_fpn": "Faster R-CNN MobileNet V3 Large FPN",
    "fasterrcnn_mobilenet_v3_large_320_fpn": "Faster R-CNN MobileNet V3 Large 320 FPN",
    "retinanet": "RetinaNet ResNet50 FPN",
    "fcos": "FCOS ResNet50 FPN",
}

CLASS_NAMES: dict[int, str] = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}

CLASS_LABELS_ES: dict[int, str] = {
    1: "Abolladura",
    2: "Rayon",
    3: "Grieta",
    4: "Vidrio roto",
    5: "Faro roto",
    6: "Rueda pinchada",
}

CLASS_COLORS: dict[int, str] = {
    1: "#df6b57",
    2: "#1f9d9b",
    3: "#3974b8",
    4: "#ec8f2f",
    5: "#51a36f",
    6: "#b94b5d",
}

REPAIR_COST_RANGE: dict[int, tuple[int, int]] = {
    1: (200, 800),
    2: (100, 500),
    3: (50, 200),
    4: (300, 1500),
    5: (150, 600),
    6: (50, 200),
}

INSURANCE_COVERAGE: dict[str, set[int]] = {
    "Responsabilidad civil": set(),
    "Terceros completo": {4, 5, 6},
    "Todo riesgo": {1, 2, 3, 4, 5, 6},
}

DEFAULT_LABOR_RATE_USD = 139
LABOR_RATE_RANGE_USD = (120, 159)

SHOP_PROFILE_MULTIPLIERS: dict[str, float] = {
    "Economico": 0.92,
    "Estandar": 1.00,
    "Premium": 1.15,
}

PARTS_POLICY_MULTIPLIERS: dict[str, float] = {
    "Aftermarket": 0.95,
    "Mixto": 1.00,
    "OEM": 1.15,
}

IMPACT_SOURCES: dict[str, dict[str, str]] = {
    "aaa_labor": {
        "source": "AAA 2026 labor rate guide",
        "url": "https://www.aaa.com/autorepair/articles/average-mechanic-labor-rate-repair-costs-in-your-state-2026",
        "note": "Base laboral usada por defecto: USD 139/h sobre un rango publicado de USD 120 a 159 por hora.",
    },
    "dent_pdr": {
        "source": "Dent Express AZ - paintless dent repair",
        "url": "https://dentexpressaz.com/how-much-does-paintless-dent-repair-cost/",
        "note": "Referencia para abolladuras: PDR pequeno USD 150-200, mediano USD 200-300 y grande USD 300+.",
    },
    "scratch_prices": {
        "source": "Apex Auto Pros - scratch repair guide",
        "url": "https://apexautopros.com/car-scratch-repair-cost-a-complete-2025-price-guide/",
        "note": "Rayones profesionales: buff USD 75-150, retoque/repintado parcial USD 150-400, profundo USD 400-800+.",
    },
    "glass_prices": {
        "source": "Caliber Auto Glass - windshield replacement without insurance",
        "url": "https://www.caliber.com/services/auto-glass/mobile-auto-glass-repair/what-is-the-cost-of-windshield-replacement-without-insurance",
        "note": "Vidrio: reemplazo tipico USD 210-500; con ADAS puede subir hasta USD 1500.",
    },
    "safelite_glass": {
        "source": "Safelite cost overview",
        "url": "https://www.safelite.com/auto-glass-repair-replacement-cost",
        "note": "Safelite aclara que el costo depende de dano, vehiculo, seguro y ubicacion; con cobertura integral puede ser tan bajo como USD 0 de bolsillo.",
    },
    "lamp_prices": {
        "source": "RepairPal - headlight bulb replacement",
        "url": "https://repairpal.com/estimator/headlight-bulb-replacement-cost",
        "note": "Faro: cambio de lampara promedio USD 172-204; conjuntos completos pueden costar mas.",
    },
    "tire_repair": {
        "source": "Kelley Blue Book - tire repair costs",
        "url": "https://www.kbb.com/service-repair-guide/tire-repair-costs/",
        "note": "Pinchadura reparable: promedio USD 54-64; si el dano es severo puede requerir reemplazo.",
    },
    "tire_replacement": {
        "source": "Kelley Blue Book - tire replacement guidance",
        "url": "https://www.kbb.com/car-advice/do-my-tires-need-to-be-replaced/",
        "note": "Neumatico nuevo de marca conocida: cerca de USD 200 por unidad con instalacion.",
    },
}

SEVERITY_BASE: dict[int, int] = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 2,
    6: 3,
}


def _scaled_range(
    base_low: float,
    base_high: float,
    shop_multiplier: float = 1.0,
    parts_multiplier: float = 1.0,
) -> tuple[int, int]:
    scaled_low = int(round(base_low * shop_multiplier * parts_multiplier))
    scaled_high = int(round(base_high * shop_multiplier * parts_multiplier))
    return scaled_low, max(scaled_high, scaled_low)


def _bodywork_range(
    labor_rate: int,
    labor_hours: tuple[float, float],
    material_cost: tuple[int, int],
    shop_multiplier: float,
) -> tuple[int, int]:
    low = labor_rate * labor_hours[0] * shop_multiplier + material_cost[0]
    high = labor_rate * labor_hours[1] * shop_multiplier + material_cost[1]
    return int(round(low)), int(round(high))


def _split_range(
    total_low: int,
    total_high: int,
    labor_share: float,
    parts_share: float,
) -> dict[str, int]:
    labor_low = int(round(total_low * labor_share))
    labor_high = int(round(total_high * labor_share))
    parts_low = int(round(total_low * parts_share))
    parts_high = int(round(total_high * parts_share))
    materials_low = max(total_low - labor_low - parts_low, 0)
    materials_high = max(total_high - labor_high - parts_high, 0)
    return {
        "labor_min": labor_low,
        "labor_max": labor_high,
        "parts_min": parts_low,
        "parts_max": parts_high,
        "materials_min": materials_low,
        "materials_max": materials_high,
    }


def estimate_detection_impact(
    detection: dict,
    labor_rate: int = DEFAULT_LABOR_RATE_USD,
    shop_profile: str = "Estandar",
    parts_policy: str = "Mixto",
) -> dict:
    label_id = int(detection["label"])
    severity_label = str(detection.get("severity_label", "Leve"))
    area_pct = float(detection.get("area_pct", 0.0))
    shop_multiplier = SHOP_PROFILE_MULTIPLIERS.get(shop_profile, 1.0)
    parts_multiplier = PARTS_POLICY_MULTIPLIERS.get(parts_policy, 1.0)

    benchmark_low = 0
    benchmark_high = 0
    repair_path = ""
    methodology_note = ""
    source_ids: list[str] = ["aaa_labor"]
    component_split = {
        "labor_min": 0,
        "labor_max": 0,
        "parts_min": 0,
        "parts_max": 0,
        "materials_min": 0,
        "materials_max": 0,
    }

    if label_id == 1:  # dent
        source_ids.append("dent_pdr")
        dent_rules = {
            "Leve": ((1.0, 1.5), (20, 45), "Paintless dent repair o correccion localizada"),
            "Moderada": ((1.6, 2.8), (45, 110), "Desabollado con preparacion y retoque"),
            "Alta": ((3.0, 5.0), (120, 240), "Chapa, correccion de panel y repintado parcial"),
            "Critica": ((4.5, 7.0), (180, 360), "Reparacion pesada de panel y repintado completo"),
        }
        labor_hours, materials, repair_path = dent_rules.get(severity_label, dent_rules["Moderada"])
        benchmark_low, benchmark_high = _bodywork_range(labor_rate, labor_hours, materials, shop_multiplier)
        methodology_note = "Abolladura modelada con referencia de PDR y mano de obra base AAA segun severidad."
        component_split = {
            "labor_min": int(round(labor_rate * labor_hours[0] * shop_multiplier)),
            "labor_max": int(round(labor_rate * labor_hours[1] * shop_multiplier)),
            "parts_min": 0,
            "parts_max": 0,
            "materials_min": int(materials[0]),
            "materials_max": int(materials[1]),
        }

    elif label_id == 2:  # scratch
        source_ids.append("scratch_prices")
        scratch_rules = {
            "Leve": ((0.4, 0.8), (15, 50), "Pulido o buff superficial"),
            "Moderada": ((1.0, 2.0), (45, 110), "Retoque de pintura o reparacion localizada"),
            "Alta": ((2.0, 4.0), (110, 260), "Preparacion de panel y repintado parcial"),
            "Critica": ((3.5, 5.5), (180, 420), "Reparacion profunda y repintado completo de panel"),
        }
        labor_hours, materials, repair_path = scratch_rules.get(severity_label, scratch_rules["Moderada"])
        benchmark_low, benchmark_high = _bodywork_range(labor_rate, labor_hours, materials, shop_multiplier)
        methodology_note = "Rayon calibrado con guia publica de reparacion superficial, media y profunda."
        component_split = {
            "labor_min": int(round(labor_rate * labor_hours[0] * shop_multiplier)),
            "labor_max": int(round(labor_rate * labor_hours[1] * shop_multiplier)),
            "parts_min": 0,
            "parts_max": 0,
            "materials_min": int(materials[0]),
            "materials_max": int(materials[1]),
        }

    elif label_id == 3:  # crack
        source_ids.append("scratch_prices")
        crack_rules = {
            "Leve": ((0.8, 1.6), (35, 90), "Sellado fino y terminacion localizada"),
            "Moderada": ((1.6, 3.0), (80, 170), "Sellado o relleno con refinish localizado"),
            "Alta": ((3.0, 4.8), (150, 300), "Reparacion de panel con trabajo de superficie y pintura"),
            "Critica": ((4.5, 6.8), (220, 460), "Reparacion compleja y repintado amplio de panel"),
        }
        labor_hours, materials, repair_path = crack_rules.get(severity_label, crack_rules["Moderada"])
        benchmark_low, benchmark_high = _bodywork_range(labor_rate, labor_hours, materials, shop_multiplier)
        methodology_note = (
            "Grieta modelada como dano de superficie/panel. No existe un benchmark publico exacto para la clase CarDD, "
            "por eso se usa mano de obra AAA y rangos de refinish similares a rayones profundos."
        )
        component_split = {
            "labor_min": int(round(labor_rate * labor_hours[0] * shop_multiplier)),
            "labor_max": int(round(labor_rate * labor_hours[1] * shop_multiplier)),
            "parts_min": 0,
            "parts_max": 0,
            "materials_min": int(materials[0]),
            "materials_max": int(materials[1]),
        }

    elif label_id == 4:  # glass shatter
        source_ids.extend(["glass_prices", "safelite_glass"])
        if severity_label in {"Alta", "Critica"} or area_pct >= 6.0:
            base_low, base_high = 500, 1500
        else:
            base_low, base_high = 210, 500
        benchmark_low, benchmark_high = _scaled_range(base_low, base_high, shop_multiplier, parts_multiplier)
        repair_path = "Reemplazo de vidrio y posible recalibracion"
        methodology_note = "Vidrio roto calibrado con rangos publicos de reemplazo estandar y vehiculos con ADAS."
        component_split = _split_range(benchmark_low, benchmark_high, labor_share=0.18, parts_share=0.72)

    elif label_id == 5:  # lamp broken
        source_ids.append("lamp_prices")
        if severity_label == "Leve":
            base_low, base_high = 172, 204
            repair_path = "Reemplazo de lampara o reparacion menor"
        elif severity_label == "Moderada":
            base_low, base_high = 220, 380
            repair_path = "Reemplazo de lampara o carcasa parcial"
        elif severity_label == "Alta":
            base_low, base_high = 320, 560
            repair_path = "Reemplazo de conjunto de faro"
        else:
            base_low, base_high = 450, 820
            repair_path = "Reemplazo de conjunto de faro y ajustes asociados"
        benchmark_low, benchmark_high = _scaled_range(base_low, base_high, shop_multiplier, parts_multiplier)
        methodology_note = "Faro roto parte del benchmark publico de cambio de lampara y lo expande cuando la severidad sugiere dano de conjunto."
        component_split = _split_range(benchmark_low, benchmark_high, labor_share=0.28, parts_share=0.62)

    elif label_id == 6:  # tire flat
        source_ids.extend(["tire_repair", "tire_replacement"])
        if severity_label == "Leve" and area_pct < 3.0:
            base_low, base_high = 0, 64
            repair_path = "Parche o reparacion de pinchadura si la zona es reparable"
        elif severity_label == "Moderada":
            base_low, base_high = 180, 250
            repair_path = "Reemplazo de un neumatico"
        elif severity_label == "Alta":
            base_low, base_high = 200, 320
            repair_path = "Reemplazo de neumatico y balanceo"
        else:
            base_low, base_high = 240, 380
            repair_path = "Reemplazo de neumatico con revision adicional"
        benchmark_low, benchmark_high = _scaled_range(base_low, base_high, shop_multiplier, parts_multiplier)
        methodology_note = "Rueda pinchada distingue entre pinchadura reparable y reemplazo completo de neumatico."
        if severity_label == "Leve" and area_pct < 3.0:
            component_split = _split_range(benchmark_low, benchmark_high, labor_share=0.55, parts_share=0.10)
        else:
            component_split = _split_range(benchmark_low, benchmark_high, labor_share=0.18, parts_share=0.72)

    else:
        benchmark_low, benchmark_high = _scaled_range(150, 450, shop_multiplier, parts_multiplier)
        repair_path = "Inspeccion manual y presupuesto especifico"
        methodology_note = "Clase no mapeada a benchmark especifico."
        component_split = _split_range(benchmark_low, benchmark_high, labor_share=0.45, parts_share=0.25)

    return {
        "cost_min": benchmark_low,
        "cost_max": benchmark_high,
        "repair_path": repair_path,
        "methodology_note": methodology_note,
        "source_ids": source_ids,
        "labor_rate_used": labor_rate,
        "shop_profile": shop_profile,
        "parts_policy": parts_policy,
        **component_split,
    }


def default_cost_range(label_id: int, severity_label: str, area_pct: float) -> tuple[int, int]:
    impact = estimate_detection_impact(
        {
            "label": label_id,
            "severity_label": severity_label,
            "area_pct": area_pct,
        }
    )
    return int(impact["cost_min"]), int(impact["cost_max"])


def estimate_impact_report(
    detections: list[dict],
    labor_rate: int = DEFAULT_LABOR_RATE_USD,
    shop_profile: str = "Estandar",
    parts_policy: str = "Mixto",
    selected_policy: str = "Todo riesgo",
    deductible: int = 500,
) -> dict:
    rows = []
    total_min = 0
    total_max = 0
    covered_mid = 0.0
    uncovered_mid = 0.0
    source_ids: list[str] = []

    for det in detections:
        impact = estimate_detection_impact(
            det,
            labor_rate=labor_rate,
            shop_profile=shop_profile,
            parts_policy=parts_policy,
        )
        row = {
            "Dano": det.get("class_name_es", "-"),
            "Clase tecnica": det.get("class_name", "-"),
            "Severidad": det.get("severity_label", "-"),
            "Score detector": float(det.get("score", 0.0)),
            "Area relativa": float(det.get("area_pct", 0.0)) / 100.0,
            "Ruta de reparacion": impact["repair_path"],
            "Costo minimo": int(impact["cost_min"]),
            "Costo maximo": int(impact["cost_max"]),
            "Mano de obra minimo": int(impact["labor_min"]),
            "Mano de obra maximo": int(impact["labor_max"]),
            "Materiales minimo": int(impact["materials_min"]),
            "Materiales maximo": int(impact["materials_max"]),
            "Repuestos minimo": int(impact["parts_min"]),
            "Repuestos maximo": int(impact["parts_max"]),
            "Metodo": impact["methodology_note"],
            "Fuentes": ", ".join(impact["source_ids"]),
            "label_id": int(det["label"]),
        }
        rows.append(row)
        total_min += int(impact["cost_min"])
        total_max += int(impact["cost_max"])
        midpoint = (int(impact["cost_min"]) + int(impact["cost_max"])) / 2
        if int(det["label"]) in INSURANCE_COVERAGE.get(selected_policy, set()):
            covered_mid += midpoint
        else:
            uncovered_mid += midpoint
        for source_id in impact["source_ids"]:
            if source_id not in source_ids:
                source_ids.append(source_id)

    midpoint_total = (total_min + total_max) / 2 if rows else 0.0
    deductible_component = min(covered_mid, float(deductible)) if covered_mid > 0 else 0.0
    out_of_pocket_mid = uncovered_mid + deductible_component

    return {
        "rows": rows,
        "summary": {
            "total_min": total_min,
            "total_max": total_max,
            "midpoint_total": midpoint_total,
            "covered_mid": covered_mid,
            "uncovered_mid": uncovered_mid,
            "out_of_pocket_mid": out_of_pocket_mid,
            "selected_policy": selected_policy,
            "deductible": deductible,
            "labor_rate": labor_rate,
            "shop_profile": shop_profile,
            "parts_policy": parts_policy,
        },
        "source_rows": [IMPACT_SOURCES[source_id] | {"id": source_id} for source_id in source_ids if source_id in IMPACT_SOURCES],
    }


def build_impact_scenarios(
    detections: list[dict],
    labor_rate: int = DEFAULT_LABOR_RATE_USD,
    parts_policy: str = "Mixto",
) -> list[dict]:
    scenarios = []
    for profile_name in SHOP_PROFILE_MULTIPLIERS:
        scenario = estimate_impact_report(
            detections,
            labor_rate=labor_rate,
            shop_profile=profile_name,
            parts_policy=parts_policy,
        )
        scenarios.append(
            {
                "Perfil de taller": profile_name,
                "Costo minimo": scenario["summary"]["total_min"],
                "Costo maximo": scenario["summary"]["total_max"],
                "Costo medio": scenario["summary"]["midpoint_total"],
            }
        )
    return scenarios


def _streamlit_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value).strip() if value else None


def _model_source() -> tuple[str | None, str | None]:
    drive_id = _streamlit_secret(MODEL_DRIVE_ID_ENV) or os.getenv(MODEL_DRIVE_ID_ENV)
    drive_url = _streamlit_secret(MODEL_DRIVE_URL_ENV) or os.getenv(MODEL_DRIVE_URL_ENV)
    return (
        str(drive_id).strip() if drive_id else None,
        str(drive_url).strip() if drive_url else None,
    )


def _is_valid_checkpoint(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_CHECKPOINT_BYTES


def ensure_checkpoint() -> Path:
    if _is_valid_checkpoint(LOCAL_CHECKPOINT_PATH):
        return LOCAL_CHECKPOINT_PATH

    if _is_valid_checkpoint(CACHED_CHECKPOINT_PATH):
        return CACHED_CHECKPOINT_PATH

    drive_id, drive_url = _model_source()
    if not drive_id and not drive_url:
        raise RuntimeError(
            "No se encontro el checkpoint local ni una fuente de descarga. "
            f"Configura {MODEL_DRIVE_ID_ENV} o {MODEL_DRIVE_URL_ENV} con el archivo modelo.pth publicado en Google Drive."
        )

    CACHED_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        if drive_id:
            gdown.download(id=drive_id, output=str(CACHED_CHECKPOINT_PATH), quiet=False)
        else:
            gdown.download(url=drive_url, output=str(CACHED_CHECKPOINT_PATH), quiet=False, fuzzy=True)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo descargar modelo.pth desde Google Drive. "
            "Verifica que el link sea publico o que no haya limite de cuota."
        ) from exc

    if not _is_valid_checkpoint(CACHED_CHECKPOINT_PATH):
        raise RuntimeError(
            "La descarga del checkpoint termino, pero el archivo no parece valido. "
            "Revisa permisos del archivo en Drive y vuelve a intentar."
        )

    return CACHED_CHECKPOINT_PATH


def _torch_load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"El checkpoint no tiene el formato esperado: {path}")
    return checkpoint


def _resolve_project_path(path_value: str | os.PathLike) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _normalize_model_config(config: dict) -> dict:
    normalized = dict(config)
    if "model_name" not in normalized:
        raise RuntimeError("La metadata del modelo no incluye 'model_name'.")
    if "num_classes" not in normalized:
        normalized["num_classes"] = NUM_CLASSES

    image_size = normalized.get("image_size")
    if image_size is not None:
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            raise RuntimeError(f"image_size invalido en metadata: {image_size!r}")
        normalized["image_size"] = (int(image_size[0]), int(image_size[1]))

    normalized["resize"] = bool(normalized.get("resize", False))
    return normalized


@st.cache_data(show_spinner=False)
def load_model_metadata(test_result_path: str | None = None) -> dict:
    result_path = Path(test_result_path) if test_result_path else TEST_RESULT_PATH
    if not result_path.exists():
        raise RuntimeError(
            "No se encontro dev/best_test_result.json. "
            "La UI necesita ese archivo para saber con que metadata reconstruir el modelo."
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint_value = result.get("checkpoint_path")
    if not checkpoint_value:
        raise RuntimeError(
            "dev/best_test_result.json no incluye 'checkpoint_path'. "
            "La UI lo necesita para leer la metadata del modelo ganador."
        )

    metadata_checkpoint_path = _resolve_project_path(checkpoint_value)
    if not metadata_checkpoint_path.exists():
        raise RuntimeError(
            "El checkpoint declarado en dev/best_test_result.json no existe: "
            f"{metadata_checkpoint_path}"
        )

    metadata_checkpoint = _torch_load_checkpoint(metadata_checkpoint_path)
    config = metadata_checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(
            "El checkpoint declarado en dev/best_test_result.json no incluye un 'config' valido."
        )

    model_config = _normalize_model_config(config)
    model_name = model_config.get("model_name", "")
    return {
        "run_id": result.get("run_id"),
        "best_experiment": result.get("best_experiment") or metadata_checkpoint.get("experiment_name"),
        "checkpoint_path": str(metadata_checkpoint_path),
        "config": model_config,
        "display_name": MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
    }


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str | None = None):
    resolved_path = Path(checkpoint_path) if checkpoint_path else ensure_checkpoint()
    metadata = load_model_metadata()
    model_config = metadata["config"]
    model = create_model_from_config(model_config, pretrained=False)

    checkpoint = _torch_load_checkpoint(resolved_path)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError(f"El checkpoint de pesos no incluye 'model_state_dict': {resolved_path}")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "El archivo dev/modelo.pth no coincide con la metadata del modelo ganador "
            "registrada en dev/best_test_result.json. Actualiza dev/modelo.pth con los pesos "
            "del checkpoint ganador o regenera best_test_result.json para que apunte al modelo correcto."
        ) from exc

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.cardd_config = model_config
    model.cardd_metadata = metadata
    return model


@st.cache_data(show_spinner=False)
def load_evaluation_summary(path: str | None = None) -> dict:
    result_path = Path(path) if path else TEST_RESULT_PATH
    if not result_path.exists():
        return {}
    return json.loads(result_path.read_text(encoding="utf-8"))


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    return to_tensor(pil_image.convert("RGB"))


def _prepare_inference_image(model, pil_image: Image.Image) -> tuple[Image.Image, float, float]:
    config = getattr(model, "cardd_config", {}) or {}
    image_size = config.get("image_size")
    if not config.get("resize") or image_size is None:
        return pil_image, 1.0, 1.0

    target_height, target_width = image_size
    original_width, original_height = pil_image.size
    if original_width == target_width and original_height == target_height:
        return pil_image, 1.0, 1.0

    resized_image = pil_image.resize((target_width, target_height), Image.BILINEAR)
    return (
        resized_image,
        original_width / target_width,
        original_height / target_height,
    )


def _scale_box_to_original(
    box: torch.Tensor,
    scale_x: float,
    scale_y: float,
    image_size: tuple[int, int],
) -> list[int]:
    image_width, image_height = image_size
    x0, y0, x1, y1 = box.tolist()
    scaled = [
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    ]
    scaled[0] = max(0, min(scaled[0], image_width))
    scaled[2] = max(0, min(scaled[2], image_width))
    scaled[1] = max(0, min(scaled[1], image_height))
    scaled[3] = max(0, min(scaled[3], image_height))
    return scaled


def _enrich_detection(det: dict, image_size: tuple[int, int]) -> dict:
    image_width, image_height = image_size
    x0, y0, x1, y1 = det["box"]
    area_px = max(x1 - x0, 0) * max(y1 - y0, 0)
    area_pct = (area_px / max(image_width * image_height, 1)) * 100.0
    severity_score, severity_label = estimate_severity(det["label"], area_pct)
    cost_min, cost_max = default_cost_range(int(det["label"]), severity_label, area_pct)
    return {
        **det,
        "class_name_es": CLASS_LABELS_ES.get(det["label"], det["class_name"]),
        "area_px": int(area_px),
        "area_pct": area_pct,
        "severity_score": severity_score,
        "severity_label": severity_label,
        "cost_min": cost_min,
        "cost_max": cost_max,
    }


def run_inference(
    model,
    pil_image: Image.Image,
    score_threshold: float = 0.4,
) -> list[dict]:
    inference_image, scale_x, scale_y = _prepare_inference_image(model, pil_image)
    tensor = preprocess_image(inference_image)

    with torch.inference_mode():
        outputs = model([tensor])

    prediction = outputs[0]
    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        score_value = float(score.item())
        if score_value < score_threshold:
            continue
        label_id = int(label.item())
        det = {
            "box": _scale_box_to_original(box, scale_x, scale_y, pil_image.size),
            "label": label_id,
            "class_name": CLASS_NAMES.get(label_id, f"clase {label_id}"),
            "score": round(score_value, 4),
        }
        detections.append(_enrich_detection(det, pil_image.size))

    return detections


def apply_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    if len(detections) < 2:
        return detections

    import torchvision.ops as ops

    boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
    keep = ops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep.tolist()]


def run_inference_tiled(
    model,
    pil_image: Image.Image,
    score_threshold: float = 0.4,
    overlap: float = 0.1,
    grid_size: int = 2,
) -> list[dict]:
    width, height = pil_image.size
    step_x = width / grid_size
    step_y = height / grid_size
    tile_width = int(step_x * (1 + overlap))
    tile_height = int(step_y * (1 + overlap))

    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            ox = int(step_x * col)
            oy = int(step_y * row)
            tiles.append((
                pil_image.crop((ox, oy, min(width, ox + tile_width), min(height, oy + tile_height))),
                ox,
                oy,
            ))

    detections = []
    for tile_image, ox, oy in tiles:
        for det in run_inference(model, tile_image, score_threshold=score_threshold):
            x0, y0, x1, y1 = det["box"]
            shifted = {
                **det,
                "box": [x0 + ox, y0 + oy, x1 + ox, y1 + oy],
            }
            detections.append(_enrich_detection(shifted, pil_image.size))

    return apply_nms(detections, iou_threshold=0.5)


def verify_detections(
    model,
    pil_image: Image.Image,
    candidates: list[dict],
    score_threshold: float = 0.4,
    padding: float = 0.5,
) -> list[dict]:
    if not candidates:
        return []

    width, height = pil_image.size
    confirmed = []
    for det in candidates:
        x0, y0, x1, y1 = det["box"]
        box_width = x1 - x0
        box_height = y1 - y0
        crop_x0 = max(0, int(x0 - box_width * padding))
        crop_y0 = max(0, int(y0 - box_height * padding))
        crop_x1 = min(width, int(x1 + box_width * padding))
        crop_y1 = min(height, int(y1 + box_height * padding))
        crop = pil_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        crop_detections = run_inference(model, crop, score_threshold=score_threshold)
        if any(int(d.get("label", -1)) == int(det["label"]) for d in crop_detections):
            confirmed.append(det)

    return confirmed


def run_inference_two_pass(
    model,
    pil_image: Image.Image,
    score_threshold: float = 0.4,
    padding: float = 0.5,
) -> list[dict]:
    candidates = run_inference(model, pil_image, score_threshold=score_threshold)
    return verify_detections(model, pil_image, candidates, score_threshold=score_threshold, padding=padding)


def estimate_severity(label_id: int, area_pct: float) -> tuple[int, str]:
    score = SEVERITY_BASE.get(int(label_id), 1)
    if area_pct >= 8.0:
        score += 2
    elif area_pct >= 2.0:
        score += 1

    score = min(score, 5)
    if score <= 2:
        return score, "Leve"
    if score == 3:
        return score, "Moderada"
    if score == 4:
        return score, "Alta"
    return score, "Critica"


def summarize_detections(detections: list[dict]) -> dict:
    if not detections:
        return {
            "count": 0,
            "max_severity": "Sin hallazgos",
            "dominant_class": "Sin detecciones",
            "cost_min": 0,
            "cost_max": 0,
            "priority_index": 0,
        }

    class_counts: dict[str, int] = {}
    for det in detections:
        class_counts[det["class_name_es"]] = class_counts.get(det["class_name_es"], 0) + 1

    max_severity_det = max(detections, key=lambda item: item["severity_score"])
    cost_min = sum(int(det["cost_min"]) for det in detections)
    cost_max = sum(int(det["cost_max"]) for det in detections)
    priority_index = min(
        100,
        int(round((max_severity_det["severity_score"] / 5) * 65 + min(len(detections), 5) * 7)),
    )

    return {
        "count": len(detections),
        "max_severity": max_severity_det["severity_label"],
        "dominant_class": max(class_counts, key=class_counts.get),
        "cost_min": cost_min,
        "cost_max": cost_max,
        "priority_index": priority_index,
    }


def build_detection_table(detections: list[dict]) -> list[dict]:
    rows = []
    for det in detections:
        rows.append(
            {
                "Dano": det["class_name_es"],
                "Clase tecnica": det["class_name"],
                "Score detector": f"{det['score']:.1%}",
                "Area imagen": f"{det['area_pct']:.1f}%",
                "Severidad": det["severity_label"],
                "Costo estimado": f"USD {det['cost_min']:,} - {det['cost_max']:,}",
                "Caja [x0,y0,x1,y1]": str(det["box"]),
            }
        )
    return rows


def build_coverage_table(detections: list[dict]) -> list[dict]:
    detected_labels = {int(det["label"]) for det in detections}
    rows = []
    for policy, covered_labels in INSURANCE_COVERAGE.items():
        covered = [CLASS_LABELS_ES[label] for label in detected_labels if label in covered_labels]
        not_covered = [CLASS_LABELS_ES[label] for label in detected_labels if label not in covered_labels]
        rows.append(
            {
                "Cobertura": policy,
                "Cubre": ", ".join(covered) if covered else "Ninguno",
                "No cubre": ", ".join(not_covered) if not_covered else "-",
            }
        )
    return rows


def draw_predictions(pil_image: Image.Image, detections: list[dict]) -> Image.Image:
    result = pil_image.copy().convert("RGB")
    if not detections:
        return result

    draw = ImageDraw.Draw(result, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", size=max(15, pil_image.width // 70))
    except OSError:
        font = ImageFont.load_default()

    img_width, img_height = pil_image.size
    label_padding = 5
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        color = CLASS_COLORS.get(det["label"], "#ffffff")
        label_text = f"{det['class_name_es']} {det['score']:.0%} | {det['severity_label']}"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(3, pil_image.width // 320))
        draw.rectangle([x0, y0, x1, y1], fill=color + "20")

        text_size = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_size[2] - text_size[0]
        text_anchor_x = max(0, min(x0, img_width - text_width - 2 * label_padding))
        text_anchor_y = max(0, y0 - 26)

        text_bbox = draw.textbbox((text_anchor_x, text_anchor_y), label_text, font=font)
        bg_bbox = [
            text_bbox[0] - label_padding,
            text_bbox[1] - label_padding,
            text_bbox[2] + label_padding,
            text_bbox[3] + label_padding,
        ]
        draw.rectangle(bg_bbox, fill=color)
        draw.text((text_bbox[0], text_bbox[1]), label_text, fill="#101820", font=font)

    return result
