from __future__ import annotations

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

CHECKPOINT_PATH = "dev/modelo.pth"

CLASS_NAMES: dict[int, str] = {
    1: "dent",
    2: "scratch",
    3: "crack",
    4: "glass shatter",
    5: "lamp broken",
    6: "tire flat",
}

CLASS_COLORS: dict[int, str] = {
    1: "#FF6B6B",
    2: "#4ECDC4",
    3: "#45B7D1",
    4: "#FFA07A",
    5: "#98D8C8",
    6: "#DDA0DD",
}

NUM_CLASSES = 7  # background + 6 clases de daño


@st.cache_resource
def load_model(checkpoint_path: str = CHECKPOINT_PATH):
    # weights=None y trainable_backbone_layers=0 evitan cualquier descarga de pesos preentrenados
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    # Idéntico a ToTensorDetection usado en val/test: convierte PIL → [3,H,W] en [0,1]
    return to_tensor(pil_image.convert("RGB"))


def run_inference(
    model,
    pil_image: Image.Image,
    score_threshold: float = 0.4,
) -> list[dict]:
    tensor = preprocess_image(pil_image)

    with torch.no_grad():
        outputs = model([tensor])

    prediction = outputs[0]
    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()
    scores = prediction["scores"].cpu()

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score.item() < score_threshold:
            continue
        label_id = label.item()
        detections.append({
            "box": [round(v) for v in box.tolist()],
            "label": label_id,
            "class_name": CLASS_NAMES.get(label_id, f"clase {label_id}"),
            "score": round(score.item(), 3),
        })

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
) -> list[dict]:
    w, h = pil_image.size
    half_w = int(w * (0.5 + overlap / 2))
    half_h = int(h * (0.5 + overlap / 2))
    offset_x = int(w * (0.5 - overlap / 2))
    offset_y = int(h * (0.5 - overlap / 2))

    tiles = [
        (pil_image.crop((0,        0,        half_w, half_h)), 0,        0),
        (pil_image.crop((offset_x, 0,        w,      half_h)), offset_x, 0),
        (pil_image.crop((0,        offset_y, half_w, h)),      0,        offset_y),
        (pil_image.crop((offset_x, offset_y, w,      h)),      offset_x, offset_y),
    ]

    all_detections = []
    for tile_img, ox, oy in tiles:
        tile_dets = run_inference(model, tile_img, score_threshold=score_threshold)
        for det in tile_dets:
            x0, y0, x1, y1 = det["box"]
            all_detections.append({**det, "box": [x0 + ox, y0 + oy, x1 + ox, y1 + oy]})

    return apply_nms(all_detections)


def run_inference_two_pass(
    model,
    pil_image: Image.Image,
    score_threshold: float = 0.4,
    padding: float = 0.5,
) -> list[dict]:
    # Primera pasada al mismo umbral — solo verificamos las detecciones que ya superan el threshold
    candidates = run_inference(model, pil_image, score_threshold=score_threshold)
    if not candidates:
        return []

    w, h = pil_image.size
    confirmed = []
    for det in candidates:
        x0, y0, x1, y1 = det["box"]
        bw, bh = x1 - x0, y1 - y0
        cx0 = max(0, int(x0 - bw * padding))
        cy0 = max(0, int(y0 - bh * padding))
        cx1 = min(w, int(x1 + bw * padding))
        cy1 = min(h, int(y1 + bh * padding))
        crop = pil_image.crop((cx0, cy0, cx1, cy1))
        if run_inference(model, crop, score_threshold=score_threshold):
            confirmed.append(det)

    return confirmed


def draw_predictions(pil_image: Image.Image, detections: list[dict]) -> Image.Image:
    result = pil_image.copy().convert("RGB")
    if not detections:
        return result

    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        color = CLASS_COLORS.get(det["label"], "#FFFFFF")
        label_text = f"{det['class_name']} {det['score']:.0%}"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        text_bbox = draw.textbbox((x0, y0 - 20), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x0, y0 - 20), label_text, fill="black", font=font)

    return result
