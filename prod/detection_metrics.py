from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from statistics import median

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


_DEFAULT_AREA_LABELS = ["all", "small", "medium", "large"]


# Crea la métrica mAP intentando usar `faster_coco_eval` cuando está disponible.
def create_map_metric(
    class_metrics: bool = True,
    extended_summary: bool = False,
    iou_thresholds=None,
    rec_thresholds=None,
    max_detection_thresholds=None,
):
    metric_kwargs = {
        "box_format": "xyxy",
        "iou_type": "bbox",
        "class_metrics": class_metrics,
        "extended_summary": extended_summary,
    }

    if iou_thresholds is not None:
        metric_kwargs["iou_thresholds"] = iou_thresholds
    if rec_thresholds is not None:
        metric_kwargs["rec_thresholds"] = rec_thresholds
    if max_detection_thresholds is not None:
        metric_kwargs["max_detection_thresholds"] = max_detection_thresholds

    try:
        return MeanAveragePrecision(backend="faster_coco_eval", **metric_kwargs)
    except TypeError:
        return MeanAveragePrecision(**metric_kwargs)


# Lleva todas las tensores de un diccionario al CPU antes de pasarlos a torchmetrics.
def _move_dict_to_cpu(d: dict) -> dict:
    return {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in d.items()
    }


def _tensor_to_python(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


# Convierte los tensores del resultado final a tipos Python serializables.
def summarize_map_results(results: dict) -> dict:
    return {k: _tensor_to_python(v) for k, v in results.items()}


def _collect_predictions_and_targets(model, dataloader, device, max_batches=None):
    predictions_all = []
    targets_all = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = [image.to(device) for image in images]
            predictions = model(images)

            predictions_all.extend(_move_dict_to_cpu(prediction) for prediction in predictions)
            targets_all.extend(_move_dict_to_cpu(target) for target in targets)

    if was_training:
        model.train()

    return predictions_all, targets_all


def _compute_map_results(
    predictions,
    targets,
    class_metrics: bool = True,
    extended_summary: bool = False,
    iou_thresholds=None,
    rec_thresholds=None,
    max_detection_thresholds=None,
):
    metric = create_map_metric(
        class_metrics=class_metrics,
        extended_summary=extended_summary,
        iou_thresholds=iou_thresholds,
        rec_thresholds=rec_thresholds,
        max_detection_thresholds=max_detection_thresholds,
    )
    metric.update(predictions, targets)
    raw_results = metric.compute()
    return metric, raw_results, summarize_map_results(raw_results)


def _build_class_metrics_rows(map_results: dict, idx_to_class: dict | None = None) -> list[dict]:
    classes = map_results.get("classes") or []
    map_per_class = map_results.get("map_per_class") or []
    mar_per_class = map_results.get("mar_100_per_class") or []

    rows = []
    for index, class_id in enumerate(classes):
        class_id = int(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": (idx_to_class or {}).get(class_id, f"class_{class_id}"),
                "map_per_class": round(float(map_per_class[index]), 4),
                "mar_100_per_class": round(float(mar_per_class[index]), 4) if index < len(mar_per_class) else None,
            }
        )

    return sorted(rows, key=lambda row: row["map_per_class"], reverse=True)


def _mean_of_valid_precision(precision_values: list[float]) -> float | None:
    valid_values = [float(value) for value in precision_values if value >= 0]
    if not valid_values:
        return None
    return float(sum(valid_values) / len(valid_values))


def _resolve_iou_index(metric, requested_iou: float) -> int:
    iou_thresholds = [float(value) for value in metric.iou_thresholds]
    for index, iou_threshold in enumerate(iou_thresholds):
        if abs(iou_threshold - requested_iou) < 1e-9:
            return index
    raise ValueError(f"IoU {requested_iou} no esta disponible en la metrica: {iou_thresholds}")


def _resolve_max_dets_index(metric, requested_max_dets: int) -> int:
    max_detection_thresholds = [int(value) for value in metric.max_detection_thresholds]
    for index, max_dets in enumerate(max_detection_thresholds):
        if max_dets == requested_max_dets:
            return index
    raise ValueError(
        f"max_dets={requested_max_dets} no esta disponible en la metrica: {max_detection_thresholds}"
    )


def _build_pr_curves(
    metric,
    raw_results: dict,
    summarized_results: dict,
    idx_to_class: dict | None = None,
    pr_iou: float = 0.5,
    pr_area: str = "all",
    pr_max_dets: int = 100,
):
    classes = summarized_results.get("classes") or []
    if not classes:
        return []

    area_labels = list(_DEFAULT_AREA_LABELS)
    if pr_area not in area_labels:
        raise ValueError(f"Area no soportada: {pr_area}. Esperadas: {', '.join(area_labels)}")

    iou_index = _resolve_iou_index(metric, pr_iou)
    area_index = area_labels.index(pr_area)
    max_dets_index = _resolve_max_dets_index(metric, pr_max_dets)
    recall_thresholds = [float(value) for value in metric.rec_thresholds]
    precision_tensor = raw_results["precision"].detach().cpu()

    curves = []
    for class_index, class_id in enumerate(classes):
        class_id = int(class_id)
        precision_values = precision_tensor[iou_index, :, class_index, area_index, max_dets_index].tolist()
        valid_pairs = [
            (float(recall_value), float(precision_value))
            for recall_value, precision_value in zip(recall_thresholds, precision_values)
            if precision_value >= 0
        ]
        recall_values = [recall_value for recall_value, _ in valid_pairs]
        precision_values = [precision_value for _, precision_value in valid_pairs]

        curves.append(
            {
                "class_id": class_id,
                "class_name": (idx_to_class or {}).get(class_id, f"class_{class_id}"),
                "iou": pr_iou,
                "area": pr_area,
                "max_dets": pr_max_dets,
                "recall": recall_values,
                "precision": precision_values,
                "ap_50": _mean_of_valid_precision(precision_values),
            }
        )

    return curves


def _build_dataset_diagnostics(dataset) -> dict:
    if dataset is None:
        return {"split": None, "num_images": 0, "num_annotations": 0, "per_class": []}

    annotations = dataset.annotation_data.get("annotations", [])
    images_by_id = {
        int(image["id"]): image
        for image in dataset.annotation_data.get("images", [])
    }
    counts_by_class = defaultdict(int)
    image_ids_by_class = defaultdict(set)
    bbox_area_by_class = defaultdict(list)
    bbox_area_ratio_by_class = defaultdict(list)
    per_image_count_by_class = defaultdict(lambda: defaultdict(int))

    for annotation in annotations:
        category_name = dataset.category_id_to_name[int(annotation["category_id"])]
        class_id = int(dataset.class_to_idx[category_name])
        image_id = int(annotation["image_id"])
        image_info = images_by_id.get(image_id, {})
        image_width = max(int(image_info.get("width", 1)), 1)
        image_height = max(int(image_info.get("height", 1)), 1)
        image_area = float(image_width * image_height)
        bbox = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox_width = float(bbox[2]) if len(bbox) > 2 else 0.0
        bbox_height = float(bbox[3]) if len(bbox) > 3 else 0.0
        bbox_area = float(annotation.get("area", bbox_width * bbox_height))

        counts_by_class[class_id] += 1
        image_ids_by_class[class_id].add(image_id)
        bbox_area_by_class[class_id].append(bbox_area)
        bbox_area_ratio_by_class[class_id].append((bbox_area / image_area) if image_area > 0 else 0.0)
        per_image_count_by_class[class_id][image_id] += 1

    per_class_rows = []
    for class_id, class_name in sorted(dataset.idx_to_class.items()):
        if class_id == 0:
            continue

        instance_counts = list(per_image_count_by_class[class_id].values())
        per_class_rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "annotation_count": int(counts_by_class[class_id]),
                "image_count": int(len(image_ids_by_class[class_id])),
                "median_bbox_area": round(float(median(bbox_area_by_class[class_id])), 1)
                if bbox_area_by_class[class_id]
                else None,
                "median_bbox_area_pct": round(float(median(bbox_area_ratio_by_class[class_id]) * 100.0), 3)
                if bbox_area_ratio_by_class[class_id]
                else None,
                "mean_instances_per_image": round(
                    float(sum(instance_counts) / len(instance_counts)),
                    2,
                )
                if instance_counts
                else 0.0,
                "max_instances_per_image": int(max(instance_counts)) if instance_counts else 0,
            }
        )

    return {
        "split": getattr(dataset, "split", None),
        "num_images": int(len(dataset)),
        "num_annotations": int(len(annotations)),
        "per_class": per_class_rows,
    }


def _get_attr_by_path(obj, attr_path):
    current = obj
    for attr_name in attr_path:
        current = getattr(current, attr_name)
    return current


def _set_attr_by_path(obj, attr_path, value):
    parent = obj
    for attr_name in attr_path[:-1]:
        parent = getattr(parent, attr_name)
    setattr(parent, attr_path[-1], value)


def _resolve_postprocess_attr_paths(model):
    attr_candidates = {
        "nms_threshold": [
            ("roi_heads", "nms_thresh"),
            ("nms_thresh",),
        ],
        "score_threshold": [
            ("roi_heads", "score_thresh"),
            ("score_thresh",),
        ],
        "detections_per_img": [
            ("roi_heads", "detections_per_img"),
            ("detections_per_img",),
        ],
    }

    resolved = {}
    for key, candidates in attr_candidates.items():
        resolved[key] = None
        for attr_path in candidates:
            try:
                _get_attr_by_path(model, attr_path)
                resolved[key] = attr_path
                break
            except AttributeError:
                continue

    return resolved


def _restore_postprocess_attrs(model, attr_paths, original_values):
    for key, attr_path in attr_paths.items():
        if attr_path is None or key not in original_values:
            continue
        _set_attr_by_path(model, attr_path, original_values[key])


def _build_nms_sensitivity(
    model,
    dataloader,
    device,
    idx_to_class: dict | None = None,
    nms_thresholds=(0.3, 0.5, 0.7),
    max_batches=None,
):
    attr_paths = _resolve_postprocess_attr_paths(model)
    nms_attr_path = attr_paths.get("nms_threshold")
    if nms_attr_path is None:
        return {
            "supported": False,
            "thresholds": [float(value) for value in nms_thresholds],
            "results": [],
            "conclusion": "El modelo no expone un atributo configurable de NMS para automatizar el barrido.",
        }

    original_values = {}
    for key, attr_path in attr_paths.items():
        if attr_path is not None:
            original_values[key] = deepcopy(_get_attr_by_path(model, attr_path))

    baseline_threshold = float(original_values["nms_threshold"])
    score_threshold = (
        float(original_values["score_threshold"])
        if "score_threshold" in original_values
        else None
    )
    detections_per_img = (
        int(original_values["detections_per_img"])
        if "detections_per_img" in original_values
        else None
    )

    results = []
    try:
        for nms_threshold in nms_thresholds:
            _set_attr_by_path(model, nms_attr_path, float(nms_threshold))
            metric_results = evaluate_map(
                model=model,
                dataloader=dataloader,
                device=device,
                class_metrics=True,
                max_batches=max_batches,
            )
            results.append(
                {
                    "nms_threshold": float(nms_threshold),
                    "map": metric_results.get("map"),
                    "map_50": metric_results.get("map_50"),
                    "map_75": metric_results.get("map_75"),
                    "mar_100": metric_results.get("mar_100"),
                    "class_metrics": _build_class_metrics_rows(metric_results, idx_to_class=idx_to_class),
                }
            )
    finally:
        _restore_postprocess_attrs(model, attr_paths, original_values)

    map_values = [float(row["map"]) for row in results if row.get("map") is not None]
    map_range = (max(map_values) - min(map_values)) if map_values else None

    if map_range is not None and map_range <= 0.02:
        conclusion = (
            "El barrido de NMS apenas mueve el mAP global, por lo que NMS no parece ser la causa principal "
            "de la brecha entre clases."
        )
    else:
        conclusion = (
            "El barrido de NMS cambia el mAP de forma no trivial; conviene revisar el postprocesado junto con "
            "otros factores del dataset."
        )

    return {
        "supported": True,
        "baseline_nms_threshold": baseline_threshold,
        "score_threshold": score_threshold,
        "detections_per_img": detections_per_img,
        "thresholds": [float(value) for value in nms_thresholds],
        "results": results,
        "map_range": map_range,
        "conclusion": conclusion,
    }


def collect_detection_report(
    model,
    dataloader,
    device,
    idx_to_class: dict | None = None,
    dataset=None,
    nms_thresholds=(0.3, 0.5, 0.7),
    include_nms_sensitivity: bool = True,
    pr_iou: float = 0.5,
    pr_area: str = "all",
    pr_max_dets: int = 100,
    max_batches=None,
):
    if dataset is None:
        dataset = getattr(dataloader, "dataset", None)
    if idx_to_class is None and dataset is not None:
        idx_to_class = getattr(dataset, "idx_to_class", None)

    predictions, targets = _collect_predictions_and_targets(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )
    extended_metric, raw_results, summarized_results = _compute_map_results(
        predictions=predictions,
        targets=targets,
        class_metrics=True,
        extended_summary=True,
    )

    summary = {
        "map": summarized_results.get("map"),
        "map_50": summarized_results.get("map_50"),
        "map_75": summarized_results.get("map_75"),
        "mar_100": summarized_results.get("mar_100"),
    }

    nms_sensitivity = (
        _build_nms_sensitivity(
            model=model,
            dataloader=dataloader,
            device=device,
            idx_to_class=idx_to_class,
            nms_thresholds=nms_thresholds,
            max_batches=max_batches,
        )
        if include_nms_sensitivity
        else {
            "supported": False,
            "skipped": True,
            "thresholds": [float(value) for value in nms_thresholds],
            "results": [],
            "conclusion": "Barrido de NMS omitido para esta evaluacion.",
        }
    )

    return {
        "summary": summary,
        "class_metrics": _build_class_metrics_rows(summarized_results, idx_to_class=idx_to_class),
        "pr_curves": _build_pr_curves(
            metric=extended_metric,
            raw_results=raw_results,
            summarized_results=summarized_results,
            idx_to_class=idx_to_class,
            pr_iou=pr_iou,
            pr_area=pr_area,
            pr_max_dets=pr_max_dets,
        ),
        "dataset_diagnostics": _build_dataset_diagnostics(dataset),
        "nms_sensitivity": nms_sensitivity,
    }


def evaluate_map(
    model,
    dataloader,
    device,
    class_metrics: bool = True,
    max_batches=None,
):
    predictions, targets = _collect_predictions_and_targets(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
    )
    _, _, summarized_results = _compute_map_results(
        predictions=predictions,
        targets=targets,
        class_metrics=class_metrics,
        extended_summary=False,
    )
    return summarized_results


# Extrae las métricas principales usadas en las tablas comparativas del proyecto.
def extract_main_map_metrics(results):
    return {
        "map": results.get("map"),
        "map_50": results.get("map_50"),
        "map_75": results.get("map_75"),
        "mar_100": results.get("mar_100"),
    }
