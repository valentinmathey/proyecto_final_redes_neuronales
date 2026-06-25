from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import pandas as pd

from utils import (
    archive_canonical_detection_test_result,
    build_detection_test_results_comparison_df,
    detection_test_result_paths,
    export_detection_test_report_html,
    export_model_comparison_html,
    export_results_comparison_html,
    is_detection_test_report_complete,
    save_detection_test_result_artifacts,
)


class ExportResultsComparisonHtmlTest(unittest.TestCase):
    def test_export_results_comparison_html_generates_expected_table(self):
        # Simula un resumen mínimo de corridas para ejercitar la exportación HTML.
        results_df = pd.DataFrame(
            [
                {
                    "name": "fasterrcnn_head_only",
                    "model_name": "fasterrcnn",
                    "best_epoch": 3,
                    "best_map": 0.41,
                    "best_map_50": 0.67,
                    "best_val_loss": 1.23,
                },
                {
                    "name": "retinanet_partial_backbone_fixed_size",
                    "model_name": "retinanet",
                    "best_epoch": 4,
                    "best_map": 0.38,
                    "best_map_50": 0.61,
                    "best_val_loss": 1.35,
                },
            ]
        )

        output_path = Path(__file__).resolve().parent / "artifacts" / "results_comparison.html"

        # Genera el reporte visual que luego consumen los notebooks.
        generated_path = export_results_comparison_html(
            results_df,
            output_path,
            title="Comparación mock de resultados",
        )

        self.assertEqual(generated_path, output_path)
        self.assertTrue(output_path.exists())

        # Verifica que el archivo tenga la estructura básica esperada.
        html_content = output_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("<table", html_content)
        self.assertIn("Comparación mock de resultados", html_content)
        self.assertIn("fasterrcnn_head_only", html_content)
        self.assertIn("retinanet", html_content)
        self.assertIn("mAP", html_content)

    def test_export_detection_test_report_html_renders_sections_and_pr_charts(self):
        report = {
            "run_id": "run_001",
            "best_experiment": "fasterrcnn_mobilenet_v3_large_partial_backbone",
            "checkpoint_path": "dev/experiments/model_best.pth",
            "summary": {
                "map": 0.4466,
                "map_50": 0.6479,
                "map_75": 0.4777,
                "mar_100": 0.5713,
            },
            "class_metrics": [
                {"class_id": 4, "class_name": "glass shatter", "map_per_class": 0.8056, "mar_100_per_class": 0.8380},
                {"class_id": 3, "class_name": "crack", "map_per_class": 0.0866, "mar_100_per_class": 0.2986},
            ],
            "pr_curves": [
                {
                    "class_id": 4,
                    "class_name": "glass shatter",
                    "iou": 0.5,
                    "area": "all",
                    "max_dets": 100,
                    "recall": [0.0, 0.5, 1.0],
                    "precision": [1.0, 0.95, 0.9],
                    "ap_50": 0.95,
                },
                {
                    "class_id": 3,
                    "class_name": "crack",
                    "iou": 0.5,
                    "area": "all",
                    "max_dets": 100,
                    "recall": [0.0, 0.5, 1.0],
                    "precision": [0.4, 0.3, 0.2],
                    "ap_50": 0.3,
                },
            ],
            "dataset_diagnostics": {
                "split": "test",
                "num_images": 374,
                "num_annotations": 785,
                "per_class": [
                    {
                        "class_id": 4,
                        "class_name": "glass shatter",
                        "annotation_count": 71,
                        "image_count": 71,
                        "median_bbox_area": 375222.0,
                        "median_bbox_area_pct": 56.255,
                        "mean_instances_per_image": 1.0,
                        "max_instances_per_image": 1,
                    }
                ],
            },
            "nms_sensitivity": {
                "supported": True,
                "baseline_nms_threshold": 0.5,
                "score_threshold": 0.05,
                "detections_per_img": 100,
                "conclusion": "El barrido de NMS apenas mueve el mAP global.",
                "results": [
                    {"nms_threshold": 0.3, "map": 0.4609, "map_50": 0.6725, "map_75": 0.5130, "mar_100": 0.57},
                    {"nms_threshold": 0.5, "map": 0.4620, "map_50": 0.6810, "map_75": 0.5106, "mar_100": 0.58},
                ],
            },
        }

        output_path = Path(__file__).resolve().parent / "artifacts" / "best_test_result_report.html"

        generated_path = export_detection_test_report_html(
            report,
            output_path,
            title="Reporte final mock",
        )

        self.assertEqual(generated_path, output_path)
        self.assertTrue(output_path.exists())

        html_content = output_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Reporte final mock", html_content)
        self.assertIn("Sensibilidad a NMS", html_content)
        self.assertIn("Curvas precision-recall por clase", html_content)
        self.assertIn("glass shatter", html_content)
        self.assertIn("crack", html_content)
        self.assertIn("pr-chart", html_content)

    def test_export_model_comparison_html_renders_runs_transforms_pr_and_selected_nms(self):
        comparison_runs = [
            {
                "run_id": "run_001",
                "created_at": "2026-06-12T12:00:00",
                "name": "fasterrcnn_baseline",
                "optimizer_name": "sgd",
                "trainable_backbone_layers": 5,
                "best_epoch": 2,
                "checkpoint_path": "dev/experiments/run_001_best.pth",
                "training_duration_seconds": 120.0,
                "config": {
                    "model_name": "fasterrcnn",
                    "optimizer_name": "sgd",
                    "num_epochs": 2,
                    "trainable_backbone_layers": 2,
                    "resize": False,
                    "image_size": None,
                },
                "history": [
                    {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "map": 0.1, "map_50": 0.2},
                    {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9, "map": 0.3, "map_50": 0.4},
                ],
                "validation_report": {
                    "summary": {"map": 0.3, "map_50": 0.4, "map_75": 0.25, "mar_100": 0.5},
                    "class_metrics": [
                        {"class_id": 1, "class_name": "dent", "map_per_class": 0.2, "mar_100_per_class": 0.4},
                    ],
                    "pr_curves": [
                        {
                            "class_id": 1,
                            "class_name": "dent",
                            "iou": 0.5,
                            "area": "all",
                            "max_dets": 100,
                            "recall": [0.0, 1.0],
                            "precision": [1.0, 0.5],
                            "ap_50": 0.75,
                        }
                    ],
                    "nms_sensitivity": {"skipped": True, "results": []},
                },
            },
            {
                "run_id": "run_002",
                "created_at": "2026-06-12T13:00:00",
                "name": "oversample_crop",
                "optimizer_name": "sgd",
                "best_epoch": 3,
                "checkpoint_path": "dev/experiments/run_002_best.pth",
                "training_duration_seconds": 180.0,
                "config": {
                    "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
                    "optimizer_name": "sgd",
                    "num_epochs": 3,
                    "trainable_backbone_layers": 2,
                    "resize": True,
                    "image_size": [640, 640],
                    "use_object_crop": True,
                    "object_crop_probability": 0.5,
                    "oversample_target_factor": 2.5,
                    "target_classes": ["dent", "scratch"],
                },
                "history": [
                    {"epoch": 1, "train_loss": 1.2, "val_loss": 1.0, "map": 0.2, "map_50": 0.3},
                    {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "map": 0.35, "map_50": 0.45},
                    {"epoch": 3, "train_loss": 0.7, "val_loss": 0.75, "map": 0.38, "map_50": 0.5},
                ],
                "validation_report": {
                    "summary": {"map": 0.38, "map_50": 0.5, "map_75": 0.3, "mar_100": 0.55},
                    "class_metrics": [
                        {"class_id": 2, "class_name": "scratch", "map_per_class": 0.25, "mar_100_per_class": 0.45},
                    ],
                    "pr_curves": [
                        {
                            "class_id": 2,
                            "class_name": "scratch",
                            "iou": 0.5,
                            "area": "all",
                            "max_dets": 100,
                            "recall": [0.0, 1.0],
                            "precision": [0.9, 0.4],
                            "ap_50": 0.65,
                        }
                    ],
                    "nms_sensitivity": {
                        "supported": True,
                        "baseline_nms_threshold": 0.5,
                        "score_threshold": 0.05,
                        "detections_per_img": 100,
                        "conclusion": "El barrido de NMS apenas mueve el mAP global.",
                        "results": [
                            {"nms_threshold": 0.3, "map": 0.37, "map_50": 0.49, "map_75": 0.29, "mar_100": 0.53},
                            {"nms_threshold": 0.5, "map": 0.38, "map_50": 0.50, "map_75": 0.30, "mar_100": 0.55},
                            {"nms_threshold": 0.7, "map": 0.37, "map_50": 0.48, "map_75": 0.31, "mar_100": 0.57},
                        ],
                    },
                },
            },
        ]

        output_path = Path(__file__).resolve().parent / "artifacts" / "model_comparison.html"
        generated_path = export_model_comparison_html(
            comparison_runs,
            output_path,
            selected_run_id="run_002",
            selection_reason="mock",
            comparison_split="val",
        )

        self.assertEqual(generated_path, output_path)
        html_content = output_path.read_text(encoding="utf-8")
        self.assertIn("Prueba 1", html_content)
        self.assertIn("Prueba 2", html_content)
        self.assertLess(html_content.index("Prueba 1"), html_content.index("Prueba 2"))
        self.assertIn("carousel-controls", html_content)
        self.assertIn("prev-slide", html_content)
        self.assertIn("next-slide", html_content)
        self.assertIn("slide-counter", html_content)
        self.assertIn('data-slide-index="0"', html_content)
        self.assertIn('data-slide-index="1"', html_content)
        self.assertIn("run-content-layout", html_content)
        self.assertIn("run-tables-column", html_content)
        self.assertIn("run-charts-column", html_content)
        self.assertIn("chart-stack", html_content)
        self.assertIn("class-charts-section", html_content)
        self.assertIn("table-pair-grid", html_content)
        self.assertIn("run-tables-column .pr-grid", html_content)
        self.assertIn("repeat(3, minmax(170px, 1fr))", html_content)
        self.assertIn("height: 142px", html_content)
        self.assertLess(
            html_content.index('class="carousel-shell"'),
            html_content.index('data-slide-index="0"'),
        )
        self.assertLess(html_content.index("<h1>"), html_content.index('class="carousel-shell"'))
        self.assertLess(html_content.index('class="carousel-shell"'), html_content.index("</section>"))
        self.assertIn("Tabla resumen", html_content)
        self.assertIn("Nombre experimento", html_content)
        self.assertIn("best_mAP", html_content)
        self.assertIn("mAP@50", html_content)
        self.assertIn("fasterrcnn_baseline", html_content)
        self.assertIn("oversample_crop", html_content)
        self.assertEqual(len(re.findall(r'<section class="run-card[^"]*"', html_content)), 2)
        self.assertEqual(len(re.findall(r'<section class="run-card[^"]*is-active[^"]*"', html_content)), 1)
        self.assertIn("Faster R-CNN ResNet50 FPN · SGD · 2 capas del backbone entrenables", html_content)
        self.assertIn(
            "Faster R-CNN MobileNet V3 Large FPN · SGD · 2 capas del backbone entrenables · "
            "Object crop + Oversampling dent/scratch",
            html_content,
        )
        self.assertIn("Dataset y transforms", html_content)
        self.assertIn("RandomObjectCropDetection", html_content)
        self.assertIn("mAP por clase en validacion", html_content)
        self.assertIn("Abolladura (dent)", html_content)
        self.assertIn("Rayón (scratch)", html_content)
        self.assertIn("Curvas precision-recall por clase", html_content)
        self.assertIn("history-chart", html_content)
        self.assertIn("Sensibilidad a NMS del modelo seleccionado", html_content)
        self.assertIn("NMS threshold", html_content)
        self.assertNotIn("El barrido de NMS apenas mueve el mAP global.", html_content)
        self.assertNotIn("Run ID", html_content)
        self.assertNotIn("Fecha", html_content)
        self.assertNotIn("mAR@100", html_content)
        self.assertNotIn("mar_100", html_content)
        self.assertNotIn("Checkpoint", html_content)
        self.assertNotIn("Split de comparacion", html_content)
        self.assertNotIn("Comparacion en val", html_content)
        self.assertNotIn("Resumen global", html_content)
        self.assertNotIn("<td>Dataset</td>", html_content)
        self.assertNotIn("<td>CarDD_COCO</td>", html_content)
        self.assertNotIn("Transform validacion", html_content)
        self.assertNotIn("Transform entrenamiento base", html_content)
        self.assertNotIn("<td>ToTensorDetection</td>", html_content)
        self.assertNotIn("ToTensorDetection + RandomHorizontalFlipDetection", html_content)
        self.assertNotIn("RandomHorizontalFlipDetection(p=0.5)", html_content)
        self.assertNotIn("Flip horizontal", html_content)

    def test_is_detection_test_report_complete_detects_missing_sections(self):
        complete_report = {
            "run_id": "run_001",
            "checkpoint_path": "dev/experiments/model_best.pth",
            "summary": {"map": 0.1, "map_50": 0.2, "map_75": 0.3, "mar_100": 0.4},
            "class_metrics": [],
            "pr_curves": [],
            "dataset_diagnostics": {},
            "nms_sensitivity": {},
        }
        incomplete_report = {
            "run_id": "run_001",
            "checkpoint_path": "dev/experiments/model_best.pth",
            "summary": {"map": 0.1},
            "class_metrics": [],
        }

        self.assertTrue(is_detection_test_report_complete(complete_report))
        self.assertFalse(is_detection_test_report_complete(incomplete_report))

    def test_save_detection_test_result_artifacts_preserves_canonical_by_default(self):
        output_dir = Path(__file__).resolve().parent / "artifacts" / "test_results"
        canonical_path = Path(__file__).resolve().parent / "artifacts" / "canonical_best_test_result.json"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text('{"run_id": "baseline"}', encoding="utf-8")
        report = {
            "run_id": "new_run",
            "best_experiment": "oversample_crop",
            "checkpoint_path": "dev/experiments/new_run_best.pth",
            "test_map": 0.5,
            "test_map_50": 0.7,
            "summary": {"map": 0.5, "map_50": 0.7, "map_75": 0.4, "mar_100": 0.6},
            "class_metrics": [
                {"class_id": 1, "class_name": "dent", "map_per_class": 0.3, "mar_100_per_class": 0.5},
                {"class_id": 2, "class_name": "scratch", "map_per_class": 0.31, "mar_100_per_class": 0.52},
                {"class_id": 3, "class_name": "crack", "map_per_class": 0.09, "mar_100_per_class": 0.3},
            ],
            "pr_curves": [],
            "dataset_diagnostics": {"split": "test", "num_images": 1, "num_annotations": 1, "per_class": []},
            "nms_sensitivity": {"results": [], "conclusion": "mock"},
        }

        paths = save_detection_test_result_artifacts(
            report,
            output_dir,
            canonical_json_path=canonical_path,
            update_canonical=False,
        )

        self.assertEqual(paths, detection_test_result_paths("new_run", output_dir))
        self.assertTrue(paths["json"].exists())
        self.assertTrue(paths["html"].exists())
        self.assertEqual(canonical_path.read_text(encoding="utf-8"), '{"run_id": "baseline"}')

    def test_archive_and_compare_detection_test_results(self):
        artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        output_dir = artifacts_dir / "comparison_test_results"
        manifest_path = artifacts_dir / "comparison_runs_manifest.jsonl"
        canonical_path = artifacts_dir / "comparison_best_test_result.json"
        canonical_report = {
            "run_id": "baseline_run",
            "best_experiment": "baseline",
            "checkpoint_path": "dev/experiments/baseline_best.pth",
            "test_map": 0.44,
            "test_map_50": 0.64,
            "summary": {"map": 0.44, "map_50": 0.64, "map_75": 0.47, "mar_100": 0.57},
            "class_metrics": [
                {"class_name": "dent", "map_per_class": 0.24},
                {"class_name": "scratch", "map_per_class": 0.23},
                {"class_name": "crack", "map_per_class": 0.08},
            ],
            "pr_curves": [],
            "dataset_diagnostics": {},
            "nms_sensitivity": {},
        }
        canonical_path.write_text(json.dumps(canonical_report), encoding="utf-8")
        manifest_path.write_text(
            json.dumps({"run_id": "baseline_run", "name": "baseline"}) + "\n",
            encoding="utf-8",
        )

        archived_path = archive_canonical_detection_test_result(canonical_path, output_dir)
        comparison_df = build_detection_test_results_comparison_df(manifest_path, output_dir)

        self.assertTrue(archived_path.exists())
        self.assertEqual(len(comparison_df), 1)
        self.assertEqual(comparison_df.iloc[0]["run_id"], "baseline_run")
        self.assertEqual(comparison_df.iloc[0]["dent_map"], 0.24)
        self.assertIn("result_html", comparison_df.columns)


if __name__ == "__main__":
    unittest.main()
