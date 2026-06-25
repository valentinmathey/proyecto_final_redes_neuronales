from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

import prod.utils as app_utils


class _DummyDetectionModel:
    def __init__(self):
        self.cardd_config = {
            "resize": True,
            "image_size": (640, 640),
        }
        self.input_shape = None

    def __call__(self, images):
        self.input_shape = tuple(images[0].shape)
        return [
            {
                "boxes": torch.tensor([[160.0, 320.0, 640.0, 640.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
        ]


class AppInferenceUtilsTest(unittest.TestCase):
    def tearDown(self):
        app_utils.load_model_metadata.clear()
        app_utils.load_model.clear()

    def test_run_inference_resizes_to_training_size_and_scales_boxes_back(self):
        model = _DummyDetectionModel()
        image = Image.new("RGB", (1280, 960), color="white")

        detections = app_utils.run_inference(model, image, score_threshold=0.4)

        self.assertEqual(model.input_shape, (3, 640, 640))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["box"], [320, 480, 1280, 960])
        self.assertEqual(detections[0]["area_px"], 460800)
        self.assertAlmostEqual(detections[0]["area_pct"], 37.5)

    def test_load_model_metadata_reads_config_from_checkpoint_declared_in_best_result(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            metadata_checkpoint_path = project_root / "dev" / "experiments" / "winner.pth"
            metadata_checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "experiment_name": "fasterrcnn_partial_backbone",
                    "config": {
                        "model_name": "fasterrcnn",
                        "num_classes": 7,
                        "resize": True,
                        "image_size": (640, 640),
                    },
                },
                metadata_checkpoint_path,
            )

            best_result_path = project_root / "dev" / "best_test_result.json"
            best_result_path.write_text(
                json.dumps(
                    {
                        "run_id": "run_001",
                        "best_experiment": "fasterrcnn_partial_backbone",
                        "checkpoint_path": "dev/experiments/winner.pth",
                    }
                ),
                encoding="utf-8",
            )

            original_project_root = app_utils.PROJECT_ROOT
            app_utils.PROJECT_ROOT = project_root
            try:
                metadata = app_utils.load_model_metadata(str(best_result_path))
            finally:
                app_utils.PROJECT_ROOT = original_project_root

        self.assertEqual(metadata["run_id"], "run_001")
        self.assertEqual(metadata["checkpoint_path"], str(metadata_checkpoint_path.resolve()))
        self.assertEqual(metadata["config"]["model_name"], "fasterrcnn")
        self.assertEqual(metadata["config"]["image_size"], (640, 640))
        self.assertTrue(metadata["config"]["resize"])

    def test_load_model_reports_clear_error_when_modelo_pth_does_not_match_metadata(self):
        source_model = torch.nn.Linear(3, 3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "modelo.pth"
            torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

            target_model = torch.nn.Linear(2, 2)
            fake_metadata = {
                "config": {
                    "model_name": "fasterrcnn",
                    "num_classes": 7,
                    "resize": True,
                    "image_size": (640, 640),
                }
            }

            with mock.patch.object(app_utils, "load_model_metadata", return_value=fake_metadata):
                with mock.patch.object(app_utils, "create_model_from_config", return_value=target_model):
                    with self.assertRaisesRegex(RuntimeError, "dev/modelo.pth no coincide"):
                        app_utils.load_model(str(checkpoint_path))


if __name__ == "__main__":
    unittest.main()
