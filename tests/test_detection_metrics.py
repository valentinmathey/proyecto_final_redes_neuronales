from __future__ import annotations

import json
import types
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from prod.detection_dataset import collate_fn
from prod.detection_metrics import collect_detection_report


class _DummyDetectionDataset(Dataset):
    def __init__(self):
        class_names = {
            0: "background",
            1: "dent",
            2: "scratch",
            3: "crack",
            4: "glass shatter",
            5: "lamp broken",
            6: "tire flat",
        }
        self.split = "test"
        self.idx_to_class = class_names
        self.class_to_idx = {name: class_id for class_id, name in class_names.items()}
        self.category_id_to_name = {
            1: "dent",
            2: "scratch",
            3: "crack",
            4: "glass shatter",
            5: "lamp broken",
            6: "tire flat",
        }
        self.annotation_data = {"images": [], "annotations": []}
        self.samples = []

        for class_id in range(1, 7):
            image_id = class_id
            image = torch.zeros((3, 64, 64), dtype=torch.float32)
            image[0, 0, 0] = float(class_id)
            target = {
                "boxes": torch.tensor([[8.0, 8.0, 32.0, 32.0]], dtype=torch.float32),
                "labels": torch.tensor([class_id], dtype=torch.int64),
                "image_id": torch.tensor([image_id], dtype=torch.int64),
                "area": torch.tensor([576.0], dtype=torch.float32),
                "iscrowd": torch.tensor([0], dtype=torch.int64),
            }
            self.samples.append((image, target))
            self.annotation_data["images"].append(
                {
                    "id": image_id,
                    "width": 64,
                    "height": 64,
                    "file_name": f"{image_id}.jpg",
                }
            )
            self.annotation_data["annotations"].append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [8.0, 8.0, 24.0, 24.0],
                    "area": 576.0,
                    "iscrowd": 0,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _DummyDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_heads = types.SimpleNamespace(
            nms_thresh=0.5,
            score_thresh=0.05,
            detections_per_img=100,
        )

    def forward(self, images):
        predictions = []
        for image in images:
            class_id = int(image[0, 0, 0].item())
            predictions.append(
                {
                    "boxes": torch.tensor([[8.0, 8.0, 32.0, 32.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.99], dtype=torch.float32),
                    "labels": torch.tensor([class_id], dtype=torch.int64),
                }
            )
        return predictions


class CollectDetectionReportTest(unittest.TestCase):
    def test_collect_detection_report_builds_pr_curves_and_nms_summary(self):
        dataset = _DummyDetectionDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        model = _DummyDetectionModel()

        report = collect_detection_report(
            model=model,
            dataloader=dataloader,
            device="cpu",
            dataset=dataset,
            idx_to_class=dataset.idx_to_class,
        )

        self.assertIn("summary", report)
        self.assertIn("pr_curves", report)
        self.assertIn("dataset_diagnostics", report)
        self.assertIn("nms_sensitivity", report)
        self.assertGreater(report["summary"]["map"], 0.99)
        self.assertEqual(len(report["pr_curves"]), 6)
        self.assertEqual(len(report["class_metrics"]), 6)

        for curve in report["pr_curves"]:
            self.assertEqual(curve["iou"], 0.5)
            self.assertEqual(curve["area"], "all")
            self.assertEqual(curve["max_dets"], 100)
            self.assertEqual(len(curve["recall"]), len(curve["precision"]))
            self.assertTrue(
                all(left <= right for left, right in zip(curve["recall"], curve["recall"][1:])),
                msg=f"Recall no es monotono para {curve['class_name']}",
            )

        nms_sensitivity = report["nms_sensitivity"]
        self.assertTrue(nms_sensitivity["supported"])
        self.assertEqual(len(nms_sensitivity["results"]), 3)
        self.assertEqual(nms_sensitivity["score_threshold"], 0.05)
        self.assertEqual(nms_sensitivity["detections_per_img"], 100)

        json.dumps(report)

    def test_collect_detection_report_can_skip_nms_sensitivity(self):
        dataset = _DummyDetectionDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        model = _DummyDetectionModel()

        report = collect_detection_report(
            model=model,
            dataloader=dataloader,
            device="cpu",
            dataset=dataset,
            idx_to_class=dataset.idx_to_class,
            include_nms_sensitivity=False,
        )

        self.assertIn("summary", report)
        self.assertEqual(len(report["class_metrics"]), 6)
        self.assertEqual(len(report["pr_curves"]), 6)
        self.assertTrue(report["nms_sensitivity"]["skipped"])
        self.assertEqual(report["nms_sensitivity"]["results"], [])


if __name__ == "__main__":
    unittest.main()
