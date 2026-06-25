from __future__ import annotations

import random
import unittest

import torch
from PIL import Image

from prod.detection_dataset import (
    RandomObjectCropDetection,
    build_oversampling_sampler,
    build_oversampling_weights,
    sample_contains_target_classes,
)


class _MiniDataset:
    def __init__(self):
        self.image_ids = [10, 20, 30]
        self.class_to_idx = {
            "background": 0,
            "dent": 1,
            "scratch": 2,
            "crack": 3,
        }
        self.category_id_to_name = {
            1: "dent",
            2: "scratch",
            3: "crack",
        }
        self.annotations_by_image = {
            10: [{"category_id": 1}],
            20: [{"category_id": 3}],
            30: [{"category_id": 2}],
        }

    def __len__(self):
        return len(self.image_ids)


class RandomObjectCropDetectionTest(unittest.TestCase):
    def test_random_object_crop_adjusts_boxes_and_keeps_valid_targets(self):
        random.seed(7)
        image = Image.new("RGB", (100, 100), color="white")
        target = {
            "boxes": torch.tensor(
                [
                    [40.0, 40.0, 60.0, 60.0],
                    [0.0, 0.0, 10.0, 10.0],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
            "area": torch.tensor([400.0, 100.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0, 0], dtype=torch.int64),
            "image_id": torch.tensor([1], dtype=torch.int64),
        }
        transform = RandomObjectCropDetection(
            p=1.0,
            target_class_ids={1},
            crop_scale_range=(2.0, 2.0),
            min_crop_size=(40, 40),
            min_visible_fraction=0.5,
            center_jitter=0.0,
        )

        cropped_image, cropped_target = transform(image, target)

        self.assertEqual(cropped_image.size, (40, 40))
        self.assertEqual(cropped_target["boxes"].shape, (1, 4))
        self.assertEqual(cropped_target["labels"].tolist(), [1])
        self.assertEqual(cropped_target["iscrowd"].tolist(), [0])
        self.assertEqual(cropped_target["image_id"].tolist(), [1])
        self.assertTrue(torch.all(cropped_target["boxes"] >= 0))
        self.assertTrue(torch.all(cropped_target["boxes"][:, [0, 2]] <= cropped_image.width))
        self.assertTrue(torch.all(cropped_target["boxes"][:, [1, 3]] <= cropped_image.height))
        self.assertGreater(float(cropped_target["area"][0]), 0.0)

    def test_oversampling_helpers_prioritize_target_classes(self):
        dataset = _MiniDataset()

        self.assertTrue(sample_contains_target_classes(dataset, 0, ["dent", "scratch"]))
        self.assertFalse(sample_contains_target_classes(dataset, 1, ["dent", "scratch"]))
        self.assertTrue(sample_contains_target_classes(dataset, 2, ["dent", "scratch"]))

        weights = build_oversampling_weights(
            dataset,
            target_classes=["dent", "scratch"],
            target_factor=2.5,
        )
        self.assertEqual(weights.tolist(), [2.5, 1.0, 2.5])

        sampler = build_oversampling_sampler(
            dataset,
            target_classes=["dent", "scratch"],
            target_factor=2.5,
        )
        self.assertEqual(sampler.num_samples, len(dataset))
        self.assertTrue(sampler.replacement)


if __name__ == "__main__":
    unittest.main()
