from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from prod.detection_training import load_checkpoint


class LoadCheckpointTest(unittest.TestCase):
    def test_load_checkpoint_reads_project_checkpoint_format(self):
        # Crea un modelo origen y otro destino para verificar la restauración completa.
        source_model = torch.nn.Linear(4, 2)
        target_model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(source_model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "checkpoint.pth"
            # Replica el formato de checkpoint amplio usado por este proyecto.
            checkpoint_payload = {
                "experiment_name": "test_experiment",
                "epoch": 3,
                "model_state_dict": source_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": [{"epoch": 1, "map": 0.1}],
                "config": {"model_name": "mock"},
            }
            torch.save(checkpoint_payload, checkpoint_path)

            # Carga el checkpoint y valida tanto metadatos como pesos del modelo.
            loaded_checkpoint = load_checkpoint(target_model, checkpoint_path, device="cpu")

            self.assertEqual(loaded_checkpoint["experiment_name"], "test_experiment")
            self.assertEqual(loaded_checkpoint["epoch"], 3)

            for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
                self.assertTrue(torch.equal(source_param, target_param))


if __name__ == "__main__":
    unittest.main()
