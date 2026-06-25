from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils import resolve_portable_path, to_portable_path


class PortablePathUtilsTest(unittest.TestCase):
    def test_to_portable_path_returns_repo_relative_posix_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            checkpoint_path = project_root / "dev" / "experiments" / "model_best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")

            # Debe persistir la ruta relativa al repo y no una absoluta de la máquina.
            portable_path = to_portable_path(checkpoint_path, base_dir=project_root)

            self.assertEqual(portable_path, "dev/experiments/model_best.pth")

    def test_resolve_portable_path_resolves_repo_relative_path_from_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            checkpoint_path = project_root / "dev" / "experiments" / "model_best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")

            # Debe reconstruir la ruta real a partir de la versión portable guardada.
            resolved_path = resolve_portable_path(
                "dev/experiments/model_best.pth",
                base_dir=project_root,
            )

            self.assertEqual(resolved_path, checkpoint_path.resolve())

    def test_resolve_portable_path_recovers_legacy_windows_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            artifacts_dir = project_root / "dev" / "experiments"
            checkpoint_path = artifacts_dir / "model_best.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"checkpoint")

            # También debe rescatar manifests viejos con rutas absolutas de Windows.
            resolved_path = resolve_portable_path(
                r"C:\Users\Usuario\Documents\Proyecto Final Joaco\dev\experiments\model_best.pth",
                base_dir=project_root,
                fallback_dir=artifacts_dir,
            )

            self.assertEqual(resolved_path, checkpoint_path.resolve())


if __name__ == "__main__":
    unittest.main()
