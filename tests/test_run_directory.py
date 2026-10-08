import json
import tempfile
import unittest
from pathlib import Path

from baselines.run_directory import prepare_fresh_run


class RunDirectoryTests(unittest.TestCase):
    def test_archive_incomplete_run_preserves_log_and_training_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "UNet"
            directory.mkdir()
            (directory / "config.json").write_text('{}')
            (directory / "train.log").write_text('iter=50 loss=0.7\n')
            (directory / "results.json").write_text(json.dumps({"training": [{"iteration": 50}], "validation": [], "test": []}))
            (directory / "checkpoints").mkdir()
            archive = prepare_fresh_run(directory)
            self.assertFalse(directory.exists())
            self.assertTrue(archive.name.startswith("UNet_incomplete_"))
            self.assertEqual((archive / "train.log").read_text(), 'iter=50 loss=0.7\n')
            self.assertEqual(json.loads((archive / "results.json").read_text())["training"][0]["iteration"], 50)

    def test_checkpoints_validation_and_unknown_files_are_protected(self):
        for evidence in ("checkpoints/last.pth", "validation/iter_0004000.json", "metrics.jsonl", "my_notes.txt"):
            with self.subTest(evidence=evidence), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "UNet"
                directory.mkdir()
                (directory / "config.json").write_text('{}')
                path = directory / evidence
                path.parent.mkdir(exist_ok=True)
                path.write_text('preserve')
                with self.assertRaises(FileExistsError):
                    prepare_fresh_run(directory)
                self.assertEqual(path.read_text(), 'preserve')


if __name__ == "__main__":
    unittest.main()
