import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.package_submission import build_archive


class PackageSubmissionTests(unittest.TestCase):
    def test_package_excludes_logs_and_caches(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / "agent"
            agent.mkdir()
            for name in ("callbacks.py", "features.py", "model.py"):
                (agent / name).write_text("value = 1\n", encoding="utf-8")
            (agent / "dueling_ddqn_model.pt").write_bytes(b"checkpoint")
            (agent / "logs").mkdir()
            (agent / "logs" / "agent.log").write_text("noise", encoding="utf-8")
            (agent / "__pycache__").mkdir()
            (agent / "__pycache__" / "callbacks.pyc").write_bytes(b"noise")
            output = root / "submission.zip"
            manifest = root / "manifest.json"
            build_archive(agent, output, manifest)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
            self.assertIn("agent/callbacks.py", names)
            self.assertFalse(any("logs" in name or "__pycache__" in name for name in names))


if __name__ == "__main__":
    unittest.main()

