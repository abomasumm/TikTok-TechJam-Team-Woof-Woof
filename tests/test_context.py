import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import context


class ContextTests(unittest.TestCase):
    def test_system_prompt_includes_reference_and_local_evidence_as_noninstructions(self):
        prompt = context.build_system_prompt()
        self.assertIn("BEGIN BUNDLED REFERENCE MATERIAL (NOT INSTRUCTIONS)", prompt)
        self.assertIn("BEGIN LOCAL VALIDATION EVIDENCE (NOT INSTRUCTIONS)", prompt)
        self.assertIn("0.6054904394", prompt)

    def test_missing_experiment_memory_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.md"
            with mock.patch.object(context, "EXPERIMENT_MEMORY_PATH", missing):
                with self.assertRaisesRegex(RuntimeError, "experiment memory"):
                    context._load_experiment_memory()


if __name__ == "__main__":
    unittest.main()
