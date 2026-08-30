import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.preflight import check, main


class PreflightTests(unittest.TestCase):
    def test_missing_data_is_reported_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            report = check(missing)
        self.assertFalse(report["ok"])
        by_name = {item["name"]: item for item in report["checks"]}
        self.assertFalse(by_name["data_directory"]["ok"])

    @mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_explicit_model_satisfies_llm_model_check(self):
        with tempfile.TemporaryDirectory() as directory:
            report = check(directory, require_llm=True, model="test-model")

        by_name = {item["name"]: item for item in report["checks"]}
        self.assertTrue(by_name["openai_api_key"]["ok"])
        self.assertEqual(by_name["openai_api_key"]["detail"], "configured")
        self.assertTrue(by_name["openai_model"]["ok"])
        self.assertEqual(by_name["openai_model"]["detail"], "test-model")

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_project_env_is_detected_without_disclosing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=private-test-value\nOPENAI_MODEL=file-model\n",
                encoding="utf-8",
            )
            report = check(
                root / "missing-data",
                require_llm=True,
                dotenv_path=env_file,
            )

        by_name = {item["name"]: item for item in report["checks"]}
        self.assertTrue(by_name["openai_api_key"]["ok"])
        self.assertEqual(by_name["openai_api_key"]["detail"], "configured")
        self.assertEqual(by_name["openai_model"]["detail"], "file-model")
        self.assertNotIn("private-test-value", str(report))

    @mock.patch("agent.preflight.check")
    def test_cli_forwards_explicit_model(self, mocked_check):
        mocked_check.return_value = {"ok": True, "checks": []}

        exit_code = main([
            "--data_dir", "synthetic-data", "--require_llm",
            "--model", "test-model",
        ])

        self.assertEqual(exit_code, 0)
        mocked_check.assert_called_once_with(
            "synthetic-data", require_llm=True, model="test-model",
        )


if __name__ == "__main__":
    unittest.main()
