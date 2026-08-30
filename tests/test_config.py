import tempfile
import unittest
from pathlib import Path

from agent.config import (
    TEMPLATE_API_KEY,
    TEMPLATE_MODEL,
    load_openai_config,
    read_allowlisted_env,
)


class OpenAIConfigTests(unittest.TestCase):
    def test_project_env_reads_only_allowlisted_names(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# local settings\n"
                "OPENAI_API_KEY='file-key'\n"
                "export OPENAI_MODEL=\"file-model\"\n"
                "UNRELATED_SECRET=do-not-load\n",
                encoding="utf-8",
            )
            values = read_allowlisted_env(env_file)

        self.assertEqual(
            values,
            {"OPENAI_API_KEY": "file-key", "OPENAI_MODEL": "file-model"},
        )

    def test_environment_takes_precedence_over_project_env(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\n",
                encoding="utf-8",
            )
            config = load_openai_config(
                dotenv_path=env_file,
                environ={"OPENAI_API_KEY": "shell-key", "OPENAI_MODEL": "shell-model"},
            )

        self.assertEqual(config.api_key, "shell-key")
        self.assertEqual(config.model, "shell-model")

    def test_explicit_model_has_highest_precedence(self):
        config = load_openai_config(
            model="cli-model",
            dotenv_path=Path("does-not-exist"),
            environ={"OPENAI_API_KEY": "shell-key", "OPENAI_MODEL": "shell-model"},
        )
        self.assertEqual(config.model, "cli-model")

    def test_configuration_repr_redacts_key(self):
        config = load_openai_config(
            dotenv_path=Path("does-not-exist"),
            environ={"OPENAI_API_KEY": "sensitive-value", "OPENAI_MODEL": "model"},
        )
        self.assertNotIn("sensitive-value", repr(config))
        self.assertTrue(config.has_api_key)

    def test_template_placeholders_are_not_treated_as_configured(self):
        config = load_openai_config(
            dotenv_path=Path("does-not-exist"),
            environ={
                "OPENAI_API_KEY": TEMPLATE_API_KEY,
                "OPENAI_MODEL": TEMPLATE_MODEL,
            },
        )
        self.assertFalse(config.has_api_key)
        self.assertFalse(config.has_model)


if __name__ == "__main__":
    unittest.main()
