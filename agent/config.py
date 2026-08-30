"""Minimal, allowlisted OpenAI configuration loading.

The project intentionally avoids a general-purpose dotenv loader: candidate
pipelines must not inherit arbitrary secrets from a developer's local file.
Only the two OpenAI settings used by the trusted orchestrator are parsed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
ALLOWED_ENV_NAMES = frozenset({"OPENAI_API_KEY", "OPENAI_MODEL"})
TEMPLATE_API_KEY = "replace-with-your-openai-api-key"
TEMPLATE_MODEL = "replace-with-an-openai-model-id-you-can-access"


@dataclass(frozen=True)
class OpenAIConfig:
    """Resolved trusted-process configuration with a redacted representation."""

    api_key: Optional[str] = field(repr=False)
    model: Optional[str]

    @property
    def has_api_key(self):
        return bool(self.api_key and self.api_key != TEMPLATE_API_KEY)

    @property
    def has_model(self):
        return bool(self.model and self.model != TEMPLATE_MODEL)


def _clean(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def read_allowlisted_env(path=DEFAULT_ENV_FILE):
    """Read only OpenAI settings from a simple project-local env file.

    Blank lines, comments, and an optional ``export`` prefix are supported.
    Unknown names are ignored. This deliberately does not perform variable
    interpolation, command substitution, or shell evaluation.
    """

    path = Path(path)
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or name not in ALLOWED_ENV_NAMES:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        value = _clean(value)
        if value is not None:
            values[name] = value
    return values


def load_openai_config(model=None, dotenv_path=None, environ=None):
    """Resolve OpenAI settings without mutating the process environment.

    Precedence is: explicit ``model`` argument, existing process environment,
    then the project ``.env`` file. The API key uses environment then ``.env``.
    """

    environment: Mapping[str, str] = os.environ if environ is None else environ
    file_values = read_allowlisted_env(dotenv_path or DEFAULT_ENV_FILE)

    api_key = _clean(environment.get("OPENAI_API_KEY")) or file_values.get("OPENAI_API_KEY")
    resolved_model = (
        _clean(model)
        or _clean(environment.get("OPENAI_MODEL"))
        or file_values.get("OPENAI_MODEL")
    )
    return OpenAIConfig(api_key=api_key, model=resolved_model)
