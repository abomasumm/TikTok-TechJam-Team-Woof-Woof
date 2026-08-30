"""Fast, non-training readiness check for the KuaiRand autonomous agent."""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

try:
    from .config import load_openai_config
except ImportError:  # Direct execution: python agent/preflight.py
    from config import load_openai_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = {
    "log_standard_4_08_to_4_21_pure.csv": {"user_id", "video_id", "date", "long_view"},
    "log_standard_4_22_to_5_08_pure.csv": {"user_id", "video_id", "date", "long_view"},
    "video_features_basic_pure.csv": {"video_id", "author_id"},
}


def check(data_dir, require_llm=False, model=None, dotenv_path=None):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    add("python", sys.version_info >= (3, 9), sys.version.split()[0])
    add("numpy", importlib.util.find_spec("numpy") is not None, "required")
    if require_llm:
        config = load_openai_config(model=model, dotenv_path=dotenv_path)
        add("openai_sdk", importlib.util.find_spec("openai") is not None, "required for LLM iterations")
        add(
            "openai_api_key",
            config.has_api_key,
            "configured" if config.has_api_key else "set OPENAI_API_KEY in the shell or project .env",
        )
        add(
            "openai_model",
            config.has_model,
            config.model if config.has_model else "set OPENAI_MODEL in the shell/project .env or pass --model",
        )

    data_dir = Path(data_dir)
    add("data_directory", data_dir.is_dir(), data_dir)
    if data_dir.is_dir():
        for filename, required_columns in REQUIRED_FILES.items():
            path = data_dir / filename
            if not path.is_file():
                add(filename, False, "missing")
                continue
            try:
                with open(path, newline="", encoding="utf-8") as handle:
                    header = set(next(csv.reader(handle)))
                missing = sorted(required_columns - header)
                add(filename, not missing, "header ok" if not missing else f"missing columns {missing}")
            except Exception as exc:
                add(filename, False, repr(exc))
    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "KuaiRand-Pure" / "data"))
    parser.add_argument("--require_llm", action="store_true")
    parser.add_argument(
        "--model",
        help="OpenAI model ID; overrides OPENAI_MODEL for this readiness check",
    )
    args = parser.parse_args(argv)
    report = check(args.data_dir, require_llm=args.require_llm, model=args.model)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
