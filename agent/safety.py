"""Defense-in-depth static check on LLM-generated pipeline.py before it's ever
executed. This is a denylist, NOT a sandbox -- it catches the obvious cases
(network calls, shelling out, deleting files, dynamic exec, installing
packages) but does not guarantee isolation. It exists because this harness
runs model-authored code directly on the operator's machine with no container
boundary; treat it as a tripwire, not a security boundary.
"""
import re

DENYLIST_PATTERNS = [
    r"\bsubprocess\b", r"\bos\.system\b", r"\bos\.popen\b",
    r"\bos\.remove\b", r"\bos\.unlink\b", r"\bos\.rmdir\b",
    r"\bshutil\.rmtree\b", r"\bshutil\.move\b",
    r"\bsocket\b", r"\burllib\b", r"\brequests\b", r"\bhttp\.client\b",
    r"\bftplib\b", r"\bsmtplib\b",
    r"\beval\s*\(", r"\bexec\s*\(", r"\b__import__\s*\(",
    r"\bctypes\b", r"\bpip\b", r"\bimportlib\b",
    r"\bopen\s*\([^)]*['\"]w['\"][^)]*\)\s*.*\.\./",  # writes escaping via ../
]

ALLOWED_IMPORT_MODULES = {
    "numpy", "np", "math", "collections", "itertools", "functools", "random",
    "json", "argparse", "time", "sys", "os", "csv", "heapq", "dataclasses",
    "typing", "data", "evaluate", "re", "statistics", "array",
}

IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.MULTILINE)


def check_code(code: str):
    """Returns (ok: bool, reason: str | None)."""
    for pat in DENYLIST_PATTERNS:
        if re.search(pat, code):
            return False, f"denylisted pattern matched: {pat}"

    for m in IMPORT_RE.finditer(code):
        root = m.group(1).split(".")[0]
        if root not in ALLOWED_IMPORT_MODULES:
            return False, (
                f"import of '{root}' is not on the allowed list "
                f"({sorted(ALLOWED_IMPORT_MODULES)}). Only numpy + stdlib + "
                f"local data/evaluate modules are available in this environment."
            )

    if "def main(" not in code and "__main__" not in code:
        return False, "no `if __name__ == '__main__':` entry point found"

    for required in ("--data_dir", "--out_dir", "--seed"):
        if required not in code:
            return False, f"missing required CLI argument: {required}"

    return True, None
