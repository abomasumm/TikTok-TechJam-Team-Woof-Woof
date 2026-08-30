"""Static validation for model-authored ``pipeline.py`` files.

This module is deliberately a *tripwire*, not a sandbox. It parses source with
``ast`` so import aliases and ``from ... import ...`` forms cannot bypass the
checks as they could with the old regular-expression implementation. It also
checks that the pipeline contains the CLI contract the orchestrator relies on.

Important limitations: static analysis cannot prove that arbitrary Python is
safe. Dynamic attribute construction, monkey-patching, native-code behaviour,
and writes through otherwise permitted APIs (for example ``open`` and
``numpy.save``) remain outside this check. The orchestrator adds a scrubbed
environment, isolated working directory, timeout, and resource limits, but that
is still not a hard container/VM security boundary.
"""

import ast
from typing import Dict, List, Optional, Set, Tuple


# Keep this list intentionally small. In particular, modules that can open a
# network connection, launch processes, dynamically import code, or offer broad
# filesystem mutation are absent. A few safe submodules are named explicitly;
# allowing every submodule of an approved root would also approve things such as
# numpy.ctypeslib.
ALLOWED_IMPORT_MODULES = {
    "argparse",
    "array",
    "collections",
    "collections.abc",
    "csv",
    "data",
    "dataclasses",
    "evaluate",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "numpy",
    "numpy.linalg",
    "numpy.random",
    "numpy.typing",
    "os",
    "os.path",
    "random",
    "re",
    "statistics",
    "sys",
    "time",
    "typing",
}


# Qualified references rejected even when the containing module is imported
# under an alias or the member is imported directly under another name.
_FORBIDDEN_REFERENCES = {
    # Process creation/control and shell execution.
    "os.abort",
    "os._exit",
    "os.execv",
    "os.execve",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execvp",
    "os.execvpe",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.popen",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.startfile",
    "os.system",
    # Destructive or low-level filesystem operations. Normal output creation
    # via open()/numpy.save() remains permitted for the pipeline contract.
    "os.chmod",
    "os.chown",
    "os.fchmod",
    "os.fchown",
    "os.ftruncate",
    "os.lchown",
    "os.link",
    "os.mkfifo",
    "os.mknod",
    "os.open",
    "os.remove",
    "os.removedirs",
    "os.rename",
    "os.renames",
    "os.replace",
    "os.rmdir",
    "os.symlink",
    "os.truncate",
    "os.unlink",
    "pathlib.Path.chmod",
    "pathlib.Path.hardlink_to",
    "pathlib.Path.rename",
    "pathlib.Path.replace",
    "pathlib.Path.rmdir",
    "pathlib.Path.symlink_to",
    "pathlib.Path.unlink",
    "shutil.move",
    "shutil.rmtree",
    # Environment access can expose the orchestrator's API credentials.
    "os.environ",
    "os.environb",
    "os.getenv",
    "os.getenvb",
    "os.putenv",
    "os.unsetenv",
    # Import/reflection surfaces commonly used to route around a denylist.
    "sys.modules",
}

_FORBIDDEN_MODULE_PREFIXES = {
    "aiohttp",
    "asyncio",
    "builtins",
    "ctypes",
    "ftplib",
    "http",
    "httpx",
    "importlib",
    "multiprocessing",
    "numpy.ctypeslib",
    "numpy.distutils",
    "numpy.f2py",
    "pathlib",
    "pkgutil",
    "requests",
    "runpy",
    "shutil",
    "smtplib",
    "socket",
    "ssl",
    "subprocess",
    "telnetlib",
    "urllib",
    "webbrowser",
}

_FORBIDDEN_DIRECT_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
}

_FORBIDDEN_REFLECTION_ATTRIBUTES = {
    "__bases__",
    "__class__",
    "__code__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
}

_REQUIRED_CLI_FLAGS = {"--data_dir", "--out_dir", "--seed", "--target_split"}


def _location(node: ast.AST) -> str:
    lineno = getattr(node, "lineno", None)
    return "" if lineno is None else " at line {}".format(lineno)


def _qualified_name(node: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    """Return a best-effort dotted name, resolving imported aliases."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value, aliases)
        if base:
            return "{}.{}".format(base, node.attr)
    return None


def _forbidden_reason(qualified_name: str) -> Optional[str]:
    if qualified_name in _FORBIDDEN_REFERENCES:
        if qualified_name.startswith("os.environ") or qualified_name in {
            "os.getenv", "os.getenvb", "os.putenv", "os.unsetenv"
        }:
            return "environment access is forbidden: '{}'".format(qualified_name)
        return "dangerous API is forbidden: '{}'".format(qualified_name)

    for prefix in _FORBIDDEN_MODULE_PREFIXES:
        if qualified_name == prefix or qualified_name.startswith(prefix + "."):
            return "forbidden module/API reference: '{}'".format(qualified_name)

    final_component = qualified_name.rsplit(".", 1)[-1]
    if final_component in _FORBIDDEN_REFLECTION_ATTRIBUTES:
        return "reflection attribute is forbidden: '{}'".format(qualified_name)
    return None


class _ImportCollector(ast.NodeVisitor):
    """Validate every import and record the name it binds in local code."""

    def __init__(self) -> None:
        self.aliases: Dict[str, str] = {}
        self.issues: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            module = imported.name
            if module not in ALLOWED_IMPORT_MODULES:
                self.issues.append(
                    "import of '{}' is not allowed{}".format(module, _location(node))
                )
                continue

            # ``import os.path`` binds ``os``; ``import os.path as p`` binds p.
            if imported.asname:
                self.aliases[imported.asname] = module
            else:
                root = module.split(".", 1)[0]
                self.aliases[root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.issues.append("relative imports are not allowed{}".format(_location(node)))
            return

        module = node.module or ""
        if module not in ALLOWED_IMPORT_MODULES:
            self.issues.append(
                "import from '{}' is not allowed{}".format(module or "<unknown>", _location(node))
            )
            return

        for imported in node.names:
            if imported.name == "*":
                self.issues.append("wildcard imports are not allowed{}".format(_location(node)))
                continue
            qualified = "{}.{}".format(module, imported.name)
            reason = _forbidden_reason(qualified)
            if reason:
                self.issues.append(reason + _location(node))
                continue
            self.aliases[imported.asname or imported.name] = qualified


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, aliases: Dict[str, str]) -> None:
        self.aliases = aliases
        self.issues: List[str] = []

    def _record_reference(self, node: ast.AST) -> None:
        qualified = _qualified_name(node, self.aliases)
        if not qualified:
            return
        reason = _forbidden_reason(qualified)
        if reason:
            self.issues.append(reason + _location(node))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._record_reference(node)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            return
        qualified = self.aliases.get(node.id, node.id)
        if qualified in _FORBIDDEN_DIRECT_CALLS or qualified == "__builtins__":
            self.issues.append(
                "dynamic execution builtin reference is forbidden: '{}'{}".format(
                    qualified, _location(node)
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        qualified = _qualified_name(node.func, self.aliases)
        if qualified:
            if qualified in _FORBIDDEN_DIRECT_CALLS:
                self.issues.append(
                    "dynamic execution builtin is forbidden: '{}'{}".format(
                        qualified, _location(node)
                    )
                )
            else:
                reason = _forbidden_reason(qualified)
                if reason:
                    self.issues.append(reason + _location(node))

        # Catch straightforward getattr(os_alias, "system")-style attempts.
        if qualified == "getattr" and len(node.args) >= 2:
            target = _qualified_name(node.args[0], self.aliases)
            member_node = node.args[1]
            if (
                target
                and isinstance(member_node, ast.Constant)
                and isinstance(member_node.value, str)
            ):
                reflected = "{}.{}".format(target, member_node.value)
                reason = _forbidden_reason(reflected)
                if reason:
                    self.issues.append(reason + _location(node))

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # On systems that expose it, /proc/.../environ bypasses os.environ.
        if isinstance(node.value, str):
            normalized = node.value.replace("\\", "/").lower()
            if "/proc/" in normalized and normalized.endswith("/environ"):
                self.issues.append(
                    "reading a process environment file is forbidden{}".format(_location(node))
                )


class _CliContractVisitor(ast.NodeVisitor):
    """Find literal argparse flags on parser objects created inside main()."""

    def __init__(self, aliases: Dict[str, str]) -> None:
        self.aliases = aliases
        self.parser_names: Set[str] = set()
        self.flags: Set[str] = set()

    def _capture_parser_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.parser_names.add(target.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = node.value
        if isinstance(value, ast.Call):
            constructor = _qualified_name(value.func, self.aliases)
            if constructor == "argparse.ArgumentParser":
                for target in node.targets:
                    self._capture_parser_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = node.value
        if isinstance(value, ast.Call):
            constructor = _qualified_name(value.func, self.aliases)
            if constructor == "argparse.ArgumentParser":
                self._capture_parser_target(node.target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "add_argument"
            and isinstance(func.value, ast.Name)
            and func.value.id in self.parser_names
        ):
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value in _REQUIRED_CLI_FLAGS:
                        self.flags.add(argument.value)
        self.generic_visit(node)


def _is_main_guard(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False

    left, right = test.left, test.comparators[0]

    def is_name(value: ast.AST) -> bool:
        return isinstance(value, ast.Name) and value.id == "__name__"

    def is_main_literal(value: ast.AST) -> bool:
        return isinstance(value, ast.Constant) and value.value == "__main__"

    return (is_name(left) and is_main_literal(right)) or (
        is_main_literal(left) and is_name(right)
    )


def _guard_calls_main(node: ast.If, aliases: Dict[str, str]) -> bool:
    for statement in node.body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Call) and _qualified_name(child.func, aliases) == "main":
                return True
    return False


def check_code(code: str) -> Tuple[bool, Optional[str]]:
    """Return ``(accepted, rejection_reason)`` for candidate source code."""
    try:
        tree = ast.parse(code, filename="pipeline.py")
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        suffix = "" if line is None else " at line {}".format(line)
        return False, "invalid Python syntax{}: {}".format(suffix, exc)

    imports = _ImportCollector()
    imports.visit(tree)
    if imports.issues:
        return False, imports.issues[0]

    safety = _SafetyVisitor(imports.aliases)
    safety.visit(tree)
    if safety.issues:
        return False, safety.issues[0]

    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    if not main_functions:
        return False, "missing top-level main() function"

    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If) and _is_main_guard(node.test)
    ]
    if not any(_guard_calls_main(guard, imports.aliases) for guard in guards):
        return False, "missing real `if __name__ == '__main__': main()` entry point"

    cli = _CliContractVisitor(imports.aliases)
    for function in main_functions:
        cli.visit(function)
    missing = sorted(_REQUIRED_CLI_FLAGS - cli.flags)
    if missing:
        return False, "missing literal argparse CLI argument(s): {}".format(", ".join(missing))

    return True, None
