"""Tests for the static candidate-code safety tripwire."""

import textwrap
import unittest
from pathlib import Path

from agent.safety import check_code


def candidate(imports="", body="pass", flags=None, entrypoint=True):
    if flags is None:
        flags = ("--data_dir", "--out_dir", "--seed", "--target_split")
    flag_lines = "\n".join("parser.add_argument({!r})".format(flag) for flag in flags)
    main_body = "\n".join(part for part in (flag_lines, body) if part)
    guard = "\nif __name__ == '__main__':\n    main()\n" if entrypoint else "\n"
    return textwrap.dedent(
        """\
        {imports}
        import argparse

        def main():
            parser = argparse.ArgumentParser()
        {main_body}
        {guard}
        """
    ).format(
        imports=imports,
        main_body=textwrap.indent(main_body, "    "),
        guard=guard,
    )


class SafetyTests(unittest.TestCase):
    def assert_rejected(self, code, expected=None):
        accepted, reason = check_code(code)
        self.assertFalse(accepted, reason)
        self.assertIsInstance(reason, str)
        if expected is not None:
            self.assertIn(expected, reason)

    def test_seed_pipeline_is_accepted(self):
        root = Path(__file__).resolve().parents[1]
        seed = (root / "agent" / "seed_pipeline.py").read_text(encoding="utf-8")
        self.assertEqual((True, None), check_code(seed))

    def test_intended_output_writes_are_accepted(self):
        code = candidate(
            imports="import json, os\nimport numpy as np\nfrom data import load\nfrom evaluate import evaluate",
            body="""\
os.makedirs(parser.parse_args().out_dir, exist_ok=True)
with open(os.path.join('.', 'metrics.json'), 'w') as handle:
    json.dump({'ok': True}, handle)
np.save(os.path.join('.', 'scores.npy'), np.asarray([1.0]))""",
        )
        self.assertEqual((True, None), check_code(code))

    def test_rejects_unapproved_import_even_when_aliased(self):
        self.assert_rejected(candidate("import socket as harmless"), "socket")

    def test_rejects_every_member_of_multi_import(self):
        self.assert_rejected(candidate("import os, socket as net"), "socket")

    def test_rejects_unapproved_nested_module(self):
        self.assert_rejected(candidate("import numpy.ctypeslib as native"), "numpy.ctypeslib")

    def test_rejects_native_loading_through_numpy_namespace(self):
        code = candidate("import numpy as np", "np.ctypeslib.load_library('x', '.')")
        self.assert_rejected(code, "numpy.ctypeslib")

    def test_rejects_wildcard_and_relative_imports(self):
        with self.subTest("wildcard"):
            self.assert_rejected(candidate("from os import *"), "wildcard")
        with self.subTest("relative"):
            self.assert_rejected(candidate("from .data import load"), "relative")

    def test_rejects_dangerous_os_call_through_module_alias(self):
        self.assert_rejected(candidate("import os as harmless", "harmless.system('id')"), "os.system")

    def test_rejects_dangerous_from_imports_even_when_aliased(self):
        cases = (
            "from os import system as calculate",
            "from os import remove as clean",
            "from os import spawnv as run_model",
        )
        for imports in cases:
            with self.subTest(imports=imports):
                self.assert_rejected(candidate(imports), "forbidden")

    def test_rejects_direct_dynamic_execution_builtins(self):
        for name, expression in (
            ("eval", "eval('1 + 1')"),
            ("exec", "exec('x = 1')"),
            ("compile", "compile('1', '<x>', 'eval')"),
            ("__import__", "__import__('socket')"),
        ):
            with self.subTest(name=name):
                self.assert_rejected(candidate(body=expression), name)

    def test_rejects_references_that_alias_dynamic_execution_builtins(self):
        self.assert_rejected(candidate(body="runner = eval\nrunner('1 + 1')"), "eval")

    def test_rejects_network_and_process_modules(self):
        for module_import in (
            "import subprocess as worker",
            "import urllib.request as web",
            "from http.client import HTTPConnection as Client",
            "import multiprocessing as parallel",
        ):
            with self.subTest(module_import=module_import):
                self.assert_rejected(candidate(module_import), "not allowed")

    def test_rejects_process_and_destructive_filesystem_calls(self):
        for expression in (
            "os.fork()",
            "os.unlink('result')",
            "os.replace('a', 'b')",
            "os.rmdir('outputs')",
            "os.open('raw', 0)",
        ):
            with self.subTest(expression=expression):
                self.assert_rejected(candidate("import os", expression), "forbidden")

    def test_rejects_environment_access(self):
        cases = (
            candidate("import os", "token = os.environ.get('OPENAI_API_KEY')"),
            candidate("import os as harmless", "token = harmless.getenv('TOKEN')"),
            candidate("from os import environ as config", "token = config['TOKEN']"),
            candidate(body="token = open('/proc/self/environ').read()"),
        )
        for code in cases:
            with self.subTest(code=code):
                self.assert_rejected(code, "environment")

    def test_rejects_getattr_and_dunder_reflection_bypasses(self):
        cases = (
            candidate("import os as harmless", "getattr(harmless, 'system')('id')"),
            candidate("import os", "os.__dict__['system']('id')"),
        )
        for code in cases:
            with self.subTest(code=code):
                self.assert_rejected(code, "forbidden")

    def test_cli_names_in_comments_or_strings_do_not_satisfy_contract(self):
        code = textwrap.dedent(
            """\
            import argparse
            # --data_dir --out_dir --seed --target_split
            UNUSED = '--data_dir --out_dir --seed --target_split'

            def main():
                argparse.ArgumentParser()

            if __name__ == '__main__':
                main()
            """
        )
        self.assert_rejected(code, "missing literal argparse")

    def test_cli_flags_must_be_literal_arguments_on_the_parser(self):
        code = textwrap.dedent(
            """\
            import argparse

            def main():
                parser = argparse.ArgumentParser()
                data_flag = '--data_dir'
                parser.add_argument(data_flag)
                parser.add_argument('--out_dir')
                parser.add_argument('--seed')
                parser.add_argument('--target_split')

            if __name__ == '__main__':
                main()
            """
        )
        self.assert_rejected(code, "--data_dir")

    def test_target_split_is_required(self):
        code = candidate(flags=("--data_dir", "--out_dir", "--seed"))
        self.assert_rejected(code, "--target_split")

    def test_requires_top_level_main_function(self):
        code = "if __name__ == '__main__':\n    print('no main')\n"
        self.assert_rejected(code, "top-level main")

    def test_requires_real_main_guard_that_calls_main(self):
        no_guard = candidate(entrypoint=False)
        self.assert_rejected(no_guard, "entry point")

        no_call = candidate(entrypoint=False) + "\nif __name__ == '__main__':\n    print(main)\n"
        self.assert_rejected(no_call, "entry point")

    def test_accepts_reversed_real_main_guard(self):
        code = candidate(entrypoint=False) + "\nif '__main__' == __name__:\n    main()\n"
        self.assertEqual((True, None), check_code(code))

    def test_rejects_invalid_syntax(self):
        self.assert_rejected("def main(:\n    pass", "invalid Python syntax")


if __name__ == "__main__":
    unittest.main()
