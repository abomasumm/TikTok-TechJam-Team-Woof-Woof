import json
import os
import csv
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agent import orchestrator


class ArtifactTests(unittest.TestCase):
    def test_data_views_remove_development_test_rows_and_mask_final_outcomes(self):
        header = [
            "user_id", "video_id", "date", "hourmin", "time_ms", "is_click",
            "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
            "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
            "comment_stay_time", "is_profile_enter", "is_rand", "tab",
        ]
        valid = ["u", "v1", "20220428", "1200", "1", "1", "1", "0", "0", "0", "0", "1", "50", "100", "4", "3", "1", "0", "1"]
        heldout = ["u", "v2", "20220429", "1201", "2", "1", "1", "1", "1", "1", "1", "1", "99", "100", "4", "3", "1", "0", "1"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in orchestrator.STATIC_DATA_FILES:
                (source / name).write_text("placeholder\n", encoding="utf-8")
            for name, rows in (
                ("log_standard_4_08_to_4_21_pure.csv", []),
                ("log_standard_4_22_to_5_08_pure.csv", [valid, heldout]),
            ):
                with open(source / name, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(rows)

            development = orchestrator.prepare_data_view(
                source, root / "development", "development"
            )
            manifest = json.loads((development / "view_manifest.json").read_text())
            self.assertNotIn("source_dir", manifest)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(set(manifest["source_files_sha256"]), set(orchestrator.SOURCE_DATA_FILES))
            with open(development / "log_standard_4_22_to_5_08_pure.csv", newline="") as handle:
                dev_rows = list(csv.DictReader(handle))
            self.assertEqual([row["date"] for row in dev_rows], ["20220428"])

            final = orchestrator.prepare_data_view(source, root / "final", "final")
            with open(final / "log_standard_4_22_to_5_08_pure.csv", newline="") as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual(final_rows[0]["long_view"], "1")
            for column in orchestrator.OUTCOME_COLUMNS:
                self.assertEqual(final_rows[1][column], "0")

            tampered = development / "user_features_pure.csv"
            tampered.chmod(0o644)
            tampered.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                orchestrator.prepare_data_view(source, development, "development")

    def test_validate_scores_requires_exact_flat_finite_numeric_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npy"
            np.save(path, np.asarray([0.1, 0.2]))
            np.testing.assert_allclose(orchestrator.validate_scores(path, 2), [0.1, 0.2])

            np.save(path, np.asarray([[0.1, 0.2]]))
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                orchestrator.validate_scores(path, 2)

            np.save(path, np.asarray([0.1]))
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                orchestrator.validate_scores(path, 2)

            np.save(path, np.asarray([0.1, np.nan]))
            with self.assertRaisesRegex(ValueError, "NaN"):
                orchestrator.validate_scores(path, 2)

    def test_validate_scores_rejects_oversized_file_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npy"
            with open(path, "wb") as handle:
                handle.truncate(
                    orchestrator.MAX_SCORE_HEADER_BYTES
                    + 2 * orchestrator.MAX_SCORE_BYTES_PER_ROW
                    + 1
                )
            with mock.patch.object(orchestrator.np, "load") as numpy_load:
                with self.assertRaisesRegex(ValueError, "size limit"):
                    orchestrator.validate_scores(path, 2)
            numpy_load.assert_not_called()

    def test_candidate_cannot_submit_validation_labels_verbatim(self):
        rows = [
            (20220422, "u", "a", "x", "0", 1.0, 1),
            (20220422, "u", "b", "x", "0", 1.0, 0),
        ]
        with self.assertRaisesRegex(ValueError, "label leakage"):
            orchestrator.score_validation(rows, np.asarray([1.0, 0.0]))

    def test_candidate_cannot_submit_scaled_or_nearly_direct_labels(self):
        labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float64)
        rows = [
            (20220422, f"u{i // 2}", f"v{i}", "x", "0", 1.0, int(label))
            for i, label in enumerate(labels)
        ]
        with self.assertRaisesRegex(ValueError, "label leakage"):
            orchestrator.score_validation(rows, 2.0 * labels)
        tiny_noise = np.linspace(-1e-10, 1e-10, len(labels))
        with self.assertRaisesRegex(ValueError, "label leakage"):
            orchestrator.score_validation(rows, labels + tiny_noise)

    def test_promotes_the_actual_successful_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            iteration = Path(directory) / "iter_0001"
            artifacts = iteration / "attempt_01" / "artifacts"
            artifacts.mkdir(parents=True)
            np.save(artifacts / "scores.npy", np.asarray([0.3, 0.4]))
            (artifacts / "diagnostics.json").write_text('{"fixed": true}', encoding="utf-8")
            result = {"artifact_dir": str(artifacts)}
            orchestrator.promote_attempt(iteration, "# repaired\n", result, 1)
            self.assertEqual((iteration / "pipeline.py").read_text(), "# repaired\n")
            np.testing.assert_allclose(np.load(iteration / "valid_scores.npy"), [0.3, 0.4])
            accepted = json.loads((iteration / "accepted_attempt.json").read_text())
            self.assertEqual(accepted["attempt"], 1)

    def test_child_environment_does_not_forward_secrets(self):
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "secret", "OTHER_TOKEN": "also-secret", "PATH": "/bin"},
            clear=True,
        ):
            environment = orchestrator.child_environment(9)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("OTHER_TOKEN", environment)
        self.assertEqual(environment["PYTHONHASHSEED"], "9")

    def test_child_environment_uses_only_private_attempt_tmp(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"PATH": "/bin", "TMPDIR": "/shared", "TMP": "/shared", "TEMP": "/shared"},
            clear=True,
        ):
            private = Path(directory) / "private"
            private.mkdir()
            environment = orchestrator.child_environment(3, temporary_dir=private)
        self.assertEqual(environment["TMPDIR"], str(private.resolve()))
        self.assertEqual(environment["TMP"], str(private.resolve()))
        self.assertEqual(environment["TEMP"], str(private.resolve()))

    def test_runtime_view_detects_source_and_copy_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            for name in orchestrator.RUNTIME_FILES:
                (project / name).write_text(f"# {name}\n", encoding="utf-8")
            with mock.patch.object(orchestrator, "PROJECT_ROOT", project):
                runtime = orchestrator.prepare_runtime_view(root / "run")
                orchestrator.prepare_runtime_view(root / "run")
                copied = runtime / "data.py"
                copied.chmod(0o644)
                copied.write_text("# tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "integrity"):
                    orchestrator.prepare_runtime_view(root / "run")

    def test_run_lock_is_exclusive_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            first = orchestrator.RunLock(directory).acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    orchestrator.RunLock(directory).acquire()
            finally:
                first.release()
            second = orchestrator.RunLock(directory).acquire()
            second.release()

    def test_finalization_marker_precedes_any_source_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            best_dir = run_dir / "iterations" / "iter_0000"
            best_dir.mkdir(parents=True)
            code = orchestrator.BASELINE_PIPELINE.read_text(encoding="utf-8")
            best_code = best_dir / "pipeline.py"
            best_code.write_text(code, encoding="utf-8")
            state = {
                "finalized": False,
                "best_iteration": 0,
                "best_code_sha256": orchestrator.sha256_file(best_code),
                "source_data_sha256": {"never-read.csv": "unused"},
                "seed": 0,
            }
            marker = run_dir / "finalization_started.json"

            def fail_after_marker(*args, **kwargs):
                self.assertTrue(marker.is_file())
                raise RuntimeError("stop before held-out access")

            with mock.patch.object(
                orchestrator, "verify_named_hashes", side_effect=fail_after_marker
            ):
                with self.assertRaisesRegex(RuntimeError, "stop before"):
                    orchestrator.finalize_once(
                        state, run_dir, run_dir / "data", 10, 128,
                        orchestrator.time.time() + 30,
                    )

            source_check = mock.Mock(side_effect=AssertionError("must not access source"))
            with mock.patch.object(orchestrator, "verify_named_hashes", source_check):
                with self.assertRaisesRegex(RuntimeError, "already finalized"):
                    orchestrator.finalize_once(
                        state, run_dir, run_dir / "data", 10, 128,
                        orchestrator.time.time() + 30,
                    )
            source_check.assert_not_called()

    def test_run_candidate_checks_output_contract(self):
        code = textwrap.dedent(
            """
            import argparse, os
            import numpy as np
            parser = argparse.ArgumentParser()
            parser.add_argument('--data_dir')
            parser.add_argument('--out_dir')
            parser.add_argument('--seed')
            parser.add_argument('--target_split')
            args = parser.parse_args()
            os.makedirs(args.out_dir, exist_ok=True)
            np.save(os.path.join(args.out_dir, 'scores.npy'), np.asarray([0.2, 0.1]))
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt"
            attempt.mkdir()
            code_path = attempt / "pipeline.py"
            code_path.write_text(code, encoding="utf-8")
            result = orchestrator.run_candidate(
                code_path,
                attempt,
                directory,
                0,
                "valid",
                2,
                10,
                0,
            )
            self.assertTrue(result["ok"], result)
            np.testing.assert_allclose(result["scores"], [0.2, 0.1])

    def test_run_candidate_rejects_nested_artifact_with_allowed_basename(self):
        code = textwrap.dedent(
            """
            import argparse, os
            import numpy as np
            parser = argparse.ArgumentParser()
            parser.add_argument('--data_dir')
            parser.add_argument('--out_dir')
            parser.add_argument('--seed')
            parser.add_argument('--target_split')
            args = parser.parse_args()
            os.makedirs(os.path.join(args.out_dir, 'nested'), exist_ok=True)
            np.save(os.path.join(args.out_dir, 'scores.npy'), np.asarray([0.2, 0.1]))
            np.save(os.path.join(args.out_dir, 'nested', 'scores.npy'), np.asarray([9.0]))
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt"
            attempt.mkdir()
            code_path = attempt / "pipeline.py"
            code_path.write_text(code, encoding="utf-8")
            result = orchestrator.run_candidate(
                code_path, attempt, directory, 0, "valid", 2, 10, 0
            )
        self.assertFalse(result["ok"])
        self.assertIn("nested/scores.npy", result["error"])

    def test_failed_candidate_is_repaired_and_promoted_canonically(self):
        def pipeline(values):
            return textwrap.dedent(
                f"""
                import argparse, os
                import numpy as np

                def main():
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--data_dir')
                    parser.add_argument('--out_dir')
                    parser.add_argument('--seed')
                    parser.add_argument('--target_split')
                    args = parser.parse_args()
                    os.makedirs(args.out_dir, exist_ok=True)
                    np.save(os.path.join(args.out_dir, 'scores.npy'), np.asarray({values!r}))

                if __name__ == '__main__':
                    main()
                """
            )

        fixed_code = pipeline([0.8, 0.2])

        class FakeProposer:
            def repair(self, code, error, parent_id, deadline):
                proposal = {
                    "parent_id": parent_id,
                    "operation": "debug",
                    "direction": "loss/test",
                    "hypothesis": "repair score shape",
                    "reasoning": "emit one score per row",
                    "sources": ["test fixture"],
                    "code": fixed_code,
                }
                return proposal, {"input_tokens": 3, "output_tokens": 2}, []

        proposal = {
            "parent_id": 0,
            "operation": "improve",
            "direction": "loss/test",
            "hypothesis": "exercise repair",
            "reasoning": "first artifact has wrong length",
            "sources": ["test fixture"],
            "code": pipeline([0.5]),
        }
        rows = [
            (20220422, "u", "a", "x", "0", 1.0, 1),
            (20220422, "u", "b", "x", "0", 1.0, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            iteration = Path(directory) / "iter_0001"
            iteration.mkdir()
            result = orchestrator.execute_with_repairs(
                proposal,
                iteration,
                FakeProposer(),
                rows,
                directory,
                0,
                10,
                0,
                2,
                set(),
                orchestrator.time.time() + 30,
                orchestrator.PROJECT_ROOT,
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(json.loads((iteration / "accepted_attempt.json").read_text())["attempt"], 1)
            self.assertEqual((iteration / "pipeline.py").read_text(), fixed_code)


class OpenAIProposerTests(unittest.TestCase):
    @staticmethod
    def proposal():
        return {
            "parent_id": 0,
            "operation": "improve",
            "direction": "loss/test",
            "hypothesis": "test the OpenAI function contract",
            "reasoning": "exercise request and response parsing",
            "sources": ["test fixture"],
            "code": "print('candidate')\n",
        }

    @staticmethod
    def fake_module(response, capture):
        class FakeResponses:
            def create(self, **kwargs):
                capture["request"] = kwargs
                return response

        class FakeOpenAI:
            def __init__(self, **kwargs):
                capture["client"] = kwargs
                self.responses = FakeResponses()

        return types.SimpleNamespace(OpenAI=FakeOpenAI)

    def test_responses_request_shape_parsing_and_token_accounting(self):
        proposal = self.proposal()
        response = types.SimpleNamespace(
            output=[
                types.SimpleNamespace(
                    type="function_call",
                    name="propose_iteration",
                    arguments=json.dumps(proposal),
                )
            ],
            usage=types.SimpleNamespace(input_tokens=123, output_tokens=45),
        )
        capture = {}
        fake_openai = self.fake_module(response, capture)
        config = orchestrator.OpenAIConfig(api_key="test-key", model="test-model")
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            proposer = orchestrator.OpenAIProposer(
                api_retries=0, max_tokens=777, config=config
            )
            parsed, usage, events = proposer.propose("try one change")

        self.assertEqual(parsed, proposal)
        self.assertEqual(usage, {"input_tokens": 123, "output_tokens": 45})
        self.assertEqual(events, [])
        self.assertEqual(
            capture["client"],
            {"api_key": "test-key", "timeout": 90.0, "max_retries": 0},
        )
        request = capture["request"]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["max_output_tokens"], 777)
        self.assertEqual(request["input"], [{"role": "user", "content": "try one change"}])
        self.assertEqual(request["tool_choice"], {"type": "function", "name": "propose_iteration"})
        self.assertFalse(request["parallel_tool_calls"])
        self.assertFalse(request["store"])
        self.assertEqual(request["tools"], [orchestrator.PROPOSE_TOOL])
        self.assertEqual(orchestrator.PROPOSE_TOOL["type"], "function")
        self.assertTrue(orchestrator.PROPOSE_TOOL["strict"])
        self.assertFalse(
            orchestrator.PROPOSE_TOOL["parameters"]["additionalProperties"]
        )
        self.assertIn("validation-only", request["instructions"])

    def test_missing_openai_key_and_model_have_actionable_errors(self):
        capture = {}
        fake_openai = self.fake_module(None, capture)
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                orchestrator.OpenAIProposer(
                    config=orchestrator.OpenAIConfig(api_key=None, model="test-model")
                )
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_MODEL"):
                orchestrator.OpenAIProposer(
                    config=orchestrator.OpenAIConfig(api_key="test-key", model=None)
                )

    def test_malformed_function_arguments_fail_without_live_call(self):
        response = types.SimpleNamespace(
            output=[
                types.SimpleNamespace(
                    type="function_call",
                    name="propose_iteration",
                    arguments="{not-json",
                )
            ],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        fake_openai = self.fake_module(response, {})
        config = orchestrator.OpenAIConfig(api_key="test-key", model="test-model")
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            proposer = orchestrator.OpenAIProposer(api_retries=0, config=config)
            with self.assertRaisesRegex(RuntimeError, "LLM request failed after retries"):
                proposer.propose("test")

    def test_missing_function_call_fails_without_live_call(self):
        response = types.SimpleNamespace(
            output=[types.SimpleNamespace(type="message", name=None, arguments=None)],
            usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        fake_openai = self.fake_module(response, {})
        config = orchestrator.OpenAIConfig(api_key="test-key", model="test-model")
        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            proposer = orchestrator.OpenAIProposer(api_retries=0, config=config)
            with self.assertRaisesRegex(RuntimeError, "exactly one propose_iteration"):
                proposer.propose("test")


class SearchStateTests(unittest.TestCase):
    def test_convergence_matches_epsilon_window(self):
        self.assertTrue(orchestrator.has_converged([0.60, 0.601, 0.6015, 0.6019], 0.002, 3))
        self.assertFalse(orchestrator.has_converged([0.60, 0.601, 0.602, 0.6021], 0.002, 3))

    def test_official_mode_keeps_official_convergence_authoritative(self):
        curve = [0.6000, 0.6010, 0.6015, 0.6019]
        self.assertTrue(
            orchestrator.should_stop_search([], curve, 0.002, 3, "official")
        )
        self.assertFalse(
            orchestrator.should_stop_search([], curve, 0.002, 3, "campaign")
        )

    def test_campaign_counts_only_successfully_evaluated_directions(self):
        failed = [
            {
                "iteration": index + 1,
                "direction": direction,
                "status": "failed",
                "metrics": None,
            }
            for index, direction in enumerate(orchestrator.DIRECTION_PRIORITY)
        ]
        self.assertTrue(orchestrator.has_unexplored_priority_direction(failed))

        successful = [
            {
                "iteration": index + 1,
                "direction": direction,
                "status": "ok",
                "metrics": {"valid": {"primary": 0.6}},
            }
            for index, direction in enumerate(orchestrator.DIRECTION_PRIORITY)
        ]
        self.assertFalse(orchestrator.has_unexplored_priority_direction(successful))
        curve = [0.6000, 0.6010, 0.6015, 0.6019]
        self.assertTrue(
            orchestrator.should_stop_search(successful, curve, 0.002, 3, "campaign")
        )

    def test_first_pass_rotates_after_a_validated_family(self):
        history = [
            {
                "iteration": 1,
                "direction": "loss/bpr",
                "status": "ok",
                "metrics": {"valid": {"primary": 0.603}},
                "is_best": True,
            }
        ]
        self.assertEqual(orchestrator.recommend_direction(history), orchestrator.DIRECTION_PRIORITY[1])

    def test_first_pass_routes_around_repeated_proposal_failures(self):
        history = [
            {
                "iteration": 1,
                "direction": orchestrator.DIRECTION_PRIORITY[0],
                "status": "ok",
                "metrics": {"valid": {"primary": 0.603}},
                "is_best": True,
            },
            {
                "iteration": 2,
                "direction": orchestrator.DIRECTION_PRIORITY[1],
                "status": "failed",
                "metrics": None,
                "is_best": False,
            },
            {
                "iteration": 3,
                "direction": orchestrator.DIRECTION_PRIORITY[1],
                "status": "failed",
                "metrics": None,
                "is_best": False,
            },
        ]
        self.assertEqual(orchestrator.recommend_direction(history), orchestrator.DIRECTION_PRIORITY[2])

    def test_rejected_nodes_are_never_selected_as_parents(self):
        history = [
            {"iteration": 0, "status": "ok"},
            {"iteration": 1, "status": "ok"},
            {"iteration": 2, "status": "ok"},
        ]
        self.assertEqual(orchestrator.select_parent(history, 1, 5, 0.002), 1)

    def test_reconcile_log_ahead_of_state(self):
        state = {
            "next_iteration": 1,
            "best_iteration": 0,
            "best_score": 0.6,
            "best_curve": [0.6],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
        }
        history = [
            {
                "iteration": 0,
                "is_best": True,
                "metrics": {"valid": {"primary": 0.6}},
                "tokens_input": 0,
                "tokens_output": 0,
            },
            {
                "iteration": 1,
                "is_best": True,
                "metrics": {"valid": {"primary": 0.61}},
                "tokens_input": 12,
                "tokens_output": 4,
            },
        ]
        self.assertTrue(orchestrator.reconcile_state_from_log(state, history))
        self.assertEqual(state["next_iteration"], 2)
        self.assertEqual(state["best_iteration"], 1)
        self.assertEqual(state["total_tokens"]["input_tokens"], 12)

    def test_reconcile_ignores_failed_nodes_for_convergence_curve(self):
        state = {
            "next_iteration": 3,
            "best_iteration": 0,
            "best_score": 0.6,
            "best_curve": [0.6, 0.6, 0.6],
            "total_tokens": {"input_tokens": 0, "output_tokens": 0},
        }
        history = [
            {
                "iteration": 0, "status": "ok", "is_best": True,
                "metrics": {"valid": {"primary": 0.6}}, "code_sha256": "a",
            },
            {
                "iteration": 1, "status": "failed", "is_best": False,
                "metrics": None, "code_sha256": None,
            },
            {
                "iteration": 2, "status": "ok", "is_best": False,
                "metrics": {"valid": {"primary": 0.59}}, "code_sha256": "b",
            },
        ]
        self.assertTrue(orchestrator.reconcile_state_from_log(state, history))
        self.assertEqual(state["best_curve"], [0.6, 0.6])

    def test_baseline_gate_checks_each_metric_not_only_the_mean(self):
        payload = {
            "scores": {
                "fm_official": {
                    "valid": {"GAUC": 0.6, "nDCG@5": 0.4, "primary": 0.5}
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(orchestrator, "BASELINE_SCORES", baseline):
                with self.assertRaisesRegex(RuntimeError, "GAUC"):
                    orchestrator.validate_baseline(
                        {"GAUC": 0.6021, "nDCG@5": 0.3979, "primary": 0.5},
                        0.002,
                    )


if __name__ == "__main__":
    unittest.main()
