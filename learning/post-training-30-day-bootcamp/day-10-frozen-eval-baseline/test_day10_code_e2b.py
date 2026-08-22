import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).with_name("score_day10_code_e2b.py")
SPEC = importlib.util.spec_from_file_location("score_day10_code_e2b_for_test", MODULE_PATH)
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


class FakeTimeout(Exception):
    pass


class FakeCommandExit(Exception):
    def __init__(self, exit_code):
        super().__init__(f"exit {exit_code}")
        self.exit_code = exit_code


class FakeFiles:
    def __init__(self, result_text, *, read_error=None, write_error=None):
        self.result_text = result_text
        self.read_error = read_error
        self.write_error = write_error
        self.writes = []

    def write(self, path, content):
        if self.write_error:
            raise self.write_error
        self.writes.append((path, content))

    def read(self, path):
        if self.read_error:
            raise self.read_error
        return self.result_text


class FakeCommands:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def run(self, command, *, timeout):
        self.calls.append((command, timeout))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class FakeSandbox:
    def __init__(
        self,
        result,
        *,
        command_outcome=None,
        read_error=None,
        write_error=None,
        kill_error=None,
        kill_result=True,
        info=None,
        info_error=None,
    ):
        self.files = FakeFiles(
            json.dumps(result), read_error=read_error, write_error=write_error
        )
        self.commands = FakeCommands(
            command_outcome
            if command_outcome is not None
            else SimpleNamespace(exit_code=0)
        )
        self.kill_error = kill_error
        self.kill_result = kill_result
        self.killed = False
        self.info = info or SimpleNamespace(
            template_id="rki5dems9wqfm4r03t7g",
            envd_version="0.6.10",
            cpu_count=2,
            memory_mb=512,
            allow_internet_access=False,
            volume_mounts=[],
            network={
                "allow_public_traffic": False,
                "deny_out": ["0.0.0.0/0"],
            },
            lifecycle={"on_timeout": "kill", "auto_resume": False},
        )
        self.info_error = info_error

    def get_info(self):
        if self.info_error:
            raise self.info_error
        return self.info

    def kill(self):
        self.killed = True
        if self.kill_error:
            raise self.kill_error
        return self.kill_result


class FakeFactory:
    def __init__(self, sandboxes=None, error=None):
        self.sandboxes = list(sandboxes or [])
        self.error = error
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.sandboxes.pop(0)


def bindings(factory):
    return {
        "create": factory,
        "timeout_exception": FakeTimeout,
        "command_exit_exception": FakeCommandExit,
    }


def sandbox_result(status="passed", failure_type=None, failure_message=None):
    return {
        "status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "stdout": "bounded stdout",
        "stderr": "bounded stderr",
    }


class Day10CodeE2BTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, cls.context = SCORER.prepare_tasks()

    def test_frozen_inputs_produce_exact_dev_code_order(self):
        self.assertEqual(len(self.tasks), 28)
        self.assertEqual(len({task["sample_id"] for task in self.tasks}), 28)
        self.assertTrue(all(task["sample_id"].startswith("eval:code:") for task in self.tasks))
        self.assertEqual(
            self.context["source_file_sha256"],
            "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
        )
        program = SCORER.compose_humaneval_program(self.tasks[0])
        expected = (
            self.tasks[0]["source_prompt"]
            + self.tasks[0]["completion"]
            + "\n"
            + self.tasks[0]["source_test"]
            + f"\ncheck({self.tasks[0]['entry_point']})\n"
        )
        self.assertEqual(program, expected)

    def test_contract_pins_template_and_complete_network_deny(self):
        config = self.context["config"]
        SCORER.verify_config(config)
        kwargs = SCORER.sandbox_create_kwargs(config)
        self.assertEqual(kwargs["template"], "rki5dems9wqfm4r03t7g")
        self.assertTrue(kwargs["secure"])
        self.assertFalse(kwargs["allow_internet_access"])
        self.assertEqual(
            kwargs["network"],
            {"allow_public_traffic": False, "deny_out": ["0.0.0.0/0"]},
        )
        self.assertEqual(kwargs["envs"], {})
        self.assertEqual(
            kwargs["lifecycle"], {"on_timeout": "kill", "auto_resume": False}
        )
        self.assertIsNone(kwargs["mcp"])
        self.assertEqual(kwargs["volume_mounts"], {})

        unsafe = copy.deepcopy(config)
        unsafe["network"]["deny_out"] = []
        with self.assertRaisesRegex(SCORER.Day10CodeScorerError, "network deny"):
            SCORER.verify_config(unsafe)

    def test_runner_uses_fixed_paths_limits_and_no_candidate_shell_interpolation(self):
        task = copy.deepcopy(self.tasks[0])
        task["completion"] = "UNIQUE_CANDIDATE_SENTINEL"
        runner = SCORER.build_runner_source(task, self.context["config"])
        command = SCORER.build_command(self.context["config"])
        compile(runner, "<controlled-runner>", "exec")
        self.assertNotIn(task["completion"], runner)
        self.assertNotIn(task["completion"], command)
        self.assertIn("base64.b64decode", runner)
        self.assertEqual(runner.count("self.limit = limit"), 1)
        for limit in (
            "RLIMIT_CORE",
            "RLIMIT_CPU",
            "RLIMIT_AS",
            "RLIMIT_FSIZE",
            "RLIMIT_NOFILE",
            "RLIMIT_NPROC",
        ):
            self.assertIn(limit, runner)
        self.assertIn("timeout -s TERM -k 1s 8s", command)
        self.assertIn(">/dev/null 2>/dev/null", command)

    def test_pass_failure_timeout_and_infrastructure_classification(self):
        cases = [
            (
                "passed",
                FakeFactory([FakeSandbox(sandbox_result())]),
                "passed",
                1.0,
            ),
            (
                "assertion",
                FakeFactory(
                    [
                        FakeSandbox(
                            sandbox_result(
                                "failed", "assertion_error", "expected value differs"
                            )
                        )
                    ]
                ),
                "failed",
                0.0,
            ),
            (
                "sdk_timeout",
                FakeFactory(
                    [FakeSandbox(sandbox_result(), command_outcome=FakeTimeout())]
                ),
                "infrastructure_error",
                None,
            ),
            (
                "gnu_timeout",
                FakeFactory(
                    [
                        FakeSandbox(
                            sandbox_result(), command_outcome=FakeCommandExit(124)
                        )
                    ]
                ),
                "timeout",
                0.0,
            ),
            (
                "cpu_timeout",
                FakeFactory(
                    [
                        FakeSandbox(
                            sandbox_result(), command_outcome=FakeCommandExit(152)
                        )
                    ]
                ),
                "timeout",
                0.0,
            ),
            (
                "create_error",
                FakeFactory(error=RuntimeError("unavailable")),
                "infrastructure_error",
                None,
            ),
            (
                "missing_result",
                FakeFactory(
                    [
                        FakeSandbox(
                            sandbox_result(), read_error=FileNotFoundError("missing")
                        )
                    ]
                ),
                "infrastructure_error",
                None,
            ),
        ]
        for name, factory, expected_status, expected_score in cases:
            with self.subTest(name=name):
                row = SCORER.execute_task(
                    self.tasks[0], self.context, bindings(factory)
                )
                self.assertEqual(row["execution_status"], expected_status)
                self.assertEqual(row["score"], expected_score)
                self.assertNotIn("raw_output", row)
                if factory.sandboxes:
                    self.fail("fake sandbox was not consumed")

    def test_malformed_result_channel_is_infrastructure_not_candidate_failure(self):
        malformed_results = [
            ("not-json", "invalid_result_json"),
            (123, "invalid_result_type"),
            ("x" * 32769, "oversized_result"),
        ]
        for result_text, expected_error in malformed_results:
            with self.subTest(expected_error=expected_error):
                sandbox = FakeSandbox(sandbox_result())
                sandbox.files.result_text = result_text
                row = SCORER.execute_task(
                    self.tasks[0],
                    self.context,
                    bindings(FakeFactory([sandbox])),
                )
                self.assertEqual(row["execution_status"], "infrastructure_error")
                self.assertEqual(row["score_status"], "infrastructure_error")
                self.assertIsNone(row["score"])
                self.assertEqual(row["error_type"], expected_error)

    def test_kill_failure_invalidates_an_otherwise_passing_score(self):
        sandbox = FakeSandbox(
            sandbox_result(), kill_error=RuntimeError("cleanup unavailable")
        )
        row = SCORER.execute_task(
            self.tasks[0], self.context, bindings(FakeFactory([sandbox]))
        )
        self.assertTrue(sandbox.killed)
        self.assertEqual(row["execution_status"], "infrastructure_error")
        self.assertIsNone(row["score"])
        self.assertEqual(row["error_type"], "sandbox_kill_RuntimeError")

    def test_false_kill_and_runtime_attestation_mismatch_are_infrastructure(self):
        false_kill = FakeSandbox(sandbox_result(), kill_result=False)
        row = SCORER.execute_task(
            self.tasks[0], self.context, bindings(FakeFactory([false_kill]))
        )
        self.assertEqual(row["execution_status"], "infrastructure_error")
        self.assertIsNone(row["score"])
        self.assertEqual(row["error_type"], "sandbox_kill_returned_false")

        bad_info = copy.copy(FakeSandbox(sandbox_result()).info)
        bad_info.template_id = "drifted-template"
        mismatch = FakeSandbox(sandbox_result(), info=bad_info)
        row = SCORER.execute_task(
            self.tasks[0], self.context, bindings(FakeFactory([mismatch]))
        )
        self.assertEqual(row["execution_status"], "infrastructure_error")
        self.assertIsNone(row["score"])
        self.assertEqual(row["error_type"], "sandbox_attestation_mismatch")
        self.assertEqual(row["failure_message"], "template_id")

    def test_score_tasks_creates_and_kills_a_fresh_sandbox_per_sample(self):
        selected = self.tasks[:2]
        sandboxes = [FakeSandbox(sandbox_result()), FakeSandbox(sandbox_result())]
        factory = FakeFactory(sandboxes)
        rows = SCORER.score_tasks(selected, self.context, bindings(factory))
        self.assertEqual(len(factory.calls), 2)
        self.assertTrue(all(sandbox.killed for sandbox in sandboxes))
        self.assertEqual(
            [row["sample_id"] for row in rows],
            [task["sample_id"] for task in selected],
        )
        self.assertTrue(all(call["template"] == "rki5dems9wqfm4r03t7g" for call in factory.calls))
        for ordinal, row in enumerate(rows):
            self.assertEqual(row["code_ordinal"], ordinal)
            self.assertEqual(row["code_run_hash"], self.context["code_run_hash"])
            self.assertEqual(
                row["complete_comparison_key"],
                self.context["complete_comparison_key"],
            )
            self.assertEqual(
                row["evaluator_source_sha256"],
                self.context["evaluator_source_sha256"],
            )

    def test_atomic_writer_refuses_unapproved_overwrite(self):
        rows = [{"sample_id": "one", "score": 1.0}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            SCORER.write_jsonl_atomic(rows, output, overwrite=False)
            self.assertEqual(json.loads(output.read_text()), rows[0])
            with self.assertRaisesRegex(
                SCORER.Day10CodeScorerError, "already exists"
            ):
                SCORER.write_jsonl_atomic(rows, output, overwrite=False)

    def test_main_does_not_publish_a_sidecar_when_infrastructure_failed(self):
        infra_row = {
            "sample_id": self.tasks[0]["sample_id"],
            "execution_status": "infrastructure_error",
            "score": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sidecar.jsonl"
            with (
                mock.patch.object(
                    SCORER,
                    "prepare_tasks",
                    return_value=([self.tasks[0]], self.context),
                ),
                mock.patch.object(SCORER, "load_e2b_bindings", return_value={}),
                mock.patch.object(SCORER, "score_tasks", return_value=[infra_row]),
            ):
                with self.assertRaisesRegex(
                    SCORER.Day10CodeScorerError, "no sidecar was published"
                ):
                    SCORER.main(["--output", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
