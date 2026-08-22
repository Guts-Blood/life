#!/usr/bin/env python3
"""Replay Day 22 MBPP preference pairs and emit self-hashed evidence.

The default is deliberately non-executing.  Use ``--backend e2b`` for
untrusted model output.  ``--trusted-local`` is only a smoke-test escape hatch
for the pinned MBPP canonical answer and this repository's deterministic
mutant; it is not a security sandbox and is marked as such in every run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_E2B_CREDENTIAL = REPO_ROOT / ".env.e2b"

SCHEMA_VERSION = 1
RUN_DOMAIN = "day22.mbpp_sandbox_run"
PAIR_DOMAIN = "day22.mbpp_sandbox_pair_evidence"
STATUS_VALUES = {
    "pass",
    "wrong_answer",
    "syntax_error",
    "runtime_error",
    "timeout",
    "infra_error",
}
TRUSTED_LOCAL_ORIGINS = {
    "canonical",
    "mbpp_canonical",
    "mbpp_canonical_solution",
    "deterministic_mutation",
}
EXPECTED_E2B_SDK_VERSION = "2.37.0"
E2B_TEMPLATE_ID = "rki5dems9wqfm4r03t7g"
E2B_TEMPLATE_RUNTIME = {
    "envd_version": "0.6.10",
    "python_version": "3.11.6",
    "vcpus": 2,
    "memory_mib": 512,
}
DEFAULT_LIMITS = {
    "cpu_seconds": 2,
    "address_space_bytes": 536_870_912,
    "file_size_bytes": 1_048_576,
    "open_files": 64,
    "processes": 32,
    "core_bytes": 0,
}


class Day22SandboxError(ValueError):
    """A scorer input, safety gate, or evidence invariant failed."""


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Day22SandboxError(f"{label} must be non-empty text without NUL")
    return value


def _canonical_tests(record: Mapping[str, Any]) -> dict[str, Any]:
    tests = record.get("tests")
    if not isinstance(tests, Mapping):
        raise Day22SandboxError("record.tests must be an object")
    result: dict[str, Any] = {}
    for key in ("test_list", "challenge_test_list"):
        value = tests.get(key, [])
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() and "\x00" not in item
            for item in value
        ):
            raise Day22SandboxError(f"tests.{key} must be a list of non-empty text")
        result[key] = [item.strip() for item in value]
    setup = tests.get("test_setup_code", tests.get("setup_code", ""))
    if isinstance(setup, (list, tuple)):
        if not all(isinstance(item, str) for item in setup):
            raise Day22SandboxError("tests setup list must contain only text")
        setup = "\n".join(item.strip() for item in setup if item.strip())
    if not isinstance(setup, str) or "\x00" in setup:
        raise Day22SandboxError("tests setup must be text")
    result["test_setup_code"] = setup.strip()
    if not result["test_list"]:
        raise Day22SandboxError("tests.test_list must not be empty")
    expected = record.get("tests_sha256", tests.get("sha256"))
    actual = object_sha256(result)
    if expected is not None and expected != actual:
        raise Day22SandboxError("tests_sha256 does not bind the supplied MBPP row")
    return result


def execution_test_bundle(
    record: Mapping[str, Any], *, include_challenge_tests: bool = False
) -> tuple[dict[str, Any], str, str]:
    """Return executed tests plus full-source and executed-subset hashes."""

    source = _canonical_tests(record)
    executed = {
        "test_list": list(source["test_list"]),
        "challenge_test_list": (
            list(source["challenge_test_list"])
            if include_challenge_tests
            else []
        ),
        "test_setup_code": source["test_setup_code"],
    }
    return executed, object_sha256(source), object_sha256(executed)


def _candidate(record: Mapping[str, Any], side: str) -> dict[str, str]:
    value = record.get(side)
    if isinstance(value, str):
        response = value
        origin = record.get(f"{side}_origin")
        candidate_id = record.get(f"{side}_candidate_id")
    elif isinstance(value, Mapping):
        response = value.get("response", value.get("text"))
        origin = value.get("origin", record.get(f"{side}_origin"))
        candidate_id = value.get(
            "candidate_id", record.get(f"{side}_candidate_id")
        )
    else:
        raise Day22SandboxError(f"record.{side} must be text or an object")
    return {
        "response": _required_text(response, f"{side}.response"),
        "origin": _required_text(origin, f"{side}.origin"),
        "candidate_id": _required_text(candidate_id, f"{side}.candidate_id"),
    }


def compose_mbpp_program(
    record: Mapping[str, Any],
    response: str,
    *,
    include_challenge_tests: bool = False,
) -> str:
    """Compose one atomic MBPP variant without trusting embedded full code."""

    prefix = _required_text(record.get("code_prefix"), "code_prefix")
    response = _required_text(response, "response")
    tests, _, _ = execution_test_bundle(
        record, include_challenge_tests=include_challenge_tests
    )
    parts: list[str] = []
    if tests["test_setup_code"]:
        parts.append(tests["test_setup_code"].rstrip())
    parts.append(prefix.rstrip())
    parts.append(response.rstrip())
    parts.extend(tests["test_list"])
    parts.extend(tests["challenge_test_list"])
    return "\n".join(parts) + "\n"


def build_runner_source(
    program: str,
    *,
    result_path: str,
    max_output_bytes: int,
    limits: Mapping[str, int] = DEFAULT_LIMITS,
) -> str:
    """Build a fixed runner; candidate bytes never enter a shell command."""

    if max_output_bytes <= 0:
        raise Day22SandboxError("max_output_bytes must be positive")
    required_limits = set(DEFAULT_LIMITS)
    if set(limits) != required_limits or any(
        not isinstance(limits[key], int) or limits[key] < 0 for key in limits
    ):
        raise Day22SandboxError("resource limits are incomplete or invalid")
    program_b64 = base64.b64encode(program.encode("utf-8")).decode("ascii")
    return f'''#!/usr/bin/env python3
import base64
import io
import json
import os
import resource
import sys

PROGRAM_B64 = {json.dumps(program_b64)}
RESULT_PATH = {json.dumps(result_path)}
MAX_OUTPUT_BYTES = {max_output_bytes}
LIMITS = {json.dumps(dict(limits), sort_keys=True)}
SAFE_OPEN = os.open
SAFE_WRITE = os.write
SAFE_FSYNC = os.fsync
SAFE_CLOSE = os.close


class BoundedText(io.TextIOBase):
    encoding = "utf-8"

    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.total = 0

    def writable(self):
        return True

    def write(self, value):
        if not isinstance(value, str):
            value = str(value)
        encoded = value.encode("utf-8", "replace")
        self.total += len(encoded)
        remaining = max(0, self.limit - self.size)
        if remaining:
            chunk = encoded[:remaining]
            self.parts.append(chunk.decode("utf-8", "ignore"))
            self.size += len(chunk)
        return len(value)

    def flush(self):
        return None

    def value(self):
        return "".join(self.parts)


def write_result(payload):
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    descriptor = SAFE_OPEN(RESULT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        SAFE_WRITE(descriptor, data)
        SAFE_FSYNC(descriptor)
    finally:
        SAFE_CLOSE(descriptor)


def set_limits():
    # macOS rejects lowering an infinite soft+hard RLIMIT_AS in one call.
    # Lower the soft limit and preserve the inherited hard ceiling instead.
    policies = (
        (resource.RLIMIT_CORE, LIMITS["core_bytes"]),
        (resource.RLIMIT_CPU, LIMITS["cpu_seconds"]),
        (resource.RLIMIT_AS, LIMITS["address_space_bytes"]),
        (resource.RLIMIT_FSIZE, LIMITS["file_size_bytes"]),
        (resource.RLIMIT_NOFILE, LIMITS["open_files"]),
        (resource.RLIMIT_NPROC, LIMITS["processes"]),
    )
    warnings = []
    for resource_id, requested in policies:
        _, inherited_hard = resource.getrlimit(resource_id)
        soft = requested if inherited_hard == resource.RLIM_INFINITY else min(requested, inherited_hard)
        try:
            resource.setrlimit(resource_id, (soft, inherited_hard))
        except (OSError, ValueError):
            # Darwin exposes RLIMIT_AS but rejects every finite value.  This
            # exception is confined to explicitly non-secure trusted-local
            # smoke runs; the pinned Linux E2B runner must apply the limit.
            if sys.platform == "darwin" and resource_id == resource.RLIMIT_AS:
                warnings.append("RLIMIT_AS_UNSUPPORTED_ON_DARWIN")
                continue
            raise
    return warnings


try:
    limit_warnings = set_limits()
except BaseException as error:
    write_result({{
        "result": "runner_error",
        "failure_type": type(error).__name__,
        "failure_message": "resource policy setup failed",
        "stdout": "",
        "stdout_total_bytes": 0,
        "stderr": "",
        "stderr_total_bytes": 0,
    }})
    raise SystemExit(70)

captured_stdout = BoundedText(MAX_OUTPUT_BYTES)
captured_stderr = BoundedText(MAX_OUTPUT_BYTES)
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = captured_stdout
sys.stderr = captured_stderr
result = {{
    "result": "pass",
    "failure_type": None,
    "failure_message": None,
    "limit_warnings": limit_warnings,
}}
try:
    program = base64.b64decode(PROGRAM_B64, validate=True).decode("utf-8")
    namespace = {{"__name__": "__main__"}}
    exec(compile(program, "<day22-mbpp>", "exec"), namespace, namespace)
except BaseException as error:
    if isinstance(error, SyntaxError):
        failure_type = "syntax_error"
    elif isinstance(error, AssertionError):
        failure_type = "assertion_error"
    elif isinstance(error, MemoryError):
        failure_type = "memory_limit"
    else:
        failure_type = "runtime_error"
    try:
        failure_message = str(error)[:4096]
    except BaseException:
        failure_message = "exception stringification failed"
    result.update({{
        "result": "failed",
        "failure_type": failure_type,
        "failure_message": failure_message,
    }})
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
result.update({{
    "stdout": captured_stdout.value(),
    "stdout_total_bytes": captured_stdout.total,
    "stderr": captured_stderr.value(),
    "stderr_total_bytes": captured_stderr.total,
}})
write_result(result)
raise SystemExit(0 if result["result"] == "pass" else 1)
'''


def _bounded_output(value: bytes | str, limit: int) -> dict[str, Any]:
    raw = value.encode("utf-8", "replace") if isinstance(value, str) else value
    excerpt = raw[:limit].decode("utf-8", "replace")
    return {
        "text": excerpt,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "truncated": len(raw) > limit,
    }


def _parse_runner_result(path: Path, max_result_bytes: int = 32_768) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Day22SandboxError("runner result is missing") from error
    if len(raw) > max_result_bytes:
        raise Day22SandboxError("runner result exceeds its byte limit")
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Day22SandboxError("runner result is invalid JSON") from error
    if not isinstance(result, dict):
        raise Day22SandboxError("runner result must be an object")
    return result


def _classify_runner_result(result: Mapping[str, Any]) -> dict[str, Any]:
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return _raw_outcome("infra_error", error_type="invalid_runner_output")
    base = {
        "stdout": stdout,
        "stdout_total_bytes": result.get("stdout_total_bytes"),
        "stderr": stderr,
        "stderr_total_bytes": result.get("stderr_total_bytes"),
    }
    if result.get("result") == "pass":
        return _raw_outcome("pass", **base)
    if result.get("result") == "runner_error":
        return _raw_outcome(
            "infra_error",
            error_type="runner_error",
            failure_message=result.get("failure_message"),
            **base,
        )
    if result.get("result") != "failed":
        return _raw_outcome("infra_error", error_type="invalid_runner_status", **base)
    failure_type = result.get("failure_type")
    status = {
        "assertion_error": "wrong_answer",
        "syntax_error": "syntax_error",
        "runtime_error": "runtime_error",
        "memory_limit": "runtime_error",
    }.get(failure_type)
    if status is None:
        return _raw_outcome("infra_error", error_type="invalid_failure_type", **base)
    return _raw_outcome(
        status,
        error_type=failure_type,
        failure_message=result.get("failure_message"),
        **base,
    )


def _raw_outcome(
    status: str,
    *,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    stdout_total_bytes: int | None = None,
    stderr_total_bytes: int | None = None,
    error_type: str | None = None,
    failure_message: Any = None,
) -> dict[str, Any]:
    if status not in STATUS_VALUES:
        raise Day22SandboxError(f"unknown execution status: {status}")
    return {
        "status": status,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_total_bytes": stdout_total_bytes,
        "stderr_total_bytes": stderr_total_bytes,
        "error_type": error_type,
        "failure_message": (
            str(failure_message)[:4096] if failure_message is not None else None
        ),
    }


def _terminate_process_group(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=0.25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate()


def execute_trusted_local_program(
    program: str,
    *,
    wall_timeout_seconds: float = 4.0,
    max_output_bytes: int = 8192,
    limits: Mapping[str, int] = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Explicitly execute trusted fixture code in a limited local subprocess.

    This blocks common accidents but does not disable networking and therefore
    must never be used for arbitrary or model-generated code.
    """

    if wall_timeout_seconds <= 0:
        raise Day22SandboxError("wall_timeout_seconds must be positive")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="day22-mbpp-") as temporary:
        directory = Path(temporary)
        runner_path = directory / "runner.py"
        result_path = directory / "result.json"
        runner_path.write_text(
            build_runner_source(
                program,
                result_path=str(result_path),
                max_output_bytes=max_output_bytes,
                limits=limits,
            ),
            encoding="utf-8",
        )
        runner_path.chmod(0o700)
        environment = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", runner_path.name],
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            outcome = _raw_outcome(
                "infra_error", error_type=f"spawn_{type(error).__name__}"
            )
        else:
            try:
                parent_stdout, parent_stderr = process.communicate(
                    timeout=wall_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                parent_stdout, parent_stderr = _terminate_process_group(process)
                outcome = _raw_outcome(
                    "timeout",
                    exit_code=process.returncode,
                    stdout=parent_stdout.decode("utf-8", "replace"),
                    stderr=parent_stderr.decode("utf-8", "replace"),
                    error_type="wall_timeout",
                )
            else:
                if process.returncode in {
                    -getattr(signal, "SIGXCPU", 24),
                    -signal.SIGKILL,
                    124,
                    137,
                }:
                    outcome = _raw_outcome(
                        "timeout",
                        exit_code=process.returncode,
                        stdout=parent_stdout.decode("utf-8", "replace"),
                        stderr=parent_stderr.decode("utf-8", "replace"),
                        error_type="cpu_or_wall_limit",
                    )
                else:
                    try:
                        runner_result = _parse_runner_result(result_path)
                    except Day22SandboxError:
                        outcome = _raw_outcome(
                            "runtime_error" if process.returncode else "infra_error",
                            exit_code=process.returncode,
                            stdout=parent_stdout.decode("utf-8", "replace"),
                            stderr=parent_stderr.decode("utf-8", "replace"),
                            error_type="missing_or_invalid_result",
                        )
                    else:
                        outcome = _classify_runner_result(runner_result)
                        outcome["exit_code"] = process.returncode
        outcome["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return outcome


def trusted_local_sandbox_identity(
    *,
    wall_timeout_seconds: float,
    max_output_bytes: int,
    limits: Mapping[str, int] = DEFAULT_LIMITS,
) -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "backend": "trusted_local_subprocess",
        "isolation": "trusted_local_not_security_sandbox",
        "network_isolation": False,
        "fresh_temporary_cwd_per_run": True,
        "stdin": "devnull",
        "clean_minimal_environment": True,
        "start_new_session": True,
        "process_group_kill_on_timeout": True,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_sha256": (
            file_sha256(executable) if executable.is_file() else None
        ),
        "platform": platform.platform(),
        "wall_timeout_seconds": wall_timeout_seconds,
        "max_output_bytes": max_output_bytes,
        "resource_limits": dict(limits),
    }


def read_e2b_api_key(path: Path) -> str:
    try:
        metadata = path.stat()
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError as error:
        raise Day22SandboxError("cannot read E2B credential file") from error
    if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_uid != os.getuid():
        raise Day22SandboxError("E2B credential must be owned by the user and mode 0600")
    if len(lines) != 1 or not lines[0].startswith("E2B_API_KEY="):
        raise Day22SandboxError("E2B credential must contain one E2B_API_KEY assignment")
    secret = lines[0].removeprefix("E2B_API_KEY=")
    if not secret or any(character.isspace() for character in secret):
        raise Day22SandboxError("E2B_API_KEY is empty or malformed")
    return secret


def load_e2b_bindings() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("e2b")
    except importlib.metadata.PackageNotFoundError as error:
        raise Day22SandboxError(
            "e2b is unavailable; use the post_training_sandbox environment"
        ) from error
    if version != EXPECTED_E2B_SDK_VERSION:
        raise Day22SandboxError(
            f"e2b version mismatch: expected {EXPECTED_E2B_SDK_VERSION}, got {version}"
        )
    try:
        from e2b import CommandExitException, Sandbox, TimeoutException
    except Exception as error:
        raise Day22SandboxError("cannot import the pinned e2b SDK") from error
    return {
        "create": Sandbox.create,
        "command_exit_exception": CommandExitException,
        "timeout_exception": TimeoutException,
        "sdk_version": version,
    }


def e2b_create_kwargs(api_key: str) -> dict[str, Any]:
    return {
        "template": E2B_TEMPLATE_ID,
        "timeout": 60,
        "secure": True,
        "allow_internet_access": False,
        "network": {
            "allow_public_traffic": False,
            "deny_out": ["0.0.0.0/0"],
        },
        "lifecycle": {"on_timeout": "kill", "auto_resume": False},
        "envs": {},
        "mcp": None,
        "volume_mounts": {},
        "api_key": api_key,
    }


def e2b_sandbox_identity(
    *, wall_timeout_seconds: float, max_output_bytes: int
) -> dict[str, Any]:
    return {
        "backend": "e2b_firecracker",
        "isolation": "fresh_security_sandbox",
        "sdk": {"package": "e2b", "version": EXPECTED_E2B_SDK_VERSION},
        "template_id": E2B_TEMPLATE_ID,
        "template_runtime": E2B_TEMPLATE_RUNTIME,
        "secure": True,
        "allow_internet_access": False,
        "allow_public_traffic": False,
        "deny_out": ["0.0.0.0/0"],
        "fresh_sandbox_per_run": True,
        "mounts": [],
        "envs": {},
        "mcp": None,
        "wall_timeout_seconds": wall_timeout_seconds,
        "max_output_bytes": max_output_bytes,
        "resource_limits": dict(DEFAULT_LIMITS),
    }


def _e2b_info_mismatch(info: Any) -> str | None:
    expected = {
        "template_id": E2B_TEMPLATE_ID,
        "envd_version": E2B_TEMPLATE_RUNTIME["envd_version"],
        "cpu_count": E2B_TEMPLATE_RUNTIME["vcpus"],
        "memory_mb": E2B_TEMPLATE_RUNTIME["memory_mib"],
        "allow_internet_access": False,
        "volume_mounts": [],
    }
    for field, value in expected.items():
        if getattr(info, field, None) != value:
            return field
    network = getattr(info, "network", None)
    if not isinstance(network, dict) or network.get("allow_public_traffic") is not False:
        return "network"
    if network.get("deny_out") != ["0.0.0.0/0"]:
        return "network.deny_out"
    if getattr(info, "lifecycle", None) != {
        "on_timeout": "kill",
        "auto_resume": False,
    }:
        return "lifecycle"
    return None


def _command_exit_code(value: Any) -> int | None:
    code = getattr(value, "exit_code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def execute_e2b_program(
    program: str,
    *,
    api_key: str,
    bindings: Mapping[str, Any],
    wall_timeout_seconds: float = 4.0,
    max_output_bytes: int = 8192,
) -> dict[str, Any]:
    """Execute one program in one fresh, attested, network-denied E2B sandbox."""

    started = time.perf_counter()
    sandbox = None
    outcome: dict[str, Any] | None = None
    runner_path = "/home/user/day22_mbpp_runner.py"
    result_path = "/home/user/day22_mbpp_result.json"
    try:
        try:
            sandbox = bindings["create"](**e2b_create_kwargs(api_key))
        except Exception as error:
            return _raw_outcome(
                "infra_error", error_type=f"sandbox_create_{type(error).__name__}"
            )
        try:
            mismatch = _e2b_info_mismatch(sandbox.get_info())
        except Exception as error:
            outcome = _raw_outcome(
                "infra_error", error_type=f"sandbox_info_{type(error).__name__}"
            )
            return outcome
        if mismatch is not None:
            outcome = _raw_outcome(
                "infra_error",
                error_type="sandbox_attestation_mismatch",
                failure_message=mismatch,
            )
            return outcome
        runner = build_runner_source(
            program,
            result_path=result_path,
            max_output_bytes=max_output_bytes,
        )
        try:
            sandbox.files.write(runner_path, runner)
        except Exception as error:
            outcome = _raw_outcome(
                "infra_error", error_type=f"runner_write_{type(error).__name__}"
            )
            return outcome
        command = (
            f"timeout -s TERM -k 1s {wall_timeout_seconds}s "
            f"python3 {runner_path} >/dev/null 2>/dev/null"
        )
        try:
            command_result = sandbox.commands.run(
                command, timeout=wall_timeout_seconds + 7
            )
        except bindings["timeout_exception"]:
            outcome = _raw_outcome("infra_error", error_type="sdk_timeout")
            return outcome
        except bindings["command_exit_exception"] as error:
            command_result = error
        except Exception as error:
            outcome = _raw_outcome(
                "infra_error", error_type=f"command_{type(error).__name__}"
            )
            return outcome
        exit_code = _command_exit_code(command_result)
        if exit_code in {124, 137}:
            outcome = _raw_outcome(
                "timeout", exit_code=exit_code, error_type="cpu_or_wall_limit"
            )
            return outcome
        try:
            result_text = sandbox.files.read(result_path)
            if not isinstance(result_text, str) or len(result_text.encode("utf-8")) > 32_768:
                raise Day22SandboxError("invalid E2B runner result")
            result = json.loads(result_text)
            if not isinstance(result, dict):
                raise Day22SandboxError("invalid E2B runner result")
        except Exception:
            outcome = _raw_outcome(
                "runtime_error" if exit_code not in (None, 0) else "infra_error",
                exit_code=exit_code,
                error_type="missing_or_invalid_result",
            )
            return outcome
        outcome = _classify_runner_result(result)
        outcome["exit_code"] = exit_code
        return outcome
    finally:
        if sandbox is not None:
            try:
                killed = sandbox.kill()
                if killed is False and outcome is not None:
                    outcome.update(
                        _raw_outcome(
                            "infra_error", error_type="sandbox_kill_returned_false"
                        )
                    )
            except Exception as error:
                if outcome is not None:
                    outcome.update(
                        _raw_outcome(
                            "infra_error",
                            error_type=f"sandbox_kill_{type(error).__name__}",
                        )
                    )
        if outcome is not None:
            outcome["duration_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )


def _validated_record_identity(record: Mapping[str, Any]) -> dict[str, str]:
    pair_id = _required_text(record.get("pair_id"), "pair_id")
    task_value = record.get("task_id")
    if isinstance(task_value, bool) or not isinstance(task_value, (str, int)):
        raise Day22SandboxError("task_id must be text or an integer")
    task_id = str(task_value)
    if not task_id.strip():
        raise Day22SandboxError("task_id must not be empty")
    return {
        "pair_id": pair_id,
        "task_id": task_id,
        "family_id": _required_text(record.get("family_id"), "family_id"),
    }


def _finalize_run(
    *,
    identity: Mapping[str, str],
    side: str,
    attempt: int,
    response: str,
    program: str,
    source_tests_sha256: str,
    test_sha256: str,
    sandbox_identity: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    status = outcome.get("status")
    if status not in STATUS_VALUES:
        raise Day22SandboxError("executor returned an unknown status")
    stdout = _bounded_output(str(outcome.get("stdout") or ""), 8192)
    stderr = _bounded_output(str(outcome.get("stderr") or ""), 8192)
    if isinstance(outcome.get("stdout_total_bytes"), int):
        stdout["bytes"] = outcome["stdout_total_bytes"]
        stdout["truncated"] = stdout["bytes"] > len(
            stdout["text"].encode("utf-8")
        )
    if isinstance(outcome.get("stderr_total_bytes"), int):
        stderr["bytes"] = outcome["stderr_total_bytes"]
        stderr["truncated"] = stderr["bytes"] > len(
            stderr["text"].encode("utf-8")
        )
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": RUN_DOMAIN,
        **identity,
        "run_id": f"{identity['pair_id']}:{side}:{attempt}:{uuid.uuid4().hex}",
        "side": side,
        "attempt": attempt,
        "status": status,
        "exit_code": outcome.get("exit_code"),
        "response_sha256": text_sha256(response),
        "program_sha256": text_sha256(program),
        "source_tests_sha256": source_tests_sha256,
        "test_sha256": test_sha256,
        "sandbox_digest": object_sha256(sandbox_identity),
        "sandbox": dict(sandbox_identity),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_ms": outcome.get("duration_ms"),
        "stdout": stdout["text"],
        "stdout_sha256": stdout["sha256"],
        "stdout_bytes": stdout["bytes"],
        "stdout_truncated": stdout["truncated"],
        "stderr": stderr["text"],
        "stderr_sha256": stderr["sha256"],
        "stderr_bytes": stderr["bytes"],
        "stderr_truncated": stderr["truncated"],
        "error_type": outcome.get("error_type"),
        "failure_message": outcome.get("failure_message"),
    }
    row["run_sha256"] = object_sha256(row)
    return row


def score_pair(
    record: Mapping[str, Any],
    *,
    backend: str | None = None,
    include_challenge_tests: bool = False,
    wall_timeout_seconds: float = 4.0,
    max_output_bytes: int = 8192,
    e2b_api_key: str | None = None,
    e2b_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay chosen/rejected twice.  No backend means no code execution."""

    if backend not in {"trusted_local", "e2b"}:
        raise Day22SandboxError(
            "execution is disabled by default; select E2B or explicitly trusted_local"
        )
    identity = _validated_record_identity(record)
    candidates = {side: _candidate(record, side) for side in ("chosen", "rejected")}
    if backend == "trusted_local":
        if (
            candidates["chosen"]["origin"]
            not in {"canonical", "mbpp_canonical", "mbpp_canonical_solution"}
            or candidates["rejected"]["origin"] != "deterministic_mutation"
        ):
            raise Day22SandboxError(
                "trusted_local only accepts canonical vs deterministic_mutation pairs"
            )
        sandbox_identity = trusted_local_sandbox_identity(
            wall_timeout_seconds=wall_timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    else:
        if not e2b_api_key or e2b_bindings is None:
            raise Day22SandboxError("E2B credentials and pinned bindings are required")
        sandbox_identity = e2b_sandbox_identity(
            wall_timeout_seconds=wall_timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    _, source_tests_sha256, test_sha256 = execution_test_bundle(
        record, include_challenge_tests=include_challenge_tests
    )
    execution: dict[str, Any] = {}
    for side in ("chosen", "rejected"):
        candidate = candidates[side]
        program = compose_mbpp_program(
            record,
            candidate["response"],
            include_challenge_tests=include_challenge_tests,
        )
        runs = []
        for attempt in (1, 2):
            if backend == "trusted_local":
                outcome = execute_trusted_local_program(
                    program,
                    wall_timeout_seconds=wall_timeout_seconds,
                    max_output_bytes=max_output_bytes,
                )
            else:
                outcome = execute_e2b_program(
                    program,
                    api_key=e2b_api_key or "",
                    bindings=e2b_bindings or {},
                    wall_timeout_seconds=wall_timeout_seconds,
                    max_output_bytes=max_output_bytes,
                )
            runs.append(
                _finalize_run(
                    identity=identity,
                    side=side,
                    attempt=attempt,
                    response=candidate["response"],
                    program=program,
                    source_tests_sha256=source_tests_sha256,
                    test_sha256=test_sha256,
                    sandbox_identity=sandbox_identity,
                    outcome=outcome,
                )
            )
        execution[side] = {
            "candidate_id": candidate["candidate_id"],
            "origin": candidate["origin"],
            "response_sha256": text_sha256(candidate["response"]),
            "runs": runs,
        }
    chosen_statuses = [run["status"] for run in execution["chosen"]["runs"]]
    rejected_statuses = [run["status"] for run in execution["rejected"]["runs"]]
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": PAIR_DOMAIN,
        **identity,
        "backend": backend,
        "source_tests_sha256": source_tests_sha256,
        "test_sha256": test_sha256,
        "include_challenge_tests": include_challenge_tests,
        "sandbox_digest": object_sha256(sandbox_identity),
        "execution": execution,
        "pair_execution_status": (
            "eligible"
            if chosen_statuses == ["pass", "pass"]
            and rejected_statuses == ["wrong_answer", "wrong_answer"]
            else "quarantine"
        ),
    }
    row["evidence_sha256"] = object_sha256(row)
    return row


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Day22SandboxError(f"cannot read input: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise Day22SandboxError(f"invalid JSON on line {line_number}") from error
        if not isinstance(value, dict):
            raise Day22SandboxError(f"line {line_number} must contain an object")
        rows.append(value)
    if not rows:
        raise Day22SandboxError("input JSONL is empty")
    return rows


def write_jsonl_atomic(rows: Sequence[Mapping[str, Any]], path: Path, *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise Day22SandboxError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="pair JSONL")
    parser.add_argument("--output", type=Path, required=True, help="evidence JSONL")
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument(
        "--backend", choices=("e2b",), help="secure backend for untrusted code"
    )
    execution.add_argument(
        "--trusted-local",
        action="store_true",
        help="UNSAFE: only canonical/deterministic fixture smoke tests",
    )
    parser.add_argument("--e2b-credential", type=Path, default=DEFAULT_E2B_CREDENTIAL)
    parser.add_argument("--include-challenge-tests", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=4.0)
    parser.add_argument("--max-output-bytes", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backend = "trusted_local" if args.trusted_local else args.backend
        api_key = None
        bindings = None
        if backend == "e2b":
            api_key = read_e2b_api_key(args.e2b_credential.resolve())
            bindings = load_e2b_bindings()
        rows = load_jsonl(args.input.resolve())
        evidence = [
            score_pair(
                row,
                backend=backend,
                include_challenge_tests=args.include_challenge_tests,
                wall_timeout_seconds=args.timeout_seconds,
                max_output_bytes=args.max_output_bytes,
                e2b_api_key=api_key,
                e2b_bindings=bindings,
            )
            for row in rows
        ]
        write_jsonl_atomic(evidence, args.output, overwrite=args.overwrite)
    except Day22SandboxError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
