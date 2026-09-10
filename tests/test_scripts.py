"""The commands the README documents, run the way the README documents them.

Why this file exists. Python puts the SCRIPT's directory on `sys.path`, not the
working directory, so `python scripts/run_demo.py` cannot import the package
sitting beside `scripts/`. The suite never caught it because pytest supplies
its own path, and the capture in SAMPLE_RUN.md was produced with a PYTHONPATH
that the README does not mention. A reviewer following the quickstart got a
traceback.

These tests run the scripts as SUBPROCESSES with a clean environment, which is
the only way to reproduce what a reviewer actually types.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _clean_env() -> dict[str, str]:
    """The reviewer's environment: no PYTHONPATH, no provider, no key."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTHONPATH",
            "ENV_FILE",
            "BOUNDARY_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "AUDIT_LOG_PATH",
        }
    }
    # A stray .env beside the repo must not select a provider for a test that
    # is about imports.
    env["ENV_FILE"] = str(ROOT / "does-not-exist.env")
    return env


def test_the_documented_demo_command_runs():
    """`python scripts/run_demo.py`, exactly as the README prints it."""
    result = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ai-data-boundary-proxy demo" in result.stdout
    assert "provider (effective)                   mock" in result.stdout


def test_the_demo_runs_from_any_working_directory(tmp_path):
    """Mutation check for the bootstrap: remove the sys.path insert and this is
    the test that fails, because the repo root is no longer the CWD either.
    """
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_demo.py")],
        cwd=tmp_path,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_demo_writes_the_audit_file_the_configuration_names(tmp_path):
    """AUDIT_LOG_PATH names a file, so the demo has to produce it.

    The file must hold the chain the demo printed, and it must verify when
    read back by something that was not the process that wrote it.

    Mutation check, executed against the log: make `AuditLog.record` stop
    writing and the file is never created, which the first assertion catches.
    """
    from boundary.audit import AuditLog

    target = tmp_path / "audit" / "boundary.audit.jsonl"
    env = _clean_env()
    env["AUDIT_LOG_PATH"] = str(target)
    result = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert target.exists(), "AUDIT_LOG_PATH was set and nothing was written"
    on_disk = AuditLog.read(target)
    assert len(on_disk) == 4
    assert on_disk.verify_chain() == (True, None, "chain intact")
    assert f"written to                             {target} (4 records" in result.stdout

    unset = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=ROOT, env=_clean_env(), capture_output=True, text=True, timeout=120,
    )
    assert "written to                             nowhere" in unset.stdout


def test_the_readme_test_count_is_the_collected_count():
    """The count the quickstart prints is re-derived, not remembered.

    `pytest --collect-only` is the source; the README quotes it. A test added
    without the README moving fails here rather than in a reviewer's terminal.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quoted = re.search(r"pytest -q\s+# (\d+) tests", readme)
    assert quoted, "the README quickstart states no test count"
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT, env=_clean_env(), capture_output=True, text=True, timeout=120,
    )
    assert collected.returncode == 0, collected.stderr
    tail = re.search(r"(\d+) tests collected", collected.stdout)
    assert tail, collected.stdout[-300:]
    assert int(quoted.group(1)) == int(tail.group(1)), (
        f"README says {quoted.group(1)} tests; pytest collects {tail.group(1)}"
    )


def test_the_completion_call_count_is_read_off_the_prompt_table():
    """The README's "2 completion calls per provider" is derived, not typed.

    One call is made per entry in `COMPLETION_PROMPTS`, so the table's length
    is the count, and the README figure is compared to it here.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("real_run", ROOT / "scripts" / "real_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = len(module.COMPLETION_PROMPTS)
    assert calls == 2
    assert "one call per entry in COMPLETION_PROMPTS, two as shipped" in " ".join(
        module.__doc__.split()
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{calls} completion calls per provider" in readme


def test_the_real_run_refuses_to_capture_a_mock():
    """A capture file that looks real and is not is worse than no capture."""
    result = subprocess.run(
        [sys.executable, "scripts/real_run.py"],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2
    assert "resolved to the mock" in result.stderr
    assert result.stdout == ""


def test_the_gate_is_runnable_as_a_module():
    result = subprocess.run(
        [sys.executable, "-m", "boundary.gate"],
        cwd=ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_demo_prints_no_ciphertext_or_key_material():
    """Determinism depends on it, and so does the byte-identity claim.

    Fernet uses a random IV, so every seal produces different bytes on every
    run. The demo stays reproducible only because it prints counts, verdicts
    and source names: never a token, a key, or a hash of either. Do NOT make
    the encryption deterministic to fix a future violation of this: that leaks
    equality, and choosing a real weakness to tidy a capture is the error this
    repository spends its length warning about.
    """
    result = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=ROOT, env=_clean_env(), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "gAAAAAB" not in result.stdout, "a Fernet token reached the output"
    # Fernet keys are 44-character url-safe base64 ending in '='.
    assert not re.search(r"[A-Za-z0-9_-]{43}=", result.stdout), "key-shaped string"
    assert "5. Crypto-shredding" in result.stdout


def test_the_demo_is_byte_identical_across_runs():
    """The crypto section could have broken this and does not."""
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "scripts/run_demo.py"],
            cwd=ROOT, env=_clean_env(), capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
