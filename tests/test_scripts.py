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
    assert "model                                  offline mock; no provider is called" in result.stdout


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


def _sample_block(heading: str) -> str:
    """The captured output under `heading` in SAMPLE_RUN.md: the fenced block
    after the one holding the command."""
    text = (ROOT / "SAMPLE_RUN.md").read_text(encoding="utf-8")
    section = text[text.index(heading):]
    fences = section.split("```")
    return fences[3].strip("\n")


def test_sample_run_holds_the_demo_output_the_code_prints():
    """SAMPLE_RUN.md is the capture the README's figures are read from, so a
    change that moves a printed figure must fail here, not only change the
    capture nobody re-runs."""
    result = subprocess.run(
        [sys.executable, "scripts/run_demo.py"],
        cwd=ROOT, env=_clean_env(), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip("\n") == _sample_block("## The demonstration")


def test_sample_run_holds_the_gate_output_the_code_prints():
    result = subprocess.run(
        [sys.executable, "-m", "boundary.gate"],
        cwd=ROOT, env={**_clean_env(), "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    block = _sample_block("## The boundary gate")
    # The capture ends with the exit status the captured command echoed; the
    # rest of the block is the gate's output and must equal it line for line.
    captured, _, status = block.rpartition("\n")
    assert status == "exit=0", status
    assert result.stdout.strip("\n") == captured.strip("\n"), (
        "the gate's output differs from its capture in SAMPLE_RUN.md")


def test_the_demo_uses_the_configured_pseudonym_salt():
    """The configured salt reaches the pseudonymizer: the same address gets
    a different token under a different salt."""
    runs = []
    for salt in ("", "2026-08"):
        env = _clean_env()
        env["BOUNDARY_PSEUDONYM_SALT"] = salt
        result = subprocess.run(
            [sys.executable, "scripts/run_demo.py"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        runs.append([line for line in result.stdout.splitlines()
                     if line.strip().startswith("token for the address")])
    assert runs[0] and runs[1] and runs[0] != runs[1]


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


def test_a_second_run_into_the_same_audit_file_prints_the_same_records_figure(tmp_path):
    """The audit file is resumed, so its total grows from run to run. The
    demo's records line counts what this run made and stays at 4; only the
    file total beside the path moves."""
    target = tmp_path / "boundary.audit.jsonl"
    env = _clean_env()
    env["AUDIT_LOG_PATH"] = str(target)
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "scripts/run_demo.py"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.splitlines())
    for lines in outputs:
        assert "  records made by this run               4" in lines
    written = [[line for line in lines if line.lstrip().startswith("written to")]
               for lines in outputs]
    assert f"{target} (4 records in the file" in written[0][0]
    assert f"{target} (8 records in the file" in written[1][0]
    rest = [[line for line in lines if not line.lstrip().startswith("written to")]
            for lines in outputs]
    assert rest[0] == rest[1]


# The README's published figures, each compared with the output it is read
# from. SAMPLE_RUN.md's demo block equals the live demo output (the test
# above), so a figure that matches the block matches the code; a README edit
# that moves a figure fails here.

def _norm(text: str) -> str:
    return " ".join(text.split())


def _fenced_lines(text: str, first_line: str) -> list[str]:
    """The lines of the fenced block whose first line is `first_line`."""
    start = text.index(first_line)
    assert text.count(first_line) == 1, f"{first_line!r} is not unique"
    return [line.strip() for line in text[start:text.index("```", start)].splitlines()
            if line.strip()]


def _one(pattern: str, text: str) -> tuple[str, ...]:
    found = re.findall(pattern, text)
    assert len(found) == 1, f"{pattern!r} matched {len(found)} times"
    return found[0] if isinstance(found[0], tuple) else (found[0],)


_PLAIN_ROW = r"^(source_only|source_and_index|mention_sweep|full)\s+(\d+)\s+(\d+)\s+(\d+)$"
_CRYPTO_ROW = r"^(crypto_shred_(?:text|full)(?:, no backup)?)\s+(\d+)\s+(\d+)\s+(\d+)$"


def _rows(lines, pattern: str) -> dict[str, tuple[str, ...]]:
    rows = {}
    for line in lines:
        m = re.match(pattern, line.strip())
        if m:
            assert m.group(1) not in rows, f"{m.group(1)} appears twice"
            rows[m.group(1)] = m.groups()[1:]
    return rows


def _demo_lines() -> list[str]:
    return _sample_block("## The demonstration").splitlines()


def test_the_readme_erasure_tables_are_the_demo_output():
    """The headline table, the crypto table, the key-blind and key-aware
    pair and the rotation block, against the demo's own lines."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    sample = (ROOT / "SAMPLE_RUN.md").read_text(encoding="utf-8")
    demo = _demo_lines()

    plain = _rows(demo, _PLAIN_ROW)
    assert len(plain) == 4
    header = "strategy            artifacts   leaking probes   after a routine re-ingest"
    assert _rows(_fenced_lines(readme, header), _PLAIN_ROW) == plain
    assert _rows(_fenced_lines(sample, header), _PLAIN_ROW) == plain

    crypto = _rows(demo, _CRYPTO_ROW)
    assert len(crypto) == 4
    crypto_header = "strategy                     artifacts   leaking probes   after a re-ingest"
    assert _rows(_fenced_lines(readme, crypto_header), _CRYPTO_ROW) == crypto

    sweep = r"full sweep, (key-blind|key-aware)\s+(\d+) artifacts,\s+(\d+) probes? +leaking"
    demo_sweep = dict((k, (a, n)) for k, a, n in re.findall(sweep, "\n".join(demo)))
    assert len(demo_sweep) == 2
    assert dict((k, (a, n)) for k, a, n in re.findall(sweep, readme)) == demo_sweep

    rotation = _fenced_lines(readme, "leaking probes after the erasure")
    assert len(rotation) == 5
    start = next(i for i, line in enumerate(demo)
                 if line.strip().startswith("leaking probes after the erasure"))
    assert [_norm(line) for line in rotation] == [
        _norm(line) for line in demo[start:start + 5]]


def test_the_readme_detection_figures_are_the_demo_output():
    """Recall, the two marginal-gain pairs, the identifier counts and the
    neighbor-threshold bounds. The strict/context pair is not printed by the
    demo, so it is computed here from the same labeled set."""
    from boundary.corpus import LABELED
    from boundary.detectors import detect_context, detect_strict
    from boundary.metrics import correlate

    readme = _norm((ROOT / "README.md").read_text(encoding="utf-8"))
    sample = _norm((ROOT / "SAMPLE_RUN.md").read_text(encoding="utf-8"))
    demo = _norm("\n".join(_demo_lines()))

    recall = _one(r"ALL (\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]) n=", demo)[0]
    false_pos = _one(r"false positives: (\d+)", demo)[0]
    words = {"zero": "0"}
    got = _one(r"Overall recall on the labeled set is `([^`]+)` with (\w+) false positives", readme)
    assert (got[0], words.get(got[1], got[1])) == (recall, false_pos)
    got = _one(r"Overall recall `([^`]+)` with (\w+) false positives", sample)
    assert (got[0], words.get(got[1], got[1])) == (recall, false_pos)

    gain = _one(r"marginal gain of loose over strict (\d\.\d{3})", demo)[0]
    overlap = _one(r"error overlap (\d\.\d{3})", demo)[0]
    assert _one(r"marginal gain of the `loose` family over `strict` is (\d\.\d+) for the "
                r"`loose` family, with an error overlap of (\d\.\d+)", readme) == (gain, overlap)
    assert _one(r"`loose` family adds a marginal gain of (\d\.\d+) over `strict`, with an "
                r"error overlap of (\d\.\d+)", sample) == (gain, overlap)

    ctx = correlate(LABELED, detect_strict, detect_context, "strict", "context")
    assert _one(r"Pairing `strict` with `context` instead measures (\d\.\d+) at an overlap "
                r"of (\d\.\d+)", readme) == (
        f"{ctx.marginal_gain:.3f}", f"{ctx.error_overlap:.3f}")

    removed = _one(r"direct identifiers removed (\d+)", demo)[0]
    surviving = _one(r"quasi-identifiers surviving (\d+)", demo)[0]
    assert _one(r"redaction removes (\d+) direct identifiers and leaves (\d+) "
                r"quasi-identifiers", readme) == (removed, surviving)

    best_without = _one(r"best score among non-mentions (\d\.\d{3})", demo)[0]
    worst_with = _one(r"worst score among real mentions (\d\.\d{3})", demo)[0]
    assert _one(r"Worst genuine mention `(\d\.\d+)`, best non-mention `(\d\.\d+)`",
                readme) == (worst_with, best_without)
    assert _one(r"`(\d\.\d+) < 0\.10 < (\d\.\d+)` on this corpus", readme) == (
        best_without, worst_with)
    assert _one(r"\(`max_without (\d\.\d+)`, `min_with (\d\.\d+)`\)", sample) == (
        best_without, worst_with)


def test_the_readme_real_run_figures_are_the_captured_ones():
    """The real-model figures the README quotes, against the verbatim
    captures in SAMPLE_RUN.md, and the header labels it quotes, against
    the lines real_run.py prints."""
    readme_raw = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = _norm(readme_raw)
    sample = (ROOT / "SAMPLE_RUN.md").read_text(encoding="utf-8")

    real = _one(r"(?m)^\s+real\s+1536\s+(\d\.\d{3})\s+(\d\.\d{3})\s+False$",
                "\n".join(sample.splitlines()))
    max_without, min_with = real
    assert _one(r"`min_with = (\d\.\d+)`, `max_without = (\d\.\d+)`", readme) == (
        min_with, max_without)
    assert _one(r"\(`min_with (\d\.\d+)` below `max_without (\d\.\d+)`\)", readme) == (
        min_with, max_without)

    def reach(heading: str) -> dict[str, tuple[str, str]]:
        part = sample[sample.index(heading):]
        part = part[:part.index("\n\n")]
        return {m.group(1).strip(): (m.group(2), m.group(3)) for m in re.finditer(
            r"^\s{4}(\S.*?)\s+acme-[\w-]+#\d\s+(\d\.\d{3})\s+(\d\.\d{3})\s+(?:True|False)$",
            part, re.M)}

    mock = reach("  mock embedder (256d)")
    live = reach("  real embedder (1536d)")
    assert len(mock) == len(live) == 4
    table = _fenced_lines(readme_raw, "probe                     real@1536d   mock@256d   token overlap")
    for name in ("derived-data question", "standing payroll access", "locked out after cutover"):
        row = _one(re.escape(name) + r"\s+(\d\.\d{3}) \*\s+(\d\.\d{3}) x\s+(\d\.\d{3})",
                   "\n".join(table))
        assert row == (live[name][0], mock[name][0], live[name][1])
    assert _one(r"\(control\) noise floor\s+(\d\.\d{3})\s+(\d\.\d{3})", "\n".join(table)) == (
        live["(control)"][0], mock["(control)"][0])

    billed = _one(r"embedding calls billed\s+(\d+)", sample)[0]
    assert _one(r"on OpenAI only, (\d+) embedding calls", readme) == (billed,)

    printed = (ROOT / "scripts" / "real_run.py").read_text(encoding="utf-8")
    for label in ("completion provider (effective)", "embedding provider (effective)"):
        assert f'print(f"  {label}' in printed
        assert f"`{label}`" in readme
        assert f"`{label}`" in _norm(sample)


def test_the_no_sdk_suite_figures_are_derived():
    """The skip count is measured with the provider SDKs hidden, and every
    place that states the suite's size with or without them is compared to
    the collected count and to that measurement."""
    readme = _norm((ROOT / "README.md").read_text(encoding="utf-8"))
    sample = _norm((ROOT / "SAMPLE_RUN.md").read_text(encoding="utf-8"))
    total = int(_one(r"pytest -q # (\d+) tests", readme)[0])

    import tempfile
    with tempfile.TemporaryDirectory() as hide:
        Path(hide, "sitecustomize.py").write_text(
            'import sys\nsys.modules["openai"] = None\nsys.modules["anthropic"] = None\n',
            encoding="utf-8")
        env = {**_clean_env(), "PYTHONPATH": hide}
        # The whole suite except this test, which would otherwise run itself.
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--deselect",
             "tests/test_scripts.py::test_the_no_sdk_suite_figures_are_derived"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
        )
    assert run.returncode == 0, run.stdout[-500:]
    passed, skipped = (int(n) for n in _one(r"(\d+) passed, (\d+) skipped", run.stdout))
    assert skipped >= 1
    assert passed + skipped + 1 == total

    assert _one(r"offline; (\d+) skip without", readme) == (str(skipped),)
    assert _one(r"the count is (\d+) with nothing skipped", readme) == (str(total),)
    assert _one(r"Without them the same suite is (\d+) passed and (\d+) skipped", sample) == (
        str(total - skipped), str(skipped))
    assert _one(r"^(\d+) passed in", _sample_block("## The test suite").splitlines()[-1]) == (
        str(total),)
