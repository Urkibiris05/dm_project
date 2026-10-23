"""Integration tests for the Bash command dispatcher."""

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dm.sh"


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    """Create a harmless executable that prints the arguments it receives."""
    executable = tmp_path / "python-recorder"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf '<%s>\\n' \"$@\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def run_cli(
    *args: str,
    python: Path | None = None,
) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    if python is not None:
        environment["PYTHON"] = str(python)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_txt_assignment_and_model_preparation() -> None:
    result = run_cli("help")

    assert result.returncode == 0
    assert "prepare-assigner" in result.stdout
    assert "assign --input mensajes.txt" in result.stdout


@pytest.mark.parametrize(
    ("command", "expected_args"),
    [
        (
            ("vectorize", "transformer", "--model", "beto-sts"),
            ["-m", "vectorization", "transformer", "--model", "beto-sts"],
        ),
        (
            (
                "assign",
                "--input",
                "input messages.txt",
                "--model",
                "model.joblib",
            ),
            [
                "-m",
                "E8.assign_messages",
                "assign",
                "--input",
                "input messages.txt",
                "--model",
                "model.joblib",
            ],
        ),
        (
            ("prepare-assigner", "--min-samples", "5", "--percentile", "50"),
            [
                "-m",
                "E8.assign_messages",
                "prepare",
                "--min-samples",
                "5",
                "--percentile",
                "50",
            ],
        ),
    ],
)
def test_commands_forward_arguments_without_shell_splitting(
    fake_python: Path,
    command: tuple[str, ...],
    expected_args: list[str],
) -> None:
    result = run_cli(*command, python=fake_python)

    assert result.returncode == 0
    assert result.stdout.splitlines() == [f"<{arg}>" for arg in expected_args]


@pytest.mark.parametrize(
    ("command", "expected_tail"),
    [
        (("e0", "--for-the-impatient"), ["--rapido"]),
        (("e1", "--for-the-impatient", "--rapido"), ["--rapido"]),
        (
            ("cluster-messages", "--all", "--for-the-impatient"),
            ["--all", "--sample-size", "1000"],
        ),
        (
            ("cluster-messages", "--sample-size", "30", "--for-the-impatient"),
            ["--sample-size", "30"],
        ),
        (
            ("evaluate-e4", "--for-the-impatient"),
            ["--max-samples", "1000", "--min-samples", "5"],
        ),
        (
            (
                "evaluate-e4",
                "--for-the-impatient",
                "--max-samples",
                "200",
                "--min-samples",
                "10",
            ),
            ["--max-samples", "200", "--min-samples", "10"],
        ),
    ],
)
def test_impatient_profile_adds_defaults_without_overriding_user_values(
    fake_python: Path,
    command: tuple[str, ...],
    expected_tail: list[str],
) -> None:
    result = run_cli(*command, python=fake_python)

    assert result.returncode == 0
    forwarded = [
        line[1:-1] for line in result.stdout.splitlines()
    ]
    assert "--for-the-impatient" not in forwarded
    assert forwarded[-len(expected_tail):] == expected_tail


@pytest.mark.parametrize("alias", ["quick", "for-the-impatient"])
def test_quick_alias_runs_fast_test_suite(
    fake_python: Path,
    alias: str,
) -> None:
    result = run_cli(alias, python=fake_python)

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "<-m>",
        "<pytest>",
        "<-q>",
        "<dbscan_implementation/tests>",
        "<evaluation/tests>",
        "<vectorization/tests>",
        "<E4/tests>",
        "<E8/tests>",
    ]


def test_invalid_impatient_usage_fails_with_clear_message(
    fake_python: Path,
) -> None:
    result = run_cli("assign", "--for-the-impatient", python=fake_python)

    assert result.returncode == 2
    assert (
        "--for-the-impatient no está disponible para assign."
        in result.stderr
    )


def test_unknown_command_fails_with_help_hint() -> None:
    result = run_cli("not-a-command")

    assert result.returncode == 2
    assert "usa './dm.sh help'" in result.stderr
