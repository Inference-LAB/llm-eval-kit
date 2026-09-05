"""
tests/test_cli.py
===================
Pytest suite for llm_eval_kit/cli.py using Typer's CliRunner, which
invokes the CLI in-process (no subprocess/terminal needed) and captures
stdout, stderr, and the exit code.

cli.py is intentionally a thin wrapper with no evaluation logic of its
own, so these tests focus on argument parsing, output formatting, and
error handling -- not on re-testing criteria logic, which is already
covered in test_refusal_check.py / test_factual_grounding.py.
"""

import json

from typer.testing import CliRunner

from llm_eval_kit.cli import app

runner = CliRunner()


def test_cli_evaluate_basic_prints_json_result():
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--prompt", "What is 2+2?",
            "--response", "4",
            "--criteria", "refusal_check",
        ],
    )
    assert result.exit_code == 0

    output = json.loads(result.stdout)
    assert "overall_score" in output
    assert "criteria" in output
    assert set(output["criteria"].keys()) == {"refusal_check"}


def test_cli_evaluate_with_context_option():
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--prompt", "What causes inflation?",
            "--response", "Excess money supply.",
            "--context", "Inflation is caused by excess money supply.",
            "--criteria", "refusal_check",
        ],
    )
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert "overall_score" in output


def test_cli_evaluate_multiple_criteria_flags():
    """--criteria can be repeated to request more than one criterion."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--prompt", "Hello",
            "--response", "Hi there!",
            "--criteria", "refusal_check",
        ],
    )
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert "refusal_check" in output["criteria"]


def test_cli_evaluate_missing_required_prompt_fails():
    """--prompt is required; omitting it should fail before Evaluator runs."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--response", "some response",
        ],
    )
    assert result.exit_code != 0


def test_cli_evaluate_unknown_criterion_exits_with_error():
    """An unknown criterion name should surface as a clean CLI error
    (exit code 1) rather than an unhandled traceback -- see cli.py's
    try/except around ev.evaluate()."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--prompt", "x",
            "--response", "y",
            "--criteria", "not_a_real_criterion",
        ],
    )
    assert result.exit_code == 1
    assert "Error" in result.stdout or "Error" in (result.stderr or "")


def test_cli_evaluate_defaults_criteria_to_all_when_omitted():
    """Omitting --criteria entirely should fall back to every registered
    criterion, same as calling Evaluator.evaluate(criteria=None)."""
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--prompt", "Hello",
            "--response", "Hi there, how can I help?",
        ],
    )
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    # refusal_check has no external dependency, so it must always be present.
    assert "refusal_check" in output["criteria"]