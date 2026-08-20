"""
llm_eval_kit.cli
==================

A thin command-line wrapper around Evaluator.evaluate(). This file
performs no evaluation logic itself -- it only parses arguments, calls
Evaluator, and prints the result as JSON.

Usage:
    llm-eval evaluate --prompt "..." --response "..." --context "..." \
        --criteria factual_grounding --criteria relevance
"""

import json
from typing import List, Optional

import typer

from llm_eval_kit.evaluator import Evaluator

app = typer.Typer(help="llm-eval-kit: evaluate LLM responses against structured criteria.")


@app.callback()
def main() -> None:
    """llm-eval-kit command-line interface."""
    pass


@app.command()
def evaluate(
    prompt: str = typer.Option(..., "--prompt", help="The prompt sent to the LLM."),
    response: str = typer.Option(..., "--response", help="The LLM's response to evaluate."),
    context: str = typer.Option("", "--context", help="Optional reference text for grounding checks."),
    criteria: Optional[List[str]] = typer.Option(
        None,
        "--criteria",
        help="Criterion name(s) to run. Repeat the flag for multiple. "
             "Defaults to all registered criteria if omitted.",
    ),
) -> None:
    """
    Evaluate a single (prompt, response, context) triple and print the
    structured result as JSON.
    """
    ev = Evaluator()
    try:
        result = ev.evaluate(prompt=prompt, response=response, context=context, criteria=criteria)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()