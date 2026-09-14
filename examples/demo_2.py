import os
import sys
import json
import warnings
import io
import contextlib

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
    from llm_eval_kit import Evaluator
    evaluator = Evaluator()
    _ = evaluator.evaluate(prompt="w", response="w", context="w", criteria=["relevance", "factual_grounding"])

print("\nScenario 2\n")

bad_numeric = evaluator.evaluate(
    prompt="What is the boiling point of water at sea level?",
    response="Water boils at 50 degrees Celsius at sea level.",
    context="At sea level, the boiling point of water is 100°C (212°F).",
    criteria=["factual_grounding"]
)

print(json.dumps(bad_numeric["criteria"]["factual_grounding"], indent=2, ensure_ascii=False))
