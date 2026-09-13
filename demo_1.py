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

print("\nScenario 1\n")

result = evaluator.evaluate(
    prompt="What causes climate change and how can we reduce it?",
    response=(
        "Climate change is mainly caused by greenhouse gas emissions from burning fossil fuels. "
        "It can be reduced through renewable energy and reforestation."
    ),
    context=(
        "Climate change is primarily driven by greenhouse gas emissions, especially from fossil fuels. "
        "It can be reduced through renewable energy adoption and reforestation."
    ),
    criteria=["refusal_check", "factual_grounding", "relevance", "completeness"]
)

print(json.dumps(result, indent=2, ensure_ascii=False))
