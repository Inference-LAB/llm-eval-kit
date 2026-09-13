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

print("\nScenario 3\n")

result = evaluator.evaluate(
    prompt="Tell me about deadlines",
    response="I cannot stress enough how important this deadline is -- please submit on time.",
    criteria=["refusal_check"]
)

print(json.dumps(result["criteria"]["refusal_check"], indent=2, ensure_ascii=False))
