import os
import sys
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

print("\nScenario 5\n")

try:
    evaluator.evaluate(prompt="test", response="test", criteria=["made_up_criterion"])
except ValueError as e:
    print(f"Caught early by registry check:\n--> {e}\n")
