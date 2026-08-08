import os
from collections import defaultdict
from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.grounded import GroundedHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler,
    "qwen2.5:3b": OpenAIHandler,
    "llama3.1:latest": OpenAIHandler,
    "llama3.2-vision:latest": OpenAIHandler,
    "llama2:latest": OpenAIHandler,
}

# The earlier ask-vs-act experiment, kept so its arm can still be run and
# compared. It only exists on the machine it was written on, so its absence
# is not an error.
try:
    from .api_inference.oai_m2 import AskActDiscriminationHandler

    api_inference_handler_map["Qwen/Qwen2.5-7B-Instruct-m2ask"] = AskActDiscriminationHandler
except ImportError:
    pass

# Set WTB_METHOD=grounded to run the wrapper described in
# wtb/model_handler/api_inference/grounded.py. Anything else runs the plain
# baseline, so the same command scores both with one variable changed.
_DEFAULT = GroundedHandler if os.getenv("WTB_METHOD", "").lower() == "grounded" else OpenAIHandler

HANDLER_MAP = defaultdict(lambda: _DEFAULT, api_inference_handler_map if _DEFAULT is OpenAIHandler else {})
