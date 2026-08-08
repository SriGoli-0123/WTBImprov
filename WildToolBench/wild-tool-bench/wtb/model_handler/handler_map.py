import os
from collections import defaultdict
from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.grounded import GroundedHandler
from .api_inference.oai_m2 import AskActDiscriminationHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler,
    "qwen2.5:3b": OpenAIHandler,
    "llama3.1:latest": OpenAIHandler,
    "llama3.2-vision:latest": OpenAIHandler,
    "llama2:latest": OpenAIHandler,
    # M2 ask-vs-act discrimination variant; routes to the same served model
    # (AskActDiscriminationHandler.WIRE_MODEL_NAME), gated by WTB_M2_ASK_ACT.
    "Qwen/Qwen2.5-7B-Instruct-m2ask": AskActDiscriminationHandler,
}

# Unnamed models fall through to the default arm:
#   WTB_METHOD=grounded -> GroundedHandler, else OpenAIHandler.
_DEFAULT = GroundedHandler if os.getenv("WTB_METHOD", "").lower() == "grounded" else OpenAIHandler

HANDLER_MAP = defaultdict(lambda: _DEFAULT, api_inference_handler_map)
