import os
from collections import defaultdict
from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.grounded import GroundedHandler
from .api_inference.gavel import GavelHandler
from .api_inference.concord import ConcordHandler
from .api_inference.prism import PrismHandler


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

# The method switch is deliberately external to the benchmark input.  It
# changes the handler, never the system prompt or tool descriptions.
_METHOD = os.getenv("WTB_METHOD", "").lower()
if _METHOD == "prism":
    _DEFAULT = PrismHandler
elif _METHOD == "concord":
    _DEFAULT = ConcordHandler
elif _METHOD == "gavel":
    _DEFAULT = GavelHandler
elif _METHOD == "grounded":
    _DEFAULT = GroundedHandler
else:
    _DEFAULT = OpenAIHandler

HANDLER_MAP = defaultdict(lambda: _DEFAULT, api_inference_handler_map if _DEFAULT is OpenAIHandler else {})
