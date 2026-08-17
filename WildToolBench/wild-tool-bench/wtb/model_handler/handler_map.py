import os
from collections import defaultdict
from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.grounded import GroundedHandler
from .api_inference.gavel import GavelHandler
from .api_inference.concord import ConcordHandler
from .api_inference.prism import PrismHandler
from .api_inference.igar_v25 import IGARV25Handler
from .api_inference.igar_v26 import IGARV26Handler


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

# The method switch is external to the benchmark input. IGAR-v26 additionally
# verifies that its model-facing messages are the unmodified stock WTB form.
_METHOD = os.getenv("WTB_METHOD", "").lower()
if _METHOD in {"igar", "igar_v24"}:
    # IGAR-v24 is implemented in BaseHandler, exactly as in commit ff3d4a1.
    # Selecting the stock OpenAI-compatible handler activates that runtime.
    _DEFAULT = OpenAIHandler
elif _METHOD == "igar_v25":
    _DEFAULT = IGARV25Handler
elif _METHOD == "igar_v26":
    _DEFAULT = IGARV26Handler
elif _METHOD == "prism":
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
