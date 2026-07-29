from collections import defaultdict
from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .local_inference.hf_qwen import HuggingFaceQwenHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler,
    "Qwen/Qwen2.5-7B-Instruct": HuggingFaceQwenHandler,
    "qwen2.5:3b": OpenAIHandler,
    "llama3.1:latest": OpenAIHandler,
    "llama3.2-vision:latest": OpenAIHandler,
    "llama2:latest": OpenAIHandler
}

HANDLER_MAP = defaultdict(lambda: OpenAIHandler, api_inference_handler_map)
