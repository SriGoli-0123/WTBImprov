from .oai import OpenAIHandler
from .deepseek import DeepSeekAPIHandler
from .hunyuan import HunYuanAPIHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler
}

agent_handle_map = {**api_inference_handler_map}
