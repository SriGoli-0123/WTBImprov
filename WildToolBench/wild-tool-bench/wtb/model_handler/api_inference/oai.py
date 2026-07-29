import json
import os
import time

from wtb.model_handler.base_handler import BaseHandler
from wtb.model_handler.utils import retry_with_backoff
from openai import OpenAI, RateLimitError


class OpenAIHandler(BaseHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=60.0,
        )

    @retry_with_backoff(RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        # Measured on result_igar_v3 (1767 generations): median 90 tokens,
        # 99th pct 716, but 5 generations reached the old 2048 cap on long
        # nested tool calls. Headroom is free - nothing else generates near it.
        # Note this does NOT address the common truncations in the vLLM log:
        # those stop around the median, i.e. the model self-terminates
        # mid-JSON far below any cap. Only constrained decoding fixes those.
        kwargs.setdefault("max_tokens", int(os.getenv("WTB_MAX_TOKENS", "4096")))
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    def _request_tool_call(self, inference_data):
        messages = inference_data["messages"]
        tools = inference_data["tools"]
        api_response, latency = self.generate_with_backoff(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            tools=tools
        )

        return api_response, latency

    def _request_text_candidate(self, inference_data):
        """Generate model-authored text on the unchanged conversation.

        COG-FS uses this only when every causally supported tool candidate is
        non-executable.  No corrective prompt, benchmark label, or evaluator
        feedback is added.
        """
        api_response, latency = self.generate_with_backoff(
            messages=inference_data["messages"],
            model=self.model_name,
            temperature=self.temperature,
            tools=inference_data["tools"],
            tool_choice="none",
        )
        parsed = self._parse_api_response(api_response)
        parsed["latency"] = latency
        parsed["tool_calls"] = None
        return parsed

    def _request_text_only(self, inference_data):
        """Compatibility wrapper for retired experimental gates."""
        return self._request_text_candidate(inference_data).get("content")

    def _request_candidates(self, inference_data, n, temperature):
        '''
        Draw n diversity samples for consensus decoding. Tries a single request
        with the OpenAI `n` parameter (vLLM prefills the prompt once), falling
        back to sequential requests if the server rejects it.
        '''
        try:
            api_response, latency = self.generate_with_backoff(
                messages=inference_data["messages"],
                model=self.model_name,
                temperature=temperature,
                tools=inference_data["tools"],
                n=n,
            )
            response_data = json.loads(api_response.json())
            usage = response_data.get("usage") or {}
            candidates = []
            for i, choice in enumerate(response_data["choices"]):
                message = choice["message"]
                candidates.append({
                    "reasoning_content": message.get("reasoning_content", None),
                    "content": message.get("content", None),
                    "tool_calls": message.get("tool_calls", None),
                    # The batched request reports usage once; attribute it to the
                    # first candidate so aggregated totals stay accurate.
                    "input_token": usage.get("prompt_tokens", 0) if i == 0 else 0,
                    "output_token": usage.get("completion_tokens", 0) if i == 0 else 0,
                    "latency": latency if i == 0 else 0,
                })
            return candidates
        except Exception as e:
            print(f"Batched n={n} sampling failed ({e}); sampling sequentially.", flush=True)

        candidates = []
        for _ in range(n):
            api_response, latency = self.generate_with_backoff(
                messages=inference_data["messages"],
                model=self.model_name,
                temperature=temperature,
                tools=inference_data["tools"],
            )
            candidate = self._parse_api_response(api_response)
            candidate["latency"] = latency
            candidates.append(candidate)
        return candidates

    def _parse_api_response(self, api_response):
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        reasoning_content = message.get("reasoning_content", None)
        content = message["content"]
        tool_calls = message.get("tool_calls", None)
        input_token = api_response["usage"]["prompt_tokens"]
        output_token = api_response["usage"]["completion_tokens"]

        return {
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": tool_calls,
            "input_token": input_token,
            "output_token": output_token
        }


def main():
    from wtb.constant import DOTENV_PATH
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file
    handler = OpenAIHandler("gpt-4o-2024-11-20", 0.1)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "celsius",
                                "fahrenheit"
                            ]
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in the two cities of Boston and San Francisco?"
        }
    ]
    inference_data = {
        "messages": messages,
        "tools": tools
    }
    api_response, latency = handler._request_tool_call(inference_data)
    result = handler._parse_api_response(api_response)
    print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
