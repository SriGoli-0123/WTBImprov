import json
import sys
import unittest

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wtb.model_handler.handler_map import HANDLER_MAP
from wtb.model_handler.local_inference.hf_qwen import (
    HuggingFaceQwenHandler,
    NativeGeneration,
)


class NativeQwenHandlerTests(unittest.TestCase):
    def test_exact_qwen_model_uses_native_handler(self):
        self.assertIs(
            HANDLER_MAP["Qwen/Qwen2.5-7B-Instruct"],
            HuggingFaceQwenHandler,
        )

    def test_history_arguments_are_objects_for_native_template(self):
        messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "getWeather",
                    "arguments": '{"city": "Phoenix"}',
                },
            }],
        }]

        converted = HuggingFaceQwenHandler._native_messages(messages)

        self.assertEqual(
            converted[0]["tool_calls"][0]["function"]["arguments"],
            {"city": "Phoenix"},
        )
        self.assertIsInstance(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            str,
        )

    def test_multiple_native_tool_calls_are_preserved(self):
        text = """
<tool_call>
{"name": "getWeather", "arguments": {"city": "Phoenix"}}
</tool_call>
<tool_call>
{"name": "getWeather", "arguments": {"city": "Boston"}}
</tool_call>
"""

        calls = HuggingFaceQwenHandler._parse_native_tool_calls(text)

        self.assertEqual(
            [call["function"]["name"] for call in calls],
            ["getWeather", "getWeather"],
        )
        self.assertEqual(
            [
                json.loads(call["function"]["arguments"])["city"]
                for call in calls
            ],
            ["Phoenix", "Boston"],
        )

    def test_string_encoded_arguments_are_supported(self):
        text = (
            '<tool_call>{"name":"lookup","arguments":'
            '"{\\"id\\": 7}"}</tool_call>'
        )

        calls = HuggingFaceQwenHandler._parse_native_tool_calls(text)

        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"id": 7},
        )

    def test_malformed_native_call_is_not_repaired(self):
        text = (
            '<tool_call>{"name":"lookup","arguments":{"id":7}}</tool_call>'
            '<tool_call>{"name":"broken","arguments":</tool_call>'
        )

        self.assertIsNone(
            HuggingFaceQwenHandler._parse_native_tool_calls(text)
        )

    def test_text_response_remains_text(self):
        handler = HuggingFaceQwenHandler.__new__(
            HuggingFaceQwenHandler
        )
        raw = NativeGeneration(
            text="I need the city before I can check.",
            prompt_tokens=100,
            completion_tokens=12,
        )

        parsed = handler._parse_api_response(raw)

        self.assertIsNone(parsed["tool_calls"])
        self.assertEqual(
            parsed["content"],
            "I need the city before I can check.",
        )
        self.assertEqual(parsed["input_token"], 100)
        self.assertEqual(parsed["output_token"], 12)

    def test_tool_response_removes_only_complete_call_blocks_from_content(self):
        handler = HuggingFaceQwenHandler.__new__(
            HuggingFaceQwenHandler
        )
        raw = NativeGeneration(
            text=(
                "Checking now.\n"
                "<tool_call>\n"
                '{"name":"getWeather","arguments":{"city":"Phoenix"}}\n'
                "</tool_call>"
            ),
            prompt_tokens=90,
            completion_tokens=20,
        )

        parsed = handler._parse_api_response(raw)

        self.assertEqual(parsed["content"], "Checking now.")
        self.assertEqual(len(parsed["tool_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
