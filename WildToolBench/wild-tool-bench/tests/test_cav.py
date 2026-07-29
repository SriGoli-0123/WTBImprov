import sys
import unittest

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wtb.model_handler.base_handler import BaseHandler
from wtb.model_handler.cav import CAVController, RuntimeFirewallError


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

COUNTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_country",
        "description": "Look up a country by ISO code.",
        "parameters": {
            "type": "object",
            "properties": {"country_code": {"type": "string"}},
            "required": ["country_code"],
        },
    },
}


def response(content=None, calls=None):
    return {
        "reasoning_content": None,
        "content": content,
        "tool_calls": calls,
        "input_token": 1,
        "output_token": 1,
        "latency": 0.01,
    }


def tool_call(location, call_id="call-current", extra=None):
    arguments = {"location": location}
    if extra:
        arguments.update(extra)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": arguments,
        },
    }


class CAVTests(unittest.TestCase):
    def test_model_input_keeps_original_minimal_system_prompt(self):
        handler = BaseHandler("test-model", 0)
        messages = handler._pre_messages_processing(
            env_info="2026-07-29",
            current_task="Check Phoenix weather.",
            history_tasks=[],
            history_answer_lists=[],
            tools=[WEATHER_TOOL],
        )
        self.assertEqual(messages, [
            {"role": "system", "content": "Current Date: 2026-07-29"},
            {"role": "user", "content": "Check Phoenix weather."},
        ])

    def test_firewall_rejects_evaluator_data(self):
        with self.assertRaises(RuntimeFirewallError):
            CAVController.runtime_view({
                "messages": [],
                "tools": [],
                "answer_list": [{"action": {"name": "get_weather"}}],
            })

    def test_supported_anchor_is_preserved(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check the weather in Phoenix."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            self.assertEqual(set(view), {"messages", "tools"})
            if not any(
                m.get("role") == "user" and m.get("content")
                for m in view["messages"]
            ):
                return response("How can I help?")
            return response(calls=[tool_call("Phoenix")])

        result = CAVController(generate, max_workers=2).decide(runtime)
        self.assertEqual(
            result["cav_log"]["decision"],
            "preserve_tool_anchor",
        )
        self.assertEqual(
            CAVController._arguments(result["tool_calls"][0])["location"],
            "Phoenix",
        )

    def test_counterfactuals_support_derived_representation(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Look up China."},
            ],
            "tools": [COUNTRY_TOOL],
        }
        derived_call = {
            "id": "country-call",
            "type": "function",
            "function": {
                "name": "lookup_country",
                "arguments": {"country_code": "CN"},
            },
        }

        def generate(view):
            if not any(
                m.get("role") == "user" and m.get("content")
                for m in view["messages"]
            ):
                return response("Which country?")
            return response(calls=[derived_call])

        result = CAVController(generate, max_workers=2).decide(runtime)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertTrue(result["cav_log"]["causal_support"]["proven"])

    def test_action_momentum_falls_back_to_model_text(self):
        prior = tool_call("Phoenix", call_id="call-prior")
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
                {"role": "assistant", "content": "", "tool_calls": [prior]},
                {
                    "role": "tool",
                    "content": '{"humidity": 20}',
                    "tool_call_id": "call-prior",
                },
                {"role": "user", "content": "What does humidity mean?"},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            has_tool_role = any(
                message.get("role") == "tool" for message in view["messages"]
            )
            if has_tool_role:
                return response(calls=[tool_call("Tucson")])
            return response("Humidity is the amount of water vapor in air.")

        def generate_text(_view):
            return response("Humidity is the amount of water vapor in air.")

        result = CAVController(
            generate,
            generate_text=generate_text,
            max_workers=2,
        ).decide(runtime)
        self.assertIsNone(result["tool_calls"])
        self.assertTrue(result["cav_log"]["action_momentum"])
        self.assertEqual(
            result["cav_log"]["decision"],
            "replace_action_momentum_with_model_text",
        )

    def test_exact_completed_call_is_not_reexecuted(self):
        prior = tool_call("Phoenix", call_id="call-prior")
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
                {"role": "assistant", "content": "", "tool_calls": [prior]},
                {
                    "role": "tool",
                    "content": '{"temperature": 38}',
                    "tool_call_id": "call-prior",
                },
                {"role": "user", "content": "What was the temperature?"},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(_view):
            return response(calls=[tool_call("Phoenix")])

        def generate_text(_view):
            return response("The visible result is 38.")

        result = CAVController(
            generate,
            generate_text=generate_text,
            max_workers=2,
        ).decide(runtime)
        self.assertIsNone(result["tool_calls"])
        suppressed = result["cav_log"]["audit"]["suppressed_calls"]
        self.assertIn("already_completed", suppressed[0]["reasons"])

    def test_causally_reauthorized_repeat_is_allowed(self):
        prior = tool_call("Phoenix", call_id="call-prior")
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
                {"role": "assistant", "content": "", "tool_calls": [prior]},
                {
                    "role": "tool",
                    "content": '{"temperature": 38}',
                    "tool_call_id": "call-prior",
                },
                {"role": "user", "content": "Run the same check once more."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            latest_users = [
                message.get("content")
                for message in view["messages"]
                if message.get("role") == "user"
            ]
            if latest_users and latest_users[-1]:
                return response(calls=[tool_call("Phoenix")])
            return response("The previous check is complete.")

        result = CAVController(generate, max_workers=2).decide(runtime)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertTrue(result["cav_log"]["causal_support"]["proven"])

    def test_executable_frontier_keeps_independent_supported_call(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
            ],
            "tools": [WEATHER_TOOL],
        }
        anchor = response(calls=[
            tool_call("Phoenix", call_id="supported"),
            tool_call("Atlantis", call_id="unsupported"),
        ])

        def generate(_view):
            return anchor

        result = CAVController(generate, max_workers=2).decide(runtime)
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(
            CAVController._arguments(result["tool_calls"][0])["location"],
            "Phoenix",
        )
        self.assertEqual(
            result["cav_log"]["decision"],
            "emit_executable_frontier",
        )

    def test_unknown_optional_argument_is_removed(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(_view):
            return response(calls=[
                tool_call("Phoenix", extra={"invented_transport_flag": True})
            ])

        result = CAVController(generate, max_workers=2).decide(runtime)
        arguments = CAVController._arguments(result["tool_calls"][0])
        self.assertEqual(arguments, {"location": "Phoenix"})
        self.assertEqual(
            result["cav_log"]["audit"]["dropped_optional_arguments"],
            [{"tool": "get_weather", "argument": "invented_transport_flag"}],
        )

    def test_schema_valid_optional_without_receipt_is_removed(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(_view):
            return response(calls=[
                tool_call("Phoenix", extra={"unit": "celsius"})
            ])

        result = CAVController(generate, max_workers=2).decide(runtime)
        arguments = CAVController._arguments(result["tool_calls"][0])
        self.assertEqual(arguments, {"location": "Phoenix"})


if __name__ == "__main__":
    unittest.main()
