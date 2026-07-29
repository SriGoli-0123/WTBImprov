import json
import sys
import unittest

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wtb.model_handler.base_handler import BaseHandler
from wtb.model_handler.cogfs import COGFSController, RuntimeFirewallError


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The requested location.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
                "include_details": {"type": "boolean"},
            },
            "required": ["location"],
        },
    },
}

COUNTRY_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_country",
        "description": "Look up country information from a country code.",
        "parameters": {
            "type": "object",
            "properties": {
                "country_code": {
                    "type": "string",
                    "description": "The ISO country code.",
                },
            },
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


def call(name, arguments, call_id=None):
    return {
        "id": call_id or f"call-{name}-{len(json.dumps(arguments))}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def latest_user(view):
    for message in reversed(view["messages"]):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def tool_names(result):
    return [
        item.get("function", {}).get("name")
        for item in result.get("tool_calls") or []
    ]


class COGFSTests(unittest.TestCase):
    def test_model_input_keeps_only_original_date_system_prompt(self):
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

    def test_firewall_rejects_answers_ids_graphs_and_scores(self):
        forbidden = [
            "answer_list",
            "history_answer_lists",
            "test_entry_id",
            "task_idx",
            "tool_call_graph",
            "score",
        ]
        for key in forbidden:
            with self.subTest(key=key), self.assertRaises(RuntimeFirewallError):
                COGFSController.runtime_view({
                    "messages": [],
                    "tools": [],
                    key: "forbidden",
                })

    def test_plain_text_anchor_is_preserved(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Explain atmospheric humidity."},
            ],
            "tools": [WEATHER_TOOL],
        }

        result = COGFSController(
            lambda _view: response("Humidity describes water vapor in air."),
            max_workers=8,
        ).decide(runtime)

        self.assertIsNone(result["tool_calls"])
        self.assertEqual(
            result["cogfs_log"]["decision"],
            "preserve_text_anchor",
        )

    def test_supported_tool_anchor_is_preserved(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            if "Phoenix" in latest_user(view):
                return response(calls=[
                    call("get_weather", {"location": "Phoenix"})
                ])
            return response("Which location?")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertEqual(tool_names(result), ["get_weather"])
        self.assertEqual(
            result["cogfs_log"]["decision"],
            "preserve_tool_anchor",
        )

    def test_causal_required_value_keeps_derived_country_code(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Look up China."},
            ],
            "tools": [COUNTRY_TOOL],
        }

        def generate(view):
            if "China" in latest_user(view):
                return response(calls=[
                    call("lookup_country", {"country_code": "CN"})
                ])
            return response("Which country?")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertEqual(tool_names(result), ["lookup_country"])
        arguments = COGFSController._arguments(result["tool_calls"][0])
        self.assertEqual(arguments, {"country_code": "CN"})

    def test_clause_ablation_recovers_suppressed_independent_call(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {
                    "role": "user",
                    "content": "Check Phoenix weather, and look up China.",
                },
            ],
            "tools": [WEATHER_TOOL, COUNTRY_TOOL],
        }

        def generate(view):
            content = latest_user(view)
            names = {
                tool["function"]["name"] for tool in view["tools"]
            }
            has_phoenix = "Phoenix" in content
            has_china = "China" in content
            if names == {"lookup_country"}:
                return (
                    response(calls=[
                        call("lookup_country", {"country_code": "CN"})
                    ])
                    if has_china else response("Which country?")
                )
            if has_phoenix:
                return response(calls=[
                    call("get_weather", {"location": "Phoenix"})
                ])
            if has_china:
                return response(calls=[
                    call("lookup_country", {"country_code": "CN"})
                ])
            return response("What should I do?")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertCountEqual(
            tool_names(result),
            ["get_weather", "lookup_country"],
        )
        self.assertEqual(
            result["cogfs_log"]["decision"],
            "synthesize_executable_frontier",
        )
        self.assertIn(
            "span:1",
            result["cogfs_log"]["frontier"]["covered_obligations"],
        )

    def test_unconfirmed_shadow_call_is_not_added(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {
                    "role": "user",
                    "content": "Check Phoenix weather, and look up China.",
                },
            ],
            "tools": [WEATHER_TOOL, COUNTRY_TOOL],
        }

        def generate(view):
            content = latest_user(view)
            names = {
                tool["function"]["name"] for tool in view["tools"]
            }
            if names == {"lookup_country"}:
                return response("I cannot confirm that action.")
            if "Phoenix" in content:
                return response(calls=[
                    call("get_weather", {"location": "Phoenix"})
                ])
            if "China" in content:
                return response(calls=[
                    call("lookup_country", {"country_code": "CN"})
                ])
            return response("What should I do?")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertEqual(tool_names(result), ["get_weather"])

    def test_schema_projection_recovers_single_tool_from_text_anchor(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
            ],
            "tools": [WEATHER_TOOL, COUNTRY_TOOL],
        }

        def generate(view):
            content = latest_user(view)
            names = {
                tool["function"]["name"] for tool in view["tools"]
            }
            if names == {"get_weather"} and "Phoenix" in content:
                return response(calls=[
                    call("get_weather", {"location": "Phoenix"})
                ])
            return response("Let me think about that.")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertEqual(tool_names(result), ["get_weather"])
        self.assertEqual(
            result["cogfs_log"]["decision"],
            "recover_tool_frontier_from_text_anchor",
        )
        self.assertEqual(result["cogfs_log"]["projection"]["admitted"], 1)

    def test_projection_does_not_turn_unsupported_chat_into_tool_call(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {
                    "role": "user",
                    "content": "What does weather forecasting mean?",
                },
            ],
            "tools": [WEATHER_TOOL, COUNTRY_TOOL],
        }

        def generate(view):
            names = {
                tool["function"]["name"] for tool in view["tools"]
            }
            if names == {"get_weather"} and latest_user(view).strip():
                return response(calls=[
                    call("get_weather", {"location": "Phoenix"})
                ])
            return response("Forecasting estimates future conditions.")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertIsNone(result["tool_calls"])
        self.assertEqual(result["cogfs_log"]["projection"]["admitted"], 0)

    def test_unmentioned_optional_string_is_removed(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Check Phoenix weather."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            if "Phoenix" in latest_user(view):
                return response(calls=[
                    call("get_weather", {
                        "location": "Phoenix",
                        "unit": "celsius",
                    })
                ])
            return response("Which location?")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        arguments = COGFSController._arguments(result["tool_calls"][0])
        self.assertEqual(arguments, {"location": "Phoenix"})

    def test_causal_boolean_intent_is_kept(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {
                    "role": "user",
                    "content": "Check Phoenix weather with full details.",
                },
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            if "Phoenix" in latest_user(view):
                return response(calls=[
                    call("get_weather", {
                        "location": "Phoenix",
                        "include_details": True,
                    })
                ])
            return response("Which location?")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        arguments = COGFSController._arguments(result["tool_calls"][0])
        self.assertEqual(arguments["include_details"], True)

    def test_completed_call_without_new_obligation_falls_back_to_text(self):
        prior = call(
            "get_weather",
            {"location": "Phoenix"},
            call_id="call-prior",
        )
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
            return response(calls=[
                call("get_weather", {"location": "Phoenix"})
            ])

        result = COGFSController(
            generate,
            generate_text=lambda _view: response(
                "The visible result says 38."
            ),
            max_workers=8,
        ).decide(runtime)

        self.assertIsNone(result["tool_calls"])
        self.assertEqual(
            result["cogfs_log"]["decision"],
            "replace_unsupported_action_with_model_text",
        )

    def test_latest_user_can_causally_reauthorize_completed_call(self):
        prior = call(
            "get_weather",
            {"location": "Phoenix"},
            call_id="call-prior",
        )
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
                {"role": "user", "content": "Check Phoenix weather again."},
            ],
            "tools": [WEATHER_TOOL],
        }

        def generate(view):
            if "again" in latest_user(view):
                return response(calls=[
                    call("get_weather", {"location": "Phoenix"})
                ])
            return response("The prior check is complete.")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertEqual(tool_names(result), ["get_weather"])

    def test_visible_historical_tool_value_can_fill_current_call(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {"role": "user", "content": "Find the country code."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        call("lookup_country", {"country_code": "CN"}, "old")
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"country_code": "CN", "name": "China"}',
                    "tool_call_id": "old",
                },
                {"role": "user", "content": "Look it up again."},
            ],
            "tools": [COUNTRY_TOOL],
        }

        def generate(view):
            if "again" in latest_user(view):
                return response(calls=[
                    call("lookup_country", {"country_code": "CN"})
                ])
            return response("The previous lookup is complete.")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertEqual(tool_names(result), ["lookup_country"])
        self.assertEqual(
            COGFSController._arguments(
                result["tool_calls"][0]
            )["country_code"],
            "CN",
        )

    def test_hard_model_call_budget_is_never_exceeded(self):
        runtime = {
            "messages": [
                {"role": "system", "content": "Current Date: 2026-07-29"},
                {
                    "role": "user",
                    "content": (
                        "Check Phoenix weather, and look up China. "
                        "Also explain the result; then summarize it."
                    ),
                },
            ],
            "tools": [WEATHER_TOOL, COUNTRY_TOOL],
        }
        count = {"value": 0}

        def generate(_view):
            count["value"] += 1
            return response("Working on it.")

        result = COGFSController(generate, max_workers=8).decide(runtime)

        self.assertLessEqual(count["value"], 8)
        self.assertLessEqual(result["cogfs_log"]["model_calls_used"], 8)


if __name__ == "__main__":
    unittest.main()
