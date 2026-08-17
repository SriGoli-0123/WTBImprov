"""Model-free regression tests for IGAR-v26's prompt invariant and continuity."""

import json
import unittest

from wtb.model_handler.base_handler import BaseHandler
from wtb.model_handler.api_inference.igar_v26 import IGARV26Handler


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def handler(cls=IGARV26Handler):
    instance = object.__new__(cls)
    BaseHandler.__init__(instance, "test-model", 0.6)
    return instance


class StubV26(IGARV26Handler):
    def _request_tool_call(self, inference_data):
        return self.anchor, 0.1

    def _parse_api_response(self, api_response):
        return dict(api_response)

    def _request_documented_tool(self, inference_data, target_tool):
        self.forced_target = target_tool
        return {
            "reasoning_content": None,
            "content": None,
            "tool_calls": [call(target_tool, {"account_id": "acct-7"})],
            "input_token": 11,
            "output_token": 4,
            "latency": 0.2,
        }


class IGARV26InvariantTests(unittest.TestCase):
    def setUp(self):
        self.handler = handler()
        self.lookup = tool(
            "lookupAccount",
            "Look up a user's account details.",
            {
                "account_id": {
                    "type": "string",
                    "description": "Unique account identifier supplied by the user.",
                }
            },
            ["account_id"],
        )

    def test_model_input_has_only_the_stock_system_message(self):
        messages, state = self.handler._pre_messages_processing(
            "2024-07-11",
            "Look up my account.",
            [],
            [],
            tools=[self.lookup],
        )
        system_messages = [m for m in messages if m["role"] == "system"]
        self.assertEqual(system_messages, [
            {"role": "system", "content": "Current Date: 2024-07-11"}
        ])
        self.assertEqual(messages[-1], {"role": "user", "content": "Look up my account."})
        serialized = json.dumps(messages)
        self.assertNotIn("Ledger", serialized)
        self.assertNotIn("Dialogue State", serialized)
        self.assertEqual(state["_v26_current_task_index"], 1)

    def test_repair_does_not_make_a_second_prompted_model_call(self):
        chosen = {
            "reasoning_content": None,
            "content": None,
            "tool_calls": [call("lookupAccount", {"account_id": "guessed"})],
        }
        self.assertEqual(
            self.handler._maybe_repair(chosen, {"messages": [], "tools": [self.lookup]}),
            chosen,
        )

    def test_documentation_maps_question_to_unique_field_and_tool(self):
        unrelated = tool(
            "listAccounts",
            "List account summaries.",
            {"limit": {"type": "integer", "description": "Maximum results."}},
            [],
        )
        messages = [
            {"role": "system", "content": "Current Date: 2024-07-11"},
            {"role": "user", "content": "Look up my account details."},
        ]
        state = {"_v26_current_task_index": 1}
        contract = self.handler._infer_question_contract(
            "What is the account ID?", messages, [self.lookup, unrelated], state
        )
        self.assertEqual(contract["target_tool"], "lookupAccount")
        self.assertEqual(contract["open_field"], "account_id")

    def test_generic_question_is_not_mistaken_for_a_tool_field(self):
        messages = [
            {"role": "system", "content": "Current Date: 2024-07-11"},
            {"role": "user", "content": "Thanks for your help."},
        ]
        contract = self.handler._infer_question_contract(
            "Would you like anything else?",
            messages,
            [self.lookup],
            {"_v26_current_task_index": 1},
        )
        self.assertIsNone(contract)

    def test_informative_reply_closes_repeated_question_with_same_task_retry(self):
        instance = handler(StubV26)
        instance.anchor = {
            "reasoning_content": None,
            "content": "Could you provide the account ID?",
            "tool_calls": None,
            "input_token": 20,
            "output_token": 8,
        }
        messages = [
            {"role": "system", "content": "Current Date: 2024-07-11"},
            {"role": "user", "content": "Look up my account."},
            {"role": "assistant", "content": "What is the account ID?"},
            {"role": "user", "content": "It is acct-7."},
        ]
        state = {
            "_v26_current_task_index": 1,
            "_v26_open_question": {
                "target_tool": "lookupAccount",
                "open_field": "account_id",
                "asked_fields": ["account_id"],
                "user_count": 1,
            },
        }
        result = instance._consensus_generate({
            "messages": messages,
            "tools": [self.lookup],
            "dialogue_state": state,
        })
        self.assertEqual(instance.forced_target, "lookupAccount")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "lookupAccount")
        self.assertEqual(result["input_token"], 31)
        self.assertIsNone(state["_v26_open_question"])

    def test_deferral_never_forces_execution(self):
        self.assertFalse(self.handler._is_informative_reply("Wait a minute"))
        self.assertFalse(self.handler._is_informative_reply("I don't know yet"))
        self.assertFalse(self.handler._is_informative_reply("Never mind"))

    def test_referential_turn_preserves_unmentioned_executed_value(self):
        messages = [
            {"role": "system", "content": "Current Date: 2024-07-11"},
            {"role": "user", "content": "Show quests at difficulty 9."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("getQuests", {"difficulty": 9}, "old")],
            },
            {"role": "tool", "tool_call_id": "old", "content": '{"quests": []}'},
            {"role": "user", "content": "Show more at that same difficulty."},
        ]
        state = {"_v26_current_task_index": 4}
        log = {}
        repaired = self.handler._preserve_referential_values(
            "getQuests", {"difficulty": 0}, messages, state, log, 0
        )
        self.assertEqual(repaired, {"difficulty": 9})
        self.assertEqual(log["v26_continuity_notes"][0]["changes"][0]["from"], 0)

    def test_explicit_new_value_is_never_overwritten(self):
        messages = [
            {"role": "user", "content": "Show quests at difficulty 9."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("getQuests", {"difficulty": 9}, "old")],
            },
            {"role": "tool", "tool_call_id": "old", "content": "{}"},
            {"role": "user", "content": "Use difficulty 4 instead of that one."},
        ]
        repaired = self.handler._preserve_referential_values(
            "getQuests",
            {"difficulty": 4},
            messages,
            {"_v26_current_task_index": 3},
            {},
            0,
        )
        self.assertEqual(repaired, {"difficulty": 4})

    def test_current_tool_observation_is_stronger_than_old_history(self):
        messages = [
            {"role": "user", "content": "Show account old-1."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("lookupAccount", {"account_id": "old-1"}, "old")],
            },
            {"role": "tool", "tool_call_id": "old", "content": '{"account_id":"old-1"}'},
            {"role": "user", "content": "Use that linked account."},
            {"role": "tool", "tool_call_id": "new", "content": '{"account_id":"new-2"}'},
        ]
        repaired = self.handler._preserve_referential_values(
            "lookupAccount",
            {"account_id": "new-2"},
            messages,
            {"_v26_current_task_index": 3},
            {},
            1,
        )
        self.assertEqual(repaired, {"account_id": "new-2"})


if __name__ == "__main__":
    unittest.main()
